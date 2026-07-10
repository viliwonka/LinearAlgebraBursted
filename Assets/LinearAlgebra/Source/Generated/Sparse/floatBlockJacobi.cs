using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra;
using LinearAlgebra.Internal;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Block-Jacobi preconditioner for a square BSR (<c>BlockRows==BlockCols</c>, <c>BR==BC</c>):
    /// z = M⁻¹ r where M = blockdiag(A_00, A_11, ..., A_{nb-1,nb-1}), i.e. each diagonal block
    /// inverted independently and applied block-wise. The <c>BR==1</c> case degenerates to
    /// point-Jacobi (z_i = r_i / A_ii) -- no special-cased code path is needed, the general
    /// BR x BR inverse-and-multiply reduces to that automatically for BR=1.
    ///
    /// Built ONCE from a compressed <see cref="floatBSR"/> (an O(nb * BR^3) one-time cost via
    /// LU decomposition on each tiny diagonal block -- reuses <see cref="LU.decompInPlace(ref floatMxN, ref Pivot)"/>
    /// / <see cref="LU.decompSolve(ref floatMxN, in Pivot, ref floatN)"/>, no new inverse
    /// primitive), then <see cref="Apply"/> is a zero-alloc block-diagonal matvec every PCG
    /// iteration.
    ///
    /// Readonly: nothing mutates after the constructor fills DInv (it is never grown/resized).
    /// The two IfloatLinearOperator wrappers (floatBSROperator/floatDenseOperator) are
    /// already readonly structs, so passing them through `in TOp` in the generic solvers makes
    /// no defensive copy; being non-readonly here would force the compiler to snapshot-copy the
    /// whole preconditioner (the DInv UnsafeList header, at least) on every `M.Apply(in r, ref
    /// z)` call inside pcg -- undermining the zero-cost-dispatch claim. See floatBSROperator's
    /// own doc comment for the same reasoning.
    /// </summary>
    public readonly partial struct floatBlockJacobi : IfloatPreconditioner, IDisposable
    {
        public readonly int BlockRows;  // nb: number of diagonal blocks (== BlockCols of the source BSR)
        public readonly int BR;         // block dimension (== BC of the source BSR)

        public int Rows => BlockRows * BR;

        // Arena-tracked path: a stable pointer into the arena's
        // ChunkedRecordTable<floatBlockJacobiRecord> (docs/dev/rfc-memory-model.md §4 Option A). null
        // for a standalone (non-arena) preconditioner, in which case DInv resolves to the inline
        // field below instead. Replaces the old `Arena _arena` handle field -- same size trade as
        // floatBSR/floatN. Readonly (this struct is `readonly partial struct`): assigned once per
        // constructor, never reassigned afterward -- see Dispose()'s comment for what that costs.
        [NativeDisableUnsafePtrRestriction] private readonly unsafe floatBlockJacobiRecord* _rec;

        // Standalone-path backing store -- stays default(UnsafeList<float>) whenever _rec != null.
        private readonly UnsafeList<float> _inlineDInv;

        /// <summary>Inverted diagonal blocks, flat row-major per block: DInv[i*BR*BR + r*BR + c]
        /// holds (A_ii⁻¹)[r,c]. Length nb*BR*BR. Dual-mode, mirrors floatBSR.RowPtr/ColInd/Values:
        /// get-only (no setter is possible on a readonly struct) -- both constructors below write
        /// the computed inverse directly to whichever backing field is live instead of going
        /// through a property setter.</summary>
        public unsafe UnsafeList<float> DInv
        {
            get
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                AssertRecordAlive();
#endif
                return _rec != null ? _rec->DInv : _inlineDInv;
            }
        }

#if ENABLE_UNITY_COLLECTIONS_CHECKS
        // Editor/test-only guard (ENABLE_UNITY_COLLECTIONS_CHECKS is defined automatically by the
        // Unity Editor, including every test run, and compiles out of player builds entirely --
        // struct size is identical in both configs either way, since this adds no field).
        // floatBlockJacobi has no spare bits to pack a
        // generation stamp into (40B = 4 BlockRows + 4 BR + 8 _rec + 24 UnsafeList<float>, exactly
        // -- see docs/dev/rfc-memory-model.md §6.2 and ArenaLayoutTests.SparseStructsAreExpectedSize),
        // so this only checks Alive: it catches a read after Dispose() on THIS record, but not a
        // stale handle into a slot that has since been recycled by a fresh Allocate() (that needs a
        // generation stamp, which floatBSR itself carries in its own padding hole). Readonly
        // struct: this method doesn't (and, being readonly, couldn't) mutate any field.
        //
        // Uses ChunkedRecordTable's IsAliveFast(TRecord*) -- a direct pointer cast, no index, no
        // chunk-scan lookup -- rather than the index-based IsAlive(int) (i.e. NOT _rec->Table->
        // IsAlive(_rec->SelfIndex)): Apply() reads DInv every PCG iteration (i.e. per element in
        // the sense that matters here), so the index-based path's chunk scan would be a real
        // per-call cost. See IsAliveFast's own doc comment (ChunkedRecordTable.cs) for the
        // container-of rationale.
        private unsafe void AssertRecordAlive()
        {
            if (_rec != null && !ChunkedRecordTable<floatBlockJacobiRecord>.IsAliveFast(_rec))
                throw new InvalidOperationException("floatBlockJacobi.DInv: use of disposed/cleared arena allocation");
        }
#endif

        /// <summary>
        /// Builds the preconditioner from A's diagonal blocks. A must be square
        /// (BlockRows==BlockCols, BR==BC). Throws ArgumentException if a diagonal block is
        /// missing from the stored pattern or is singular.
        /// </summary>
        public unsafe floatBlockJacobi(in floatBSR A, Allocator allocator)
        {
            _rec = null;
            _inlineDInv = default;

            if (A.BlockRows != A.BlockCols || A.BR != A.BC)
                throw new ArgumentException("floatBlockJacobi: A must be square (BlockRows==BlockCols, BR==BC)");

            BlockRows = A.BlockRows;
            BR = A.BR;

            int blockLen = BR * BR;
            var dinv = new UnsafeList<float>(BlockRows * blockLen, allocator, NativeArrayOptions.ClearMemory);
            dinv.Resize(BlockRows * blockLen, NativeArrayOptions.ClearMemory);

            for (int i = 0; i < BlockRows; i++)
            {
                // Blocks within a block-row are stored in ascending ColInd (BSR invariant) --
                // scan forward and stop as soon as we pass column i.
                int s = A.RowPtr[i], e = A.RowPtr[i + 1];
                int found = -1;
                for (int k = s; k < e; k++)
                {
                    int blockCol = A.ColInd[k];
                    if (blockCol == i) { found = k; break; }
                    if (blockCol > i) break;
                }

                if (found < 0)
                {
                    dinv.Dispose();
                    throw new ArgumentException("floatBlockJacobi: missing diagonal block in A");
                }

                // Copy the diagonal block into scratch (LU factorization is destructive).
                var Dcopy = new floatMxN(BR, BR, Allocator.Temp, true);
                int srcOff = found * blockLen;
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Dcopy[r, c] = A.Values[srcOff + r * BR + c];

                var P = new Pivot(BR, Allocator.Temp);
                bool ok = LU.decompInPlace(ref Dcopy, ref P);

                if (!ok)
                {
                    P.Dispose();
                    Dcopy.Dispose();
                    dinv.Dispose();
                    throw new ArgumentException("floatBlockJacobi: diagonal block is singular");
                }

                // Column-by-column solve against unit vectors -> the explicit BR x BR inverse.
                int dstOff = i * blockLen;
                var col = new floatN(BR, Allocator.Temp, true);
                for (int c = 0; c < BR; c++)
                {
                    for (int r = 0; r < BR; r++)
                        col[r] = (r == c) ? (float)1 : (float)0;

                    LU.decompSolve(ref Dcopy, in P, ref col);

                    for (int r = 0; r < BR; r++)
                        dinv[dstOff + r * BR + c] = col[r];
                }

                col.Dispose();
                P.Dispose();
                Dcopy.Dispose();
            }

            _inlineDInv = dinv;
        }

        /// <summary>
        /// Arena-tracked constructor. <paramref name="rec"/> is a slot already carved from the
        /// arena's record table by the caller (Arena.floatBlockJacobi) -- same pre-allocated-
        /// record contract as floatBSR's arena-tracked ctor. Chains into the Allocator ctor above
        /// to reuse its diagonal-block LU-inversion loop unchanged (readonly fields may be
        /// assigned in ANY instance constructor of the declaring type, including one reached via
        /// `: this(...)` -- so re-assigning `_rec`/adopting the computed list here, after the
        /// chained-to ctor already ran, is legal), then adopts the freshly-built list into the
        /// record instead of leaving it on the standalone path.
        /// </summary>
        internal unsafe floatBlockJacobi(in floatBSR A, floatBlockJacobiRecord* rec, Allocator allocator) : this(in A, allocator)
        {
            rec->DInv = _inlineDInv;
            _inlineDInv = default;
            _rec = rec;
        }

        /// <summary>
        /// z = M⁻¹ r, applied block-wise: z_i = A_ii⁻¹ · r_i. z must not alias r (each z_i read
        /// draws on the full r_i block; overwriting r in place mid-block would corrupt later
        /// rows of the same block's product). Runs every PCG/LOBPCG iteration, so b in
        /// {1,2,3,4,6} (the same square sizes the spMV kernels specialize) dispatches to a fully
        /// unrolled dense b x b matvec (<see cref="UnsafeOP.blockJacobiApplyB1"/>..B6, mirroring
        /// <c>bsrMatVecB{b}</c>'s unroll -- Krylov R2, docs/draft-spec-krylov-optimization.md) --
        /// bit-identical to the general loop below (same left-to-right term order, just named
        /// locals instead of a runtime-trip-count inner loop Burst can't unroll). Any other BR
        /// falls through to the general runtime-BR loop, unchanged.
        /// </summary>
        public unsafe void Apply(in floatN r, ref floatN z)
        {
            int n = Rows;

            if (r.N != n)
                throw new ArgumentException("floatBlockJacobi.Apply: r.N must equal Rows");
            if (z.N != n)
                throw new ArgumentException("floatBlockJacobi.Apply: z.N must equal Rows");

            if (z.Data.Ptr == r.Data.Ptr)
                throw new ArgumentException("floatBlockJacobi.Apply: z must not alias r");

            float* rp = r.Data.Ptr;
            float* zp = z.Data.Ptr;
            float* dp = DInv.Ptr;

            switch (BR)
            {
                case 1: UnsafeOP.blockJacobiApplyB1(dp, rp, zp, BlockRows); return;
                case 2: UnsafeOP.blockJacobiApplyB2(dp, rp, zp, BlockRows); return;
                case 3: UnsafeOP.blockJacobiApplyB3(dp, rp, zp, BlockRows); return;
                case 4: UnsafeOP.blockJacobiApplyB4(dp, rp, zp, BlockRows); return;
                case 6: UnsafeOP.blockJacobiApplyB6(dp, rp, zp, BlockRows); return;
            }

            int blockLen = BR * BR;

            for (int i = 0; i < BlockRows; i++)
            {
                int rowBase = i * BR;
                int blockOff = i * blockLen;

                for (int lr = 0; lr < BR; lr++)
                {
                    float sum = 0;
                    for (int lc = 0; lc < BR; lc++)
                        sum += dp[blockOff + lr * BR + lc] * rp[rowBase + lc];
                    zp[rowBase + lr] = sum;
                }
            }
        }

        /// <summary>
        /// Disposes the DInv buffer. Note: unlike floatN/floatBSR's mutable-struct Dispose(),
        /// this CANNOT null <c>_rec</c> afterward (the struct is `readonly`, so no instance method
        /// may reassign a field, not even its own). Consequence: an ALIASED double-dispose (a
        /// different struct copy sharing this SAME record) still throws here, from the table's own
        /// double-Free guard, exactly like floatN/floatBSR -- but a SAME-COPY double-dispose
        /// (calling Dispose() twice on the identical variable) also throws here instead of
        /// degrading to a safe no-op the second time, since `_rec` is still non-null on the second
        /// call. This is a strictly-no-worse-than-before tradeoff: the pre-migration Dispose() had
        /// no double-dispose protection at all (silent double-free UB); the standalone
        /// (non-arena) path is unchanged either way, for the same readonly-field reason.
        /// </summary>
        public unsafe void Dispose()
        {
            // LINALG_DEBUG NaN-poison-on-dispose removed (2026-07-05): the symbol was defined
            // nowhere in the project, so that block was dead code that had never executed.
            // Superseded by the record table's own unconditional guard below -- a double-dispose
            // throws deterministically via Free()'s double-Free check, in every build config, not
            // just a debug one -- plus the ENABLE_UNITY_COLLECTIONS_CHECKS generational overlay on
            // DInv, which catches a stale read (use-after-dispose/Clear) instead of returning
            // garbage.
            if (_rec != null)
            {
                var dinv = _rec->DInv;
                _rec->Table->Free(_rec->SelfIndex);
                dinv.Dispose();
            }
            else
            {
                _inlineDInv.Dispose();
            }
        }
    }
}
