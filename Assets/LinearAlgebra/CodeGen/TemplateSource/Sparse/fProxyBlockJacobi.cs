using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Block-Jacobi preconditioner for a square BSR (<c>BlockRows==BlockCols</c>, <c>BR==BC</c>):
    /// z = M⁻¹ r where M = blockdiag(A_00, A_11, ..., A_{nb-1,nb-1}), i.e. each diagonal block
    /// inverted independently and applied block-wise. The <c>BR==1</c> case degenerates to
    /// point-Jacobi (z_i = r_i / A_ii) -- no special-cased code path is needed, the general
    /// BR x BR inverse-and-multiply reduces to that automatically for BR=1.
    ///
    /// Built ONCE from a compressed <see cref="fProxyBSR"/> (an O(nb * BR^3) one-time cost via
    /// LU decomposition on each tiny diagonal block -- reuses <see cref="LU.luDecompositionInPlace"/>
    /// / <see cref="LU.luSolve(ref fProxyMxN, in Pivot, ref fProxyN)"/>, no new inverse
    /// primitive), then <see cref="Apply"/> is a zero-alloc block-diagonal matvec every PCG
    /// iteration.
    ///
    /// Readonly: nothing mutates after the constructor fills DInv (it is never grown/resized).
    /// The two IfProxyLinearOperator wrappers (fProxyBSROperator/fProxyDenseOperator) are
    /// already readonly structs, so passing them through `in TOp` in the generic solvers makes
    /// no defensive copy; being non-readonly here would force the compiler to snapshot-copy the
    /// whole preconditioner (the DInv UnsafeList header, at least) on every `M.Apply(in r, ref
    /// z)` call inside pcg -- undermining the zero-cost-dispatch claim. See fProxyBSROperator's
    /// own doc comment for the same reasoning.
    /// </summary>
    public readonly partial struct fProxyBlockJacobi : IfProxyPreconditioner, IDisposable
    {
        public readonly int BlockRows;  // nb: number of diagonal blocks (== BlockCols of the source BSR)
        public readonly int BR;         // block dimension (== BC of the source BSR)

        public int Rows => BlockRows * BR;

        // Arena-tracked path: a stable pointer into the arena's
        // ChunkedRecordTable<fProxyBlockJacobiRecord> (docs/rfc-memory-model.md §4 Option A). null
        // for a standalone (non-arena) preconditioner, in which case DInv resolves to the inline
        // field below instead. Replaces the old `Arena _arena` handle field -- same size trade as
        // fProxyBSR/fProxyN. Readonly (this struct is `readonly partial struct`): assigned once per
        // constructor, never reassigned afterward -- see Dispose()'s comment for what that costs.
        [NativeDisableUnsafePtrRestriction] private readonly unsafe fProxyBlockJacobiRecord* _rec;

        // Standalone-path backing store -- stays default(UnsafeList<fProxy>) whenever _rec != null.
        private readonly UnsafeList<fProxy> _inlineDInv;

        /// <summary>Inverted diagonal blocks, flat row-major per block: DInv[i*BR*BR + r*BR + c]
        /// holds (A_ii⁻¹)[r,c]. Length nb*BR*BR. Dual-mode, mirrors fProxyBSR.RowPtr/ColInd/Values:
        /// get-only (no setter is possible on a readonly struct) -- both constructors below write
        /// the computed inverse directly to whichever backing field is live instead of going
        /// through a property setter.</summary>
        public unsafe UnsafeList<fProxy> DInv => _rec != null ? _rec->DInv : _inlineDInv;

        /// <summary>
        /// Builds the preconditioner from A's diagonal blocks. A must be square
        /// (BlockRows==BlockCols, BR==BC). Throws ArgumentException if a diagonal block is
        /// missing from the stored pattern or is singular.
        /// </summary>
        public unsafe fProxyBlockJacobi(in fProxyBSR A, Allocator allocator)
        {
            _rec = null;
            _inlineDInv = default;

            if (A.BlockRows != A.BlockCols || A.BR != A.BC)
                throw new ArgumentException("fProxyBlockJacobi: A must be square (BlockRows==BlockCols, BR==BC)");

            BlockRows = A.BlockRows;
            BR = A.BR;

            int blockLen = BR * BR;
            var dinv = new UnsafeList<fProxy>(BlockRows * blockLen, allocator, NativeArrayOptions.ClearMemory);
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
                    throw new ArgumentException("fProxyBlockJacobi: missing diagonal block in A");
                }

                // Copy the diagonal block into scratch (LU factorization is destructive).
                var Dcopy = new fProxyMxN(BR, BR, Allocator.Temp, true);
                int srcOff = found * blockLen;
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Dcopy[r, c] = A.Values[srcOff + r * BR + c];

                var P = new Pivot(BR, Allocator.Temp);
                bool ok = LU.luDecompositionInPlace(ref Dcopy, ref P);

                if (!ok)
                {
                    P.Dispose();
                    Dcopy.Dispose();
                    dinv.Dispose();
                    throw new ArgumentException("fProxyBlockJacobi: diagonal block is singular");
                }

                // Column-by-column solve against unit vectors -> the explicit BR x BR inverse.
                int dstOff = i * blockLen;
                var col = new fProxyN(BR, Allocator.Temp, true);
                for (int c = 0; c < BR; c++)
                {
                    for (int r = 0; r < BR; r++)
                        col[r] = (r == c) ? (fProxy)1 : (fProxy)0;

                    LU.luSolve(ref Dcopy, in P, ref col);

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
        /// arena's record table by the caller (Arena.fProxyBlockJacobi) -- same pre-allocated-
        /// record contract as fProxyBSR's arena-tracked ctor. Chains into the Allocator ctor above
        /// to reuse its diagonal-block LU-inversion loop unchanged (readonly fields may be
        /// assigned in ANY instance constructor of the declaring type, including one reached via
        /// `: this(...)` -- so re-assigning `_rec`/adopting the computed list here, after the
        /// chained-to ctor already ran, is legal), then adopts the freshly-built list into the
        /// record instead of leaving it on the standalone path.
        /// </summary>
        internal unsafe fProxyBlockJacobi(in fProxyBSR A, fProxyBlockJacobiRecord* rec, Allocator allocator) : this(in A, allocator)
        {
            rec->DInv = _inlineDInv;
            _inlineDInv = default;
            _rec = rec;
        }

        /// <summary>
        /// z = M⁻¹ r, applied block-wise: z_i = A_ii⁻¹ · r_i. z must not alias r (each z_i read
        /// draws on the full r_i block; overwriting r in place mid-block would corrupt later
        /// rows of the same block's product).
        /// </summary>
        public unsafe void Apply(in fProxyN r, ref fProxyN z)
        {
            int n = Rows;

            if (r.N != n)
                throw new ArgumentException("fProxyBlockJacobi.Apply: r.N must equal Rows");
            if (z.N != n)
                throw new ArgumentException("fProxyBlockJacobi.Apply: z.N must equal Rows");

            if (z.Data.Ptr == r.Data.Ptr)
                throw new ArgumentException("fProxyBlockJacobi.Apply: z must not alias r");

            fProxy* rp = r.Data.Ptr;
            fProxy* zp = z.Data.Ptr;
            fProxy* dp = DInv.Ptr;

            int blockLen = BR * BR;

            for (int i = 0; i < BlockRows; i++)
            {
                int rowBase = i * BR;
                int blockOff = i * blockLen;

                for (int lr = 0; lr < BR; lr++)
                {
                    fProxy sum = 0;
                    for (int lc = 0; lc < BR; lc++)
                        sum += dp[blockOff + lr * BR + lc] * rp[rowBase + lc];
                    zp[rowBase + lr] = sum;
                }
            }
        }

        /// <summary>
        /// Disposes the DInv buffer. Note: unlike fProxyN/fProxyBSR's mutable-struct Dispose(),
        /// this CANNOT null <c>_rec</c> afterward (the struct is `readonly`, so no instance method
        /// may reassign a field, not even its own). Consequence: an ALIASED double-dispose (a
        /// different struct copy sharing this SAME record) still throws here, from the table's own
        /// double-Free guard, exactly like fProxyN/fProxyBSR -- but a SAME-COPY double-dispose
        /// (calling Dispose() twice on the identical variable) also throws here instead of
        /// degrading to a safe no-op the second time, since `_rec` is still non-null on the second
        /// call. This is a strictly-no-worse-than-before tradeoff: the pre-migration Dispose() had
        /// no double-dispose protection at all (silent double-free UB); the standalone
        /// (non-arena) path is unchanged either way, for the same readonly-field reason.
        /// </summary>
        public unsafe void Dispose()
        {
#if LINALG_DEBUG
            // poison the buffer so a read-after-dispose surfaces as NaN instead of stale data
            for (int i = 0; i < DInv.Length; i++) DInv[i] = fProxy.NaN;
#endif
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
