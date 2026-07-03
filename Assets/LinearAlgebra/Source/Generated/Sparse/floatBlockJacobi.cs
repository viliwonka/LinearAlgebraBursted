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
    /// Built ONCE from a compressed <see cref="floatBSR"/> (an O(nb * BR^3) one-time cost via
    /// LU decomposition on each tiny diagonal block -- reuses <see cref="LU.luDecompositionInPlace"/>
    /// / <see cref="LU.luSolve(ref floatMxN, in Pivot, ref floatN)"/>, no new inverse
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

        /// <summary>Inverted diagonal blocks, flat row-major per block: DInv[i*BR*BR + r*BR + c]
        /// holds (A_ii⁻¹)[r,c]. Length nb*BR*BR.</summary>
        public readonly UnsafeList<float> DInv;

        // Value handle to the shared ArenaCore, not a raw pointer (see Arena.cs); copies stay live (FM2).
        private readonly Arena _arena;

        /// <summary>
        /// Builds the preconditioner from A's diagonal blocks. A must be square
        /// (BlockRows==BlockCols, BR==BC). Throws ArgumentException if a diagonal block is
        /// missing from the stored pattern or is singular.
        /// </summary>
        public unsafe floatBlockJacobi(in floatBSR A, Allocator allocator)
        {
            _arena = default;

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
                bool ok = LU.luDecompositionInPlace(ref Dcopy, ref P);

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

                    LU.luSolve(ref Dcopy, in P, ref col);

                    for (int r = 0; r < BR; r++)
                        dinv[dstOff + r * BR + c] = col[r];
                }

                col.Dispose();
                P.Dispose();
                Dcopy.Dispose();
            }

            DInv = dinv;
        }

        /// <summary>
        /// Same construction, tracked by an arena (disposed with the arena). Takes the arena by
        /// `in`: it only reads arena.Allocator and stores the (currently unused, future-
        /// proofing) `_arena` handle -- a plain value copy, safe regardless of `in`/`ref` now
        /// that Arena is a thin copyable handle to a heap-allocated ArenaCore (see Arena.cs).
        /// The arena-OWNING factory that actually registers this instance for disposal is
        /// `Arena.floatBlockJacobi(in floatBSR)`.
        /// </summary>
        public unsafe floatBlockJacobi(in floatBSR A, in Arena arena) : this(in A, arena.Allocator)
        {
            _arena = arena;
        }

        /// <summary>
        /// z = M⁻¹ r, applied block-wise: z_i = A_ii⁻¹ · r_i. z must not alias r (each z_i read
        /// draws on the full r_i block; overwriting r in place mid-block would corrupt later
        /// rows of the same block's product).
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

        public void Dispose()
        {
#if LINALG_DEBUG
            // poison the buffer so a read-after-dispose surfaces as NaN instead of stale data
            for (int i = 0; i < DInv.Length; i++) DInv[i] = float.NaN;
#endif
            DInv.Dispose();
        }
    }
}
