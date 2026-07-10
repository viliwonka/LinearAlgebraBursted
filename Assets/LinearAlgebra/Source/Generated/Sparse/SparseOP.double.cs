using System;
using System.Runtime.CompilerServices;
using Unity.Collections.LowLevel.Unsafe;
using LinearAlgebra;
using LinearAlgebra.Internal;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Sparse matvec kernels over doubleBSR (block-CSR). The shape mirrors Blas.dot's
    /// dense matVec overloads (in A, in x, ref y) on purpose -- a future generic
    /// IdoubleLinearOperator wrapper (Phase 2) can forward Apply/ApplyT straight to spMV/spMVT.
    /// </summary>
    public static partial class BSR
    {
        // ---- y = A * x ----

        // ref-dest primitive. Guard: y must not alias x (each x[k] feeds every block-row that
        // stores a block in column-block k).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void spMV(in doubleBSR A, in doubleN x, ref doubleN y)
        {
            Assume.SameDim(A.N_Cols, x.N);

            if (y.N != A.M_Rows)
                throw new ArgumentException("spMV: y.N must equal A.M_Rows");

            unsafe
            {
                if (y.Data.Ptr == x.Data.Ptr)
                    throw new ArgumentException("spMV: y must not alias x");

                // bsrMatVec accumulates (+=), so the destination must start zeroed.
                UnsafeUtility.MemClear(y.Data.Ptr, (long)y.Data.Length * UnsafeUtility.SizeOf<double>());

                int* rowPtr = A.RowPtr.Ptr;
                int* colInd = A.ColInd.Ptr;
                double* values = A.Values.Ptr;
                double* xPtr = x.Data.Ptr;
                double* yPtr = y.Data.Ptr;

                if (A.Symmetric)
                {
                    // Symmetric storage requires BR==BC by construction (doubleBSR ctor), so
                    // dispatching on BR alone is sufficient here.
                    switch (A.BR)
                    {
                        case 1: UnsafeOP.bsrMatVecSymB1(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 2: UnsafeOP.bsrMatVecSymB2(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 3: UnsafeOP.bsrMatVecSymB3(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 4: UnsafeOP.bsrMatVecSymB4(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 6: UnsafeOP.bsrMatVecSymB6(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        default: UnsafeOP.bsrMatVecSym(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows, A.BR); break;
                    }
                }
                else if (A.BR == A.BC)
                {
                    // Register-tile specializations only apply to square blocks -- rectangular
                    // BR != BC always falls through to the general kernel below.
                    switch (A.BR)
                    {
                        case 1: UnsafeOP.bsrMatVecB1(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 2: UnsafeOP.bsrMatVecB2(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 3: UnsafeOP.bsrMatVecB3(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 4: UnsafeOP.bsrMatVecB4(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 6: UnsafeOP.bsrMatVecB6(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        default: UnsafeOP.bsrMatVec(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows, A.BR, A.BC); break;
                    }
                }
                else
                {
                    UnsafeOP.bsrMatVec(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows, A.BR, A.BC);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN spMV(in doubleBSR A, in doubleN x)
        {
            doubleN result = x.doubleTempVec(A.M_Rows);
            spMV(in A, in x, ref result);
            return result;
        }

        // y = A x, PLUS dot(x, y) computed as part of the same call -- Krylov R2's ApplyDot
        // (docs/draft-spec-krylov-optimization.md; see doubleBSROperator.ApplyDot, the sole
        // caller). COMPOSES: a plain spMV, then one Blas.dot(x,y) pass (still the 2x-accumulator
        // vecDot kernel, just not folded into spMV). Non-square (Rows != Cols) can't pair x[i]
        // with y[i] at all -- Blas.dot below throws in that case, same as a caller doing Apply
        // then Blas.dot(x,y) by hand would get.
        //
        // MEASURED, not assumed: an earlier version of this method dispatched genuinely-fused
        // "Dot" kernels (bsrMatVecB1Dot..B6Dot) for full-storage square BSR at a specialized
        // block size, folding dot(x,y) into the same per-block-row pass that computes y. A/B'd at
        // the b=1 stencil section of LargeSparseBenchmark (the cleanest-signal section from
        // Round 1) against this compose form: CG at N=5120/float went from ~0.245ms (this
        // compose form, matching the pre-ApplyDot baseline) to ~0.359ms with the fused B1Dot
        // kernel -- a reproducible ~45% REGRESSION, not a win. Root cause: B1Dot's per-row
        // arithmetic is trivial (the b=1 stencil is a tridiagonal, ~3 stored blocks per row), so
        // the kernel's cost is dominated by its OUTER cross-row dot fold -- which, lacking a
        // contiguous 4-wide block to reinterpret as double4 (row results arrive one at a time),
        // used two alternating SCALAR accumulators instead. That scalar fold is far slower than
        // simply calling the already-tuned SIMD vecDot separately (2x double4, 8 lane-chains) --
        // exactly what composing does. Per the spec's own instruction ("try 2 accumulators,
        // measure, stop... if it doesn't measurably win, keep the original"): reverted to compose
        // for every case rather than ship a kernel that loses on its own designed-to-be-clearest
        // benchmark. The fused kernels were deleted, not merely unused, to avoid maintaining
        // known-worse code.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double spMVDot(in doubleBSR A, in doubleN x, ref doubleN y)
        {
            spMV(in A, in x, ref y);
            return Blas.dot(x, y);
        }

        /// <summary>
        /// Squared L2 norm of each column of the block-sparse A: d2[j] = Σ_i A[i,j]² = diag(AᵀA)[j],
        /// computed directly from the stored blocks in a single pass over the nonzeros (no AᵀA
        /// formed, no transpose-matvecs). Written into the caller's d2 (length A.N_Cols), no
        /// allocation. Feeds an AᵀA-Jacobi (column-equilibration) least-squares preconditioner
        /// (see <see cref="doubleColScaledOperator{TInner}"/> / <c>Blas.buildJacobiScale</c>).
        ///
        /// NOT supported for Symmetric (upper-block-triangle-only) storage: the implicit lower
        /// blocks are not materialized, so a single pass would under-count every column -- throws
        /// in that case. Jacobi-LS preconditioning targets rectangular / non-symmetric least
        /// squares, where Symmetric is false.
        /// </summary>
        public static void columnNormsSquared(in doubleBSR A, ref doubleN d2)
        {
            if (d2.N != A.N_Cols)
                throw new ArgumentException("columnNormsSquared: d2.N must equal A.N_Cols");
            if (A.Symmetric)
                throw new ArgumentException("columnNormsSquared: not supported for Symmetric (upper-block-only) storage -- the implicit lower blocks would be under-counted");

            int BR = A.BR, BC = A.BC;
            int blockSize = BR * BC;

            unsafe
            {
                int* rowPtr = A.RowPtr.Ptr;
                int* colInd = A.ColInd.Ptr;
                double* values = A.Values.Ptr;
                double* d2Ptr = d2.Data.Ptr;

                UnsafeUtility.MemClear(d2Ptr, (long)d2.Data.Length * UnsafeUtility.SizeOf<double>());

                for (int bi = 0; bi < A.BlockRows; bi++)
                {
                    for (int k = rowPtr[bi]; k < rowPtr[bi + 1]; k++)
                    {
                        int colBase = colInd[k] * BC;         // global column of block-interior col 0
                        double* block = values + (long)k * blockSize;
                        for (int r = 0; r < BR; r++)
                            for (int c = 0; c < BC; c++)
                            {
                                double v = block[r * BC + c]; // row-major block interior
                                d2Ptr[colBase + c] += v * v;
                            }
                    }
                }
            }
        }

        // ---- y = A^T * x ----

        // ref-dest primitive. Guard: y must not alias x (each x[k] feeds every block-column
        // that stores a block in block-row k).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void spMVT(in doubleBSR A, in doubleN x, ref doubleN y)
        {
            if (A.Symmetric)
            {
                // A == A^T for symmetric upper-block storage -- forward straight to spMV. Its guards
                // (Assume.SameDim(A.N_Cols, x.N), y.N != A.M_Rows) are equivalent to spMVT's own
                // (Assume.SameDim(A.M_Rows, x.N), y.N != A.N_Cols) here because Symmetric implies
                // A.M_Rows == A.N_Cols.
                spMV(in A, in x, ref y);
                return;
            }

            Assume.SameDim(A.M_Rows, x.N);

            if (y.N != A.N_Cols)
                throw new ArgumentException("spMVT: y.N must equal A.N_Cols");

            unsafe
            {
                if (y.Data.Ptr == x.Data.Ptr)
                    throw new ArgumentException("spMVT: y must not alias x");

                // bsrMatVecT accumulates (+=), so the destination must start zeroed.
                UnsafeUtility.MemClear(y.Data.Ptr, (long)y.Data.Length * UnsafeUtility.SizeOf<double>());

                int* rowPtr = A.RowPtr.Ptr;
                int* colInd = A.ColInd.Ptr;
                double* values = A.Values.Ptr;
                double* xPtr = x.Data.Ptr;
                double* yPtr = y.Data.Ptr;

                if (A.BR == A.BC)
                {
                    // Register-tile specializations only apply to square blocks -- rectangular
                    // BR != BC always falls through to the general kernel below.
                    switch (A.BR)
                    {
                        case 1: UnsafeOP.bsrMatVecTB1(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 2: UnsafeOP.bsrMatVecTB2(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 3: UnsafeOP.bsrMatVecTB3(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 4: UnsafeOP.bsrMatVecTB4(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        case 6: UnsafeOP.bsrMatVecTB6(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows); break;
                        default: UnsafeOP.bsrMatVecT(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows, A.BR, A.BC); break;
                    }
                }
                else
                {
                    UnsafeOP.bsrMatVecT(rowPtr, colInd, values, xPtr, yPtr, A.BlockRows, A.BR, A.BC);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static doubleN spMVT(in doubleBSR A, in doubleN x)
        {
            doubleN result = x.doubleTempVec(A.N_Cols);
            spMVT(in A, in x, ref result);
            return result;
        }

        // ---- block triangular sweeps (Krylov R3, docs/draft-spec-krylov-optimization.md) ----

        /// <summary>
        /// Block forward substitution: solves (D/diagScale + L) y = r, where D is A's block
        /// diagonal (inverted via <paramref name="Jacobi"/>, reused unchanged -- no separate
        /// inversion here) and L is A's STRICTLY LOWER stored blocks (ColInd &lt; block-row).
        /// Rows solved in ascending order; each row's off-diagonal contribution is a b x b matvec
        /// against already-solved earlier rows, the diagonal solve is Jacobi's explicit block
        /// inverse (no per-row factorization). FULL-storage BSR only (Q4 ruling) -- throws on
        /// Symmetric (upper-block-triangle-only) storage; mirror it first via
        /// <see cref="Arena.doubleBSRMirrorToFull"/>. diagScale=1 is the plain (unscaled)
        /// Gauss-Seidel forward sweep; <see cref="doubleSSOR"/> drives this with diagScale=Omega.
        /// y must not alias r (same "read the full row before any row's write" reasoning as
        /// <see cref="doubleBlockJacobi.Apply"/>).
        /// </summary>
        public static void sweepLower(in doubleBSR A, in doubleBlockJacobi Jacobi, double diagScale, in doubleN r, ref doubleN y)
        {
            if (A.BlockRows != A.BlockCols || A.BR != A.BC)
                throw new ArgumentException("sweepLower: A must be square (BlockRows==BlockCols, BR==BC)");
            if (A.Symmetric)
                throw new ArgumentException("sweepLower: A must be full-storage BSR (Symmetric upper-block-triangle storage is not supported here -- mirror it first via Arena.doubleBSRMirrorToFull)");
            if (Jacobi.BlockRows != A.BlockRows || Jacobi.BR != A.BR)
                throw new ArgumentException("sweepLower: Jacobi must be built from A (BlockRows/BR mismatch)");
            if (r.N != A.M_Rows)
                throw new ArgumentException("sweepLower: r.N must equal A.M_Rows");
            if (y.N != A.M_Rows)
                throw new ArgumentException("sweepLower: y.N must equal A.M_Rows");

            unsafe
            {
                if (y.Data.Ptr == r.Data.Ptr)
                    throw new ArgumentException("sweepLower: y must not alias r");

                int* rowPtr = A.RowPtr.Ptr;
                int* colInd = A.ColInd.Ptr;
                double* values = A.Values.Ptr;
                double* dInv = Jacobi.DInv.Ptr;
                double* rPtr = r.Data.Ptr;
                double* yPtr = y.Data.Ptr;

                switch (A.BR)
                {
                    case 1: UnsafeOP.sweepLowerB1(rowPtr, colInd, values, dInv, diagScale, rPtr, yPtr, A.BlockRows); break;
                    case 2: UnsafeOP.sweepLowerB2(rowPtr, colInd, values, dInv, diagScale, rPtr, yPtr, A.BlockRows); break;
                    case 3: UnsafeOP.sweepLowerB3(rowPtr, colInd, values, dInv, diagScale, rPtr, yPtr, A.BlockRows); break;
                    case 4: UnsafeOP.sweepLowerB4(rowPtr, colInd, values, dInv, diagScale, rPtr, yPtr, A.BlockRows); break;
                    case 6: UnsafeOP.sweepLowerB6(rowPtr, colInd, values, dInv, diagScale, rPtr, yPtr, A.BlockRows); break;
                    default: UnsafeOP.sweepLower(rowPtr, colInd, values, dInv, diagScale, rPtr, yPtr, A.BlockRows, A.BR); break;
                }
            }
        }

        /// <summary>sweepLower with diagScale=1 -- the plain (unscaled) Gauss-Seidel forward sweep: solves (D+L) y = r.</summary>
        public static void sweepLower(in doubleBSR A, in doubleBlockJacobi Jacobi, in doubleN r, ref doubleN y)
            => sweepLower(in A, in Jacobi, (double)1, in r, ref y);

        /// <summary>
        /// Block backward substitution: solves (D/diagScale + U) y = r, where U is A's STRICTLY
        /// UPPER stored blocks (ColInd &gt; block-row). Mirror of <see cref="sweepLower"/> --
        /// rows solved in DESCENDING order, same diagonal-inverse/full-storage/no-alias contract.
        /// </summary>
        public static void sweepUpper(in doubleBSR A, in doubleBlockJacobi Jacobi, double diagScale, in doubleN r, ref doubleN y)
        {
            if (A.BlockRows != A.BlockCols || A.BR != A.BC)
                throw new ArgumentException("sweepUpper: A must be square (BlockRows==BlockCols, BR==BC)");
            if (A.Symmetric)
                throw new ArgumentException("sweepUpper: A must be full-storage BSR (Symmetric upper-block-triangle storage is not supported here -- mirror it first via Arena.doubleBSRMirrorToFull)");
            if (Jacobi.BlockRows != A.BlockRows || Jacobi.BR != A.BR)
                throw new ArgumentException("sweepUpper: Jacobi must be built from A (BlockRows/BR mismatch)");
            if (r.N != A.M_Rows)
                throw new ArgumentException("sweepUpper: r.N must equal A.M_Rows");
            if (y.N != A.M_Rows)
                throw new ArgumentException("sweepUpper: y.N must equal A.M_Rows");

            unsafe
            {
                if (y.Data.Ptr == r.Data.Ptr)
                    throw new ArgumentException("sweepUpper: y must not alias r");

                int* rowPtr = A.RowPtr.Ptr;
                int* colInd = A.ColInd.Ptr;
                double* values = A.Values.Ptr;
                double* dInv = Jacobi.DInv.Ptr;
                double* rPtr = r.Data.Ptr;
                double* yPtr = y.Data.Ptr;

                switch (A.BR)
                {
                    case 1: UnsafeOP.sweepUpperB1(rowPtr, colInd, values, dInv, diagScale, rPtr, yPtr, A.BlockRows); break;
                    case 2: UnsafeOP.sweepUpperB2(rowPtr, colInd, values, dInv, diagScale, rPtr, yPtr, A.BlockRows); break;
                    case 3: UnsafeOP.sweepUpperB3(rowPtr, colInd, values, dInv, diagScale, rPtr, yPtr, A.BlockRows); break;
                    case 4: UnsafeOP.sweepUpperB4(rowPtr, colInd, values, dInv, diagScale, rPtr, yPtr, A.BlockRows); break;
                    case 6: UnsafeOP.sweepUpperB6(rowPtr, colInd, values, dInv, diagScale, rPtr, yPtr, A.BlockRows); break;
                    default: UnsafeOP.sweepUpper(rowPtr, colInd, values, dInv, diagScale, rPtr, yPtr, A.BlockRows, A.BR); break;
                }
            }
        }

        /// <summary>sweepUpper with diagScale=1 -- the plain (unscaled) Gauss-Seidel backward sweep: solves (D+U) y = r.</summary>
        public static void sweepUpper(in doubleBSR A, in doubleBlockJacobi Jacobi, in doubleN r, ref doubleN y)
            => sweepUpper(in A, in Jacobi, (double)1, in r, ref y);
    }
}
