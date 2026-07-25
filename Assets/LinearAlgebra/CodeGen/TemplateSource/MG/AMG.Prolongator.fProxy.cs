using System;
using Unity.Collections;
using Unity.Mathematics;
using BULA.Sparse;

namespace BULA.Sparse
{
    public static partial class AMG
    {
        /// <summary>
        /// Tentative (unsmoothed) prolongator T from an aggregation, carrying the near-nullspace B
        /// (n x m, row-major, n = A.M_Rows) exactly onto the coarse grid: for each aggregate the
        /// local rows of B are orthonormalized by modified Gram–Schmidt into Q (T's block column)
        /// and the m x m coefficient R becomes that coarse node's near-nullspace block, so that
        /// T · Bcoarse == B to working precision (the defining tentative-prolongator identity).
        ///
        /// T is a BSR with A.BR x m blocks, BlockRows = A.BlockRows, BlockCols = numAgg, exactly one
        /// stored block per fine block-row (column aggId[i]). Bcoarse is the (numAgg*m) x m coarse
        /// near-nullspace (block-stacked R factors). A rank-deficient aggregate column (norm collapses
        /// under MGS, e.g. more near-nullspace modes than the aggregate can represent) drops to a zero
        /// Q column with R = 0 -- an honest loss of that coarse dof rather than a divide-by-zero.
        /// Deterministic: ascending aggregate and member order, fixed MGS loop order.
        /// aggId.N must equal A.BlockRows and B.M_Rows must equal A.M_Rows. Allocates <paramref
        /// name="Bcoarse"/> and the returned T from <paramref name="allocator"/> (default Temp;
        /// caller owns disposal); all other assembly scratch is Allocator.Temp.
        /// </summary>
        public static fProxyBSR tentativeProlongator(in fProxyBSR A, in Indices aggId, int numAgg,
            in fProxyMxN B, out fProxyMxN Bcoarse, Allocator allocator = Allocator.Temp)
        {
            int nb = A.BlockRows;
            int BR = A.BR;
            int m = B.N_Cols;

            if (aggId.N != nb)
                throw new ArgumentException("AMG.tentativeProlongator: aggId.N must equal A.BlockRows");
            if (B.M_Rows != nb * BR)
                throw new ArgumentException("AMG.tentativeProlongator: B.M_Rows must equal A.M_Rows");
            if (numAgg < 1)
                throw new ArgumentException("AMG.tentativeProlongator: numAgg must be >= 1");

            // Group block-rows by aggregate (counting sort); members stay ascending within each
            // aggregate because block-rows are scanned in ascending order.
            var aggPtr = new NativeArray<int>(numAgg + 1, Allocator.Temp, NativeArrayOptions.ClearMemory);
            for (int i = 0; i < nb; i++) aggPtr[aggId[i] + 1]++;
            for (int a = 0; a < numAgg; a++) aggPtr[a + 1] += aggPtr[a];
            var aggMembers = new NativeArray<int>(nb, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var cursor = new NativeArray<int>(numAgg, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            for (int a = 0; a < numAgg; a++) cursor[a] = aggPtr[a];
            for (int i = 0; i < nb; i++) { int a = aggId[i]; aggMembers[cursor[a]] = i; cursor[a]++; }
            cursor.Dispose();

            Bcoarse = new fProxyMxN(numAgg * m, m, allocator);   // fresh alloc clears (uninit=false default)

            var builder = new fProxyBSRBuilder(nb, numAgg, BR, m, Allocator.Temp, math.max(1, nb));

            fProxy relTol = Consts.fProxySqrtEps;

            for (int a = 0; a < numAgg; a++)
            {
                int s = aggPtr[a], e = aggPtr[a + 1];
                int k = e - s;                 // block-rows in this aggregate
                int rows = k * BR;

                // Local near-nullspace L = B restricted to this aggregate's rows (rows x m).
                var L = new fProxyMxN(rows, m, Allocator.Temp, true);
                for (int l = 0; l < k; l++)
                {
                    int gi = aggMembers[s + l];
                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < m; c++)
                            L[l * BR + r, c] = B[gi * BR + r, c];
                }

                // uninit=false: R is UPPER-triangular, its lower triangle is never written but is
                // copied into Bcoarse, so it must start zeroed (the ctor's bool is `uninit`, not
                // `clear`).
                var R = new fProxyMxN(m, m, Allocator.Temp, false);

                // Modified Gram–Schmidt: L (in place) -> Q (orthonormal columns), R upper m x m,
                // L_original = Q R.
                for (int j = 0; j < m; j++)
                {
                    fProxy initNorm = 0;
                    for (int row = 0; row < rows; row++) initNorm += L[row, j] * L[row, j];
                    initNorm = math.sqrt(initNorm);

                    for (int i = 0; i < j; i++)
                    {
                        fProxy dot = 0;
                        for (int row = 0; row < rows; row++) dot += L[row, i] * L[row, j];
                        R[i, j] = dot;
                        for (int row = 0; row < rows; row++) L[row, j] -= dot * L[row, i];
                    }

                    fProxy nrm = 0;
                    for (int row = 0; row < rows; row++) nrm += L[row, j] * L[row, j];
                    nrm = math.sqrt(nrm);

                    // Relative gate against the column's own initial norm, plus an absolute floor so
                    // a machine-tiny initNorm (whose relTol*initNorm underflows to ~0) cannot let an
                    // FP-noise residual survive as a spurious coarse dof.
                    if (nrm > relTol * initNorm && nrm > Consts.fProxyEpsilon)
                    {
                        R[j, j] = nrm;
                        fProxy inv = (fProxy)1 / nrm;
                        for (int row = 0; row < rows; row++) L[row, j] *= inv;
                    }
                    else
                    {
                        R[j, j] = 0;
                        for (int row = 0; row < rows; row++) L[row, j] = (fProxy)0;
                    }
                }

                // Scatter Q into T (one BR x m block per member row) and R into the coarse
                // near-nullspace block for aggregate a.
                var blk = new fProxyMxN(BR, m, Allocator.Temp, true);
                for (int l = 0; l < k; l++)
                {
                    int gi = aggMembers[s + l];
                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < m; c++)
                            blk[r, c] = L[l * BR + r, c];
                    builder.AddBlock(gi, a, in blk);
                }
                blk.Dispose();

                for (int r = 0; r < m; r++)
                    for (int c = 0; c < m; c++)
                        Bcoarse[a * m + r, c] = R[r, c];

                R.Dispose();
                L.Dispose();
            }

            aggMembers.Dispose();
            aggPtr.Dispose();

            var result = builder.ToBSR(allocator);
            builder.Dispose();
            return result;
        }

        /// <summary>
        /// Tentative prolongator for the scalar default near-nullspace B = 1 (m = 1): each aggregate's
        /// block-column is the normalized constant vector. Bcoarse is numAgg x 1. Allocates via
        /// <paramref name="allocator"/> (default Temp).
        /// </summary>
        public static fProxyBSR tentativeProlongator(in fProxyBSR A, in Indices aggId, int numAgg,
            out fProxyMxN Bcoarse, Allocator allocator = Allocator.Temp)
        {
            int n = A.M_Rows;
            var ones = new fProxyMxN(n, 1, Allocator.Temp);
            for (int i = 0; i < n; i++) ones[i, 0] = (fProxy)1;
            var result = tentativeProlongator(in A, in aggId, numAgg, in ones, out Bcoarse, allocator);
            ones.Dispose();
            return result;
        }
    }
}
