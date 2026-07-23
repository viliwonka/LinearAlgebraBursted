using System;
using Unity.Collections;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra.Sparse
{
    public static partial class AMG
    {
        /// <summary>
        /// Unsmoothed Galerkin coarse operator A_c = Tᵀ A T for the tentative prolongator T (one
        /// A.BR x m block per fine block-row). Because each fine block-row maps to exactly one
        /// aggregate, the triple product collapses to a segmented scatter-add over A's stored blocks:
        /// each fine block A_ij contributes T[i]ᵀ A_ij T[j] (m x m) to coarse block
        /// (aggId[i], aggId[j]) — no general sparse matrix-matrix product needed. A_c is a full-storage
        /// numAgg x numAgg BSR with m x m blocks. Deterministic: fine blocks visited in ascending
        /// (row, col) order and summed in the builder's stable sorted order.
        ///
        /// A must be FULL storage with square blocks (BR == BC, BlockRows == BlockCols). T.BlockRows
        /// must equal A.BlockRows, T.BlockCols == numAgg, T.BR == A.BR. aggId.N == A.BlockRows.
        /// </summary>
        public static fProxyBSR galerkinRAP(in fProxyBSR A, in fProxyBSR T, in Indices aggId, int numAgg, ref Arena arena)
        {
            if (A.Symmetric)
                throw new ArgumentException("AMG.galerkinRAP: A must be full storage (mirror a Symmetric BSR first)");
            if (A.BR != A.BC || A.BlockRows != A.BlockCols)
                throw new ArgumentException("AMG.galerkinRAP: A must have square blocks on a square block grid");

            int nb = A.BlockRows;
            int BR = A.BR;
            int m = T.BC;

            if (T.BlockRows != nb) throw new ArgumentException("AMG.galerkinRAP: T.BlockRows must equal A.BlockRows");
            if (T.BlockCols != numAgg) throw new ArgumentException("AMG.galerkinRAP: T.BlockCols must equal numAgg");
            if (T.BR != BR) throw new ArgumentException("AMG.galerkinRAP: T.BR must equal A.BR");
            if (aggId.N != nb) throw new ArgumentException("AMG.galerkinRAP: aggId.N must equal A.BlockRows");

            int blockLenA = BR * BR;
            int blockLenT = BR * m;

            var builder = arena.fProxyBSRBuilder(numAgg, numAgg, m, m, math.max(1, A.Nnzb));
            var contrib = new fProxyMxN(m, m, Allocator.Temp, true);   // fully written each block

            for (int i = 0; i < nb; i++)
            {
                int ti = T.RowPtr[i] * blockLenT;      // T[i] block base
                int aI = aggId[i];

                int s = A.RowPtr[i], e = A.RowPtr[i + 1];
                for (int k = s; k < e; k++)
                {
                    int j = A.ColInd[k];
                    int tj = T.RowPtr[j] * blockLenT;   // T[j] block base
                    int aoff = k * blockLenA;

                    // contrib[p,q] = sum_r sum_c T[i][r,p] * A_ij[r,c] * T[j][c,q].
                    for (int p = 0; p < m; p++)
                        for (int q = 0; q < m; q++)
                        {
                            fProxy acc = 0;
                            for (int r = 0; r < BR; r++)
                            {
                                fProxy tip = T.Values[ti + r * m + p];
                                if (tip == (fProxy)0) continue;
                                fProxy inner = 0;
                                for (int c = 0; c < BR; c++)
                                    inner += A.Values[aoff + r * BR + c] * T.Values[tj + c * m + q];
                                acc += tip * inner;
                            }
                            contrib[p, q] = acc;
                        }

                    builder.AddBlock(aI, aggId[j], in contrib);
                }
            }

            contrib.Dispose();
            return builder.ToBSR(ref arena);
        }

        /// <summary>Standalone twin of <see cref="galerkinRAP(in fProxyBSR, in fProxyBSR, in Indices, int, ref Arena)"/>:
        /// allocates the coarse operator from <paramref name="allocator"/> instead of an arena;
        /// caller owns disposing the returned <see cref="fProxyBSR"/>. Internal assembly scratch is
        /// Temp and disposed before returning.</summary>
        public static fProxyBSR galerkinRAP(in fProxyBSR A, in fProxyBSR T, in Indices aggId, int numAgg, Allocator allocator = Allocator.Temp)
        {
            if (A.Symmetric)
                throw new ArgumentException("AMG.galerkinRAP: A must be full storage (mirror a Symmetric BSR first)");
            if (A.BR != A.BC || A.BlockRows != A.BlockCols)
                throw new ArgumentException("AMG.galerkinRAP: A must have square blocks on a square block grid");

            int nb = A.BlockRows;
            int BR = A.BR;
            int m = T.BC;

            if (T.BlockRows != nb) throw new ArgumentException("AMG.galerkinRAP: T.BlockRows must equal A.BlockRows");
            if (T.BlockCols != numAgg) throw new ArgumentException("AMG.galerkinRAP: T.BlockCols must equal numAgg");
            if (T.BR != BR) throw new ArgumentException("AMG.galerkinRAP: T.BR must equal A.BR");
            if (aggId.N != nb) throw new ArgumentException("AMG.galerkinRAP: aggId.N must equal A.BlockRows");

            int blockLenA = BR * BR;
            int blockLenT = BR * m;

            var builder = new fProxyBSRBuilder(numAgg, numAgg, m, m, Allocator.Temp, math.max(1, A.Nnzb));
            var contrib = new fProxyMxN(m, m, Allocator.Temp, true);   // fully written each block

            for (int i = 0; i < nb; i++)
            {
                int ti = T.RowPtr[i] * blockLenT;      // T[i] block base
                int aI = aggId[i];

                int s = A.RowPtr[i], e = A.RowPtr[i + 1];
                for (int k = s; k < e; k++)
                {
                    int j = A.ColInd[k];
                    int tj = T.RowPtr[j] * blockLenT;   // T[j] block base
                    int aoff = k * blockLenA;

                    // contrib[p,q] = sum_r sum_c T[i][r,p] * A_ij[r,c] * T[j][c,q].
                    for (int p = 0; p < m; p++)
                        for (int q = 0; q < m; q++)
                        {
                            fProxy acc = 0;
                            for (int r = 0; r < BR; r++)
                            {
                                fProxy tip = T.Values[ti + r * m + p];
                                if (tip == (fProxy)0) continue;
                                fProxy inner = 0;
                                for (int c = 0; c < BR; c++)
                                    inner += A.Values[aoff + r * BR + c] * T.Values[tj + c * m + q];
                                acc += tip * inner;
                            }
                            contrib[p, q] = acc;
                        }

                    builder.AddBlock(aI, aggId[j], in contrib);
                }
            }

            contrib.Dispose();
            var result = builder.ToBSR(allocator);
            builder.Dispose();
            return result;
        }
    }
}
