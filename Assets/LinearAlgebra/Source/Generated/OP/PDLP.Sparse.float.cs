#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class LP
    {
        // ============================================================================================
        // SPARSE (BSR) PDLP -- the matrix-free first-order LP (restarted PDHG) over a block-sparse
        // constraint matrix. This is the regime PDLP is built for: the whole solver only ever calls
        // spMV / spMVT on A, so nothing scales with a dense N² and there is no factorization / normal
        // equation to precondition. Shares pdlpCore / pdlpScaledSolve with the dense entry point (see
        // PDLP.float.cs); only the equilibration reads the sparse blocks directly. See docs/spec-pdlp.md.
        // ============================================================================================

        // Ruiz (ℓ∞) equilibration for a BSR matrix, filling Dr (length m) / Dc (length n) from ONE block
        // traversal per sweep (O(nnz)) -- the sparse mirror of pdlpEquilibrateDense (Ruiz only; a
        // Pock-Chambolle ℓ1 pass on top over-shrinks ‖Â‖→0). Symmetric (upper-block-only) storage is
        // skipped: its implicit lower blocks are not stored, so a single pass would under-count the
        // row/column norms; leaving Dr=Dc=1 is safe (a positive diagonal never moves the optimum). Clamped.
        static unsafe void pdlpEquilibrateBSR(in floatBSR A, ref floatN Dr, ref floatN Dc)
        {
            int m = A.M_Rows, n = A.N_Cols;
            for (int i = 0; i < m; i++) Dr[i] = (float)1;
            for (int j = 0; j < n; j++) Dc[j] = (float)1;
            if (A.Symmetric) return;

            int BR = A.BR, BC = A.BC, blockSize = BR * BC, blockRows = A.BlockRows;
            int* rowPtr = A.RowPtr.Ptr;
            int* colInd = A.ColInd.Ptr;
            float* values = A.Values.Ptr;
            float* dr = Dr.Data.Ptr;
            float* dc = Dc.Data.Ptr;

            var racc = new floatN(m, Allocator.Temp);
            var cacc = new floatN(n, Allocator.Temp);
            float* ra = racc.Data.Ptr;
            float* ca = cacc.Data.Ptr;
            long fsz = UnsafeUtility.SizeOf<float>();

            for (int it = 0; it < 10; it++)
            {
                // row inf-norms of the current Â: ra[i] = max_j |A_ij|·dc[j]
                UnsafeUtility.MemClear(ra, (long)m * fsz);
                for (int bi = 0; bi < blockRows; bi++)
                    for (int k = rowPtr[bi]; k < rowPtr[bi + 1]; k++)
                    {
                        int colBase = colInd[k] * BC;
                        float* blk = values + (long)k * blockSize;
                        for (int r = 0; r < BR; r++)
                        {
                            int gr = bi * BR + r; if (gr >= m) continue;
                            for (int c = 0; c < BC; c++)
                            {
                                int gc = colBase + c; if (gc >= n) continue;
                                double v = math.abs((double)blk[r * BC + c]) * (double)dc[gc];
                                if (v > (double)ra[gr]) ra[gr] = (float)v;
                            }
                        }
                    }
                for (int i = 0; i < m; i++) { double R = (double)dr[i] * (double)ra[i]; if (R > 1e-30) dr[i] = (float)((double)dr[i] / math.sqrt(R)); }

                // column inf-norms (using the updated Dr): ca[j] = max_i |A_ij|·dr[i]
                UnsafeUtility.MemClear(ca, (long)n * fsz);
                for (int bi = 0; bi < blockRows; bi++)
                    for (int k = rowPtr[bi]; k < rowPtr[bi + 1]; k++)
                    {
                        int colBase = colInd[k] * BC;
                        float* blk = values + (long)k * blockSize;
                        for (int r = 0; r < BR; r++)
                        {
                            int gr = bi * BR + r; if (gr >= m) continue;
                            for (int c = 0; c < BC; c++)
                            {
                                int gc = colBase + c; if (gc >= n) continue;
                                double v = math.abs((double)blk[r * BC + c]) * (double)dr[gr];
                                if (v > (double)ca[gc]) ca[gc] = (float)v;
                            }
                        }
                    }
                for (int j = 0; j < n; j++) { double C = (double)dc[j] * (double)ca[j]; if (C > 1e-30) dc[j] = (float)((double)dc[j] / math.sqrt(C)); }
            }

            for (int i = 0; i < m; i++) dr[i] = (float)math.clamp((double)dr[i], 1e-12, 1e12);
            for (int j = 0; j < n; j++) dc[j] = (float)math.clamp((double)dc[j], 1e-12, 1e12);

            racc.Dispose();
            cacc.Dispose();
        }

        /// <summary>
        /// Sparse (BSR) PDLP (matrix-free first-order LP): minimize cᵀx s.t. ℓ_c ≤ A x ≤ u_c,
        /// ℓ_v ≤ x ≤ u_v, with A a block-sparse <see cref="floatBSR"/>. Use ±<c>1e30</c> in a bound to mean
        /// unbounded (ℓ_c=u_c ⇒ equality row, ℓ_v=0/u_v=1e30 ⇒ x ≥ 0). Restarted PDHG with adaptive step
        /// size + primal weight + Ruiz/Pock-Chambolle preconditioning; every iteration is a single
        /// spMV + spMVT, so cost scales with nnz, not N². <paramref name="x"/> (length A.N_Cols) is
        /// overwritten with the solution. This is the large-scale sibling of the dense
        /// <see cref="pdlp(in floatMxN, in floatN, in floatN, in floatN, in floatN, in floatN, ref floatN, out double, int, double)"/>.
        /// </summary>
        public static LPInfo pdlp(in floatBSR A, in floatN lc, in floatN uc, in floatN lv, in floatN uv,
                                  in floatN c, ref floatN x, out double objective, int maxIter = 0, double epsOpt = 1e-6)
        {
            int m = A.M_Rows, n = A.N_Cols;
            if (lc.N != m || uc.N != m) throw new System.ArgumentException("LP.pdlp(BSR): lc/uc length must equal A.M_Rows");
            if (lv.N != n || uv.N != n) throw new System.ArgumentException("LP.pdlp(BSR): lv/uv length must equal A.N_Cols");
            if (c.N != n) throw new System.ArgumentException("LP.pdlp(BSR): c length must equal A.N_Cols");
            if (x.N != n) throw new System.ArgumentException("LP.pdlp(BSR): x length must equal A.N_Cols");

            var Dr = new floatN(m, Allocator.Temp);
            var Dc = new floatN(n, Allocator.Temp);
            pdlpEquilibrateBSR(in A, ref Dr, ref Dc);

            var op = new floatBSROperator(in A);
            var info = pdlpScaledSolve(in op, in Dr, in Dc, in lc, in uc, in lv, in uv, in c, ref x, out objective, maxIter, epsOpt);

            Dr.Dispose();
            Dc.Dispose();
            return info;
        }
    }
}
