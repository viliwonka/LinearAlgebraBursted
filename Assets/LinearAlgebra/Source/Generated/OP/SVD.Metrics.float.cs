#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra
{
    public static partial class SVD {

        /// <summary>
        /// Computes the singular values of any-shape A into S (length k = min(M_Rows, N_Cols)),
        /// sorted in descending order. A is NOT modified — an internal copy (tall A) or transpose
        /// (wide A) is decomposed, since A and Aᵀ share the same singular values. Allocates SVD
        /// scratch from A's arena. Returns k (= S.N). Shared by matrixL2 / cond / rank.
        /// </summary>
        public static int singularValues(in floatMxN A, ref floatN S)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;
            int k = math.min(m, n);

            if (S.N != k)
                throw new ArgumentException("singularValues: S.N must equal min(A.M_Rows, A.N_Cols)");

            if (k == 0)
                return 0;

            if (m >= n) {
                // U holds a copy of A (svdDecomposition overwrites it with the left vectors).
                // TempCopy (not Copy) so the scratch lands in the temp pool reclaimed by ClearTemp,
                // matching the wide branch's trans() and the sibling SVD solvers.
                floatMxN U = A.TempCopy();
                floatMxN V = A.tempfloatMat(n, n);
                svdDecomposition(ref U, ref S, ref V, 30);
            }
            else {
                // Wide: decompose Aᵀ (n x m, tall); same singular values as A.
                floatMxN At = floatOP.trans(A);
                floatMxN V = A.tempfloatMat(m, m);
                svdDecomposition(ref At, ref S, ref V, 30);
            }

            return k;
        }
    }
}
