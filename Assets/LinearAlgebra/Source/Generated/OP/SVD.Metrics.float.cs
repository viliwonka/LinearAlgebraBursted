#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra
{
    public static partial class SVD {

        /// <summary>
        /// Computes the singular values of any-shape A into S (length k = min(M_Rows, N_Cols)),
        /// sorted in descending order. A is NOT modified — for wide A its transpose is decomposed,
        /// since A and Aᵀ share the same singular values. Uses the fast values-only Golub-Kahan path
        /// (svdValues), which needs no orthogonal factors. Allocates SVD scratch from A's arena.
        /// Returns k (= S.N). Shared by matrixL2 / cond / rank.
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
                // svdValues takes A as input (not modified) — no copy needed.
                svdValues(in A, ref S);
            }
            else {
                // Wide: decompose Aᵀ (n x m, tall); same singular values as A.
                floatMxN At = Linear_OP.trans(A);
                svdValues(in At, ref S);
            }

            return k;
        }
    }
}
