#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

namespace LinearAlgebra
{
    // Structural predicates for the SIGNED integer family (int/short/long -- uint is
    // deliberately excluded from this surface, see docs/naming-style-guide.md). Merges into the
    // SAME bare partial class as Analysis.fProxy.cs's `Analysis` (safe: every method here takes
    // a concrete shortN/shortMxN parameter, never a bare generic <T>, so this follows the same
    // merge rule that already lets float and double coexist in one Analysis -- see
    // docs/naming-style-guide.md's "Split vs merge safety").
    //
    // DELIBERATELY NO EPSILON/TOLERANCE PARAMETER: integer arithmetic is exact (no rounding
    // error to tolerate), so every predicate here is an exact-equality check. float/double
    // Analysis offers both a bare (exact) and an epsilon-taking overload for isIdentity/
    // isSymmetric/isDiagonal/isUpperTriangular/isLowerTriangular; the integer surface
    // intentionally has ONLY the bare form -- an epsilon-taking sibling would just mask real
    // off-by-one bugs instead of tolerating legitimate floating-point roundoff, which doesn't
    // exist for integers.
    public static partial class Analysis
    {
        public static bool isZero(in shortN a)
        {
            for (int i = 0; i < a.N; i++) {
                if (a[i] != 0)
                    return false;
            }

            return true;
        }

        public static bool isZero(in shortMxN m)
        {
            for (int i = 0; i < m.Length; i++) {
                if (m[i] != 0)
                    return false;
            }

            return true;
        }

        public static bool isIdentity(in shortMxN A)
        {
            if (A.M_Rows != A.N_Cols)
                return false;

            for (int r = 0; r < A.M_Rows; r++)
            for (int c = 0; c < A.M_Rows; c++)
            {
                if (r == c)
                {
                    if (A[r, c] != 1)
                        return false;
                }
                else if (A[r, c] != 0)
                    return false;
            }
            return true;
        }

        public static bool isSymmetric(in shortMxN A)
        {
            if (A.M_Rows != A.N_Cols)
                return false;

            for (int r = 0; r < A.M_Rows; r++)
            for (int c = 0; c < A.M_Rows; c++)
            {
                if (A[r, c] != A[c, r])
                    return false;
            }
            return true;
        }

        public static bool isDiagonal(in shortMxN A)
        {
            if (A.M_Rows != A.N_Cols)
                return false;

            for (int r = 0; r < A.M_Rows; r++)
            for (int c = 0; c < A.M_Rows; c++)
            {
                if (r != c && A[r, c] != 0)
                    return false;
            }
            return true;
        }

        public static bool isUpperTriangular(in shortMxN A)
        {
            if (A.M_Rows != A.N_Cols)
                return false;

            for (int r = 0; r < A.M_Rows; r++)
            for (int c = 0; c < r; c++)
            {
                if (A[r, c] != 0)
                    return false;
            }
            return true;
        }

        public static bool isLowerTriangular(in shortMxN A)
        {
            if (A.M_Rows != A.N_Cols)
                return false;

            for (int r = 0; r < A.M_Rows; r++)
            for (int c = r + 1; c < A.M_Rows; c++)
            {
                if (A[r, c] != 0)
                    return false;
            }
            return true;
        }
    }
}
