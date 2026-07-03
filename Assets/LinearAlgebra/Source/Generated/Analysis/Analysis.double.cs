#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Mathematics;
using Unity.Burst;
using LinearAlgebra.Internal;
namespace LinearAlgebra
{

    public static partial class Analysis_OP {

        public static bool isAnyNan(in doubleN a) {
            
            for (int i = 0; i < a.N; i++) {
                if (a[i] != a[i])
                    return true;
            }
            return false;
        }

        public static bool isAnyNan(in doubleMxN m) {
            
            for (int i = 0; i < m.Length; i++) {
                if (m[i] != m[i])
                    return true;
            }

            return false;
        }

        public static bool isAnyInf(in doubleN a) {
                        
            for (int i = 0; i < a.N; i++) {
                if (math.isinf(a[i]))
                    return true;
            }

            return false;
        }

        public static bool isAnyInf(in doubleMxN m) {

            for (int i = 0; i < m.Length; i++) {
                if (math.isinf(m[i]))
                    return true;
            }

            return false;
        }

        public static bool isZero(in doubleN a, double epsilon)
        {
            for (int i = 0; i < a.N; i++) {
                if (math.abs(a[i]) > epsilon)
                    return false;
            }

            return true;
        }

        public static bool isZero(in doubleMxN m, double epsilon)
        {
            for (int i = 0; i < m.Length; i++) {
                if (math.abs(m[i]) > epsilon)
                    return false;
            }

            return true;
        }

        public static double MaxZeroError(in doubleMxN m)
        {
            double maxError = 0f;
            for (int i = 0; i < m.Length; i++)
                maxError = math.max(maxError, math.abs(m[i]));
            
            return maxError;
        }

        public static double MaxZeroError(in doubleN v)
        {
            double maxError = 0f;
            for (int i = 0; i < v.N; i++)
                maxError = math.max(maxError, math.abs(v[i]));
            
            return maxError;
        }

        public static bool isIdentity(in doubleMxN A)
        {
            if(A.M_Rows != A.N_Cols)
                return false;

            for (int r = 0; r < A.M_Rows; r++)
            for (int c = 0; c < A.M_Rows; c++)
            {
                if (r == c)
                {
                    if (A[r, c] != 1f) 
                        return false;
                }
                else if (A[r, c] != 0f)
                    return false;
            }
            return true;
        }

        public static bool isIdentity(in doubleMxN A, double epsilon)
        {
            if (A.M_Rows != A.N_Cols)
                return false;

            for (int r = 0; r < A.M_Rows; r++)
            for (int c = 0; c < A.M_Rows; c++)
            {
                if (r == c) {  
                    if(math.abs(A[r, c] - 1f) > epsilon)
                        return false;
                }
                else if (math.abs(A[r, c]) > epsilon)
                    return false;
            }
            return true;
        }

        public static bool isSymmetric(in doubleMxN A)
        {
            if(A.M_Rows != A.N_Cols)
                return false;

            for (int r = 0; r < A.M_Rows; r++)
            for (int c = 0; c < A.M_Rows; c++)
            {
                if (A[r, c] != A[c, r])
                    return false;
            }
            return true;
        }

        public static bool isSymmetric(in doubleMxN A, double epsilon)
        {
            if(A.M_Rows != A.N_Cols)
                return false;

            for (int r = 0; r < A.M_Rows; r++)
            for (int c = 0; c < A.M_Rows; c++)
            {
                if (math.abs(A[r, c] - A[c, r]) > epsilon)
                    return false;
            }
            return true;
        }

        public static bool isDiagonal(in doubleMxN A)
        {
            if(A.M_Rows != A.N_Cols)
                return false;

            for (int r = 0; r < A.M_Rows; r++)
            for (int c = 0; c < A.M_Rows; c++)
            {
                if (r != c && A[r, c] != 0f)
                    return false;
            }
            return true;
        }

        public static bool isDiagonal(in doubleMxN A, double epsilon)
        {
            if(A.M_Rows != A.N_Cols)
                return false;

            for (int r = 0; r < A.M_Rows; r++)
            for (int c = 0; c < A.M_Rows; c++)
            {
                if (r != c && math.abs(A[r, c]) > epsilon)
                    return false;
            }
            return true;
        }

        public static bool isUpperTriangular(in doubleMxN A)
        {
            if(A.M_Rows != A.N_Cols)
                return false;

            for (int r = 0; r < A.M_Rows; r++)
            for (int c = 0; c < r; c++)
            {
                if (A[r, c] != 0f)
                    return false;
            }
            return true;
        }

        public static bool isUpperTriangular(in doubleMxN A, double epsilon)
        {
            if(A.M_Rows != A.N_Cols)
                return false;

            for (int r = 0; r < A.M_Rows; r++)
            for (int c = 0; c < r; c++)
            {
                if (math.abs(A[r, c]) > epsilon)
                    return false;
            }
            return true;
        }

        public static bool isLowerTriangular(in doubleMxN A)
        {
            if(A.M_Rows != A.N_Cols)
                return false;

            for (int r = 0; r < A.M_Rows; r++)
            for (int c = r + 1; c < A.M_Rows; c++)
            {
                if (A[r, c] != 0f)
                    return false;
            }
            return true;
        }

        public static bool isLowerTriangular(in doubleMxN A, double epsilon)
        {
            if(A.M_Rows != A.N_Cols)
                return false;

            for (int r = 0; r < A.M_Rows; r++)
            for (int c = r + 1; c < A.M_Rows; c++)
            {
                if (math.abs(A[r, c]) > epsilon)
                    return false;
            }

            return true;
        }

        public static bool isOrthogonal(in doubleMxN A, double epsilon)
        {
            doubleMxN B = new doubleMxN(A.N_Cols, A.N_Cols, Unity.Collections.Allocator.Temp);

            // B = A^T * A
            unsafe {
                Unsafe_OP.matMatDotTransA(A.Data.Ptr, A.Data.Ptr, B.Data.Ptr, A.N_Cols, A.M_Rows, B.N_Cols);
            }

            bool valid = true;

            // NaN-reject: a NaN in B = AᵀA would otherwise slip through isIdentity's epsilon test
            // (abs(NaN) > eps is false), wrongly reporting a NaN-poisoned matrix as orthogonal.
            if (isAnyNan(in B) || !isIdentity(B, epsilon))
            {
                valid = false;
            }

            B.Dispose();

            return valid;
        }
    }
}
