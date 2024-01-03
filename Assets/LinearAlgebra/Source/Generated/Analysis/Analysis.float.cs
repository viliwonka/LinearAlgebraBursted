#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Mathematics;
using Unity.Burst;
namespace LinearAlgebra
{

    public static partial class Analysis {

        public static bool IsAnyNan(in floatN a) {
            
            for (int i = 0; i < a.N; i++) {
                if (a[i] != a[i])
                    return true;
            }
            return false;
        }

        public static bool IsAnyNan(in floatMxN m) {
            
            for (int i = 0; i < m.Length; i++) {
                if (m[i] != m[i])
                    return true;
            }

            return false;
        }

        public static bool IsAnyInf(in floatN a) {
                        
            for (int i = 0; i < a.N; i++) {
                if (math.isinf(a[i]))
                    return true;
            }

            return false;
        }

        public static bool IsAnyInf(in floatMxN m) {

            for (int i = 0; i < m.Length; i++) {
                if (math.isinf(m[i]))
                    return true;
            }

            return false;
        }

        public static bool IsZero(in floatN a, float epsilon)
        {
            for (int i = 0; i < a.N; i++) {
                if (math.abs(a[i]) > epsilon)
                    return false;
            }

            return true;
        }

        public static bool IsZero(in floatMxN m, float epsilon)
        {
            for (int i = 0; i < m.Length; i++) {
                if (math.abs(m[i]) > epsilon)
                    return false;
            }

            return true;
        }

        public static float MaxZeroError(in floatMxN m)
        {
            float maxError = 0f;
            for (int i = 0; i < m.Length; i++)
                maxError = math.max(maxError, math.abs(m[i]));
            
            return maxError;
        }

        public static float MaxZeroError(in floatN v)
        {
            float maxError = 0f;
            for (int i = 0; i < v.N; i++)
                maxError = math.max(maxError, math.abs(v[i]));
            
            return maxError;
        }

        public static bool IsIdentity(in floatMxN A)
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

        public static bool IsIdentity(in floatMxN A, float epsilon)
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

        public static bool IsSymmetric(in floatMxN A)
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

        public static bool IsSymmetric(in floatMxN A, float epsilon)
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

        public static bool IsDiagonal(in floatMxN A)
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

        public static bool IsDiagonal(in floatMxN A, float epsilon)
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

        public static bool IsUpperTriangular(in floatMxN A)
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

        public static bool IsUpperTriangular(in floatMxN A, float epsilon)
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

        public static bool IsLowerTriangular(in floatMxN A)
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

        public static bool IsLowerTriangular(in floatMxN A, float epsilon)
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

        // could be done in-place with dot products and comparisons
        public static bool IsOrthogonal(in floatMxN A, float epsilon)
        {
            /*if (A.M_Rows != A.N_Cols)
                return false;*/

            floatMxN B = new floatMxN(A.N_Cols, A.N_Cols, Unity.Collections.Allocator.Temp);

            // B = A^T * A
            unsafe {
                UnsafeOP.matMatDotTransA(A.Data.Ptr, A.Data.Ptr, B.Data.Ptr, A.N_Cols, A.M_Rows, B.N_Cols);
            }

            bool valid = true;

            if (!IsIdentity(B, epsilon))
            {
                valid = false;
            }

            B.Dispose();

            return valid;
        }
    }
}
