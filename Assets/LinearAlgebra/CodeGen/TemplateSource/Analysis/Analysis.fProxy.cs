using Unity.Mathematics;
using Unity.Burst;
using LinearAlgebra.Internal;
namespace LinearAlgebra
{

    public static partial class Analysis {

        public static bool isAnyNan(in fProxyN a) {
            
            for (int i = 0; i < a.N; i++) {
                if (a[i] != a[i])
                    return true;
            }
            return false;
        }

        public static bool isAnyNan(in fProxyMxN m) {
            
            for (int i = 0; i < m.Length; i++) {
                if (m[i] != m[i])
                    return true;
            }

            return false;
        }

        public static bool isAnyInf(in fProxyN a) {
                        
            for (int i = 0; i < a.N; i++) {
                if (math.isinf(a[i]))
                    return true;
            }

            return false;
        }

        public static bool isAnyInf(in fProxyMxN m) {

            for (int i = 0; i < m.Length; i++) {
                if (math.isinf(m[i]))
                    return true;
            }

            return false;
        }

        public static bool isZero(in fProxyN a, fProxy epsilon)
        {
            for (int i = 0; i < a.N; i++) {
                if (math.abs(a[i]) > epsilon)
                    return false;
            }

            return true;
        }

        public static bool isZero(in fProxyMxN m, fProxy epsilon)
        {
            for (int i = 0; i < m.Length; i++) {
                if (math.abs(m[i]) > epsilon)
                    return false;
            }

            return true;
        }

        public static fProxy MaxZeroError(in fProxyMxN m)
        {
            fProxy maxError = 0f;
            for (int i = 0; i < m.Length; i++)
                maxError = math.max(maxError, math.abs(m[i]));
            
            return maxError;
        }

        public static fProxy MaxZeroError(in fProxyN v)
        {
            fProxy maxError = 0f;
            for (int i = 0; i < v.N; i++)
                maxError = math.max(maxError, math.abs(v[i]));
            
            return maxError;
        }

        public static bool isIdentity(in fProxyMxN A)
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

        public static bool isIdentity(in fProxyMxN A, fProxy epsilon)
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

        public static bool isSymmetric(in fProxyMxN A)
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

        public static bool isSymmetric(in fProxyMxN A, fProxy epsilon)
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

        public static bool isDiagonal(in fProxyMxN A)
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

        public static bool isDiagonal(in fProxyMxN A, fProxy epsilon)
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

        public static bool isUpperTriangular(in fProxyMxN A)
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

        public static bool isUpperTriangular(in fProxyMxN A, fProxy epsilon)
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

        public static bool isLowerTriangular(in fProxyMxN A)
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

        public static bool isLowerTriangular(in fProxyMxN A, fProxy epsilon)
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

        public static bool isOrthogonal(in fProxyMxN A, fProxy epsilon)
        {
            fProxyMxN B = new fProxyMxN(A.N_Cols, A.N_Cols, Unity.Collections.Allocator.Temp);

            // B = A^T * A. The GEMM kernel promises [NoAlias] on every pointer, so the second
            // A goes in as a Temp copy (O(n²) copy against the O(n³) product).
            var Acopy = new fProxyMxN(A.M_Rows, A.N_Cols, Unity.Collections.Allocator.Temp, true);
            Acopy.Data.CopyFrom(A.Data);
            unsafe {
                UnsafeOP.matMatDotTransA(A.Data.Ptr, Acopy.Data.Ptr, B.Data.Ptr, A.N_Cols, A.M_Rows, B.N_Cols);
            }
            Acopy.Dispose();

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
