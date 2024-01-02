using Unity.Mathematics;

using UnityEngine.UIElements;

namespace LinearAlgebra
{
    public static partial class ArenaExtensions {

        #region VECTOR
        public static iProxyN iProxyIndexZeroVector(this ref Arena arena, int N)
        {
            var vec = arena.iProxyVec(N, true);

            unsafe {
                mathUnsafeiProxy.setIndexZero(vec.Data.Ptr, N);
                
            }
            return vec;
        }

        public static iProxyN iProxyIndexOneVector(this ref Arena arena, int N)
        {
            var vec = arena.iProxyVec(N, true);

            unsafe {
                mathUnsafeiProxy.setIndexOne(vec.Data.Ptr, N);
            }
            return vec;
        }

        // all zero but the index is one
        public static iProxyN iProxyBasisVector(this ref Arena arena, int N, int index)
        {
            var vec = arena.iProxyVec(N);

            if(index < 0 || index >= N)
                throw new System.Exception("BasisVector: Index out of bounds");

            vec[index] = (iProxy)1;

            return vec;
        }

        public static iProxyN iProxyRandomVector(this ref Arena arena, int N, iProxy min, iProxy max, uint seed = 84115)
        {
            var vec = arena.iProxyVec(N, true);

            Random random = new Random(seed);

            if (max >= min) {
                for (int i = 0; i < N; i++)
                    vec[i] = (iProxy)random.NextInt((int)min, (int)max);
            }
            else {
                for (int i = N - 1; i >= 0; i--)
                    vec[i] = (iProxy)random.NextInt((int)min, (int)max);
            }

            return vec;
        }

        //linspace
        public static iProxyN iProxyLinVector(this ref Arena arena, int N, iProxy start, iProxy end)
        {
            var vec = arena.iProxyVec(N);

            float scale = 1 / (float)(N - 1);
            for(int i = 0; i < N; i++) {
                vec[i] = (iProxy)math.lerp((iProxy)start, (iProxy)end, i * scale);
            }

            return vec;
        }

        #endregion

        #region MATRIX
        // constructs identity matrix
        public static iProxyMxN iProxyIdentityMatrix(this ref Arena arena, int N)
        {
            var matrix = arena.iProxyMat(N, N);

            for (int i = 0; i < N; i++)
                matrix[i, i] = 1;
            
            return matrix;
        }

        // constructs diagonal matrix with scalar s on diagonal
        public static iProxyMxN iProxyDiagonalMatrix(this ref Arena arena, int N, iProxy s)
        {
            var matrix = arena.iProxyMat(N, N);

            for (int i = 0; i < N; i++)
                matrix[i, i] = s;

            return matrix;
        }

        // constructs diagonal matrix based on vector
        public static iProxyMxN iProxyDiagonalMatrix(this ref Arena arena, in iProxyN vec)
        {
            var matrix = arena.iProxyMat(vec.N, vec.N);

            for (int i = 0; i < vec.N; i++)
                matrix[i, i] = vec[i];

            return matrix;
        }

        // constructs matrix with indexes that start at 0
        public static iProxyMxN iProxyIndexZeroMatrix(this ref Arena arena, int M_rows, int N_cols)
        {
            var mat = arena.iProxyMat(M_rows, N_cols, true);

            int len = mat.Length;

            unsafe
            {
                mathUnsafeiProxy.setIndexZero(mat.Data.Ptr, len);
            }
            
            return mat;
        }

        // constructs matrix with indexes that start at 1
        public static iProxyMxN iProxyIndexOneMatrix(this ref Arena arena, int M_rows, int N_cols)
        {
            var mat = arena.iProxyMat(M_rows, N_cols, true);

            int len = mat.Length;

            unsafe
            {
                mathUnsafeiProxy.setIndexOne(mat.Data.Ptr, len);
            }

            return mat;
        }

        // random matrix

        public static iProxyMxN iProxyRandomMatrix(this ref Arena arena, int M_rows, int N_cols, uint seed = 121312)
        {
            return iProxyRandomMatrix(ref arena, M_rows, N_cols, -1, 1, seed);
        }

        // constructs diagonal matrix with scalar s on diagonal
        public static iProxyMxN iProxyRandomDiagonalMatrix(this ref Arena arena, int N, iProxy min, iProxy max, uint seed = 65792)
        {
            var matrix = arena.iProxyMat(N, N);

            Random rand = new Random(seed);
            if (max >= min) {
                for (int i = 0; i < N; i++)
                    matrix[i, i] = (iProxy)rand.NextInt((int)min, (int)max);
            }
            else {
                for (int i = N - 1; i >= 0; i--)
                    matrix[i, i] = (iProxy)rand.NextInt((int)min, (int)max);
            }

            return matrix;
        }

        public static iProxyMxN iProxyRandomMatrix(this ref Arena arena, int M_rows, int N_cols, iProxy min, iProxy max, uint seed = 121312)
        {
            var matrix = arena.iProxyMat(M_rows, N_cols, true);

            Random random = new Random(seed);

            int len = matrix.Length;

            if (max >= min) {
                for (int i = 0; i < len; i++)
                    matrix[i] = (iProxy)random.NextInt((int)min, (int)max);
            }
            else {

                for (int i = len - 1; i >= 0; i--)
                    matrix[i] = (iProxy)random.NextInt((int)max, (int)min);
            }

            return matrix;
        }

        // i and j are axis indexes to swap
        public static iProxyMxN iProxyPermutationMatrix(this ref Arena arena, int M, int i, int j)
        {
            var matrix = arena.iProxyIdentityMatrix(M);

            if (M < 2)
                throw new System.Exception("PermutationMatrix: Matrix must be at least 2x2");

            if (i < 0 || i >= M || j < 0 || j >= M)
                throw new System.Exception("PermutationMatrix: Index out of bounds");

            if (i == j)
            {
                return matrix;
            }

            matrix[i, j] = 1;
            matrix[j, i] = 1;
            matrix[i, i] = 0;
            matrix[j, j] = 0;
            
            return matrix;
        }

        #endregion

    }

}