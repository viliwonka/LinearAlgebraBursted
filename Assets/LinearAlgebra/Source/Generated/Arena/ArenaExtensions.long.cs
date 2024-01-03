using Unity.Mathematics;

using UnityEngine.UIElements;

namespace LinearAlgebra
{
    public static partial class ArenaExtensions {

        #region VECTOR
        public static longN longIndexZeroVector(this ref Arena arena, int N)
        {
            var vec = arena.longVec(N, true);

            unsafe {
                mathUnsafelong.setIndexZero(vec.Data.Ptr, N);
                
            }
            return vec;
        }

        public static longN longIndexOneVector(this ref Arena arena, int N)
        {
            var vec = arena.longVec(N, true);

            unsafe {
                mathUnsafelong.setIndexOne(vec.Data.Ptr, N);
            }
            return vec;
        }

        // all zero but the index is one
        public static longN longBasisVector(this ref Arena arena, int N, int index)
        {
            var vec = arena.longVec(N);

            if(index < 0 || index >= N)
                throw new System.Exception("BasisVector: Index out of bounds");

            vec[index] = (long)1;

            return vec;
        }

        public static longN longRandomVector(this ref Arena arena, int N, long min, long max, uint seed = 84115)
        {
            var vec = arena.longVec(N, true);

            Random random = new Random(seed);

            if (max >= min) {
                for (int i = 0; i < N; i++)
                    vec[i] = (long)random.NextInt((int)min, (int)max);
            }
            else {
                for (int i = N - 1; i >= 0; i--)
                    vec[i] = (long)random.NextInt((int)min, (int)max);
            }

            return vec;
        }

        //linspace
        public static longN longLinVector(this ref Arena arena, int N, long start, long end)
        {
            var vec = arena.longVec(N);

            float scale = 1 / (float)(N - 1);
            for(int i = 0; i < N; i++) {
                vec[i] = (long)math.lerp((long)start, (long)end, i * scale);
            }

            return vec;
        }

        #endregion

        #region MATRIX
        // constructs identity matrix
        public static longMxN longIdentityMatrix(this ref Arena arena, int N)
        {
            var matrix = arena.longMat(N, N);

            for (int i = 0; i < N; i++)
                matrix[i, i] = 1;
            
            return matrix;
        }

        // constructs diagonal matrix with scalar s on diagonal
        public static longMxN longDiagonalMatrix(this ref Arena arena, int N, long s)
        {
            var matrix = arena.longMat(N, N);

            for (int i = 0; i < N; i++)
                matrix[i, i] = s;

            return matrix;
        }

        // constructs diagonal matrix based on vector
        public static longMxN longDiagonalMatrix(this ref Arena arena, in longN vec)
        {
            var matrix = arena.longMat(vec.N, vec.N);

            for (int i = 0; i < vec.N; i++)
                matrix[i, i] = vec[i];

            return matrix;
        }

        // constructs matrix with indexes that start at 0
        public static longMxN longIndexZeroMatrix(this ref Arena arena, int M_rows, int N_cols)
        {
            var mat = arena.longMat(M_rows, N_cols, true);

            int len = mat.Length;

            unsafe
            {
                mathUnsafelong.setIndexZero(mat.Data.Ptr, len);
            }
            
            return mat;
        }

        // constructs matrix with indexes that start at 1
        public static longMxN longIndexOneMatrix(this ref Arena arena, int M_rows, int N_cols)
        {
            var mat = arena.longMat(M_rows, N_cols, true);

            int len = mat.Length;

            unsafe
            {
                mathUnsafelong.setIndexOne(mat.Data.Ptr, len);
            }

            return mat;
        }

        // random matrix

        public static longMxN longRandomMatrix(this ref Arena arena, int M_rows, int N_cols, uint seed = 121312)
        {
            return longRandomMatrix(ref arena, M_rows, N_cols, -1, 1, seed);
        }

        // constructs diagonal matrix with scalar s on diagonal
        public static longMxN longRandomDiagonalMatrix(this ref Arena arena, int N, long min, long max, uint seed = 65792)
        {
            var matrix = arena.longMat(N, N);

            Random rand = new Random(seed);
            if (max >= min) {
                for (int i = 0; i < N; i++)
                    matrix[i, i] = (long)rand.NextInt((int)min, (int)max);
            }
            else {
                for (int i = N - 1; i >= 0; i--)
                    matrix[i, i] = (long)rand.NextInt((int)min, (int)max);
            }

            return matrix;
        }

        public static longMxN longRandomMatrix(this ref Arena arena, int M_rows, int N_cols, long min, long max, uint seed = 121312)
        {
            var matrix = arena.longMat(M_rows, N_cols, true);

            Random random = new Random(seed);

            int len = matrix.Length;

            if (max >= min) {
                for (int i = 0; i < len; i++)
                    matrix[i] = (long)random.NextInt((int)min, (int)max);
            }
            else {

                for (int i = len - 1; i >= 0; i--)
                    matrix[i] = (long)random.NextInt((int)max, (int)min);
            }

            return matrix;
        }

        // i and j are axis indexes to swap
        public static longMxN longPermutationMatrix(this ref Arena arena, int M, int i, int j)
        {
            var matrix = arena.longIdentityMatrix(M);

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