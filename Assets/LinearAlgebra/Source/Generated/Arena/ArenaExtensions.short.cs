using Unity.Mathematics;

using UnityEngine.UIElements;

namespace LinearAlgebra
{
    public static partial class ArenaExtensions {

        #region VECTOR
        public static shortN shortIndexZeroVector(this ref Arena arena, int N)
        {
            var vec = arena.shortVec(N, true);

            unsafe {
                mathUnsafeshort.setIndexZero(vec.Data.Ptr, N);
                
            }
            return vec;
        }

        public static shortN shortIndexOneVector(this ref Arena arena, int N)
        {
            var vec = arena.shortVec(N, true);

            unsafe {
                mathUnsafeshort.setIndexOne(vec.Data.Ptr, N);
            }
            return vec;
        }

        // all zero but the index is one
        public static shortN shortBasisVector(this ref Arena arena, int N, int index)
        {
            var vec = arena.shortVec(N);

            if(index < 0 || index >= N)
                throw new System.Exception("BasisVector: Index out of bounds");

            vec[index] = (short)1;

            return vec;
        }

        public static shortN shortRandomVector(this ref Arena arena, int N, short min, short max, uint seed = 84115)
        {
            var vec = arena.shortVec(N, true);

            Random random = new Random(seed);

            if (max >= min) {
                for (int i = 0; i < N; i++)
                    vec[i] = (short)random.NextInt((int)min, (int)max);
            }
            else {
                for (int i = N - 1; i >= 0; i--)
                    vec[i] = (short)random.NextInt((int)min, (int)max);
            }

            return vec;
        }

        //linspace
        public static shortN shortLinVector(this ref Arena arena, int N, short start, short end)
        {
            var vec = arena.shortVec(N);

            float scale = 1 / (float)(N - 1);
            for(int i = 0; i < N; i++) {
                vec[i] = (short)math.lerp((short)start, (short)end, i * scale);
            }

            return vec;
        }

        #endregion

        #region MATRIX
        // constructs identity matrix
        public static shortMxN shortIdentityMatrix(this ref Arena arena, int N)
        {
            var matrix = arena.shortMat(N, N);

            for (int i = 0; i < N; i++)
                matrix[i, i] = 1;
            
            return matrix;
        }

        // constructs diagonal matrix with scalar s on diagonal
        public static shortMxN shortDiagonalMatrix(this ref Arena arena, int N, short s)
        {
            var matrix = arena.shortMat(N, N);

            for (int i = 0; i < N; i++)
                matrix[i, i] = s;

            return matrix;
        }

        // constructs diagonal matrix based on vector
        public static shortMxN shortDiagonalMatrix(this ref Arena arena, in shortN vec)
        {
            var matrix = arena.shortMat(vec.N, vec.N);

            for (int i = 0; i < vec.N; i++)
                matrix[i, i] = vec[i];

            return matrix;
        }

        // constructs matrix with indexes that start at 0
        public static shortMxN shortIndexZeroMatrix(this ref Arena arena, int M_rows, int N_cols)
        {
            var mat = arena.shortMat(M_rows, N_cols, true);

            int len = mat.Length;

            unsafe
            {
                mathUnsafeshort.setIndexZero(mat.Data.Ptr, len);
            }
            
            return mat;
        }

        // constructs matrix with indexes that start at 1
        public static shortMxN shortIndexOneMatrix(this ref Arena arena, int M_rows, int N_cols)
        {
            var mat = arena.shortMat(M_rows, N_cols, true);

            int len = mat.Length;

            unsafe
            {
                mathUnsafeshort.setIndexOne(mat.Data.Ptr, len);
            }

            return mat;
        }

        // random matrix

        public static shortMxN shortRandomMatrix(this ref Arena arena, int M_rows, int N_cols, uint seed = 121312)
        {
            return shortRandomMatrix(ref arena, M_rows, N_cols, -1, 1, seed);
        }

        // constructs diagonal matrix with scalar s on diagonal
        public static shortMxN shortRandomDiagonalMatrix(this ref Arena arena, int N, short min, short max, uint seed = 65792)
        {
            var matrix = arena.shortMat(N, N);

            Random rand = new Random(seed);
            if (max >= min) {
                for (int i = 0; i < N; i++)
                    matrix[i, i] = (short)rand.NextInt((int)min, (int)max);
            }
            else {
                for (int i = N - 1; i >= 0; i--)
                    matrix[i, i] = (short)rand.NextInt((int)min, (int)max);
            }

            return matrix;
        }

        public static shortMxN shortRandomMatrix(this ref Arena arena, int M_rows, int N_cols, short min, short max, uint seed = 121312)
        {
            var matrix = arena.shortMat(M_rows, N_cols, true);

            Random random = new Random(seed);

            int len = matrix.Length;

            if (max >= min) {
                for (int i = 0; i < len; i++)
                    matrix[i] = (short)random.NextInt((int)min, (int)max);
            }
            else {

                for (int i = len - 1; i >= 0; i--)
                    matrix[i] = (short)random.NextInt((int)max, (int)min);
            }

            return matrix;
        }

        // i and j are axis indexes to swap
        public static shortMxN shortPermutationMatrix(this ref Arena arena, int M, int i, int j)
        {
            var matrix = arena.shortIdentityMatrix(M);

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