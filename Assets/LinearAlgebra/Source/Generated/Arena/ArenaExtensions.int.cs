using Unity.Mathematics;

using UnityEngine.UIElements;

namespace LinearAlgebra
{
    public static partial class ArenaExtensions {

        #region VECTOR
        public static intN intIndexZeroVector(this ref Arena arena, int N)
        {
            var vec = arena.intVec(N, true);

            unsafe {
                mathUnsafeint.setIndexZero(vec.Data.Ptr, N);
                
            }
            return vec;
        }

        public static intN intIndexOneVector(this ref Arena arena, int N)
        {
            var vec = arena.intVec(N, true);

            unsafe {
                mathUnsafeint.setIndexOne(vec.Data.Ptr, N);
            }
            return vec;
        }

        // all zero but the index is one
        public static intN intBasisVector(this ref Arena arena, int N, int index)
        {
            var vec = arena.intVec(N);

            if(index < 0 || index >= N)
                throw new System.Exception("BasisVector: Index out of bounds");

            vec[index] = (int)1;

            return vec;
        }

        public static intN intRandomVector(this ref Arena arena, int N, int min, int max, uint seed = 84115)
        {
            var vec = arena.intVec(N, true);

            Random random = new Random(seed);

            if (max >= min) {
                for (int i = 0; i < N; i++)
                    vec[i] = (int)random.NextInt((int)min, (int)max);
            }
            else {
                // max < min: pass the smaller bound first so NextInt gets a valid [lo, hi) range.
                // Previously passed (min, max) here, where min > max — Unity.Mathematics NextInt
                // then computed (max - min) as a negative span and returned garbage.
                for (int i = N - 1; i >= 0; i--)
                    vec[i] = (int)random.NextInt((int)max, (int)min);
            }

            return vec;
        }

        //linspace
        public static intN intLinVector(this ref Arena arena, int N, int start, int end)
        {
            var vec = arena.intVec(N);

            // N == 1 would divide by (N-1) == 0 -> Inf -> NaN -> garbage int. Match the guarded
            // fProxyGenOP.linspace convention: a single sample returns {start}.
            if (N == 1) { vec[0] = start; return vec; }

            float scale = 1 / (float)(N - 1);
            for(int i = 0; i < N; i++) {
                vec[i] = (int)math.lerp((int)start, (int)end, i * scale);
            }
            // Pin endpoints exactly (the lerp at the last index lands ~1 ulp short of end,
            // which can truncate to the wrong integer).
            vec[0] = start;
            vec[N - 1] = end;

            return vec;
        }

        #endregion

        #region MATRIX
        // constructs identity matrix
        public static intMxN intIdentityMatrix(this ref Arena arena, int N)
        {
            var matrix = arena.intMat(N, N);

            for (int i = 0; i < N; i++)
                matrix[i, i] = 1;
            
            return matrix;
        }

        // constructs diagonal matrix with scalar s on diagonal
        public static intMxN intDiagonalMatrix(this ref Arena arena, int N, int s)
        {
            var matrix = arena.intMat(N, N);

            for (int i = 0; i < N; i++)
                matrix[i, i] = s;

            return matrix;
        }

        // constructs diagonal matrix based on vector
        public static intMxN intDiagonalMatrix(this ref Arena arena, in intN vec)
        {
            var matrix = arena.intMat(vec.N, vec.N);

            for (int i = 0; i < vec.N; i++)
                matrix[i, i] = vec[i];

            return matrix;
        }

        // constructs matrix with indexes that start at 0
        public static intMxN intIndexZeroMatrix(this ref Arena arena, int M_rows, int N_cols)
        {
            var mat = arena.intMat(M_rows, N_cols, true);

            int len = mat.Length;

            unsafe
            {
                mathUnsafeint.setIndexZero(mat.Data.Ptr, len);
            }
            
            return mat;
        }

        // constructs matrix with indexes that start at 1
        public static intMxN intIndexOneMatrix(this ref Arena arena, int M_rows, int N_cols)
        {
            var mat = arena.intMat(M_rows, N_cols, true);

            int len = mat.Length;

            unsafe
            {
                mathUnsafeint.setIndexOne(mat.Data.Ptr, len);
            }

            return mat;
        }

        // random matrix

        public static intMxN intRandomMatrix(this ref Arena arena, int M_rows, int N_cols, uint seed = 121312)
        {
            return intRandomMatrix(ref arena, M_rows, N_cols, -1, 1, seed);
        }

        // constructs diagonal matrix with scalar s on diagonal
        public static intMxN intRandomDiagonalMatrix(this ref Arena arena, int N, int min, int max, uint seed = 65792)
        {
            var matrix = arena.intMat(N, N);

            Random rand = new Random(seed);
            if (max >= min) {
                for (int i = 0; i < N; i++)
                    matrix[i, i] = (int)rand.NextInt((int)min, (int)max);
            }
            else {
                // max < min: pass the smaller bound first so NextInt gets a valid [lo, hi) range
                // (previously passed the inverted (min, max), yielding garbage).
                for (int i = N - 1; i >= 0; i--)
                    matrix[i, i] = (int)rand.NextInt((int)max, (int)min);
            }

            return matrix;
        }

        public static intMxN intRandomMatrix(this ref Arena arena, int M_rows, int N_cols, int min, int max, uint seed = 121312)
        {
            var matrix = arena.intMat(M_rows, N_cols, true);

            Random random = new Random(seed);

            int len = matrix.Length;

            if (max >= min) {
                for (int i = 0; i < len; i++)
                    matrix[i] = (int)random.NextInt((int)min, (int)max);
            }
            else {

                for (int i = len - 1; i >= 0; i--)
                    matrix[i] = (int)random.NextInt((int)max, (int)min);
            }

            return matrix;
        }

        // i and j are axis indexes to swap
        public static intMxN intPermutationMatrix(this ref Arena arena, int M, int i, int j)
        {
            var matrix = arena.intIdentityMatrix(M);

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