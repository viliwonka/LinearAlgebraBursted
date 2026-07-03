using Unity.Mathematics;

namespace LinearAlgebra
{
    public static partial class ArenaExtensions {

        #region VECTOR
        public static longN longIndexZeroVec(this ref Arena arena, int N)
        {
            var vec = arena.longVec(N, true);

            unsafe {
                mathUnsafelong.setIndexZero(vec.Data.Ptr, N);
                
            }
            return vec;
        }

        public static longN longIndexOneVec(this ref Arena arena, int N)
        {
            var vec = arena.longVec(N, true);

            unsafe {
                mathUnsafelong.setIndexOne(vec.Data.Ptr, N);
            }
            return vec;
        }

        // all zero but the index is one
        public static longN longBasisVec(this ref Arena arena, int N, int index)
        {
            var vec = arena.longVec(N);

            if(index < 0 || index >= N)
                throw new System.ArgumentOutOfRangeException("BasisVector: Index out of bounds");

            vec[index] = (long)1;

            return vec;
        }

        public static longN longRandomVec(this ref Arena arena, int N, long min, long max, uint seed = 84115)
        {
            var vec = arena.longVec(N, true);

            Random random = new Random(seed);

            if (max >= min) {
                for (int i = 0; i < N; i++)
                    vec[i] = (long)random.NextInt((int)min, (int)max);
            }
            else {
                // max < min: pass the smaller bound first so NextInt gets a valid [lo, hi) range.
                // Previously passed (min, max) here, where min > max — Unity.Mathematics NextInt
                // then computed (max - min) as a negative span and returned garbage.
                for (int i = N - 1; i >= 0; i--)
                    vec[i] = (long)random.NextInt((int)max, (int)min);
            }

            return vec;
        }

        public static longN longLinVec(this ref Arena arena, int N, long start, long end)
        {
            var vec = arena.longVec(N);

            // N == 1 would divide by (N-1) == 0 -> Inf -> NaN -> garbage int. Match the guarded
            // fProxyGen_OP.linspace convention: a single sample returns {start}.
            if (N == 1) { vec[0] = start; return vec; }

            float scale = 1 / (float)(N - 1);
            for(int i = 0; i < N; i++) {
                vec[i] = (long)math.lerp((long)start, (long)end, i * scale);
            }
            // Pin endpoints exactly (the lerp at the last index lands ~1 ulp short of end,
            // which can truncate to the wrong integer).
            vec[0] = start;
            vec[N - 1] = end;

            return vec;
        }

        #endregion

        #region MATRIX
        public static longMxN longIdentityMat(this ref Arena arena, int N)
        {
            var matrix = arena.longMat(N, N);

            for (int i = 0; i < N; i++)
                matrix[i, i] = 1;
            
            return matrix;
        }

        public static longMxN longDiagonalMat(this ref Arena arena, int N, long s)
        {
            var matrix = arena.longMat(N, N);

            for (int i = 0; i < N; i++)
                matrix[i, i] = s;

            return matrix;
        }

        public static longMxN longDiagonalMat(this ref Arena arena, in longN vec)
        {
            var matrix = arena.longMat(vec.N, vec.N);

            for (int i = 0; i < vec.N; i++)
                matrix[i, i] = vec[i];

            return matrix;
        }

        public static longMxN longIndexZeroMat(this ref Arena arena, int M_rows, int N_cols)
        {
            var mat = arena.longMat(M_rows, N_cols, true);

            int len = mat.Length;

            unsafe
            {
                mathUnsafelong.setIndexZero(mat.Data.Ptr, len);
            }
            
            return mat;
        }

        public static longMxN longIndexOneMat(this ref Arena arena, int M_rows, int N_cols)
        {
            var mat = arena.longMat(M_rows, N_cols, true);

            int len = mat.Length;

            unsafe
            {
                mathUnsafelong.setIndexOne(mat.Data.Ptr, len);
            }

            return mat;
        }

        public static longMxN longRandomMat(this ref Arena arena, int M_rows, int N_cols, uint seed = 121312)
        {
            return longRandomMat(ref arena, M_rows, N_cols, -1, 1, seed);
        }

        // constructs diagonal matrix with scalar s on diagonal
        public static longMxN longRandomDiagonalMat(this ref Arena arena, int N, long min, long max, uint seed = 65792)
        {
            var matrix = arena.longMat(N, N);

            Random rand = new Random(seed);
            if (max >= min) {
                for (int i = 0; i < N; i++)
                    matrix[i, i] = (long)rand.NextInt((int)min, (int)max);
            }
            else {
                // max < min: pass the smaller bound first so NextInt gets a valid [lo, hi) range
                // (previously passed the inverted (min, max), yielding garbage).
                for (int i = N - 1; i >= 0; i--)
                    matrix[i, i] = (long)rand.NextInt((int)max, (int)min);
            }

            return matrix;
        }

        public static longMxN longRandomMat(this ref Arena arena, int M_rows, int N_cols, long min, long max, uint seed = 121312)
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

        public static longMxN longPermutationMat(this ref Arena arena, int M, int i, int j)
        {
            var matrix = arena.longIdentityMat(M);

            if (M < 2)
                throw new System.ArgumentException("PermutationMatrix: Matrix must be at least 2x2");

            if (i < 0 || i >= M || j < 0 || j >= M)
                throw new System.ArgumentOutOfRangeException("PermutationMatrix: Index out of bounds");

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