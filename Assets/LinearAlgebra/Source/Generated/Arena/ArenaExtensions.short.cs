using Unity.Mathematics;

using UnityEngine.UIElements;

namespace LinearAlgebra
{
    public static partial class ArenaExtensions {

        #region VECTOR
        public static shortN shortIndexZeroVec(this ref Arena arena, int N)
        {
            var vec = arena.shortVec(N, true);

            unsafe {
                mathUnsafeshort.setIndexZero(vec.Data.Ptr, N);
                
            }
            return vec;
        }

        public static shortN shortIndexOneVec(this ref Arena arena, int N)
        {
            var vec = arena.shortVec(N, true);

            unsafe {
                mathUnsafeshort.setIndexOne(vec.Data.Ptr, N);
            }
            return vec;
        }

        // all zero but the index is one
        public static shortN shortBasisVec(this ref Arena arena, int N, int index)
        {
            var vec = arena.shortVec(N);

            if(index < 0 || index >= N)
                throw new System.ArgumentOutOfRangeException("BasisVector: Index out of bounds");

            vec[index] = (short)1;

            return vec;
        }

        public static shortN shortRandomVec(this ref Arena arena, int N, short min, short max, uint seed = 84115)
        {
            var vec = arena.shortVec(N, true);

            Random random = new Random(seed);

            if (max >= min) {
                for (int i = 0; i < N; i++)
                    vec[i] = (short)random.NextInt((int)min, (int)max);
            }
            else {
                // max < min: pass the smaller bound first so NextInt gets a valid [lo, hi) range.
                // Previously passed (min, max) here, where min > max — Unity.Mathematics NextInt
                // then computed (max - min) as a negative span and returned garbage.
                for (int i = N - 1; i >= 0; i--)
                    vec[i] = (short)random.NextInt((int)max, (int)min);
            }

            return vec;
        }

        public static shortN shortLinVec(this ref Arena arena, int N, short start, short end)
        {
            var vec = arena.shortVec(N);

            // N == 1 would divide by (N-1) == 0 -> Inf -> NaN -> garbage int. Match the guarded
            // fProxyGen_OP.linspace convention: a single sample returns {start}.
            if (N == 1) { vec[0] = start; return vec; }

            float scale = 1 / (float)(N - 1);
            for(int i = 0; i < N; i++) {
                vec[i] = (short)math.lerp((short)start, (short)end, i * scale);
            }
            // Pin endpoints exactly (the lerp at the last index lands ~1 ulp short of end,
            // which can truncate to the wrong integer).
            vec[0] = start;
            vec[N - 1] = end;

            return vec;
        }

        #endregion

        #region MATRIX
        public static shortMxN shortIdentityMat(this ref Arena arena, int N)
        {
            var matrix = arena.shortMat(N, N);

            for (int i = 0; i < N; i++)
                matrix[i, i] = 1;
            
            return matrix;
        }

        public static shortMxN shortDiagonalMat(this ref Arena arena, int N, short s)
        {
            var matrix = arena.shortMat(N, N);

            for (int i = 0; i < N; i++)
                matrix[i, i] = s;

            return matrix;
        }

        public static shortMxN shortDiagonalMat(this ref Arena arena, in shortN vec)
        {
            var matrix = arena.shortMat(vec.N, vec.N);

            for (int i = 0; i < vec.N; i++)
                matrix[i, i] = vec[i];

            return matrix;
        }

        public static shortMxN shortIndexZeroMat(this ref Arena arena, int M_rows, int N_cols)
        {
            var mat = arena.shortMat(M_rows, N_cols, true);

            int len = mat.Length;

            unsafe
            {
                mathUnsafeshort.setIndexZero(mat.Data.Ptr, len);
            }
            
            return mat;
        }

        public static shortMxN shortIndexOneMat(this ref Arena arena, int M_rows, int N_cols)
        {
            var mat = arena.shortMat(M_rows, N_cols, true);

            int len = mat.Length;

            unsafe
            {
                mathUnsafeshort.setIndexOne(mat.Data.Ptr, len);
            }

            return mat;
        }

        public static shortMxN shortRandomMat(this ref Arena arena, int M_rows, int N_cols, uint seed = 121312)
        {
            return shortRandomMat(ref arena, M_rows, N_cols, -1, 1, seed);
        }

        // constructs diagonal matrix with scalar s on diagonal
        public static shortMxN shortRandomDiagonalMat(this ref Arena arena, int N, short min, short max, uint seed = 65792)
        {
            var matrix = arena.shortMat(N, N);

            Random rand = new Random(seed);
            if (max >= min) {
                for (int i = 0; i < N; i++)
                    matrix[i, i] = (short)rand.NextInt((int)min, (int)max);
            }
            else {
                // max < min: pass the smaller bound first so NextInt gets a valid [lo, hi) range
                // (previously passed the inverted (min, max), yielding garbage).
                for (int i = N - 1; i >= 0; i--)
                    matrix[i, i] = (short)rand.NextInt((int)max, (int)min);
            }

            return matrix;
        }

        public static shortMxN shortRandomMat(this ref Arena arena, int M_rows, int N_cols, short min, short max, uint seed = 121312)
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

        public static shortMxN shortPermutationMat(this ref Arena arena, int M, int i, int j)
        {
            var matrix = arena.shortIdentityMat(M);

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