using Unity.Mathematics;

namespace LinearAlgebra
{

    public static partial class ArenaExtensions {

        #region VECTOR
        
        public static boolN boolRandomVec(this ref Arena arena, int N, uint seed = 34215)
        {
            var vec = arena.BoolVector(N, true);

            Random random = new Random(seed);

            for (int i = 0; i < vec.N; i++)
                vec[i] = random.NextBool();

            return vec;
        }

        #endregion

        #region MATRIX

        public static boolMxN boolRandomMat(this ref Arena arena, int M_rows, int N_cols, uint seed = 121312)
        {
            var matrix = arena.BoolMatrix(M_rows, N_cols, true);

            Random random = new Random(seed);

            int len = matrix.Length;
            for (int i = 0; i < len; i++)
                matrix[i] = random.NextBool();

            return matrix;
        }

        #endregion

    }

}