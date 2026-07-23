using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // Standalone (non-arena) vector/matrix generators. Same semantics as ArenaExtensions.cs's
    // bool factories but allocate their own buffer via allocator instead of drawing from an arena.
    public static partial class GenerateOP {

        #region VECTOR

        public static boolN boolRandomVec(int N, uint seed = 34215, Allocator allocator = Allocator.Temp)
        {
            var vec = new boolN(N, allocator, true);

            Random random = new Random(seed);

            for (int i = 0; i < vec.N; i++)
                vec[i] = random.NextBool();

            return vec;
        }

        #endregion

        #region MATRIX

        public static boolMxN boolRandomMat(int M_rows, int N_cols, uint seed = 121312, Allocator allocator = Allocator.Temp)
        {
            var matrix = new boolMxN(M_rows, N_cols, allocator, true);

            Random random = new Random(seed);

            int len = matrix.Length;
            for (int i = 0; i < len; i++)
                matrix[i] = random.NextBool();

            return matrix;
        }

        #endregion

    }

}
