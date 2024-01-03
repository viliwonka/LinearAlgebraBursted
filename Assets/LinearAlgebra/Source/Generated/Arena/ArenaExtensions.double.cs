using Unity.Mathematics;

namespace LinearAlgebra
{

    public static partial class ArenaExtensions {

        #region VECTOR
        public static doubleN doubleIndexZeroVector(this ref Arena arena, int N)
        {
            var vec = arena.doubleVec(N, true);

            unsafe {
                mathUnsafedouble.setIndexZero(vec.Data.Ptr, N);
                
            }
            return vec;
        }

        public static doubleN doubleIndexOneVector(this ref Arena arena, int N)
        {
            var vec = arena.doubleVec(N, true);

            unsafe {
                mathUnsafedouble.setIndexOne(vec.Data.Ptr, N);
            }
            return vec;
        }

        // all zero but the index is one
        public static doubleN doubleBasisVector(this ref Arena arena, int N, int index)
        {
            var vec = arena.doubleVec(N);

            if(index < 0 || index >= N)
                throw new System.Exception("BasisVector: Index out of bounds");

            vec[index] = 1f;

            return vec;
        }

        public static doubleN doubleRandomUnitVector(this ref Arena arena, int N, uint seed = 34215)
        {
            var vec = arena.doubleVec(N, true);

            Random random = new Random(seed);

            double sum = 0;
            for (int i = 0; i < vec.N; i++)
            {
                double p = random.NextDouble(-1f, 1f);
                sum += p*p;
                vec[i] = p;
            }

            double scale = 1 / math.sqrt(sum);

            doubleOP.mulInpl(vec, scale);

            return vec;
        }

        public static doubleN doubleRandomVector(this ref Arena arena, int N, double min, double max, uint seed = 34215)
        {
            var vec = arena.doubleVec(N, true);

            Random random = new Random(seed);

            for (int i = 0; i < vec.N; i++)
                vec[i] = random.NextDouble(min, max);

            return vec;
        }

        //linspace
        public static doubleN doubleLinVector(this ref Arena arena, int N, double start, double end)
        {
            var vec = arena.doubleVec(N);

            double scale = 1 / (double)(N - 1);
            for(int i = 0; i < N; i++) {
                vec[i] = math.lerp(start, end, i * scale);
            }

            return vec;
        }

        #endregion

        #region MATRIX
        // constructs identity matrix
        public static doubleMxN doubleIdentityMatrix(this ref Arena arena, int N)
        {
            var matrix = arena.doubleMat(N, N);

            for (int i = 0; i < N; i++)
                matrix[i, i] = 1;
            

            return matrix;
        }

        // constructs diagonal matrix with scalar s on diagonal
        public static doubleMxN doubleDiagonalMatrix(this ref Arena arena, int N, double s)
        {
            var matrix = arena.doubleMat(N, N);

            for (int i = 0; i < N; i++)
                matrix[i, i] = s;

            return matrix;
        }

        // constructs diagonal matrix based on vector
        public static doubleMxN doubleDiagonalMatrix(this ref Arena arena, in doubleN vec)
        {
            var matrix = arena.doubleMat(vec.N, vec.N);

            for (int i = 0; i < vec.N; i++)
                matrix[i, i] = vec[i];

            return matrix;
        }

        // constructs matrix with indexes that start at 0
        public static doubleMxN doubleIndexZeroMatrix(this ref Arena arena, int M_rows, int N_cols)
        {
            var mat = arena.doubleMat(M_rows, N_cols, true);

            int len = mat.Length;

            unsafe
            {
                mathUnsafedouble.setIndexZero(mat.Data.Ptr, len);
            }
            
            return mat;
        }

        // constructs matrix with indexes that start at 1
        public static doubleMxN doubleIndexOneMatrix(this ref Arena arena, int M_rows, int N_cols)
        {
            var mat = arena.doubleMat(M_rows, N_cols, true);

            int len = mat.Length;

            unsafe
            {
                mathUnsafedouble.setIndexOne(mat.Data.Ptr, len);
            }

            return mat;
        }

        // random matrix

        public static doubleMxN doubleRandomMatrix(this ref Arena arena, int M_rows, int N_cols, uint seed = 121312)
        {
            return doubleRandomMatrix(ref arena, M_rows, N_cols, -1, 1, seed);
        }

        // constructs diagonal matrix with scalar s on diagonal
        public static doubleMxN doubleRandomDiagonalMatrix(this ref Arena arena, int N, double min, double max, uint seed = 65792)
        {
            var matrix = arena.doubleMat(N, N);

            Random rand = new Random(seed);

            for (int i = 0; i < N; i++)
                matrix[i, i] = rand.NextDouble(min, max);

            return matrix;
        }

        public static doubleMxN doubleRandomMatrix(this ref Arena arena, int M_rows, int N_cols, double min, double max, uint seed = 121312)
        {
            var matrix = arena.doubleMat(M_rows, N_cols, true);

            Random random = new Random(seed);

            int len = matrix.Length;
            for (int i = 0; i < len; i++)
                matrix[i] = random.NextDouble(min, max);

            return matrix;
        }

        // i and j are axis indexes to rotate
        public static doubleMxN doubleRotationMatrix(this ref Arena arena, int M, int i, int j, double radians)
        {
            var matrix = arena.doubleIdentityMatrix(M);

            if (M < 2)
                throw new System.Exception("RotationMatrix: Matrix must be at least 2x2");

            if(i < 0 || i >= M || j < 0 || j >= M)
                throw new System.Exception("RotationMatrix: Index out of bounds");

            if(i == j) {
                return matrix;
            }

            double c = math.cos(radians);
            double s = math.sin(radians);

            matrix[i, i] = c;
            matrix[j, j] = c;
            matrix[i, j] = -s;
            matrix[j, i] = s;

            return matrix;
        }

        // i and j are axis indexes to swap
        public static doubleMxN doublePermutationMatrix(this ref Arena arena, int M, int i, int j)
        {
            var matrix = arena.doubleIdentityMatrix(M);

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

        public static doubleMxN doubleHouseholderMatrix(this ref Arena arena, int M, in doubleN v)
        {
            if(M < 2)
                throw new System.Exception("HouseholderMatrix: Matrix must be at least 2x2");

            // Compute the Householder matrix: H = I - 2 * vvT / (vTv)
            if (v.N != M)
                throw new System.Exception("HouseholderMatrix: Vector length must match matrix dimension.");

            var matrix = arena.doubleIdentityMatrix(M);

            // Compute the outer product of v
            double vTv = doubleOP.dot(v, v);
            
            double scaleFactor = 2 / vTv;
            
            // Rank 1 update
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < M; j++)
                {
                    double vvT_element = scaleFactor * v[i] * v[j];
                    matrix[i, j] -= vvT_element;
                }
            }

            return matrix;
        }

        // very ill conditioned matrix, used for testing numerical stability
        public static doubleMxN doubleHilbertMatrix(this ref Arena arena, int M)
        {
            if (M < 2)
                throw new System.Exception("HilbertMatrix: Matrix must be at least 2x2");

            var hilbert = arena.doubleMat(M, true);

            for(int i = 0; i < M; i++) {
                for (int j = 0; j < M; j++) {
                    hilbert[i, j] = (double) 1.0 / (double)(i + j + 1);
                }
            }

            return hilbert;
        }

        #endregion

    }

}