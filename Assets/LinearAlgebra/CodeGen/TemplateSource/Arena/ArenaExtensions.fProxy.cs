using Unity.Mathematics;

namespace LinearAlgebra
{

    public static partial class ArenaExtensions {

        #region VECTOR
        public static fProxyN fProxyIndexZeroVector(this ref Arena arena, int N)
        {
            var vec = arena.fProxyVec(N, true);

            unsafe {
                mathUnsafefProxy.setIndexZero(vec.Data.Ptr, N);
                
            }
            return vec;
        }

        public static fProxyN fProxyIndexOneVector(this ref Arena arena, int N)
        {
            var vec = arena.fProxyVec(N, true);

            unsafe {
                mathUnsafefProxy.setIndexOne(vec.Data.Ptr, N);
            }
            return vec;
        }

        // all zero but the index is one
        public static fProxyN fProxyBasisVector(this ref Arena arena, int N, int index)
        {
            var vec = arena.fProxyVec(N);

            if(index < 0 || index >= N)
                throw new System.Exception("BasisVector: Index out of bounds");

            vec[index] = 1f;

            return vec;
        }

        public static fProxyN fProxyRandomUnitVector(this ref Arena arena, int N, uint seed = 34215)
        {
            var vec = arena.fProxyVec(N, true);

            Random random = new Random(seed);

            fProxy sum = 0;
            for (int i = 0; i < vec.N; i++)
            {
                fProxy p = random.NextFProxy(-1f, 1f);
                sum += p*p;
                vec[i] = p;
            }

            fProxy scale = 1 / math.sqrt(sum);

            fProxyOP.mulInpl(vec, scale);

            return vec;
        }

        public static fProxyN fProxyRandomVector(this ref Arena arena, int N, fProxy min, fProxy max, uint seed = 34215)
        {
            var vec = arena.fProxyVec(N, true);

            Random random = new Random(seed);

            for (int i = 0; i < vec.N; i++)
                vec[i] = random.NextFProxy(min, max);

            return vec;
        }

        //linspace
        public static fProxyN fProxyLinVector(this ref Arena arena, int N, fProxy start, fProxy end)
        {
            var vec = arena.fProxyVec(N);

            fProxy scale = 1 / (fProxy)(N - 1);
            for(int i = 0; i < N; i++) {
                vec[i] = math.lerp(start, end, i * scale);
            }

            return vec;
        }

        #endregion

        #region MATRIX
        // constructs identity matrix
        public static fProxyMxN fProxyIdentityMatrix(this ref Arena arena, int N)
        {
            var matrix = arena.fProxyMat(N, N);

            for (int i = 0; i < N; i++)
                matrix[i, i] = 1;
            

            return matrix;
        }

        // constructs diagonal matrix with scalar s on diagonal
        public static fProxyMxN fProxyDiagonalMatrix(this ref Arena arena, int N, fProxy s)
        {
            var matrix = arena.fProxyMat(N, N);

            for (int i = 0; i < N; i++)
                matrix[i, i] = s;

            return matrix;
        }

        // constructs diagonal matrix based on vector
        public static fProxyMxN fProxyDiagonalMatrix(this ref Arena arena, in fProxyN vec)
        {
            var matrix = arena.fProxyMat(vec.N, vec.N);

            for (int i = 0; i < vec.N; i++)
                matrix[i, i] = vec[i];

            return matrix;
        }

        // constructs matrix with indexes that start at 0
        public static fProxyMxN fProxyIndexZeroMatrix(this ref Arena arena, int M_rows, int N_cols)
        {
            var mat = arena.fProxyMat(M_rows, N_cols, true);

            int len = mat.Length;

            unsafe
            {
                mathUnsafefProxy.setIndexZero(mat.Data.Ptr, len);
            }
            
            return mat;
        }

        // constructs matrix with indexes that start at 1
        public static fProxyMxN fProxyIndexOneMatrix(this ref Arena arena, int M_rows, int N_cols)
        {
            var mat = arena.fProxyMat(M_rows, N_cols, true);

            int len = mat.Length;

            unsafe
            {
                mathUnsafefProxy.setIndexOne(mat.Data.Ptr, len);
            }

            return mat;
        }

        // random matrix

        public static fProxyMxN fProxyRandomMatrix(this ref Arena arena, int M_rows, int N_cols, uint seed = 121312)
        {
            return fProxyRandomMatrix(ref arena, M_rows, N_cols, -1, 1, seed);
        }

        // constructs diagonal matrix with scalar s on diagonal
        public static fProxyMxN fProxyRandomDiagonalMatrix(this ref Arena arena, int N, fProxy min, fProxy max, uint seed = 65792)
        {
            var matrix = arena.fProxyMat(N, N);

            Random rand = new Random(seed);

            for (int i = 0; i < N; i++)
                matrix[i, i] = rand.NextFProxy(min, max);

            return matrix;
        }

        public static fProxyMxN fProxyRandomMatrix(this ref Arena arena, int M_rows, int N_cols, fProxy min, fProxy max, uint seed = 121312)
        {
            var matrix = arena.fProxyMat(M_rows, N_cols, true);

            Random random = new Random(seed);

            int len = matrix.Length;
            for (int i = 0; i < len; i++)
                matrix[i] = random.NextFProxy(min, max);

            return matrix;
        }

        // i and j are axis indexes to rotate
        public static fProxyMxN fProxyRotationMatrix(this ref Arena arena, int M, int i, int j, fProxy radians)
        {
            var matrix = arena.fProxyIdentityMatrix(M);

            if (M < 2)
                throw new System.Exception("RotationMatrix: Matrix must be at least 2x2");

            if(i < 0 || i >= M || j < 0 || j >= M)
                throw new System.Exception("RotationMatrix: Index out of bounds");

            if(i == j) {
                return matrix;
            }

            fProxy c = math.cos(radians);
            fProxy s = math.sin(radians);

            matrix[i, i] = c;
            matrix[j, j] = c;
            matrix[i, j] = -s;
            matrix[j, i] = s;

            return matrix;
        }

        // i and j are axis indexes to swap
        public static fProxyMxN fProxyPermutationMatrix(this ref Arena arena, int M, int i, int j)
        {
            var matrix = arena.fProxyIdentityMatrix(M);

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

        public static fProxyMxN fProxyHouseholderMatrix(this ref Arena arena, int M, in fProxyN v)
        {
            if(M < 2)
                throw new System.Exception("HouseholderMatrix: Matrix must be at least 2x2");

            // Compute the Householder matrix: H = I - 2 * vvT / (vTv)
            if (v.N != M)
                throw new System.Exception("HouseholderMatrix: Vector length must match matrix dimension.");

            var matrix = arena.fProxyIdentityMatrix(M);

            // Compute the outer product of v
            fProxy vTv = fProxyOP.dot(v, v);
            
            fProxy scaleFactor = 2 / vTv;
            
            // Rank 1 update
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < M; j++)
                {
                    fProxy vvT_element = scaleFactor * v[i] * v[j];
                    matrix[i, j] -= vvT_element;
                }
            }

            return matrix;
        }

        // very ill conditioned matrix, used for testing numerical stability
        public static fProxyMxN fProxyHilbertMatrix(this ref Arena arena, int M)
        {
            if (M < 2)
                throw new System.Exception("HilbertMatrix: Matrix must be at least 2x2");

            var hilbert = arena.fProxyMat(M, true);

            for(int i = 0; i < M; i++) {
                for (int j = 0; j < M; j++) {
                    hilbert[i, j] = (fProxy) 1.0 / (fProxy)(i + j + 1);
                }
            }

            return hilbert;
        }

        #endregion

    }

}