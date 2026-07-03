using Unity.Mathematics;

namespace LinearAlgebra
{

    public static partial class ArenaExtensions {

        #region VECTOR
        public static fProxyN fProxyIndexZeroVec(this ref Arena arena, int N)
        {
            var vec = arena.fProxyVec(N, true);

            unsafe {
                mathUnsafefProxy.setIndexZero(vec.Data.Ptr, N);
                
            }
            return vec;
        }

        public static fProxyN fProxyIndexOneVec(this ref Arena arena, int N)
        {
            var vec = arena.fProxyVec(N, true);

            unsafe {
                mathUnsafefProxy.setIndexOne(vec.Data.Ptr, N);
            }
            return vec;
        }

        // all zero but the index is one
        public static fProxyN fProxyBasisVec(this ref Arena arena, int N, int index)
        {
            var vec = arena.fProxyVec(N);

            if(index < 0 || index >= N)
                throw new System.ArgumentOutOfRangeException("BasisVector: Index out of bounds");

            vec[index] = 1f;

            return vec;
        }

        public static fProxyN fProxyRandomUnitVec(this ref Arena arena, int N, uint seed = 34215)
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

            fProxyComp.mulInpl(vec, scale);

            return vec;
        }

        public static fProxyN fProxyRandomVec(this ref Arena arena, int N, fProxy min, fProxy max, uint seed = 34215)
        {
            var vec = arena.fProxyVec(N, true);

            Random random = new Random(seed);

            for (int i = 0; i < vec.N; i++)
                vec[i] = random.NextFProxy(min, max);

            return vec;
        }

        // Legacy name for fProxyLinspace(a, b, N); delegates to the guarded Generate.linspace (handles N==1, pins both endpoints exactly).
        public static fProxyN fProxyLinVec(this ref Arena arena, int N, fProxy start, fProxy end)
        {
            var vec = arena.fProxyVec(N);
            Generate.linspace(ref vec, start, end);
            return vec;
        }

        #endregion

        #region MATRIX
        public static fProxyMxN fProxyIdentityMat(this ref Arena arena, int N)
        {
            var matrix = arena.fProxyMat(N, N);

            for (int i = 0; i < N; i++)
                matrix[i, i] = 1;
            

            return matrix;
        }

        public static fProxyMxN fProxyDiagonalMat(this ref Arena arena, int N, fProxy s)
        {
            var matrix = arena.fProxyMat(N, N);

            for (int i = 0; i < N; i++)
                matrix[i, i] = s;

            return matrix;
        }

        public static fProxyMxN fProxyDiagonalMat(this ref Arena arena, in fProxyN vec)
        {
            var matrix = arena.fProxyMat(vec.N, vec.N);

            for (int i = 0; i < vec.N; i++)
                matrix[i, i] = vec[i];

            return matrix;
        }

        public static fProxyMxN fProxyIndexZeroMat(this ref Arena arena, int M_rows, int N_cols)
        {
            var mat = arena.fProxyMat(M_rows, N_cols, true);

            int len = mat.Length;

            unsafe
            {
                mathUnsafefProxy.setIndexZero(mat.Data.Ptr, len);
            }
            
            return mat;
        }

        public static fProxyMxN fProxyIndexOneMat(this ref Arena arena, int M_rows, int N_cols)
        {
            var mat = arena.fProxyMat(M_rows, N_cols, true);

            int len = mat.Length;

            unsafe
            {
                mathUnsafefProxy.setIndexOne(mat.Data.Ptr, len);
            }

            return mat;
        }

        public static fProxyMxN fProxyRandomMat(this ref Arena arena, int M_rows, int N_cols, uint seed = 121312)
        {
            return fProxyRandomMat(ref arena, M_rows, N_cols, -1, 1, seed);
        }

        // constructs diagonal matrix with scalar s on diagonal
        public static fProxyMxN fProxyRandomDiagonalMat(this ref Arena arena, int N, fProxy min, fProxy max, uint seed = 65792)
        {
            var matrix = arena.fProxyMat(N, N);

            Random rand = new Random(seed);

            for (int i = 0; i < N; i++)
                matrix[i, i] = rand.NextFProxy(min, max);

            return matrix;
        }

        public static fProxyMxN fProxyRandomMat(this ref Arena arena, int M_rows, int N_cols, fProxy min, fProxy max, uint seed = 121312)
        {
            var matrix = arena.fProxyMat(M_rows, N_cols, true);

            Random random = new Random(seed);

            int len = matrix.Length;
            for (int i = 0; i < len; i++)
                matrix[i] = random.NextFProxy(min, max);

            return matrix;
        }

        public static fProxyMxN fProxyRotationMat(this ref Arena arena, int M, int i, int j, fProxy radians)
        {
            var matrix = arena.fProxyIdentityMat(M);

            if (M < 2)
                throw new System.ArgumentException("RotationMatrix: Matrix must be at least 2x2");

            if(i < 0 || i >= M || j < 0 || j >= M)
                throw new System.ArgumentOutOfRangeException("RotationMatrix: Index out of bounds");

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

        public static fProxyMxN fProxyPermutationMat(this ref Arena arena, int M, int i, int j)
        {
            var matrix = arena.fProxyIdentityMat(M);

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

        public static fProxyMxN fProxyHouseholderMat(this ref Arena arena, int M, in fProxyN v)
        {
            if(M < 2)
                throw new System.ArgumentException("HouseholderMatrix: Matrix must be at least 2x2");

            // Compute the Householder matrix: H = I - 2 * vvT / (vTv)
            if (v.N != M)
                throw new System.ArgumentException("HouseholderMatrix: Vector length must match matrix dimension.");

            var matrix = arena.fProxyIdentityMat(M);

            // Compute the outer product of v
            fProxy vTv = Blas.dot(v, v);
            
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
        public static fProxyMxN fProxyHilbertMat(this ref Arena arena, int M)
        {
            if (M < 2)
                throw new System.ArgumentException("HilbertMatrix: Matrix must be at least 2x2");

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