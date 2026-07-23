using Unity.Collections;
using Unity.Mathematics;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    // Standalone (non-arena) vector/matrix generators. Same semantics as ArenaExtensions.fProxy
    // but allocate their own buffer via allocator instead of drawing from an arena.
    public static partial class GenerateOP {

        #region VECTOR
        public static fProxyN fProxyIndexZeroVec(int N, Allocator allocator = Allocator.Temp)
        {
            var vec = new fProxyN(N, allocator, true);

            unsafe {
                UnsafeMathOP.setIndexZero(vec.Data.Ptr, N);

            }
            return vec;
        }

        public static fProxyN fProxyIndexOneVec(int N, Allocator allocator = Allocator.Temp)
        {
            var vec = new fProxyN(N, allocator, true);

            unsafe {
                UnsafeMathOP.setIndexOne(vec.Data.Ptr, N);
            }
            return vec;
        }

        // all zero but the index is one
        public static fProxyN fProxyBasisVec(int N, int index, Allocator allocator = Allocator.Temp)
        {
            var vec = new fProxyN(N, allocator);

            if(index < 0 || index >= N)
                throw new System.ArgumentOutOfRangeException("fProxyBasisVec: Index out of bounds");

            vec[index] = 1f;

            return vec;
        }

        public static fProxyN fProxyRandomUnitVec(int N, uint seed = 34215, Allocator allocator = Allocator.Temp)
        {
            var vec = new fProxyN(N, allocator, true);

            Random random = new Random(seed);

            fProxy sum = 0;
            for (int i = 0; i < vec.N; i++)
            {
                fProxy p = random.NextFProxy(-1f, 1f);
                sum += p*p;
                vec[i] = p;
            }

            fProxy scale = 1 / math.sqrt(sum);

            fProxyComp.mulInPlace(vec, scale);

            return vec;
        }

        public static fProxyN fProxyRandomVec(int N, fProxy min, fProxy max, uint seed = 34215, Allocator allocator = Allocator.Temp)
        {
            var vec = new fProxyN(N, allocator, true);

            Random random = new Random(seed);

            for (int i = 0; i < vec.N; i++)
                vec[i] = random.NextFProxy(min, max);

            return vec;
        }

        // Alias for fProxyLinspace(a, b, N); delegates to the guarded Generate.linspace (handles N==1, pins both endpoints exactly).
        public static fProxyN fProxyLinVec(int N, fProxy start, fProxy end, Allocator allocator = Allocator.Temp)
        {
            var vec = new fProxyN(N, allocator);
            Generate.linspace(ref vec, start, end);
            return vec;
        }

        // Returns a new N-vector with every element set to s.
        public static fProxyN fProxyVec(int N, fProxy s, Allocator allocator = Allocator.Temp)
        {
            var vec = new fProxyN(N, allocator, true);
            unsafe {
                UnsafeMathOP.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        #endregion

        #region MATRIX
        public static fProxyMxN fProxyIdentityMat(int N, Allocator allocator = Allocator.Temp)
        {
            var matrix = new fProxyMxN(N, N, allocator);

            for (int i = 0; i < N; i++)
                matrix[i, i] = 1;


            return matrix;
        }

        public static fProxyMxN fProxyDiagonalMat(int N, fProxy s, Allocator allocator = Allocator.Temp)
        {
            var matrix = new fProxyMxN(N, N, allocator);

            for (int i = 0; i < N; i++)
                matrix[i, i] = s;

            return matrix;
        }

        public static fProxyMxN fProxyDiagonalMat(in fProxyN vec, Allocator allocator = Allocator.Temp)
        {
            var matrix = new fProxyMxN(vec.N, vec.N, allocator);

            for (int i = 0; i < vec.N; i++)
                matrix[i, i] = vec[i];

            return matrix;
        }

        public static fProxyMxN fProxyIndexZeroMat(int M_rows, int N_cols, Allocator allocator = Allocator.Temp)
        {
            var mat = new fProxyMxN(M_rows, N_cols, allocator, true);

            int len = mat.Length;

            unsafe
            {
                UnsafeMathOP.setIndexZero(mat.Data.Ptr, len);
            }

            return mat;
        }

        public static fProxyMxN fProxyIndexOneMat(int M_rows, int N_cols, Allocator allocator = Allocator.Temp)
        {
            var mat = new fProxyMxN(M_rows, N_cols, allocator, true);

            int len = mat.Length;

            unsafe
            {
                UnsafeMathOP.setIndexOne(mat.Data.Ptr, len);
            }

            return mat;
        }

        public static fProxyMxN fProxyRandomMat(int M_rows, int N_cols, uint seed = 121312, Allocator allocator = Allocator.Temp)
        {
            return fProxyRandomMat(M_rows, N_cols, -1, 1, seed, allocator);
        }

        // constructs diagonal matrix with random diagonal entries in [min, max]
        public static fProxyMxN fProxyRandomDiagonalMat(int N, fProxy min, fProxy max, uint seed = 65792, Allocator allocator = Allocator.Temp)
        {
            var matrix = new fProxyMxN(N, N, allocator);

            Random rand = new Random(seed);

            for (int i = 0; i < N; i++)
                matrix[i, i] = rand.NextFProxy(min, max);

            return matrix;
        }

        public static fProxyMxN fProxyRandomMat(int M_rows, int N_cols, fProxy min, fProxy max, uint seed = 121312, Allocator allocator = Allocator.Temp)
        {
            var matrix = new fProxyMxN(M_rows, N_cols, allocator, true);

            Random random = new Random(seed);

            int len = matrix.Length;
            for (int i = 0; i < len; i++)
                matrix[i] = random.NextFProxy(min, max);

            return matrix;
        }

        public static fProxyMxN fProxyRotationMat(int M, int i, int j, fProxy radians, Allocator allocator = Allocator.Temp)
        {
            var matrix = fProxyIdentityMat(M, allocator);

            if (M < 2)
                throw new System.ArgumentException("fProxyRotationMat: Matrix must be at least 2x2");

            if(i < 0 || i >= M || j < 0 || j >= M)
                throw new System.ArgumentOutOfRangeException("fProxyRotationMat: Index out of bounds");

            if(i == j) {
                return matrix;
            }

            fProxy c = DetMath.Cos(radians);
            fProxy s = DetMath.Sin(radians);

            matrix[i, i] = c;
            matrix[j, j] = c;
            matrix[i, j] = -s;
            matrix[j, i] = s;

            return matrix;
        }

        public static fProxyMxN fProxyPermutationMat(int M, int i, int j, Allocator allocator = Allocator.Temp)
        {
            var matrix = fProxyIdentityMat(M, allocator);

            if (M < 2)
                throw new System.ArgumentException("fProxyPermutationMat: Matrix must be at least 2x2");

            if (i < 0 || i >= M || j < 0 || j >= M)
                throw new System.ArgumentOutOfRangeException("fProxyPermutationMat: Index out of bounds");

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

        public static fProxyMxN fProxyHouseholderMat(int M, in fProxyN v, Allocator allocator = Allocator.Temp)
        {
            if(M < 2)
                throw new System.ArgumentException("fProxyHouseholderMat: Matrix must be at least 2x2");

            // Compute the Householder matrix: H = I - 2 * vvT / (vTv)
            if (v.N != M)
                throw new System.ArgumentException("fProxyHouseholderMat: Vector length must match matrix dimension.");

            var matrix = fProxyIdentityMat(M, allocator);

            // Compute the outer product of v
            fProxy vTv = Blas.dot(v, v);

            // Degenerate (zero / near-zero) v -> identity transform; matrix is already I. NaN-safe
            // (!(vTv > t) is true for NaN); avoids 2/0 = Inf poisoning the matrix.
            if (!(vTv > Consts.fProxyZeroThreshold))
                return matrix;

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
        public static fProxyMxN fProxyHilbertMat(int M, Allocator allocator = Allocator.Temp)
        {
            if (M < 2)
                throw new System.ArgumentException("fProxyHilbertMat: Matrix must be at least 2x2");

            var hilbert = new fProxyMxN(M, M, allocator, true);

            for(int i = 0; i < M; i++) {
                for (int j = 0; j < M; j++) {
                    hilbert[i, j] = (fProxy) 1.0 / (fProxy)(i + j + 1);
                }
            }

            return hilbert;
        }

        // Returns a new M_rows x N_cols matrix with every element set to s.
        public static fProxyMxN fProxyMat(int M_rows, int N_cols, fProxy s, Allocator allocator = Allocator.Temp)
        {
            var matrix = new fProxyMxN(M_rows, N_cols, allocator, false);
            unsafe
            {
                UnsafeMathOP.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        #endregion

    }

}
