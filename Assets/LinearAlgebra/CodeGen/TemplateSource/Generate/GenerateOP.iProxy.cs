using Unity.Collections;
using Unity.Mathematics;
using BULA.Internal;

//alsoExpand[uint]// standalone construction convenience methods. The default-range RandomMat overload
//hardcodes a signed [-1, 1] range that has no unsigned equivalent - see the skipFor-marked block
//below (do not write that marker's literal token here - the codegen parser is content-sensitive,
//not comment-aware, and would treat this doc comment as a real marker); everything else here takes
//its range as explicit params, so it's unsigned-clean.

namespace BULA
{
    // Standalone vector/matrix generators: allocate their own buffer via allocator.
    public static partial class GenerateOP {

        #region VECTOR
        public static iProxyN iProxyIndexZeroVec(int N, Allocator allocator = Allocator.Temp)
        {
            var vec = new iProxyN(N, allocator, true);

            unsafe {
                UnsafeMathOP.setIndexZero(vec.Data.Ptr, N);

            }
            return vec;
        }

        public static iProxyN iProxyIndexOneVec(int N, Allocator allocator = Allocator.Temp)
        {
            var vec = new iProxyN(N, allocator, true);

            unsafe {
                UnsafeMathOP.setIndexOne(vec.Data.Ptr, N);
            }
            return vec;
        }

        // all zero but the index is one
        public static iProxyN iProxyBasisVec(int N, int index, Allocator allocator = Allocator.Temp)
        {
            var vec = new iProxyN(N, allocator);

            if(index < 0 || index >= N)
                throw new System.ArgumentOutOfRangeException("iProxyBasisVec: Index out of bounds");

            vec[index] = (iProxy)1;

            return vec;
        }

        // NOTE: min/max are cast to int for NextInt - for uint, bounds above int.MaxValue are unsupported (the cast wraps).
        public static iProxyN iProxyRandomVec(int N, iProxy min, iProxy max, uint seed = 84115, Allocator allocator = Allocator.Temp)
        {
            var vec = new iProxyN(N, allocator, true);

            Random random = new Random(seed);

            if (max >= min) {
                for (int i = 0; i < N; i++)
                    vec[i] = (iProxy)random.NextInt((int)min, (int)max);
            }
            else {
                // max < min: pass the smaller bound first so NextInt gets a valid [lo, hi) range.
                for (int i = N - 1; i >= 0; i--)
                    vec[i] = (iProxy)random.NextInt((int)max, (int)min);
            }

            return vec;
        }

        public static iProxyN iProxyLinVec(int N, iProxy start, iProxy end, Allocator allocator = Allocator.Temp)
        {
            var vec = new iProxyN(N, allocator);

            // N == 1 would divide by (N-1) == 0 -> Inf -> NaN -> garbage int. Match the guarded
            // Generate.linspace convention: a single sample returns {start}.
            if (N == 1) { vec[0] = start; return vec; }

            // Interpolate in double: exact for every int/uint value and for long magnitudes
            // up to 2^53 (float's 24-bit mantissa corrupts interior values).
            double scale = 1.0 / (N - 1);
            for(int i = 0; i < N; i++) {
                vec[i] = (iProxy)math.lerp((double)start, (double)end, i * scale);
            }
            // Pin endpoints exactly (the lerp at the last index lands ~1 ulp short of end,
            // which can truncate to the wrong integer).
            vec[0] = start;
            vec[N - 1] = end;

            return vec;
        }

        // Returns a new N-vector with every element set to s.
        public static iProxyN iProxyVec(int N, iProxy s, Allocator allocator = Allocator.Temp)
        {
            var vec = new iProxyN(N, allocator, true);
            unsafe {
                UnsafeMathOP.setAll(vec.Data.Ptr, N, s);
            }
            return vec;
        }

        #endregion

        #region MATRIX
        public static iProxyMxN iProxyIdentityMat(int N, Allocator allocator = Allocator.Temp)
        {
            var matrix = new iProxyMxN(N, N, allocator);

            for (int i = 0; i < N; i++)
                matrix[i, i] = 1;

            return matrix;
        }

        public static iProxyMxN iProxyDiagonalMat(int N, iProxy s, Allocator allocator = Allocator.Temp)
        {
            var matrix = new iProxyMxN(N, N, allocator);

            for (int i = 0; i < N; i++)
                matrix[i, i] = s;

            return matrix;
        }

        public static iProxyMxN iProxyDiagonalMat(in iProxyN vec, Allocator allocator = Allocator.Temp)
        {
            var matrix = new iProxyMxN(vec.N, vec.N, allocator);

            for (int i = 0; i < vec.N; i++)
                matrix[i, i] = vec[i];

            return matrix;
        }

        public static iProxyMxN iProxyIndexZeroMat(int M_rows, int N_cols, Allocator allocator = Allocator.Temp)
        {
            var mat = new iProxyMxN(M_rows, N_cols, allocator, true);

            int len = mat.Length;

            unsafe
            {
                UnsafeMathOP.setIndexZero(mat.Data.Ptr, len);
            }

            return mat;
        }

        public static iProxyMxN iProxyIndexOneMat(int M_rows, int N_cols, Allocator allocator = Allocator.Temp)
        {
            var mat = new iProxyMxN(M_rows, N_cols, allocator, true);

            int len = mat.Length;

            unsafe
            {
                UnsafeMathOP.setIndexOne(mat.Data.Ptr, len);
            }

            return mat;
        }

        // This default-range overload hardcodes a symmetric [-1, 1] range - literal -1 is out of
        // range for an unsigned type, so this overload doesn't exist for uint; uint callers must
        // use the explicit min/max overload below instead.
        //+skipFor[u]
        public static iProxyMxN iProxyRandomMat(int M_rows, int N_cols, uint seed = 121312, Allocator allocator = Allocator.Temp)
        {
            return iProxyRandomMat(M_rows, N_cols, -1, 1, seed, allocator);
        }
        //-skipFor

        // constructs diagonal matrix with random diagonal entries in [min, max]
        // NOTE: min/max are cast to int for NextInt - for uint, bounds above int.MaxValue are unsupported (the cast wraps).
        public static iProxyMxN iProxyRandomDiagonalMat(int N, iProxy min, iProxy max, uint seed = 65792, Allocator allocator = Allocator.Temp)
        {
            var matrix = new iProxyMxN(N, N, allocator);

            Random rand = new Random(seed);
            if (max >= min) {
                for (int i = 0; i < N; i++)
                    matrix[i, i] = (iProxy)rand.NextInt((int)min, (int)max);
            }
            else {
                // max < min: pass the smaller bound first so NextInt gets a valid [lo, hi) range.
                for (int i = N - 1; i >= 0; i--)
                    matrix[i, i] = (iProxy)rand.NextInt((int)max, (int)min);
            }

            return matrix;
        }

        // NOTE: min/max are cast to int for NextInt - for uint, bounds above int.MaxValue are unsupported (the cast wraps).
        public static iProxyMxN iProxyRandomMat(int M_rows, int N_cols, iProxy min, iProxy max, uint seed = 121312, Allocator allocator = Allocator.Temp)
        {
            var matrix = new iProxyMxN(M_rows, N_cols, allocator, true);

            Random random = new Random(seed);

            int len = matrix.Length;

            if (max >= min) {
                for (int i = 0; i < len; i++)
                    matrix[i] = (iProxy)random.NextInt((int)min, (int)max);
            }
            else {

                for (int i = len - 1; i >= 0; i--)
                    matrix[i] = (iProxy)random.NextInt((int)max, (int)min);
            }

            return matrix;
        }

        public static iProxyMxN iProxyPermutationMat(int M, int i, int j, Allocator allocator = Allocator.Temp)
        {
            var matrix = iProxyIdentityMat(M, allocator);

            if (M < 2)
                throw new System.ArgumentException("iProxyPermutationMat: Matrix must be at least 2x2");

            if (i < 0 || i >= M || j < 0 || j >= M)
                throw new System.ArgumentOutOfRangeException("iProxyPermutationMat: Index out of bounds");

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

        // Returns a new M_rows x N_cols matrix with every element set to s.
        public static iProxyMxN iProxyMat(int M_rows, int N_cols, iProxy s, Allocator allocator = Allocator.Temp)
        {
            var matrix = new iProxyMxN(M_rows, N_cols, allocator, false);
            unsafe
            {
                UnsafeMathOP.setAll(matrix.Data.Ptr, matrix.Length, s);
            }
            return matrix;
        }

        #endregion

    }

}
