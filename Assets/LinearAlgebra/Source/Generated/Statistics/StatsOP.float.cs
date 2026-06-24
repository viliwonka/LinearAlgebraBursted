#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Collections;
using Unity.Mathematics;
using Unity.Collections.LowLevel.Unsafe;
using System;

namespace LinearAlgebra.Stats
{

    // just a prototype, needs matrices handling too
    public static partial class floatStatsOP  {

        public static float sum<T>(in T x) where T : unmanaged, IUnsafefloatArray {

            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute sum of an empty array.");

            if (x.Data.Length == 1)
                return x.Data[0];

            float sum = 0f;
            for (int i = 0; i < x.Data.Length; i++) 
                sum += x.Data[i];
            
            return sum;
        }

        public static float mean<T>(in T x) where T : unmanaged, IUnsafefloatArray {
            return sum(in x) / x.Data.Length;
        }

        public static float variance<T>(in T x) where T : unmanaged, IUnsafefloatArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute variance of an empty array.");

            if (x.Data.Length == 1)
                return 0f;

            float m = mean(x);
            float sum = 0f;
            for (int i = 0; i < x.Data.Length; i++)
            {
                float d = x.Data[i] - m;
                sum += d*d;
            }
            return sum / x.Data.Length;
        }

        public static float stdDev<T>(in T x) where T : unmanaged, IUnsafefloatArray {
            return math.sqrt(variance(x));
        }

        // Sample variance: Σ(xᵢ−mean)²/(n−1). Two-pass (compute mean first, then squared deviations).
        // n==0 → throws; n==1 → throws.
        public static float varianceSample<T>(in T x) where T : unmanaged, IUnsafefloatArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute sample variance of an empty array.");

            if (x.Data.Length == 1)
                throw new InvalidOperationException("Sample variance requires at least 2 elements.");

            float m = mean(in x);
            float sum = 0f;
            for (int i = 0; i < x.Data.Length; i++)
            {
                float d = x.Data[i] - m;
                sum += d * d;
            }
            return sum / (x.Data.Length - 1);
        }

        public static float stdDevSample<T>(in T x) where T : unmanaged, IUnsafefloatArray {
            return math.sqrt(varianceSample(in x));
        }

        // Returns the index of the smallest element (first occurrence on ties).
        // For a matrix (IUnsafefloatArray over row-major data), the index is the linear index r*N_Cols+c.
        // Note: if element 0 is NaN, the result may be index 0 (NaN comparisons are all false); behavior with NaN data is unspecified.
        public static int argmin<T>(in T x) where T : unmanaged, IUnsafefloatArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute argmin of an empty array.");

            float best = x.Data[0];
            int bestIdx = 0;
            for (int i = 1; i < x.Data.Length; i++)
            {
                if (x.Data[i] < best)
                {
                    best = x.Data[i];
                    bestIdx = i;
                }
            }
            return bestIdx;
        }

        // Returns the index of the largest element (first occurrence on ties).
        // For a matrix (IUnsafefloatArray over row-major data), the index is the linear index r*N_Cols+c.
        // Note: if element 0 is NaN, the result may be index 0 (NaN comparisons are all false); behavior with NaN data is unspecified.
        public static int argmax<T>(in T x) where T : unmanaged, IUnsafefloatArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute argmax of an empty array.");

            float best = x.Data[0];
            int bestIdx = 0;
            for (int i = 1; i < x.Data.Length; i++)
            {
                if (x.Data[i] > best)
                {
                    best = x.Data[i];
                    bestIdx = i;
                }
            }
            return bestIdx;
        }

        public static float min<T>(in T x) where T : unmanaged, IUnsafefloatArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute min of an empty array.");

            if (x.Data.Length == 1)
                return x.Data[0];

            float min = float.MaxValue;
            for (int i = 0; i < x.Data.Length; i++)
                min = math.min(min, x.Data[i]);
            
            return min;
        }

        public static float max<T>(in T x) where T : unmanaged, IUnsafefloatArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute max of an empty array.");

            if (x.Data.Length == 1)
                return x.Data[0];

            float max = float.MinValue;
            for (int i = 0; i < x.Data.Length; i++)
                max = math.max(max, x.Data[i]);
            
            return max;
        }

        // needs to handle even / odd case
        public static float median<T>(in T x) where T : unmanaged, IUnsafefloatArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute median of an empty array.");
            
            if (x.Data.Length == 1)
                return x.Data[0];

            var copy = new UnsafeList<float>(x.Data.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            copy.AddRange(x.Data);
            copy.Sort();

            float res;

            // Odd case! e.g.: 5 % 2 = 1
            if (copy.Length % 2 != 0) {
                res = copy[copy.Length / 2];
            }
            else { // Even case!
                var n = copy.Length / 2;
                res = (copy[n-1] + copy[n]) / 2f;
            }

            copy.Dispose();

            return res;
        }

        public static float range<T>(in T x) where T : unmanaged, IUnsafefloatArray
        {
            return max(x) - min(x);
        }

        public static floatMeanMinMaxRangeStats meanMinMaxRange<T>(in T x) where T : unmanaged, IUnsafefloatArray {

            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute meanMinMaxRange of an empty array.");

            if (x.Data.Length == 1)
                return new floatMeanMinMaxRangeStats(x.Data[0], x.Data[0], x.Data[0], 0f);

            float min = float.MaxValue;
            float max = float.MinValue;
            float sum = 0f;
            for (int i = 0; i < x.Data.Length; i++)
            {
                float val = x.Data[i];
                min = math.min(min, val);
                max = math.max(max, val);
                sum += val;
            }

            float mean = sum / x.Data.Length;
            float range = max - min;

            return new floatMeanMinMaxRangeStats(mean, min, max, range);
        }

        // p-th (0..1) percentile of a SORTED list via linear interpolation (numpy 'linear' method).
        // pos = p*(n-1) is always in [0, n-1], so both indices are in bounds for any n >= 1.
        static float Percentile(UnsafeList<float> sorted, float p)
        {
            int n = sorted.Length;
            float pos = p * (float)(n - 1);
            int lo = (int)math.floor(pos);
            int hi = (int)math.ceil(pos);
            return sorted[lo] + (pos - (float)lo) * (sorted[hi] - sorted[lo]);
        }

        public static floatFullStats meanMinMaxRange_medianIQRstdDevVariance<T>(in T x) where T : unmanaged, IUnsafefloatArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute meanMinMaxRange_medianIQRstdDevVariance of an empty array.");

            if (x.Data.Length == 1)
            {
                return new floatFullStats(x.Data.Length, x.Data[0], x.Data[0], x.Data[0], 0f, x.Data[0], 0f, 0f, 0f, x.Data[0], x.Data[0]);
            }
            var copy = new UnsafeList<float>(x.Data.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            copy.AddRange(x.Data);
            copy.Sort();

            float min = copy[0];
            float max = copy[copy.Length - 1];
            float sum = 0f;

            // sum
            for (int i = 0; i < x.Data.Length; i++) {
                sum += x.Data[i];
            }

            float mean = sum / x.Data.Length;
            float range = max - min;

            float variance = 0f;
            for (int i = 0; i < x.Data.Length; i++) {
                float d = x.Data[i] - mean;
                variance += d * d;
            }
            variance /= x.Data.Length;

            // Quartiles via linear-interpolation percentile (numpy 'linear'). Bounds-safe for all
            // n >= 2 — the previous index arithmetic read out of bounds for n==2 (q1Index = -1) and
            // collapsed Q3 onto the median for small odd n.
            float median = Percentile(copy, (float)0.5);
            float q1 = Percentile(copy, (float)0.25);
            float q3 = Percentile(copy, (float)0.75);
            float iqr = q3 - q1;

            copy.Dispose();
            
            float stdDev = math.sqrt(variance);

            return new floatFullStats(x.Data.Length, mean, min, max, range, median, stdDev, variance, iqr, q1, q3);
        }

        #region MATRIX

        // sum along rows of matrix
        public static floatN rowSum(in floatMxN A)
        {
            var vec = A.floatVec(A.M_Rows);

            for (int r = 0; r < A.M_Rows; r++)
            {
                float sum = 0f;
                for (int c = 0; c < A.N_Cols; c++)
                    sum += A[r, c];
                
                vec[r] = sum;
            }
            
            return vec;
        }

        // sum along cols of matrix
        public static floatN colSum(in floatMxN A)
        {
            var vec = A.floatVec(A.N_Cols);

            for (int r = 0; r < A.M_Rows; r++)
            {
                for (int c = 0; c < A.N_Cols; c++)
                    vec[c] += A[r, c];
            }

            return vec;
        }

        // mean along rows of matrix
        public static floatN rowMean(in floatMxN A)
        {
            var vec = rowSum(in A);

            floatOP.divInpl(vec, A.N_Cols);

            return vec;
        }

        // mean along cols of matrix
        public static floatN colMean(in floatMxN A)
        {
            var vec = colSum(in A);

            floatOP.divInpl(vec, A.M_Rows);

            return vec;
        }

        // min along rows of matrix (length M_Rows)
        public static floatN rowMin(in floatMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            var vec = A.floatVec(A.M_Rows);

            for (int r = 0; r < A.M_Rows; r++)
            {
                float m = A[r, 0];
                for (int c = 1; c < A.N_Cols; c++)
                    m = math.min(m, A[r, c]);
                vec[r] = m;
            }

            return vec;
        }

        // max along rows of matrix (length M_Rows)
        public static floatN rowMax(in floatMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            var vec = A.floatVec(A.M_Rows);

            for (int r = 0; r < A.M_Rows; r++)
            {
                float m = A[r, 0];
                for (int c = 1; c < A.N_Cols; c++)
                    m = math.max(m, A[r, c]);
                vec[r] = m;
            }

            return vec;
        }

        // min along cols of matrix (length N_Cols)
        public static floatN colMin(in floatMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            var vec = A.floatVec(A.N_Cols);

            for (int c = 0; c < A.N_Cols; c++)
                vec[c] = A[0, c];

            for (int r = 1; r < A.M_Rows; r++)
            {
                for (int c = 0; c < A.N_Cols; c++)
                    vec[c] = math.min(vec[c], A[r, c]);
            }

            return vec;
        }

        // max along cols of matrix (length N_Cols)
        public static floatN colMax(in floatMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            var vec = A.floatVec(A.N_Cols);

            for (int c = 0; c < A.N_Cols; c++)
                vec[c] = A[0, c];

            for (int r = 1; r < A.M_Rows; r++)
            {
                for (int c = 0; c < A.N_Cols; c++)
                    vec[c] = math.max(vec[c], A[r, c]);
            }

            return vec;
        }

        // population variance along rows (÷N_Cols). Two-pass: compute each row mean inline as a scalar, then accumulate squared deviations.
        // A 1-column matrix produces all-zero result.
        public static floatN rowVariance(in floatMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            var vec = A.floatVec(A.M_Rows);

            for (int r = 0; r < A.M_Rows; r++)
            {
                // First pass: compute row mean inline (no allocation).
                float rowSum = 0f;
                for (int c = 0; c < A.N_Cols; c++)
                    rowSum += A[r, c];
                float m = rowSum / A.N_Cols;

                // Second pass: accumulate squared deviations.
                float sum = 0f;
                for (int c = 0; c < A.N_Cols; c++)
                {
                    float d = A[r, c] - m;
                    sum += d * d;
                }
                vec[r] = sum;
            }

            floatOP.divInpl(vec, A.N_Cols);

            return vec;
        }

        // population variance along cols (÷M_Rows). Two-pass: allocate temp vector for col means, then accumulate squared deviations.
        public static floatN colVariance(in floatMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            // Allocate means from the temp pool (reclaimed by ClearTemp, not persistent).
            var means = A.tempfloatVec(A.N_Cols);
            var vec = A.floatVec(A.N_Cols);

            // First pass: accumulate column sums into means.
            for (int r = 0; r < A.M_Rows; r++)
            {
                for (int c = 0; c < A.N_Cols; c++)
                    means[c] += A[r, c];
            }

            // Divide to get means.
            floatOP.divInpl(means, A.M_Rows);

            // Second pass: accumulate squared deviations.
            for (int r = 0; r < A.M_Rows; r++)
            {
                for (int c = 0; c < A.N_Cols; c++)
                {
                    float d = A[r, c] - means[c];
                    vec[c] += d * d;
                }
            }

            floatOP.divInpl(vec, A.M_Rows);

            return vec;
        }

        // population std dev along rows (sqrt of rowVariance)
        public static floatN rowStdDev(in floatMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            var vec = rowVariance(in A);

            for (int r = 0; r < A.M_Rows; r++)
                vec[r] = math.sqrt(vec[r]);

            return vec;
        }

        // population std dev along cols (sqrt of colVariance)
        public static floatN colStdDev(in floatMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            var vec = colVariance(in A);

            for (int c = 0; c < A.N_Cols; c++)
                vec[c] = math.sqrt(vec[c]);

            return vec;
        }

        // Core covariance computation: fills caller-provided N×N matrix C (already allocated).
        // Assumes A.M_Rows >= 2 and A.N_Cols == N (no guard). Uses a temp vector for column
        // means (reclaimed by ClearTemp). Fills all N×N cells symmetric with ÷(M−1).
        private static void covarianceInto(in floatMxN A, ref floatMxN C)
        {
            int N = A.N_Cols;
            int M = A.M_Rows;

            // Temp vector for column means (reclaimed by ClearTemp, not persistent).
            var means = A.tempfloatVec(N);

            // First pass: accumulate column sums, then divide to get means.
            for (int r = 0; r < M; r++)
            {
                for (int c = 0; c < N; c++)
                    means[c] += A[r, c];
            }
            for (int c = 0; c < N; c++)
                means[c] /= (float)M;

            // Second pass: compute upper triangle and mirror for symmetry.
            float invDenom = 1f / (float)(M - 1);
            for (int i = 0; i < N; i++)
            {
                for (int j = i; j < N; j++)
                {
                    float acc = 0f;
                    for (int r = 0; r < M; r++)
                        acc += (A[r, i] - means[i]) * (A[r, j] - means[j]);

                    float cov = acc * invDenom;
                    C[i, j] = cov;
                    C[j, i] = cov;
                }
            }
        }

        // Sample covariance matrix (N_Cols × N_Cols). Columns = variables, rows = observations.
        // Normalization: ÷ (M_Rows − 1) (Bessel-corrected / sample covariance).
        // C[i,i] equals varianceSample of column i. Symmetric: C[i,j] == C[j,i].
        // A zero-variance column produces a zero row/column in C (including C[i,i] == 0).
        public static floatMxN covariance(in floatMxN A)
        {
            if (A.M_Rows < 2 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Covariance requires at least 2 observations (rows) and 1 variable (column).");

            int N = A.N_Cols;

            // Allocate output covariance matrix (persistent arena allocation).
            var C = A.floatMat(N, N);
            covarianceInto(in A, ref C);
            return C;
        }

        // Pearson correlation matrix (N_Cols × N_Cols). Columns = variables, rows = observations.
        // Diagonal entries are 1. Off-diagonal: R[i,j] = C[i,j] / (s_i * s_j) where s_i = sqrt(C[i,i]).
        // A zero-variance column (s_i == 0) yields 0 for off-diagonal entries in that row/column
        // and 1 on the diagonal by convention. Off-diagonal entries are clamped to [-1, 1] to suppress
        // floating-point roundoff overshoot. Note: off-diagonal correlations are 0 only when a column's
        // computed sample variance is exactly 0; near-constant columns (with tiny roundoff variance)
        // may produce noisy correlations.
        public static floatMxN correlation(in floatMxN A)
        {
            if (A.M_Rows < 2 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Correlation requires at least 2 observations (rows) and 1 variable (column).");

            int N = A.N_Cols;

            // Compute covariance into a temp matrix (reclaimed by ClearTemp, not persistent).
            var C = A.tempfloatMat(N, N);
            covarianceInto(in A, ref C);

            // Precompute all N standard deviations into a temp vector (reclaimed by ClearTemp).
            var s = A.tempfloatVec(N);
            for (int i = 0; i < N; i++)
                s[i] = math.sqrt(C[i, i]);

            // Allocate output correlation matrix (persistent arena allocation).
            var R = A.floatMat(N, N);

            // Fill correlation matrix (upper triangle mirrored for symmetry).
            for (int i = 0; i < N; i++)
            {
                // Diagonal is always 1 by convention.
                R[i, i] = 1f;

                for (int j = i + 1; j < N; j++)
                {
                    float rij;
                    if (s[i] > 0f && s[j] > 0f)
                        rij = math.clamp(C[i, j] / (s[i] * s[j]), -1f, 1f);
                    else
                        rij = 0f;

                    R[i, j] = rij;
                    R[j, i] = rij;
                }
            }

            return R;
        }

        #endregion
    }
}
