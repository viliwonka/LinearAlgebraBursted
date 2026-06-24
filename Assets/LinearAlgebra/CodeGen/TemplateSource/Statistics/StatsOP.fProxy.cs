#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Collections;
using Unity.Mathematics;
using Unity.Collections.LowLevel.Unsafe;
using System;

namespace LinearAlgebra.Stats
{

    // just a prototype, needs matrices handling too
    public static partial class fProxyStatsOP  {

        public static fProxy sum<T>(in T x) where T : unmanaged, IUnsafefProxyArray {

            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute sum of an empty array.");

            if (x.Data.Length == 1)
                return x.Data[0];

            fProxy sum = 0f;
            for (int i = 0; i < x.Data.Length; i++) 
                sum += x.Data[i];
            
            return sum;
        }

        public static fProxy mean<T>(in T x) where T : unmanaged, IUnsafefProxyArray {
            return sum(in x) / x.Data.Length;
        }

        public static fProxy variance<T>(in T x) where T : unmanaged, IUnsafefProxyArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute variance of an empty array.");

            if (x.Data.Length == 1)
                return 0f;

            fProxy m = mean(x);
            fProxy sum = 0f;
            for (int i = 0; i < x.Data.Length; i++)
            {
                fProxy d = x.Data[i] - m;
                sum += d*d;
            }
            return sum / x.Data.Length;
        }

        public static fProxy stdDev<T>(in T x) where T : unmanaged, IUnsafefProxyArray {
            return math.sqrt(variance(x));
        }

        // Sample variance: Σ(xᵢ−mean)²/(n−1). Two-pass (compute mean first, then squared deviations).
        // n==0 → throws; n==1 → throws.
        public static fProxy varianceSample<T>(in T x) where T : unmanaged, IUnsafefProxyArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute sample variance of an empty array.");

            if (x.Data.Length == 1)
                throw new InvalidOperationException("Sample variance requires at least 2 elements.");

            fProxy m = mean(in x);
            fProxy sum = 0f;
            for (int i = 0; i < x.Data.Length; i++)
            {
                fProxy d = x.Data[i] - m;
                sum += d * d;
            }
            return sum / (x.Data.Length - 1);
        }

        public static fProxy stdDevSample<T>(in T x) where T : unmanaged, IUnsafefProxyArray {
            return math.sqrt(varianceSample(in x));
        }

        // Returns the index of the smallest element (first occurrence on ties).
        // For a matrix (IUnsafefProxyArray over row-major data), the index is the linear index r*N_Cols+c.
        // Note: if element 0 is NaN, the result may be index 0 (NaN comparisons are all false); behavior with NaN data is unspecified.
        public static int argmin<T>(in T x) where T : unmanaged, IUnsafefProxyArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute argmin of an empty array.");

            fProxy best = x.Data[0];
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
        // For a matrix (IUnsafefProxyArray over row-major data), the index is the linear index r*N_Cols+c.
        // Note: if element 0 is NaN, the result may be index 0 (NaN comparisons are all false); behavior with NaN data is unspecified.
        public static int argmax<T>(in T x) where T : unmanaged, IUnsafefProxyArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute argmax of an empty array.");

            fProxy best = x.Data[0];
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

        public static fProxy min<T>(in T x) where T : unmanaged, IUnsafefProxyArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute min of an empty array.");

            if (x.Data.Length == 1)
                return x.Data[0];

            fProxy min = fProxy.MaxValue;
            for (int i = 0; i < x.Data.Length; i++)
                min = math.min(min, x.Data[i]);
            
            return min;
        }

        public static fProxy max<T>(in T x) where T : unmanaged, IUnsafefProxyArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute max of an empty array.");

            if (x.Data.Length == 1)
                return x.Data[0];

            fProxy max = fProxy.MinValue;
            for (int i = 0; i < x.Data.Length; i++)
                max = math.max(max, x.Data[i]);
            
            return max;
        }

        // needs to handle even / odd case
        public static fProxy median<T>(in T x) where T : unmanaged, IUnsafefProxyArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute median of an empty array.");
            
            if (x.Data.Length == 1)
                return x.Data[0];

            var copy = new UnsafeList<fProxy>(x.Data.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            copy.AddRange(x.Data);
            copy.Sort();

            fProxy res;

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

        public static fProxy range<T>(in T x) where T : unmanaged, IUnsafefProxyArray
        {
            return max(x) - min(x);
        }

        public static fProxyMeanMinMaxRangeStats meanMinMaxRange<T>(in T x) where T : unmanaged, IUnsafefProxyArray {

            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute meanMinMaxRange of an empty array.");

            if (x.Data.Length == 1)
                return new fProxyMeanMinMaxRangeStats(x.Data[0], x.Data[0], x.Data[0], 0f);

            fProxy min = fProxy.MaxValue;
            fProxy max = fProxy.MinValue;
            fProxy sum = 0f;
            for (int i = 0; i < x.Data.Length; i++)
            {
                fProxy val = x.Data[i];
                min = math.min(min, val);
                max = math.max(max, val);
                sum += val;
            }

            fProxy mean = sum / x.Data.Length;
            fProxy range = max - min;

            return new fProxyMeanMinMaxRangeStats(mean, min, max, range);
        }

        // p-th (0..1) percentile of a SORTED list via linear interpolation (numpy 'linear' method).
        // pos = p*(n-1) is always in [0, n-1], so both indices are in bounds for any n >= 1.
        static fProxy Percentile(UnsafeList<fProxy> sorted, fProxy p)
        {
            int n = sorted.Length;
            fProxy pos = p * (fProxy)(n - 1);
            int lo = (int)math.floor(pos);
            int hi = (int)math.ceil(pos);
            return sorted[lo] + (pos - (fProxy)lo) * (sorted[hi] - sorted[lo]);
        }

        public static fProxyFullStats meanMinMaxRange_medianIQRstdDevVariance<T>(in T x) where T : unmanaged, IUnsafefProxyArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute meanMinMaxRange_medianIQRstdDevVariance of an empty array.");

            if (x.Data.Length == 1)
            {
                return new fProxyFullStats(x.Data.Length, x.Data[0], x.Data[0], x.Data[0], 0f, x.Data[0], 0f, 0f, 0f, x.Data[0], x.Data[0]);
            }
            var copy = new UnsafeList<fProxy>(x.Data.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            copy.AddRange(x.Data);
            copy.Sort();

            fProxy min = copy[0];
            fProxy max = copy[copy.Length - 1];
            fProxy sum = 0f;

            // sum
            for (int i = 0; i < x.Data.Length; i++) {
                sum += x.Data[i];
            }

            fProxy mean = sum / x.Data.Length;
            fProxy range = max - min;

            fProxy variance = 0f;
            for (int i = 0; i < x.Data.Length; i++) {
                fProxy d = x.Data[i] - mean;
                variance += d * d;
            }
            variance /= x.Data.Length;

            // Quartiles via linear-interpolation percentile (numpy 'linear'). Bounds-safe for all
            // n >= 2 — the previous index arithmetic read out of bounds for n==2 (q1Index = -1) and
            // collapsed Q3 onto the median for small odd n.
            fProxy median = Percentile(copy, (fProxy)0.5);
            fProxy q1 = Percentile(copy, (fProxy)0.25);
            fProxy q3 = Percentile(copy, (fProxy)0.75);
            fProxy iqr = q3 - q1;

            copy.Dispose();
            
            fProxy stdDev = math.sqrt(variance);

            return new fProxyFullStats(x.Data.Length, mean, min, max, range, median, stdDev, variance, iqr, q1, q3);
        }

        #region MATRIX

        // sum along rows of matrix
        public static fProxyN rowSum(in fProxyMxN A)
        {
            var vec = A.fProxyVec(A.M_Rows);

            for (int r = 0; r < A.M_Rows; r++)
            {
                fProxy sum = 0f;
                for (int c = 0; c < A.N_Cols; c++)
                    sum += A[r, c];
                
                vec[r] = sum;
            }
            
            return vec;
        }

        // sum along cols of matrix
        public static fProxyN colSum(in fProxyMxN A)
        {
            var vec = A.fProxyVec(A.N_Cols);

            for (int r = 0; r < A.M_Rows; r++)
            {
                for (int c = 0; c < A.N_Cols; c++)
                    vec[c] += A[r, c];
            }

            return vec;
        }

        // mean along rows of matrix
        public static fProxyN rowMean(in fProxyMxN A)
        {
            var vec = rowSum(in A);

            fProxyOP.divInpl(vec, A.N_Cols);

            return vec;
        }

        // mean along cols of matrix
        public static fProxyN colMean(in fProxyMxN A)
        {
            var vec = colSum(in A);

            fProxyOP.divInpl(vec, A.M_Rows);

            return vec;
        }

        // min along rows of matrix (length M_Rows)
        public static fProxyN rowMin(in fProxyMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            var vec = A.fProxyVec(A.M_Rows);

            for (int r = 0; r < A.M_Rows; r++)
            {
                fProxy m = A[r, 0];
                for (int c = 1; c < A.N_Cols; c++)
                    m = math.min(m, A[r, c]);
                vec[r] = m;
            }

            return vec;
        }

        // max along rows of matrix (length M_Rows)
        public static fProxyN rowMax(in fProxyMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            var vec = A.fProxyVec(A.M_Rows);

            for (int r = 0; r < A.M_Rows; r++)
            {
                fProxy m = A[r, 0];
                for (int c = 1; c < A.N_Cols; c++)
                    m = math.max(m, A[r, c]);
                vec[r] = m;
            }

            return vec;
        }

        // min along cols of matrix (length N_Cols)
        public static fProxyN colMin(in fProxyMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            var vec = A.fProxyVec(A.N_Cols);

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
        public static fProxyN colMax(in fProxyMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            var vec = A.fProxyVec(A.N_Cols);

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
        public static fProxyN rowVariance(in fProxyMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            var vec = A.fProxyVec(A.M_Rows);

            for (int r = 0; r < A.M_Rows; r++)
            {
                // First pass: compute row mean inline (no allocation).
                fProxy rowSum = 0f;
                for (int c = 0; c < A.N_Cols; c++)
                    rowSum += A[r, c];
                fProxy m = rowSum / A.N_Cols;

                // Second pass: accumulate squared deviations.
                fProxy sum = 0f;
                for (int c = 0; c < A.N_Cols; c++)
                {
                    fProxy d = A[r, c] - m;
                    sum += d * d;
                }
                vec[r] = sum;
            }

            fProxyOP.divInpl(vec, A.N_Cols);

            return vec;
        }

        // population variance along cols (÷M_Rows). Two-pass: allocate temp vector for col means, then accumulate squared deviations.
        public static fProxyN colVariance(in fProxyMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            // Allocate means from the temp pool (reclaimed by ClearTemp, not persistent).
            var means = A.tempfProxyVec(A.N_Cols);
            var vec = A.fProxyVec(A.N_Cols);

            // First pass: accumulate column sums into means.
            for (int r = 0; r < A.M_Rows; r++)
            {
                for (int c = 0; c < A.N_Cols; c++)
                    means[c] += A[r, c];
            }

            // Divide to get means.
            fProxyOP.divInpl(means, A.M_Rows);

            // Second pass: accumulate squared deviations.
            for (int r = 0; r < A.M_Rows; r++)
            {
                for (int c = 0; c < A.N_Cols; c++)
                {
                    fProxy d = A[r, c] - means[c];
                    vec[c] += d * d;
                }
            }

            fProxyOP.divInpl(vec, A.M_Rows);

            return vec;
        }

        // population std dev along rows (sqrt of rowVariance)
        public static fProxyN rowStdDev(in fProxyMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            var vec = rowVariance(in A);

            for (int r = 0; r < A.M_Rows; r++)
                vec[r] = math.sqrt(vec[r]);

            return vec;
        }

        // population std dev along cols (sqrt of colVariance)
        public static fProxyN colStdDev(in fProxyMxN A)
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
        private static void covarianceInto(in fProxyMxN A, ref fProxyMxN C)
        {
            int N = A.N_Cols;
            int M = A.M_Rows;

            // Temp vector for column means (reclaimed by ClearTemp, not persistent).
            var means = A.tempfProxyVec(N);

            // First pass: accumulate column sums, then divide to get means.
            for (int r = 0; r < M; r++)
            {
                for (int c = 0; c < N; c++)
                    means[c] += A[r, c];
            }
            for (int c = 0; c < N; c++)
                means[c] /= (fProxy)M;

            // Second pass: compute upper triangle and mirror for symmetry.
            fProxy invDenom = 1f / (fProxy)(M - 1);
            for (int i = 0; i < N; i++)
            {
                for (int j = i; j < N; j++)
                {
                    fProxy acc = 0f;
                    for (int r = 0; r < M; r++)
                        acc += (A[r, i] - means[i]) * (A[r, j] - means[j]);

                    fProxy cov = acc * invDenom;
                    C[i, j] = cov;
                    C[j, i] = cov;
                }
            }
        }

        // Sample covariance matrix (N_Cols × N_Cols). Columns = variables, rows = observations.
        // Normalization: ÷ (M_Rows − 1) (Bessel-corrected / sample covariance).
        // C[i,i] equals varianceSample of column i. Symmetric: C[i,j] == C[j,i].
        // A zero-variance column produces a zero row/column in C (including C[i,i] == 0).
        public static fProxyMxN covariance(in fProxyMxN A)
        {
            if (A.M_Rows < 2 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Covariance requires at least 2 observations (rows) and 1 variable (column).");

            int N = A.N_Cols;

            // Allocate output covariance matrix (persistent arena allocation).
            var C = A.fProxyMat(N, N);
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
        public static fProxyMxN correlation(in fProxyMxN A)
        {
            if (A.M_Rows < 2 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Correlation requires at least 2 observations (rows) and 1 variable (column).");

            int N = A.N_Cols;

            // Compute covariance into a temp matrix (reclaimed by ClearTemp, not persistent).
            var C = A.tempfProxyMat(N, N);
            covarianceInto(in A, ref C);

            // Precompute all N standard deviations into a temp vector (reclaimed by ClearTemp).
            var s = A.tempfProxyVec(N);
            for (int i = 0; i < N; i++)
                s[i] = math.sqrt(C[i, i]);

            // Allocate output correlation matrix (persistent arena allocation).
            var R = A.fProxyMat(N, N);

            // Fill correlation matrix (upper triangle mirrored for symmetry).
            for (int i = 0; i < N; i++)
            {
                // Diagonal is always 1 by convention.
                R[i, i] = 1f;

                for (int j = i + 1; j < N; j++)
                {
                    fProxy rij;
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
