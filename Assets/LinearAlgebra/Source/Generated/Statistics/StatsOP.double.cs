#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Collections;
using Unity.Mathematics;
using Unity.Collections.LowLevel.Unsafe;
using System;

namespace LinearAlgebra.Stats
{

    // just a prototype, needs matrices handling too
    public static partial class doubleStatsOP  {

        public static double sum<T>(in T x) where T : unmanaged, IUnsafedoubleArray {

            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute sum of an empty array.");

            if (x.Data.Length == 1)
                return x.Data[0];

            double sum = 0f;
            for (int i = 0; i < x.Data.Length; i++) 
                sum += x.Data[i];
            
            return sum;
        }

        public static double mean<T>(in T x) where T : unmanaged, IUnsafedoubleArray {
            return sum(in x) / x.Data.Length;
        }

        public static double variance<T>(in T x) where T : unmanaged, IUnsafedoubleArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute variance of an empty array.");

            if (x.Data.Length == 1)
                return 0f;

            double m = mean(x);
            double sum = 0f;
            for (int i = 0; i < x.Data.Length; i++)
            {
                double d = x.Data[i] - m;
                sum += d*d;
            }
            return sum / x.Data.Length;
        }

        public static double stdDev<T>(in T x) where T : unmanaged, IUnsafedoubleArray {
            return math.sqrt(variance(x));
        }

        // Sample variance: Σ(xᵢ−mean)²/(n−1). Two-pass (compute mean first, then squared deviations).
        // n==0 → throws; n==1 → throws.
        public static double varianceSample<T>(in T x) where T : unmanaged, IUnsafedoubleArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute sample variance of an empty array.");

            if (x.Data.Length == 1)
                throw new InvalidOperationException("Sample variance requires at least 2 elements.");

            double m = mean(in x);
            double sum = 0f;
            for (int i = 0; i < x.Data.Length; i++)
            {
                double d = x.Data[i] - m;
                sum += d * d;
            }
            return sum / (x.Data.Length - 1);
        }

        public static double stdDevSample<T>(in T x) where T : unmanaged, IUnsafedoubleArray {
            return math.sqrt(varianceSample(in x));
        }

        // Returns the index of the smallest element (first occurrence on ties).
        // For a matrix (IUnsafedoubleArray over row-major data), the index is the linear index r*N_Cols+c.
        // Note: if element 0 is NaN, the result may be index 0 (NaN comparisons are all false); behavior with NaN data is unspecified.
        public static int argmin<T>(in T x) where T : unmanaged, IUnsafedoubleArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute argmin of an empty array.");

            double best = x.Data[0];
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
        // For a matrix (IUnsafedoubleArray over row-major data), the index is the linear index r*N_Cols+c.
        // Note: if element 0 is NaN, the result may be index 0 (NaN comparisons are all false); behavior with NaN data is unspecified.
        public static int argmax<T>(in T x) where T : unmanaged, IUnsafedoubleArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute argmax of an empty array.");

            double best = x.Data[0];
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

        public static double min<T>(in T x) where T : unmanaged, IUnsafedoubleArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute min of an empty array.");

            if (x.Data.Length == 1)
                return x.Data[0];

            double min = double.MaxValue;
            for (int i = 0; i < x.Data.Length; i++)
                min = math.min(min, x.Data[i]);
            
            return min;
        }

        public static double max<T>(in T x) where T : unmanaged, IUnsafedoubleArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute max of an empty array.");

            if (x.Data.Length == 1)
                return x.Data[0];

            double max = double.MinValue;
            for (int i = 0; i < x.Data.Length; i++)
                max = math.max(max, x.Data[i]);
            
            return max;
        }

        // needs to handle even / odd case
        public static double median<T>(in T x) where T : unmanaged, IUnsafedoubleArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute median of an empty array.");
            
            if (x.Data.Length == 1)
                return x.Data[0];

            var copy = new UnsafeList<double>(x.Data.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            copy.AddRange(x.Data);
            copy.Sort();

            double res;

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

        public static double range<T>(in T x) where T : unmanaged, IUnsafedoubleArray
        {
            return max(x) - min(x);
        }

        public static doubleMeanMinMaxRangeStats meanMinMaxRange<T>(in T x) where T : unmanaged, IUnsafedoubleArray {

            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute meanMinMaxRange of an empty array.");

            if (x.Data.Length == 1)
                return new doubleMeanMinMaxRangeStats(x.Data[0], x.Data[0], x.Data[0], 0f);

            double min = double.MaxValue;
            double max = double.MinValue;
            double sum = 0f;
            for (int i = 0; i < x.Data.Length; i++)
            {
                double val = x.Data[i];
                min = math.min(min, val);
                max = math.max(max, val);
                sum += val;
            }

            double mean = sum / x.Data.Length;
            double range = max - min;

            return new doubleMeanMinMaxRangeStats(mean, min, max, range);
        }

        public static doubleFullStats meanMinMaxRange_medianIQRstdDevVariance<T>(in T x) where T : unmanaged, IUnsafedoubleArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute meanMinMaxRange_medianIQRstdDevVariance of an empty array.");

            if (x.Data.Length == 1)
            {
                return new doubleFullStats(x.Data.Length, x.Data[0], x.Data[0], x.Data[0], 0f, x.Data[0], 0f, 0f, 0f, x.Data[0], x.Data[0]);
            }
            var copy = new UnsafeList<double>(x.Data.Length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            copy.AddRange(x.Data);
            copy.Sort();

            double min = copy[0];
            double max = copy[copy.Length - 1];
            double sum = 0f;

            // sum
            for (int i = 0; i < x.Data.Length; i++) {
                sum += x.Data[i];
            }

            double mean = sum / x.Data.Length;
            double range = max - min;

            double variance = 0f;
            for (int i = 0; i < x.Data.Length; i++) {
                double d = x.Data[i] - mean;
                variance += d * d;
            }
            variance /= x.Data.Length;

            double median;
            double q1;
            double q3;
            if (copy.Length % 2 != 0)
            {
                int midIndex = copy.Length / 2;
                median = copy[midIndex];
                int q1Index = midIndex / 2;
                int q3Index = midIndex + q1Index;
                q1 = copy[q1Index];
                q3 = copy[q3Index];
            }
            else
            {
                int midIndex = copy.Length / 2;
                median = (copy[midIndex - 1] + copy[midIndex]) / 2f;
                int q1Index = midIndex / 2 - 1;
                int q3Index = midIndex + q1Index;
                q1 = (copy[q1Index] + copy[q1Index + 1]) / 2f;
                q3 = (copy[q3Index] + copy[q3Index + 1]) / 2f;
            }
            double iqr = q3 - q1;

            copy.Dispose();
            
            double stdDev = math.sqrt(variance);

            return new doubleFullStats(x.Data.Length, mean, min, max, range, median, stdDev, variance, iqr, q1, q3);
        }

        #region MATRIX

        // sum along rows of matrix
        public static doubleN rowSum(in doubleMxN A)
        {
            var vec = A.doubleVec(A.M_Rows);

            for (int r = 0; r < A.M_Rows; r++)
            {
                double sum = 0f;
                for (int c = 0; c < A.N_Cols; c++)
                    sum += A[r, c];
                
                vec[r] = sum;
            }
            
            return vec;
        }

        // sum along cols of matrix
        public static doubleN colSum(in doubleMxN A)
        {
            var vec = A.doubleVec(A.N_Cols);

            for (int r = 0; r < A.M_Rows; r++)
            {
                for (int c = 0; c < A.N_Cols; c++)
                    vec[c] += A[r, c];
            }

            return vec;
        }

        // mean along rows of matrix
        public static doubleN rowMean(in doubleMxN A)
        {
            var vec = rowSum(in A);

            doubleOP.divInpl(vec, A.N_Cols);

            return vec;
        }

        // mean along cols of matrix
        public static doubleN colMean(in doubleMxN A)
        {
            var vec = colSum(in A);

            doubleOP.divInpl(vec, A.M_Rows);

            return vec;
        }

        // min along rows of matrix (length M_Rows)
        public static doubleN rowMin(in doubleMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            var vec = A.doubleVec(A.M_Rows);

            for (int r = 0; r < A.M_Rows; r++)
            {
                double m = A[r, 0];
                for (int c = 1; c < A.N_Cols; c++)
                    m = math.min(m, A[r, c]);
                vec[r] = m;
            }

            return vec;
        }

        // max along rows of matrix (length M_Rows)
        public static doubleN rowMax(in doubleMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            var vec = A.doubleVec(A.M_Rows);

            for (int r = 0; r < A.M_Rows; r++)
            {
                double m = A[r, 0];
                for (int c = 1; c < A.N_Cols; c++)
                    m = math.max(m, A[r, c]);
                vec[r] = m;
            }

            return vec;
        }

        // min along cols of matrix (length N_Cols)
        public static doubleN colMin(in doubleMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            var vec = A.doubleVec(A.N_Cols);

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
        public static doubleN colMax(in doubleMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            var vec = A.doubleVec(A.N_Cols);

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
        public static doubleN rowVariance(in doubleMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            var vec = A.doubleVec(A.M_Rows);

            for (int r = 0; r < A.M_Rows; r++)
            {
                // First pass: compute row mean inline (no allocation).
                double rowSum = 0f;
                for (int c = 0; c < A.N_Cols; c++)
                    rowSum += A[r, c];
                double m = rowSum / A.N_Cols;

                // Second pass: accumulate squared deviations.
                double sum = 0f;
                for (int c = 0; c < A.N_Cols; c++)
                {
                    double d = A[r, c] - m;
                    sum += d * d;
                }
                vec[r] = sum;
            }

            doubleOP.divInpl(vec, A.N_Cols);

            return vec;
        }

        // population variance along cols (÷M_Rows). Two-pass: allocate temp vector for col means, then accumulate squared deviations.
        public static doubleN colVariance(in doubleMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            // Allocate means from the temp pool (reclaimed by ClearTemp, not persistent).
            var means = A.tempdoubleVec(A.N_Cols);
            var vec = A.doubleVec(A.N_Cols);

            // First pass: accumulate column sums into means.
            for (int r = 0; r < A.M_Rows; r++)
            {
                for (int c = 0; c < A.N_Cols; c++)
                    means[c] += A[r, c];
            }

            // Divide to get means.
            doubleOP.divInpl(means, A.M_Rows);

            // Second pass: accumulate squared deviations.
            for (int r = 0; r < A.M_Rows; r++)
            {
                for (int c = 0; c < A.N_Cols; c++)
                {
                    double d = A[r, c] - means[c];
                    vec[c] += d * d;
                }
            }

            doubleOP.divInpl(vec, A.M_Rows);

            return vec;
        }

        // population std dev along rows (sqrt of rowVariance)
        public static doubleN rowStdDev(in doubleMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");

            var vec = rowVariance(in A);

            for (int r = 0; r < A.M_Rows; r++)
                vec[r] = math.sqrt(vec[r]);

            return vec;
        }

        // population std dev along cols (sqrt of colVariance)
        public static doubleN colStdDev(in doubleMxN A)
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
        private static void covarianceInto(in doubleMxN A, ref doubleMxN C)
        {
            int N = A.N_Cols;
            int M = A.M_Rows;

            // Temp vector for column means (reclaimed by ClearTemp, not persistent).
            var means = A.tempdoubleVec(N);

            // First pass: accumulate column sums, then divide to get means.
            for (int r = 0; r < M; r++)
            {
                for (int c = 0; c < N; c++)
                    means[c] += A[r, c];
            }
            for (int c = 0; c < N; c++)
                means[c] /= (double)M;

            // Second pass: compute upper triangle and mirror for symmetry.
            double invDenom = 1f / (double)(M - 1);
            for (int i = 0; i < N; i++)
            {
                for (int j = i; j < N; j++)
                {
                    double acc = 0f;
                    for (int r = 0; r < M; r++)
                        acc += (A[r, i] - means[i]) * (A[r, j] - means[j]);

                    double cov = acc * invDenom;
                    C[i, j] = cov;
                    C[j, i] = cov;
                }
            }
        }

        // Sample covariance matrix (N_Cols × N_Cols). Columns = variables, rows = observations.
        // Normalization: ÷ (M_Rows − 1) (Bessel-corrected / sample covariance).
        // C[i,i] equals varianceSample of column i. Symmetric: C[i,j] == C[j,i].
        // A zero-variance column produces a zero row/column in C (including C[i,i] == 0).
        public static doubleMxN covariance(in doubleMxN A)
        {
            if (A.M_Rows < 2 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Covariance requires at least 2 observations (rows) and 1 variable (column).");

            int N = A.N_Cols;

            // Allocate output covariance matrix (persistent arena allocation).
            var C = A.doubleMat(N, N);
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
        public static doubleMxN correlation(in doubleMxN A)
        {
            if (A.M_Rows < 2 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Correlation requires at least 2 observations (rows) and 1 variable (column).");

            int N = A.N_Cols;

            // Compute covariance into a temp matrix (reclaimed by ClearTemp, not persistent).
            var C = A.tempdoubleMat(N, N);
            covarianceInto(in A, ref C);

            // Precompute all N standard deviations into a temp vector (reclaimed by ClearTemp).
            var s = A.tempdoubleVec(N);
            for (int i = 0; i < N; i++)
                s[i] = math.sqrt(C[i, i]);

            // Allocate output correlation matrix (persistent arena allocation).
            var R = A.doubleMat(N, N);

            // Fill correlation matrix (upper triangle mirrored for symmetry).
            for (int i = 0; i < N; i++)
            {
                // Diagonal is always 1 by convention.
                R[i, i] = 1f;

                for (int j = i + 1; j < N; j++)
                {
                    double rij;
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
