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

        // p-th (0..1) percentile of a SORTED list via linear interpolation (numpy 'linear' method).
        // pos = p*(n-1) is always in [0, n-1], so both indices are in bounds for any n >= 1.
        static double Percentile(UnsafeList<double> sorted, double p)
        {
            int n = sorted.Length;
            double pos = p * (double)(n - 1);
            int lo = (int)math.floor(pos);
            int hi = (int)math.ceil(pos);
            return sorted[lo] + (pos - (double)lo) * (sorted[hi] - sorted[lo]);
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

            // Quartiles via linear-interpolation percentile (numpy 'linear'). Bounds-safe for all
            // n >= 2 — the previous index arithmetic read out of bounds for n==2 (q1Index = -1) and
            // collapsed Q3 onto the median for small odd n.
            double median = Percentile(copy, (double)0.5);
            double q1 = Percentile(copy, (double)0.25);
            double q3 = Percentile(copy, (double)0.75);
            double iqr = q3 - q1;

            copy.Dispose();
            
            double stdDev = math.sqrt(variance);

            return new doubleFullStats(x.Data.Length, mean, min, max, range, median, stdDev, variance, iqr, q1, q3);
        }

        #region MATRIX

        // --- Every row*/col* reduction below comes in two forms: a zero-alloc ref-DESTINATION
        //     primitive `(in A, ref doubleN dest)` that writes into a caller-provided vector
        //     (length A.M_Rows for row* ops, A.N_Cols for col* ops), and an allocating wrapper
        //     `(in A)` that returns a fresh arena vector. Use the ref form in per-frame / realtime
        //     loops (e.g. over a rolling window) to avoid allocating a result vector every call.
        //     The col* accumulating ops (colSum / colVariance / colNorm*) clear dest first, so dest
        //     may hold garbage on entry. ---

        // sum along rows of matrix (dest length M_Rows)
        public static void rowSum(in doubleMxN A, ref doubleN dest)
        {
            if (dest.N != A.M_Rows)
                throw new System.ArgumentException("StatsOP.rowSum: dest.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                double sum = 0f;
                for (int c = 0; c < A.N_Cols; c++)
                    sum += A[r, c];
                dest[r] = sum;
            }
        }

        public static doubleN rowSum(in doubleMxN A)
        {
            var vec = A.doubleVec(A.M_Rows);
            rowSum(in A, ref vec);
            return vec;
        }

        // sum along cols of matrix (dest length N_Cols)
        public static void colSum(in doubleMxN A, ref doubleN dest)
        {
            if (dest.N != A.N_Cols)
                throw new System.ArgumentException("StatsOP.colSum: dest.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
                dest[c] = 0f;

            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    dest[c] += A[r, c];
        }

        public static doubleN colSum(in doubleMxN A)
        {
            var vec = A.doubleVec(A.N_Cols);
            colSum(in A, ref vec);
            return vec;
        }

        // mean along rows of matrix (dest length M_Rows)
        public static void rowMean(in doubleMxN A, ref doubleN dest)
        {
            rowSum(in A, ref dest);
            doubleOP.divInpl(dest, A.N_Cols);
        }

        public static doubleN rowMean(in doubleMxN A)
        {
            var vec = A.doubleVec(A.M_Rows);
            rowMean(in A, ref vec);
            return vec;
        }

        // mean along cols of matrix (dest length N_Cols)
        public static void colMean(in doubleMxN A, ref doubleN dest)
        {
            colSum(in A, ref dest);
            doubleOP.divInpl(dest, A.M_Rows);
        }

        public static doubleN colMean(in doubleMxN A)
        {
            var vec = A.doubleVec(A.N_Cols);
            colMean(in A, ref vec);
            return vec;
        }

        // min along rows of matrix (dest length M_Rows)
        public static void rowMin(in doubleMxN A, ref doubleN dest)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");
            if (dest.N != A.M_Rows)
                throw new System.ArgumentException("StatsOP.rowMin: dest.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                double m = A[r, 0];
                for (int c = 1; c < A.N_Cols; c++)
                    m = math.min(m, A[r, c]);
                dest[r] = m;
            }
        }

        public static doubleN rowMin(in doubleMxN A)
        {
            var vec = A.doubleVec(A.M_Rows);
            rowMin(in A, ref vec);
            return vec;
        }

        // max along rows of matrix (dest length M_Rows)
        public static void rowMax(in doubleMxN A, ref doubleN dest)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");
            if (dest.N != A.M_Rows)
                throw new System.ArgumentException("StatsOP.rowMax: dest.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                double m = A[r, 0];
                for (int c = 1; c < A.N_Cols; c++)
                    m = math.max(m, A[r, c]);
                dest[r] = m;
            }
        }

        public static doubleN rowMax(in doubleMxN A)
        {
            var vec = A.doubleVec(A.M_Rows);
            rowMax(in A, ref vec);
            return vec;
        }

        // min along cols of matrix (dest length N_Cols)
        public static void colMin(in doubleMxN A, ref doubleN dest)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");
            if (dest.N != A.N_Cols)
                throw new System.ArgumentException("StatsOP.colMin: dest.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
                dest[c] = A[0, c];

            for (int r = 1; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    dest[c] = math.min(dest[c], A[r, c]);
        }

        public static doubleN colMin(in doubleMxN A)
        {
            var vec = A.doubleVec(A.N_Cols);
            colMin(in A, ref vec);
            return vec;
        }

        // max along cols of matrix (dest length N_Cols)
        public static void colMax(in doubleMxN A, ref doubleN dest)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");
            if (dest.N != A.N_Cols)
                throw new System.ArgumentException("StatsOP.colMax: dest.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
                dest[c] = A[0, c];

            for (int r = 1; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    dest[c] = math.max(dest[c], A[r, c]);
        }

        public static doubleN colMax(in doubleMxN A)
        {
            var vec = A.doubleVec(A.N_Cols);
            colMax(in A, ref vec);
            return vec;
        }

        // population variance along rows (÷N_Cols), dest length M_Rows. Two-pass per row: compute the
        // row mean inline (scalar, no alloc), then accumulate squared deviations. 1-column => all zero.
        public static void rowVariance(in doubleMxN A, ref doubleN dest)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");
            if (dest.N != A.M_Rows)
                throw new System.ArgumentException("StatsOP.rowVariance: dest.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                double rsum = 0f;
                for (int c = 0; c < A.N_Cols; c++)
                    rsum += A[r, c];
                double m = rsum / A.N_Cols;

                double sum = 0f;
                for (int c = 0; c < A.N_Cols; c++)
                {
                    double d = A[r, c] - m;
                    sum += d * d;
                }
                dest[r] = sum / A.N_Cols;
            }
        }

        public static doubleN rowVariance(in doubleMxN A)
        {
            var vec = A.doubleVec(A.M_Rows);
            rowVariance(in A, ref vec);
            return vec;
        }

        // population variance along cols (÷M_Rows), dest length N_Cols. Two-pass; needs one N_Cols
        // scratch vector for the column means. The scratch is a function-local Allocator.Temp
        // allocation disposed before return — it does NOT persist in the arena temp pool, so calling
        // this every frame leaks nothing (unlike a tempdoubleVec, which lives until ClearTemp).
        public static void colVariance(in doubleMxN A, ref doubleN dest)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");
            if (dest.N != A.N_Cols)
                throw new System.ArgumentException("StatsOP.colVariance: dest.N must equal A.N_Cols");

            // column means in a self-disposing local temp (zero-initialised; freed on return).
            var means = new doubleN(A.N_Cols, Allocator.Temp);
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    means[c] += A[r, c];
            doubleOP.divInpl(means, A.M_Rows);

            for (int c = 0; c < A.N_Cols; c++)
                dest[c] = 0f;

            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                {
                    double d = A[r, c] - means[c];
                    dest[c] += d * d;
                }

            doubleOP.divInpl(dest, A.M_Rows);

            means.Dispose();
        }

        public static doubleN colVariance(in doubleMxN A)
        {
            var vec = A.doubleVec(A.N_Cols);
            colVariance(in A, ref vec);
            return vec;
        }

        // population std dev along rows (sqrt of rowVariance), dest length M_Rows
        public static void rowStdDev(in doubleMxN A, ref doubleN dest)
        {
            rowVariance(in A, ref dest);
            for (int r = 0; r < A.M_Rows; r++)
                dest[r] = math.sqrt(dest[r]);
        }

        public static doubleN rowStdDev(in doubleMxN A)
        {
            var vec = A.doubleVec(A.M_Rows);
            rowStdDev(in A, ref vec);
            return vec;
        }

        // population std dev along cols (sqrt of colVariance), dest length N_Cols
        public static void colStdDev(in doubleMxN A, ref doubleN dest)
        {
            colVariance(in A, ref dest);
            for (int c = 0; c < A.N_Cols; c++)
                dest[c] = math.sqrt(dest[c]);
        }

        public static doubleN colStdDev(in doubleMxN A)
        {
            var vec = A.doubleVec(A.N_Cols);
            colStdDev(in A, ref vec);
            return vec;
        }

        // L1 norm of each row (Σ|·| across columns), dest length M_Rows
        public static void rowNormL1(in doubleMxN A, ref doubleN dest)
        {
            if (dest.N != A.M_Rows)
                throw new System.ArgumentException("StatsOP.rowNormL1: dest.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                double s = 0f;
                for (int c = 0; c < A.N_Cols; c++)
                    s += math.abs(A[r, c]);
                dest[r] = s;
            }
        }

        public static doubleN rowNormL1(in doubleMxN A)
        {
            var vec = A.doubleVec(A.M_Rows);
            rowNormL1(in A, ref vec);
            return vec;
        }

        // L2 norm of each row (sqrt Σ·² across columns), dest length M_Rows
        public static void rowNormL2(in doubleMxN A, ref doubleN dest)
        {
            if (dest.N != A.M_Rows)
                throw new System.ArgumentException("StatsOP.rowNormL2: dest.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                double s = 0f;
                for (int c = 0; c < A.N_Cols; c++)
                    s += A[r, c] * A[r, c];
                dest[r] = math.sqrt(s);
            }
        }

        public static doubleN rowNormL2(in doubleMxN A)
        {
            var vec = A.doubleVec(A.M_Rows);
            rowNormL2(in A, ref vec);
            return vec;
        }

        // L1 norm of each column (Σ|·| across rows), dest length N_Cols
        public static void colNormL1(in doubleMxN A, ref doubleN dest)
        {
            if (dest.N != A.N_Cols)
                throw new System.ArgumentException("StatsOP.colNormL1: dest.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
                dest[c] = 0f;

            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    dest[c] += math.abs(A[r, c]);
        }

        public static doubleN colNormL1(in doubleMxN A)
        {
            var vec = A.doubleVec(A.N_Cols);
            colNormL1(in A, ref vec);
            return vec;
        }

        // L2 norm of each column (sqrt Σ·² across rows), dest length N_Cols
        public static void colNormL2(in doubleMxN A, ref doubleN dest)
        {
            if (dest.N != A.N_Cols)
                throw new System.ArgumentException("StatsOP.colNormL2: dest.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
                dest[c] = 0f;

            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    dest[c] += A[r, c] * A[r, c];

            for (int c = 0; c < A.N_Cols; c++)
                dest[c] = math.sqrt(dest[c]);
        }

        public static doubleN colNormL2(in doubleMxN A)
        {
            var vec = A.doubleVec(A.N_Cols);
            colNormL2(in A, ref vec);
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
