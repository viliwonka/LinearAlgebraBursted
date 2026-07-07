#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Collections;
using Unity.Mathematics;
using Unity.Collections.LowLevel.Unsafe;
using System;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{

    internal static partial class doubleStatsCore {

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

            // Quartiles via linear-interpolation percentile (numpy 'linear'). Bounds-safe for all n >= 2.
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

        public static void rowMean(in doubleMxN A, ref doubleN dest)
        {
            rowSum(in A, ref dest);
            doubleComp.divInPlace(dest, A.N_Cols);
        }

        public static doubleN rowMean(in doubleMxN A)
        {
            var vec = A.doubleVec(A.M_Rows);
            rowMean(in A, ref vec);
            return vec;
        }

        public static void colMean(in doubleMxN A, ref doubleN dest)
        {
            colSum(in A, ref dest);
            doubleComp.divInPlace(dest, A.M_Rows);
        }

        public static doubleN colMean(in doubleMxN A)
        {
            var vec = A.doubleVec(A.N_Cols);
            colMean(in A, ref vec);
            return vec;
        }

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
        // this every frame leaks nothing (unlike a doubleTempVec, which lives until ClearTemp).
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
            doubleComp.divInPlace(means, A.M_Rows);

            for (int c = 0; c < A.N_Cols; c++)
                dest[c] = 0f;

            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                {
                    double d = A[r, c] - means[c];
                    dest[c] += d * d;
                }

            doubleComp.divInPlace(dest, A.M_Rows);

            means.Dispose();
        }

        public static doubleN colVariance(in doubleMxN A)
        {
            var vec = A.doubleVec(A.N_Cols);
            colVariance(in A, ref vec);
            return vec;
        }

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
        // M_Rows < 2 → zero-fills C and returns gracefully (no NaN). Uses temp vectors/matrix
        // for column means and centered data (both reclaimed by ClearTemp). Fills all N×N cells
        // via Gram formulation (centeredᵀ·centered ÷ (M−1)), which is exactly symmetric.
        // Public so zero-alloc callers (e.g. the realtime rolling window) can reuse it with a
        // preallocated C instead of going through the allocating covariance(in A) wrapper.
        public static void covarianceInto(in doubleMxN A, ref doubleMxN C)
        {
            int N = A.N_Cols;
            int M = A.M_Rows;

            // Guard: M < 2 makes 1/(M−1) = 1/0 = Inf, and 0·Inf = NaN-fills every cell.
            // Zero-fill C and return gracefully — the wrappers (covariance / correlation /
            // RollingWindow.Covariance) all throw for M < 2; this primitive degrades without
            // NaN for any zero-alloc realtime caller that pre-screens count.
            if (M < 2)
            {
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++)
                        C[i, j] = (double)0;
                return;
            }

            // Temp vector for column means (reclaimed by ClearTemp, not persistent).
            var means = A.doubleTempVec(N);

            // First pass: accumulate column sums (row-major), then divide to get means.
            for (int r = 0; r < M; r++)
                for (int c = 0; c < N; c++)
                    means[c] += A[r, c];
            for (int c = 0; c < N; c++)
                means[c] /= (double)M;

            // Second pass: build centered M×N matrix in one row-major sweep (reclaimed by ClearTemp).
            // centered[r, c] = A[r, c] − mean[c]
            var centered = A.doubleTempMat(M, N);
            for (int r = 0; r < M; r++)
                for (int c = 0; c < N; c++)
                    centered[r, c] = A[r, c] - means[c];

            // C = centeredᵀ · centered (Gram formulation). dot(..., transposeA:true) dispatches
            // to matMatDotTransA; inner read is unit-stride over columns of centered. Zeros C first.
            Blas.dot(in centered, in centered, ref C, transposeA: true);

            // Scale by 1/(M−1) (Bessel correction). The Gram matrix is exactly symmetric under
            // IEEE 754 (mul(a,b)==mul(b,a)), so no explicit symmetrization pass is needed.
            double invDenom = 1f / (double)(M - 1);
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    C[i, j] *= invDenom;
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
            var C = A.doubleTempMat(N, N);
            covarianceInto(in A, ref C);

            // Precompute all N standard deviations into a temp vector (reclaimed by ClearTemp).
            var s = A.doubleTempVec(N);
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

        // --- Distribution transforms (in-place) ---
        //
        // Each verb comes in three scopes:
        //   <T> flat  — generic over vec + matrix (matrix is IUnsafedoubleArray over flat row-major data).
        //   *Rows     — per row of a doubleMxN.
        //   *Columns  — per column of a doubleMxN (strided).
        //
        // Matrix variants use function-local Allocator.Temp scratch vectors disposed before
        // return — callers on per-frame paths leak nothing (unlike doubleTempVec / arena temp).
        //
        // Zero/constant-axis guards use !(x > 0) which is NaN-safe (catches 0 AND NaN).

        /// <summary>z-score standardize every element in-place: x ← (x − mean) / stdDev (population ÷ N).
        /// Constant input (stdDev == 0) → zero-fill. Empty → no-op.</summary>
        /// <remarks>Flat form — treats the input as one 1-D array. For a matrix this is the
        /// <b>whole-matrix</b> scope (all elements as a single distribution); use the
        /// <c>Rows</c>/<c>Columns</c> variants for per-axis.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void standardize<T>(in T x) where T : unmanaged, IUnsafedoubleArray
        {
            if (x.Data.Length == 0) return;
            unsafe
            {
                double* ptr = x.Data.Ptr;
                int n = x.Data.Length;
                double m = (double)0;
                for (int i = 0; i < n; i++) m += ptr[i];
                m /= n;
                double varAcc = (double)0;
                for (int i = 0; i < n; i++) { double d = ptr[i] - m; varAcc += d * d; }
                double sd = math.sqrt(varAcc / n);
                if (!(sd > (double)0)) { for (int i = 0; i < n; i++) ptr[i] = (double)0; }
                else                   { for (int i = 0; i < n; i++) ptr[i] = (ptr[i] - m) / sd; }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void standardizeRows(ref doubleMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;
            // Process each row in a single combined pass: compute mean once (eliminating the
            // duplicate computed by rowMean + rowVariance), then variance, then apply.
            // No Temp allocations — all scalars; rows are independently standardized.
            for (int r = 0; r < A.M_Rows; r++)
            {
                double rsum = (double)0;
                for (int c = 0; c < A.N_Cols; c++) rsum += A[r, c];
                double mu = rsum / A.N_Cols;
                double varAcc = (double)0;
                for (int c = 0; c < A.N_Cols; c++) { double d = A[r, c] - mu; varAcc += d * d; }
                double sd = math.sqrt(varAcc / A.N_Cols);
                if (!(sd > (double)0)) { for (int c = 0; c < A.N_Cols; c++) A[r, c] = (double)0; }
                else                   { for (int c = 0; c < A.N_Cols; c++) A[r, c] = (A[r, c] - mu) / sd; }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void standardizeColumns(ref doubleMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;
            var s0 = new doubleN(A.N_Cols, Allocator.Temp); // column means
            var s1 = new doubleN(A.N_Cols, Allocator.Temp); // column std devs
            // pass 1: compute column means into s0 (no extra Temp alloc — colMean uses ref dest)
            colMean(in A, ref s0);
            // pass 2: inline column variance using s0, store std dev into s1
            // (avoids the extra Temp alloc that colStdDev → colVariance would make for its own means)
            for (int c = 0; c < A.N_Cols; c++)
                s1[c] = (double)0;
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                {
                    double d = A[r, c] - s0[c];
                    s1[c] += d * d;
                }
            for (int c = 0; c < A.N_Cols; c++)
                s1[c] = math.sqrt(s1[c] / A.M_Rows);
            // pass 3: apply z-score in-place (row-major traverse for unit-stride writes)
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                {
                    double mu = s0[c], sd = s1[c];
                    if (!(sd > (double)0)) A[r, c] = (double)0;
                    else                   A[r, c] = (A[r, c] - mu) / sd;
                }
            s0.Dispose(); s1.Dispose();
        }

        /// <summary>Rescale every element in-place to [lo, hi] via min-max normalization.
        /// The no-arg overload maps to [0, 1]. Constant input (max == min) → every element set to lo. Empty → no-op.</summary>
        /// <remarks>Flat form; matrix scope details as in <see cref="standardize{T}(in T)"/>.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void rescale<T>(in T x) where T : unmanaged, IUnsafedoubleArray
            => rescale(in x, (double)0, (double)1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void rescale<T>(in T x, double lo, double hi) where T : unmanaged, IUnsafedoubleArray
        {
            if (x.Data.Length == 0) return;
            unsafe
            {
                double* ptr = x.Data.Ptr;
                int n = x.Data.Length;
                double mn = ptr[0], mx = ptr[0];
                for (int i = 1; i < n; i++)
                {
                    if (ptr[i] < mn) mn = ptr[i];
                    if (ptr[i] > mx) mx = ptr[i];
                }
                double rng = mx - mn;
                if (!(rng > (double)0)) { for (int i = 0; i < n; i++) ptr[i] = lo; }
                else
                {
                    double sc = hi - lo;
                    for (int i = 0; i < n; i++) ptr[i] = lo + ((ptr[i] - mn) / rng) * sc;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void rescaleRows(ref doubleMxN A) => rescaleRows(ref A, (double)0, (double)1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void rescaleRows(ref doubleMxN A, double lo, double hi)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;
            var s0 = new doubleN(A.M_Rows, Allocator.Temp);
            var s1 = new doubleN(A.M_Rows, Allocator.Temp);
            rowMin(in A, ref s0); rowMax(in A, ref s1);
            for (int r = 0; r < A.M_Rows; r++)
            {
                double mn = s0[r], rng = s1[r] - mn;
                if (!(rng > (double)0)) { for (int c = 0; c < A.N_Cols; c++) A[r, c] = lo; }
                else
                {
                    double sc = hi - lo;
                    for (int c = 0; c < A.N_Cols; c++) A[r, c] = lo + ((A[r, c] - mn) / rng) * sc;
                }
            }
            s0.Dispose(); s1.Dispose();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void rescaleColumns(ref doubleMxN A) => rescaleColumns(ref A, (double)0, (double)1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void rescaleColumns(ref doubleMxN A, double lo, double hi)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;
            var s0 = new doubleN(A.N_Cols, Allocator.Temp);
            var s1 = new doubleN(A.N_Cols, Allocator.Temp);
            colMin(in A, ref s0); colMax(in A, ref s1);
            // Apply: row-major traverse for unit-stride writes.
            double sc = hi - lo;
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                {
                    double mn = s0[c], rng = s1[c] - mn;
                    if (!(rng > (double)0)) A[r, c] = lo;
                    else                    A[r, c] = lo + ((A[r, c] - mn) / rng) * sc;
                }
            s0.Dispose(); s1.Dispose();
        }

        /// <summary>Center every element in-place by subtracting the mean: x ← x − mean. Empty → no-op.</summary>
        /// <remarks>Flat form; matrix scope details as in <see cref="standardize{T}(in T)"/>.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void center<T>(in T x) where T : unmanaged, IUnsafedoubleArray
        {
            if (x.Data.Length == 0) return;
            unsafe
            {
                double* ptr = x.Data.Ptr;
                int n = x.Data.Length;
                double m = (double)0;
                for (int i = 0; i < n; i++) m += ptr[i];
                m /= n;
                for (int i = 0; i < n; i++) ptr[i] -= m;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void centerRows(ref doubleMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;
            var s0 = new doubleN(A.M_Rows, Allocator.Temp);
            rowMean(in A, ref s0);
            for (int r = 0; r < A.M_Rows; r++)
            {
                double m = s0[r];
                for (int c = 0; c < A.N_Cols; c++) A[r, c] -= m;
            }
            s0.Dispose();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void centerColumns(ref doubleMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;
            var s0 = new doubleN(A.N_Cols, Allocator.Temp);
            colMean(in A, ref s0);
            // Apply: row-major traverse for unit-stride writes.
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    A[r, c] -= s0[c];
            s0.Dispose();
        }

        /// <summary>Divide every element in-place by max|x|, mapping data into [−1, 1].
        /// All-zero (or NaN-only) input → left unchanged. Empty → no-op.</summary>
        /// <remarks>Flat form; matrix scope details as in <see cref="standardize{T}(in T)"/>.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void maxAbs<T>(in T x) where T : unmanaged, IUnsafedoubleArray
        {
            if (x.Data.Length == 0) return;
            unsafe
            {
                double* ptr = x.Data.Ptr;
                int n = x.Data.Length;
                double mAbs = (double)0;
                for (int i = 0; i < n; i++) mAbs = math.max(mAbs, math.abs(ptr[i]));
                if (!(mAbs > (double)0)) return;
                for (int i = 0; i < n; i++) ptr[i] /= mAbs;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void maxAbsRows(ref doubleMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;
            for (int r = 0; r < A.M_Rows; r++)
            {
                double mAbs = (double)0;
                for (int c = 0; c < A.N_Cols; c++) mAbs = math.max(mAbs, math.abs(A[r, c]));
                if (!(mAbs > (double)0)) continue;
                for (int c = 0; c < A.N_Cols; c++) A[r, c] /= mAbs;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void maxAbsColumns(ref doubleMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;
            // Pre-compute per-column max|x| into a temp array via row-major stats pass.
            var mAbsArr = new doubleN(A.N_Cols, Allocator.Temp);
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    mAbsArr[c] = math.max(mAbsArr[c], math.abs(A[r, c]));
            // Apply: row-major traverse for unit-stride writes.
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    if (mAbsArr[c] > (double)0) A[r, c] /= mAbsArr[c];
            mAbsArr.Dispose();
        }

        /// <summary>Apply numerically stable softmax in-place: x ← exp(x − max(x)) / Σ exp(x − max(x)).
        /// No allocation. Empty → no-op.</summary>
        /// <remarks>Flat form; matrix scope details as in <see cref="standardize{T}(in T)"/>.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void softmax<T>(in T x) where T : unmanaged, IUnsafedoubleArray
        {
            if (x.Data.Length == 0) return;
            unsafe
            {
                double* ptr = x.Data.Ptr;
                int n = x.Data.Length;
                double maxVal = ptr[0];
                for (int i = 1; i < n; i++) if (ptr[i] > maxVal) maxVal = ptr[i];
                double expSum = (double)0;
                for (int i = 0; i < n; i++) { ptr[i] = math.exp(ptr[i] - maxVal); expSum += ptr[i]; }
                for (int i = 0; i < n; i++) ptr[i] /= expSum;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void softmaxRows(ref doubleMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;
            for (int r = 0; r < A.M_Rows; r++)
            {
                double maxVal = A[r, 0];
                for (int c = 1; c < A.N_Cols; c++) if (A[r, c] > maxVal) maxVal = A[r, c];
                double expSum = (double)0;
                for (int c = 0; c < A.N_Cols; c++) { A[r, c] = math.exp(A[r, c] - maxVal); expSum += A[r, c]; }
                for (int c = 0; c < A.N_Cols; c++) A[r, c] /= expSum;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void softmaxColumns(ref doubleMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;
            for (int c = 0; c < A.N_Cols; c++)
            {
                double maxVal = A[0, c];
                for (int r = 1; r < A.M_Rows; r++) if (A[r, c] > maxVal) maxVal = A[r, c];
                double expSum = (double)0;
                for (int r = 0; r < A.M_Rows; r++) { A[r, c] = math.exp(A[r, c] - maxVal); expSum += A[r, c]; }
                for (int r = 0; r < A.M_Rows; r++) A[r, c] /= expSum;
            }
        }

        #endregion
    }
}
