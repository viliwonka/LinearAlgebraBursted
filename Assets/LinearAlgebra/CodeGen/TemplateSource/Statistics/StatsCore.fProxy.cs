using Unity.Collections;
using Unity.Mathematics;
using Unity.Collections.LowLevel.Unsafe;
using System;
using System.Runtime.CompilerServices;

using BULA.Internal;

namespace BULA
{

    internal static partial class fProxyStatsCore {

        public static fProxy sum<T>(in T x) where T : unmanaged, IUnsafefProxyArray {

            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute sum of an empty array.");

            if (x.Data.Length == 1)
                return x.Data[0];

            // The SIMD reduction lives in UnsafeOP.sum (2x width-4 accumulators, frozen fold); Stats just
            // guards the empty/single-element cases and forwards. See UnsafeOP reductions / matVecDot.
            unsafe { return Internal.UnsafeOP.sum(x.Data.Ptr, x.Data.Length); }
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

            unsafe { return UnsafeOP.min(x.Data.Ptr, x.Data.Length); }
        }

        public static fProxy max<T>(in T x) where T : unmanaged, IUnsafefProxyArray {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute max of an empty array.");

            unsafe { return UnsafeOP.max(x.Data.Ptr, x.Data.Length); }
        }

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

            // Quartiles via linear-interpolation percentile (numpy 'linear'). Bounds-safe for all n >= 2.
            fProxy median = Percentile(copy, (fProxy)0.5);
            fProxy q1 = Percentile(copy, (fProxy)0.25);
            fProxy q3 = Percentile(copy, (fProxy)0.75);
            fProxy iqr = q3 - q1;

            copy.Dispose();
            
            fProxy stdDev = math.sqrt(variance);

            return new fProxyFullStats(x.Data.Length, mean, min, max, range, median, stdDev, variance, iqr, q1, q3);
        }

        #region MATRIX

        // --- Every row*/col* reduction below comes in two forms: a zero-alloc ref-DESTINATION
        //     primitive `(in A, ref fProxyN dest)` that writes into a caller-provided vector
        //     (length A.M_Rows for row* ops, A.N_Cols for col* ops), and an allocating wrapper
        //     `(in A)` that returns a fresh vector (from Allocator.Temp). Use the ref form in per-frame / realtime
        //     loops (e.g. over a rolling window) to avoid allocating a result vector every call.
        //     The col* accumulating ops (colSum / colVariance / colNorm*) clear dest first, so dest
        //     may hold garbage on entry. ---

        public static void rowSum(in fProxyMxN A, ref fProxyN dest)
        {
            if (dest.N != A.M_Rows)
                throw new System.ArgumentException("Stats.rowSum: dest.N must equal A.M_Rows");

            unsafe
            {
                fProxy* ap = A.Data.Ptr;
                int nc = A.N_Cols;
                for (int r = 0; r < A.M_Rows; r++)
                    dest[r] = UnsafeOP.sum(ap + (long)r * nc, nc);
            }
        }

        public static fProxyN rowSum(in fProxyMxN A)
        {
            var vec = A.fProxyVec(A.M_Rows);
            rowSum(in A, ref vec);
            return vec;
        }

        public static void colSum(in fProxyMxN A, ref fProxyN dest)
        {
            if (dest.N != A.N_Cols)
                throw new System.ArgumentException("Stats.colSum: dest.N must equal A.N_Cols");

            unsafe
            {
                fProxy* ap = A.Data.Ptr; fProxy* dp = dest.Data.Ptr;
                int nc = A.N_Cols;
                for (int c = 0; c < nc; c++)
                    dp[c] = 0f;

                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    for (int c = 0; c < nc; c++)
                        dp[c] += row[c];
                }
            }
        }

        public static fProxyN colSum(in fProxyMxN A)
        {
            var vec = A.fProxyVec(A.N_Cols);
            colSum(in A, ref vec);
            return vec;
        }

        public static void rowMean(in fProxyMxN A, ref fProxyN dest)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");
            rowSum(in A, ref dest);
            fProxyComp.divInPlace(dest, A.N_Cols);
        }

        public static fProxyN rowMean(in fProxyMxN A)
        {
            var vec = A.fProxyVec(A.M_Rows);
            rowMean(in A, ref vec);
            return vec;
        }

        public static void colMean(in fProxyMxN A, ref fProxyN dest)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");
            colSum(in A, ref dest);
            fProxyComp.divInPlace(dest, A.M_Rows);
        }

        public static fProxyN colMean(in fProxyMxN A)
        {
            var vec = A.fProxyVec(A.N_Cols);
            colMean(in A, ref vec);
            return vec;
        }

        public static void rowMin(in fProxyMxN A, ref fProxyN dest)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");
            if (dest.N != A.M_Rows)
                throw new System.ArgumentException("Stats.rowMin: dest.N must equal A.M_Rows");

            unsafe
            {
                fProxy* ap = A.Data.Ptr;
                int nc = A.N_Cols;
                for (int r = 0; r < A.M_Rows; r++)
                    dest[r] = UnsafeOP.min(ap + (long)r * nc, nc);
            }
        }

        public static fProxyN rowMin(in fProxyMxN A)
        {
            var vec = A.fProxyVec(A.M_Rows);
            rowMin(in A, ref vec);
            return vec;
        }

        public static void rowMax(in fProxyMxN A, ref fProxyN dest)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");
            if (dest.N != A.M_Rows)
                throw new System.ArgumentException("Stats.rowMax: dest.N must equal A.M_Rows");

            unsafe
            {
                fProxy* ap = A.Data.Ptr;
                int nc = A.N_Cols;
                for (int r = 0; r < A.M_Rows; r++)
                    dest[r] = UnsafeOP.max(ap + (long)r * nc, nc);
            }
        }

        public static fProxyN rowMax(in fProxyMxN A)
        {
            var vec = A.fProxyVec(A.M_Rows);
            rowMax(in A, ref vec);
            return vec;
        }

        public static void colMin(in fProxyMxN A, ref fProxyN dest)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");
            if (dest.N != A.N_Cols)
                throw new System.ArgumentException("Stats.colMin: dest.N must equal A.N_Cols");

            unsafe
            {
                fProxy* ap = A.Data.Ptr; fProxy* dp = dest.Data.Ptr;
                int nc = A.N_Cols;
                for (int c = 0; c < nc; c++)
                    dp[c] = ap[c];

                for (int r = 1; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    for (int c = 0; c < nc; c++)
                        dp[c] = math.min(dp[c], row[c]);
                }
            }
        }

        public static fProxyN colMin(in fProxyMxN A)
        {
            var vec = A.fProxyVec(A.N_Cols);
            colMin(in A, ref vec);
            return vec;
        }

        public static void colMax(in fProxyMxN A, ref fProxyN dest)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");
            if (dest.N != A.N_Cols)
                throw new System.ArgumentException("Stats.colMax: dest.N must equal A.N_Cols");

            unsafe
            {
                fProxy* ap = A.Data.Ptr; fProxy* dp = dest.Data.Ptr;
                int nc = A.N_Cols;
                for (int c = 0; c < nc; c++)
                    dp[c] = ap[c];

                for (int r = 1; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    for (int c = 0; c < nc; c++)
                        dp[c] = math.max(dp[c], row[c]);
                }
            }
        }

        public static fProxyN colMax(in fProxyMxN A)
        {
            var vec = A.fProxyVec(A.N_Cols);
            colMax(in A, ref vec);
            return vec;
        }

        // population variance along rows (÷N_Cols), dest length M_Rows. Two-pass per row: compute the
        // row mean inline (scalar, no alloc), then accumulate squared deviations. 1-column => all zero.
        public static void rowVariance(in fProxyMxN A, ref fProxyN dest)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");
            if (dest.N != A.M_Rows)
                throw new System.ArgumentException("Stats.rowVariance: dest.N must equal A.M_Rows");

            unsafe
            {
                fProxy* ap = A.Data.Ptr;
                int nc = A.N_Cols;
                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    fProxy rsum = 0f;
                    for (int c = 0; c < nc; c++)
                        rsum += row[c];
                    fProxy m = rsum / nc;

                    fProxy sum = 0f;
                    for (int c = 0; c < nc; c++)
                    {
                        fProxy d = row[c] - m;
                        sum += d * d;
                    }
                    dest[r] = sum / nc;
                }
            }
        }

        public static fProxyN rowVariance(in fProxyMxN A)
        {
            var vec = A.fProxyVec(A.M_Rows);
            rowVariance(in A, ref vec);
            return vec;
        }

        // population variance along cols (÷M_Rows), dest length N_Cols. Two-pass; needs one N_Cols
        // scratch vector for the column means. The scratch is a function-local Allocator.Temp
        // allocation disposed before return, so calling this every frame leaks nothing (unlike a
        // fProxyTempVec, which is never explicitly disposed).
        public static void colVariance(in fProxyMxN A, ref fProxyN dest)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Cannot compute statistics of an empty matrix.");
            if (dest.N != A.N_Cols)
                throw new System.ArgumentException("Stats.colVariance: dest.N must equal A.N_Cols");

            // column means in a self-disposing local temp (zero-initialised; freed on return).
            var means = new fProxyN(A.N_Cols, Allocator.Temp);
            unsafe
            {
                fProxy* ap = A.Data.Ptr; fProxy* mp = means.Data.Ptr; fProxy* dp = dest.Data.Ptr;
                int nc = A.N_Cols;
                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    for (int c = 0; c < nc; c++)
                        mp[c] += row[c];
                }
                fProxyComp.divInPlace(means, A.M_Rows);

                for (int c = 0; c < nc; c++)
                    dp[c] = 0f;

                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    for (int c = 0; c < nc; c++)
                    {
                        fProxy d = row[c] - mp[c];
                        dp[c] += d * d;
                    }
                }
            }

            fProxyComp.divInPlace(dest, A.M_Rows);

            means.Dispose();
        }

        public static fProxyN colVariance(in fProxyMxN A)
        {
            var vec = A.fProxyVec(A.N_Cols);
            colVariance(in A, ref vec);
            return vec;
        }

        public static void rowStdDev(in fProxyMxN A, ref fProxyN dest)
        {
            rowVariance(in A, ref dest);
            unsafe
            {
                fProxy* dp = dest.Data.Ptr;
                int m = A.M_Rows;
                for (int r = 0; r < m; r++)
                    dp[r] = math.sqrt(dp[r]);
            }
        }

        public static fProxyN rowStdDev(in fProxyMxN A)
        {
            var vec = A.fProxyVec(A.M_Rows);
            rowStdDev(in A, ref vec);
            return vec;
        }

        public static void colStdDev(in fProxyMxN A, ref fProxyN dest)
        {
            colVariance(in A, ref dest);
            unsafe
            {
                fProxy* dp = dest.Data.Ptr;
                int nc = A.N_Cols;
                for (int c = 0; c < nc; c++)
                    dp[c] = math.sqrt(dp[c]);
            }
        }

        public static fProxyN colStdDev(in fProxyMxN A)
        {
            var vec = A.fProxyVec(A.N_Cols);
            colStdDev(in A, ref vec);
            return vec;
        }

        public static void rowNormL1(in fProxyMxN A, ref fProxyN dest)
        {
            if (dest.N != A.M_Rows)
                throw new System.ArgumentException("Stats.rowNormL1: dest.N must equal A.M_Rows");

            unsafe
            {
                fProxy* ap = A.Data.Ptr;
                int nc = A.N_Cols;
                for (int r = 0; r < A.M_Rows; r++)
                    dest[r] = UnsafeOP.sumAbs(ap + (long)r * nc, nc);
            }
        }

        public static fProxyN rowNormL1(in fProxyMxN A)
        {
            var vec = A.fProxyVec(A.M_Rows);
            rowNormL1(in A, ref vec);
            return vec;
        }

        public static void rowNormL2(in fProxyMxN A, ref fProxyN dest)
        {
            if (dest.N != A.M_Rows)
                throw new System.ArgumentException("Stats.rowNormL2: dest.N must equal A.M_Rows");

            unsafe
            {
                fProxy* ap = A.Data.Ptr;
                int nc = A.N_Cols;
                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    dest[r] = math.sqrt(UnsafeOP.vecDot(row, row, nc));
                }
            }
        }

        public static fProxyN rowNormL2(in fProxyMxN A)
        {
            var vec = A.fProxyVec(A.M_Rows);
            rowNormL2(in A, ref vec);
            return vec;
        }

        public static void colNormL1(in fProxyMxN A, ref fProxyN dest)
        {
            if (dest.N != A.N_Cols)
                throw new System.ArgumentException("Stats.colNormL1: dest.N must equal A.N_Cols");

            unsafe
            {
                fProxy* ap = A.Data.Ptr; fProxy* dp = dest.Data.Ptr;
                int nc = A.N_Cols;
                for (int c = 0; c < nc; c++)
                    dp[c] = 0f;

                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    for (int c = 0; c < nc; c++)
                        dp[c] += math.abs(row[c]);
                }
            }
        }

        public static fProxyN colNormL1(in fProxyMxN A)
        {
            var vec = A.fProxyVec(A.N_Cols);
            colNormL1(in A, ref vec);
            return vec;
        }

        public static void colNormL2(in fProxyMxN A, ref fProxyN dest)
        {
            if (dest.N != A.N_Cols)
                throw new System.ArgumentException("Stats.colNormL2: dest.N must equal A.N_Cols");

            unsafe
            {
                fProxy* ap = A.Data.Ptr; fProxy* dp = dest.Data.Ptr;
                int nc = A.N_Cols;
                for (int c = 0; c < nc; c++)
                    dp[c] = 0f;

                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    for (int c = 0; c < nc; c++)
                        dp[c] += row[c] * row[c];
                }

                for (int c = 0; c < nc; c++)
                    dp[c] = math.sqrt(dp[c]);
            }
        }

        public static fProxyN colNormL2(in fProxyMxN A)
        {
            var vec = A.fProxyVec(A.N_Cols);
            colNormL2(in A, ref vec);
            return vec;
        }

        // Core covariance computation: fills caller-provided N×N matrix C (already allocated).
        // M_Rows < 2 → zero-fills C and returns gracefully (no NaN). Uses temp vectors/matrix
        // for column means and centered data (both from Allocator.Temp). Fills all N×N cells
        // via Gram formulation (centeredᵀ·centered ÷ (M−1)), which is exactly symmetric.
        // Public so zero-alloc callers (e.g. the realtime rolling window) can reuse it with a
        // preallocated C instead of going through the allocating covariance(in A) wrapper.
        public static void covarianceInto(in fProxyMxN A, ref fProxyMxN C)
        {
            int N = A.N_Cols;
            int M = A.M_Rows;

            if (C.M_Rows != N || C.N_Cols != N)
                throw new System.ArgumentException("covarianceInto: C must be N_Cols x N_Cols");

            // Guard: M < 2 makes 1/(M−1) = 1/0 = Inf, and 0·Inf = NaN-fills every cell.
            // Zero-fill C and return gracefully — the wrappers (covariance / correlation /
            // RollingWindow.Covariance) all throw for M < 2; this primitive degrades without
            // NaN for any zero-alloc realtime caller that pre-screens count.
            if (M < 2)
            {
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++)
                        C[i, j] = (fProxy)0;
                return;
            }

            // Temp vector for column means (from Allocator.Temp).
            var means = A.fProxyTempVec(N);

            // First pass: accumulate column sums (row-major), then divide to get means.
            // Second pass: build centered M×N matrix in one row-major sweep (from Allocator.Temp).
            // centered[r, c] = A[r, c] − mean[c]
            var centered = A.fProxyTempMat(M, N);
            unsafe
            {
                fProxy* ap = A.Data.Ptr; fProxy* mp = means.Data.Ptr; fProxy* cp = centered.Data.Ptr;
                for (int r = 0; r < M; r++)
                {
                    fProxy* row = ap + (long)r * N;
                    for (int c = 0; c < N; c++)
                        mp[c] += row[c];
                }
                for (int c = 0; c < N; c++)
                    mp[c] /= (fProxy)M;

                for (int r = 0; r < M; r++)
                {
                    fProxy* arow = ap + (long)r * N; fProxy* crow = cp + (long)r * N;
                    for (int c = 0; c < N; c++)
                        crow[c] = arow[c] - mp[c];
                }
            }

            // C = centeredᵀ · centered (Gram formulation). dot(..., transposeA:true) dispatches
            // to matMatDotTransA; inner read is unit-stride over columns of centered. Zeros C first.
            Blas.dot(in centered, in centered, ref C, transposeA: true);

            // Scale by 1/(M−1) (Bessel correction). The Gram matrix is exactly symmetric under
            // IEEE 754 (mul(a,b)==mul(b,a)), so no explicit symmetrization pass is needed.
            fProxy invDenom = 1f / (fProxy)(M - 1);
            unsafe
            {
                fProxy* cp = C.Data.Ptr;
                long total = (long)N * N;
                for (long k = 0; k < total; k++)
                    cp[k] *= invDenom;
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

            // Allocate output covariance matrix (from Allocator.Temp; returned to the caller).
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

            // Compute covariance into a temp matrix (from Allocator.Temp).
            var C = A.fProxyTempMat(N, N);
            covarianceInto(in A, ref C);

            // Precompute all N standard deviations into a temp vector (from Allocator.Temp).
            var s = A.fProxyTempVec(N);
            for (int i = 0; i < N; i++)
                s[i] = math.sqrt(C[i, i]);

            // Allocate output correlation matrix (from Allocator.Temp; returned to the caller).
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

        // --- Distribution transforms (in-place) ---
        //
        // Each verb comes in three scopes:
        //   <T> flat  — generic over vec + matrix (matrix is IUnsafefProxyArray over flat row-major data).
        //   *Rows     — per row of a fProxyMxN.
        //   *Columns  — per column of a fProxyMxN (strided).
        //
        // Matrix variants use function-local Allocator.Temp scratch vectors disposed before
        // return — callers on per-frame paths leak nothing (unlike fProxyTempVec, which is never
        // explicitly disposed).
        //
        // Zero/constant-axis guards use !(x > 0) which is NaN-safe (catches 0 AND NaN).

        /// <summary>z-score standardize every element in-place: x ← (x − mean) / stdDev (population ÷ N).
        /// Constant input (stdDev == 0) → zero-fill. Empty → no-op.</summary>
        /// <remarks>Flat form — treats the input as one 1-D array. For a matrix this is the
        /// <b>whole-matrix</b> scope (all elements as a single distribution); use the
        /// <c>Rows</c>/<c>Columns</c> variants for per-axis.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void standardize<T>(in T x) where T : unmanaged, IUnsafefProxyArray
        {
            if (x.Data.Length == 0) return;
            unsafe
            {
                fProxy* ptr = x.Data.Ptr;
                int n = x.Data.Length;
                fProxy m = (fProxy)0;
                for (int i = 0; i < n; i++) m += ptr[i];
                m /= n;
                fProxy varAcc = (fProxy)0;
                for (int i = 0; i < n; i++) { fProxy d = ptr[i] - m; varAcc += d * d; }
                fProxy sd = math.sqrt(varAcc / n);
                if (!(sd > (fProxy)0)) { for (int i = 0; i < n; i++) ptr[i] = (fProxy)0; }
                else                   { for (int i = 0; i < n; i++) ptr[i] = (ptr[i] - m) / sd; }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void standardizeRows(ref fProxyMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;
            // Process each row in a single combined pass: compute mean once (eliminating the
            // duplicate computed by rowMean + rowVariance), then variance, then apply.
            // No Temp allocations — all scalars; rows are independently standardized.
            unsafe
            {
                fProxy* ap = A.Data.Ptr;
                int nc = A.N_Cols;
                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    fProxy rsum = (fProxy)0;
                    for (int c = 0; c < nc; c++) rsum += row[c];
                    fProxy mu = rsum / nc;
                    fProxy varAcc = (fProxy)0;
                    for (int c = 0; c < nc; c++) { fProxy d = row[c] - mu; varAcc += d * d; }
                    fProxy sd = math.sqrt(varAcc / nc);
                    if (!(sd > (fProxy)0)) { for (int c = 0; c < nc; c++) row[c] = (fProxy)0; }
                    else                   { for (int c = 0; c < nc; c++) row[c] = (row[c] - mu) / sd; }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void standardizeColumns(ref fProxyMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;
            var s0 = new fProxyN(A.N_Cols, Allocator.Temp); // column means
            var s1 = new fProxyN(A.N_Cols, Allocator.Temp); // column std devs
            // pass 1: compute column means into s0 (no extra Temp alloc — colMean uses ref dest)
            colMean(in A, ref s0);
            // pass 2: inline column variance using s0, store std dev into s1
            // (avoids the extra Temp alloc that colStdDev → colVariance would make for its own means)
            unsafe
            {
                fProxy* ap = A.Data.Ptr; fProxy* s0p = s0.Data.Ptr; fProxy* s1p = s1.Data.Ptr;
                int nc = A.N_Cols;
                for (int c = 0; c < nc; c++)
                    s1p[c] = (fProxy)0;
                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    for (int c = 0; c < nc; c++)
                    {
                        fProxy d = row[c] - s0p[c];
                        s1p[c] += d * d;
                    }
                }
                for (int c = 0; c < nc; c++)
                    s1p[c] = math.sqrt(s1p[c] / A.M_Rows);
                // pass 3: apply z-score in-place (row-major traverse for unit-stride writes)
                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    for (int c = 0; c < nc; c++)
                    {
                        fProxy mu = s0p[c], sd = s1p[c];
                        if (!(sd > (fProxy)0)) row[c] = (fProxy)0;
                        else                   row[c] = (row[c] - mu) / sd;
                    }
                }
            }
            s0.Dispose(); s1.Dispose();
        }

        /// <summary>Rescale every element in-place to [lo, hi] via min-max normalization.
        /// The no-arg overload maps to [0, 1]. Constant input (max == min) → every element set to lo. Empty → no-op.</summary>
        /// <remarks>Flat form; matrix scope details as in <see cref="standardize{T}(in T)"/>.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void rescale<T>(in T x) where T : unmanaged, IUnsafefProxyArray
            => rescale(in x, (fProxy)0, (fProxy)1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void rescale<T>(in T x, fProxy lo, fProxy hi) where T : unmanaged, IUnsafefProxyArray
        {
            if (x.Data.Length == 0) return;
            unsafe
            {
                fProxy* ptr = x.Data.Ptr;
                int n = x.Data.Length;
                fProxy mn = ptr[0], mx = ptr[0];
                for (int i = 1; i < n; i++)
                {
                    if (ptr[i] < mn) mn = ptr[i];
                    if (ptr[i] > mx) mx = ptr[i];
                }
                fProxy rng = mx - mn;
                if (!(rng > (fProxy)0)) { for (int i = 0; i < n; i++) ptr[i] = lo; }
                else
                {
                    fProxy sc = hi - lo;
                    for (int i = 0; i < n; i++) ptr[i] = lo + ((ptr[i] - mn) / rng) * sc;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void rescaleRows(ref fProxyMxN A) => rescaleRows(ref A, (fProxy)0, (fProxy)1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void rescaleRows(ref fProxyMxN A, fProxy lo, fProxy hi)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;
            var s0 = new fProxyN(A.M_Rows, Allocator.Temp);
            var s1 = new fProxyN(A.M_Rows, Allocator.Temp);
            rowMin(in A, ref s0); rowMax(in A, ref s1);
            unsafe
            {
                fProxy* ap = A.Data.Ptr;
                int nc = A.N_Cols;
                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    fProxy mn = s0[r], rng = s1[r] - mn;
                    if (!(rng > (fProxy)0)) { for (int c = 0; c < nc; c++) row[c] = lo; }
                    else
                    {
                        fProxy sc = hi - lo;
                        for (int c = 0; c < nc; c++) row[c] = lo + ((row[c] - mn) / rng) * sc;
                    }
                }
            }
            s0.Dispose(); s1.Dispose();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void rescaleColumns(ref fProxyMxN A) => rescaleColumns(ref A, (fProxy)0, (fProxy)1);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void rescaleColumns(ref fProxyMxN A, fProxy lo, fProxy hi)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;
            var s0 = new fProxyN(A.N_Cols, Allocator.Temp);
            var s1 = new fProxyN(A.N_Cols, Allocator.Temp);
            colMin(in A, ref s0); colMax(in A, ref s1);
            // Apply: row-major traverse for unit-stride writes.
            fProxy sc = hi - lo;
            unsafe
            {
                fProxy* ap = A.Data.Ptr; fProxy* s0p = s0.Data.Ptr; fProxy* s1p = s1.Data.Ptr;
                int nc = A.N_Cols;
                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    for (int c = 0; c < nc; c++)
                    {
                        fProxy mn = s0p[c], rng = s1p[c] - mn;
                        if (!(rng > (fProxy)0)) row[c] = lo;
                        else                    row[c] = lo + ((row[c] - mn) / rng) * sc;
                    }
                }
            }
            s0.Dispose(); s1.Dispose();
        }

        /// <summary>Center every element in-place by subtracting the mean: x ← x − mean. Empty → no-op.</summary>
        /// <remarks>Flat form; matrix scope details as in <see cref="standardize{T}(in T)"/>.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void center<T>(in T x) where T : unmanaged, IUnsafefProxyArray
        {
            if (x.Data.Length == 0) return;
            unsafe
            {
                fProxy* ptr = x.Data.Ptr;
                int n = x.Data.Length;
                fProxy m = (fProxy)0;
                for (int i = 0; i < n; i++) m += ptr[i];
                m /= n;
                for (int i = 0; i < n; i++) ptr[i] -= m;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void centerRows(ref fProxyMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;
            var s0 = new fProxyN(A.M_Rows, Allocator.Temp);
            rowMean(in A, ref s0);
            unsafe
            {
                fProxy* ap = A.Data.Ptr;
                int nc = A.N_Cols;
                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    fProxy m = s0[r];
                    for (int c = 0; c < nc; c++) row[c] -= m;
                }
            }
            s0.Dispose();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void centerColumns(ref fProxyMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;
            var s0 = new fProxyN(A.N_Cols, Allocator.Temp);
            colMean(in A, ref s0);
            // Apply: row-major traverse for unit-stride writes.
            unsafe
            {
                fProxy* ap = A.Data.Ptr; fProxy* s0p = s0.Data.Ptr;
                int nc = A.N_Cols;
                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    for (int c = 0; c < nc; c++)
                        row[c] -= s0p[c];
                }
            }
            s0.Dispose();
        }

        /// <summary>Divide every element in-place by max|x|, mapping data into [−1, 1].
        /// All-zero (or NaN-only) input → left unchanged. Empty → no-op.</summary>
        /// <remarks>Flat form; matrix scope details as in <see cref="standardize{T}(in T)"/>.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void maxAbs<T>(in T x) where T : unmanaged, IUnsafefProxyArray
        {
            if (x.Data.Length == 0) return;
            unsafe
            {
                fProxy* ptr = x.Data.Ptr;
                int n = x.Data.Length;
                fProxy mAbs = (fProxy)0;
                for (int i = 0; i < n; i++) mAbs = math.max(mAbs, math.abs(ptr[i]));
                if (!(mAbs > (fProxy)0)) return;
                for (int i = 0; i < n; i++) ptr[i] /= mAbs;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void maxAbsRows(ref fProxyMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;
            unsafe
            {
                fProxy* ap = A.Data.Ptr;
                int nc = A.N_Cols;
                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    fProxy mAbs = (fProxy)0;
                    for (int c = 0; c < nc; c++) mAbs = math.max(mAbs, math.abs(row[c]));
                    if (!(mAbs > (fProxy)0)) continue;
                    for (int c = 0; c < nc; c++) row[c] /= mAbs;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void maxAbsColumns(ref fProxyMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;
            // Pre-compute per-column max|x| into a temp array via row-major stats pass.
            var mAbsArr = new fProxyN(A.N_Cols, Allocator.Temp);
            unsafe
            {
                fProxy* ap = A.Data.Ptr; fProxy* mp = mAbsArr.Data.Ptr;
                int nc = A.N_Cols;
                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    for (int c = 0; c < nc; c++)
                        mp[c] = math.max(mp[c], math.abs(row[c]));
                }
                // Apply: row-major traverse for unit-stride writes.
                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    for (int c = 0; c < nc; c++)
                        if (mp[c] > (fProxy)0) row[c] /= mp[c];
                }
            }
            mAbsArr.Dispose();
        }

        /// <summary>Apply numerically stable softmax in-place: x ← exp(x − max(x)) / Σ exp(x − max(x)).
        /// No allocation. Empty → no-op.</summary>
        /// <remarks>Flat form; matrix scope details as in <see cref="standardize{T}(in T)"/>.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void softmax<T>(in T x) where T : unmanaged, IUnsafefProxyArray
        {
            if (x.Data.Length == 0) return;
            unsafe
            {
                fProxy* ptr = x.Data.Ptr;
                int n = x.Data.Length;
                fProxy maxVal = ptr[0];
                for (int i = 1; i < n; i++) if (ptr[i] > maxVal) maxVal = ptr[i];
                fProxy expSum = (fProxy)0;
                for (int i = 0; i < n; i++) { ptr[i] = DetMath.Exp(ptr[i] - maxVal); expSum += ptr[i]; }
                for (int i = 0; i < n; i++) ptr[i] /= expSum;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void softmaxRows(ref fProxyMxN A)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0) return;
            unsafe
            {
                fProxy* ap = A.Data.Ptr;
                int nc = A.N_Cols;
                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    fProxy maxVal = row[0];
                    for (int c = 1; c < nc; c++) if (row[c] > maxVal) maxVal = row[c];
                    fProxy expSum = (fProxy)0;
                    for (int c = 0; c < nc; c++) { row[c] = DetMath.Exp(row[c] - maxVal); expSum += row[c]; }
                    for (int c = 0; c < nc; c++) row[c] /= expSum;
                }
            }
        }

        public static void softmaxColumns(ref fProxyMxN A)
        {
            int nr = A.M_Rows, nc = A.N_Cols;
            if (nr == 0 || nc == 0) return;

            // Column softmax via three row-major passes (unit-stride inner loops -> good cache locality
            // + the max/divide passes vectorise; the exp pass stays exp-bound but reads contiguously
            // instead of column-strided). Each column's max and exp-sum still visit rows in ascending
            // order, and pass 3 divides (not *reciprocal), so the result is bit-identical to the strided
            // per-column form. Two length-N_Cols Temps hold the per-column max then the per-column exp-sum.
            fProxyN colMax = A.fProxyTempVec(nc);
            fProxyN colSum = A.fProxyTempVec(nc);
            unsafe
            {
                fProxy* ap = A.Data.Ptr;
                fProxy* mp = colMax.Data.Ptr;
                fProxy* sp = colSum.Data.Ptr;

                // Pass 1: per-column max (init from row 0; strict > so NaN never displaces).
                for (int c = 0; c < nc; c++) { mp[c] = ap[c]; sp[c] = (fProxy)0; }
                for (int r = 1; r < nr; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    for (int c = 0; c < nc; c++) if (row[c] > mp[c]) mp[c] = row[c];
                }

                // Pass 2: exp(A - colMax) in place, accumulate the per-column sum (ascending rows).
                for (int r = 0; r < nr; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    for (int c = 0; c < nc; c++) { fProxy e = DetMath.Exp(row[c] - mp[c]); row[c] = e; sp[c] += e; }
                }

                // Pass 3: divide each column by its exp-sum.
                for (int r = 0; r < nr; r++)
                {
                    fProxy* row = ap + (long)r * nc;
                    for (int c = 0; c < nc; c++) row[c] /= sp[c];
                }
            }
        }

        #endregion
    }
}
