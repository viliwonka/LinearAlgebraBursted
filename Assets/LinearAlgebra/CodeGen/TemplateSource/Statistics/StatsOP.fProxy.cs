#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Collections;
using Unity.Mathematics;
using Unity.Collections.LowLevel.Unsafe;
using System;

namespace LinearAlgebra.Stats
{

    // just a prototype, needs matrices handling too
    public static partial class fProxyStatsOP<T> where T : unmanaged, IUnsafefProxyArray {

        public static fProxy sum(in T x) {

            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute sum of an empty array.");

            if (x.Data.Length == 1)
                return x.Data[0];

            fProxy sum = 0f;
            for (int i = 0; i < x.Data.Length; i++)
                sum += x.Data[i];
            
            return sum;
        }

        public static fProxy mean(in T x) {
            return sum(in x) / x.Data.Length;
        }

        public static fProxy variance(in T x)
        {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute variance of an empty array.");

            if (x.Data.Length == 1)
                return x.Data[0];

            fProxy m = mean(x);
            fProxy sum = 0f;
            for (int i = 0; i < x.Data.Length; i++)
            {
                fProxy d = x.Data[i] - m;
                sum += d*d;
            }
            return sum / x.Data.Length;
        }

        public static fProxy stdDev(in T x)
        {
            return math.sqrt(variance(x));
        }

        public static fProxy min(in T x)
        {
            if (x.Data.Length == 0)
                throw new InvalidOperationException("Cannot compute min of an empty array.");

            if (x.Data.Length == 1)
                return x.Data[0];

            fProxy min = fProxy.MaxValue;
            for (int i = 0; i < x.Data.Length; i++)
                min = math.min(min, x.Data[i]);
            
            return min;
        }

        public static fProxy max(in T x)
        {
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
        public static fProxy median(in T x)
        {
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

        public static fProxy range(in T x)
        {
            return max(x) - min(x);
        }

        public static fProxyMeanMinMaxRangeStats meanMinMaxRange(in T x)
        {
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

        public static fProxyFullStats meanMinMaxRange_medianIQRstdDevVariance(in T x)
        {
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

            fProxy median;
            fProxy q1;
            fProxy q3;
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

        #endregion
    }
}
