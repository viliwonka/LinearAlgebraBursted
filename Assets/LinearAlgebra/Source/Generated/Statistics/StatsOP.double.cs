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
                return x.Data[0];

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

        #endregion
    }
}
