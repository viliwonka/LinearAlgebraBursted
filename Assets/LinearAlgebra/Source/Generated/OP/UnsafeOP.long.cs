#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Mathematics;
using System.Runtime.CompilerServices;
using Unity.Burst;

namespace LinearAlgebra
{
    public static unsafe partial class UnsafeOP {

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static long sum([NoAlias] long* a, int n) {

            long sum = 0;

            for (int i = 0; i < n; i++)
                sum += a[i];
            
            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static long sumAbs([NoAlias] long* a, int n)
        {
            long sum = 0;

            for (int i = 0; i < n; i++)
                sum += (long)(a[i] < 0? -a[i] : a[i]);
            
            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static long maxAbs([NoAlias] long* a, int n)
        {
            long max = 0;

            for (int i = 0; i < n; i++) {
                var abs = (a[i] < 0 ? -a[i] : a[i]);
                max = (long)(max < abs? abs : max);
            }
            return max;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static long vecDot([NoAlias] long* vA, [NoAlias] long* vB, int n) {

            long sum = 0;

            for (int i = 0; i < n; i++) {
                sum += (long)(vA[i] * vB[i]);
            }

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static long vecDotRange([NoAlias] long* vA, [NoAlias] long* vB, int start, int end)
        {
            long sum = 0;

            for (int i = start; i < end; i++)
            {
                sum += (long)(vA[i] * vB[i]);
            }

            return sum;
        }



        // outer dot product (vec x vec => mat)
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void vecOuterDot([NoAlias] long* vA, [NoAlias] long* vB, [NoAlias] long* mat, int m, int n)
        {
            //mat doesn't need to be initialized to zero
            for (int r = 0; r < m; r++)
            for (int c = 0; c < n; c++)
            {
                mat[r * n + c] = (long)(vA[r] * vB[c]);
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matVecDot([NoAlias] long* mat, [NoAlias] long* x, [NoAlias] long* y, int m, int n)
        {
            // mat = m x n
            // x = n
            // y = m, needs to be initialized to zero
            for (int r = 0; r < m; r++)
            {
                for(int c = 0; c < n; c++)
                {
                    y[r] += (long)(mat[r * n + c] * x[c]);
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void vecMatDot([NoAlias] long* y, [NoAlias] long* mat, [NoAlias] long* x, int m, int n)
        {
            // mat = m x n
            // y = inVec = m
            // x = outVec = n, needs to be initialized to zero
            for (int c = 0; c < n; c++)
            {
                for (int r = 0; r < m; r++)
                {
                    x[c] += (long)(mat[r * n + c] * y[r]);
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matTrans([NoAlias] long* matA, [NoAlias] long* matB, int m, int n)
        {
            // matB doesn't need to be initialized to zero

            // matA = m x n, in
            // matB = n x m, out
            for(int r = 0; r < m; r++)
            for(int c = 0; c < n; c++)
            {
                matB[c * m + r] = matA[r * n + c];
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matMatDot([NoAlias] long* matA, [NoAlias] long* matB, [NoAlias] long* matC, int m, int n, int k)
        {
            // matA = m x n
            // matB = n x k
            // matC = outMat = m x k, needs to be initialized to zero

            for (int r = 0; r < m; r++)
            {
                for (int nCols = 0; nCols < n; nCols++)
                {
                    long temp = matA[r * n + nCols]; // Cache the value from matA
                    for (int kCols = 0; kCols < k; kCols++)
                    {
                        matC[r * k + kCols] += (long)(temp * matB[nCols * k + kCols]);
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matMatDotTransA([NoAlias] long* matA, [NoAlias] long* matB, [NoAlias] long* matC, int m, int n, int k)
        {
            // matA = m x n, but treated as n x m due to transposition
            // matB = n x k
            // matC = outMat = m x k, needs to be initialized to zero
            for (int r = 0; r < m; r++)
            {
                for (int nCols = 0; nCols < n; nCols++)
                {
                    long temp = matA[nCols * m + r];
                    for (int kCols = 0; kCols < k; kCols++)
                    {
                        matC[r * k + kCols] += (long)(temp * matB[nCols * k + kCols]);
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void signFlip([NoAlias] long* target, [NoAlias] long* from, int n) {

            for (int i = 0; i < n; i++)
                target[i] = (long)(-from[i]);
        }
        
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compAdd([NoAlias] long* target, [NoAlias] long* from, int n) {

            for (int i = 0; i < n; i++)
                target[i] += from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compSub([NoAlias] long* from, [NoAlias] long* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] -= from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalAdd([NoAlias] long* target, int n, long s) {

            for (int i = 0; i < n; i++)
                target[i] += s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalSub(long s, [NoAlias] long* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = (long)(s - target[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compMul([NoAlias] long* from, [NoAlias] long* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] *= from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compDiv([NoAlias] long* targetDividend, [NoAlias] long* fromDivisor, int n)
        {            
            for (int i = 0; i < n; i++)
                targetDividend[i] /= fromDivisor[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compMod([NoAlias] long* targetDividend, [NoAlias] long* fromDivisor, int n)
        {
            for (int i = 0; i < n; i++)
                targetDividend[i] %= fromDivisor[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMul([NoAlias] long* target, int n, long s)
        {
            for (int i = 0; i < n; i++)
                target[i] *= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalDiv([NoAlias] long* target, int n, long s)
        {
            for (int i = 0; i < n; i++)
                target[i] /= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalDiv(long s, [NoAlias] long* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = (long)(s / target[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMod([NoAlias] long* target, int n, long s)
        {
            for (int i = 0; i < n; i++)
                target[i] %= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMod(long s, [NoAlias] long* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = (long)(s % target[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseComplement([NoAlias] long* target, int n) {
            for (int i = 0; i < n; i++)
                target[i] = (long)(~target[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseAnd([NoAlias] long* target, int n, long value) {
            for (int i = 0; i < n; i++)
                target[i] &= value;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseOr([NoAlias] long* target, int n, long value) {
            for (int i = 0; i < n; i++)
                target[i] |= value;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseXor([NoAlias] long* target, int n, long value) {
            for (int i = 0; i < n; i++)
                target[i] ^= value;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseLeftShift([NoAlias] long* target, int n, int shift) {
            for (int i = 0; i < n; i++)
                target[i] <<= shift;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseRightShift([NoAlias] long* target, int n, int shift) {
            for (int i = 0; i < n; i++)
                target[i] >>= shift;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseLeftShift(int value, [NoAlias] long* TargetWShift, int n) {
            for (int i = 0; i < n; i++)
                TargetWShift[i] = (long)(value << (int)TargetWShift[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseRightShift(int value, [NoAlias] long* TargetWShift, int n) {
            for (int i = 0; i < n; i++)
                TargetWShift[i] = (long)(value >> (int)TargetWShift[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseAndComp([NoAlias] long* targetA, [NoAlias] long* b, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] &= b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseOrComp([NoAlias] long* targetA, [NoAlias] long* b, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] |= b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseXorComp([NoAlias] long* targetA, [NoAlias] long* b, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] ^= b[i];
        }

        // do something about bitwise, it is breaking compatibility
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseLeftShiftComp([NoAlias] long* targetA, [NoAlias] long* shiftValues, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] <<= (int)shiftValues[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseRightShiftComp([NoAlias] long* targetA, [NoAlias] long* shiftValues, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] >>= (int)shiftValues[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        [BurstCompile]
        // Swap rows in a matrix 
        public static void swapRows([NoAlias] long* target, int rowA, int rowB, int nCols, int colStart = 0, int colEnd = -1) {

            int rowIndexA = rowA * nCols;
            int rowIndexB = rowB * nCols;

            if (colEnd == -1)
                colEnd = nCols;

            for (int i = colStart; i < colEnd; i++) {
                long temp = target[rowIndexA + i];
                target[rowIndexA + i] = target[rowIndexB + i];
                target[rowIndexB + i] = temp;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]

        // Swap columns in a matrix
        public static void swapColumns([NoAlias] long* target, int colA, int colB, int nRows, int nCols, int start = 0, int end = -1) {
            int startA = colA;
            int startB = colB;

            if (end == -1)
                end = nRows;

            for (int i = start; i < end; i++) {
                long temp = target[startA + i * nCols];
                target[startA + i * nCols] = target[startB + i * nCols];
                target[startB + i * nCols] = temp;
            }
        }
    }
}