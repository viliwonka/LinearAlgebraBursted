#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Mathematics;
using System.Runtime.CompilerServices;
using Unity.Burst;

namespace LinearAlgebra
{
    public static unsafe partial class UnsafeOP {

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static int sum([NoAlias] int* a, int n) {

            int sum = 0;

            for (int i = 0; i < n; i++)
                sum += a[i];
            
            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static int sumAbs([NoAlias] int* a, int n)
        {
            int sum = 0;

            for (int i = 0; i < n; i++)
                sum += (int)(a[i] < 0? -a[i] : a[i]);
            
            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static int maxAbs([NoAlias] int* a, int n)
        {
            int max = 0;

            for (int i = 0; i < n; i++) {
                var abs = (a[i] < 0 ? -a[i] : a[i]);
                max = (int)(max < abs? abs : max);
            }
            return max;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static int vecDot([NoAlias] int* vA, [NoAlias] int* vB, int n) {

            int sum = 0;

            for (int i = 0; i < n; i++) {
                sum += (int)(vA[i] * vB[i]);
            }

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static int vecDotRange([NoAlias] int* vA, [NoAlias] int* vB, int start, int end)
        {
            int sum = 0;

            for (int i = start; i < end; i++)
            {
                sum += (int)(vA[i] * vB[i]);
            }

            return sum;
        }



        // outer dot product (vec x vec => mat)
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void vecOuterDot([NoAlias] int* vA, [NoAlias] int* vB, [NoAlias] int* mat, int m, int n)
        {
            //mat doesn't need to be initialized to zero
            for (int r = 0; r < m; r++)
            for (int c = 0; c < n; c++)
            {
                mat[r * n + c] = (int)(vA[r] * vB[c]);
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matVecDot([NoAlias] int* mat, [NoAlias] int* x, [NoAlias] int* y, int m, int n)
        {
            // mat = m x n
            // x = n
            // y = m, needs to be initialized to zero
            for (int r = 0; r < m; r++)
            {
                for(int c = 0; c < n; c++)
                {
                    y[r] += (int)(mat[r * n + c] * x[c]);
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void vecMatDot([NoAlias] int* y, [NoAlias] int* mat, [NoAlias] int* x, int m, int n)
        {
            // mat = m x n
            // y = inVec = m
            // x = outVec = n, needs to be initialized to zero
            for (int c = 0; c < n; c++)
            {
                for (int r = 0; r < m; r++)
                {
                    x[c] += (int)(mat[r * n + c] * y[r]);
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matTrans([NoAlias] int* matA, [NoAlias] int* matB, int m, int n)
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
        public static void matMatDot([NoAlias] int* matA, [NoAlias] int* matB, [NoAlias] int* matC, int m, int n, int k)
        {
            // matA = m x n
            // matB = n x k
            // matC = outMat = m x k, needs to be initialized to zero

            for (int r = 0; r < m; r++)
            {
                for (int nCols = 0; nCols < n; nCols++)
                {
                    int temp = matA[r * n + nCols]; // Cache the value from matA
                    for (int kCols = 0; kCols < k; kCols++)
                    {
                        matC[r * k + kCols] += (int)(temp * matB[nCols * k + kCols]);
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matMatDotTransA([NoAlias] int* matA, [NoAlias] int* matB, [NoAlias] int* matC, int m, int n, int k)
        {
            // matA = m x n, but treated as n x m due to transposition
            // matB = n x k
            // matC = outMat = m x k, needs to be initialized to zero
            for (int r = 0; r < m; r++)
            {
                for (int nCols = 0; nCols < n; nCols++)
                {
                    int temp = matA[nCols * m + r];
                    for (int kCols = 0; kCols < k; kCols++)
                    {
                        matC[r * k + kCols] += (int)(temp * matB[nCols * k + kCols]);
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void signFlip([NoAlias] int* target, [NoAlias] int* from, int n) {

            for (int i = 0; i < n; i++)
                target[i] = (int)(-from[i]);
        }
        
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compAdd([NoAlias] int* target, [NoAlias] int* from, int n) {

            for (int i = 0; i < n; i++)
                target[i] += from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compSub([NoAlias] int* from, [NoAlias] int* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] -= from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalAdd([NoAlias] int* target, int n, int s) {

            for (int i = 0; i < n; i++)
                target[i] += s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalSub(int s, [NoAlias] int* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = (int)(s - target[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compMul([NoAlias] int* from, [NoAlias] int* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] *= from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compDiv([NoAlias] int* targetDividend, [NoAlias] int* fromDivisor, int n)
        {            
            for (int i = 0; i < n; i++)
                targetDividend[i] /= fromDivisor[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compMod([NoAlias] int* targetDividend, [NoAlias] int* fromDivisor, int n)
        {
            for (int i = 0; i < n; i++)
                targetDividend[i] %= fromDivisor[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMul([NoAlias] int* target, int n, int s)
        {
            for (int i = 0; i < n; i++)
                target[i] *= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalDiv([NoAlias] int* target, int n, int s)
        {
            for (int i = 0; i < n; i++)
                target[i] /= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalDiv(int s, [NoAlias] int* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = (int)(s / target[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMod([NoAlias] int* target, int n, int s)
        {
            for (int i = 0; i < n; i++)
                target[i] %= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMod(int s, [NoAlias] int* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = (int)(s % target[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseComplement([NoAlias] int* target, int n) {
            for (int i = 0; i < n; i++)
                target[i] = (int)(~target[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseAnd([NoAlias] int* target, int n, int value) {
            for (int i = 0; i < n; i++)
                target[i] &= value;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseOr([NoAlias] int* target, int n, int value) {
            for (int i = 0; i < n; i++)
                target[i] |= value;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseXor([NoAlias] int* target, int n, int value) {
            for (int i = 0; i < n; i++)
                target[i] ^= value;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseLeftShift([NoAlias] int* target, int n, int shift) {
            for (int i = 0; i < n; i++)
                target[i] <<= shift;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseRightShift([NoAlias] int* target, int n, int shift) {
            for (int i = 0; i < n; i++)
                target[i] >>= shift;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseLeftShift(int value, [NoAlias] int* TargetWShift, int n) {
            for (int i = 0; i < n; i++)
                TargetWShift[i] = (int)(value << (int)TargetWShift[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseRightShift(int value, [NoAlias] int* TargetWShift, int n) {
            for (int i = 0; i < n; i++)
                TargetWShift[i] = (int)(value >> (int)TargetWShift[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseAndComp([NoAlias] int* targetA, [NoAlias] int* b, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] &= b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseOrComp([NoAlias] int* targetA, [NoAlias] int* b, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] |= b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseXorComp([NoAlias] int* targetA, [NoAlias] int* b, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] ^= b[i];
        }

        // do something about bitwise, it is breaking compatibility
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseLeftShiftComp([NoAlias] int* targetA, [NoAlias] int* shiftValues, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] <<= (int)shiftValues[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseRightShiftComp([NoAlias] int* targetA, [NoAlias] int* shiftValues, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] >>= (int)shiftValues[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        [BurstCompile]
        // Swap rows in a matrix 
        public static void swapRows([NoAlias] int* target, int rowA, int rowB, int nCols, int colStart = 0, int colEnd = -1) {

            int rowIndexA = rowA * nCols;
            int rowIndexB = rowB * nCols;

            if (colEnd == -1)
                colEnd = nCols;

            for (int i = colStart; i < colEnd; i++) {
                int temp = target[rowIndexA + i];
                target[rowIndexA + i] = target[rowIndexB + i];
                target[rowIndexB + i] = temp;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]

        // Swap columns in a matrix
        public static void swapColumns([NoAlias] int* target, int colA, int colB, int nRows, int nCols, int start = 0, int end = -1) {
            int startA = colA;
            int startB = colB;

            if (end == -1)
                end = nRows;

            for (int i = start; i < end; i++) {
                int temp = target[startA + i * nCols];
                target[startA + i * nCols] = target[startB + i * nCols];
                target[startB + i * nCols] = temp;
            }
        }
    }
}