#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Mathematics;
using System.Runtime.CompilerServices;
using Unity.Burst;

namespace LinearAlgebra
{
    public static unsafe partial class UnsafeOP {

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static iProxy sum([NoAlias] iProxy* a, int n) {

            iProxy sum = 0;

            for (int i = 0; i < n; i++)
                sum += a[i];
            
            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static iProxy sumAbs([NoAlias] iProxy* a, int n)
        {
            iProxy sum = 0;

            for (int i = 0; i < n; i++)
                sum += (iProxy)(a[i] < 0? -a[i] : a[i]);
            
            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static iProxy maxAbs([NoAlias] iProxy* a, int n)
        {
            iProxy max = 0;

            for (int i = 0; i < n; i++) {
                var abs = (a[i] < 0 ? -a[i] : a[i]);
                max = (iProxy)(max < abs? abs : max);
            }
            return max;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static iProxy vecDot([NoAlias] iProxy* vA, [NoAlias] iProxy* vB, int n) {

            iProxy sum = 0;

            for (int i = 0; i < n; i++) {
                sum += (iProxy)(vA[i] * vB[i]);
            }

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static iProxy vecDotRange([NoAlias] iProxy* vA, [NoAlias] iProxy* vB, int start, int end)
        {
            iProxy sum = 0;

            for (int i = start; i < end; i++)
            {
                sum += (iProxy)(vA[i] * vB[i]);
            }

            return sum;
        }



        // outer dot product (vec x vec => mat)
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void vecOuterDot([NoAlias] iProxy* vA, [NoAlias] iProxy* vB, [NoAlias] iProxy* mat, int m, int n)
        {
            //mat doesn't need to be initialized to zero
            for (int r = 0; r < m; r++)
            for (int c = 0; c < n; c++)
            {
                mat[r * n + c] = (iProxy)(vA[r] * vB[c]);
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matVecDot([NoAlias] iProxy* mat, [NoAlias] iProxy* x, [NoAlias] iProxy* y, int m, int n)
        {
            // mat = m x n
            // x = n
            // y = m, needs to be initialized to zero
            for (int r = 0; r < m; r++)
            {
                for(int c = 0; c < n; c++)
                {
                    y[r] += (iProxy)(mat[r * n + c] * x[c]);
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void vecMatDot([NoAlias] iProxy* y, [NoAlias] iProxy* mat, [NoAlias] iProxy* x, int m, int n)
        {
            // mat = m x n
            // y = inVec = m
            // x = outVec = n, needs to be initialized to zero
            for (int c = 0; c < n; c++)
            {
                for (int r = 0; r < m; r++)
                {
                    x[c] += (iProxy)(mat[r * n + c] * y[r]);
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matTrans([NoAlias] iProxy* matA, [NoAlias] iProxy* matB, int m, int n)
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
        public static void matMatDot([NoAlias] iProxy* matA, [NoAlias] iProxy* matB, [NoAlias] iProxy* matC, int m, int n, int k)
        {
            // matA = m x n
            // matB = n x k
            // matC = outMat = m x k, needs to be initialized to zero

            for (int r = 0; r < m; r++)
            {
                for (int nCols = 0; nCols < n; nCols++)
                {
                    iProxy temp = matA[r * n + nCols]; // Cache the value from matA
                    for (int kCols = 0; kCols < k; kCols++)
                    {
                        matC[r * k + kCols] += (iProxy)(temp * matB[nCols * k + kCols]);
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matMatDotTransA([NoAlias] iProxy* matA, [NoAlias] iProxy* matB, [NoAlias] iProxy* matC, int m, int n, int k)
        {
            // matA = m x n, but treated as n x m due to transposition
            // matB = n x k
            // matC = outMat = m x k, needs to be initialized to zero
            for (int r = 0; r < m; r++)
            {
                for (int nCols = 0; nCols < n; nCols++)
                {
                    iProxy temp = matA[nCols * m + r];
                    for (int kCols = 0; kCols < k; kCols++)
                    {
                        matC[r * k + kCols] += (iProxy)(temp * matB[nCols * k + kCols]);
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void signFlip([NoAlias] iProxy* target, [NoAlias] iProxy* from, int n) {

            for (int i = 0; i < n; i++)
                target[i] = (iProxy)(-from[i]);
        }
        
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compAdd([NoAlias] iProxy* target, [NoAlias] iProxy* from, int n) {

            for (int i = 0; i < n; i++)
                target[i] += from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compSub([NoAlias] iProxy* from, [NoAlias] iProxy* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] -= from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalAdd([NoAlias] iProxy* target, int n, iProxy s) {

            for (int i = 0; i < n; i++)
                target[i] += s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalSub(iProxy s, [NoAlias] iProxy* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = (iProxy)(s - target[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compMul([NoAlias] iProxy* from, [NoAlias] iProxy* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] *= from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compDiv([NoAlias] iProxy* targetDividend, [NoAlias] iProxy* fromDivisor, int n)
        {            
            for (int i = 0; i < n; i++)
                targetDividend[i] /= fromDivisor[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compMod([NoAlias] iProxy* targetDividend, [NoAlias] iProxy* fromDivisor, int n)
        {
            for (int i = 0; i < n; i++)
                targetDividend[i] %= fromDivisor[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMul([NoAlias] iProxy* target, int n, iProxy s)
        {
            for (int i = 0; i < n; i++)
                target[i] *= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalDiv([NoAlias] iProxy* target, int n, iProxy s)
        {
            for (int i = 0; i < n; i++)
                target[i] /= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalDiv(iProxy s, [NoAlias] iProxy* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = (iProxy)(s / target[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMod([NoAlias] iProxy* target, int n, iProxy s)
        {
            for (int i = 0; i < n; i++)
                target[i] %= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMod(iProxy s, [NoAlias] iProxy* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = (iProxy)(s % target[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseComplement([NoAlias] iProxy* target, int n) {
            for (int i = 0; i < n; i++)
                target[i] = (iProxy)(~target[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseAnd([NoAlias] iProxy* target, int n, iProxy value) {
            for (int i = 0; i < n; i++)
                target[i] &= value;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseOr([NoAlias] iProxy* target, int n, iProxy value) {
            for (int i = 0; i < n; i++)
                target[i] |= value;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseXor([NoAlias] iProxy* target, int n, iProxy value) {
            for (int i = 0; i < n; i++)
                target[i] ^= value;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseLeftShift([NoAlias] iProxy* target, int n, int shift) {
            for (int i = 0; i < n; i++)
                target[i] <<= shift;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseRightShift([NoAlias] iProxy* target, int n, int shift) {
            for (int i = 0; i < n; i++)
                target[i] >>= shift;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseLeftShift(int value, [NoAlias] iProxy* TargetWShift, int n) {
            for (int i = 0; i < n; i++)
                TargetWShift[i] = (iProxy)(value << (int)TargetWShift[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseRightShift(int value, [NoAlias] iProxy* TargetWShift, int n) {
            for (int i = 0; i < n; i++)
                TargetWShift[i] = (iProxy)(value >> (int)TargetWShift[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseAndComp([NoAlias] iProxy* targetA, [NoAlias] iProxy* b, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] &= b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseOrComp([NoAlias] iProxy* targetA, [NoAlias] iProxy* b, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] |= b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseXorComp([NoAlias] iProxy* targetA, [NoAlias] iProxy* b, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] ^= b[i];
        }

        // do something about bitwise, it is breaking compatibility
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseLeftShiftComp([NoAlias] iProxy* targetA, [NoAlias] iProxy* shiftValues, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] <<= (int)shiftValues[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseRightShiftComp([NoAlias] iProxy* targetA, [NoAlias] iProxy* shiftValues, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] >>= (int)shiftValues[i];
        }

    }
}