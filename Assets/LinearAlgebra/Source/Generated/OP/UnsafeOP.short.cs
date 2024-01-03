#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Mathematics;
using System.Runtime.CompilerServices;
using Unity.Burst;

namespace LinearAlgebra
{
    public static unsafe partial class UnsafeOP {

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static short sum([NoAlias] short* a, int n) {

            short sum = 0;

            for (int i = 0; i < n; i++)
                sum += a[i];
            
            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static short sumAbs([NoAlias] short* a, int n)
        {
            short sum = 0;

            for (int i = 0; i < n; i++)
                sum += (short)(a[i] < 0? -a[i] : a[i]);
            
            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static short maxAbs([NoAlias] short* a, int n)
        {
            short max = 0;

            for (int i = 0; i < n; i++) {
                var abs = (a[i] < 0 ? -a[i] : a[i]);
                max = (short)(max < abs? abs : max);
            }
            return max;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static short vecDot([NoAlias] short* vA, [NoAlias] short* vB, int n) {

            short sum = 0;

            for (int i = 0; i < n; i++) {
                sum += (short)(vA[i] * vB[i]);
            }

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static short vecDotRange([NoAlias] short* vA, [NoAlias] short* vB, int start, int end)
        {
            short sum = 0;

            for (int i = start; i < end; i++)
            {
                sum += (short)(vA[i] * vB[i]);
            }

            return sum;
        }



        // outer dot product (vec x vec => mat)
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void vecOuterDot([NoAlias] short* vA, [NoAlias] short* vB, [NoAlias] short* mat, int m, int n)
        {
            //mat doesn't need to be initialized to zero
            for (int r = 0; r < m; r++)
            for (int c = 0; c < n; c++)
            {
                mat[r * n + c] = (short)(vA[r] * vB[c]);
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matVecDot([NoAlias] short* mat, [NoAlias] short* x, [NoAlias] short* y, int m, int n)
        {
            // mat = m x n
            // x = n
            // y = m, needs to be initialized to zero
            for (int r = 0; r < m; r++)
            {
                for(int c = 0; c < n; c++)
                {
                    y[r] += (short)(mat[r * n + c] * x[c]);
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void vecMatDot([NoAlias] short* y, [NoAlias] short* mat, [NoAlias] short* x, int m, int n)
        {
            // mat = m x n
            // y = inVec = m
            // x = outVec = n, needs to be initialized to zero
            for (int c = 0; c < n; c++)
            {
                for (int r = 0; r < m; r++)
                {
                    x[c] += (short)(mat[r * n + c] * y[r]);
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matTrans([NoAlias] short* matA, [NoAlias] short* matB, int m, int n)
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
        public static void matMatDot([NoAlias] short* matA, [NoAlias] short* matB, [NoAlias] short* matC, int m, int n, int k)
        {
            // matA = m x n
            // matB = n x k
            // matC = outMat = m x k, needs to be initialized to zero

            for (int r = 0; r < m; r++)
            {
                for (int nCols = 0; nCols < n; nCols++)
                {
                    short temp = matA[r * n + nCols]; // Cache the value from matA
                    for (int kCols = 0; kCols < k; kCols++)
                    {
                        matC[r * k + kCols] += (short)(temp * matB[nCols * k + kCols]);
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matMatDotTransA([NoAlias] short* matA, [NoAlias] short* matB, [NoAlias] short* matC, int m, int n, int k)
        {
            // matA = m x n, but treated as n x m due to transposition
            // matB = n x k
            // matC = outMat = m x k, needs to be initialized to zero
            for (int r = 0; r < m; r++)
            {
                for (int nCols = 0; nCols < n; nCols++)
                {
                    short temp = matA[nCols * m + r];
                    for (int kCols = 0; kCols < k; kCols++)
                    {
                        matC[r * k + kCols] += (short)(temp * matB[nCols * k + kCols]);
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void signFlip([NoAlias] short* target, [NoAlias] short* from, int n) {

            for (int i = 0; i < n; i++)
                target[i] = (short)(-from[i]);
        }
        
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compAdd([NoAlias] short* target, [NoAlias] short* from, int n) {

            for (int i = 0; i < n; i++)
                target[i] += from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compSub([NoAlias] short* from, [NoAlias] short* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] -= from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalAdd([NoAlias] short* target, int n, short s) {

            for (int i = 0; i < n; i++)
                target[i] += s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalSub(short s, [NoAlias] short* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = (short)(s - target[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compMul([NoAlias] short* from, [NoAlias] short* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] *= from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compDiv([NoAlias] short* targetDividend, [NoAlias] short* fromDivisor, int n)
        {            
            for (int i = 0; i < n; i++)
                targetDividend[i] /= fromDivisor[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compMod([NoAlias] short* targetDividend, [NoAlias] short* fromDivisor, int n)
        {
            for (int i = 0; i < n; i++)
                targetDividend[i] %= fromDivisor[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMul([NoAlias] short* target, int n, short s)
        {
            for (int i = 0; i < n; i++)
                target[i] *= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalDiv([NoAlias] short* target, int n, short s)
        {
            for (int i = 0; i < n; i++)
                target[i] /= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalDiv(short s, [NoAlias] short* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = (short)(s / target[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMod([NoAlias] short* target, int n, short s)
        {
            for (int i = 0; i < n; i++)
                target[i] %= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMod(short s, [NoAlias] short* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = (short)(s % target[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseComplement([NoAlias] short* target, int n) {
            for (int i = 0; i < n; i++)
                target[i] = (short)(~target[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseAnd([NoAlias] short* target, int n, short value) {
            for (int i = 0; i < n; i++)
                target[i] &= value;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseOr([NoAlias] short* target, int n, short value) {
            for (int i = 0; i < n; i++)
                target[i] |= value;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseXor([NoAlias] short* target, int n, short value) {
            for (int i = 0; i < n; i++)
                target[i] ^= value;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseLeftShift([NoAlias] short* target, int n, int shift) {
            for (int i = 0; i < n; i++)
                target[i] <<= shift;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseRightShift([NoAlias] short* target, int n, int shift) {
            for (int i = 0; i < n; i++)
                target[i] >>= shift;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseLeftShift(int value, [NoAlias] short* TargetWShift, int n) {
            for (int i = 0; i < n; i++)
                TargetWShift[i] = (short)(value << (int)TargetWShift[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseRightShift(int value, [NoAlias] short* TargetWShift, int n) {
            for (int i = 0; i < n; i++)
                TargetWShift[i] = (short)(value >> (int)TargetWShift[i]);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseAndComp([NoAlias] short* targetA, [NoAlias] short* b, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] &= b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseOrComp([NoAlias] short* targetA, [NoAlias] short* b, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] |= b[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseXorComp([NoAlias] short* targetA, [NoAlias] short* b, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] ^= b[i];
        }

        // do something about bitwise, it is breaking compatibility
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseLeftShiftComp([NoAlias] short* targetA, [NoAlias] short* shiftValues, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] <<= (int)shiftValues[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bitwiseRightShiftComp([NoAlias] short* targetA, [NoAlias] short* shiftValues, int n) {
            for (int i = 0; i < n; i++)
                targetA[i] >>= (int)shiftValues[i];
        }

    }
}