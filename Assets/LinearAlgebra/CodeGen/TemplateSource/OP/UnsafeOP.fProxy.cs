#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Mathematics;
using System.Runtime.CompilerServices;
using Unity.Burst;

namespace LinearAlgebra
{
    public static unsafe partial class UnsafeOP {

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static fProxy sum([NoAlias] fProxy* a, int n) {

            fProxy sum = 0f;

            for (int i = 0; i < n; i++)
                sum += a[i];
            
            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static fProxy sumAbs([NoAlias] fProxy* a, int n)
        {
            fProxy sum = 0f;

            for (int i = 0; i < n; i++)
                sum += math.abs(a[i]);
            
            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static fProxy maxAbs([NoAlias] fProxy* a, int n)
        {
            fProxy max = 0f;

            for (int i = 0; i < n; i++)
                max = math.max(max, math.abs(a[i]));

            return max;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static fProxy vecDot([NoAlias] fProxy* vA, [NoAlias] fProxy* vB, int n) {

            fProxy sum = 0f;

            for (int i = 0; i < n; i++) {
                sum += vA[i] * vB[i];
            }

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static fProxy vecDotRange([NoAlias] fProxy* vA, [NoAlias] fProxy* vB, int start, int end)
        {
            fProxy sum = 0f;

            for (int i = start; i < end; i++)
            {
                sum += vA[i] * vB[i];
            }

            return sum;
        }



        // outer dot product (vec x vec => mat)
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void vecOuterDot([NoAlias] fProxy* vA, [NoAlias] fProxy* vB, [NoAlias] fProxy* mat, int m, int n)
        {
            //mat doesn't need to be initialized to zero
            for (int r = 0; r < m; r++)
            for (int c = 0; c < n; c++)
            {
                mat[r * n + c] = vA[r] * vB[c];
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matVecDot([NoAlias] fProxy* mat, [NoAlias] fProxy* x, [NoAlias] fProxy* y, int m, int n)
        {
            // mat = m x n
            // x = n
            // y = m, needs to be initialized to zero
            // y = mat * x
            for (int r = 0; r < m; r++)
            {
                for(int c = 0; c < n; c++)
                {
                    y[r] += mat[r * n + c] * x[c];
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void vecMatDot([NoAlias] fProxy* y, [NoAlias] fProxy* mat, [NoAlias] fProxy* x, int m, int n)
        {
            // mat = m x n
            // y = inVec = m
            // x = outVec = n, needs to be initialized to zero
            // x = y^T * mat
            for (int c = 0; c < n; c++)
            {
                for (int r = 0; r < m; r++)
                {
                    x[c] += y[r] * mat[r * n + c];
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matTrans([NoAlias] fProxy* matA, [NoAlias] fProxy* matB, int m, int n)
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
        public static void matMatDot([NoAlias] fProxy* matA, [NoAlias] fProxy* matB, [NoAlias] fProxy* matC, int m, int n, int k)
        {
            // matA = m x n
            // matB = n x k
            // matC = outMat = m x k, needs to be initialized to zero

            for (int r = 0; r < m; r++)
            {
                for (int nCols = 0; nCols < n; nCols++)
                {
                    fProxy temp = matA[r * n + nCols]; // Cache the value from matA
                    for (int kCols = 0; kCols < k; kCols++)
                    {
                        matC[r * k + kCols] += temp * matB[nCols * k + kCols];
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matMatDotTransA([NoAlias] fProxy* matA, [NoAlias] fProxy* matB, [NoAlias] fProxy* matC, int m, int n, int k)
        {
            // matA = m x n, but treated as n x m due to transposition
            // matB = n x k
            // matC = outMat = m x k, needs to be initialized to zero
            for (int r = 0; r < m; r++)
            {
                for (int nCols = 0; nCols < n; nCols++)
                {
                    fProxy temp = matA[nCols * m + r];
                    for (int kCols = 0; kCols < k; kCols++)
                    {
                        matC[r * k + kCols] += temp * matB[nCols * k + kCols];
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void signFlip([NoAlias] fProxy* target, [NoAlias] fProxy* from, int n) {

            for (int i = 0; i < n; i++)
                target[i] = -from[i];
        }
        
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compAdd([NoAlias] fProxy* target, [NoAlias] fProxy* from, int n) {

            for (int i = 0; i < n; i++)
                target[i] += from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compSub([NoAlias] fProxy* from, [NoAlias] fProxy* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] -= from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalAdd([NoAlias] fProxy* target, int n, fProxy s) {

            for (int i = 0; i < n; i++)
                target[i] += s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalSub(fProxy s, [NoAlias] fProxy* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = s - target[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compMul([NoAlias] fProxy* from, [NoAlias] fProxy* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] *= from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compDiv([NoAlias] fProxy* targetDividend, [NoAlias] fProxy* fromDivisor, int n)
        {            
            for (int i = 0; i < n; i++)
                targetDividend[i] /= fromDivisor[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compMod([NoAlias] fProxy* targetDividend, [NoAlias] fProxy* fromDivisor, int n)
        {
            for (int i = 0; i < n; i++)
                targetDividend[i] %= fromDivisor[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMul([NoAlias] fProxy* target, int n, fProxy s)
        {
            for (int i = 0; i < n; i++)
                target[i] *= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalDiv([NoAlias] fProxy* target, int n, fProxy s)
        {
            for (int i = 0; i < n; i++)
                target[i] /= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalDiv(fProxy s, [NoAlias] fProxy* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = s / target[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMod([NoAlias] fProxy* target, int n, fProxy s)
        {
            for (int i = 0; i < n; i++)
                target[i] %= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMod(fProxy s, [NoAlias] fProxy* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = s % target[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void normalizeL2Inpl([NoAlias] fProxy* target, int n)
        {
            fProxy sum = 0f;

            for (int i = 0; i < n; i++)
                sum += target[i] * target[i];
            
            sum = math.sqrt(sum);

            for (int i = 0; i < n; i++)
                target[i] /= sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static fProxy normalizeL2Inpl([NoAlias] fProxy* target, int start, int end)
        {
            fProxy sum = 0f;

            for (int i = start; i < end; i++)
                sum += target[i] * target[i];

            sum = math.sqrt(sum);

            for (int i = start; i < end; i++)
                target[i] = (target[i] / sum);

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static fProxy normalizeL1([NoAlias] fProxy* target, int n)
        {
            fProxy sum = 0f;

            for (int i = 0; i < n; i++)
                sum += math.abs(target[i]);

            for (int i = 0; i < n; i++)
                target[i] /= sum;

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static fProxy normalizeL1([NoAlias] fProxy* target, int start, int end)
        {
            fProxy sum = 0f;

            for (int i = start; i < end; i++)
                sum += math.abs(target[i]);

            for (int i = start; i < end; i++)
                target[i] = (target[i] / sum);

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static fProxy normalizeLMax([NoAlias] fProxy* target, int n)
        {
            fProxy max = 0f;

            for (int i = 0; i < n; i++)
                max = math.max(max, math.abs(target[i]));

            for (int i = 0; i < n; i++)
                target[i] /= max;

            return max;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static fProxy normalizeLMax([NoAlias] fProxy* target, int start, int end)
        {
            fProxy max = 0f;

            for (int i = start; i < end; i++)
                max = math.max(max, math.abs(target[i]));

            for (int i = start; i < end; i++)
                target[i] /= max;

            return max;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static fProxy normalizeLP([NoAlias] fProxy* target, int n, fProxy p)
        {
            fProxy sum = 0f;

            for (int i = 0; i < n; i++)
                sum += math.pow(target[i], p);

            sum = math.pow(sum, 1f / p);

            for (int i = 0; i < n; i++)
                target[i] /= sum;

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static fProxy normalizeLP([NoAlias] fProxy* target, int start, int end, fProxy p)
        {
            fProxy sum = 0f;

            for (int i = start; i < end; i++)
                sum += math.pow(target[i], p);

            sum = math.pow(sum, 1f / p);

            for (int i = start; i < end; i++)
                target[i] = (target[i] / sum);

            return sum;
        }
    }
}