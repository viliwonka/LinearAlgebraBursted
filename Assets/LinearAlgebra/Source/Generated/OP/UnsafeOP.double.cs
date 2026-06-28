#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Mathematics;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Collections;

namespace LinearAlgebra
{
    public static unsafe partial class UnsafeOP {

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static double sum([NoAlias] double* a, int n) {

            double sum = 0f;

            for (int i = 0; i < n; i++)
                sum += a[i];
            
            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static double sumAbs([NoAlias] double* a, int n)
        {
            double sum = 0f;

            for (int i = 0; i < n; i++)
                sum += math.abs(a[i]);
            
            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static double maxAbs([NoAlias] double* a, int n)
        {
            double max = 0f;

            for (int i = 0; i < n; i++)
                max = math.max(max, math.abs(a[i]));

            return max;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static double vecDot([NoAlias] double* vA, [NoAlias] double* vB, int n) {

            double sum = 0f;

            for (int i = 0; i < n; i++) {
                sum += vA[i] * vB[i];
            }

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static double vecDotRange([NoAlias] double* vA, [NoAlias] double* vB, int start, int end)
        {
            double sum = 0f;

            for (int i = start; i < end; i++)
            {
                sum += vA[i] * vB[i];
            }

            return sum;
        }



        // outer dot product (vec x vec => mat)
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void vecOuterDot([NoAlias] double* vA, [NoAlias] double* vB, [NoAlias] double* mat, int m, int n)
        {
            //mat doesn't need to be initialized to zero
            for (int r = 0; r < m; r++)
            for (int c = 0; c < n; c++)
            {
                mat[r * n + c] = vA[r] * vB[c];
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matVecDot([NoAlias] double* mat, [NoAlias] double* x, [NoAlias] double* y, int m, int n)
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
        public static void vecMatDot([NoAlias] double* y, [NoAlias] double* mat, [NoAlias] double* x, int m, int n)
        {
            // mat = m x n
            // y = inVec = m
            // x = outVec = n
            // x = y^T * mat
            // Zero result first, then accumulate row-wise so mat[baseIdx + c] is unit-stride in c.
            UnsafeUtility.MemClear(x, (long)n * UnsafeUtility.SizeOf<double>());
            for (int r = 0; r < m; r++)
            {
                double yr = y[r];
                int baseIdx = r * n;
                for (int c = 0; c < n; c++)
                {
                    x[c] += yr * mat[baseIdx + c];
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matTrans([NoAlias] double* matA, [NoAlias] double* matB, int m, int n)
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
        public static void matMatDot([NoAlias] double* matA, [NoAlias] double* matB, [NoAlias] double* matC, int m, int n, int k)
        {
            // matA = m x n
            // matB = n x k
            // matC = outMat = m x k, needs to be initialized to zero

            for (int r = 0; r < m; r++)
            {
                for (int nCols = 0; nCols < n; nCols++)
                {
                    double temp = matA[r * n + nCols]; // Cache the value from matA
                    for (int kCols = 0; kCols < k; kCols++)
                    {
                        matC[r * k + kCols] += temp * matB[nCols * k + kCols];
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matMatDotTransA([NoAlias] double* matA, [NoAlias] double* matB, [NoAlias] double* matC, int m, int n, int k)
        {
            // matA = m x n, but treated as n x m due to transposition
            // matB = n x k
            // matC = outMat = m x k, needs to be initialized to zero
            for (int r = 0; r < m; r++)
            {
                for (int nCols = 0; nCols < n; nCols++)
                {
                    double temp = matA[nCols * m + r];
                    for (int kCols = 0; kCols < k; kCols++)
                    {
                        matC[r * k + kCols] += temp * matB[nCols * k + kCols];
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void signFlip([NoAlias] double* target, [NoAlias] double* from, int n) {

            for (int i = 0; i < n; i++)
                target[i] = -from[i];
        }
        
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compAdd([NoAlias] double* target, [NoAlias] double* from, int n) {

            for (int i = 0; i < n; i++)
                target[i] += from[i];
        }

        // y[i] += a * x[i]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void axpy([NoAlias] double* y, [NoAlias] double* x, double a, int n) {

            for (int i = 0; i < n; i++)
                y[i] += a * x[i];
        }

        // y[i] = a * y[i] + x[i]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void aypx([NoAlias] double* y, [NoAlias] double* x, double a, int n) {

            for (int i = 0; i < n; i++)
                y[i] = a * y[i] + x[i];
        }

        // acc[i] += x[i] * x[i]  — accumulate squares. Independent across i (no reduction), so it
        // vectorises; used to build per-column squared norms in a row-major sweep (QRCP pivoting).
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void addSquares([NoAlias] double* acc, [NoAlias] double* x, int n) {

            for (int i = 0; i < n; i++)
                acc[i] += x[i] * x[i];
        }

        // Plane (Givens / Jacobi) rotation of two vectors:
        //   a[i] = c*a[i] - s*b[i];   b[i] = s*a[i_old] + c*b[i]
        // a and b MUST be distinct, non-overlapping ranges (e.g. two different rows of a matrix —
        // p != q in Jacobi, i != i+1 in QL), which is what makes the [NoAlias] truthful and lets
        // Burst vectorise the butterfly (each i is independent). The SVD/eigen sweeps transpose
        // first so the rotated pair is row-contiguous, then call this.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void jacobiRotate([NoAlias] double* a, [NoAlias] double* b, double c, double s, int n) {

            for (int i = 0; i < n; i++)
            {
                double ai = a[i];
                double bi = b[i];
                a[i] = c * ai - s * bi;
                b[i] = s * ai + c * bi;
            }
        }

        // Fused 2x2 Gram dots of a vector pair: aa = a·a, bb = b·b, ab = a·b over [0,n).
        // 4-way unrolled with INDEPENDENT partial accumulators so the three reductions are not
        // latency-bound on a single loop-carried chain (recovers FMA-pipeline ILP, and lets Burst
        // pack lanes). a,b must be distinct non-overlapping ranges. NOTE: the 4-way partial-sum order
        // differs from a sequential sum, so results are rounding-level (not bitwise) different from a
        // naive accumulate — fine for the one-sided Jacobi SVD Gram step. Used by svdDecomposition.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void gram2x2([NoAlias] double* a, [NoAlias] double* b,
                                   out double aa, out double bb, out double ab, int n) {

            double a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            double b0 = 0, b1 = 0, b2 = 0, b3 = 0;
            double g0 = 0, g1 = 0, g2 = 0, g3 = 0;

            int i = 0;
            int n4 = n & ~3;
            for (; i < n4; i += 4)
            {
                double p0 = a[i],     q0 = b[i];
                double p1 = a[i + 1], q1 = b[i + 1];
                double p2 = a[i + 2], q2 = b[i + 2];
                double p3 = a[i + 3], q3 = b[i + 3];
                a0 += p0 * p0; b0 += q0 * q0; g0 += p0 * q0;
                a1 += p1 * p1; b1 += q1 * q1; g1 += p1 * q1;
                a2 += p2 * p2; b2 += q2 * q2; g2 += p2 * q2;
                a3 += p3 * p3; b3 += q3 * q3; g3 += p3 * q3;
            }

            double aSum = (a0 + a1) + (a2 + a3);
            double bSum = (b0 + b1) + (b2 + b3);
            double gSum = (g0 + g1) + (g2 + g3);

            for (; i < n; i++)
            {
                double p = a[i], q = b[i];
                aSum += p * p; bSum += q * q; gSum += p * q;
            }

            aa = aSum; bb = bSum; ab = gSum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compSub([NoAlias] double* target, [NoAlias] double* from, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] -= from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalAdd([NoAlias] double* target, int n, double s) {

            for (int i = 0; i < n; i++)
                target[i] += s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalSub(double s, [NoAlias] double* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = s - target[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compMul([NoAlias] double* from, [NoAlias] double* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] *= from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compDiv([NoAlias] double* targetDividend, [NoAlias] double* fromDivisor, int n)
        {            
            for (int i = 0; i < n; i++)
                targetDividend[i] /= fromDivisor[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compMod([NoAlias] double* targetDividend, [NoAlias] double* fromDivisor, int n)
        {
            for (int i = 0; i < n; i++)
                targetDividend[i] %= fromDivisor[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMul([NoAlias] double* target, int n, double s)
        {
            for (int i = 0; i < n; i++)
                target[i] *= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalDiv([NoAlias] double* target, int n, double s)
        {
            for (int i = 0; i < n; i++)
                target[i] /= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalDiv(double s, [NoAlias] double* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = s / target[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMod([NoAlias] double* target, int n, double s)
        {
            for (int i = 0; i < n; i++)
                target[i] %= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMod(double s, [NoAlias] double* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = s % target[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void normalizeL2Inpl([NoAlias] double* target, int n)
        {
            double sum = 0f;

            for (int i = 0; i < n; i++)
                sum += target[i] * target[i];
            
            sum = math.sqrt(sum);

            for (int i = 0; i < n; i++)
                target[i] /= sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static double normalizeL2Inpl([NoAlias] double* target, int start, int end)
        {
            double sum = 0f;

            for (int i = start; i < end; i++)
                sum += target[i] * target[i];

            sum = math.sqrt(sum);

            for (int i = start; i < end; i++)
                target[i] = (target[i] / sum);

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static double normalizeL1([NoAlias] double* target, int n)
        {
            double sum = 0f;

            for (int i = 0; i < n; i++)
                sum += math.abs(target[i]);

            for (int i = 0; i < n; i++)
                target[i] /= sum;

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static double normalizeL1([NoAlias] double* target, int start, int end)
        {
            double sum = 0f;

            for (int i = start; i < end; i++)
                sum += math.abs(target[i]);

            for (int i = start; i < end; i++)
                target[i] = (target[i] / sum);

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static double normalizeLMax([NoAlias] double* target, int n)
        {
            double max = 0f;

            for (int i = 0; i < n; i++)
                max = math.max(max, math.abs(target[i]));

            for (int i = 0; i < n; i++)
                target[i] /= max;

            return max;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static double normalizeLMax([NoAlias] double* target, int start, int end)
        {
            double max = 0f;

            for (int i = start; i < end; i++)
                max = math.max(max, math.abs(target[i]));

            for (int i = start; i < end; i++)
                target[i] /= max;

            return max;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static double normalizeLP([NoAlias] double* target, int n, double p)
        {
            double sum = 0f;

            for (int i = 0; i < n; i++)
                sum += math.pow(math.abs(target[i]), p);   // Lp norm uses |x_i|^p

            sum = math.pow(sum, 1f / p);

            for (int i = 0; i < n; i++)
                target[i] /= sum;

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static double normalizeLP([NoAlias] double* target, int start, int end, double p)
        {
            double sum = 0f;

            for (int i = start; i < end; i++)
                sum += math.pow(math.abs(target[i]), p);   // Lp norm uses |x_i|^p

            sum = math.pow(sum, 1f / p);

            for (int i = start; i < end; i++)
                target[i] = (target[i] / sum);

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void swap([NoAlias] double* target, int startA, int startB, int n) {
            
            for (int i = 0; i < n; i++) {
                double temp = target[startA + i];
                target[startA + i] = target[startB + i];
                target[startB + i] = temp;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void swap([NoAlias] double* target, int startA, int startB, int n, int stride) {

            for (int i = 0; i < n; i++) {
                double temp = target[startA + i * stride];
                target[startA + i * stride] = target[startB + i * stride];
                target[startB + i * stride] = temp;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        [BurstCompile]
        // Swap rows in a matrix 
        public static void swapRows([NoAlias] double* target, int rowA, int rowB, int nCols, int colStart = 0, int colEnd = -1) {
            
            int rowIndexA = rowA * nCols;
            int rowIndexB = rowB * nCols; 

            if(colEnd == -1)
                colEnd = nCols;

            for (int i = colStart; i < colEnd; i++) {
                double temp = target[rowIndexA + i];
                target[rowIndexA + i] = target[rowIndexB + i];
                target[rowIndexB + i] = temp;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        [BurstCompile]
        // Swap columns in a matrix
        public static void swapColumns([NoAlias] double* target, int colA, int colB, int nRows, int nCols, int start = 0, int end = -1) {
            int startA = colA;
            int startB = colB;

            if(end == -1)
                end = nRows;

            for (int i = start; i < end; i++) {
                double temp = target[startA + i * nCols];
                target[startA + i * nCols] = target[startB + i * nCols];
                target[startB + i * nCols] = temp;
            }
        }
    }
}