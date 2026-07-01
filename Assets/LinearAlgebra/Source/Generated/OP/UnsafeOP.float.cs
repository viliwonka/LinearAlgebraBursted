#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Mathematics;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Collections;

namespace LinearAlgebra.Internal
{
    public static unsafe partial class Unsafe_OP {

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static float sum([NoAlias] float* a, int n) {

            float sum = 0f;

            for (int i = 0; i < n; i++)
                sum += a[i];
            
            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static float sumAbs([NoAlias] float* a, int n)
        {
            float sum = 0f;

            for (int i = 0; i < n; i++)
                sum += math.abs(a[i]);
            
            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static float maxAbs([NoAlias] float* a, int n)
        {
            float max = 0f;

            for (int i = 0; i < n; i++)
                max = math.max(max, math.abs(a[i]));

            return max;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static float vecDot([NoAlias] float* vA, [NoAlias] float* vB, int n) {

            float sum = 0f;

            for (int i = 0; i < n; i++) {
                sum += vA[i] * vB[i];
            }

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static float vecDotRange([NoAlias] float* vA, [NoAlias] float* vB, int start, int end)
        {
            float sum = 0f;

            for (int i = start; i < end; i++)
            {
                sum += vA[i] * vB[i];
            }

            return sum;
        }



        // outer dot product (vec x vec => mat)
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void vecOuterDot([NoAlias] float* vA, [NoAlias] float* vB, [NoAlias] float* mat, int m, int n)
        {
            //mat doesn't need to be initialized to zero
            for (int r = 0; r < m; r++)
            for (int c = 0; c < n; c++)
            {
                mat[r * n + c] = vA[r] * vB[c];
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matVecDot([NoAlias] float* mat, [NoAlias] float* x, [NoAlias] float* y, int m, int n)
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
        public static void vecMatDot([NoAlias] float* y, [NoAlias] float* mat, [NoAlias] float* x, int m, int n)
        {
            // mat = m x n
            // y = inVec = m
            // x = outVec = n
            // x = y^T * mat
            // Zero result first, then accumulate row-wise so mat[baseIdx + c] is unit-stride in c.
            UnsafeUtility.MemClear(x, (long)n * UnsafeUtility.SizeOf<float>());
            for (int r = 0; r < m; r++)
            {
                float yr = y[r];
                int baseIdx = r * n;
                for (int c = 0; c < n; c++)
                {
                    x[c] += yr * mat[baseIdx + c];
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matTrans([NoAlias] float* matA, [NoAlias] float* matB, int m, int n)
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
        public static void matMatDot([NoAlias] float* matA, [NoAlias] float* matB, [NoAlias] float* matC, int m, int n, int k)
        {
            // matA = m x n
            // matB = n x k
            // matC = outMat = m x k, needs to be initialized to zero

            for (int r = 0; r < m; r++)
            {
                for (int nCols = 0; nCols < n; nCols++)
                {
                    float temp = matA[r * n + nCols]; // Cache the value from matA
                    for (int kCols = 0; kCols < k; kCols++)
                    {
                        matC[r * k + kCols] += temp * matB[nCols * k + kCols];
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matMatDotTransA([NoAlias] float* matA, [NoAlias] float* matB, [NoAlias] float* matC, int m, int n, int k)
        {
            // matA = m x n, but treated as n x m due to transposition
            // matB = n x k
            // matC = outMat = m x k, needs to be initialized to zero
            for (int r = 0; r < m; r++)
            {
                for (int nCols = 0; nCols < n; nCols++)
                {
                    float temp = matA[nCols * m + r];
                    for (int kCols = 0; kCols < k; kCols++)
                    {
                        matC[r * k + kCols] += temp * matB[nCols * k + kCols];
                    }
                }
            }
        }

        // Row-wise forward substitution ("TRSM", lower-triangular panel solve, applied one row at a
        // time): for every row t of B, solve L11 * B[t,:]ᵀ = B[t,:]ᵀ_old for B[t,:] IN PLACE, where
        // L11 is the jb x jb lower-triangular block at leading dimension Lld (row-major,
        // L11[p,k] at L11[p*Lld+k]). B is nrows x jb, row-major, leading dimension Bld; B[t,p] is
        // only read for k<p at the point column p is solved (forward substitution), so each row can
        // be overwritten in place left-to-right. Used by Cholesky's blocked (level-3) factorization
        // to compute the below-panel strip L21 from the already-factored diagonal block L11 (DTRSM):
        // ONE call solves every below-panel row for the whole panel, instead of a rank-1 update per
        // (row, column) pair — the latter keeps the same O(n^2) NoInlining-call count as the
        // unblocked sweep it's replacing (just doing less work per call), which was measured to eat
        // the paired SYRK's savings at mid-range n. [NoAlias] is truthful: L11 (rows [j0,j0+jb)) and
        // B (rows [rStart,n), rStart=j0+jb) are disjoint row ranges of the same underlying L matrix.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void trsmLowerPanel([NoAlias] float* L11, int Lld, [NoAlias] float* B, int Bld, int nrows, int jb)
        {
            for (int t = 0; t < nrows; t++)
            {
                float* Brow = B + (long)t * Bld;
                for (int p = 0; p < jb; p++)
                {
                    float s = Brow[p];
                    float* L11row = L11 + (long)p * Lld;
                    for (int k = 0; k < p; k++)
                        s -= L11row[k] * Brow[k];
                    Brow[p] = s / L11row[p];
                }
            }
        }

        // Lower-triangular SYRK subtract into the trailing diagonal block of a row-major n x n
        // matrix L:
        //   L[i,k] -= Σ_{p=0..jb-1} P[i',p] * PT[p,k']   for rStart <= k <= i < n   (i'=i-rStart, k'=k-rStart)
        // P = the panel strip L[rStart:n, j0:j0+jb] read in place from L (strided, cols j0..j0+jb-1).
        // PT = transpose of P, contiguous jb x ntrail (ntrail = n-rStart): PT[p*ntrail + k'] = P[k',p].
        // Inner loop over k' is unit-stride in both L's row and PT's row p, and TRIANGULAR (k' in
        // [0,i']), so it costs the SYRK's n^3/6, not a full-rectangular n^3/3 — do NOT extend it to
        // the strict upper triangle (that would double the flops and write past L's diagonal). Used
        // by Cholesky's blocked (level-3) factorization for the trailing-block update A22 -= L21*L21ᵀ.
        // [NoAlias] is truthful: PT is a separate Temp buffer; the P read-region (cols [j0,j0+jb)) and
        // the write-region (cols [rStart=j0+jb, i]) are disjoint column ranges of L, and the P entry is
        // loaded into `temp` before any write in that same iteration.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void syrkLowerSub([NoAlias] float* Lp, int n, int rStart, int j0, int jb, [NoAlias] float* PT)
        {
            int ntrail = n - rStart;
            for (int ip = 0; ip < ntrail; ip++)          // i' ; i = rStart+ip
            {
                int i = rStart + ip;
                float* Lrow = Lp + (long)i * n;         // write cols [rStart, i]
                float* Pi = Lp + (long)i * n + j0;      // P[ip, 0..jb)
                for (int p = 0; p < jb; p++)
                {
                    float temp = Pi[p];                  // scalar, loaded before the k' write loop (no hazard)
                    float* PTp = PT + (long)p * ntrail;
                    for (int kp = 0; kp <= ip; kp++)      // unit stride, lower triangle incl diagonal
                        Lrow[rStart + kp] -= temp * PTp[kp];
                }
            }
        }

        // ---- Compact-WY (block-reflector) helpers, τ≡1 convention (H_i = I - u_i u_iᵀ) ----
        // Used by QR's blocked (level-3) factorization/reconstruction to batch nb reflectors into
        // one GEMM-shaped trailing update  C -= V·(T·(Vᵀ·C))  instead of nb rank-1 passes. V is a
        // clean contiguous panel (rows masked to zero above each reflector's own diagonal), leading
        // dimension Vld == the panel's CURRENT pb (not necessarily QR_BLOCK — the last panel of a
        // matrix can be narrower). C is a strided sub-block of the matrix being updated (leading
        // dimension Cld == that matrix's N_Cols). W is a dense contiguous pb×cw scratch buffer (row
        // stride cw).

        // W[i,j] += Σ_{t=0..rows-1} Vp[t*Vld+i] * Cp[t*Cld+j]   (i in [0,pb), j in [0,cw)).
        // Caller must zero W first. Loop order t (outer) / i (middle) / j (inner): the j loop walks
        // Crow and Wi left-to-right (unit stride in both), the same "walk rows" trick as
        // applyReflectorRight/vecMatDot, so Burst vectorises it.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void wyVtC([NoAlias] float* Vp, int Vld, [NoAlias] float* Cp, int Cld, int rows, int pb, int cw, [NoAlias] float* W)
        {
            for (int t = 0; t < rows; t++)
            {
                float* Vrow = Vp + (long)t * Vld;
                float* Crow = Cp + (long)t * Cld;
                for (int i = 0; i < pb; i++)
                {
                    float temp = Vrow[i];
                    float* Wi = W + (long)i * cw;
                    for (int j = 0; j < cw; j++)
                        Wi[j] += temp * Crow[j];
                }
            }
        }

        // C[t,j] -= Σ_{i=0..pb-1} Vp[t*Vld+i] * W[i,j]   — the second GEMM half of the block
        // reflector apply, C -= V·W. Same unit-stride-j vectorisation shape as wyVtC.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void wySubVW([NoAlias] float* Vp, int Vld, [NoAlias] float* Cp, int Cld, int rows, int pb, int cw, [NoAlias] float* W)
        {
            for (int t = 0; t < rows; t++)
            {
                float* Vrow = Vp + (long)t * Vld;
                float* Crow = Cp + (long)t * Cld;
                for (int i = 0; i < pb; i++)
                {
                    float temp = Vrow[i];
                    float* Wi = W + (long)i * cw;
                    for (int j = 0; j < cw; j++)
                        Crow[j] -= temp * Wi[j];
                }
            }
        }

        // Used by LQ's blocked (level-3) factorization/reconstruction to compute the "folding" term
        // Y = C·Vᵀ needed for a RIGHT-multiply block reflector update  C -= Y·(T·V)  (equivalently
        // C·(I - Vᵀ T V)) — see LQ.lqDecompositionBlockedCore. This is the right-multiply mirror of
        // wyVtC's left-multiply W = Vᵀ·C. Vt is the TRANSPOSE of the clean reflector panel V (shape
        // cn x pb, contiguous, row stride pb) rather than V itself (shape pb x cn): computing Y = C·Vᵀ
        // needs to walk Vᵀ's rows (== V's columns) contiguously, so the caller pre-transposes V into
        // Vt once per panel. C is a strided sub-block of the matrix being updated (leading dimension
        // Cld). Y is dense contiguous rows×pb (row stride pb).
        //
        // An alternative that skipped the Vt transpose and instead computed each Y[t,i] as a direct
        // (4-accumulator) dot product of C's row t against Vpanel's row i was tried and measured ~2x
        // SLOWER — despite matching "row-major right-multiply is reduction-bound" (see LQ.dot4's doc
        // comment), the huge (rows*cn) outer trip count of that formulation, each doing only a
        // pb=32-wide reduction, lost badly to this version's more moderate (rows*cn) outer trip count
        // whose innermost pass is a long (pb-wide) UNIT-STRIDE accumulation with no reduction
        // dependency chain — the same axpy-shaped pattern QR's wyVtC already exploits.
        //
        // Y[t,i] += Σ_{c=0..cn-1} Cp[t*Cld+c] * Vt[c*pb+i]   (t in [0,rows), i in [0,pb)).
        // Caller must zero Y first. Loop order t (outer) / c (middle) / i (inner): the i loop walks
        // Vtrow and Yrow left-to-right (unit stride in both), the same "walk rows" trick as wyVtC.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void lqYeqCVt([NoAlias] float* Cp, int Cld, [NoAlias] float* Vt, int cn, int rows, int pb, [NoAlias] float* Y)
        {
            for (int t = 0; t < rows; t++)
            {
                float* Crow = Cp + (long)t * Cld;
                float* Yrow = Y + (long)t * pb;
                for (int c = 0; c < cn; c++)
                {
                    float temp = Crow[c];
                    float* Vtrow = Vt + (long)c * pb;
                    for (int i = 0; i < pb; i++)
                        Yrow[i] += temp * Vtrow[i];
                }
            }
        }

        // W := Tᵀ · W in place (QR's FACTORIZATION direction — applies the block product in
        // reverse reflector order, H_{pb-1}···H_0). T is pb×pb upper-triangular contiguous
        // (row-major, T[i,k] at T[i*pb+k]), diagonal T[i,i] = 1 (τ≡1 convention, not LAPACK's
        // τ-scaled diagonal). (Tᵀ)[i,k] = T[k,i], nonzero only for k <= i, so row i of the result
        // only needs W's rows 0..i — iterate i DOWNWARD so W[k] for k < i hasn't been overwritten
        // yet when it's read.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void wyTriTransMul([NoAlias] float* T, int pb, [NoAlias] float* W, int cw)
        {
            for (int i = pb - 1; i >= 0; i--)
            {
                float* Wi = W + (long)i * cw;
                float tii = T[i * pb + i];
                for (int j = 0; j < cw; j++)
                    Wi[j] = tii * Wi[j];
                for (int k = 0; k < i; k++)
                {
                    float tki = T[k * pb + i];
                    float* Wk = W + (long)k * cw;
                    for (int j = 0; j < cw; j++)
                        Wi[j] += tki * Wk[j];
                }
            }
        }

        // W := T · W in place (QR's RECONSTRUCTION direction — applies the block product in
        // forward reflector order, H_0···H_{pb-1}, un-transposed). (T·W)[i] = Σ_{k>=i} T[i,k]·W[k],
        // so row i needs rows i..pb-1 — iterate i UPWARD so W[k] for k > i hasn't been overwritten
        // yet when it's read.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void wyTriMul([NoAlias] float* T, int pb, [NoAlias] float* W, int cw)
        {
            for (int i = 0; i < pb; i++)
            {
                float* Wi = W + (long)i * cw;
                float tii = T[i * pb + i];
                for (int j = 0; j < cw; j++)
                    Wi[j] = tii * Wi[j];
                for (int k = i + 1; k < pb; k++)
                {
                    float tik = T[i * pb + k];
                    float* Wk = W + (long)k * cw;
                    for (int j = 0; j < cw; j++)
                        Wi[j] += tik * Wk[j];
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void signFlip([NoAlias] float* target, [NoAlias] float* from, int n) {

            for (int i = 0; i < n; i++)
                target[i] = -from[i];
        }
        
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compAdd([NoAlias] float* target, [NoAlias] float* from, int n) {

            for (int i = 0; i < n; i++)
                target[i] += from[i];
        }

        // y[i] += a * x[i]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void axpy([NoAlias] float* y, [NoAlias] float* x, float a, int n) {

            for (int i = 0; i < n; i++)
                y[i] += a * x[i];
        }

        // y[i] = a * y[i] + x[i]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void aypx([NoAlias] float* y, [NoAlias] float* x, float a, int n) {

            for (int i = 0; i < n; i++)
                y[i] = a * y[i] + x[i];
        }

        // acc[i] += x[i] * x[i]  — accumulate squares. Independent across i (no reduction), so it
        // vectorises; used to build per-column squared norms in a row-major sweep (QRCP pivoting).
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void addSquares([NoAlias] float* acc, [NoAlias] float* x, int n) {

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
        public static void jacobiRotate([NoAlias] float* a, [NoAlias] float* b, float c, float s, int n) {

            for (int i = 0; i < n; i++)
            {
                float ai = a[i];
                float bi = b[i];
                a[i] = c * ai - s * bi;
                b[i] = s * ai + c * bi;
            }
        }

        // Francis double-shift QR ROW update (3-row form), over a contiguous column range [0,n):
        //   p = a[i] + q*b[i] + r*c[i];   c[i] -= p*zz;   b[i] -= p*yy;   a[i] -= p*xx;
        // a, b, c are three DISTINCT rows of the Hessenberg matrix (rows k, k+1, k+2), so the
        // [NoAlias] is truthful and Burst can SIMD the (otherwise can't-prove-non-aliasing) butterfly.
        // p is read from the old values before any write, so the three stores are independent per i.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void francisRow3([NoAlias] float* a, [NoAlias] float* b, [NoAlias] float* c,
                                       float q, float r, float xx, float yy, float zz, int n) {

            for (int i = 0; i < n; i++)
            {
                float p = a[i] + q * b[i] + r * c[i];
                c[i] -= p * zz;
                b[i] -= p * yy;
                a[i] -= p * xx;
            }
        }

        // Francis double-shift QR ROW update (2-row form, used when k == nn-1, no third row):
        //   p = a[i] + q*b[i];   b[i] -= p*yy;   a[i] -= p*xx;
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void francisRow2([NoAlias] float* a, [NoAlias] float* b,
                                       float q, float xx, float yy, int n) {

            for (int i = 0; i < n; i++)
            {
                float p = a[i] + q * b[i];
                b[i] -= p * yy;
                a[i] -= p * xx;
            }
        }

        // Fused 2x2 Gram dots of a vector pair: aa = a·a, bb = b·b, ab = a·b over [0,n).
        // 4-way unrolled with INDEPENDENT partial accumulators so the three reductions are not
        // latency-bound on a single loop-carried chain (recovers FMA-pipeline ILP, and lets Burst
        // pack lanes). a,b must be distinct non-overlapping ranges. NOTE: the 4-way partial-sum order
        // differs from a sequential sum, so results are rounding-level (not bitwise) different from a
        // naive accumulate — fine for the one-sided Jacobi SVD Gram step. Used by svdDecomposition.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void gram2x2([NoAlias] float* a, [NoAlias] float* b,
                                   out float aa, out float bb, out float ab, int n) {

            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            float b0 = 0, b1 = 0, b2 = 0, b3 = 0;
            float g0 = 0, g1 = 0, g2 = 0, g3 = 0;

            int i = 0;
            int n4 = n & ~3;
            for (; i < n4; i += 4)
            {
                float p0 = a[i],     q0 = b[i];
                float p1 = a[i + 1], q1 = b[i + 1];
                float p2 = a[i + 2], q2 = b[i + 2];
                float p3 = a[i + 3], q3 = b[i + 3];
                a0 += p0 * p0; b0 += q0 * q0; g0 += p0 * q0;
                a1 += p1 * p1; b1 += q1 * q1; g1 += p1 * q1;
                a2 += p2 * p2; b2 += q2 * q2; g2 += p2 * q2;
                a3 += p3 * p3; b3 += q3 * q3; g3 += p3 * q3;
            }

            float aSum = (a0 + a1) + (a2 + a3);
            float bSum = (b0 + b1) + (b2 + b3);
            float gSum = (g0 + g1) + (g2 + g3);

            for (; i < n; i++)
            {
                float p = a[i], q = b[i];
                aSum += p * p; bSum += q * q; gSum += p * q;
            }

            aa = aSum; bb = bSum; ab = gSum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compSub([NoAlias] float* target, [NoAlias] float* from, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] -= from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalAdd([NoAlias] float* target, int n, float s) {

            for (int i = 0; i < n; i++)
                target[i] += s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalSub(float s, [NoAlias] float* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = s - target[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compMul([NoAlias] float* from, [NoAlias] float* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] *= from[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compDiv([NoAlias] float* targetDividend, [NoAlias] float* fromDivisor, int n)
        {            
            for (int i = 0; i < n; i++)
                targetDividend[i] /= fromDivisor[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void compMod([NoAlias] float* targetDividend, [NoAlias] float* fromDivisor, int n)
        {
            for (int i = 0; i < n; i++)
                targetDividend[i] %= fromDivisor[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMul([NoAlias] float* target, int n, float s)
        {
            for (int i = 0; i < n; i++)
                target[i] *= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalDiv([NoAlias] float* target, int n, float s)
        {
            for (int i = 0; i < n; i++)
                target[i] /= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalDiv(float s, [NoAlias] float* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = s / target[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMod([NoAlias] float* target, int n, float s)
        {
            for (int i = 0; i < n; i++)
                target[i] %= s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void scalMod(float s, [NoAlias] float* target, int n)
        {
            for (int i = 0; i < n; i++)
                target[i] = s % target[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void normalizeL2Inpl([NoAlias] float* target, int n)
        {
            float sum = 0f;

            for (int i = 0; i < n; i++)
                sum += target[i] * target[i];
            
            sum = math.sqrt(sum);

            for (int i = 0; i < n; i++)
                target[i] /= sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static float normalizeL2Inpl([NoAlias] float* target, int start, int end)
        {
            float sum = 0f;

            for (int i = start; i < end; i++)
                sum += target[i] * target[i];

            sum = math.sqrt(sum);

            for (int i = start; i < end; i++)
                target[i] = (target[i] / sum);

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static float normalizeL1([NoAlias] float* target, int n)
        {
            float sum = 0f;

            for (int i = 0; i < n; i++)
                sum += math.abs(target[i]);

            for (int i = 0; i < n; i++)
                target[i] /= sum;

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static float normalizeL1([NoAlias] float* target, int start, int end)
        {
            float sum = 0f;

            for (int i = start; i < end; i++)
                sum += math.abs(target[i]);

            for (int i = start; i < end; i++)
                target[i] = (target[i] / sum);

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static float normalizeLMax([NoAlias] float* target, int n)
        {
            float max = 0f;

            for (int i = 0; i < n; i++)
                max = math.max(max, math.abs(target[i]));

            for (int i = 0; i < n; i++)
                target[i] /= max;

            return max;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static float normalizeLMax([NoAlias] float* target, int start, int end)
        {
            float max = 0f;

            for (int i = start; i < end; i++)
                max = math.max(max, math.abs(target[i]));

            for (int i = start; i < end; i++)
                target[i] /= max;

            return max;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static float normalizeLP([NoAlias] float* target, int n, float p)
        {
            float sum = 0f;

            for (int i = 0; i < n; i++)
                sum += math.pow(math.abs(target[i]), p);   // Lp norm uses |x_i|^p

            sum = math.pow(sum, 1f / p);

            for (int i = 0; i < n; i++)
                target[i] /= sum;

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static float normalizeLP([NoAlias] float* target, int start, int end, float p)
        {
            float sum = 0f;

            for (int i = start; i < end; i++)
                sum += math.pow(math.abs(target[i]), p);   // Lp norm uses |x_i|^p

            sum = math.pow(sum, 1f / p);

            for (int i = start; i < end; i++)
                target[i] = (target[i] / sum);

            return sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void swap([NoAlias] float* target, int startA, int startB, int n) {
            
            for (int i = 0; i < n; i++) {
                float temp = target[startA + i];
                target[startA + i] = target[startB + i];
                target[startB + i] = temp;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void swap([NoAlias] float* target, int startA, int startB, int n, int stride) {

            for (int i = 0; i < n; i++) {
                float temp = target[startA + i * stride];
                target[startA + i * stride] = target[startB + i * stride];
                target[startB + i * stride] = temp;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        [BurstCompile]
        // Swap rows in a matrix 
        public static void swapRows([NoAlias] float* target, int rowA, int rowB, int nCols, int colStart = 0, int colEnd = -1) {
            
            int rowIndexA = rowA * nCols;
            int rowIndexB = rowB * nCols; 

            if(colEnd == -1)
                colEnd = nCols;

            for (int i = colStart; i < colEnd; i++) {
                float temp = target[rowIndexA + i];
                target[rowIndexA + i] = target[rowIndexB + i];
                target[rowIndexB + i] = temp;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        [BurstCompile]
        // Swap columns in a matrix
        public static void swapColumns([NoAlias] float* target, int colA, int colB, int nRows, int nCols, int start = 0, int end = -1) {
            int startA = colA;
            int startB = colB;

            if(end == -1)
                end = nRows;

            for (int i = start; i < end; i++) {
                float temp = target[startA + i * nCols];
                target[startA + i * nCols] = target[startB + i * nCols];
                target[startB + i * nCols] = temp;
            }
        }
    }
}