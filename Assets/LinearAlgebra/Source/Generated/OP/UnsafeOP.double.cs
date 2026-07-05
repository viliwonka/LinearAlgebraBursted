#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Mathematics;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Collections;

namespace LinearAlgebra.Internal
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

        // Register-tiled GEMM: an MR x NR block of C is held in NAMED SCALAR LOCALS (NOT an array —
        // arrays don't reliably register-promote) across the WHOLE k-reduction (p = 0..n-1), so each
        // A value is reused NR times and each B value MR times, and each C element is written ONCE at
        // the end instead of being read-modified-written every p (the untiled fallback below re-touches
        // a whole C row per p, AND re-streams the whole of matB once per output row — that double
        // re-streaming, not the FLOP count, is why the untiled kernel is bandwidth- not compute-bound).
        //
        // Determinism (see the matMatDot spec / docs/level3-blocking-guide.md): every C[i,j] is STILL
        // exactly one running accumulator summing p ascending 0..n-1 with the SAME `c += a*b`
        // expression as the fallback. Tiling only changes WHICH independent accumulators run
        // interleaved (ILP across the MR*NR chains) — never how any ONE accumulator sums (no
        // k-splitting) — so results are bit-identical to the fallback at every tile size and on every
        // SIMD width (SIMD, if any, runs across the NR/column axis, never across the p-reduction).
        //
        // Tile constants are METHOD-LOCAL (a class-level const collides across the float/double
        // partial-class generated files -> CS0102). MR/NR are the SAME for float and double: the
        // //+choose codegen marker only substitutes literal VALUES, not the number of unrolled named
        // locals, and this template's text is emitted verbatim for both the float and double outputs
        // — so one tile shape has to serve both.
        //
        // Tile-size sweep (float/double, square N, GemmBenchmark): 4x4 -> 4x8 -> 8x8 -> 6x16 -> 8x16.
        // There is NO cache-level (k-panel) blocking here — the hard determinism rule above forbids
        // splitting one element's k-reduction into partial sums, which is exactly what k-panel
        // blocking would do — so B re-streaming is controlled purely by MR (B is re-read m/MR times).
        // 4x8 won in-cache (N<=256) but REGRESSED 2.2x vs the untiled fallback at N=1024 (69 -> 31
        // GFLOP/s, float) once that re-streamed strip of B stopped fitting in cache. A bigger MR is
        // the only lever without adding k-panel blocking: 8x16 (MR=8, NR=16 => 8*ceil(16/8) = 16
        // AVX2 accumulator vectors — the upper edge before spilling; do not go to 16x16, that's 32) won
        // at every measured size up to N=2048 for both types (float 1024: 86 vs baseline's 69 GFLOP/s;
        // float 2048: 71 vs baseline's 54; double tracks the same shape) and is what's left in place —
        // no size gate needed. See docs/level3-blocking-guide.md for the general blocking background.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matMatDot([NoAlias] double* matA, [NoAlias] double* matB, [NoAlias] double* matC, int m, int n, int k)
        {
            // matA = m x n
            // matB = n x k
            // matC = outMat = m x k, needs to be initialized to zero (this kernel ACCUMULATES, +=)

            const int MR = 8;
            const int NR = 16;

            int mTiles = (m / MR) * MR;
            int kTiles = (k / NR) * NR;

            for (int i = 0; i < mTiles; i += MR)
            {
                double* Arow0 = matA + (long)(i + 0) * n;
                double* Arow1 = matA + (long)(i + 1) * n;
                double* Arow2 = matA + (long)(i + 2) * n;
                double* Arow3 = matA + (long)(i + 3) * n;
                double* Arow4 = matA + (long)(i + 4) * n;
                double* Arow5 = matA + (long)(i + 5) * n;
                double* Arow6 = matA + (long)(i + 6) * n;
                double* Arow7 = matA + (long)(i + 7) * n;

                for (int j = 0; j < kTiles; j += NR)
                {
                    double c000 = 0f, c001 = 0f, c002 = 0f, c003 = 0f, c004 = 0f, c005 = 0f, c006 = 0f, c007 = 0f, c008 = 0f, c009 = 0f, c010 = 0f, c011 = 0f, c012 = 0f, c013 = 0f, c014 = 0f, c015 = 0f;
                    double c100 = 0f, c101 = 0f, c102 = 0f, c103 = 0f, c104 = 0f, c105 = 0f, c106 = 0f, c107 = 0f, c108 = 0f, c109 = 0f, c110 = 0f, c111 = 0f, c112 = 0f, c113 = 0f, c114 = 0f, c115 = 0f;
                    double c200 = 0f, c201 = 0f, c202 = 0f, c203 = 0f, c204 = 0f, c205 = 0f, c206 = 0f, c207 = 0f, c208 = 0f, c209 = 0f, c210 = 0f, c211 = 0f, c212 = 0f, c213 = 0f, c214 = 0f, c215 = 0f;
                    double c300 = 0f, c301 = 0f, c302 = 0f, c303 = 0f, c304 = 0f, c305 = 0f, c306 = 0f, c307 = 0f, c308 = 0f, c309 = 0f, c310 = 0f, c311 = 0f, c312 = 0f, c313 = 0f, c314 = 0f, c315 = 0f;
                    double c400 = 0f, c401 = 0f, c402 = 0f, c403 = 0f, c404 = 0f, c405 = 0f, c406 = 0f, c407 = 0f, c408 = 0f, c409 = 0f, c410 = 0f, c411 = 0f, c412 = 0f, c413 = 0f, c414 = 0f, c415 = 0f;
                    double c500 = 0f, c501 = 0f, c502 = 0f, c503 = 0f, c504 = 0f, c505 = 0f, c506 = 0f, c507 = 0f, c508 = 0f, c509 = 0f, c510 = 0f, c511 = 0f, c512 = 0f, c513 = 0f, c514 = 0f, c515 = 0f;
                    double c600 = 0f, c601 = 0f, c602 = 0f, c603 = 0f, c604 = 0f, c605 = 0f, c606 = 0f, c607 = 0f, c608 = 0f, c609 = 0f, c610 = 0f, c611 = 0f, c612 = 0f, c613 = 0f, c614 = 0f, c615 = 0f;
                    double c700 = 0f, c701 = 0f, c702 = 0f, c703 = 0f, c704 = 0f, c705 = 0f, c706 = 0f, c707 = 0f, c708 = 0f, c709 = 0f, c710 = 0f, c711 = 0f, c712 = 0f, c713 = 0f, c714 = 0f, c715 = 0f;

                    for (int p = 0; p < n; p++)
                    {
                        double a0 = Arow0[p];
                        double a1 = Arow1[p];
                        double a2 = Arow2[p];
                        double a3 = Arow3[p];
                        double a4 = Arow4[p];
                        double a5 = Arow5[p];
                        double a6 = Arow6[p];
                        double a7 = Arow7[p];

                        double* Brow = matB + (long)p * k + j;
                        double b0 = Brow[0];
                        double b1 = Brow[1];
                        double b2 = Brow[2];
                        double b3 = Brow[3];
                        double b4 = Brow[4];
                        double b5 = Brow[5];
                        double b6 = Brow[6];
                        double b7 = Brow[7];
                        double b8 = Brow[8];
                        double b9 = Brow[9];
                        double b10 = Brow[10];
                        double b11 = Brow[11];
                        double b12 = Brow[12];
                        double b13 = Brow[13];
                        double b14 = Brow[14];
                        double b15 = Brow[15];

                        c000 += a0 * b0; c001 += a0 * b1; c002 += a0 * b2; c003 += a0 * b3; c004 += a0 * b4; c005 += a0 * b5; c006 += a0 * b6; c007 += a0 * b7; c008 += a0 * b8; c009 += a0 * b9; c010 += a0 * b10; c011 += a0 * b11; c012 += a0 * b12; c013 += a0 * b13; c014 += a0 * b14; c015 += a0 * b15;
                        c100 += a1 * b0; c101 += a1 * b1; c102 += a1 * b2; c103 += a1 * b3; c104 += a1 * b4; c105 += a1 * b5; c106 += a1 * b6; c107 += a1 * b7; c108 += a1 * b8; c109 += a1 * b9; c110 += a1 * b10; c111 += a1 * b11; c112 += a1 * b12; c113 += a1 * b13; c114 += a1 * b14; c115 += a1 * b15;
                        c200 += a2 * b0; c201 += a2 * b1; c202 += a2 * b2; c203 += a2 * b3; c204 += a2 * b4; c205 += a2 * b5; c206 += a2 * b6; c207 += a2 * b7; c208 += a2 * b8; c209 += a2 * b9; c210 += a2 * b10; c211 += a2 * b11; c212 += a2 * b12; c213 += a2 * b13; c214 += a2 * b14; c215 += a2 * b15;
                        c300 += a3 * b0; c301 += a3 * b1; c302 += a3 * b2; c303 += a3 * b3; c304 += a3 * b4; c305 += a3 * b5; c306 += a3 * b6; c307 += a3 * b7; c308 += a3 * b8; c309 += a3 * b9; c310 += a3 * b10; c311 += a3 * b11; c312 += a3 * b12; c313 += a3 * b13; c314 += a3 * b14; c315 += a3 * b15;
                        c400 += a4 * b0; c401 += a4 * b1; c402 += a4 * b2; c403 += a4 * b3; c404 += a4 * b4; c405 += a4 * b5; c406 += a4 * b6; c407 += a4 * b7; c408 += a4 * b8; c409 += a4 * b9; c410 += a4 * b10; c411 += a4 * b11; c412 += a4 * b12; c413 += a4 * b13; c414 += a4 * b14; c415 += a4 * b15;
                        c500 += a5 * b0; c501 += a5 * b1; c502 += a5 * b2; c503 += a5 * b3; c504 += a5 * b4; c505 += a5 * b5; c506 += a5 * b6; c507 += a5 * b7; c508 += a5 * b8; c509 += a5 * b9; c510 += a5 * b10; c511 += a5 * b11; c512 += a5 * b12; c513 += a5 * b13; c514 += a5 * b14; c515 += a5 * b15;
                        c600 += a6 * b0; c601 += a6 * b1; c602 += a6 * b2; c603 += a6 * b3; c604 += a6 * b4; c605 += a6 * b5; c606 += a6 * b6; c607 += a6 * b7; c608 += a6 * b8; c609 += a6 * b9; c610 += a6 * b10; c611 += a6 * b11; c612 += a6 * b12; c613 += a6 * b13; c614 += a6 * b14; c615 += a6 * b15;
                        c700 += a7 * b0; c701 += a7 * b1; c702 += a7 * b2; c703 += a7 * b3; c704 += a7 * b4; c705 += a7 * b5; c706 += a7 * b6; c707 += a7 * b7; c708 += a7 * b8; c709 += a7 * b9; c710 += a7 * b10; c711 += a7 * b11; c712 += a7 * b12; c713 += a7 * b13; c714 += a7 * b14; c715 += a7 * b15;
                    }

                    double* Crow0 = matC + (long)(i + 0) * k + j;
                    double* Crow1 = matC + (long)(i + 1) * k + j;
                    double* Crow2 = matC + (long)(i + 2) * k + j;
                    double* Crow3 = matC + (long)(i + 3) * k + j;
                    double* Crow4 = matC + (long)(i + 4) * k + j;
                    double* Crow5 = matC + (long)(i + 5) * k + j;
                    double* Crow6 = matC + (long)(i + 6) * k + j;
                    double* Crow7 = matC + (long)(i + 7) * k + j;

                    Crow0[0] += c000; Crow0[1] += c001; Crow0[2] += c002; Crow0[3] += c003; Crow0[4] += c004; Crow0[5] += c005; Crow0[6] += c006; Crow0[7] += c007; Crow0[8] += c008; Crow0[9] += c009; Crow0[10] += c010; Crow0[11] += c011; Crow0[12] += c012; Crow0[13] += c013; Crow0[14] += c014; Crow0[15] += c015;
                    Crow1[0] += c100; Crow1[1] += c101; Crow1[2] += c102; Crow1[3] += c103; Crow1[4] += c104; Crow1[5] += c105; Crow1[6] += c106; Crow1[7] += c107; Crow1[8] += c108; Crow1[9] += c109; Crow1[10] += c110; Crow1[11] += c111; Crow1[12] += c112; Crow1[13] += c113; Crow1[14] += c114; Crow1[15] += c115;
                    Crow2[0] += c200; Crow2[1] += c201; Crow2[2] += c202; Crow2[3] += c203; Crow2[4] += c204; Crow2[5] += c205; Crow2[6] += c206; Crow2[7] += c207; Crow2[8] += c208; Crow2[9] += c209; Crow2[10] += c210; Crow2[11] += c211; Crow2[12] += c212; Crow2[13] += c213; Crow2[14] += c214; Crow2[15] += c215;
                    Crow3[0] += c300; Crow3[1] += c301; Crow3[2] += c302; Crow3[3] += c303; Crow3[4] += c304; Crow3[5] += c305; Crow3[6] += c306; Crow3[7] += c307; Crow3[8] += c308; Crow3[9] += c309; Crow3[10] += c310; Crow3[11] += c311; Crow3[12] += c312; Crow3[13] += c313; Crow3[14] += c314; Crow3[15] += c315;
                    Crow4[0] += c400; Crow4[1] += c401; Crow4[2] += c402; Crow4[3] += c403; Crow4[4] += c404; Crow4[5] += c405; Crow4[6] += c406; Crow4[7] += c407; Crow4[8] += c408; Crow4[9] += c409; Crow4[10] += c410; Crow4[11] += c411; Crow4[12] += c412; Crow4[13] += c413; Crow4[14] += c414; Crow4[15] += c415;
                    Crow5[0] += c500; Crow5[1] += c501; Crow5[2] += c502; Crow5[3] += c503; Crow5[4] += c504; Crow5[5] += c505; Crow5[6] += c506; Crow5[7] += c507; Crow5[8] += c508; Crow5[9] += c509; Crow5[10] += c510; Crow5[11] += c511; Crow5[12] += c512; Crow5[13] += c513; Crow5[14] += c514; Crow5[15] += c515;
                    Crow6[0] += c600; Crow6[1] += c601; Crow6[2] += c602; Crow6[3] += c603; Crow6[4] += c604; Crow6[5] += c605; Crow6[6] += c606; Crow6[7] += c607; Crow6[8] += c608; Crow6[9] += c609; Crow6[10] += c610; Crow6[11] += c611; Crow6[12] += c612; Crow6[13] += c613; Crow6[14] += c614; Crow6[15] += c615;
                    Crow7[0] += c700; Crow7[1] += c701; Crow7[2] += c702; Crow7[3] += c703; Crow7[4] += c704; Crow7[5] += c705; Crow7[6] += c706; Crow7[7] += c707; Crow7[8] += c708; Crow7[9] += c709; Crow7[10] += c710; Crow7[11] += c711; Crow7[12] += c712; Crow7[13] += c713; Crow7[14] += c714; Crow7[15] += c715;
                }

                // Remainder columns [kTiles, k) for these MR rows: same p-ascending order, plain fallback.
                if (kTiles < k)
                    matMatDotRange(matA, matB, matC, i, i + MR, n, k, kTiles, k);
            }

            // Remainder rows [mTiles, m) — and, when m < MR, the WHOLE matrix (mTiles == 0 routes
            // every row through here): plain fallback, zero seam risk vs the tiled bulk above.
            if (mTiles < m)
                matMatDotRange(matA, matB, matC, mTiles, m, n, k, 0, k);
        }

        // Plain (untiled) GEMM restricted to an explicit row/column sub-range, in the SAME
        // p-ascending accumulation order as matMatDot's tiled bulk above (this literally IS the
        // pre-tiling kernel, just row/col-bounded) — used both for the remainder rows/cols the
        // MR x NR tiling doesn't evenly cover, and as the whole-matrix path for matrices smaller than
        // one tile (mTiles==0 or kTiles==0 routes every row/col through here), so there is zero risk
        // of a seam between the tiled and fallback regions, and small matrices provably do not regress
        // (they take exactly the pre-existing kernel).
        [MethodImpl(MethodImplOptions.NoInlining)]
        static void matMatDotRange([NoAlias] double* matA, [NoAlias] double* matB, [NoAlias] double* matC,
                                    int rowStart, int rowEnd, int n, int k, int colStart, int colEnd)
        {
            for (int r = rowStart; r < rowEnd; r++)
            {
                for (int nCols = 0; nCols < n; nCols++)
                {
                    double temp = matA[r * n + nCols];
                    for (int kCols = colStart; kCols < colEnd; kCols++)
                    {
                        matC[r * k + kCols] += temp * matB[nCols * k + kCols];
                    }
                }
            }
        }

        // Same register-tile treatment as matMatDot, applied to the transposed-A read (Aᵀ·B). Only
        // the A access pattern differs: A[p, i+t] rather than A[i+t, p], because A is stored m x n but
        // read as n x m here. The MR row-values for a fixed p are CONTIGUOUS
        // (matA[p*m + i .. i+MR-1]) — an even better load pattern than matMatDot's per-row strided
        // reads. Same determinism argument and same MR/NR-shared-across-types constraint as above.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matMatDotTransA([NoAlias] double* matA, [NoAlias] double* matB, [NoAlias] double* matC, int m, int n, int k)
        {
            // matA = m x n, but treated as n x m due to transposition
            // matB = n x k
            // matC = outMat = m x k, needs to be initialized to zero (this kernel ACCUMULATES, +=)

            const int MR = 8;
            const int NR = 16;

            int mTiles = (m / MR) * MR;
            int kTiles = (k / NR) * NR;

            for (int i = 0; i < mTiles; i += MR)
            {
                for (int j = 0; j < kTiles; j += NR)
                {
                    double c000 = 0f, c001 = 0f, c002 = 0f, c003 = 0f, c004 = 0f, c005 = 0f, c006 = 0f, c007 = 0f, c008 = 0f, c009 = 0f, c010 = 0f, c011 = 0f, c012 = 0f, c013 = 0f, c014 = 0f, c015 = 0f;
                    double c100 = 0f, c101 = 0f, c102 = 0f, c103 = 0f, c104 = 0f, c105 = 0f, c106 = 0f, c107 = 0f, c108 = 0f, c109 = 0f, c110 = 0f, c111 = 0f, c112 = 0f, c113 = 0f, c114 = 0f, c115 = 0f;
                    double c200 = 0f, c201 = 0f, c202 = 0f, c203 = 0f, c204 = 0f, c205 = 0f, c206 = 0f, c207 = 0f, c208 = 0f, c209 = 0f, c210 = 0f, c211 = 0f, c212 = 0f, c213 = 0f, c214 = 0f, c215 = 0f;
                    double c300 = 0f, c301 = 0f, c302 = 0f, c303 = 0f, c304 = 0f, c305 = 0f, c306 = 0f, c307 = 0f, c308 = 0f, c309 = 0f, c310 = 0f, c311 = 0f, c312 = 0f, c313 = 0f, c314 = 0f, c315 = 0f;
                    double c400 = 0f, c401 = 0f, c402 = 0f, c403 = 0f, c404 = 0f, c405 = 0f, c406 = 0f, c407 = 0f, c408 = 0f, c409 = 0f, c410 = 0f, c411 = 0f, c412 = 0f, c413 = 0f, c414 = 0f, c415 = 0f;
                    double c500 = 0f, c501 = 0f, c502 = 0f, c503 = 0f, c504 = 0f, c505 = 0f, c506 = 0f, c507 = 0f, c508 = 0f, c509 = 0f, c510 = 0f, c511 = 0f, c512 = 0f, c513 = 0f, c514 = 0f, c515 = 0f;
                    double c600 = 0f, c601 = 0f, c602 = 0f, c603 = 0f, c604 = 0f, c605 = 0f, c606 = 0f, c607 = 0f, c608 = 0f, c609 = 0f, c610 = 0f, c611 = 0f, c612 = 0f, c613 = 0f, c614 = 0f, c615 = 0f;
                    double c700 = 0f, c701 = 0f, c702 = 0f, c703 = 0f, c704 = 0f, c705 = 0f, c706 = 0f, c707 = 0f, c708 = 0f, c709 = 0f, c710 = 0f, c711 = 0f, c712 = 0f, c713 = 0f, c714 = 0f, c715 = 0f;

                    for (int p = 0; p < n; p++)
                    {
                        double* Ap = matA + (long)p * m + i;
                        double a0 = Ap[0];
                        double a1 = Ap[1];
                        double a2 = Ap[2];
                        double a3 = Ap[3];
                        double a4 = Ap[4];
                        double a5 = Ap[5];
                        double a6 = Ap[6];
                        double a7 = Ap[7];

                        double* Brow = matB + (long)p * k + j;
                        double b0 = Brow[0];
                        double b1 = Brow[1];
                        double b2 = Brow[2];
                        double b3 = Brow[3];
                        double b4 = Brow[4];
                        double b5 = Brow[5];
                        double b6 = Brow[6];
                        double b7 = Brow[7];
                        double b8 = Brow[8];
                        double b9 = Brow[9];
                        double b10 = Brow[10];
                        double b11 = Brow[11];
                        double b12 = Brow[12];
                        double b13 = Brow[13];
                        double b14 = Brow[14];
                        double b15 = Brow[15];

                        c000 += a0 * b0; c001 += a0 * b1; c002 += a0 * b2; c003 += a0 * b3; c004 += a0 * b4; c005 += a0 * b5; c006 += a0 * b6; c007 += a0 * b7; c008 += a0 * b8; c009 += a0 * b9; c010 += a0 * b10; c011 += a0 * b11; c012 += a0 * b12; c013 += a0 * b13; c014 += a0 * b14; c015 += a0 * b15;
                        c100 += a1 * b0; c101 += a1 * b1; c102 += a1 * b2; c103 += a1 * b3; c104 += a1 * b4; c105 += a1 * b5; c106 += a1 * b6; c107 += a1 * b7; c108 += a1 * b8; c109 += a1 * b9; c110 += a1 * b10; c111 += a1 * b11; c112 += a1 * b12; c113 += a1 * b13; c114 += a1 * b14; c115 += a1 * b15;
                        c200 += a2 * b0; c201 += a2 * b1; c202 += a2 * b2; c203 += a2 * b3; c204 += a2 * b4; c205 += a2 * b5; c206 += a2 * b6; c207 += a2 * b7; c208 += a2 * b8; c209 += a2 * b9; c210 += a2 * b10; c211 += a2 * b11; c212 += a2 * b12; c213 += a2 * b13; c214 += a2 * b14; c215 += a2 * b15;
                        c300 += a3 * b0; c301 += a3 * b1; c302 += a3 * b2; c303 += a3 * b3; c304 += a3 * b4; c305 += a3 * b5; c306 += a3 * b6; c307 += a3 * b7; c308 += a3 * b8; c309 += a3 * b9; c310 += a3 * b10; c311 += a3 * b11; c312 += a3 * b12; c313 += a3 * b13; c314 += a3 * b14; c315 += a3 * b15;
                        c400 += a4 * b0; c401 += a4 * b1; c402 += a4 * b2; c403 += a4 * b3; c404 += a4 * b4; c405 += a4 * b5; c406 += a4 * b6; c407 += a4 * b7; c408 += a4 * b8; c409 += a4 * b9; c410 += a4 * b10; c411 += a4 * b11; c412 += a4 * b12; c413 += a4 * b13; c414 += a4 * b14; c415 += a4 * b15;
                        c500 += a5 * b0; c501 += a5 * b1; c502 += a5 * b2; c503 += a5 * b3; c504 += a5 * b4; c505 += a5 * b5; c506 += a5 * b6; c507 += a5 * b7; c508 += a5 * b8; c509 += a5 * b9; c510 += a5 * b10; c511 += a5 * b11; c512 += a5 * b12; c513 += a5 * b13; c514 += a5 * b14; c515 += a5 * b15;
                        c600 += a6 * b0; c601 += a6 * b1; c602 += a6 * b2; c603 += a6 * b3; c604 += a6 * b4; c605 += a6 * b5; c606 += a6 * b6; c607 += a6 * b7; c608 += a6 * b8; c609 += a6 * b9; c610 += a6 * b10; c611 += a6 * b11; c612 += a6 * b12; c613 += a6 * b13; c614 += a6 * b14; c615 += a6 * b15;
                        c700 += a7 * b0; c701 += a7 * b1; c702 += a7 * b2; c703 += a7 * b3; c704 += a7 * b4; c705 += a7 * b5; c706 += a7 * b6; c707 += a7 * b7; c708 += a7 * b8; c709 += a7 * b9; c710 += a7 * b10; c711 += a7 * b11; c712 += a7 * b12; c713 += a7 * b13; c714 += a7 * b14; c715 += a7 * b15;
                    }

                    double* Crow0 = matC + (long)(i + 0) * k + j;
                    double* Crow1 = matC + (long)(i + 1) * k + j;
                    double* Crow2 = matC + (long)(i + 2) * k + j;
                    double* Crow3 = matC + (long)(i + 3) * k + j;
                    double* Crow4 = matC + (long)(i + 4) * k + j;
                    double* Crow5 = matC + (long)(i + 5) * k + j;
                    double* Crow6 = matC + (long)(i + 6) * k + j;
                    double* Crow7 = matC + (long)(i + 7) * k + j;

                    Crow0[0] += c000; Crow0[1] += c001; Crow0[2] += c002; Crow0[3] += c003; Crow0[4] += c004; Crow0[5] += c005; Crow0[6] += c006; Crow0[7] += c007; Crow0[8] += c008; Crow0[9] += c009; Crow0[10] += c010; Crow0[11] += c011; Crow0[12] += c012; Crow0[13] += c013; Crow0[14] += c014; Crow0[15] += c015;
                    Crow1[0] += c100; Crow1[1] += c101; Crow1[2] += c102; Crow1[3] += c103; Crow1[4] += c104; Crow1[5] += c105; Crow1[6] += c106; Crow1[7] += c107; Crow1[8] += c108; Crow1[9] += c109; Crow1[10] += c110; Crow1[11] += c111; Crow1[12] += c112; Crow1[13] += c113; Crow1[14] += c114; Crow1[15] += c115;
                    Crow2[0] += c200; Crow2[1] += c201; Crow2[2] += c202; Crow2[3] += c203; Crow2[4] += c204; Crow2[5] += c205; Crow2[6] += c206; Crow2[7] += c207; Crow2[8] += c208; Crow2[9] += c209; Crow2[10] += c210; Crow2[11] += c211; Crow2[12] += c212; Crow2[13] += c213; Crow2[14] += c214; Crow2[15] += c215;
                    Crow3[0] += c300; Crow3[1] += c301; Crow3[2] += c302; Crow3[3] += c303; Crow3[4] += c304; Crow3[5] += c305; Crow3[6] += c306; Crow3[7] += c307; Crow3[8] += c308; Crow3[9] += c309; Crow3[10] += c310; Crow3[11] += c311; Crow3[12] += c312; Crow3[13] += c313; Crow3[14] += c314; Crow3[15] += c315;
                    Crow4[0] += c400; Crow4[1] += c401; Crow4[2] += c402; Crow4[3] += c403; Crow4[4] += c404; Crow4[5] += c405; Crow4[6] += c406; Crow4[7] += c407; Crow4[8] += c408; Crow4[9] += c409; Crow4[10] += c410; Crow4[11] += c411; Crow4[12] += c412; Crow4[13] += c413; Crow4[14] += c414; Crow4[15] += c415;
                    Crow5[0] += c500; Crow5[1] += c501; Crow5[2] += c502; Crow5[3] += c503; Crow5[4] += c504; Crow5[5] += c505; Crow5[6] += c506; Crow5[7] += c507; Crow5[8] += c508; Crow5[9] += c509; Crow5[10] += c510; Crow5[11] += c511; Crow5[12] += c512; Crow5[13] += c513; Crow5[14] += c514; Crow5[15] += c515;
                    Crow6[0] += c600; Crow6[1] += c601; Crow6[2] += c602; Crow6[3] += c603; Crow6[4] += c604; Crow6[5] += c605; Crow6[6] += c606; Crow6[7] += c607; Crow6[8] += c608; Crow6[9] += c609; Crow6[10] += c610; Crow6[11] += c611; Crow6[12] += c612; Crow6[13] += c613; Crow6[14] += c614; Crow6[15] += c615;
                    Crow7[0] += c700; Crow7[1] += c701; Crow7[2] += c702; Crow7[3] += c703; Crow7[4] += c704; Crow7[5] += c705; Crow7[6] += c706; Crow7[7] += c707; Crow7[8] += c708; Crow7[9] += c709; Crow7[10] += c710; Crow7[11] += c711; Crow7[12] += c712; Crow7[13] += c713; Crow7[14] += c714; Crow7[15] += c715;
                }

                // Remainder columns [kTiles, k) for these MR rows: same p-ascending order, plain fallback.
                if (kTiles < k)
                    matMatDotTransARange(matA, matB, matC, i, i + MR, m, n, k, kTiles, k);
            }

            // Remainder rows [mTiles, m) — and, when m < MR, the WHOLE matrix: plain fallback, zero
            // seam risk vs the tiled bulk above.
            if (mTiles < m)
                matMatDotTransARange(matA, matB, matC, mTiles, m, m, n, k, 0, k);
        }

        // Plain (untiled) Aᵀ·B restricted to an explicit row/column sub-range — the transposed-A
        // mirror of matMatDotRange, same rationale (remainder coverage + whole-matrix small-size
        // fallback with zero seam risk).
        [MethodImpl(MethodImplOptions.NoInlining)]
        static void matMatDotTransARange([NoAlias] double* matA, [NoAlias] double* matB, [NoAlias] double* matC,
                                          int rowStart, int rowEnd, int m, int n, int k, int colStart, int colEnd)
        {
            for (int r = rowStart; r < rowEnd; r++)
            {
                for (int nCols = 0; nCols < n; nCols++)
                {
                    double temp = matA[nCols * m + r];
                    for (int kCols = colStart; kCols < colEnd; kCols++)
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
        public static void trsmLowerPanel([NoAlias] double* L11, int Lld, [NoAlias] double* B, int Bld, int nrows, int jb)
        {
            for (int t = 0; t < nrows; t++)
            {
                double* Brow = B + (long)t * Bld;
                for (int p = 0; p < jb; p++)
                {
                    double s = Brow[p];
                    double* L11row = L11 + (long)p * Lld;
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
        public static void syrkLowerSub([NoAlias] double* Lp, int n, int rStart, int j0, int jb, [NoAlias] double* PT)
        {
            int ntrail = n - rStart;
            for (int ip = 0; ip < ntrail; ip++)          // i' ; i = rStart+ip
            {
                int i = rStart + ip;
                double* Lrow = Lp + (long)i * n;         // write cols [rStart, i]
                double* Pi = Lp + (long)i * n + j0;      // P[ip, 0..jb)
                for (int p = 0; p < jb; p++)
                {
                    double temp = Pi[p];                  // scalar, loaded before the k' write loop (no hazard)
                    double* PTp = PT + (long)p * ntrail;
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
        public static void wyVtC([NoAlias] double* Vp, int Vld, [NoAlias] double* Cp, int Cld, int rows, int pb, int cw, [NoAlias] double* W)
        {
            for (int t = 0; t < rows; t++)
            {
                double* Vrow = Vp + (long)t * Vld;
                double* Crow = Cp + (long)t * Cld;
                for (int i = 0; i < pb; i++)
                {
                    double temp = Vrow[i];
                    double* Wi = W + (long)i * cw;
                    for (int j = 0; j < cw; j++)
                        Wi[j] += temp * Crow[j];
                }
            }
        }

        // C[t,j] -= Σ_{i=0..pb-1} Vp[t*Vld+i] * W[i,j]   — the second GEMM half of the block
        // reflector apply, C -= V·W. Same unit-stride-j vectorisation shape as wyVtC.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void wySubVW([NoAlias] double* Vp, int Vld, [NoAlias] double* Cp, int Cld, int rows, int pb, int cw, [NoAlias] double* W)
        {
            for (int t = 0; t < rows; t++)
            {
                double* Vrow = Vp + (long)t * Vld;
                double* Crow = Cp + (long)t * Cld;
                for (int i = 0; i < pb; i++)
                {
                    double temp = Vrow[i];
                    double* Wi = W + (long)i * cw;
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
        public static void lqYeqCVt([NoAlias] double* Cp, int Cld, [NoAlias] double* Vt, int cn, int rows, int pb, [NoAlias] double* Y)
        {
            for (int t = 0; t < rows; t++)
            {
                double* Crow = Cp + (long)t * Cld;
                double* Yrow = Y + (long)t * pb;
                for (int c = 0; c < cn; c++)
                {
                    double temp = Crow[c];
                    double* Vtrow = Vt + (long)c * pb;
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
        public static void wyTriTransMul([NoAlias] double* T, int pb, [NoAlias] double* W, int cw)
        {
            for (int i = pb - 1; i >= 0; i--)
            {
                double* Wi = W + (long)i * cw;
                double tii = T[i * pb + i];
                for (int j = 0; j < cw; j++)
                    Wi[j] = tii * Wi[j];
                for (int k = 0; k < i; k++)
                {
                    double tki = T[k * pb + i];
                    double* Wk = W + (long)k * cw;
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
        public static void wyTriMul([NoAlias] double* T, int pb, [NoAlias] double* W, int cw)
        {
            for (int i = 0; i < pb; i++)
            {
                double* Wi = W + (long)i * cw;
                double tii = T[i * pb + i];
                for (int j = 0; j < cw; j++)
                    Wi[j] = tii * Wi[j];
                for (int k = i + 1; k < pb; k++)
                {
                    double tik = T[i * pb + k];
                    double* Wk = W + (long)k * cw;
                    for (int j = 0; j < cw; j++)
                        Wi[j] += tik * Wk[j];
                }
            }
        }

        // Compact-WY block-reflector T factor (LARFT). Builds the pb×pb upper-triangular T from the
        // panel V (pb reflectors) so a batch of Householder reflectors applies as one GEMM-shaped block
        // update  (I - V T Vᵀ). τ≡1 folded-reflector convention → T[i,i] = 1 (NOT LAPACK's τ-scaled
        // diagonal). Direction-agnostic (it only contracts a Gram matrix from a clean, masked panel),
        // so it is shared by QR (left-multiply) and LQ (right-multiply) — and reused by the blocked
        // symmetric/bidiagonal reductions.
        //
        //   Vp    panel base pointer, row-major, leading dimension Vld; reflector v_i in column i.
        //         v_i is masked to zero for local rows t < i (callers guarantee this), so a dot
        //         v_k·v_i (k < i) over the FULL row range [0,rows) restricts itself to t >= i.
        //   rows  panel row count (local t = 0..rows-1); v_i occupies Vp[t*Vld+i].
        //   pb    number of reflectors in the panel.
        //   T     pb×pb contiguous output, row-major (T[i,k] at T[i*pb+k]), upper-triangular.
        //   tcol  scratch, length >= pb.
        //   G     pb×pb scratch for the Gram matrix VᵀV.
        //
        // Two passes rather than pb²/2 direct dot products:
        //   1) G = VᵀV via a GEMM-shaped unit-stride loop (t outer, i middle, j INNER unit-stride) —
        //      reaches GEMM throughput. The naive per-(k,i) dot form (t as the reduction axis, stride
        //      Vld between consecutive t) does NOT vectorise and was measured far slower.
        //   2) The T recursion reads G's entries instead of recomputing dots — O(pb³/6), negligible.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void formT([NoAlias] double* Vp, int Vld, int rows, int pb, [NoAlias] double* T, [NoAlias] double* tcol, [NoAlias] double* G)
        {
            UnsafeUtility.MemClear(G, (long)pb * pb * UnsafeUtility.SizeOf<double>());
            for (int t = 0; t < rows; t++)
            {
                double* Vrow = Vp + (long)t * Vld;
                for (int i = 0; i < pb; i++)
                {
                    double temp = Vrow[i];
                    double* Gi = G + (long)i * pb;
                    for (int j = 0; j < pb; j++)
                        Gi[j] += temp * Vrow[j];
                }
            }

            for (int i = 0; i < pb; i++)
            {
                T[i * pb + i] = 1;
                if (i > 0)
                {
                    // tcol[k] = -G[k,i] = -(v_k · v_i), k in [0, i)
                    for (int k = 0; k < i; k++)
                        tcol[k] = -G[k * pb + i];
                    // T[k,i] = Σ_{l=k..i-1} T[k,l] * tcol[l], k in [0, i)  (T[0:i,0:i] · tcol)
                    for (int k = 0; k < i; k++)
                    {
                        double sum = 0;
                        for (int l = k; l < i; l++)
                            sum += T[k * pb + l] * tcol[l];
                        T[k * pb + i] = sum;
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

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void axpy([NoAlias] double* y, [NoAlias] double* x, double a, int n) {

            for (int i = 0; i < n; i++)
                y[i] += a * x[i];
        }

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

        // Francis double-shift QR ROW update (3-row form), over a contiguous column range [0,n):
        //   p = a[i] + q*b[i] + r*c[i];   c[i] -= p*zz;   b[i] -= p*yy;   a[i] -= p*xx;
        // a, b, c are three DISTINCT rows of the Hessenberg matrix (rows k, k+1, k+2), so the
        // [NoAlias] is truthful and Burst can SIMD the (otherwise can't-prove-non-aliasing) butterfly.
        // p is read from the old values before any write, so the three stores are independent per i.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void francisRow3([NoAlias] double* a, [NoAlias] double* b, [NoAlias] double* c,
                                       double q, double r, double xx, double yy, double zz, int n) {

            for (int i = 0; i < n; i++)
            {
                double p = a[i] + q * b[i] + r * c[i];
                c[i] -= p * zz;
                b[i] -= p * yy;
                a[i] -= p * xx;
            }
        }

        // Francis double-shift QR ROW update (2-row form, used when k == nn-1, no third row):
        //   p = a[i] + q*b[i];   b[i] -= p*yy;   a[i] -= p*xx;
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void francisRow2([NoAlias] double* a, [NoAlias] double* b,
                                       double q, double xx, double yy, int n) {

            for (int i = 0; i < n; i++)
            {
                double p = a[i] + q * b[i];
                b[i] -= p * yy;
                a[i] -= p * xx;
            }
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
        public static void normalizeL2InPlace([NoAlias] double* target, int n)
        {
            double sum = 0f;

            for (int i = 0; i < n; i++)
                sum += target[i] * target[i];
            
            sum = math.sqrt(sum);

            for (int i = 0; i < n; i++)
                target[i] /= sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static double normalizeL2InPlace([NoAlias] double* target, int start, int end)
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