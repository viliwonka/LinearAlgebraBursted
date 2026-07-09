#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Mathematics;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Collections;


namespace LinearAlgebra.Internal
{
    public static unsafe partial class UnsafeOP {

        // These Level-1 reductions use TWO width-4 SIMD accumulators (float4 -> float4/double4):
        // 2x4 = 8 fixed per-lane add-chains -- enough independent chains to keep the FP add ports busy
        // (one 4-lane accumulator left them ~half idle in-cache; the 2nd accumulator measured ~2x). SIMD
        // packing is not reassociation, so this is Strict-safe and bit-identical on SSE/AVX/NEON; the
        // summation TREE (which accumulator/lane sums which elements, then acc0+acc1, then the balanced
        // (x+y)+(z+w) fold) is fixed by the source == the FROZEN numeric contract -- do not reshuffle it.
        // See matVecDot for why Burst can't do this itself under FloatMode.Default (reduction
        // vectorization needs reassociation, which Strict forbids). Reinterpret loads are unaligned
        // (rows/vectors are element-aligned only) -- Burst emits unaligned loads; an intrinsic rewrite
        // must too. No FMA under Strict by design.

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static float sumAbs([NoAlias] float* a, int n)
        {
            var pa = (float4*)a;
            int nQ = n >> 2;
            float4 acc0 = default, acc1 = default;
            int q = 0;
            for (; q + 2 <= nQ; q += 2)
            {
                acc0 += floatM.abs(pa[q]);
                acc1 += floatM.abs(pa[q + 1]);
            }
            if (q < nQ) acc0 += floatM.abs(pa[q]);
            float4 acc = acc0 + acc1;
            float s = (acc.x + acc.y) + (acc.z + acc.w);
            for (int i = nQ << 2; i < n; i++)
                s += math.abs(a[i]);
            return s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static float sum([NoAlias] float* a, int n)
        {
            var pa = (float4*)a;
            int nQ = n >> 2;
            float4 acc0 = default, acc1 = default;
            int q = 0;
            for (; q + 2 <= nQ; q += 2)
            {
                acc0 += pa[q];
                acc1 += pa[q + 1];
            }
            if (q < nQ) acc0 += pa[q];
            float4 acc = acc0 + acc1;
            float s = (acc.x + acc.y) + (acc.z + acc.w);
            for (int i = nQ << 2; i < n; i++)
                s += a[i];
            return s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static float maxAbs([NoAlias] float* a, int n)
        {
            // max is exact (no rounding), so accumulator/lane order changes nothing but NaN propagation,
            // which is still identical on every machine. Accumulators seed at 0 == the old max=0 (abs>=0).
            var pa = (float4*)a;
            int nQ = n >> 2;
            float4 acc0 = default, acc1 = default;
            int q = 0;
            for (; q + 2 <= nQ; q += 2)
            {
                acc0 = floatM.max(acc0, floatM.abs(pa[q]));
                acc1 = floatM.max(acc1, floatM.abs(pa[q + 1]));
            }
            if (q < nQ) acc0 = floatM.max(acc0, floatM.abs(pa[q]));
            float4 acc = floatM.max(acc0, acc1);
            float m = math.max(math.max(acc.x, acc.y), math.max(acc.z, acc.w));
            for (int i = nQ << 2; i < n; i++)
                m = math.max(m, math.abs(a[i]));
            return m;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static float vecDot([NoAlias] float* vA, [NoAlias] float* vB, int n) {

            var pa = (float4*)vA;
            var pb = (float4*)vB;
            int nQ = n >> 2;
            float4 acc0 = default, acc1 = default;
            int q = 0;
            for (; q + 2 <= nQ; q += 2)
            {
                acc0 += pa[q]     * pb[q];
                acc1 += pa[q + 1] * pb[q + 1];
            }
            if (q < nQ) acc0 += pa[q] * pb[q];
            float4 acc = acc0 + acc1;
            float s = (acc.x + acc.y) + (acc.z + acc.w);
            for (int i = nQ << 2; i < n; i++)
                s += vA[i] * vB[i];
            return s;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static float vecDotRange([NoAlias] float* vA, [NoAlias] float* vB, int start, int end)
        {
            // Base the vector pointers at `start` (element-aligned only, fine for unaligned loads).
            int n = end - start;
            var pa = (float4*)(vA + start);
            var pb = (float4*)(vB + start);
            int nQ = n >> 2;
            float4 acc0 = default, acc1 = default;
            int q = 0;
            for (; q + 2 <= nQ; q += 2)
            {
                acc0 += pa[q]     * pb[q];
                acc1 += pa[q + 1] * pb[q + 1];
            }
            if (q < nQ) acc0 += pa[q] * pb[q];
            float4 acc = acc0 + acc1;
            float s = (acc.x + acc.y) + (acc.z + acc.w);
            for (int i = start + (nQ << 2); i < end; i++)
                s += vA[i] * vB[i];
            return s;
        }



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
            // y = m, needs to be initialized to zero (this accumulates into y)
            // y += mat * x
            //
            // Each row is a dot product (reduction). A single running accumulator is a serial FP-add
            // dependency chain that strict-FloatMode Burst cannot split into SIMD lanes (that would be
            // reassociation). We give it an EXPLICIT width-4 SIMD accumulator (float4 -> float4/double4):
            // four independent lane-chains packed into one register, advancing in parallel WITHOUT asking
            // the compiler to reassociate -- the per-lane summation order is fixed by the source and stays
            // deterministic. The balanced (x+y)+(z+w) fold matches the old scalar (s0+s1)+(s2+s3) exactly
            // (same lanes, same order) -> bit-identical result. Scalar multi-accumulator source
            // (s0..s3) does NOT get Burst to emit this; the vector type + reinterpret load does. (n%4 tail
            // handled scalar.)
            // TWO float4 accumulators (8 lane-chains): ~2x over a single 4-lane accumulator in-cache
            // (4 chains left the FP add ports half idle). 4 accumulators measured NO further gain
            // (memory/port-bound). Frozen fold: acc0+acc1 then (x+y)+(z+w).
            var xp = (float4*)x;
            int nQ = n >> 2;      // number of full width-4 blocks
            int tail = nQ << 2;   // first index of the scalar n%4 tail
            for (int r = 0; r < m; r++)
            {
                int baseIdx = r * n;
                var mp = (float4*)(mat + baseIdx);
                float4 acc0 = default, acc1 = default;
                int q = 0;
                for (; q + 2 <= nQ; q += 2)
                {
                    acc0 += mp[q]     * xp[q];
                    acc1 += mp[q + 1] * xp[q + 1];
                }
                if (q < nQ) acc0 += mp[q] * xp[q];
                float4 acc = acc0 + acc1;
                float sum = (acc.x + acc.y) + (acc.z + acc.w);
                for (int c = tail; c < n; c++)
                    sum += mat[baseIdx + c] * x[c];
                y[r] += sum;
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

        // Register-tiled GEMM: an MR x NR block of C is held in named scalar locals across the whole
        // k-reduction (p = 0..n-1), so each A value is reused NR times and each B value MR times, and
        // each C element is written once at the end. This keeps the kernel compute-bound; the untiled
        // fallback below re-streams matB once per output row and is bandwidth-bound.
        //
        // Determinism: every C[i,j] is still one running accumulator summing p ascending 0..n-1 with
        // the same `c += a*b` expression as the fallback. Tiling only interleaves independent
        // accumulators (ILP across the MR*NR chains) — it never splits an individual element's
        // k-reduction — so results are bit-identical to the fallback at every tile size and SIMD width.
        // (This determinism rule is also why there is no cache-level k-panel blocking.)
        //
        // Tile constants are method-local: a class-level const would collide across the generated
        // float/double partial-class files (CS0102). MR=8, NR=16 (16 AVX2 accumulator vectors, the
        // edge before register spilling) is used for both types and every size, no size gate. See
        // docs/dev/level3-blocking-guide.md for the blocking background and GemmBenchmark for the sweep.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void matMatDot([NoAlias] float* matA, [NoAlias] float* matB, [NoAlias] float* matC, int m, int n, int k)
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
                float* Arow0 = matA + (long)(i + 0) * n;
                float* Arow1 = matA + (long)(i + 1) * n;
                float* Arow2 = matA + (long)(i + 2) * n;
                float* Arow3 = matA + (long)(i + 3) * n;
                float* Arow4 = matA + (long)(i + 4) * n;
                float* Arow5 = matA + (long)(i + 5) * n;
                float* Arow6 = matA + (long)(i + 6) * n;
                float* Arow7 = matA + (long)(i + 7) * n;

                for (int j = 0; j < kTiles; j += NR)
                {
                    float c000 = 0f, c001 = 0f, c002 = 0f, c003 = 0f, c004 = 0f, c005 = 0f, c006 = 0f, c007 = 0f, c008 = 0f, c009 = 0f, c010 = 0f, c011 = 0f, c012 = 0f, c013 = 0f, c014 = 0f, c015 = 0f;
                    float c100 = 0f, c101 = 0f, c102 = 0f, c103 = 0f, c104 = 0f, c105 = 0f, c106 = 0f, c107 = 0f, c108 = 0f, c109 = 0f, c110 = 0f, c111 = 0f, c112 = 0f, c113 = 0f, c114 = 0f, c115 = 0f;
                    float c200 = 0f, c201 = 0f, c202 = 0f, c203 = 0f, c204 = 0f, c205 = 0f, c206 = 0f, c207 = 0f, c208 = 0f, c209 = 0f, c210 = 0f, c211 = 0f, c212 = 0f, c213 = 0f, c214 = 0f, c215 = 0f;
                    float c300 = 0f, c301 = 0f, c302 = 0f, c303 = 0f, c304 = 0f, c305 = 0f, c306 = 0f, c307 = 0f, c308 = 0f, c309 = 0f, c310 = 0f, c311 = 0f, c312 = 0f, c313 = 0f, c314 = 0f, c315 = 0f;
                    float c400 = 0f, c401 = 0f, c402 = 0f, c403 = 0f, c404 = 0f, c405 = 0f, c406 = 0f, c407 = 0f, c408 = 0f, c409 = 0f, c410 = 0f, c411 = 0f, c412 = 0f, c413 = 0f, c414 = 0f, c415 = 0f;
                    float c500 = 0f, c501 = 0f, c502 = 0f, c503 = 0f, c504 = 0f, c505 = 0f, c506 = 0f, c507 = 0f, c508 = 0f, c509 = 0f, c510 = 0f, c511 = 0f, c512 = 0f, c513 = 0f, c514 = 0f, c515 = 0f;
                    float c600 = 0f, c601 = 0f, c602 = 0f, c603 = 0f, c604 = 0f, c605 = 0f, c606 = 0f, c607 = 0f, c608 = 0f, c609 = 0f, c610 = 0f, c611 = 0f, c612 = 0f, c613 = 0f, c614 = 0f, c615 = 0f;
                    float c700 = 0f, c701 = 0f, c702 = 0f, c703 = 0f, c704 = 0f, c705 = 0f, c706 = 0f, c707 = 0f, c708 = 0f, c709 = 0f, c710 = 0f, c711 = 0f, c712 = 0f, c713 = 0f, c714 = 0f, c715 = 0f;

                    for (int p = 0; p < n; p++)
                    {
                        float a0 = Arow0[p];
                        float a1 = Arow1[p];
                        float a2 = Arow2[p];
                        float a3 = Arow3[p];
                        float a4 = Arow4[p];
                        float a5 = Arow5[p];
                        float a6 = Arow6[p];
                        float a7 = Arow7[p];

                        float* Brow = matB + (long)p * k + j;
                        float b0 = Brow[0];
                        float b1 = Brow[1];
                        float b2 = Brow[2];
                        float b3 = Brow[3];
                        float b4 = Brow[4];
                        float b5 = Brow[5];
                        float b6 = Brow[6];
                        float b7 = Brow[7];
                        float b8 = Brow[8];
                        float b9 = Brow[9];
                        float b10 = Brow[10];
                        float b11 = Brow[11];
                        float b12 = Brow[12];
                        float b13 = Brow[13];
                        float b14 = Brow[14];
                        float b15 = Brow[15];

                        c000 += a0 * b0; c001 += a0 * b1; c002 += a0 * b2; c003 += a0 * b3; c004 += a0 * b4; c005 += a0 * b5; c006 += a0 * b6; c007 += a0 * b7; c008 += a0 * b8; c009 += a0 * b9; c010 += a0 * b10; c011 += a0 * b11; c012 += a0 * b12; c013 += a0 * b13; c014 += a0 * b14; c015 += a0 * b15;
                        c100 += a1 * b0; c101 += a1 * b1; c102 += a1 * b2; c103 += a1 * b3; c104 += a1 * b4; c105 += a1 * b5; c106 += a1 * b6; c107 += a1 * b7; c108 += a1 * b8; c109 += a1 * b9; c110 += a1 * b10; c111 += a1 * b11; c112 += a1 * b12; c113 += a1 * b13; c114 += a1 * b14; c115 += a1 * b15;
                        c200 += a2 * b0; c201 += a2 * b1; c202 += a2 * b2; c203 += a2 * b3; c204 += a2 * b4; c205 += a2 * b5; c206 += a2 * b6; c207 += a2 * b7; c208 += a2 * b8; c209 += a2 * b9; c210 += a2 * b10; c211 += a2 * b11; c212 += a2 * b12; c213 += a2 * b13; c214 += a2 * b14; c215 += a2 * b15;
                        c300 += a3 * b0; c301 += a3 * b1; c302 += a3 * b2; c303 += a3 * b3; c304 += a3 * b4; c305 += a3 * b5; c306 += a3 * b6; c307 += a3 * b7; c308 += a3 * b8; c309 += a3 * b9; c310 += a3 * b10; c311 += a3 * b11; c312 += a3 * b12; c313 += a3 * b13; c314 += a3 * b14; c315 += a3 * b15;
                        c400 += a4 * b0; c401 += a4 * b1; c402 += a4 * b2; c403 += a4 * b3; c404 += a4 * b4; c405 += a4 * b5; c406 += a4 * b6; c407 += a4 * b7; c408 += a4 * b8; c409 += a4 * b9; c410 += a4 * b10; c411 += a4 * b11; c412 += a4 * b12; c413 += a4 * b13; c414 += a4 * b14; c415 += a4 * b15;
                        c500 += a5 * b0; c501 += a5 * b1; c502 += a5 * b2; c503 += a5 * b3; c504 += a5 * b4; c505 += a5 * b5; c506 += a5 * b6; c507 += a5 * b7; c508 += a5 * b8; c509 += a5 * b9; c510 += a5 * b10; c511 += a5 * b11; c512 += a5 * b12; c513 += a5 * b13; c514 += a5 * b14; c515 += a5 * b15;
                        c600 += a6 * b0; c601 += a6 * b1; c602 += a6 * b2; c603 += a6 * b3; c604 += a6 * b4; c605 += a6 * b5; c606 += a6 * b6; c607 += a6 * b7; c608 += a6 * b8; c609 += a6 * b9; c610 += a6 * b10; c611 += a6 * b11; c612 += a6 * b12; c613 += a6 * b13; c614 += a6 * b14; c615 += a6 * b15;
                        c700 += a7 * b0; c701 += a7 * b1; c702 += a7 * b2; c703 += a7 * b3; c704 += a7 * b4; c705 += a7 * b5; c706 += a7 * b6; c707 += a7 * b7; c708 += a7 * b8; c709 += a7 * b9; c710 += a7 * b10; c711 += a7 * b11; c712 += a7 * b12; c713 += a7 * b13; c714 += a7 * b14; c715 += a7 * b15;
                    }

                    float* Crow0 = matC + (long)(i + 0) * k + j;
                    float* Crow1 = matC + (long)(i + 1) * k + j;
                    float* Crow2 = matC + (long)(i + 2) * k + j;
                    float* Crow3 = matC + (long)(i + 3) * k + j;
                    float* Crow4 = matC + (long)(i + 4) * k + j;
                    float* Crow5 = matC + (long)(i + 5) * k + j;
                    float* Crow6 = matC + (long)(i + 6) * k + j;
                    float* Crow7 = matC + (long)(i + 7) * k + j;

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
        static void matMatDotRange([NoAlias] float* matA, [NoAlias] float* matB, [NoAlias] float* matC,
                                    int rowStart, int rowEnd, int n, int k, int colStart, int colEnd)
        {
            for (int r = rowStart; r < rowEnd; r++)
            {
                for (int nCols = 0; nCols < n; nCols++)
                {
                    float temp = matA[r * n + nCols];
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
        public static void matMatDotTransA([NoAlias] float* matA, [NoAlias] float* matB, [NoAlias] float* matC, int m, int n, int k)
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
                    float c000 = 0f, c001 = 0f, c002 = 0f, c003 = 0f, c004 = 0f, c005 = 0f, c006 = 0f, c007 = 0f, c008 = 0f, c009 = 0f, c010 = 0f, c011 = 0f, c012 = 0f, c013 = 0f, c014 = 0f, c015 = 0f;
                    float c100 = 0f, c101 = 0f, c102 = 0f, c103 = 0f, c104 = 0f, c105 = 0f, c106 = 0f, c107 = 0f, c108 = 0f, c109 = 0f, c110 = 0f, c111 = 0f, c112 = 0f, c113 = 0f, c114 = 0f, c115 = 0f;
                    float c200 = 0f, c201 = 0f, c202 = 0f, c203 = 0f, c204 = 0f, c205 = 0f, c206 = 0f, c207 = 0f, c208 = 0f, c209 = 0f, c210 = 0f, c211 = 0f, c212 = 0f, c213 = 0f, c214 = 0f, c215 = 0f;
                    float c300 = 0f, c301 = 0f, c302 = 0f, c303 = 0f, c304 = 0f, c305 = 0f, c306 = 0f, c307 = 0f, c308 = 0f, c309 = 0f, c310 = 0f, c311 = 0f, c312 = 0f, c313 = 0f, c314 = 0f, c315 = 0f;
                    float c400 = 0f, c401 = 0f, c402 = 0f, c403 = 0f, c404 = 0f, c405 = 0f, c406 = 0f, c407 = 0f, c408 = 0f, c409 = 0f, c410 = 0f, c411 = 0f, c412 = 0f, c413 = 0f, c414 = 0f, c415 = 0f;
                    float c500 = 0f, c501 = 0f, c502 = 0f, c503 = 0f, c504 = 0f, c505 = 0f, c506 = 0f, c507 = 0f, c508 = 0f, c509 = 0f, c510 = 0f, c511 = 0f, c512 = 0f, c513 = 0f, c514 = 0f, c515 = 0f;
                    float c600 = 0f, c601 = 0f, c602 = 0f, c603 = 0f, c604 = 0f, c605 = 0f, c606 = 0f, c607 = 0f, c608 = 0f, c609 = 0f, c610 = 0f, c611 = 0f, c612 = 0f, c613 = 0f, c614 = 0f, c615 = 0f;
                    float c700 = 0f, c701 = 0f, c702 = 0f, c703 = 0f, c704 = 0f, c705 = 0f, c706 = 0f, c707 = 0f, c708 = 0f, c709 = 0f, c710 = 0f, c711 = 0f, c712 = 0f, c713 = 0f, c714 = 0f, c715 = 0f;

                    for (int p = 0; p < n; p++)
                    {
                        float* Ap = matA + (long)p * m + i;
                        float a0 = Ap[0];
                        float a1 = Ap[1];
                        float a2 = Ap[2];
                        float a3 = Ap[3];
                        float a4 = Ap[4];
                        float a5 = Ap[5];
                        float a6 = Ap[6];
                        float a7 = Ap[7];

                        float* Brow = matB + (long)p * k + j;
                        float b0 = Brow[0];
                        float b1 = Brow[1];
                        float b2 = Brow[2];
                        float b3 = Brow[3];
                        float b4 = Brow[4];
                        float b5 = Brow[5];
                        float b6 = Brow[6];
                        float b7 = Brow[7];
                        float b8 = Brow[8];
                        float b9 = Brow[9];
                        float b10 = Brow[10];
                        float b11 = Brow[11];
                        float b12 = Brow[12];
                        float b13 = Brow[13];
                        float b14 = Brow[14];
                        float b15 = Brow[15];

                        c000 += a0 * b0; c001 += a0 * b1; c002 += a0 * b2; c003 += a0 * b3; c004 += a0 * b4; c005 += a0 * b5; c006 += a0 * b6; c007 += a0 * b7; c008 += a0 * b8; c009 += a0 * b9; c010 += a0 * b10; c011 += a0 * b11; c012 += a0 * b12; c013 += a0 * b13; c014 += a0 * b14; c015 += a0 * b15;
                        c100 += a1 * b0; c101 += a1 * b1; c102 += a1 * b2; c103 += a1 * b3; c104 += a1 * b4; c105 += a1 * b5; c106 += a1 * b6; c107 += a1 * b7; c108 += a1 * b8; c109 += a1 * b9; c110 += a1 * b10; c111 += a1 * b11; c112 += a1 * b12; c113 += a1 * b13; c114 += a1 * b14; c115 += a1 * b15;
                        c200 += a2 * b0; c201 += a2 * b1; c202 += a2 * b2; c203 += a2 * b3; c204 += a2 * b4; c205 += a2 * b5; c206 += a2 * b6; c207 += a2 * b7; c208 += a2 * b8; c209 += a2 * b9; c210 += a2 * b10; c211 += a2 * b11; c212 += a2 * b12; c213 += a2 * b13; c214 += a2 * b14; c215 += a2 * b15;
                        c300 += a3 * b0; c301 += a3 * b1; c302 += a3 * b2; c303 += a3 * b3; c304 += a3 * b4; c305 += a3 * b5; c306 += a3 * b6; c307 += a3 * b7; c308 += a3 * b8; c309 += a3 * b9; c310 += a3 * b10; c311 += a3 * b11; c312 += a3 * b12; c313 += a3 * b13; c314 += a3 * b14; c315 += a3 * b15;
                        c400 += a4 * b0; c401 += a4 * b1; c402 += a4 * b2; c403 += a4 * b3; c404 += a4 * b4; c405 += a4 * b5; c406 += a4 * b6; c407 += a4 * b7; c408 += a4 * b8; c409 += a4 * b9; c410 += a4 * b10; c411 += a4 * b11; c412 += a4 * b12; c413 += a4 * b13; c414 += a4 * b14; c415 += a4 * b15;
                        c500 += a5 * b0; c501 += a5 * b1; c502 += a5 * b2; c503 += a5 * b3; c504 += a5 * b4; c505 += a5 * b5; c506 += a5 * b6; c507 += a5 * b7; c508 += a5 * b8; c509 += a5 * b9; c510 += a5 * b10; c511 += a5 * b11; c512 += a5 * b12; c513 += a5 * b13; c514 += a5 * b14; c515 += a5 * b15;
                        c600 += a6 * b0; c601 += a6 * b1; c602 += a6 * b2; c603 += a6 * b3; c604 += a6 * b4; c605 += a6 * b5; c606 += a6 * b6; c607 += a6 * b7; c608 += a6 * b8; c609 += a6 * b9; c610 += a6 * b10; c611 += a6 * b11; c612 += a6 * b12; c613 += a6 * b13; c614 += a6 * b14; c615 += a6 * b15;
                        c700 += a7 * b0; c701 += a7 * b1; c702 += a7 * b2; c703 += a7 * b3; c704 += a7 * b4; c705 += a7 * b5; c706 += a7 * b6; c707 += a7 * b7; c708 += a7 * b8; c709 += a7 * b9; c710 += a7 * b10; c711 += a7 * b11; c712 += a7 * b12; c713 += a7 * b13; c714 += a7 * b14; c715 += a7 * b15;
                    }

                    float* Crow0 = matC + (long)(i + 0) * k + j;
                    float* Crow1 = matC + (long)(i + 1) * k + j;
                    float* Crow2 = matC + (long)(i + 2) * k + j;
                    float* Crow3 = matC + (long)(i + 3) * k + j;
                    float* Crow4 = matC + (long)(i + 4) * k + j;
                    float* Crow5 = matC + (long)(i + 5) * k + j;
                    float* Crow6 = matC + (long)(i + 6) * k + j;
                    float* Crow7 = matC + (long)(i + 7) * k + j;

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
        static void matMatDotTransARange([NoAlias] float* matA, [NoAlias] float* matB, [NoAlias] float* matC,
                                          int rowStart, int rowEnd, int m, int n, int k, int colStart, int colEnd)
        {
            for (int r = rowStart; r < rowEnd; r++)
            {
                for (int nCols = 0; nCols < n; nCols++)
                {
                    float temp = matA[nCols * m + r];
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
        // (row, column) pair (which would keep the unblocked sweep's O(n^2) call count).
        // [NoAlias] is truthful: L11 (rows [j0,j0+jb)) and
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
        // The caller pre-transposes V into Vt so the inner pass over pb is a long unit-stride
        // accumulation (axpy-shaped, no reduction dependency chain — the same pattern QR's wyVtC
        // exploits), rather than a per-element reduction dot of C's row against V's row.
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
        public static void formT([NoAlias] float* Vp, int Vld, int rows, int pb, [NoAlias] float* T, [NoAlias] float* tcol, [NoAlias] float* G)
        {
            UnsafeUtility.MemClear(G, (long)pb * pb * UnsafeUtility.SizeOf<float>());
            for (int t = 0; t < rows; t++)
            {
                float* Vrow = Vp + (long)t * Vld;
                for (int i = 0; i < pb; i++)
                {
                    float temp = Vrow[i];
                    float* Gi = G + (long)i * pb;
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
                        float sum = 0;
                        for (int l = k; l < i; l++)
                            sum += T[k * pb + l] * tcol[l];
                        T[k * pb + i] = sum;
                    }
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

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void axpy([NoAlias] float* y, [NoAlias] float* x, float a, int n) {

            for (int i = 0; i < n; i++)
                y[i] += a * x[i];
        }

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
        public static void normalizeL2InPlace([NoAlias] float* target, int n)
        {
            float sum = 0f;

            for (int i = 0; i < n; i++)
                sum += target[i] * target[i];
            
            sum = math.sqrt(sum);

            for (int i = 0; i < n; i++)
                target[i] /= sum;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static float normalizeL2InPlace([NoAlias] float* target, int start, int end)
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

        // In-place ASCENDING heapsort of `val[]`, keyed by `key[]` (parallel arrays -- the same
        // permutation is applied to both). O(n log n) WORST CASE, unlike a quicksort's O(n^2) worst
        // case, which matters here because this exists specifically to replace an O(n^2) pattern (see
        // caller). Used by LP.ladBR's weighted-median ratio-test scan
        // (LP.BarrodaleRoberts.float.cs): that scan used to repeatedly linear-scan the REMAINING
        // candidates for the current minimum ratio, removing the winner by swap-with-last each round --
        // an O(k) scan repeated up to k times is O(k^2), and at large m the candidate count k (bounded
        // by m) made this the dominant cost of the whole solve even though the reported pivot count
        // stayed small (each round can "fold" a candidate without registering as a pivot). Sorting once
        // up front costs O(k log k) and the caller then walks the result in a single linear pass.
        //
        // NOT a stable sort (heapsort never is): candidates with EXACTLY EQUAL keys may end up in a
        // different relative order than the linear-scan-with-swap-removal approach it replaces would
        // have produced. Callers that need the same tie-break behavior in the presence of exact key
        // ties (e.g. to stay bit-identical to existing test coverage) should gate this kernel out below
        // whatever size makes exact ties negligible for their data, and keep the tie-break-preserving
        // path for everything else -- see LP.BarrodaleRoberts.float.cs's own gate for the rationale.
        //
        // [NoAlias]: key and val are genuinely distinct arrays (the caller's parallel ratio/row-index
        // buffers) -- both are permuted in lockstep by every swap below.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void sortByKeyAscending([NoAlias] float* key, [NoAlias] int* val, int n)
        {
            // Build a max-heap over [0,n) (by key), then repeatedly swap the max to the tail and
            // shrink the heap -- the standard textbook heapsort, specialized to carry a parallel
            // int payload alongside each float key.
            for (int start = (n >> 1) - 1; start >= 0; start--)
                SiftDown(key, val, start, n);

            for (int end = n - 1; end > 0; end--)
            {
                float tk = key[0]; key[0] = key[end]; key[end] = tk;
                int tv = val[0]; val[0] = val[end]; val[end] = tv;
                SiftDown(key, val, 0, end);
            }
        }

        // Restores the max-heap property for the subtree rooted at `root`, over the live heap range
        // [0,heapLen). Standard sift-down: repeatedly swap `root` with its larger child until it is
        // >= both children (or a leaf).
        [MethodImpl(MethodImplOptions.NoInlining)]
        static void SiftDown([NoAlias] float* key, [NoAlias] int* val, int root, int heapLen)
        {
            while (true)
            {
                int child = 2 * root + 1;
                if (child >= heapLen) break;
                if (child + 1 < heapLen && key[child + 1] > key[child]) child++;
                if (key[root] >= key[child]) break;

                float tk = key[root]; key[root] = key[child]; key[child] = tk;
                int tv = val[root]; val[root] = val[child]; val[child] = tv;
                root = child;
            }
        }
    }
}