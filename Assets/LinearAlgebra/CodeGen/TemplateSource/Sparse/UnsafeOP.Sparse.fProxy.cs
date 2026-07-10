using System.Runtime.CompilerServices;
using Unity.Burst;

namespace LinearAlgebra.Internal
{
    public static unsafe partial class UnsafeOP
    {
        // General BR x BC block-CSR matvec: y = A * x. y must already be zeroed by the caller
        // (accumulates into y, like matVecDot). Correctness-first fallback -- BSR.spMV
        // routes here for rectangular blocks (BR != BC) and any square block size NOT covered
        // by the register-tile specializations below (bsrMatVecB1/B2/B3/B4/B6). BR/BC are
        // ordinary runtime fields here, so Burst cannot unroll/vectorize the inner block-multiply
        // loops -- that is exactly what the specializations below fix for the common square sizes.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVec([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] fProxy* values,
                                      [NoAlias] fProxy* x, [NoAlias] fProxy* y,
                                      int blockRows, int BR, int BC)
        {
            int blockLen = BR * BC;

            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int yBase = br * BR;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bc = colInd[k];
                    int xBase = bc * BC;
                    fProxy* block = values + k * blockLen;

                    for (int r = 0; r < BR; r++)
                    {
                        fProxy sum = 0;
                        int rowOff = r * BC;
                        for (int c = 0; c < BC; c++)
                            sum += block[rowOff + c] * x[xBase + c];
                        y[yBase + r] += sum;
                    }
                }
            }
        }

        // General BR x BC block-CSR transpose matvec: y = A^T * x. y must already be zeroed by
        // the caller. Walks the SAME row-major block-CSR storage as bsrMatVec (no separate
        // transposed copy) -- each stored block K contributes y[j-block] += K^T * x[i-block].
        // Safe single-threaded (no scatter race): every (br,k) pair touches a distinct block,
        // and different blocks in the same block-row write to DIFFERENT y[j-block] ranges only
        // when ColInd is duplicate-free per row, which ToBSR guarantees.
        // Correctness-first fallback -- BSR.spMVT routes here for rectangular blocks and
        // any square size not covered by bsrMatVecTB1/B2/B3/B4/B6 below.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecT([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] fProxy* values,
                                       [NoAlias] fProxy* x, [NoAlias] fProxy* y,
                                       int blockRows, int BR, int BC)
        {
            int blockLen = BR * BC;

            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int xBase = br * BR;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bc = colInd[k];
                    int yBase = bc * BC;
                    fProxy* block = values + k * blockLen;

                    for (int c = 0; c < BC; c++)
                    {
                        fProxy sum = 0;
                        for (int r = 0; r < BR; r++)
                            sum += block[r * BC + c] * x[xBase + r];
                        y[yBase + c] += sum;
                    }
                }
            }
        }

        // Symmetric BR x BR block-CSR matvec: y = A * x where A is stored as its UPPER block-
        // triangle only (ColInd >= blockRow for every stored block) and the strictly-lower triangle
        // is IMPLICIT (block (bj,bi) == transpose of stored block (bi,bj)). y must already be zeroed
        // by the caller (accumulates into y). For each stored block K at (bi,bj):
        //   - diagonal (bi==bj): y_i += K * x_j   (once -- K is used as the full BR x BR block as
        //     given, no packing/half-storage assumption; see fProxyBSR.Symmetric doc)
        //   - off-diagonal (bi<bj, guaranteed since only the upper triangle is stored): y_i += K * x_j
        //     AND y_j += K^T * x_i (the implicit mirrored lower block)
        // Single-threaded caller (IJob.Run, no parallel-for) -> the y_j scatter write from an
        // off-diagonal block is race-free, matching every other kernel in this file. Correctness-
        // first fallback -- BSR.spMV routes here for symmetric matrices whose BR is not one
        // of the register-tile specializations below (bsrMatVecSymB1/B2/B3/B4/B6).
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecSym([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] fProxy* values,
                                         [NoAlias] fProxy* x, [NoAlias] fProxy* y,
                                         int blockRows, int BR)
        {
            int blockLen = BR * BR;

            for (int bi = 0; bi < blockRows; bi++)
            {
                int rowStart = rowPtr[bi];
                int rowEnd = rowPtr[bi + 1];
                int yBaseI = bi * BR;

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bj = colInd[k];
                    int xBaseJ = bj * BR;
                    fProxy* block = values + k * blockLen;

                    // y_i += K * x_j  (always -- diagonal or off-diagonal)
                    for (int r = 0; r < BR; r++)
                    {
                        fProxy sum = 0;
                        int rowOff = r * BR;
                        for (int c = 0; c < BR; c++)
                            sum += block[rowOff + c] * x[xBaseJ + c];
                        y[yBaseI + r] += sum;
                    }

                    if (bi != bj)
                    {
                        // y_j += K^T * x_i  (the implicit mirrored lower block)
                        int yBaseJ = bj * BR;
                        int xBaseI = bi * BR;
                        for (int c = 0; c < BR; c++)
                        {
                            fProxy sum = 0;
                            for (int r = 0; r < BR; r++)
                                sum += block[r * BR + c] * x[xBaseI + r];
                            y[yBaseJ + c] += sum;
                        }
                    }
                }
            }
        }

        // =====================================================================================
        // Milestone D: block-size-specialized (fully unrolled) square-block kernels, b in
        // {1,2,3,4,6} (3x3 is the FEM/cloth workhorse). The general kernels above have RUNTIME
        // trip counts for the inner BR x BC loops (BR/BC are ordinary int fields), which Burst
        // cannot unroll or register-allocate. These kernels hardcode b as a literal constant in
        // the method body, so the whole block-multiply is a straight-line sequence of named
        // scalar locals -- Burst can register-allocate and (auto-)vectorize it.
        //
        // Krylov R2 (docs/draft-spec-krylov-optimization.md, R2): each of these ALSO had a single
        // per-output-row dependency chain across the row's stored blocks (one running accumulator
        // in memory, `y[...] += ...` executed once per k, so k=0..len-1 formed a serial FP-add
        // chain the CPU could not pipeline). Fixed the same way the SIMD reduction campaign fixed
        // matVecDot/vecDot: process the row's stored blocks in PAIRS (k, k+1) into two INDEPENDENT
        // local accumulators (one for even pair-slots, one for odd), summed once at row end (plus
        // a scalar tail for an odd-length row) -- two lane-chains instead of one, same trade the
        // 2x fProxy4 dense reductions made. This is a DIFFERENT split axis than the dense
        // reductions (those pack 4 contiguous elements into one SIMD register; a BSR row's stored
        // blocks are not contiguous in x/y, so there is nothing to reinterpret-load as fProxy4 --
        // the two chains here are plain scalar/named-local accumulators, same idiom, different
        // mechanism). Consequence: the row's TOTAL is now (evenSum + oddSum) [+ tail] instead of a
        // strict left-to-right fold -- ROUNDING-ONLY (not bit-identical; floating-point addition is
        // not associative), same pass over the same data, same "same-day" campaign lesson: try 2
        // accumulators, measure, stop (see LargeSparseBenchmark's spMV section and the round's
        // commit message for the A/B numbers -- do NOT add a 3rd/4th accumulator without a fresh
        // measurement motivating it). MEASURED EXCEPTION: b=1 (bsrMatVecB1/TB1/SymB1) is NOT
        // paired -- A/B'd at the clean-signal b=1 stencil benchmark and showed no measurable win
        // (flat within noise) over the original single-accumulator form, plausibly because a b=1
        // row here has only ~3 stored blocks (one fma each), too little work to amortize the
        // pairing bookkeeping against. Kept as the original kernel for that block size per the
        // spec's own instruction -- see each B1 kernel's own comment.
        //
        // bsrMatVecTB* (transpose) has no such row-local chain (each stored block in a row scatters
        // to a DIFFERENT y block, distinct addresses within one row) -- these still get the
        // mechanical 2-wide pairing (independent per-block locals computed before either store) for
        // the same reason register-tiling helps ANY tight loop (more independent in-flight work per
        // iteration), with writes issued in the original k-order so any cross-row scatter
        // accumulation this feeds elsewhere stays in the same order (bit-identical for T).
        //
        // bsrMatVecSymB* mirrors both: the "y_i += K*x_j" always-part is the same per-row chain as
        // the forward kernels (paired, rounding-only) while the "y_j += K^T*x_i" mirrored scatter
        // part keeps its original per-k position/order (bit-identical for that part) -- see each
        // kernel below.
        //
        // Dispatch lives in BSR.spMV / spMVT / spMVDot (SparseOP.fProxy.cs).
        // =====================================================================================

        // ---- bsrMatVec: y = A * x, square block b -----------------------------------------

        // b=1 is deliberately NOT accumulator-split (unlike B2..B6 below): A/B'd at the b=1
        // stencil section of LargeSparseBenchmark (Krylov R2, docs/draft-spec-krylov-optimization.md)
        // -- CG/MINRES showed no measurable win (flat within noise, 0 to +2%) over this original
        // single-accumulator form. Root cause judged architectural: a b=1 row here has ~3 stored
        // blocks (tridiagonal stencil), each contributing exactly ONE fma -- there is barely
        // enough per-row work to amortize the extra bookkeeping (len/kPairEnd, the paired-loop
        // branch, the acc0+acc1 fold) the pairing needs, unlike B2..B6 where each stored block is
        // several fma's (more real work per loop-control-overhead unit). Per the spec's own
        // instruction ("try 2 accumulators, measure, stop -- if it doesn't measurably win, keep
        // the original kernel for that block size"): kept as the original single-chain form.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecB1([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] fProxy* values,
                                        [NoAlias] fProxy* x, [NoAlias] fProxy* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];

                for (int k = rowStart; k < rowEnd; k++)
                    y[br] += values[k] * x[colInd[k]];
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecB2([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] fProxy* values,
                                        [NoAlias] fProxy* x, [NoAlias] fProxy* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int yBase = br * 2;
                int len = rowEnd - rowStart;
                int kPairEnd = rowStart + ((len >> 1) << 1);

                fProxy acc0_0 = default, acc0_1 = default;
                fProxy acc1_0 = default, acc1_1 = default;

                int k = rowStart;
                for (; k < kPairEnd; k += 2)
                {
                    int xBase0 = colInd[k] * 2;
                    fProxy* block0 = values + k * 4;
                    fProxy x00 = x[xBase0 + 0];
                    fProxy x01 = x[xBase0 + 1];
                    acc0_0 += block0[0] * x00 + block0[1] * x01;
                    acc0_1 += block0[2] * x00 + block0[3] * x01;

                    int xBase1 = colInd[k + 1] * 2;
                    fProxy* block1 = values + (k + 1) * 4;
                    fProxy x10 = x[xBase1 + 0];
                    fProxy x11 = x[xBase1 + 1];
                    acc1_0 += block1[0] * x10 + block1[1] * x11;
                    acc1_1 += block1[2] * x10 + block1[3] * x11;
                }

                fProxy s0 = acc0_0 + acc1_0;
                fProxy s1 = acc0_1 + acc1_1;
                if (k < rowEnd)
                {
                    int xBase = colInd[k] * 2;
                    fProxy* block = values + k * 4;
                    fProxy x0 = x[xBase + 0];
                    fProxy x1 = x[xBase + 1];
                    s0 += block[0] * x0 + block[1] * x1;
                    s1 += block[2] * x0 + block[3] * x1;
                }

                y[yBase + 0] += s0;
                y[yBase + 1] += s1;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecB3([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] fProxy* values,
                                        [NoAlias] fProxy* x, [NoAlias] fProxy* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int yBase = br * 3;
                int len = rowEnd - rowStart;
                int kPairEnd = rowStart + ((len >> 1) << 1);

                fProxy acc0_0 = default, acc0_1 = default, acc0_2 = default;
                fProxy acc1_0 = default, acc1_1 = default, acc1_2 = default;

                int k = rowStart;
                for (; k < kPairEnd; k += 2)
                {
                    int xBase0 = colInd[k] * 3;
                    fProxy* block0 = values + k * 9;
                    fProxy x00 = x[xBase0 + 0];
                    fProxy x01 = x[xBase0 + 1];
                    fProxy x02 = x[xBase0 + 2];
                    acc0_0 += block0[0] * x00 + block0[1] * x01 + block0[2] * x02;
                    acc0_1 += block0[3] * x00 + block0[4] * x01 + block0[5] * x02;
                    acc0_2 += block0[6] * x00 + block0[7] * x01 + block0[8] * x02;

                    int xBase1 = colInd[k + 1] * 3;
                    fProxy* block1 = values + (k + 1) * 9;
                    fProxy x10 = x[xBase1 + 0];
                    fProxy x11 = x[xBase1 + 1];
                    fProxy x12 = x[xBase1 + 2];
                    acc1_0 += block1[0] * x10 + block1[1] * x11 + block1[2] * x12;
                    acc1_1 += block1[3] * x10 + block1[4] * x11 + block1[5] * x12;
                    acc1_2 += block1[6] * x10 + block1[7] * x11 + block1[8] * x12;
                }

                fProxy s0 = acc0_0 + acc1_0;
                fProxy s1 = acc0_1 + acc1_1;
                fProxy s2 = acc0_2 + acc1_2;
                if (k < rowEnd)
                {
                    int xBase = colInd[k] * 3;
                    fProxy* block = values + k * 9;
                    fProxy x0 = x[xBase + 0];
                    fProxy x1 = x[xBase + 1];
                    fProxy x2 = x[xBase + 2];
                    s0 += block[0] * x0 + block[1] * x1 + block[2] * x2;
                    s1 += block[3] * x0 + block[4] * x1 + block[5] * x2;
                    s2 += block[6] * x0 + block[7] * x1 + block[8] * x2;
                }

                y[yBase + 0] += s0;
                y[yBase + 1] += s1;
                y[yBase + 2] += s2;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecB4([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] fProxy* values,
                                        [NoAlias] fProxy* x, [NoAlias] fProxy* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int yBase = br * 4;
                int len = rowEnd - rowStart;
                int kPairEnd = rowStart + ((len >> 1) << 1);

                fProxy acc0_0 = default, acc0_1 = default, acc0_2 = default, acc0_3 = default;
                fProxy acc1_0 = default, acc1_1 = default, acc1_2 = default, acc1_3 = default;

                int k = rowStart;
                for (; k < kPairEnd; k += 2)
                {
                    int xBase0 = colInd[k] * 4;
                    fProxy* block0 = values + k * 16;
                    fProxy x00 = x[xBase0 + 0];
                    fProxy x01 = x[xBase0 + 1];
                    fProxy x02 = x[xBase0 + 2];
                    fProxy x03 = x[xBase0 + 3];
                    acc0_0 += block0[0]  * x00 + block0[1]  * x01 + block0[2]  * x02 + block0[3]  * x03;
                    acc0_1 += block0[4]  * x00 + block0[5]  * x01 + block0[6]  * x02 + block0[7]  * x03;
                    acc0_2 += block0[8]  * x00 + block0[9]  * x01 + block0[10] * x02 + block0[11] * x03;
                    acc0_3 += block0[12] * x00 + block0[13] * x01 + block0[14] * x02 + block0[15] * x03;

                    int xBase1 = colInd[k + 1] * 4;
                    fProxy* block1 = values + (k + 1) * 16;
                    fProxy x10 = x[xBase1 + 0];
                    fProxy x11 = x[xBase1 + 1];
                    fProxy x12 = x[xBase1 + 2];
                    fProxy x13 = x[xBase1 + 3];
                    acc1_0 += block1[0]  * x10 + block1[1]  * x11 + block1[2]  * x12 + block1[3]  * x13;
                    acc1_1 += block1[4]  * x10 + block1[5]  * x11 + block1[6]  * x12 + block1[7]  * x13;
                    acc1_2 += block1[8]  * x10 + block1[9]  * x11 + block1[10] * x12 + block1[11] * x13;
                    acc1_3 += block1[12] * x10 + block1[13] * x11 + block1[14] * x12 + block1[15] * x13;
                }

                fProxy s0 = acc0_0 + acc1_0;
                fProxy s1 = acc0_1 + acc1_1;
                fProxy s2 = acc0_2 + acc1_2;
                fProxy s3 = acc0_3 + acc1_3;
                if (k < rowEnd)
                {
                    int xBase = colInd[k] * 4;
                    fProxy* block = values + k * 16;
                    fProxy x0 = x[xBase + 0];
                    fProxy x1 = x[xBase + 1];
                    fProxy x2 = x[xBase + 2];
                    fProxy x3 = x[xBase + 3];
                    s0 += block[0]  * x0 + block[1]  * x1 + block[2]  * x2 + block[3]  * x3;
                    s1 += block[4]  * x0 + block[5]  * x1 + block[6]  * x2 + block[7]  * x3;
                    s2 += block[8]  * x0 + block[9]  * x1 + block[10] * x2 + block[11] * x3;
                    s3 += block[12] * x0 + block[13] * x1 + block[14] * x2 + block[15] * x3;
                }

                y[yBase + 0] += s0;
                y[yBase + 1] += s1;
                y[yBase + 2] += s2;
                y[yBase + 3] += s3;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecB6([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] fProxy* values,
                                        [NoAlias] fProxy* x, [NoAlias] fProxy* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int yBase = br * 6;
                int len = rowEnd - rowStart;
                int kPairEnd = rowStart + ((len >> 1) << 1);

                fProxy acc0_0 = default, acc0_1 = default, acc0_2 = default, acc0_3 = default, acc0_4 = default, acc0_5 = default;
                fProxy acc1_0 = default, acc1_1 = default, acc1_2 = default, acc1_3 = default, acc1_4 = default, acc1_5 = default;

                int k = rowStart;
                for (; k < kPairEnd; k += 2)
                {
                    int xBase0 = colInd[k] * 6;
                    fProxy* block0 = values + k * 36;
                    fProxy x00 = x[xBase0 + 0];
                    fProxy x01 = x[xBase0 + 1];
                    fProxy x02 = x[xBase0 + 2];
                    fProxy x03 = x[xBase0 + 3];
                    fProxy x04 = x[xBase0 + 4];
                    fProxy x05 = x[xBase0 + 5];
                    acc0_0 += block0[0]  * x00 + block0[1]  * x01 + block0[2]  * x02 + block0[3]  * x03 + block0[4]  * x04 + block0[5]  * x05;
                    acc0_1 += block0[6]  * x00 + block0[7]  * x01 + block0[8]  * x02 + block0[9]  * x03 + block0[10] * x04 + block0[11] * x05;
                    acc0_2 += block0[12] * x00 + block0[13] * x01 + block0[14] * x02 + block0[15] * x03 + block0[16] * x04 + block0[17] * x05;
                    acc0_3 += block0[18] * x00 + block0[19] * x01 + block0[20] * x02 + block0[21] * x03 + block0[22] * x04 + block0[23] * x05;
                    acc0_4 += block0[24] * x00 + block0[25] * x01 + block0[26] * x02 + block0[27] * x03 + block0[28] * x04 + block0[29] * x05;
                    acc0_5 += block0[30] * x00 + block0[31] * x01 + block0[32] * x02 + block0[33] * x03 + block0[34] * x04 + block0[35] * x05;

                    int xBase1 = colInd[k + 1] * 6;
                    fProxy* block1 = values + (k + 1) * 36;
                    fProxy x10 = x[xBase1 + 0];
                    fProxy x11 = x[xBase1 + 1];
                    fProxy x12 = x[xBase1 + 2];
                    fProxy x13 = x[xBase1 + 3];
                    fProxy x14 = x[xBase1 + 4];
                    fProxy x15 = x[xBase1 + 5];
                    acc1_0 += block1[0]  * x10 + block1[1]  * x11 + block1[2]  * x12 + block1[3]  * x13 + block1[4]  * x14 + block1[5]  * x15;
                    acc1_1 += block1[6]  * x10 + block1[7]  * x11 + block1[8]  * x12 + block1[9]  * x13 + block1[10] * x14 + block1[11] * x15;
                    acc1_2 += block1[12] * x10 + block1[13] * x11 + block1[14] * x12 + block1[15] * x13 + block1[16] * x14 + block1[17] * x15;
                    acc1_3 += block1[18] * x10 + block1[19] * x11 + block1[20] * x12 + block1[21] * x13 + block1[22] * x14 + block1[23] * x15;
                    acc1_4 += block1[24] * x10 + block1[25] * x11 + block1[26] * x12 + block1[27] * x13 + block1[28] * x14 + block1[29] * x15;
                    acc1_5 += block1[30] * x10 + block1[31] * x11 + block1[32] * x12 + block1[33] * x13 + block1[34] * x14 + block1[35] * x15;
                }

                fProxy s0 = acc0_0 + acc1_0;
                fProxy s1 = acc0_1 + acc1_1;
                fProxy s2 = acc0_2 + acc1_2;
                fProxy s3 = acc0_3 + acc1_3;
                fProxy s4 = acc0_4 + acc1_4;
                fProxy s5 = acc0_5 + acc1_5;
                if (k < rowEnd)
                {
                    int xBase = colInd[k] * 6;
                    fProxy* block = values + k * 36;
                    fProxy x0 = x[xBase + 0];
                    fProxy x1 = x[xBase + 1];
                    fProxy x2 = x[xBase + 2];
                    fProxy x3 = x[xBase + 3];
                    fProxy x4 = x[xBase + 4];
                    fProxy x5 = x[xBase + 5];
                    s0 += block[0]  * x0 + block[1]  * x1 + block[2]  * x2 + block[3]  * x3 + block[4]  * x4 + block[5]  * x5;
                    s1 += block[6]  * x0 + block[7]  * x1 + block[8]  * x2 + block[9]  * x3 + block[10] * x4 + block[11] * x5;
                    s2 += block[12] * x0 + block[13] * x1 + block[14] * x2 + block[15] * x3 + block[16] * x4 + block[17] * x5;
                    s3 += block[18] * x0 + block[19] * x1 + block[20] * x2 + block[21] * x3 + block[22] * x4 + block[23] * x5;
                    s4 += block[24] * x0 + block[25] * x1 + block[26] * x2 + block[27] * x3 + block[28] * x4 + block[29] * x5;
                    s5 += block[30] * x0 + block[31] * x1 + block[32] * x2 + block[33] * x3 + block[34] * x4 + block[35] * x5;
                }

                y[yBase + 0] += s0;
                y[yBase + 1] += s1;
                y[yBase + 2] += s2;
                y[yBase + 3] += s3;
                y[yBase + 4] += s4;
                y[yBase + 5] += s5;
            }
        }

        // ---- bsrMatVecT: y = A^T * x, square block b ---------------------------------------

        // b=1 not paired -- see bsrMatVecB1's comment (same A/B finding applies: trivial
        // per-block work at b=1 leaves nothing for pairing to amortize against).
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecTB1([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] fProxy* values,
                                         [NoAlias] fProxy* x, [NoAlias] fProxy* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                fProxy xv = x[br];

                for (int k = rowStart; k < rowEnd; k++)
                    y[colInd[k]] += values[k] * xv;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecTB2([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] fProxy* values,
                                         [NoAlias] fProxy* x, [NoAlias] fProxy* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int xBase = br * 2;
                fProxy x0 = x[xBase + 0];
                fProxy x1 = x[xBase + 1];
                int len = rowEnd - rowStart;
                int kPairEnd = rowStart + ((len >> 1) << 1);

                int k = rowStart;
                for (; k < kPairEnd; k += 2)
                {
                    fProxy* block0 = values + k * 4;
                    fProxy y00 = block0[0] * x0 + block0[2] * x1;
                    fProxy y01 = block0[1] * x0 + block0[3] * x1;

                    fProxy* block1 = values + (k + 1) * 4;
                    fProxy y10 = block1[0] * x0 + block1[2] * x1;
                    fProxy y11 = block1[1] * x0 + block1[3] * x1;

                    int yBase0 = colInd[k] * 2;
                    y[yBase0 + 0] += y00;
                    y[yBase0 + 1] += y01;

                    int yBase1 = colInd[k + 1] * 2;
                    y[yBase1 + 0] += y10;
                    y[yBase1 + 1] += y11;
                }
                if (k < rowEnd)
                {
                    int yBase = colInd[k] * 2;
                    fProxy* block = values + k * 4;
                    y[yBase + 0] += block[0] * x0 + block[2] * x1;
                    y[yBase + 1] += block[1] * x0 + block[3] * x1;
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecTB3([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] fProxy* values,
                                         [NoAlias] fProxy* x, [NoAlias] fProxy* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int xBase = br * 3;
                fProxy x0 = x[xBase + 0];
                fProxy x1 = x[xBase + 1];
                fProxy x2 = x[xBase + 2];
                int len = rowEnd - rowStart;
                int kPairEnd = rowStart + ((len >> 1) << 1);

                int k = rowStart;
                for (; k < kPairEnd; k += 2)
                {
                    fProxy* block0 = values + k * 9;
                    fProxy y00 = block0[0] * x0 + block0[3] * x1 + block0[6] * x2;
                    fProxy y01 = block0[1] * x0 + block0[4] * x1 + block0[7] * x2;
                    fProxy y02 = block0[2] * x0 + block0[5] * x1 + block0[8] * x2;

                    fProxy* block1 = values + (k + 1) * 9;
                    fProxy y10 = block1[0] * x0 + block1[3] * x1 + block1[6] * x2;
                    fProxy y11 = block1[1] * x0 + block1[4] * x1 + block1[7] * x2;
                    fProxy y12 = block1[2] * x0 + block1[5] * x1 + block1[8] * x2;

                    int yBase0 = colInd[k] * 3;
                    y[yBase0 + 0] += y00;
                    y[yBase0 + 1] += y01;
                    y[yBase0 + 2] += y02;

                    int yBase1 = colInd[k + 1] * 3;
                    y[yBase1 + 0] += y10;
                    y[yBase1 + 1] += y11;
                    y[yBase1 + 2] += y12;
                }
                if (k < rowEnd)
                {
                    int yBase = colInd[k] * 3;
                    fProxy* block = values + k * 9;
                    y[yBase + 0] += block[0] * x0 + block[3] * x1 + block[6] * x2;
                    y[yBase + 1] += block[1] * x0 + block[4] * x1 + block[7] * x2;
                    y[yBase + 2] += block[2] * x0 + block[5] * x1 + block[8] * x2;
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecTB4([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] fProxy* values,
                                         [NoAlias] fProxy* x, [NoAlias] fProxy* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int xBase = br * 4;
                fProxy x0 = x[xBase + 0];
                fProxy x1 = x[xBase + 1];
                fProxy x2 = x[xBase + 2];
                fProxy x3 = x[xBase + 3];
                int len = rowEnd - rowStart;
                int kPairEnd = rowStart + ((len >> 1) << 1);

                int k = rowStart;
                for (; k < kPairEnd; k += 2)
                {
                    fProxy* block0 = values + k * 16;
                    fProxy y00 = block0[0] * x0 + block0[4]  * x1 + block0[8]  * x2 + block0[12] * x3;
                    fProxy y01 = block0[1] * x0 + block0[5]  * x1 + block0[9]  * x2 + block0[13] * x3;
                    fProxy y02 = block0[2] * x0 + block0[6]  * x1 + block0[10] * x2 + block0[14] * x3;
                    fProxy y03 = block0[3] * x0 + block0[7]  * x1 + block0[11] * x2 + block0[15] * x3;

                    fProxy* block1 = values + (k + 1) * 16;
                    fProxy y10 = block1[0] * x0 + block1[4]  * x1 + block1[8]  * x2 + block1[12] * x3;
                    fProxy y11 = block1[1] * x0 + block1[5]  * x1 + block1[9]  * x2 + block1[13] * x3;
                    fProxy y12 = block1[2] * x0 + block1[6]  * x1 + block1[10] * x2 + block1[14] * x3;
                    fProxy y13 = block1[3] * x0 + block1[7]  * x1 + block1[11] * x2 + block1[15] * x3;

                    int yBase0 = colInd[k] * 4;
                    y[yBase0 + 0] += y00;
                    y[yBase0 + 1] += y01;
                    y[yBase0 + 2] += y02;
                    y[yBase0 + 3] += y03;

                    int yBase1 = colInd[k + 1] * 4;
                    y[yBase1 + 0] += y10;
                    y[yBase1 + 1] += y11;
                    y[yBase1 + 2] += y12;
                    y[yBase1 + 3] += y13;
                }
                if (k < rowEnd)
                {
                    int yBase = colInd[k] * 4;
                    fProxy* block = values + k * 16;
                    y[yBase + 0] += block[0] * x0 + block[4]  * x1 + block[8]  * x2 + block[12] * x3;
                    y[yBase + 1] += block[1] * x0 + block[5]  * x1 + block[9]  * x2 + block[13] * x3;
                    y[yBase + 2] += block[2] * x0 + block[6]  * x1 + block[10] * x2 + block[14] * x3;
                    y[yBase + 3] += block[3] * x0 + block[7]  * x1 + block[11] * x2 + block[15] * x3;
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecTB6([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] fProxy* values,
                                         [NoAlias] fProxy* x, [NoAlias] fProxy* y, int blockRows)
        {
            for (int br = 0; br < blockRows; br++)
            {
                int rowStart = rowPtr[br];
                int rowEnd = rowPtr[br + 1];
                int xBase = br * 6;
                fProxy x0 = x[xBase + 0];
                fProxy x1 = x[xBase + 1];
                fProxy x2 = x[xBase + 2];
                fProxy x3 = x[xBase + 3];
                fProxy x4 = x[xBase + 4];
                fProxy x5 = x[xBase + 5];
                int len = rowEnd - rowStart;
                int kPairEnd = rowStart + ((len >> 1) << 1);

                int k = rowStart;
                for (; k < kPairEnd; k += 2)
                {
                    fProxy* block0 = values + k * 36;
                    fProxy y00 = block0[0] * x0 + block0[6]  * x1 + block0[12] * x2 + block0[18] * x3 + block0[24] * x4 + block0[30] * x5;
                    fProxy y01 = block0[1] * x0 + block0[7]  * x1 + block0[13] * x2 + block0[19] * x3 + block0[25] * x4 + block0[31] * x5;
                    fProxy y02 = block0[2] * x0 + block0[8]  * x1 + block0[14] * x2 + block0[20] * x3 + block0[26] * x4 + block0[32] * x5;
                    fProxy y03 = block0[3] * x0 + block0[9]  * x1 + block0[15] * x2 + block0[21] * x3 + block0[27] * x4 + block0[33] * x5;
                    fProxy y04 = block0[4] * x0 + block0[10] * x1 + block0[16] * x2 + block0[22] * x3 + block0[28] * x4 + block0[34] * x5;
                    fProxy y05 = block0[5] * x0 + block0[11] * x1 + block0[17] * x2 + block0[23] * x3 + block0[29] * x4 + block0[35] * x5;

                    fProxy* block1 = values + (k + 1) * 36;
                    fProxy y10 = block1[0] * x0 + block1[6]  * x1 + block1[12] * x2 + block1[18] * x3 + block1[24] * x4 + block1[30] * x5;
                    fProxy y11 = block1[1] * x0 + block1[7]  * x1 + block1[13] * x2 + block1[19] * x3 + block1[25] * x4 + block1[31] * x5;
                    fProxy y12 = block1[2] * x0 + block1[8]  * x1 + block1[14] * x2 + block1[20] * x3 + block1[26] * x4 + block1[32] * x5;
                    fProxy y13 = block1[3] * x0 + block1[9]  * x1 + block1[15] * x2 + block1[21] * x3 + block1[27] * x4 + block1[33] * x5;
                    fProxy y14 = block1[4] * x0 + block1[10] * x1 + block1[16] * x2 + block1[22] * x3 + block1[28] * x4 + block1[34] * x5;
                    fProxy y15 = block1[5] * x0 + block1[11] * x1 + block1[17] * x2 + block1[23] * x3 + block1[29] * x4 + block1[35] * x5;

                    int yBase0 = colInd[k] * 6;
                    y[yBase0 + 0] += y00;
                    y[yBase0 + 1] += y01;
                    y[yBase0 + 2] += y02;
                    y[yBase0 + 3] += y03;
                    y[yBase0 + 4] += y04;
                    y[yBase0 + 5] += y05;

                    int yBase1 = colInd[k + 1] * 6;
                    y[yBase1 + 0] += y10;
                    y[yBase1 + 1] += y11;
                    y[yBase1 + 2] += y12;
                    y[yBase1 + 3] += y13;
                    y[yBase1 + 4] += y14;
                    y[yBase1 + 5] += y15;
                }
                if (k < rowEnd)
                {
                    int yBase = colInd[k] * 6;
                    fProxy* block = values + k * 36;
                    y[yBase + 0] += block[0] * x0 + block[6]  * x1 + block[12] * x2 + block[18] * x3 + block[24] * x4 + block[30] * x5;
                    y[yBase + 1] += block[1] * x0 + block[7]  * x1 + block[13] * x2 + block[19] * x3 + block[25] * x4 + block[31] * x5;
                    y[yBase + 2] += block[2] * x0 + block[8]  * x1 + block[14] * x2 + block[20] * x3 + block[26] * x4 + block[32] * x5;
                    y[yBase + 3] += block[3] * x0 + block[9]  * x1 + block[15] * x2 + block[21] * x3 + block[27] * x4 + block[33] * x5;
                    y[yBase + 4] += block[4] * x0 + block[10] * x1 + block[16] * x2 + block[22] * x3 + block[28] * x4 + block[34] * x5;
                    y[yBase + 5] += block[5] * x0 + block[11] * x1 + block[17] * x2 + block[23] * x3 + block[29] * x4 + block[35] * x5;
                }
            }
        }

        // ---- bsrMatVecSym: y = A * x, symmetric upper-block-triangle storage, square block b --
        //
        // Each specialization is the fused pair of the corresponding bsrMatVecB{b} row-pass
        // (y_i += K * x_j, always -- the per-row chain, PAIRED below like bsrMatVecB{b}) and
        // bsrMatVecTB{b} column-pass (y_j += K^T * x_i, only when bi != bj -- kept at its original
        // per-k position/order, bit-identical) applied to the SAME stored block K -- see
        // bsrMatVecSym above for the general version this mirrors. xi0.. (row bi's own x slice) is
        // hoisted out of the k-loop (used by every off-diagonal scatter in the row) -- a harmless,
        // bit-identical read-once instead of the original's read-every-scatter.

        // b=1 not paired -- see bsrMatVecB1's comment (same A/B finding applies).
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecSymB1([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] fProxy* values,
                                           [NoAlias] fProxy* x, [NoAlias] fProxy* y, int blockRows)
        {
            for (int bi = 0; bi < blockRows; bi++)
            {
                int rowStart = rowPtr[bi];
                int rowEnd = rowPtr[bi + 1];

                for (int k = rowStart; k < rowEnd; k++)
                {
                    int bj = colInd[k];
                    fProxy v = values[k];

                    y[bi] += v * x[bj];
                    if (bi != bj)
                        y[bj] += v * x[bi];
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecSymB2([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] fProxy* values,
                                           [NoAlias] fProxy* x, [NoAlias] fProxy* y, int blockRows)
        {
            for (int bi = 0; bi < blockRows; bi++)
            {
                int rowStart = rowPtr[bi];
                int rowEnd = rowPtr[bi + 1];
                int yBaseI = bi * 2;
                int xBaseI = bi * 2;
                fProxy xi0 = x[xBaseI + 0];
                fProxy xi1 = x[xBaseI + 1];
                int len = rowEnd - rowStart;
                int kPairEnd = rowStart + ((len >> 1) << 1);

                fProxy acc0_0 = default, acc0_1 = default;
                fProxy acc1_0 = default, acc1_1 = default;

                int k = rowStart;
                for (; k < kPairEnd; k += 2)
                {
                    int bj0 = colInd[k];
                    int xBaseJ0 = bj0 * 2;
                    fProxy* block0 = values + k * 4;
                    fProxy xj00 = x[xBaseJ0 + 0];
                    fProxy xj01 = x[xBaseJ0 + 1];
                    acc0_0 += block0[0] * xj00 + block0[1] * xj01;
                    acc0_1 += block0[2] * xj00 + block0[3] * xj01;
                    if (bi != bj0)
                    {
                        int yBaseJ0 = bj0 * 2;
                        y[yBaseJ0 + 0] += block0[0] * xi0 + block0[2] * xi1;
                        y[yBaseJ0 + 1] += block0[1] * xi0 + block0[3] * xi1;
                    }

                    int bj1 = colInd[k + 1];
                    int xBaseJ1 = bj1 * 2;
                    fProxy* block1 = values + (k + 1) * 4;
                    fProxy xj10 = x[xBaseJ1 + 0];
                    fProxy xj11 = x[xBaseJ1 + 1];
                    acc1_0 += block1[0] * xj10 + block1[1] * xj11;
                    acc1_1 += block1[2] * xj10 + block1[3] * xj11;
                    if (bi != bj1)
                    {
                        int yBaseJ1 = bj1 * 2;
                        y[yBaseJ1 + 0] += block1[0] * xi0 + block1[2] * xi1;
                        y[yBaseJ1 + 1] += block1[1] * xi0 + block1[3] * xi1;
                    }
                }

                fProxy s0 = acc0_0 + acc1_0;
                fProxy s1 = acc0_1 + acc1_1;
                if (k < rowEnd)
                {
                    int bj = colInd[k];
                    int xBaseJ = bj * 2;
                    fProxy* block = values + k * 4;
                    fProxy xj0 = x[xBaseJ + 0];
                    fProxy xj1 = x[xBaseJ + 1];
                    s0 += block[0] * xj0 + block[1] * xj1;
                    s1 += block[2] * xj0 + block[3] * xj1;
                    if (bi != bj)
                    {
                        int yBaseJ = bj * 2;
                        y[yBaseJ + 0] += block[0] * xi0 + block[2] * xi1;
                        y[yBaseJ + 1] += block[1] * xi0 + block[3] * xi1;
                    }
                }

                y[yBaseI + 0] += s0;
                y[yBaseI + 1] += s1;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecSymB3([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] fProxy* values,
                                           [NoAlias] fProxy* x, [NoAlias] fProxy* y, int blockRows)
        {
            for (int bi = 0; bi < blockRows; bi++)
            {
                int rowStart = rowPtr[bi];
                int rowEnd = rowPtr[bi + 1];
                int yBaseI = bi * 3;
                int xBaseI = bi * 3;
                fProxy xi0 = x[xBaseI + 0];
                fProxy xi1 = x[xBaseI + 1];
                fProxy xi2 = x[xBaseI + 2];
                int len = rowEnd - rowStart;
                int kPairEnd = rowStart + ((len >> 1) << 1);

                fProxy acc0_0 = default, acc0_1 = default, acc0_2 = default;
                fProxy acc1_0 = default, acc1_1 = default, acc1_2 = default;

                int k = rowStart;
                for (; k < kPairEnd; k += 2)
                {
                    int bj0 = colInd[k];
                    int xBaseJ0 = bj0 * 3;
                    fProxy* block0 = values + k * 9;
                    fProxy xj00 = x[xBaseJ0 + 0];
                    fProxy xj01 = x[xBaseJ0 + 1];
                    fProxy xj02 = x[xBaseJ0 + 2];
                    acc0_0 += block0[0] * xj00 + block0[1] * xj01 + block0[2] * xj02;
                    acc0_1 += block0[3] * xj00 + block0[4] * xj01 + block0[5] * xj02;
                    acc0_2 += block0[6] * xj00 + block0[7] * xj01 + block0[8] * xj02;
                    if (bi != bj0)
                    {
                        int yBaseJ0 = bj0 * 3;
                        y[yBaseJ0 + 0] += block0[0] * xi0 + block0[3] * xi1 + block0[6] * xi2;
                        y[yBaseJ0 + 1] += block0[1] * xi0 + block0[4] * xi1 + block0[7] * xi2;
                        y[yBaseJ0 + 2] += block0[2] * xi0 + block0[5] * xi1 + block0[8] * xi2;
                    }

                    int bj1 = colInd[k + 1];
                    int xBaseJ1 = bj1 * 3;
                    fProxy* block1 = values + (k + 1) * 9;
                    fProxy xj10 = x[xBaseJ1 + 0];
                    fProxy xj11 = x[xBaseJ1 + 1];
                    fProxy xj12 = x[xBaseJ1 + 2];
                    acc1_0 += block1[0] * xj10 + block1[1] * xj11 + block1[2] * xj12;
                    acc1_1 += block1[3] * xj10 + block1[4] * xj11 + block1[5] * xj12;
                    acc1_2 += block1[6] * xj10 + block1[7] * xj11 + block1[8] * xj12;
                    if (bi != bj1)
                    {
                        int yBaseJ1 = bj1 * 3;
                        y[yBaseJ1 + 0] += block1[0] * xi0 + block1[3] * xi1 + block1[6] * xi2;
                        y[yBaseJ1 + 1] += block1[1] * xi0 + block1[4] * xi1 + block1[7] * xi2;
                        y[yBaseJ1 + 2] += block1[2] * xi0 + block1[5] * xi1 + block1[8] * xi2;
                    }
                }

                fProxy s0 = acc0_0 + acc1_0;
                fProxy s1 = acc0_1 + acc1_1;
                fProxy s2 = acc0_2 + acc1_2;
                if (k < rowEnd)
                {
                    int bj = colInd[k];
                    int xBaseJ = bj * 3;
                    fProxy* block = values + k * 9;
                    fProxy xj0 = x[xBaseJ + 0];
                    fProxy xj1 = x[xBaseJ + 1];
                    fProxy xj2 = x[xBaseJ + 2];
                    s0 += block[0] * xj0 + block[1] * xj1 + block[2] * xj2;
                    s1 += block[3] * xj0 + block[4] * xj1 + block[5] * xj2;
                    s2 += block[6] * xj0 + block[7] * xj1 + block[8] * xj2;
                    if (bi != bj)
                    {
                        int yBaseJ = bj * 3;
                        y[yBaseJ + 0] += block[0] * xi0 + block[3] * xi1 + block[6] * xi2;
                        y[yBaseJ + 1] += block[1] * xi0 + block[4] * xi1 + block[7] * xi2;
                        y[yBaseJ + 2] += block[2] * xi0 + block[5] * xi1 + block[8] * xi2;
                    }
                }

                y[yBaseI + 0] += s0;
                y[yBaseI + 1] += s1;
                y[yBaseI + 2] += s2;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecSymB4([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] fProxy* values,
                                           [NoAlias] fProxy* x, [NoAlias] fProxy* y, int blockRows)
        {
            for (int bi = 0; bi < blockRows; bi++)
            {
                int rowStart = rowPtr[bi];
                int rowEnd = rowPtr[bi + 1];
                int yBaseI = bi * 4;
                int xBaseI = bi * 4;
                fProxy xi0 = x[xBaseI + 0];
                fProxy xi1 = x[xBaseI + 1];
                fProxy xi2 = x[xBaseI + 2];
                fProxy xi3 = x[xBaseI + 3];
                int len = rowEnd - rowStart;
                int kPairEnd = rowStart + ((len >> 1) << 1);

                fProxy acc0_0 = default, acc0_1 = default, acc0_2 = default, acc0_3 = default;
                fProxy acc1_0 = default, acc1_1 = default, acc1_2 = default, acc1_3 = default;

                int k = rowStart;
                for (; k < kPairEnd; k += 2)
                {
                    int bj0 = colInd[k];
                    int xBaseJ0 = bj0 * 4;
                    fProxy* block0 = values + k * 16;
                    fProxy xj00 = x[xBaseJ0 + 0];
                    fProxy xj01 = x[xBaseJ0 + 1];
                    fProxy xj02 = x[xBaseJ0 + 2];
                    fProxy xj03 = x[xBaseJ0 + 3];
                    acc0_0 += block0[0]  * xj00 + block0[1]  * xj01 + block0[2]  * xj02 + block0[3]  * xj03;
                    acc0_1 += block0[4]  * xj00 + block0[5]  * xj01 + block0[6]  * xj02 + block0[7]  * xj03;
                    acc0_2 += block0[8]  * xj00 + block0[9]  * xj01 + block0[10] * xj02 + block0[11] * xj03;
                    acc0_3 += block0[12] * xj00 + block0[13] * xj01 + block0[14] * xj02 + block0[15] * xj03;
                    if (bi != bj0)
                    {
                        int yBaseJ0 = bj0 * 4;
                        y[yBaseJ0 + 0] += block0[0] * xi0 + block0[4]  * xi1 + block0[8]  * xi2 + block0[12] * xi3;
                        y[yBaseJ0 + 1] += block0[1] * xi0 + block0[5]  * xi1 + block0[9]  * xi2 + block0[13] * xi3;
                        y[yBaseJ0 + 2] += block0[2] * xi0 + block0[6]  * xi1 + block0[10] * xi2 + block0[14] * xi3;
                        y[yBaseJ0 + 3] += block0[3] * xi0 + block0[7]  * xi1 + block0[11] * xi2 + block0[15] * xi3;
                    }

                    int bj1 = colInd[k + 1];
                    int xBaseJ1 = bj1 * 4;
                    fProxy* block1 = values + (k + 1) * 16;
                    fProxy xj10 = x[xBaseJ1 + 0];
                    fProxy xj11 = x[xBaseJ1 + 1];
                    fProxy xj12 = x[xBaseJ1 + 2];
                    fProxy xj13 = x[xBaseJ1 + 3];
                    acc1_0 += block1[0]  * xj10 + block1[1]  * xj11 + block1[2]  * xj12 + block1[3]  * xj13;
                    acc1_1 += block1[4]  * xj10 + block1[5]  * xj11 + block1[6]  * xj12 + block1[7]  * xj13;
                    acc1_2 += block1[8]  * xj10 + block1[9]  * xj11 + block1[10] * xj12 + block1[11] * xj13;
                    acc1_3 += block1[12] * xj10 + block1[13] * xj11 + block1[14] * xj12 + block1[15] * xj13;
                    if (bi != bj1)
                    {
                        int yBaseJ1 = bj1 * 4;
                        y[yBaseJ1 + 0] += block1[0] * xi0 + block1[4]  * xi1 + block1[8]  * xi2 + block1[12] * xi3;
                        y[yBaseJ1 + 1] += block1[1] * xi0 + block1[5]  * xi1 + block1[9]  * xi2 + block1[13] * xi3;
                        y[yBaseJ1 + 2] += block1[2] * xi0 + block1[6]  * xi1 + block1[10] * xi2 + block1[14] * xi3;
                        y[yBaseJ1 + 3] += block1[3] * xi0 + block1[7]  * xi1 + block1[11] * xi2 + block1[15] * xi3;
                    }
                }

                fProxy s0 = acc0_0 + acc1_0;
                fProxy s1 = acc0_1 + acc1_1;
                fProxy s2 = acc0_2 + acc1_2;
                fProxy s3 = acc0_3 + acc1_3;
                if (k < rowEnd)
                {
                    int bj = colInd[k];
                    int xBaseJ = bj * 4;
                    fProxy* block = values + k * 16;
                    fProxy xj0 = x[xBaseJ + 0];
                    fProxy xj1 = x[xBaseJ + 1];
                    fProxy xj2 = x[xBaseJ + 2];
                    fProxy xj3 = x[xBaseJ + 3];
                    s0 += block[0]  * xj0 + block[1]  * xj1 + block[2]  * xj2 + block[3]  * xj3;
                    s1 += block[4]  * xj0 + block[5]  * xj1 + block[6]  * xj2 + block[7]  * xj3;
                    s2 += block[8]  * xj0 + block[9]  * xj1 + block[10] * xj2 + block[11] * xj3;
                    s3 += block[12] * xj0 + block[13] * xj1 + block[14] * xj2 + block[15] * xj3;
                    if (bi != bj)
                    {
                        int yBaseJ = bj * 4;
                        y[yBaseJ + 0] += block[0] * xi0 + block[4]  * xi1 + block[8]  * xi2 + block[12] * xi3;
                        y[yBaseJ + 1] += block[1] * xi0 + block[5]  * xi1 + block[9]  * xi2 + block[13] * xi3;
                        y[yBaseJ + 2] += block[2] * xi0 + block[6]  * xi1 + block[10] * xi2 + block[14] * xi3;
                        y[yBaseJ + 3] += block[3] * xi0 + block[7]  * xi1 + block[11] * xi2 + block[15] * xi3;
                    }
                }

                y[yBaseI + 0] += s0;
                y[yBaseI + 1] += s1;
                y[yBaseI + 2] += s2;
                y[yBaseI + 3] += s3;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsrMatVecSymB6([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] fProxy* values,
                                           [NoAlias] fProxy* x, [NoAlias] fProxy* y, int blockRows)
        {
            for (int bi = 0; bi < blockRows; bi++)
            {
                int rowStart = rowPtr[bi];
                int rowEnd = rowPtr[bi + 1];
                int yBaseI = bi * 6;
                int xBaseI = bi * 6;
                fProxy xi0 = x[xBaseI + 0];
                fProxy xi1 = x[xBaseI + 1];
                fProxy xi2 = x[xBaseI + 2];
                fProxy xi3 = x[xBaseI + 3];
                fProxy xi4 = x[xBaseI + 4];
                fProxy xi5 = x[xBaseI + 5];
                int len = rowEnd - rowStart;
                int kPairEnd = rowStart + ((len >> 1) << 1);

                fProxy acc0_0 = default, acc0_1 = default, acc0_2 = default, acc0_3 = default, acc0_4 = default, acc0_5 = default;
                fProxy acc1_0 = default, acc1_1 = default, acc1_2 = default, acc1_3 = default, acc1_4 = default, acc1_5 = default;

                int k = rowStart;
                for (; k < kPairEnd; k += 2)
                {
                    int bj0 = colInd[k];
                    int xBaseJ0 = bj0 * 6;
                    fProxy* block0 = values + k * 36;
                    fProxy xj00 = x[xBaseJ0 + 0];
                    fProxy xj01 = x[xBaseJ0 + 1];
                    fProxy xj02 = x[xBaseJ0 + 2];
                    fProxy xj03 = x[xBaseJ0 + 3];
                    fProxy xj04 = x[xBaseJ0 + 4];
                    fProxy xj05 = x[xBaseJ0 + 5];
                    acc0_0 += block0[0]  * xj00 + block0[1]  * xj01 + block0[2]  * xj02 + block0[3]  * xj03 + block0[4]  * xj04 + block0[5]  * xj05;
                    acc0_1 += block0[6]  * xj00 + block0[7]  * xj01 + block0[8]  * xj02 + block0[9]  * xj03 + block0[10] * xj04 + block0[11] * xj05;
                    acc0_2 += block0[12] * xj00 + block0[13] * xj01 + block0[14] * xj02 + block0[15] * xj03 + block0[16] * xj04 + block0[17] * xj05;
                    acc0_3 += block0[18] * xj00 + block0[19] * xj01 + block0[20] * xj02 + block0[21] * xj03 + block0[22] * xj04 + block0[23] * xj05;
                    acc0_4 += block0[24] * xj00 + block0[25] * xj01 + block0[26] * xj02 + block0[27] * xj03 + block0[28] * xj04 + block0[29] * xj05;
                    acc0_5 += block0[30] * xj00 + block0[31] * xj01 + block0[32] * xj02 + block0[33] * xj03 + block0[34] * xj04 + block0[35] * xj05;
                    if (bi != bj0)
                    {
                        int yBaseJ0 = bj0 * 6;
                        y[yBaseJ0 + 0] += block0[0] * xi0 + block0[6]  * xi1 + block0[12] * xi2 + block0[18] * xi3 + block0[24] * xi4 + block0[30] * xi5;
                        y[yBaseJ0 + 1] += block0[1] * xi0 + block0[7]  * xi1 + block0[13] * xi2 + block0[19] * xi3 + block0[25] * xi4 + block0[31] * xi5;
                        y[yBaseJ0 + 2] += block0[2] * xi0 + block0[8]  * xi1 + block0[14] * xi2 + block0[20] * xi3 + block0[26] * xi4 + block0[32] * xi5;
                        y[yBaseJ0 + 3] += block0[3] * xi0 + block0[9]  * xi1 + block0[15] * xi2 + block0[21] * xi3 + block0[27] * xi4 + block0[33] * xi5;
                        y[yBaseJ0 + 4] += block0[4] * xi0 + block0[10] * xi1 + block0[16] * xi2 + block0[22] * xi3 + block0[28] * xi4 + block0[34] * xi5;
                        y[yBaseJ0 + 5] += block0[5] * xi0 + block0[11] * xi1 + block0[17] * xi2 + block0[23] * xi3 + block0[29] * xi4 + block0[35] * xi5;
                    }

                    int bj1 = colInd[k + 1];
                    int xBaseJ1 = bj1 * 6;
                    fProxy* block1 = values + (k + 1) * 36;
                    fProxy xj10 = x[xBaseJ1 + 0];
                    fProxy xj11 = x[xBaseJ1 + 1];
                    fProxy xj12 = x[xBaseJ1 + 2];
                    fProxy xj13 = x[xBaseJ1 + 3];
                    fProxy xj14 = x[xBaseJ1 + 4];
                    fProxy xj15 = x[xBaseJ1 + 5];
                    acc1_0 += block1[0]  * xj10 + block1[1]  * xj11 + block1[2]  * xj12 + block1[3]  * xj13 + block1[4]  * xj14 + block1[5]  * xj15;
                    acc1_1 += block1[6]  * xj10 + block1[7]  * xj11 + block1[8]  * xj12 + block1[9]  * xj13 + block1[10] * xj14 + block1[11] * xj15;
                    acc1_2 += block1[12] * xj10 + block1[13] * xj11 + block1[14] * xj12 + block1[15] * xj13 + block1[16] * xj14 + block1[17] * xj15;
                    acc1_3 += block1[18] * xj10 + block1[19] * xj11 + block1[20] * xj12 + block1[21] * xj13 + block1[22] * xj14 + block1[23] * xj15;
                    acc1_4 += block1[24] * xj10 + block1[25] * xj11 + block1[26] * xj12 + block1[27] * xj13 + block1[28] * xj14 + block1[29] * xj15;
                    acc1_5 += block1[30] * xj10 + block1[31] * xj11 + block1[32] * xj12 + block1[33] * xj13 + block1[34] * xj14 + block1[35] * xj15;
                    if (bi != bj1)
                    {
                        int yBaseJ1 = bj1 * 6;
                        y[yBaseJ1 + 0] += block1[0] * xi0 + block1[6]  * xi1 + block1[12] * xi2 + block1[18] * xi3 + block1[24] * xi4 + block1[30] * xi5;
                        y[yBaseJ1 + 1] += block1[1] * xi0 + block1[7]  * xi1 + block1[13] * xi2 + block1[19] * xi3 + block1[25] * xi4 + block1[31] * xi5;
                        y[yBaseJ1 + 2] += block1[2] * xi0 + block1[8]  * xi1 + block1[14] * xi2 + block1[20] * xi3 + block1[26] * xi4 + block1[32] * xi5;
                        y[yBaseJ1 + 3] += block1[3] * xi0 + block1[9]  * xi1 + block1[15] * xi2 + block1[21] * xi3 + block1[27] * xi4 + block1[33] * xi5;
                        y[yBaseJ1 + 4] += block1[4] * xi0 + block1[10] * xi1 + block1[16] * xi2 + block1[22] * xi3 + block1[28] * xi4 + block1[34] * xi5;
                        y[yBaseJ1 + 5] += block1[5] * xi0 + block1[11] * xi1 + block1[17] * xi2 + block1[23] * xi3 + block1[29] * xi4 + block1[35] * xi5;
                    }
                }

                fProxy s0 = acc0_0 + acc1_0;
                fProxy s1 = acc0_1 + acc1_1;
                fProxy s2 = acc0_2 + acc1_2;
                fProxy s3 = acc0_3 + acc1_3;
                fProxy s4 = acc0_4 + acc1_4;
                fProxy s5 = acc0_5 + acc1_5;
                if (k < rowEnd)
                {
                    int bj = colInd[k];
                    int xBaseJ = bj * 6;
                    fProxy* block = values + k * 36;
                    fProxy xj0 = x[xBaseJ + 0];
                    fProxy xj1 = x[xBaseJ + 1];
                    fProxy xj2 = x[xBaseJ + 2];
                    fProxy xj3 = x[xBaseJ + 3];
                    fProxy xj4 = x[xBaseJ + 4];
                    fProxy xj5 = x[xBaseJ + 5];
                    s0 += block[0]  * xj0 + block[1]  * xj1 + block[2]  * xj2 + block[3]  * xj3 + block[4]  * xj4 + block[5]  * xj5;
                    s1 += block[6]  * xj0 + block[7]  * xj1 + block[8]  * xj2 + block[9]  * xj3 + block[10] * xj4 + block[11] * xj5;
                    s2 += block[12] * xj0 + block[13] * xj1 + block[14] * xj2 + block[15] * xj3 + block[16] * xj4 + block[17] * xj5;
                    s3 += block[18] * xj0 + block[19] * xj1 + block[20] * xj2 + block[21] * xj3 + block[22] * xj4 + block[23] * xj5;
                    s4 += block[24] * xj0 + block[25] * xj1 + block[26] * xj2 + block[27] * xj3 + block[28] * xj4 + block[29] * xj5;
                    s5 += block[30] * xj0 + block[31] * xj1 + block[32] * xj2 + block[33] * xj3 + block[34] * xj4 + block[35] * xj5;
                    if (bi != bj)
                    {
                        int yBaseJ = bj * 6;
                        y[yBaseJ + 0] += block[0] * xi0 + block[6]  * xi1 + block[12] * xi2 + block[18] * xi3 + block[24] * xi4 + block[30] * xi5;
                        y[yBaseJ + 1] += block[1] * xi0 + block[7]  * xi1 + block[13] * xi2 + block[19] * xi3 + block[25] * xi4 + block[31] * xi5;
                        y[yBaseJ + 2] += block[2] * xi0 + block[8]  * xi1 + block[14] * xi2 + block[20] * xi3 + block[26] * xi4 + block[32] * xi5;
                        y[yBaseJ + 3] += block[3] * xi0 + block[9]  * xi1 + block[15] * xi2 + block[21] * xi3 + block[27] * xi4 + block[33] * xi5;
                        y[yBaseJ + 4] += block[4] * xi0 + block[10] * xi1 + block[16] * xi2 + block[22] * xi3 + block[28] * xi4 + block[34] * xi5;
                        y[yBaseJ + 5] += block[5] * xi0 + block[11] * xi1 + block[17] * xi2 + block[23] * xi3 + block[29] * xi4 + block[35] * xi5;
                    }
                }

                y[yBaseI + 0] += s0;
                y[yBaseI + 1] += s1;
                y[yBaseI + 2] += s2;
                y[yBaseI + 3] += s3;
                y[yBaseI + 4] += s4;
                y[yBaseI + 5] += s5;
            }
        }

        // Krylov R2's ApplyDot (docs/draft-spec-krylov-optimization.md) does NOT have a fused
        // kernel family here. A "Dot" variant (fold dot(x,y) into the same pass as y=A*x, tried
        // for the full-storage square-block kernels above) was A/B'd against simply composing
        // (spMV then Blas.dot(x,y)) at the b=1 stencil section of LargeSparseBenchmark and lost
        // by a wide, reproducible margin (~45% SLOWER at N=5120/float) -- see BSR.spMVDot's doc
        // comment (SparseOP.fProxy.cs) for the root cause and the full writeup. Composing wins
        // because it reuses the already 2x-fProxy4-SIMD-tuned vecDot kernel instead of a bespoke
        // scalar cross-row fold. Kept out per the spec's own instruction: measure, and if it
        // doesn't win, don't ship it.

        // =====================================================================================
        // Krylov R2, fProxyBlockJacobi.Apply specialization (docs/draft-spec-krylov-optimization.md,
        // R2): z = DInv * r, one dense b x b matvec per block-row (DInv holds ONE explicit inverse
        // block per block-row, no stored-block loop like the spMV kernels above -- there is nothing
        // to accumulator-split here, each output is already a single BR-term sum). Mirrors the
        // spMV unroll structure (b hardcoded as a literal so Burst can register-allocate the whole
        // block-multiply) purely for the same reason bsrMatVecB{b} exists: BR is a runtime field in
        // the general loop, so Burst cannot unroll/vectorize it. Left-to-right term order matches
        // the general loop's `sum = 0; sum += ...` fold exactly -> BIT-IDENTICAL to the general
        // fallback, not just rounding-equivalent. Dispatch lives in fProxyBlockJacobi.Apply.
        // =====================================================================================

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void blockJacobiApplyB1([NoAlias] fProxy* dp, [NoAlias] fProxy* rp, [NoAlias] fProxy* zp, int blockRows)
        {
            for (int i = 0; i < blockRows; i++)
                zp[i] = dp[i] * rp[i];
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void blockJacobiApplyB2([NoAlias] fProxy* dp, [NoAlias] fProxy* rp, [NoAlias] fProxy* zp, int blockRows)
        {
            for (int i = 0; i < blockRows; i++)
            {
                int rowBase = i * 2;
                int blockOff = i * 4;
                fProxy r0 = rp[rowBase + 0];
                fProxy r1 = rp[rowBase + 1];
                zp[rowBase + 0] = dp[blockOff + 0] * r0 + dp[blockOff + 1] * r1;
                zp[rowBase + 1] = dp[blockOff + 2] * r0 + dp[blockOff + 3] * r1;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void blockJacobiApplyB3([NoAlias] fProxy* dp, [NoAlias] fProxy* rp, [NoAlias] fProxy* zp, int blockRows)
        {
            for (int i = 0; i < blockRows; i++)
            {
                int rowBase = i * 3;
                int blockOff = i * 9;
                fProxy r0 = rp[rowBase + 0];
                fProxy r1 = rp[rowBase + 1];
                fProxy r2 = rp[rowBase + 2];
                zp[rowBase + 0] = dp[blockOff + 0] * r0 + dp[blockOff + 1] * r1 + dp[blockOff + 2] * r2;
                zp[rowBase + 1] = dp[blockOff + 3] * r0 + dp[blockOff + 4] * r1 + dp[blockOff + 5] * r2;
                zp[rowBase + 2] = dp[blockOff + 6] * r0 + dp[blockOff + 7] * r1 + dp[blockOff + 8] * r2;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void blockJacobiApplyB4([NoAlias] fProxy* dp, [NoAlias] fProxy* rp, [NoAlias] fProxy* zp, int blockRows)
        {
            for (int i = 0; i < blockRows; i++)
            {
                int rowBase = i * 4;
                int blockOff = i * 16;
                fProxy r0 = rp[rowBase + 0];
                fProxy r1 = rp[rowBase + 1];
                fProxy r2 = rp[rowBase + 2];
                fProxy r3 = rp[rowBase + 3];
                zp[rowBase + 0] = dp[blockOff + 0]  * r0 + dp[blockOff + 1]  * r1 + dp[blockOff + 2]  * r2 + dp[blockOff + 3]  * r3;
                zp[rowBase + 1] = dp[blockOff + 4]  * r0 + dp[blockOff + 5]  * r1 + dp[blockOff + 6]  * r2 + dp[blockOff + 7]  * r3;
                zp[rowBase + 2] = dp[blockOff + 8]  * r0 + dp[blockOff + 9]  * r1 + dp[blockOff + 10] * r2 + dp[blockOff + 11] * r3;
                zp[rowBase + 3] = dp[blockOff + 12] * r0 + dp[blockOff + 13] * r1 + dp[blockOff + 14] * r2 + dp[blockOff + 15] * r3;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void blockJacobiApplyB6([NoAlias] fProxy* dp, [NoAlias] fProxy* rp, [NoAlias] fProxy* zp, int blockRows)
        {
            for (int i = 0; i < blockRows; i++)
            {
                int rowBase = i * 6;
                int blockOff = i * 36;
                fProxy r0 = rp[rowBase + 0];
                fProxy r1 = rp[rowBase + 1];
                fProxy r2 = rp[rowBase + 2];
                fProxy r3 = rp[rowBase + 3];
                fProxy r4 = rp[rowBase + 4];
                fProxy r5 = rp[rowBase + 5];
                zp[rowBase + 0] = dp[blockOff + 0]  * r0 + dp[blockOff + 1]  * r1 + dp[blockOff + 2]  * r2 + dp[blockOff + 3]  * r3 + dp[blockOff + 4]  * r4 + dp[blockOff + 5]  * r5;
                zp[rowBase + 1] = dp[blockOff + 6]  * r0 + dp[blockOff + 7]  * r1 + dp[blockOff + 8]  * r2 + dp[blockOff + 9]  * r3 + dp[blockOff + 10] * r4 + dp[blockOff + 11] * r5;
                zp[rowBase + 2] = dp[blockOff + 12] * r0 + dp[blockOff + 13] * r1 + dp[blockOff + 14] * r2 + dp[blockOff + 15] * r3 + dp[blockOff + 16] * r4 + dp[blockOff + 17] * r5;
                zp[rowBase + 3] = dp[blockOff + 18] * r0 + dp[blockOff + 19] * r1 + dp[blockOff + 20] * r2 + dp[blockOff + 21] * r3 + dp[blockOff + 22] * r4 + dp[blockOff + 23] * r5;
                zp[rowBase + 4] = dp[blockOff + 24] * r0 + dp[blockOff + 25] * r1 + dp[blockOff + 26] * r2 + dp[blockOff + 27] * r3 + dp[blockOff + 28] * r4 + dp[blockOff + 29] * r5;
                zp[rowBase + 5] = dp[blockOff + 30] * r0 + dp[blockOff + 31] * r1 + dp[blockOff + 32] * r2 + dp[blockOff + 33] * r3 + dp[blockOff + 34] * r4 + dp[blockOff + 35] * r5;
            }
        }
    }
}
