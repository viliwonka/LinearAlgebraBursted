using System.Runtime.CompilerServices;
using Unity.Burst;

namespace LinearAlgebra.Internal
{
    public static unsafe partial class Unsafe_OP
    {
        // General BR x BC block-CSR matvec: y = A * x. y must already be zeroed by the caller
        // (accumulates into y, like matVecDot). Correctness-first fallback -- every block size
        // routes through this same general loop today.
        // TODO Phase-7 perf: unrolled register-tile specializations for BR,BC in {1,2,3,4,6}
        // (switch-dispatch on (BR,BC) here; reuse the register-tile GEMM lessons).
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsmMatVec([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                      [NoAlias] float* x, [NoAlias] float* y,
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
                    float* block = values + k * blockLen;

                    for (int r = 0; r < BR; r++)
                    {
                        float sum = 0;
                        int rowOff = r * BC;
                        for (int c = 0; c < BC; c++)
                            sum += block[rowOff + c] * x[xBase + c];
                        y[yBase + r] += sum;
                    }
                }
            }
        }

        // General BR x BC block-CSR transpose matvec: y = A^T * x. y must already be zeroed by
        // the caller. Walks the SAME row-major block-CSR storage as bsmMatVec (no separate
        // transposed copy) -- each stored block K contributes y[j-block] += K^T * x[i-block].
        // Safe single-threaded (no scatter race): every (br,k) pair touches a distinct block,
        // and different blocks in the same block-row write to DIFFERENT y[j-block] ranges only
        // when ColInd is duplicate-free per row, which ToBSM guarantees.
        // TODO Phase-7 perf: unrolled register-tile specializations for BR,BC in {1,2,3,4,6}.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void bsmMatVecT([NoAlias] int* rowPtr, [NoAlias] int* colInd, [NoAlias] float* values,
                                       [NoAlias] float* x, [NoAlias] float* y,
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
                    float* block = values + k * blockLen;

                    for (int c = 0; c < BC; c++)
                    {
                        float sum = 0;
                        for (int r = 0; r < BR; r++)
                            sum += block[r * BC + c] * x[xBase + r];
                        y[yBase + c] += sum;
                    }
                }
            }
        }
    }
}
