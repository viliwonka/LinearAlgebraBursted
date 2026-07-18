using System;
using LinearAlgebra;
using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Row-oriented sparse approximate inverse (SPAI) over a square BSR: builds M ≈ A⁻¹ directly
    /// (no factorization) by minimizing ‖M A − I‖_F row by row over a STATIC pattern (default:
    /// A's own row pattern) -- the nonsymmetric sibling of <see cref="fProxyIC0"/>/<see
    /// cref="fProxyFSAI"/>'s counterpart <see cref="fProxyILU0"/>, intended for
    /// <see cref="LinearAlgebra.Krylov"/>.pbiCGStab. M is NOT symmetric even for symmetric A --
    /// it is NOT a valid CG/MINRES preconditioner.
    ///
    /// Each row is an independent small least-squares problem (row i's support J_i, shadow column
    /// pattern I_i, dense normal-equations solve via <see cref="CHO"/>) -- no row-to-row
    /// dependency, no global breakdown cascade. A rank-deficient local system retries with an
    /// escalating diagonal (Tikhonov) shift, recorded in <see cref="Shift"/>; throws if the largest
    /// shift still fails. Apply is a single BSR spMV.
    ///
    /// Symmetric-storage A pays a one-time mirror-to-full copy (SPAI needs full rows), same as
    /// ILU0. A must store every diagonal block. Arena-composed -- no record table of its own, no
    /// Dispose().
    /// </summary>
    public readonly struct fProxySPAI : IfProxyPreconditioner
    {
        /// <summary>M's storage: A's own full block pattern (MVP default).</summary>
        public readonly fProxyBSR M;

        /// <summary>Worst per-row Tikhonov shift that made a local least-squares solve succeed; 0 if every row was clean.</summary>
        public readonly fProxy Shift;

        public int Rows => M.M_Rows;

        /// <summary>
        /// Builds SPAI with <see cref="SaiOptions.Default"/>. Throws if A is not square
        /// (BlockRows==BlockCols, BR==BC), if a diagonal block is absent, or if a row's local
        /// least-squares solve still breaks down at the largest Tikhonov shift. Use the out-info
        /// overload for a non-throwing build.
        /// </summary>
        public fProxySPAI(in fProxyBSR a, ref Arena arena)
        {
            this = new fProxySPAI(in a, ref arena, out PreconditionerInfo info);
            if (!info.Solved)
                throw new ArgumentException("fProxySPAI: a row's local least-squares solve broke down at every Tikhonov shift");
        }

        /// <summary>Non-throwing build with <see cref="SaiOptions.Default"/>; see the out-info +
        /// SaiOptions overload for the full contract.</summary>
        public fProxySPAI(in fProxyBSR a, ref Arena arena, out PreconditionerInfo info)
        {
            this = new fProxySPAI(in a, ref arena, SaiOptions.Default, out info);
        }

        /// <summary>Builds SPAI with the given <see cref="SaiOptions"/>. Same throw contract as the
        /// options-less overload.</summary>
        public fProxySPAI(in fProxyBSR a, ref Arena arena, in SaiOptions opts)
        {
            this = new fProxySPAI(in a, ref arena, in opts, out PreconditionerInfo info);
            if (!info.Solved)
                throw new ArgumentException("fProxySPAI: a row's local least-squares solve broke down at every Tikhonov shift");
        }

        /// <summary>
        /// Non-throwing build: info.status is Success, or Singular when some row's local
        /// least-squares solve broke down at every Tikhonov shift (the preconditioner is then
        /// unusable -- do not Apply); info also carries the worst rescuing shift and the worst
        /// attempts consumed by any single row. Caller-contract violations (non-square, missing
        /// diagonal block, opts.patternPower != 1) still throw.
        /// </summary>
        public fProxySPAI(in fProxyBSR a, ref Arena arena, in SaiOptions opts, out PreconditionerInfo info)
        {
            if (a.BlockRows != a.BlockCols || a.BR != a.BC)
                throw new ArgumentException("fProxySPAI: A must be square (BlockRows==BlockCols, BR==BC)");
            if (opts.patternPower != 1)
                throw new ArgumentException("fProxySPAI: opts.patternPower must be 1 (pattern(A^2) is not implemented yet)");

            var A = a.Symmetric ? arena.fProxyBSRMirrorToFull(in a) : a;

            int nb = A.BlockRows;
            int BR = A.BR;
            int blockLen = BR * BR;

            // Every diagonal block must exist (required in J_i, and it locates i within I_i).
            // Shift scale mirrors IC0/ILU0's: the largest |diagonal entry| of A.
            fProxy diagMax = 0;
            for (int i = 0; i < nb; i++)
            {
                int s = A.RowPtr[i], e = A.RowPtr[i + 1];
                bool hasDiag = false;
                for (int k = s; k < e; k++)
                {
                    if (A.ColInd[k] != i) continue;
                    hasDiag = true;
                    int off = k * blockLen;
                    for (int r = 0; r < BR; r++)
                    {
                        fProxy av = math.abs(A.Values[off + r * BR + r]);
                        if (av > diagMax) diagMax = av;
                    }
                    break;
                }
                if (!hasDiag)
                    throw new ArgumentException("fProxySPAI: missing diagonal block in A");
            }
            if (diagMax <= (fProxy)0) diagMax = (fProxy)1;

            // M's pattern = A's own full pattern (MVP default).
            var Mm = arena.fProxyBSR(nb, nb, BR, BR, A.Nnzb, true);
            var mRowPtr = Mm.RowPtr; var mColInd = Mm.ColInd; var mValues = Mm.Values;
            {
                for (int i = 0; i <= nb; i++) mRowPtr[i] = A.RowPtr[i];
                for (int k = 0; k < A.Nnzb; k++) mColInd[k] = A.ColInd[k];
            }

            fProxy worstShift = 0;
            int worstAttempts = 1;
            bool ok = true;

            // Widest possible shadow pattern I_i is every block-column of A (nb) -- one fixed-size
            // Temp buffer reused (overwritten) across every row's union-merge below.
            var shadow = new NativeArray<int>(nb, Allocator.Temp, NativeArrayOptions.UninitializedMemory);

            for (int i = 0; i < nb; i++)
            {
                int jS = A.RowPtr[i], jE = A.RowPtr[i + 1];
                int m = jE - jS;   // |J_i|

                // Shadow pattern I_i: sorted, deduplicated union of the row patterns of every
                // j in J_i. Ascending insertion into the running sorted prefix shadow[0:shadowCount)
                // -- deterministic regardless of insertion order (converges to the unique sorted set).
                int shadowCount = 0;
                for (int aI = 0; aI < m; aI++)
                {
                    int j = A.ColInd[jS + aI];
                    int rs = A.RowPtr[j], re = A.RowPtr[j + 1];
                    for (int k = rs; k < re; k++)
                    {
                        int col = A.ColInd[k];
                        int p = shadowCount - 1;
                        while (p >= 0 && shadow[p] > col) p--;
                        bool dup = p >= 0 && shadow[p] == col;
                        if (!dup)
                        {
                            for (int q = shadowCount; q > p + 1; q--) shadow[q] = shadow[q - 1];
                            shadow[p + 1] = col;
                            shadowCount++;
                        }
                    }
                }
                int kCount = shadowCount;   // |I_i|
                int nJ = m * BR, nI = kCount * BR;

                int iLocal = -1;
                for (int p = 0; p < kCount; p++) if (shadow[p] == i) { iLocal = p; break; }
                // i in J_i (diagonal required) -> row i's own pattern was unioned in -> iLocal >= 0.

                var Ahat = new fProxyMxN(nJ, nI, Allocator.Temp, true);
                for (int aI = 0; aI < m; aI++)
                {
                    int ga = A.ColInd[jS + aI];
                    for (int bI = 0; bI < kCount; bI++)
                    {
                        int gb = shadow[bI];
                        fProxyFSAI.GatherBlockInto(in A, ga, gb, ref Ahat, aI * BR, bI * BR, (fProxy)0);
                    }
                }

                var Nbase = new fProxyMxN(nJ, nJ, Allocator.Temp, true);
                Blas.dotSymT(in Ahat, in Ahat, ref Nbase);   // N = A_hat . A_hat^T (SPD by construction)

                // RHS base = A_hat . e_shadow(iLocal) = A_hat's own iLocal-th BR-column block.
                var RHSbase = new fProxyMxN(nJ, BR, Allocator.Temp, true);
                for (int r = 0; r < nJ; r++)
                    for (int c = 0; c < BR; c++)
                        RHSbase[r, c] = Ahat[r, iLocal * BR + c];

                var Nwork = new fProxyMxN(nJ, nJ, Allocator.Temp, true);
                var RHSwork = new fProxyMxN(nJ, BR, Allocator.Temp, true);

                fProxy shift = 0;
                bool rowOk = false;
                int rowAttempts = 0;

                for (int attempt = 0; attempt < 6; attempt++)
                {
                    rowAttempts = attempt + 1;

                    Nwork.Data.CopyFrom(Nbase.Data);
                    if (shift != (fProxy)0)
                        for (int r = 0; r < nJ; r++) Nwork[r, r] += shift;
                    RHSwork.Data.CopyFrom(RHSbase.Data);

                    var chInfo = CHO.decompInPlace(ref Nwork);
                    if (chInfo.Solved)
                    {
                        CHO.decompSolve(ref Nwork, ref RHSwork);   // RHSwork := m (nJ x BR) == this row of M^T restricted to J_i

                        for (int aI = 0; aI < m; aI++)
                        {
                            int dstOff = (jS + aI) * blockLen;   // M's row-i slots share A's RowPtr layout 1:1
                            for (int q = 0; q < BR; q++)
                                for (int p = 0; p < BR; p++)
                                    mValues[dstOff + q * BR + p] = RHSwork[aI * BR + p, q];
                        }

                        rowOk = true;
                        break;
                    }

                    shift = shift == (fProxy)0 ? (fProxy)1e-3 * diagMax : shift * (fProxy)10;
                }

                RHSwork.Dispose();
                Nwork.Dispose();
                RHSbase.Dispose();
                Nbase.Dispose();
                Ahat.Dispose();

                if (rowAttempts > worstAttempts) worstAttempts = rowAttempts;
                if (shift > worstShift) worstShift = shift;
                if (!rowOk) { ok = false; break; }
            }

            shadow.Dispose();

            M = Mm;
            Shift = worstShift;

            info = new PreconditionerInfo
            {
                status = ok ? DirectSolveStatus.Success : DirectSolveStatus.Singular,
                shift = (double)worstShift,
                attempts = worstAttempts,
            };
        }

        /// <summary>z = M r: one BSR spMV. z must not alias r.</summary>
        public unsafe void Apply(in fProxyN r, ref fProxyN z)
        {
            int n = Rows;
            if (r.N != n)
                throw new ArgumentException("fProxySPAI.Apply: r.N must equal Rows");
            if (z.N != n)
                throw new ArgumentException("fProxySPAI.Apply: z.N must equal Rows");
            if (z.Data.Ptr == r.Data.Ptr)
                throw new ArgumentException("fProxySPAI.Apply: z must not alias r");

            BSR.spMV(in M, in r, ref z);
        }
    }
}
