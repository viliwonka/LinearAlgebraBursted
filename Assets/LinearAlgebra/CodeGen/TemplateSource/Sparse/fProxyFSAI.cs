using System;
using LinearAlgebra;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Factored sparse approximate inverse (FSAI, Kolotilina-Yeremin/Kaporin) over a square SPD
    /// BSR: builds a block-lower-triangular G on a STATIC pattern S (default: A's lower block
    /// pattern, diagonal included -- same as <see cref="fProxyIC0"/>'s) such that M = Gᵀ G ≈ A⁻¹.
    /// Each block-row of G is computed by an INDEPENDENT dense local SPD solve (no row-to-row
    /// factorization dependency, unlike IC0) -- setup has no global breakdown cascade, and Apply
    /// is two plain BSR spMVs (no triangular solve). M is symmetric by construction and SPD
    /// whenever G is nonsingular (guaranteed by the Cholesky-scaled construction below), so FSAI
    /// is a valid <see cref="LinearAlgebra.Krylov"/>.cg/minres preconditioner.
    ///
    /// A local system that is not SPD retries with an escalating diagonal shift on JUST that row
    /// (rows are independent, so no global refactorization is needed); the worst shift used across
    /// all rows is recorded in <see cref="Shift"/> (0 = every row clean). Throws if a row still
    /// breaks down at the largest shift.
    ///
    /// Symmetric-storage A needs no mirror (consumed zero-copy, same as IC0); full-storage A is
    /// accepted too (only its lower blocks feed the pattern). A must store every diagonal block.
    /// Arena-composed -- no record table of its own, no Dispose().
    /// </summary>
    public readonly struct fProxyFSAI : IfProxyPreconditioner
    {
        /// <summary>Block-lower-triangular factor G over the static pattern S.</summary>
        public readonly fProxyBSR G;

        /// <summary>Gᵀ, materialized once so Apply is two forward spMVs.</summary>
        public readonly fProxyBSR Gt;

        /// <summary>Owned apply scratch, length Rows: holds y = G r during Apply.</summary>
        public readonly fProxyN Scratch;

        /// <summary>Worst per-row diagonal shift that made a local solve succeed; 0 if every row was clean.</summary>
        public readonly fProxy Shift;

        public int Rows => G.M_Rows;

        /// <summary>
        /// Builds FSAI with <see cref="SaiOptions.Default"/>. Throws if A is not square
        /// (BlockRows==BlockCols, BR==BC), if a diagonal block is absent, or if a row's local
        /// solve still breaks down at the largest diagonal shift. Use the out-info overload for a
        /// non-throwing build.
        /// </summary>
        public fProxyFSAI(in fProxyBSR a, ref Arena arena)
        {
            this = new fProxyFSAI(in a, ref arena, out PreconditionerInfo info);
            if (!info.Solved)
                throw new ArgumentException("fProxyFSAI: a row's local solve broke down at every diagonal shift — is A symmetric positive definite?");
        }

        /// <summary>Non-throwing build with <see cref="SaiOptions.Default"/>; see the out-info +
        /// SaiOptions overload for the full contract.</summary>
        public fProxyFSAI(in fProxyBSR a, ref Arena arena, out PreconditionerInfo info)
        {
            this = new fProxyFSAI(in a, ref arena, SaiOptions.Default, out info);
        }

        /// <summary>Builds FSAI with the given <see cref="SaiOptions"/>. Same throw contract as the
        /// options-less overload.</summary>
        public fProxyFSAI(in fProxyBSR a, ref Arena arena, in SaiOptions opts)
        {
            this = new fProxyFSAI(in a, ref arena, in opts, out PreconditionerInfo info);
            if (!info.Solved)
                throw new ArgumentException("fProxyFSAI: a row's local solve broke down at every diagonal shift — is A symmetric positive definite?");
        }

        /// <summary>
        /// Non-throwing build: info.status is Success, or NotPositiveDefinite when some row's local
        /// solve broke down at every diagonal shift (the preconditioner is then unusable -- do not
        /// Apply); info also carries the worst rescuing shift and the worst attempts consumed by
        /// any single row. Caller-contract violations (non-square, missing diagonal block,
        /// opts.patternPower != 1) still throw.
        /// </summary>
        public fProxyFSAI(in fProxyBSR a, ref Arena arena, in SaiOptions opts, out PreconditionerInfo info)
        {
            if (a.BlockRows != a.BlockCols || a.BR != a.BC)
                throw new ArgumentException("fProxyFSAI: A must be square (BlockRows==BlockCols, BR==BC)");
            if (opts.patternPower != 1)
                throw new ArgumentException("fProxyFSAI: opts.patternPower must be 1 (pattern(A^2) is not implemented yet)");

            // No mirror needed either way: full-storage A's lower blocks are read directly below,
            // and symmetric-storage A already stores exactly the lower-block pattern (diagonal
            // included) that S defaults to.
            var A = a;

            int nb = A.BlockRows;
            int BR = A.BR;
            int blockLen = BR * BR;
            fProxy dropTol = (fProxy)opts.dropTol;

            // Diagonal-block Frobenius norms (dropTol filter) and the largest |diagonal entry|
            // (shift scale), computed once; also validates every diagonal block is present.
            var diagNormF = new NativeArray<fProxy>(nb, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
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
                    fProxy normSq = 0;
                    for (int t = 0; t < blockLen; t++)
                    {
                        fProxy v = A.Values[off + t];
                        normSq += v * v;
                    }
                    diagNormF[i] = math.sqrt(normSq);
                    for (int r = 0; r < BR; r++)
                    {
                        fProxy av = math.abs(A.Values[off + r * BR + r]);
                        if (av > diagMax) diagMax = av;
                    }
                    break;
                }
                if (!hasDiag)
                {
                    diagNormF.Dispose();
                    throw new ArgumentException("fProxyFSAI: missing diagonal block in A");
                }
            }
            if (diagMax <= (fProxy)0) diagMax = (fProxy)1;

            // ---- build S: A's blocks with col <= row, diagonal required, optional dropTol filter ----
            int nnzbG = 0;
            for (int i = 0; i < nb; i++)
            {
                int s = A.RowPtr[i], e = A.RowPtr[i + 1];
                for (int k = s; k < e; k++)
                {
                    int col = A.ColInd[k];
                    if (col > i) break;
                    if (col != i && dropTol > (fProxy)0 && BelowDropTol(in A, k, blockLen, diagNormF[i], diagNormF[col], dropTol))
                        continue;
                    nnzbG++;
                }
            }

            var Gm = arena.fProxyBSR(nb, nb, BR, BR, nnzbG, true);
            var gRowPtr = Gm.RowPtr; var gColInd = Gm.ColInd; var gValues = Gm.Values;
            {
                int outIdx = 0;
                for (int i = 0; i < nb; i++)
                {
                    gRowPtr[i] = outIdx;
                    int s = A.RowPtr[i], e = A.RowPtr[i + 1];
                    for (int k = s; k < e; k++)
                    {
                        int col = A.ColInd[k];
                        if (col > i) break;
                        if (col != i && dropTol > (fProxy)0 && BelowDropTol(in A, k, blockLen, diagNormF[i], diagNormF[col], dropTol))
                            continue;
                        gColInd[outIdx] = col;
                        outIdx++;
                    }
                }
                gRowPtr[nb] = outIdx;
            }
            diagNormF.Dispose();

            // ---- per block-row: independent dense SPD solve, escalating a LOCAL diagonal shift ----
            fProxy worstShift = 0;
            int worstAttempts = 1;
            bool ok = true;

            for (int i = 0; i < nb; i++)
            {
                int rowStart = gRowPtr[i], rowEnd = gRowPtr[i + 1];
                int m = rowEnd - rowStart;   // |J_i|
                int n = m * BR;

                var Ahat = new fProxyMxN(n, n, Allocator.Temp, true);
                var X = new fProxyMxN(n, BR, Allocator.Temp, true);

                fProxy shift = 0;
                bool rowOk = false;
                int rowAttempts = 0;

                for (int attempt = 0; attempt < 6; attempt++)
                {
                    rowAttempts = attempt + 1;

                    FillLowerGather(in A, in Gm, rowStart, m, BR, shift, ref Ahat);
                    FillIdentityLastBlock(ref X, m, BR);

                    var chInfo = CHO.decompInPlace(ref Ahat);
                    if (chInfo.Solved)
                    {
                        CHO.decompSolve(ref Ahat, ref X);   // X := A_hat^{-1} E

                        var D = new fProxyMxN(BR, BR, Allocator.Temp, true);
                        CopyLastBlock(in X, m, BR, ref D);

                        var dInfo = CHO.decompInPlace(ref D);
                        if (dInfo.Solved)
                        {
                            var Xt = new fProxyMxN(BR, n, Allocator.Temp, true);
                            TransposeInto(in X, ref Xt);
                            Blas.triLower(ref D, ref Xt);   // Xt := C^{-1} X^T = g (BR x n)
                            ScatterRow(in Xt, m, BR, gValues, rowStart);
                            Xt.Dispose();
                            D.Dispose();
                            rowOk = true;
                            break;
                        }
                        D.Dispose();
                    }

                    shift = shift == (fProxy)0 ? (fProxy)1e-3 * diagMax : shift * (fProxy)10;
                }

                X.Dispose();
                Ahat.Dispose();

                if (rowAttempts > worstAttempts) worstAttempts = rowAttempts;
                if (shift > worstShift) worstShift = shift;
                if (!rowOk) { ok = false; break; }
            }

            G = Gm;
            Gt = arena.fProxyBSRTranspose(in Gm);
            Scratch = arena.fProxyVec(nb * BR, true);
            Shift = worstShift;

            info = new PreconditionerInfo
            {
                status = ok ? DirectSolveStatus.Success : DirectSolveStatus.NotPositiveDefinite,
                shift = (double)worstShift,
                attempts = worstAttempts,
            };
        }

        // True when off-diagonal block A[row(k),col(k)] is small enough (relative to its diagonal
        // blocks' Frobenius norms) to drop from S under opts.dropTol.
        static bool BelowDropTol(in fProxyBSR A, int k, int blockLen, fProxy diagNormI, fProxy diagNormJ, fProxy dropTol)
        {
            int off = k * blockLen;
            fProxy normSq = 0;
            for (int t = 0; t < blockLen; t++)
            {
                fProxy v = A.Values[off + t];
                normSq += v * v;
            }
            fProxy threshold = dropTol * math.sqrt(diagNormI * diagNormJ);
            return math.sqrt(normSq) <= threshold;
        }

        // Fills the lower triangle (b <= a) of the n x n dense A_hat = A[J_i, J_i] from G's row-i
        // pattern (J_i); CHO only ever reads a factor's lower triangle, so the strict upper is left
        // untouched. Adds `shift` to A_hat's own diagonal blocks (mirrors shifting A -> A + shift*I).
        static void FillLowerGather(in fProxyBSR A, in fProxyBSR G, int rowStart, int m, int BR, fProxy shift, ref fProxyMxN Ahat)
        {
            for (int aI = 0; aI < m; aI++)
            {
                int ga = G.ColInd[rowStart + aI];
                for (int bI = 0; bI <= aI; bI++)
                {
                    int gb = G.ColInd[rowStart + bI];
                    GatherBlockInto(in A, ga, gb, ref Ahat, aI * BR, bI * BR, aI == bI ? shift : (fProxy)0);
                }
            }
        }

        // E (n x BR): zero except an I_BR block in the LAST block-row slot.
        static void FillIdentityLastBlock(ref fProxyMxN X, int m, int BR)
        {
            int n = X.M_Rows;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < BR; c++)
                    X[r, c] = (fProxy)0;
            int last = (m - 1) * BR;
            for (int c = 0; c < BR; c++)
                X[last + c, c] = (fProxy)1;
        }

        // D := X's last BR rows (the diagonal BR x BR slot of X = A_hat^{-1} E).
        static void CopyLastBlock(in fProxyMxN X, int m, int BR, ref fProxyMxN D)
        {
            int last = (m - 1) * BR;
            for (int r = 0; r < BR; r++)
                for (int c = 0; c < BR; c++)
                    D[r, c] = X[last + r, c];
        }

        static void TransposeInto(in fProxyMxN X, ref fProxyMxN Xt)
        {
            int n = X.M_Rows, BR = X.N_Cols;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < BR; c++)
                    Xt[c, r] = X[r, c];
        }

        // Scatters g (BR x n, already in G's row-block layout) into G's row-i slots, one BR x BR
        // block per J_i entry, in the SAME ascending order G's pattern was built in.
        static void ScatterRow(in fProxyMxN Xt, int m, int BR, UnsafeList<fProxy> gValues, int rowStart)
        {
            int blockLen = BR * BR;
            for (int aI = 0; aI < m; aI++)
            {
                int dstOff = (rowStart + aI) * blockLen;
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        gValues[dstOff + r * BR + c] = Xt[r, aI * BR + c];
            }
        }

        // ---- shared gather primitives (also used by fProxySPAI) ----

        // Storage index of block (row,col) in A's RowPtr/ColInd, or -1 if not stored. Row degree is
        // small (sparse pattern, setup-only) -- a plain ascending scan.
        internal static int FindBlockIndex(in fProxyBSR A, int row, int col)
        {
            int s = A.RowPtr[row], e = A.RowPtr[row + 1];
            for (int k = s; k < e; k++)
                if (A.ColInd[k] == col) return k;
            return -1;
        }

        // Gathers A's (gr,gc) block (BR x BC) into dst at [rowOff:rowOff+BR, colOff:colOff+BC),
        // zero-filled when the block isn't stored. Handles Symmetric (lower-block-triangle-only)
        // storage by transposing the mirrored stored block when gc > gr. Adds diagShift to the
        // block's own scalar diagonal when gr == gc.
        internal static void GatherBlockInto(in fProxyBSR A, int gr, int gc, ref fProxyMxN dst, int rowOff, int colOff, fProxy diagShift)
        {
            int BR = A.BR, BC = A.BC;
            bool transpose = A.Symmetric && gc > gr;
            int idx = transpose ? FindBlockIndex(in A, gc, gr) : FindBlockIndex(in A, gr, gc);

            if (idx < 0)
            {
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BC; c++)
                        dst[rowOff + r, colOff + c] = (fProxy)0;
            }
            else
            {
                int off = idx * BR * BC;
                var v = A.Values;
                if (transpose)
                {
                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BC; c++)
                            dst[rowOff + r, colOff + c] = v[off + c * BC + r];
                }
                else
                {
                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BC; c++)
                            dst[rowOff + r, colOff + c] = v[off + r * BC + c];
                }
            }

            if (gr == gc && diagShift != (fProxy)0)
                for (int r = 0; r < BR; r++)
                    dst[rowOff + r, colOff + r] += diagShift;
        }

        /// <summary>z = Gᵀ (G r): two forward BSR spMVs through <see cref="Scratch"/>. z must not
        /// alias r or Scratch.</summary>
        public bool IsIdentity => false;

        public unsafe void Apply(in fProxyN r, ref fProxyN z)
        {
            int n = Rows;
            if (r.N != n)
                throw new ArgumentException("fProxyFSAI.Apply: r.N must equal Rows");
            if (z.N != n)
                throw new ArgumentException("fProxyFSAI.Apply: z.N must equal Rows");
            if (z.Data.Ptr == r.Data.Ptr)
                throw new ArgumentException("fProxyFSAI.Apply: z must not alias r");
            if (z.Data.Ptr == Scratch.Data.Ptr)
                throw new ArgumentException("fProxyFSAI.Apply: z must not alias Scratch");
            if (r.Data.Ptr == Scratch.Data.Ptr)
                throw new ArgumentException("fProxyFSAI.Apply: r must not alias Scratch");

            fProxyN scratch = Scratch;   // local alias sharing Scratch's buffer -- Scratch itself is
                                          // a readonly field and cannot be passed by ref directly.
            BSR.spMV(in G, in r, ref scratch);     // scratch = G r
            BSR.spMV(in Gt, in scratch, ref z);    // z = Gt (G r)
        }
    }
}
