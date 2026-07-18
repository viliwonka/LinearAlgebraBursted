using System;
using LinearAlgebra;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Shared setup for the one-level Schwarz preconditioners: contiguous block-row partitioning,
    /// delta-layer overlap over the symmetrized block-adjacency graph, and dense local-matrix
    /// gather. Both <see cref="fProxyAdditiveSchwarz"/> and <see cref="fProxyRestrictedSchwarz"/>
    /// build their topology through here; they differ only in local factor kind and scatter.
    /// </summary>
    internal static class fProxySchwarzShared
    {
        /// <summary>
        /// Builds the subdomain topology into fresh arena buffers. Partitions block-rows
        /// 0..BlockRows-1 into K contiguous ranges of blocksPerSub = max(1, subdomainSize/BR) blocks,
        /// then extends each by opts.overlap adjacency layers. Returns the effective full-storage A
        /// to gather values from (a transient arena mirror for Symmetric-storage input, else A
        /// itself). Throws if A is not square (BlockRows==BlockCols, BR==BC).
        /// </summary>
        internal static fProxyBSR BuildTopology(in fProxyBSR a, ref Arena arena, in SchwarzOptions opts,
            out Indices subStart, out Indices subBlocks, out Indices ownedLo, out Indices ownedHi,
            out int K, out int maxBlocks)
        {
            if (a.BlockRows != a.BlockCols || a.BR != a.BC)
                throw new ArgumentException("fProxySchwarz: A must be square (BlockRows==BlockCols, BR==BC)");

            fProxyBSR A = a.Symmetric ? arena.fProxyBSRMirrorToFull(in a) : a;

            int nb = A.BlockRows;
            int BR = A.BR;

            int blocksPerSub = math.max(1, opts.subdomainSize / BR);
            K = (nb + blocksPerSub - 1) / blocksPerSub;
            int delta = math.max(0, opts.overlap);

            ownedLo = arena.Indices(K);
            ownedHi = arena.Indices(K);
            subStart = arena.Indices(K + 1);
            for (int i = 0; i < K; i++)
            {
                int lo = i * blocksPerSub;
                int hi = math.min((i + 1) * blocksPerSub, nb);
                ownedLo[i] = lo;
                ownedHi[i] = hi;
            }

            // Transient symmetrized adjacency: A's own block pattern (out-neighbors) plus its
            // transpose (in-neighbors). Duplicate edges are harmless -- the BFS mark dedups.
            var tPtr = new NativeArray<int>(nb + 1, Allocator.Temp, NativeArrayOptions.ClearMemory);
            int nnzb = A.Nnzb;
            var tIdx = new NativeArray<int>(nnzb, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            {
                var aRowPtr = A.RowPtr; var aColInd = A.ColInd;
                for (int k = 0; k < nnzb; k++) tPtr[aColInd[k] + 1]++;
                for (int c = 0; c < nb; c++) tPtr[c + 1] += tPtr[c];
                var fill = new NativeArray<int>(nb, Allocator.Temp, NativeArrayOptions.ClearMemory);
                for (int r = 0; r < nb; r++)
                {
                    int s = aRowPtr[r], e = aRowPtr[r + 1];
                    for (int k = s; k < e; k++)
                    {
                        int c = aColInd[k];
                        tIdx[tPtr[c] + fill[c]] = r;
                        fill[c]++;
                    }
                }
                fill.Dispose();
            }

            // Multi-source BFS per subdomain, bounded to delta layers. mark[g]==stamp marks members;
            // dist[g] the layer (trusted only when mark[g]==stamp).
            var mark = new NativeArray<int>(nb, Allocator.Temp, NativeArrayOptions.ClearMemory);
            var dist = new NativeArray<int>(nb, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var queue = new UnsafeList<int>(math.max(1, blocksPerSub), Allocator.Temp);
            var allMembers = new UnsafeList<int>(math.max(1, nb), Allocator.Temp);

            var aRp = A.RowPtr; var aCi = A.ColInd;
            maxBlocks = 0;
            subStart[0] = 0;
            for (int i = 0; i < K; i++)
            {
                int stamp = i + 1;
                queue.Clear();
                int lo = ownedLo[i], hi = ownedHi[i];
                for (int g = lo; g < hi; g++)
                {
                    mark[g] = stamp;
                    dist[g] = 0;
                    queue.Add(g);
                }

                int head = 0;
                while (head < queue.Length)
                {
                    int r = queue[head++];
                    int d = dist[r];
                    if (d >= delta) continue;

                    int s = aRp[r], e = aRp[r + 1];
                    for (int k = s; k < e; k++)
                    {
                        int j = aCi[k];
                        if (mark[j] != stamp) { mark[j] = stamp; dist[j] = d + 1; queue.Add(j); }
                    }
                    int ts = tPtr[r], te = tPtr[r + 1];
                    for (int k = ts; k < te; k++)
                    {
                        int j = tIdx[k];
                        if (mark[j] != stamp) { mark[j] = stamp; dist[j] = d + 1; queue.Add(j); }
                    }
                }

                // Collect members ascending (canonical local order) by scanning the whole index space.
                int count = 0;
                for (int g = 0; g < nb; g++)
                    if (mark[g] == stamp) { allMembers.Add(g); count++; }

                subStart[i + 1] = subStart[i] + count;
                if (count > maxBlocks) maxBlocks = count;
            }

            subBlocks = arena.Indices(allMembers.Length);
            for (int t = 0; t < allMembers.Length; t++) subBlocks[t] = allMembers[t];

            allMembers.Dispose();
            queue.Dispose();
            dist.Dispose();
            mark.Dispose();
            tIdx.Dispose();
            tPtr.Dispose();

            return A;
        }

        /// <summary>Gathers A's (gr,gc) BR x BR block into the dense local matrix M at
        /// (rowOff,colOff), zero-filled when the block is not stored; adds diagShift to the block's
        /// own scalar diagonal when gr==gc. A must be full storage (ascending ColInd per row).</summary>
        internal static void GatherBlock(in fProxyBSR A, int gr, int gc, ref fProxyMxN M, int rowOff, int colOff, fProxy diagShift, int BR)
        {
            int s = A.RowPtr[gr], e = A.RowPtr[gr + 1];
            int idx = -1;
            for (int k = s; k < e; k++)
            {
                int col = A.ColInd[k];
                if (col == gc) { idx = k; break; }
                if (col > gc) break;
            }

            if (idx < 0)
            {
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        M[rowOff + r, colOff + c] = (fProxy)0;
            }
            else
            {
                int off = idx * BR * BR;
                var v = A.Values;
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        M[rowOff + r, colOff + c] = v[off + r * BR + c];
            }

            if (gr == gc && diagShift != (fProxy)0)
                for (int r = 0; r < BR; r++)
                    M[rowOff + r, colOff + r] += diagShift;
        }

        /// <summary>Largest |scalar diagonal entry| across A's stored diagonal blocks (shift scale);
        /// 1 if none is positive.</summary>
        internal static fProxy DiagMax(in fProxyBSR A)
        {
            int nb = A.BlockRows, BR = A.BR, blockLen = BR * BR;
            fProxy diagMax = 0;
            for (int i = 0; i < nb; i++)
            {
                int s = A.RowPtr[i], e = A.RowPtr[i + 1];
                for (int k = s; k < e; k++)
                {
                    if (A.ColInd[k] != i) continue;
                    int off = k * blockLen;
                    for (int r = 0; r < BR; r++)
                    {
                        fProxy av = math.abs(A.Values[off + r * BR + r]);
                        if (av > diagMax) diagMax = av;
                    }
                    break;
                }
            }
            if (diagMax <= (fProxy)0) diagMax = (fProxy)1;
            return diagMax;
        }
    }

    /// <summary>
    /// One-level SYMMETRIC additive Schwarz preconditioner over a square SPD BSR:
    /// M^-1 = sum_i R_i^T A_i^-1 R_i, where R_i restricts onto overlapped subdomain i and
    /// A_i = R_i A R_i^T is factored densely (Cholesky) once at build and reused by every Apply.
    /// The overlapped scatter (R_i^T sums each dof over every subdomain that owns it) makes M
    /// symmetric; whenever the build reports Success, M is SPD by construction -- valid for
    /// <see cref="LinearAlgebra.Krylov"/>.cg AND minres.
    ///
    /// Subdomains are contiguous block-row ranges of size opts.subdomainSize (in scalar unknowns,
    /// rounded to whole blocks) extended by opts.overlap adjacency layers. A local Cholesky that
    /// breaks down retries with the IC0-style escalating diagonal shift on just that subdomain
    /// (1e-3*diagMax, x10, up to 6 attempts); the worst shift used is in <see cref="Shift"/>.
    /// Cached-factor memory is O(N * overlapFactor^2 * subdomainSize) -- see
    /// <see cref="SchwarzOptions"/>. Symmetric-storage A is mirrored to full transiently at setup
    /// (A is read only at setup, never at Apply). Arena-composed -- no record table of its own, no
    /// Dispose(). A single instance is not safe for concurrent Apply (the local scratch is shared).
    /// </summary>
    public readonly struct fProxyAdditiveSchwarz : IfProxyPreconditioner
    {
        /// <summary>Prefix offsets into <see cref="SubBlocks"/> (CSR-of-subdomains), length K+1.</summary>
        public readonly Indices SubStart;
        /// <summary>Each subdomain's overlapped block ids, ascending; concatenated per SubStart.</summary>
        public readonly Indices SubBlocks;
        /// <summary>Owned (non-overlapped) block range [lo,hi) per subdomain, length K each.</summary>
        public readonly Indices OwnedLo;
        public readonly Indices OwnedHi;
        /// <summary>Prefix offsets into <see cref="Factors"/> (entries = sum n_i^2), length K+1.</summary>
        public readonly Indices FactorStart;
        /// <summary>Cached dense lower Cholesky factors L_i, row-major n_i x n_i per subdomain.</summary>
        public readonly fProxyN Factors;
        /// <summary>Apply-time local vector, length <see cref="MaxLocalN"/> (contents mutate; no
        /// struct field does -- survives an IJob struct-copy).</summary>
        public readonly fProxyN Scratch;
        /// <summary>Worst per-subdomain diagonal shift that made a local factor succeed; 0 if clean.</summary>
        public readonly fProxy Shift;

        public readonly int BlockRows;
        public readonly int BR;
        public readonly int K;
        public readonly int MaxLocalN;

        public int Rows => BlockRows * BR;

        /// <summary>Builds AS with <see cref="SchwarzOptions.Default"/>. Throws if A is not square or
        /// if a local factor breaks down at every diagonal shift; use the out-info overload for a
        /// non-throwing build.</summary>
        public fProxyAdditiveSchwarz(in fProxyBSR a, ref Arena arena)
        {
            this = new fProxyAdditiveSchwarz(in a, ref arena, out PreconditionerInfo info);
            if (!info.Solved)
                throw new ArgumentException("fProxyAdditiveSchwarz: a local factor broke down at every diagonal shift — is A symmetric positive definite?");
        }

        /// <summary>Non-throwing build with <see cref="SchwarzOptions.Default"/>.</summary>
        public fProxyAdditiveSchwarz(in fProxyBSR a, ref Arena arena, out PreconditionerInfo info)
        {
            this = new fProxyAdditiveSchwarz(in a, ref arena, SchwarzOptions.Default, out info);
        }

        /// <summary>Builds AS with the given options. Same throw contract as the options-less
        /// overload.</summary>
        public fProxyAdditiveSchwarz(in fProxyBSR a, ref Arena arena, in SchwarzOptions opts)
        {
            this = new fProxyAdditiveSchwarz(in a, ref arena, in opts, out PreconditionerInfo info);
            if (!info.Solved)
                throw new ArgumentException("fProxyAdditiveSchwarz: a local factor broke down at every diagonal shift — is A symmetric positive definite?");
        }

        /// <summary>
        /// Non-throwing build: info.status is Success, or NotPositiveDefinite when some subdomain's
        /// local Cholesky broke down at every diagonal shift (the preconditioner is then unusable --
        /// do not Apply); info also carries the worst rescuing shift and the worst attempts across
        /// subdomains. A non-square A still throws.
        /// </summary>
        public fProxyAdditiveSchwarz(in fProxyBSR a, ref Arena arena, in SchwarzOptions opts, out PreconditionerInfo info)
        {
            // Everything computed into locals; all readonly fields assigned once at the end (avoids
            // reading a not-yet-fully-assigned `this` in the struct ctor).
            fProxyBSR A = fProxySchwarzShared.BuildTopology(in a, ref arena, in opts,
                out Indices subStart, out Indices subBlocks, out Indices ownedLo, out Indices ownedHi,
                out int k, out int maxBlocks);

            int br = A.BR;
            int maxLocalN = maxBlocks * br;

            var factorStart = arena.Indices(k + 1);
            factorStart[0] = 0;
            for (int i = 0; i < k; i++)
            {
                int nn = (subStart[i + 1] - subStart[i]) * br;
                factorStart[i + 1] = factorStart[i] + nn * nn;
            }
            var factors = arena.fProxyVec(factorStart[k]);
            var scratch = arena.fProxyVec(math.max(1, maxLocalN), true);

            fProxy diagMax = fProxySchwarzShared.DiagMax(in A);

            fProxy worstShift = 0;
            int worstAttempts = 1;
            bool ok = true;

            for (int i = 0; i < k; i++)
            {
                int start = subStart[i];
                int count = subStart[i + 1] - start;
                int n = count * br;
                int fbase = factorStart[i];

                var M = new fProxyMxN(n, n, Allocator.Temp, true);
                fProxy shift = 0;
                bool subOk = false;
                int subAttempts = 0;

                for (int attempt = 0; attempt < 6; attempt++)
                {
                    subAttempts = attempt + 1;
                    for (int aI = 0; aI < count; aI++)
                    {
                        int gr = subBlocks[start + aI];
                        for (int bI = 0; bI <= aI; bI++)
                        {
                            int gc = subBlocks[start + bI];
                            fProxySchwarzShared.GatherBlock(in A, gr, gc, ref M, aI * br, bI * br, aI == bI ? shift : (fProxy)0, br);
                        }
                    }

                    var chInfo = CHO.decompInPlace(ref M);
                    if (chInfo.Solved)
                    {
                        var md = M.Data;
                        for (int t = 0; t < n * n; t++) factors[fbase + t] = md[t];
                        subOk = true;
                        break;
                    }
                    shift = shift == (fProxy)0 ? (fProxy)1e-3 * diagMax : shift * (fProxy)10;
                }

                M.Dispose();

                if (subAttempts > worstAttempts) worstAttempts = subAttempts;
                if (shift > worstShift) worstShift = shift;
                if (!subOk) { ok = false; break; }
            }

            SubStart = subStart;
            SubBlocks = subBlocks;
            OwnedLo = ownedLo;
            OwnedHi = ownedHi;
            FactorStart = factorStart;
            Factors = factors;
            Scratch = scratch;
            Shift = worstShift;
            BlockRows = A.BlockRows;
            BR = br;
            K = k;
            MaxLocalN = maxLocalN;

            info = new PreconditionerInfo
            {
                status = ok ? DirectSolveStatus.Success : DirectSolveStatus.NotPositiveDefinite,
                shift = (double)worstShift,
                attempts = worstAttempts,
            };
        }

        /// <summary>z = M^-1 r = sum_i R_i^T A_i^-1 R_i r: zero z, then for each subdomain gather r,
        /// solve against the cached Cholesky factor, and add the local solution back into z over the
        /// overlapped block set. z must not alias r or Scratch.</summary>
        public bool IsIdentity => false;

        public unsafe void Apply(in fProxyN r, ref fProxyN z)
        {
            int n = Rows;
            if (r.N != n) throw new ArgumentException("fProxyAdditiveSchwarz.Apply: r.N must equal Rows");
            if (z.N != n) throw new ArgumentException("fProxyAdditiveSchwarz.Apply: z.N must equal Rows");
            if (z.Data.Ptr == r.Data.Ptr) throw new ArgumentException("fProxyAdditiveSchwarz.Apply: z must not alias r");
            if (z.Data.Ptr == Scratch.Data.Ptr) throw new ArgumentException("fProxyAdditiveSchwarz.Apply: z must not alias Scratch");
            if (r.Data.Ptr == Scratch.Data.Ptr) throw new ArgumentException("fProxyAdditiveSchwarz.Apply: r must not alias Scratch");

            fProxyN buf = Scratch;
            fProxy* zp = z.Data.Ptr;
            fProxy* rp = r.Data.Ptr;
            fProxy* bufp = buf.Data.Ptr;

            for (int i = 0; i < n; i++) zp[i] = (fProxy)0;

            for (int sub = 0; sub < K; sub++)
            {
                int start = SubStart[sub];
                int count = SubStart[sub + 1] - start;
                int nn = count * BR;
                int fbase = FactorStart[sub];

                for (int l = 0; l < count; l++)
                {
                    int g = SubBlocks[start + l];
                    int lb = l * BR, gb = g * BR;
                    for (int t = 0; t < BR; t++) bufp[lb + t] = rp[gb + t];
                }

                CholSolveInPlace(Factors, fbase, nn, buf);

                for (int l = 0; l < count; l++)
                {
                    int g = SubBlocks[start + l];
                    int lb = l * BR, gb = g * BR;
                    for (int t = 0; t < BR; t++) zp[gb + t] += bufp[lb + t];
                }
            }
        }

        // Solves (L L^T) x = b in place on b[0..nn), L the row-major nn x nn factor at Factors[fbase].
        static unsafe void CholSolveInPlace(fProxyN factors, int fbase, int nn, fProxyN b)
        {
            fProxy* f = factors.Data.Ptr;
            fProxy* bp = b.Data.Ptr;
            for (int r = 0; r < nn; r++)
            {
                fProxy s = bp[r];
                int rowOff = fbase + r * nn;
                for (int c = 0; c < r; c++) s -= f[rowOff + c] * bp[c];
                bp[r] = s / f[rowOff + r];
            }
            for (int r = nn - 1; r >= 0; r--)
            {
                fProxy s = bp[r];
                for (int c = r + 1; c < nn; c++) s -= f[fbase + c * nn + r] * bp[c];
                bp[r] = s / f[fbase + r * nn + r];
            }
        }
    }

    /// <summary>
    /// One-level RESTRICTED additive Schwarz (RAS) preconditioner over a square BSR:
    /// M^-1 = sum_i R~_i^T A_i^-1 R_i, where R_i gathers the overlapped subdomain and R~_i^T scatters
    /// back only its OWNED (non-overlapped) cell (Cai &amp; Sarkis 1999). Because the owned cells
    /// partition the index space, every dof is written exactly once -- no overlap summation. This
    /// makes M NON-SYMMETRIC even for SPD A: never use it with cg/minres/CG (there is no such
    /// overload -- the type is the guard); use <see cref="LinearAlgebra.Krylov"/>.pbiCGStab. The
    /// nonsymmetric sibling of <see cref="fProxyAdditiveSchwarz"/>, mirroring the IC0/ILU0 split.
    ///
    /// Local matrices A_i are factored densely once (LU with partial pivoting, since the target is
    /// general square A) and reused by every Apply. A numerically singular local block reports
    /// Singular via the out-info twin (no diagonal-shift retry -- RAS targets general A). Cached-
    /// factor memory is O(N * overlapFactor^2 * subdomainSize) -- see <see cref="SchwarzOptions"/>.
    /// Symmetric-storage A is mirrored to full transiently at setup. Arena-composed -- no record
    /// table of its own, no Dispose(). A single instance is not safe for concurrent Apply.
    /// </summary>
    public readonly struct fProxyRestrictedSchwarz : IfProxyPreconditioner
    {
        /// <summary>Prefix offsets into <see cref="SubBlocks"/> (CSR-of-subdomains), length K+1.</summary>
        public readonly Indices SubStart;
        /// <summary>Each subdomain's overlapped block ids, ascending; concatenated per SubStart.</summary>
        public readonly Indices SubBlocks;
        /// <summary>Owned (non-overlapped) block range [lo,hi) per subdomain, length K each; the RAS
        /// scatter restricts to this range.</summary>
        public readonly Indices OwnedLo;
        public readonly Indices OwnedHi;
        /// <summary>Prefix offsets into <see cref="Factors"/> (entries = sum n_i^2), length K+1.</summary>
        public readonly Indices FactorStart;
        /// <summary>Cached compact LU factors in logical (pivoted) row order, row-major per subdomain
        /// (unit-lower L below the diagonal, U on/above it).</summary>
        public readonly fProxyN Factors;
        /// <summary>Per-subdomain row-permutation P (logical row r reads gathered entry P[r]);
        /// subdomain i occupies [SubStart[i]*BR, SubStart[i+1]*BR).</summary>
        public readonly Indices Piv;
        /// <summary>Apply-time gather buffer, length <see cref="MaxLocalN"/>.</summary>
        public readonly fProxyN Scratch;
        /// <summary>Apply-time solve buffer (permuted RHS then solution), length <see cref="MaxLocalN"/>.</summary>
        public readonly fProxyN Scratch2;

        public readonly int BlockRows;
        public readonly int BR;
        public readonly int K;
        public readonly int MaxLocalN;

        public int Rows => BlockRows * BR;

        /// <summary>Builds RAS with <see cref="SchwarzOptions.Default"/>. Throws if A is not square or
        /// if a local LU is singular; use the out-info overload for a non-throwing build.</summary>
        public fProxyRestrictedSchwarz(in fProxyBSR a, ref Arena arena)
        {
            this = new fProxyRestrictedSchwarz(in a, ref arena, out PreconditionerInfo info);
            if (!info.Solved)
                throw new ArgumentException("fProxyRestrictedSchwarz: a local LU factor is numerically singular");
        }

        /// <summary>Non-throwing build with <see cref="SchwarzOptions.Default"/>.</summary>
        public fProxyRestrictedSchwarz(in fProxyBSR a, ref Arena arena, out PreconditionerInfo info)
        {
            this = new fProxyRestrictedSchwarz(in a, ref arena, SchwarzOptions.Default, out info);
        }

        /// <summary>Builds RAS with the given options. Same throw contract as the options-less
        /// overload.</summary>
        public fProxyRestrictedSchwarz(in fProxyBSR a, ref Arena arena, in SchwarzOptions opts)
        {
            this = new fProxyRestrictedSchwarz(in a, ref arena, in opts, out PreconditionerInfo info);
            if (!info.Solved)
                throw new ArgumentException("fProxyRestrictedSchwarz: a local LU factor is numerically singular");
        }

        /// <summary>
        /// Non-throwing build: info.status is Success, or Singular when some subdomain's local LU
        /// hits a zero pivot (the preconditioner is then unusable -- do not Apply). shift is always 0
        /// and attempts always 1 (RAS does not retry with a diagonal shift). A non-square A still
        /// throws.
        /// </summary>
        public fProxyRestrictedSchwarz(in fProxyBSR a, ref Arena arena, in SchwarzOptions opts, out PreconditionerInfo info)
        {
            // Everything computed into locals; all readonly fields assigned once at the end (avoids
            // reading a not-yet-fully-assigned `this` in the struct ctor).
            fProxyBSR A = fProxySchwarzShared.BuildTopology(in a, ref arena, in opts,
                out Indices subStart, out Indices subBlocks, out Indices ownedLo, out Indices ownedHi,
                out int k, out int maxBlocks);

            int br = A.BR;
            int maxLocalN = maxBlocks * br;

            var factorStart = arena.Indices(k + 1);
            factorStart[0] = 0;
            for (int i = 0; i < k; i++)
            {
                int nn = (subStart[i + 1] - subStart[i]) * br;
                factorStart[i + 1] = factorStart[i] + nn * nn;
            }
            var factors = arena.fProxyVec(factorStart[k]);
            var piv = arena.Indices(subStart[k] * br);
            var scratch = arena.fProxyVec(math.max(1, maxLocalN), true);
            var scratch2 = arena.fProxyVec(math.max(1, maxLocalN), true);

            bool ok = true;
            for (int i = 0; i < k; i++)
            {
                int start = subStart[i];
                int count = subStart[i + 1] - start;
                int n = count * br;
                int fbase = factorStart[i];
                int pbase = start * br;

                var M = new fProxyMxN(n, n, Allocator.Temp, true);
                for (int aI = 0; aI < count; aI++)
                {
                    int gr = subBlocks[start + aI];
                    for (int bI = 0; bI < count; bI++)
                    {
                        int gc = subBlocks[start + bI];
                        fProxySchwarzShared.GatherBlock(in A, gr, gc, ref M, aI * br, bI * br, (fProxy)0, br);
                    }
                }

                var P = new Pivot(n, Allocator.Temp);
                var luInfo = LU.decompInPlace(ref M, ref P);
                if (luInfo.Solved)
                {
                    // Store the compact LU in logical row order: F[r,c] = M[P[r], c].
                    for (int rr = 0; rr < n; rr++)
                    {
                        int pr = P[rr];
                        piv[pbase + rr] = pr;
                        int dst = fbase + rr * n;
                        for (int c = 0; c < n; c++) factors[dst + c] = M[pr, c];
                    }
                }
                else ok = false;

                P.Dispose();
                M.Dispose();
                if (!ok) break;
            }

            SubStart = subStart;
            SubBlocks = subBlocks;
            OwnedLo = ownedLo;
            OwnedHi = ownedHi;
            FactorStart = factorStart;
            Factors = factors;
            Piv = piv;
            Scratch = scratch;
            Scratch2 = scratch2;
            BlockRows = A.BlockRows;
            BR = br;
            K = k;
            MaxLocalN = maxLocalN;

            info = new PreconditionerInfo
            {
                status = ok ? DirectSolveStatus.Success : DirectSolveStatus.Singular,
                shift = 0,
                attempts = 1,
            };
        }

        /// <summary>z = M^-1 r = sum_i R~_i^T A_i^-1 R_i r: for each subdomain gather r, solve against
        /// the cached LU factor, and write the owned-cell entries into z (each dof exactly once). z
        /// must not alias r, Scratch, or Scratch2.</summary>
        public bool IsIdentity => false;

        public unsafe void Apply(in fProxyN r, ref fProxyN z)
        {
            int n = Rows;
            if (r.N != n) throw new ArgumentException("fProxyRestrictedSchwarz.Apply: r.N must equal Rows");
            if (z.N != n) throw new ArgumentException("fProxyRestrictedSchwarz.Apply: z.N must equal Rows");
            if (z.Data.Ptr == r.Data.Ptr) throw new ArgumentException("fProxyRestrictedSchwarz.Apply: z must not alias r");
            if (z.Data.Ptr == Scratch.Data.Ptr || z.Data.Ptr == Scratch2.Data.Ptr) throw new ArgumentException("fProxyRestrictedSchwarz.Apply: z must not alias Scratch");
            if (r.Data.Ptr == Scratch.Data.Ptr || r.Data.Ptr == Scratch2.Data.Ptr) throw new ArgumentException("fProxyRestrictedSchwarz.Apply: r must not alias Scratch");

            fProxyN gatherBuf = Scratch;
            fProxyN solveBuf = Scratch2;
            fProxy* rp = r.Data.Ptr;
            fProxy* zp = z.Data.Ptr;
            fProxy* gatherp = gatherBuf.Data.Ptr;
            fProxy* solvep = solveBuf.Data.Ptr;

            for (int sub = 0; sub < K; sub++)
            {
                int start = SubStart[sub];
                int count = SubStart[sub + 1] - start;
                int nn = count * BR;
                int fbase = FactorStart[sub];
                int pbase = start * BR;

                for (int l = 0; l < count; l++)
                {
                    int g = SubBlocks[start + l];
                    int lb = l * BR, gb = g * BR;
                    for (int t = 0; t < BR; t++) gatherp[lb + t] = rp[gb + t];
                }

                for (int rr = 0; rr < nn; rr++) solvep[rr] = gatherp[Piv[pbase + rr]];

                LUSolveInPlace(Factors, fbase, nn, solveBuf);

                int lo = OwnedLo[sub], hi = OwnedHi[sub];
                for (int l = 0; l < count; l++)
                {
                    int g = SubBlocks[start + l];
                    if (g < lo || g >= hi) continue;
                    int lb = l * BR, gb = g * BR;
                    for (int t = 0; t < BR; t++) zp[gb + t] = solvep[lb + t];
                }
            }
        }

        // Solves (L U) x = b in place on b[0..nn) (already row-permuted): unit-lower forward sweep,
        // then upper backward sweep. F the row-major compact LU at Factors[fbase].
        static unsafe void LUSolveInPlace(fProxyN factors, int fbase, int nn, fProxyN b)
        {
            fProxy* f = factors.Data.Ptr;
            fProxy* bp = b.Data.Ptr;
            for (int r = 0; r < nn; r++)
            {
                fProxy s = bp[r];
                int rowOff = fbase + r * nn;
                for (int c = 0; c < r; c++) s -= f[rowOff + c] * bp[c];
                bp[r] = s;
            }
            for (int r = nn - 1; r >= 0; r--)
            {
                fProxy s = bp[r];
                int rowOff = fbase + r * nn;
                for (int c = r + 1; c < nn; c++) s -= f[rowOff + c] * bp[c];
                bp[r] = s / f[rowOff + r];
            }
        }
    }
}
