using System.Globalization;
using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using BULA;
using BULA.Sparse;

namespace BULA.Benchmarks
{
    // GENERATED per-dtype half of SparseSolverBenchmark (timed IJobs + residual/build helpers + the
    // Section0..4 build+measure methods). The dtype-agnostic harness (config constants, seed helper, row
    // formatters, block-pattern choosers, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/SparseSolverBenchmark.cs (SparseSolverFmt + the partial class).
    //
    // Density is passed as `float` in BOTH the float and double builders (it is a block-fill fraction, not
    // matrix data), so `float density` is kept literal here rather than templated. The only genuinely
    // dtype-sensitive literals are the off-diagonal/diagonal block magnitudes (0.3, 0.2, 2), wrapped in
    // (fProxy) casts so the double build gets the true double value rather than a widened float literal.

    // ---- CG scratch: r, p, Ap (all A.Rows length) --------------------------------------------------

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CGDenseJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN b, x, r, p, Ap;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.cg(in A, in b, ref x, ref r, ref p, ref Ap, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CGSparseJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x, r, p, Ap;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.cg(in A, in b, ref x, ref r, ref p, ref Ap, K, 0f);
        }
    }

    // ---- MINRES scratch: y, r1, r2, v, w, w1, w2 (all A.Rows length) -------------------------------

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MinresDenseJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN b, x, y, r1, r2, v, w, w1, w2;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MinresSparseJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x, y, r1, r2, v, w, w1, w2;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, K, 0f);
        }
    }

    // ---- BiCGSTAB scratch: r, rHat0, p, v, t (all A.Rows length) -----------------------------------

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct BiCGStabDenseJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN b, x, r, rHat0, p, v, t;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct BiCGStabSparseJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x, r, rHat0, p, v, t;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, K, 0f);
        }
    }

    // ---- LSQR scratch: u, tmpM (A.Rows length), v, w, tmpN (A.Cols length) ------------------------

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LsqrDenseJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN b, x, u, v, w, tmpM, tmpN;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LsqrSparseJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN b, x, u, v, w, tmpM, tmpN;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, K, 0f);
        }
    }

    // ---- transpose-optimized sparse LSQR jobs: use a materialized Aᵀ so ApplyT runs as a forward
    //      spMV over Aᵀ instead of the cache-unfriendly on-the-fly spMVT. Aᵀ is built ONCE outside the
    //      timed region (a real solve builds it once and reuses it every iteration). ------------------

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LsqrSparseTJobFProxy : IJob
    {
        public fProxyBSR A, AT;
        public fProxyN b, x, u, v, w, tmpM, tmpN;
        public int K;
        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.lsqr(in A, in AT, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, K, 0f);
        }
    }

    // ---- Section 4: hand-inlined dense CG (no operator interface, no cg<TOp> generic dispatch -- a raw
    //      GEMV loop + axpy/dot written directly in Execute()). Same algorithm as Krylov.cg<TOp>, just with
    //      every step spelled out inline against raw pointers. x is reset to zero and tol is effectively 0
    //      (K fixed iterations), matching the other jobs. --------------------------------------------------

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CGHandInlinedJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN b, x, r, p, Ap;
        public int K;

        public unsafe void Execute()
        {
            int n = x.N;
            fProxy* Ad = A.Data.Ptr;
            fProxy* bd = b.Data.Ptr;
            fProxy* xd = x.Data.Ptr;
            fProxy* rd = r.Data.Ptr;
            fProxy* pd = p.Data.Ptr;
            fProxy* Apd = Ap.Data.Ptr;

            for (int i = 0; i < n; i++) xd[i] = 0f;
            for (int i = 0; i < n; i++) rd[i] = bd[i];       // r = b - A*0 = b
            for (int i = 0; i < n; i++) pd[i] = rd[i];       // p = r

            fProxy rsold = 0f;
            for (int i = 0; i < n; i++) rsold += rd[i] * rd[i];

            for (int k = 0; k < K; k++)
            {
                for (int row = 0; row < n; row++)
                {
                    fProxy sum = 0f;
                    int baseIdx = row * n;
                    for (int col = 0; col < n; col++)
                        sum += Ad[baseIdx + col] * pd[col];
                    Apd[row] = sum;
                }

                fProxy pAp = 0f;
                for (int i = 0; i < n; i++) pAp += pd[i] * Apd[i];
                if (!(pAp > 0f)) break;

                fProxy alpha = rsold / pAp;
                for (int i = 0; i < n; i++) xd[i] += alpha * pd[i];
                for (int i = 0; i < n; i++) rd[i] -= alpha * Apd[i];

                fProxy rsnew = 0f;
                for (int i = 0; i < n; i++) rsnew += rd[i] * rd[i];

                fProxy beta = rsnew / rsold;
                for (int i = 0; i < n; i++) pd[i] = beta * pd[i] + rd[i];

                rsold = rsnew;
            }
        }
    }

    // ---- operator matvec microbench jobs (REPS back-to-back matvecs, zero-alloc) -------------------
    //
    // Isolate the per-iteration operator cost -- dense GEMV (Blas.dot) vs sparse spMV. The reps loop
    // PING-PONGS x<->y (each matvec feeds the next) to defeat Burst dead-store elimination. Values may
    // diverge to Inf across the chain (diagonally dominant, radius >> 1) -- irrelevant to TIMING; the
    // numerical cross-check (maxAbsDiff) is computed separately from a clean single untimed matvec.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MatvecDenseJobFProxy : IJob
    {
        public fProxyMxN A;
        public fProxyN x, y;
        public int reps;
        public void Execute()
        {
            for (int k = 0; k < reps; k++)
            {
                if ((k & 1) == 0) Blas.dot(in A, in x, ref y);
                else              Blas.dot(in A, in y, ref x);
            }
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MatvecSparseJobFProxy : IJob
    {
        public fProxyBSR A;
        public fProxyN x, y;
        public int reps;
        public void Execute()
        {
            for (int k = 0; k < reps; k++)
            {
                if ((k & 1) == 0) BSR.spMV(in A, in x, ref y);
                else              BSR.spMV(in A, in y, ref x);
            }
        }
    }

    public static partial class SparseSolverBenchmark
    {
        // ==== residual helpers (always evaluated against the DENSE reference matrix) ===================

        static double ResidualLinSys(in fProxyMxN A, in fProxyN x, in fProxyN b)
        {
            var Ax = Blas.dot(A, x);
            double num = 0, den = 0;
            for (int i = 0; i < b.N; i++)
            {
                double diff = (double)Ax[i] - (double)b[i];
                num += diff * diff;
                den += (double)b[i] * (double)b[i];
            }
            return math.sqrt(num) / math.sqrt(math.max(den, 1e-30));
        }

        // Least-squares optimality: ||A^T(Ax-b)|| / ||A^T b|| -- the correct acceptance criterion for a
        // (possibly inconsistent) rectangular system, NOT ||Ax-b|| (nonzero even at the LS optimum).
        static double ResidualLS(in fProxyMxN A, in fProxyN x, in fProxyN b)
        {
            var Ax = Blas.dot(A, x);
            var res = new fProxyN(in Ax, Allocator.Persistent);
            fProxyComp.subInPlace(res, b);
            var atr = Blas.dot(res, A);
            var atb = Blas.dot(b, A);
            double num = 0, den = 0;
            for (int i = 0; i < atr.N; i++) num += (double)atr[i] * (double)atr[i];
            for (int i = 0; i < atb.N; i++) den += (double)atb[i] * (double)atb[i];
            res.Dispose();
            return math.sqrt(num) / math.sqrt(math.max(den, 1e-30));
        }

        // ==== block-matrix builders ====================================================================

        static void BuildBlockSPDFProxy(int nb, float density, uint seed, out fProxyMxN dense, out fProxyBSR sparse)
        {
            const int BR = SparseSolverFmt.BR;
            int dim = nb * BR;
            dense = new fProxyMxN(dim, dim, Allocator.Persistent);
            var pairs = SparseSolverFmt.ChooseOffDiagPairsSymmetric(nb, density, seed, out int nnzb);
            var builder = new fProxyBSRBuilder(nb, nb, BR, BR, Allocator.Persistent, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);
            fProxy strong = dim;
            fProxy offScale = (fProxy)0.3;

            for (int i = 0; i < nb; i++)
            {
                var Mi = new fProxyMxN(BR, BR, Allocator.Persistent);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Mi[r, c] = rng.NextFProxy(-1f, 1f);
                var Di = Blas.dot(Mi, Mi, true);
                for (int d = 0; d < BR; d++) Di[d, d] += strong;

                builder.AddBlock(i, i, in Di);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[i * BR + r, i * BR + c] = Di[r, c];

                Mi.Dispose();
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = new fProxyMxN(BR, BR, Allocator.Persistent);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextFProxy(-offScale, offScale);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];

                var blockT = new fProxyMxN(BR, BR, Allocator.Persistent);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        blockT[r, c] = block[c, r];

                builder.AddBlock(bj, bi, in blockT);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bj * BR + r, bi * BR + c] = blockT[r, c];

                block.Dispose();
                blockT.Dispose();
            }

            sparse = builder.ToBSR(Allocator.Persistent);
            builder.Dispose();
        }

        // Same recipe as BuildBlockSPDFProxy (identical rng sequence), but assembles TWO block-CSR encodings
        // of the SAME dense SPD matrix side by side: `full` (every stored block, incl. the explicit mirrored
        // upper block) and `sym` (lower-triangle + diagonal ONLY, via ToBSRSymmetric). Used by Section 0b to
        // isolate the symmetric-storage spMV win on a byte-for-byte identical matrix.
        static void BuildBlockSPDPairFProxy(int nb, float density, uint seed,
                                            out fProxyMxN dense, out fProxyBSR full, out fProxyBSR sym)
        {
            const int BR = SparseSolverFmt.BR;
            int dim = nb * BR;
            dense = new fProxyMxN(dim, dim, Allocator.Persistent);
            var pairs = SparseSolverFmt.ChooseOffDiagPairsSymmetric(nb, density, seed, out int nnzbFull);
            int nnzbSym = nb + pairs.Count;
            var fullBuilder = new fProxyBSRBuilder(nb, nb, BR, BR, Allocator.Persistent, nnzbFull);
            var symBuilder = new fProxyBSRBuilder(nb, nb, BR, BR, Allocator.Persistent, nnzbSym);
            var rng = new Random(seed ^ 0x9E3779B9u);
            fProxy strong = dim;
            fProxy offScale = (fProxy)0.3;

            for (int i = 0; i < nb; i++)
            {
                var Mi = new fProxyMxN(BR, BR, Allocator.Persistent);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Mi[r, c] = rng.NextFProxy(-1f, 1f);
                var Di = Blas.dot(Mi, Mi, true);
                for (int d = 0; d < BR; d++) Di[d, d] += strong;

                fullBuilder.AddBlock(i, i, in Di);
                symBuilder.AddBlock(i, i, in Di);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[i * BR + r, i * BR + c] = Di[r, c];

                Mi.Dispose();
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = new fProxyMxN(BR, BR, Allocator.Persistent);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextFProxy(-offScale, offScale);

                fullBuilder.AddBlock(bi, bj, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];

                var blockT = new fProxyMxN(BR, BR, Allocator.Persistent);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        blockT[r, c] = block[c, r];

                fullBuilder.AddBlock(bj, bi, in blockT); // mirrored upper block -- FULL storage only
                symBuilder.AddBlock(bj, bi, in blockT);  // lower only -- sym never sees the mirror
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bj * BR + r, bi * BR + c] = blockT[r, c];

                block.Dispose();
                blockT.Dispose();
            }

            full = fullBuilder.ToBSR(Allocator.Persistent);
            sym = symBuilder.ToBSRSymmetric(Allocator.Persistent);
            fullBuilder.Dispose();
            symBuilder.Dispose();
        }

        static void BuildBlockNonSymFProxy(int nb, float density, uint seed, out fProxyMxN dense, out fProxyBSR sparse)
        {
            const int BR = SparseSolverFmt.BR;
            int dim = nb * BR;
            dense = new fProxyMxN(dim, dim, Allocator.Persistent);
            var pairs = SparseSolverFmt.ChooseOffDiagPairsAsymmetric(nb, density, seed, out int nnzb);
            var builder = new fProxyBSRBuilder(nb, nb, BR, BR, Allocator.Persistent, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);
            fProxy strong = dim;
            fProxy offScale = (fProxy)0.3;

            for (int i = 0; i < nb; i++)
            {
                var Mi = new fProxyMxN(BR, BR, Allocator.Persistent);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Mi[r, c] = rng.NextFProxy(-1f, 1f);
                var Di = Blas.dot(Mi, Mi, true);
                for (int d = 0; d < BR; d++) Di[d, d] += strong;

                builder.AddBlock(i, i, in Di);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[i * BR + r, i * BR + c] = Di[r, c];

                Mi.Dispose();
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = new fProxyMxN(BR, BR, Allocator.Persistent);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextFProxy(-offScale, offScale);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];

                block.Dispose();
            }

            sparse = builder.ToBSR(Allocator.Persistent);
            builder.Dispose();
        }

        static void BuildBlockRectFProxy(int mb, int nb, float density, uint seed, out fProxyMxN dense, out fProxyBSR sparse)
        {
            const int BR = SparseSolverFmt.BR;
            int rows = mb * BR, cols = nb * BR;
            dense = new fProxyMxN(rows, cols, Allocator.Persistent);
            int diagCount = math.min(mb, nb);
            var pairs = SparseSolverFmt.ChooseOffDiagPairsRect(mb, nb, density, seed, out int nnzb);
            var builder = new fProxyBSRBuilder(mb, nb, BR, BR, Allocator.Persistent, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);

            for (int i = 0; i < diagCount; i++)
            {
                var block = new fProxyMxN(BR, BR, Allocator.Persistent);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = (r == c ? (fProxy)2 : (fProxy)0) + rng.NextFProxy(-(fProxy)0.2, (fProxy)0.2);

                builder.AddBlock(i, i, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[i * BR + r, i * BR + c] = block[r, c];

                block.Dispose();
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = new fProxyMxN(BR, BR, Allocator.Persistent);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextFProxy(-(fProxy)0.3, (fProxy)0.3);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];

                block.Dispose();
            }

            sparse = builder.ToBSR(Allocator.Persistent);
            builder.Dispose();
        }

        // Block-matrix builder, parameterized block size (used ONLY by the dedicated b=4/N=1024 Section 1x --
        // Section 1's builders keep BR hardcoded since their numbers are cited in docs). Same recipe as
        // BuildBlockSPDFProxy, generalized so the block size isn't tied to the file-wide BR=3 constant.
        static void BuildBlockSPDFProxySized(int nb, int br, float density, uint seed, out fProxyMxN dense, out fProxyBSR sparse)
        {
            int dim = nb * br;
            dense = new fProxyMxN(dim, dim, Allocator.Persistent);
            var pairs = SparseSolverFmt.ChooseOffDiagPairsSymmetric(nb, density, seed, out int nnzb);
            var builder = new fProxyBSRBuilder(nb, nb, br, br, Allocator.Persistent, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);
            fProxy strong = dim;
            fProxy offScale = (fProxy)0.3;

            for (int i = 0; i < nb; i++)
            {
                var Mi = new fProxyMxN(br, br, Allocator.Persistent);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        Mi[r, c] = rng.NextFProxy(-1f, 1f);
                var Di = Blas.dot(Mi, Mi, true);
                for (int d = 0; d < br; d++) Di[d, d] += strong;

                builder.AddBlock(i, i, in Di);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        dense[i * br + r, i * br + c] = Di[r, c];

                Mi.Dispose();
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = new fProxyMxN(br, br, Allocator.Persistent);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        block[r, c] = rng.NextFProxy(-offScale, offScale);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        dense[bi * br + r, bj * br + c] = block[r, c];

                var blockT = new fProxyMxN(br, br, Allocator.Persistent);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        blockT[r, c] = block[c, r];

                builder.AddBlock(bj, bi, in blockT);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        dense[bj * br + r, bi * br + c] = blockT[r, c];

                block.Dispose();
                blockT.Dispose();
            }

            sparse = builder.ToBSR(Allocator.Persistent);
            builder.Dispose();
        }

        // ==== Section 0: operator matvec throughput (dense GEMV vs sparse spMV) =========================

        static void Section0FProxy(StringBuilder sb)
        {
            int BR = SparseSolverFmt.BR, REPS = SparseSolverFmt.REPS_MATVEC;
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 0. Operator matvec throughput (b={0}): dense GEMV vs sparse spMV, REPS={1} [fProxy] ---", BR, REPS));
            sb.AppendLine(SparseSolverFmt.MatvecHeader());

            foreach (var n in SparseSolverFmt.BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in SparseSolverFmt.Densities)
                {
                    BuildBlockSPDFProxy(nb, density, SparseSolverFmt.Seed(n, density, 91), out var dense, out var sparse);
                    uint sx = SparseSolverFmt.Seed(n, density, 92);

                    var xd = GenerateOP.fProxyRandomVec(n, -1f, 1f, sx, Allocator.Persistent);   // ping-pong clobbers input -> fresh copy per timing
                    var yd = new fProxyN(n, Allocator.Persistent);
                    var denseJob = new MatvecDenseJobFProxy { A = dense, x = xd, y = yd, reps = REPS };
                    var denseStat = Bench.Time(() => denseJob.Run());
                    sb.AppendLine(SparseSolverFmt.MatvecRow("fProxy", n, density, "GEMV-dense", denseStat, 1.0, null));

                    var xs = GenerateOP.fProxyRandomVec(n, -1f, 1f, sx, Allocator.Persistent);   // identical contents to xd
                    var ys = new fProxyN(n, Allocator.Persistent);
                    var sparseJob = new MatvecSparseJobFProxy { A = sparse, x = xs, y = ys, reps = REPS };
                    var sparseStat = Bench.Time(() => sparseJob.Run());

                    // clean single-matvec numerical cross-check (untimed; identical input)
                    var xc = GenerateOP.fProxyRandomVec(n, -1f, 1f, sx, Allocator.Persistent);
                    var yDc = Blas.dot(dense, xc);
                    var ySc = BSR.spMV(sparse, xc);
                    double md = 0;
                    for (int i = 0; i < n; i++) md = math.max(md, math.abs((double)yDc[i] - (double)ySc[i]));
                    double speedup = denseStat.Median / math.max(sparseStat.Median, 1e-30);
                    sb.AppendLine(SparseSolverFmt.MatvecRow("fProxy", n, density, "spMV-sparse", sparseStat, speedup, md));

                    dense.Dispose();
                    sparse.Dispose();
                    xd.Dispose();
                    yd.Dispose();
                    xs.Dispose();
                    ys.Dispose();
                    xc.Dispose();
                }
            }
        }

        // ==== Section 0b: symmetric-storage spMV vs full-storage spMV on the SAME SPD matrix ============
        //
        // Isolates the symmetric-storage cost: lower-triangle-only storage
        // (ToBSRSymmetric) vs full block-CSR, on the identical matrix (BuildBlockSPDPairFProxy pins the rng
        // so `full` and `sym` encode byte-for-byte the same SPD system). bsrMatVecSym touches half as many
        // STORED blocks as bsrMatVec's full traversal -- expected ~2x at denser fill, less at sparse fill.

        static void Section0bFProxy(StringBuilder sb)
        {
            int BR = SparseSolverFmt.BR, REPS = SparseSolverFmt.REPS_MATVEC;
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 0b. Symmetric-storage spMV vs full-storage spMV, SAME SPD matrix (b={0}), REPS={1} [fProxy] ---", BR, REPS));
            sb.AppendLine(SparseSolverFmt.MatvecHeader());

            foreach (var n in SparseSolverFmt.BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in SparseSolverFmt.Densities)
                {
                    BuildBlockSPDPairFProxy(nb, density, SparseSolverFmt.Seed(n, density, 95), out var dense, out var full, out var sym);
                    uint sx = SparseSolverFmt.Seed(n, density, 96);

                    var xf = GenerateOP.fProxyRandomVec(n, -1f, 1f, sx, Allocator.Persistent);
                    var yf = new fProxyN(n, Allocator.Persistent);
                    var fullJob = new MatvecSparseJobFProxy { A = full, x = xf, y = yf, reps = REPS };
                    var fullStat = Bench.Time(() => fullJob.Run());
                    sb.AppendLine(SparseSolverFmt.MatvecRow("fProxy", n, density, "spMV-full", fullStat, 1.0, null));

                    var xs = GenerateOP.fProxyRandomVec(n, -1f, 1f, sx, Allocator.Persistent);   // identical contents to xf
                    var ys = new fProxyN(n, Allocator.Persistent);
                    var symJob = new MatvecSparseJobFProxy { A = sym, x = xs, y = ys, reps = REPS };
                    var symStat = Bench.Time(() => symJob.Run());

                    // clean single-matvec numerical cross-check (untimed; identical input)
                    var xc = GenerateOP.fProxyRandomVec(n, -1f, 1f, sx, Allocator.Persistent);
                    var yFc = BSR.spMV(full, xc);
                    var ySc = BSR.spMV(sym, xc);
                    double md = 0;
                    for (int i = 0; i < n; i++) md = math.max(md, math.abs((double)yFc[i] - (double)ySc[i]));
                    double speedup = fullStat.Median / math.max(symStat.Median, 1e-30);
                    sb.AppendLine(SparseSolverFmt.MatvecRow("fProxy", n, density, "spMV-sym", symStat, speedup, md));

                    dense.Dispose();
                    full.Dispose();
                    sym.Dispose();
                    xf.Dispose();
                    yf.Dispose();
                    xs.Dispose();
                    ys.Dispose();
                    xc.Dispose();
                }
            }
        }

        // ==== Section 1: SPD -> cg & minres ============================================================

        static void Section1FProxy(StringBuilder sb)
        {
            int BR = SparseSolverFmt.BR, K = SparseSolverFmt.K_CG;
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 1. SPD block-sparse (b={0}): cg & minres, K={1}, tol=0 [fProxy] ---", BR, K));
            sb.AppendLine(SparseSolverFmt.RowHeader());

            foreach (var n in SparseSolverFmt.BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in SparseSolverFmt.Densities)
                {
                    BuildBlockSPDFProxy(nb, density, SparseSolverFmt.Seed(n, density, 11), out var dense, out var sparse);
                    var b = GenerateOP.fProxyRandomVec(n, -1f, 1f, SparseSolverFmt.Seed(n, density, 12), Allocator.Persistent);

                    var xCgD = new fProxyN(n, Allocator.Persistent); var rCgD = new fProxyN(n, Allocator.Persistent); var pCgD = new fProxyN(n, Allocator.Persistent); var ApCgD = new fProxyN(n, Allocator.Persistent);
                    var cgDenseJob = new CGDenseJobFProxy { A = dense, b = b, x = xCgD, r = rCgD, p = pCgD, Ap = ApCgD, K = K };
                    var cgDenseStat = Bench.Time(() => cgDenseJob.Run());
                    sb.AppendLine(SparseSolverFmt.Row("fProxy", n, density, "CG-dense", cgDenseStat, ResidualLinSys(in dense, in xCgD, in b)));

                    var xCgS = new fProxyN(n, Allocator.Persistent); var rCgS = new fProxyN(n, Allocator.Persistent); var pCgS = new fProxyN(n, Allocator.Persistent); var ApCgS = new fProxyN(n, Allocator.Persistent);
                    var cgSparseJob = new CGSparseJobFProxy { A = sparse, b = b, x = xCgS, r = rCgS, p = pCgS, Ap = ApCgS, K = K };
                    var cgSparseStat = Bench.Time(() => cgSparseJob.Run());
                    sb.AppendLine(SparseSolverFmt.Row("fProxy", n, density, "CG-sparse", cgSparseStat, ResidualLinSys(in dense, in xCgS, in b)));

                    var xMrD = new fProxyN(n, Allocator.Persistent);
                    var yD = new fProxyN(n, Allocator.Persistent); var r1D = new fProxyN(n, Allocator.Persistent); var r2D = new fProxyN(n, Allocator.Persistent); var vD = new fProxyN(n, Allocator.Persistent);
                    var wD = new fProxyN(n, Allocator.Persistent); var w1D = new fProxyN(n, Allocator.Persistent); var w2D = new fProxyN(n, Allocator.Persistent);
                    var mrDenseJob = new MinresDenseJobFProxy { A = dense, b = b, x = xMrD, y = yD, r1 = r1D, r2 = r2D, v = vD, w = wD, w1 = w1D, w2 = w2D, K = K };
                    var mrDenseStat = Bench.Time(() => mrDenseJob.Run());
                    sb.AppendLine(SparseSolverFmt.Row("fProxy", n, density, "MINRES-dense", mrDenseStat, ResidualLinSys(in dense, in xMrD, in b)));

                    var xMrS = new fProxyN(n, Allocator.Persistent);
                    var yS = new fProxyN(n, Allocator.Persistent); var r1S = new fProxyN(n, Allocator.Persistent); var r2S = new fProxyN(n, Allocator.Persistent); var vS = new fProxyN(n, Allocator.Persistent);
                    var wS = new fProxyN(n, Allocator.Persistent); var w1S = new fProxyN(n, Allocator.Persistent); var w2S = new fProxyN(n, Allocator.Persistent);
                    var mrSparseJob = new MinresSparseJobFProxy { A = sparse, b = b, x = xMrS, y = yS, r1 = r1S, r2 = r2S, v = vS, w = wS, w1 = w1S, w2 = w2S, K = K };
                    var mrSparseStat = Bench.Time(() => mrSparseJob.Run());
                    sb.AppendLine(SparseSolverFmt.Row("fProxy", n, density, "MINRES-sparse", mrSparseStat, ResidualLinSys(in dense, in xMrS, in b)));

                    dense.Dispose(); sparse.Dispose(); b.Dispose();
                    xCgD.Dispose(); rCgD.Dispose(); pCgD.Dispose(); ApCgD.Dispose();
                    xCgS.Dispose(); rCgS.Dispose(); pCgS.Dispose(); ApCgS.Dispose();
                    xMrD.Dispose(); yD.Dispose(); r1D.Dispose(); r2D.Dispose(); vD.Dispose(); wD.Dispose(); w1D.Dispose(); w2D.Dispose();
                    xMrS.Dispose(); yS.Dispose(); r1S.Dispose(); r2S.Dispose(); vS.Dispose(); wS.Dispose(); w1S.Dispose(); w2S.Dispose();
                }
            }
        }

        // ==== Section 1x: SPD block-sparse, b=4, N=1024 (CG only, 7% fill) ==============================
        //
        // The b=3 sweep (Section 1) tops out at N=768 because 1024 isn't divisible by 3. b=4 IS one of the
        // compile-time-unrolled bsrMatVecB4 kernel sizes, so nb=256 blocks of 4x4 gives a genuine 1024x1024
        // CG dense-vs-sparse comparison at the same convention (7% block density, K=40, tol=0). Only CG is
        // timed here -- this subsection backfills the README's 1024x1024 CG row.

        static void Section1xFProxy(StringBuilder sb)
        {
            int N = SparseSolverFmt.N_B4, BR4 = SparseSolverFmt.BR4, NB = SparseSolverFmt.NB_B4, K = SparseSolverFmt.K_CG;
            float density = SparseSolverFmt.Densities[0]; // 7%
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 1x. SPD block-sparse (b={0}, N={1}): cg, K={2}, tol=0 [fProxy] ---", BR4, N, K));
            sb.AppendLine(SparseSolverFmt.RowHeader());

            BuildBlockSPDFProxySized(NB, BR4, density, SparseSolverFmt.Seed(N, density, 111), out var dense, out var sparse);
            var b = GenerateOP.fProxyRandomVec(N, -1f, 1f, SparseSolverFmt.Seed(N, density, 112), Allocator.Persistent);

            var xCgD = new fProxyN(N, Allocator.Persistent); var rCgD = new fProxyN(N, Allocator.Persistent); var pCgD = new fProxyN(N, Allocator.Persistent); var ApCgD = new fProxyN(N, Allocator.Persistent);
            var cgDenseJob = new CGDenseJobFProxy { A = dense, b = b, x = xCgD, r = rCgD, p = pCgD, Ap = ApCgD, K = K };
            var cgDenseStat = Bench.Time(() => cgDenseJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("fProxy", N, density, "CG-dense", cgDenseStat, ResidualLinSys(in dense, in xCgD, in b)));

            var xCgS = new fProxyN(N, Allocator.Persistent); var rCgS = new fProxyN(N, Allocator.Persistent); var pCgS = new fProxyN(N, Allocator.Persistent); var ApCgS = new fProxyN(N, Allocator.Persistent);
            var cgSparseJob = new CGSparseJobFProxy { A = sparse, b = b, x = xCgS, r = rCgS, p = pCgS, Ap = ApCgS, K = K };
            var cgSparseStat = Bench.Time(() => cgSparseJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("fProxy", N, density, "CG-sparse", cgSparseStat, ResidualLinSys(in dense, in xCgS, in b)));

            dense.Dispose(); sparse.Dispose(); b.Dispose();
            xCgD.Dispose(); rCgD.Dispose(); pCgD.Dispose(); ApCgD.Dispose();
            xCgS.Dispose(); rCgS.Dispose(); pCgS.Dispose(); ApCgS.Dispose();
        }

        // ==== Section 2: non-symmetric -> biCGStab =====================================================

        static void Section2FProxy(StringBuilder sb)
        {
            int BR = SparseSolverFmt.BR, K = SparseSolverFmt.K_BICGSTAB;
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 2. Non-symmetric block-sparse (b={0}): biCGStab, K={1}, tol=0 [fProxy] ---", BR, K));
            sb.AppendLine(SparseSolverFmt.RowHeader());

            foreach (var n in SparseSolverFmt.BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in SparseSolverFmt.Densities)
                {
                    BuildBlockNonSymFProxy(nb, density, SparseSolverFmt.Seed(n, density, 21), out var dense, out var sparse);
                    var b = GenerateOP.fProxyRandomVec(n, -1f, 1f, SparseSolverFmt.Seed(n, density, 22), Allocator.Persistent);

                    var xD = new fProxyN(n, Allocator.Persistent); var rD = new fProxyN(n, Allocator.Persistent); var rh0D = new fProxyN(n, Allocator.Persistent);
                    var pD = new fProxyN(n, Allocator.Persistent); var vD = new fProxyN(n, Allocator.Persistent); var tD = new fProxyN(n, Allocator.Persistent);
                    var jobD = new BiCGStabDenseJobFProxy { A = dense, b = b, x = xD, r = rD, rHat0 = rh0D, p = pD, v = vD, t = tD, K = K };
                    var statD = Bench.Time(() => jobD.Run());
                    sb.AppendLine(SparseSolverFmt.Row("fProxy", n, density, "BiCGStab-dense", statD, ResidualLinSys(in dense, in xD, in b)));

                    var xS = new fProxyN(n, Allocator.Persistent); var rS = new fProxyN(n, Allocator.Persistent); var rh0S = new fProxyN(n, Allocator.Persistent);
                    var pS = new fProxyN(n, Allocator.Persistent); var vS = new fProxyN(n, Allocator.Persistent); var tS = new fProxyN(n, Allocator.Persistent);
                    var jobS = new BiCGStabSparseJobFProxy { A = sparse, b = b, x = xS, r = rS, rHat0 = rh0S, p = pS, v = vS, t = tS, K = K };
                    var statS = Bench.Time(() => jobS.Run());
                    sb.AppendLine(SparseSolverFmt.Row("fProxy", n, density, "BiCGStab-sparse", statS, ResidualLinSys(in dense, in xS, in b)));

                    dense.Dispose(); sparse.Dispose(); b.Dispose();
                    xD.Dispose(); rD.Dispose(); rh0D.Dispose(); pD.Dispose(); vD.Dispose(); tD.Dispose();
                    xS.Dispose(); rS.Dispose(); rh0S.Dispose(); pS.Dispose(); vS.Dispose(); tS.Dispose();
                }
            }
        }

        // ==== Section 3: rectangular -> lsqr (over- and under-determined) ========================

        static void RunRectCaseFProxy(int nRef, int mb, int nb, float density, string tag, int tagSeed, StringBuilder sb)
        {
            int BR = SparseSolverFmt.BR, K = SparseSolverFmt.K_LS;
            BuildBlockRectFProxy(mb, nb, density, SparseSolverFmt.Seed(nRef, density, tagSeed), out var dense, out var sparse);
            int rows = mb * BR, cols = nb * BR;
            var b = GenerateOP.fProxyRandomVec(rows, -1f, 1f, SparseSolverFmt.Seed(nRef, density, tagSeed + 1), Allocator.Persistent);

            var xLD = new fProxyN(cols, Allocator.Persistent); var uLD = new fProxyN(rows, Allocator.Persistent); var vLD = new fProxyN(cols, Allocator.Persistent); var wLD = new fProxyN(cols, Allocator.Persistent);
            var tmMLD = new fProxyN(rows, Allocator.Persistent); var tmNLD = new fProxyN(cols, Allocator.Persistent);
            var lsqrDenseJob = new LsqrDenseJobFProxy { A = dense, b = b, x = xLD, u = uLD, v = vLD, w = wLD, tmpM = tmMLD, tmpN = tmNLD, K = K };
            var lsqrDenseStat = Bench.Time(() => lsqrDenseJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("fProxy", nRef, density, "LSQR-dense-" + tag, lsqrDenseStat, ResidualLS(in dense, in xLD, in b)));

            var xLS = new fProxyN(cols, Allocator.Persistent); var uLS = new fProxyN(rows, Allocator.Persistent); var vLS = new fProxyN(cols, Allocator.Persistent); var wLS = new fProxyN(cols, Allocator.Persistent);
            var tmMLS = new fProxyN(rows, Allocator.Persistent); var tmNLS = new fProxyN(cols, Allocator.Persistent);
            var lsqrSparseJob = new LsqrSparseJobFProxy { A = sparse, b = b, x = xLS, u = uLS, v = vLS, w = wLS, tmpM = tmMLS, tmpN = tmNLS, K = K };
            var lsqrSparseStat = Bench.Time(() => lsqrSparseJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("fProxy", nRef, density, "LSQR-sparse-" + tag, lsqrSparseStat, ResidualLS(in dense, in xLS, in b)));

            // Transpose-optimized variants -- Aᵀ materialized ONCE (outside timing), ApplyT
            // becomes a forward spMV over Aᵀ. Compare "sparseT" rows against the "sparse" rows above.
            var AT = sparse.Transpose(Allocator.Persistent);

            var xLST = new fProxyN(cols, Allocator.Persistent); var uLST = new fProxyN(rows, Allocator.Persistent); var vLST = new fProxyN(cols, Allocator.Persistent); var wLST = new fProxyN(cols, Allocator.Persistent);
            var tmMLST = new fProxyN(rows, Allocator.Persistent); var tmNLST = new fProxyN(cols, Allocator.Persistent);
            var lsqrSparseTJob = new LsqrSparseTJobFProxy { A = sparse, AT = AT, b = b, x = xLST, u = uLST, v = vLST, w = wLST, tmpM = tmMLST, tmpN = tmNLST, K = K };
            var lsqrSparseTStat = Bench.Time(() => lsqrSparseTJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("fProxy", nRef, density, "LSQR-sparseT-" + tag, lsqrSparseTStat, ResidualLS(in dense, in xLST, in b)));

            dense.Dispose(); sparse.Dispose(); AT.Dispose(); b.Dispose();
            xLD.Dispose(); uLD.Dispose(); vLD.Dispose(); wLD.Dispose(); tmMLD.Dispose(); tmNLD.Dispose();
            xLS.Dispose(); uLS.Dispose(); vLS.Dispose(); wLS.Dispose(); tmMLS.Dispose(); tmNLS.Dispose();
            xLST.Dispose(); uLST.Dispose(); vLST.Dispose(); wLST.Dispose(); tmMLST.Dispose(); tmNLST.Dispose();
        }

        static void Section3FProxy(StringBuilder sb)
        {
            int BR = SparseSolverFmt.BR, K = SparseSolverFmt.K_LS;
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 3. Rectangular block-sparse (b={0}): lsqr, K={1}, tol=0 [fProxy] ---", BR, K));
            sb.AppendLine("    over: rows=2xcols block grid (m=2n, overdetermined); under: cols=2xrows block grid (m=n/2, underdetermined).");
            sb.AppendLine("    residual = ||A^T(Ax-b)|| / ||A^T b|| (least-squares optimality, not ||Ax-b||).");
            sb.AppendLine(SparseSolverFmt.RowHeader());

            foreach (var n in SparseSolverFmt.BlockSizesN)
            {
                int nb0 = n / BR;
                foreach (var density in SparseSolverFmt.Densities)
                {
                    RunRectCaseFProxy(n, 2 * nb0, nb0, density, "over", 31, sb);
                    RunRectCaseFProxy(n, nb0, 2 * nb0, density, "under", 33, sb);
                }
            }
        }

        // ==== Section 4: zero-cost-abstraction probe (THE fork datapoint) ===============================
        //
        // Generic Krylov.cg(in fProxyMxN,...) -- which internally wraps A in fProxyDenseOperator and calls
        // the generic cg<TOp> loop -- vs a hand-inlined dense CG written directly against raw pointers in
        // CGHandInlinedJobFProxy, no operator interface, no generic dispatch. Same matrix, same K, same
        // algorithm. If Burst fully inlines the generic operator call the two times are ~equal (ratio ~1) --
        // the operator abstraction is then free and there is no perf case for forking dense/sparse solver
        // bodies. A material gap would argue otherwise.

        static void Section4FProxy(StringBuilder sb)
        {
            int K = SparseSolverFmt.K_CG;
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 4. Zero-cost-abstraction probe (dense SPD, K={0}, tol=0) [fProxy] ---", K));
            sb.AppendLine("    generic = Krylov.cg(in fProxyMxN,...) via cg<fProxyDenseOperator>;");
            sb.AppendLine("    hand-inlined = raw-pointer GEMV/axpy/dot written directly in the job, no operator interface.");
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-16} {3,11} {4,11} {5,10}",
                "dtype", "N", "path", "med(ms)", "min(ms)", "ratio"));

            foreach (var n in SparseSolverFmt.BlockSizesN)
            {
                var M = GenerateOP.fProxyRandomMat(n, n, -1f, 1f, SparseSolverFmt.Seed(n, 0f, 41), Allocator.Persistent);
                var A = Blas.dot(M, M, true);
                for (int d = 0; d < n; d++) A[d, d] += n;
                var b = GenerateOP.fProxyRandomVec(n, -1f, 1f, SparseSolverFmt.Seed(n, 0f, 42), Allocator.Persistent);

                var xG = new fProxyN(n, Allocator.Persistent); var rG = new fProxyN(n, Allocator.Persistent); var pG = new fProxyN(n, Allocator.Persistent); var ApG = new fProxyN(n, Allocator.Persistent);
                var genericJob = new CGDenseJobFProxy { A = A, b = b, x = xG, r = rG, p = pG, Ap = ApG, K = K };
                var genericStat = Bench.Time(() => genericJob.Run());

                var xH = new fProxyN(n, Allocator.Persistent); var rH = new fProxyN(n, Allocator.Persistent); var pH = new fProxyN(n, Allocator.Persistent); var ApH = new fProxyN(n, Allocator.Persistent);
                var handJob = new CGHandInlinedJobFProxy { A = A, b = b, x = xH, r = rH, p = pH, Ap = ApH, K = K };
                var handStat = Bench.Time(() => handJob.Run());

                double ratio = genericStat.Median / math.max(handStat.Median, 1e-9);
                sb.AppendLine(string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-16} {3,11:F4} {4,11:F4} {5,10:F3}",
                    "fProxy", n, "generic", genericStat.Median, genericStat.Min, ratio));
                sb.AppendLine(string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-16} {3,11:F4} {4,11:F4} {5,10}",
                    "fProxy", n, "hand-inlined", handStat.Median, handStat.Min, "--"));

                M.Dispose(); b.Dispose();
                xG.Dispose(); rG.Dispose(); pG.Dispose(); ApG.Dispose();
                xH.Dispose(); rH.Dispose(); pH.Dispose(); ApH.Dispose();
            }
        }
    }
}
