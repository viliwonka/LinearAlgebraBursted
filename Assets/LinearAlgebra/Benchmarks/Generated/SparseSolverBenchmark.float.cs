using System.Globalization;
using System.Text;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;
using LinearAlgebra.Sparse;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of SparseSolverBenchmark (timed IJobs + residual/build helpers + the
    // Section0..4 build+measure methods). The dtype-agnostic harness (config constants, seed helper, row
    // formatters, block-pattern choosers, Run, Section) is hand-written in
    // Assets/LinearAlgebra/Benchmarks/SparseSolverBenchmark.cs (SparseSolverFmt + the partial class).
    //
    // Density is passed as `float` in BOTH the float and double builders (it is a block-fill fraction, not
    // matrix data), so `float density` is kept literal here rather than templated. The only genuinely
    // dtype-sensitive literals are the off-diagonal/diagonal block magnitudes (0.3, 0.2, 2), wrapped in
    // (float) casts so the double build gets the true double value rather than a widened float literal.

    // ---- CG scratch: r, p, Ap (all A.Rows length) --------------------------------------------------

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CGDenseJobFloat : IJob
    {
        public floatMxN A;
        public floatN b, x, r, p, Ap;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.cg(in A, in b, ref x, ref r, ref p, ref Ap, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CGSparseJobFloat : IJob
    {
        public floatBSR A;
        public floatN b, x, r, p, Ap;
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
    public struct MinresDenseJobFloat : IJob
    {
        public floatMxN A;
        public floatN b, x, y, r1, r2, v, w, w1, w2;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MinresSparseJobFloat : IJob
    {
        public floatBSR A;
        public floatN b, x, y, r1, r2, v, w, w1, w2;
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
    public struct BiCGStabDenseJobFloat : IJob
    {
        public floatMxN A;
        public floatN b, x, r, rHat0, p, v, t;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct BiCGStabSparseJobFloat : IJob
    {
        public floatBSR A;
        public floatN b, x, r, rHat0, p, v, t;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, K, 0f);
        }
    }

    // ---- CGLS scratch: r, q (A.Rows length), s, p (A.Cols length) ----------------------------------

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CglsDenseJobFloat : IJob
    {
        public floatMxN A;
        public floatN b, x, r, s, p, q;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.cgls(in A, in b, ref x, ref r, ref s, ref p, ref q, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CglsSparseJobFloat : IJob
    {
        public floatBSR A;
        public floatN b, x, r, s, p, q;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.cgls(in A, in b, ref x, ref r, ref s, ref p, ref q, K, 0f);
        }
    }

    // ---- LSQR scratch: u, tmpM (A.Rows length), v, w, tmpN (A.Cols length) ------------------------

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LsqrDenseJobFloat : IJob
    {
        public floatMxN A;
        public floatN b, x, u, v, w, tmpM, tmpN;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LsqrSparseJobFloat : IJob
    {
        public floatBSR A;
        public floatN b, x, u, v, w, tmpM, tmpN;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, K, 0f);
        }
    }

    // ---- transpose-optimized sparse CGLS/LSQR jobs (Milestone B): use a materialized Aᵀ so ApplyT runs
    //      as a forward spMV over Aᵀ instead of the cache-unfriendly on-the-fly spMVT. Aᵀ is built ONCE
    //      outside the timed region (a real solve builds it once and reuses it every iteration). --------

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CglsSparseTJobFloat : IJob
    {
        public floatBSR A, AT;
        public floatN b, x, r, s, p, q;
        public int K;
        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.cgls(in A, in AT, in b, ref x, ref r, ref s, ref p, ref q, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LsqrSparseTJobFloat : IJob
    {
        public floatBSR A, AT;
        public floatN b, x, u, v, w, tmpM, tmpN;
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
    public struct CGHandInlinedJobFloat : IJob
    {
        public floatMxN A;
        public floatN b, x, r, p, Ap;
        public int K;

        public unsafe void Execute()
        {
            int n = x.N;
            float* Ad = A.Data.Ptr;
            float* bd = b.Data.Ptr;
            float* xd = x.Data.Ptr;
            float* rd = r.Data.Ptr;
            float* pd = p.Data.Ptr;
            float* Apd = Ap.Data.Ptr;

            for (int i = 0; i < n; i++) xd[i] = 0f;
            for (int i = 0; i < n; i++) rd[i] = bd[i];       // r = b - A*0 = b
            for (int i = 0; i < n; i++) pd[i] = rd[i];       // p = r

            float rsold = 0f;
            for (int i = 0; i < n; i++) rsold += rd[i] * rd[i];

            for (int k = 0; k < K; k++)
            {
                for (int row = 0; row < n; row++)
                {
                    float sum = 0f;
                    int baseIdx = row * n;
                    for (int col = 0; col < n; col++)
                        sum += Ad[baseIdx + col] * pd[col];
                    Apd[row] = sum;
                }

                float pAp = 0f;
                for (int i = 0; i < n; i++) pAp += pd[i] * Apd[i];
                if (!(pAp > 0f)) break;

                float alpha = rsold / pAp;
                for (int i = 0; i < n; i++) xd[i] += alpha * pd[i];
                for (int i = 0; i < n; i++) rd[i] -= alpha * Apd[i];

                float rsnew = 0f;
                for (int i = 0; i < n; i++) rsnew += rd[i] * rd[i];

                float beta = rsnew / rsold;
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
    public struct MatvecDenseJobFloat : IJob
    {
        public floatMxN A;
        public floatN x, y;
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
    public struct MatvecSparseJobFloat : IJob
    {
        public floatBSR A;
        public floatN x, y;
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

        static double ResidualLinSys(in floatMxN A, in floatN x, in floatN b)
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
        static double ResidualLS(in floatMxN A, in floatN x, in floatN b)
        {
            var Ax = Blas.dot(A, x);
            var res = Ax - b;
            var atr = Blas.dot(res, A);
            var atb = Blas.dot(b, A);
            double num = 0, den = 0;
            for (int i = 0; i < atr.N; i++) num += (double)atr[i] * (double)atr[i];
            for (int i = 0; i < atb.N; i++) den += (double)atb[i] * (double)atb[i];
            return math.sqrt(num) / math.sqrt(math.max(den, 1e-30));
        }

        // ==== block-matrix builders ====================================================================

        static void BuildBlockSPDFloat(ref Arena arena, int nb, float density, uint seed, out floatMxN dense, out floatBSR sparse)
        {
            const int BR = SparseSolverFmt.BR;
            int dim = nb * BR;
            dense = arena.floatMat(dim, dim);
            var pairs = SparseSolverFmt.ChooseOffDiagPairsSymmetric(nb, density, seed, out int nnzb);
            var builder = arena.floatBSRBuilder(nb, nb, BR, BR, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);
            float strong = dim;
            float offScale = (float)0.3;

            for (int i = 0; i < nb; i++)
            {
                var Mi = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Mi[r, c] = rng.NextFloat(-1f, 1f);
                var Di = Blas.dot(Mi, Mi, true);
                for (int d = 0; d < BR; d++) Di[d, d] += strong;

                builder.AddBlock(i, i, in Di);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[i * BR + r, i * BR + c] = Di[r, c];
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextFloat(-offScale, offScale);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];

                var blockT = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        blockT[r, c] = block[c, r];

                builder.AddBlock(bj, bi, in blockT);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bj * BR + r, bi * BR + c] = blockT[r, c];
            }

            sparse = builder.ToBSR(ref arena);
        }

        // Same recipe as BuildBlockSPDFloat (identical rng sequence), but assembles TWO block-CSR encodings
        // of the SAME dense SPD matrix side by side: `full` (every stored block, incl. the explicit mirrored
        // lower block) and `sym` (upper-triangle + diagonal ONLY, via ToBSRSymmetric). Used by Section 0b to
        // isolate the symmetric-storage spMV win on a byte-for-byte identical matrix.
        static void BuildBlockSPDPairFloat(ref Arena arena, int nb, float density, uint seed,
                                            out floatMxN dense, out floatBSR full, out floatBSR sym)
        {
            const int BR = SparseSolverFmt.BR;
            int dim = nb * BR;
            dense = arena.floatMat(dim, dim);
            var pairs = SparseSolverFmt.ChooseOffDiagPairsSymmetric(nb, density, seed, out int nnzbFull);
            int nnzbSym = nb + pairs.Count;
            var fullBuilder = arena.floatBSRBuilder(nb, nb, BR, BR, nnzbFull);
            var symBuilder = arena.floatBSRBuilder(nb, nb, BR, BR, nnzbSym);
            var rng = new Random(seed ^ 0x9E3779B9u);
            float strong = dim;
            float offScale = (float)0.3;

            for (int i = 0; i < nb; i++)
            {
                var Mi = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Mi[r, c] = rng.NextFloat(-1f, 1f);
                var Di = Blas.dot(Mi, Mi, true);
                for (int d = 0; d < BR; d++) Di[d, d] += strong;

                fullBuilder.AddBlock(i, i, in Di);
                symBuilder.AddBlock(i, i, in Di);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[i * BR + r, i * BR + c] = Di[r, c];
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextFloat(-offScale, offScale);

                fullBuilder.AddBlock(bi, bj, in block);
                symBuilder.AddBlock(bi, bj, in block);   // upper only -- sym never sees the mirror
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];

                var blockT = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        blockT[r, c] = block[c, r];

                fullBuilder.AddBlock(bj, bi, in blockT); // mirrored lower block -- FULL storage only
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bj * BR + r, bi * BR + c] = blockT[r, c];
            }

            full = fullBuilder.ToBSR(ref arena);
            sym = symBuilder.ToBSRSymmetric(ref arena);
        }

        static void BuildBlockNonSymFloat(ref Arena arena, int nb, float density, uint seed, out floatMxN dense, out floatBSR sparse)
        {
            const int BR = SparseSolverFmt.BR;
            int dim = nb * BR;
            dense = arena.floatMat(dim, dim);
            var pairs = SparseSolverFmt.ChooseOffDiagPairsAsymmetric(nb, density, seed, out int nnzb);
            var builder = arena.floatBSRBuilder(nb, nb, BR, BR, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);
            float strong = dim;
            float offScale = (float)0.3;

            for (int i = 0; i < nb; i++)
            {
                var Mi = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Mi[r, c] = rng.NextFloat(-1f, 1f);
                var Di = Blas.dot(Mi, Mi, true);
                for (int d = 0; d < BR; d++) Di[d, d] += strong;

                builder.AddBlock(i, i, in Di);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[i * BR + r, i * BR + c] = Di[r, c];
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextFloat(-offScale, offScale);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];
            }

            sparse = builder.ToBSR(ref arena);
        }

        static void BuildBlockRectFloat(ref Arena arena, int mb, int nb, float density, uint seed, out floatMxN dense, out floatBSR sparse)
        {
            const int BR = SparseSolverFmt.BR;
            int rows = mb * BR, cols = nb * BR;
            dense = arena.floatMat(rows, cols);
            int diagCount = math.min(mb, nb);
            var pairs = SparseSolverFmt.ChooseOffDiagPairsRect(mb, nb, density, seed, out int nnzb);
            var builder = arena.floatBSRBuilder(mb, nb, BR, BR, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);

            for (int i = 0; i < diagCount; i++)
            {
                var block = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = (r == c ? (float)2 : (float)0) + rng.NextFloat(-(float)0.2, (float)0.2);

                builder.AddBlock(i, i, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[i * BR + r, i * BR + c] = block[r, c];
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = arena.floatMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextFloat(-(float)0.3, (float)0.3);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];
            }

            sparse = builder.ToBSR(ref arena);
        }

        // Block-matrix builder, parameterized block size (used ONLY by the dedicated b=4/N=1024 Section 1x --
        // Section 1's builders keep BR hardcoded since their numbers are cited in docs). Same recipe as
        // BuildBlockSPDFloat, generalized so the block size isn't tied to the file-wide BR=3 constant.
        static void BuildBlockSPDFloatSized(ref Arena arena, int nb, int br, float density, uint seed, out floatMxN dense, out floatBSR sparse)
        {
            int dim = nb * br;
            dense = arena.floatMat(dim, dim);
            var pairs = SparseSolverFmt.ChooseOffDiagPairsSymmetric(nb, density, seed, out int nnzb);
            var builder = arena.floatBSRBuilder(nb, nb, br, br, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);
            float strong = dim;
            float offScale = (float)0.3;

            for (int i = 0; i < nb; i++)
            {
                var Mi = arena.floatMat(br, br);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        Mi[r, c] = rng.NextFloat(-1f, 1f);
                var Di = Blas.dot(Mi, Mi, true);
                for (int d = 0; d < br; d++) Di[d, d] += strong;

                builder.AddBlock(i, i, in Di);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        dense[i * br + r, i * br + c] = Di[r, c];
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = arena.floatMat(br, br);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        block[r, c] = rng.NextFloat(-offScale, offScale);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        dense[bi * br + r, bj * br + c] = block[r, c];

                var blockT = arena.floatMat(br, br);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        blockT[r, c] = block[c, r];

                builder.AddBlock(bj, bi, in blockT);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        dense[bj * br + r, bi * br + c] = blockT[r, c];
            }

            sparse = builder.ToBSR(ref arena);
        }

        // ==== Section 0: operator matvec throughput (dense GEMV vs sparse spMV) =========================

        static void Section0Float(StringBuilder sb)
        {
            int BR = SparseSolverFmt.BR, REPS = SparseSolverFmt.REPS_MATVEC;
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 0. Operator matvec throughput (b={0}): dense GEMV vs sparse spMV, REPS={1} [float] ---", BR, REPS));
            sb.AppendLine(SparseSolverFmt.MatvecHeader());

            foreach (var n in SparseSolverFmt.BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in SparseSolverFmt.Densities)
                {
                    var arena = new Arena(Allocator.Persistent);
                    BuildBlockSPDFloat(ref arena, nb, density, SparseSolverFmt.Seed(n, density, 91), out var dense, out var sparse);
                    uint sx = SparseSolverFmt.Seed(n, density, 92);

                    var xd = arena.floatRandomVec(n, -1f, 1f, sx);   // ping-pong clobbers input -> fresh copy per timing
                    var yd = arena.floatVec(n);
                    var denseJob = new MatvecDenseJobFloat { A = dense, x = xd, y = yd, reps = REPS };
                    var denseStat = Bench.Time(() => denseJob.Run());
                    sb.AppendLine(SparseSolverFmt.MatvecRow("float", n, density, "GEMV-dense", denseStat, 1.0, null));

                    var xs = arena.floatRandomVec(n, -1f, 1f, sx);   // identical contents to xd
                    var ys = arena.floatVec(n);
                    var sparseJob = new MatvecSparseJobFloat { A = sparse, x = xs, y = ys, reps = REPS };
                    var sparseStat = Bench.Time(() => sparseJob.Run());

                    // clean single-matvec numerical cross-check (untimed; identical input)
                    var xc = arena.floatRandomVec(n, -1f, 1f, sx);
                    var yDc = Blas.dot(dense, xc);
                    var ySc = BSR.spMV(sparse, xc);
                    double md = 0;
                    for (int i = 0; i < n; i++) md = math.max(md, math.abs((double)yDc[i] - (double)ySc[i]));
                    double speedup = denseStat.Median / math.max(sparseStat.Median, 1e-30);
                    sb.AppendLine(SparseSolverFmt.MatvecRow("float", n, density, "spMV-sparse", sparseStat, speedup, md));

                    arena.Dispose();
                }
            }
        }

        // ==== Section 0b: symmetric-storage spMV vs full-storage spMV on the SAME SPD matrix ============
        //
        // Isolates the symmetric-storage half of the Milestone-A story: upper-triangle-only storage
        // (ToBSRSymmetric) vs full block-CSR, on the identical matrix (BuildBlockSPDPairFloat pins the rng
        // so `full` and `sym` encode byte-for-byte the same SPD system). bsrMatVecSym touches half as many
        // STORED blocks as bsrMatVec's full traversal -- expected ~2x at denser fill, less at sparse fill.

        static void Section0bFloat(StringBuilder sb)
        {
            int BR = SparseSolverFmt.BR, REPS = SparseSolverFmt.REPS_MATVEC;
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 0b. Symmetric-storage spMV vs full-storage spMV, SAME SPD matrix (b={0}), REPS={1} [float] ---", BR, REPS));
            sb.AppendLine(SparseSolverFmt.MatvecHeader());

            foreach (var n in SparseSolverFmt.BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in SparseSolverFmt.Densities)
                {
                    var arena = new Arena(Allocator.Persistent);
                    BuildBlockSPDPairFloat(ref arena, nb, density, SparseSolverFmt.Seed(n, density, 95), out _, out var full, out var sym);
                    uint sx = SparseSolverFmt.Seed(n, density, 96);

                    var xf = arena.floatRandomVec(n, -1f, 1f, sx);
                    var yf = arena.floatVec(n);
                    var fullJob = new MatvecSparseJobFloat { A = full, x = xf, y = yf, reps = REPS };
                    var fullStat = Bench.Time(() => fullJob.Run());
                    sb.AppendLine(SparseSolverFmt.MatvecRow("float", n, density, "spMV-full", fullStat, 1.0, null));

                    var xs = arena.floatRandomVec(n, -1f, 1f, sx);   // identical contents to xf
                    var ys = arena.floatVec(n);
                    var symJob = new MatvecSparseJobFloat { A = sym, x = xs, y = ys, reps = REPS };
                    var symStat = Bench.Time(() => symJob.Run());

                    // clean single-matvec numerical cross-check (untimed; identical input)
                    var xc = arena.floatRandomVec(n, -1f, 1f, sx);
                    var yFc = BSR.spMV(full, xc);
                    var ySc = BSR.spMV(sym, xc);
                    double md = 0;
                    for (int i = 0; i < n; i++) md = math.max(md, math.abs((double)yFc[i] - (double)ySc[i]));
                    double speedup = fullStat.Median / math.max(symStat.Median, 1e-30);
                    sb.AppendLine(SparseSolverFmt.MatvecRow("float", n, density, "spMV-sym", symStat, speedup, md));

                    arena.Dispose();
                }
            }
        }

        // ==== Section 1: SPD -> cg & minres ============================================================

        static void Section1Float(StringBuilder sb)
        {
            int BR = SparseSolverFmt.BR, K = SparseSolverFmt.K_CG;
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 1. SPD block-sparse (b={0}): cg & minres, K={1}, tol=0 [float] ---", BR, K));
            sb.AppendLine(SparseSolverFmt.RowHeader());

            foreach (var n in SparseSolverFmt.BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in SparseSolverFmt.Densities)
                {
                    var arena = new Arena(Allocator.Persistent);
                    BuildBlockSPDFloat(ref arena, nb, density, SparseSolverFmt.Seed(n, density, 11), out var dense, out var sparse);
                    var b = arena.floatRandomVec(n, -1f, 1f, SparseSolverFmt.Seed(n, density, 12));

                    var xCgD = arena.floatVec(n); var rCgD = arena.floatVec(n); var pCgD = arena.floatVec(n); var ApCgD = arena.floatVec(n);
                    var cgDenseJob = new CGDenseJobFloat { A = dense, b = b, x = xCgD, r = rCgD, p = pCgD, Ap = ApCgD, K = K };
                    var cgDenseStat = Bench.Time(() => cgDenseJob.Run());
                    sb.AppendLine(SparseSolverFmt.Row("float", n, density, "CG-dense", cgDenseStat, ResidualLinSys(in dense, in xCgD, in b)));

                    var xCgS = arena.floatVec(n); var rCgS = arena.floatVec(n); var pCgS = arena.floatVec(n); var ApCgS = arena.floatVec(n);
                    var cgSparseJob = new CGSparseJobFloat { A = sparse, b = b, x = xCgS, r = rCgS, p = pCgS, Ap = ApCgS, K = K };
                    var cgSparseStat = Bench.Time(() => cgSparseJob.Run());
                    sb.AppendLine(SparseSolverFmt.Row("float", n, density, "CG-sparse", cgSparseStat, ResidualLinSys(in dense, in xCgS, in b)));

                    var xMrD = arena.floatVec(n);
                    var yD = arena.floatVec(n); var r1D = arena.floatVec(n); var r2D = arena.floatVec(n); var vD = arena.floatVec(n);
                    var wD = arena.floatVec(n); var w1D = arena.floatVec(n); var w2D = arena.floatVec(n);
                    var mrDenseJob = new MinresDenseJobFloat { A = dense, b = b, x = xMrD, y = yD, r1 = r1D, r2 = r2D, v = vD, w = wD, w1 = w1D, w2 = w2D, K = K };
                    var mrDenseStat = Bench.Time(() => mrDenseJob.Run());
                    sb.AppendLine(SparseSolverFmt.Row("float", n, density, "MINRES-dense", mrDenseStat, ResidualLinSys(in dense, in xMrD, in b)));

                    var xMrS = arena.floatVec(n);
                    var yS = arena.floatVec(n); var r1S = arena.floatVec(n); var r2S = arena.floatVec(n); var vS = arena.floatVec(n);
                    var wS = arena.floatVec(n); var w1S = arena.floatVec(n); var w2S = arena.floatVec(n);
                    var mrSparseJob = new MinresSparseJobFloat { A = sparse, b = b, x = xMrS, y = yS, r1 = r1S, r2 = r2S, v = vS, w = wS, w1 = w1S, w2 = w2S, K = K };
                    var mrSparseStat = Bench.Time(() => mrSparseJob.Run());
                    sb.AppendLine(SparseSolverFmt.Row("float", n, density, "MINRES-sparse", mrSparseStat, ResidualLinSys(in dense, in xMrS, in b)));

                    arena.Dispose();
                }
            }
        }

        // ==== Section 1x: SPD block-sparse, b=4, N=1024 (CG only, 7% fill) ==============================
        //
        // The b=3 sweep (Section 1) tops out at N=768 because 1024 isn't divisible by 3. b=4 IS one of the
        // compile-time-unrolled bsrMatVecB4 kernel sizes, so nb=256 blocks of 4x4 gives a genuine 1024x1024
        // CG dense-vs-sparse comparison at the same convention (7% block density, K=40, tol=0). Only CG is
        // timed here -- this subsection backfills the README's 1024x1024 CG row.

        static void Section1xFloat(StringBuilder sb)
        {
            int N = SparseSolverFmt.N_B4, BR4 = SparseSolverFmt.BR4, NB = SparseSolverFmt.NB_B4, K = SparseSolverFmt.K_CG;
            float density = SparseSolverFmt.Densities[0]; // 7%
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 1x. SPD block-sparse (b={0}, N={1}): cg, K={2}, tol=0 [float] ---", BR4, N, K));
            sb.AppendLine(SparseSolverFmt.RowHeader());

            var arena = new Arena(Allocator.Persistent);
            BuildBlockSPDFloatSized(ref arena, NB, BR4, density, SparseSolverFmt.Seed(N, density, 111), out var dense, out var sparse);
            var b = arena.floatRandomVec(N, -1f, 1f, SparseSolverFmt.Seed(N, density, 112));

            var xCgD = arena.floatVec(N); var rCgD = arena.floatVec(N); var pCgD = arena.floatVec(N); var ApCgD = arena.floatVec(N);
            var cgDenseJob = new CGDenseJobFloat { A = dense, b = b, x = xCgD, r = rCgD, p = pCgD, Ap = ApCgD, K = K };
            var cgDenseStat = Bench.Time(() => cgDenseJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("float", N, density, "CG-dense", cgDenseStat, ResidualLinSys(in dense, in xCgD, in b)));

            var xCgS = arena.floatVec(N); var rCgS = arena.floatVec(N); var pCgS = arena.floatVec(N); var ApCgS = arena.floatVec(N);
            var cgSparseJob = new CGSparseJobFloat { A = sparse, b = b, x = xCgS, r = rCgS, p = pCgS, Ap = ApCgS, K = K };
            var cgSparseStat = Bench.Time(() => cgSparseJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("float", N, density, "CG-sparse", cgSparseStat, ResidualLinSys(in dense, in xCgS, in b)));

            arena.Dispose();
        }

        // ==== Section 2: non-symmetric -> biCGStab =====================================================

        static void Section2Float(StringBuilder sb)
        {
            int BR = SparseSolverFmt.BR, K = SparseSolverFmt.K_BICGSTAB;
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 2. Non-symmetric block-sparse (b={0}): biCGStab, K={1}, tol=0 [float] ---", BR, K));
            sb.AppendLine(SparseSolverFmt.RowHeader());

            foreach (var n in SparseSolverFmt.BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in SparseSolverFmt.Densities)
                {
                    var arena = new Arena(Allocator.Persistent);
                    BuildBlockNonSymFloat(ref arena, nb, density, SparseSolverFmt.Seed(n, density, 21), out var dense, out var sparse);
                    var b = arena.floatRandomVec(n, -1f, 1f, SparseSolverFmt.Seed(n, density, 22));

                    var xD = arena.floatVec(n); var rD = arena.floatVec(n); var rh0D = arena.floatVec(n);
                    var pD = arena.floatVec(n); var vD = arena.floatVec(n); var tD = arena.floatVec(n);
                    var jobD = new BiCGStabDenseJobFloat { A = dense, b = b, x = xD, r = rD, rHat0 = rh0D, p = pD, v = vD, t = tD, K = K };
                    var statD = Bench.Time(() => jobD.Run());
                    sb.AppendLine(SparseSolverFmt.Row("float", n, density, "BiCGStab-dense", statD, ResidualLinSys(in dense, in xD, in b)));

                    var xS = arena.floatVec(n); var rS = arena.floatVec(n); var rh0S = arena.floatVec(n);
                    var pS = arena.floatVec(n); var vS = arena.floatVec(n); var tS = arena.floatVec(n);
                    var jobS = new BiCGStabSparseJobFloat { A = sparse, b = b, x = xS, r = rS, rHat0 = rh0S, p = pS, v = vS, t = tS, K = K };
                    var statS = Bench.Time(() => jobS.Run());
                    sb.AppendLine(SparseSolverFmt.Row("float", n, density, "BiCGStab-sparse", statS, ResidualLinSys(in dense, in xS, in b)));

                    arena.Dispose();
                }
            }
        }

        // ==== Section 3: rectangular -> cgls & lsqr (over- and under-determined) ========================

        static void RunRectCaseFloat(int nRef, int mb, int nb, float density, string tag, int tagSeed, StringBuilder sb)
        {
            int BR = SparseSolverFmt.BR, K = SparseSolverFmt.K_LS;
            var arena = new Arena(Allocator.Persistent);
            BuildBlockRectFloat(ref arena, mb, nb, density, SparseSolverFmt.Seed(nRef, density, tagSeed), out var dense, out var sparse);
            int rows = mb * BR, cols = nb * BR;
            var b = arena.floatRandomVec(rows, -1f, 1f, SparseSolverFmt.Seed(nRef, density, tagSeed + 1));

            var xCD = arena.floatVec(cols); var rCD = arena.floatVec(rows); var sCD = arena.floatVec(cols); var pCD = arena.floatVec(cols); var qCD = arena.floatVec(rows);
            var cglsDenseJob = new CglsDenseJobFloat { A = dense, b = b, x = xCD, r = rCD, s = sCD, p = pCD, q = qCD, K = K };
            var cglsDenseStat = Bench.Time(() => cglsDenseJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("float", nRef, density, "CGLS-dense-" + tag, cglsDenseStat, ResidualLS(in dense, in xCD, in b)));

            var xCS = arena.floatVec(cols); var rCS = arena.floatVec(rows); var sCS = arena.floatVec(cols); var pCS = arena.floatVec(cols); var qCS = arena.floatVec(rows);
            var cglsSparseJob = new CglsSparseJobFloat { A = sparse, b = b, x = xCS, r = rCS, s = sCS, p = pCS, q = qCS, K = K };
            var cglsSparseStat = Bench.Time(() => cglsSparseJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("float", nRef, density, "CGLS-sparse-" + tag, cglsSparseStat, ResidualLS(in dense, in xCS, in b)));

            var xLD = arena.floatVec(cols); var uLD = arena.floatVec(rows); var vLD = arena.floatVec(cols); var wLD = arena.floatVec(cols);
            var tmMLD = arena.floatVec(rows); var tmNLD = arena.floatVec(cols);
            var lsqrDenseJob = new LsqrDenseJobFloat { A = dense, b = b, x = xLD, u = uLD, v = vLD, w = wLD, tmpM = tmMLD, tmpN = tmNLD, K = K };
            var lsqrDenseStat = Bench.Time(() => lsqrDenseJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("float", nRef, density, "LSQR-dense-" + tag, lsqrDenseStat, ResidualLS(in dense, in xLD, in b)));

            var xLS = arena.floatVec(cols); var uLS = arena.floatVec(rows); var vLS = arena.floatVec(cols); var wLS = arena.floatVec(cols);
            var tmMLS = arena.floatVec(rows); var tmNLS = arena.floatVec(cols);
            var lsqrSparseJob = new LsqrSparseJobFloat { A = sparse, b = b, x = xLS, u = uLS, v = vLS, w = wLS, tmpM = tmMLS, tmpN = tmNLS, K = K };
            var lsqrSparseStat = Bench.Time(() => lsqrSparseJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("float", nRef, density, "LSQR-sparse-" + tag, lsqrSparseStat, ResidualLS(in dense, in xLS, in b)));

            // Milestone B: transpose-optimized variants -- Aᵀ materialized ONCE (outside timing), ApplyT
            // becomes a forward spMV over Aᵀ. Compare "sparseT" rows against the "sparse" rows above.
            var AT = arena.floatBSRTranspose(in sparse);

            var xCST = arena.floatVec(cols); var rCST = arena.floatVec(rows); var sCST = arena.floatVec(cols); var pCST = arena.floatVec(cols); var qCST = arena.floatVec(rows);
            var cglsSparseTJob = new CglsSparseTJobFloat { A = sparse, AT = AT, b = b, x = xCST, r = rCST, s = sCST, p = pCST, q = qCST, K = K };
            var cglsSparseTStat = Bench.Time(() => cglsSparseTJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("float", nRef, density, "CGLS-sparseT-" + tag, cglsSparseTStat, ResidualLS(in dense, in xCST, in b)));

            var xLST = arena.floatVec(cols); var uLST = arena.floatVec(rows); var vLST = arena.floatVec(cols); var wLST = arena.floatVec(cols);
            var tmMLST = arena.floatVec(rows); var tmNLST = arena.floatVec(cols);
            var lsqrSparseTJob = new LsqrSparseTJobFloat { A = sparse, AT = AT, b = b, x = xLST, u = uLST, v = vLST, w = wLST, tmpM = tmMLST, tmpN = tmNLST, K = K };
            var lsqrSparseTStat = Bench.Time(() => lsqrSparseTJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("float", nRef, density, "LSQR-sparseT-" + tag, lsqrSparseTStat, ResidualLS(in dense, in xLST, in b)));

            arena.Dispose();
        }

        static void Section3Float(StringBuilder sb)
        {
            int BR = SparseSolverFmt.BR, K = SparseSolverFmt.K_LS;
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 3. Rectangular block-sparse (b={0}): cgls & lsqr, K={1}, tol=0 [float] ---", BR, K));
            sb.AppendLine("    over: rows=2xcols block grid (m=2n, overdetermined); under: cols=2xrows block grid (m=n/2, underdetermined).");
            sb.AppendLine("    residual = ||A^T(Ax-b)|| / ||A^T b|| (least-squares optimality, not ||Ax-b||).");
            sb.AppendLine(SparseSolverFmt.RowHeader());

            foreach (var n in SparseSolverFmt.BlockSizesN)
            {
                int nb0 = n / BR;
                foreach (var density in SparseSolverFmt.Densities)
                {
                    RunRectCaseFloat(n, 2 * nb0, nb0, density, "over", 31, sb);
                    RunRectCaseFloat(n, nb0, 2 * nb0, density, "under", 33, sb);
                }
            }
        }

        // ==== Section 4: zero-cost-abstraction probe (THE fork datapoint) ===============================
        //
        // Generic Krylov.cg(in floatMxN,...) -- which internally wraps A in floatDenseOperator and calls
        // the generic cg<TOp> loop -- vs a hand-inlined dense CG written directly against raw pointers in
        // CGHandInlinedJobFloat, no operator interface, no generic dispatch. Same matrix, same K, same
        // algorithm. If Burst fully inlines the generic operator call the two times are ~equal (ratio ~1) --
        // the operator abstraction is then free and there is no perf case for forking dense/sparse solver
        // bodies. A material gap would argue otherwise.

        static void Section4Float(StringBuilder sb)
        {
            int K = SparseSolverFmt.K_CG;
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 4. Zero-cost-abstraction probe (dense SPD, K={0}, tol=0) [float] ---", K));
            sb.AppendLine("    generic = Krylov.cg(in floatMxN,...) via cg<floatDenseOperator>;");
            sb.AppendLine("    hand-inlined = raw-pointer GEMV/axpy/dot written directly in the job, no operator interface.");
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-16} {3,11} {4,11} {5,10}",
                "dtype", "N", "path", "med(ms)", "min(ms)", "ratio"));

            foreach (var n in SparseSolverFmt.BlockSizesN)
            {
                var arena = new Arena(Allocator.Persistent);
                var M = arena.floatRandomMat(n, n, -1f, 1f, SparseSolverFmt.Seed(n, 0f, 41));
                var A = Blas.dot(M, M, true);
                for (int d = 0; d < n; d++) A[d, d] += n;
                var b = arena.floatRandomVec(n, -1f, 1f, SparseSolverFmt.Seed(n, 0f, 42));

                var xG = arena.floatVec(n); var rG = arena.floatVec(n); var pG = arena.floatVec(n); var ApG = arena.floatVec(n);
                var genericJob = new CGDenseJobFloat { A = A, b = b, x = xG, r = rG, p = pG, Ap = ApG, K = K };
                var genericStat = Bench.Time(() => genericJob.Run());

                var xH = arena.floatVec(n); var rH = arena.floatVec(n); var pH = arena.floatVec(n); var ApH = arena.floatVec(n);
                var handJob = new CGHandInlinedJobFloat { A = A, b = b, x = xH, r = rH, p = pH, Ap = ApH, K = K };
                var handStat = Bench.Time(() => handJob.Run());

                double ratio = genericStat.Median / math.max(handStat.Median, 1e-9);
                sb.AppendLine(string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-16} {3,11:F4} {4,11:F4} {5,10:F3}",
                    "float", n, "generic", genericStat.Median, genericStat.Min, ratio));
                sb.AppendLine(string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-16} {3,11:F4} {4,11:F4} {5,10}",
                    "float", n, "hand-inlined", handStat.Median, handStat.Min, "--"));

                arena.Dispose();
            }
        }
    }
}
