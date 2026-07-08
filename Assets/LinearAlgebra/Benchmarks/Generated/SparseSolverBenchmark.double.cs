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
    // (double) casts so the double build gets the true double value rather than a widened float literal.

    // ---- CG scratch: r, p, Ap (all A.Rows length) --------------------------------------------------

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CGDenseJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b, x, r, p, Ap;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.cg(in A, in b, ref x, ref r, ref p, ref Ap, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CGSparseJobDouble : IJob
    {
        public doubleBSR A;
        public doubleN b, x, r, p, Ap;
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
    public struct MinresDenseJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b, x, y, r1, r2, v, w, w1, w2;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct MinresSparseJobDouble : IJob
    {
        public doubleBSR A;
        public doubleN b, x, y, r1, r2, v, w, w1, w2;
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
    public struct BiCGStabDenseJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b, x, r, rHat0, p, v, t;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct BiCGStabSparseJobDouble : IJob
    {
        public doubleBSR A;
        public doubleN b, x, r, rHat0, p, v, t;
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
    public struct CglsDenseJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b, x, r, s, p, q;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.cgls(in A, in b, ref x, ref r, ref s, ref p, ref q, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct CglsSparseJobDouble : IJob
    {
        public doubleBSR A;
        public doubleN b, x, r, s, p, q;
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
    public struct LsqrDenseJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b, x, u, v, w, tmpM, tmpN;
        public int K;

        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LsqrSparseJobDouble : IJob
    {
        public doubleBSR A;
        public doubleN b, x, u, v, w, tmpM, tmpN;
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
    public struct CglsSparseTJobDouble : IJob
    {
        public doubleBSR A, AT;
        public doubleN b, x, r, s, p, q;
        public int K;
        public void Execute()
        {
            int n = x.N;
            for (int i = 0; i < n; i++) x[i] = 0f;
            Krylov.cgls(in A, in AT, in b, ref x, ref r, ref s, ref p, ref q, K, 0f);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LsqrSparseTJobDouble : IJob
    {
        public doubleBSR A, AT;
        public doubleN b, x, u, v, w, tmpM, tmpN;
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
    public struct CGHandInlinedJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN b, x, r, p, Ap;
        public int K;

        public unsafe void Execute()
        {
            int n = x.N;
            double* Ad = A.Data.Ptr;
            double* bd = b.Data.Ptr;
            double* xd = x.Data.Ptr;
            double* rd = r.Data.Ptr;
            double* pd = p.Data.Ptr;
            double* Apd = Ap.Data.Ptr;

            for (int i = 0; i < n; i++) xd[i] = 0f;
            for (int i = 0; i < n; i++) rd[i] = bd[i];       // r = b - A*0 = b
            for (int i = 0; i < n; i++) pd[i] = rd[i];       // p = r

            double rsold = 0f;
            for (int i = 0; i < n; i++) rsold += rd[i] * rd[i];

            for (int k = 0; k < K; k++)
            {
                for (int row = 0; row < n; row++)
                {
                    double sum = 0f;
                    int baseIdx = row * n;
                    for (int col = 0; col < n; col++)
                        sum += Ad[baseIdx + col] * pd[col];
                    Apd[row] = sum;
                }

                double pAp = 0f;
                for (int i = 0; i < n; i++) pAp += pd[i] * Apd[i];
                if (!(pAp > 0f)) break;

                double alpha = rsold / pAp;
                for (int i = 0; i < n; i++) xd[i] += alpha * pd[i];
                for (int i = 0; i < n; i++) rd[i] -= alpha * Apd[i];

                double rsnew = 0f;
                for (int i = 0; i < n; i++) rsnew += rd[i] * rd[i];

                double beta = rsnew / rsold;
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
    public struct MatvecDenseJobDouble : IJob
    {
        public doubleMxN A;
        public doubleN x, y;
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
    public struct MatvecSparseJobDouble : IJob
    {
        public doubleBSR A;
        public doubleN x, y;
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

        static double ResidualLinSys(in doubleMxN A, in doubleN x, in doubleN b)
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
        static double ResidualLS(in doubleMxN A, in doubleN x, in doubleN b)
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

        static void BuildBlockSPDDouble(ref Arena arena, int nb, float density, uint seed, out doubleMxN dense, out doubleBSR sparse)
        {
            const int BR = SparseSolverFmt.BR;
            int dim = nb * BR;
            dense = arena.doubleMat(dim, dim);
            var pairs = SparseSolverFmt.ChooseOffDiagPairsSymmetric(nb, density, seed, out int nnzb);
            var builder = arena.doubleBSRBuilder(nb, nb, BR, BR, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);
            double strong = dim;
            double offScale = (double)0.3;

            for (int i = 0; i < nb; i++)
            {
                var Mi = arena.doubleMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Mi[r, c] = rng.NextDouble(-1f, 1f);
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
                var block = arena.doubleMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextDouble(-offScale, offScale);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];

                var blockT = arena.doubleMat(BR, BR);
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

        // Same recipe as BuildBlockSPDDouble (identical rng sequence), but assembles TWO block-CSR encodings
        // of the SAME dense SPD matrix side by side: `full` (every stored block, incl. the explicit mirrored
        // lower block) and `sym` (upper-triangle + diagonal ONLY, via ToBSRSymmetric). Used by Section 0b to
        // isolate the symmetric-storage spMV win on a byte-for-byte identical matrix.
        static void BuildBlockSPDPairDouble(ref Arena arena, int nb, float density, uint seed,
                                            out doubleMxN dense, out doubleBSR full, out doubleBSR sym)
        {
            const int BR = SparseSolverFmt.BR;
            int dim = nb * BR;
            dense = arena.doubleMat(dim, dim);
            var pairs = SparseSolverFmt.ChooseOffDiagPairsSymmetric(nb, density, seed, out int nnzbFull);
            int nnzbSym = nb + pairs.Count;
            var fullBuilder = arena.doubleBSRBuilder(nb, nb, BR, BR, nnzbFull);
            var symBuilder = arena.doubleBSRBuilder(nb, nb, BR, BR, nnzbSym);
            var rng = new Random(seed ^ 0x9E3779B9u);
            double strong = dim;
            double offScale = (double)0.3;

            for (int i = 0; i < nb; i++)
            {
                var Mi = arena.doubleMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Mi[r, c] = rng.NextDouble(-1f, 1f);
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
                var block = arena.doubleMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextDouble(-offScale, offScale);

                fullBuilder.AddBlock(bi, bj, in block);
                symBuilder.AddBlock(bi, bj, in block);   // upper only -- sym never sees the mirror
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];

                var blockT = arena.doubleMat(BR, BR);
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

        static void BuildBlockNonSymDouble(ref Arena arena, int nb, float density, uint seed, out doubleMxN dense, out doubleBSR sparse)
        {
            const int BR = SparseSolverFmt.BR;
            int dim = nb * BR;
            dense = arena.doubleMat(dim, dim);
            var pairs = SparseSolverFmt.ChooseOffDiagPairsAsymmetric(nb, density, seed, out int nnzb);
            var builder = arena.doubleBSRBuilder(nb, nb, BR, BR, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);
            double strong = dim;
            double offScale = (double)0.3;

            for (int i = 0; i < nb; i++)
            {
                var Mi = arena.doubleMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        Mi[r, c] = rng.NextDouble(-1f, 1f);
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
                var block = arena.doubleMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextDouble(-offScale, offScale);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];
            }

            sparse = builder.ToBSR(ref arena);
        }

        static void BuildBlockRectDouble(ref Arena arena, int mb, int nb, float density, uint seed, out doubleMxN dense, out doubleBSR sparse)
        {
            const int BR = SparseSolverFmt.BR;
            int rows = mb * BR, cols = nb * BR;
            dense = arena.doubleMat(rows, cols);
            int diagCount = math.min(mb, nb);
            var pairs = SparseSolverFmt.ChooseOffDiagPairsRect(mb, nb, density, seed, out int nnzb);
            var builder = arena.doubleBSRBuilder(mb, nb, BR, BR, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);

            for (int i = 0; i < diagCount; i++)
            {
                var block = arena.doubleMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = (r == c ? (double)2 : (double)0) + rng.NextDouble(-(double)0.2, (double)0.2);

                builder.AddBlock(i, i, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[i * BR + r, i * BR + c] = block[r, c];
            }

            foreach (var pos in pairs)
            {
                int bi = pos.Bi, bj = pos.Bj;
                var block = arena.doubleMat(BR, BR);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        block[r, c] = rng.NextDouble(-(double)0.3, (double)0.3);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        dense[bi * BR + r, bj * BR + c] = block[r, c];
            }

            sparse = builder.ToBSR(ref arena);
        }

        // Block-matrix builder, parameterized block size (used ONLY by the dedicated b=4/N=1024 Section 1x --
        // Section 1's builders keep BR hardcoded since their numbers are cited in docs). Same recipe as
        // BuildBlockSPDDouble, generalized so the block size isn't tied to the file-wide BR=3 constant.
        static void BuildBlockSPDDoubleSized(ref Arena arena, int nb, int br, float density, uint seed, out doubleMxN dense, out doubleBSR sparse)
        {
            int dim = nb * br;
            dense = arena.doubleMat(dim, dim);
            var pairs = SparseSolverFmt.ChooseOffDiagPairsSymmetric(nb, density, seed, out int nnzb);
            var builder = arena.doubleBSRBuilder(nb, nb, br, br, nnzb);
            var rng = new Random(seed ^ 0x9E3779B9u);
            double strong = dim;
            double offScale = (double)0.3;

            for (int i = 0; i < nb; i++)
            {
                var Mi = arena.doubleMat(br, br);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        Mi[r, c] = rng.NextDouble(-1f, 1f);
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
                var block = arena.doubleMat(br, br);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        block[r, c] = rng.NextDouble(-offScale, offScale);

                builder.AddBlock(bi, bj, in block);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        dense[bi * br + r, bj * br + c] = block[r, c];

                var blockT = arena.doubleMat(br, br);
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

        static void Section0Double(StringBuilder sb)
        {
            int BR = SparseSolverFmt.BR, REPS = SparseSolverFmt.REPS_MATVEC;
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 0. Operator matvec throughput (b={0}): dense GEMV vs sparse spMV, REPS={1} [double] ---", BR, REPS));
            sb.AppendLine(SparseSolverFmt.MatvecHeader());

            foreach (var n in SparseSolverFmt.BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in SparseSolverFmt.Densities)
                {
                    var arena = new Arena(Allocator.Persistent);
                    BuildBlockSPDDouble(ref arena, nb, density, SparseSolverFmt.Seed(n, density, 91), out var dense, out var sparse);
                    uint sx = SparseSolverFmt.Seed(n, density, 92);

                    var xd = arena.doubleRandomVec(n, -1f, 1f, sx);   // ping-pong clobbers input -> fresh copy per timing
                    var yd = arena.doubleVec(n);
                    var denseJob = new MatvecDenseJobDouble { A = dense, x = xd, y = yd, reps = REPS };
                    var denseStat = Bench.Time(() => denseJob.Run());
                    sb.AppendLine(SparseSolverFmt.MatvecRow("double", n, density, "GEMV-dense", denseStat, 1.0, null));

                    var xs = arena.doubleRandomVec(n, -1f, 1f, sx);   // identical contents to xd
                    var ys = arena.doubleVec(n);
                    var sparseJob = new MatvecSparseJobDouble { A = sparse, x = xs, y = ys, reps = REPS };
                    var sparseStat = Bench.Time(() => sparseJob.Run());

                    // clean single-matvec numerical cross-check (untimed; identical input)
                    var xc = arena.doubleRandomVec(n, -1f, 1f, sx);
                    var yDc = Blas.dot(dense, xc);
                    var ySc = BSR.spMV(sparse, xc);
                    double md = 0;
                    for (int i = 0; i < n; i++) md = math.max(md, math.abs((double)yDc[i] - (double)ySc[i]));
                    double speedup = denseStat.Median / math.max(sparseStat.Median, 1e-30);
                    sb.AppendLine(SparseSolverFmt.MatvecRow("double", n, density, "spMV-sparse", sparseStat, speedup, md));

                    arena.Dispose();
                }
            }
        }

        // ==== Section 0b: symmetric-storage spMV vs full-storage spMV on the SAME SPD matrix ============
        //
        // Isolates the symmetric-storage half of the Milestone-A story: upper-triangle-only storage
        // (ToBSRSymmetric) vs full block-CSR, on the identical matrix (BuildBlockSPDPairDouble pins the rng
        // so `full` and `sym` encode byte-for-byte the same SPD system). bsrMatVecSym touches half as many
        // STORED blocks as bsrMatVec's full traversal -- expected ~2x at denser fill, less at sparse fill.

        static void Section0bDouble(StringBuilder sb)
        {
            int BR = SparseSolverFmt.BR, REPS = SparseSolverFmt.REPS_MATVEC;
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 0b. Symmetric-storage spMV vs full-storage spMV, SAME SPD matrix (b={0}), REPS={1} [double] ---", BR, REPS));
            sb.AppendLine(SparseSolverFmt.MatvecHeader());

            foreach (var n in SparseSolverFmt.BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in SparseSolverFmt.Densities)
                {
                    var arena = new Arena(Allocator.Persistent);
                    BuildBlockSPDPairDouble(ref arena, nb, density, SparseSolverFmt.Seed(n, density, 95), out _, out var full, out var sym);
                    uint sx = SparseSolverFmt.Seed(n, density, 96);

                    var xf = arena.doubleRandomVec(n, -1f, 1f, sx);
                    var yf = arena.doubleVec(n);
                    var fullJob = new MatvecSparseJobDouble { A = full, x = xf, y = yf, reps = REPS };
                    var fullStat = Bench.Time(() => fullJob.Run());
                    sb.AppendLine(SparseSolverFmt.MatvecRow("double", n, density, "spMV-full", fullStat, 1.0, null));

                    var xs = arena.doubleRandomVec(n, -1f, 1f, sx);   // identical contents to xf
                    var ys = arena.doubleVec(n);
                    var symJob = new MatvecSparseJobDouble { A = sym, x = xs, y = ys, reps = REPS };
                    var symStat = Bench.Time(() => symJob.Run());

                    // clean single-matvec numerical cross-check (untimed; identical input)
                    var xc = arena.doubleRandomVec(n, -1f, 1f, sx);
                    var yFc = BSR.spMV(full, xc);
                    var ySc = BSR.spMV(sym, xc);
                    double md = 0;
                    for (int i = 0; i < n; i++) md = math.max(md, math.abs((double)yFc[i] - (double)ySc[i]));
                    double speedup = fullStat.Median / math.max(symStat.Median, 1e-30);
                    sb.AppendLine(SparseSolverFmt.MatvecRow("double", n, density, "spMV-sym", symStat, speedup, md));

                    arena.Dispose();
                }
            }
        }

        // ==== Section 1: SPD -> cg & minres ============================================================

        static void Section1Double(StringBuilder sb)
        {
            int BR = SparseSolverFmt.BR, K = SparseSolverFmt.K_CG;
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 1. SPD block-sparse (b={0}): cg & minres, K={1}, tol=0 [double] ---", BR, K));
            sb.AppendLine(SparseSolverFmt.RowHeader());

            foreach (var n in SparseSolverFmt.BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in SparseSolverFmt.Densities)
                {
                    var arena = new Arena(Allocator.Persistent);
                    BuildBlockSPDDouble(ref arena, nb, density, SparseSolverFmt.Seed(n, density, 11), out var dense, out var sparse);
                    var b = arena.doubleRandomVec(n, -1f, 1f, SparseSolverFmt.Seed(n, density, 12));

                    var xCgD = arena.doubleVec(n); var rCgD = arena.doubleVec(n); var pCgD = arena.doubleVec(n); var ApCgD = arena.doubleVec(n);
                    var cgDenseJob = new CGDenseJobDouble { A = dense, b = b, x = xCgD, r = rCgD, p = pCgD, Ap = ApCgD, K = K };
                    var cgDenseStat = Bench.Time(() => cgDenseJob.Run());
                    sb.AppendLine(SparseSolverFmt.Row("double", n, density, "CG-dense", cgDenseStat, ResidualLinSys(in dense, in xCgD, in b)));

                    var xCgS = arena.doubleVec(n); var rCgS = arena.doubleVec(n); var pCgS = arena.doubleVec(n); var ApCgS = arena.doubleVec(n);
                    var cgSparseJob = new CGSparseJobDouble { A = sparse, b = b, x = xCgS, r = rCgS, p = pCgS, Ap = ApCgS, K = K };
                    var cgSparseStat = Bench.Time(() => cgSparseJob.Run());
                    sb.AppendLine(SparseSolverFmt.Row("double", n, density, "CG-sparse", cgSparseStat, ResidualLinSys(in dense, in xCgS, in b)));

                    var xMrD = arena.doubleVec(n);
                    var yD = arena.doubleVec(n); var r1D = arena.doubleVec(n); var r2D = arena.doubleVec(n); var vD = arena.doubleVec(n);
                    var wD = arena.doubleVec(n); var w1D = arena.doubleVec(n); var w2D = arena.doubleVec(n);
                    var mrDenseJob = new MinresDenseJobDouble { A = dense, b = b, x = xMrD, y = yD, r1 = r1D, r2 = r2D, v = vD, w = wD, w1 = w1D, w2 = w2D, K = K };
                    var mrDenseStat = Bench.Time(() => mrDenseJob.Run());
                    sb.AppendLine(SparseSolverFmt.Row("double", n, density, "MINRES-dense", mrDenseStat, ResidualLinSys(in dense, in xMrD, in b)));

                    var xMrS = arena.doubleVec(n);
                    var yS = arena.doubleVec(n); var r1S = arena.doubleVec(n); var r2S = arena.doubleVec(n); var vS = arena.doubleVec(n);
                    var wS = arena.doubleVec(n); var w1S = arena.doubleVec(n); var w2S = arena.doubleVec(n);
                    var mrSparseJob = new MinresSparseJobDouble { A = sparse, b = b, x = xMrS, y = yS, r1 = r1S, r2 = r2S, v = vS, w = wS, w1 = w1S, w2 = w2S, K = K };
                    var mrSparseStat = Bench.Time(() => mrSparseJob.Run());
                    sb.AppendLine(SparseSolverFmt.Row("double", n, density, "MINRES-sparse", mrSparseStat, ResidualLinSys(in dense, in xMrS, in b)));

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

        static void Section1xDouble(StringBuilder sb)
        {
            int N = SparseSolverFmt.N_B4, BR4 = SparseSolverFmt.BR4, NB = SparseSolverFmt.NB_B4, K = SparseSolverFmt.K_CG;
            float density = SparseSolverFmt.Densities[0]; // 7%
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 1x. SPD block-sparse (b={0}, N={1}): cg, K={2}, tol=0 [double] ---", BR4, N, K));
            sb.AppendLine(SparseSolverFmt.RowHeader());

            var arena = new Arena(Allocator.Persistent);
            BuildBlockSPDDoubleSized(ref arena, NB, BR4, density, SparseSolverFmt.Seed(N, density, 111), out var dense, out var sparse);
            var b = arena.doubleRandomVec(N, -1f, 1f, SparseSolverFmt.Seed(N, density, 112));

            var xCgD = arena.doubleVec(N); var rCgD = arena.doubleVec(N); var pCgD = arena.doubleVec(N); var ApCgD = arena.doubleVec(N);
            var cgDenseJob = new CGDenseJobDouble { A = dense, b = b, x = xCgD, r = rCgD, p = pCgD, Ap = ApCgD, K = K };
            var cgDenseStat = Bench.Time(() => cgDenseJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("double", N, density, "CG-dense", cgDenseStat, ResidualLinSys(in dense, in xCgD, in b)));

            var xCgS = arena.doubleVec(N); var rCgS = arena.doubleVec(N); var pCgS = arena.doubleVec(N); var ApCgS = arena.doubleVec(N);
            var cgSparseJob = new CGSparseJobDouble { A = sparse, b = b, x = xCgS, r = rCgS, p = pCgS, Ap = ApCgS, K = K };
            var cgSparseStat = Bench.Time(() => cgSparseJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("double", N, density, "CG-sparse", cgSparseStat, ResidualLinSys(in dense, in xCgS, in b)));

            arena.Dispose();
        }

        // ==== Section 2: non-symmetric -> biCGStab =====================================================

        static void Section2Double(StringBuilder sb)
        {
            int BR = SparseSolverFmt.BR, K = SparseSolverFmt.K_BICGSTAB;
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 2. Non-symmetric block-sparse (b={0}): biCGStab, K={1}, tol=0 [double] ---", BR, K));
            sb.AppendLine(SparseSolverFmt.RowHeader());

            foreach (var n in SparseSolverFmt.BlockSizesN)
            {
                int nb = n / BR;
                foreach (var density in SparseSolverFmt.Densities)
                {
                    var arena = new Arena(Allocator.Persistent);
                    BuildBlockNonSymDouble(ref arena, nb, density, SparseSolverFmt.Seed(n, density, 21), out var dense, out var sparse);
                    var b = arena.doubleRandomVec(n, -1f, 1f, SparseSolverFmt.Seed(n, density, 22));

                    var xD = arena.doubleVec(n); var rD = arena.doubleVec(n); var rh0D = arena.doubleVec(n);
                    var pD = arena.doubleVec(n); var vD = arena.doubleVec(n); var tD = arena.doubleVec(n);
                    var jobD = new BiCGStabDenseJobDouble { A = dense, b = b, x = xD, r = rD, rHat0 = rh0D, p = pD, v = vD, t = tD, K = K };
                    var statD = Bench.Time(() => jobD.Run());
                    sb.AppendLine(SparseSolverFmt.Row("double", n, density, "BiCGStab-dense", statD, ResidualLinSys(in dense, in xD, in b)));

                    var xS = arena.doubleVec(n); var rS = arena.doubleVec(n); var rh0S = arena.doubleVec(n);
                    var pS = arena.doubleVec(n); var vS = arena.doubleVec(n); var tS = arena.doubleVec(n);
                    var jobS = new BiCGStabSparseJobDouble { A = sparse, b = b, x = xS, r = rS, rHat0 = rh0S, p = pS, v = vS, t = tS, K = K };
                    var statS = Bench.Time(() => jobS.Run());
                    sb.AppendLine(SparseSolverFmt.Row("double", n, density, "BiCGStab-sparse", statS, ResidualLinSys(in dense, in xS, in b)));

                    arena.Dispose();
                }
            }
        }

        // ==== Section 3: rectangular -> cgls & lsqr (over- and under-determined) ========================

        static void RunRectCaseDouble(int nRef, int mb, int nb, float density, string tag, int tagSeed, StringBuilder sb)
        {
            int BR = SparseSolverFmt.BR, K = SparseSolverFmt.K_LS;
            var arena = new Arena(Allocator.Persistent);
            BuildBlockRectDouble(ref arena, mb, nb, density, SparseSolverFmt.Seed(nRef, density, tagSeed), out var dense, out var sparse);
            int rows = mb * BR, cols = nb * BR;
            var b = arena.doubleRandomVec(rows, -1f, 1f, SparseSolverFmt.Seed(nRef, density, tagSeed + 1));

            var xCD = arena.doubleVec(cols); var rCD = arena.doubleVec(rows); var sCD = arena.doubleVec(cols); var pCD = arena.doubleVec(cols); var qCD = arena.doubleVec(rows);
            var cglsDenseJob = new CglsDenseJobDouble { A = dense, b = b, x = xCD, r = rCD, s = sCD, p = pCD, q = qCD, K = K };
            var cglsDenseStat = Bench.Time(() => cglsDenseJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("double", nRef, density, "CGLS-dense-" + tag, cglsDenseStat, ResidualLS(in dense, in xCD, in b)));

            var xCS = arena.doubleVec(cols); var rCS = arena.doubleVec(rows); var sCS = arena.doubleVec(cols); var pCS = arena.doubleVec(cols); var qCS = arena.doubleVec(rows);
            var cglsSparseJob = new CglsSparseJobDouble { A = sparse, b = b, x = xCS, r = rCS, s = sCS, p = pCS, q = qCS, K = K };
            var cglsSparseStat = Bench.Time(() => cglsSparseJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("double", nRef, density, "CGLS-sparse-" + tag, cglsSparseStat, ResidualLS(in dense, in xCS, in b)));

            var xLD = arena.doubleVec(cols); var uLD = arena.doubleVec(rows); var vLD = arena.doubleVec(cols); var wLD = arena.doubleVec(cols);
            var tmMLD = arena.doubleVec(rows); var tmNLD = arena.doubleVec(cols);
            var lsqrDenseJob = new LsqrDenseJobDouble { A = dense, b = b, x = xLD, u = uLD, v = vLD, w = wLD, tmpM = tmMLD, tmpN = tmNLD, K = K };
            var lsqrDenseStat = Bench.Time(() => lsqrDenseJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("double", nRef, density, "LSQR-dense-" + tag, lsqrDenseStat, ResidualLS(in dense, in xLD, in b)));

            var xLS = arena.doubleVec(cols); var uLS = arena.doubleVec(rows); var vLS = arena.doubleVec(cols); var wLS = arena.doubleVec(cols);
            var tmMLS = arena.doubleVec(rows); var tmNLS = arena.doubleVec(cols);
            var lsqrSparseJob = new LsqrSparseJobDouble { A = sparse, b = b, x = xLS, u = uLS, v = vLS, w = wLS, tmpM = tmMLS, tmpN = tmNLS, K = K };
            var lsqrSparseStat = Bench.Time(() => lsqrSparseJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("double", nRef, density, "LSQR-sparse-" + tag, lsqrSparseStat, ResidualLS(in dense, in xLS, in b)));

            // Milestone B: transpose-optimized variants -- Aᵀ materialized ONCE (outside timing), ApplyT
            // becomes a forward spMV over Aᵀ. Compare "sparseT" rows against the "sparse" rows above.
            var AT = arena.doubleBSRTranspose(in sparse);

            var xCST = arena.doubleVec(cols); var rCST = arena.doubleVec(rows); var sCST = arena.doubleVec(cols); var pCST = arena.doubleVec(cols); var qCST = arena.doubleVec(rows);
            var cglsSparseTJob = new CglsSparseTJobDouble { A = sparse, AT = AT, b = b, x = xCST, r = rCST, s = sCST, p = pCST, q = qCST, K = K };
            var cglsSparseTStat = Bench.Time(() => cglsSparseTJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("double", nRef, density, "CGLS-sparseT-" + tag, cglsSparseTStat, ResidualLS(in dense, in xCST, in b)));

            var xLST = arena.doubleVec(cols); var uLST = arena.doubleVec(rows); var vLST = arena.doubleVec(cols); var wLST = arena.doubleVec(cols);
            var tmMLST = arena.doubleVec(rows); var tmNLST = arena.doubleVec(cols);
            var lsqrSparseTJob = new LsqrSparseTJobDouble { A = sparse, AT = AT, b = b, x = xLST, u = uLST, v = vLST, w = wLST, tmpM = tmMLST, tmpN = tmNLST, K = K };
            var lsqrSparseTStat = Bench.Time(() => lsqrSparseTJob.Run());
            sb.AppendLine(SparseSolverFmt.Row("double", nRef, density, "LSQR-sparseT-" + tag, lsqrSparseTStat, ResidualLS(in dense, in xLST, in b)));

            arena.Dispose();
        }

        static void Section3Double(StringBuilder sb)
        {
            int BR = SparseSolverFmt.BR, K = SparseSolverFmt.K_LS;
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 3. Rectangular block-sparse (b={0}): cgls & lsqr, K={1}, tol=0 [double] ---", BR, K));
            sb.AppendLine("    over: rows=2xcols block grid (m=2n, overdetermined); under: cols=2xrows block grid (m=n/2, underdetermined).");
            sb.AppendLine("    residual = ||A^T(Ax-b)|| / ||A^T b|| (least-squares optimality, not ||Ax-b||).");
            sb.AppendLine(SparseSolverFmt.RowHeader());

            foreach (var n in SparseSolverFmt.BlockSizesN)
            {
                int nb0 = n / BR;
                foreach (var density in SparseSolverFmt.Densities)
                {
                    RunRectCaseDouble(n, 2 * nb0, nb0, density, "over", 31, sb);
                    RunRectCaseDouble(n, nb0, 2 * nb0, density, "under", 33, sb);
                }
            }
        }

        // ==== Section 4: zero-cost-abstraction probe (THE fork datapoint) ===============================
        //
        // Generic Krylov.cg(in doubleMxN,...) -- which internally wraps A in doubleDenseOperator and calls
        // the generic cg<TOp> loop -- vs a hand-inlined dense CG written directly against raw pointers in
        // CGHandInlinedJobDouble, no operator interface, no generic dispatch. Same matrix, same K, same
        // algorithm. If Burst fully inlines the generic operator call the two times are ~equal (ratio ~1) --
        // the operator abstraction is then free and there is no perf case for forking dense/sparse solver
        // bodies. A material gap would argue otherwise.

        static void Section4Double(StringBuilder sb)
        {
            int K = SparseSolverFmt.K_CG;
            sb.AppendLine();
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "--- 4. Zero-cost-abstraction probe (dense SPD, K={0}, tol=0) [double] ---", K));
            sb.AppendLine("    generic = Krylov.cg(in doubleMxN,...) via cg<doubleDenseOperator>;");
            sb.AppendLine("    hand-inlined = raw-pointer GEMV/axpy/dot written directly in the job, no operator interface.");
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-16} {3,11} {4,11} {5,10}",
                "dtype", "N", "path", "med(ms)", "min(ms)", "ratio"));

            foreach (var n in SparseSolverFmt.BlockSizesN)
            {
                var arena = new Arena(Allocator.Persistent);
                var M = arena.doubleRandomMat(n, n, -1f, 1f, SparseSolverFmt.Seed(n, 0f, 41));
                var A = Blas.dot(M, M, true);
                for (int d = 0; d < n; d++) A[d, d] += n;
                var b = arena.doubleRandomVec(n, -1f, 1f, SparseSolverFmt.Seed(n, 0f, 42));

                var xG = arena.doubleVec(n); var rG = arena.doubleVec(n); var pG = arena.doubleVec(n); var ApG = arena.doubleVec(n);
                var genericJob = new CGDenseJobDouble { A = A, b = b, x = xG, r = rG, p = pG, Ap = ApG, K = K };
                var genericStat = Bench.Time(() => genericJob.Run());

                var xH = arena.doubleVec(n); var rH = arena.doubleVec(n); var pH = arena.doubleVec(n); var ApH = arena.doubleVec(n);
                var handJob = new CGHandInlinedJobDouble { A = A, b = b, x = xH, r = rH, p = pH, Ap = ApH, K = K };
                var handStat = Bench.Time(() => handJob.Run());

                double ratio = genericStat.Median / math.max(handStat.Median, 1e-9);
                sb.AppendLine(string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-16} {3,11:F4} {4,11:F4} {5,10:F3}",
                    "double", n, "generic", genericStat.Median, genericStat.Min, ratio));
                sb.AppendLine(string.Format(CultureInfo.InvariantCulture, "{0,-7} {1,-6} {2,-16} {3,11:F4} {4,11:F4} {5,10}",
                    "double", n, "hand-inlined", handStat.Median, handStat.Min, "--"));

                arena.Dispose();
            }
        }
    }
}
