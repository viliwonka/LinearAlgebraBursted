using System;
using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// One-level Schwarz preconditioners: fProxyAdditiveSchwarz (symmetric AS, valid for pcg/pminres)
// and fProxyRestrictedSchwarz (RAS, non-symmetric, pbiCGStab only).
// Correctness anchors (spec docs/dev/spec-additive-schwarz-preconditioner.md, section Tests):
//   (1) EXACTNESS: one subdomain covering the whole matrix (subdomainSize >= N), overlap 0 -> the
//       local factor IS A's factor -> M = A^-1: Apply matches a dense solve and pcg converges in 1
//       iteration.
//   (2) AS SYMMETRY: <M r1, r2> == <r1, M r2> on Laplacian2D and PenalizedGrid3D, several
//       (subdomainSize, overlap) combos including ragged last subdomains -- the CG-validity contract.
//   (3) RAS SOLVES: pbiCGStab + RAS converges (residual-checked) on a general diag-dominant square
//       and on an SPD matrix.
//   (4) HEADLINE (iterations): AS-pcg reaches tol in fewer iterations than plain CG AND than
//       block-Jacobi-pcg, aggregated over 3 random right-hand sides (residual-based, counts read
//       back to the managed thread). IC0 is NOT compared (reference-only per spec).
//   (5) SYMMETRIC-storage A produces bit-identical M^-1 r to the same A in full storage.
//   (6) DETERMINISM: two builds + applies -> byte-identical z.
//   (7) THROUGH-IJOB: AS built once on the main thread, pcg run inside a Burst IJob.Run() twice,
//       gives bit-identical iteration count and x; the managed path matches to tolerance.
//   (8) BREAKDOWN: an indefinite local block makes AS's out-info report NotPositiveDefinite; a
//       singular local block makes RAS's out-info report Singular -- both without throwing, while
//       the throwing ctors throw.
//   CG-SAFETY: RAS has NO pcg/pminres overload (it is not symmetric) -- Krylov.pcg(A, ras, ...)
//       would not compile. That absence is the type-system guard; it cannot be asserted at runtime.
//
// "Beats X" is always RESIDUAL-based (||b - A x||^2 <= (C*tol)^2 ||b||^2), never a per-element
// solution-error bound (solution error ~ cond(A)*residual, unsatisfiable for cond>1). Iteration
// comparisons are multi-trial with the counts written to NativeArrays and every Assert run on the
// managed thread, so a failure prints the actual numbers.
public class fProxySchwarzPreconditionerTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct SchwarzTestJob : IJob
    {
        public enum TestType
        {
            ExactWholeMatrix,
            AsSymmetry,
            RasSolves,
            SymmetricStorageMatchesFull,
            DeterministicBuild,
            PminresConverges,
        }

        public TestType Type;

        // Cholesky/LU based (+ - * / sqrt only) -> IC0-class tolerances.
        static fProxy Tol() => /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ExactWholeMatrix: ExactWholeMatrix(); break;
                case TestType.AsSymmetry: AsSymmetry(); break;
                case TestType.RasSolves: RasSolves(); break;
                case TestType.SymmetricStorageMatchesFull: SymmetricStorageMatchesFull(); break;
                case TestType.DeterministicBuild: DeterministicBuild(); break;
                case TestType.PminresConverges: PminresConverges(); break;
            }
        }

        // ---- helpers ---------------------------------------------------------------------

        // SPD block-tridiagonal chain (fill-free), full or symmetric (lower) storage.
        static fProxyBSR BuildBlockTridiag(ref Arena arena, int nb, int BR, bool symmetric)
        {
            var builder = arena.fProxyBSRBuilder(nb, nb, BR, BR);
            var diag = arena.fProxyMat(BR, BR);
            var off = arena.fProxyMat(BR, BR);
            for (int r = 0; r < BR; r++)
                for (int c = 0; c < BR; c++)
                {
                    diag[r, c] = (r == c ? (fProxy)(2 * BR + 2) : (fProxy)0) + (fProxy)0.25f;
                    off[r, c] = r == c ? (fProxy)(-1) : (fProxy)0;
                }

            for (int i = 0; i < nb; i++)
            {
                builder.AddBlock(i, i, in diag);
                if (i + 1 < nb)
                {
                    builder.AddBlock(i + 1, i, in off);
                    if (!symmetric) builder.AddBlock(i, i + 1, in off);
                }
            }
            return symmetric ? builder.ToBSRSymmetric(ref arena) : builder.ToBSR(ref arena);
        }

        static void AssertVecMatchesInverse(in fProxyN got, in fProxyMxN Adense, in fProxyN r, ref Arena arena, fProxy tol)
        {
            int n = r.N;
            var D = Adense.Copy();
            var zRef = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) zRef[i] = r[i];
            var info = CHO.solveInPlace(ref D, ref zRef);
            Assert.IsTrue(info.Solved);
            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(got[i] - zRef[i]) < tol * ((fProxy)1 + math.abs(zRef[i])));
        }

        static fProxy Asymmetry(in fProxyBSR A, in fProxyN r1, in fProxyN r2, in fProxyAdditiveSchwarz M, ref Arena arena)
        {
            int n = A.M_Rows;
            var Mr1 = arena.fProxyVec(n);
            var Mr2 = arena.fProxyVec(n);
            M.Apply(in r1, ref Mr1);
            M.Apply(in r2, ref Mr2);
            fProxy a = Blas.dot(r1, Mr2);
            fProxy b = Blas.dot(r2, Mr1);
            return math.abs(a - b) / ((fProxy)1 + math.abs(a) + math.abs(b));
        }

        // ================================================================================
        // (1) Exactness: one subdomain, no overlap -> M = A^-1.
        // ================================================================================

        void ExactWholeMatrix()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyRandomSparseSPD(10, 2, (fProxy)0.5, 950001u);
            int n = A.M_Rows;

            // subdomainSize >= N -> a single subdomain covering the whole matrix; overlap 0.
            var opts = new SchwarzOptions { subdomainSize = n + 8, overlap = 0 };
            var M = arena.fProxyAdditiveSchwarz(in A, in opts);
            Assert.IsTrue(M.K == 1);
            Assert.IsTrue(M.Shift == (fProxy)0);

            var r = arena.fProxyRandomVec(n, -1f, 1f, 950002u);
            var z = arena.fProxyVec(n);
            M.Apply(in r, ref z);
            var Adense = A.ToDense(ref arena);
            AssertVecMatchesInverse(in z, in Adense, in r, ref arena, Tol());

            // Exact preconditioner -> pcg converges in one step.
            var xTrue = arena.fProxyRandomVec(n, 0.5f, 1.5f, 950003u);
            var b = BSR.spMV(in A, in xTrue);
            var x = arena.fProxyVec(n);
            var info = Krylov.pcg(in A, in M, in b, ref x, 4 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(info.Solved);
            Assert.IsTrue(info.iterations <= 1);

            arena.Dispose();
        }

        // ================================================================================
        // (2) AS symmetry: <M r1, r2> == <r1, M r2> across several (subdomainSize, overlap).
        // ================================================================================

        // Static readonly, not constructed inline: Burst forbids creating a managed array at runtime
        // inside a job (BC1028); a statically-initialized readonly array reference compiles.
        static readonly int[] AsSizes = { 12, 20, 31 };
        static readonly int[] AsOverlaps = { 0, 1, 2 };

        void AsSymmetry()
        {
            var arena = new Arena(Allocator.Persistent);

            var lap = arena.fProxyLaplacian2D(8, 8);         // 64 dof, BR=1
            var pen = arena.fProxyPenalizedGrid3D(3, 3, 3, (fProxy)1, (fProxy)10);   // 27 blocks, BR=3

            // (subdomainSize, overlap) combos: some force ragged last subdomains (nb not a multiple).
            for (int si = 0; si < AsSizes.Length; si++)
            {
                for (int oi = 0; oi < AsOverlaps.Length; oi++)
                {
                    var opts = new SchwarzOptions { subdomainSize = AsSizes[si], overlap = AsOverlaps[oi] };

                    CheckSym(in lap, in opts, ref arena, 951000u + (uint)(si * 10 + oi));
                    CheckSym(in pen, in opts, ref arena, 952000u + (uint)(si * 10 + oi));
                }
            }

            arena.Dispose();
        }

        void CheckSym(in fProxyBSR A, in SchwarzOptions opts, ref Arena arena, uint seed)
        {
            var M = arena.fProxyAdditiveSchwarz(in A, in opts);
            int n = A.M_Rows;
            for (int t = 0; t < 3; t++)
            {
                var r1 = arena.fProxyRandomVec(n, -1f, 1f, seed + (uint)t);
                var r2 = arena.fProxyRandomVec(n, -1f, 1f, seed + 100u + (uint)t);
                Assert.IsTrue(Asymmetry(in A, in r1, in r2, in M, ref arena) < Tol());
            }
        }

        // ================================================================================
        // (3) RAS solves via pbiCGStab on a general square and an SPD matrix (residual-checked).
        // ================================================================================

        void RasSolves()
        {
            var arena = new Arena(Allocator.Persistent);

            var gen = arena.fProxyRandomSparse(16, 16, 2, (fProxy)0.4, 953001u);   // nonsymmetric diag-dominant
            var spd = arena.fProxyRandomSparseSPD(16, 2, (fProxy)0.4, 953101u);
            var opts = new SchwarzOptions { subdomainSize = 12, overlap = 1 };     // multiple subdomains

            RasResidualOk(in gen, in opts, ref arena, 953201u);
            RasResidualOk(in spd, in opts, ref arena, 953301u);

            arena.Dispose();
        }

        void RasResidualOk(in fProxyBSR A, in SchwarzOptions opts, ref Arena arena, uint seed)
        {
            var M = arena.fProxyRestrictedSchwarz(in A, in opts);
            int n = A.M_Rows;

            var xTrue = arena.fProxyRandomVec(n, 0.5f, 1.5f, seed);
            var b = BSR.spMV(in A, in xTrue);
            var x = arena.fProxyVec(n);
            var info = Krylov.pbiCGStab(in A, in M, in b, ref x, 20 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(info.Solved);

            var Ax = arena.fProxyVec(n);
            BSR.spMV(in A, in x, ref Ax);
            var resid = arena.fProxyVec(n);
            resid.Data.CopyFrom(b.Data);
            resid.addScaledInPlace((fProxy)(-1), Ax);
            fProxy resNormSq = Blas.dot(resid, resid);
            fProxy bNormSq = Blas.dot(b, b);
            fProxy residualC = (fProxy)8;
            fProxy tol = Consts.fProxySqrtEps;
            Assert.IsTrue(resNormSq <= residualC * residualC * tol * tol * bNormSq);
        }

        // ================================================================================
        // (5) Symmetric- vs full-storage A produce bit-identical M^-1 r (AS and RAS).
        // ================================================================================

        void SymmetricStorageMatchesFull()
        {
            var arena = new Arena(Allocator.Persistent);
            const int nb = 9, BR = 2;
            var full = BuildBlockTridiag(ref arena, nb, BR, false);
            var sym = BuildBlockTridiag(ref arena, nb, BR, true);
            int n = full.M_Rows;
            var opts = new SchwarzOptions { subdomainSize = 6, overlap = 1 };

            var mFull = arena.fProxyAdditiveSchwarz(in full, in opts);
            var mSym = arena.fProxyAdditiveSchwarz(in sym, in opts);
            var rFull = arena.fProxyRestrictedSchwarz(in full, in opts);
            var rSym = arena.fProxyRestrictedSchwarz(in sym, in opts);

            var r = arena.fProxyRandomVec(n, -1f, 1f, 954001u);
            var za = arena.fProxyVec(n);
            var zb = arena.fProxyVec(n);
            mFull.Apply(in r, ref za);
            mSym.Apply(in r, ref zb);
            for (int i = 0; i < n; i++) Assert.IsTrue(za[i] == zb[i]);

            var zc = arena.fProxyVec(n);
            var zd = arena.fProxyVec(n);
            rFull.Apply(in r, ref zc);
            rSym.Apply(in r, ref zd);
            for (int i = 0; i < n; i++) Assert.IsTrue(zc[i] == zd[i]);

            arena.Dispose();
        }

        // ================================================================================
        // (6) Determinism: two builds + applies -> byte-identical z (AS and RAS).
        // ================================================================================

        void DeterministicBuild()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomSparseSPD(20, 2, (fProxy)0.35, 955001u);
            var Ag = arena.fProxyRandomSparse(20, 20, 2, (fProxy)0.35, 955101u);
            int n = A.M_Rows;
            var opts = new SchwarzOptions { subdomainSize = 10, overlap = 1 };

            var r = arena.fProxyRandomVec(n, -1f, 1f, 955002u);

            var m1 = arena.fProxyAdditiveSchwarz(in A, in opts);
            var m2 = arena.fProxyAdditiveSchwarz(in A, in opts);
            var z1 = arena.fProxyVec(n);
            var z2 = arena.fProxyVec(n);
            m1.Apply(in r, ref z1);
            m2.Apply(in r, ref z2);
            for (int i = 0; i < n; i++) Assert.IsTrue(z1[i] == z2[i]);

            var g1 = arena.fProxyRestrictedSchwarz(in Ag, in opts);
            var g2 = arena.fProxyRestrictedSchwarz(in Ag, in opts);
            var w1 = arena.fProxyVec(n);
            var w2 = arena.fProxyVec(n);
            g1.Apply(in r, ref w1);
            g2.Apply(in r, ref w2);
            for (int i = 0; i < n; i++) Assert.IsTrue(w1[i] == w2[i]);

            arena.Dispose();
        }

        // ================================================================================
        // AS is valid for pminres too (SPD, symmetric M). Exercises the pminres rung.
        // ================================================================================

        void PminresConverges()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyLaplacian2D(6, 6);   // 36 dof SPD
            int n = A.M_Rows;
            var opts = new SchwarzOptions { subdomainSize = 12, overlap = 1 };
            var M = arena.fProxyAdditiveSchwarz(in A, in opts);

            var xTrue = arena.fProxyRandomVec(n, 0.5f, 1.5f, 956001u);
            var b = BSR.spMV(in A, in xTrue);
            var x = arena.fProxyVec(n);
            var info = Krylov.pminres(in A, in M, in b, ref x, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(info.Solved);

            var Ax = arena.fProxyVec(n);
            BSR.spMV(in A, in x, ref Ax);
            var resid = arena.fProxyVec(n);
            resid.Data.CopyFrom(b.Data);
            resid.addScaledInPlace((fProxy)(-1), Ax);
            fProxy resNormSq = Blas.dot(resid, resid);
            fProxy bNormSq = Blas.dot(b, b);
            fProxy tol = Consts.fProxySqrtEps;
            Assert.IsTrue(resNormSq <= (fProxy)64 * tol * tol * bNormSq);

            arena.Dispose();
        }
    }

    // ================================================================================
    // (4) Headline: AS-pcg beats plain CG AND block-Jacobi-pcg in iterations, aggregated over
    // 3 random right-hand sides. All numbers written to NativeArrays; asserts on managed thread.
    // ================================================================================
    [BurstCompile(CompileSynchronously = true)]
    public struct SchwarzBeatsJob : IJob
    {
        public const int TRIALS = 3;

        public enum Mode { Laplacian, Penalized }
        public Mode mode;
        public int subdomainSize;
        public int overlap;

        // iters[trial*3 + solver], solved[...]: solver 0=plain CG, 1=block-Jacobi, 2=AS.
        public NativeArray<int> iters;
        public NativeArray<int> solved;
        public NativeArray<int> accOk;       // length TRIALS: 1 if AS residual <= RESIDUAL_C*tol*||b||
        public NativeArray<double> asInfo;   // [0]=attempts, [1]=shift, [2]=status, [3]=K

        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            fProxyBSR A = mode == Mode.Laplacian
                ? arena.fProxyLaplacian2D(16, 16)
                : arena.fProxyPenalizedGrid3D(4, 4, 4, (fProxy)1, (fProxy)10);

            var bJ = arena.fProxyBlockJacobi(in A);
            var opts = new SchwarzOptions { subdomainSize = subdomainSize, overlap = overlap };
            var asM = arena.fProxyAdditiveSchwarz(in A, in opts, out PreconditionerInfo info);
            int n = A.M_Rows;

            asInfo[0] = info.attempts;
            asInfo[1] = info.shift;
            asInfo[2] = (double)(int)info.status;
            asInfo[3] = asM.K;

            fProxy tol = Consts.fProxySqrtEps;
            int maxIter = 20 * n;

            RunTrial(arena, in A, in bJ, in asM, n, maxIter, tol, 0, 957001u, (fProxy)0.5, (fProxy)1.5);
            RunTrial(arena, in A, in bJ, in asM, n, maxIter, tol, 1, 957011u, (fProxy)(-1), (fProxy)1);
            RunTrial(arena, in A, in bJ, in asM, n, maxIter, tol, 2, 957021u, (fProxy)(-3), (fProxy)3);

            arena.Dispose();
        }

        void RunTrial(Arena arena, in fProxyBSR A, in fProxyBlockJacobi bJ, in fProxyAdditiveSchwarz asM,
                      int n, int maxIter, fProxy tol, int trial, uint seed, fProxy lo, fProxy hi)
        {
            var xTrue = arena.fProxyRandomVec(n, lo, hi, seed);
            var b = BSR.spMV(in A, in xTrue);

            var xCG = arena.fProxyVec(n);
            var infoCG = Krylov.cg(in A, in b, ref xCG, maxIter, tol);
            var xJ = arena.fProxyVec(n);
            var infoJ = Krylov.pcg(in A, in bJ, in b, ref xJ, maxIter, tol);
            var xA = arena.fProxyVec(n);
            var infoA = Krylov.pcg(in A, in asM, in b, ref xA, maxIter, tol);

            iters[trial * 3 + 0] = infoCG.iterations;
            iters[trial * 3 + 1] = infoJ.iterations;
            iters[trial * 3 + 2] = infoA.iterations;
            solved[trial * 3 + 0] = infoCG.Solved ? 1 : 0;
            solved[trial * 3 + 1] = infoJ.Solved ? 1 : 0;
            solved[trial * 3 + 2] = infoA.Solved ? 1 : 0;

            var AxA = arena.fProxyVec(n);
            BSR.spMV(in A, in xA, ref AxA);
            var resid = arena.fProxyVec(n);
            resid.Data.CopyFrom(b.Data);
            resid.addScaledInPlace((fProxy)(-1), AxA);
            fProxy resNormSq = Blas.dot(resid, resid);
            fProxy bNormSq = Blas.dot(b, b);
            fProxy residualC = (fProxy)8;
            accOk[trial] = (resNormSq <= residualC * residualC * tol * tol * bNormSq) ? 1 : 0;
        }
    }

    // Solve-only job for the through-IJob determinism test: AS is built ONCE on the main thread and
    // passed in by value; the job runs pcg and records the iteration count.
    [BurstCompile(CompileSynchronously = true)]
    public struct SchwarzSolveJob : IJob
    {
        public fProxyBSR A;
        public fProxyAdditiveSchwarz M;
        public fProxyN b;
        public fProxyN x;                 // output (arena-backed; written through its pointer)
        public NativeArray<int> iters;    // length 1
        public int maxIter;
        public fProxy tol;

        public void Execute()
        {
            var info = Krylov.pcg(in A, in M, in b, ref x, maxIter, tol);
            iters[0] = info.iterations;
        }
    }

    // ---- correctness cases (Burst) -------------------------------------------------------

    [Test] public void ExactWholeMatrixTest()
        => new SchwarzTestJob { Type = SchwarzTestJob.TestType.ExactWholeMatrix }.Run();
    [Test] public void AsSymmetryTest()
        => new SchwarzTestJob { Type = SchwarzTestJob.TestType.AsSymmetry }.Run();
    [Test] public void RasSolvesTest()
        => new SchwarzTestJob { Type = SchwarzTestJob.TestType.RasSolves }.Run();
    [Test] public void SymmetricStorageMatchesFullTest()
        => new SchwarzTestJob { Type = SchwarzTestJob.TestType.SymmetricStorageMatchesFull }.Run();
    [Test] public void DeterministicBuildTest()
        => new SchwarzTestJob { Type = SchwarzTestJob.TestType.DeterministicBuild }.Run();
    [Test] public void PminresConvergesTest()
        => new SchwarzTestJob { Type = SchwarzTestJob.TestType.PminresConverges }.Run();

    // ---- (4) headline iteration comparison (managed orchestration) -----------------------

    [Test]
    public void BeatsPlainCgAndJacobiOnLaplacian()
        => RunBeats(SchwarzBeatsJob.Mode.Laplacian, subdomainSize: 32, overlap: 1);

    [Test]
    public void BeatsPlainCgAndJacobiOnPenalizedGrid3D()
        => RunBeats(SchwarzBeatsJob.Mode.Penalized, subdomainSize: 24, overlap: 1);

    static void RunBeats(SchwarzBeatsJob.Mode mode, int subdomainSize, int overlap)
    {
        int trials = SchwarzBeatsJob.TRIALS;
        var iters = new NativeArray<int>(trials * 3, Allocator.Persistent);
        var solved = new NativeArray<int>(trials * 3, Allocator.Persistent);
        var accOk = new NativeArray<int>(trials, Allocator.Persistent);
        var asInfo = new NativeArray<double>(4, Allocator.Persistent);

        new SchwarzBeatsJob { mode = mode, subdomainSize = subdomainSize, overlap = overlap,
                              iters = iters, solved = solved, accOk = accOk, asInfo = asInfo }.Run();

        var msg = $"AS build ({mode}): attempts={asInfo[0]} shift={asInfo[1]:E3} status={(DirectSolveStatus)(int)asInfo[2]} K={(int)asInfo[3]}\n";
        int cgSum = 0, jSum = 0, aSum = 0, beatsCG = 0, beatsJ = 0;
        for (int t = 0; t < trials; t++)
        {
            int cg = iters[t * 3 + 0], j = iters[t * 3 + 1], a = iters[t * 3 + 2];
            msg += $"trial {t}: CG={cg}(s={solved[t * 3 + 0] == 1}) Jacobi={j}(s={solved[t * 3 + 1] == 1}) AS={a}(s={solved[t * 3 + 2] == 1}) accOk={accOk[t] == 1}\n";
        }

        for (int t = 0; t < trials; t++)
        {
            Assert.IsTrue(solved[t * 3 + 0] == 1, "plain CG did not converge -- " + msg);
            Assert.IsTrue(solved[t * 3 + 1] == 1, "block-Jacobi did not converge -- " + msg);
            Assert.IsTrue(solved[t * 3 + 2] == 1, "AS did not converge -- " + msg);
            Assert.IsTrue(accOk[t] == 1, "AS residual did not meet RESIDUAL_C*tol*||b|| -- " + msg);
            int cg = iters[t * 3 + 0], j = iters[t * 3 + 1], a = iters[t * 3 + 2];
            cgSum += cg; jSum += j; aSum += a;
            if (a < cg) beatsCG++;
            if (a < j) beatsJ++;
        }

        // Aggregate beats (robust to a single-trial tie) AND wins on >= 2 of 3 trials.
        Assert.IsTrue(aSum < cgSum, "AS total iterations did not beat plain CG's -- " + msg);
        Assert.IsTrue(aSum < jSum, "AS total iterations did not beat block-Jacobi's -- " + msg);
        Assert.IsTrue(beatsCG >= trials - 1, "AS must strictly beat plain CG on all but at most one trial -- " + msg);
        Assert.IsTrue(beatsJ >= trials - 1, "AS must strictly beat block-Jacobi on all but at most one trial -- " + msg);

        iters.Dispose();
        solved.Dispose();
        accOk.Dispose();
        asInfo.Dispose();
    }

    // ---- (7) through-IJob determinism ----------------------------------------------------

    [Test]
    public void ThroughIJobDeterminismTest()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyLaplacian2D(8, 8);   // 64 dof SPD
        int n = A.M_Rows;
        var opts = new SchwarzOptions { subdomainSize = 16, overlap = 1 };
        var M = arena.fProxyAdditiveSchwarz(in A, in opts);

        var xTrue = arena.fProxyRandomVec(n, 0.5f, 1.5f, 958001u);
        var b = BSR.spMV(in A, in xTrue);
        fProxy tol = Consts.fProxySqrtEps;
        int maxIter = 8 * n;

        var x1 = arena.fProxyVec(n);
        var x2 = arena.fProxyVec(n);
        var it1 = new NativeArray<int>(1, Allocator.Persistent);
        var it2 = new NativeArray<int>(1, Allocator.Persistent);

        new SchwarzSolveJob { A = A, M = M, b = b, x = x1, iters = it1, maxIter = maxIter, tol = tol }.Run();
        new SchwarzSolveJob { A = A, M = M, b = b, x = x2, iters = it2, maxIter = maxIter, tol = tol }.Run();

        Assert.AreEqual(it1[0], it2[0]);
        for (int i = 0; i < n; i++)
            Assert.IsTrue(x1[i] == x2[i]);

        // Managed (non-Burst) path: consistent to tolerance (SIMD reassociation may differ).
        var x3 = arena.fProxyVec(n);
        var infoM = Krylov.pcg(in A, in M, in b, ref x3, maxIter, tol);
        Assert.IsTrue(infoM.Solved);
        fProxy consistencyTol = /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;
        for (int i = 0; i < n; i++)
            Assert.IsTrue(math.abs(x1[i] - x3[i]) < consistencyTol * ((fProxy)1 + math.abs(x3[i])));

        it1.Dispose();
        it2.Dispose();
        arena.Dispose();
    }

    // ---- (8) breakdown: non-throwing twin vs throwing ctor -------------------------------

    [Test]
    public void AsIndefiniteBreaksDown()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            // Symmetric INDEFINITE 2x2 (BR=1): [[1,20],[20,1]] has eigenvalues 21, -19. diagMax=1,
            // so no diagonal shift up to 10*diagMax rescues the local Cholesky -> build fails.
            var builder = arena.fProxyBSRBuilder(2, 2, 1, 1, 4);
            builder.AddValue(0, 0, (fProxy)1);
            builder.AddValue(0, 1, (fProxy)20);
            builder.AddValue(1, 0, (fProxy)20);
            builder.AddValue(1, 1, (fProxy)1);
            var A = builder.ToBSR(ref arena);

            var opts = new SchwarzOptions { subdomainSize = 8, overlap = 0 };
            var M = arena.fProxyAdditiveSchwarz(in A, in opts, out PreconditionerInfo info);
            Assert.IsFalse(info.Solved);
            Assert.IsTrue(info.status == DirectSolveStatus.NotPositiveDefinite);

            Assert.Throws<ArgumentException>(() => { var m2 = arena.fProxyAdditiveSchwarz(in A, in opts); });
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void RasSingularBreaksDown()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            // Structurally singular 2x2 (BR=1): column 1 is entirely zero -> local LU hits a zero
            // pivot -> Singular (RAS does not shift-retry).
            var builder = arena.fProxyBSRBuilder(2, 2, 1, 1, 4);
            builder.AddValue(0, 0, (fProxy)1);
            builder.AddValue(1, 0, (fProxy)2);
            var A = builder.ToBSR(ref arena);

            var opts = new SchwarzOptions { subdomainSize = 8, overlap = 0 };
            var M = arena.fProxyRestrictedSchwarz(in A, in opts, out PreconditionerInfo info);
            Assert.IsFalse(info.Solved);
            Assert.IsTrue(info.status == DirectSolveStatus.Singular);

            Assert.Throws<ArgumentException>(() => { var m2 = arena.fProxyRestrictedSchwarz(in A, in opts); });
        }
        finally { arena.Dispose(); }
    }

    // ---- guard cases (managed thread) ----------------------------------------------------

    [Test]
    public void NonSquareThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var builder = arena.fProxyBSRBuilder(2, 3, 2, 2);
            var block = arena.fProxyMat(2, 2, (fProxy)1);
            builder.AddBlock(0, 0, in block);
            var A = builder.ToBSR(ref arena);
            Assert.Throws<ArgumentException>(() => { var m = arena.fProxyAdditiveSchwarz(in A); });
            Assert.Throws<ArgumentException>(() => { var m = arena.fProxyRestrictedSchwarz(in A); });
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void ApplyAliasingThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyRandomSparseSPD(8, 2, (fProxy)0.5, 959001u);
            int n = A.M_Rows;
            var opts = new SchwarzOptions { subdomainSize = 6, overlap = 1 };
            var M = arena.fProxyAdditiveSchwarz(in A, in opts);
            var R = arena.fProxyRestrictedSchwarz(in A, in opts);

            var r = arena.fProxyVec(n, (fProxy)1);
            var z = arena.fProxyVec(n);

            // AS: z aliases r, z aliases Scratch, r aliases Scratch.
            Assert.Throws<ArgumentException>(() => M.Apply(in r, ref r));
            var asScratch = M.Scratch;
            Assert.Throws<ArgumentException>(() => M.Apply(in r, ref asScratch));
            Assert.Throws<ArgumentException>(() => M.Apply(in asScratch, ref z));

            // RAS: z aliases r, z aliases a scratch, r aliases a scratch.
            Assert.Throws<ArgumentException>(() => R.Apply(in r, ref r));
            var rScratch = R.Scratch;
            var rScratch2 = R.Scratch2;
            Assert.Throws<ArgumentException>(() => R.Apply(in r, ref rScratch));
            Assert.Throws<ArgumentException>(() => R.Apply(in r, ref rScratch2));
            Assert.Throws<ArgumentException>(() => R.Apply(in rScratch, ref z));
        }
        finally { arena.Dispose(); }
    }
}
