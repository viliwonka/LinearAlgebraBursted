using System;
using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// One-level Schwarz preconditioners: fProxyAdditiveSchwarz (symmetric AS, valid for cg/minres)
// and fProxyRestrictedSchwarz (RAS, non-symmetric, biCGStab only).
// Correctness anchors (spec docs/dev/spec-additive-schwarz-preconditioner.md, section Tests):
//   (1) EXACTNESS: one subdomain covering the whole matrix (subdomainSize >= N), overlap 0 -> the
//       local factor IS A's factor -> M = A^-1: Apply matches a dense solve and cg converges in 1
//       iteration.
//   (2) AS SYMMETRY: <M r1, r2> == <r1, M r2> on Laplacian2D and PenalizedGrid3D, several
//       (subdomainSize, overlap) combos including ragged last subdomains -- the CG-validity contract.
//   (3) RAS SOLVES: biCGStab + RAS converges (residual-checked) on a general diag-dominant square
//       and on an SPD matrix.
//   (4) HEADLINE (iterations): AS-cg reaches tol in fewer iterations than plain CG AND than
//       block-Jacobi-cg, aggregated over 3 random right-hand sides (residual-based, counts read
//       back to the managed thread). IC0 is NOT compared (reference-only per spec).
//   (5) SYMMETRIC-storage A produces bit-identical M^-1 r to the same A in full storage.
//   (6) DETERMINISM: two builds + applies -> byte-identical z.
//   (7) THROUGH-IJOB: AS built once on the main thread, cg run inside a Burst IJob.Run() twice,
//       gives bit-identical iteration count and x; the managed path matches to tolerance.
//   (8) BREAKDOWN: an indefinite local block makes AS's out-info report NotPositiveDefinite; a
//       singular local block makes RAS's out-info report Singular -- both without throwing, while
//       the throwing ctors throw.
//   CG-SAFETY: RAS has NO cg/minres overload (it is not symmetric) -- Krylov.cg(A, ras, ...)
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
        static fProxyBSR BuildBlockTridiag(int nb, int BR, bool symmetric)
        {
            var builder = new fProxyBSRBuilder(nb, nb, BR, BR, Allocator.Temp);
            var diag = new fProxyMxN(BR, BR, Allocator.Temp);
            var off = new fProxyMxN(BR, BR, Allocator.Temp);
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
            return symmetric ? builder.ToBSRSymmetric(Allocator.Temp) : builder.ToBSR(Allocator.Temp);
        }

        static void AssertVecMatchesInverse(in fProxyN got, in fProxyMxN Adense, in fProxyN r, fProxy tol)
        {
            int n = r.N;
            var D = Adense.Copy();
            var zRef = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) zRef[i] = r[i];
            var info = CHO.solveInPlace(ref D, ref zRef);
            Assert.IsTrue(info.Solved);
            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(got[i] - zRef[i]) < tol * ((fProxy)1 + math.abs(zRef[i])));
        }

        static fProxy Asymmetry(in fProxyBSR A, in fProxyN r1, in fProxyN r2, in fProxyAdditiveSchwarz M)
        {
            int n = A.M_Rows;
            var Mr1 = new fProxyN(n, Allocator.Temp);
            var Mr2 = new fProxyN(n, Allocator.Temp);
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
            var A = fProxyGallery.fProxyRandomSparseSPD(10, 2, (fProxy)0.5, 950001u, allocator: Allocator.Temp);
            int n = A.M_Rows;

            // subdomainSize >= N -> a single subdomain covering the whole matrix; overlap 0.
            var opts = new SchwarzOptions { subdomainSize = n + 8, overlap = 0 };
            var M = new fProxyAdditiveSchwarz(in A, Allocator.Temp, in opts);
            Assert.IsTrue(M.K == 1);
            Assert.IsTrue(M.Shift == (fProxy)0);

            var r = GenerateOP.fProxyRandomVec(n, -1f, 1f, 950002u, allocator: Allocator.Temp);
            var z = new fProxyN(n, Allocator.Temp);
            M.Apply(in r, ref z);
            var Adense = A.ToDense(Allocator.Temp);
            AssertVecMatchesInverse(in z, in Adense, in r, Tol());

            // Exact preconditioner -> cg converges in one step.
            var xTrue = GenerateOP.fProxyRandomVec(n, 0.5f, 1.5f, 950003u, allocator: Allocator.Temp);
            var b = BSR.spMV(in A, in xTrue);
            var x = new fProxyN(n, Allocator.Temp);
            var info = Krylov.cg(in A, in M, in b, ref x, 4 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(info.Solved);
            Assert.IsTrue(info.iterations <= 1);
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
            var lap = fProxyGallery.fProxyLaplacian2D(8, 8, allocator: Allocator.Temp);         // 64 dof, BR=1
            var pen = fProxyGallery.fProxyPenalizedGrid3D(3, 3, 3, (fProxy)1, (fProxy)10, allocator: Allocator.Temp);   // 27 blocks, BR=3

            // (subdomainSize, overlap) combos: some force ragged last subdomains (nb not a multiple).
            for (int si = 0; si < AsSizes.Length; si++)
            {
                for (int oi = 0; oi < AsOverlaps.Length; oi++)
                {
                    var opts = new SchwarzOptions { subdomainSize = AsSizes[si], overlap = AsOverlaps[oi] };

                    CheckSym(in lap, in opts, 951000u + (uint)(si * 10 + oi));
                    CheckSym(in pen, in opts, 952000u + (uint)(si * 10 + oi));
                }
            }
        }

        void CheckSym(in fProxyBSR A, in SchwarzOptions opts, uint seed)
        {
            var M = new fProxyAdditiveSchwarz(in A, Allocator.Temp, in opts);
            int n = A.M_Rows;
            for (int t = 0; t < 3; t++)
            {
                var r1 = GenerateOP.fProxyRandomVec(n, -1f, 1f, seed + (uint)t, allocator: Allocator.Temp);
                var r2 = GenerateOP.fProxyRandomVec(n, -1f, 1f, seed + 100u + (uint)t, allocator: Allocator.Temp);
                Assert.IsTrue(Asymmetry(in A, in r1, in r2, in M) < Tol());
            }
        }

        // ================================================================================
        // (3) RAS solves via biCGStab on a general square and an SPD matrix (residual-checked).
        // ================================================================================

        void RasSolves()
        {
            var gen = fProxyGallery.fProxyRandomSparse(16, 16, 2, (fProxy)0.4, 953001u, allocator: Allocator.Temp);   // nonsymmetric diag-dominant
            var spd = fProxyGallery.fProxyRandomSparseSPD(16, 2, (fProxy)0.4, 953101u, allocator: Allocator.Temp);
            var opts = new SchwarzOptions { subdomainSize = 12, overlap = 1 };     // multiple subdomains

            RasResidualOk(in gen, in opts, 953201u);
            RasResidualOk(in spd, in opts, 953301u);
        }

        void RasResidualOk(in fProxyBSR A, in SchwarzOptions opts, uint seed)
        {
            var M = new fProxyRestrictedSchwarz(in A, Allocator.Temp, in opts);
            int n = A.M_Rows;

            var xTrue = GenerateOP.fProxyRandomVec(n, 0.5f, 1.5f, seed, allocator: Allocator.Temp);
            var b = BSR.spMV(in A, in xTrue);
            var x = new fProxyN(n, Allocator.Temp);
            var info = Krylov.biCGStab(in A, in M, in b, ref x, 20 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(info.Solved);

            var Ax = new fProxyN(n, Allocator.Temp);
            BSR.spMV(in A, in x, ref Ax);
            var resid = new fProxyN(n, Allocator.Temp);
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
            const int nb = 9, BR = 2;
            var full = BuildBlockTridiag(nb, BR, false);
            var sym = BuildBlockTridiag(nb, BR, true);
            int n = full.M_Rows;
            var opts = new SchwarzOptions { subdomainSize = 6, overlap = 1 };

            var mFull = new fProxyAdditiveSchwarz(in full, Allocator.Temp, in opts);
            var mSym = new fProxyAdditiveSchwarz(in sym, Allocator.Temp, in opts);
            var rFull = new fProxyRestrictedSchwarz(in full, Allocator.Temp, in opts);
            var rSym = new fProxyRestrictedSchwarz(in sym, Allocator.Temp, in opts);

            var r = GenerateOP.fProxyRandomVec(n, -1f, 1f, 954001u, allocator: Allocator.Temp);
            var za = new fProxyN(n, Allocator.Temp);
            var zb = new fProxyN(n, Allocator.Temp);
            mFull.Apply(in r, ref za);
            mSym.Apply(in r, ref zb);
            for (int i = 0; i < n; i++) Assert.IsTrue(za[i] == zb[i]);

            var zc = new fProxyN(n, Allocator.Temp);
            var zd = new fProxyN(n, Allocator.Temp);
            rFull.Apply(in r, ref zc);
            rSym.Apply(in r, ref zd);
            for (int i = 0; i < n; i++) Assert.IsTrue(zc[i] == zd[i]);
        }

        // ================================================================================
        // (6) Determinism: two builds + applies -> byte-identical z (AS and RAS).
        // ================================================================================

        void DeterministicBuild()
        {
            var A = fProxyGallery.fProxyRandomSparseSPD(20, 2, (fProxy)0.35, 955001u, allocator: Allocator.Temp);
            var Ag = fProxyGallery.fProxyRandomSparse(20, 20, 2, (fProxy)0.35, 955101u, allocator: Allocator.Temp);
            int n = A.M_Rows;
            var opts = new SchwarzOptions { subdomainSize = 10, overlap = 1 };

            var r = GenerateOP.fProxyRandomVec(n, -1f, 1f, 955002u, allocator: Allocator.Temp);

            var m1 = new fProxyAdditiveSchwarz(in A, Allocator.Temp, in opts);
            var m2 = new fProxyAdditiveSchwarz(in A, Allocator.Temp, in opts);
            var z1 = new fProxyN(n, Allocator.Temp);
            var z2 = new fProxyN(n, Allocator.Temp);
            m1.Apply(in r, ref z1);
            m2.Apply(in r, ref z2);
            for (int i = 0; i < n; i++) Assert.IsTrue(z1[i] == z2[i]);

            var g1 = new fProxyRestrictedSchwarz(in Ag, Allocator.Temp, in opts);
            var g2 = new fProxyRestrictedSchwarz(in Ag, Allocator.Temp, in opts);
            var w1 = new fProxyN(n, Allocator.Temp);
            var w2 = new fProxyN(n, Allocator.Temp);
            g1.Apply(in r, ref w1);
            g2.Apply(in r, ref w2);
            for (int i = 0; i < n; i++) Assert.IsTrue(w1[i] == w2[i]);
        }

        // ================================================================================
        // AS is valid for minres too (SPD, symmetric M). Exercises the minres rung.
        // ================================================================================

        void PminresConverges()
        {
            var A = fProxyGallery.fProxyLaplacian2D(6, 6, allocator: Allocator.Temp);   // 36 dof SPD
            int n = A.M_Rows;
            var opts = new SchwarzOptions { subdomainSize = 12, overlap = 1 };
            var M = new fProxyAdditiveSchwarz(in A, Allocator.Temp, in opts);

            var xTrue = GenerateOP.fProxyRandomVec(n, 0.5f, 1.5f, 956001u, allocator: Allocator.Temp);
            var b = BSR.spMV(in A, in xTrue);
            var x = new fProxyN(n, Allocator.Temp);
            var info = Krylov.minres(in A, in M, in b, ref x, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(info.Solved);

            var Ax = new fProxyN(n, Allocator.Temp);
            BSR.spMV(in A, in x, ref Ax);
            var resid = new fProxyN(n, Allocator.Temp);
            resid.Data.CopyFrom(b.Data);
            resid.addScaledInPlace((fProxy)(-1), Ax);
            fProxy resNormSq = Blas.dot(resid, resid);
            fProxy bNormSq = Blas.dot(b, b);
            fProxy tol = Consts.fProxySqrtEps;
            Assert.IsTrue(resNormSq <= (fProxy)64 * tol * tol * bNormSq);
        }
    }

    // ================================================================================
    // (4) Headline: AS-cg beats plain CG AND block-Jacobi-cg in iterations, aggregated over
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
            fProxyBSR A = mode == Mode.Laplacian
                ? fProxyGallery.fProxyLaplacian2D(16, 16, allocator: Allocator.Temp)
                : fProxyGallery.fProxyPenalizedGrid3D(4, 4, 4, (fProxy)1, (fProxy)10, allocator: Allocator.Temp);

            var bJ = new fProxyBlockJacobi(in A, Allocator.Temp);
            var opts = new SchwarzOptions { subdomainSize = subdomainSize, overlap = overlap };
            var asM = new fProxyAdditiveSchwarz(in A, Allocator.Temp, in opts, out PreconditionerInfo info);
            int n = A.M_Rows;

            asInfo[0] = info.attempts;
            asInfo[1] = info.shift;
            asInfo[2] = (double)(int)info.status;
            asInfo[3] = asM.K;

            fProxy tol = Consts.fProxySqrtEps;
            int maxIter = 20 * n;

            RunTrial(in A, in bJ, in asM, n, maxIter, tol, 0, 957001u, (fProxy)0.5, (fProxy)1.5);
            RunTrial(in A, in bJ, in asM, n, maxIter, tol, 1, 957011u, (fProxy)(-1), (fProxy)1);
            RunTrial(in A, in bJ, in asM, n, maxIter, tol, 2, 957021u, (fProxy)(-3), (fProxy)3);
        }

        void RunTrial(in fProxyBSR A, in fProxyBlockJacobi bJ, in fProxyAdditiveSchwarz asM,
                      int n, int maxIter, fProxy tol, int trial, uint seed, fProxy lo, fProxy hi)
        {
            var xTrue = GenerateOP.fProxyRandomVec(n, lo, hi, seed, allocator: Allocator.Temp);
            var b = BSR.spMV(in A, in xTrue);

            var xCG = new fProxyN(n, Allocator.Temp);
            var infoCG = Krylov.cg(in A, in b, ref xCG, maxIter, tol);
            var xJ = new fProxyN(n, Allocator.Temp);
            var infoJ = Krylov.cg(in A, in bJ, in b, ref xJ, maxIter, tol);
            var xA = new fProxyN(n, Allocator.Temp);
            var infoA = Krylov.cg(in A, in asM, in b, ref xA, maxIter, tol);

            iters[trial * 3 + 0] = infoCG.iterations;
            iters[trial * 3 + 1] = infoJ.iterations;
            iters[trial * 3 + 2] = infoA.iterations;
            solved[trial * 3 + 0] = infoCG.Solved ? 1 : 0;
            solved[trial * 3 + 1] = infoJ.Solved ? 1 : 0;
            solved[trial * 3 + 2] = infoA.Solved ? 1 : 0;

            var AxA = new fProxyN(n, Allocator.Temp);
            BSR.spMV(in A, in xA, ref AxA);
            var resid = new fProxyN(n, Allocator.Temp);
            resid.Data.CopyFrom(b.Data);
            resid.addScaledInPlace((fProxy)(-1), AxA);
            fProxy resNormSq = Blas.dot(resid, resid);
            fProxy bNormSq = Blas.dot(b, b);
            fProxy residualC = (fProxy)8;
            accOk[trial] = (resNormSq <= residualC * residualC * tol * tol * bNormSq) ? 1 : 0;
        }
    }

    // Solve-only job for the through-IJob determinism test: AS is built ONCE on the main thread and
    // passed in by value; the job runs cg and records the iteration count.
    [BurstCompile(CompileSynchronously = true)]
    public struct SchwarzSolveJob : IJob
    {
        public fProxyBSR A;
        public fProxyAdditiveSchwarz M;
        public fProxyN b;
        public fProxyN x;                 // output (written through its pointer)
        public NativeArray<int> iters;    // length 1
        public int maxIter;
        public fProxy tol;

        public void Execute()
        {
            var info = Krylov.cg(in A, in M, in b, ref x, maxIter, tol);
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
        var A = fProxyGallery.fProxyLaplacian2D(8, 8, allocator: Allocator.Temp);   // 64 dof SPD
        int n = A.M_Rows;
        var opts = new SchwarzOptions { subdomainSize = 16, overlap = 1 };
        var M = new fProxyAdditiveSchwarz(in A, Allocator.Temp, in opts);

        var xTrue = GenerateOP.fProxyRandomVec(n, 0.5f, 1.5f, 958001u, allocator: Allocator.Temp);
        var b = BSR.spMV(in A, in xTrue);
        fProxy tol = Consts.fProxySqrtEps;
        int maxIter = 8 * n;

        var x1 = new fProxyN(n, Allocator.Temp);
        var x2 = new fProxyN(n, Allocator.Temp);
        var it1 = new NativeArray<int>(1, Allocator.Persistent);
        var it2 = new NativeArray<int>(1, Allocator.Persistent);

        new SchwarzSolveJob { A = A, M = M, b = b, x = x1, iters = it1, maxIter = maxIter, tol = tol }.Run();
        new SchwarzSolveJob { A = A, M = M, b = b, x = x2, iters = it2, maxIter = maxIter, tol = tol }.Run();

        Assert.AreEqual(it1[0], it2[0]);
        for (int i = 0; i < n; i++)
            Assert.IsTrue(x1[i] == x2[i]);

        // Managed (non-Burst) path: consistent to tolerance (SIMD reassociation may differ).
        var x3 = new fProxyN(n, Allocator.Temp);
        var infoM = Krylov.cg(in A, in M, in b, ref x3, maxIter, tol);
        Assert.IsTrue(infoM.Solved);
        fProxy consistencyTol = /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;
        for (int i = 0; i < n; i++)
            Assert.IsTrue(math.abs(x1[i] - x3[i]) < consistencyTol * ((fProxy)1 + math.abs(x3[i])));

        it1.Dispose();
        it2.Dispose();
    }

    // ---- (8) breakdown: non-throwing twin vs throwing ctor -------------------------------

    [Test]
    public void AsIndefiniteBreaksDown()
    {
        // Symmetric INDEFINITE 2x2 (BR=1): [[1,20],[20,1]] has eigenvalues 21, -19. diagMax=1,
        // so no diagonal shift up to 10*diagMax rescues the local Cholesky -> build fails.
        var builder = new fProxyBSRBuilder(2, 2, 1, 1, Allocator.Temp, 4);
        builder.AddValue(0, 0, (fProxy)1);
        builder.AddValue(0, 1, (fProxy)20);
        builder.AddValue(1, 0, (fProxy)20);
        builder.AddValue(1, 1, (fProxy)1);
        var A = builder.ToBSR(Allocator.Temp);

        var opts = new SchwarzOptions { subdomainSize = 8, overlap = 0 };
        var M = new fProxyAdditiveSchwarz(in A, Allocator.Temp, in opts, out PreconditionerInfo info);
        Assert.IsFalse(info.Solved);
        Assert.IsTrue(info.status == DirectSolveStatus.NotPositiveDefinite);

        Assert.Throws<ArgumentException>(() => { var m2 = new fProxyAdditiveSchwarz(in A, Allocator.Temp, in opts); });
    }

    [Test]
    public void RasSingularBreaksDown()
    {
        // Structurally singular 2x2 (BR=1): column 1 is entirely zero -> local LU hits a zero
        // pivot -> Singular (RAS does not shift-retry).
        var builder = new fProxyBSRBuilder(2, 2, 1, 1, Allocator.Temp, 4);
        builder.AddValue(0, 0, (fProxy)1);
        builder.AddValue(1, 0, (fProxy)2);
        var A = builder.ToBSR(Allocator.Temp);

        var opts = new SchwarzOptions { subdomainSize = 8, overlap = 0 };
        var M = new fProxyRestrictedSchwarz(in A, Allocator.Temp, in opts, out PreconditionerInfo info);
        Assert.IsFalse(info.Solved);
        Assert.IsTrue(info.status == DirectSolveStatus.Singular);

        Assert.Throws<ArgumentException>(() => { var m2 = new fProxyRestrictedSchwarz(in A, Allocator.Temp, in opts); });
    }

    // ---- guard cases (managed thread) ----------------------------------------------------

    [Test]
    public void NonSquareThrows()
    {
        var builder = new fProxyBSRBuilder(2, 3, 2, 2, Allocator.Temp);
        var block = GenerateOP.fProxyMat(2, 2, (fProxy)1, allocator: Allocator.Temp);
        builder.AddBlock(0, 0, in block);
        var A = builder.ToBSR(Allocator.Temp);
        Assert.Throws<ArgumentException>(() => { var m = new fProxyAdditiveSchwarz(in A, Allocator.Temp); });
        Assert.Throws<ArgumentException>(() => { var m = new fProxyRestrictedSchwarz(in A, Allocator.Temp); });
    }

    [Test]
    public void ApplyAliasingThrows()
    {
        var A = fProxyGallery.fProxyRandomSparseSPD(8, 2, (fProxy)0.5, 959001u, allocator: Allocator.Temp);
        int n = A.M_Rows;
        var opts = new SchwarzOptions { subdomainSize = 6, overlap = 1 };
        var M = new fProxyAdditiveSchwarz(in A, Allocator.Temp, in opts);
        var R = new fProxyRestrictedSchwarz(in A, Allocator.Temp, in opts);

        var r = GenerateOP.fProxyVec(n, (fProxy)1, allocator: Allocator.Temp);
        var z = new fProxyN(n, Allocator.Temp);

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
}
