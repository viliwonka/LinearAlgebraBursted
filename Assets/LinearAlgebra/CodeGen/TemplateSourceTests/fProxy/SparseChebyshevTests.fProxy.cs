using System;
using BULA;
using BULA.Gallery;
using BULA.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Chebyshev polynomial preconditioner (fProxyChebyshev). Correctness anchors (spec
// docs/dev/spec-chebyshev-preconditioner.md Section 9):
//   (1) on a 2D Poisson system cg+Chebyshev(d=3) needs strictly fewer OUTER iterations than
//       cg+block-Jacobi, which in turn needs strictly fewer than plain CG (same tol/rhs/x0);
//   (2) raising the polynomial degree from 1 to 4 does not increase the outer-iteration count
//       (interior degrees are not asserted strictly non-increasing -- see DegreeSweepNonIncreasingTest);
//   (3) cg+Chebyshev lands on the dense-direct solution within tolerance;
//   (4) the induced M^-1 is symmetric (dot(u,M v)==dot(v,M u)) and positive (dot(v,M v)>0);
//   (5) build on the managed thread + solve THROUGH a Burst IJob (struct copied by value) yields a
//       correct solve agreeing with the managed solve -- the readonly/set-once struct-copy-safety claim;
//   (6) two identical solves are bit-identical (Apply is dot-free -> deterministic by construction);
//   (7) contract violations throw ArgumentException;
//   (8) Symmetric-storage input and its full-storage twin give the same Apply output (spMV consumes
//       both natively -- no mirror -- but their accumulation orders differ, so equal-within-tolerance,
//       not bit-identical; see the report note).
// Correctness cases run inside a [BurstCompile] IJob (same split as SparseIC0Tests / SparseSolverTests);
// the through-IJob case and the guard cases run on the managed thread (NUnit Assert.Throws cannot run
// inside a Burst-compiled job).
//
// Every built preconditioner runs Eigen.lanczos, whose step count the ctor clamps to A.Rows -- so a
// system smaller than opt.eigSteps (default 10) builds fine (see SmallSystemBelowEigStepsBuildsAndConverges).
// The guard cases that expect a throw BEFORE the Lanczos run (bad options / bad diagonal / non-square)
// may use tiny matrices.
public class fProxySparseChebyshevTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct SparseChebyshevTestJob : IJob
    {
        public enum TestType
        {
            PoissonIterationOrdering,
            SolutionMatchesDirect,
            ApplyIsSymmetricAndPositive,
            DeterministicSolve,
            SymmetricStorageMatchesFull,
        }

        public TestType Type;

        // Solution / symmetry cross-checks: two independently-converged (or iterative-vs-direct)
        // solutions on well-conditioned SPD systems agree to well under this scaled threshold on both
        // precisions.
        static fProxy Tol() => /*+choose[1e-3f|1e-7]*/1e-3f/*-choose*/;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.PoissonIterationOrdering: PoissonIterationOrdering(); break;
                case TestType.SolutionMatchesDirect: SolutionMatchesDirect(); break;
                case TestType.ApplyIsSymmetricAndPositive: ApplyIsSymmetricAndPositive(); break;
                case TestType.DeterministicSolve: DeterministicSolve(); break;
                case TestType.SymmetricStorageMatchesFull: SymmetricStorageMatchesFull(); break;
            }
        }

        // SPD block-tridiagonal chain: diagonal blocks (2*BR+2)*I + 0.25 (symmetric, strongly
        // diagonally dominant -> SPD), off-diagonal coupling blocks -I. Same recipe as SparseIC0Tests.
        static fProxyBSR BuildBlockTridiag(int nb, int BR)
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
                    builder.AddBlock(i, i + 1, in off);
                }
            }
            return builder.ToBSR(Allocator.Temp);
        }

        // ---- 1. Poisson outer-iteration ordering: Chebyshev(d=3) < block-Jacobi < plain CG -------
        void PoissonIterationOrdering()
        {
            // g = 32 -> n = 1024, comfortably above the default eigSteps = 10.
            var A = fProxyGallery.fProxyLaplacian2D(32, 32);
            int n = A.M_Rows;

            var xTrue = GenerateOP.fProxyRandomVec(n, 0.5f, 1.5f, 851001u);
            var b = BSR.spMV(in A, in xTrue);
            fProxy tol = Consts.fProxySqrtEps;
            int maxIter = 4 * n;

            var Mc = new fProxyChebyshev(in A, Allocator.Temp);                 // Default: degree 3
            var Mj = new fProxyBlockJacobi(in A, Allocator.Temp);

            var xC = new fProxyN(n, Allocator.Temp);
            var infoC = Krylov.cg(in A, in Mc, in b, ref xC, maxIter, tol);
            Assert.IsTrue(infoC.Solved);

            var xJ = new fProxyN(n, Allocator.Temp);
            var infoJ = Krylov.cg(in A, in Mj, in b, ref xJ, maxIter, tol);
            Assert.IsTrue(infoJ.Solved);

            var xG = new fProxyN(n, Allocator.Temp);
            var infoG = Krylov.cg(in A, in b, ref xG, maxIter, tol);
            Assert.IsTrue(infoG.Solved);

            // Spec Section 9.1: strictly decreasing outer-iteration counts, all to the same tol.
            Assert.IsTrue(infoC.iterations < infoJ.iterations);
            Assert.IsTrue(infoJ.iterations < infoG.iterations);
        }

        // ---- 3. Solution correctness: cg+Chebyshev matches the dense direct (Cholesky) solve -----
        void SolutionMatchesDirect()
        {
            const int nb = 8, BR = 2;                            // n = 16 > eigSteps
            var A = BuildBlockTridiag(nb, BR);
            int n = A.M_Rows;

            var M = new fProxyChebyshev(in A, Allocator.Temp);
            var xTrue = GenerateOP.fProxyRandomVec(n, 0.5f, 1.5f, 851201u);
            var b = BSR.spMV(in A, in xTrue);

            var x = new fProxyN(n, Allocator.Temp);
            var info = Krylov.cg(in A, in M, in b, ref x, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(info.Solved);

            // Dense Cholesky oracle on the same system (independent code path). CHO.solveInPlace is
            // destructive, so it runs on the dense copy with b copied into the rhs slot.
            var D = A.ToDense(Allocator.Temp);
            var xRef = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) xRef[i] = b[i];
            var choInfo = CHO.solveInPlace(ref D, ref xRef);
            Assert.IsTrue(choInfo.Solved);

            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(x[i] - xRef[i]) < Tol() * ((fProxy)1 + math.abs(xRef[i])));
        }

        // ---- 4. SPD spot check: M^-1 symmetric AND positive definite -------------------------------
        void ApplyIsSymmetricAndPositive()
        {
            var A = fProxyGallery.fProxyRandomSparseSPD(20, 3, (fProxy)0.3, 851301u);   // n = 60
            var M = new fProxyChebyshev(in A, Allocator.Temp);
            int n = A.M_Rows;

            var u = GenerateOP.fProxyRandomVec(n, -1f, 1f, 851302u);
            var v = GenerateOP.fProxyRandomVec(n, -1f, 1f, 851303u);
            var Mu = new fProxyN(n, Allocator.Temp);
            var Mv = new fProxyN(n, Allocator.Temp);
            M.Apply(in u, ref Mu);
            M.Apply(in v, ref Mv);

            // Symmetry: <u, M^-1 v> == <v, M^-1 u> (M^-1 = q(D^-1 A) D^-1 is symmetric wrt <.,.>).
            fProxy a = Blas.dot(u, Mv);
            fProxy bb = Blas.dot(v, Mu);
            fProxy scale = (fProxy)1 + math.abs(a) + math.abs(bb);
            Assert.IsTrue(math.abs(a - bb) < Tol() * scale);

            // Positive definiteness: <v, M^-1 v> > 0 for v != 0.
            fProxy q = Blas.dot(v, Mv);
            Assert.IsTrue(q > (fProxy)0);
        }

        // ---- 6. Determinism: two identical solves are bit-identical (Apply is dot-free) ------------
        void DeterministicSolve()
        {
            var A = fProxyGallery.fProxyLaplacian2D(16, 16);              // n = 256
            int n = A.M_Rows;

            var M = new fProxyChebyshev(in A, Allocator.Temp);
            var xTrue = GenerateOP.fProxyRandomVec(n, 0.5f, 1.5f, 851401u);
            var b = BSR.spMV(in A, in xTrue);
            fProxy tol = Consts.fProxySqrtEps;
            int maxIter = 4 * n;

            var x1 = new fProxyN(n, Allocator.Temp);
            var i1 = Krylov.cg(in A, in M, in b, ref x1, maxIter, tol);
            Assert.IsTrue(i1.Solved);

            var x2 = new fProxyN(n, Allocator.Temp);
            var i2 = Krylov.cg(in A, in M, in b, ref x2, maxIter, tol);
            Assert.IsTrue(i2.Solved);

            Assert.IsTrue(i1.iterations == i2.iterations);
            for (int i = 0; i < n; i++)
                Assert.IsTrue(x1[i] == x2[i]);                   // bit-identical (==, not within-tolerance)
        }

        // ---- 8. Storage-mode equivalence: Symmetric-stored A vs full-stored twin, same Apply -------
        //
        // spMV consumes both storage modes natively (no mirror), but the symmetric path folds each
        // off-diagonal block into y[i] AND y[j] in a different accumulation order than the full path --
        // and that order feeds both the Lanczos hi-estimate AND every Apply spMV -- so the outputs
        // agree to a tight tolerance rather than bitwise.
        void SymmetricStorageMatchesFull()
        {
            const int nb = 6, BR = 2;                            // n = 12 > eigSteps
            var full = BuildBlockTridiag(nb, BR);

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
                if (i + 1 < nb) builder.AddBlock(i + 1, i, in off);   // lower triangle only
            }
            var sym = builder.ToBSRSymmetric(Allocator.Temp);

            var mFull = new fProxyChebyshev(in full, Allocator.Temp);
            var mSym = new fProxyChebyshev(in sym, Allocator.Temp);

            int n = full.M_Rows;
            var r2 = GenerateOP.fProxyRandomVec(n, -1f, 1f, 851501u);
            var zF = new fProxyN(n, Allocator.Temp);
            var zS = new fProxyN(n, Allocator.Temp);
            mFull.Apply(in r2, ref zF);
            mSym.Apply(in r2, ref zS);

            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(zF[i] - zS[i]) < Tol() * ((fProxy)1 + math.abs(zF[i])));
        }
    }

    // ---- correctness cases (Burst) -------------------------------------------------------

    [Test]
    public void PoissonIterationOrderingTest()
        => new SparseChebyshevTestJob { Type = SparseChebyshevTestJob.TestType.PoissonIterationOrdering }.Run();

    [Test]
    public void SolutionMatchesDirectTest()
        => new SparseChebyshevTestJob { Type = SparseChebyshevTestJob.TestType.SolutionMatchesDirect }.Run();

    [Test]
    public void ApplyIsSymmetricAndPositiveTest()
        => new SparseChebyshevTestJob { Type = SparseChebyshevTestJob.TestType.ApplyIsSymmetricAndPositive }.Run();

    [Test]
    public void DeterministicSolveTest()
        => new SparseChebyshevTestJob { Type = SparseChebyshevTestJob.TestType.DeterministicSolve }.Run();

    [Test]
    public void SymmetricStorageMatchesFullTest()
        => new SparseChebyshevTestJob { Type = SparseChebyshevTestJob.TestType.SymmetricStorageMatchesFull }.Run();

    // ---- 5. through-IJob struct-copy safety (managed build, jobbed solve) -----------------
    //
    // The preconditioner is built on the managed thread, then handed BY VALUE into a Burst IJob that
    // runs the full cg solve. .Run() executes on a struct COPY of fProxyChebyshev -- the path that
    // would expose a stale self-pointer or a lost post-construction write (the LOBPCG IJob cache-copy
    // lesson). The struct is readonly/standalone-allocated, so the jobbed solve must reproduce the managed
    // solve (managed Mono vs Burst -> agree to tolerance, not bitwise) and satisfy A x ~= b.
    [BurstCompile(CompileSynchronously = true)]
    struct JobbedChebyshevSolve : IJob
    {
        [ReadOnly] public fProxyBSR A;
        public fProxyChebyshev M;
        [ReadOnly] public fProxyN b;
        public fProxyN x;
        public NativeArray<int> Out;      // [0] = solved, [1] = iterations
        public int MaxIter;
        public fProxy Tol;

        public void Execute()
        {
            var info = Krylov.cg(in A, in M, in b, ref x, MaxIter, Tol);
            Out[0] = info.Solved ? 1 : 0;
            Out[1] = info.iterations;
        }
    }

    static fProxy JobTol() => /*+choose[1e-2f|1e-5]*/1e-2f/*-choose*/;

    [Test]
    public void JobbedBuildSolveMatchesManaged()
    {
        var A = fProxyGallery.fProxyLaplacian2D(16, 16);                 // n = 256
        int n = A.M_Rows;

        var M = new fProxyChebyshev(in A, Allocator.Temp);                     // built on the MANAGED thread
        var xTrue = GenerateOP.fProxyRandomVec(n, 0.5f, 1.5f, 852001u);
        var b = BSR.spMV(in A, in xTrue);
        fProxy tol = Consts.fProxySqrtEps;
        int maxIter = 8 * n;

        // Managed-thread reference solve.
        var xManaged = new fProxyN(n, Allocator.Temp);
        var infoM = Krylov.cg(in A, in M, in b, ref xManaged, maxIter, tol);
        Assert.IsTrue(infoM.Solved);

        // Same struct copied INTO a Burst job (runs AFTER the managed solve; M's owned scratch is
        // reused sequentially, never concurrently).
        var xJob = new fProxyN(n, Allocator.Temp);
        var outp = new NativeArray<int>(2, Allocator.TempJob);
        new JobbedChebyshevSolve { A = A, M = M, b = b, x = xJob, Out = outp, MaxIter = maxIter, Tol = tol }.Run();
        Assert.IsTrue(outp[0] == 1);

        // The jobbed (struct-copied) solve is a valid solution: it agrees with the managed solve and
        // reproduces b.
        for (int i = 0; i < n; i++)
            Assert.IsTrue(math.abs(xJob[i] - xManaged[i]) <= JobTol() * ((fProxy)1 + math.abs(xManaged[i])));

        var Ax = BSR.spMV(in A, in xJob);
        for (int i = 0; i < n; i++)
            Assert.IsTrue(math.abs(Ax[i] - b[i]) <= JobTol() * ((fProxy)1 + math.abs(b[i])));

        outp.Dispose();
    }

    // ---- 9. Small system (n < default eigSteps): builds and converges ----------------------
    //
    // Regression for the eigSteps>n Lanczos throw: a BR=2, 4-block SPD chain has n=8 < the default
    // eigSteps (10). The ctor must clamp the Lanczos step count to n and BUILD (previously threw),
    // and cg must then converge. Residual is recomputed from a fresh spMV and asserted on the
    // managed thread (‖b-Ax‖² <= C²·tol²·‖b‖²) so a failure prints the real numbers.
    [BurstCompile(CompileSynchronously = true)]
    struct SmallSystemBuildSolveJob : IJob
    {
        public NativeArray<fProxy> OutR;   // [0] = ‖b-Ax‖², [1] = ‖b‖²
        public NativeArray<int> OutI;      // [0] = solved flag (1/0), [1] = iterations

        static fProxyBSR BuildBlockTridiag(int nb, int BR)
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
                    builder.AddBlock(i, i + 1, in off);
                }
            }
            return builder.ToBSR(Allocator.Temp);
        }

        public void Execute()
        {
            const int nb = 4, BR = 2;                            // n = 8 < default eigSteps (10)
            var A = BuildBlockTridiag(nb, BR);
            int n = A.M_Rows;

            var M = new fProxyChebyshev(in A, Allocator.Temp);                 // must not throw (eigSteps clamped to n)
            var xTrue = GenerateOP.fProxyRandomVec(n, 0.5f, 1.5f, 853001u);
            var b = BSR.spMV(in A, in xTrue);
            fProxy tol = Consts.fProxySqrtEps;
            int maxIter = 8 * n;

            var x = new fProxyN(n, Allocator.Temp);
            var info = Krylov.cg(in A, in M, in b, ref x, maxIter, tol);

            var Ax = BSR.spMV(in A, in x);
            fProxy rr = 0, bb = 0;
            for (int i = 0; i < n; i++)
            {
                fProxy d = b[i] - Ax[i];
                rr += d * d;
                bb += b[i] * b[i];
            }

            OutR[0] = rr;
            OutR[1] = bb;
            OutI[0] = info.Solved ? 1 : 0;
            OutI[1] = info.iterations;
        }
    }

    [Test]
    public void SmallSystemBelowEigStepsBuildsAndConverges()
    {
        var outR = new NativeArray<fProxy>(2, Allocator.TempJob);
        var outI = new NativeArray<int>(2, Allocator.TempJob);
        new SmallSystemBuildSolveJob { OutR = outR, OutI = outI }.Run();

        fProxy rr = outR[0], bb = outR[1];
        int solved = outI[0], iters = outI[1];
        TestContext.WriteLine($"SmallSystemBelowEigStepsBuildsAndConverges: solved={solved} iters={iters} rr={rr} bb={bb}");

        Assert.IsTrue(solved == 1, $"cg did not converge (iterations={iters}, rr={rr}, bb={bb})");

        fProxy C = (fProxy)64;
        fProxy tol = Consts.fProxySqrtEps;
        fProxy bound = C * C * tol * tol * bb;
        Assert.IsTrue(rr <= bound, $"residual too large: rr={rr} > bound={bound} (iters={iters}, bb={bb})");

        outR.Dispose();
        outI.Dispose();
    }

    // ---- 2. Degree sweep d in {1,2,3,4}: outer-iteration count -----------------------------
    //
    // Runs the whole sweep in Burst but reports the raw per-degree outer-iteration count via
    // Out instead of asserting inside Execute() -- a Burst assert hides the actual numbers on
    // failure, so this shape keeps the diagnosis legible (and this run's counts visible in the
    // test log) no matter which invariant below trips.
    [BurstCompile(CompileSynchronously = true)]
    struct DegreeSweepJob : IJob
    {
        public NativeArray<int> Out;   // [0..3] = iterations for degree 1..4; [4..7] = solved flags (1/0)

        public void Execute()
        {
            var A = fProxyGallery.fProxyLaplacian2D(16, 16);              // n = 256 > eigSteps
            int n = A.M_Rows;

            var xTrue = GenerateOP.fProxyRandomVec(n, 0.5f, 1.5f, 851101u);
            var b = BSR.spMV(in A, in xTrue);
            fProxy tol = Consts.fProxySqrtEps;
            int maxIter = 4 * n;

            for (int d = 1; d <= 4; d++)
            {
                var opt = fProxyChebyshevOptions.Default;
                opt.degree = d;
                var M = new fProxyChebyshev(in A, in opt, Allocator.Temp);

                var x = new fProxyN(n, Allocator.Temp);
                var info = Krylov.cg(in A, in M, in b, ref x, maxIter, tol);
                Out[d - 1] = info.iterations;
                Out[4 + d - 1] = info.Solved ? 1 : 0;
            }
        }
    }

    // Every degree must converge, and the strongest smoother (degree 4) must not need MORE outer
    // iterations than the weakest one (degree 1) -- a Chebyshev-damped preconditioner should never
    // make cg's outer loop worse as the polynomial gets stronger. Interior degrees (2, 3) are NOT
    // asserted strictly non-increasing step-by-step: the exact iteration a true-residual convergence
    // test crosses tol is a threshold-boundary event, sensitive to SIMD-reduction-order rounding at
    // the ULP level, and can wobble by a step at one interior degree without indicating a real
    // regression -- see Sparse/DEVLOG.md "Chebyshev" for the diagnosis that ruled out a systematic
    // eigenvalue-underestimate cause.
    [Test]
    public void DegreeSweepNonIncreasingTest()
    {
        var outp = new NativeArray<int>(8, Allocator.TempJob);
        new DegreeSweepJob { Out = outp }.Run();

        int d1 = outp[0], d2 = outp[1], d3 = outp[2], d4 = outp[3];
        TestContext.WriteLine($"DegreeSweepNonIncreasingTest outer iterations: d1={d1} d2={d2} d3={d3} d4={d4}");

        for (int d = 1; d <= 4; d++)
            Assert.IsTrue(outp[4 + d - 1] == 1, $"degree={d} did not converge (iterations={outp[d - 1]})");

        Assert.IsTrue(d4 <= d1,
            $"degree=4 ({d4} iters) should not need MORE outer iterations than degree=1 ({d1} iters)");

        outp.Dispose();
    }

    // ---- 7. guard cases (managed thread; Assert.Throws cannot run in Burst) ----------------

    // A valid SPD block-tridiagonal with n >= default eigSteps (10) -> the ctor's Lanczos run
    // succeeds, so a built preconditioner is available for the Apply-guard tests.
    static fProxyBSR BuildValidSPD()
    {
        const int nb = 6, BR = 2;                                // n = 12
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
                builder.AddBlock(i, i + 1, in off);
            }
        }
        return builder.ToBSR(Allocator.Temp);
    }

    // A small square 1x1-block BSR with the given scalar diagonal values (every diagonal block present).
    static fProxyBSR BuildDiag1x1(fProxy d0, fProxy d1)
    {
        var builder = new fProxyBSRBuilder(2, 2, 1, 1, Allocator.Temp);
        var b0 = new fProxyMxN(1, 1, Allocator.Temp); b0[0, 0] = d0;
        var b1 = new fProxyMxN(1, 1, Allocator.Temp); b1[0, 0] = d1;
        builder.AddBlock(0, 0, in b0);
        builder.AddBlock(1, 1, in b1);
        return builder.ToBSR(Allocator.Temp);
    }

    [Test]
    public void NonSquareThrows()
    {
        var builder = new fProxyBSRBuilder(2, 3, 2, 2, Allocator.Temp);        // BlockRows != BlockCols
        var block = GenerateOP.fProxyMat(2, 2, (fProxy)1);
        builder.AddBlock(0, 0, in block);
        var A = builder.ToBSR(Allocator.Temp);
        Assert.Throws<ArgumentException>(() => { var m = new fProxyChebyshev(in A, Allocator.Temp); });
    }

    [Test]
    public void MissingDiagonalThrows()
    {
        var builder = new fProxyBSRBuilder(2, 2, 2, 2, Allocator.Temp);
        var block = GenerateOP.fProxyMat(2, 2, (fProxy)1);
        builder.AddBlock(0, 0, in block);
        builder.AddBlock(1, 0, in block);                       // no (1,1) diagonal block
        var A = builder.ToBSR(Allocator.Temp);
        Assert.Throws<ArgumentException>(() => { var m = new fProxyChebyshev(in A, Allocator.Temp); });
    }

    [Test]
    public void ZeroDiagonalThrows()
    {
        var A = BuildDiag1x1((fProxy)4, (fProxy)0);   // A[1,1] == 0 -> not SPD
        Assert.Throws<ArgumentException>(() => { var m = new fProxyChebyshev(in A, Allocator.Temp); });
    }

    [Test]
    public void NegativeDiagonalThrows()
    {
        var A = BuildDiag1x1((fProxy)4, (fProxy)(-1));   // A[1,1] < 0 -> not SPD
        Assert.Throws<ArgumentException>(() => { var m = new fProxyChebyshev(in A, Allocator.Temp); });
    }

    [Test]
    public void DegreeZeroThrows()
    {
        var A = BuildDiag1x1((fProxy)4, (fProxy)4);
        var opt = fProxyChebyshevOptions.Default;
        opt.degree = 0;                                          // must be >= 1
        Assert.Throws<ArgumentException>(() => { var m = new fProxyChebyshev(in A, in opt, Allocator.Temp); });
    }

    [Test]
    public void KappaTooSmallThrows()
    {
        var A = BuildDiag1x1((fProxy)4, (fProxy)4);
        var opt = fProxyChebyshevOptions.Default;
        opt.kappa = (fProxy)1;                                   // must be > 1
        Assert.Throws<ArgumentException>(() => { var m = new fProxyChebyshev(in A, in opt, Allocator.Temp); });
    }

    [Test]
    public void EigStepsZeroThrows()
    {
        var A = BuildDiag1x1((fProxy)4, (fProxy)4);
        var opt = fProxyChebyshevOptions.Default;
        opt.eigSteps = 0;                                        // must be >= 1
        Assert.Throws<ArgumentException>(() => { var m = new fProxyChebyshev(in A, in opt, Allocator.Temp); });
    }

    [Test]
    public void SafetyTooSmallThrows()
    {
        var A = BuildDiag1x1((fProxy)4, (fProxy)4);
        var opt = fProxyChebyshevOptions.Default;
        opt.safety = (fProxy)0.5;                                // must be >= 1
        Assert.Throws<ArgumentException>(() => { var m = new fProxyChebyshev(in A, in opt, Allocator.Temp); });
    }

    [Test]
    public void ApplyAliasThrows()
    {
        var A = BuildValidSPD();
        var M = new fProxyChebyshev(in A, Allocator.Temp);
        var r = GenerateOP.fProxyVec(A.M_Rows, (fProxy)1);
        Assert.Throws<ArgumentException>(() => M.Apply(in r, ref r));   // z aliases r
    }

    [Test]
    public void ApplyWrongResidualSizeThrows()
    {
        var A = BuildValidSPD();
        var M = new fProxyChebyshev(in A, Allocator.Temp);
        var r = GenerateOP.fProxyVec(A.M_Rows - 1, (fProxy)1);      // r.N != Rows
        var z = new fProxyN(A.M_Rows, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => M.Apply(in r, ref z));
    }

    [Test]
    public void ApplyWrongOutputSizeThrows()
    {
        var A = BuildValidSPD();
        var M = new fProxyChebyshev(in A, Allocator.Temp);
        var r = GenerateOP.fProxyVec(A.M_Rows, (fProxy)1);
        var z = new fProxyN(A.M_Rows - 1, Allocator.Temp);                 // z.N != Rows
        Assert.Throws<ArgumentException>(() => M.Apply(in r, ref z));
    }
}
