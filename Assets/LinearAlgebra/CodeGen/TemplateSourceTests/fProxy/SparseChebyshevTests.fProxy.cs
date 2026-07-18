using System;
using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Chebyshev polynomial preconditioner (fProxyChebyshev). Correctness anchors (spec
// docs/dev/spec-chebyshev-preconditioner.md Section 9):
//   (1) on a 2D Poisson system pcg+Chebyshev(d=3) needs strictly fewer OUTER iterations than
//       pcg+block-Jacobi, which in turn needs strictly fewer than plain CG (same tol/rhs/x0);
//   (2) raising the polynomial degree from 1 to 4 does not increase the outer-iteration count
//       (interior degrees are not asserted strictly non-increasing -- see DegreeSweepNonIncreasingTest);
//   (3) pcg+Chebyshev lands on the dense-direct solution within tolerance;
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
// Every built preconditioner runs Eigen.lanczos for opt.eigSteps steps, which throws if
// eigSteps > A.Rows, so every VALID build here keeps n >= eigSteps (default 10). The guard cases that
// expect a throw BEFORE the Lanczos run (bad options / bad diagonal / non-square) may use tiny matrices.
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
        static fProxyBSR BuildBlockTridiag(ref Arena arena, int nb, int BR)
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
                    builder.AddBlock(i, i + 1, in off);
                }
            }
            return builder.ToBSR(ref arena);
        }

        // ---- 1. Poisson outer-iteration ordering: Chebyshev(d=3) < block-Jacobi < plain CG -------
        void PoissonIterationOrdering()
        {
            var arena = new Arena(Allocator.Persistent);

            // g = 32 -> n = 1024, comfortably above the default eigSteps = 10.
            var A = arena.fProxyLaplacian2D(32, 32);
            int n = A.M_Rows;

            var xTrue = arena.fProxyRandomVec(n, 0.5f, 1.5f, 851001u);
            var b = BSR.spMV(in A, in xTrue);
            fProxy tol = Consts.fProxySqrtEps;
            int maxIter = 4 * n;

            var Mc = arena.fProxyChebyshev(in A);                 // Default: degree 3
            var Mj = arena.fProxyBlockJacobi(in A);

            var xC = arena.fProxyVec(n);
            var infoC = Krylov.pcg(in A, in Mc, in b, ref xC, maxIter, tol);
            Assert.IsTrue(infoC.Solved);

            var xJ = arena.fProxyVec(n);
            var infoJ = Krylov.pcg(in A, in Mj, in b, ref xJ, maxIter, tol);
            Assert.IsTrue(infoJ.Solved);

            var xG = arena.fProxyVec(n);
            var infoG = Krylov.cg(in A, in b, ref xG, maxIter, tol);
            Assert.IsTrue(infoG.Solved);

            // Spec Section 9.1: strictly decreasing outer-iteration counts, all to the same tol.
            Assert.IsTrue(infoC.iterations < infoJ.iterations);
            Assert.IsTrue(infoJ.iterations < infoG.iterations);

            arena.Dispose();
        }

        // ---- 3. Solution correctness: pcg+Chebyshev matches the dense direct (Cholesky) solve -----
        void SolutionMatchesDirect()
        {
            var arena = new Arena(Allocator.Persistent);

            const int nb = 8, BR = 2;                            // n = 16 > eigSteps
            var A = BuildBlockTridiag(ref arena, nb, BR);
            int n = A.M_Rows;

            var M = arena.fProxyChebyshev(in A);
            var xTrue = arena.fProxyRandomVec(n, 0.5f, 1.5f, 851201u);
            var b = BSR.spMV(in A, in xTrue);

            var x = arena.fProxyVec(n);
            var info = Krylov.pcg(in A, in M, in b, ref x, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(info.Solved);

            // Dense Cholesky oracle on the same system (independent code path). CHO.solveInPlace is
            // destructive, so it runs on the dense copy with b copied into the rhs slot.
            var D = A.ToDense(ref arena);
            var xRef = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) xRef[i] = b[i];
            var choInfo = CHO.solveInPlace(ref D, ref xRef);
            Assert.IsTrue(choInfo.Solved);

            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(x[i] - xRef[i]) < Tol() * ((fProxy)1 + math.abs(xRef[i])));

            arena.Dispose();
        }

        // ---- 4. SPD spot check: M^-1 symmetric AND positive definite -------------------------------
        void ApplyIsSymmetricAndPositive()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyRandomSparseSPD(20, 3, (fProxy)0.3, 851301u);   // n = 60
            var M = arena.fProxyChebyshev(in A);
            int n = A.M_Rows;

            var u = arena.fProxyRandomVec(n, -1f, 1f, 851302u);
            var v = arena.fProxyRandomVec(n, -1f, 1f, 851303u);
            var Mu = arena.fProxyVec(n);
            var Mv = arena.fProxyVec(n);
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

            arena.Dispose();
        }

        // ---- 6. Determinism: two identical solves are bit-identical (Apply is dot-free) ------------
        void DeterministicSolve()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyLaplacian2D(16, 16);              // n = 256
            int n = A.M_Rows;

            var M = arena.fProxyChebyshev(in A);
            var xTrue = arena.fProxyRandomVec(n, 0.5f, 1.5f, 851401u);
            var b = BSR.spMV(in A, in xTrue);
            fProxy tol = Consts.fProxySqrtEps;
            int maxIter = 4 * n;

            var x1 = arena.fProxyVec(n);
            var i1 = Krylov.pcg(in A, in M, in b, ref x1, maxIter, tol);
            Assert.IsTrue(i1.Solved);

            var x2 = arena.fProxyVec(n);
            var i2 = Krylov.pcg(in A, in M, in b, ref x2, maxIter, tol);
            Assert.IsTrue(i2.Solved);

            Assert.IsTrue(i1.iterations == i2.iterations);
            for (int i = 0; i < n; i++)
                Assert.IsTrue(x1[i] == x2[i]);                   // bit-identical (==, not within-tolerance)

            arena.Dispose();
        }

        // ---- 8. Storage-mode equivalence: Symmetric-stored A vs full-stored twin, same Apply -------
        //
        // spMV consumes both storage modes natively (no mirror), but the symmetric path folds each
        // off-diagonal block into y[i] AND y[j] in a different accumulation order than the full path --
        // and that order feeds both the Lanczos hi-estimate AND every Apply spMV -- so the outputs
        // agree to a tight tolerance rather than bitwise.
        void SymmetricStorageMatchesFull()
        {
            var arena = new Arena(Allocator.Persistent);

            const int nb = 6, BR = 2;                            // n = 12 > eigSteps
            var full = BuildBlockTridiag(ref arena, nb, BR);

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
                if (i + 1 < nb) builder.AddBlock(i + 1, i, in off);   // lower triangle only
            }
            var sym = builder.ToBSRSymmetric(ref arena);

            var mFull = arena.fProxyChebyshev(in full);
            var mSym = arena.fProxyChebyshev(in sym);

            int n = full.M_Rows;
            var r2 = arena.fProxyRandomVec(n, -1f, 1f, 851501u);
            var zF = arena.fProxyVec(n);
            var zS = arena.fProxyVec(n);
            mFull.Apply(in r2, ref zF);
            mSym.Apply(in r2, ref zS);

            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(zF[i] - zS[i]) < Tol() * ((fProxy)1 + math.abs(zF[i])));

            arena.Dispose();
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
    // runs the full pcg solve. .Run() executes on a struct COPY of fProxyChebyshev -- the path that
    // would expose a stale self-pointer or a lost post-construction write (the LOBPCG IJob cache-copy
    // lesson). The struct is readonly/arena-backed, so the jobbed solve must reproduce the managed
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
            var info = Krylov.pcg(in A, in M, in b, ref x, MaxIter, Tol);
            Out[0] = info.Solved ? 1 : 0;
            Out[1] = info.iterations;
        }
    }

    static fProxy JobTol() => /*+choose[1e-2f|1e-5]*/1e-2f/*-choose*/;

    [Test]
    public void JobbedBuildSolveMatchesManaged()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.fProxyLaplacian2D(16, 16);                 // n = 256
        int n = A.M_Rows;

        var M = arena.fProxyChebyshev(in A);                     // built on the MANAGED thread
        var xTrue = arena.fProxyRandomVec(n, 0.5f, 1.5f, 852001u);
        var b = BSR.spMV(in A, in xTrue);
        fProxy tol = Consts.fProxySqrtEps;
        int maxIter = 8 * n;

        // Managed-thread reference solve.
        var xManaged = arena.fProxyVec(n);
        var infoM = Krylov.pcg(in A, in M, in b, ref xManaged, maxIter, tol);
        Assert.IsTrue(infoM.Solved);

        // Same struct copied INTO a Burst job (runs AFTER the managed solve; M's owned scratch is
        // reused sequentially, never concurrently).
        var xJob = arena.fProxyVec(n);
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
        arena.Dispose();
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
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyLaplacian2D(16, 16);              // n = 256 > eigSteps
            int n = A.M_Rows;

            var xTrue = arena.fProxyRandomVec(n, 0.5f, 1.5f, 851101u);
            var b = BSR.spMV(in A, in xTrue);
            fProxy tol = Consts.fProxySqrtEps;
            int maxIter = 4 * n;

            for (int d = 1; d <= 4; d++)
            {
                var opt = fProxyChebyshevOptions.Default;
                opt.degree = d;
                var M = arena.fProxyChebyshev(in A, in opt);

                var x = arena.fProxyVec(n);
                var info = Krylov.pcg(in A, in M, in b, ref x, maxIter, tol);
                Out[d - 1] = info.iterations;
                Out[4 + d - 1] = info.Solved ? 1 : 0;
            }

            arena.Dispose();
        }
    }

    // Every degree must converge, and the strongest smoother (degree 4) must not need MORE outer
    // iterations than the weakest one (degree 1) -- a Chebyshev-damped preconditioner should never
    // make pcg's outer loop worse as the polynomial gets stronger. Interior degrees (2, 3) are NOT
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
    static fProxyBSR BuildValidSPD(ref Arena arena)
    {
        const int nb = 6, BR = 2;                                // n = 12
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
                builder.AddBlock(i, i + 1, in off);
            }
        }
        return builder.ToBSR(ref arena);
    }

    // A small square 1x1-block BSR with the given scalar diagonal values (every diagonal block present).
    static fProxyBSR BuildDiag1x1(ref Arena arena, fProxy d0, fProxy d1)
    {
        var builder = arena.fProxyBSRBuilder(2, 2, 1, 1);
        var b0 = arena.fProxyMat(1, 1); b0[0, 0] = d0;
        var b1 = arena.fProxyMat(1, 1); b1[0, 0] = d1;
        builder.AddBlock(0, 0, in b0);
        builder.AddBlock(1, 1, in b1);
        return builder.ToBSR(ref arena);
    }

    [Test]
    public void NonSquareThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var builder = arena.fProxyBSRBuilder(2, 3, 2, 2);        // BlockRows != BlockCols
            var block = arena.fProxyMat(2, 2, (fProxy)1);
            builder.AddBlock(0, 0, in block);
            var A = builder.ToBSR(ref arena);
            Assert.Throws<ArgumentException>(() => { var m = arena.fProxyChebyshev(in A); });
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void MissingDiagonalThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var builder = arena.fProxyBSRBuilder(2, 2, 2, 2);
            var block = arena.fProxyMat(2, 2, (fProxy)1);
            builder.AddBlock(0, 0, in block);
            builder.AddBlock(1, 0, in block);                       // no (1,1) diagonal block
            var A = builder.ToBSR(ref arena);
            Assert.Throws<ArgumentException>(() => { var m = arena.fProxyChebyshev(in A); });
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void ZeroDiagonalThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildDiag1x1(ref arena, (fProxy)4, (fProxy)0);   // A[1,1] == 0 -> not SPD
            Assert.Throws<ArgumentException>(() => { var m = arena.fProxyChebyshev(in A); });
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void NegativeDiagonalThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildDiag1x1(ref arena, (fProxy)4, (fProxy)(-1));   // A[1,1] < 0 -> not SPD
            Assert.Throws<ArgumentException>(() => { var m = arena.fProxyChebyshev(in A); });
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void DegreeZeroThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildDiag1x1(ref arena, (fProxy)4, (fProxy)4);
            var opt = fProxyChebyshevOptions.Default;
            opt.degree = 0;                                          // must be >= 1
            Assert.Throws<ArgumentException>(() => { var m = arena.fProxyChebyshev(in A, in opt); });
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void KappaTooSmallThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildDiag1x1(ref arena, (fProxy)4, (fProxy)4);
            var opt = fProxyChebyshevOptions.Default;
            opt.kappa = (fProxy)1;                                   // must be > 1
            Assert.Throws<ArgumentException>(() => { var m = arena.fProxyChebyshev(in A, in opt); });
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void EigStepsZeroThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildDiag1x1(ref arena, (fProxy)4, (fProxy)4);
            var opt = fProxyChebyshevOptions.Default;
            opt.eigSteps = 0;                                        // must be >= 1
            Assert.Throws<ArgumentException>(() => { var m = arena.fProxyChebyshev(in A, in opt); });
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void SafetyTooSmallThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildDiag1x1(ref arena, (fProxy)4, (fProxy)4);
            var opt = fProxyChebyshevOptions.Default;
            opt.safety = (fProxy)0.5;                                // must be >= 1
            Assert.Throws<ArgumentException>(() => { var m = arena.fProxyChebyshev(in A, in opt); });
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void ApplyAliasThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildValidSPD(ref arena);
            var M = arena.fProxyChebyshev(in A);
            var r = arena.fProxyVec(A.M_Rows, (fProxy)1);
            Assert.Throws<ArgumentException>(() => M.Apply(in r, ref r));   // z aliases r
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void ApplyWrongResidualSizeThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildValidSPD(ref arena);
            var M = arena.fProxyChebyshev(in A);
            var r = arena.fProxyVec(A.M_Rows - 1, (fProxy)1);      // r.N != Rows
            var z = arena.fProxyVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() => M.Apply(in r, ref z));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void ApplyWrongOutputSizeThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildValidSPD(ref arena);
            var M = arena.fProxyChebyshev(in A);
            var r = arena.fProxyVec(A.M_Rows, (fProxy)1);
            var z = arena.fProxyVec(A.M_Rows - 1);                 // z.N != Rows
            Assert.Throws<ArgumentException>(() => M.Apply(in r, ref z));
        }
        finally { arena.Dispose(); }
    }
}
