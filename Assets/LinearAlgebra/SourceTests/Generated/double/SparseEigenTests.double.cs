using System;

using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Sparse (BSM) power-iteration test suite for the matrix-free Eigen.powerIteration<TOp> refactor:
// the new powerIteration(in doubleBSM, ...) overloads forward through doubleBSMOperator into the
// same generic core the dense powerIteration(in doubleMxN, ...) path uses. Every sparse result is
// cross-checked against the pre-existing dense path (same recipe as doubleSparseSolverTests: build
// the SAME operator in both forms and compare), plus one literature known-spectrum case.
//
// Correctness cases run inside a [BurstCompile] IJob (matches doubleSparseSolverTests /
// doubleEigenTests) and use doubleEigenTests' Fail-NativeArray diagnostic convention: a failed
// Assert inside a Burst job aborts silently without surfacing to the runner, so every check first
// records [0]=flag, [1]=got, [2]=expected/limit, [3]=diff into Fail. Guard/exception cases run on
// the managed test thread with Assert.Throws, since NUnit's Assert.Throws cannot execute in Burst.
public class doubleSparseEigenTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SparseEigenTestJob : IJob
    {
        public enum TestType
        {
            DenseVsSparseCrossCheck,
            LaplacianKnownSpectrum,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff/code
        public NativeArray<double> Fail;

        // Two INDEPENDENTLY-converged iterative eigenpairs (dense manual-matvec vs BSM spMV) are
        // compared, so a machine-epsilon threshold is inappropriate: mirror doubleSparseSolverTests'
        // choose-marker tolerance for iterative-vs-iterative cross-checks (looser on float).
        static double LooseTol() => 1e-5;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.DenseVsSparseCrossCheck: DenseVsSparseCrossCheck(); break;
                case TestType.LaplacianKnownSpectrum: LaplacianKnownSpectrum(); break;
            }
        }

        // ---- helpers ---------------------------------------------------------------------

        // Same recipe as doubleSparseSolverTests.BuildDenseSPD: A = M^T M + dim*I -> strictly SPD /
        // diagonally dominant, so it has a single clearly dominant (positive) eigenvalue that power
        // iteration converges to unambiguously.
        static doubleMxN BuildDenseSPD(ref Arena arena, int dim, uint seed)
        {
            var M = arena.doubleRandomMat(dim, dim, -1f, 1f, seed);
            var A = Linear_OP.dot(M, M, true);
            for (int d = 0; d < dim; d++)
                A[d, d] += dim;
            return A;
        }

        // 1x1-block BSM built from a dense matrix's nonzero entries via AddValue (identical to
        // doubleSparseSolverTests.DenseToBSM1x1). nnzHint bounds the known nonzero pattern purely as
        // a perf hint; growth past it is safe (the builder's triplet state lives behind a shared
        // heap pointer). Encodes the SAME numeric operator as the dense form, so spMV(bsm,.) and the
        // dense matvec agree up to floating-point reassociation only.
        static doubleBSM DenseToBSM1x1(ref Arena arena, in doubleMxN A, int nnzHint)
        {
            var builder = arena.doubleBSMBuilder(A.M_Rows, A.N_Cols, 1, 1, math.max(nnzHint, 1));
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    if (A[r, c] != (double)0)
                        builder.AddValue(r, c, A[r, c]);
            return builder.ToBSM(ref arena);
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff
        void AssertClose(double a, double b, double precision)
        {
            double diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        // Boolean-return guard: record a distinguishing `code` in [1] so a silent Burst abort is
        // still diagnosable ([2]/[3] unused). Used for the powerIteration convergence flags.
        void AssertTrue(bool cond, double code)
        {
            if (!cond && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = code;
                Fail[2] = (double)0;
                Fail[3] = (double)0;
            }
            Assert.IsTrue(cond);
        }

        // Two unit eigenvectors are equal up to an overall sign: align the sign on a's
        // largest-magnitude component (robust to a near-zero pivot), then compare elementwise.
        void AssertVecEqUpToSign(in doubleN a, in doubleN b, int n, double absTol)
        {
            int piv = 0;
            double best = (double)0;
            for (int i = 0; i < n; i++)
            {
                double m = math.abs(a[i]);
                if (m > best) { best = m; piv = i; }
            }
            double sign = (a[piv] * b[piv] >= (double)0) ? (double)1 : (double)(-1);

            double maxErr = (double)0;
            for (int i = 0; i < n; i++)
            {
                double e = math.abs(a[i] - sign * b[i]);
                if (e > maxErr) maxErr = e;
            }
            if (!(maxErr <= absTol) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = maxErr;
                Fail[2] = absTol;
                Fail[3] = maxErr - absTol;
            }
            Assert.IsTrue(maxErr <= absTol);
        }

        // Residual property ||Av - lambda*v||_inf <= limit, where Av is supplied precomputed (here
        // from Sparse_OP.spMV on the BSM) and limit scales with max(1,|lambda|). Mirrors
        // doubleEigenTests.AssertPowerResidual but takes Av directly so the BSM matvec is the thing
        // under test.
        void AssertResidual(in doubleN Av, in doubleN v, double lambda, double limitBase, int n)
        {
            double maxRes = (double)0;
            for (int i = 0; i < n; i++)
            {
                double ri = math.abs(Av[i] - lambda * v[i]);
                if (ri > maxRes) maxRes = ri;
            }
            double scale = math.abs(lambda);
            if (scale < (double)1) scale = (double)1;
            double limit = limitBase * scale;
            if (!(maxRes <= limit) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = maxRes;
                Fail[2] = limit;
                Fail[3] = maxRes - limit;
            }
            Assert.IsTrue(maxRes <= limit);
        }

        // ---- (a) dense-vs-sparse cross-check ---------------------------------------------
        //
        // Build one SPD operator, run powerIteration on the dense form and on the 1x1-block BSM form
        // from the SAME zero-seeded v (deterministic internal seeding -> both iterations start from
        // the identical vector). Both must converge; the eigenvalues must agree closely and the
        // eigenvectors up to an overall sign. This is the sparse path's core acceptance criterion:
        // the BSM overload must reproduce the trusted dense overload's dominant eigenpair.
        void DenseVsSparseCrossCheck()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 12;
            var A = BuildDenseSPD(ref arena, dim, 20240702);
            var bsm = DenseToBSM1x1(ref arena, in A, dim * dim);

            double tol = (double)10 * Consts.doubleZeroThreshold;

            // Dense reference (v starts at zero -> deterministic seeding).
            var vDense = arena.doubleVec(dim);
            var wDense = arena.doubleVec(dim);
            bool okDense = Eigen.powerIteration(in A, ref vDense, ref wDense, out double lamDense, tol, 2000);
            AssertTrue(okDense, (double)1);

            // Sparse (BSM) path from an identically zero-seeded v.
            var vSparse = arena.doubleVec(dim);
            var wSparse = arena.doubleVec(dim);
            bool okSparse = Eigen.powerIteration(in bsm, ref vSparse, ref wSparse, out double lamSparse, tol, 2000);
            AssertTrue(okSparse, (double)2);

            // Eigenvalues agree (magnitude up to ~dim+order-of-M; scale the loose tolerance).
            double scale = (double)1 + math.abs(lamDense);
            AssertClose(lamSparse, lamDense, LooseTol() * scale);

            // Eigenvectors agree up to an overall sign (both are unit vectors).
            AssertVecEqUpToSign(in vDense, in vSparse, dim, LooseTol());

            arena.Dispose();
        }

        // ---- (b) literature known-spectrum on the BSM path -------------------------------
        //
        // n x n 1D-Laplacian tridiagonal (diag 2, off-diag -1): eigenvalues are EXACTLY
        // lambda_k = 2 - 2*cos(k*pi/(n+1)), k=1..n. The DOMINANT (largest) is k=n. Encode it as a
        // 1x1-block BSM (tridiagonal -> nnzHint = 3*n bounds the pattern) and run powerIteration on
        // the BSM form. Assert convergence, the closed-form dominant eigenvalue (computed in double
        // then cast, mirroring doubleEigenTests.EvSymLaplacian), and the residual A*v ~= lambda*v
        // using Sparse_OP.spMV on the BSM itself.
        void LaplacianKnownSpectrum()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 16;
            var A = arena.doubleLaplacian1D(n);
            var bsm = DenseToBSM1x1(ref arena, in A, 3 * n);

            double tol = (double)10 * Consts.doubleZeroThreshold;

            var v = arena.doubleVec(n);   // zero -> deterministic seeding
            var w = arena.doubleVec(n);
            bool ok = Eigen.powerIteration(in bsm, ref v, ref w, out double lambda, tol, 4000);
            AssertTrue(ok, (double)1);

            // Closed-form dominant eigenvalue (k = n), computed in double precision then cast.
            double lamD = 2.0 - 2.0 * math.cos(n * math.PI_DBL / (n + 1));
            double scale = (double)1 + math.abs((double)lamD);
            AssertClose(lambda, (double)lamD, (double)1000 * Consts.doubleZeroThreshold * scale);

            // Residual property on the BSM operator: A*v ~= lambda*v (A*v via spMV on the BSM).
            var Av = Sparse_OP.spMV(in bsm, in v);
            AssertResidual(in Av, in v, lambda, (double)100 * Consts.doubleZeroThreshold, n);

            arena.Dispose();
        }
    }

    // ---- correctness entry points (Burst job + Fail-array surfacing) ----------------------

    void RunCase(SparseEigenTestJob.TestType type)
    {
        var fail = new NativeArray<double>(4, Allocator.TempJob);
        try
        {
            new SparseEigenTestJob { Type = type, Fail = fail }.Run();
            // A failed in-job Assert aborts the Burst job WITHOUT throwing to the caller; surface
            // the recorded diagnostics here (same convention as doubleEigenTests).
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/code {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/code {fail[3]} ({e.Message})");
            throw;
        }
        finally { fail.Dispose(); }
    }

    [Test]
    public void DenseVsSparseCrossCheckTest()
        => RunCase(SparseEigenTestJob.TestType.DenseVsSparseCrossCheck);

    [Test]
    public void LaplacianKnownSpectrumTest()
        => RunCase(SparseEigenTestJob.TestType.LaplacianKnownSpectrum);

    // ---- guard / exception cases (managed thread; Assert.Throws can't run inside Burst) ----
    //
    // The BSM overloads forward into the same generic powerIteration<TOp> core, whose argument
    // guards throw ArgumentException on: A.Rows != A.Cols, v.N != A.Rows, w.N != A.Rows, v/w
    // aliasing, and maxIter < 1. Not exhaustive -- these just prove each guard fires on the BSM
    // entry point (matching doubleEigenTests' Power* throw tests, but via doubleBSM).

    // A square 4x4 (two 2x2 diagonal blocks) BSM -- both diagonal blocks present, well-formed.
    static doubleBSM BuildSquareBSM(ref Arena arena)
    {
        const int BR = 2, BC = 2;
        var builder = arena.doubleBSMBuilder(2, 2, BR, BC, 2);
        builder.AddBlock(0, 0, arena.doubleRandomMat(BR, BC, -1f, 1f, 71001));
        builder.AddBlock(1, 1, arena.doubleRandomMat(BR, BC, -1f, 1f, 71002));
        return builder.ToBSM(ref arena);
    }

    [Test]
    public void Power_NonSquareBSM_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            // 2x3 block grid of 2x2 blocks -> 4x6 (Rows != Cols). One block suffices; the
            // Rows != Cols guard fires before v/w are examined.
            const int BR = 2, BC = 2;
            var builder = arena.doubleBSMBuilder(2, 3, BR, BC, 1);
            builder.AddBlock(0, 0, arena.doubleRandomMat(BR, BC, -1f, 1f, 71101));
            var A = builder.ToBSM(ref arena);

            var v = arena.doubleVec(A.M_Rows);
            var w = arena.doubleVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref w, out double lambda, Consts.doubleZeroThreshold, 1000));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Power_WrongVLength_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareBSM(ref arena);   // 4x4
            var v = arena.doubleVec(A.M_Rows - 1); // wrong length
            var w = arena.doubleVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref w, out double lambda, Consts.doubleZeroThreshold, 1000));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Power_AliasingVAndW_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareBSM(ref arena);   // 4x4
            var v = arena.doubleVec(A.M_Rows);
            var wAlias = v; // w aliases v (struct copy shares Data.Ptr) -> guard must fire
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref wAlias, out double lambda, Consts.doubleZeroThreshold, 1000));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Power_BadMaxIter_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareBSM(ref arena);   // 4x4
            var v = arena.doubleVec(A.M_Rows);
            var w = arena.doubleVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref w, out double lambda, Consts.doubleZeroThreshold, 0));
        }
        finally { arena.Dispose(); }
    }
}
