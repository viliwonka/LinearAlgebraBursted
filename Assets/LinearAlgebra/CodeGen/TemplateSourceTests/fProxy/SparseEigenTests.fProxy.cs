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
// the new powerIteration(in fProxyBSM, ...) overloads forward through fProxyBSMOperator into the
// same generic core the dense powerIteration(in fProxyMxN, ...) path uses. Every sparse result is
// cross-checked against the pre-existing dense path (same recipe as fProxySparseSolverTests: build
// the SAME operator in both forms and compare), plus one literature known-spectrum case.
//
// Correctness cases run inside a [BurstCompile] IJob (matches fProxySparseSolverTests /
// fProxyEigenTests) and use fProxyEigenTests' Fail-NativeArray diagnostic convention: a failed
// Assert inside a Burst job aborts silently without surfacing to the runner, so every check first
// records [0]=flag, [1]=got, [2]=expected/limit, [3]=diff into Fail. Guard/exception cases run on
// the managed test thread with Assert.Throws, since NUnit's Assert.Throws cannot execute in Burst.
public class fProxySparseEigenTests
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
        public NativeArray<fProxy> Fail;

        // Two INDEPENDENTLY-converged iterative eigenpairs (dense manual-matvec vs BSM spMV) are
        // compared, so a machine-epsilon threshold is inappropriate: mirror fProxySparseSolverTests'
        // choose-marker tolerance for iterative-vs-iterative cross-checks (looser on float).
        static fProxy LooseTol() => /*+choose[1e-2f|1e-5]*/1e-2f/*-choose*/;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.DenseVsSparseCrossCheck: DenseVsSparseCrossCheck(); break;
                case TestType.LaplacianKnownSpectrum: LaplacianKnownSpectrum(); break;
            }
        }

        // ---- helpers ---------------------------------------------------------------------

        // Same recipe as fProxySparseSolverTests.BuildDenseSPD: A = M^T M + dim*I -> strictly SPD /
        // diagonally dominant, so it has a single clearly dominant (positive) eigenvalue that power
        // iteration converges to unambiguously.
        static fProxyMxN BuildDenseSPD(ref Arena arena, int dim, uint seed)
        {
            var M = arena.fProxyRandomMat(dim, dim, -1f, 1f, seed);
            var A = Linear_OP.dot(M, M, true);
            for (int d = 0; d < dim; d++)
                A[d, d] += dim;
            return A;
        }

        // 1x1-block BSM built from a dense matrix's nonzero entries via AddValue (identical to
        // fProxySparseSolverTests.DenseToBSM1x1). nnzHint bounds the known nonzero pattern purely as
        // a perf hint; growth past it is safe (the builder's triplet state lives behind a shared
        // heap pointer). Encodes the SAME numeric operator as the dense form, so spMV(bsm,.) and the
        // dense matvec agree up to floating-point reassociation only.
        static fProxyBSM DenseToBSM1x1(ref Arena arena, in fProxyMxN A, int nnzHint)
        {
            var builder = arena.fProxyBSMBuilder(A.M_Rows, A.N_Cols, 1, 1, math.max(nnzHint, 1));
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    if (A[r, c] != (fProxy)0)
                        builder.AddValue(r, c, A[r, c]);
            return builder.ToBSM(ref arena);
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff
        void AssertClose(fProxy a, fProxy b, fProxy precision)
        {
            fProxy diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        // Boolean-return guard: record a distinguishing `code` in [1] so a silent Burst abort is
        // still diagnosable ([2]/[3] unused). Used for the powerIteration convergence flags.
        void AssertTrue(bool cond, fProxy code)
        {
            if (!cond && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = code;
                Fail[2] = (fProxy)0;
                Fail[3] = (fProxy)0;
            }
            Assert.IsTrue(cond);
        }

        // Two unit eigenvectors are equal up to an overall sign: align the sign on a's
        // largest-magnitude component (robust to a near-zero pivot), then compare elementwise.
        void AssertVecEqUpToSign(in fProxyN a, in fProxyN b, int n, fProxy absTol)
        {
            int piv = 0;
            fProxy best = (fProxy)0;
            for (int i = 0; i < n; i++)
            {
                fProxy m = math.abs(a[i]);
                if (m > best) { best = m; piv = i; }
            }
            fProxy sign = (a[piv] * b[piv] >= (fProxy)0) ? (fProxy)1 : (fProxy)(-1);

            fProxy maxErr = (fProxy)0;
            for (int i = 0; i < n; i++)
            {
                fProxy e = math.abs(a[i] - sign * b[i]);
                if (e > maxErr) maxErr = e;
            }
            if (!(maxErr <= absTol) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = maxErr;
                Fail[2] = absTol;
                Fail[3] = maxErr - absTol;
            }
            Assert.IsTrue(maxErr <= absTol);
        }

        // Residual property ||Av - lambda*v||_inf <= limit, where Av is supplied precomputed (here
        // from Sparse_OP.spMV on the BSM) and limit scales with max(1,|lambda|). Mirrors
        // fProxyEigenTests.AssertPowerResidual but takes Av directly so the BSM matvec is the thing
        // under test.
        void AssertResidual(in fProxyN Av, in fProxyN v, fProxy lambda, fProxy limitBase, int n)
        {
            fProxy maxRes = (fProxy)0;
            for (int i = 0; i < n; i++)
            {
                fProxy ri = math.abs(Av[i] - lambda * v[i]);
                if (ri > maxRes) maxRes = ri;
            }
            fProxy scale = math.abs(lambda);
            if (scale < (fProxy)1) scale = (fProxy)1;
            fProxy limit = limitBase * scale;
            if (!(maxRes <= limit) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
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

            fProxy tol = (fProxy)10 * Consts.fProxyZeroThreshold;

            // Dense reference (v starts at zero -> deterministic seeding).
            var vDense = arena.fProxyVec(dim);
            var wDense = arena.fProxyVec(dim);
            bool okDense = Eigen.powerIteration(in A, ref vDense, ref wDense, out fProxy lamDense, tol, 2000);
            AssertTrue(okDense, (fProxy)1);

            // Sparse (BSM) path from an identically zero-seeded v.
            var vSparse = arena.fProxyVec(dim);
            var wSparse = arena.fProxyVec(dim);
            bool okSparse = Eigen.powerIteration(in bsm, ref vSparse, ref wSparse, out fProxy lamSparse, tol, 2000);
            AssertTrue(okSparse, (fProxy)2);

            // Eigenvalues agree (magnitude up to ~dim+order-of-M; scale the loose tolerance).
            fProxy scale = (fProxy)1 + math.abs(lamDense);
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
        // then cast, mirroring fProxyEigenTests.EvSymLaplacian), and the residual A*v ~= lambda*v
        // using Sparse_OP.spMV on the BSM itself.
        void LaplacianKnownSpectrum()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 16;
            var A = arena.fProxyLaplacian1D(n);
            var bsm = DenseToBSM1x1(ref arena, in A, 3 * n);

            fProxy tol = (fProxy)10 * Consts.fProxyZeroThreshold;

            var v = arena.fProxyVec(n);   // zero -> deterministic seeding
            var w = arena.fProxyVec(n);
            bool ok = Eigen.powerIteration(in bsm, ref v, ref w, out fProxy lambda, tol, 4000);
            AssertTrue(ok, (fProxy)1);

            // Closed-form dominant eigenvalue (k = n), computed in double precision then cast.
            double lamD = 2.0 - 2.0 * math.cos(n * math.PI_DBL / (n + 1));
            fProxy scale = (fProxy)1 + math.abs((fProxy)lamD);
            AssertClose(lambda, (fProxy)lamD, (fProxy)1000 * Consts.fProxyZeroThreshold * scale);

            // Residual property on the BSM operator: A*v ~= lambda*v (A*v via spMV on the BSM).
            var Av = Sparse_OP.spMV(in bsm, in v);
            AssertResidual(in Av, in v, lambda, (fProxy)100 * Consts.fProxyZeroThreshold, n);

            arena.Dispose();
        }
    }

    // ---- correctness entry points (Burst job + Fail-array surfacing) ----------------------

    void RunCase(SparseEigenTestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new SparseEigenTestJob { Type = type, Fail = fail }.Run();
            // A failed in-job Assert aborts the Burst job WITHOUT throwing to the caller; surface
            // the recorded diagnostics here (same convention as fProxyEigenTests).
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/code {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (fProxy)0)
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
    // entry point (matching fProxyEigenTests' Power* throw tests, but via fProxyBSM).

    // A square 4x4 (two 2x2 diagonal blocks) BSM -- both diagonal blocks present, well-formed.
    static fProxyBSM BuildSquareBSM(ref Arena arena)
    {
        const int BR = 2, BC = 2;
        var builder = arena.fProxyBSMBuilder(2, 2, BR, BC, 2);
        builder.AddBlock(0, 0, arena.fProxyRandomMat(BR, BC, -1f, 1f, 71001));
        builder.AddBlock(1, 1, arena.fProxyRandomMat(BR, BC, -1f, 1f, 71002));
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
            var builder = arena.fProxyBSMBuilder(2, 3, BR, BC, 1);
            builder.AddBlock(0, 0, arena.fProxyRandomMat(BR, BC, -1f, 1f, 71101));
            var A = builder.ToBSM(ref arena);

            var v = arena.fProxyVec(A.M_Rows);
            var w = arena.fProxyVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref w, out fProxy lambda, Consts.fProxyZeroThreshold, 1000));
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
            var v = arena.fProxyVec(A.M_Rows - 1); // wrong length
            var w = arena.fProxyVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref w, out fProxy lambda, Consts.fProxyZeroThreshold, 1000));
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
            var v = arena.fProxyVec(A.M_Rows);
            var wAlias = v; // w aliases v (struct copy shares Data.Ptr) -> guard must fire
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref wAlias, out fProxy lambda, Consts.fProxyZeroThreshold, 1000));
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
            var v = arena.fProxyVec(A.M_Rows);
            var w = arena.fProxyVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref w, out fProxy lambda, Consts.fProxyZeroThreshold, 0));
        }
        finally { arena.Dispose(); }
    }
}
