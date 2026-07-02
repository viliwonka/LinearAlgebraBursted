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
// the new powerIteration(in floatBSM, ...) overloads forward through floatBSMOperator into the
// same generic core the dense powerIteration(in floatMxN, ...) path uses. Every sparse result is
// cross-checked against the pre-existing dense path (same recipe as floatSparseSolverTests: build
// the SAME operator in both forms and compare), plus one literature known-spectrum case.
//
// Correctness cases run inside a [BurstCompile] IJob (matches floatSparseSolverTests /
// floatEigenTests) and use floatEigenTests' Fail-NativeArray diagnostic convention: a failed
// Assert inside a Burst job aborts silently without surfacing to the runner, so every check first
// records [0]=flag, [1]=got, [2]=expected/limit, [3]=diff into Fail. Guard/exception cases run on
// the managed test thread with Assert.Throws, since NUnit's Assert.Throws cannot execute in Burst.
public class floatSparseEigenTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SparseEigenTestJob : IJob
    {
        public enum TestType
        {
            DenseVsSparseCrossCheck,
            LaplacianKnownSpectrum,
            InverseLaplacianCrossCheck,
            InverseVsEigenvaluesSymmetric,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff/code
        public NativeArray<float> Fail;

        // Two INDEPENDENTLY-converged iterative eigenpairs (dense manual-matvec vs BSM spMV) are
        // compared, so a machine-epsilon threshold is inappropriate: mirror floatSparseSolverTests'
        // choose-marker tolerance for iterative-vs-iterative cross-checks (looser on float).
        static float LooseTol() => 1e-2f;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.DenseVsSparseCrossCheck: DenseVsSparseCrossCheck(); break;
                case TestType.LaplacianKnownSpectrum: LaplacianKnownSpectrum(); break;
                case TestType.InverseLaplacianCrossCheck: InverseLaplacianCrossCheck(); break;
                case TestType.InverseVsEigenvaluesSymmetric: InverseVsEigenvaluesSymmetric(); break;
            }
        }

        // ---- helpers ---------------------------------------------------------------------

        // Same recipe as floatSparseSolverTests.BuildDenseSPD: A = M^T M + dim*I -> strictly SPD /
        // diagonally dominant, so it has a single clearly dominant (positive) eigenvalue that power
        // iteration converges to unambiguously.
        static floatMxN BuildDenseSPD(ref Arena arena, int dim, uint seed)
        {
            var M = arena.floatRandomMat(dim, dim, -1f, 1f, seed);
            var A = Linear_OP.dot(M, M, true);
            for (int d = 0; d < dim; d++)
                A[d, d] += dim;
            return A;
        }

        // 1x1-block BSM built from a dense matrix's nonzero entries via AddValue (identical to
        // floatSparseSolverTests.DenseToBSM1x1). nnzHint bounds the known nonzero pattern purely as
        // a perf hint; growth past it is safe (the builder's triplet state lives behind a shared
        // heap pointer). Encodes the SAME numeric operator as the dense form, so spMV(bsm,.) and the
        // dense matvec agree up to floating-point reassociation only.
        static floatBSM DenseToBSM1x1(ref Arena arena, in floatMxN A, int nnzHint)
        {
            var builder = arena.floatBSMBuilder(A.M_Rows, A.N_Cols, 1, 1, math.max(nnzHint, 1));
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    if (A[r, c] != (float)0)
                        builder.AddValue(r, c, A[r, c]);
            return builder.ToBSM(ref arena);
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff
        void AssertClose(float a, float b, float precision)
        {
            float diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        // Boolean-return guard: record a distinguishing `code` in [1] so a silent Burst abort is
        // still diagnosable ([2]/[3] unused). Used for the powerIteration convergence flags.
        void AssertTrue(bool cond, float code)
        {
            if (!cond && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = code;
                Fail[2] = (float)0;
                Fail[3] = (float)0;
            }
            Assert.IsTrue(cond);
        }

        // Two unit eigenvectors are equal up to an overall sign: align the sign on a's
        // largest-magnitude component (robust to a near-zero pivot), then compare elementwise.
        void AssertVecEqUpToSign(in floatN a, in floatN b, int n, float absTol)
        {
            int piv = 0;
            float best = (float)0;
            for (int i = 0; i < n; i++)
            {
                float m = math.abs(a[i]);
                if (m > best) { best = m; piv = i; }
            }
            float sign = (a[piv] * b[piv] >= (float)0) ? (float)1 : (float)(-1);

            float maxErr = (float)0;
            for (int i = 0; i < n; i++)
            {
                float e = math.abs(a[i] - sign * b[i]);
                if (e > maxErr) maxErr = e;
            }
            if (!(maxErr <= absTol) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = maxErr;
                Fail[2] = absTol;
                Fail[3] = maxErr - absTol;
            }
            Assert.IsTrue(maxErr <= absTol);
        }

        // Residual property ||Av - lambda*v||_inf <= limit, where Av is supplied precomputed (here
        // from Sparse_OP.spMV on the BSM) and limit scales with max(1,|lambda|). Mirrors
        // floatEigenTests.AssertPowerResidual but takes Av directly so the BSM matvec is the thing
        // under test.
        void AssertResidual(in floatN Av, in floatN v, float lambda, float limitBase, int n)
        {
            float maxRes = (float)0;
            for (int i = 0; i < n; i++)
            {
                float ri = math.abs(Av[i] - lambda * v[i]);
                if (ri > maxRes) maxRes = ri;
            }
            float scale = math.abs(lambda);
            if (scale < (float)1) scale = (float)1;
            float limit = limitBase * scale;
            if (!(maxRes <= limit) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
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

            float tol = (float)10 * Consts.floatZeroThreshold;

            // Dense reference (v starts at zero -> deterministic seeding).
            var vDense = arena.floatVec(dim);
            var wDense = arena.floatVec(dim);
            bool okDense = Eigen.powerIteration(in A, ref vDense, ref wDense, out float lamDense, tol, 2000);
            AssertTrue(okDense, (float)1);

            // Sparse (BSM) path from an identically zero-seeded v.
            var vSparse = arena.floatVec(dim);
            var wSparse = arena.floatVec(dim);
            bool okSparse = Eigen.powerIteration(in bsm, ref vSparse, ref wSparse, out float lamSparse, tol, 2000);
            AssertTrue(okSparse, (float)2);

            // Eigenvalues agree (magnitude up to ~dim+order-of-M; scale the loose tolerance).
            float scale = (float)1 + math.abs(lamDense);
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
        // then cast, mirroring floatEigenTests.EvSymLaplacian), and the residual A*v ~= lambda*v
        // using Sparse_OP.spMV on the BSM itself.
        void LaplacianKnownSpectrum()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 16;
            var A = arena.floatLaplacian1D(n);
            var bsm = DenseToBSM1x1(ref arena, in A, 3 * n);

            float tol = (float)10 * Consts.floatZeroThreshold;

            var v = arena.floatVec(n);   // zero -> deterministic seeding
            var w = arena.floatVec(n);
            bool ok = Eigen.powerIteration(in bsm, ref v, ref w, out float lambda, tol, 4000);
            AssertTrue(ok, (float)1);

            // Closed-form dominant eigenvalue (k = n), computed in double precision then cast.
            double lamD = 2.0 - 2.0 * math.cos(n * math.PI_DBL / (n + 1));
            float scale = (float)1 + math.abs((float)lamD);
            AssertClose(lambda, (float)lamD, (float)1000 * Consts.floatZeroThreshold * scale);

            // Residual property on the BSM operator: A*v ~= lambda*v (A*v via spMV on the BSM).
            var Av = Sparse_OP.spMV(in bsm, in v);
            AssertResidual(in Av, in v, lambda, (float)100 * Consts.floatZeroThreshold, n);

            arena.Dispose();
        }

        // ---- Milestone C2: Eigen.inversePowerIteration<TOp> (smallest eigenpair, generic over
        // IfloatLinearOperator, inner solve via Solvers.cg<TOp>) -----------------------------
        //
        // (a)+(b) literature known-spectrum AND dense-vs-BSM cross-check on the 1D Laplacian.
        //
        // The 1D Laplacian's SMALLEST eigenvalues are well-separated (lambda_2/lambda_1 ~= 4 for
        // small k, since lambda_k ~= (k*pi/(n+1))^2 for small k/n), so inverse iteration converges
        // quickly and reliably. This is deliberately NOT built from BuildDenseSPD (M^T M + dim*I):
        // that construction is great for the DOMINANT-eigenvalue powerIteration tests above (the
        // largest eigenvalues of a square Wishart-like M^T M are well separated) but is a poor
        // fixture for inverse iteration -- a square Wishart matrix's smallest eigenvalues cluster
        // near zero, so the ratio driving inverse iteration's convergence rate is close to 1 and
        // convergence can be arbitrarily slow. The Laplacian avoids that pitfall entirely.
        //
        // Runs inversePowerIteration on BOTH the dense matrix and an equivalent 1x1-block BSM
        // (same recipe as LaplacianKnownSpectrum/DenseVsSparseCrossCheck above): asserts both
        // converge, both match the closed-form smallest eigenvalue
        // lambda_1 = 2 - 2*cos(pi/(n+1)), the two eigenvector estimates agree up to an overall
        // sign, and the BSM path's own residual A*v ~= lambda*v holds via Sparse_OP.spMV.
        //
        // Tolerances here use LooseTol() (NOT the tight "1000*zeroThreshold"/"100*zeroThreshold"
        // constants LaplacianKnownSpectrum uses for pure-matvec powerIteration): inverse iteration
        // is mediated by an INEXACT inner CG solve (bounded by cgTol, not machine epsilon), so its
        // eigenpair floor is many orders coarser than a solver that only ever does matvecs.
        void InverseLaplacianCrossCheck()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 12;
            var Adense = arena.floatLaplacian1D(n);
            var bsm = DenseToBSM1x1(ref arena, in Adense, 3 * n);

            // tol is a multiple of cgTol (not the much tighter Consts.floatZeroThreshold): the
            // outer convergence checks compare consecutive eigenpair estimates, each from its own
            // fresh CG solve accurate only to ~cgTol, so an outer tolerance tighter than that noise
            // floor could spin to maxIter without ever detecting convergence (see
            // Eigen.inversePowerIteration's no-scratch convenience overload doc comment).
            float cgTol = Consts.floatSqrtEps;
            float tol = (float)10 * cgTol;

            // Closed-form smallest eigenvalue (k = 1), computed in double precision then cast.
            double lamD = 2.0 - 2.0 * math.cos(1.0 * math.PI_DBL / (n + 1));
            float scale = (float)1 + math.abs((float)lamD);

            // Dense inverse power iteration (v starts at zero -> deterministic seeding).
            var vDense = arena.floatVec(n);
            bool okDense = Eigen.inversePowerIteration(in Adense, ref vDense, out float lamDense, tol, 200, n, cgTol);
            AssertTrue(okDense, (float)1);
            AssertClose(lamDense, (float)lamD, LooseTol() * scale);

            // Sparse (BSM) inverse power iteration, from an identically zero-seeded v.
            var vSparse = arena.floatVec(n);
            bool okSparse = Eigen.inversePowerIteration(in bsm, ref vSparse, out float lamSparse, tol, 200, n, cgTol);
            AssertTrue(okSparse, (float)2);
            AssertClose(lamSparse, (float)lamD, LooseTol() * scale);

            // Dense-vs-BSM agreement: two INDEPENDENTLY-converged eigenvectors, up to overall sign.
            AssertVecEqUpToSign(in vDense, in vSparse, n, LooseTol());

            // Residual property on the BSM operator: A*v ~= lambda*v (A*v via spMV on the BSM).
            var Av = Sparse_OP.spMV(in bsm, in vSparse);
            AssertResidual(in Av, in vSparse, lamSparse, LooseTol(), n);

            arena.Dispose();
        }

        // ---- (c) cross-check inversePowerIteration's lambda_min against the dense full-spectrum
        // eigenvaluesSymmetric (Householder tridiagonalization + QL) on the SAME operator.
        // eigenvaluesSymmetric DESTROYS its input matrix, so it runs on an independently-built
        // copy of the Laplacian -- floatLaplacian1D is a pure generator, so calling it twice
        // yields two separate floatMxN instances encoding the identical numeric operator.
        void InverseVsEigenvaluesSymmetric()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 10;
            var A = arena.floatLaplacian1D(n);
            var ARef = arena.floatLaplacian1D(n);   // independent copy; destroyed below

            // tol is a multiple of cgTol (not the much tighter Consts.floatZeroThreshold): the
            // outer convergence checks compare consecutive eigenpair estimates, each from its own
            // fresh CG solve accurate only to ~cgTol, so an outer tolerance tighter than that noise
            // floor could spin to maxIter without ever detecting convergence (see
            // Eigen.inversePowerIteration's no-scratch convenience overload doc comment).
            float cgTol = Consts.floatSqrtEps;
            float tol = (float)10 * cgTol;

            var v = arena.floatVec(n);   // zero -> deterministic seeding
            bool ok = Eigen.inversePowerIteration(in A, ref v, out float lambda, tol, 200, n, cgTol);
            AssertTrue(ok, (float)1);

            var eigenvalues = arena.floatVec(n);
            bool okEig = Eigen.eigenvaluesSymmetric(ref ARef, ref eigenvalues);
            AssertTrue(okEig, (float)2);

            // eigenvaluesSymmetric sorts DESCENDING -> the smallest eigenvalue is the last entry.
            float smallestRef = eigenvalues[n - 1];

            float scale = (float)1 + math.abs(smallestRef);
            AssertClose(lambda, smallestRef, LooseTol() * scale);

            arena.Dispose();
        }
    }

    // ---- correctness entry points (Burst job + Fail-array surfacing) ----------------------

    void RunCase(SparseEigenTestJob.TestType type)
    {
        var fail = new NativeArray<float>(4, Allocator.TempJob);
        try
        {
            new SparseEigenTestJob { Type = type, Fail = fail }.Run();
            // A failed in-job Assert aborts the Burst job WITHOUT throwing to the caller; surface
            // the recorded diagnostics here (same convention as floatEigenTests).
            if (fail[0] != (float)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/code {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (float)0)
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

    [Test]
    public void InverseLaplacianCrossCheckTest()
        => RunCase(SparseEigenTestJob.TestType.InverseLaplacianCrossCheck);

    [Test]
    public void InverseVsEigenvaluesSymmetricTest()
        => RunCase(SparseEigenTestJob.TestType.InverseVsEigenvaluesSymmetric);

    // ---- guard / exception cases (managed thread; Assert.Throws can't run inside Burst) ----
    //
    // The BSM overloads forward into the same generic powerIteration<TOp> core, whose argument
    // guards throw ArgumentException on: A.Rows != A.Cols, v.N != A.Rows, w.N != A.Rows, v/w
    // aliasing, and maxIter < 1. Not exhaustive -- these just prove each guard fires on the BSM
    // entry point (matching floatEigenTests' Power* throw tests, but via floatBSM).

    // A square 4x4 (two 2x2 diagonal blocks) BSM -- both diagonal blocks present, well-formed.
    static floatBSM BuildSquareBSM(ref Arena arena)
    {
        const int BR = 2, BC = 2;
        var builder = arena.floatBSMBuilder(2, 2, BR, BC, 2);
        builder.AddBlock(0, 0, arena.floatRandomMat(BR, BC, -1f, 1f, 71001));
        builder.AddBlock(1, 1, arena.floatRandomMat(BR, BC, -1f, 1f, 71002));
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
            var builder = arena.floatBSMBuilder(2, 3, BR, BC, 1);
            builder.AddBlock(0, 0, arena.floatRandomMat(BR, BC, -1f, 1f, 71101));
            var A = builder.ToBSM(ref arena);

            var v = arena.floatVec(A.M_Rows);
            var w = arena.floatVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref w, out float lambda, Consts.floatZeroThreshold, 1000));
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
            var v = arena.floatVec(A.M_Rows - 1); // wrong length
            var w = arena.floatVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref w, out float lambda, Consts.floatZeroThreshold, 1000));
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
            var v = arena.floatVec(A.M_Rows);
            var wAlias = v; // w aliases v (struct copy shares Data.Ptr) -> guard must fire
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref wAlias, out float lambda, Consts.floatZeroThreshold, 1000));
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
            var v = arena.floatVec(A.M_Rows);
            var w = arena.floatVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.powerIteration(in A, ref v, ref w, out float lambda, Consts.floatZeroThreshold, 0));
        }
        finally { arena.Dispose(); }
    }

    // ---- inversePowerIteration guard / exception cases (managed thread) -------------------
    //
    // The BSM overloads forward into the same generic inversePowerIteration<TOp> core, whose
    // argument guards throw ArgumentException on: A.Rows != A.Cols, v/y/r/p/Ap length mismatch,
    // v/y/r/p/Ap aliasing, and maxIter < 1. Not exhaustive -- these just prove each guard fires
    // on the BSM entry point (mirrors the Power_*_Throws tests above).

    [Test]
    public void InversePower_NonSquareBSM_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            // 2x3 block grid of 2x2 blocks -> 4x6 (Rows != Cols). One block suffices; the
            // Rows != Cols guard fires before v is examined.
            const int BR = 2, BC = 2;
            var builder = arena.floatBSMBuilder(2, 3, BR, BC, 1);
            builder.AddBlock(0, 0, arena.floatRandomMat(BR, BC, -1f, 1f, 71201));
            var A = builder.ToBSM(ref arena);

            var v = arena.floatVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.inversePowerIteration(in A, ref v, out float lambda));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void InversePower_WrongVLength_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareBSM(ref arena);      // 4x4
            var v = arena.floatVec(A.M_Rows - 1);  // wrong length
            Assert.Throws<ArgumentException>(() =>
                Eigen.inversePowerIteration(in A, ref v, out float lambda));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void InversePower_AliasingVAndY_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareBSM(ref arena);   // 4x4
            var v  = arena.floatVec(A.M_Rows);
            var r  = arena.floatVec(A.M_Rows);
            var p  = arena.floatVec(A.M_Rows);
            var Ap = arena.floatVec(A.M_Rows);
            var yAlias = v; // y aliases v (struct copy shares Data.Ptr) -> guard must fire
            Assert.Throws<ArgumentException>(() =>
                Eigen.inversePowerIteration(in A, ref v, ref yAlias, ref r, ref p, ref Ap, out float lambda,
                    Consts.floatZeroThreshold, 1000, A.M_Rows, Consts.floatSqrtEps));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void InversePower_BadMaxIter_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareBSM(ref arena);   // 4x4
            var v = arena.floatVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() =>
                Eigen.inversePowerIteration(in A, ref v, out float lambda,
                    Consts.floatZeroThreshold, 0, A.M_Rows, Consts.floatSqrtEps));
        }
        finally { arena.Dispose(); }
    }
}
