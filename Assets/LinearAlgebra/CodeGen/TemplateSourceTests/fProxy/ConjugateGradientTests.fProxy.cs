using LinearAlgebra;
using LinearAlgebra.Gallery;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

public class fProxyConjugateGradientTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct ConjugateGradientTestJob : IJob
    {
        public enum TestType
        {
            AddScaledInPlace,
            ScaleAddInPlace,
            SolveDefaults,
            CrossCheckCholesky,
            OverloadsAgree,
            ZeroRhs,
            NotSPD,
            SingularConsistent,
            AlreadyConverged,
            Tiny,
            GalleryLaplacian1D,
            GalleryMinIJ,
        }

        public TestType Type;

        // Float expansion needs a generous tolerance; double is far tighter.
        // The CG residual default tolerance is fProxySqrtEps and the stop test is on
        // the squared residual relative to ||b||^2, so the per-component error in
        // A*x - b is comfortably below this scaled threshold on both precisions.
        static fProxy Tol() => 1024 * Consts.fProxySqrtEps;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.AddScaledInPlace:
                    AddScaledInPlace();
                    break;
                case TestType.ScaleAddInPlace:
                    ScaleAddInPlace();
                    break;
                case TestType.SolveDefaults:
                    SolveDefaults();
                    break;
                case TestType.CrossCheckCholesky:
                    CrossCheckCholesky();
                    break;
                case TestType.OverloadsAgree:
                    OverloadsAgree();
                    break;
                case TestType.ZeroRhs:
                    ZeroRhs();
                    break;
                case TestType.NotSPD:
                    NotSPD();
                    break;
                case TestType.SingularConsistent:
                    SingularConsistent();
                    break;
                case TestType.AlreadyConverged:
                    AlreadyConverged();
                    break;
                case TestType.Tiny:
                    Tiny();
                    break;
                case TestType.GalleryLaplacian1D:
                    GalleryLaplacian1D();
                    break;
                case TestType.GalleryMinIJ:
                    GalleryMinIJ();
                    break;
            }
        }

        // Build an SPD matrix reliably as A = MᵀM + n·I.
        // MᵀM is symmetric positive-semidefinite; adding n·I (n = dim) makes it
        // strictly positive-definite and diagonally dominant, so CG must converge.
        static fProxyMxN BuildSPD(ref Arena arena, int dim, uint seed)
        {
            var M = arena.fProxyRandomMat(dim, dim, -1f, 1f, seed);

            // dot(M, M, transposeA:true) == Mᵀ·M
            var A = Blas.dot(M, M, true);

            for (int d = 0; d < dim; d++)
                A[d, d] += dim;

            return A;
        }

        // ---- Load-bearing primitives ------------------------------------------------

        // y += a * x : each element must become y0 + a*x0.
        void AddScaledInPlace()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 16;

            var y = arena.fProxyRandomVec(n, -1f, 1f, 11111);
            var x = arena.fProxyRandomVec(n, -1f, 1f, 22222);
            fProxy a = (fProxy)(-0.75f);

            // Snapshot of the original y before the in-place update.
            var y0 = y.Copy();

            y.addScaledInPlace(a, x);

            fProxy tol = Tol();
            for (int i = 0; i < n; i++)
            {
                fProxy expected = y0[i] + a * x[i];
                Assert.IsTrue(math.abs(y[i] - expected) < tol);
            }

            arena.Dispose();
        }

        // y = a * y + x : each element must become a*y0 + x0.
        void ScaleAddInPlace()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 16;

            var y = arena.fProxyRandomVec(n, -1f, 1f, 33333);
            var x = arena.fProxyRandomVec(n, -1f, 1f, 44444);
            fProxy a = (fProxy)1.5f;

            var y0 = y.Copy();

            y.scaleAddInPlace(a, x);

            fProxy tol = Tol();
            for (int i = 0; i < n; i++)
            {
                fProxy expected = a * y0[i] + x[i];
                Assert.IsTrue(math.abs(y[i] - expected) < tol);
            }

            arena.Dispose();
        }

        // ---- Solver -----------------------------------------------------------------

        // Defaults overload: SPD A, random b, x = 0 -> A·x ≈ b and converged.
        void SolveDefaults()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 12;

            var A = BuildSPD(ref arena, dim, 90125);
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 4242);

            var x = arena.fProxyVec(dim); // zero initial guess

            bool ok = Krylov.cg(in A, in b, ref x);
            Assert.IsTrue(ok);

            var Ax = Blas.dot(A, x);
            Assert.IsTrue(Analysis.isZero(b - Ax, Tol()));

            arena.Dispose();
        }

        // CG solution must match the Cholesky solution on the same SPD system.
        void CrossCheckCholesky()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 11;

            var A = BuildSPD(ref arena, dim, 707070);
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 8181);

            // CG solve
            var xCG = arena.fProxyVec(dim);
            bool ok = Krylov.cg(in A, in b, ref xCG);
            Assert.IsTrue(ok);

            // Cholesky solve on the same system (b overwritten with x), as the explicit two-call
            // composition.
            var bChol = b.Copy();
            var L = arena.fProxyMat(dim, dim);
            DirectSolveInfo cholInfo = CHO.decomp(in A, ref L);
            if (cholInfo.Solved) cholInfo = CHO.decompSolve(ref L, ref bChol);
            bool cholOk = cholInfo.Solved;
            Assert.IsTrue(cholOk);

            Assert.IsTrue(Analysis.isZero(xCG - bChol, Tol()));

            arena.Dispose();
        }

        // The caller-scratch primitive and the explicit maxIter/tol
        // overload must produce the same solution as the defaults overload.
        void OverloadsAgree()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 10;

            var A = BuildSPD(ref arena, dim, 271828);
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 5151);

            // Reference: defaults overload.
            var xDef = arena.fProxyVec(dim);
            bool okDef = Krylov.cg(in A, in b, ref xDef);
            Assert.IsTrue(okDef);

            // Explicit maxIter/tol overload.
            var xExpl = arena.fProxyVec(dim);
            bool okExpl = Krylov.cg(in A, in b, ref xExpl, dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okExpl);
            Assert.IsTrue(Analysis.isZero(xDef - xExpl, Tol()));

            // Zero-alloc primitive with caller-provided scratch r, p, Ap.
            var xPrim = arena.fProxyVec(dim);
            var r = arena.fProxyVec(dim);
            var p = arena.fProxyVec(dim);
            var Ap = arena.fProxyVec(dim);
            bool okPrim = Krylov.cg(in A, in b, ref xPrim,
                                                    ref r, ref p, ref Ap,
                                                    dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okPrim);
            Assert.IsTrue(Analysis.isZero(xDef - xPrim, Tol()));

            arena.Dispose();
        }

        // b = 0 -> returns true and x is all zeros.
        void ZeroRhs()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 7;

            var A = BuildSPD(ref arena, dim, 424242);
            var b = arena.fProxyVec(dim); // zero vector

            // Non-zero initial guess must still be driven to zero.
            var x = arena.fProxyRandomVec(dim, -1f, 1f, 9999);

            bool ok = Krylov.cg(in A, in b, ref x);
            Assert.IsTrue(ok);
            Assert.IsTrue(Analysis.isZero(in x, Tol()));

            arena.Dispose();
        }

        // Indefinite symmetric matrix [[1,2],[2,1]] (eigenvalues 3, -1) -> returns false.
        void NotSPD()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(2, 2);
            A[0, 0] = 1f; A[0, 1] = 2f;
            A[1, 0] = 2f; A[1, 1] = 1f;

            var b = arena.fProxyRandomVec(2, -1f, 1f, 13);
            var x = arena.fProxyVec(2);

            bool ok = Krylov.cg(in A, in b, ref x);
            Assert.IsFalse(ok);

            arena.Dispose();
        }

        // Singular but CONSISTENT SPD-semidefinite system: A = [[1,1],[1,1]] (rank 1, eigenvalues
        // 2 and 0), b = [2,2] is in range(A). CG must stay well-behaved on the rank-deficient input:
        // never NaN, and if it reports convergence the returned x must actually solve A x = b. (A
        // search direction entering the null space trips the p·Ap<=0 guard -> clean false; otherwise
        // the residual is annihilated and it converges to a valid solution.)
        void SingularConsistent()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(2, 2);
            A[0, 0] = 1f; A[0, 1] = 1f;
            A[1, 0] = 1f; A[1, 1] = 1f;

            var b = arena.fProxyVec(2);
            b[0] = 2f; b[1] = 2f;

            var x = arena.fProxyVec(2);
            bool ok = Krylov.cg(in A, in b, ref x);

            // never produces NaN/Inf on the rank-deficient input...
            Assert.IsFalse(Analysis.isAnyNan(in x));
            // ...and a reported convergence must be a genuine solution.
            if (ok)
            {
                var Ax = Blas.dot(A, x);
                Assert.IsTrue(Analysis.isZero(b - Ax, Tol()));
            }

            arena.Dispose();
        }

        // Initial guess already correct: feed the converged solution back -> returns
        // true immediately and x is unchanged.
        void AlreadyConverged()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 9;

            var A = BuildSPD(ref arena, dim, 31337);
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 6363);

            // First solve from zero.
            var x = arena.fProxyVec(dim);
            bool ok = Krylov.cg(in A, in b, ref x);
            Assert.IsTrue(ok);

            // Feed the solution back as the initial guess.
            var xWarm = x.Copy();
            bool ok2 = Krylov.cg(in A, in b, ref xWarm);
            Assert.IsTrue(ok2);

            // x must be unchanged (still solves the system).
            Assert.IsTrue(Analysis.isZero(x - xWarm, Tol()));

            var Ax = Blas.dot(A, xWarm);
            Assert.IsTrue(Analysis.isZero(b - Ax, Tol()));

            arena.Dispose();
        }

        // 1x1 degenerate path: A = [[k]] (k > 0) -> single-variable solve k·x = b.
        void Tiny()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(1, 1);
            A[0, 0] = 4f;

            var b = arena.fProxyRandomVec(1, -1f, 1f, 77);
            var x = arena.fProxyVec(1);

            bool ok = Krylov.cg(in A, in b, ref x);
            Assert.IsTrue(ok);

            var Ax = Blas.dot(A, x);
            Assert.IsTrue(Analysis.isZero(b - Ax, Tol()));

            arena.Dispose();
        }

        // GALLERY KNOWN-ANSWER (Gallery.SPD): the n×n 1D Laplacian (tridiagonal 2,-1) is SPD — the
        // canonical CG benchmark. CG must solve A·x = b accurately (A·x ≈ b). Iterations capped at
        // 4n to cover the conditioning (cond ≈ (2(n+1)/π)²) plus float round-off.
        void GalleryLaplacian1D()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;

            var A = arena.fProxyLaplacian1D(dim);
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 4242);

            var x = arena.fProxyVec(dim);

            bool ok = Krylov.cg(in A, in b, ref x, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);

            var Ax = Blas.dot(A, x);
            Assert.IsTrue(Analysis.isZero(b - Ax, Tol()));

            arena.Dispose();
        }

        // GALLERY KNOWN-ANSWER (Gallery.Phase2): the n×n MinIJ matrix A[i,j]=min(i,j)+1 is SPD, so
        // CG must solve A·x = b accurately. Iterations capped at 4n for conditioning + round-off.
        void GalleryMinIJ()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 10;

            var A = arena.fProxyMinIJ(dim);
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 8181);

            var x = arena.fProxyVec(dim);

            bool ok = Krylov.cg(in A, in b, ref x, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);

            var Ax = Blas.dot(A, x);
            Assert.IsTrue(Analysis.isZero(b - Ax, Tol()));

            arena.Dispose();
        }
    }

    [Test]
    public void AddScaledInPlaceTest()
    {
        new ConjugateGradientTestJob() { Type = ConjugateGradientTestJob.TestType.AddScaledInPlace }.Run();
    }

    [Test]
    public void ScaleAddInPlaceTest()
    {
        new ConjugateGradientTestJob() { Type = ConjugateGradientTestJob.TestType.ScaleAddInPlace }.Run();
    }

    [Test]
    public void SolveDefaultsTest()
    {
        new ConjugateGradientTestJob() { Type = ConjugateGradientTestJob.TestType.SolveDefaults }.Run();
    }

    [Test]
    public void CrossCheckCholeskyTest()
    {
        new ConjugateGradientTestJob() { Type = ConjugateGradientTestJob.TestType.CrossCheckCholesky }.Run();
    }

    [Test]
    public void OverloadsAgreeTest()
    {
        new ConjugateGradientTestJob() { Type = ConjugateGradientTestJob.TestType.OverloadsAgree }.Run();
    }

    [Test]
    public void ZeroRhsTest()
    {
        new ConjugateGradientTestJob() { Type = ConjugateGradientTestJob.TestType.ZeroRhs }.Run();
    }

    [Test]
    public void NotSPDTest()
    {
        new ConjugateGradientTestJob() { Type = ConjugateGradientTestJob.TestType.NotSPD }.Run();
    }

    [Test]
    public void SingularConsistentTest()
    {
        new ConjugateGradientTestJob() { Type = ConjugateGradientTestJob.TestType.SingularConsistent }.Run();
    }

    [Test]
    public void AlreadyConvergedTest()
    {
        new ConjugateGradientTestJob() { Type = ConjugateGradientTestJob.TestType.AlreadyConverged }.Run();
    }

    [Test]
    public void TinyTest()
    {
        new ConjugateGradientTestJob() { Type = ConjugateGradientTestJob.TestType.Tiny }.Run();
    }

    [Test]
    public void GalleryLaplacian1DTest()
    {
        new ConjugateGradientTestJob() { Type = ConjugateGradientTestJob.TestType.GalleryLaplacian1D }.Run();
    }

    [Test]
    public void GalleryMinIJTest()
    {
        new ConjugateGradientTestJob() { Type = ConjugateGradientTestJob.TestType.GalleryMinIJ }.Run();
    }
}
