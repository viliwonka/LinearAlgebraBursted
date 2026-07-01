using LinearAlgebra;
using LinearAlgebra.Gallery;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

public class fProxyCholeskyTests
{
    [BurstCompile]
    public struct CholeskyTestJob : IJob
    {
        public enum TestType
        {
            RoundTrip,
            SolveOneStep,
            SolveTwoStep,
            KnownSmall,
            Identity,
            NotSPD,
            CrossCheckLU,
            Tiny,
            Aliasing,
            GalleryMinIJ,
            GalleryGCD,
            GalleryFiedlerRejects,
        }

        public TestType Type;

        // Float expansion needs a generous tolerance; double is far tighter.
        // fProxyZeroThreshold is per-precision (1e-6 float, 1e-14 double).
        static fProxy Tol() => 256 * Consts.fProxySqrtEps;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.RoundTrip:
                    RoundTrip();
                    break;
                case TestType.SolveOneStep:
                    SolveOneStep();
                    break;
                case TestType.SolveTwoStep:
                    SolveTwoStep();
                    break;
                case TestType.KnownSmall:
                    KnownSmall();
                    break;
                case TestType.Identity:
                    Identity();
                    break;
                case TestType.NotSPD:
                    NotSPD();
                    break;
                case TestType.CrossCheckLU:
                    CrossCheckLU();
                    break;
                case TestType.Tiny:
                    Tiny();
                    break;
                case TestType.Aliasing:
                    Aliasing();
                    break;
                case TestType.GalleryMinIJ:
                    GalleryMinIJ();
                    break;
                case TestType.GalleryGCD:
                    GalleryGCD();
                    break;
                case TestType.GalleryFiedlerRejects:
                    GalleryFiedlerRejects();
                    break;
            }
        }

        // Build an SPD matrix reliably as A = MᵀM + n·I.
        // MᵀM is symmetric positive-semidefinite; adding n·I (n = dim) makes it
        // strictly positive-definite and diagonally dominant, so Cholesky must succeed.
        static fProxyMxN BuildSPD(ref Arena arena, int dim, uint seed)
        {
            var M = arena.fProxyRandomMat(dim, dim, -1f, 1f, seed);

            // dot(M, M, transposeA:true) == Mᵀ·M
            var A = fProxy_OP.dot(M, M, true);

            for (int d = 0; d < dim; d++)
                A[d, d] += dim;

            return A;
        }

        void RoundTrip()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 10;

            var A = BuildSPD(ref arena, dim, 90125);
            var L = arena.fProxyMat(dim, dim);

            bool ok = Cholesky.choleskyDecomposition(in A, ref L);
            Assert.IsTrue(ok);

            // L must be lower triangular (strict upper zeroed).
            Assert.IsTrue(Analysis_OP.isLowerTriangular(L, Tol()));

            // Reconstruct A = L·Lᵀ and compare. Build Lᵀ explicitly then L·Lᵀ.
            var Lt = fProxy_OP.trans(L);
            var recon = fProxy_OP.dot(L, Lt, false);

            Assert.IsTrue(Analysis_OP.isZero(A - recon, Tol()));

            arena.Dispose();
        }

        void SolveOneStep()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 12;

            var A = BuildSPD(ref arena, dim, 31337);
            var L = arena.fProxyMat(dim, dim);

            var b = arena.fProxyRandomVec(dim, -1f, 1f, 4242);
            var bOrig = b.Copy();

            // factor + solve in one call; b is overwritten with x.
            bool ok = Cholesky.choleskySolve(in A, ref L, ref b);
            Assert.IsTrue(ok);

            // Verify A·x ≈ bOrig
            var Ax = fProxy_OP.dot(A, b);
            Assert.IsTrue(Analysis_OP.isZero(bOrig - Ax, Tol()));

            arena.Dispose();
        }

        void SolveTwoStep()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 9;

            var A = BuildSPD(ref arena, dim, 271828);
            var L = arena.fProxyMat(dim, dim);

            var b = arena.fProxyRandomVec(dim, -1f, 1f, 5151);
            var bOrig = b.Copy();

            bool ok = Cholesky.choleskyDecomposition(in A, ref L);
            Assert.IsTrue(ok);

            // Solve using the pre-computed factor.
            Cholesky.choleskySolve(ref L, ref b);

            var Ax = fProxy_OP.dot(A, b);
            Assert.IsTrue(Analysis_OP.isZero(bOrig - Ax, Tol()));

            arena.Dispose();
        }

        void KnownSmall()
        {
            var arena = new Arena(Allocator.Persistent);

            // A = [[4,2],[2,3]] -> L = [[2,0],[1,sqrt(2)]]
            var A = arena.fProxyMat(2, 2);
            A[0, 0] = 4f; A[0, 1] = 2f;
            A[1, 0] = 2f; A[1, 1] = 3f;

            var L = arena.fProxyMat(2, 2);

            bool ok = Cholesky.choleskyDecomposition(in A, ref L);
            Assert.IsTrue(ok);

            fProxy tol = Tol();
            Assert.IsTrue(math.abs(L[0, 0] - 2f) < tol);
            Assert.IsTrue(math.abs(L[0, 1] - 0f) < tol);
            Assert.IsTrue(math.abs(L[1, 0] - 1f) < tol);
            Assert.IsTrue(math.abs(L[1, 1] - math.sqrt((fProxy)2f)) < tol);

            // Reconstruct as a second check.
            var Lt = fProxy_OP.trans(L);
            var recon = fProxy_OP.dot(L, Lt, false);
            Assert.IsTrue(Analysis_OP.isZero(A - recon, tol));

            arena.Dispose();
        }

        void Identity()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var A = arena.fProxyIdentityMat(dim);
            var L = arena.fProxyMat(dim, dim);

            bool ok = Cholesky.choleskyDecomposition(in A, ref L);
            Assert.IsTrue(ok);

            // chol(I) = I
            Assert.IsTrue(Analysis_OP.isIdentity(L, Tol()));

            // Solving I x = b returns x = b.
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 9090);
            var bOrig = b.Copy();
            Cholesky.choleskySolve(ref L, ref b);
            Assert.IsTrue(Analysis_OP.isZero(bOrig - b, Tol()));

            arena.Dispose();
        }

        void NotSPD()
        {
            var arena = new Arena(Allocator.Persistent);

            // Case 1: symmetric but indefinite. [[1,2],[2,1]] has eigenvalues 3 and -1.
            {
                var A = arena.fProxyMat(2, 2);
                A[0, 0] = 1f; A[0, 1] = 2f;
                A[1, 0] = 2f; A[1, 1] = 1f;

                var L = arena.fProxyMat(2, 2);

                bool ok = Cholesky.choleskyDecomposition(in A, ref L);
                Assert.IsFalse(ok);
                // On false, no NaN must be produced.
                Assert.IsFalse(Analysis_OP.isAnyNan(in L));

                // choleskySolve factor+solve overload must also report failure.
                var b = arena.fProxyRandomVec(2, -1f, 1f, 13);
                bool solved = Cholesky.choleskySolve(in A, ref L, ref b);
                Assert.IsFalse(solved);
            }

            // Case 2: zero matrix (first pivot is 0, not > 0) -> not positive-definite.
            {
                int dim = 5;
                var A = arena.fProxyMat(dim, dim);
                var L = arena.fProxyMat(dim, dim);

                bool ok = Cholesky.choleskyDecomposition(in A, ref L);
                Assert.IsFalse(ok);
                Assert.IsFalse(Analysis_OP.isAnyNan(in L));
            }

            // Case 3: negative diagonal -> not positive-definite.
            {
                var A = arena.fProxyMat(3, 3);
                A[0, 0] = 2f;  A[0, 1] = 0f;  A[0, 2] = 0f;
                A[1, 0] = 0f;  A[1, 1] = -3f; A[1, 2] = 0f;
                A[2, 0] = 0f;  A[2, 1] = 0f;  A[2, 2] = 1f;

                var L = arena.fProxyMat(3, 3);

                bool ok = Cholesky.choleskyDecomposition(in A, ref L);
                Assert.IsFalse(ok);
                Assert.IsFalse(Analysis_OP.isAnyNan(in L));
            }

            arena.Dispose();
        }

        void CrossCheckLU()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 11;

            var A = BuildSPD(ref arena, dim, 707070);

            var b = arena.fProxyRandomVec(dim, -1f, 1f, 8181);

            // Cholesky solve
            var bChol = b.Copy();
            var L = arena.fProxyMat(dim, dim);
            bool ok = Cholesky.choleskySolve(in A, ref L, ref bChol);
            Assert.IsTrue(ok);

            // LU solve on the same system (inplace LU with pivot).
            var lu = A.Copy();
            var pivot = new Pivot(dim, Allocator.Temp);
            bool luOk = LinearAlgebra.LU.luDecompositionInpl(ref lu, ref pivot);
            Assert.IsTrue(luOk);

            var bLU = b.Copy();
            LinearAlgebra.LU.luSolve(ref lu, in pivot, ref bLU);
            pivot.Dispose();

            // The two solutions must agree.
            Assert.IsTrue(Analysis_OP.isZero(bChol - bLU, Tol()));

            arena.Dispose();
        }

        // n == 1 degenerate path: A = [[k]] (k > 0) -> L = [[sqrt(k)]].
        void Tiny()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(1, 1);
            A[0, 0] = 9f;

            var L = arena.fProxyMat(1, 1);

            bool ok = Cholesky.choleskyDecomposition(in A, ref L);
            Assert.IsTrue(ok);
            Assert.IsTrue(math.abs(L[0, 0] - 3f) < Tol());

            // Solve 9·x = b -> A·x ≈ b.
            var b = arena.fProxyRandomVec(1, -1f, 1f, 77);
            var bOrig = b.Copy();
            Cholesky.choleskySolve(ref L, ref b);
            var Ax = fProxy_OP.dot(A, b);
            Assert.IsTrue(Analysis_OP.isZero(bOrig - Ax, Tol()));

            arena.Dispose();
        }

        // L aliasing A must be safe: only A's lower triangle is read, and each
        // entry is read before it is overwritten, so factoring in place is valid.
        void Aliasing()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 7;

            var A = BuildSPD(ref arena, dim, 424242);
            var Aorig = A.Copy();

            // L and A are distinct handles over the SAME underlying data.
            var L = A;
            bool ok = Cholesky.choleskyDecomposition(in A, ref L);
            Assert.IsTrue(ok);

            // Reconstruct L·Lᵀ and compare against the ORIGINAL A.
            var Lt = fProxy_OP.trans(L);
            var recon = fProxy_OP.dot(L, Lt, false);
            Assert.IsTrue(Analysis_OP.isZero(Aorig - recon, Tol()));

            arena.Dispose();
        }

        // GALLERY KNOWN-ANSWER (Gallery.Phase2 / SPD): the n×n MinIJ matrix A[i,j]=min(i,j)+1 is SPD
        // (and det=1), so Cholesky must succeed and reconstruct: L lower triangular with A = L·Lᵀ.
        void GalleryMinIJ()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 6;

            var A = arena.fProxyMinIJ(dim);
            var L = arena.fProxyMat(dim, dim);

            bool ok = Cholesky.choleskyDecomposition(in A, ref L);
            Assert.IsTrue(ok);

            Assert.IsTrue(Analysis_OP.isLowerTriangular(L, Tol()));

            var Lt = fProxy_OP.trans(L);
            var recon = fProxy_OP.dot(L, Lt, false);
            Assert.IsTrue(Analysis_OP.isZero(A - recon, Tol()));

            arena.Dispose();
        }

        // GALLERY KNOWN-ANSWER (Gallery.Phase2): the n×n GCD matrix A[i,j]=gcd(i+1,j+1) is SPD
        // (Smith's theorem, det = ∏ φ(k) > 0), so Cholesky must succeed and reconstruct A = L·Lᵀ.
        void GalleryGCD()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 6;

            var A = arena.fProxyGCD(dim);
            var L = arena.fProxyMat(dim, dim);

            bool ok = Cholesky.choleskyDecomposition(in A, ref L);
            Assert.IsTrue(ok);

            Assert.IsTrue(Analysis_OP.isLowerTriangular(L, Tol()));

            var Lt = fProxy_OP.trans(L);
            var recon = fProxy_OP.dot(L, Lt, false);
            Assert.IsTrue(Analysis_OP.isZero(A - recon, Tol()));

            arena.Dispose();
        }

        // GALLERY KNOWN-ANSWER (Gallery.Special): the Fiedler matrix F[i,j]=|i-j| is symmetric but
        // INDEFINITE (one positive eigenvalue, n-1 negative). Cholesky must REJECT it (return false)
        // and produce no NaN. For n=3 the leading entry is 0, so the very first pivot is non-positive.
        void GalleryFiedlerRejects()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 3;

            var A = arena.fProxyFiedler(dim);
            var L = arena.fProxyMat(dim, dim);

            bool ok = Cholesky.choleskyDecomposition(in A, ref L);
            Assert.IsFalse(ok);
            Assert.IsFalse(Analysis_OP.isAnyNan(in L));

            // factor+solve overload must also report failure.
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 17);
            bool solved = Cholesky.choleskySolve(in A, ref L, ref b);
            Assert.IsFalse(solved);

            arena.Dispose();
        }
    }

    [Test]
    public void RoundTripTest()
    {
        new CholeskyTestJob() { Type = CholeskyTestJob.TestType.RoundTrip }.Run();
    }

    [Test]
    public void SolveOneStepTest()
    {
        new CholeskyTestJob() { Type = CholeskyTestJob.TestType.SolveOneStep }.Run();
    }

    [Test]
    public void SolveTwoStepTest()
    {
        new CholeskyTestJob() { Type = CholeskyTestJob.TestType.SolveTwoStep }.Run();
    }

    [Test]
    public void KnownSmallTest()
    {
        new CholeskyTestJob() { Type = CholeskyTestJob.TestType.KnownSmall }.Run();
    }

    [Test]
    public void IdentityTest()
    {
        new CholeskyTestJob() { Type = CholeskyTestJob.TestType.Identity }.Run();
    }

    [Test]
    public void NotSPDTest()
    {
        new CholeskyTestJob() { Type = CholeskyTestJob.TestType.NotSPD }.Run();
    }

    [Test]
    public void CrossCheckLUTest()
    {
        new CholeskyTestJob() { Type = CholeskyTestJob.TestType.CrossCheckLU }.Run();
    }

    [Test]
    public void TinyTest()
    {
        new CholeskyTestJob() { Type = CholeskyTestJob.TestType.Tiny }.Run();
    }

    [Test]
    public void AliasingTest()
    {
        new CholeskyTestJob() { Type = CholeskyTestJob.TestType.Aliasing }.Run();
    }

    [Test]
    public void GalleryMinIJTest()
    {
        new CholeskyTestJob() { Type = CholeskyTestJob.TestType.GalleryMinIJ }.Run();
    }

    [Test]
    public void GalleryGCDTest()
    {
        new CholeskyTestJob() { Type = CholeskyTestJob.TestType.GalleryGCD }.Run();
    }

    [Test]
    public void GalleryFiedlerRejectsTest()
    {
        new CholeskyTestJob() { Type = CholeskyTestJob.TestType.GalleryFiedlerRejects }.Run();
    }
}
