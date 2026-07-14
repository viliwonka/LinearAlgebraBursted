using LinearAlgebra;
using LinearAlgebra.Gallery;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

public class fProxyCHOTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct CHOTestJob : IJob
    {
        public enum TestType
        {
            RoundTrip,
            SolveOneStep,
            SolveTwoStep,
            KnownSmall,
            Identity,
            NotSPD,
            NotSPDStatus,
            CrossCheckLU,
            Tiny,
            Aliasing,
            GalleryMinIJ,
            GalleryGCD,
            GalleryFiedlerRejects,
            // Blocked (level-3 / TRSM+SYRK trailing-update) path, engaged when
            // n >= CHOL_BLOCK_MIN_N (a measured per-dtype crossover — see Consts.fProxyCholBlockMinN;
            // 512 for both dtypes as of the post-axpy4 retune). The rest of the suite above tops out
            // at dim=12, so these are the ONLY tests that reach the blocked core. 545 and 600 are NOT
            // multiples of CHOL_BLOCK=32, so their last panel is narrower than a full block
            // (545 % 32 = 1, 600 % 32 = 24) and their TRSM row counts land on the wide kernel's
            // scalar-remainder seam; 512 and 576 are aligned.
            BlockedRoundTrip512,
            BlockedRoundTrip545,
            BlockedRoundTrip576,
            BlockedRoundTrip600,
            BlockedNotSPD,
            BlockedAliasing,
            // CHO.solveInPlace's exit factor is a valid decompSolve input, bit-identical to a fresh
            // decomp + decompSolve on the same original A.
            SolveInPlaceExitIsUsableFactor,
            // Driver short-circuit purity: non-PD input leaves b_to_x untouched.
            SolveInPlaceShortCircuitPurity,
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
                case TestType.NotSPDStatus:
                    NotSPDStatus();
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
                case TestType.BlockedRoundTrip512:
                    BlockedRoundTripAt(512, 2560001);
                    break;
                case TestType.BlockedRoundTrip545:
                    BlockedRoundTripAt(545, 3000001);
                    break;
                case TestType.BlockedRoundTrip576:
                    BlockedRoundTripAt(576, 3200001);
                    break;
                case TestType.BlockedRoundTrip600:
                    BlockedRoundTripAt(600, 4000001);
                    break;
                case TestType.BlockedNotSPD:
                    BlockedNotSPD();
                    break;
                case TestType.BlockedAliasing:
                    BlockedAliasing();
                    break;
                case TestType.SolveInPlaceExitIsUsableFactor:
                    SolveInPlaceExitIsUsableFactor();
                    break;
                case TestType.SolveInPlaceShortCircuitPurity:
                    SolveInPlaceShortCircuitPurity();
                    break;
            }
        }

        // Build an SPD matrix reliably as A = MᵀM + n·I.
        // MᵀM is symmetric positive-semidefinite; adding n·I (n = dim) makes it
        // strictly positive-definite and diagonally dominant, so Cholesky must succeed.
        static fProxyMxN BuildSPD(ref Arena arena, int dim, uint seed)
        {
            var M = arena.fProxyRandomMat(dim, dim, -1f, 1f, seed);

            var A = Blas.dot(M, M, true);

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

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);

            // L must be lower triangular (strict upper zeroed).
            Assert.IsTrue(Analysis.isLowerTriangular(L, Tol()));

            // Reconstruct A = L·Lᵀ and compare. Build Lᵀ explicitly then L·Lᵀ.
            var Lt = Blas.trans(L);
            var recon = Blas.dot(L, Lt, false);

            Assert.IsTrue(Analysis.isZero(A - recon, Tol()));

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

            // factor + solve, as the explicit two-call composition; b is overwritten with x.
            DirectSolveInfo info = CHO.decomp(in A, ref L);
            if (info.Solved) info = CHO.decompSolve(ref L, ref b);
            bool ok = info.Solved;
            Assert.IsTrue(ok);

            // Verify A·x ≈ bOrig
            var Ax = Blas.dot(A, b);
            Assert.IsTrue(Analysis.isZero(bOrig - Ax, Tol()));

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

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);

            // Solve using the pre-computed factor.
            CHO.decompSolve(ref L, ref b);

            var Ax = Blas.dot(A, b);
            Assert.IsTrue(Analysis.isZero(bOrig - Ax, Tol()));

            arena.Dispose();
        }

        // CHO.solveInPlace's exit (A_to_L) must be a valid decompSolve
        // input -- solving a SECOND right-hand side through it must be bit-identical to a completely
        // independent decomp + decompSolve on the same original matrix.
        void SolveInPlaceExitIsUsableFactor()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 11;
            var A = BuildSPD(ref arena, dim, 909091);

            var b1 = arena.fProxyRandomVec(dim, -1f, 1f, 1212);
            var b2 = arena.fProxyRandomVec(dim, -1f, 1f, 3434);

            // path under test: solveInPlace (first RHS), then decompSolve (second RHS) off its exit.
            var Afused = A.Copy();
            var x1 = b1.Copy();
            var info = CHO.solveInPlace(ref Afused, ref x1);
            Assert.IsTrue(info.Solved);

            var x2 = b2.Copy();
            CHO.decompSolve(ref Afused, ref x2);

            // oracle: fresh decomp + decompSolve on an independent copy, same second RHS.
            var L = arena.fProxyMat(dim, dim);
            var infoRef = CHO.decomp(in A, ref L);
            Assert.IsTrue(infoRef.Solved);

            var x2ref = b2.Copy();
            CHO.decompSolve(ref L, ref x2ref);

            for (int i = 0; i < dim; i++)
                Assert.IsTrue(x2[i] == x2ref[i]);

            arena.Dispose();
        }

        // Driver short-circuit purity: CHO.solveInPlace on a NON-PD matrix must
        // (a) return the NotPositiveDefinite failure status and (b) leave b_to_x BIT-IDENTICAL to its
        // pre-call snapshot. Guards the `if (!info.Solved) return info;` early return in the fused
        // POSV driver: without it, decompSolve would run on a garbage/partial factor and corrupt b.
        void SolveInPlaceShortCircuitPurity()
        {
            var arena = new Arena(Allocator.Persistent);

            // Indefinite [[1,2],[2,1]] (eigenvalues 3, -1) -- reused from NotSPD/NotSPDStatus.
            var A = arena.fProxyMat(2, 2);
            A[0, 0] = 1f; A[0, 1] = 2f;
            A[1, 0] = 2f; A[1, 1] = 1f;

            var b = arena.fProxyRandomVec(2, -1f, 1f, 13579);
            var bSnapshot = b.Copy(); // capture BEFORE the call

            DirectSolveInfo info = CHO.solveInPlace(ref A, ref b);

            Assert.IsTrue(info.status == DirectSolveStatus.NotPositiveDefinite);
            Assert.IsFalse(info.Solved);

            // b_to_x untouched: bit-identical (==, not within-tolerance) to its snapshot.
            for (int i = 0; i < 2; i++)
                Assert.IsTrue(b[i] == bSnapshot[i]);

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

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);

            fProxy tol = Tol();
            Assert.IsTrue(math.abs(L[0, 0] - 2f) < tol);
            Assert.IsTrue(math.abs(L[0, 1] - 0f) < tol);
            Assert.IsTrue(math.abs(L[1, 0] - 1f) < tol);
            Assert.IsTrue(math.abs(L[1, 1] - math.sqrt((fProxy)2f)) < tol);

            var Lt = Blas.trans(L);
            var recon = Blas.dot(L, Lt, false);
            Assert.IsTrue(Analysis.isZero(A - recon, tol));

            arena.Dispose();
        }

        void Identity()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var A = arena.fProxyIdentityMat(dim);
            var L = arena.fProxyMat(dim, dim);

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);

            // chol(I) = I
            Assert.IsTrue(Analysis.isIdentity(L, Tol()));

            // Solving I x = b returns x = b.
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 9090);
            var bOrig = b.Copy();
            CHO.decompSolve(ref L, ref b);
            Assert.IsTrue(Analysis.isZero(bOrig - b, Tol()));

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

                bool ok = CHO.decomp(in A, ref L);
                Assert.IsFalse(ok);
                // On false, no NaN must be produced.
                Assert.IsFalse(Analysis.isAnyNan(in L));

                // factor+solve composition must also report failure.
                var b = arena.fProxyRandomVec(2, -1f, 1f, 13);
                DirectSolveInfo solveInfo = CHO.decomp(in A, ref L);
                if (solveInfo.Solved) solveInfo = CHO.decompSolve(ref L, ref b);
                bool solved = solveInfo.Solved;
                Assert.IsFalse(solved);
            }

            // Case 2: zero matrix (first pivot is 0, not > 0) -> not positive-definite.
            {
                int dim = 5;
                var A = arena.fProxyMat(dim, dim);
                var L = arena.fProxyMat(dim, dim);

                bool ok = CHO.decomp(in A, ref L);
                Assert.IsFalse(ok);
                Assert.IsFalse(Analysis.isAnyNan(in L));
            }

            // Case 3: negative diagonal -> not positive-definite.
            {
                var A = arena.fProxyMat(3, 3);
                A[0, 0] = 2f;  A[0, 1] = 0f;  A[0, 2] = 0f;
                A[1, 0] = 0f;  A[1, 1] = -3f; A[1, 2] = 0f;
                A[2, 0] = 0f;  A[2, 1] = 0f;  A[2, 2] = 1f;

                var L = arena.fProxyMat(3, 3);

                bool ok = CHO.decomp(in A, ref L);
                Assert.IsFalse(ok);
                Assert.IsFalse(Analysis.isAnyNan(in L));
            }

            arena.Dispose();
        }

        // Direct-solve-status coverage: a non-PD matrix must report
        // DirectSolveStatus.NotPositiveDefinite (not just a falsy implicit-bool) from both
        // decomp and the factor-and-solve composition.
        void NotSPDStatus()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(2, 2);
            A[0, 0] = 1f; A[0, 1] = 2f;
            A[1, 0] = 2f; A[1, 1] = 1f; // same matrix as NotSPD's Case 1 (indefinite)

            var L = arena.fProxyMat(2, 2);
            DirectSolveInfo decompInfo = CHO.decomp(in A, ref L);
            Assert.IsTrue(decompInfo.status == DirectSolveStatus.NotPositiveDefinite);
            Assert.IsFalse(decompInfo.Solved);
            Assert.IsFalse(decompInfo);

            var b = arena.fProxyRandomVec(2, -1f, 1f, 17);
            DirectSolveInfo solveInfo = CHO.decomp(in A, ref L);
            if (solveInfo.Solved) solveInfo = CHO.decompSolve(ref L, ref b);
            Assert.IsTrue(solveInfo.status == DirectSolveStatus.NotPositiveDefinite);
            Assert.IsFalse(solveInfo.Solved);

            arena.Dispose();
        }

        void CrossCheckLU()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 11;

            var A = BuildSPD(ref arena, dim, 707070);

            var b = arena.fProxyRandomVec(dim, -1f, 1f, 8181);

            var bChol = b.Copy();
            var L = arena.fProxyMat(dim, dim);
            DirectSolveInfo info = CHO.decomp(in A, ref L);
            if (info.Solved) info = CHO.decompSolve(ref L, ref bChol);
            bool ok = info.Solved;
            Assert.IsTrue(ok);

            // LU solve on the same system (in-place LU with pivot).
            var lu = A.Copy();
            var pivot = new Pivot(dim, Allocator.Temp);
            bool luOk = LinearAlgebra.LU.decompInPlace(ref lu, ref pivot);
            Assert.IsTrue(luOk);

            var bLU = b.Copy();
            LinearAlgebra.LU.decompSolve(ref lu, in pivot, ref bLU);
            pivot.Dispose();

            // The two solutions must agree.
            Assert.IsTrue(Analysis.isZero(bChol - bLU, Tol()));

            arena.Dispose();
        }

        // n == 1 degenerate path: A = [[k]] (k > 0) -> L = [[sqrt(k)]].
        void Tiny()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(1, 1);
            A[0, 0] = 9f;

            var L = arena.fProxyMat(1, 1);

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);
            Assert.IsTrue(math.abs(L[0, 0] - 3f) < Tol());

            // Solve 9·x = b -> A·x ≈ b.
            var b = arena.fProxyRandomVec(1, -1f, 1f, 77);
            var bOrig = b.Copy();
            CHO.decompSolve(ref L, ref b);
            var Ax = Blas.dot(A, b);
            Assert.IsTrue(Analysis.isZero(bOrig - Ax, Tol()));

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
            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);

            // Reconstruct L·Lᵀ and compare against the ORIGINAL A.
            var Lt = Blas.trans(L);
            var recon = Blas.dot(L, Lt, false);
            Assert.IsTrue(Analysis.isZero(Aorig - recon, Tol()));

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

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);

            Assert.IsTrue(Analysis.isLowerTriangular(L, Tol()));

            var Lt = Blas.trans(L);
            var recon = Blas.dot(L, Lt, false);
            Assert.IsTrue(Analysis.isZero(A - recon, Tol()));

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

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);

            Assert.IsTrue(Analysis.isLowerTriangular(L, Tol()));

            var Lt = Blas.trans(L);
            var recon = Blas.dot(L, Lt, false);
            Assert.IsTrue(Analysis.isZero(A - recon, Tol()));

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

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsFalse(ok);
            Assert.IsFalse(Analysis.isAnyNan(in L));

            // factor+solve composition must also report failure.
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 17);
            DirectSolveInfo solveInfo = CHO.decomp(in A, ref L);
            if (solveInfo.Solved) solveInfo = CHO.decompSolve(ref L, ref b);
            bool solved = solveInfo.Solved;
            Assert.IsFalse(solved);

            arena.Dispose();
        }

        // Round-trip at sizes that reach the BLOCKED (level-3 / TRSM+SYRK trailing-update) path
        // (n >= CHOL_BLOCK_MIN_N = 256). Same invariants as RoundTrip: L lower-triangular, A ≈ L·Lᵀ.
        void BlockedRoundTripAt(int dim, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);

            var A = BuildSPD(ref arena, dim, seed);
            var L = arena.fProxyMat(dim, dim);

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);

            Assert.IsTrue(Analysis.isLowerTriangular(L, Tol()));

            var Lt = Blas.trans(L);
            var recon = Blas.dot(L, Lt, false);
            Assert.IsTrue(Analysis.isZero(A - recon, Tol()));

            arena.Dispose();
        }

        // Non-PD rejection WITHIN the blocked path, past the first panel/TRSM/SYRK trailing update
        // (dim=300 -> panels [0,32) [32,64) ... [256,288) [288,300); index 260 sits in the ninth
        // panel). The rank-1/SYRK updates only ever SUBTRACT squares from a diagonal entry, so
        // seeding it very negative guarantees the running pivot stays non-positive however earlier
        // panels' deferred trailing updates land on it — verifying the !(d>0) check still fires at
        // the right column when the SYRK update has been deferred across whole panels.
        void BlockedNotSPD()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 545;
            var A = BuildSPD(ref arena, dim, 555555);
            A[520, 520] = -1000000f;

            var L = arena.fProxyMat(dim, dim);

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsFalse(ok);
            Assert.IsFalse(Analysis.isAnyNan(in L));

            arena.Dispose();
        }

        // In-place (L aliases A) factorization through the BLOCKED path (dim=288 = 9*CHOL_BLOCK).
        void BlockedAliasing()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 544;
            var A = BuildSPD(ref arena, dim, 909090);
            var Aorig = A.Copy();

            var L = A;
            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);

            var Lt = Blas.trans(L);
            var recon = Blas.dot(L, Lt, false);
            Assert.IsTrue(Analysis.isZero(Aorig - recon, Tol()));

            arena.Dispose();
        }
    }

    [Test]
    public void RoundTripTest()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.RoundTrip }.Run();
    }

    [Test]
    public void SolveOneStepTest()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.SolveOneStep }.Run();
    }

    [Test]
    public void SolveTwoStepTest()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.SolveTwoStep }.Run();
    }

    [Test]
    public void KnownSmallTest()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.KnownSmall }.Run();
    }

    [Test]
    public void IdentityTest()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.Identity }.Run();
    }

    [Test]
    public void NotSPDTest()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.NotSPD }.Run();
    }

    [Test]
    public void NotSPDStatusTest()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.NotSPDStatus }.Run();
    }

    [Test]
    public void CrossCheckLUTest()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.CrossCheckLU }.Run();
    }

    [Test]
    public void TinyTest()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.Tiny }.Run();
    }

    [Test]
    public void AliasingTest()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.Aliasing }.Run();
    }

    [Test]
    public void GalleryMinIJTest()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.GalleryMinIJ }.Run();
    }

    [Test]
    public void GalleryGCDTest()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.GalleryGCD }.Run();
    }

    [Test]
    public void GalleryFiedlerRejectsTest()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.GalleryFiedlerRejects }.Run();
    }

    [Test]
    public void BlockedRoundTrip512Test()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.BlockedRoundTrip512 }.Run();
    }

    [Test]
    public void BlockedRoundTrip545Test()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.BlockedRoundTrip545 }.Run();
    }

    [Test]
    public void BlockedRoundTrip576Test()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.BlockedRoundTrip576 }.Run();
    }

    [Test]
    public void BlockedRoundTrip600Test()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.BlockedRoundTrip600 }.Run();
    }

    [Test]
    public void BlockedNotSPDTest()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.BlockedNotSPD }.Run();
    }

    [Test]
    public void BlockedAliasingTest()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.BlockedAliasing }.Run();
    }

    [Test]
    public void SolveInPlaceExitIsUsableFactorTest()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.SolveInPlaceExitIsUsableFactor }.Run();
    }

    [Test]
    public void SolveInPlaceShortCircuitPurityTest()
    {
        new CHOTestJob() { Type = CHOTestJob.TestType.SolveInPlaceShortCircuitPurity }.Run();
    }
}
