using BULA;
using BULA.Gallery;
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
        static fProxyMxN BuildSPD(int dim, uint seed)
        {
            var M = GenerateOP.fProxyRandomMat(dim, dim, -1f, 1f, seed, Allocator.Temp);

            var A = Blas.dot(M, M, true);

            for (int d = 0; d < dim; d++)
                A[d, d] += dim;

            return A;
        }

        void RoundTrip()
        {
            int dim = 10;

            var A = BuildSPD(dim, 90125);
            var L = new fProxyMxN(dim, dim, Allocator.Temp);

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);

            // L must be lower triangular (strict upper zeroed).
            Assert.IsTrue(Analysis.isLowerTriangular(L, Tol()));

            // Reconstruct A = L·Lᵀ and compare. Build Lᵀ explicitly then L·Lᵀ.
            var Lt = Blas.trans(L);
            var recon = Blas.dot(L, Lt, false);

            var AMinusRecon = new fProxyMxN(in A, Allocator.Temp);
            fProxyComp.subInPlace(AMinusRecon, recon);
            Assert.IsTrue(Analysis.isZero(AMinusRecon, Tol()));
        }

        void SolveOneStep()
        {
            int dim = 12;

            var A = BuildSPD(dim, 31337);
            var L = new fProxyMxN(dim, dim, Allocator.Temp);

            var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 4242, Allocator.Temp);
            var bOrig = new fProxyN(in b, Allocator.Temp);

            // factor + solve, as the explicit two-call composition; b is overwritten with x.
            DirectSolveInfo info = CHO.decomp(in A, ref L);
            if (info.Solved) info = CHO.decompSolve(ref L, ref b);
            bool ok = info.Solved;
            Assert.IsTrue(ok);

            // Verify A·x ≈ bOrig
            var Ax = Blas.dot(A, b);
            var bOrigMinusAx = new fProxyN(in bOrig, Allocator.Temp);
            fProxyComp.subInPlace(bOrigMinusAx, Ax);
            Assert.IsTrue(Analysis.isZero(bOrigMinusAx, Tol()));
        }

        void SolveTwoStep()
        {
            int dim = 9;

            var A = BuildSPD(dim, 271828);
            var L = new fProxyMxN(dim, dim, Allocator.Temp);

            var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 5151, Allocator.Temp);
            var bOrig = new fProxyN(in b, Allocator.Temp);

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);

            // Solve using the pre-computed factor.
            CHO.decompSolve(ref L, ref b);

            var Ax = Blas.dot(A, b);
            var bOrigMinusAx = new fProxyN(in bOrig, Allocator.Temp);
            fProxyComp.subInPlace(bOrigMinusAx, Ax);
            Assert.IsTrue(Analysis.isZero(bOrigMinusAx, Tol()));
        }

        // CHO.solveInPlace's exit (A_to_L) must be a valid decompSolve
        // input -- solving a SECOND right-hand side through it must be bit-identical to a completely
        // independent decomp + decompSolve on the same original matrix.
        void SolveInPlaceExitIsUsableFactor()
        {
            int dim = 11;
            var A = BuildSPD(dim, 909091);

            var b1 = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 1212, Allocator.Temp);
            var b2 = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 3434, Allocator.Temp);

            // path under test: solveInPlace (first RHS), then decompSolve (second RHS) off its exit.
            var Afused = new fProxyMxN(in A, Allocator.Temp);
            var x1 = new fProxyN(in b1, Allocator.Temp);
            var info = CHO.solveInPlace(ref Afused, ref x1);
            Assert.IsTrue(info.Solved);

            var x2 = new fProxyN(in b2, Allocator.Temp);
            CHO.decompSolve(ref Afused, ref x2);

            // oracle: fresh decomp + decompSolve on an independent copy, same second RHS.
            var L = new fProxyMxN(dim, dim, Allocator.Temp);
            var infoRef = CHO.decomp(in A, ref L);
            Assert.IsTrue(infoRef.Solved);

            var x2ref = new fProxyN(in b2, Allocator.Temp);
            CHO.decompSolve(ref L, ref x2ref);

            for (int i = 0; i < dim; i++)
                Assert.IsTrue(x2[i] == x2ref[i]);
        }

        // Driver short-circuit purity: CHO.solveInPlace on a NON-PD matrix must
        // (a) return the NotPositiveDefinite failure status and (b) leave b_to_x BIT-IDENTICAL to its
        // pre-call snapshot. Guards the `if (!info.Solved) return info;` early return in the fused
        // POSV driver: without it, decompSolve would run on a garbage/partial factor and corrupt b.
        void SolveInPlaceShortCircuitPurity()
        {
            // Indefinite [[1,2],[2,1]] (eigenvalues 3, -1) -- reused from NotSPD/NotSPDStatus.
            var A = new fProxyMxN(2, 2, Allocator.Temp);
            A[0, 0] = 1f; A[0, 1] = 2f;
            A[1, 0] = 2f; A[1, 1] = 1f;

            var b = GenerateOP.fProxyRandomVec(2, -1f, 1f, 13579, Allocator.Temp);
            var bSnapshot = new fProxyN(in b, Allocator.Temp); // capture BEFORE the call

            DirectSolveInfo info = CHO.solveInPlace(ref A, ref b);

            Assert.IsTrue(info.status == DirectSolveStatus.NotPositiveDefinite);
            Assert.IsFalse(info.Solved);

            // b_to_x untouched: bit-identical (==, not within-tolerance) to its snapshot.
            for (int i = 0; i < 2; i++)
                Assert.IsTrue(b[i] == bSnapshot[i]);
        }

        void KnownSmall()
        {
            // A = [[4,2],[2,3]] -> L = [[2,0],[1,sqrt(2)]]
            var A = new fProxyMxN(2, 2, Allocator.Temp);
            A[0, 0] = 4f; A[0, 1] = 2f;
            A[1, 0] = 2f; A[1, 1] = 3f;

            var L = new fProxyMxN(2, 2, Allocator.Temp);

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);

            fProxy tol = Tol();
            Assert.IsTrue(math.abs(L[0, 0] - 2f) < tol);
            Assert.IsTrue(math.abs(L[0, 1] - 0f) < tol);
            Assert.IsTrue(math.abs(L[1, 0] - 1f) < tol);
            Assert.IsTrue(math.abs(L[1, 1] - math.sqrt((fProxy)2f)) < tol);

            var Lt = Blas.trans(L);
            var recon = Blas.dot(L, Lt, false);
            var AMinusRecon = new fProxyMxN(in A, Allocator.Temp);
            fProxyComp.subInPlace(AMinusRecon, recon);
            Assert.IsTrue(Analysis.isZero(AMinusRecon, tol));
        }

        void Identity()
        {
            int dim = 8;

            var A = GenerateOP.fProxyIdentityMat(dim, Allocator.Temp);
            var L = new fProxyMxN(dim, dim, Allocator.Temp);

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);

            // chol(I) = I
            Assert.IsTrue(Analysis.isIdentity(L, Tol()));

            // Solving I x = b returns x = b.
            var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 9090, Allocator.Temp);
            var bOrig = new fProxyN(in b, Allocator.Temp);
            CHO.decompSolve(ref L, ref b);
            var bOrigMinusB = new fProxyN(in bOrig, Allocator.Temp);
            fProxyComp.subInPlace(bOrigMinusB, b);
            Assert.IsTrue(Analysis.isZero(bOrigMinusB, Tol()));
        }

        void NotSPD()
        {
            // Case 1: symmetric but indefinite. [[1,2],[2,1]] has eigenvalues 3 and -1.
            {
                var A = new fProxyMxN(2, 2, Allocator.Temp);
                A[0, 0] = 1f; A[0, 1] = 2f;
                A[1, 0] = 2f; A[1, 1] = 1f;

                var L = new fProxyMxN(2, 2, Allocator.Temp);

                bool ok = CHO.decomp(in A, ref L);
                Assert.IsFalse(ok);
                // On false, no NaN must be produced.
                Assert.IsFalse(Analysis.isAnyNan(in L));

                // factor+solve composition must also report failure.
                var b = GenerateOP.fProxyRandomVec(2, -1f, 1f, 13, Allocator.Temp);
                DirectSolveInfo solveInfo = CHO.decomp(in A, ref L);
                if (solveInfo.Solved) solveInfo = CHO.decompSolve(ref L, ref b);
                bool solved = solveInfo.Solved;
                Assert.IsFalse(solved);
            }

            // Case 2: zero matrix (first pivot is 0, not > 0) -> not positive-definite.
            {
                int dim = 5;
                var A = new fProxyMxN(dim, dim, Allocator.Temp);
                var L = new fProxyMxN(dim, dim, Allocator.Temp);

                bool ok = CHO.decomp(in A, ref L);
                Assert.IsFalse(ok);
                Assert.IsFalse(Analysis.isAnyNan(in L));
            }

            // Case 3: negative diagonal -> not positive-definite.
            {
                var A = new fProxyMxN(3, 3, Allocator.Temp);
                A[0, 0] = 2f;  A[0, 1] = 0f;  A[0, 2] = 0f;
                A[1, 0] = 0f;  A[1, 1] = -3f; A[1, 2] = 0f;
                A[2, 0] = 0f;  A[2, 1] = 0f;  A[2, 2] = 1f;

                var L = new fProxyMxN(3, 3, Allocator.Temp);

                bool ok = CHO.decomp(in A, ref L);
                Assert.IsFalse(ok);
                Assert.IsFalse(Analysis.isAnyNan(in L));
            }
        }

        // Direct-solve-status coverage: a non-PD matrix must report
        // DirectSolveStatus.NotPositiveDefinite (not just a falsy implicit-bool) from both
        // decomp and the factor-and-solve composition.
        void NotSPDStatus()
        {
            var A = new fProxyMxN(2, 2, Allocator.Temp);
            A[0, 0] = 1f; A[0, 1] = 2f;
            A[1, 0] = 2f; A[1, 1] = 1f; // same matrix as NotSPD's Case 1 (indefinite)

            var L = new fProxyMxN(2, 2, Allocator.Temp);
            DirectSolveInfo decompInfo = CHO.decomp(in A, ref L);
            Assert.IsTrue(decompInfo.status == DirectSolveStatus.NotPositiveDefinite);
            Assert.IsFalse(decompInfo.Solved);
            Assert.IsFalse(decompInfo);

            var b = GenerateOP.fProxyRandomVec(2, -1f, 1f, 17, Allocator.Temp);
            DirectSolveInfo solveInfo = CHO.decomp(in A, ref L);
            if (solveInfo.Solved) solveInfo = CHO.decompSolve(ref L, ref b);
            Assert.IsTrue(solveInfo.status == DirectSolveStatus.NotPositiveDefinite);
            Assert.IsFalse(solveInfo.Solved);
        }

        void CrossCheckLU()
        {
            int dim = 11;

            var A = BuildSPD(dim, 707070);

            var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 8181, Allocator.Temp);

            var bChol = new fProxyN(in b, Allocator.Temp);
            var L = new fProxyMxN(dim, dim, Allocator.Temp);
            DirectSolveInfo info = CHO.decomp(in A, ref L);
            if (info.Solved) info = CHO.decompSolve(ref L, ref bChol);
            bool ok = info.Solved;
            Assert.IsTrue(ok);

            // LU solve on the same system (in-place LU with pivot).
            var lu = new fProxyMxN(in A, Allocator.Temp);
            var pivot = new Pivot(dim, Allocator.Temp);
            bool luOk = BULA.LU.decompInPlace(ref lu, ref pivot);
            Assert.IsTrue(luOk);

            var bLU = new fProxyN(in b, Allocator.Temp);
            BULA.LU.decompSolve(ref lu, in pivot, ref bLU);
            pivot.Dispose();

            // The two solutions must agree.
            var bCholMinusBLU = new fProxyN(in bChol, Allocator.Temp);
            fProxyComp.subInPlace(bCholMinusBLU, bLU);
            Assert.IsTrue(Analysis.isZero(bCholMinusBLU, Tol()));
        }

        // n == 1 degenerate path: A = [[k]] (k > 0) -> L = [[sqrt(k)]].
        void Tiny()
        {
            var A = new fProxyMxN(1, 1, Allocator.Temp);
            A[0, 0] = 9f;

            var L = new fProxyMxN(1, 1, Allocator.Temp);

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);
            Assert.IsTrue(math.abs(L[0, 0] - 3f) < Tol());

            // Solve 9·x = b -> A·x ≈ b.
            var b = GenerateOP.fProxyRandomVec(1, -1f, 1f, 77, Allocator.Temp);
            var bOrig = new fProxyN(in b, Allocator.Temp);
            CHO.decompSolve(ref L, ref b);
            var Ax = Blas.dot(A, b);
            var bOrigMinusAx = new fProxyN(in bOrig, Allocator.Temp);
            fProxyComp.subInPlace(bOrigMinusAx, Ax);
            Assert.IsTrue(Analysis.isZero(bOrigMinusAx, Tol()));
        }

        // L aliasing A must be safe: only A's lower triangle is read, and each
        // entry is read before it is overwritten, so factoring in place is valid.
        void Aliasing()
        {
            int dim = 7;

            var A = BuildSPD(dim, 424242);
            var Aorig = new fProxyMxN(in A, Allocator.Temp);

            // L and A are distinct handles over the SAME underlying data.
            var L = A;
            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);

            // Reconstruct L·Lᵀ and compare against the ORIGINAL A.
            var Lt = Blas.trans(L);
            var recon = Blas.dot(L, Lt, false);
            var AorigMinusRecon = new fProxyMxN(in Aorig, Allocator.Temp);
            fProxyComp.subInPlace(AorigMinusRecon, recon);
            Assert.IsTrue(Analysis.isZero(AorigMinusRecon, Tol()));
        }

        // GALLERY KNOWN-ANSWER (Gallery.Phase2 / SPD): the n×n MinIJ matrix A[i,j]=min(i,j)+1 is SPD
        // (and det=1), so Cholesky must succeed and reconstruct: L lower triangular with A = L·Lᵀ.
        void GalleryMinIJ()
        {
            int dim = 6;

            var A = fProxyGallery.fProxyMinIJ(dim, Allocator.Temp);
            var L = new fProxyMxN(dim, dim, Allocator.Temp);

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);

            Assert.IsTrue(Analysis.isLowerTriangular(L, Tol()));

            var Lt = Blas.trans(L);
            var recon = Blas.dot(L, Lt, false);
            var AMinusRecon = new fProxyMxN(in A, Allocator.Temp);
            fProxyComp.subInPlace(AMinusRecon, recon);
            Assert.IsTrue(Analysis.isZero(AMinusRecon, Tol()));
        }

        // GALLERY KNOWN-ANSWER (Gallery.Phase2): the n×n GCD matrix A[i,j]=gcd(i+1,j+1) is SPD
        // (Smith's theorem, det = ∏ φ(k) > 0), so Cholesky must succeed and reconstruct A = L·Lᵀ.
        void GalleryGCD()
        {
            int dim = 6;

            var A = fProxyGallery.fProxyGCD(dim, Allocator.Temp);
            var L = new fProxyMxN(dim, dim, Allocator.Temp);

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);

            Assert.IsTrue(Analysis.isLowerTriangular(L, Tol()));

            var Lt = Blas.trans(L);
            var recon = Blas.dot(L, Lt, false);
            var AMinusRecon = new fProxyMxN(in A, Allocator.Temp);
            fProxyComp.subInPlace(AMinusRecon, recon);
            Assert.IsTrue(Analysis.isZero(AMinusRecon, Tol()));
        }

        // GALLERY KNOWN-ANSWER (Gallery.Special): the Fiedler matrix F[i,j]=|i-j| is symmetric but
        // INDEFINITE (one positive eigenvalue, n-1 negative). Cholesky must REJECT it (return false)
        // and produce no NaN. For n=3 the leading entry is 0, so the very first pivot is non-positive.
        void GalleryFiedlerRejects()
        {
            int dim = 3;

            var A = fProxyGallery.fProxyFiedler(dim, Allocator.Temp);
            var L = new fProxyMxN(dim, dim, Allocator.Temp);

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsFalse(ok);
            Assert.IsFalse(Analysis.isAnyNan(in L));

            // factor+solve composition must also report failure.
            var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 17, Allocator.Temp);
            DirectSolveInfo solveInfo = CHO.decomp(in A, ref L);
            if (solveInfo.Solved) solveInfo = CHO.decompSolve(ref L, ref b);
            bool solved = solveInfo.Solved;
            Assert.IsFalse(solved);
        }

        // Round-trip at sizes that reach the BLOCKED (level-3 / TRSM+SYRK trailing-update) path
        // (n >= CHOL_BLOCK_MIN_N = 256). Same invariants as RoundTrip: L lower-triangular, A ≈ L·Lᵀ.
        void BlockedRoundTripAt(int dim, uint seed)
        {
            var A = BuildSPD(dim, seed);
            var L = new fProxyMxN(dim, dim, Allocator.Temp);

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);

            Assert.IsTrue(Analysis.isLowerTriangular(L, Tol()));

            var Lt = Blas.trans(L);
            var recon = Blas.dot(L, Lt, false);
            var AMinusRecon = new fProxyMxN(in A, Allocator.Temp);
            fProxyComp.subInPlace(AMinusRecon, recon);
            Assert.IsTrue(Analysis.isZero(AMinusRecon, Tol()));
        }

        // Non-PD rejection WITHIN the blocked path, past the first panel/TRSM/SYRK trailing update
        // (dim=300 -> panels [0,32) [32,64) ... [256,288) [288,300); index 260 sits in the ninth
        // panel). The rank-1/SYRK updates only ever SUBTRACT squares from a diagonal entry, so
        // seeding it very negative guarantees the running pivot stays non-positive however earlier
        // panels' deferred trailing updates land on it — verifying the !(d>0) check still fires at
        // the right column when the SYRK update has been deferred across whole panels.
        void BlockedNotSPD()
        {
            int dim = 545;
            var A = BuildSPD(dim, 555555);
            A[520, 520] = -1000000f;

            var L = new fProxyMxN(dim, dim, Allocator.Temp);

            bool ok = CHO.decomp(in A, ref L);
            Assert.IsFalse(ok);
            Assert.IsFalse(Analysis.isAnyNan(in L));
        }

        // In-place (L aliases A) factorization through the BLOCKED path (dim=288 = 9*CHOL_BLOCK).
        void BlockedAliasing()
        {
            int dim = 544;
            var A = BuildSPD(dim, 909090);
            var Aorig = new fProxyMxN(in A, Allocator.Temp);

            var L = A;
            bool ok = CHO.decomp(in A, ref L);
            Assert.IsTrue(ok);

            var Lt = Blas.trans(L);
            var recon = Blas.dot(L, Lt, false);
            var AorigMinusRecon = new fProxyMxN(in Aorig, Allocator.Temp);
            fProxyComp.subInPlace(AorigMinusRecon, recon);
            Assert.IsTrue(Analysis.isZero(AorigMinusRecon, Tol()));
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
