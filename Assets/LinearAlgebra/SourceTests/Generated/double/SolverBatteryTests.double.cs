using System;
#pragma warning disable 618 // intentionally exercises the deprecated cyclic-Jacobi Eigen.decompInPlace (kept for reference)

using LinearAlgebra;
using LinearAlgebra.Gallery;   // opt-in: arena.doubleHilbert(n), arena.doubleKahan(n,θ), ...

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// "Solver battery" — systematic matrix×solver cross-coverage. Each library solver / decomposition is
// driven against a standard spread of gallery matrices spanning regimes (well-conditioned, ill-
// conditioned, indefinite, rank-stress, known-spectrum), every case pinned to a known/verifiable
// result: a closed-form determinant / eigenvalue / singular value, a reconstruction-error norm
// (decompose-then-rebuild), or a residual norm (solve-then-verify A x ≈ b). Tests are grouped BY
// SOLVER so each solver is exercised across multiple regimes.
//
// Verification reuses the library's own ops (Cholesky, LU, QR/QRCP, SVD, Eigen, MatrixMetrics).
// Tolerances are per-precision: they scale with Consts.doubleSqrtEps (float ≈ 3.45e-4, double ≈ 1.49e-8)
// so the SAME expression is loose for float and tight for double, matching the GalleryTests idiom.
// The tightest near-degenerate facts (Rosser spectrum, cond(Hilbert), Lauchli pinv accuracy) are
// precision-gated via IsDouble(): a tight band for double, a generous band for float. Reconstruction /
// residual errors are backward-stable (≈ eps·‖A‖, NOT cond-amplified), so they stay tight even for the
// ill-conditioned generators; the cond-amplified facts (LU/QR/CG solution accuracy on ill-conditioned A)
// use generous, sqrtEps-scaled bands. Reference values were cross-checked offline.
public class doubleSolverBatteryTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            // Cholesky
            CholeskySPDBattery,
            CholeskyRejectIndefinite,
            // LU
            LUDeterminantBattery,
            LUSolveBattery,
            // QR (Householder)
            QRDirectSolveSquare,
            QRLeastSquares,
            // QR column-pivot (QRCP)
            QRCPKahan,
            QRCPRankDeficient,
            // SVD
            SVDHadamardSigma,
            SVDParterCluster,
            SVDLauchliRankStress,
            // Symmetric eigen (Jacobi)
            EigenSymmetricReconstruct,
            EigenLaplacianSpectrum,
            EigenClementSpectrum,
            EigenFiedlerInertia,
            EigenDingDongBand,
            EigenRosserSpectrum,
            // Non-symmetric eigen (QR algorithm)
            EigenQRFrank,
            EigenQRCompanion,
            // Conjugate Gradient
            CGSPDBattery,
            // Condition number
            CondBattery,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<double> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.CholeskySPDBattery:        CholeskySPDBattery();        break;
                case TestType.CholeskyRejectIndefinite:  CholeskyRejectIndefinite();  break;
                case TestType.LUDeterminantBattery:      LUDeterminantBattery();      break;
                case TestType.LUSolveBattery:            LUSolveBattery();            break;
                case TestType.QRDirectSolveSquare:       QRDirectSolveSquare();       break;
                case TestType.QRLeastSquares:            QRLeastSquares();            break;
                case TestType.QRCPKahan:                 QRCPKahan();                 break;
                case TestType.QRCPRankDeficient:         QRCPRankDeficient();         break;
                case TestType.SVDHadamardSigma:          SVDHadamardSigma();          break;
                case TestType.SVDParterCluster:          SVDParterCluster();          break;
                case TestType.SVDLauchliRankStress:      SVDLauchliRankStress();      break;
                case TestType.EigenSymmetricReconstruct: EigenSymmetricReconstruct(); break;
                case TestType.EigenLaplacianSpectrum:    EigenLaplacianSpectrum();    break;
                case TestType.EigenClementSpectrum:      EigenClementSpectrum();      break;
                case TestType.EigenFiedlerInertia:       EigenFiedlerInertia();       break;
                case TestType.EigenDingDongBand:         EigenDingDongBand();         break;
                case TestType.EigenRosserSpectrum:       EigenRosserSpectrum();       break;
                case TestType.EigenQRFrank:              EigenQRFrank();              break;
                case TestType.EigenQRCompanion:          EigenQRCompanion();          break;
                case TestType.CGSPDBattery:              CGSPDBattery();              break;
                case TestType.CondBattery:               CondBattery();               break;
            }
        }

        // =====================================================================
        // Cholesky — succeed + reconstruct LLᵀ ≈ A on the SPD battery;
        //            reject (return false) on indefinite inputs.
        // =====================================================================

        // SPD battery: Hilbert (small n), Pascal, Lehmer, MinIJ, Pei(α>0), Moler, Laplacian1D, GCD.
        // Cholesky succeeds AND L·Lᵀ reconstructs A (backward-stable, so the error is tiny even for the
        // ill-conditioned Hilbert/Moler at these small n).
        void CholeskySPDBattery()
        {
            var arena = new Arena(Allocator.Persistent);

            var H = arena.doubleHilbert(4);            CheckCholeskyReconstruct(ref arena, in H, (double)50);
            var P = arena.doublePascal(5);             CheckCholeskyReconstruct(ref arena, in P, (double)50);
            var L = arena.doubleLehmer(5);             CheckCholeskyReconstruct(ref arena, in L, (double)50);
            var M = arena.doubleMinIJ(5);              CheckCholeskyReconstruct(ref arena, in M, (double)50);
            var Pe = arena.doublePei(5, (double)2);    CheckCholeskyReconstruct(ref arena, in Pe, (double)50);
            var Mo = arena.doubleMoler(5);             CheckCholeskyReconstruct(ref arena, in Mo, (double)100);
            var T = arena.doubleLaplacian1D(6);        CheckCholeskyReconstruct(ref arena, in T, (double)50);
            var G = arena.doubleGCD(5);                CheckCholeskyReconstruct(ref arena, in G, (double)50);

            arena.Dispose();
        }

        void CheckCholeskyReconstruct(ref Arena arena, in doubleMxN A, double factor)
        {
            int n = A.M_Rows;

            var L = arena.doubleMat(n, n);
            AssertTrue(CHO.decomp(in A, ref L));

            // rec = L · Lᵀ
            var Lt = arena.doubleMat(n, n);
            Blas.trans(in L, ref Lt);
            var rec = arena.doubleMat(n, n);
            Blas.dot(in L, in Lt, ref rec);

            double tol = (MatMaxAbs(in A) + (double)1) * Consts.doubleSqrtEps * factor;
            AssertTrue(MaxAbsDiff(in A, in rec) <= tol);
        }

        // Indefinite inputs MUST be rejected (decomp returns false): Fiedler(n≥2) and
        // Clement(n≥2) both have a zero diagonal → first pivot non-positive; Rosser is indefinite
        // (negative eigenvalues) → a later pivot goes non-positive.
        void CholeskyRejectIndefinite()
        {
            var arena = new Arena(Allocator.Persistent);

            var F = arena.doubleFiedler(3);
            var Lf = arena.doubleMat(3, 3);
            AssertTrue(!CHO.decomp(in F, ref Lf));

            var C = arena.doubleClement(3);
            var Lc = arena.doubleMat(3, 3);
            AssertTrue(!CHO.decomp(in C, ref Lc));

            var R = arena.doubleRosser();
            var Lr = arena.doubleMat(8, 8);
            AssertTrue(!CHO.decomp(in R, ref Lr));

            arena.Dispose();
        }

        // =====================================================================
        // LU — known determinants + solve A x = b on well-conditioned matrices.
        // =====================================================================

        // Determinant battery against documented closed forms:
        //   Pascal = 1, MinIJ = 1, Moler = 1, Frank = 1, Triw = 1,
        //   Vandermonde({1,2,3,4}) = ∏_{i<j}(nodes[j]−nodes[i]) = 12,
        //   GCD(5) = ∏ φ(k) = 1·1·2·2·4 = 16,
        //   Redheffer(5) = Mertens M(5) = −2.
        void LUDeterminantBattery()
        {
            var arena = new Arena(Allocator.Persistent);

            var P = arena.doublePascal(5);
            AssertClose(Determinant(in P), (double)1, (double)150 * Consts.doubleSqrtEps);

            var M = arena.doubleMinIJ(5);
            AssertClose(Determinant(in M), (double)1, (double)50 * Consts.doubleSqrtEps);

            var Mo = arena.doubleMoler(5);
            AssertClose(Determinant(in Mo), (double)1, (double)150 * Consts.doubleSqrtEps);

            var Fr = arena.doubleFrank(4);
            AssertClose(Determinant(in Fr), (double)1, (double)0.05);

            var Tw = arena.doubleTriw(5, (double)(-2));
            AssertClose(Determinant(in Tw), (double)1, (double)1E-3);

            var nodes = arena.doubleVec(4);
            nodes[0] = (double)1; nodes[1] = (double)2; nodes[2] = (double)3; nodes[3] = (double)4;
            var V = arena.doubleVandermonde(in nodes);
            AssertClose(Determinant(in V), (double)12, (double)0.2);

            var G = arena.doubleGCD(5);
            AssertClose(Determinant(in G), (double)16, (double)0.5);

            var Rh = arena.doubleRedheffer(5);
            AssertClose(Determinant(in Rh), (double)(-2), (double)0.1);

            arena.Dispose();
        }

        // LU-solve reconstructs x on well-conditioned matrices (Laplacian1D, Pascal).
        // xtol is cond-amplified: Pascal(5) (cond ≈ 8.5e3) needs a looser band than Laplacian1D.
        void LUSolveBattery()
        {
            var arena = new Arena(Allocator.Persistent);

            var T = arena.doubleLaplacian1D(8);
            CheckLUSolve(ref arena, in T, (double)200 * Consts.doubleSqrtEps);

            var P = arena.doublePascal(5);
            CheckLUSolve(ref arena, in P, (double)5E-2);

            arena.Dispose();
        }

        void CheckLUSolve(ref Arena arena, in doubleMxN A, double xtol)
        {
            int n = A.M_Rows;

            var xTrue = arena.doubleVec(n);
            for (int i = 0; i < n; i++) xTrue[i] = (double)(i + 1);

            var b = Blas.dot(A, xTrue);   // consistent RHS

            var LUm = A.Copy();
            var P = new Pivot(n, Allocator.Temp);
            AssertTrue(LU.decompInPlace(ref LUm, ref P));

            var x = b.Copy();                 // decompSolve overwrites b with x
            LU.decompSolve(ref LUm, in P, ref x);

            // residual ‖A x − b‖ with the ORIGINAL A,b (backward-stable ⇒ tiny)
            double resTol = (MatMaxAbs(in A) + (double)1) * (double)100 * Consts.doubleSqrtEps;
            AssertTrue(ResidualNorm(in A, in x, in b) <= resTol);

            for (int i = 0; i < n; i++)
                AssertClose(x[i], xTrue[i], xtol);

            P.Dispose();
        }

        // =====================================================================
        // QR (Householder) — square solve + overdetermined least squares.
        // =====================================================================

        // Well-conditioned square systems solved via QR.solveInPlace: residual small.
        // Laplacian1D(8) (cond ≈ 41) and Pei(5,2) (eigenvalues {7,2,2,2,2}, cond ≈ 3.5).
        void QRDirectSolveSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            var T = arena.doubleLaplacian1D(8);
            CheckQRSquare(ref arena, in T, (double)200 * Consts.doubleSqrtEps);

            var Pe = arena.doublePei(5, (double)2);
            CheckQRSquare(ref arena, in Pe, (double)50 * Consts.doubleSqrtEps);

            arena.Dispose();
        }

        void CheckQRSquare(ref Arena arena, in doubleMxN A, double xtol)
        {
            int n = A.M_Rows;

            var xTrue = arena.doubleVec(n);
            for (int i = 0; i < n; i++) xTrue[i] = (double)(i + 1);

            var b = Blas.dot(A, xTrue);

            var Aw = A.Copy();   // solveInPlace destroys A and b
            var bw = b.Copy();
            var x = arena.doubleVec(n);
            QR.solveInPlace(ref Aw, ref bw, ref x);

            double resTol = (MatMaxAbs(in A) + (double)1) * (double)100 * Consts.doubleSqrtEps;
            AssertTrue(ResidualNorm(in A, in x, in b) <= resTol);

            for (int i = 0; i < n; i++)
                AssertClose(x[i], xTrue[i], xtol);
        }

        // Overdetermined least squares on a tall full-column-rank gallery matrix: Läuchli(3, ε=0.5) is
        // 4×3 and well-conditioned at this ε. With a consistent RHS (b = A·xTrue) the LS solution is
        // exactly xTrue and the residual is ≈ 0.
        void QRLeastSquares()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 3;
            var A = arena.doubleLauchli(n, (double)0.5);   // 4×3

            var xTrue = arena.doubleVec(n);
            xTrue[0] = (double)1; xTrue[1] = (double)(-2); xTrue[2] = (double)3;

            var b = Blas.dot(A, xTrue);   // length 4, in range(A)

            var Aw = A.Copy();
            var bw = b.Copy();
            var x = arena.doubleVec(n);
            QR.solveInPlace(ref Aw, ref bw, ref x);   // x has length A.N_Cols = 3

            double xtol = (double)50 * Consts.doubleSqrtEps;
            for (int i = 0; i < n; i++)
                AssertClose(x[i], xTrue[i], xtol);

            // residual ‖A x − b‖ ≈ 0 for a consistent system
            double resTol = (MatMaxAbs(in A) + (double)1) * (double)100 * Consts.doubleSqrtEps;
            AssertTrue(ResidualNorm(in A, in x, in b) <= resTol);

            arena.Dispose();
        }

        // =====================================================================
        // QR column-pivot (QRCP) — rank-revealing factorization.
        // =====================================================================

        // Kahan is the classic QRCP counterexample: every column has norm 1, so it is provably invariant
        // under column pivoting (P = identity). We verify A·P ≈ Q·R reconstructs and |R[0,0]| ≥ |R[1,1]|
        // ≥ … (the rank-revealing diagonal ordering).
        void QRCPKahan()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            var A = arena.doubleKahan(n, (double)1.2);
            CheckQRCPReconstruct(ref arena, in A);

            arena.Dispose();
        }

        // Rank-deficient input: Pei(4, 0) = αI + J with α = 0 is the all-ones matrix (rank 1). QRCP must
        // reconstruct A·P ≈ Q·R, keep |R diag| non-increasing, and drive R[1,1]… to ≈ 0 (revealing rank 1).
        void QRCPRankDeficient()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;
            var A = arena.doublePei(n, (double)0);   // all-ones, rank 1
            CheckQRCPReconstruct(ref arena, in A);

            // rank 1 ⇒ trailing diagonal entries are numerically zero
            var Q = A.Copy();
            var R = arena.doubleMat(n, n);
            var P = new Pivot(n, Allocator.Temp);
            QRCP.decompInPlace(ref Q, ref R, ref P);

            double rankTol = (double)50 * Consts.doubleSqrtEps;
            for (int d = 1; d < n; d++)
                AssertTrue(math.abs(R[d, d]) <= rankTol);

            P.Dispose();
            arena.Dispose();
        }

        // A·P ≈ Q·R reconstruction + non-increasing |R diag|. Result column j is original column P[j],
        // so (Q·R)[:, j] must equal A[:, P[j]].
        void CheckQRCPReconstruct(ref Arena arena, in doubleMxN A)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            var Q = A.Copy();   // overwritten with Q (m × n)
            var R = arena.doubleMat(n, n);
            var P = new Pivot(n, Allocator.Temp);
            QRCP.decompInPlace(ref Q, ref R, ref P);

            // QRProduct = Q · R (m × n)
            var QRProduct = arena.doubleMat(m, n);
            Blas.dot(in Q, in R, ref QRProduct);

            double tol = (MatMaxAbs(in A) + (double)1) * (double)100 * Consts.doubleSqrtEps;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    AssertClose(QRProduct[i, j], A[i, P[j]], tol);

            // |R[0,0]| ≥ |R[1,1]| ≥ … (Businger–Golub ordering)
            double slack = (double)10 * Consts.doubleSqrtEps;
            for (int d = 0; d < n - 1; d++)
                AssertTrue(math.abs(R[d, d]) >= math.abs(R[d + 1, d + 1]) - slack);

            P.Dispose();
        }

        // =====================================================================
        // SVD — singular values / pinv least squares.
        // =====================================================================

        // Hadamard: HᵀH = n·I ⇒ every singular value = √n and cond = 1 exactly. n = 4 and n = 8.
        void SVDHadamardSigma()
        {
            var arena = new Arena(Allocator.Persistent);

            CheckHadamardSVD(ref arena, 4);
            CheckHadamardSVD(ref arena, 8);

            arena.Dispose();
        }

        void CheckHadamardSVD(ref Arena arena, int n)
        {
            var A = arena.doubleHadamard(n);

            var S = arena.doubleVec(n);
            SVD.singularValues(in A, ref S);

            double sq = math.sqrt((double)n);
            double band = (double)50 * Consts.doubleSqrtEps * sq;
            for (int i = 0; i < n; i++)
                AssertClose(S[i], sq, band);

            double condBand = IsDouble() ? (double)1E-5 : (double)1E-2;
            AssertClose(Analysis.cond(in A), (double)1, condBand);
        }

        // Parter: nonsymmetric Toeplitz; all singular values < π and cluster near π. n = 8.
        void SVDParterCluster()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 8;
            var A = arena.doubleParter(n);

            var S = arena.doubleVec(n);
            SVD.singularValues(in A, ref S);

            double pi = (double)math.PI_DBL;
            double band = (double)50 * Consts.doubleSqrtEps;
            for (int i = 0; i < n; i++)
                AssertTrue(S[i] < pi + band);

            AssertTrue(S[0] > pi - (double)0.5);   // largest clusters near π

            arena.Dispose();
        }

        // Läuchli rank-stress: A = (n+1)×n, σ = {√(n+ε²), ε, …, ε}. SVD resolves the tiny ε singular
        // values accurately (absolute error ≈ eps·σ_max), the numerical rank stays n, and the SVD
        // pseudo-inverse solve recovers a consistent xTrue where the system is ill-conditioned
        // (κ ≈ √n/ε). pinv accuracy is precision-gated.
        void SVDLauchliRankStress()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 3;
            double eps = (double)1E-3;
            var A = arena.doubleLauchli(n, eps);   // 4×3

            // singular values
            var S = arena.doubleVec(n);
            SVD.singularValues(in A, ref S);

            double sMax = math.sqrt((double)n + eps * eps);
            AssertClose(S[0], sMax, (double)50 * Consts.doubleSqrtEps * sMax);
            AssertClose(S[1], eps, (double)1 * Consts.doubleSqrtEps);
            AssertClose(S[2], eps, (double)1 * Consts.doubleSqrtEps);

            // numerical rank is full (n) at this ε
            RecordEq(Analysis.rank(in A), n);

            // pinv least squares recovers a consistent xTrue
            var xTrue = arena.doubleVec(n);
            xTrue[0] = (double)1; xTrue[1] = (double)2; xTrue[2] = (double)3;

            var b = Blas.dot(A, xTrue);   // length 4, in range(A)

            var Aw = A.Copy();                // pinvSolve no longer modifies A (copy kept for clarity)
            var x = arena.doubleVec(n);
            RankInfo pinvInfo = SVD.pinvSolve(ref Aw, in b, ref x);
            bool conv = pinvInfo;
            int r = pinvInfo.rank;
            AssertTrue(conv);
            RecordEq(r, n);

            double xtol = IsDouble() ? (double)1E-7 : (double)1E-2;
            for (int i = 0; i < n; i++)
                AssertClose(x[i], xTrue[i], xtol);

            arena.Dispose();
        }

        // =====================================================================
        // Symmetric eigen (Jacobi).
        // =====================================================================

        // Reconstruct V·diag(λ)·Vᵀ ≈ A and orthogonality VᵀV ≈ I on SPD/symmetric inputs.
        void EigenSymmetricReconstruct()
        {
            var arena = new Arena(Allocator.Persistent);

            var T = arena.doubleLaplacian1D(6);     CheckEigenReconstruct(ref arena, in T);
            var Pe = arena.doublePei(5, (double)2);  CheckEigenReconstruct(ref arena, in Pe);
            var P = arena.doublePascal(5);           CheckEigenReconstruct(ref arena, in P);

            arena.Dispose();
        }

        void CheckEigenReconstruct(ref Arena arena, in doubleMxN A)
        {
            int n = A.M_Rows;

            var Ac = A.Copy();
            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);
            AssertTrue(Eigen.decompInPlace(ref Ac, ref eig, ref V));

            // VᵀV ≈ I
            var VtV = arena.doubleMat(n, n);
            Blas.dot(in V, in V, ref VtV, transposeA: true);
            double orthoTol = (double)50 * Consts.doubleSqrtEps;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    AssertClose(VtV[i, j], (i == j) ? (double)1 : (double)0, orthoTol);

            // rec = V · diag(eig) · Vᵀ
            var D = arena.doubleMat(n, n);   // zero-initialized
            for (int i = 0; i < n; i++) D[i, i] = eig[i];
            var VD = arena.doubleMat(n, n);
            Blas.dot(in V, in D, ref VD);
            var Vt = arena.doubleMat(n, n);
            Blas.trans(in V, ref Vt);
            var rec = arena.doubleMat(n, n);
            Blas.dot(in VD, in Vt, ref rec);

            double tol = (MatMaxAbs(in A) + (double)1) * (double)50 * Consts.doubleSqrtEps;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    AssertClose(rec[i, j], A[i, j], tol);
        }

        // Laplacian1D eigenvalues λ_k = 2 − 2cos(kπ/(n+1)). n = 6 (descending).
        void EigenLaplacianSpectrum()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;
            var T = arena.doubleLaplacian1D(n);

            var Tc = T.Copy();
            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);
            AssertTrue(Eigen.decompInPlace(ref Tc, ref eig, ref V));

            double pi = (double)math.PI_DBL;
            double tol = (double)50 * Consts.doubleSqrtEps;
            for (int i = 0; i < n; i++)
            {
                double expected = (double)2 - (double)2 * math.cos((double)(n - i) * pi / (double)(n + 1));
                AssertClose(eig[i], expected, tol);
            }

            arena.Dispose();
        }

        // Clement: eigenvalues exactly {n−1, n−3, …, −(n−1)}. n = 4 ⇒ {3, 1, −1, −3}.
        void EigenClementSpectrum()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;
            var C = arena.doubleClement(n);

            var Cc = C.Copy();
            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);
            AssertTrue(Eigen.decompInPlace(ref Cc, ref eig, ref V));

            double tol = (double)50 * Consts.doubleSqrtEps;
            AssertClose(eig[0], (double)3, tol);
            AssertClose(eig[1], (double)1, tol);
            AssertClose(eig[2], (double)(-1), tol);
            AssertClose(eig[3], (double)(-3), tol);

            arena.Dispose();
        }

        // Fiedler: exactly one positive eigenvalue, n−1 negative (indefinite inertia). n = 5.
        void EigenFiedlerInertia()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            var F = arena.doubleFiedler(n);

            var Fc = F.Copy();
            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);
            AssertTrue(Eigen.decompInPlace(ref Fc, ref eig, ref V));

            // smallest |λ| ≈ 0.56 ⇒ a small gate cleanly separates signs.
            double gate = (double)1E-2;
            int pos = 0, neg = 0;
            for (int i = 0; i < n; i++)
            {
                if (eig[i] > gate) pos++;
                else if (eig[i] < -gate) neg++;
            }
            RecordEq(pos, 1);
            RecordEq(neg, n - 1);

            arena.Dispose();
        }

        // DingDong: all eigenvalues in (−π/2, π/2), clustering near ±π/2. n = 6.
        void EigenDingDongBand()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;
            var D = arena.doubleDingDong(n);

            var Dc = D.Copy();
            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);
            AssertTrue(Eigen.decompInPlace(ref Dc, ref eig, ref V));

            double halfPi = (double)(math.PI_DBL * 0.5);
            double band = (double)10 * Consts.doubleSqrtEps;
            for (int i = 0; i < n; i++)
            {
                AssertTrue(eig[i] <= halfPi + band);
                AssertTrue(eig[i] >= -halfPi - band);
            }
            AssertTrue(eig[0] > halfPi - (double)0.1);
            AssertTrue(eig[n - 1] < -halfPi + (double)0.1);

            arena.Dispose();
        }

        // Rosser 8×8: near-degenerate spectrum, the canonical eigensolver stress test. Documented
        // spectrum matched in a precision-gated band (tight for double, generous for float), plus the
        // robust invariant Σλ = trace = 4040.
        void EigenRosserSpectrum()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleRosser();

            var expected = arena.doubleVec(8);
            expected[0] = (double)1020.4202;
            expected[1] = (double)1019.9936;
            expected[2] = (double)1019.5244;
            expected[3] = (double)1000.1207;
            expected[4] = (double)999.9469;
            expected[5] = (double)0.2180;
            expected[6] = (double)(-0.1705);
            expected[7] = (double)(-1020.0532);

            var Ac = A.Copy();
            var eig = arena.doubleVec(8);
            var V = arena.doubleMat(8, 8);
            AssertTrue(Eigen.decompInPlace(ref Ac, ref eig, ref V));

            double esum = (double)0;
            for (int i = 0; i < 8; i++) esum += eig[i];
            AssertClose(esum, (double)4040, IsDouble() ? (double)1E-3 : (double)0.5);

            double band = IsDouble() ? (double)0.5 : (double)3.0;
            for (int i = 0; i < 8; i++)
                AssertClose(eig[i], expected[i], band);

            arena.Dispose();
        }

        // =====================================================================
        // Non-symmetric eigen (QR algorithm) — valuesQR.
        // =====================================================================

        // Frank(4): all eigenvalues real (imag ≈ 0) and positive (≈ {7.31, 2.07, 0.48, 0.137}).
        void EigenQRFrank()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;
            var F = arena.doubleFrank(n);

            var Fc = F.Copy();
            var re = arena.doubleVec(n);
            var im = arena.doubleVec(n);
            AssertTrue(Eigen.valuesQR(ref Fc, ref re, ref im));

            for (int i = 0; i < n; i++)
            {
                AssertClose(im[i], (double)0, (double)1E-2);
                AssertTrue(re[i] > (double)0);
            }

            arena.Dispose();
        }

        // Companion of (x−1)(x−2)(x−3) = x³ − 6x² + 11x − 6 ⇒ coeffs {−6, 11, −6}.
        // valuesQR returns the roots {3, 2, 1} (descending, real).
        void EigenQRCompanion()
        {
            var arena = new Arena(Allocator.Persistent);

            var coeffs = arena.doubleVec(3);
            coeffs[0] = (double)(-6); coeffs[1] = (double)11; coeffs[2] = (double)(-6);
            var C = arena.doubleCompanion(in coeffs);

            var re = arena.doubleVec(3);
            var im = arena.doubleVec(3);
            AssertTrue(Eigen.valuesQR(ref C, ref re, ref im));

            double tol = (double)1E-2;
            AssertClose(re[0], (double)3, tol);
            AssertClose(re[1], (double)2, tol);
            AssertClose(re[2], (double)1, tol);
            for (int i = 0; i < 3; i++) AssertClose(im[i], (double)0, tol);

            arena.Dispose();
        }

        // =====================================================================
        // Conjugate Gradient — SPD solves.
        // =====================================================================

        // CG on the SPD battery (Laplacian1D, MinIJ, Pei): converges and the residual ‖A x − b‖ is
        // small relative to ‖b‖. xtol is cond-amplified per matrix.
        void CGSPDBattery()
        {
            var arena = new Arena(Allocator.Persistent);

            var T = arena.doubleLaplacian1D(8);
            CheckCG(ref arena, in T, (double)200 * Consts.doubleSqrtEps);

            // MinIJ(5) cond ≈ 44 ⇒ the CG solution error is cond-amplified (the rigorous check here is
            // the residual norm inside CheckCG; this per-component band is a generous sanity bound).
            var M = arena.doubleMinIJ(5);
            CheckCG(ref arena, in M, (double)5E-2);

            var Pe = arena.doublePei(5, (double)2);
            CheckCG(ref arena, in Pe, (double)50 * Consts.doubleSqrtEps);

            arena.Dispose();
        }

        void CheckCG(ref Arena arena, in doubleMxN A, double xtol)
        {
            int n = A.M_Rows;

            var xTrue = arena.doubleVec(n);
            for (int i = 0; i < n; i++) xTrue[i] = (double)(i + 1);

            var b = Blas.dot(A, xTrue);

            var x = arena.doubleVec(n);
            bool conv = Krylov.cg(in A, in b, ref x, 200, Consts.doubleSqrtEps);
            AssertTrue(conv);

            // relative residual ‖A x − b‖ ≤ 100·sqrtEps·‖b‖ (CG guarantees ≤ sqrtEps·‖b‖)
            double relResTol = (double)100 * Consts.doubleSqrtEps * VecNorm(in b);
            AssertTrue(ResidualNorm(in A, in x, in b) <= relResTol);

            for (int i = 0; i < n; i++)
                AssertClose(x[i], xTrue[i], xtol);
        }

        // =====================================================================
        // Condition number (MatrixMetrics.cond).
        // =====================================================================

        // Hadamard cond = 1 (orthogonal up to scale); Hilbert cond grows fast: cond(H₃) ≈ 524.06.
        void CondBattery()
        {
            var arena = new Arena(Allocator.Persistent);

            var Hd = arena.doubleHadamard(4);
            AssertClose(Analysis.cond(in Hd), (double)1, IsDouble() ? (double)1E-5 : (double)1E-2);

            var H3 = arena.doubleHilbert(3);
            double c = Analysis.cond(in H3);
            AssertClose(c, (double)524.0568, IsDouble() ? (double)1 : (double)10);

            arena.Dispose();
        }

        // =====================================================================
        // helpers
        // =====================================================================

        // det via LU on a copy (decompInPlace destroys its input).
        double Determinant(in doubleMxN M)
        {
            int n = M.M_Rows;
            var LUmat = M.Copy();
            var pivot = new Pivot(n, Allocator.Temp);
            LU.decompInPlace(ref LUmat, ref pivot);
            double det = LU.determinant(in LUmat, in pivot);
            pivot.Dispose();
            return det;
        }

        // max |A[i,j]| (matrix magnitude, used to scale backward-stable tolerances).
        double MatMaxAbs(in doubleMxN A)
        {
            double mx = (double)0;
            for (int i = 0; i < A.M_Rows; i++)
                for (int j = 0; j < A.N_Cols; j++)
                {
                    double v = math.abs(A[i, j]);
                    if (v > mx) mx = v;
                }
            return mx;
        }

        // max |A[i,j] − B[i,j]| over same-shape matrices.
        double MaxAbsDiff(in doubleMxN A, in doubleMxN B)
        {
            double mx = (double)0;
            for (int i = 0; i < A.M_Rows; i++)
                for (int j = 0; j < A.N_Cols; j++)
                {
                    double v = math.abs(A[i, j] - B[i, j]);
                    if (v > mx) mx = v;
                }
            return mx;
        }

        double VecNorm(in doubleN v)
        {
            double s = (double)0;
            for (int i = 0; i < v.N; i++) s += v[i] * v[i];
            return math.sqrt(s);
        }

        // ‖A x − b‖₂
        double ResidualNorm(in doubleMxN A, in doubleN x, in doubleN b)
        {
            var Ax = Blas.dot(A, x);
            double s = (double)0;
            for (int i = 0; i < b.N; i++)
            {
                double d = Ax[i] - b[i];
                s += d * d;
            }
            return math.sqrt(s);
        }

        // true only when double expands to double (doubleEpsilon ≈ 2.2e-16 < 1e-10).
        bool IsDouble() => (double)Consts.doubleEpsilon < 1e-10;

        void AssertClose(double a, double b, double precision)
        {
            double diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1; Fail[1] = a; Fail[2] = b; Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (double)0)
            {
                Fail[0] = (double)1; Fail[1] = (double)0; Fail[2] = (double)1; Fail[3] = (double)1;
            }
            Assert.IsTrue(ok);
        }

        void RecordEq(int got, int expected)
        {
            if (got != expected && Fail[0] == (double)0)
            {
                Fail[0] = (double)1; Fail[1] = got; Fail[2] = expected; Fail[3] = got - expected;
            }
            Assert.AreEqual(expected, got);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void SolverBatteryTests(TestJob.TestType type)
    {
        var fail = new NativeArray<double>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff {fail[3]} ({e.Message})");
            throw;
        }
        finally { fail.Dispose(); }
    }
}
