using System;
#pragma warning disable 618 // intentionally exercises the deprecated cyclic-Jacobi Eigen.decompInPlace (kept for reference)

using LinearAlgebra;
using LinearAlgebra.Gallery;   // opt-in: fProxyGallery.fProxyHilbert(n), fProxyGallery.fProxyKahan(n,θ), ...

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
// Verification reuses the library's own ops (Cholesky, LU, QR/QRCP, SVD, Eigen, Analysis).
// Tolerances are per-precision: they scale with Consts.fProxySqrtEps (float ≈ 3.45e-4, double ≈ 1.49e-8)
// so the SAME expression is loose for float and tight for double, matching the GalleryTests idiom.
// The tightest near-degenerate facts (Rosser spectrum, cond(Hilbert), Lauchli pinv accuracy) are
// precision-gated via IsDouble(): a tight band for double, a generous band for float. Reconstruction /
// residual errors are backward-stable (≈ eps·‖A‖, NOT cond-amplified), so they stay tight even for the
// ill-conditioned generators; the cond-amplified facts (LU/QR/CG solution accuracy on ill-conditioned A)
// use generous, sqrtEps-scaled bands. Reference values were cross-checked offline.
public class fProxySolverBatteryTests
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
        public NativeArray<fProxy> Fail;

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
            var H = fProxyGallery.fProxyHilbert(4);            CheckCholeskyReconstruct(in H, (fProxy)50);
            var P = fProxyGallery.fProxyPascal(5);             CheckCholeskyReconstruct(in P, (fProxy)50);
            var L = fProxyGallery.fProxyLehmer(5);             CheckCholeskyReconstruct(in L, (fProxy)50);
            var M = fProxyGallery.fProxyMinIJ(5);              CheckCholeskyReconstruct(in M, (fProxy)50);
            var Pe = fProxyGallery.fProxyPei(5, (fProxy)2);    CheckCholeskyReconstruct(in Pe, (fProxy)50);
            var Mo = fProxyGallery.fProxyMoler(5);             CheckCholeskyReconstruct(in Mo, (fProxy)100);
            var T = fProxyGallery.fProxyLaplacian1D(6);        CheckCholeskyReconstruct(in T, (fProxy)50);
            var G = fProxyGallery.fProxyGCD(5);                CheckCholeskyReconstruct(in G, (fProxy)50);
        }

        void CheckCholeskyReconstruct(in fProxyMxN A, fProxy factor)
        {
            int n = A.M_Rows;

            var L = new fProxyMxN(n, n, Allocator.Temp);
            AssertTrue(CHO.decomp(in A, ref L));

            // rec = L · Lᵀ
            var Lt = new fProxyMxN(n, n, Allocator.Temp);
            Blas.trans(in L, ref Lt);
            var rec = new fProxyMxN(n, n, Allocator.Temp);
            Blas.dot(in L, in Lt, ref rec);

            fProxy tol = (MatMaxAbs(in A) + (fProxy)1) * Consts.fProxySqrtEps * factor;
            AssertTrue(MaxAbsDiff(in A, in rec) <= tol);
        }

        // Indefinite inputs MUST be rejected (decomp returns false): Fiedler(n≥2) and
        // Clement(n≥2) both have a zero diagonal → first pivot non-positive; Rosser is indefinite
        // (negative eigenvalues) → a later pivot goes non-positive.
        void CholeskyRejectIndefinite()
        {
            var F = fProxyGallery.fProxyFiedler(3);
            var Lf = new fProxyMxN(3, 3, Allocator.Temp);
            AssertTrue(!CHO.decomp(in F, ref Lf));

            var C = fProxyGallery.fProxyClement(3);
            var Lc = new fProxyMxN(3, 3, Allocator.Temp);
            AssertTrue(!CHO.decomp(in C, ref Lc));

            var R = fProxyGallery.fProxyRosser();
            var Lr = new fProxyMxN(8, 8, Allocator.Temp);
            AssertTrue(!CHO.decomp(in R, ref Lr));
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
            var P = fProxyGallery.fProxyPascal(5);
            AssertClose(Determinant(in P), (fProxy)1, (fProxy)150 * Consts.fProxySqrtEps);

            var M = fProxyGallery.fProxyMinIJ(5);
            AssertClose(Determinant(in M), (fProxy)1, (fProxy)50 * Consts.fProxySqrtEps);

            var Mo = fProxyGallery.fProxyMoler(5);
            AssertClose(Determinant(in Mo), (fProxy)1, (fProxy)150 * Consts.fProxySqrtEps);

            var Fr = fProxyGallery.fProxyFrank(4);
            AssertClose(Determinant(in Fr), (fProxy)1, (fProxy)0.05);

            var Tw = fProxyGallery.fProxyTriw(5, (fProxy)(-2));
            AssertClose(Determinant(in Tw), (fProxy)1, (fProxy)1E-3);

            var nodes = new fProxyN(4, Allocator.Temp);
            nodes[0] = (fProxy)1; nodes[1] = (fProxy)2; nodes[2] = (fProxy)3; nodes[3] = (fProxy)4;
            var V = fProxyGallery.fProxyVandermonde(in nodes);
            AssertClose(Determinant(in V), (fProxy)12, (fProxy)0.2);

            var G = fProxyGallery.fProxyGCD(5);
            AssertClose(Determinant(in G), (fProxy)16, (fProxy)0.5);

            var Rh = fProxyGallery.fProxyRedheffer(5);
            AssertClose(Determinant(in Rh), (fProxy)(-2), (fProxy)0.1);
        }

        // LU-solve reconstructs x on well-conditioned matrices (Laplacian1D, Pascal).
        // xtol is cond-amplified: Pascal(5) (cond ≈ 8.5e3) needs a looser band than Laplacian1D.
        void LUSolveBattery()
        {
            var T = fProxyGallery.fProxyLaplacian1D(8);
            CheckLUSolve(in T, (fProxy)200 * Consts.fProxySqrtEps);

            var P = fProxyGallery.fProxyPascal(5);
            CheckLUSolve(in P, (fProxy)5E-2);
        }

        void CheckLUSolve(in fProxyMxN A, fProxy xtol)
        {
            int n = A.M_Rows;

            var xTrue = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) xTrue[i] = (fProxy)(i + 1);

            var b = Blas.dot(A, xTrue);   // consistent RHS

            var LUm = A.Copy();
            var P = new Pivot(n, Allocator.Temp);
            AssertTrue(LU.decompInPlace(ref LUm, ref P));

            var x = b.Copy();                 // decompSolve overwrites b with x
            LU.decompSolve(ref LUm, in P, ref x);

            // residual ‖A x − b‖ with the ORIGINAL A,b (backward-stable ⇒ tiny)
            fProxy resTol = (MatMaxAbs(in A) + (fProxy)1) * (fProxy)100 * Consts.fProxySqrtEps;
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
            var T = fProxyGallery.fProxyLaplacian1D(8);
            CheckQRSquare(in T, (fProxy)200 * Consts.fProxySqrtEps);

            var Pe = fProxyGallery.fProxyPei(5, (fProxy)2);
            CheckQRSquare(in Pe, (fProxy)50 * Consts.fProxySqrtEps);
        }

        void CheckQRSquare(in fProxyMxN A, fProxy xtol)
        {
            int n = A.M_Rows;

            var xTrue = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) xTrue[i] = (fProxy)(i + 1);

            var b = Blas.dot(A, xTrue);

            var Aw = A.Copy();   // solveInPlace destroys A and b
            var bw = b.Copy();
            var x = new fProxyN(n, Allocator.Temp);
            QR.solveInPlace(ref Aw, ref bw, ref x);

            fProxy resTol = (MatMaxAbs(in A) + (fProxy)1) * (fProxy)100 * Consts.fProxySqrtEps;
            AssertTrue(ResidualNorm(in A, in x, in b) <= resTol);

            for (int i = 0; i < n; i++)
                AssertClose(x[i], xTrue[i], xtol);
        }

        // Overdetermined least squares on a tall full-column-rank gallery matrix: Läuchli(3, ε=0.5) is
        // 4×3 and well-conditioned at this ε. With a consistent RHS (b = A·xTrue) the LS solution is
        // exactly xTrue and the residual is ≈ 0.
        void QRLeastSquares()
        {
            int n = 3;
            var A = fProxyGallery.fProxyLauchli(n, (fProxy)0.5);   // 4×3

            var xTrue = new fProxyN(n, Allocator.Temp);
            xTrue[0] = (fProxy)1; xTrue[1] = (fProxy)(-2); xTrue[2] = (fProxy)3;

            var b = Blas.dot(A, xTrue);   // length 4, in range(A)

            var Aw = A.Copy();
            var bw = b.Copy();
            var x = new fProxyN(n, Allocator.Temp);
            QR.solveInPlace(ref Aw, ref bw, ref x);   // x has length A.N_Cols = 3

            fProxy xtol = (fProxy)50 * Consts.fProxySqrtEps;
            for (int i = 0; i < n; i++)
                AssertClose(x[i], xTrue[i], xtol);

            // residual ‖A x − b‖ ≈ 0 for a consistent system
            fProxy resTol = (MatMaxAbs(in A) + (fProxy)1) * (fProxy)100 * Consts.fProxySqrtEps;
            AssertTrue(ResidualNorm(in A, in x, in b) <= resTol);
        }

        // =====================================================================
        // QR column-pivot (QRCP) — rank-revealing factorization.
        // =====================================================================

        // Kahan is the classic QRCP counterexample: every column has norm 1, so it is provably invariant
        // under column pivoting (P = identity). We verify A·P ≈ Q·R reconstructs and |R[0,0]| ≥ |R[1,1]|
        // ≥ … (the rank-revealing diagonal ordering).
        void QRCPKahan()
        {
            int n = 5;
            var A = fProxyGallery.fProxyKahan(n, (fProxy)0.36235775);
            CheckQRCPReconstruct(in A);
        }

        // Rank-deficient input: Pei(4, 0) = αI + J with α = 0 is the all-ones matrix (rank 1). QRCP must
        // reconstruct A·P ≈ Q·R, keep |R diag| non-increasing, and drive R[1,1]… to ≈ 0 (revealing rank 1).
        void QRCPRankDeficient()
        {
            int n = 4;
            var A = fProxyGallery.fProxyPei(n, (fProxy)0);   // all-ones, rank 1
            CheckQRCPReconstruct(in A);

            // rank 1 ⇒ trailing diagonal entries are numerically zero
            var Q = A.Copy();
            var R = new fProxyMxN(n, n, Allocator.Temp);
            var P = new Pivot(n, Allocator.Temp);
            QRCP.decompInPlace(ref Q, ref R, ref P);

            fProxy rankTol = (fProxy)50 * Consts.fProxySqrtEps;
            for (int d = 1; d < n; d++)
                AssertTrue(math.abs(R[d, d]) <= rankTol);

            P.Dispose();
        }

        // A·P ≈ Q·R reconstruction + non-increasing |R diag|. Result column j is original column P[j],
        // so (Q·R)[:, j] must equal A[:, P[j]].
        void CheckQRCPReconstruct(in fProxyMxN A)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            var Q = A.Copy();   // overwritten with Q (m × n)
            var R = new fProxyMxN(n, n, Allocator.Temp);
            var P = new Pivot(n, Allocator.Temp);
            QRCP.decompInPlace(ref Q, ref R, ref P);

            // QRProduct = Q · R (m × n)
            var QRProduct = new fProxyMxN(m, n, Allocator.Temp);
            Blas.dot(in Q, in R, ref QRProduct);

            fProxy tol = (MatMaxAbs(in A) + (fProxy)1) * (fProxy)100 * Consts.fProxySqrtEps;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    AssertClose(QRProduct[i, j], A[i, P[j]], tol);

            // |R[0,0]| ≥ |R[1,1]| ≥ … (Businger–Golub ordering)
            fProxy slack = (fProxy)10 * Consts.fProxySqrtEps;
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
            CheckHadamardSVD(4);
            CheckHadamardSVD(8);
        }

        void CheckHadamardSVD(int n)
        {
            var A = fProxyGallery.fProxyHadamard(n);

            var S = new fProxyN(n, Allocator.Temp);
            SVD.singularValues(in A, ref S);

            fProxy sq = math.sqrt((fProxy)n);
            fProxy band = (fProxy)50 * Consts.fProxySqrtEps * sq;
            for (int i = 0; i < n; i++)
                AssertClose(S[i], sq, band);

            fProxy condBand = IsDouble() ? (fProxy)1E-5 : (fProxy)1E-2;
            AssertClose(Analysis.cond(in A), (fProxy)1, condBand);
        }

        // Parter: nonsymmetric Toeplitz; all singular values < π and cluster near π. n = 8.
        void SVDParterCluster()
        {
            int n = 8;
            var A = fProxyGallery.fProxyParter(n);

            var S = new fProxyN(n, Allocator.Temp);
            SVD.singularValues(in A, ref S);

            fProxy pi = (fProxy)math.PI_DBL;
            fProxy band = (fProxy)50 * Consts.fProxySqrtEps;
            for (int i = 0; i < n; i++)
                AssertTrue(S[i] < pi + band);

            AssertTrue(S[0] > pi - (fProxy)0.5);   // largest clusters near π
        }

        // Läuchli rank-stress: A = (n+1)×n, σ = {√(n+ε²), ε, …, ε}. SVD resolves the tiny ε singular
        // values accurately (absolute error ≈ eps·σ_max), the numerical rank stays n, and the SVD
        // pseudo-inverse solve recovers a consistent xTrue where the system is ill-conditioned
        // (κ ≈ √n/ε). pinv accuracy is precision-gated.
        void SVDLauchliRankStress()
        {
            int n = 3;
            fProxy eps = (fProxy)1E-3;
            var A = fProxyGallery.fProxyLauchli(n, eps);   // 4×3

            // singular values
            var S = new fProxyN(n, Allocator.Temp);
            SVD.singularValues(in A, ref S);

            fProxy sMax = math.sqrt((fProxy)n + eps * eps);
            AssertClose(S[0], sMax, (fProxy)50 * Consts.fProxySqrtEps * sMax);
            AssertClose(S[1], eps, (fProxy)1 * Consts.fProxySqrtEps);
            AssertClose(S[2], eps, (fProxy)1 * Consts.fProxySqrtEps);

            // numerical rank is full (n) at this ε
            RecordEq(Analysis.rank(in A), n);

            // pinv least squares recovers a consistent xTrue
            var xTrue = new fProxyN(n, Allocator.Temp);
            xTrue[0] = (fProxy)1; xTrue[1] = (fProxy)2; xTrue[2] = (fProxy)3;

            var b = Blas.dot(A, xTrue);   // length 4, in range(A)

            var Aw = A.Copy();                // pinvSolve no longer modifies A (copy kept for clarity)
            var x = new fProxyN(n, Allocator.Temp);
            RankInfo pinvInfo = SVD.pinvSolve(ref Aw, in b, ref x);
            bool conv = pinvInfo;
            int r = pinvInfo.rank;
            AssertTrue(conv);
            RecordEq(r, n);

            fProxy xtol = IsDouble() ? (fProxy)1E-7 : (fProxy)1E-2;
            for (int i = 0; i < n; i++)
                AssertClose(x[i], xTrue[i], xtol);
        }

        // =====================================================================
        // Symmetric eigen (Jacobi).
        // =====================================================================

        // Reconstruct V·diag(λ)·Vᵀ ≈ A and orthogonality VᵀV ≈ I on SPD/symmetric inputs.
        void EigenSymmetricReconstruct()
        {
            var T = fProxyGallery.fProxyLaplacian1D(6);     CheckEigenReconstruct(in T);
            var Pe = fProxyGallery.fProxyPei(5, (fProxy)2);  CheckEigenReconstruct(in Pe);
            var P = fProxyGallery.fProxyPascal(5);           CheckEigenReconstruct(in P);
        }

        void CheckEigenReconstruct(in fProxyMxN A)
        {
            int n = A.M_Rows;

            var Ac = A.Copy();
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            AssertTrue(Eigen.decompInPlace(ref Ac, ref eig, ref V));

            // VᵀV ≈ I
            var VtV = new fProxyMxN(n, n, Allocator.Temp);
            Blas.dot(in V, in V, ref VtV, transposeA: true);
            fProxy orthoTol = (fProxy)50 * Consts.fProxySqrtEps;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    AssertClose(VtV[i, j], (i == j) ? (fProxy)1 : (fProxy)0, orthoTol);

            // rec = V · diag(eig) · Vᵀ
            var D = new fProxyMxN(n, n, Allocator.Temp);   // zero-initialized
            for (int i = 0; i < n; i++) D[i, i] = eig[i];
            var VD = new fProxyMxN(n, n, Allocator.Temp);
            Blas.dot(in V, in D, ref VD);
            var Vt = new fProxyMxN(n, n, Allocator.Temp);
            Blas.trans(in V, ref Vt);
            var rec = new fProxyMxN(n, n, Allocator.Temp);
            Blas.dot(in VD, in Vt, ref rec);

            fProxy tol = (MatMaxAbs(in A) + (fProxy)1) * (fProxy)50 * Consts.fProxySqrtEps;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    AssertClose(rec[i, j], A[i, j], tol);
        }

        // Laplacian1D eigenvalues λ_k = 2 − 2cos(kπ/(n+1)). n = 6 (descending).
        void EigenLaplacianSpectrum()
        {
            int n = 6;
            var T = fProxyGallery.fProxyLaplacian1D(n);

            var Tc = T.Copy();
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            AssertTrue(Eigen.decompInPlace(ref Tc, ref eig, ref V));

            fProxy pi = (fProxy)math.PI_DBL;
            fProxy tol = (fProxy)50 * Consts.fProxySqrtEps;
            for (int i = 0; i < n; i++)
            {
                fProxy expected = (fProxy)2 - (fProxy)2 * math.cos((fProxy)(n - i) * pi / (fProxy)(n + 1));
                AssertClose(eig[i], expected, tol);
            }
        }

        // Clement: eigenvalues exactly {n−1, n−3, …, −(n−1)}. n = 4 ⇒ {3, 1, −1, −3}.
        void EigenClementSpectrum()
        {
            int n = 4;
            var C = fProxyGallery.fProxyClement(n);

            var Cc = C.Copy();
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            AssertTrue(Eigen.decompInPlace(ref Cc, ref eig, ref V));

            fProxy tol = (fProxy)50 * Consts.fProxySqrtEps;
            AssertClose(eig[0], (fProxy)3, tol);
            AssertClose(eig[1], (fProxy)1, tol);
            AssertClose(eig[2], (fProxy)(-1), tol);
            AssertClose(eig[3], (fProxy)(-3), tol);
        }

        // Fiedler: exactly one positive eigenvalue, n−1 negative (indefinite inertia). n = 5.
        void EigenFiedlerInertia()
        {
            int n = 5;
            var F = fProxyGallery.fProxyFiedler(n);

            var Fc = F.Copy();
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            AssertTrue(Eigen.decompInPlace(ref Fc, ref eig, ref V));

            // smallest |λ| ≈ 0.56 ⇒ a small gate cleanly separates signs.
            fProxy gate = (fProxy)1E-2;
            int pos = 0, neg = 0;
            for (int i = 0; i < n; i++)
            {
                if (eig[i] > gate) pos++;
                else if (eig[i] < -gate) neg++;
            }
            RecordEq(pos, 1);
            RecordEq(neg, n - 1);
        }

        // DingDong: all eigenvalues in (−π/2, π/2), clustering near ±π/2. n = 6.
        void EigenDingDongBand()
        {
            int n = 6;
            var D = fProxyGallery.fProxyDingDong(n);

            var Dc = D.Copy();
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            AssertTrue(Eigen.decompInPlace(ref Dc, ref eig, ref V));

            fProxy halfPi = (fProxy)(math.PI_DBL * 0.5);
            fProxy band = (fProxy)10 * Consts.fProxySqrtEps;
            for (int i = 0; i < n; i++)
            {
                AssertTrue(eig[i] <= halfPi + band);
                AssertTrue(eig[i] >= -halfPi - band);
            }
            AssertTrue(eig[0] > halfPi - (fProxy)0.1);
            AssertTrue(eig[n - 1] < -halfPi + (fProxy)0.1);
        }

        // Rosser 8×8: near-degenerate spectrum, the canonical eigensolver stress test. Documented
        // spectrum matched in a precision-gated band (tight for double, generous for float), plus the
        // robust invariant Σλ = trace = 4040.
        void EigenRosserSpectrum()
        {
            var A = fProxyGallery.fProxyRosser();

            var expected = new fProxyN(8, Allocator.Temp);
            expected[0] = (fProxy)1020.4202;
            expected[1] = (fProxy)1019.9936;
            expected[2] = (fProxy)1019.5244;
            expected[3] = (fProxy)1000.1207;
            expected[4] = (fProxy)999.9469;
            expected[5] = (fProxy)0.2180;
            expected[6] = (fProxy)(-0.1705);
            expected[7] = (fProxy)(-1020.0532);

            var Ac = A.Copy();
            var eig = new fProxyN(8, Allocator.Temp);
            var V = new fProxyMxN(8, 8, Allocator.Temp);
            AssertTrue(Eigen.decompInPlace(ref Ac, ref eig, ref V));

            fProxy esum = (fProxy)0;
            for (int i = 0; i < 8; i++) esum += eig[i];
            AssertClose(esum, (fProxy)4040, IsDouble() ? (fProxy)1E-3 : (fProxy)0.5);

            fProxy band = IsDouble() ? (fProxy)0.5 : (fProxy)3.0;
            for (int i = 0; i < 8; i++)
                AssertClose(eig[i], expected[i], band);
        }

        // =====================================================================
        // Non-symmetric eigen (QR algorithm) — valuesQRInPlace.
        // =====================================================================

        // Frank(4): all eigenvalues real (imag ≈ 0) and positive (≈ {7.31, 2.07, 0.48, 0.137}).
        void EigenQRFrank()
        {
            int n = 4;
            var F = fProxyGallery.fProxyFrank(n);

            var Fc = F.Copy();
            var re = new fProxyN(n, Allocator.Temp);
            var im = new fProxyN(n, Allocator.Temp);
            AssertTrue(Eigen.valuesQRInPlace(ref Fc, ref re, ref im));

            for (int i = 0; i < n; i++)
            {
                AssertClose(im[i], (fProxy)0, (fProxy)1E-2);
                AssertTrue(re[i] > (fProxy)0);
            }
        }

        // Companion of (x−1)(x−2)(x−3) = x³ − 6x² + 11x − 6 ⇒ coeffs {−6, 11, −6}.
        // valuesQRInPlace returns the roots {3, 2, 1} (descending, real).
        void EigenQRCompanion()
        {
            var coeffs = new fProxyN(3, Allocator.Temp);
            coeffs[0] = (fProxy)(-6); coeffs[1] = (fProxy)11; coeffs[2] = (fProxy)(-6);
            var C = fProxyGallery.fProxyCompanion(in coeffs);

            var re = new fProxyN(3, Allocator.Temp);
            var im = new fProxyN(3, Allocator.Temp);
            AssertTrue(Eigen.valuesQRInPlace(ref C, ref re, ref im));

            fProxy tol = (fProxy)1E-2;
            AssertClose(re[0], (fProxy)3, tol);
            AssertClose(re[1], (fProxy)2, tol);
            AssertClose(re[2], (fProxy)1, tol);
            for (int i = 0; i < 3; i++) AssertClose(im[i], (fProxy)0, tol);
        }

        // =====================================================================
        // Conjugate Gradient — SPD solves.
        // =====================================================================

        // CG on the SPD battery (Laplacian1D, MinIJ, Pei): converges and the residual ‖A x − b‖ is
        // small relative to ‖b‖. xtol is cond-amplified per matrix.
        void CGSPDBattery()
        {
            var T = fProxyGallery.fProxyLaplacian1D(8);
            CheckCG(in T, (fProxy)200 * Consts.fProxySqrtEps);

            // MinIJ(5) cond ≈ 44 ⇒ the CG solution error is cond-amplified (the rigorous check here is
            // the residual norm inside CheckCG; this per-component band is a generous sanity bound).
            var M = fProxyGallery.fProxyMinIJ(5);
            CheckCG(in M, (fProxy)5E-2);

            var Pe = fProxyGallery.fProxyPei(5, (fProxy)2);
            CheckCG(in Pe, (fProxy)50 * Consts.fProxySqrtEps);
        }

        void CheckCG(in fProxyMxN A, fProxy xtol)
        {
            int n = A.M_Rows;

            var xTrue = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) xTrue[i] = (fProxy)(i + 1);

            var b = Blas.dot(A, xTrue);

            var x = new fProxyN(n, Allocator.Temp);
            bool conv = Krylov.cg(in A, in b, ref x, 200, Consts.fProxySqrtEps);
            AssertTrue(conv);

            // relative residual ‖A x − b‖ ≤ 100·sqrtEps·‖b‖ (CG guarantees ≤ sqrtEps·‖b‖)
            fProxy relResTol = (fProxy)100 * Consts.fProxySqrtEps * VecNorm(in b);
            AssertTrue(ResidualNorm(in A, in x, in b) <= relResTol);

            for (int i = 0; i < n; i++)
                AssertClose(x[i], xTrue[i], xtol);
        }

        // =====================================================================
        // Condition number (Analysis.cond).
        // =====================================================================

        // Hadamard cond = 1 (orthogonal up to scale); Hilbert cond grows fast: cond(H₃) ≈ 524.06.
        void CondBattery()
        {
            var Hd = fProxyGallery.fProxyHadamard(4);
            AssertClose(Analysis.cond(in Hd), (fProxy)1, IsDouble() ? (fProxy)1E-5 : (fProxy)1E-2);

            var H3 = fProxyGallery.fProxyHilbert(3);
            fProxy c = Analysis.cond(in H3);
            AssertClose(c, (fProxy)524.0568, IsDouble() ? (fProxy)1 : (fProxy)10);
        }

        // =====================================================================
        // helpers
        // =====================================================================

        // det via LU on a copy (decompInPlace destroys its input).
        fProxy Determinant(in fProxyMxN M)
        {
            int n = M.M_Rows;
            var LUmat = M.Copy();
            var pivot = new Pivot(n, Allocator.Temp);
            LU.decompInPlace(ref LUmat, ref pivot);
            fProxy det = Analysis.determinant(in LUmat, in pivot);
            pivot.Dispose();
            return det;
        }

        // max |A[i,j]| (matrix magnitude, used to scale backward-stable tolerances).
        fProxy MatMaxAbs(in fProxyMxN A)
        {
            fProxy mx = (fProxy)0;
            for (int i = 0; i < A.M_Rows; i++)
                for (int j = 0; j < A.N_Cols; j++)
                {
                    fProxy v = math.abs(A[i, j]);
                    if (v > mx) mx = v;
                }
            return mx;
        }

        // max |A[i,j] − B[i,j]| over same-shape matrices.
        fProxy MaxAbsDiff(in fProxyMxN A, in fProxyMxN B)
        {
            fProxy mx = (fProxy)0;
            for (int i = 0; i < A.M_Rows; i++)
                for (int j = 0; j < A.N_Cols; j++)
                {
                    fProxy v = math.abs(A[i, j] - B[i, j]);
                    if (v > mx) mx = v;
                }
            return mx;
        }

        fProxy VecNorm(in fProxyN v)
        {
            fProxy s = (fProxy)0;
            for (int i = 0; i < v.N; i++) s += v[i] * v[i];
            return math.sqrt(s);
        }

        // ‖A x − b‖₂
        fProxy ResidualNorm(in fProxyMxN A, in fProxyN x, in fProxyN b)
        {
            var Ax = Blas.dot(A, x);
            fProxy s = (fProxy)0;
            for (int i = 0; i < b.N; i++)
            {
                fProxy d = Ax[i] - b[i];
                s += d * d;
            }
            return math.sqrt(s);
        }

        // true only when fProxy expands to double (doubleEpsilon ≈ 2.2e-16 < 1e-10).
        bool IsDouble() => (double)Consts.fProxyEpsilon < 1e-10;

        void AssertClose(fProxy a, fProxy b, fProxy precision)
        {
            fProxy diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = a; Fail[2] = b; Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = (fProxy)0; Fail[2] = (fProxy)1; Fail[3] = (fProxy)1;
            }
            Assert.IsTrue(ok);
        }

        void RecordEq(int got, int expected)
        {
            if (got != expected && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = got; Fail[2] = expected; Fail[3] = got - expected;
            }
            Assert.AreEqual(expected, got);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void SolverBatteryTests(TestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff {fail[3]} ({e.Message})");
            throw;
        }
        finally { fail.Dispose(); }
    }
}
