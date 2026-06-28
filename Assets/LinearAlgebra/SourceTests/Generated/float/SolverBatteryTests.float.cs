using System;

using LinearAlgebra;
using LinearAlgebra.Gallery;   // opt-in: arena.floatHilbert(n), arena.floatKahan(n,θ), ...

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
// Verification reuses the library's own ops (Cholesky, LU, OrthoOP QR/QRCP, SVD, Eigen, MatrixMetrics).
// Tolerances are per-precision: they scale with Consts.floatSqrtEps (float ≈ 3.45e-4, double ≈ 1.49e-8)
// so the SAME expression is loose for float and tight for double, matching the GalleryTests idiom.
// The tightest near-degenerate facts (Rosser spectrum, cond(Hilbert), Lauchli pinv accuracy) are
// precision-gated via IsDouble(): a tight band for double, a generous band for float. Reconstruction /
// residual errors are backward-stable (≈ eps·‖A‖, NOT cond-amplified), so they stay tight even for the
// ill-conditioned generators; the cond-amplified facts (LU/QR/CG solution accuracy on ill-conditioned A)
// use generous, sqrtEps-scaled bands. Reference values were cross-checked offline.
public class floatSolverBatteryTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
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
        public NativeArray<float> Fail;

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

            var H = arena.floatHilbert(4);            CheckCholeskyReconstruct(ref arena, in H, (float)50);
            var P = arena.floatPascal(5);             CheckCholeskyReconstruct(ref arena, in P, (float)50);
            var L = arena.floatLehmer(5);             CheckCholeskyReconstruct(ref arena, in L, (float)50);
            var M = arena.floatMinIJ(5);              CheckCholeskyReconstruct(ref arena, in M, (float)50);
            var Pe = arena.floatPei(5, (float)2);    CheckCholeskyReconstruct(ref arena, in Pe, (float)50);
            var Mo = arena.floatMoler(5);             CheckCholeskyReconstruct(ref arena, in Mo, (float)100);
            var T = arena.floatLaplacian1D(6);        CheckCholeskyReconstruct(ref arena, in T, (float)50);
            var G = arena.floatGCD(5);                CheckCholeskyReconstruct(ref arena, in G, (float)50);

            arena.Dispose();
        }

        void CheckCholeskyReconstruct(ref Arena arena, in floatMxN A, float factor)
        {
            int n = A.M_Rows;

            var L = arena.floatMat(n, n);
            AssertTrue(Cholesky.choleskyDecomposition(in A, ref L));

            // rec = L · Lᵀ
            var Lt = arena.floatMat(n, n);
            floatOP.trans(in L, ref Lt);
            var rec = arena.floatMat(n, n);
            floatOP.dot(in L, in Lt, ref rec);

            float tol = (MatMaxAbs(in A) + (float)1) * Consts.floatSqrtEps * factor;
            AssertTrue(MaxAbsDiff(in A, in rec) <= tol);
        }

        // Indefinite inputs MUST be rejected (choleskyDecomposition returns false): Fiedler(n≥2) and
        // Clement(n≥2) both have a zero diagonal → first pivot non-positive; Rosser is indefinite
        // (negative eigenvalues) → a later pivot goes non-positive.
        void CholeskyRejectIndefinite()
        {
            var arena = new Arena(Allocator.Persistent);

            var F = arena.floatFiedler(3);
            var Lf = arena.floatMat(3, 3);
            AssertTrue(!Cholesky.choleskyDecomposition(in F, ref Lf));

            var C = arena.floatClement(3);
            var Lc = arena.floatMat(3, 3);
            AssertTrue(!Cholesky.choleskyDecomposition(in C, ref Lc));

            var R = arena.floatRosser();
            var Lr = arena.floatMat(8, 8);
            AssertTrue(!Cholesky.choleskyDecomposition(in R, ref Lr));

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

            var P = arena.floatPascal(5);
            AssertClose(Determinant(in P), (float)1, (float)150 * Consts.floatSqrtEps);

            var M = arena.floatMinIJ(5);
            AssertClose(Determinant(in M), (float)1, (float)50 * Consts.floatSqrtEps);

            var Mo = arena.floatMoler(5);
            AssertClose(Determinant(in Mo), (float)1, (float)150 * Consts.floatSqrtEps);

            var Fr = arena.floatFrank(4);
            AssertClose(Determinant(in Fr), (float)1, (float)0.05);

            var Tw = arena.floatTriw(5, (float)(-2));
            AssertClose(Determinant(in Tw), (float)1, (float)1E-3);

            var nodes = arena.floatVec(4);
            nodes[0] = (float)1; nodes[1] = (float)2; nodes[2] = (float)3; nodes[3] = (float)4;
            var V = arena.floatVandermonde(in nodes);
            AssertClose(Determinant(in V), (float)12, (float)0.2);

            var G = arena.floatGCD(5);
            AssertClose(Determinant(in G), (float)16, (float)0.5);

            var Rh = arena.floatRedheffer(5);
            AssertClose(Determinant(in Rh), (float)(-2), (float)0.1);

            arena.Dispose();
        }

        // LU-solve reconstructs x on well-conditioned matrices (Laplacian1D, Pascal).
        // xtol is cond-amplified: Pascal(5) (cond ≈ 8.5e3) needs a looser band than Laplacian1D.
        void LUSolveBattery()
        {
            var arena = new Arena(Allocator.Persistent);

            var T = arena.floatLaplacian1D(8);
            CheckLUSolve(ref arena, in T, (float)200 * Consts.floatSqrtEps);

            var P = arena.floatPascal(5);
            CheckLUSolve(ref arena, in P, (float)5E-2);

            arena.Dispose();
        }

        void CheckLUSolve(ref Arena arena, in floatMxN A, float xtol)
        {
            int n = A.M_Rows;

            var xTrue = arena.floatVec(n);
            for (int i = 0; i < n; i++) xTrue[i] = (float)(i + 1);

            var b = floatOP.dot(A, xTrue);   // consistent RHS

            var LUm = A.Copy();
            var P = new Pivot(n, Allocator.Temp);
            AssertTrue(LU.luDecompositionInplace(ref LUm, ref P));

            var x = b.Copy();                 // LUSolve overwrites b with x
            LU.LUSolve(ref LUm, in P, ref x);

            // residual ‖A x − b‖ with the ORIGINAL A,b (backward-stable ⇒ tiny)
            float resTol = (MatMaxAbs(in A) + (float)1) * (float)100 * Consts.floatSqrtEps;
            AssertTrue(ResidualNorm(in A, in x, in b) <= resTol);

            for (int i = 0; i < n; i++)
                AssertClose(x[i], xTrue[i], xtol);

            P.Dispose();
        }

        // =====================================================================
        // QR (Householder) — square solve + overdetermined least squares.
        // =====================================================================

        // Well-conditioned square systems solved via qrDirectSolve (Solvers.SolveQR): residual small.
        // Laplacian1D(8) (cond ≈ 41) and Pei(5,2) (eigenvalues {7,2,2,2,2}, cond ≈ 3.5).
        void QRDirectSolveSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            var T = arena.floatLaplacian1D(8);
            CheckQRSquare(ref arena, in T, (float)200 * Consts.floatSqrtEps);

            var Pe = arena.floatPei(5, (float)2);
            CheckQRSquare(ref arena, in Pe, (float)50 * Consts.floatSqrtEps);

            arena.Dispose();
        }

        void CheckQRSquare(ref Arena arena, in floatMxN A, float xtol)
        {
            int n = A.M_Rows;

            var xTrue = arena.floatVec(n);
            for (int i = 0; i < n; i++) xTrue[i] = (float)(i + 1);

            var b = floatOP.dot(A, xTrue);

            var Aw = A.Copy();   // qrDirectSolve destroys A and b
            var bw = b.Copy();
            var x = arena.floatVec(n);
            Solvers.SolveQR(ref Aw, ref bw, ref x);

            float resTol = (MatMaxAbs(in A) + (float)1) * (float)100 * Consts.floatSqrtEps;
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
            var A = arena.floatLauchli(n, (float)0.5);   // 4×3

            var xTrue = arena.floatVec(n);
            xTrue[0] = (float)1; xTrue[1] = (float)(-2); xTrue[2] = (float)3;

            var b = floatOP.dot(A, xTrue);   // length 4, in range(A)

            var Aw = A.Copy();
            var bw = b.Copy();
            var x = arena.floatVec(n);
            Solvers.SolveQR(ref Aw, ref bw, ref x);   // x has length A.N_Cols = 3

            float xtol = (float)50 * Consts.floatSqrtEps;
            for (int i = 0; i < n; i++)
                AssertClose(x[i], xTrue[i], xtol);

            // residual ‖A x − b‖ ≈ 0 for a consistent system
            float resTol = (MatMaxAbs(in A) + (float)1) * (float)100 * Consts.floatSqrtEps;
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
            var A = arena.floatKahan(n, (float)1.2);
            CheckQRCPReconstruct(ref arena, in A);

            arena.Dispose();
        }

        // Rank-deficient input: Pei(4, 0) = αI + J with α = 0 is the all-ones matrix (rank 1). QRCP must
        // reconstruct A·P ≈ Q·R, keep |R diag| non-increasing, and drive R[1,1]… to ≈ 0 (revealing rank 1).
        void QRCPRankDeficient()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;
            var A = arena.floatPei(n, (float)0);   // all-ones, rank 1
            CheckQRCPReconstruct(ref arena, in A);

            // rank 1 ⇒ trailing diagonal entries are numerically zero
            var Q = A.Copy();
            var R = arena.floatMat(n, n);
            var P = new Pivot(n, Allocator.Temp);
            OrthoOP.qrDecompositionColumnPivot(ref Q, ref R, ref P);

            float rankTol = (float)50 * Consts.floatSqrtEps;
            for (int d = 1; d < n; d++)
                AssertTrue(math.abs(R[d, d]) <= rankTol);

            P.Dispose();
            arena.Dispose();
        }

        // A·P ≈ Q·R reconstruction + non-increasing |R diag|. Result column j is original column P[j],
        // so (Q·R)[:, j] must equal A[:, P[j]].
        void CheckQRCPReconstruct(ref Arena arena, in floatMxN A)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            var Q = A.Copy();   // overwritten with Q (m × n)
            var R = arena.floatMat(n, n);
            var P = new Pivot(n, Allocator.Temp);
            OrthoOP.qrDecompositionColumnPivot(ref Q, ref R, ref P);

            // QR = Q · R (m × n)
            var QR = arena.floatMat(m, n);
            floatOP.dot(in Q, in R, ref QR);

            float tol = (MatMaxAbs(in A) + (float)1) * (float)100 * Consts.floatSqrtEps;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    AssertClose(QR[i, j], A[i, P[j]], tol);

            // |R[0,0]| ≥ |R[1,1]| ≥ … (Businger–Golub ordering)
            float slack = (float)10 * Consts.floatSqrtEps;
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
            var A = arena.floatHadamard(n);

            var S = arena.floatVec(n);
            SVD.singularValues(in A, ref S);

            float sq = math.sqrt((float)n);
            float band = (float)50 * Consts.floatSqrtEps * sq;
            for (int i = 0; i < n; i++)
                AssertClose(S[i], sq, band);

            float condBand = IsDouble() ? (float)1E-5 : (float)1E-2;
            AssertClose(floatOP.cond(in A), (float)1, condBand);
        }

        // Parter: nonsymmetric Toeplitz; all singular values < π and cluster near π. n = 8.
        void SVDParterCluster()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 8;
            var A = arena.floatParter(n);

            var S = arena.floatVec(n);
            SVD.singularValues(in A, ref S);

            float pi = (float)math.PI_DBL;
            float band = (float)50 * Consts.floatSqrtEps;
            for (int i = 0; i < n; i++)
                AssertTrue(S[i] < pi + band);

            AssertTrue(S[0] > pi - (float)0.5);   // largest clusters near π

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
            float eps = (float)1E-3;
            var A = arena.floatLauchli(n, eps);   // 4×3

            // singular values
            var S = arena.floatVec(n);
            SVD.singularValues(in A, ref S);

            float sMax = math.sqrt((float)n + eps * eps);
            AssertClose(S[0], sMax, (float)50 * Consts.floatSqrtEps * sMax);
            AssertClose(S[1], eps, (float)1 * Consts.floatSqrtEps);
            AssertClose(S[2], eps, (float)1 * Consts.floatSqrtEps);

            // numerical rank is full (n) at this ε
            RecordEq(floatOP.rank(in A), n);

            // pinv least squares recovers a consistent xTrue
            var xTrue = arena.floatVec(n);
            xTrue[0] = (float)1; xTrue[1] = (float)2; xTrue[2] = (float)3;

            var b = floatOP.dot(A, xTrue);   // length 4, in range(A)

            var Aw = A.Copy();                // pinvSolve no longer modifies A (copy kept for clarity)
            var x = arena.floatVec(n);
            int r = SVD.pinvSolve(ref Aw, in b, ref x, out bool conv);
            AssertTrue(conv);
            RecordEq(r, n);

            float xtol = IsDouble() ? (float)1E-7 : (float)1E-2;
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

            var T = arena.floatLaplacian1D(6);     CheckEigenReconstruct(ref arena, in T);
            var Pe = arena.floatPei(5, (float)2);  CheckEigenReconstruct(ref arena, in Pe);
            var P = arena.floatPascal(5);           CheckEigenReconstruct(ref arena, in P);

            arena.Dispose();
        }

        void CheckEigenReconstruct(ref Arena arena, in floatMxN A)
        {
            int n = A.M_Rows;

            var Ac = A.Copy();
            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            AssertTrue(Eigen.eigenDecomposition(ref Ac, ref eig, ref V));

            // VᵀV ≈ I
            var VtV = arena.floatMat(n, n);
            floatOP.dot(in V, in V, ref VtV, transposeA: true);
            float orthoTol = (float)50 * Consts.floatSqrtEps;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    AssertClose(VtV[i, j], (i == j) ? (float)1 : (float)0, orthoTol);

            // rec = V · diag(eig) · Vᵀ
            var D = arena.floatMat(n, n);   // zero-initialized
            for (int i = 0; i < n; i++) D[i, i] = eig[i];
            var VD = arena.floatMat(n, n);
            floatOP.dot(in V, in D, ref VD);
            var Vt = arena.floatMat(n, n);
            floatOP.trans(in V, ref Vt);
            var rec = arena.floatMat(n, n);
            floatOP.dot(in VD, in Vt, ref rec);

            float tol = (MatMaxAbs(in A) + (float)1) * (float)50 * Consts.floatSqrtEps;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    AssertClose(rec[i, j], A[i, j], tol);
        }

        // Laplacian1D eigenvalues λ_k = 2 − 2cos(kπ/(n+1)). n = 6 (descending).
        void EigenLaplacianSpectrum()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;
            var T = arena.floatLaplacian1D(n);

            var Tc = T.Copy();
            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            AssertTrue(Eigen.eigenDecomposition(ref Tc, ref eig, ref V));

            float pi = (float)math.PI_DBL;
            float tol = (float)50 * Consts.floatSqrtEps;
            for (int i = 0; i < n; i++)
            {
                float expected = (float)2 - (float)2 * math.cos((float)(n - i) * pi / (float)(n + 1));
                AssertClose(eig[i], expected, tol);
            }

            arena.Dispose();
        }

        // Clement: eigenvalues exactly {n−1, n−3, …, −(n−1)}. n = 4 ⇒ {3, 1, −1, −3}.
        void EigenClementSpectrum()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;
            var C = arena.floatClement(n);

            var Cc = C.Copy();
            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            AssertTrue(Eigen.eigenDecomposition(ref Cc, ref eig, ref V));

            float tol = (float)50 * Consts.floatSqrtEps;
            AssertClose(eig[0], (float)3, tol);
            AssertClose(eig[1], (float)1, tol);
            AssertClose(eig[2], (float)(-1), tol);
            AssertClose(eig[3], (float)(-3), tol);

            arena.Dispose();
        }

        // Fiedler: exactly one positive eigenvalue, n−1 negative (indefinite inertia). n = 5.
        void EigenFiedlerInertia()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            var F = arena.floatFiedler(n);

            var Fc = F.Copy();
            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            AssertTrue(Eigen.eigenDecomposition(ref Fc, ref eig, ref V));

            // smallest |λ| ≈ 0.56 ⇒ a small gate cleanly separates signs.
            float gate = (float)1E-2;
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
            var D = arena.floatDingDong(n);

            var Dc = D.Copy();
            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            AssertTrue(Eigen.eigenDecomposition(ref Dc, ref eig, ref V));

            float halfPi = (float)(math.PI_DBL * 0.5);
            float band = (float)10 * Consts.floatSqrtEps;
            for (int i = 0; i < n; i++)
            {
                AssertTrue(eig[i] <= halfPi + band);
                AssertTrue(eig[i] >= -halfPi - band);
            }
            AssertTrue(eig[0] > halfPi - (float)0.1);
            AssertTrue(eig[n - 1] < -halfPi + (float)0.1);

            arena.Dispose();
        }

        // Rosser 8×8: near-degenerate spectrum, the canonical eigensolver stress test. Documented
        // spectrum matched in a precision-gated band (tight for double, generous for float), plus the
        // robust invariant Σλ = trace = 4040.
        void EigenRosserSpectrum()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.floatRosser();

            var expected = arena.floatVec(8);
            expected[0] = (float)1020.4202;
            expected[1] = (float)1019.9936;
            expected[2] = (float)1019.5244;
            expected[3] = (float)1000.1207;
            expected[4] = (float)999.9469;
            expected[5] = (float)0.2180;
            expected[6] = (float)(-0.1705);
            expected[7] = (float)(-1020.0532);

            var Ac = A.Copy();
            var eig = arena.floatVec(8);
            var V = arena.floatMat(8, 8);
            AssertTrue(Eigen.eigenDecomposition(ref Ac, ref eig, ref V));

            float esum = (float)0;
            for (int i = 0; i < 8; i++) esum += eig[i];
            AssertClose(esum, (float)4040, IsDouble() ? (float)1E-3 : (float)0.5);

            float band = IsDouble() ? (float)0.5 : (float)3.0;
            for (int i = 0; i < 8; i++)
                AssertClose(eig[i], expected[i], band);

            arena.Dispose();
        }

        // =====================================================================
        // Non-symmetric eigen (QR algorithm) — eigenvaluesQR.
        // =====================================================================

        // Frank(4): all eigenvalues real (imag ≈ 0) and positive (≈ {7.31, 2.07, 0.48, 0.137}).
        void EigenQRFrank()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;
            var F = arena.floatFrank(n);

            var Fc = F.Copy();
            var re = arena.floatVec(n);
            var im = arena.floatVec(n);
            AssertTrue(Eigen.eigenvaluesQR(ref Fc, ref re, ref im));

            for (int i = 0; i < n; i++)
            {
                AssertClose(im[i], (float)0, (float)1E-2);
                AssertTrue(re[i] > (float)0);
            }

            arena.Dispose();
        }

        // Companion of (x−1)(x−2)(x−3) = x³ − 6x² + 11x − 6 ⇒ coeffs {−6, 11, −6}.
        // eigenvaluesQR returns the roots {3, 2, 1} (descending, real).
        void EigenQRCompanion()
        {
            var arena = new Arena(Allocator.Persistent);

            var coeffs = arena.floatVec(3);
            coeffs[0] = (float)(-6); coeffs[1] = (float)11; coeffs[2] = (float)(-6);
            var C = arena.floatCompanion(in coeffs);

            var re = arena.floatVec(3);
            var im = arena.floatVec(3);
            AssertTrue(Eigen.eigenvaluesQR(ref C, ref re, ref im));

            float tol = (float)1E-2;
            AssertClose(re[0], (float)3, tol);
            AssertClose(re[1], (float)2, tol);
            AssertClose(re[2], (float)1, tol);
            for (int i = 0; i < 3; i++) AssertClose(im[i], (float)0, tol);

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

            var T = arena.floatLaplacian1D(8);
            CheckCG(ref arena, in T, (float)200 * Consts.floatSqrtEps);

            // MinIJ(5) cond ≈ 44 ⇒ the CG solution error is cond-amplified (the rigorous check here is
            // the residual norm inside CheckCG; this per-component band is a generous sanity bound).
            var M = arena.floatMinIJ(5);
            CheckCG(ref arena, in M, (float)5E-2);

            var Pe = arena.floatPei(5, (float)2);
            CheckCG(ref arena, in Pe, (float)50 * Consts.floatSqrtEps);

            arena.Dispose();
        }

        void CheckCG(ref Arena arena, in floatMxN A, float xtol)
        {
            int n = A.M_Rows;

            var xTrue = arena.floatVec(n);
            for (int i = 0; i < n; i++) xTrue[i] = (float)(i + 1);

            var b = floatOP.dot(A, xTrue);

            var x = arena.floatVec(n);
            bool conv = Solvers.conjugateGradient(in A, in b, ref x, 200, Consts.floatSqrtEps);
            AssertTrue(conv);

            // relative residual ‖A x − b‖ ≤ 100·sqrtEps·‖b‖ (CG guarantees ≤ sqrtEps·‖b‖)
            float relResTol = (float)100 * Consts.floatSqrtEps * VecNorm(in b);
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

            var Hd = arena.floatHadamard(4);
            AssertClose(floatOP.cond(in Hd), (float)1, IsDouble() ? (float)1E-5 : (float)1E-2);

            var H3 = arena.floatHilbert(3);
            float c = floatOP.cond(in H3);
            AssertClose(c, (float)524.0568, IsDouble() ? (float)1 : (float)10);

            arena.Dispose();
        }

        // =====================================================================
        // helpers
        // =====================================================================

        // det via LU on a copy (luDecompositionInplace destroys its input).
        float Determinant(in floatMxN M)
        {
            int n = M.M_Rows;
            var LUmat = M.Copy();
            var pivot = new Pivot(n, Allocator.Temp);
            LU.luDecompositionInplace(ref LUmat, ref pivot);
            float det = LU.determinant(in LUmat, in pivot);
            pivot.Dispose();
            return det;
        }

        // max |A[i,j]| (matrix magnitude, used to scale backward-stable tolerances).
        float MatMaxAbs(in floatMxN A)
        {
            float mx = (float)0;
            for (int i = 0; i < A.M_Rows; i++)
                for (int j = 0; j < A.N_Cols; j++)
                {
                    float v = math.abs(A[i, j]);
                    if (v > mx) mx = v;
                }
            return mx;
        }

        // max |A[i,j] − B[i,j]| over same-shape matrices.
        float MaxAbsDiff(in floatMxN A, in floatMxN B)
        {
            float mx = (float)0;
            for (int i = 0; i < A.M_Rows; i++)
                for (int j = 0; j < A.N_Cols; j++)
                {
                    float v = math.abs(A[i, j] - B[i, j]);
                    if (v > mx) mx = v;
                }
            return mx;
        }

        float VecNorm(in floatN v)
        {
            float s = (float)0;
            for (int i = 0; i < v.N; i++) s += v[i] * v[i];
            return math.sqrt(s);
        }

        // ‖A x − b‖₂
        float ResidualNorm(in floatMxN A, in floatN x, in floatN b)
        {
            var Ax = floatOP.dot(A, x);
            float s = (float)0;
            for (int i = 0; i < b.N; i++)
            {
                float d = Ax[i] - b[i];
                s += d * d;
            }
            return math.sqrt(s);
        }

        // true only when float expands to double (doubleEpsilon ≈ 2.2e-16 < 1e-10).
        bool IsDouble() => (double)Consts.floatEpsilon < 1e-10;

        void AssertClose(float a, float b, float precision)
        {
            float diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1; Fail[1] = a; Fail[2] = b; Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (float)0)
            {
                Fail[0] = (float)1; Fail[1] = (float)0; Fail[2] = (float)1; Fail[3] = (float)1;
            }
            Assert.IsTrue(ok);
        }

        void RecordEq(int got, int expected)
        {
            if (got != expected && Fail[0] == (float)0)
            {
                Fail[0] = (float)1; Fail[1] = got; Fail[2] = expected; Fail[3] = got - expected;
            }
            Assert.AreEqual(expected, got);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void SolverBatteryTests(TestJob.TestType type)
    {
        var fail = new NativeArray<float>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (float)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (float)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff {fail[3]} ({e.Message})");
            throw;
        }
        finally { fail.Dispose(); }
    }
}
