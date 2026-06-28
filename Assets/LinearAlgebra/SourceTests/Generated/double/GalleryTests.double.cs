using System;
#pragma warning disable 618 // intentionally exercises the deprecated Jacobi svdDecomposition / eigenDecomposition (kept for reference)

using LinearAlgebra;
using LinearAlgebra.Gallery;   // opt-in: arena.doublePascal(n), arena.doubleFrank(n), ...

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Property + algorithm-exercise tests for the famous-test-matrix gallery (docs/spec-gallery.md).
// Each case pins a generator against its DOCUMENTED closed form (determinant, eigenvalues, definiteness,
// FFT cross-check) rather than a self-consistency check, then a few cases feed the generators into the
// existing solvers (CG, eigenvaluesQR) as honest inputs.
//
// Verification reuses the library's own ops (LU.determinant, Cholesky, Eigen.eigenDecomposition /
// eigenvaluesQR, doubleFFT.fft). Tolerances are per-precision: they scale with Consts.doubleSqrtEps
// (float ≈ 3.45e-4, double ≈ 1.49e-8) so the SAME expression is loose for float and tight for double,
// matching the LiteratureTests / RandomMatrixTests idiom. Ill-conditioned generators (Hilbert, Frank,
// Pascal at larger n, Moler) use small n and generous multiples; exact-in-float properties (Hadamard
// HᵀH = nI) use a tight tolerance. Reference eigenvalues/determinants were cross-checked offline.
//
// Argument-validation throws run on the managed thread (Assert.Throws), like the sibling guard tests.
public class doubleGalleryTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            // Batch A — SPD / symmetric
            PascalProps,
            MinIJProps,
            MolerProps,
            Laplacian1DProps,
            KMSProps,
            PeiProps,
            LehmerHilbertSpd,
            // Batch B — eigenvalue / nonsymmetric / structured / rank
            ClementEig,
            FiedlerProps,
            DingDongEig,
            FrankProps,
            VandermondeDet,
            CompanionEig,
            HadamardOrthogonal,
            CirculantFFT,
            TriwProps,
            WilkinsonNearPair,
            LauchliDims,
            // Algorithm-exercise
            CGLaplacian,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<double> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.PascalProps:        PascalProps();        break;
                case TestType.MinIJProps:         MinIJProps();         break;
                case TestType.MolerProps:         MolerProps();         break;
                case TestType.Laplacian1DProps:   Laplacian1DProps();   break;
                case TestType.KMSProps:           KMSProps();           break;
                case TestType.PeiProps:           PeiProps();           break;
                case TestType.LehmerHilbertSpd:   LehmerHilbertSpd();   break;
                case TestType.ClementEig:         ClementEig();         break;
                case TestType.FiedlerProps:       FiedlerProps();       break;
                case TestType.DingDongEig:        DingDongEig();        break;
                case TestType.FrankProps:         FrankProps();         break;
                case TestType.VandermondeDet:     VandermondeDet();     break;
                case TestType.CompanionEig:       CompanionEig();       break;
                case TestType.HadamardOrthogonal: HadamardOrthogonal(); break;
                case TestType.CirculantFFT:       CirculantFFT();       break;
                case TestType.TriwProps:          TriwProps();          break;
                case TestType.WilkinsonNearPair:  WilkinsonNearPair();  break;
                case TestType.LauchliDims:        LauchliDims();        break;
                case TestType.CGLaplacian:        CGLaplacian();        break;
            }
        }

        // =====================================================================
        // Batch A
        // =====================================================================

        // Pascal: symmetric, det = 1 (exact integer), SPD (Cholesky succeeds). n = 5.
        void PascalProps()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            var P = arena.doublePascal(n);

            AssertSymmetric(in P, (double)1E-5);
            AssertCholeskyOk(ref arena, in P);

            // det = 1; Pascal(5) cond ≈ 8.5e3, so float det error ≈ a few e-3.
            AssertClose(Determinant(in P), (double)1, (double)150 * Consts.doubleSqrtEps);

            arena.Dispose();
        }

        // MinIJ A[i,j]=min(i,j)+1: symmetric, det = 1, SPD. n = 5.
        void MinIJProps()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            var A = arena.doubleMinIJ(n);

            AssertSymmetric(in A, (double)1E-5);
            AssertCholeskyOk(ref arena, in A);
            AssertClose(Determinant(in A), (double)1, (double)50 * Consts.doubleSqrtEps);

            arena.Dispose();
        }

        // Moler = UᵀU (U = triw 1-diag, α-super): SPD and det = 1 for ALL α.
        // α = −1 (mild, n = 5) for the det assert; α = 2 (n = 4) just for SPD + det with a looser band.
        void MolerProps()
        {
            var arena = new Arena(Allocator.Persistent);

            // α = −1 (default overload), n = 5
            var M1 = arena.doubleMoler(5);
            AssertSymmetric(in M1, (double)1E-5);
            AssertCholeskyOk(ref arena, in M1);
            AssertClose(Determinant(in M1), (double)1, (double)150 * Consts.doubleSqrtEps);

            // α = 2, n = 4 — still SPD with det = 1 (more ill-conditioned ⇒ looser det band)
            var M2 = arena.doubleMoler(4, (double)2);
            AssertSymmetric(in M2, (double)1E-5);
            AssertCholeskyOk(ref arena, in M2);
            AssertClose(Determinant(in M2), (double)1, (double)300 * Consts.doubleSqrtEps);

            arena.Dispose();
        }

        // Laplacian1D (Strang 2nd-difference): SPD, det = n+1, eigenvalues 2−2cos(kπ/(n+1)). n = 6.
        void Laplacian1DProps()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;
            var T = arena.doubleLaplacian1D(n);

            AssertSymmetric(in T, (double)1E-5);
            AssertCholeskyOk(ref arena, in T);

            // det = n + 1 = 7 (read-only path destroys a copy inside Determinant)
            AssertClose(Determinant(in T), (double)(n + 1), (double)0.05);

            // eigenvalues (descending) eig[i] = 2 − 2cos((n−i)π/(n+1))
            var Tc = T.Copy();
            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);
            AssertTrue(Eigen.eigenDecomposition(ref Tc, ref eig, ref V));

            double pi = (double)math.PI_DBL;
            for (int i = 0; i < n; i++)
            {
                double expected = (double)2 - (double)2 * math.cos((double)(n - i) * pi / (double)(n + 1));
                AssertClose(eig[i], expected, (double)50 * Consts.doubleSqrtEps);
            }

            arena.Dispose();
        }

        // KMS A[i,j]=ρ^|i−j|: det = (1−ρ²)^(n−1) for ρ = 0.5 (SPD); and the integer-power path produces
        // sign-correct negative entries for ρ = −0.5.
        void KMSProps()
        {
            var arena = new Arena(Allocator.Persistent);

            // ρ = 0.5, n = 5: det = (1 − 0.25)^4 = 0.31640625; SPD ⇒ Cholesky succeeds.
            int n = 5;
            var A = arena.doubleKMS(n, (double)0.5);
            AssertSymmetric(in A, (double)1E-5);
            AssertCholeskyOk(ref arena, in A);

            double expDet = (double)1;
            for (int k = 0; k < n - 1; k++) expDet *= (double)0.75;   // (1 − ρ²)^(n−1)
            AssertClose(Determinant(in A), expDet, (double)50 * Consts.doubleSqrtEps);

            // ρ = −0.5, n = 3: K[0,1] = ρ^1 = −0.5, K[0,2] = ρ^2 = +0.25 (post-fix integer power).
            var B = arena.doubleKMS(3, (double)(-0.5));
            AssertClose(B[0, 1], (double)(-0.5), (double)1E-5);
            AssertClose(B[0, 2], (double)0.25, (double)1E-5);
            AssertClose(B[1, 2], (double)(-0.5), (double)1E-5);

            arena.Dispose();
        }

        // Pei = αI + J: eigenvalues {α+n (×1), α (×n−1)}; det = αⁿ⁻¹(α+n). α = 2, n = 5 ⇒ {7,2,2,2,2}, det 112.
        void PeiProps()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            double alpha = (double)2;
            var A = arena.doublePei(n, alpha);

            AssertSymmetric(in A, (double)1E-5);

            // det = α^(n−1)·(α+n) = 16·7 = 112
            AssertClose(Determinant(in A), (double)112, (double)0.5);

            var Ac = A.Copy();
            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);
            AssertTrue(Eigen.eigenDecomposition(ref Ac, ref eig, ref V));

            double tol = (double)50 * Consts.doubleSqrtEps;
            AssertClose(eig[0], alpha + (double)n, tol);   // α + n = 7
            for (int i = 1; i < n; i++)
                AssertClose(eig[i], alpha, tol);           // α = 2

            arena.Dispose();
        }

        // Lehmer and Hilbert: both SPD at small n ⇒ symmetric and Cholesky succeeds.
        void LehmerHilbertSpd()
        {
            var arena = new Arena(Allocator.Persistent);

            var L = arena.doubleLehmer(5);
            AssertSymmetric(in L, (double)1E-5);
            AssertCholeskyOk(ref arena, in L);

            var H = arena.doubleHilbert(4);
            AssertSymmetric(in H, (double)1E-5);
            AssertCholeskyOk(ref arena, in H);

            arena.Dispose();
        }

        // =====================================================================
        // Batch B
        // =====================================================================

        // Clement: symmetric tridiag, zero diagonal. Eigenvalues exactly {n−1,…,−(n−1)}; trace = 0.
        // n = 4 ⇒ {3,1,−1,−3}.
        void ClementEig()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;
            var C = arena.doubleClement(n);

            AssertSymmetric(in C, (double)1E-5);

            // trace (diagonal) is exactly 0
            double tr = (double)0;
            for (int i = 0; i < n; i++) tr += C[i, i];
            AssertClose(tr, (double)0, (double)1E-5);

            var Cc = C.Copy();
            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);
            AssertTrue(Eigen.eigenDecomposition(ref Cc, ref eig, ref V));

            double tol = (double)50 * Consts.doubleSqrtEps;
            AssertClose(eig[0], (double)3, tol);
            AssertClose(eig[1], (double)1, tol);
            AssertClose(eig[2], (double)(-1), tol);
            AssertClose(eig[3], (double)(-3), tol);

            arena.Dispose();
        }

        // Fiedler F[i,j]=|i−j|: symmetric; exactly one positive eigenvalue, n−1 negative;
        // det = (−1)^(n−1)(n−1)2^(n−2). n = 5 ⇒ 1 positive / 4 negative, det = 4·8 = 32.
        void FiedlerProps()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            var F = arena.doubleFiedler(n);

            AssertSymmetric(in F, (double)1E-5);
            AssertClose(Determinant(in F), (double)32, (double)0.5);

            var Fc = F.Copy();
            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);
            AssertTrue(Eigen.eigenDecomposition(ref Fc, ref eig, ref V));

            // eigenvalues are bounded away from 0 (smallest |λ| ≈ 0.56), so a small gate is safe.
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

        // DingDong: symmetric Hankel; all eigenvalues in (−π/2, π/2), clustering near ±π/2.
        // The extreme eigenvalues sit ~1e-7 from ±π/2, so bound by π/2 + tol (numerical) rather than strictly.
        void DingDongEig()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;
            var D = arena.doubleDingDong(n);
            AssertSymmetric(in D, (double)1E-5);

            var Dc = D.Copy();
            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);
            AssertTrue(Eigen.eigenDecomposition(ref Dc, ref eig, ref V));

            double halfPi = (double)(math.PI_DBL * 0.5);
            double band = (double)10 * Consts.doubleSqrtEps;   // covers the ~1e-7 boundary margin
            for (int i = 0; i < n; i++)
            {
                AssertTrue(eig[i] <= halfPi + band);
                AssertTrue(eig[i] >= -halfPi - band);
            }
            // clustering near ±π/2: the extreme eigenvalues are within 0.1 of ±π/2
            AssertTrue(eig[0] > halfPi - (double)0.1);
            AssertTrue(eig[n - 1] < -halfPi + (double)0.1);

            arena.Dispose();
        }

        // Frank: upper Hessenberg, det = 1, eigenvalues real + positive. n=3 entries exact; det at n=4;
        // eigenvaluesQR at n=4 returns all-real all-positive.
        void FrankProps()
        {
            var arena = new Arena(Allocator.Persistent);

            // n = 3 exact matrix [[3,2,1],[2,2,1],[0,1,1]]
            var F3 = arena.doubleFrank(3);
            AssertClose(F3[0, 0], (double)3, (double)1E-5); AssertClose(F3[0, 1], (double)2, (double)1E-5); AssertClose(F3[0, 2], (double)1, (double)1E-5);
            AssertClose(F3[1, 0], (double)2, (double)1E-5); AssertClose(F3[1, 1], (double)2, (double)1E-5); AssertClose(F3[1, 2], (double)1, (double)1E-5);
            AssertClose(F3[2, 0], (double)0, (double)1E-5); AssertClose(F3[2, 1], (double)1, (double)1E-5); AssertClose(F3[2, 2], (double)1, (double)1E-5);

            // det = 1 at n = 4
            var F4 = arena.doubleFrank(4);
            AssertClose(Determinant(in F4), (double)1, (double)0.05);

            // eigenvaluesQR: all real (imag ≈ 0) and positive (Frank4 ≈ {7.31, 2.07, 0.48, 0.137})
            var Fc = F4.Copy();
            var re = arena.doubleVec(4);
            var im = arena.doubleVec(4);
            AssertTrue(Eigen.eigenvaluesQR(ref Fc, ref re, ref im));
            for (int i = 0; i < 4; i++)
            {
                AssertClose(im[i], (double)0, (double)1E-2);   // real spectrum
                AssertTrue(re[i] > (double)0);                 // positive
            }

            arena.Dispose();
        }

        // Vandermonde: det = ∏_{i<j}(nodes[j]−nodes[i]). nodes {1,2,3,4} ⇒ 12.
        // A node = 0 still yields an all-ones column 0 (the 0⁰ = 1 path).
        void VandermondeDet()
        {
            var arena = new Arena(Allocator.Persistent);

            var nodes = arena.doubleVec(4);
            nodes[0] = (double)1; nodes[1] = (double)2; nodes[2] = (double)3; nodes[3] = (double)4;
            var V = arena.doubleVandermonde(in nodes);
            AssertClose(Determinant(in V), (double)12, (double)0.2);

            // node 0 ⇒ column 0 is all ones (0^0 = 1)
            var nodes0 = arena.doubleVec(3);
            nodes0[0] = (double)0; nodes0[1] = (double)1; nodes0[2] = (double)2;
            var V0 = arena.doubleVandermonde(in nodes0);
            for (int i = 0; i < 3; i++)
                AssertClose(V0[i, 0], (double)1, (double)1E-6);

            arena.Dispose();
        }

        // Companion of (x−1)(x−2)(x−3) = x³ − 6x² + 11x − 6 ⇒ coeffs {−6, 11, −6} (coeffs[k] = coeff of xᵏ).
        // eigenvaluesQR returns the roots {3,2,1} (descending, real).
        void CompanionEig()
        {
            var arena = new Arena(Allocator.Persistent);

            var coeffs = arena.doubleVec(3);
            coeffs[0] = (double)(-6); coeffs[1] = (double)11; coeffs[2] = (double)(-6);
            var C = arena.doubleCompanion(in coeffs);

            var re = arena.doubleVec(3);
            var im = arena.doubleVec(3);
            AssertTrue(Eigen.eigenvaluesQR(ref C, ref re, ref im));

            double tol = (double)1E-2;
            AssertClose(re[0], (double)3, tol);
            AssertClose(re[1], (double)2, tol);
            AssertClose(re[2], (double)1, tol);
            for (int i = 0; i < 3; i++) AssertClose(im[i], (double)0, tol);

            arena.Dispose();
        }

        // Hadamard: HᵀH = n·I exactly (entries ±1, exact in float for n ≤ 8). n = 4 and n = 8.
        void HadamardOrthogonal()
        {
            CheckHadamard(4);
            CheckHadamard(8);
        }

        void CheckHadamard(int n)
        {
            var arena = new Arena(Allocator.Persistent);

            var H = arena.doubleHadamard(n);

            // HᵀH
            var HtH = arena.doubleMat(n, n);
            doubleOP.dot(in H, in H, ref HtH, transposeA: true);

            double tol = (double)1E-4;   // exact arithmetic; tiny tolerance
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                {
                    double expected = (r == c) ? (double)n : (double)0;
                    AssertClose(HtH[r, c], expected, tol);
                }

            arena.Dispose();
        }

        // Circulant ↔ FFT cross-check. Using a SYMMETRIC first row (c[k]=c[n−k]) makes the spectrum REAL,
        // so eigenvalues and DFT values can be matched unambiguously by sorting the two real spectra.
        // Also: every row sum equals sum(c).
        void CirculantFFT()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;
            var c = arena.doubleVec(n);
            c[0] = (double)2; c[1] = (double)0.5; c[2] = (double)(-0.3); c[3] = (double)0.5;   // symmetric: c[1]==c[3]
            double csum = (double)2 + (double)0.5 - (double)0.3 + (double)0.5;                  // 2.7

            var C = arena.doubleCirculant(in c);

            // every row sums to sum(c)
            for (int i = 0; i < n; i++)
            {
                double rs = (double)0;
                for (int j = 0; j < n; j++) rs += C[i, j];
                AssertClose(rs, csum, (double)50 * Consts.doubleSqrtEps);
            }

            // eigenvalues via QR (matrix destroyed ⇒ copy)
            var Cc = C.Copy();
            var evRe = arena.doubleVec(n);
            var evIm = arena.doubleVec(n);
            AssertTrue(Eigen.eigenvaluesQR(ref Cc, ref evRe, ref evIm));

            // DFT of c via the library FFT (in place ⇒ copy the real part, zero imag)
            var fRe = c.Copy();
            var fIm = arena.doubleVec(n);
            doubleFFT.fft(ref fRe, ref fIm);

            double spectralTol = (double)100 * Consts.doubleSqrtEps;

            // symmetric circulant ⇒ both spectra are real
            for (int i = 0; i < n; i++)
            {
                AssertClose(evIm[i], (double)0, spectralTol);
                AssertClose(fIm[i], (double)0, spectralTol);
            }

            // compare the two real spectra as sorted sets (descending)
            SortDescending(ref evRe);
            SortDescending(ref fRe);
            for (int i = 0; i < n; i++)
                AssertClose(evRe[i], fRe[i], spectralTol);

            arena.Dispose();
        }

        // Triw: upper-triangular, 1-diagonal, α super. det = 1; all eigenvalues = 1. n = 5, α = −2.
        void TriwProps()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            var T = arena.doubleTriw(n, (double)(-2));

            // det = product of unit diagonal = 1 (triangular ⇒ exact)
            AssertClose(Determinant(in T), (double)1, (double)1E-3);

            // eigenvalues: matrix is already upper-triangular (real Schur form) ⇒ diagonal = all 1
            var Tc = T.Copy();
            var re = arena.doubleVec(n);
            var im = arena.doubleVec(n);
            AssertTrue(Eigen.eigenvaluesQR(ref Tc, ref re, ref im));
            for (int i = 0; i < n; i++)
            {
                AssertClose(re[i], (double)1, (double)1E-3);
                AssertClose(im[i], (double)0, (double)1E-3);
            }

            arena.Dispose();
        }

        // WilkinsonPlus (n odd ≥ 3): the two largest eigenvalues form a near-pair. n = 7:
        // reference eig ≈ {3.7616, 3.7321, 2.3633, …} ⇒ top gap ≈ 0.0295, next gap ≈ 1.37.
        void WilkinsonNearPair()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 7;
            var W = arena.doubleWilkinsonPlus(n);
            AssertSymmetric(in W, (double)1E-5);

            var Wc = W.Copy();
            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);
            AssertTrue(Eigen.eigenDecomposition(ref Wc, ref eig, ref V));

            // near-pair: the top two are far closer to each other than to the third eigenvalue.
            double topGap = math.abs(eig[0] - eig[1]);
            double nextGap = eig[1] - eig[2];
            AssertTrue(topGap < (double)0.05);
            AssertTrue(nextGap > (double)1);

            arena.Dispose();
        }

        // Läuchli is rectangular (n+1)×n with row 0 ones and rows 1..n = ε·I.
        void LauchliDims()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 3;
            double eps = (double)1E-3;
            var A = arena.doubleLauchli(n, eps);

            RecordEq(A.M_Rows, n + 1);
            RecordEq(A.N_Cols, n);

            // row 0 all ones
            for (int j = 0; j < n; j++) AssertClose(A[0, j], (double)1, (double)1E-6);
            // rows 1..n form ε·I
            for (int r = 1; r <= n; r++)
                for (int j = 0; j < n; j++)
                    AssertClose(A[r, j], (r - 1 == j) ? eps : (double)0, (double)1E-6);

            arena.Dispose();
        }

        // =====================================================================
        // Algorithm-exercise
        // =====================================================================

        // CG solves Laplacian1D·x = b accurately (SPD, well-conditioned at n = 8).
        void CGLaplacian()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 8;
            var A = arena.doubleLaplacian1D(n);

            var xTrue = arena.doubleVec(n);
            for (int i = 0; i < n; i++) xTrue[i] = (double)(i + 1);   // 1,2,...,n

            var b = doubleOP.dot(A, xTrue);   // consistent RHS

            var x = arena.doubleVec(n);
            bool conv = Solvers.conjugateGradient(in A, in b, ref x, 200, Consts.doubleSqrtEps);
            AssertTrue(conv);

            double tol = (double)100 * Consts.doubleSqrtEps;
            for (int i = 0; i < n; i++)
                AssertClose(x[i], xTrue[i], tol);

            arena.Dispose();
        }

        // =====================================================================
        // helpers
        // =====================================================================

        // det via LU on a copy (luDecompositionInplace destroys its input).
        double Determinant(in doubleMxN M)
        {
            int n = M.M_Rows;
            var LUmat = M.Copy();
            var pivot = new Pivot(n, Allocator.Temp);
            LU.luDecompositionInplace(ref LUmat, ref pivot);
            double det = LU.determinant(in LUmat, in pivot);
            pivot.Dispose();
            return det;
        }

        void AssertSymmetric(in doubleMxN A, double tol)
        {
            int n = A.N_Cols;
            for (int r = 0; r < n; r++)
                for (int c = r + 1; c < n; c++)
                    AssertClose(A[r, c], A[c, r], tol);
        }

        void AssertCholeskyOk(ref Arena arena, in doubleMxN A)
        {
            var L = arena.doubleMat(A.M_Rows, A.N_Cols);
            AssertTrue(Cholesky.choleskyDecomposition(in A, ref L));
        }

        // descending selection sort, in place
        void SortDescending(ref doubleN v)
        {
            int n = v.N;
            for (int i = 0; i < n - 1; i++)
            {
                int best = i;
                for (int j = i + 1; j < n; j++)
                    if (v[j] > v[best]) best = j;
                if (best != i)
                {
                    double t = v[i]; v[i] = v[best]; v[best] = t;
                }
            }
        }

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
    public void GalleryTests(TestJob.TestType type)
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

    // ---------------- Managed argument-validation throws (main thread) ----------------

    // Hadamard requires n to be a power of two.
    [Test]
    public void HadamardNonPowerOfTwoThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            Assert.Throws<ArgumentException>(() => arena.doubleHadamard(3));
            Assert.Throws<ArgumentException>(() => arena.doubleHadamard(0));
            Assert.Throws<ArgumentException>(() => arena.doubleHadamard(6));
        }
        finally { arena.Dispose(); }
    }

    // WilkinsonPlus requires n odd and >= 3.
    [Test]
    public void WilkinsonPlusInvalidNThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            Assert.Throws<ArgumentException>(() => arena.doubleWilkinsonPlus(1));  // < 3
            Assert.Throws<ArgumentException>(() => arena.doubleWilkinsonPlus(2));  // even
            Assert.Throws<ArgumentException>(() => arena.doubleWilkinsonPlus(4));  // even
        }
        finally { arena.Dispose(); }
    }
}
