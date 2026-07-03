using System;
#pragma warning disable 618 // intentionally exercises the deprecated Jacobi svdDecomposition / eigenDecomposition (kept for reference)

using LinearAlgebra;
using LinearAlgebra.Gallery;   // opt-in: arena.fProxyPascal(n), arena.fProxyFrank(n), ...

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
// eigenvaluesQR, FFT.fft). Tolerances are per-precision: they scale with Consts.fProxySqrtEps
// (float ≈ 3.45e-4, double ≈ 1.49e-8) so the SAME expression is loose for float and tight for double,
// matching the LiteratureTests / RandomMatrixTests idiom. Ill-conditioned generators (Hilbert, Frank,
// Pascal at larger n, Moler) use small n and generous multiples; exact-in-float properties (Hadamard
// HᵀH = nI) use a tight tolerance. Reference eigenvalues/determinants were cross-checked offline.
//
// Argument-validation throws run on the managed thread (Assert.Throws), like the sibling guard tests.
public class fProxyGalleryTests
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
        public NativeArray<fProxy> Fail;

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
            var P = arena.fProxyPascal(n);

            AssertSymmetric(in P, (fProxy)1E-5);
            AssertCholeskyOk(ref arena, in P);

            // det = 1; Pascal(5) cond ≈ 8.5e3, so float det error ≈ a few e-3.
            AssertClose(Determinant(in P), (fProxy)1, (fProxy)150 * Consts.fProxySqrtEps);

            arena.Dispose();
        }

        // MinIJ A[i,j]=min(i,j)+1: symmetric, det = 1, SPD. n = 5.
        void MinIJProps()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            var A = arena.fProxyMinIJ(n);

            AssertSymmetric(in A, (fProxy)1E-5);
            AssertCholeskyOk(ref arena, in A);
            AssertClose(Determinant(in A), (fProxy)1, (fProxy)50 * Consts.fProxySqrtEps);

            arena.Dispose();
        }

        // Moler = UᵀU (U = triw 1-diag, α-super): SPD and det = 1 for ALL α.
        // α = −1 (mild, n = 5) for the det assert; α = 2 (n = 4) just for SPD + det with a looser band.
        void MolerProps()
        {
            var arena = new Arena(Allocator.Persistent);

            // α = −1 (default overload), n = 5
            var M1 = arena.fProxyMoler(5);
            AssertSymmetric(in M1, (fProxy)1E-5);
            AssertCholeskyOk(ref arena, in M1);
            AssertClose(Determinant(in M1), (fProxy)1, (fProxy)150 * Consts.fProxySqrtEps);

            // α = 2, n = 4 — still SPD with det = 1 (more ill-conditioned ⇒ looser det band)
            var M2 = arena.fProxyMoler(4, (fProxy)2);
            AssertSymmetric(in M2, (fProxy)1E-5);
            AssertCholeskyOk(ref arena, in M2);
            AssertClose(Determinant(in M2), (fProxy)1, (fProxy)300 * Consts.fProxySqrtEps);

            arena.Dispose();
        }

        // Laplacian1D (Strang 2nd-difference): SPD, det = n+1, eigenvalues 2−2cos(kπ/(n+1)). n = 6.
        void Laplacian1DProps()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;
            var T = arena.fProxyLaplacian1D(n);

            AssertSymmetric(in T, (fProxy)1E-5);
            AssertCholeskyOk(ref arena, in T);

            // det = n + 1 = 7 (read-only path destroys a copy inside Determinant)
            AssertClose(Determinant(in T), (fProxy)(n + 1), (fProxy)0.05);

            // eigenvalues (descending) eig[i] = 2 − 2cos((n−i)π/(n+1))
            var Tc = T.Copy();
            var eig = arena.fProxyVec(n);
            var V = arena.fProxyMat(n, n);
            AssertTrue(Eigen.eigenDecomposition(ref Tc, ref eig, ref V));

            fProxy pi = (fProxy)math.PI_DBL;
            for (int i = 0; i < n; i++)
            {
                fProxy expected = (fProxy)2 - (fProxy)2 * math.cos((fProxy)(n - i) * pi / (fProxy)(n + 1));
                AssertClose(eig[i], expected, (fProxy)50 * Consts.fProxySqrtEps);
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
            var A = arena.fProxyKMS(n, (fProxy)0.5);
            AssertSymmetric(in A, (fProxy)1E-5);
            AssertCholeskyOk(ref arena, in A);

            fProxy expDet = (fProxy)1;
            for (int k = 0; k < n - 1; k++) expDet *= (fProxy)0.75;   // (1 − ρ²)^(n−1)
            AssertClose(Determinant(in A), expDet, (fProxy)50 * Consts.fProxySqrtEps);

            // ρ = −0.5, n = 3: K[0,1] = ρ^1 = −0.5, K[0,2] = ρ^2 = +0.25 (post-fix integer power).
            var B = arena.fProxyKMS(3, (fProxy)(-0.5));
            AssertClose(B[0, 1], (fProxy)(-0.5), (fProxy)1E-5);
            AssertClose(B[0, 2], (fProxy)0.25, (fProxy)1E-5);
            AssertClose(B[1, 2], (fProxy)(-0.5), (fProxy)1E-5);

            arena.Dispose();
        }

        // Pei = αI + J: eigenvalues {α+n (×1), α (×n−1)}; det = αⁿ⁻¹(α+n). α = 2, n = 5 ⇒ {7,2,2,2,2}, det 112.
        void PeiProps()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            fProxy alpha = (fProxy)2;
            var A = arena.fProxyPei(n, alpha);

            AssertSymmetric(in A, (fProxy)1E-5);

            // det = α^(n−1)·(α+n) = 16·7 = 112
            AssertClose(Determinant(in A), (fProxy)112, (fProxy)0.5);

            var Ac = A.Copy();
            var eig = arena.fProxyVec(n);
            var V = arena.fProxyMat(n, n);
            AssertTrue(Eigen.eigenDecomposition(ref Ac, ref eig, ref V));

            fProxy tol = (fProxy)50 * Consts.fProxySqrtEps;
            AssertClose(eig[0], alpha + (fProxy)n, tol);   // α + n = 7
            for (int i = 1; i < n; i++)
                AssertClose(eig[i], alpha, tol);           // α = 2

            arena.Dispose();
        }

        // Lehmer and Hilbert: both SPD at small n ⇒ symmetric and Cholesky succeeds.
        void LehmerHilbertSpd()
        {
            var arena = new Arena(Allocator.Persistent);

            var L = arena.fProxyLehmer(5);
            AssertSymmetric(in L, (fProxy)1E-5);
            AssertCholeskyOk(ref arena, in L);

            var H = arena.fProxyHilbert(4);
            AssertSymmetric(in H, (fProxy)1E-5);
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
            var C = arena.fProxyClement(n);

            AssertSymmetric(in C, (fProxy)1E-5);

            // trace (diagonal) is exactly 0
            fProxy tr = (fProxy)0;
            for (int i = 0; i < n; i++) tr += C[i, i];
            AssertClose(tr, (fProxy)0, (fProxy)1E-5);

            var Cc = C.Copy();
            var eig = arena.fProxyVec(n);
            var V = arena.fProxyMat(n, n);
            AssertTrue(Eigen.eigenDecomposition(ref Cc, ref eig, ref V));

            fProxy tol = (fProxy)50 * Consts.fProxySqrtEps;
            AssertClose(eig[0], (fProxy)3, tol);
            AssertClose(eig[1], (fProxy)1, tol);
            AssertClose(eig[2], (fProxy)(-1), tol);
            AssertClose(eig[3], (fProxy)(-3), tol);

            arena.Dispose();
        }

        // Fiedler F[i,j]=|i−j|: symmetric; exactly one positive eigenvalue, n−1 negative;
        // det = (−1)^(n−1)(n−1)2^(n−2). n = 5 ⇒ 1 positive / 4 negative, det = 4·8 = 32.
        void FiedlerProps()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            var F = arena.fProxyFiedler(n);

            AssertSymmetric(in F, (fProxy)1E-5);
            AssertClose(Determinant(in F), (fProxy)32, (fProxy)0.5);

            var Fc = F.Copy();
            var eig = arena.fProxyVec(n);
            var V = arena.fProxyMat(n, n);
            AssertTrue(Eigen.eigenDecomposition(ref Fc, ref eig, ref V));

            // eigenvalues are bounded away from 0 (smallest |λ| ≈ 0.56), so a small gate is safe.
            fProxy gate = (fProxy)1E-2;
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
            var D = arena.fProxyDingDong(n);
            AssertSymmetric(in D, (fProxy)1E-5);

            var Dc = D.Copy();
            var eig = arena.fProxyVec(n);
            var V = arena.fProxyMat(n, n);
            AssertTrue(Eigen.eigenDecomposition(ref Dc, ref eig, ref V));

            fProxy halfPi = (fProxy)(math.PI_DBL * 0.5);
            fProxy band = (fProxy)10 * Consts.fProxySqrtEps;   // covers the ~1e-7 boundary margin
            for (int i = 0; i < n; i++)
            {
                AssertTrue(eig[i] <= halfPi + band);
                AssertTrue(eig[i] >= -halfPi - band);
            }
            // clustering near ±π/2: the extreme eigenvalues are within 0.1 of ±π/2
            AssertTrue(eig[0] > halfPi - (fProxy)0.1);
            AssertTrue(eig[n - 1] < -halfPi + (fProxy)0.1);

            arena.Dispose();
        }

        // Frank: upper Hessenberg, det = 1, eigenvalues real + positive. n=3 entries exact; det at n=4;
        // eigenvaluesQR at n=4 returns all-real all-positive.
        void FrankProps()
        {
            var arena = new Arena(Allocator.Persistent);

            // n = 3 exact matrix [[3,2,1],[2,2,1],[0,1,1]]
            var F3 = arena.fProxyFrank(3);
            AssertClose(F3[0, 0], (fProxy)3, (fProxy)1E-5); AssertClose(F3[0, 1], (fProxy)2, (fProxy)1E-5); AssertClose(F3[0, 2], (fProxy)1, (fProxy)1E-5);
            AssertClose(F3[1, 0], (fProxy)2, (fProxy)1E-5); AssertClose(F3[1, 1], (fProxy)2, (fProxy)1E-5); AssertClose(F3[1, 2], (fProxy)1, (fProxy)1E-5);
            AssertClose(F3[2, 0], (fProxy)0, (fProxy)1E-5); AssertClose(F3[2, 1], (fProxy)1, (fProxy)1E-5); AssertClose(F3[2, 2], (fProxy)1, (fProxy)1E-5);

            // det = 1 at n = 4
            var F4 = arena.fProxyFrank(4);
            AssertClose(Determinant(in F4), (fProxy)1, (fProxy)0.05);

            // eigenvaluesQR: all real (imag ≈ 0) and positive (Frank4 ≈ {7.31, 2.07, 0.48, 0.137})
            var Fc = F4.Copy();
            var re = arena.fProxyVec(4);
            var im = arena.fProxyVec(4);
            AssertTrue(Eigen.eigenvaluesQR(ref Fc, ref re, ref im));
            for (int i = 0; i < 4; i++)
            {
                AssertClose(im[i], (fProxy)0, (fProxy)1E-2);   // real spectrum
                AssertTrue(re[i] > (fProxy)0);                 // positive
            }

            arena.Dispose();
        }

        // Vandermonde: det = ∏_{i<j}(nodes[j]−nodes[i]). nodes {1,2,3,4} ⇒ 12.
        // A node = 0 still yields an all-ones column 0 (the 0⁰ = 1 path).
        void VandermondeDet()
        {
            var arena = new Arena(Allocator.Persistent);

            var nodes = arena.fProxyVec(4);
            nodes[0] = (fProxy)1; nodes[1] = (fProxy)2; nodes[2] = (fProxy)3; nodes[3] = (fProxy)4;
            var V = arena.fProxyVandermonde(in nodes);
            AssertClose(Determinant(in V), (fProxy)12, (fProxy)0.2);

            // node 0 ⇒ column 0 is all ones (0^0 = 1)
            var nodes0 = arena.fProxyVec(3);
            nodes0[0] = (fProxy)0; nodes0[1] = (fProxy)1; nodes0[2] = (fProxy)2;
            var V0 = arena.fProxyVandermonde(in nodes0);
            for (int i = 0; i < 3; i++)
                AssertClose(V0[i, 0], (fProxy)1, (fProxy)1E-6);

            arena.Dispose();
        }

        // Companion of (x−1)(x−2)(x−3) = x³ − 6x² + 11x − 6 ⇒ coeffs {−6, 11, −6} (coeffs[k] = coeff of xᵏ).
        // eigenvaluesQR returns the roots {3,2,1} (descending, real).
        void CompanionEig()
        {
            var arena = new Arena(Allocator.Persistent);

            var coeffs = arena.fProxyVec(3);
            coeffs[0] = (fProxy)(-6); coeffs[1] = (fProxy)11; coeffs[2] = (fProxy)(-6);
            var C = arena.fProxyCompanion(in coeffs);

            var re = arena.fProxyVec(3);
            var im = arena.fProxyVec(3);
            AssertTrue(Eigen.eigenvaluesQR(ref C, ref re, ref im));

            fProxy tol = (fProxy)1E-2;
            AssertClose(re[0], (fProxy)3, tol);
            AssertClose(re[1], (fProxy)2, tol);
            AssertClose(re[2], (fProxy)1, tol);
            for (int i = 0; i < 3; i++) AssertClose(im[i], (fProxy)0, tol);

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

            var H = arena.fProxyHadamard(n);

            // HᵀH
            var HtH = arena.fProxyMat(n, n);
            Blas.dot(in H, in H, ref HtH, transposeA: true);

            fProxy tol = (fProxy)1E-4;   // exact arithmetic; tiny tolerance
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                {
                    fProxy expected = (r == c) ? (fProxy)n : (fProxy)0;
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
            var c = arena.fProxyVec(n);
            c[0] = (fProxy)2; c[1] = (fProxy)0.5; c[2] = (fProxy)(-0.3); c[3] = (fProxy)0.5;   // symmetric: c[1]==c[3]
            fProxy csum = (fProxy)2 + (fProxy)0.5 - (fProxy)0.3 + (fProxy)0.5;                  // 2.7

            var C = arena.fProxyCirculant(in c);

            // every row sums to sum(c)
            for (int i = 0; i < n; i++)
            {
                fProxy rs = (fProxy)0;
                for (int j = 0; j < n; j++) rs += C[i, j];
                AssertClose(rs, csum, (fProxy)50 * Consts.fProxySqrtEps);
            }

            // eigenvalues via QR (matrix destroyed ⇒ copy)
            var Cc = C.Copy();
            var evRe = arena.fProxyVec(n);
            var evIm = arena.fProxyVec(n);
            AssertTrue(Eigen.eigenvaluesQR(ref Cc, ref evRe, ref evIm));

            // DFT of c via the library FFT (in place ⇒ copy the real part, zero imag)
            var fRe = c.Copy();
            var fIm = arena.fProxyVec(n);
            FFT.fft(ref fRe, ref fIm);

            fProxy spectralTol = (fProxy)100 * Consts.fProxySqrtEps;

            // symmetric circulant ⇒ both spectra are real
            for (int i = 0; i < n; i++)
            {
                AssertClose(evIm[i], (fProxy)0, spectralTol);
                AssertClose(fIm[i], (fProxy)0, spectralTol);
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
            var T = arena.fProxyTriw(n, (fProxy)(-2));

            // det = product of unit diagonal = 1 (triangular ⇒ exact)
            AssertClose(Determinant(in T), (fProxy)1, (fProxy)1E-3);

            // eigenvalues: matrix is already upper-triangular (real Schur form) ⇒ diagonal = all 1
            var Tc = T.Copy();
            var re = arena.fProxyVec(n);
            var im = arena.fProxyVec(n);
            AssertTrue(Eigen.eigenvaluesQR(ref Tc, ref re, ref im));
            for (int i = 0; i < n; i++)
            {
                AssertClose(re[i], (fProxy)1, (fProxy)1E-3);
                AssertClose(im[i], (fProxy)0, (fProxy)1E-3);
            }

            arena.Dispose();
        }

        // WilkinsonPlus (n odd ≥ 3): the two largest eigenvalues form a near-pair. n = 7:
        // reference eig ≈ {3.7616, 3.7321, 2.3633, …} ⇒ top gap ≈ 0.0295, next gap ≈ 1.37.
        void WilkinsonNearPair()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 7;
            var W = arena.fProxyWilkinsonPlus(n);
            AssertSymmetric(in W, (fProxy)1E-5);

            var Wc = W.Copy();
            var eig = arena.fProxyVec(n);
            var V = arena.fProxyMat(n, n);
            AssertTrue(Eigen.eigenDecomposition(ref Wc, ref eig, ref V));

            // near-pair: the top two are far closer to each other than to the third eigenvalue.
            fProxy topGap = math.abs(eig[0] - eig[1]);
            fProxy nextGap = eig[1] - eig[2];
            AssertTrue(topGap < (fProxy)0.05);
            AssertTrue(nextGap > (fProxy)1);

            arena.Dispose();
        }

        // Läuchli is rectangular (n+1)×n with row 0 ones and rows 1..n = ε·I.
        void LauchliDims()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 3;
            fProxy eps = (fProxy)1E-3;
            var A = arena.fProxyLauchli(n, eps);

            RecordEq(A.M_Rows, n + 1);
            RecordEq(A.N_Cols, n);

            // row 0 all ones
            for (int j = 0; j < n; j++) AssertClose(A[0, j], (fProxy)1, (fProxy)1E-6);
            // rows 1..n form ε·I
            for (int r = 1; r <= n; r++)
                for (int j = 0; j < n; j++)
                    AssertClose(A[r, j], (r - 1 == j) ? eps : (fProxy)0, (fProxy)1E-6);

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
            var A = arena.fProxyLaplacian1D(n);

            var xTrue = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) xTrue[i] = (fProxy)(i + 1);   // 1,2,...,n

            var b = Blas.dot(A, xTrue);   // consistent RHS

            var x = arena.fProxyVec(n);
            bool conv = Solvers.cg(in A, in b, ref x, 200, Consts.fProxySqrtEps);
            AssertTrue(conv);

            fProxy tol = (fProxy)100 * Consts.fProxySqrtEps;
            for (int i = 0; i < n; i++)
                AssertClose(x[i], xTrue[i], tol);

            arena.Dispose();
        }

        // =====================================================================
        // helpers
        // =====================================================================

        // det via LU on a copy (luDecompositionInPlace destroys its input).
        fProxy Determinant(in fProxyMxN M)
        {
            int n = M.M_Rows;
            var LUmat = M.Copy();
            var pivot = new Pivot(n, Allocator.Temp);
            LU.luDecompositionInPlace(ref LUmat, ref pivot);
            fProxy det = LU.determinant(in LUmat, in pivot);
            pivot.Dispose();
            return det;
        }

        void AssertSymmetric(in fProxyMxN A, fProxy tol)
        {
            int n = A.N_Cols;
            for (int r = 0; r < n; r++)
                for (int c = r + 1; c < n; c++)
                    AssertClose(A[r, c], A[c, r], tol);
        }

        void AssertCholeskyOk(ref Arena arena, in fProxyMxN A)
        {
            var L = arena.fProxyMat(A.M_Rows, A.N_Cols);
            AssertTrue(Cholesky.choleskyDecomposition(in A, ref L));
        }

        // descending selection sort, in place
        void SortDescending(ref fProxyN v)
        {
            int n = v.N;
            for (int i = 0; i < n - 1; i++)
            {
                int best = i;
                for (int j = i + 1; j < n; j++)
                    if (v[j] > v[best]) best = j;
                if (best != i)
                {
                    fProxy t = v[i]; v[i] = v[best]; v[best] = t;
                }
            }
        }

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
    public void GalleryTests(TestJob.TestType type)
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

    // ---------------- Managed argument-validation throws (main thread) ----------------

    // Hadamard requires n to be a power of two.
    [Test]
    public void HadamardNonPowerOfTwoThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            Assert.Throws<ArgumentException>(() => arena.fProxyHadamard(3));
            Assert.Throws<ArgumentException>(() => arena.fProxyHadamard(0));
            Assert.Throws<ArgumentException>(() => arena.fProxyHadamard(6));
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
            Assert.Throws<ArgumentException>(() => arena.fProxyWilkinsonPlus(1));  // < 3
            Assert.Throws<ArgumentException>(() => arena.fProxyWilkinsonPlus(2));  // even
            Assert.Throws<ArgumentException>(() => arena.fProxyWilkinsonPlus(4));  // even
        }
        finally { arena.Dispose(); }
    }
}
