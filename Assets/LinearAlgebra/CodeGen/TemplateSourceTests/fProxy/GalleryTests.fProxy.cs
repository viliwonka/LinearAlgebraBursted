using System;
#pragma warning disable 618 // intentionally exercises the deprecated cyclic-Jacobi Eigen.decompInPlace (kept for reference)

using BULA;
using BULA.Gallery;   // opt-in: fProxyGallery.fProxyPascal(n), fProxyGallery.fProxyFrank(n), ...

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Property + algorithm-exercise tests for the famous-test-matrix gallery.
// Each case pins a generator against its DOCUMENTED closed form (determinant, eigenvalues, definiteness,
// FFT cross-check) rather than a self-consistency check, then a few cases feed the generators into the
// existing solvers (CG, Eigen.valuesQRInPlace) as honest inputs.
//
// Verification reuses the library's own ops (Analysis.determinant, Cholesky, Eigen.decompInPlace /
// Eigen.valuesQRInPlace, FFT.fft). Tolerances are per-precision: they scale with Consts.fProxySqrtEps
// (float ≈ 3.45e-4, double ≈ 1.49e-8) so the SAME expression is loose for float and tight for double,
// matching the LiteratureTests / RandomMatrixTests idiom. Ill-conditioned generators (Hilbert, Frank,
// Pascal at larger n, Moler) use small n and generous multiples; exact-in-float properties (Hadamard
// HᵀH = nI) use a tight tolerance. Reference eigenvalues/determinants were cross-checked offline.
//
// Argument-validation throws run on the managed thread (Assert.Throws), like the sibling guard tests.
public class fProxyGalleryTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
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
            WilkinsonMinusSignSymmetric,
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
                case TestType.WilkinsonMinusSignSymmetric: WilkinsonMinusSignSymmetric(); break;
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
            int n = 5;
            var P = fProxyGallery.fProxyPascal(n);

            AssertSymmetric(in P, (fProxy)1E-5);
            AssertCholeskyOk(in P);

            // det = 1; Pascal(5) cond ≈ 8.5e3, so float det error ≈ a few e-3.
            AssertClose(Determinant(in P), (fProxy)1, (fProxy)150 * Consts.fProxySqrtEps);
        }

        // MinIJ A[i,j]=min(i,j)+1: symmetric, det = 1, SPD. n = 5.
        void MinIJProps()
        {
            int n = 5;
            var A = fProxyGallery.fProxyMinIJ(n);

            AssertSymmetric(in A, (fProxy)1E-5);
            AssertCholeskyOk(in A);
            AssertClose(Determinant(in A), (fProxy)1, (fProxy)50 * Consts.fProxySqrtEps);
        }

        // Moler = UᵀU (U = triw 1-diag, α-super): SPD and det = 1 for ALL α.
        // α = −1 (mild, n = 5) for the det assert; α = 2 (n = 4) just for SPD + det with a looser band.
        void MolerProps()
        {
            // α = −1 (default overload), n = 5
            var M1 = fProxyGallery.fProxyMoler(5);
            AssertSymmetric(in M1, (fProxy)1E-5);
            AssertCholeskyOk(in M1);
            AssertClose(Determinant(in M1), (fProxy)1, (fProxy)150 * Consts.fProxySqrtEps);

            // α = 2, n = 4 — still SPD with det = 1 (more ill-conditioned ⇒ looser det band)
            var M2 = fProxyGallery.fProxyMoler(4, (fProxy)2);
            AssertSymmetric(in M2, (fProxy)1E-5);
            AssertCholeskyOk(in M2);
            AssertClose(Determinant(in M2), (fProxy)1, (fProxy)300 * Consts.fProxySqrtEps);
        }

        // Laplacian1D (Strang 2nd-difference): SPD, det = n+1, eigenvalues 2−2cos(kπ/(n+1)). n = 6.
        void Laplacian1DProps()
        {
            int n = 6;
            var T = fProxyGallery.fProxyLaplacian1D(n);

            AssertSymmetric(in T, (fProxy)1E-5);
            AssertCholeskyOk(in T);

            // det = n + 1 = 7 (read-only path destroys a copy inside Determinant)
            AssertClose(Determinant(in T), (fProxy)(n + 1), (fProxy)0.05);

            // eigenvalues (descending) eig[i] = 2 − 2cos((n−i)π/(n+1))
            var Tc = new fProxyMxN(in T, Allocator.Temp);
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            AssertTrue(Eigen.decompInPlace(ref Tc, ref eig, ref V));

            fProxy pi = (fProxy)math.PI_DBL;
            for (int i = 0; i < n; i++)
            {
                fProxy expected = (fProxy)2 - (fProxy)2 * math.cos((fProxy)(n - i) * pi / (fProxy)(n + 1));
                AssertClose(eig[i], expected, (fProxy)50 * Consts.fProxySqrtEps);
            }
        }

        // KMS A[i,j]=ρ^|i−j|: det = (1−ρ²)^(n−1) for ρ = 0.5 (SPD); and the integer-power path produces
        // sign-correct negative entries for ρ = −0.5.
        void KMSProps()
        {
            // ρ = 0.5, n = 5: det = (1 − 0.25)^4 = 0.31640625; SPD ⇒ Cholesky succeeds.
            int n = 5;
            var A = fProxyGallery.fProxyKMS(n, (fProxy)0.5);
            AssertSymmetric(in A, (fProxy)1E-5);
            AssertCholeskyOk(in A);

            fProxy expDet = (fProxy)1;
            for (int k = 0; k < n - 1; k++) expDet *= (fProxy)0.75;   // (1 − ρ²)^(n−1)
            AssertClose(Determinant(in A), expDet, (fProxy)50 * Consts.fProxySqrtEps);

            // ρ = −0.5, n = 3: K[0,1] = ρ^1 = −0.5, K[0,2] = ρ^2 = +0.25 (post-fix integer power).
            var B = fProxyGallery.fProxyKMS(3, (fProxy)(-0.5));
            AssertClose(B[0, 1], (fProxy)(-0.5), (fProxy)1E-5);
            AssertClose(B[0, 2], (fProxy)0.25, (fProxy)1E-5);
            AssertClose(B[1, 2], (fProxy)(-0.5), (fProxy)1E-5);
        }

        // Pei = αI + J: eigenvalues {α+n (×1), α (×n−1)}; det = αⁿ⁻¹(α+n). α = 2, n = 5 ⇒ {7,2,2,2,2}, det 112.
        void PeiProps()
        {
            int n = 5;
            fProxy alpha = (fProxy)2;
            var A = fProxyGallery.fProxyPei(n, alpha);

            AssertSymmetric(in A, (fProxy)1E-5);

            // det = α^(n−1)·(α+n) = 16·7 = 112
            AssertClose(Determinant(in A), (fProxy)112, (fProxy)0.5);

            var Ac = new fProxyMxN(in A, Allocator.Temp);
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            AssertTrue(Eigen.decompInPlace(ref Ac, ref eig, ref V));

            fProxy tol = (fProxy)50 * Consts.fProxySqrtEps;
            AssertClose(eig[0], alpha + (fProxy)n, tol);   // α + n = 7
            for (int i = 1; i < n; i++)
                AssertClose(eig[i], alpha, tol);           // α = 2
        }

        // Lehmer and Hilbert: both SPD at small n ⇒ symmetric and Cholesky succeeds.
        void LehmerHilbertSpd()
        {
            var L = fProxyGallery.fProxyLehmer(5);
            AssertSymmetric(in L, (fProxy)1E-5);
            AssertCholeskyOk(in L);

            var H = fProxyGallery.fProxyHilbert(4);
            AssertSymmetric(in H, (fProxy)1E-5);
            AssertCholeskyOk(in H);
        }

        // =====================================================================
        // Batch B
        // =====================================================================

        // Clement: symmetric tridiag, zero diagonal. Eigenvalues exactly {n−1,…,−(n−1)}; trace = 0.
        // n = 4 ⇒ {3,1,−1,−3}.
        void ClementEig()
        {
            int n = 4;
            var C = fProxyGallery.fProxyClement(n);

            AssertSymmetric(in C, (fProxy)1E-5);

            // trace (diagonal) is exactly 0
            fProxy tr = (fProxy)0;
            for (int i = 0; i < n; i++) tr += C[i, i];
            AssertClose(tr, (fProxy)0, (fProxy)1E-5);

            var Cc = new fProxyMxN(in C, Allocator.Temp);
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            AssertTrue(Eigen.decompInPlace(ref Cc, ref eig, ref V));

            fProxy tol = (fProxy)50 * Consts.fProxySqrtEps;
            AssertClose(eig[0], (fProxy)3, tol);
            AssertClose(eig[1], (fProxy)1, tol);
            AssertClose(eig[2], (fProxy)(-1), tol);
            AssertClose(eig[3], (fProxy)(-3), tol);
        }

        // Fiedler F[i,j]=|i−j|: symmetric; exactly one positive eigenvalue, n−1 negative;
        // det = (−1)^(n−1)(n−1)2^(n−2). n = 5 ⇒ 1 positive / 4 negative, det = 4·8 = 32.
        void FiedlerProps()
        {
            int n = 5;
            var F = fProxyGallery.fProxyFiedler(n);

            AssertSymmetric(in F, (fProxy)1E-5);
            AssertClose(Determinant(in F), (fProxy)32, (fProxy)0.5);

            var Fc = new fProxyMxN(in F, Allocator.Temp);
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            AssertTrue(Eigen.decompInPlace(ref Fc, ref eig, ref V));

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
        }

        // DingDong: symmetric Hankel; all eigenvalues in (−π/2, π/2), clustering near ±π/2.
        // The extreme eigenvalues sit ~1e-7 from ±π/2, so bound by π/2 + tol (numerical) rather than strictly.
        void DingDongEig()
        {
            int n = 6;
            var D = fProxyGallery.fProxyDingDong(n);
            AssertSymmetric(in D, (fProxy)1E-5);

            var Dc = new fProxyMxN(in D, Allocator.Temp);
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            AssertTrue(Eigen.decompInPlace(ref Dc, ref eig, ref V));

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
        }

        // Frank: upper Hessenberg, det = 1, eigenvalues real + positive. n=3 entries exact; det at n=4;
        // Eigen.valuesQRInPlace at n=4 returns all-real all-positive.
        void FrankProps()
        {
            // n = 3 exact matrix [[3,2,1],[2,2,1],[0,1,1]]
            var F3 = fProxyGallery.fProxyFrank(3);
            AssertClose(F3[0, 0], (fProxy)3, (fProxy)1E-5); AssertClose(F3[0, 1], (fProxy)2, (fProxy)1E-5); AssertClose(F3[0, 2], (fProxy)1, (fProxy)1E-5);
            AssertClose(F3[1, 0], (fProxy)2, (fProxy)1E-5); AssertClose(F3[1, 1], (fProxy)2, (fProxy)1E-5); AssertClose(F3[1, 2], (fProxy)1, (fProxy)1E-5);
            AssertClose(F3[2, 0], (fProxy)0, (fProxy)1E-5); AssertClose(F3[2, 1], (fProxy)1, (fProxy)1E-5); AssertClose(F3[2, 2], (fProxy)1, (fProxy)1E-5);

            // det = 1 at n = 4
            var F4 = fProxyGallery.fProxyFrank(4);
            AssertClose(Determinant(in F4), (fProxy)1, (fProxy)0.05);

            // Eigen.valuesQRInPlace: all real (imag ≈ 0) and positive (Frank4 ≈ {7.31, 2.07, 0.48, 0.137})
            var Fc = new fProxyMxN(in F4, Allocator.Temp);
            var re = new fProxyN(4, Allocator.Temp);
            var im = new fProxyN(4, Allocator.Temp);
            AssertTrue(Eigen.valuesQRInPlace(ref Fc, ref re, ref im));
            for (int i = 0; i < 4; i++)
            {
                AssertClose(im[i], (fProxy)0, (fProxy)1E-2);   // real spectrum
                AssertTrue(re[i] > (fProxy)0);                 // positive
            }
        }

        // Vandermonde: det = ∏_{i<j}(nodes[j]−nodes[i]). nodes {1,2,3,4} ⇒ 12.
        // A node = 0 still yields an all-ones column 0 (the 0⁰ = 1 path).
        void VandermondeDet()
        {
            var nodes = new fProxyN(4, Allocator.Temp);
            nodes[0] = (fProxy)1; nodes[1] = (fProxy)2; nodes[2] = (fProxy)3; nodes[3] = (fProxy)4;
            var V = fProxyGallery.fProxyVandermonde(in nodes);
            AssertClose(Determinant(in V), (fProxy)12, (fProxy)0.2);

            // node 0 ⇒ column 0 is all ones (0^0 = 1)
            var nodes0 = new fProxyN(3, Allocator.Temp);
            nodes0[0] = (fProxy)0; nodes0[1] = (fProxy)1; nodes0[2] = (fProxy)2;
            var V0 = fProxyGallery.fProxyVandermonde(in nodes0);
            for (int i = 0; i < 3; i++)
                AssertClose(V0[i, 0], (fProxy)1, (fProxy)1E-6);
        }

        // Companion of (x−1)(x−2)(x−3) = x³ − 6x² + 11x − 6 ⇒ coeffs {−6, 11, −6} (coeffs[k] = coeff of xᵏ).
        // Eigen.valuesQRInPlace returns the roots {3,2,1} (descending, real).
        void CompanionEig()
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

        // Hadamard: HᵀH = n·I exactly (entries ±1, exact in float for n ≤ 8). n = 4 and n = 8.
        void HadamardOrthogonal()
        {
            CheckHadamard(4);
            CheckHadamard(8);
        }

        void CheckHadamard(int n)
        {
            var H = fProxyGallery.fProxyHadamard(n);

            // HᵀH
            var HtH = new fProxyMxN(n, n, Allocator.Temp);
            Blas.dot(in H, in H, ref HtH, transposeA: true);

            fProxy tol = (fProxy)1E-4;   // exact arithmetic; tiny tolerance
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                {
                    fProxy expected = (r == c) ? (fProxy)n : (fProxy)0;
                    AssertClose(HtH[r, c], expected, tol);
                }
        }

        // Circulant ↔ FFT cross-check. Using a SYMMETRIC first row (c[k]=c[n−k]) makes the spectrum REAL,
        // so eigenvalues and DFT values can be matched unambiguously by sorting the two real spectra.
        // Also: every row sum equals sum(c).
        void CirculantFFT()
        {
            int n = 4;
            var c = new fProxyN(n, Allocator.Temp);
            c[0] = (fProxy)2; c[1] = (fProxy)0.5; c[2] = (fProxy)(-0.3); c[3] = (fProxy)0.5;   // symmetric: c[1]==c[3]
            fProxy csum = (fProxy)2 + (fProxy)0.5 - (fProxy)0.3 + (fProxy)0.5;                  // 2.7

            var C = fProxyGallery.fProxyCirculant(in c);

            // every row sums to sum(c)
            for (int i = 0; i < n; i++)
            {
                fProxy rs = (fProxy)0;
                for (int j = 0; j < n; j++) rs += C[i, j];
                AssertClose(rs, csum, (fProxy)50 * Consts.fProxySqrtEps);
            }

            // eigenvalues via QR (matrix destroyed ⇒ copy)
            var Cc = new fProxyMxN(in C, Allocator.Temp);
            var evRe = new fProxyN(n, Allocator.Temp);
            var evIm = new fProxyN(n, Allocator.Temp);
            AssertTrue(Eigen.valuesQRInPlace(ref Cc, ref evRe, ref evIm));

            // DFT of c via the library FFT (in place ⇒ copy the real part, zero imag)
            var fRe = new fProxyN(in c, Allocator.Temp);
            var fIm = new fProxyN(n, Allocator.Temp);
            var fftWs = new fProxyFFTCache(n, Allocator.Temp);
            FFT.fft(ref fRe, ref fIm, in fftWs);

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
        }

        // Triw: upper-triangular, 1-diagonal, α super. det = 1; all eigenvalues = 1. n = 5, α = −2.
        void TriwProps()
        {
            int n = 5;
            var T = fProxyGallery.fProxyTriw(n, (fProxy)(-2));

            // det = product of unit diagonal = 1 (triangular ⇒ exact)
            AssertClose(Determinant(in T), (fProxy)1, (fProxy)1E-3);

            // eigenvalues: matrix is already upper-triangular (real Schur form) ⇒ diagonal = all 1
            var Tc = new fProxyMxN(in T, Allocator.Temp);
            var re = new fProxyN(n, Allocator.Temp);
            var im = new fProxyN(n, Allocator.Temp);
            AssertTrue(Eigen.valuesQRInPlace(ref Tc, ref re, ref im));
            for (int i = 0; i < n; i++)
            {
                AssertClose(re[i], (fProxy)1, (fProxy)1E-3);
                AssertClose(im[i], (fProxy)0, (fProxy)1E-3);
            }
        }

        // WilkinsonPlus (n odd ≥ 3): the two largest eigenvalues form a near-pair. n = 7:
        // reference eig ≈ {3.7616, 3.7321, 2.3633, …} ⇒ top gap ≈ 0.0295, next gap ≈ 1.37.
        void WilkinsonNearPair()
        {
            int n = 7;
            var W = fProxyGallery.fProxyWilkinsonPlus(n);
            AssertSymmetric(in W, (fProxy)1E-5);

            var Wc = new fProxyMxN(in W, Allocator.Temp);
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            AssertTrue(Eigen.decompInPlace(ref Wc, ref eig, ref V));

            // near-pair: the top two are far closer to each other than to the third eigenvalue.
            fProxy topGap = math.abs(eig[0] - eig[1]);
            fProxy nextGap = eig[1] - eig[2];
            AssertTrue(topGap < (fProxy)0.05);
            AssertTrue(nextGap > (fProxy)1);
        }

        // WilkinsonMinus (n odd ≥ 3): spectrum symmetric about zero with an exact zero eigenvalue.
        // Pinned two ways -- the closed form at n = 3 ({−√3, 0, √3}), and the ± pairing at n = 7.
        void WilkinsonMinusSignSymmetric()
        {
            var W3 = fProxyGallery.fProxyWilkinsonMinus(3);
            AssertSymmetric(in W3, (fProxy)1E-5);

            var W3c = new fProxyMxN(in W3, Allocator.Temp);
            var eig3 = new fProxyN(3, Allocator.Temp);
            var V3 = new fProxyMxN(3, 3, Allocator.Temp);
            AssertTrue(Eigen.decompInPlace(ref W3c, ref eig3, ref V3));   // DESCENDING

            fProxy root3 = (fProxy)math.sqrt(3.0);
            AssertClose(eig3[0], root3, (fProxy)1E-4);
            AssertClose(eig3[1], (fProxy)0, (fProxy)1E-4);
            AssertClose(eig3[2], -root3, (fProxy)1E-4);

            // n = 7: eigenvalue j pairs with eigenvalue (n-1-j) under negation, and the middle one
            // is the exact zero forced by odd n.
            int n = 7;
            var W = fProxyGallery.fProxyWilkinsonMinus(n);
            var Wc = new fProxyMxN(in W, Allocator.Temp);
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            AssertTrue(Eigen.decompInPlace(ref Wc, ref eig, ref V));

            for (int j = 0; j < n / 2; j++)
                AssertClose(eig[j], -eig[n - 1 - j], (fProxy)1E-3);
            AssertClose(eig[n / 2], (fProxy)0, (fProxy)1E-3);
        }

        // Läuchli is rectangular (n+1)×n with row 0 ones and rows 1..n = ε·I.
        void LauchliDims()
        {
            int n = 3;
            fProxy eps = (fProxy)1E-3;
            var A = fProxyGallery.fProxyLauchli(n, eps);

            RecordEq(A.M_Rows, n + 1);
            RecordEq(A.N_Cols, n);

            // row 0 all ones
            for (int j = 0; j < n; j++) AssertClose(A[0, j], (fProxy)1, (fProxy)1E-6);
            // rows 1..n form ε·I
            for (int r = 1; r <= n; r++)
                for (int j = 0; j < n; j++)
                    AssertClose(A[r, j], (r - 1 == j) ? eps : (fProxy)0, (fProxy)1E-6);
        }

        // =====================================================================
        // Algorithm-exercise
        // =====================================================================

        // CG solves Laplacian1D·x = b accurately (SPD, well-conditioned at n = 8).
        void CGLaplacian()
        {
            int n = 8;
            var A = fProxyGallery.fProxyLaplacian1D(n);

            var xTrue = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) xTrue[i] = (fProxy)(i + 1);   // 1,2,...,n

            var b = Blas.dot(A, xTrue);   // consistent RHS

            var x = new fProxyN(n, Allocator.Temp);
            bool conv = Krylov.cg(in A, in b, ref x, 200, Consts.fProxySqrtEps);
            AssertTrue(conv);

            fProxy tol = (fProxy)100 * Consts.fProxySqrtEps;
            for (int i = 0; i < n; i++)
                AssertClose(x[i], xTrue[i], tol);
        }

        // =====================================================================
        // helpers
        // =====================================================================

        // det via LU on a copy (LU.decompInPlace destroys its input).
        fProxy Determinant(in fProxyMxN M)
        {
            int n = M.M_Rows;
            var LUmat = new fProxyMxN(in M, Allocator.Temp);
            var pivot = new Pivot(n, Allocator.Temp);
            LU.decompInPlace(ref LUmat, ref pivot);
            fProxy det = Analysis.determinant(in LUmat, in pivot);
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

        void AssertCholeskyOk(in fProxyMxN A)
        {
            var L = new fProxyMxN(A.M_Rows, A.N_Cols, Allocator.Temp);
            AssertTrue(CHO.decomp(in A, ref L));
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
        Assert.Throws<ArgumentException>(() => fProxyGallery.fProxyHadamard(3));
        Assert.Throws<ArgumentException>(() => fProxyGallery.fProxyHadamard(0));
        Assert.Throws<ArgumentException>(() => fProxyGallery.fProxyHadamard(6));
    }

    // WilkinsonPlus requires n odd and >= 3.
    [Test]
    public void WilkinsonPlusInvalidNThrows()
    {
        Assert.Throws<ArgumentException>(() => fProxyGallery.fProxyWilkinsonPlus(1));  // < 3
        Assert.Throws<ArgumentException>(() => fProxyGallery.fProxyWilkinsonPlus(2));  // even
        Assert.Throws<ArgumentException>(() => fProxyGallery.fProxyWilkinsonPlus(4));  // even
    }

    // WilkinsonMinus requires n odd and >= 3.
    [Test]
    public void WilkinsonMinusInvalidNThrows()
    {
        Assert.Throws<ArgumentException>(() => fProxyGallery.fProxyWilkinsonMinus(1));  // < 3
        Assert.Throws<ArgumentException>(() => fProxyGallery.fProxyWilkinsonMinus(2));  // even
        Assert.Throws<ArgumentException>(() => fProxyGallery.fProxyWilkinsonMinus(4));  // even
    }
}
