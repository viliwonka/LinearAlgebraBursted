using System;

using LinearAlgebra;
using LinearAlgebra.Gallery;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Tests for the general (non-symmetric) QR eigenvalue algorithm: Eigen.eigenvaluesQR
// (Hessenberg reduction + Francis double-shift QR -> real Schur form; eigenvalues as real/imag
// component arrays, sorted by (real, then imag) descending).
//
// Literature test vectors (matrices with KNOWN eigenvalues):
//  - Diagonal / upper-triangular: eigenvalues are the diagonal entries.
//  - Companion matrix of (x-1)(x-2)(x-3)(x-4): eigenvalues are the roots 1,2,3,4.
//  - [[0,-1],[1,0]]: eigenvalues ±i (pure imaginary — the canonical complex-pair case that
//    power iteration and Jacobi cannot find).
//  - 2x2 rotation by θ: eigenvalues cosθ ± i·sinθ.
//  - Block diag(2, rotation): a real eigenvalue alongside a complex-conjugate pair.
//  - Random symmetric: cross-checked against eigenDecomposition (Jacobi); all eigenvalues real.
//  - Random general: sum of eigenvalues == trace, imaginary parts sum to 0 (conjugate pairs cancel).
public class fProxyEigenQRTests
{
    [BurstCompile]
    public struct AssemblyTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomMatrix(6, 6, -2f, 2f, 12345);
            var re = arena.fProxyVec(6);
            var im = arena.fProxyVec(6);
            Eigen.eigenvaluesQR(ref A, ref re, ref im);
            arena.Dispose();
        }
    }

    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            DiagonalReal,
            UpperTriangular,
            CompanionKnownRoots,
            PureImaginaryPair,
            RotationComplexPair,
            RealPlusComplexBlock,
            SymmetricCrossCheckJacobi,
            TraceInvariant,
            NilpotentJordan,
            FrankRealPositive,
            CompanionGalleryRoots,
        }

        public TestType Type;

        // [0] flag, [1] got, [2] expected/limit, [3] diff/index
        public NativeArray<fProxy> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.DiagonalReal:              DiagonalReal();              break;
                case TestType.UpperTriangular:           UpperTriangular();           break;
                case TestType.CompanionKnownRoots:       CompanionKnownRoots();       break;
                case TestType.PureImaginaryPair:         PureImaginaryPair();         break;
                case TestType.RotationComplexPair:       RotationComplexPair();       break;
                case TestType.RealPlusComplexBlock:      RealPlusComplexBlock();      break;
                case TestType.SymmetricCrossCheckJacobi: SymmetricCrossCheckJacobi(); break;
                case TestType.TraceInvariant:            TraceInvariant();            break;
                case TestType.NilpotentJordan:           NilpotentJordan();           break;
                case TestType.FrankRealPositive:         FrankRealPositive();         break;
                case TestType.CompanionGalleryRoots:     CompanionGalleryRoots();     break;
            }
        }

        // diag(5,3,-2,0.5) -> eigenvalues {5,3,0.5,-2} (sorted desc), all imag 0.
        void DiagonalReal()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;
            var A = arena.fProxyMat(n, n);
            A[0, 0] = 5f; A[1, 1] = 3f; A[2, 2] = -2f; A[3, 3] = 0.5f;

            var re = arena.fProxyVec(n);
            var im = arena.fProxyVec(n);
            bool ok = Eigen.eigenvaluesQR(ref A, ref re, ref im);
            RecordEq(ok ? 1 : 0, 1);

            fProxy tol = (fProxy)1E-4f;
            AssertClose(re[0], 5f, tol); AssertClose(re[1], 3f, tol);
            AssertClose(re[2], 0.5f, tol); AssertClose(re[3], -2f, tol);
            for (int i = 0; i < n; i++) AssertClose(im[i], 0f, tol);

            arena.Dispose();
        }

        // Upper triangular: eigenvalues are the diagonal entries {4,3,2} regardless of the
        // (nonzero) superdiagonal content.
        void UpperTriangular()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 3;
            var A = arena.fProxyMat(n, n);
            A[0, 0] = 2f; A[0, 1] = 7f; A[0, 2] = -3f;
            A[1, 1] = 4f; A[1, 2] = 5f;
            A[2, 2] = 3f;

            var re = arena.fProxyVec(n);
            var im = arena.fProxyVec(n);
            bool ok = Eigen.eigenvaluesQR(ref A, ref re, ref im);
            RecordEq(ok ? 1 : 0, 1);

            fProxy tol = (fProxy)1E-3f;
            AssertClose(re[0], 4f, tol); AssertClose(re[1], 3f, tol); AssertClose(re[2], 2f, tol);
            for (int i = 0; i < n; i++) AssertClose(im[i], 0f, tol);

            arena.Dispose();
        }

        // Companion matrix of (x-1)(x-2)(x-3)(x-4) = x^4 -10x^3 +35x^2 -50x +24.
        // Last column holds -c_i; eigenvalues are the roots {4,3,2,1}.
        void CompanionKnownRoots()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;
            var A = arena.fProxyMat(n, n);
            // subdiagonal ones
            A[1, 0] = 1f; A[2, 1] = 1f; A[3, 2] = 1f;
            // last column: -c0,-c1,-c2,-c3 = -24, 50, -35, 10
            A[0, 3] = -24f; A[1, 3] = 50f; A[2, 3] = -35f; A[3, 3] = 10f;

            var re = arena.fProxyVec(n);
            var im = arena.fProxyVec(n);
            bool ok = Eigen.eigenvaluesQR(ref A, ref re, ref im);
            RecordEq(ok ? 1 : 0, 1);

            // companion eigenproblems are mildly stiff; scale-relative tolerance.
            fProxy tol = (fProxy)1E-2f;
            AssertClose(re[0], 4f, tol); AssertClose(re[1], 3f, tol);
            AssertClose(re[2], 2f, tol); AssertClose(re[3], 1f, tol);
            for (int i = 0; i < n; i++) AssertClose(im[i], 0f, tol);

            arena.Dispose();
        }

        // [[0,-1],[1,0]]: eigenvalues are 0 ± i. Sorted desc by (real, imag) => (0,+1) then (0,-1).
        void PureImaginaryPair()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 2;
            var A = arena.fProxyMat(n, n);
            A[0, 0] = 0f; A[0, 1] = -1f;
            A[1, 0] = 1f; A[1, 1] = 0f;

            var re = arena.fProxyVec(n);
            var im = arena.fProxyVec(n);
            bool ok = Eigen.eigenvaluesQR(ref A, ref re, ref im);
            RecordEq(ok ? 1 : 0, 1);

            fProxy tol = (fProxy)1E-5f;
            AssertClose(re[0], 0f, tol); AssertClose(re[1], 0f, tol);
            AssertClose(im[0], 1f, tol);   // +i first
            AssertClose(im[1], -1f, tol);  // -i second

            arena.Dispose();
        }

        // Rotation by θ=π/4: eigenvalues cos θ ± i sin θ = 0.70710678 ± 0.70710678 i.
        void RotationComplexPair()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 2;
            fProxy c = (fProxy)math.cos(math.PI_DBL / 4.0);
            fProxy s = (fProxy)math.sin(math.PI_DBL / 4.0);

            var A = arena.fProxyMat(n, n);
            A[0, 0] = c; A[0, 1] = -s;
            A[1, 0] = s; A[1, 1] = c;

            var re = arena.fProxyVec(n);
            var im = arena.fProxyVec(n);
            bool ok = Eigen.eigenvaluesQR(ref A, ref re, ref im);
            RecordEq(ok ? 1 : 0, 1);

            fProxy tol = (fProxy)1E-4f;
            AssertClose(re[0], c, tol); AssertClose(re[1], c, tol);
            AssertClose(im[0], s, tol);    // +sin first
            AssertClose(im[1], -s, tol);

            arena.Dispose();
        }

        // Block diag( [2], [[0,-1],[1,0]] ): a real eigenvalue 2 alongside the pair 0 ± i.
        void RealPlusComplexBlock()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 3;
            var A = arena.fProxyMat(n, n);
            A[0, 0] = 2f;
            A[1, 1] = 0f; A[1, 2] = -1f;
            A[2, 1] = 1f; A[2, 2] = 0f;

            var re = arena.fProxyVec(n);
            var im = arena.fProxyVec(n);
            bool ok = Eigen.eigenvaluesQR(ref A, ref re, ref im);
            RecordEq(ok ? 1 : 0, 1);

            fProxy tol = (fProxy)1E-4f;
            // sorted desc by real: 2 (real) first, then 0+i, then 0-i.
            AssertClose(re[0], 2f, tol); AssertClose(im[0], 0f, tol);
            AssertClose(re[1], 0f, tol); AssertClose(im[1], 1f, tol);
            AssertClose(re[2], 0f, tol); AssertClose(im[2], -1f, tol);

            arena.Dispose();
        }

        // Random symmetric matrix: all eigenvalues real; cross-check eigenvaluesQR against the
        // symmetric Jacobi eigenDecomposition (both sorted descending by value).
        void SymmetricCrossCheckJacobi()
        {
            var arena = new Arena(Allocator.Persistent);

            for (uint t = 0; t < 8; t++)
            {
                int n = 6;
                var A = arena.fProxyRandomMatrix(n, n, -3f, 3f, 30000 + t * 13);
                for (int i = 0; i < n; i++)
                    for (int j = i + 1; j < n; j++)
                    {
                        fProxy avg = (A[i, j] + A[j, i]) * (fProxy)0.5;
                        A[i, j] = avg; A[j, i] = avg;
                    }

                var Aqr = A.Copy();
                var Ajac = A.Copy();

                var re = arena.fProxyVec(n);
                var im = arena.fProxyVec(n);
                bool ok = Eigen.eigenvaluesQR(ref Aqr, ref re, ref im);
                RecordEq(ok ? 1 : 0, 1);

                var jac = arena.fProxyVec(n);
                var V = arena.fProxyMat(n, n);
                Eigen.eigenDecomposition(ref Ajac, ref jac, ref V); // sorts desc by value

                fProxy tol = (fProxy)1E-3f;
                for (int i = 0; i < n; i++)
                {
                    AssertClose(im[i], 0f, tol);                 // symmetric => real spectrum
                    fProxy scale = (fProxy)1 + math.abs(jac[i]);
                    AssertClose(re[i], jac[i], tol * scale);     // matches Jacobi, same order
                }

                arena.Clear();
            }

            arena.Dispose();
        }

        // Random general (non-symmetric) matrix: sum of eigenvalue real parts == trace(A), and the
        // imaginary parts sum to 0 (complex eigenvalues occur in conjugate pairs).
        void TraceInvariant()
        {
            var arena = new Arena(Allocator.Persistent);

            for (uint t = 0; t < 12; t++)
            {
                int n = 7;
                var A = arena.fProxyRandomMatrix(n, n, -2f, 2f, 40000 + t * 17);
                fProxy trace = 0;
                for (int i = 0; i < n; i++) trace += A[i, i];

                var re = arena.fProxyVec(n);
                var im = arena.fProxyVec(n);
                bool ok = Eigen.eigenvaluesQR(ref A, ref re, ref im);
                RecordEq(ok ? 1 : 0, 1);

                fProxy sumRe = 0, sumIm = 0;
                for (int i = 0; i < n; i++) { sumRe += re[i]; sumIm += im[i]; }

                fProxy tol = (fProxy)1E-3f * ((fProxy)1 + math.abs(trace));
                AssertClose(sumRe, trace, tol);
                AssertClose(sumIm, 0f, tol);

                arena.Clear();
            }

            arena.Dispose();
        }

        // Defective (non-diagonalizable) NILPOTENT Jordan blocks: a 3x3 [[0,1,0],[0,0,1],[0,0,0]]
        // and a 2x2 [[0,1],[0,0]] each have 0 as their ONLY eigenvalue (full algebraic multiplicity,
        // deficient geometric multiplicity). A degenerate case for the QR iteration — all eigenvalues
        // must come out 0 (real and imaginary) with no NaN. Exercises the shift-finder path where the
        // p,q,r normalization can hit s2 == 0.
        void NilpotentJordan()
        {
            var arena = new Arena(Allocator.Persistent);

            // 3x3 Jordan block.
            {
                int n = 3;
                var A = arena.fProxyMat(n, n);
                A[0, 1] = 1f; A[1, 2] = 1f; // ones on the superdiagonal, all else zero

                var re = arena.fProxyVec(n);
                var im = arena.fProxyVec(n);
                bool ok = Eigen.eigenvaluesQR(ref A, ref re, ref im);
                RecordEq(ok ? 1 : 0, 1);
                for (int i = 0; i < n; i++)
                {
                    AssertClose(re[i], 0f, (fProxy)1E-5f);
                    AssertClose(im[i], 0f, (fProxy)1E-5f);
                }
            }

            // 2x2 Jordan block.
            {
                int n = 2;
                var A = arena.fProxyMat(n, n);
                A[0, 1] = 1f;

                var re = arena.fProxyVec(n);
                var im = arena.fProxyVec(n);
                bool ok = Eigen.eigenvaluesQR(ref A, ref re, ref im);
                RecordEq(ok ? 1 : 0, 1);
                for (int i = 0; i < n; i++)
                {
                    AssertClose(re[i], 0f, (fProxy)1E-5f);
                    AssertClose(im[i], 0f, (fProxy)1E-5f);
                }
            }

            arena.Dispose();
        }

        // GALLERY KNOWN-ANSWER (Gallery.Special): n=5 Frank matrix — upper Hessenberg, det=1.
        // Known property: ALL eigenvalues are real and positive (and come in reciprocal pairs).
        // For n=5 the spectrum is {10.063, 3.557, 1.0, 0.281, 0.0994}: well separated, smallest
        // ~0.0994, so a 1E-2 positivity band is robust under float QR. Also checks sum(re)==trace.
        void FrankRealPositive()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            var A = arena.fProxyFrank(n);

            // trace must be read before eigenvaluesQR destroys A.
            fProxy trace = 0;
            for (int i = 0; i < n; i++) trace += A[i, i];

            var re = arena.fProxyVec(n);
            var im = arena.fProxyVec(n);
            bool ok = Eigen.eigenvaluesQR(ref A, ref re, ref im);
            RecordEq(ok ? 1 : 0, 1);

            fProxy tol = (fProxy)1E-2f;
            fProxy posTol = (fProxy)1E-2f;
            fProxy sumRe = 0;
            for (int i = 0; i < n; i++)
            {
                AssertClose(im[i], 0f, tol);     // real spectrum
                sumRe += re[i];

                bool pos = re[i] > posTol;       // positive spectrum
                if (!pos && Fail[0] == (fProxy)0)
                {
                    Fail[0] = (fProxy)1;
                    Fail[1] = re[i];
                    Fail[2] = posTol;
                    Fail[3] = (fProxy)i;
                }
                Assert.IsTrue(pos);
            }

            AssertClose(sumRe, trace, (fProxy)1E-2f * ((fProxy)1 + math.abs(trace)));

            arena.Dispose();
        }

        // GALLERY KNOWN-ANSWER (Gallery.Special): companion matrix of the monic polynomial
        // (x-1)(x-2)(x-3)(x-4) = x^4 - 10x^3 + 35x^2 - 50x + 24, built via the gallery generator
        // from coeffs {24,-50,35,-10}. Eigenvalues equal the roots {1,2,3,4}; sorted desc -> {4,3,2,1}.
        void CompanionGalleryRoots()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;
            var coeffs = arena.fProxyVec(n);
            coeffs[0] = 24f; coeffs[1] = -50f; coeffs[2] = 35f; coeffs[3] = -10f;

            var A = arena.fProxyCompanion(in coeffs);

            var re = arena.fProxyVec(n);
            var im = arena.fProxyVec(n);
            bool ok = Eigen.eigenvaluesQR(ref A, ref re, ref im);
            RecordEq(ok ? 1 : 0, 1);

            // companion eigenproblems are mildly stiff; scale-relative tolerance.
            fProxy tol = (fProxy)1E-2f;
            AssertClose(re[0], 4f, tol); AssertClose(re[1], 3f, tol);
            AssertClose(re[2], 2f, tol); AssertClose(re[3], 1f, tol);
            for (int i = 0; i < n; i++) AssertClose(im[i], 0f, tol);

            arena.Dispose();
        }

        void AssertClose(fProxy got, fProxy expected, fProxy tol)
        {
            fProxy d = math.abs(got - expected);
            if (!(d <= tol) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = got;
                Fail[2] = expected;
                Fail[3] = d;
            }
            Assert.IsTrue(d <= tol);
        }

        void RecordEq(int got, int expected)
        {
            if (got != expected && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = got;
                Fail[2] = expected;
                Fail[3] = got - expected;
            }
            Assert.AreEqual(expected, got);
        }
    }

    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void EigenQRTests(TestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }
}
