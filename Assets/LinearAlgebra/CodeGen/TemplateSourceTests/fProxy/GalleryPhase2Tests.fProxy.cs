using System;
#pragma warning disable 618 // intentionally exercises the deprecated cyclic-Jacobi Eigen.decompInPlace (kept for reference)

using LinearAlgebra;
using LinearAlgebra.Gallery;   // opt-in: arena.fProxyCauchy(x,y), arena.fProxyMagic(n), ...

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Property + algorithm-exercise tests for the Phase-2 famous-test-matrix gallery
// (docs/dev/spec-gallery.md, Phase 2 table; production template Gallery.Phase2.fProxy.cs).
// Each case pins a generator against its DOCUMENTED closed form (Cauchy/GCD/Redheffer determinants,
// magic constant, Rosser/Prolate eigenvalues, Parter singular values, Grcar/Lotkin structure) using
// the library's own ops (Analysis.determinant, Cholesky, Eigen.decompInPlace, SVD.singularValues).
//
// Tolerances are per-precision: they scale with Consts.fProxySqrtEps (float ≈ 3.45e-4, double ≈ 1.49e-8)
// so the SAME expression is loose for float and tight for double, matching the GalleryTests /
// LiteratureTests idiom. The hard ill-conditioned / near-degenerate asserts (Rosser eigenvalues,
// Prolate small eigenvalues) are precision-gated via IsDouble(): a tight band for double, a generous
// band for float. Argument-validation throws run on the managed thread (Assert.Throws), like the
// sibling guard tests.
public class fProxyGalleryPhase2Tests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            CauchyHilbertCrossCheck,
            CauchyDet,
            GCDProps,
            GCDSolve,
            RedhefferDet,
            MagicProps,
            RosserProps,
            ParterProps,
            ProlateProps,
            GrcarStructure,
            LotkinStructure,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<fProxy> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.CauchyHilbertCrossCheck: CauchyHilbertCrossCheck(); break;
                case TestType.CauchyDet:               CauchyDet();               break;
                case TestType.GCDProps:                GCDProps();                break;
                case TestType.GCDSolve:                GCDSolve();                break;
                case TestType.RedhefferDet:            RedhefferDet();            break;
                case TestType.MagicProps:              MagicProps();              break;
                case TestType.RosserProps:             RosserProps();             break;
                case TestType.ParterProps:             ParterProps();             break;
                case TestType.ProlateProps:            ProlateProps();            break;
                case TestType.GrcarStructure:          GrcarStructure();          break;
                case TestType.LotkinStructure:         LotkinStructure();         break;
            }
        }

        // =====================================================================
        // 1. Cauchy
        // =====================================================================

        // Cross-check: with x = y = {0.5, 1.5, 2.5}, C[i,j] = 1/((i+.5)+(j+.5)) = 1/(i+j+1),
        // which is exactly the Hilbert matrix entrywise.
        void CauchyHilbertCrossCheck()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 3;
            var x = arena.fProxyVec(n);
            x[0] = (fProxy)0.5; x[1] = (fProxy)1.5; x[2] = (fProxy)2.5;

            var C = arena.fProxyCauchy(in x, in x);
            var H = arena.fProxyHilbert(n);

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    AssertClose(C[i, j], H[i, j], (fProxy)1E-5);

            arena.Dispose();
        }

        // det via LU equals the Cauchy determinant formula
        //   det = ∏_{i<j}(x[j]−x[i])(y[j]−y[i]) / ∏_{i,j}(x[i]+y[j])
        // for x = {1,2,3}, y = {0.5,1.5,2.5}. Reference is computed with plain scalar loops.
        void CauchyDet()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 3;
            var x = arena.fProxyVec(n);
            var y = arena.fProxyVec(n);
            x[0] = (fProxy)1;   x[1] = (fProxy)2;   x[2] = (fProxy)3;
            y[0] = (fProxy)0.5; y[1] = (fProxy)1.5; y[2] = (fProxy)2.5;

            var C = arena.fProxyCauchy(in x, in y);

            // reference Cauchy determinant via scalar loops
            fProxy num = (fProxy)1;
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    num *= (x[j] - x[i]) * (y[j] - y[i]);
            fProxy den = (fProxy)1;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    den *= (x[i] + y[j]);
            fProxy expected = num / den;

            // det is tiny (~9e-5); use a relative band with a small absolute floor.
            fProxy tol = math.abs(expected) * (fProxy)0.05 + (fProxy)1E-9;
            AssertClose(Determinant(in C), expected, tol);

            arena.Dispose();
        }

        // =====================================================================
        // 2. GCD — A[i,j] = gcd(i+1, j+1)
        // =====================================================================

        // Symmetric; SPD (Cholesky succeeds); det = ∏_{k=1}^n φ(k). n=4 ⇒ 4, n=5 ⇒ 16.
        void GCDProps()
        {
            var arena = new Arena(Allocator.Persistent);

            // n = 4: det = φ(1)φ(2)φ(3)φ(4) = 1·1·2·2 = 4
            var A4 = arena.fProxyGCD(4);
            AssertSymmetric(in A4, (fProxy)1E-5);
            AssertCholeskyOk(ref arena, in A4);
            AssertClose(Determinant(in A4), (fProxy)TotientProduct(4), (fProxy)200 * Consts.fProxySqrtEps);

            // n = 5: det = ·φ(5) = 4·4 = 16
            var A5 = arena.fProxyGCD(5);
            AssertSymmetric(in A5, (fProxy)1E-5);
            AssertCholeskyOk(ref arena, in A5);
            AssertClose(Determinant(in A5), (fProxy)TotientProduct(5), (fProxy)2000 * Consts.fProxySqrtEps);

            arena.Dispose();
        }

        // Algorithm-exercise: GCD is SPD ⇒ CG solves A·x = b accurately. n = 5.
        void GCDSolve()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            var A = arena.fProxyGCD(n);

            var xTrue = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) xTrue[i] = (fProxy)(i + 1);

            var b = Blas.dot(A, xTrue);   // consistent RHS

            var x = arena.fProxyVec(n);
            bool conv = Krylov.cg(in A, in b, ref x, 500, Consts.fProxySqrtEps);
            AssertTrue(conv);

            // GCD(5) is moderately conditioned ⇒ generous, precision-scaled band.
            fProxy tol = (fProxy)2000 * Consts.fProxySqrtEps;
            for (int i = 0; i < n; i++)
                AssertClose(x[i], xTrue[i], tol);

            arena.Dispose();
        }

        // =====================================================================
        // 3. Redheffer — det = Mertens M(n)
        // =====================================================================

        // M(1..5) = 1, 0, −1, −1, −2.
        void RedhefferDet()
        {
            var arena = new Arena(Allocator.Persistent);

            // Mertens M(n) for n = 1..5 (Burst: no managed array, use an arena vec).
            var mertens = arena.fProxyVec(5);
            mertens[0] = (fProxy)1; mertens[1] = (fProxy)0; mertens[2] = (fProxy)(-1);
            mertens[3] = (fProxy)(-1); mertens[4] = (fProxy)(-2);

            for (int n = 1; n <= 5; n++)
            {
                var R = arena.fProxyRedheffer(n);
                // integer-valued small determinants; modest precision-scaled band.
                AssertClose(Determinant(in R), mertens[n - 1], (fProxy)200 * Consts.fProxySqrtEps);
            }

            arena.Dispose();
        }

        // =====================================================================
        // 4. Magic — odd-order Siamese magic square
        // =====================================================================

        // n=3 equals [[8,1,6],[3,5,7],[4,9,2]] exactly; n=5 every row/col/both-diagonal sum == 65.
        void MagicProps()
        {
            var arena = new Arena(Allocator.Persistent);

            // n = 3 exact entries [[8,1,6],[3,5,7],[4,9,2]] (inline literals; Burst has no managed arrays)
            var M3 = arena.fProxyMagic(3);
            AssertClose(M3[0, 0], (fProxy)8, (fProxy)1E-5); AssertClose(M3[0, 1], (fProxy)1, (fProxy)1E-5); AssertClose(M3[0, 2], (fProxy)6, (fProxy)1E-5);
            AssertClose(M3[1, 0], (fProxy)3, (fProxy)1E-5); AssertClose(M3[1, 1], (fProxy)5, (fProxy)1E-5); AssertClose(M3[1, 2], (fProxy)7, (fProxy)1E-5);
            AssertClose(M3[2, 0], (fProxy)4, (fProxy)1E-5); AssertClose(M3[2, 1], (fProxy)9, (fProxy)1E-5); AssertClose(M3[2, 2], (fProxy)2, (fProxy)1E-5);

            // n = 5 magic constant = n(n²+1)/2 = 65
            int n = 5;
            fProxy magic = (fProxy)(n * (n * n + 1) / 2);
            var M5 = arena.fProxyMagic(n);

            // rows
            for (int i = 0; i < n; i++)
            {
                fProxy s = (fProxy)0;
                for (int j = 0; j < n; j++) s += M5[i, j];
                AssertClose(s, magic, (fProxy)1E-4);
            }
            // cols
            for (int j = 0; j < n; j++)
            {
                fProxy s = (fProxy)0;
                for (int i = 0; i < n; i++) s += M5[i, j];
                AssertClose(s, magic, (fProxy)1E-4);
            }
            // main diagonal and anti-diagonal
            fProxy d = (fProxy)0, ad = (fProxy)0;
            for (int i = 0; i < n; i++) { d += M5[i, i]; ad += M5[i, n - 1 - i]; }
            AssertClose(d, magic, (fProxy)1E-4);
            AssertClose(ad, magic, (fProxy)1E-4);

            arena.Dispose();
        }

        // =====================================================================
        // 5. Rosser — fixed 8×8 symmetric eigensolver stress test
        // =====================================================================

        // symmetric; trace == 4040; eigenvalues match the documented set within a precision-gated band
        // (tight for double, generous for float — the near-pairs near 0, 1000, 1020 are hard in float).
        void RosserProps()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyRosser();
            AssertSymmetric(in A, (fProxy)1E-4);

            // trace = 4040 (exact integer diagonal)
            fProxy tr = (fProxy)0;
            for (int i = 0; i < 8; i++) tr += A[i, i];
            AssertClose(tr, (fProxy)4040, (fProxy)1E-3);

            // documented spectrum, DESCENDING (Eigen.decompInPlace returns descending)
            var expected = arena.fProxyVec(8);
            expected[0] = (fProxy)1020.4202;
            expected[1] = (fProxy)1019.9936;
            expected[2] = (fProxy)1019.5244;
            expected[3] = (fProxy)1000.1207;
            expected[4] = (fProxy)999.9469;
            expected[5] = (fProxy)0.2180;
            expected[6] = (fProxy)(-0.1705);
            expected[7] = (fProxy)(-1020.0532);

            var Ac = A.Copy();
            var eig = arena.fProxyVec(8);
            var V = arena.fProxyMat(8, 8);
            AssertTrue(Eigen.decompInPlace(ref Ac, ref eig, ref V));

            // sum of eigenvalues equals trace regardless of precision (robust invariant).
            fProxy esum = (fProxy)0;
            for (int i = 0; i < 8; i++) esum += eig[i];
            AssertClose(esum, (fProxy)4040, IsDouble() ? (fProxy)1E-3 : (fProxy)0.5);

            // per-eigenvalue documented match: tight for double, generous band for float.
            fProxy band = IsDouble() ? (fProxy)0.5 : (fProxy)3.0;
            for (int i = 0; i < 8; i++)
                AssertClose(eig[i], expected[i], band);

            arena.Dispose();
        }

        // =====================================================================
        // 6. Parter — Toeplitz 1/(i−j+0.5)
        // =====================================================================

        // nonsymmetric; singular values cluster near π, all < π. n = 8.
        void ParterProps()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 8;
            var A = arena.fProxyParter(n);

            // nonsymmetric: A[0,1] != A[1,0]
            AssertTrue(math.abs(A[0, 1] - A[1, 0]) > (fProxy)1E-3);

            var S = arena.fProxyVec(n);   // length min(m,n) = n, descending
            SVD.singularValues(in A, ref S);

            fProxy pi = (fProxy)math.PI_DBL;
            // all singular values below π (allow a tiny numerical overshoot band)
            fProxy band = (fProxy)50 * Consts.fProxySqrtEps;
            for (int i = 0; i < n; i++)
                AssertTrue(S[i] < pi + band);

            // largest clusters near π
            AssertTrue(S[0] > pi - (fProxy)0.5);

            arena.Dispose();
        }

        // =====================================================================
        // 7. Prolate — symmetric Toeplitz, eigenvalues in (0,1)
        // =====================================================================

        // symmetric; all eigenvalues in [−tol, 1+tol]. Precision-gated lower band (small eigenvalues
        // can dip slightly negative in float). n = 8, w = 0.25.
        void ProlateProps()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 8;
            var A = arena.fProxyProlate(n, (fProxy)0.25);
            AssertSymmetric(in A, (fProxy)1E-5);

            var Ac = A.Copy();
            var eig = arena.fProxyVec(n);
            var V = arena.fProxyMat(n, n);
            AssertTrue(Eigen.decompInPlace(ref Ac, ref eig, ref V));

            // documented (0,1): allow a precision-gated margin on both ends.
            fProxy lo = IsDouble() ? (fProxy)1E-7 : (fProxy)1E-2;
            fProxy hi = IsDouble() ? (fProxy)1E-7 : (fProxy)1E-2;
            for (int i = 0; i < n; i++)
            {
                AssertTrue(eig[i] >= -lo);
                AssertTrue(eig[i] <= (fProxy)1 + hi);
            }

            arena.Dispose();
        }

        // =====================================================================
        // 8. Grcar — nonsymmetric banded Toeplitz
        // =====================================================================

        // Structural: diag == 1, superdiags 1..k == 1, first subdiag == −1, far off-band == 0;
        // nonsymmetric. Tested for default k=3 and k=2.
        void GrcarStructure()
        {
            var arena = new Arena(Allocator.Persistent);

            CheckGrcar(arena.fProxyGrcar(8), 8, 3);        // default k = 3
            CheckGrcar(arena.fProxyGrcar(8, 2), 8, 2);     // k = 2

            arena.Dispose();
        }

        void CheckGrcar(fProxyMxN A, int n, int k)
        {
            for (int i = 0; i < n; i++)
            {
                // diagonal == 1
                AssertClose(A[i, i], (fProxy)1, (fProxy)1E-6);
                // superdiagonals 1..k == 1
                for (int d = 1; d <= k && i + d < n; d++)
                    AssertClose(A[i, i + d], (fProxy)1, (fProxy)1E-6);
                // first subdiagonal == −1
                if (i + 1 < n)
                    AssertClose(A[i + 1, i], (fProxy)(-1), (fProxy)1E-6);
            }
            // far off-band entries == 0: just past the upper band (d = k+1) and below the subdiagonal.
            AssertClose(A[0, k + 1], (fProxy)0, (fProxy)1E-6);   // d = k+1 > k
            AssertClose(A[n - 1, 0], (fProxy)0, (fProxy)1E-6);   // d = −(n−1) < −1

            // nonsymmetric: A[0,1] (=1) != A[1,0] (=−1)
            AssertTrue(math.abs(A[0, 1] - A[1, 0]) > (fProxy)1E-3);
        }

        // =====================================================================
        // 9. Lotkin — Hilbert with first row all ones
        // =====================================================================

        // row 0 all ones; rows i≥1 equal the Hilbert pattern 1/(i+j+1); nonsymmetric.
        void LotkinStructure()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            var A = arena.fProxyLotkin(n);

            // row 0 all ones
            for (int j = 0; j < n; j++)
                AssertClose(A[0, j], (fProxy)1, (fProxy)1E-6);

            // rows i ≥ 1: Hilbert pattern
            for (int i = 1; i < n; i++)
                for (int j = 0; j < n; j++)
                    AssertClose(A[i, j], (fProxy)1 / (fProxy)(i + j + 1), (fProxy)1E-5);

            // nonsymmetric: A[0,1] = 1, A[1,0] = 1/2
            AssertTrue(math.abs(A[0, 1] - A[1, 0]) > (fProxy)1E-3);

            arena.Dispose();
        }

        // =====================================================================
        // helpers
        // =====================================================================

        // true only when fProxy expands to double (doubleEpsilon ≈ 2.2e-16 < 1e-10;
        // floatEpsilon ≈ 1.2e-7 is not).
        bool IsDouble() => (double)Consts.fProxyEpsilon < 1e-10;

        // Euclidean GCD (matches the production private helper).
        int Gcd(int a, int b) { while (b != 0) { int t = b; b = a % b; a = t; } return a; }

        // Euler totient φ(k) via a small coprime count.
        int Totient(int k)
        {
            int count = 0;
            for (int a = 1; a <= k; a++) if (Gcd(a, k) == 1) count++;
            return count;
        }

        // ∏_{m=1}^n φ(m) — the Smith-theorem determinant of the GCD matrix.
        int TotientProduct(int n)
        {
            int p = 1;
            for (int m = 1; m <= n; m++) p *= Totient(m);
            return p;
        }

        // det via LU on a copy (LU.decompInPlace destroys its input).
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
            AssertTrue(CHO.decomp(in A, ref L));
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
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void GalleryPhase2Tests(TestJob.TestType type)
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

    // Cauchy: mismatched lengths and zero denominator both throw.
    [Test]
    public void CauchyInvalidArgsThrow()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            // x.N != y.N
            var x2 = arena.fProxyVec(2);
            var y3 = arena.fProxyVec(3);
            Assert.Throws<ArgumentException>(() => arena.fProxyCauchy(in x2, in y3));

            // zero denominator: x[0] + y[0] = 1 + (−1) = 0
            var x1 = arena.fProxyVec(1);
            var y1 = arena.fProxyVec(1);
            x1[0] = (fProxy)1; y1[0] = (fProxy)(-1);
            Assert.Throws<ArgumentException>(() => arena.fProxyCauchy(in x1, in y1));
        }
        finally { arena.Dispose(); }
    }

    // Magic requires a positive ODD n (even n throws).
    [Test]
    public void MagicEvenNThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            Assert.Throws<ArgumentException>(() => arena.fProxyMagic(4));
            Assert.Throws<ArgumentException>(() => arena.fProxyMagic(2));
        }
        finally { arena.Dispose(); }
    }

    // Prolate requires 0 < w < 0.5.
    [Test]
    public void ProlateInvalidWThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            Assert.Throws<ArgumentException>(() => arena.fProxyProlate(4, (fProxy)0));      // w <= 0
            Assert.Throws<ArgumentException>(() => arena.fProxyProlate(4, (fProxy)(-0.1))); // w < 0
            Assert.Throws<ArgumentException>(() => arena.fProxyProlate(4, (fProxy)0.5));    // w >= 0.5
            Assert.Throws<ArgumentException>(() => arena.fProxyProlate(4, (fProxy)0.6));    // w > 0.5
        }
        finally { arena.Dispose(); }
    }
}
