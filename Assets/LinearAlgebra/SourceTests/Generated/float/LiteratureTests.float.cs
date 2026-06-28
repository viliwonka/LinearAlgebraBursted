using System;
#pragma warning disable 618 // intentionally exercises the deprecated Jacobi svdDecomposition (kept for reference)

using LinearAlgebra;
using LinearAlgebra.Gallery;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Literature test vectors with KNOWN closed-form results / documented failure modes.
// See memory note literature-test-vectors. Each case pins an algorithm against an independent
// reference value rather than a self-consistency check.
public class floatLiteratureTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            Laplacian1D,
            QRScaleInvariance,
            IndefiniteCholeskyFails,
            PascalDetAndCholesky,
            HilbertCond,
            WilkinsonEigen,
            LauchliLeastSquares,
            VandermondeDet,
            NonsymmetricSVD,
        }

        public TestType Type;

        // [0] flag, [1] got, [2] expected/limit, [3] diff
        public NativeArray<float> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.Laplacian1D:             Laplacian1D(); break;
                case TestType.QRScaleInvariance:       QRScaleInvariance(); break;
                case TestType.IndefiniteCholeskyFails: IndefiniteCholeskyFails(); break;
                case TestType.PascalDetAndCholesky:    PascalDetAndCholesky(); break;
                case TestType.HilbertCond:             HilbertCond(); break;
                case TestType.WilkinsonEigen:          WilkinsonEigen(); break;
                case TestType.LauchliLeastSquares:     LauchliLeastSquares(); break;
                case TestType.VandermondeDet:          VandermondeDet(); break;
                case TestType.NonsymmetricSVD:         NonsymmetricSVD(); break;
            }
        }

        // Vandermonde V[i,j] = x_i^j, nodes x = [1,2,3,4]. Known: det(V) = Π_{i<j}(x_j − x_i)
        // = (1)(2)(3)(1)(2)(1) = 12. Tests LU determinant on a non-trivial (ill-conditioned) matrix.
        void VandermondeDet()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;
            var nodes = arena.floatVec(n);
            for (int i = 0; i < n; i++) nodes[i] = (float)(i + 1);   // nodes 1,2,3,4
            var V = arena.floatVandermonde(in nodes);

            var LUmat = V.Copy();
            var pivot = new Pivot(n, Allocator.Temp);
            LU.luDecompositionInplace(ref LUmat, ref pivot);
            float det = LU.determinant(in LUmat, in pivot);
            pivot.Dispose();

            AssertClose(det, (float)12, (float)1E-1);

            arena.Dispose();
        }

        // Non-symmetric A = [[0,2],[-1,0]]: eigenvalues are ±i√2 (complex), but singular values are
        // REAL and known — AᵀA = diag(1,4) → σ = {2,1}. Confirms the SVD path computes singular
        // values (not |eigenvalues|): σ_max=2, σ_min=1, ‖A‖₂=2, cond=2.
        void NonsymmetricSVD()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.floatMat(2, 2);
            A[0, 0] = (float)0; A[0, 1] = (float)2;
            A[1, 0] = (float)(-1); A[1, 1] = (float)0;

            var S = arena.floatVec(2);
            SVD.singularValues(in A, ref S);   // descending
            AssertClose(S[0], (float)2, (float)1E-4);
            AssertClose(S[1], (float)1, (float)1E-4);

            AssertClose(floatNormsOP.matrixL2(in A), (float)2, (float)1E-4);
            AssertClose(floatOP.cond(in A), (float)2, (float)1E-4);

            arena.Dispose();
        }

        // Läuchli matrix A = [[1,1,1],[ε,0,0],[0,ε,0],[0,0,ε]] (4x3): columns are barely independent
        // (cond ≈ 1/ε), the classic loss-of-orthogonality least-squares stress case. With a consistent
        // RHS b = A·x_true, both the SVD pseudo-inverse and QR must recover x_true accurately — a naive
        // normal-equations solve (cond ≈ 1/ε²) would lose ~6 digits.
        void LauchliLeastSquares()
        {
            var arena = new Arena(Allocator.Persistent);

            float eps = (float)1E-3;
            var xTrue = arena.floatVec(3);
            xTrue[0] = (float)1; xTrue[1] = (float)2; xTrue[2] = (float)3;

            // --- SVD pseudo-inverse solve (pinvSolve no longer modifies A or b) ---
            var A1 = arena.floatLauchli(3, eps);   // (3+1)x3 = 4x3
            var b1 = floatOP.dot(A1, xTrue);   // length 4, exactly in range(A)
            var xSvd = arena.floatVec(3);
            SVD.pinvSolve(ref A1, in b1, ref xSvd, out bool converged);
            AssertTrue(converged);
            for (int k = 0; k < 3; k++)
                AssertClose(xSvd[k], xTrue[k], (float)1E-2);

            // --- QR direct solve (destroys A and b) ---
            var A2 = arena.floatLauchli(3, eps);   // (3+1)x3 = 4x3
            var b2 = floatOP.dot(A2, xTrue);
            var xQr = arena.floatVec(3);
            OrthoOP.qrDirectSolve(ref A2, ref b2, ref xQr);
            for (int k = 0; k < 3; k++)
                AssertClose(xQr[k], xTrue[k], (float)1E-2);

            arena.Dispose();
        }

        // Symmetric Pascal matrix P[i,j] = P[i-1,j] + P[i,j-1] (P[i,0]=P[0,j]=1). Known: det(P) = 1
        // for all n, and P is SPD. Tests LU determinant against an exact integer result + Cholesky.
        void PascalDetAndCholesky()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;
            var P = arena.floatPascal(n);

            // SPD -> Cholesky succeeds (read-only on P)
            var L = arena.floatMat(n, n);
            AssertTrue(Cholesky.choleskyDecomposition(in P, ref L));

            // det(Pascal) = 1 (LU destroys its input, so factor a copy)
            var LUmat = P.Copy();
            var pivot = new Pivot(n, Allocator.Temp);
            LU.luDecompositionInplace(ref LUmat, ref pivot);
            float det = LU.determinant(in LUmat, in pivot);
            pivot.Dispose();

            AssertClose(det, (float)1, (float)1E-2);

            arena.Dispose();
        }

        // Hilbert matrix — the canonical ill-conditioned test matrix. cond₂(H_3) ≈ 524.06 (pinned),
        // and cond grows explosively: cond₂(H_5) ≈ 4.77e5 (assert merely "huge", float can't nail it).
        void HilbertCond()
        {
            var arena = new Arena(Allocator.Persistent);

            var H3 = arena.floatHilbert(3);
            AssertClose(floatOP.cond(in H3), (float)524.0568, (float)5);

            var H5 = arena.floatHilbert(5);
            AssertBelow((float)1E5, floatOP.cond(in H5));   // cond(H_5) ≈ 4.77e5, comfortably > 1e5

            arena.Dispose();
        }

        // Wilkinson W21+ : symmetric tridiagonal, diag |i-10|, off-diag 1. Famous near-pair: the two
        // largest eigenvalues both ≈ 10.74619 (agree to ~1e-14). Stresses the cyclic-Jacobi eigensolver
        // on near-degenerate eigenvalues (power iteration could not separate them). Trace = 110.
        void WilkinsonEigen()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 21;
            var W = arena.floatWilkinsonPlus(n);   // symmetric tridiag, diag |10-i|, off 1

            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            AssertTrue(Eigen.eigenDecomposition(ref W, ref eig, ref V, 100));   // destroys W; must converge

            // eigenvectors orthonormal
            AssertTrue(Analysis.IsOrthogonal(V, (float)1E-3));

            // two spectral invariants over ALL eigenvalues (so corrupted middle ones can't hide):
            //   Σλ = trace = 2*(1+..+10) = 110;   Σλ² = ‖W‖_F² = 2*(1²+..+10²) + 40 = 810
            float sum = (float)0, sumSq = (float)0;
            for (int i = 0; i < n; i++) { sum += eig[i]; sumSq += eig[i] * eig[i]; }
            AssertClose(sum, (float)110, (float)1E-2);
            AssertClose(sumSq, (float)810, (float)1E-1);

            // the documented near-pair (two largest)
            AssertClose(eig[0], (float)10.74619, (float)1E-2);
            AssertClose(eig[1], (float)10.74619, (float)1E-2);

            arena.Dispose();
        }

        // 1D Laplacian / second-difference tridiagonal T_n (diag 2, off-diag -1), SPD. Exact
        // eigenvalues λ_k = 2 - 2cos(kπ/(n+1)). Tests eigenDecomposition (eigenvalues), cond
        // (= λ_max/λ_min since symmetric PD), and Cholesky (SPD succeeds) in one case.
        void Laplacian1D()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;
            var T = arena.floatLaplacian1D(n);   // diag 2, off-diag -1

            float pi = (float)math.PI_DBL;
            float lamMax = (float)2 - (float)2 * math.cos((float)n * pi / (float)(n + 1));
            float lamMin = (float)2 - (float)2 * math.cos(pi / (float)(n + 1));

            // condition number (read-only on T)
            AssertClose(floatOP.cond(in T), lamMax / lamMin, (float)1E-2);

            // SPD -> Cholesky succeeds (read-only on T)
            var L = arena.floatMat(n, n);
            AssertTrue(Cholesky.choleskyDecomposition(in T, ref L));

            // eigenvalues match the closed form, descending: eig[i] = 2 - 2cos((n-i)π/(n+1))
            var Tc = T.Copy();           // eigenDecomposition destroys its input
            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            AssertTrue(Eigen.eigenDecomposition(ref Tc, ref eig, ref V));   // must converge

            // eigenvectors must be orthonormal
            AssertTrue(Analysis.IsOrthogonal(V, (float)1E-3));

            for (int i = 0; i < n; i++)
            {
                float expected = (float)2 - (float)2 * math.cos((float)(n - i) * pi / (float)(n + 1));
                AssertClose(eig[i], expected, (float)1E-3);
            }

            arena.Dispose();
        }

        // QR must be scale-invariant: scaling A by 1e-7 must not change that A = Q·R reconstructs.
        // (Regression for the absolute zero-column threshold bug — pre-fix, every column of a
        // uniformly tiny matrix read as "zero" and QR produced garbage.)
        void QRScaleInvariance()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;
            float scale = (float)1E-7;

            var A = arena.floatRandomMatrix(n, n, -1f, 1f, 90211);
            floatOP.mulInpl(A, scale);   // entries now ~1e-7

            var Q = A.Copy();
            var R = arena.floatMat(n, n);
            OrthoOP.qrDecomposition(ref Q, ref R);

            floatMxN recon = floatOP.dot(Q, R);
            float err = Analysis.MaxZeroError(A - recon);

            // relative to the matrix scale; pre-fix this was O(scale) (total garbage)
            AssertBelow(err / scale, (float)1E-3);

            arena.Dispose();
        }

        // [[1,2],[2,1]] is symmetric but indefinite (eigenvalues 3, -1): Cholesky MUST return false.
        void IndefiniteCholeskyFails()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.floatMat(2, 2);
            A[0, 0] = (float)1; A[0, 1] = (float)2;
            A[1, 0] = (float)2; A[1, 1] = (float)1;

            var L = arena.floatMat(2, 2);
            bool spd = Cholesky.choleskyDecomposition(in A, ref L);

            // must be rejected as not positive-definite
            if (spd && Fail[0] == (float)0)
            {
                Fail[0] = (float)1; Fail[1] = (float)1; Fail[2] = (float)0; Fail[3] = (float)1;
            }
            Assert.IsFalse(spd);

            arena.Dispose();
        }

        void AssertClose(float a, float b, float precision)
        {
            float diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1; Fail[1] = a; Fail[2] = b; Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertBelow(float value, float limit)
        {
            if (!(value < limit) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1; Fail[1] = value; Fail[2] = limit; Fail[3] = value - limit;
            }
            Assert.IsTrue(value < limit);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (float)0)
            {
                Fail[0] = (float)1; Fail[1] = (float)0; Fail[2] = (float)1; Fail[3] = (float)1;
            }
            Assert.IsTrue(ok);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void LiteratureTests(TestJob.TestType type)
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
