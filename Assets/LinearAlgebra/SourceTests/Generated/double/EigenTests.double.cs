using System;

using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Stats;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using UnityEngine;
using UnityEngine.TestTools;

public class doubleEigenTests
{

    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            // eigenDecomposition
            EigenIdentity,
            EigenDiagonal,
            EigenKnown2x2,
            EigenRandomSymmetric,
            EigenReconstruct,
            EigenPSDvsSVD,
            EigenZero,
            EigenRank1Projection,
            EigenLaplacianSingular,
            EigenClement,
            EigenFiedler,
            EigenDingDong,
            EigenNonConvergence,
            // powerIteration
            PowerDiagonalDominant,
            PowerNegativeDominant,
            PowerSymmetricCrossCheck,
            PowerComplexPair,
            PowerZeroMatrix,
            // eigenvaluesSymmetric
            EvSymIdentity,
            EvSymDiagonal,
            EvSymKnown2x2,
            EvSymN1,
            EvSymCrossCheckJacobi,
            EvSymLaplacian
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff/index
        public NativeArray<double> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.EigenIdentity:
                    EigenIdentity();
                    break;
                case TestType.EigenDiagonal:
                    EigenDiagonal();
                    break;
                case TestType.EigenKnown2x2:
                    EigenKnown2x2();
                    break;
                case TestType.EigenRandomSymmetric:
                    EigenRandomSymmetric();
                    break;
                case TestType.EigenReconstruct:
                    EigenReconstruct();
                    break;
                case TestType.EigenPSDvsSVD:
                    EigenPSDvsSVD();
                    break;
                case TestType.EigenZero:
                    EigenZero();
                    break;
                case TestType.EigenRank1Projection:
                    EigenRank1Projection();
                    break;
                case TestType.EigenLaplacianSingular:
                    EigenLaplacianSingular();
                    break;
                case TestType.EigenClement:
                    EigenClement();
                    break;
                case TestType.EigenFiedler:
                    EigenFiedler();
                    break;
                case TestType.EigenDingDong:
                    EigenDingDong();
                    break;
                case TestType.EigenNonConvergence:
                    EigenNonConvergence();
                    break;
                case TestType.PowerDiagonalDominant:
                    PowerDiagonalDominant();
                    break;
                case TestType.PowerNegativeDominant:
                    PowerNegativeDominant();
                    break;
                case TestType.PowerSymmetricCrossCheck:
                    PowerSymmetricCrossCheck();
                    break;
                case TestType.PowerComplexPair:
                    PowerComplexPair();
                    break;
                case TestType.PowerZeroMatrix:
                    PowerZeroMatrix();
                    break;
                case TestType.EvSymIdentity:
                    EvSymIdentity();
                    break;
                case TestType.EvSymDiagonal:
                    EvSymDiagonal();
                    break;
                case TestType.EvSymKnown2x2:
                    EvSymKnown2x2();
                    break;
                case TestType.EvSymN1:
                    EvSymN1();
                    break;
                case TestType.EvSymCrossCheckJacobi:
                    EvSymCrossCheckJacobi();
                    break;
                case TestType.EvSymLaplacian:
                    EvSymLaplacian();
                    break;
            }
        }

        // ---------------------------------------------------------------------
        // eigenDecomposition tests
        // ---------------------------------------------------------------------

        // 4x4 identity: every eigenvalue == 1, V orthogonal. Exact closed form, so
        // eigenvalue tolerance 100*ZeroTreshold is comfortably above float Jacobi noise.
        public void EigenIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;

            var A = arena.doubleIdentityMatrix(n);
            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);

            bool converged = Eigen.eigenDecomposition(ref A, ref eig, ref V);

            Assert.IsTrue(converged);

            for (int i = 0; i < n; i++)
                AssertClose(eig[i], (double)1, (double)100 * Consts.doubleZeroTreshold);

            Assert.IsTrue(Analysis.IsOrthogonal(V, (double)100 * Consts.doubleZeroTreshold));

            arena.Dispose();
        }

        // diag(3, -2, 0.5, 5): eigenvalues are the diagonal, sorted DESCENDING BY VALUE
        // -> (5, 3, 0.5, -2). V orthogonal. Diagonal input is exact, tolerance generous.
        public void EigenDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;

            var A = arena.doubleMat(n, n);
            A[0, 0] = (double)3;
            A[1, 1] = (double)(-2);
            A[2, 2] = (double)0.5;
            A[3, 3] = (double)5;

            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);

            bool converged = Eigen.eigenDecomposition(ref A, ref eig, ref V);

            Assert.IsTrue(converged);

            AssertClose(eig[0], (double)5, (double)100 * Consts.doubleZeroTreshold);
            AssertClose(eig[1], (double)3, (double)100 * Consts.doubleZeroTreshold);
            AssertClose(eig[2], (double)0.5, (double)100 * Consts.doubleZeroTreshold);
            AssertClose(eig[3], (double)(-2), (double)100 * Consts.doubleZeroTreshold);

            AssertDescending(in eig, n);

            Assert.IsTrue(Analysis.IsOrthogonal(V, (double)100 * Consts.doubleZeroTreshold));

            arena.Dispose();
        }

        // [[2,1],[1,2]]: eigenvalues 3 (vector (1,1)/sqrt2) and 1 (vector (1,-1)/sqrt2).
        // Sign-agnostic: assert A_orig * v_k ~= lambda_k * v_k for each column.
        public void EigenKnown2x2()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 2;

            var A = arena.doubleMat(n, n);
            A[0, 0] = (double)2; A[0, 1] = (double)1;
            A[1, 0] = (double)1; A[1, 1] = (double)2;

            var Aorig = A.Copy();

            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);

            bool converged = Eigen.eigenDecomposition(ref A, ref eig, ref V);

            Assert.IsTrue(converged);

            AssertClose(eig[0], (double)3, (double)100 * Consts.doubleZeroTreshold);
            AssertClose(eig[1], (double)1, (double)100 * Consts.doubleZeroTreshold);

            AssertDescending(in eig, n);

            // sign-agnostic eigenvector verification: ||A*v_k - lambda_k*v_k||_inf small
            AssertEigenResidual(in Aorig, in V, in eig, n);

            // V orthogonal
            Assert.IsTrue(Analysis.IsOrthogonal(V, (double)100 * Consts.doubleZeroTreshold));

            arena.Dispose();
        }

        // 8x8 random symmetric (values ~ +-5). Check: converged, V orthogonal, eigenvalues
        // descending, per-eigenpair residual small (scaled by (1+|lambda|)), trace == sum lambda.
        // Residual/orthogonality tolerance scaled by matrix magnitude: 8x8 entries up to 5,
        // float Jacobi residual ~ few * 1e-5 absolute -> 1000*ZeroTreshold*(1+|lambda|).
        public void EigenRandomSymmetric()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 8;

            var A = arena.doubleRandomMatrix(n, n, (double)(-5), (double)5, 8123451);
            // symmetrize in place
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    double avg = (A[i, j] + A[j, i]) * (double)0.5;
                    A[i, j] = avg;
                    A[j, i] = avg;
                }

            var Aorig = A.Copy();

            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);

            bool converged = Eigen.eigenDecomposition(ref A, ref eig, ref V);

            Assert.IsTrue(converged);

            Assert.IsFalse(Analysis.IsAnyNan(in eig));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            Assert.IsTrue(Analysis.IsOrthogonal(V, (double)1000 * Consts.doubleZeroTreshold));

            AssertDescending(in eig, n);

            AssertEigenResidual(in Aorig, in V, in eig, n);

            // trace(A_orig) == sum eigenvalues
            double trace = (double)0;
            for (int i = 0; i < n; i++)
                trace += Aorig[i, i];
            double sumEig = (double)0;
            for (int i = 0; i < n; i++)
                sumEig += eig[i];
            // trace magnitude up to ~8*5 = 40; allow magnitude-scaled tolerance.
            AssertClose(trace, sumEig, (double)1000 * Consts.doubleZeroTreshold);

            arena.Dispose();
        }

        // Same setup as EigenRandomSymmetric (different seed): reconstruct V*diag(lambda)*V^T
        // and compare to A_orig elementwise. Reconstruction error for float Jacobi on an
        // 8x8 with entries up to ~5 lands around 1e-5..1e-4 absolute -> 1000*ZeroTreshold.
        public void EigenReconstruct()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 8;

            var A = arena.doubleRandomMatrix(n, n, (double)(-5), (double)5, 5571903);
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    double avg = (A[i, j] + A[j, i]) * (double)0.5;
                    A[i, j] = avg;
                    A[j, i] = avg;
                }

            var Aorig = A.Copy();

            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);

            bool converged = Eigen.eigenDecomposition(ref A, ref eig, ref V);

            Assert.IsTrue(converged);

            // Reconstruct: recon = V * diag(eig) * V^T
            var diagE = arena.doubleDiagonalMatrix(in eig);
            var Vd = doubleOP.dot(V, diagE);
            var Vt = doubleOP.trans(V);
            var recon = doubleOP.dot(Vd, Vt);

            doubleMxN shouldBeZero = Aorig - recon;

            if (Analysis.IsAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            double precision = (double)1000 * Consts.doubleZeroTreshold;
            double zeroError = Analysis.MaxZeroError(shouldBeZero);
            if (!(zeroError <= precision) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = zeroError;
                Fail[2] = precision;
                Fail[3] = zeroError - precision;
            }
            Assert.IsTrue(Analysis.IsZero(in shouldBeZero, precision));

            arena.Dispose();
        }

        // 6x6 PSD matrix A = B^T B. Eigenvalues must all be >= -tol and equal the singular
        // values of A (which for symmetric PSD equal the eigenvalues) in the same descending
        // order. Both eigenDecomposition and svdDecomposition destroy their input, so copy.
        // A = B^T B with B entries ~ +-3 -> eigenvalues up to ~ order 100; scale tolerance.
        public void EigenPSDvsSVD()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;

            var B = arena.doubleRandomMatrix(n, n, (double)(-3), (double)3, 9920017);

            // A = B^T B (manual), symmetric PSD
            var A = arena.doubleMat(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    double sum = (double)0;
                    for (int k = 0; k < n; k++)
                        sum += B[k, i] * B[k, j];
                    A[i, j] = sum;
                }
            // exact symmetrize to kill any rounding asymmetry
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    double avg = (A[i, j] + A[j, i]) * (double)0.5;
                    A[i, j] = avg;
                    A[j, i] = avg;
                }

            var Aeig = A.Copy();   // destroyed by eigenDecomposition
            var Asvd = A.Copy();   // destroyed by svdDecomposition

            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);

            bool converged = Eigen.eigenDecomposition(ref Aeig, ref eig, ref V);
            Assert.IsTrue(converged);

            // eigenvalues all >= -tol (PSD)
            double negTol = (double)1000 * Consts.doubleZeroTreshold;
            for (int i = 0; i < n; i++)
            {
                bool nonNeg = eig[i] >= -negTol;
                if (!nonNeg && Fail[0] == (double)0)
                {
                    Fail[0] = (double)1;
                    Fail[1] = eig[i];
                    Fail[2] = -negTol;
                    Fail[3] = (double)i;
                }
                Assert.IsTrue(nonNeg);
            }

            // singular values via SVD on a fresh copy
            var S = arena.doubleVec(n);
            var Vsvd = arena.doubleMat(n, n);
            bool svdOk = SVD.svdDecomposition(ref Asvd, ref S, ref Vsvd);
            Assert.IsTrue(svdOk);

            // Compare eigenvalues to singular values, same descending order.
            // Magnitude can reach ~ order 100, so scale tolerance by (1+|S[i]|).
            for (int i = 0; i < n; i++)
            {
                double scale = (double)1 + Unity.Mathematics.math.abs(S[i]);
                double tol = (double)1000 * Consts.doubleZeroTreshold * scale;
                double diff = Unity.Mathematics.math.abs(eig[i] - S[i]);
                if (!(diff <= tol) && Fail[0] == (double)0)
                {
                    Fail[0] = (double)1;
                    Fail[1] = eig[i];
                    Fail[2] = S[i];
                    Fail[3] = diff;
                }
                Assert.IsTrue(diff <= tol);
            }

            arena.Dispose();
        }

        // 5x5 zero matrix: converged, all eigenvalues 0, V orthogonal (stays identity).
        public void EigenZero()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;

            var A = arena.doubleMat(n, n);
            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);

            bool converged = Eigen.eigenDecomposition(ref A, ref eig, ref V);

            Assert.IsTrue(converged);

            Assert.IsFalse(Analysis.IsAnyNan(in eig));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            for (int i = 0; i < n; i++)
                AssertClose(eig[i], (double)0, (double)100 * Consts.doubleZeroTreshold);

            Assert.IsTrue(Analysis.IsOrthogonal(V, (double)100 * Consts.doubleZeroTreshold));

            arena.Dispose();
        }

        // Rank-1 projection A = v*vᵀ (v = (1,2,3,1)): SINGULAR symmetric matrix whose eigenvalues
        // are exactly {‖v‖² = 15, 0, 0, 0}. Tests a genuine zero eigenvalue ALONGSIDE a nonzero one
        // (the realistic rank-deficient eigen case — distinct from the all-zero EigenZero). Checks
        // the dominant eigenvalue, the exact-zero tail, descending order, reconstruction, and that
        // the trailing (null-space) eigenvectors still form an orthonormal V.
        public void EigenRank1Projection()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;

            var v = arena.doubleVec(n);
            v[0] = (double)1; v[1] = (double)2; v[2] = (double)3; v[3] = (double)1;
            double vv = (double)0;
            for (int i = 0; i < n; i++) vv += v[i] * v[i]; // = 15

            var A = arena.doubleMat(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = v[i] * v[j];

            var Aorig = A.Copy();
            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);

            bool converged = Eigen.eigenDecomposition(ref A, ref eig, ref V);
            Assert.IsTrue(converged);
            Assert.IsFalse(Analysis.IsAnyNan(in eig));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            double tol = (double)100 * Consts.doubleZeroTreshold;
            // dominant eigenvalue == ‖v‖² = 15; the other three are exactly zero.
            AssertClose(eig[0], vv, tol * ((double)1 + vv));
            for (int i = 1; i < n; i++)
                AssertClose(eig[i], (double)0, tol * ((double)1 + vv));

            AssertDescending(in eig, n);
            AssertEigenResidual(in Aorig, in V, in eig, n);
            Assert.IsTrue(Analysis.IsOrthogonal(V, tol));

            arena.Dispose();
        }

        // Triangle-graph Laplacian L = [[2,-1,-1],[-1,2,-1],[-1,-1,2]]: a classic SINGULAR symmetric
        // matrix with exact eigenvalues {3, 3, 0} (the 0 is the all-ones null vector; rank 2). A
        // known literature vector exercising a zero eigenvalue plus a repeated nonzero one.
        public void EigenLaplacianSingular()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 3;

            var A = arena.doubleMat(n, n);
            A[0, 0] = (double)2; A[0, 1] = (double)(-1); A[0, 2] = (double)(-1);
            A[1, 0] = (double)(-1); A[1, 1] = (double)2; A[1, 2] = (double)(-1);
            A[2, 0] = (double)(-1); A[2, 1] = (double)(-1); A[2, 2] = (double)2;

            var Aorig = A.Copy();
            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);

            bool converged = Eigen.eigenDecomposition(ref A, ref eig, ref V);
            Assert.IsTrue(converged);
            Assert.IsFalse(Analysis.IsAnyNan(in eig));

            double tol = (double)100 * Consts.doubleZeroTreshold;
            AssertClose(eig[0], (double)3, tol * (double)4);
            AssertClose(eig[1], (double)3, tol * (double)4);
            AssertClose(eig[2], (double)0, tol * (double)4); // singular: smallest eigenvalue is 0

            AssertDescending(in eig, n);
            AssertEigenResidual(in Aorig, in V, in eig, n);
            Assert.IsTrue(Analysis.IsOrthogonal(V, tol));

            arena.Dispose();
        }

        // GALLERY KNOWN-ANSWER (Gallery.Special): n=5 Clement matrix — symmetric tridiagonal with
        // zero diagonal whose eigenvalues are EXACTLY the integer-spaced set {n-1, n-3, ..., -(n-1)}
        // = {4, 2, 0, -2, -4} for n=5 (symmetric about 0, trace 0). Well-separated spectrum, so a
        // 1000*ZeroTreshold absolute tolerance comfortably covers float Jacobi noise.
        public void EigenClement()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;

            var A = arena.doubleClement(n);
            var Aorig = A.Copy();

            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);

            bool converged = Eigen.eigenDecomposition(ref A, ref eig, ref V);
            Assert.IsTrue(converged);
            Assert.IsFalse(Analysis.IsAnyNan(in eig));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            double tol = (double)1000 * Consts.doubleZeroTreshold;
            AssertClose(eig[0], (double)4, tol);
            AssertClose(eig[1], (double)2, tol);
            AssertClose(eig[2], (double)0, tol);
            AssertClose(eig[3], (double)(-2), tol);
            AssertClose(eig[4], (double)(-4), tol);

            AssertDescending(in eig, n);
            AssertEigenResidual(in Aorig, in V, in eig, n);
            Assert.IsTrue(Analysis.IsOrthogonal(V, tol));

            arena.Dispose();
        }

        // GALLERY KNOWN-ANSWER (Gallery.Special): n=5 Fiedler distance matrix F[i,j]=|i-j|. Known
        // inertia: EXACTLY ONE positive eigenvalue and n-1 negative ones. For n=5 the spectrum is
        // {8.288, -0.558, -0.764, -1.730, -5.236}; the smallest gap from 0 is ~0.558, so a 1E-2 band
        // cleanly separates the signs while staying far above float Jacobi noise. Descending order
        // means the single positive value lands at eig[0].
        public void EigenFiedler()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;

            var A = arena.doubleFiedler(n);
            var Aorig = A.Copy();

            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);

            bool converged = Eigen.eigenDecomposition(ref A, ref eig, ref V);
            Assert.IsTrue(converged);
            Assert.IsFalse(Analysis.IsAnyNan(in eig));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            double band = (double)1E-2f;

            // The single positive eigenvalue is the largest (descending) -> eig[0] > band.
            bool topPos = eig[0] > band;
            if (!topPos && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = eig[0];
                Fail[2] = band;
                Fail[3] = (double)0;
            }
            Assert.IsTrue(topPos);

            // The remaining n-1 eigenvalues are all strictly negative.
            for (int i = 1; i < n; i++)
            {
                bool isNeg = eig[i] < -band;
                if (!isNeg && Fail[0] == (double)0)
                {
                    Fail[0] = (double)1;
                    Fail[1] = eig[i];
                    Fail[2] = -band;
                    Fail[3] = (double)i;
                }
                Assert.IsTrue(isNeg);
            }

            AssertDescending(in eig, n);
            AssertEigenResidual(in Aorig, in V, in eig, n);
            Assert.IsTrue(Analysis.IsOrthogonal(V, (double)1000 * Consts.doubleZeroTreshold));

            arena.Dispose();
        }

        // GALLERY KNOWN-ANSWER (Gallery.Special): n=5 DingDong symmetric Hankel matrix. Known
        // property: every eigenvalue lies strictly inside (-pi/2, pi/2), clustering near +-pi/2.
        // For n=5 the extreme eigenvalues are ~+-1.5707..., ~1.7e-6 below pi/2, so a small margin
        // absorbs Jacobi error while still asserting the bound is not exceeded.
        public void EigenDingDong()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;

            var A = arena.doubleDingDong(n);
            var Aorig = A.Copy();

            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);

            bool converged = Eigen.eigenDecomposition(ref A, ref eig, ref V);
            Assert.IsTrue(converged);
            Assert.IsFalse(Analysis.IsAnyNan(in eig));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            double half = (double)(Unity.Mathematics.math.PI_DBL * 0.5);
            double margin = (double)1000 * Consts.doubleZeroTreshold;

            for (int i = 0; i < n; i++)
            {
                bool inBand = eig[i] <= half + margin && eig[i] >= -half - margin;
                if (!inBand && Fail[0] == (double)0)
                {
                    Fail[0] = (double)1;
                    Fail[1] = eig[i];
                    Fail[2] = half;
                    Fail[3] = (double)i;
                }
                Assert.IsTrue(inBand);
            }

            AssertDescending(in eig, n);
            AssertEigenResidual(in Aorig, in V, in eig, n);
            Assert.IsTrue(Analysis.IsOrthogonal(V, margin));

            arena.Dispose();
        }

        // Hilbert-like symmetric matrix with maxSweeps = 1: regardless of returned bool,
        // outputs must be finite (no NaN) and eigenvalues descending.
        public void EigenNonConvergence()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 8;

            var A = arena.doubleHilbertMatrix(n);
            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);

            // maxSweeps = 1: convergence not asserted.
            Eigen.eigenDecomposition(ref A, ref eig, ref V, 1);

            Assert.IsFalse(Analysis.IsAnyNan(in eig));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            AssertDescending(in eig, n);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // powerIteration tests
        // ---------------------------------------------------------------------

        // diag(5, 3, 1, 0.5) with v = 0 input (exercises deterministic seeding) -> true,
        // lambda ~= 5 (dominant), residual property ||A*v - lambda*v||_inf <= tol*max(1,|lambda|).
        public void PowerDiagonalDominant()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;

            var A = arena.doubleMat(n, n);
            A[0, 0] = (double)5;
            A[1, 1] = (double)3;
            A[2, 2] = (double)1;
            A[3, 3] = (double)0.5;

            var v = arena.doubleVec(n);   // zero vector -> deterministic seeding
            var w = arena.doubleVec(n);

            double tol = (double)10 * Consts.doubleZeroTreshold;
            bool ok = Eigen.powerIteration(in A, ref v, ref w, out double lambda, tol, 1000);

            Assert.IsTrue(ok);
            AssertFinite(lambda);
            AssertClose(lambda, (double)5, (double)100 * Consts.doubleZeroTreshold);

            AssertPowerResidual(in A, in v, lambda, tol, n);

            arena.Dispose();
        }

        // diag(-7, 2, 1): dominant BY MAGNITUDE is -7. lambda ~= -7, |v[0]| ~= 1 (e0 dir).
        public void PowerNegativeDominant()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 3;

            var A = arena.doubleMat(n, n);
            A[0, 0] = (double)(-7);
            A[1, 1] = (double)2;
            A[2, 2] = (double)1;

            var v = arena.doubleVec(n);
            var w = arena.doubleVec(n);

            double tol = (double)10 * Consts.doubleZeroTreshold;
            bool ok = Eigen.powerIteration(in A, ref v, ref w, out double lambda, tol, 1000);

            Assert.IsTrue(ok);
            AssertFinite(lambda);
            AssertClose(lambda, (double)(-7), (double)100 * Consts.doubleZeroTreshold);

            // eigenvector aligned with e0: |v[0]| ~= 1
            AssertClose(Unity.Mathematics.math.abs(v[0]), (double)1, (double)100 * Consts.doubleZeroTreshold);

            AssertPowerResidual(in A, in v, lambda, tol, n);

            arena.Dispose();
        }

        // 6x6 random symmetric with a forced clear dominant eigenvalue (+12 boost on one
        // diagonal). Reference lambda_max from eigenDecomposition on a copy. Power iteration
        // finds dominant BY MAGNITUDE; the boosted positive eigenvalue dominates both in
        // value and magnitude, so the reference is eig[0] (largest by value == largest |.|).
        public void PowerSymmetricCrossCheck()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;

            var A = arena.doubleRandomMatrix(n, n, (double)(-4), (double)4, 4471123);
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    double avg = (A[i, j] + A[j, i]) * (double)0.5;
                    A[i, j] = avg;
                    A[j, i] = avg;
                }
            // Force a clearly dominant positive eigenvalue (well separated in magnitude).
            A[0, 0] = A[0, 0] + (double)12;

            var Apow = A.Copy();
            var Aeig = A.Copy();

            // reference: dominant eigenvalue by value (== by magnitude here, well separated)
            var eig = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);
            bool econv = Eigen.eigenDecomposition(ref Aeig, ref eig, ref V);
            Assert.IsTrue(econv);

            // dominant by magnitude: compare |eig[0]| vs |eig[n-1]|
            double reference = eig[0];
            if (Unity.Mathematics.math.abs(eig[n - 1]) > Unity.Mathematics.math.abs(eig[0]))
                reference = eig[n - 1];

            var v = arena.doubleVec(n);
            var w = arena.doubleVec(n);

            double tol = (double)10 * Consts.doubleZeroTreshold;
            bool ok = Eigen.powerIteration(in Apow, ref v, ref w, out double lambda, tol, 2000);

            Assert.IsTrue(ok);
            AssertFinite(lambda);

            // magnitude up to ~16; scale tolerance by (1+|reference|).
            double scale = (double)1 + Unity.Mathematics.math.abs(reference);
            AssertClose(lambda, reference, (double)1000 * Consts.doubleZeroTreshold * scale);

            arena.Dispose();
        }

        // 2x2 rotation [[0,-1],[1,0]] (eigenvalues +-i): power iteration cannot converge,
        // returns false; v finite, lambda finite (no NaN).
        public void PowerComplexPair()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 2;

            var A = arena.doubleMat(n, n);
            A[0, 0] = (double)0; A[0, 1] = (double)(-1);
            A[1, 0] = (double)1; A[1, 1] = (double)0;

            var v = arena.doubleVec(n);
            var w = arena.doubleVec(n);

            double tol = (double)10 * Consts.doubleZeroTreshold;
            bool ok = Eigen.powerIteration(in A, ref v, ref w, out double lambda, tol, 200);

            Assert.IsFalse(ok);
            AssertFinite(lambda);
            for (int i = 0; i < n; i++)
                AssertFinite(v[i]);

            arena.Dispose();
        }

        // 3x3 zero matrix: A*v == 0, ||w|| == 0 path -> lambda set to 0, returns true.
        public void PowerZeroMatrix()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 3;

            var A = arena.doubleMat(n, n);

            var v = arena.doubleVec(n);
            var w = arena.doubleVec(n);

            double tol = (double)10 * Consts.doubleZeroTreshold;
            bool ok = Eigen.powerIteration(in A, ref v, ref w, out double lambda, tol, 1000);

            Assert.IsTrue(ok);
            AssertFinite(lambda);
            AssertClose(lambda, (double)0, (double)100 * Consts.doubleZeroTreshold);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // eigenvaluesSymmetric tests (Householder tridiagonalization + implicit-shift QL)
        // ---------------------------------------------------------------------

        // n=5 identity: all eigenvalues exactly 1, sorted descending. Exact closed form ->
        // 100*ZeroTreshold tolerance comfortably above QL noise. A is DESTROYED.
        public void EvSymIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;

            var A = arena.doubleIdentityMatrix(n);
            var eig = arena.doubleVec(n);

            bool ok = Eigen.eigenvaluesSymmetric(ref A, ref eig);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.IsAnyNan(in eig));

            for (int i = 0; i < n; i++)
                AssertClose(eig[i], (double)1, (double)100 * Consts.doubleZeroTreshold);

            AssertDescending(in eig, n);

            arena.Dispose();
        }

        // diag(3, -2, 0.5, 5, -7, 1): eigenvalues are the diagonal entries, sorted DESCENDING BY
        // VALUE -> (5, 3, 1, 0.5, -2, -7). Diagonal input is exact (Householder leaves it untouched),
        // so a generous tolerance applies. A is DESTROYED.
        public void EvSymDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;

            var A = arena.doubleMat(n, n);
            A[0, 0] = (double)3;
            A[1, 1] = (double)(-2);
            A[2, 2] = (double)0.5;
            A[3, 3] = (double)5;
            A[4, 4] = (double)(-7);
            A[5, 5] = (double)1;

            var eig = arena.doubleVec(n);

            bool ok = Eigen.eigenvaluesSymmetric(ref A, ref eig);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.IsAnyNan(in eig));

            double tol = (double)100 * Consts.doubleZeroTreshold;
            AssertClose(eig[0], (double)5, tol);
            AssertClose(eig[1], (double)3, tol);
            AssertClose(eig[2], (double)1, tol);
            AssertClose(eig[3], (double)0.5, tol);
            AssertClose(eig[4], (double)(-2), tol);
            AssertClose(eig[5], (double)(-7), tol);

            AssertDescending(in eig, n);

            arena.Dispose();
        }

        // [[2,1],[1,2]]: closed-form eigenvalues 3 and 1 (descending). A is DESTROYED.
        public void EvSymKnown2x2()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 2;

            var A = arena.doubleMat(n, n);
            A[0, 0] = (double)2; A[0, 1] = (double)1;
            A[1, 0] = (double)1; A[1, 1] = (double)2;

            var eig = arena.doubleVec(n);

            bool ok = Eigen.eigenvaluesSymmetric(ref A, ref eig);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.IsAnyNan(in eig));

            AssertClose(eig[0], (double)3, (double)100 * Consts.doubleZeroTreshold);
            AssertClose(eig[1], (double)1, (double)100 * Consts.doubleZeroTreshold);

            AssertDescending(in eig, n);

            arena.Dispose();
        }

        // n=1 trivial: the sole eigenvalue equals the single entry (early-return path, no iteration).
        public void EvSymN1()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 1;

            var A = arena.doubleMat(n, n);
            A[0, 0] = (double)(-3.25);

            var eig = arena.doubleVec(n);

            bool ok = Eigen.eigenvaluesSymmetric(ref A, ref eig);

            Assert.IsTrue(ok);
            AssertClose(eig[0], (double)(-3.25), (double)100 * Consts.doubleZeroTreshold);

            arena.Dispose();
        }

        // CROSS-CHECK vs the Jacobi eigenDecomposition: for n=6 and n=8 random SYMMETRIC matrices,
        // run eigenDecomposition on one copy and eigenvaluesSymmetric on a SEPARATE copy (both
        // DESTROY their input, both sort descending) and require the eigenvalue vectors to agree.
        // Tolerance scaled by (1+|lambda|): entries ~ +-5, so float values land around few*1e-5.
        public void EvSymCrossCheckJacobi()
        {
            CrossCheckOne(6, 6610337);
            CrossCheckOne(8, 1277459);
        }

        private void CrossCheckOne(int n, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleRandomMatrix(n, n, (double)(-5), (double)5, seed);
            // symmetrize in place
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    double avg = (A[i, j] + A[j, i]) * (double)0.5;
                    A[i, j] = avg;
                    A[j, i] = avg;
                }

            var Ajac = A.Copy();   // destroyed by eigenDecomposition
            var Aql = A.Copy();    // destroyed by eigenvaluesSymmetric

            var eigJac = arena.doubleVec(n);
            var V = arena.doubleMat(n, n);
            bool jacOk = Eigen.eigenDecomposition(ref Ajac, ref eigJac, ref V);
            Assert.IsTrue(jacOk);

            var eigQL = arena.doubleVec(n);
            bool qlOk = Eigen.eigenvaluesSymmetric(ref Aql, ref eigQL);
            Assert.IsTrue(qlOk);

            Assert.IsFalse(Analysis.IsAnyNan(in eigQL));
            AssertDescending(in eigQL, n);

            // both sorted descending -> compare elementwise.
            for (int i = 0; i < n; i++)
            {
                double scale = (double)1 + Unity.Mathematics.math.abs(eigJac[i]);
                double tol = (double)1000 * Consts.doubleZeroTreshold * scale;
                double diff = Unity.Mathematics.math.abs(eigQL[i] - eigJac[i]);
                if (!(diff <= tol) && Fail[0] == (double)0)
                {
                    Fail[0] = (double)1;
                    Fail[1] = eigQL[i];
                    Fail[2] = eigJac[i];
                    Fail[3] = diff;
                }
                Assert.IsTrue(diff <= tol);
            }

            arena.Dispose();
        }

        // LITERATURE KNOWN-ANSWER: n=6 path-graph (1D Laplacian) tridiagonal with diag 2 and
        // off-diagonal -1. Eigenvalues are EXACTLY lambda_k = 2 - 2*cos(k*pi/(n+1)), k=1..n. Sorted
        // descending corresponds to k = n, n-1, ..., 1. Well-separated spectrum -> 1000*ZeroTreshold
        // absolute tolerance covers float QL noise.
        public void EvSymLaplacian()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;

            var A = arena.doubleMat(n, n);
            for (int i = 0; i < n; i++)
            {
                A[i, i] = (double)2;
                if (i + 1 < n)
                {
                    A[i, i + 1] = (double)(-1);
                    A[i + 1, i] = (double)(-1);
                }
            }

            var eig = arena.doubleVec(n);

            bool ok = Eigen.eigenvaluesSymmetric(ref A, ref eig);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.IsAnyNan(in eig));

            double tol = (double)1000 * Consts.doubleZeroTreshold;
            // descending order: eig[i] corresponds to k = n - i.
            for (int i = 0; i < n; i++)
            {
                int k = n - i;
                double lamD = 2.0 - 2.0 * Unity.Mathematics.math.cos(k * Unity.Mathematics.math.PI_DBL / (n + 1));
                AssertClose(eig[i], (double)lamD, tol);
            }

            AssertDescending(in eig, n);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // Helpers
        // ---------------------------------------------------------------------

        // For every eigenpair (lambda_k = eig[k], v_k = column k of V), assert
        // ||A*v_k - lambda_k*v_k||_inf <= 1000*ZeroTreshold * (1 + |lambda_k|).
        private void AssertEigenResidual(in doubleMxN A, in doubleMxN V, in doubleN eig, int n)
        {
            for (int k = 0; k < n; k++)
            {
                double lambda = eig[k];
                double maxRes = (double)0;
                for (int i = 0; i < n; i++)
                {
                    double av = (double)0;
                    for (int j = 0; j < n; j++)
                        av += A[i, j] * V[j, k];
                    double ri = Unity.Mathematics.math.abs(av - lambda * V[i, k]);
                    if (ri > maxRes)
                        maxRes = ri;
                }
                double tol = (double)1000 * Consts.doubleZeroTreshold * ((double)1 + Unity.Mathematics.math.abs(lambda));
                if (!(maxRes <= tol) && Fail[0] == (double)0)
                {
                    Fail[0] = (double)1;
                    Fail[1] = maxRes;
                    Fail[2] = tol;
                    Fail[3] = (double)k;
                }
                Assert.IsTrue(maxRes <= tol);
            }
        }

        // Recompute residual r = A*v - lambda*v (inf-norm) and assert it satisfies the
        // documented convergence criterion r <= tol * max(1, |lambda|).
        private void AssertPowerResidual(in doubleMxN A, in doubleN v, double lambda, double tol, int n)
        {
            double maxRes = (double)0;
            for (int i = 0; i < n; i++)
            {
                double av = (double)0;
                for (int j = 0; j < n; j++)
                    av += A[i, j] * v[j];
                double ri = Unity.Mathematics.math.abs(av - lambda * v[i]);
                if (ri > maxRes)
                    maxRes = ri;
            }
            double scale = Unity.Mathematics.math.abs(lambda);
            if (scale < (double)1)
                scale = (double)1;
            double limit = tol * scale;
            if (!(maxRes <= limit) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = maxRes;
                Fail[2] = limit;
                Fail[3] = maxRes - limit;
            }
            Assert.IsTrue(maxRes <= limit);
        }

        // Eigenvalues descending by value: eig[i] <= eig[i-1] (+ slack).
        // Fail layout: [1]=eig[i], [2]=eig[i-1], [3]=index.
        private void AssertDescending(in doubleN eig, int n)
        {
            for (int i = 1; i < n; i++)
            {
                bool descending = eig[i] <= eig[i - 1] + (double)100 * Consts.doubleZeroTreshold;
                if (!descending && Fail[0] == (double)0)
                {
                    Fail[0] = (double)1;
                    Fail[1] = eig[i];
                    Fail[2] = eig[i - 1];
                    Fail[3] = (double)i;
                }
                Assert.IsTrue(descending);
            }
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff
        private void AssertClose(double a, double b, double precision)
        {
            double diff = Unity.Mathematics.math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        private void AssertFinite(double v)
        {
            if (!Unity.Mathematics.math.isfinite(v) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = v;
                Fail[2] = (double)0;
                Fail[3] = (double)0;
            }
            Assert.IsTrue(Unity.Mathematics.math.isfinite(v));
        }

    }

    public static Array GetEnums() {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void EigenSolverTests(TestJob.TestType type)
    {
        var fail = new NativeArray<double>(4, Allocator.TempJob);
        try {
            new TestJob() { Type = type, Fail = fail }.Run();
            // Under Burst a failed in-job assert logs an exception and aborts the job without
            // throwing to the caller - surface the recorded diagnostics here as well.
            if (fail[0] != (double)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");

        }
        catch (Exception e) {
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally {
            fail.Dispose();
        }
    }

    // -------------------------------------------------------------------------
    // Managed throw-tests: argument validation runs on the main thread (not Burst).
    // -------------------------------------------------------------------------

    [Test]
    public void EigenThrowsOnNonSquare()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.doubleMat(3, 4);
        var eig = arena.doubleVec(4);
        var V = arena.doubleMat(4, 4);

        Assert.Catch<ArgumentException>(() => Eigen.eigenDecomposition(ref A, ref eig, ref V));

        arena.Dispose();
    }

    [Test]
    public void EigenThrowsOnWrongEigenvalueLength()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.doubleMat(4, 4);
        var eig = arena.doubleVec(3);
        var V = arena.doubleMat(4, 4);

        Assert.Catch<ArgumentException>(() => Eigen.eigenDecomposition(ref A, ref eig, ref V));

        arena.Dispose();
    }

    [Test]
    public void EigenThrowsOnWrongVShape()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.doubleMat(4, 4);
        var eig = arena.doubleVec(4);
        var V = arena.doubleMat(3, 3);

        Assert.Catch<ArgumentException>(() => Eigen.eigenDecomposition(ref A, ref eig, ref V));

        arena.Dispose();
    }

    [Test]
    public void EigenThrowsOnBadMaxSweeps()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.doubleMat(4, 4);
        var eig = arena.doubleVec(4);
        var V = arena.doubleMat(4, 4);

        Assert.Catch<ArgumentException>(() => Eigen.eigenDecomposition(ref A, ref eig, ref V, 0));

        arena.Dispose();
    }

    [Test]
    public void EigenThrowsOnNonSymmetric()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.doubleMat(2, 2);
        A[0, 0] = (double)1; A[0, 1] = (double)2;
        A[1, 0] = (double)0; A[1, 1] = (double)1;

        var eig = arena.doubleVec(2);
        var V = arena.doubleMat(2, 2);

        Assert.Catch<ArgumentException>(() => Eigen.eigenDecomposition(ref A, ref eig, ref V));

        arena.Dispose();
    }

    [Test]
    public void EvSymThrowsOnNonSymmetric()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.doubleMat(2, 2);
        A[0, 0] = (double)1; A[0, 1] = (double)2;
        A[1, 0] = (double)0; A[1, 1] = (double)1;

        var eig = arena.doubleVec(2);

        Assert.Catch<ArgumentException>(() => Eigen.eigenvaluesSymmetric(ref A, ref eig));

        arena.Dispose();
    }

    [Test]
    public void EvSymThrowsOnNonSquare()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.doubleMat(3, 4);
        var eig = arena.doubleVec(4);

        Assert.Catch<ArgumentException>(() => Eigen.eigenvaluesSymmetric(ref A, ref eig));

        arena.Dispose();
    }

    [Test]
    public void EvSymThrowsOnWrongEigenvalueLength()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.doubleMat(4, 4);
        var eig = arena.doubleVec(3);

        Assert.Catch<ArgumentException>(() => Eigen.eigenvaluesSymmetric(ref A, ref eig));

        arena.Dispose();
    }

    [Test]
    public void PowerThrowsOnNonSquare()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.doubleMat(3, 4);
        var v = arena.doubleVec(4);
        var w = arena.doubleVec(4);

        Assert.Catch<ArgumentException>(() =>
            Eigen.powerIteration(in A, ref v, ref w, out double lambda, Consts.doubleZeroTreshold, 1000));

        arena.Dispose();
    }

    [Test]
    public void PowerThrowsOnWrongVLength()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.doubleMat(4, 4);
        var v = arena.doubleVec(3);
        var w = arena.doubleVec(4);

        Assert.Catch<ArgumentException>(() =>
            Eigen.powerIteration(in A, ref v, ref w, out double lambda, Consts.doubleZeroTreshold, 1000));

        arena.Dispose();
    }

    [Test]
    public void PowerThrowsOnWrongWLength()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.doubleMat(4, 4);
        var v = arena.doubleVec(4);
        var w = arena.doubleVec(3);

        Assert.Catch<ArgumentException>(() =>
            Eigen.powerIteration(in A, ref v, ref w, out double lambda, Consts.doubleZeroTreshold, 1000));

        arena.Dispose();
    }

    [Test]
    public void PowerThrowsOnBadMaxIter()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.doubleMat(4, 4);
        var v = arena.doubleVec(4);
        var w = arena.doubleVec(4);

        Assert.Catch<ArgumentException>(() =>
            Eigen.powerIteration(in A, ref v, ref w, out double lambda, Consts.doubleZeroTreshold, 0));

        arena.Dispose();
    }

}
