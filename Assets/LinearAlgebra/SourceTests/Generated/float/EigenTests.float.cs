using System;
#pragma warning disable 618 // intentionally exercises the deprecated cyclic-Jacobi eigenDecomposition (kept for reference)

using LinearAlgebra;
using LinearAlgebra.Gallery;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using UnityEngine;
using UnityEngine.TestTools;

public class floatEigenTests
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
            // powerIteration / inversePowerIteration EigenSolveInfo field assertions
            PowerConvergedInfo,
            PowerMaxIterationsInfo,
            InversePowerBreakdownInfo,
            InversePowerConvergedInfo,
            // eigenvaluesSymmetric
            EvSymIdentity,
            EvSymDiagonal,
            EvSymKnown2x2,
            EvSymN1,
            EvSymCrossCheckJacobi,
            EvSymLaplacian,
            // eigenSymmetric (tred2 + tql2 full decomposition)
            EsymIdentity,
            EsymDiagonal,
            EsymKnown2x2,
            EsymReconstruct,
            EsymOrthogonality,
            EsymEigenpair,
            EsymCrossCheck,
            EsymLaplacian
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff/index
        public NativeArray<float> Fail;

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
                case TestType.PowerConvergedInfo:
                    PowerConvergedInfo();
                    break;
                case TestType.PowerMaxIterationsInfo:
                    PowerMaxIterationsInfo();
                    break;
                case TestType.InversePowerBreakdownInfo:
                    InversePowerBreakdownInfo();
                    break;
                case TestType.InversePowerConvergedInfo:
                    InversePowerConvergedInfo();
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
                case TestType.EsymIdentity:
                    EsymIdentity();
                    break;
                case TestType.EsymDiagonal:
                    EsymDiagonal();
                    break;
                case TestType.EsymKnown2x2:
                    EsymKnown2x2();
                    break;
                case TestType.EsymReconstruct:
                    EsymReconstruct();
                    break;
                case TestType.EsymOrthogonality:
                    EsymOrthogonality();
                    break;
                case TestType.EsymEigenpair:
                    EsymEigenpair();
                    break;
                case TestType.EsymCrossCheck:
                    EsymCrossCheck();
                    break;
                case TestType.EsymLaplacian:
                    EsymLaplacian();
                    break;
            }
        }

        // ---------------------------------------------------------------------
        // eigenDecomposition tests
        // ---------------------------------------------------------------------

        // 4x4 identity: every eigenvalue == 1, V orthogonal. Exact closed form, so
        // eigenvalue tolerance 100*ZeroThreshold is comfortably above float Jacobi noise.
        public void EigenIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;

            var A = arena.floatIdentityMat(n);
            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);

            Assert.IsTrue(converged);

            for (int i = 0; i < n; i++)
                AssertClose(eig[i], (float)1, (float)100 * Consts.floatZeroThreshold);

            Assert.IsTrue(Analysis.isOrthogonal(V, (float)100 * Consts.floatZeroThreshold));

            arena.Dispose();
        }

        // diag(3, -2, 0.5, 5): eigenvalues are the diagonal, sorted DESCENDING BY VALUE
        // -> (5, 3, 0.5, -2). V orthogonal. Diagonal input is exact, tolerance generous.
        public void EigenDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;

            var A = arena.floatMat(n, n);
            A[0, 0] = (float)3;
            A[1, 1] = (float)(-2);
            A[2, 2] = (float)0.5;
            A[3, 3] = (float)5;

            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);

            Assert.IsTrue(converged);

            AssertClose(eig[0], (float)5, (float)100 * Consts.floatZeroThreshold);
            AssertClose(eig[1], (float)3, (float)100 * Consts.floatZeroThreshold);
            AssertClose(eig[2], (float)0.5, (float)100 * Consts.floatZeroThreshold);
            AssertClose(eig[3], (float)(-2), (float)100 * Consts.floatZeroThreshold);

            AssertDescending(in eig, n);

            Assert.IsTrue(Analysis.isOrthogonal(V, (float)100 * Consts.floatZeroThreshold));

            arena.Dispose();
        }

        // [[2,1],[1,2]]: eigenvalues 3 (vector (1,1)/sqrt2) and 1 (vector (1,-1)/sqrt2).
        // Sign-agnostic: assert A_orig * v_k ~= lambda_k * v_k for each column.
        public void EigenKnown2x2()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 2;

            var A = arena.floatMat(n, n);
            A[0, 0] = (float)2; A[0, 1] = (float)1;
            A[1, 0] = (float)1; A[1, 1] = (float)2;

            var Aorig = A.Copy();

            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);

            Assert.IsTrue(converged);

            AssertClose(eig[0], (float)3, (float)100 * Consts.floatZeroThreshold);
            AssertClose(eig[1], (float)1, (float)100 * Consts.floatZeroThreshold);

            AssertDescending(in eig, n);

            // sign-agnostic eigenvector verification: ||A*v_k - lambda_k*v_k||_inf small
            AssertEigenResidual(in Aorig, in V, in eig, n);

            // V orthogonal
            Assert.IsTrue(Analysis.isOrthogonal(V, (float)100 * Consts.floatZeroThreshold));

            arena.Dispose();
        }

        // 8x8 random symmetric (values ~ +-5). Check: converged, V orthogonal, eigenvalues
        // descending, per-eigenpair residual small (scaled by (1+|lambda|)), trace == sum lambda.
        // Residual/orthogonality tolerance scaled by matrix magnitude: 8x8 entries up to 5,
        // float Jacobi residual ~ few * 1e-5 absolute -> 1000*ZeroThreshold*(1+|lambda|).
        public void EigenRandomSymmetric()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 8;

            var A = arena.floatRandomMat(n, n, (float)(-5), (float)5, 8123451);
            // symmetrize in place
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    float avg = (A[i, j] + A[j, i]) * (float)0.5;
                    A[i, j] = avg;
                    A[j, i] = avg;
                }

            var Aorig = A.Copy();

            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);

            Assert.IsTrue(converged);

            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            Assert.IsTrue(Analysis.isOrthogonal(V, (float)1000 * Consts.floatZeroThreshold));

            AssertDescending(in eig, n);

            AssertEigenResidual(in Aorig, in V, in eig, n);

            // trace(A_orig) == sum eigenvalues
            float trace = (float)0;
            for (int i = 0; i < n; i++)
                trace += Aorig[i, i];
            float sumEig = (float)0;
            for (int i = 0; i < n; i++)
                sumEig += eig[i];
            // trace magnitude up to ~8*5 = 40; allow magnitude-scaled tolerance.
            AssertClose(trace, sumEig, (float)1000 * Consts.floatZeroThreshold);

            arena.Dispose();
        }

        // Same setup as EigenRandomSymmetric (different seed): reconstruct V*diag(lambda)*V^T
        // and compare to A_orig elementwise. Reconstruction error for float Jacobi on an
        // 8x8 with entries up to ~5 lands around 1e-5..1e-4 absolute -> 1000*ZeroThreshold.
        public void EigenReconstruct()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 8;

            var A = arena.floatRandomMat(n, n, (float)(-5), (float)5, 5571903);
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    float avg = (A[i, j] + A[j, i]) * (float)0.5;
                    A[i, j] = avg;
                    A[j, i] = avg;
                }

            var Aorig = A.Copy();

            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);

            Assert.IsTrue(converged);

            // Reconstruct: recon = V * diag(eig) * V^T
            var diagE = arena.floatDiagonalMat(in eig);
            var Vd = Blas.dot(V, diagE);
            var Vt = Blas.trans(V);
            var recon = Blas.dot(Vd, Vt);

            floatMxN shouldBeZero = Aorig - recon;

            if (Analysis.isAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            float precision = (float)1000 * Consts.floatZeroThreshold;
            float zeroError = Analysis.MaxZeroError(shouldBeZero);
            if (!(zeroError <= precision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = zeroError;
                Fail[2] = precision;
                Fail[3] = zeroError - precision;
            }
            Assert.IsTrue(Analysis.isZero(in shouldBeZero, precision));

            arena.Dispose();
        }

        // 6x6 PSD matrix A = B^T B. Eigenvalues must all be >= -tol and equal the singular
        // values of A (which for symmetric PSD equal the eigenvalues) in the same descending
        // order. eigenDecomposition destroys its input, so copy; SVD.values takes A `in`
        // (preserved), so no copy is needed for the SVD side.
        // A = B^T B with B entries ~ +-3 -> eigenvalues up to ~ order 100; scale tolerance.
        public void EigenPSDvsSVD()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;

            var B = arena.floatRandomMat(n, n, (float)(-3), (float)3, 9920017);

            // A = B^T B (manual), symmetric PSD
            var A = arena.floatMat(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    float sum = (float)0;
                    for (int k = 0; k < n; k++)
                        sum += B[k, i] * B[k, j];
                    A[i, j] = sum;
                }
            // exact symmetrize to kill any rounding asymmetry
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    float avg = (A[i, j] + A[j, i]) * (float)0.5;
                    A[i, j] = avg;
                    A[j, i] = avg;
                }

            var Aeig = A.Copy();   // destroyed by eigenDecomposition

            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            bool converged = Eigen.decompInPlace(ref Aeig, ref eig, ref V);
            Assert.IsTrue(converged);

            // eigenvalues all >= -tol (PSD)
            float negTol = (float)1000 * Consts.floatZeroThreshold;
            for (int i = 0; i < n; i++)
            {
                bool nonNeg = eig[i] >= -negTol;
                if (!nonNeg && Fail[0] == (float)0)
                {
                    Fail[0] = (float)1;
                    Fail[1] = eig[i];
                    Fail[2] = -negTol;
                    Fail[3] = (float)i;
                }
                Assert.IsTrue(nonNeg);
            }

            // singular values via SVD.values on the untouched A (preserved, no copy needed)
            var S = arena.floatVec(n);
            bool svdOk = SVD.values(in A, ref S);
            Assert.IsTrue(svdOk);

            // Compare eigenvalues to singular values, same descending order.
            // Magnitude can reach ~ order 100, so scale tolerance by (1+|S[i]|).
            for (int i = 0; i < n; i++)
            {
                float scale = (float)1 + Unity.Mathematics.math.abs(S[i]);
                float tol = (float)1000 * Consts.floatZeroThreshold * scale;
                float diff = Unity.Mathematics.math.abs(eig[i] - S[i]);
                if (!(diff <= tol) && Fail[0] == (float)0)
                {
                    Fail[0] = (float)1;
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

            var A = arena.floatMat(n, n);
            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);

            Assert.IsTrue(converged);

            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            for (int i = 0; i < n; i++)
                AssertClose(eig[i], (float)0, (float)100 * Consts.floatZeroThreshold);

            Assert.IsTrue(Analysis.isOrthogonal(V, (float)100 * Consts.floatZeroThreshold));

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

            var v = arena.floatVec(n);
            v[0] = (float)1; v[1] = (float)2; v[2] = (float)3; v[3] = (float)1;
            float vv = (float)0;
            for (int i = 0; i < n; i++) vv += v[i] * v[i]; // = 15

            var A = arena.floatMat(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = v[i] * v[j];

            var Aorig = A.Copy();
            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);
            Assert.IsTrue(converged);
            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            float tol = (float)100 * Consts.floatZeroThreshold;
            // dominant eigenvalue == ‖v‖² = 15; the other three are exactly zero.
            AssertClose(eig[0], vv, tol * ((float)1 + vv));
            for (int i = 1; i < n; i++)
                AssertClose(eig[i], (float)0, tol * ((float)1 + vv));

            AssertDescending(in eig, n);
            AssertEigenResidual(in Aorig, in V, in eig, n);
            Assert.IsTrue(Analysis.isOrthogonal(V, tol));

            arena.Dispose();
        }

        // Triangle-graph Laplacian L = [[2,-1,-1],[-1,2,-1],[-1,-1,2]]: a classic SINGULAR symmetric
        // matrix with exact eigenvalues {3, 3, 0} (the 0 is the all-ones null vector; rank 2). A
        // known literature vector exercising a zero eigenvalue plus a repeated nonzero one.
        public void EigenLaplacianSingular()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 3;

            var A = arena.floatMat(n, n);
            A[0, 0] = (float)2; A[0, 1] = (float)(-1); A[0, 2] = (float)(-1);
            A[1, 0] = (float)(-1); A[1, 1] = (float)2; A[1, 2] = (float)(-1);
            A[2, 0] = (float)(-1); A[2, 1] = (float)(-1); A[2, 2] = (float)2;

            var Aorig = A.Copy();
            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);
            Assert.IsTrue(converged);
            Assert.IsFalse(Analysis.isAnyNan(in eig));

            float tol = (float)100 * Consts.floatZeroThreshold;
            AssertClose(eig[0], (float)3, tol * (float)4);
            AssertClose(eig[1], (float)3, tol * (float)4);
            AssertClose(eig[2], (float)0, tol * (float)4); // singular: smallest eigenvalue is 0

            AssertDescending(in eig, n);
            AssertEigenResidual(in Aorig, in V, in eig, n);
            Assert.IsTrue(Analysis.isOrthogonal(V, tol));

            arena.Dispose();
        }

        // GALLERY KNOWN-ANSWER (Gallery.Special): n=5 Clement matrix — symmetric tridiagonal with
        // zero diagonal whose eigenvalues are EXACTLY the integer-spaced set {n-1, n-3, ..., -(n-1)}
        // = {4, 2, 0, -2, -4} for n=5 (symmetric about 0, trace 0). Well-separated spectrum, so a
        // 1000*ZeroThreshold absolute tolerance comfortably covers float Jacobi noise.
        public void EigenClement()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;

            var A = arena.floatClement(n);
            var Aorig = A.Copy();

            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);
            Assert.IsTrue(converged);
            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            float tol = (float)1000 * Consts.floatZeroThreshold;
            AssertClose(eig[0], (float)4, tol);
            AssertClose(eig[1], (float)2, tol);
            AssertClose(eig[2], (float)0, tol);
            AssertClose(eig[3], (float)(-2), tol);
            AssertClose(eig[4], (float)(-4), tol);

            AssertDescending(in eig, n);
            AssertEigenResidual(in Aorig, in V, in eig, n);
            Assert.IsTrue(Analysis.isOrthogonal(V, tol));

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

            var A = arena.floatFiedler(n);
            var Aorig = A.Copy();

            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);
            Assert.IsTrue(converged);
            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            float band = (float)1E-2f;

            // The single positive eigenvalue is the largest (descending) -> eig[0] > band.
            bool topPos = eig[0] > band;
            if (!topPos && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = eig[0];
                Fail[2] = band;
                Fail[3] = (float)0;
            }
            Assert.IsTrue(topPos);

            // The remaining n-1 eigenvalues are all strictly negative.
            for (int i = 1; i < n; i++)
            {
                bool isNeg = eig[i] < -band;
                if (!isNeg && Fail[0] == (float)0)
                {
                    Fail[0] = (float)1;
                    Fail[1] = eig[i];
                    Fail[2] = -band;
                    Fail[3] = (float)i;
                }
                Assert.IsTrue(isNeg);
            }

            AssertDescending(in eig, n);
            AssertEigenResidual(in Aorig, in V, in eig, n);
            Assert.IsTrue(Analysis.isOrthogonal(V, (float)1000 * Consts.floatZeroThreshold));

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

            var A = arena.floatDingDong(n);
            var Aorig = A.Copy();

            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);
            Assert.IsTrue(converged);
            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            float half = (float)(Unity.Mathematics.math.PI_DBL * 0.5);
            float margin = (float)1000 * Consts.floatZeroThreshold;

            for (int i = 0; i < n; i++)
            {
                bool inBand = eig[i] <= half + margin && eig[i] >= -half - margin;
                if (!inBand && Fail[0] == (float)0)
                {
                    Fail[0] = (float)1;
                    Fail[1] = eig[i];
                    Fail[2] = half;
                    Fail[3] = (float)i;
                }
                Assert.IsTrue(inBand);
            }

            AssertDescending(in eig, n);
            AssertEigenResidual(in Aorig, in V, in eig, n);
            Assert.IsTrue(Analysis.isOrthogonal(V, margin));

            arena.Dispose();
        }

        // Hilbert-like symmetric matrix with maxSweeps = 1: regardless of returned bool,
        // outputs must be finite (no NaN) and eigenvalues descending.
        public void EigenNonConvergence()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 8;

            var A = arena.floatHilbertMat(n);
            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            // maxSweeps = 1: convergence not asserted.
            Eigen.decompInPlace(ref A, ref eig, ref V, 1);

            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

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

            var A = arena.floatMat(n, n);
            A[0, 0] = (float)5;
            A[1, 1] = (float)3;
            A[2, 2] = (float)1;
            A[3, 3] = (float)0.5;

            var v = arena.floatVec(n);   // zero vector -> deterministic seeding
            var w = arena.floatVec(n);

            float tol = (float)10 * Consts.floatZeroThreshold;
            bool ok = Eigen.powerIteration(in A, ref v, ref w, out float lambda, tol, 1000);

            Assert.IsTrue(ok);
            AssertFinite(lambda);
            AssertClose(lambda, (float)5, (float)100 * Consts.floatZeroThreshold);

            AssertPowerResidual(in A, in v, lambda, tol, n);

            arena.Dispose();
        }

        // diag(-7, 2, 1): dominant BY MAGNITUDE is -7. lambda ~= -7, |v[0]| ~= 1 (e0 dir).
        public void PowerNegativeDominant()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 3;

            var A = arena.floatMat(n, n);
            A[0, 0] = (float)(-7);
            A[1, 1] = (float)2;
            A[2, 2] = (float)1;

            var v = arena.floatVec(n);
            var w = arena.floatVec(n);

            float tol = (float)10 * Consts.floatZeroThreshold;
            bool ok = Eigen.powerIteration(in A, ref v, ref w, out float lambda, tol, 1000);

            Assert.IsTrue(ok);
            AssertFinite(lambda);
            AssertClose(lambda, (float)(-7), (float)100 * Consts.floatZeroThreshold);

            // eigenvector aligned with e0: |v[0]| ~= 1
            AssertClose(Unity.Mathematics.math.abs(v[0]), (float)1, (float)100 * Consts.floatZeroThreshold);

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

            var A = arena.floatRandomMat(n, n, (float)(-4), (float)4, 4471123);
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    float avg = (A[i, j] + A[j, i]) * (float)0.5;
                    A[i, j] = avg;
                    A[j, i] = avg;
                }
            // Force a clearly dominant positive eigenvalue (well separated in magnitude).
            A[0, 0] = A[0, 0] + (float)12;

            var Apow = A.Copy();
            var Aeig = A.Copy();

            // reference: dominant eigenvalue by value (== by magnitude here, well separated)
            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            bool econv = Eigen.decompInPlace(ref Aeig, ref eig, ref V);
            Assert.IsTrue(econv);

            // dominant by magnitude: compare |eig[0]| vs |eig[n-1]|
            float reference = eig[0];
            if (Unity.Mathematics.math.abs(eig[n - 1]) > Unity.Mathematics.math.abs(eig[0]))
                reference = eig[n - 1];

            var v = arena.floatVec(n);
            var w = arena.floatVec(n);

            float tol = (float)10 * Consts.floatZeroThreshold;
            bool ok = Eigen.powerIteration(in Apow, ref v, ref w, out float lambda, tol, 2000);

            Assert.IsTrue(ok);
            AssertFinite(lambda);

            // magnitude up to ~16; scale tolerance by (1+|reference|).
            float scale = (float)1 + Unity.Mathematics.math.abs(reference);
            AssertClose(lambda, reference, (float)1000 * Consts.floatZeroThreshold * scale);

            arena.Dispose();
        }

        // 2x2 rotation [[0,-1],[1,0]] (eigenvalues +-i): power iteration cannot converge,
        // returns false; v finite, lambda finite (no NaN).
        public void PowerComplexPair()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 2;

            var A = arena.floatMat(n, n);
            A[0, 0] = (float)0; A[0, 1] = (float)(-1);
            A[1, 0] = (float)1; A[1, 1] = (float)0;

            var v = arena.floatVec(n);
            var w = arena.floatVec(n);

            float tol = (float)10 * Consts.floatZeroThreshold;
            bool ok = Eigen.powerIteration(in A, ref v, ref w, out float lambda, tol, 200);

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

            var A = arena.floatMat(n, n);

            var v = arena.floatVec(n);
            var w = arena.floatVec(n);

            float tol = (float)10 * Consts.floatZeroThreshold;
            bool ok = Eigen.powerIteration(in A, ref v, ref w, out float lambda, tol, 1000);

            Assert.IsTrue(ok);
            AssertFinite(lambda);
            AssertClose(lambda, (float)0, (float)100 * Consts.floatZeroThreshold);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // EigenSolveInfo field assertions (the NEW diagnostics struct the eigensolvers
        // now return alongside `out float lambda`). These pin the FIELD VALUES, not just
        // the implicit-bool success test: status, residual, and iterations. A regression
        // that hard-coded status = Converged or reported a wrong/NaN residual would slip
        // past the pre-existing bool-only tests but is caught here.
        // ---------------------------------------------------------------------

        // powerIteration CONVERGED-field check. Reuses PowerDiagonalDominant's converging setup
        // (diag(5,3,1,0.5), well-separated dominant eigenvalue 5). Beyond Solved, assert the
        // reported residual is the SAME infinity-norm the loop's convergence test used -- i.e.
        // finite, non-negative, and <= tol*max(1,|lambda|) (the exact criterion the algorithm
        // returns Converged on) -- and that at least one iteration was counted.
        public void PowerConvergedInfo()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;

            var A = arena.floatMat(n, n);
            A[0, 0] = (float)5;
            A[1, 1] = (float)3;
            A[2, 2] = (float)1;
            A[3, 3] = (float)0.5;

            var v = arena.floatVec(n);   // zero -> deterministic seeding
            var w = arena.floatVec(n);

            float tol = (float)10 * Consts.floatZeroThreshold;
            var info = Eigen.powerIteration(in A, ref v, ref w, out float lambda, tol, 1000);

            // Converged, three equivalent ways (implicit bool, Solved property, status enum).
            AssertTrue((bool)info, (float)1);
            AssertTrue(info.Solved, (float)2);
            AssertTrue(info.status == IterativeSolveStatus.Converged, (float)3);

            // residual is finite, non-negative, and satisfies the documented convergence bound
            // r <= tol*max(1,|lambda|). Reproduce the scale in float exactly as the algorithm
            // does, then widen -- so this is bit-for-bit the criterion the Converged return used.
            AssertTrue(Unity.Mathematics.math.isfinite(info.residual), (float)4);
            AssertTrue(info.residual >= 0.0, (float)5);

            float fscale = Unity.Mathematics.math.abs(lambda);
            if (fscale < (float)1) fscale = (float)1;
            double limit = (double)(tol * fscale);
            AssertTrue(info.residual <= limit, (float)6);

            // Converged counts the converging iteration too -> iterations >= 1.
            AssertTrue(info.iterations >= 1, (float)7);

            arena.Dispose();
        }

        // powerIteration MAX-ITERATIONS field check (deterministic non-convergence). The 2x2 real
        // rotation [[0,-1],[1,0]] has the complex conjugate pair +-i as its dominant eigenvalue, so
        // power iteration provably CANNOT converge (documented in powerIteration's XML notes). It
        // must exhaust maxIter and report MaxIterations -- assert !Solved, status == MaxIterations,
        // and iterations == maxIter exactly.
        public void PowerMaxIterationsInfo()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 2;

            var A = arena.floatMat(n, n);
            A[0, 0] = (float)0; A[0, 1] = (float)(-1);
            A[1, 0] = (float)1; A[1, 1] = (float)0;

            var v = arena.floatVec(n);
            var w = arena.floatVec(n);

            int maxIter = 200;
            float tol = (float)10 * Consts.floatZeroThreshold;
            var info = Eigen.powerIteration(in A, ref v, ref w, out float lambda, tol, maxIter);

            AssertTrue(!info, (float)1);          // implicit bool: false
            AssertTrue(!info.Solved, (float)2);
            AssertTrue(info.status == IterativeSolveStatus.MaxIterations, (float)3);
            AssertTrue(info.iterations == maxIter, (float)4);
            // residual on a MaxIterations return is still the finite last-iterate residual (NOT NaN).
            AssertTrue(Unity.Mathematics.math.isfinite(info.residual), (float)5);

            arena.Dispose();
        }

        // inversePowerIteration BREAKDOWN field check (deterministic). A = diag(1,-1) is INDEFINITE
        // (not SPD), so the inner CG solve hits non-positive curvature (p.Ap < 0) on its very first
        // step and breaks down; inverse iteration then bails out reporting Breakdown. The
        // deterministic seed (1,2)/sqrt5 makes p.Ap = 1/5 - 4/5 = -3/5 < 0 with certainty. Assert
        // !Solved, status == Breakdown, and residual == NaN (the documented Breakdown residual).
        public void InversePowerBreakdownInfo()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 2;

            var A = arena.floatMat(n, n);
            A[0, 0] = (float)1; A[1, 1] = (float)(-1);   // indefinite -> CG breakdown

            var v = arena.floatVec(n);   // zero -> deterministic (1,2) seeding

            var info = Eigen.inversePowerIteration(in A, ref v, out float lambda);

            AssertTrue(!info, (float)1);          // implicit bool: false
            AssertTrue(!info.Solved, (float)2);
            AssertTrue(info.status == IterativeSolveStatus.Breakdown, (float)3);
            // residual is double.NaN on a Breakdown return. Use the Burst-safe double isnan overload
            // (self-inequality residual != residual is an equally valid check under FloatMode.Default).
            AssertTrue(Unity.Mathematics.math.isnan(info.residual), (float)4);

            arena.Dispose();
        }

        // inversePowerIteration CONVERGED-residual check on an SPD operator that converges. The 1D
        // Laplacian (tridiagonal 2,-1) is SPD with well-separated small eigenvalues, so inverse
        // iteration converges reliably (same fixture family the sparse InverseLaplacianCrossCheck
        // uses). Assert Solved, and that the reported residual is finite, non-negative, and bounded.
        // Inverse iteration's residual bottoms out near cgTol (the inner CG's floor, not machine
        // epsilon), so the bound is a generous cgTol-anchored multiple scaled by max(1,|lambda|) --
        // NOT the tight Consts.floatZeroThreshold bands the pure-matvec powerIteration uses. This
        // still catches an O(1)/NaN residual regression while staying above the honest cgTol floor.
        public void InversePowerConvergedInfo()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 12;

            var A = arena.floatLaplacian1D(n);

            // tol a multiple of cgTol (see inversePowerIteration's doc comment): consecutive
            // eigenpair estimates each come from a fresh CG solve accurate only to ~cgTol.
            float cgTol = Consts.floatSqrtEps;
            float tol = (float)10 * cgTol;

            var v = arena.floatVec(n);   // zero -> deterministic seeding

            var info = Eigen.inversePowerIteration(in A, ref v, out float lambda, tol, 200, n, cgTol);

            AssertTrue(info.Solved, (float)1);
            AssertTrue(info.status == IterativeSolveStatus.Converged, (float)2);

            AssertTrue(Unity.Mathematics.math.isfinite(info.residual), (float)3);
            AssertTrue(info.residual >= 0.0, (float)4);

            // residual bottoms near cgTol; bound by a generous cgTol-anchored multiple (auto-scaled
            // per numeric type via Consts.floatSqrtEps), scaled by max(1,|lambda|).
            float fscale = Unity.Mathematics.math.abs(lambda);
            if (fscale < (float)1) fscale = (float)1;
            double limit = (double)((float)5000 * cgTol * fscale);
            AssertTrue(info.residual <= limit, (float)5);

            AssertTrue(info.iterations >= 1, (float)6);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // eigenvaluesSymmetric tests (Householder tridiagonalization + implicit-shift QL)
        // ---------------------------------------------------------------------

        // n=5 identity: same oracle as EigenIdentity (eigenvalues == 1); QL variant, A is DESTROYED.
        public void EvSymIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;

            var A = arena.floatIdentityMat(n);
            var eig = arena.floatVec(n);

            bool ok = Eigen.valuesSymmetric(ref A, ref eig);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in eig));

            for (int i = 0; i < n; i++)
                AssertClose(eig[i], (float)1, (float)100 * Consts.floatZeroThreshold);

            AssertDescending(in eig, n);

            arena.Dispose();
        }

        // diag(3, -2, 0.5, 5, -7, 1): eigenvalues == diagonal, sorted descending -> (5, 3, 1, 0.5, -2, -7)
        // (same oracle as EigenDiagonal; Householder leaves the diagonal untouched). A is DESTROYED.
        public void EvSymDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;

            var A = arena.floatMat(n, n);
            A[0, 0] = (float)3;
            A[1, 1] = (float)(-2);
            A[2, 2] = (float)0.5;
            A[3, 3] = (float)5;
            A[4, 4] = (float)(-7);
            A[5, 5] = (float)1;

            var eig = arena.floatVec(n);

            bool ok = Eigen.valuesSymmetric(ref A, ref eig);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in eig));

            float tol = (float)100 * Consts.floatZeroThreshold;
            AssertClose(eig[0], (float)5, tol);
            AssertClose(eig[1], (float)3, tol);
            AssertClose(eig[2], (float)1, tol);
            AssertClose(eig[3], (float)0.5, tol);
            AssertClose(eig[4], (float)(-2), tol);
            AssertClose(eig[5], (float)(-7), tol);

            AssertDescending(in eig, n);

            arena.Dispose();
        }

        // [[2,1],[1,2]]: same oracle as EigenKnown2x2 (eigenvalues 3, 1). A is DESTROYED.
        public void EvSymKnown2x2()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 2;

            var A = arena.floatMat(n, n);
            A[0, 0] = (float)2; A[0, 1] = (float)1;
            A[1, 0] = (float)1; A[1, 1] = (float)2;

            var eig = arena.floatVec(n);

            bool ok = Eigen.valuesSymmetric(ref A, ref eig);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in eig));

            AssertClose(eig[0], (float)3, (float)100 * Consts.floatZeroThreshold);
            AssertClose(eig[1], (float)1, (float)100 * Consts.floatZeroThreshold);

            AssertDescending(in eig, n);

            arena.Dispose();
        }

        // n=1 trivial: the sole eigenvalue equals the single entry (early-return path, no iteration).
        public void EvSymN1()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 1;

            var A = arena.floatMat(n, n);
            A[0, 0] = (float)(-3.25);

            var eig = arena.floatVec(n);

            bool ok = Eigen.valuesSymmetric(ref A, ref eig);

            Assert.IsTrue(ok);
            AssertClose(eig[0], (float)(-3.25), (float)100 * Consts.floatZeroThreshold);

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

            var A = arena.floatRandomMat(n, n, (float)(-5), (float)5, seed);
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    float avg = (A[i, j] + A[j, i]) * (float)0.5;
                    A[i, j] = avg;
                    A[j, i] = avg;
                }

            var Ajac = A.Copy();   // destroyed by eigenDecomposition
            var Aql = A.Copy();    // destroyed by eigenvaluesSymmetric

            var eigJac = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            bool jacOk = Eigen.decompInPlace(ref Ajac, ref eigJac, ref V);
            Assert.IsTrue(jacOk);

            var eigQL = arena.floatVec(n);
            bool qlOk = Eigen.valuesSymmetric(ref Aql, ref eigQL);
            Assert.IsTrue(qlOk);

            Assert.IsFalse(Analysis.isAnyNan(in eigQL));
            AssertDescending(in eigQL, n);

            // both sorted descending -> compare elementwise.
            for (int i = 0; i < n; i++)
            {
                float scale = (float)1 + Unity.Mathematics.math.abs(eigJac[i]);
                float tol = (float)1000 * Consts.floatZeroThreshold * scale;
                float diff = Unity.Mathematics.math.abs(eigQL[i] - eigJac[i]);
                if (!(diff <= tol) && Fail[0] == (float)0)
                {
                    Fail[0] = (float)1;
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
        // descending corresponds to k = n, n-1, ..., 1. Well-separated spectrum -> 1000*ZeroThreshold
        // absolute tolerance covers float QL noise.
        public void EvSymLaplacian()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;

            var A = arena.floatMat(n, n);
            for (int i = 0; i < n; i++)
            {
                A[i, i] = (float)2;
                if (i + 1 < n)
                {
                    A[i, i + 1] = (float)(-1);
                    A[i + 1, i] = (float)(-1);
                }
            }

            var eig = arena.floatVec(n);

            bool ok = Eigen.valuesSymmetric(ref A, ref eig);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in eig));

            float tol = (float)1000 * Consts.floatZeroThreshold;
            // descending order: eig[i] corresponds to k = n - i.
            for (int i = 0; i < n; i++)
            {
                int k = n - i;
                double lamD = 2.0 - 2.0 * Unity.Mathematics.math.cos(k * Unity.Mathematics.math.PI_DBL / (n + 1));
                AssertClose(eig[i], (float)lamD, tol);
            }

            AssertDescending(in eig, n);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // eigenSymmetric tests (tred2 Householder + tql2 implicit-shift QL)
        // ---------------------------------------------------------------------

        // n=5 identity: same oracle as EigenIdentity; tred2/tql2 variant, A is DESTROYED.
        public void EsymIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;

            var A = arena.floatIdentityMat(n);
            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            bool ok = Eigen.symmetric(ref A, ref eig, ref V);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            for (int i = 0; i < n; i++)
                AssertClose(eig[i], (float)1, (float)100 * Consts.floatZeroThreshold);

            AssertDescending(in eig, n);

            Assert.IsTrue(Analysis.isOrthogonal(V, (float)100 * Consts.floatZeroThreshold));

            arena.Dispose();
        }

        // diag(3, -2, 0.5, 5, -7): eigenvalues == diagonal, sorted descending -> (5, 3, 0.5, -2, -7)
        // (same oracle as EigenDiagonal). V is a permutation/sign variant of identity, so rather than
        // pin exact V we verify the decomposition reconstructs A = V*diag(eig)*V^T and that V is orthogonal.
        public void EsymDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;

            var A = arena.floatMat(n, n);
            A[0, 0] = (float)3;
            A[1, 1] = (float)(-2);
            A[2, 2] = (float)0.5;
            A[3, 3] = (float)5;
            A[4, 4] = (float)(-7);

            var Aorig = A.Copy();

            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            bool ok = Eigen.symmetric(ref A, ref eig, ref V);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            float tol = (float)100 * Consts.floatZeroThreshold;
            AssertClose(eig[0], (float)5, tol);
            AssertClose(eig[1], (float)3, tol);
            AssertClose(eig[2], (float)0.5, tol);
            AssertClose(eig[3], (float)(-2), tol);
            AssertClose(eig[4], (float)(-7), tol);

            AssertDescending(in eig, n);

            // V a permutation of identity -> check A = V diag(eig) V^T rather than exact V.
            AssertReconstruction(in Aorig, in V, in eig, n, (float)1000 * Consts.floatZeroThreshold);

            Assert.IsTrue(Analysis.isOrthogonal(V, tol));

            arena.Dispose();
        }

        // [[2,1],[1,2]]: same oracle as EigenKnown2x2 (eigenvalues 3, 1); sign-agnostic eigenvector check.
        public void EsymKnown2x2()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 2;

            var A = arena.floatMat(n, n);
            A[0, 0] = (float)2; A[0, 1] = (float)1;
            A[1, 0] = (float)1; A[1, 1] = (float)2;

            var Aorig = A.Copy();

            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            bool ok = Eigen.symmetric(ref A, ref eig, ref V);

            Assert.IsTrue(ok);

            AssertClose(eig[0], (float)3, (float)100 * Consts.floatZeroThreshold);
            AssertClose(eig[1], (float)1, (float)100 * Consts.floatZeroThreshold);

            AssertDescending(in eig, n);

            AssertEigenResidual(in Aorig, in V, in eig, n);

            Assert.IsTrue(Analysis.isOrthogonal(V, (float)100 * Consts.floatZeroThreshold));

            arena.Dispose();
        }

        // RECONSTRUCTION on random symmetric matrices (n=6, n=8): keep a copy of A before it is
        // destroyed, then assert ||A - V*diag(eig)*V^T|| small.
        public void EsymReconstruct()
        {
            ReconstructOne(6, 3310991);
            ReconstructOne(8, 7745213);
        }

        private void ReconstructOne(int n, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);

            var A = MakeRandomSymmetric(ref arena, n, seed);
            var Aorig = A.Copy();

            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            bool ok = Eigen.symmetric(ref A, ref eig, ref V);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            AssertReconstruction(in Aorig, in V, in eig, n, (float)1000 * Consts.floatZeroThreshold);

            arena.Dispose();
        }

        // ORTHOGONALITY on random symmetric matrices (n=6, n=8): ||V^T V - I|| small.
        public void EsymOrthogonality()
        {
            OrthogonalityOne(6, 5519027);
            OrthogonalityOne(8, 9081237);
        }

        private void OrthogonalityOne(int n, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);

            var A = MakeRandomSymmetric(ref arena, n, seed);

            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            bool ok = Eigen.symmetric(ref A, ref eig, ref V);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in V));

            // Explicit ||V^T V - I||_max check (Analysis.isOrthogonal also asserted for parity).
            float precision = (float)1000 * Consts.floatZeroThreshold;
            float maxErr = (float)0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    float dot = (float)0;
                    for (int k = 0; k < n; k++)
                        dot += V[k, i] * V[k, j];
                    float expected = (i == j) ? (float)1 : (float)0;
                    float err = Unity.Mathematics.math.abs(dot - expected);
                    if (err > maxErr)
                        maxErr = err;
                }
            if (!(maxErr <= precision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = maxErr;
                Fail[2] = precision;
                Fail[3] = maxErr - precision;
            }
            Assert.IsTrue(maxErr <= precision);

            Assert.IsTrue(Analysis.isOrthogonal(V, precision));

            arena.Dispose();
        }

        // EIGENPAIR residual on random symmetric matrices (n=6, n=8): for each i,
        // ||A*V[:,i] - lambda_i*V[:,i]|| small (using the saved copy of A).
        public void EsymEigenpair()
        {
            EigenpairOne(6, 2240881);
            EigenpairOne(8, 6612553);
        }

        private void EigenpairOne(int n, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);

            var A = MakeRandomSymmetric(ref arena, n, seed);
            var Aorig = A.Copy();

            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            bool ok = Eigen.symmetric(ref A, ref eig, ref V);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            AssertDescending(in eig, n);
            AssertEigenResidual(in Aorig, in V, in eig, n);

            arena.Dispose();
        }

        // CROSS-CHECK eigenvalues vs the trusted values-only eigenvaluesSymmetric on the SAME
        // random symmetric matrices (n=6, n=8). Both DESTROY their input and sort descending, so
        // run each on a separate copy and compare elementwise.
        public void EsymCrossCheck()
        {
            CrossCheckValuesOne(6, 4456121);
            CrossCheckValuesOne(8, 8123779);
        }

        private void CrossCheckValuesOne(int n, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);

            var A = MakeRandomSymmetric(ref arena, n, seed);

            var Asym = A.Copy();   // destroyed by eigenSymmetric
            var Aval = A.Copy();   // destroyed by eigenvaluesSymmetric

            var eigSym = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            bool symOk = Eigen.symmetric(ref Asym, ref eigSym, ref V);
            Assert.IsTrue(symOk);

            var eigVal = arena.floatVec(n);
            bool valOk = Eigen.valuesSymmetric(ref Aval, ref eigVal);
            Assert.IsTrue(valOk);

            Assert.IsFalse(Analysis.isAnyNan(in eigSym));
            AssertDescending(in eigSym, n);

            for (int i = 0; i < n; i++)
            {
                float scale = (float)1 + Unity.Mathematics.math.abs(eigVal[i]);
                float tol = (float)1000 * Consts.floatZeroThreshold * scale;
                float diff = Unity.Mathematics.math.abs(eigSym[i] - eigVal[i]);
                if (!(diff <= tol) && Fail[0] == (float)0)
                {
                    Fail[0] = (float)1;
                    Fail[1] = eigSym[i];
                    Fail[2] = eigVal[i];
                    Fail[3] = diff;
                }
                Assert.IsTrue(diff <= tol);
            }

            arena.Dispose();
        }

        // n=6 1D-Laplacian: same known-answer oracle as EvSymLaplacian (lambda_k = 2-2cos(k*pi/(n+1))).
        // Also verifies the eigenpairs and orthogonality of the computed V.
        public void EsymLaplacian()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;

            var A = arena.floatMat(n, n);
            for (int i = 0; i < n; i++)
            {
                A[i, i] = (float)2;
                if (i + 1 < n)
                {
                    A[i, i + 1] = (float)(-1);
                    A[i + 1, i] = (float)(-1);
                }
            }

            var Aorig = A.Copy();

            var eig = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            bool ok = Eigen.symmetric(ref A, ref eig, ref V);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            float tol = (float)1000 * Consts.floatZeroThreshold;
            for (int i = 0; i < n; i++)
            {
                int k = n - i;
                double lamD = 2.0 - 2.0 * Unity.Mathematics.math.cos(k * Unity.Mathematics.math.PI_DBL / (n + 1));
                AssertClose(eig[i], (float)lamD, tol);
            }

            AssertDescending(in eig, n);
            AssertEigenResidual(in Aorig, in V, in eig, n);
            Assert.IsTrue(Analysis.isOrthogonal(V, tol));

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // Helpers
        // ---------------------------------------------------------------------

        // Allocate a random matrix (entries ~ +-5) and symmetrize it in place.
        private floatMxN MakeRandomSymmetric(ref Arena arena, int n, uint seed)
        {
            var A = arena.floatRandomMat(n, n, (float)(-5), (float)5, seed);
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    float avg = (A[i, j] + A[j, i]) * (float)0.5;
                    A[i, j] = avg;
                    A[j, i] = avg;
                }
            return A;
        }

        // Reconstruct recon = V*diag(eig)*V^T element-by-element and assert ||A - recon||_max small.
        // No arena allocation (caller's arena is busy with A copy); uses a stack-free triple loop.
        private void AssertReconstruction(in floatMxN A, in floatMxN V, in floatN eig, int n, float precision)
        {
            float maxErr = (float)0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    float sum = (float)0;
                    for (int k = 0; k < n; k++)
                        sum += V[i, k] * eig[k] * V[j, k];
                    float err = Unity.Mathematics.math.abs(sum - A[i, j]);
                    if (err > maxErr)
                        maxErr = err;
                }
            if (!(maxErr <= precision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = maxErr;
                Fail[2] = precision;
                Fail[3] = maxErr - precision;
            }
            Assert.IsTrue(maxErr <= precision);
        }

        // For every eigenpair (lambda_k = eig[k], v_k = column k of V), assert
        // ||A*v_k - lambda_k*v_k||_inf <= 1000*ZeroThreshold * (1 + |lambda_k|).
        private void AssertEigenResidual(in floatMxN A, in floatMxN V, in floatN eig, int n)
        {
            for (int k = 0; k < n; k++)
            {
                float lambda = eig[k];
                float maxRes = (float)0;
                for (int i = 0; i < n; i++)
                {
                    float av = (float)0;
                    for (int j = 0; j < n; j++)
                        av += A[i, j] * V[j, k];
                    float ri = Unity.Mathematics.math.abs(av - lambda * V[i, k]);
                    if (ri > maxRes)
                        maxRes = ri;
                }
                float tol = (float)1000 * Consts.floatZeroThreshold * ((float)1 + Unity.Mathematics.math.abs(lambda));
                if (!(maxRes <= tol) && Fail[0] == (float)0)
                {
                    Fail[0] = (float)1;
                    Fail[1] = maxRes;
                    Fail[2] = tol;
                    Fail[3] = (float)k;
                }
                Assert.IsTrue(maxRes <= tol);
            }
        }

        // Recompute residual r = A*v - lambda*v (inf-norm) and assert it satisfies the
        // documented convergence criterion r <= tol * max(1, |lambda|).
        private void AssertPowerResidual(in floatMxN A, in floatN v, float lambda, float tol, int n)
        {
            float maxRes = (float)0;
            for (int i = 0; i < n; i++)
            {
                float av = (float)0;
                for (int j = 0; j < n; j++)
                    av += A[i, j] * v[j];
                float ri = Unity.Mathematics.math.abs(av - lambda * v[i]);
                if (ri > maxRes)
                    maxRes = ri;
            }
            float scale = Unity.Mathematics.math.abs(lambda);
            if (scale < (float)1)
                scale = (float)1;
            float limit = tol * scale;
            if (!(maxRes <= limit) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = maxRes;
                Fail[2] = limit;
                Fail[3] = maxRes - limit;
            }
            Assert.IsTrue(maxRes <= limit);
        }

        // Eigenvalues descending by value: eig[i] <= eig[i-1] (+ slack).
        private void AssertDescending(in floatN eig, int n)
        {
            for (int i = 1; i < n; i++)
            {
                bool descending = eig[i] <= eig[i - 1] + (float)100 * Consts.floatZeroThreshold;
                if (!descending && Fail[0] == (float)0)
                {
                    Fail[0] = (float)1;
                    Fail[1] = eig[i];
                    Fail[2] = eig[i - 1];
                    Fail[3] = (float)i;
                }
                Assert.IsTrue(descending);
            }
        }

        private void AssertClose(float a, float b, float precision)
        {
            float diff = Unity.Mathematics.math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        private void AssertFinite(float v)
        {
            if (!Unity.Mathematics.math.isfinite(v) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = v;
                Fail[2] = (float)0;
                Fail[3] = (float)0;
            }
            Assert.IsTrue(Unity.Mathematics.math.isfinite(v));
        }

        // Boolean-condition assert with a distinguishing `code` recorded in Fail[1] (so a silent
        // Burst abort is still diagnosable). Mirrors floatSparseEigenTests.AssertTrue; used by the
        // EigenSolveInfo field checks, where the value under test is an enum/bool, not a magnitude.
        private void AssertTrue(bool cond, float code)
        {
            if (!cond && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = code;
                Fail[2] = (float)0;
                Fail[3] = (float)0;
            }
            Assert.IsTrue(cond);
        }

    }

    public static Array GetEnums() {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void EigenSolverTests(TestJob.TestType type)
    {
        var fail = new NativeArray<float>(4, Allocator.TempJob);
        try {
            new TestJob() { Type = type, Fail = fail }.Run();
            // Under Burst a failed in-job assert logs an exception and aborts the job without
            // throwing to the caller - surface the recorded diagnostics here as well.
            if (fail[0] != (float)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");

        }
        catch (Exception e) {
            if (fail[0] != (float)0)
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

        var A = arena.floatMat(3, 4);
        var eig = arena.floatVec(4);
        var V = arena.floatMat(4, 4);

        Assert.Catch<ArgumentException>(() => Eigen.decompInPlace(ref A, ref eig, ref V));

        arena.Dispose();
    }

    [Test]
    public void EigenThrowsOnWrongEigenvalueLength()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.floatMat(4, 4);
        var eig = arena.floatVec(3);
        var V = arena.floatMat(4, 4);

        Assert.Catch<ArgumentException>(() => Eigen.decompInPlace(ref A, ref eig, ref V));

        arena.Dispose();
    }

    [Test]
    public void EigenThrowsOnWrongVShape()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.floatMat(4, 4);
        var eig = arena.floatVec(4);
        var V = arena.floatMat(3, 3);

        Assert.Catch<ArgumentException>(() => Eigen.decompInPlace(ref A, ref eig, ref V));

        arena.Dispose();
    }

    [Test]
    public void EigenThrowsOnBadMaxSweeps()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.floatMat(4, 4);
        var eig = arena.floatVec(4);
        var V = arena.floatMat(4, 4);

        Assert.Catch<ArgumentException>(() => Eigen.decompInPlace(ref A, ref eig, ref V, 0));

        arena.Dispose();
    }

    [Test]
    public void EigenThrowsOnNonSymmetric()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.floatMat(2, 2);
        A[0, 0] = (float)1; A[0, 1] = (float)2;
        A[1, 0] = (float)0; A[1, 1] = (float)1;

        var eig = arena.floatVec(2);
        var V = arena.floatMat(2, 2);

        Assert.Catch<ArgumentException>(() => Eigen.decompInPlace(ref A, ref eig, ref V));

        arena.Dispose();
    }

    [Test]
    public void EvSymThrowsOnNonSymmetric()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.floatMat(2, 2);
        A[0, 0] = (float)1; A[0, 1] = (float)2;
        A[1, 0] = (float)0; A[1, 1] = (float)1;

        var eig = arena.floatVec(2);

        Assert.Catch<ArgumentException>(() => Eigen.valuesSymmetric(ref A, ref eig));

        arena.Dispose();
    }

    [Test]
    public void EvSymThrowsOnNonSquare()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.floatMat(3, 4);
        var eig = arena.floatVec(4);

        Assert.Catch<ArgumentException>(() => Eigen.valuesSymmetric(ref A, ref eig));

        arena.Dispose();
    }

    [Test]
    public void EvSymThrowsOnWrongEigenvalueLength()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.floatMat(4, 4);
        var eig = arena.floatVec(3);

        Assert.Catch<ArgumentException>(() => Eigen.valuesSymmetric(ref A, ref eig));

        arena.Dispose();
    }

    [Test]
    public void EsymThrowsOnNonSquare()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.floatMat(3, 4);
        var eig = arena.floatVec(4);
        var V = arena.floatMat(4, 4);

        Assert.Catch<ArgumentException>(() => Eigen.symmetric(ref A, ref eig, ref V));

        arena.Dispose();
    }

    [Test]
    public void EsymThrowsOnNonSymmetric()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.floatMat(2, 2);
        A[0, 0] = (float)1; A[0, 1] = (float)2;
        A[1, 0] = (float)0; A[1, 1] = (float)1;

        var eig = arena.floatVec(2);
        var V = arena.floatMat(2, 2);

        Assert.Catch<ArgumentException>(() => Eigen.symmetric(ref A, ref eig, ref V));

        arena.Dispose();
    }

    [Test]
    public void EsymThrowsOnWrongEigenvalueLength()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.floatMat(4, 4);
        var eig = arena.floatVec(3);
        var V = arena.floatMat(4, 4);

        Assert.Catch<ArgumentException>(() => Eigen.symmetric(ref A, ref eig, ref V));

        arena.Dispose();
    }

    [Test]
    public void EsymThrowsOnWrongVShape()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.floatMat(4, 4);
        var eig = arena.floatVec(4);
        var V = arena.floatMat(3, 3);

        Assert.Catch<ArgumentException>(() => Eigen.symmetric(ref A, ref eig, ref V));

        arena.Dispose();
    }

    [Test]
    public void PowerThrowsOnNonSquare()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.floatMat(3, 4);
        var v = arena.floatVec(4);
        var w = arena.floatVec(4);

        Assert.Catch<ArgumentException>(() =>
            Eigen.powerIteration(in A, ref v, ref w, out float lambda, Consts.floatZeroThreshold, 1000));

        arena.Dispose();
    }

    [Test]
    public void PowerThrowsOnWrongVLength()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.floatMat(4, 4);
        var v = arena.floatVec(3);
        var w = arena.floatVec(4);

        Assert.Catch<ArgumentException>(() =>
            Eigen.powerIteration(in A, ref v, ref w, out float lambda, Consts.floatZeroThreshold, 1000));

        arena.Dispose();
    }

    [Test]
    public void PowerThrowsOnWrongWLength()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.floatMat(4, 4);
        var v = arena.floatVec(4);
        var w = arena.floatVec(3);

        Assert.Catch<ArgumentException>(() =>
            Eigen.powerIteration(in A, ref v, ref w, out float lambda, Consts.floatZeroThreshold, 1000));

        arena.Dispose();
    }

    [Test]
    public void PowerThrowsOnBadMaxIter()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.floatMat(4, 4);
        var v = arena.floatVec(4);
        var w = arena.floatVec(4);

        Assert.Catch<ArgumentException>(() =>
            Eigen.powerIteration(in A, ref v, ref w, out float lambda, Consts.floatZeroThreshold, 0));

        arena.Dispose();
    }

}
