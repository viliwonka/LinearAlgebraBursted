using System;
#pragma warning disable 618 // intentionally exercises the deprecated cyclic-Jacobi Eigen.decompInPlace (kept for reference)

using BULA;
using BULA.Gallery;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using UnityEngine;
using UnityEngine.TestTools;

public class fProxyEigenTests
{

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            // Eigen.decompInPlace
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
            // Eigen.valuesSymmetricInPlace
            EvSymIdentity,
            EvSymDiagonal,
            EvSymKnown2x2,
            EvSymN1,
            EvSymCrossCheckJacobi,
            EvSymLaplacian,
            // Eigen.symmetricInPlace (tred2 + tql2 full decomposition)
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
        public NativeArray<fProxy> Fail;

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
        // Eigen.decompInPlace tests
        // ---------------------------------------------------------------------

        // 4x4 identity: every eigenvalue == 1, V orthogonal. Exact closed form, so
        // eigenvalue tolerance 100*ZeroThreshold is comfortably above float Jacobi noise.
        public void EigenIdentity()
        {
            int n = 4;

            var A = GenerateOP.fProxyIdentityMat(n);
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);

            Assert.IsTrue(converged);

            for (int i = 0; i < n; i++)
                AssertClose(eig[i], (fProxy)1, (fProxy)100 * Consts.fProxyZeroThreshold);

            Assert.IsTrue(Analysis.isOrthogonal(V, (fProxy)100 * Consts.fProxyZeroThreshold));
        }

        // diag(3, -2, 0.5, 5): eigenvalues are the diagonal, sorted DESCENDING BY VALUE
        // -> (5, 3, 0.5, -2). V orthogonal. Diagonal input is exact, tolerance generous.
        public void EigenDiagonal()
        {
            int n = 4;

            var A = new fProxyMxN(n, n, Allocator.Temp);
            A[0, 0] = (fProxy)3;
            A[1, 1] = (fProxy)(-2);
            A[2, 2] = (fProxy)0.5;
            A[3, 3] = (fProxy)5;

            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);

            Assert.IsTrue(converged);

            AssertClose(eig[0], (fProxy)5, (fProxy)100 * Consts.fProxyZeroThreshold);
            AssertClose(eig[1], (fProxy)3, (fProxy)100 * Consts.fProxyZeroThreshold);
            AssertClose(eig[2], (fProxy)0.5, (fProxy)100 * Consts.fProxyZeroThreshold);
            AssertClose(eig[3], (fProxy)(-2), (fProxy)100 * Consts.fProxyZeroThreshold);

            AssertDescending(in eig, n);

            Assert.IsTrue(Analysis.isOrthogonal(V, (fProxy)100 * Consts.fProxyZeroThreshold));
        }

        // [[2,1],[1,2]]: eigenvalues 3 (vector (1,1)/sqrt2) and 1 (vector (1,-1)/sqrt2).
        // Sign-agnostic: assert A_orig * v_k ~= lambda_k * v_k for each column.
        public void EigenKnown2x2()
        {
            int n = 2;

            var A = new fProxyMxN(n, n, Allocator.Temp);
            A[0, 0] = (fProxy)2; A[0, 1] = (fProxy)1;
            A[1, 0] = (fProxy)1; A[1, 1] = (fProxy)2;

            var Aorig = new fProxyMxN(in A, Allocator.Temp);

            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);

            Assert.IsTrue(converged);

            AssertClose(eig[0], (fProxy)3, (fProxy)100 * Consts.fProxyZeroThreshold);
            AssertClose(eig[1], (fProxy)1, (fProxy)100 * Consts.fProxyZeroThreshold);

            AssertDescending(in eig, n);

            // sign-agnostic eigenvector verification: ||A*v_k - lambda_k*v_k||_inf small
            AssertEigenResidual(in Aorig, in V, in eig, n);

            // V orthogonal
            Assert.IsTrue(Analysis.isOrthogonal(V, (fProxy)100 * Consts.fProxyZeroThreshold));
        }

        // 8x8 random symmetric (values ~ +-5). Check: converged, V orthogonal, eigenvalues
        // descending, per-eigenpair residual small (scaled by (1+|lambda|)), trace == sum lambda.
        // Residual/orthogonality tolerance scaled by matrix magnitude: 8x8 entries up to 5,
        // float Jacobi residual ~ few * 1e-5 absolute -> 1000*ZeroThreshold*(1+|lambda|).
        public void EigenRandomSymmetric()
        {
            int n = 8;

            var A = GenerateOP.fProxyRandomMat(n, n, (fProxy)(-5), (fProxy)5, 8123451);
            // symmetrize in place
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    fProxy avg = (A[i, j] + A[j, i]) * (fProxy)0.5;
                    A[i, j] = avg;
                    A[j, i] = avg;
                }

            var Aorig = new fProxyMxN(in A, Allocator.Temp);

            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);

            Assert.IsTrue(converged);

            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            Assert.IsTrue(Analysis.isOrthogonal(V, (fProxy)1000 * Consts.fProxyZeroThreshold));

            AssertDescending(in eig, n);

            AssertEigenResidual(in Aorig, in V, in eig, n);

            // trace(A_orig) == sum eigenvalues
            fProxy trace = (fProxy)0;
            for (int i = 0; i < n; i++)
                trace += Aorig[i, i];
            fProxy sumEig = (fProxy)0;
            for (int i = 0; i < n; i++)
                sumEig += eig[i];
            // trace magnitude up to ~8*5 = 40; allow magnitude-scaled tolerance.
            AssertClose(trace, sumEig, (fProxy)1000 * Consts.fProxyZeroThreshold);
        }

        // Same setup as EigenRandomSymmetric (different seed): reconstruct V*diag(lambda)*V^T
        // and compare to A_orig elementwise. Reconstruction error for float Jacobi on an
        // 8x8 with entries up to ~5 lands around 1e-5..1e-4 absolute -> 1000*ZeroThreshold.
        public void EigenReconstruct()
        {
            int n = 8;

            var A = GenerateOP.fProxyRandomMat(n, n, (fProxy)(-5), (fProxy)5, 5571903);
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    fProxy avg = (A[i, j] + A[j, i]) * (fProxy)0.5;
                    A[i, j] = avg;
                    A[j, i] = avg;
                }

            var Aorig = new fProxyMxN(in A, Allocator.Temp);

            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);

            Assert.IsTrue(converged);

            // Reconstruct: recon = V * diag(eig) * V^T
            var diagE = GenerateOP.fProxyDiagonalMat(in eig);
            var Vd = Blas.dot(V, diagE);
            var Vt = Blas.trans(V);
            var recon = Blas.dot(Vd, Vt);

            var shouldBeZero = new fProxyMxN(in Aorig, Allocator.Temp);
            fProxyComp.subInPlace(shouldBeZero, recon);

            if (Analysis.isAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            fProxy precision = (fProxy)1000 * Consts.fProxyZeroThreshold;
            fProxy zeroError = Analysis.MaxZeroError(shouldBeZero);
            if (!(zeroError <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = zeroError;
                Fail[2] = precision;
                Fail[3] = zeroError - precision;
            }
            Assert.IsTrue(Analysis.isZero(in shouldBeZero, precision));
        }

        // 6x6 PSD matrix A = B^T B. Eigenvalues must all be >= -tol and equal the singular
        // values of A (which for symmetric PSD equal the eigenvalues) in the same descending
        // order. Eigen.decompInPlace destroys its input, so copy; SVD.values takes A `in`
        // (preserved), so no copy is needed for the SVD side.
        // A = B^T B with B entries ~ +-3 -> eigenvalues up to ~ order 100; scale tolerance.
        public void EigenPSDvsSVD()
        {
            int n = 6;

            var B = GenerateOP.fProxyRandomMat(n, n, (fProxy)(-3), (fProxy)3, 9920017);

            // A = B^T B (manual), symmetric PSD
            var A = new fProxyMxN(n, n, Allocator.Temp);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    fProxy sum = (fProxy)0;
                    for (int k = 0; k < n; k++)
                        sum += B[k, i] * B[k, j];
                    A[i, j] = sum;
                }
            // exact symmetrize to kill any rounding asymmetry
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    fProxy avg = (A[i, j] + A[j, i]) * (fProxy)0.5;
                    A[i, j] = avg;
                    A[j, i] = avg;
                }

            var Aeig = new fProxyMxN(in A, Allocator.Temp);   // destroyed by Eigen.decompInPlace

            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            bool converged = Eigen.decompInPlace(ref Aeig, ref eig, ref V);
            Assert.IsTrue(converged);

            // eigenvalues all >= -tol (PSD)
            fProxy negTol = (fProxy)1000 * Consts.fProxyZeroThreshold;
            for (int i = 0; i < n; i++)
            {
                bool nonNeg = eig[i] >= -negTol;
                if (!nonNeg && Fail[0] == (fProxy)0)
                {
                    Fail[0] = (fProxy)1;
                    Fail[1] = eig[i];
                    Fail[2] = -negTol;
                    Fail[3] = (fProxy)i;
                }
                Assert.IsTrue(nonNeg);
            }

            // singular values via SVD.values on the untouched A (preserved, no copy needed)
            var S = new fProxyN(n, Allocator.Temp);
            bool svdOk = SVD.values(in A, ref S);
            Assert.IsTrue(svdOk);

            // Compare eigenvalues to singular values, same descending order.
            // Magnitude can reach ~ order 100, so scale tolerance by (1+|S[i]|).
            for (int i = 0; i < n; i++)
            {
                fProxy scale = (fProxy)1 + Unity.Mathematics.math.abs(S[i]);
                fProxy tol = (fProxy)1000 * Consts.fProxyZeroThreshold * scale;
                fProxy diff = Unity.Mathematics.math.abs(eig[i] - S[i]);
                if (!(diff <= tol) && Fail[0] == (fProxy)0)
                {
                    Fail[0] = (fProxy)1;
                    Fail[1] = eig[i];
                    Fail[2] = S[i];
                    Fail[3] = diff;
                }
                Assert.IsTrue(diff <= tol);
            }        }

        // 5x5 zero matrix: converged, all eigenvalues 0, V orthogonal (stays identity).
        public void EigenZero()
        {
            int n = 5;

            var A = new fProxyMxN(n, n, Allocator.Temp);
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);

            Assert.IsTrue(converged);

            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            for (int i = 0; i < n; i++)
                AssertClose(eig[i], (fProxy)0, (fProxy)100 * Consts.fProxyZeroThreshold);

            Assert.IsTrue(Analysis.isOrthogonal(V, (fProxy)100 * Consts.fProxyZeroThreshold));
        }

        // Rank-1 projection A = v*vᵀ (v = (1,2,3,1)): SINGULAR symmetric matrix whose eigenvalues
        // are exactly {‖v‖² = 15, 0, 0, 0}. Tests a genuine zero eigenvalue ALONGSIDE a nonzero one
        // (the realistic rank-deficient eigen case — distinct from the all-zero EigenZero). Checks
        // the dominant eigenvalue, the exact-zero tail, descending order, reconstruction, and that
        // the trailing (null-space) eigenvectors still form an orthonormal V.
        public void EigenRank1Projection()
        {
            int n = 4;

            var v = new fProxyN(n, Allocator.Temp);
            v[0] = (fProxy)1; v[1] = (fProxy)2; v[2] = (fProxy)3; v[3] = (fProxy)1;
            fProxy vv = (fProxy)0;
            for (int i = 0; i < n; i++) vv += v[i] * v[i]; // = 15

            var A = new fProxyMxN(n, n, Allocator.Temp);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = v[i] * v[j];

            var Aorig = new fProxyMxN(in A, Allocator.Temp);
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);
            Assert.IsTrue(converged);
            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            fProxy tol = (fProxy)100 * Consts.fProxyZeroThreshold;
            // dominant eigenvalue == ‖v‖² = 15; the other three are exactly zero.
            AssertClose(eig[0], vv, tol * ((fProxy)1 + vv));
            for (int i = 1; i < n; i++)
                AssertClose(eig[i], (fProxy)0, tol * ((fProxy)1 + vv));

            AssertDescending(in eig, n);
            AssertEigenResidual(in Aorig, in V, in eig, n);
            Assert.IsTrue(Analysis.isOrthogonal(V, tol));
        }

        // Triangle-graph Laplacian L = [[2,-1,-1],[-1,2,-1],[-1,-1,2]]: a classic SINGULAR symmetric
        // matrix with exact eigenvalues {3, 3, 0} (the 0 is the all-ones null vector; rank 2). A
        // known literature vector exercising a zero eigenvalue plus a repeated nonzero one.
        public void EigenLaplacianSingular()
        {
            int n = 3;

            var A = new fProxyMxN(n, n, Allocator.Temp);
            A[0, 0] = (fProxy)2; A[0, 1] = (fProxy)(-1); A[0, 2] = (fProxy)(-1);
            A[1, 0] = (fProxy)(-1); A[1, 1] = (fProxy)2; A[1, 2] = (fProxy)(-1);
            A[2, 0] = (fProxy)(-1); A[2, 1] = (fProxy)(-1); A[2, 2] = (fProxy)2;

            var Aorig = new fProxyMxN(in A, Allocator.Temp);
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);
            Assert.IsTrue(converged);
            Assert.IsFalse(Analysis.isAnyNan(in eig));

            fProxy tol = (fProxy)100 * Consts.fProxyZeroThreshold;
            AssertClose(eig[0], (fProxy)3, tol * (fProxy)4);
            AssertClose(eig[1], (fProxy)3, tol * (fProxy)4);
            AssertClose(eig[2], (fProxy)0, tol * (fProxy)4); // singular: smallest eigenvalue is 0

            AssertDescending(in eig, n);
            AssertEigenResidual(in Aorig, in V, in eig, n);
            Assert.IsTrue(Analysis.isOrthogonal(V, tol));
        }

        // GALLERY KNOWN-ANSWER (Gallery.Special): n=5 Clement matrix — symmetric tridiagonal with
        // zero diagonal whose eigenvalues are EXACTLY the integer-spaced set {n-1, n-3, ..., -(n-1)}
        // = {4, 2, 0, -2, -4} for n=5 (symmetric about 0, trace 0). Well-separated spectrum, so a
        // 1000*ZeroThreshold absolute tolerance comfortably covers float Jacobi noise.
        public void EigenClement()
        {
            int n = 5;

            var A = fProxyGallery.fProxyClement(n);
            var Aorig = new fProxyMxN(in A, Allocator.Temp);

            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);
            Assert.IsTrue(converged);
            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            fProxy tol = (fProxy)1000 * Consts.fProxyZeroThreshold;
            AssertClose(eig[0], (fProxy)4, tol);
            AssertClose(eig[1], (fProxy)2, tol);
            AssertClose(eig[2], (fProxy)0, tol);
            AssertClose(eig[3], (fProxy)(-2), tol);
            AssertClose(eig[4], (fProxy)(-4), tol);

            AssertDescending(in eig, n);
            AssertEigenResidual(in Aorig, in V, in eig, n);
            Assert.IsTrue(Analysis.isOrthogonal(V, tol));
        }

        // GALLERY KNOWN-ANSWER (Gallery.Special): n=5 Fiedler distance matrix F[i,j]=|i-j|. Known
        // inertia: EXACTLY ONE positive eigenvalue and n-1 negative ones. For n=5 the spectrum is
        // {8.288, -0.558, -0.764, -1.730, -5.236}; the smallest gap from 0 is ~0.558, so a 1E-2 band
        // cleanly separates the signs while staying far above float Jacobi noise. Descending order
        // means the single positive value lands at eig[0].
        public void EigenFiedler()
        {
            int n = 5;

            var A = fProxyGallery.fProxyFiedler(n);
            var Aorig = new fProxyMxN(in A, Allocator.Temp);

            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);
            Assert.IsTrue(converged);
            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            fProxy band = (fProxy)1E-2f;

            // The single positive eigenvalue is the largest (descending) -> eig[0] > band.
            bool topPos = eig[0] > band;
            if (!topPos && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = eig[0];
                Fail[2] = band;
                Fail[3] = (fProxy)0;
            }
            Assert.IsTrue(topPos);

            // The remaining n-1 eigenvalues are all strictly negative.
            for (int i = 1; i < n; i++)
            {
                bool isNeg = eig[i] < -band;
                if (!isNeg && Fail[0] == (fProxy)0)
                {
                    Fail[0] = (fProxy)1;
                    Fail[1] = eig[i];
                    Fail[2] = -band;
                    Fail[3] = (fProxy)i;
                }
                Assert.IsTrue(isNeg);
            }

            AssertDescending(in eig, n);
            AssertEigenResidual(in Aorig, in V, in eig, n);
            Assert.IsTrue(Analysis.isOrthogonal(V, (fProxy)1000 * Consts.fProxyZeroThreshold));
        }

        // GALLERY KNOWN-ANSWER (Gallery.Special): n=5 DingDong symmetric Hankel matrix. Known
        // property: every eigenvalue lies strictly inside (-pi/2, pi/2), clustering near +-pi/2.
        // For n=5 the extreme eigenvalues are ~+-1.5707..., ~1.7e-6 below pi/2, so a small margin
        // absorbs Jacobi error while still asserting the bound is not exceeded.
        public void EigenDingDong()
        {
            int n = 5;

            var A = fProxyGallery.fProxyDingDong(n);
            var Aorig = new fProxyMxN(in A, Allocator.Temp);

            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            bool converged = Eigen.decompInPlace(ref A, ref eig, ref V);
            Assert.IsTrue(converged);
            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            fProxy half = (fProxy)(Unity.Mathematics.math.PI_DBL * 0.5);
            fProxy margin = (fProxy)1000 * Consts.fProxyZeroThreshold;

            for (int i = 0; i < n; i++)
            {
                bool inBand = eig[i] <= half + margin && eig[i] >= -half - margin;
                if (!inBand && Fail[0] == (fProxy)0)
                {
                    Fail[0] = (fProxy)1;
                    Fail[1] = eig[i];
                    Fail[2] = half;
                    Fail[3] = (fProxy)i;
                }
                Assert.IsTrue(inBand);
            }

            AssertDescending(in eig, n);
            AssertEigenResidual(in Aorig, in V, in eig, n);
            Assert.IsTrue(Analysis.isOrthogonal(V, margin));
        }

        // Hilbert-like symmetric matrix with maxSweeps = 1: regardless of returned bool,
        // outputs must be finite (no NaN) and eigenvalues descending.
        public void EigenNonConvergence()
        {
            int n = 8;

            var A = GenerateOP.fProxyHilbertMat(n);
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            // maxSweeps = 1: convergence not asserted.
            Eigen.decompInPlace(ref A, ref eig, ref V, 1);

            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            AssertDescending(in eig, n);
        }

        // ---------------------------------------------------------------------
        // powerIteration tests
        // ---------------------------------------------------------------------

        // diag(5, 3, 1, 0.5) with v = 0 input (exercises deterministic seeding) -> true,
        // lambda ~= 5 (dominant), residual property ||A*v - lambda*v||_inf <= tol*max(1,|lambda|).
        public void PowerDiagonalDominant()
        {
            int n = 4;

            var A = new fProxyMxN(n, n, Allocator.Temp);
            A[0, 0] = (fProxy)5;
            A[1, 1] = (fProxy)3;
            A[2, 2] = (fProxy)1;
            A[3, 3] = (fProxy)0.5;

            var v = new fProxyN(n, Allocator.Temp);   // zero vector -> deterministic seeding
            var w = new fProxyN(n, Allocator.Temp);

            fProxy tol = (fProxy)10 * Consts.fProxyZeroThreshold;
            bool ok = Eigen.powerIteration(in A, ref v, ref w, out fProxy lambda, tol, 1000);

            Assert.IsTrue(ok);
            AssertFinite(lambda);
            AssertClose(lambda, (fProxy)5, (fProxy)100 * Consts.fProxyZeroThreshold);

            AssertPowerResidual(in A, in v, lambda, tol, n);
        }

        // diag(-7, 2, 1): dominant BY MAGNITUDE is -7. lambda ~= -7, |v[0]| ~= 1 (e0 dir).
        public void PowerNegativeDominant()
        {
            int n = 3;

            var A = new fProxyMxN(n, n, Allocator.Temp);
            A[0, 0] = (fProxy)(-7);
            A[1, 1] = (fProxy)2;
            A[2, 2] = (fProxy)1;

            var v = new fProxyN(n, Allocator.Temp);
            var w = new fProxyN(n, Allocator.Temp);

            fProxy tol = (fProxy)10 * Consts.fProxyZeroThreshold;
            bool ok = Eigen.powerIteration(in A, ref v, ref w, out fProxy lambda, tol, 1000);

            Assert.IsTrue(ok);
            AssertFinite(lambda);
            AssertClose(lambda, (fProxy)(-7), (fProxy)100 * Consts.fProxyZeroThreshold);

            // eigenvector aligned with e0: |v[0]| ~= 1
            AssertClose(Unity.Mathematics.math.abs(v[0]), (fProxy)1, (fProxy)100 * Consts.fProxyZeroThreshold);

            AssertPowerResidual(in A, in v, lambda, tol, n);
        }

        // 6x6 random symmetric with a forced clear dominant eigenvalue (+12 boost on one
        // diagonal). Reference lambda_max from Eigen.decompInPlace on a copy. Power iteration
        // finds dominant BY MAGNITUDE; the boosted positive eigenvalue dominates both in
        // value and magnitude, so the reference is eig[0] (largest by value == largest |.|).
        public void PowerSymmetricCrossCheck()
        {
            int n = 6;

            var A = GenerateOP.fProxyRandomMat(n, n, (fProxy)(-4), (fProxy)4, 4471123);
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    fProxy avg = (A[i, j] + A[j, i]) * (fProxy)0.5;
                    A[i, j] = avg;
                    A[j, i] = avg;
                }
            // Force a clearly dominant positive eigenvalue (well separated in magnitude).
            A[0, 0] = A[0, 0] + (fProxy)12;

            var Apow = new fProxyMxN(in A, Allocator.Temp);
            var Aeig = new fProxyMxN(in A, Allocator.Temp);

            // reference: dominant eigenvalue by value (== by magnitude here, well separated)
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            bool econv = Eigen.decompInPlace(ref Aeig, ref eig, ref V);
            Assert.IsTrue(econv);

            // dominant by magnitude: compare |eig[0]| vs |eig[n-1]|
            fProxy reference = eig[0];
            if (Unity.Mathematics.math.abs(eig[n - 1]) > Unity.Mathematics.math.abs(eig[0]))
                reference = eig[n - 1];

            var v = new fProxyN(n, Allocator.Temp);
            var w = new fProxyN(n, Allocator.Temp);

            fProxy tol = (fProxy)10 * Consts.fProxyZeroThreshold;
            bool ok = Eigen.powerIteration(in Apow, ref v, ref w, out fProxy lambda, tol, 2000);

            Assert.IsTrue(ok);
            AssertFinite(lambda);

            // magnitude up to ~16; scale tolerance by (1+|reference|).
            fProxy scale = (fProxy)1 + Unity.Mathematics.math.abs(reference);
            AssertClose(lambda, reference, (fProxy)1000 * Consts.fProxyZeroThreshold * scale);
        }

        // 2x2 rotation [[0,-1],[1,0]] (eigenvalues +-i): power iteration cannot converge,
        // returns false; v finite, lambda finite (no NaN).
        public void PowerComplexPair()
        {
            int n = 2;

            var A = new fProxyMxN(n, n, Allocator.Temp);
            A[0, 0] = (fProxy)0; A[0, 1] = (fProxy)(-1);
            A[1, 0] = (fProxy)1; A[1, 1] = (fProxy)0;

            var v = new fProxyN(n, Allocator.Temp);
            var w = new fProxyN(n, Allocator.Temp);

            fProxy tol = (fProxy)10 * Consts.fProxyZeroThreshold;
            bool ok = Eigen.powerIteration(in A, ref v, ref w, out fProxy lambda, tol, 200);

            Assert.IsFalse(ok);
            AssertFinite(lambda);
            for (int i = 0; i < n; i++)
                AssertFinite(v[i]);
        }

        // 3x3 zero matrix: A*v == 0, ||w|| == 0 path -> lambda set to 0, returns true.
        public void PowerZeroMatrix()
        {
            int n = 3;

            var A = new fProxyMxN(n, n, Allocator.Temp);

            var v = new fProxyN(n, Allocator.Temp);
            var w = new fProxyN(n, Allocator.Temp);

            fProxy tol = (fProxy)10 * Consts.fProxyZeroThreshold;
            bool ok = Eigen.powerIteration(in A, ref v, ref w, out fProxy lambda, tol, 1000);

            Assert.IsTrue(ok);
            AssertFinite(lambda);
            AssertClose(lambda, (fProxy)0, (fProxy)100 * Consts.fProxyZeroThreshold);
        }

        // ---------------------------------------------------------------------
        // EigenSolveInfo field assertions (the NEW diagnostics struct the eigensolvers
        // now return alongside `out fProxy lambda`). These pin the FIELD VALUES, not just
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
            int n = 4;

            var A = new fProxyMxN(n, n, Allocator.Temp);
            A[0, 0] = (fProxy)5;
            A[1, 1] = (fProxy)3;
            A[2, 2] = (fProxy)1;
            A[3, 3] = (fProxy)0.5;

            var v = new fProxyN(n, Allocator.Temp);   // zero -> deterministic seeding
            var w = new fProxyN(n, Allocator.Temp);

            fProxy tol = (fProxy)10 * Consts.fProxyZeroThreshold;
            var info = Eigen.powerIteration(in A, ref v, ref w, out fProxy lambda, tol, 1000);

            // Converged, three equivalent ways (implicit bool, Solved property, status enum).
            AssertTrue((bool)info, (fProxy)1);
            AssertTrue(info.Solved, (fProxy)2);
            AssertTrue(info.status == IterativeSolveStatus.Converged, (fProxy)3);

            // residual is finite, non-negative, and satisfies the documented convergence bound
            // r <= tol*max(1,|lambda|). Reproduce the scale in fProxy exactly as the algorithm
            // does, then widen -- so this is bit-for-bit the criterion the Converged return used.
            AssertTrue(Unity.Mathematics.math.isfinite(info.residual), (fProxy)4);
            AssertTrue(info.residual >= 0.0, (fProxy)5);

            fProxy fscale = Unity.Mathematics.math.abs(lambda);
            if (fscale < (fProxy)1) fscale = (fProxy)1;
            double limit = (double)(tol * fscale);
            AssertTrue(info.residual <= limit, (fProxy)6);

            // Converged counts the converging iteration too -> iterations >= 1.
            AssertTrue(info.iterations >= 1, (fProxy)7);
        }

        // powerIteration MAX-ITERATIONS field check (deterministic non-convergence). The 2x2 real
        // rotation [[0,-1],[1,0]] has the complex conjugate pair +-i as its dominant eigenvalue, so
        // power iteration provably CANNOT converge (documented in powerIteration's XML notes). It
        // must exhaust maxIter and report MaxIterations -- assert !Solved, status == MaxIterations,
        // and iterations == maxIter exactly.
        public void PowerMaxIterationsInfo()
        {
            int n = 2;

            var A = new fProxyMxN(n, n, Allocator.Temp);
            A[0, 0] = (fProxy)0; A[0, 1] = (fProxy)(-1);
            A[1, 0] = (fProxy)1; A[1, 1] = (fProxy)0;

            var v = new fProxyN(n, Allocator.Temp);
            var w = new fProxyN(n, Allocator.Temp);

            int maxIter = 200;
            fProxy tol = (fProxy)10 * Consts.fProxyZeroThreshold;
            var info = Eigen.powerIteration(in A, ref v, ref w, out fProxy lambda, tol, maxIter);

            AssertTrue(!info, (fProxy)1);          // implicit bool: false
            AssertTrue(!info.Solved, (fProxy)2);
            AssertTrue(info.status == IterativeSolveStatus.MaxIterations, (fProxy)3);
            AssertTrue(info.iterations == maxIter, (fProxy)4);
            // residual on a MaxIterations return is still the finite last-iterate residual (NOT NaN).
            AssertTrue(Unity.Mathematics.math.isfinite(info.residual), (fProxy)5);
        }

        // inversePowerIteration BREAKDOWN field check (deterministic). A = diag(1,-1) is INDEFINITE
        // (not SPD), so the inner CG solve hits non-positive curvature (p.Ap < 0) on its very first
        // step and breaks down; inverse iteration then bails out reporting Breakdown. The
        // deterministic seed (1,2)/sqrt5 makes p.Ap = 1/5 - 4/5 = -3/5 < 0 with certainty. Assert
        // !Solved, status == Breakdown, and residual == NaN (the documented Breakdown residual).
        public void InversePowerBreakdownInfo()
        {
            int n = 2;

            var A = new fProxyMxN(n, n, Allocator.Temp);
            A[0, 0] = (fProxy)1; A[1, 1] = (fProxy)(-1);   // indefinite -> CG breakdown

            var v = new fProxyN(n, Allocator.Temp);   // zero -> deterministic (1,2) seeding

            var info = Eigen.inversePowerIteration(in A, ref v, out fProxy lambda);

            AssertTrue(!info, (fProxy)1);          // implicit bool: false
            AssertTrue(!info.Solved, (fProxy)2);
            AssertTrue(info.status == IterativeSolveStatus.Breakdown, (fProxy)3);
            // residual is double.NaN on a Breakdown return. Use the Burst-safe double isnan overload
            // (self-inequality residual != residual is an equally valid check under FloatMode.Default).
            AssertTrue(Unity.Mathematics.math.isnan(info.residual), (fProxy)4);
        }

        // inversePowerIteration CONVERGED-residual check on an SPD operator that converges. The 1D
        // Laplacian (tridiagonal 2,-1) is SPD with well-separated small eigenvalues, so inverse
        // iteration converges reliably (same fixture family the sparse InverseLaplacianCrossCheck
        // uses). Assert Solved, and that the reported residual is finite, non-negative, and bounded.
        // Inverse iteration's residual bottoms out near cgTol (the inner CG's floor, not machine
        // epsilon), so the bound is a generous cgTol-anchored multiple scaled by max(1,|lambda|) --
        // NOT the tight Consts.fProxyZeroThreshold bands the pure-matvec powerIteration uses. This
        // still catches an O(1)/NaN residual regression while staying above the honest cgTol floor.
        public void InversePowerConvergedInfo()
        {
            int n = 12;

            var A = fProxyGallery.fProxyLaplacian1D(n);

            // tol a multiple of cgTol (see inversePowerIteration's doc comment): consecutive
            // eigenpair estimates each come from a fresh CG solve accurate only to ~cgTol.
            fProxy cgTol = Consts.fProxySqrtEps;
            fProxy tol = (fProxy)10 * cgTol;

            var v = new fProxyN(n, Allocator.Temp);   // zero -> deterministic seeding

            var info = Eigen.inversePowerIteration(in A, ref v, out fProxy lambda, tol, 200, n, cgTol);

            AssertTrue(info.Solved, (fProxy)1);
            AssertTrue(info.status == IterativeSolveStatus.Converged, (fProxy)2);

            AssertTrue(Unity.Mathematics.math.isfinite(info.residual), (fProxy)3);
            AssertTrue(info.residual >= 0.0, (fProxy)4);

            // residual bottoms near cgTol; bound by a generous cgTol-anchored multiple (auto-scaled
            // per numeric type via Consts.fProxySqrtEps), scaled by max(1,|lambda|).
            fProxy fscale = Unity.Mathematics.math.abs(lambda);
            if (fscale < (fProxy)1) fscale = (fProxy)1;
            double limit = (double)((fProxy)5000 * cgTol * fscale);
            AssertTrue(info.residual <= limit, (fProxy)5);

            AssertTrue(info.iterations >= 1, (fProxy)6);
        }

        // ---------------------------------------------------------------------
        // Eigen.valuesSymmetricInPlace tests (Householder tridiagonalization + implicit-shift QL)
        // ---------------------------------------------------------------------

        // n=5 identity: same oracle as EigenIdentity (eigenvalues == 1); QL variant, A is DESTROYED.
        public void EvSymIdentity()
        {
            int n = 5;

            var A = GenerateOP.fProxyIdentityMat(n);
            var eig = new fProxyN(n, Allocator.Temp);

            bool ok = Eigen.valuesSymmetricInPlace(ref A, ref eig);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in eig));

            for (int i = 0; i < n; i++)
                AssertClose(eig[i], (fProxy)1, (fProxy)100 * Consts.fProxyZeroThreshold);

            AssertDescending(in eig, n);
        }

        // diag(3, -2, 0.5, 5, -7, 1): eigenvalues == diagonal, sorted descending -> (5, 3, 1, 0.5, -2, -7)
        // (same oracle as EigenDiagonal; Householder leaves the diagonal untouched). A is DESTROYED.
        public void EvSymDiagonal()
        {
            int n = 6;

            var A = new fProxyMxN(n, n, Allocator.Temp);
            A[0, 0] = (fProxy)3;
            A[1, 1] = (fProxy)(-2);
            A[2, 2] = (fProxy)0.5;
            A[3, 3] = (fProxy)5;
            A[4, 4] = (fProxy)(-7);
            A[5, 5] = (fProxy)1;

            var eig = new fProxyN(n, Allocator.Temp);

            bool ok = Eigen.valuesSymmetricInPlace(ref A, ref eig);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in eig));

            fProxy tol = (fProxy)100 * Consts.fProxyZeroThreshold;
            AssertClose(eig[0], (fProxy)5, tol);
            AssertClose(eig[1], (fProxy)3, tol);
            AssertClose(eig[2], (fProxy)1, tol);
            AssertClose(eig[3], (fProxy)0.5, tol);
            AssertClose(eig[4], (fProxy)(-2), tol);
            AssertClose(eig[5], (fProxy)(-7), tol);

            AssertDescending(in eig, n);
        }

        // [[2,1],[1,2]]: same oracle as EigenKnown2x2 (eigenvalues 3, 1). A is DESTROYED.
        public void EvSymKnown2x2()
        {
            int n = 2;

            var A = new fProxyMxN(n, n, Allocator.Temp);
            A[0, 0] = (fProxy)2; A[0, 1] = (fProxy)1;
            A[1, 0] = (fProxy)1; A[1, 1] = (fProxy)2;

            var eig = new fProxyN(n, Allocator.Temp);

            bool ok = Eigen.valuesSymmetricInPlace(ref A, ref eig);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in eig));

            AssertClose(eig[0], (fProxy)3, (fProxy)100 * Consts.fProxyZeroThreshold);
            AssertClose(eig[1], (fProxy)1, (fProxy)100 * Consts.fProxyZeroThreshold);

            AssertDescending(in eig, n);
        }

        // n=1 trivial: the sole eigenvalue equals the single entry (early-return path, no iteration).
        public void EvSymN1()
        {
            int n = 1;

            var A = new fProxyMxN(n, n, Allocator.Temp);
            A[0, 0] = (fProxy)(-3.25);

            var eig = new fProxyN(n, Allocator.Temp);

            bool ok = Eigen.valuesSymmetricInPlace(ref A, ref eig);

            Assert.IsTrue(ok);
            AssertClose(eig[0], (fProxy)(-3.25), (fProxy)100 * Consts.fProxyZeroThreshold);
        }

        // CROSS-CHECK vs the Jacobi Eigen.decompInPlace: for n=6 and n=8 random SYMMETRIC matrices,
        // run Eigen.decompInPlace on one copy and Eigen.valuesSymmetricInPlace on a SEPARATE copy (both
        // DESTROY their input, both sort descending) and require the eigenvalue vectors to agree.
        // Tolerance scaled by (1+|lambda|): entries ~ +-5, so float values land around few*1e-5.
        public void EvSymCrossCheckJacobi()
        {
            CrossCheckOne(6, 6610337);
            CrossCheckOne(8, 1277459);
        }

        private void CrossCheckOne(int n, uint seed)
        {
            var A = GenerateOP.fProxyRandomMat(n, n, (fProxy)(-5), (fProxy)5, seed);
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    fProxy avg = (A[i, j] + A[j, i]) * (fProxy)0.5;
                    A[i, j] = avg;
                    A[j, i] = avg;
                }

            var Ajac = new fProxyMxN(in A, Allocator.Temp);   // destroyed by Eigen.decompInPlace
            var Aql = new fProxyMxN(in A, Allocator.Temp);    // destroyed by Eigen.valuesSymmetricInPlace

            var eigJac = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            bool jacOk = Eigen.decompInPlace(ref Ajac, ref eigJac, ref V);
            Assert.IsTrue(jacOk);

            var eigQL = new fProxyN(n, Allocator.Temp);
            bool qlOk = Eigen.valuesSymmetricInPlace(ref Aql, ref eigQL);
            Assert.IsTrue(qlOk);

            Assert.IsFalse(Analysis.isAnyNan(in eigQL));
            AssertDescending(in eigQL, n);

            // both sorted descending -> compare elementwise.
            for (int i = 0; i < n; i++)
            {
                fProxy scale = (fProxy)1 + Unity.Mathematics.math.abs(eigJac[i]);
                fProxy tol = (fProxy)1000 * Consts.fProxyZeroThreshold * scale;
                fProxy diff = Unity.Mathematics.math.abs(eigQL[i] - eigJac[i]);
                if (!(diff <= tol) && Fail[0] == (fProxy)0)
                {
                    Fail[0] = (fProxy)1;
                    Fail[1] = eigQL[i];
                    Fail[2] = eigJac[i];
                    Fail[3] = diff;
                }
                Assert.IsTrue(diff <= tol);
            }        }

        // LITERATURE KNOWN-ANSWER: n=6 path-graph (1D Laplacian) tridiagonal with diag 2 and
        // off-diagonal -1. Eigenvalues are EXACTLY lambda_k = 2 - 2*cos(k*pi/(n+1)), k=1..n. Sorted
        // descending corresponds to k = n, n-1, ..., 1. Well-separated spectrum -> 1000*ZeroThreshold
        // absolute tolerance covers float QL noise.
        public void EvSymLaplacian()
        {
            int n = 6;

            var A = new fProxyMxN(n, n, Allocator.Temp);
            for (int i = 0; i < n; i++)
            {
                A[i, i] = (fProxy)2;
                if (i + 1 < n)
                {
                    A[i, i + 1] = (fProxy)(-1);
                    A[i + 1, i] = (fProxy)(-1);
                }
            }

            var eig = new fProxyN(n, Allocator.Temp);

            bool ok = Eigen.valuesSymmetricInPlace(ref A, ref eig);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in eig));

            fProxy tol = (fProxy)1000 * Consts.fProxyZeroThreshold;
            // descending order: eig[i] corresponds to k = n - i.
            for (int i = 0; i < n; i++)
            {
                int k = n - i;
                double lamD = 2.0 - 2.0 * Unity.Mathematics.math.cos(k * Unity.Mathematics.math.PI_DBL / (n + 1));
                AssertClose(eig[i], (fProxy)lamD, tol);
            }

            AssertDescending(in eig, n);
        }

        // ---------------------------------------------------------------------
        // Eigen.symmetricInPlace tests (tred2 Householder + tql2 implicit-shift QL)
        // ---------------------------------------------------------------------

        // n=5 identity: same oracle as EigenIdentity; tred2/tql2 variant, A is DESTROYED.
        public void EsymIdentity()
        {
            int n = 5;

            var A = GenerateOP.fProxyIdentityMat(n);
            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            bool ok = Eigen.symmetricInPlace(ref A, ref eig, ref V);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            for (int i = 0; i < n; i++)
                AssertClose(eig[i], (fProxy)1, (fProxy)100 * Consts.fProxyZeroThreshold);

            AssertDescending(in eig, n);

            Assert.IsTrue(Analysis.isOrthogonal(V, (fProxy)100 * Consts.fProxyZeroThreshold));
        }

        // diag(3, -2, 0.5, 5, -7): eigenvalues == diagonal, sorted descending -> (5, 3, 0.5, -2, -7)
        // (same oracle as EigenDiagonal). V is a permutation/sign variant of identity, so rather than
        // pin exact V we verify the decomposition reconstructs A = V*diag(eig)*V^T and that V is orthogonal.
        public void EsymDiagonal()
        {
            int n = 5;

            var A = new fProxyMxN(n, n, Allocator.Temp);
            A[0, 0] = (fProxy)3;
            A[1, 1] = (fProxy)(-2);
            A[2, 2] = (fProxy)0.5;
            A[3, 3] = (fProxy)5;
            A[4, 4] = (fProxy)(-7);

            var Aorig = new fProxyMxN(in A, Allocator.Temp);

            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            bool ok = Eigen.symmetricInPlace(ref A, ref eig, ref V);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            fProxy tol = (fProxy)100 * Consts.fProxyZeroThreshold;
            AssertClose(eig[0], (fProxy)5, tol);
            AssertClose(eig[1], (fProxy)3, tol);
            AssertClose(eig[2], (fProxy)0.5, tol);
            AssertClose(eig[3], (fProxy)(-2), tol);
            AssertClose(eig[4], (fProxy)(-7), tol);

            AssertDescending(in eig, n);

            // V a permutation of identity -> check A = V diag(eig) V^T rather than exact V.
            AssertReconstruction(in Aorig, in V, in eig, n, (fProxy)1000 * Consts.fProxyZeroThreshold);

            Assert.IsTrue(Analysis.isOrthogonal(V, tol));
        }

        // [[2,1],[1,2]]: same oracle as EigenKnown2x2 (eigenvalues 3, 1); sign-agnostic eigenvector check.
        public void EsymKnown2x2()
        {
            int n = 2;

            var A = new fProxyMxN(n, n, Allocator.Temp);
            A[0, 0] = (fProxy)2; A[0, 1] = (fProxy)1;
            A[1, 0] = (fProxy)1; A[1, 1] = (fProxy)2;

            var Aorig = new fProxyMxN(in A, Allocator.Temp);

            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            bool ok = Eigen.symmetricInPlace(ref A, ref eig, ref V);

            Assert.IsTrue(ok);

            AssertClose(eig[0], (fProxy)3, (fProxy)100 * Consts.fProxyZeroThreshold);
            AssertClose(eig[1], (fProxy)1, (fProxy)100 * Consts.fProxyZeroThreshold);

            AssertDescending(in eig, n);

            AssertEigenResidual(in Aorig, in V, in eig, n);

            Assert.IsTrue(Analysis.isOrthogonal(V, (fProxy)100 * Consts.fProxyZeroThreshold));
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
            var A = MakeRandomSymmetric(n, seed);
            var Aorig = new fProxyMxN(in A, Allocator.Temp);

            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            bool ok = Eigen.symmetricInPlace(ref A, ref eig, ref V);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            AssertReconstruction(in Aorig, in V, in eig, n, (fProxy)1000 * Consts.fProxyZeroThreshold);
        }

        // ORTHOGONALITY on random symmetric matrices (n=6, n=8): ||V^T V - I|| small.
        public void EsymOrthogonality()
        {
            OrthogonalityOne(6, 5519027);
            OrthogonalityOne(8, 9081237);
        }

        private void OrthogonalityOne(int n, uint seed)
        {
            var A = MakeRandomSymmetric(n, seed);

            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            bool ok = Eigen.symmetricInPlace(ref A, ref eig, ref V);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in V));

            // Explicit ||V^T V - I||_max check (Analysis.isOrthogonal also asserted for parity).
            fProxy precision = (fProxy)1000 * Consts.fProxyZeroThreshold;
            fProxy maxErr = (fProxy)0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    fProxy dot = (fProxy)0;
                    for (int k = 0; k < n; k++)
                        dot += V[k, i] * V[k, j];
                    fProxy expected = (i == j) ? (fProxy)1 : (fProxy)0;
                    fProxy err = Unity.Mathematics.math.abs(dot - expected);
                    if (err > maxErr)
                        maxErr = err;
                }
            if (!(maxErr <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = maxErr;
                Fail[2] = precision;
                Fail[3] = maxErr - precision;
            }
            Assert.IsTrue(maxErr <= precision);

            Assert.IsTrue(Analysis.isOrthogonal(V, precision));
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
            var A = MakeRandomSymmetric(n, seed);
            var Aorig = new fProxyMxN(in A, Allocator.Temp);

            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            bool ok = Eigen.symmetricInPlace(ref A, ref eig, ref V);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            AssertDescending(in eig, n);
            AssertEigenResidual(in Aorig, in V, in eig, n);
        }

        // CROSS-CHECK eigenvalues vs the trusted values-only Eigen.valuesSymmetricInPlace on the SAME
        // random symmetric matrices (n=6, n=8). Both DESTROY their input and sort descending, so
        // run each on a separate copy and compare elementwise.
        public void EsymCrossCheck()
        {
            CrossCheckValuesOne(6, 4456121);
            CrossCheckValuesOne(8, 8123779);
        }

        private void CrossCheckValuesOne(int n, uint seed)
        {
            var A = MakeRandomSymmetric(n, seed);

            var Asym = new fProxyMxN(in A, Allocator.Temp);   // destroyed by Eigen.symmetricInPlace
            var Aval = new fProxyMxN(in A, Allocator.Temp);   // destroyed by Eigen.valuesSymmetricInPlace

            var eigSym = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            bool symOk = Eigen.symmetricInPlace(ref Asym, ref eigSym, ref V);
            Assert.IsTrue(symOk);

            var eigVal = new fProxyN(n, Allocator.Temp);
            bool valOk = Eigen.valuesSymmetricInPlace(ref Aval, ref eigVal);
            Assert.IsTrue(valOk);

            Assert.IsFalse(Analysis.isAnyNan(in eigSym));
            AssertDescending(in eigSym, n);

            for (int i = 0; i < n; i++)
            {
                fProxy scale = (fProxy)1 + Unity.Mathematics.math.abs(eigVal[i]);
                fProxy tol = (fProxy)1000 * Consts.fProxyZeroThreshold * scale;
                fProxy diff = Unity.Mathematics.math.abs(eigSym[i] - eigVal[i]);
                if (!(diff <= tol) && Fail[0] == (fProxy)0)
                {
                    Fail[0] = (fProxy)1;
                    Fail[1] = eigSym[i];
                    Fail[2] = eigVal[i];
                    Fail[3] = diff;
                }
                Assert.IsTrue(diff <= tol);
            }        }

        // n=6 1D-Laplacian: same known-answer oracle as EvSymLaplacian (lambda_k = 2-2cos(k*pi/(n+1))).
        // Also verifies the eigenpairs and orthogonality of the computed V.
        public void EsymLaplacian()
        {
            int n = 6;

            var A = new fProxyMxN(n, n, Allocator.Temp);
            for (int i = 0; i < n; i++)
            {
                A[i, i] = (fProxy)2;
                if (i + 1 < n)
                {
                    A[i, i + 1] = (fProxy)(-1);
                    A[i + 1, i] = (fProxy)(-1);
                }
            }

            var Aorig = new fProxyMxN(in A, Allocator.Temp);

            var eig = new fProxyN(n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);

            bool ok = Eigen.symmetricInPlace(ref A, ref eig, ref V);

            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in eig));
            Assert.IsFalse(Analysis.isAnyNan(in V));

            fProxy tol = (fProxy)1000 * Consts.fProxyZeroThreshold;
            for (int i = 0; i < n; i++)
            {
                int k = n - i;
                double lamD = 2.0 - 2.0 * Unity.Mathematics.math.cos(k * Unity.Mathematics.math.PI_DBL / (n + 1));
                AssertClose(eig[i], (fProxy)lamD, tol);
            }

            AssertDescending(in eig, n);
            AssertEigenResidual(in Aorig, in V, in eig, n);
            Assert.IsTrue(Analysis.isOrthogonal(V, tol));
        }

        // ---------------------------------------------------------------------
        // Helpers
        // ---------------------------------------------------------------------

        // Allocate a random matrix (entries ~ +-5) and symmetrize it in place.
        private fProxyMxN MakeRandomSymmetric(int n, uint seed)
        {
            var A = GenerateOP.fProxyRandomMat(n, n, (fProxy)(-5), (fProxy)5, seed);
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    fProxy avg = (A[i, j] + A[j, i]) * (fProxy)0.5;
                    A[i, j] = avg;
                    A[j, i] = avg;
                }
            return A;
        }

        // Reconstruct recon = V*diag(eig)*V^T element-by-element and assert ||A - recon||_max small.
        // Uses a stack-free triple loop.
        private void AssertReconstruction(in fProxyMxN A, in fProxyMxN V, in fProxyN eig, int n, fProxy precision)
        {
            fProxy maxErr = (fProxy)0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    fProxy sum = (fProxy)0;
                    for (int k = 0; k < n; k++)
                        sum += V[i, k] * eig[k] * V[j, k];
                    fProxy err = Unity.Mathematics.math.abs(sum - A[i, j]);
                    if (err > maxErr)
                        maxErr = err;
                }
            if (!(maxErr <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = maxErr;
                Fail[2] = precision;
                Fail[3] = maxErr - precision;
            }
            Assert.IsTrue(maxErr <= precision);
        }

        // For every eigenpair (lambda_k = eig[k], v_k = column k of V), assert
        // ||A*v_k - lambda_k*v_k||_inf <= 1000*ZeroThreshold * (1 + |lambda_k|).
        private void AssertEigenResidual(in fProxyMxN A, in fProxyMxN V, in fProxyN eig, int n)
        {
            for (int k = 0; k < n; k++)
            {
                fProxy lambda = eig[k];
                fProxy maxRes = (fProxy)0;
                for (int i = 0; i < n; i++)
                {
                    fProxy av = (fProxy)0;
                    for (int j = 0; j < n; j++)
                        av += A[i, j] * V[j, k];
                    fProxy ri = Unity.Mathematics.math.abs(av - lambda * V[i, k]);
                    if (ri > maxRes)
                        maxRes = ri;
                }
                fProxy tol = (fProxy)1000 * Consts.fProxyZeroThreshold * ((fProxy)1 + Unity.Mathematics.math.abs(lambda));
                if (!(maxRes <= tol) && Fail[0] == (fProxy)0)
                {
                    Fail[0] = (fProxy)1;
                    Fail[1] = maxRes;
                    Fail[2] = tol;
                    Fail[3] = (fProxy)k;
                }
                Assert.IsTrue(maxRes <= tol);
            }
        }

        // Recompute residual r = A*v - lambda*v (inf-norm) and assert it satisfies the
        // documented convergence criterion r <= tol * max(1, |lambda|).
        private void AssertPowerResidual(in fProxyMxN A, in fProxyN v, fProxy lambda, fProxy tol, int n)
        {
            fProxy maxRes = (fProxy)0;
            for (int i = 0; i < n; i++)
            {
                fProxy av = (fProxy)0;
                for (int j = 0; j < n; j++)
                    av += A[i, j] * v[j];
                fProxy ri = Unity.Mathematics.math.abs(av - lambda * v[i]);
                if (ri > maxRes)
                    maxRes = ri;
            }
            fProxy scale = Unity.Mathematics.math.abs(lambda);
            if (scale < (fProxy)1)
                scale = (fProxy)1;
            fProxy limit = tol * scale;
            if (!(maxRes <= limit) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = maxRes;
                Fail[2] = limit;
                Fail[3] = maxRes - limit;
            }
            Assert.IsTrue(maxRes <= limit);
        }

        // Eigenvalues descending by value: eig[i] <= eig[i-1] (+ slack).
        private void AssertDescending(in fProxyN eig, int n)
        {
            for (int i = 1; i < n; i++)
            {
                bool descending = eig[i] <= eig[i - 1] + (fProxy)100 * Consts.fProxyZeroThreshold;
                if (!descending && Fail[0] == (fProxy)0)
                {
                    Fail[0] = (fProxy)1;
                    Fail[1] = eig[i];
                    Fail[2] = eig[i - 1];
                    Fail[3] = (fProxy)i;
                }
                Assert.IsTrue(descending);
            }
        }

        private void AssertClose(fProxy a, fProxy b, fProxy precision)
        {
            fProxy diff = Unity.Mathematics.math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        private void AssertFinite(fProxy v)
        {
            if (!Unity.Mathematics.math.isfinite(v) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = v;
                Fail[2] = (fProxy)0;
                Fail[3] = (fProxy)0;
            }
            Assert.IsTrue(Unity.Mathematics.math.isfinite(v));
        }

        // Boolean-condition assert with a distinguishing `code` recorded in Fail[1] (so a silent
        // Burst abort is still diagnosable). Mirrors fProxySparseEigenTests.AssertTrue; used by the
        // EigenSolveInfo field checks, where the value under test is an enum/bool, not a magnitude.
        private void AssertTrue(bool cond, fProxy code)
        {
            if (!cond && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = code;
                Fail[2] = (fProxy)0;
                Fail[3] = (fProxy)0;
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
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try {
            new TestJob() { Type = type, Fail = fail }.Run();
            // Under Burst a failed in-job assert logs an exception and aborts the job without
            // throwing to the caller - surface the recorded diagnostics here as well.
            if (fail[0] != (fProxy)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");

        }
        catch (Exception e) {
            if (fail[0] != (fProxy)0)
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
        var A = new fProxyMxN(3, 4, Allocator.Temp);
        var eig = new fProxyN(4, Allocator.Temp);
        var V = new fProxyMxN(4, 4, Allocator.Temp);

        Assert.Catch<ArgumentException>(() => Eigen.decompInPlace(ref A, ref eig, ref V));
    }

    [Test]
    public void EigenThrowsOnWrongEigenvalueLength()
    {
        var A = new fProxyMxN(4, 4, Allocator.Temp);
        var eig = new fProxyN(3, Allocator.Temp);
        var V = new fProxyMxN(4, 4, Allocator.Temp);

        Assert.Catch<ArgumentException>(() => Eigen.decompInPlace(ref A, ref eig, ref V));
    }

    [Test]
    public void EigenThrowsOnWrongVShape()
    {
        var A = new fProxyMxN(4, 4, Allocator.Temp);
        var eig = new fProxyN(4, Allocator.Temp);
        var V = new fProxyMxN(3, 3, Allocator.Temp);

        Assert.Catch<ArgumentException>(() => Eigen.decompInPlace(ref A, ref eig, ref V));
    }

    [Test]
    public void EigenThrowsOnBadMaxSweeps()
    {
        var A = new fProxyMxN(4, 4, Allocator.Temp);
        var eig = new fProxyN(4, Allocator.Temp);
        var V = new fProxyMxN(4, 4, Allocator.Temp);

        Assert.Catch<ArgumentException>(() => Eigen.decompInPlace(ref A, ref eig, ref V, 0));
    }

    [Test]
    public void EigenThrowsOnNonSymmetric()
    {
        var A = new fProxyMxN(2, 2, Allocator.Temp);
        A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)2;
        A[1, 0] = (fProxy)0; A[1, 1] = (fProxy)1;

        var eig = new fProxyN(2, Allocator.Temp);
        var V = new fProxyMxN(2, 2, Allocator.Temp);

        Assert.Catch<ArgumentException>(() => Eigen.decompInPlace(ref A, ref eig, ref V));
    }

    [Test]
    public void EvSymThrowsOnNonSymmetric()
    {
        var A = new fProxyMxN(2, 2, Allocator.Temp);
        A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)2;
        A[1, 0] = (fProxy)0; A[1, 1] = (fProxy)1;

        var eig = new fProxyN(2, Allocator.Temp);

        Assert.Catch<ArgumentException>(() => Eigen.valuesSymmetricInPlace(ref A, ref eig));
    }

    [Test]
    public void EvSymThrowsOnNonSquare()
    {
        var A = new fProxyMxN(3, 4, Allocator.Temp);
        var eig = new fProxyN(4, Allocator.Temp);

        Assert.Catch<ArgumentException>(() => Eigen.valuesSymmetricInPlace(ref A, ref eig));
    }

    [Test]
    public void EvSymThrowsOnWrongEigenvalueLength()
    {
        var A = new fProxyMxN(4, 4, Allocator.Temp);
        var eig = new fProxyN(3, Allocator.Temp);

        Assert.Catch<ArgumentException>(() => Eigen.valuesSymmetricInPlace(ref A, ref eig));
    }

    [Test]
    public void EsymThrowsOnNonSquare()
    {
        var A = new fProxyMxN(3, 4, Allocator.Temp);
        var eig = new fProxyN(4, Allocator.Temp);
        var V = new fProxyMxN(4, 4, Allocator.Temp);

        Assert.Catch<ArgumentException>(() => Eigen.symmetricInPlace(ref A, ref eig, ref V));
    }

    [Test]
    public void EsymThrowsOnNonSymmetric()
    {
        var A = new fProxyMxN(2, 2, Allocator.Temp);
        A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)2;
        A[1, 0] = (fProxy)0; A[1, 1] = (fProxy)1;

        var eig = new fProxyN(2, Allocator.Temp);
        var V = new fProxyMxN(2, 2, Allocator.Temp);

        Assert.Catch<ArgumentException>(() => Eigen.symmetricInPlace(ref A, ref eig, ref V));
    }

    [Test]
    public void EsymThrowsOnWrongEigenvalueLength()
    {
        var A = new fProxyMxN(4, 4, Allocator.Temp);
        var eig = new fProxyN(3, Allocator.Temp);
        var V = new fProxyMxN(4, 4, Allocator.Temp);

        Assert.Catch<ArgumentException>(() => Eigen.symmetricInPlace(ref A, ref eig, ref V));
    }

    [Test]
    public void EsymThrowsOnWrongVShape()
    {
        var A = new fProxyMxN(4, 4, Allocator.Temp);
        var eig = new fProxyN(4, Allocator.Temp);
        var V = new fProxyMxN(3, 3, Allocator.Temp);

        Assert.Catch<ArgumentException>(() => Eigen.symmetricInPlace(ref A, ref eig, ref V));
    }

    [Test]
    public void PowerThrowsOnNonSquare()
    {
        var A = new fProxyMxN(3, 4, Allocator.Temp);
        var v = new fProxyN(4, Allocator.Temp);
        var w = new fProxyN(4, Allocator.Temp);

        Assert.Catch<ArgumentException>(() =>
            Eigen.powerIteration(in A, ref v, ref w, out fProxy lambda, Consts.fProxyZeroThreshold, 1000));
    }

    [Test]
    public void PowerThrowsOnWrongVLength()
    {
        var A = new fProxyMxN(4, 4, Allocator.Temp);
        var v = new fProxyN(3, Allocator.Temp);
        var w = new fProxyN(4, Allocator.Temp);

        Assert.Catch<ArgumentException>(() =>
            Eigen.powerIteration(in A, ref v, ref w, out fProxy lambda, Consts.fProxyZeroThreshold, 1000));
    }

    [Test]
    public void PowerThrowsOnWrongWLength()
    {
        var A = new fProxyMxN(4, 4, Allocator.Temp);
        var v = new fProxyN(4, Allocator.Temp);
        var w = new fProxyN(3, Allocator.Temp);

        Assert.Catch<ArgumentException>(() =>
            Eigen.powerIteration(in A, ref v, ref w, out fProxy lambda, Consts.fProxyZeroThreshold, 1000));
    }

    [Test]
    public void PowerThrowsOnBadMaxIter()
    {
        var A = new fProxyMxN(4, 4, Allocator.Temp);
        var v = new fProxyN(4, Allocator.Temp);
        var w = new fProxyN(4, Allocator.Temp);

        Assert.Catch<ArgumentException>(() =>
            Eigen.powerIteration(in A, ref v, ref w, out fProxy lambda, Consts.fProxyZeroThreshold, 0));
    }

}
