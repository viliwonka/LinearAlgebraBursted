using System;
#pragma warning disable 618 // intentionally exercises the deprecated Jacobi svdDecomposition (kept for reference)

using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Stats;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Mathematics;

using Unity.Jobs;
using UnityEngine;
using UnityEngine.TestTools;

public class fProxySVDTests
{

    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            SVDIdentity,
            SVDDiagonal,
            SVDKnown2x2,
            SVDRandomSquare,
            SVDRectangularTall,
            SVDRankDeficient,
            SVDZero,
            SVDSingleColumn,
            SVDNonConvergence,
            SVDGalleryHadamard,
            SVDGalleryParter,
            SVValuesIdentity,
            SVValuesDiagonal,
            SVValuesKnown2x2,
            SVValuesRankDeficient,
            SVValuesCrossSquare6,
            SVValuesCrossSquare8,
            SVValuesCrossTall8x5,
            SVValuesCrossTall7x3,
            GolubKahanCrossSquare6,
            GolubKahanCrossSquare8,
            GolubKahanCrossTall10x6,
            GolubKahanCrossTall12x4,
            GolubKahanRankDeficient,
            GolubKahanClustered,
            GolubKahanZero,
            GolubKahanRank3
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff/index
        public NativeArray<fProxy> Fail;

        public void Execute()
        {
            switch(Type)
            {
                case TestType.SVDIdentity:
                    SVDIdentity();
                break;
                case TestType.SVDDiagonal:
                    SVDDiagonal();
                break;
                case TestType.SVDKnown2x2:
                    SVDKnown2x2();
                break;
                case TestType.SVDRandomSquare:
                    SVDRandomSquare();
                break;
                case TestType.SVDRectangularTall:
                    SVDRectangularTall();
                break;
                case TestType.SVDRankDeficient:
                    SVDRankDeficient();
                break;
                case TestType.SVDZero:
                    SVDZero();
                break;
                case TestType.SVDSingleColumn:
                    SVDSingleColumn();
                break;
                case TestType.SVDNonConvergence:
                    SVDNonConvergence();
                break;
                case TestType.SVDGalleryHadamard:
                    SVDGalleryHadamard();
                break;
                case TestType.SVDGalleryParter:
                    SVDGalleryParter();
                break;
                case TestType.SVValuesIdentity:
                    SVValuesIdentity();
                break;
                case TestType.SVValuesDiagonal:
                    SVValuesDiagonal();
                break;
                case TestType.SVValuesKnown2x2:
                    SVValuesKnown2x2();
                break;
                case TestType.SVValuesRankDeficient:
                    SVValuesRankDeficient();
                break;
                case TestType.SVValuesCrossSquare6:
                    SVValuesCross(6, 6, 9001011);
                break;
                case TestType.SVValuesCrossSquare8:
                    SVValuesCross(8, 8, 4242421);
                break;
                case TestType.SVValuesCrossTall8x5:
                    SVValuesCross(8, 5, 7733119);
                break;
                case TestType.SVValuesCrossTall7x3:
                    SVValuesCross(7, 3, 1551991);
                break;
                case TestType.GolubKahanCrossSquare6:
                    GolubKahanCross(6, 6, 9001011);
                break;
                case TestType.GolubKahanCrossSquare8:
                    GolubKahanCross(8, 8, 4242421);
                break;
                case TestType.GolubKahanCrossTall10x6:
                    GolubKahanCross(10, 6, 7733119);
                break;
                case TestType.GolubKahanCrossTall12x4:
                    GolubKahanCross(12, 4, 1551991);
                break;
                case TestType.GolubKahanRankDeficient:
                    GolubKahanRankDeficient();
                break;
                case TestType.GolubKahanClustered:
                    GolubKahanClustered();
                break;
                case TestType.GolubKahanZero:
                    GolubKahanZero();
                break;
                case TestType.GolubKahanRank3:
                    GolubKahanRank3();
                break;
            }
        }

        public void SVDIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 4;

            var U = arena.fProxyIdentityMatrix(dim);
            var S = arena.fProxyVec(dim);
            var V = arena.fProxyMat(dim, dim);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            for (int i = 0; i < dim; i++)
                AssertClose(S[i], (fProxy)1f, 1E-4f);

            Assert.IsTrue(Analysis.IsOrthogonal(U, 1E-4f));
            Assert.IsTrue(Analysis.IsOrthogonal(V, 1E-4f));

            AssertReconstruct(in A, in U, in S, in V, ref arena, 1E-4f);

            arena.Dispose();
        }

        public void SVDDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 4;

            var U = arena.fProxyMat(dim, dim);
            U[0, 0] = 3f;
            U[1, 1] = -2f;
            U[2, 2] = 0.5f;
            U[3, 3] = 5f;

            var S = arena.fProxyVec(dim);
            var V = arena.fProxyMat(dim, dim);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            // Singular values are |eigenvalues|, sorted descending: 5, 3, 2, 0.5
            AssertClose(S[0], (fProxy)5f, 1E-4f);
            AssertClose(S[1], (fProxy)3f, 1E-4f);
            AssertClose(S[2], (fProxy)2f, 1E-4f);
            AssertClose(S[3], (fProxy)0.5f, 1E-4f);

            // descending and non-negative
            AssertDescendingNonNegative(in S, dim);

            AssertReconstruct(in A, in U, in S, in V, ref arena, 1E-4f);

            arena.Dispose();
        }

        public void SVDKnown2x2()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 2;

            // A = [[3, 0], [4, 5]]
            var U = arena.fProxyMat(dim, dim);
            U[0, 0] = 3f; U[0, 1] = 0f;
            U[1, 0] = 4f; U[1, 1] = 5f;

            var S = arena.fProxyVec(dim);
            var V = arena.fProxyMat(dim, dim);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            // singular values: sqrt(45) ~ 6.7082039, sqrt(5) ~ 2.2360680
            AssertClose(S[0], (fProxy)6.7082039f, 1E-3f);
            AssertClose(S[1], (fProxy)2.2360680f, 1E-3f);

            AssertDescendingNonNegative(in S, dim);

            Assert.IsTrue(Analysis.IsOrthogonal(U, 1E-4f));
            Assert.IsTrue(Analysis.IsOrthogonal(V, 1E-4f));

            AssertReconstruct(in A, in U, in S, in V, ref arena, 1E-4f);

            arena.Dispose();
        }

        public void SVDRandomSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var U = arena.fProxyRandomMatrix(dim, dim, -10f, 10f, 314221);
            var S = arena.fProxyVec(dim);
            var V = arena.fProxyMat(dim, dim);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            Assert.IsFalse(Analysis.IsAnyNan(in U));
            Assert.IsFalse(Analysis.IsAnyNan(in S));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            Assert.IsTrue(Analysis.IsOrthogonal(U, 1E-4f));
            Assert.IsTrue(Analysis.IsOrthogonal(V, 1E-4f));

            AssertDescendingNonNegative(in S, dim);

            AssertReconstruct(in A, in U, in S, in V, ref arena, 1E-4f);

            arena.Dispose();
        }

        public void SVDRectangularTall()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 6;
            int n = 3;

            var U = arena.fProxyRandomMatrix(m, n, -10f, 10f, 778231);
            var S = arena.fProxyVec(n);
            var V = arena.fProxyMat(n, n);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            Assert.IsFalse(Analysis.IsAnyNan(in U));
            Assert.IsFalse(Analysis.IsAnyNan(in S));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            // U is 6x3 with orthonormal columns -> U^T U = I_3
            Assert.IsTrue(Analysis.IsOrthogonal(U, 1E-4f));
            Assert.IsTrue(Analysis.IsOrthogonal(V, 1E-4f));

            AssertDescendingNonNegative(in S, n);

            AssertReconstruct(in A, in U, in S, in V, ref arena, 1E-4f);

            arena.Dispose();
        }

        public void SVDRankDeficient()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 4;

            var U = arena.fProxyRandomMatrix(dim, dim, -5f, 5f, 559013);
            // make column 2 an exact copy of column 0 -> rank deficient
            for (int i = 0; i < dim; i++)
                U[i, 2] = U[i, 0];

            var S = arena.fProxyVec(dim);
            var V = arena.fProxyMat(dim, dim);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            Assert.IsFalse(Analysis.IsAnyNan(in U));
            Assert.IsFalse(Analysis.IsAnyNan(in S));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            // exactly one zero singular value (smallest), the third is non-trivial
            Assert.IsTrue(S[3] < 1E-4f);
            Assert.IsTrue(S[2] > 1E-4f);

            AssertDescendingNonNegative(in S, dim);

            AssertReconstruct(in A, in U, in S, in V, ref arena, 1E-4f);

            // first 3 columns of U are orthonormal (the column matching zero sigma is zeroed)
            for (int a = 0; a < 3; a++) {
                for (int b = a; b < 3; b++) {
                    fProxy dotcol = (fProxy)0f;
                    for (int i = 0; i < dim; i++)
                        dotcol += U[i, a] * U[i, b];
                    fProxy expected = (a == b) ? (fProxy)1f : (fProxy)0f;
                    AssertClose(dotcol, expected, 1E-4f);
                }
            }

            arena.Dispose();
        }

        public void SVDZero()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 3;

            var U = arena.fProxyMat(dim, dim);
            var S = arena.fProxyVec(dim);
            var V = arena.fProxyMat(dim, dim);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            Assert.IsFalse(Analysis.IsAnyNan(in U));
            Assert.IsFalse(Analysis.IsAnyNan(in S));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            for (int i = 0; i < dim; i++)
                AssertClose(S[i], (fProxy)0f, 1E-4f);

            // U is all zeros (every column matches a zero singular value)
            Assert.IsTrue(Analysis.IsZero(in U, 1E-4f));

            // V stays identity (no rotations applied)
            Assert.IsTrue(Analysis.IsIdentity(in V, 1E-4f));

            arena.Dispose();
        }

        public void SVDSingleColumn()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 5;
            int n = 1;

            var U = arena.fProxyMat(m, n);
            U[0, 0] = 1f;
            U[1, 0] = 2f;
            U[2, 0] = 3f;
            U[3, 0] = 4f;
            U[4, 0] = 5f;

            var S = arena.fProxyVec(n);
            var V = arena.fProxyMat(n, n);

            var A = U.Copy();

            bool success = SVD.svdDecomposition(ref U, ref S, ref V);

            Assert.IsTrue(success);

            // S[0] == column 2-norm = sqrt(1+4+9+16+25) = sqrt(55) ~ 7.4161985
            AssertClose(S[0], (fProxy)7.4161985f, 1E-3f);

            // U column has unit norm
            fProxy normSq = (fProxy)0f;
            for (int i = 0; i < m; i++)
                normSq += U[i, 0] * U[i, 0];
            AssertClose(normSq, (fProxy)1f, 1E-4f);

            AssertReconstruct(in A, in U, in S, in V, ref arena, 1E-4f);

            arena.Dispose();
        }

        public void SVDNonConvergence()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var U = arena.fProxyHilbertMatrix(dim);
            var S = arena.fProxyVec(dim);
            var V = arena.fProxyMat(dim, dim);

            // maxSweeps = 1: regardless of convergence bool, outputs must be finite,
            // S descending and non-negative.
            SVD.svdDecomposition(ref U, ref S, ref V, 1);

            // The return value is intentionally not asserted (may or may not converge).

            Assert.IsFalse(Analysis.IsAnyNan(in U));
            Assert.IsFalse(Analysis.IsAnyNan(in S));
            Assert.IsFalse(Analysis.IsAnyNan(in V));

            AssertDescendingNonNegative(in S, dim);

            arena.Dispose();
        }

        // GALLERY KNOWN-ANSWER (Gallery.Special): the 4x4 Sylvester-Walsh Hadamard matrix satisfies
        // HᵀH = n·I, so ALL singular values equal √n = √4 = 2 and the condition number is exactly 1.
        // Uses SVD.singularValues (A is not modified, S is sorted descending).
        public void SVDGalleryHadamard()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;

            var A = arena.fProxyHadamard(n);
            var S = arena.fProxyVec(n);

            int k = SVD.singularValues(in A, ref S);
            AssertClose((fProxy)k, (fProxy)n, 1E-6f);

            Assert.IsFalse(Analysis.IsAnyNan(in S));

            // every singular value == sqrt(4) == 2 (cond = 1)
            for (int i = 0; i < n; i++)
                AssertClose(S[i], (fProxy)2f, 1E-4f);

            AssertDescendingNonNegative(in S, n);

            arena.Dispose();
        }

        // GALLERY KNOWN-ANSWER (Gallery.Phase2): the 8x8 Parter matrix (Toeplitz 1/(i-j+0.5)) has
        // singular values that cluster near π, ALL strictly below π. For n=8 the largest is
        // 3.1415926534..., only ~1.1e-10 below π — far tighter than float SVD precision — so the
        // bound is asserted with a scale-aware margin that still rejects any gross overshoot.
        public void SVDGalleryParter()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 8;

            var A = arena.fProxyParter(n);
            var S = arena.fProxyVec(n);

            SVD.singularValues(in A, ref S);

            Assert.IsFalse(Analysis.IsAnyNan(in S));

            fProxy pi = (fProxy)Unity.Mathematics.math.PI_DBL;
            // boundary lies within ~1e-10 of π; absorb float SVD error without masking a real overshoot.
            fProxy margin = (fProxy)64 * Consts.fProxySqrtEps;

            for (int i = 0; i < n; i++)
            {
                bool below = S[i] <= pi + margin;
                if (!below && Fail[0] == (fProxy)0)
                {
                    Fail[0] = (fProxy)1;
                    Fail[1] = S[i];
                    Fail[2] = pi;
                    Fail[3] = (fProxy)i;
                }
                Assert.IsTrue(below);
            }

            // largest singular value clusters near π (close from below).
            AssertClose(S[0], pi, margin);

            AssertDescendingNonNegative(in S, n);

            arena.Dispose();
        }

        // ---- svdValues (singular VALUES only, A unmodified) ----

        // Identity n=5 -> all singular values 1.
        public void SVValuesIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;

            var A = arena.fProxyIdentityMatrix(n);
            var S = arena.fProxyVec(n);

            bool ok = SVD.svdValues(in A, ref S);
            Assert.IsTrue(ok);

            Assert.IsFalse(Analysis.IsAnyNan(in S));

            for (int i = 0; i < n; i++)
                AssertClose(S[i], (fProxy)1f, 1E-4f);

            AssertDescendingNonNegative(in S, n);

            // A must be unchanged (still identity).
            Assert.IsTrue(Analysis.IsIdentity(in A, 1E-5f));

            arena.Dispose();
        }

        // Diagonal diag(d) -> singular values = |d_i| sorted descending.
        public void SVValuesDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;

            var A = arena.fProxyMat(n, n);
            A[0, 0] = 3f;
            A[1, 1] = -2f;
            A[2, 2] = 0.5f;
            A[3, 3] = 5f;
            A[4, 4] = -1f;

            var Apristine = A.Copy();

            var S = arena.fProxyVec(n);

            bool ok = SVD.svdValues(in A, ref S);
            Assert.IsTrue(ok);

            Assert.IsFalse(Analysis.IsAnyNan(in S));

            // |d| sorted descending: 5, 3, 2, 1, 0.5
            AssertClose(S[0], (fProxy)5f, 1E-4f);
            AssertClose(S[1], (fProxy)3f, 1E-4f);
            AssertClose(S[2], (fProxy)2f, 1E-4f);
            AssertClose(S[3], (fProxy)1f, 1E-4f);
            AssertClose(S[4], (fProxy)0.5f, 1E-4f);

            AssertDescendingNonNegative(in S, n);

            // A must be unmodified.
            AssertMatrixUnchanged(in A, in Apristine, n, n);

            arena.Dispose();
        }

        // Known small matrix [[3,0],[0,-4]] -> singular values 4, 3.
        public void SVValuesKnown2x2()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 2;

            var A = arena.fProxyMat(n, n);
            A[0, 0] = 3f; A[0, 1] = 0f;
            A[1, 0] = 0f; A[1, 1] = -4f;

            var S = arena.fProxyVec(n);

            bool ok = SVD.svdValues(in A, ref S);
            Assert.IsTrue(ok);

            Assert.IsFalse(Analysis.IsAnyNan(in S));

            AssertClose(S[0], (fProxy)4f, 1E-4f);
            AssertClose(S[1], (fProxy)3f, 1E-4f);

            AssertDescendingNonNegative(in S, n);

            arena.Dispose();
        }

        // Rank-1 outer product u*v^T -> exactly one positive singular value (= |u|*|v|), rest ~0.
        public void SVValuesRankDeficient()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;

            var u = arena.fProxyVec(n);
            u[0] = 1f; u[1] = -2f; u[2] = 3f; u[3] = 0.5f; u[4] = -1.5f;
            var v = arena.fProxyVec(n);
            v[0] = 2f; v[1] = 1f; v[2] = -1f; v[3] = 4f; v[4] = 0.25f;

            var A = arena.fProxyOuter(in u, in v);
            var Apristine = A.Copy();

            var S = arena.fProxyVec(n);

            bool ok = SVD.svdValues(in A, ref S);
            Assert.IsTrue(ok);

            Assert.IsFalse(Analysis.IsAnyNan(in S));

            // expected sole singular value = ||u|| * ||v||
            fProxy nu = (fProxy)0f, nv = (fProxy)0f;
            for (int i = 0; i < n; i++) { nu += u[i] * u[i]; nv += v[i] * v[i]; }
            fProxy sigma = math.sqrt(nu) * math.sqrt(nv);

            AssertClose(S[0], sigma, (fProxy)1E-3f + (fProxy)1E-4f * sigma);

            // the rest collapse to ~0
            for (int i = 1; i < n; i++)
                AssertClose(S[i], (fProxy)0f, (fProxy)1E-3f + (fProxy)1E-4f * sigma);

            AssertDescendingNonNegative(in S, n);

            AssertMatrixUnchanged(in A, in Apristine, n, n);

            arena.Dispose();
        }

        // Cross-check svdValues vs the trusted svdDecomposition for m >= n (square AND tall).
        // svdDecomposition destroys its U argument; svdValues takes A `in` (must be unmodified).
        public void SVValuesCross(int m, int n, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyRandomMatrix(m, n, -10f, 10f, seed);
            var Apristine = A.Copy();

            // reference path: copy of A consumed by svdDecomposition
            var U = A.Copy();
            var Sref = arena.fProxyVec(n);
            var V = arena.fProxyMat(n, n);
            bool okRef = SVD.svdDecomposition(ref U, ref Sref, ref V);
            Assert.IsTrue(okRef);

            // values-only path on the untouched A
            var S = arena.fProxyVec(n);
            bool ok = SVD.svdValues(in A, ref S);
            Assert.IsTrue(ok);

            Assert.IsFalse(Analysis.IsAnyNan(in S));
            Assert.IsFalse(Analysis.IsAnyNan(in Sref));

            AssertDescendingNonNegative(in S, n);
            AssertDescendingNonNegative(in Sref, n);

            // agree element-wise (both descending) with a scale-aware tolerance.
            for (int i = 0; i < n; i++)
            {
                fProxy scale = math.max(math.abs(S[i]), math.abs(Sref[i]));
                fProxy tol = (fProxy)1E-3f + (fProxy)1E-3f * scale;
                AssertClose(S[i], Sref[i], tol);
            }

            // svdValues must NOT have modified A.
            AssertMatrixUnchanged(in A, in Apristine, m, n);

            arena.Dispose();
        }

        // Cross-check the Golub-Kahan full SVD vs the trusted one-sided Jacobi svdDecomposition,
        // plus reconstruction A = U diag(S) Vᵀ and orthonormal U,V. A must be unmodified.
        public void GolubKahanCross(int m, int n, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyRandomMatrix(m, n, -10f, 10f, seed);
            var Apristine = A.Copy();

            // oracle: one-sided Jacobi (destroys its U arg = copy of A)
            var Uref = A.Copy();
            var Sref = arena.fProxyVec(n);
            var Vref = arena.fProxyMat(n, n);
            bool okRef = SVD.svdDecomposition(ref Uref, ref Sref, ref Vref);
            Assert.IsTrue(okRef);

            // Golub-Kahan on the untouched A
            var U = arena.fProxyMat(m, n);
            var S = arena.fProxyVec(n);
            var V = arena.fProxyMat(n, n);
            bool ok = SVD.svdGolubKahan(in A, ref U, ref S, ref V);
            Assert.IsTrue(ok);

            Assert.IsFalse(Analysis.IsAnyNan(in S));
            AssertDescendingNonNegative(in S, n);

            // singular values agree with the oracle (both descending, scale-aware tol)
            for (int i = 0; i < n; i++)
            {
                fProxy scale = math.max(math.abs(S[i]), math.abs(Sref[i]));
                fProxy tol = (fProxy)1E-3f + (fProxy)1E-3f * scale;
                AssertClose(S[i], Sref[i], tol);
            }

            fProxy reconTol = (fProxy)1E-3f + (fProxy)1E-4f * math.abs(S[0]);
            AssertReconstruct(in A, in U, in S, in V, ref arena, reconTol);
            Assert.IsTrue(Analysis.IsOrthogonal(U, (fProxy)1E-3f));
            Assert.IsTrue(Analysis.IsOrthogonal(V, (fProxy)1E-3f));

            AssertMatrixUnchanged(in A, in Apristine, m, n);

            arena.Dispose();
        }

        // Rank-1 (rank-deficient) matrix: one nonzero singular value ||u||*||v||, the rest ~0.
        // Exercises the clustered-zero deflation in the bidiagonal QR.
        public void GolubKahanRankDeficient()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            var u = arena.fProxyVec(n);
            u[0] = 1f; u[1] = -2f; u[2] = 3f; u[3] = 0.5f; u[4] = -1.5f;
            var v = arena.fProxyVec(n);
            v[0] = 2f; v[1] = 1f; v[2] = -1f; v[3] = 4f; v[4] = 0.25f;

            var A = arena.fProxyOuter(in u, in v);
            var Apristine = A.Copy();

            var U = arena.fProxyMat(n, n);
            var S = arena.fProxyVec(n);
            var V = arena.fProxyMat(n, n);
            bool ok = SVD.svdGolubKahan(in A, ref U, ref S, ref V);
            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.IsAnyNan(in S));

            fProxy nu = (fProxy)0f, nv = (fProxy)0f;
            for (int i = 0; i < n; i++) { nu += u[i] * u[i]; nv += v[i] * v[i]; }
            fProxy sigma = math.sqrt(nu) * math.sqrt(nv);

            AssertClose(S[0], sigma, (fProxy)1E-3f + (fProxy)1E-3f * sigma);
            for (int i = 1; i < n; i++)
                AssertClose(S[i], (fProxy)0f, (fProxy)1E-3f + (fProxy)1E-3f * sigma);

            AssertDescendingNonNegative(in S, n);
            AssertReconstruct(in A, in U, in S, in V, ref arena, (fProxy)1E-3f + (fProxy)1E-4f * sigma);
            AssertMatrixUnchanged(in A, in Apristine, n, n);

            arena.Dispose();
        }

        // Fully clustered spectrum: A = 3*I has all singular values equal to 3. Stresses deflation.
        public void GolubKahanClustered()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;
            var A = arena.fProxyMat(n, n);
            for (int i = 0; i < n; i++) A[i, i] = (fProxy)3f;
            var Apristine = A.Copy();

            var U = arena.fProxyMat(n, n);
            var S = arena.fProxyVec(n);
            var V = arena.fProxyMat(n, n);
            bool ok = SVD.svdGolubKahan(in A, ref U, ref S, ref V);
            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.IsAnyNan(in S));

            for (int i = 0; i < n; i++)
                AssertClose(S[i], (fProxy)3f, (fProxy)1E-3f);

            AssertReconstruct(in A, in U, in S, in V, ref arena, (fProxy)1E-3f);
            Assert.IsTrue(Analysis.IsOrthogonal(U, (fProxy)1E-3f));
            Assert.IsTrue(Analysis.IsOrthogonal(V, (fProxy)1E-3f));
            AssertMatrixUnchanged(in A, in Apristine, n, n);

            arena.Dispose();
        }

        // Zero matrix: anorm == 0 → deflation threshold 0; must still converge (no NaN), S all 0.
        public void GolubKahanZero()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            var A = arena.fProxyMat(n, n);   // all zeros

            var U = arena.fProxyMat(n, n);
            var S = arena.fProxyVec(n);
            var V = arena.fProxyMat(n, n);
            bool ok = SVD.svdGolubKahan(in A, ref U, ref S, ref V);
            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.IsAnyNan(in S));
            for (int i = 0; i < n; i++)
                AssertClose(S[i], (fProxy)0f, (fProxy)1E-5f);

            arena.Dispose();
        }

        // Rank-3 6x6 (sum of three independent outer products) → 3 nonzero + 3 zero singular values.
        // The INTERIOR zeros exercise the cancellation branch (|d[nm]| <= thresh) of the bidiagonal QR.
        public void GolubKahanRank3()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;
            var u1 = arena.fProxyVec(n); u1[0]=1f; u1[1]=0f; u1[2]=0f; u1[3]=1f; u1[4]=0f; u1[5]=0f;
            var v1 = arena.fProxyVec(n); v1[0]=1f; v1[1]=2f; v1[2]=0f; v1[3]=0f; v1[4]=0f; v1[5]=0f;
            var u2 = arena.fProxyVec(n); u2[0]=0f; u2[1]=1f; u2[2]=0f; u2[3]=0f; u2[4]=1f; u2[5]=0f;
            var v2 = arena.fProxyVec(n); v2[0]=0f; v2[1]=0f; v2[2]=1f; v2[3]=3f; v2[4]=0f; v2[5]=0f;
            var u3 = arena.fProxyVec(n); u3[0]=0f; u3[1]=0f; u3[2]=1f; u3[3]=0f; u3[4]=0f; u3[5]=1f;
            var v3 = arena.fProxyVec(n); v3[0]=0f; v3[1]=0f; v3[2]=0f; v3[3]=0f; v3[4]=1f; v3[5]=2f;

            var A = arena.fProxyMat(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = u1[i] * v1[j] + u2[i] * v2[j] + u3[i] * v3[j];
            var Apristine = A.Copy();

            // oracle
            var Uref = A.Copy();
            var Sref = arena.fProxyVec(n);
            var Vref = arena.fProxyMat(n, n);
            bool okRef = SVD.svdDecomposition(ref Uref, ref Sref, ref Vref);
            Assert.IsTrue(okRef);

            var U = arena.fProxyMat(n, n);
            var S = arena.fProxyVec(n);
            var V = arena.fProxyMat(n, n);
            bool ok = SVD.svdGolubKahan(in A, ref U, ref S, ref V);
            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.IsAnyNan(in S));
            AssertDescendingNonNegative(in S, n);

            for (int i = 0; i < n; i++)
            {
                fProxy scale = math.max(math.abs(S[i]), math.abs(Sref[i]));
                fProxy tol = (fProxy)1E-3f + (fProxy)1E-3f * scale;
                AssertClose(S[i], Sref[i], tol);
            }
            // bottom three singular values are the (interior) zeros
            for (int i = 3; i < n; i++)
                AssertClose(S[i], (fProxy)0f, (fProxy)1E-3f + (fProxy)1E-3f * S[0]);

            AssertReconstruct(in A, in U, in S, in V, ref arena, (fProxy)1E-3f + (fProxy)1E-4f * S[0]);
            Assert.IsTrue(Analysis.IsOrthogonal(U, (fProxy)1E-3f));
            Assert.IsTrue(Analysis.IsOrthogonal(V, (fProxy)1E-3f));
            AssertMatrixUnchanged(in A, in Apristine, n, n);

            arena.Dispose();
        }

        // Fail layout: [1]=A[i,j], [2]=ref[i,j], [3]=diff
        private void AssertMatrixUnchanged(in fProxyMxN A, in fProxyMxN B, int m, int n)
        {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    fProxy diff = math.abs(A[i, j] - B[i, j]);
                    bool same = diff <= (fProxy)1E-6f;
                    if (!same && Fail[0] == (fProxy)0)
                    {
                        Fail[0] = (fProxy)1;
                        Fail[1] = A[i, j];
                        Fail[2] = B[i, j];
                        Fail[3] = diff;
                    }
                    Assert.IsTrue(same);
                }
        }

        private void AssertReconstruct(in fProxyMxN A, in fProxyMxN U, in fProxyN S, in fProxyMxN V, ref Arena arena, fProxy precision)
        {
            // A ~= U * diag(S) * V^T
            var diagS = arena.fProxyDiagonalMatrix(in S);
            var US = fProxyOP.dot(U, diagS);
            var Vt = fProxyOP.trans(V);
            var recon = fProxyOP.dot(US, Vt);

            fProxyMxN shouldBeZero = A - recon;

            if (Analysis.IsAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            var zeroError = Analysis.MaxZeroError(shouldBeZero);

            // Fail layout: [1]=zeroError, [2]=precision, [3]=diff
            if (!(zeroError <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = zeroError;
                Fail[2] = precision;
                Fail[3] = zeroError - precision;
            }
            Assert.IsTrue(Analysis.IsZero(in shouldBeZero, precision));
        }

        // Fail layout: [1]=S[i] (offending element), [2]=bound or S[i-1], [3]=index cast to fProxy
        private void AssertDescendingNonNegative(in fProxyN S, int n)
        {
            for (int i = 0; i < n; i++)
            {
                bool nonNeg = S[i] >= (fProxy)(-1E-6f);
                if (!nonNeg && Fail[0] == (fProxy)0)
                {
                    Fail[0] = (fProxy)1;
                    Fail[1] = S[i];
                    Fail[2] = (fProxy)(-1E-6f);
                    Fail[3] = (fProxy)i;
                }
                Assert.IsTrue(nonNeg);
            }

            for (int i = 1; i < n; i++)
            {
                bool descending = S[i] <= S[i - 1] + (fProxy)1E-6f;
                if (!descending && Fail[0] == (fProxy)0)
                {
                    Fail[0] = (fProxy)1;
                    Fail[1] = S[i];
                    Fail[2] = S[i - 1];
                    Fail[3] = (fProxy)i;
                }
                Assert.IsTrue(descending);
            }
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff
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

    }

    public static Array GetEnums() {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void SVDDecompTests(TestJob.TestType type)
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

    // Managed throw-tests: argument validation runs on the main thread (not in a Burst job).

    [Test]
    public void SVDThrowsOnWideMatrix()
    {
        var arena = new Arena(Allocator.Persistent);

        var U = arena.fProxyMat(2, 3);
        var S = arena.fProxyVec(3);
        var V = arena.fProxyMat(3, 3);

        Assert.Catch<ArgumentException>(() => SVD.svdDecomposition(ref U, ref S, ref V));

        arena.Dispose();
    }

    [Test]
    public void SVDThrowsOnWrongSLength()
    {
        var arena = new Arena(Allocator.Persistent);

        var U = arena.fProxyMat(4, 3);
        var S = arena.fProxyVec(2);
        var V = arena.fProxyMat(3, 3);

        Assert.Catch<ArgumentException>(() => SVD.svdDecomposition(ref U, ref S, ref V));

        arena.Dispose();
    }

    [Test]
    public void SVDThrowsOnWrongVSize()
    {
        var arena = new Arena(Allocator.Persistent);

        var U = arena.fProxyMat(4, 3);
        var S = arena.fProxyVec(3);
        var V = arena.fProxyMat(2, 2);

        Assert.Catch<ArgumentException>(() => SVD.svdDecomposition(ref U, ref S, ref V));

        arena.Dispose();
    }

    [Test]
    public void SVDThrowsOnBadMaxSweeps()
    {
        var arena = new Arena(Allocator.Persistent);

        var U = arena.fProxyMat(4, 3);
        var S = arena.fProxyVec(3);
        var V = arena.fProxyMat(3, 3);

        Assert.Catch<ArgumentException>(() => SVD.svdDecomposition(ref U, ref S, ref V, 0));

        arena.Dispose();
    }

    [Test]
    public void SVValuesThrowsOnWideMatrix()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.fProxyMat(2, 3);
        var S = arena.fProxyVec(3);

        Assert.Catch<ArgumentException>(() => SVD.svdValues(in A, ref S));

        arena.Dispose();
    }

    [Test]
    public void SVValuesThrowsOnWrongSLength()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.fProxyMat(4, 3);
        var S = arena.fProxyVec(2);

        Assert.Catch<ArgumentException>(() => SVD.svdValues(in A, ref S));

        arena.Dispose();
    }

    [Test]
    public void SVDThrowsOnBadEps()
    {
        var arena = new Arena(Allocator.Persistent);

        var U = arena.fProxyMat(4, 3);
        var S = arena.fProxyVec(3);
        var V = arena.fProxyMat(3, 3);

        Assert.Catch<ArgumentException>(() => SVD.svdDecomposition(ref U, ref S, ref V, 30, (fProxy)0f));

        arena.Dispose();
    }

}
