using System;

using LinearAlgebra;
using LinearAlgebra.Gallery;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Mathematics;

using Unity.Jobs;
using UnityEngine;
using UnityEngine.TestTools;

public class floatSVDTests
{

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
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
            GolubKahanRank3,
            // --- thin known-Σ (randsvd / Higham Test Matrix Toolbox): A = U·diag(Σ)·Vᵀ, Haar U/V,
            //     prescribed Σ → exact singular values KNOWN. Sweeps the Higham randsvd modes. ---
            ThinKnownGeometric_30x10,
            ThinKnownArithmetic_24x8,
            ThinKnownOneSmall_10x10,
            ThinKnownClustered_20x8,
            ThinKnownFlatCliff_40x12,
            ThinKnownWideViaTranspose_6x15,
            ThinGalleryHilbert_8,
            ThinGalleryKahan_12,
            // Solver API rework (commit 2): uninit-x contract.
            PinvSolveUninitXContract,
            // Commit 2.5 SVD coverage restoration:
            //  2b  independent-algorithm cross-check (σ_i^2 == eig(AᵀA)_i) + Frobenius identity.
            CrossCheckEigenSquare8,
            CrossCheckEigenTall12x8,
            CrossCheckEigenClustered12x8,
            //  2c(i)  known-Σ via Gram-Schmidt on COHERENT vectors (different statistics than randsvd).
            KnownSigmaGramSchmidtCoherent,
            //  2d  ports of deleted Golub-Kahan edge cases.
            Known2x2GolubKahan,
            SingleColumn5x1,
            NonConvergenceHilbert8,
            //  2e  determinant invariant |det A| == Π σ_i.
            DetEqualsProductSingularValues
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff/index
        public NativeArray<float> Fail;

        public void Execute()
        {
            switch(Type)
            {
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
                case TestType.ThinKnownGeometric_30x10:        ThinKnownGeometric_30x10();        break;
                case TestType.ThinKnownArithmetic_24x8:        ThinKnownArithmetic_24x8();        break;
                case TestType.ThinKnownOneSmall_10x10:         ThinKnownOneSmall_10x10();         break;
                case TestType.ThinKnownClustered_20x8:         ThinKnownClustered_20x8();         break;
                case TestType.ThinKnownFlatCliff_40x12:        ThinKnownFlatCliff_40x12();        break;
                case TestType.ThinKnownWideViaTranspose_6x15:  ThinKnownWideViaTranspose_6x15();  break;
                case TestType.ThinGalleryHilbert_8:            ThinGalleryHilbert_8();            break;
                case TestType.ThinGalleryKahan_12:             ThinGalleryKahan_12();             break;
                case TestType.PinvSolveUninitXContract:        PinvSolveUninitXContract();        break;
                case TestType.CrossCheckEigenSquare8:          CrossCheckEigenRandom(8, 8, 0x5EED0011u);   break;
                case TestType.CrossCheckEigenTall12x8:         CrossCheckEigenRandom(12, 8, 0x5EED0012u);  break;
                case TestType.CrossCheckEigenClustered12x8:    CrossCheckEigenClustered12x8();    break;
                case TestType.KnownSigmaGramSchmidtCoherent:   KnownSigmaGramSchmidtCoherent();   break;
                case TestType.Known2x2GolubKahan:              Known2x2GolubKahan();              break;
                case TestType.SingleColumn5x1:                 SingleColumn5x1();                 break;
                case TestType.NonConvergenceHilbert8:          NonConvergenceHilbert8();          break;
                case TestType.DetEqualsProductSingularValues:  DetEqualsProductSingularValues();  break;
            }
        }

        // randsvd (Higham Test Matrix Toolbox): A = U·diag(σ)·Vᵀ with Haar-random orthogonal U (m×m),
        // V (n×n) and a caller-prescribed σ (length n, descending) → the exact singular values are KNOWN.
        void BuildRandSvd(ref Unity.Mathematics.Random rng, int m, int n, in floatN sigma, ref floatMxN A)
        {
            var U = new floatMxN(m, m, Allocator.Temp, false);
            var V = new floatMxN(n, n, Allocator.Temp, false);
            Rand.orthogonalInPlace(ref rng, ref U);
            Rand.orthogonalInPlace(ref rng, ref V);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int t = 0; t < n; t++)
                        acc += (double)U[i, t] * (double)sigma[t] * (double)V[j, t];
                    A[i, j] = (float)acc;
                }
            U.Dispose(); V.Dispose();
        }

        // U/V orthonormal-columns check: max |colᵀcol − δ| ≤ tol.
        void AssertOrthoColsLocal(in floatMxN basis, int rows, int cols, float tol)
        {
            for (int a = 0; a < cols; a++)
                for (int b = a; b < cols; b++)
                {
                    float dot = (float)0;
                    for (int i = 0; i < rows; i++) dot += basis[i, a] * basis[i, b];
                    AssertClose(dot, (a == b) ? (float)1 : (float)0, tol);
                }
        }

        // Core known-Σ thin check: recovered S == prescribed σ (svTol), UᵀU=I, VᵀV=I, A == U diag(S) Vᵀ.
        void CheckThinKnown(in floatMxN A, in floatN sigma, int m, int n, ref Arena arena)
        {
            var U = arena.floatMat(m, n);
            var S = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            Assert.IsTrue(SVD.thin(in A, ref U, ref S, ref V));
            Assert.IsFalse(Analysis.isAnyNan(in S));

            // svTol & ortho/recon tolerances scale with the numeric type via Consts.floatSqrtEps
            // (float ~3.45e-4, double ~1.49e-8): the SAME bound holds for both generated expansions.
            float svTol = (float)8 * Consts.floatSqrtEps * (sigma[0] + (float)1);
            for (int t = 0; t < n; t++) AssertClose(S[t], sigma[t], svTol);

            AssertDescendingNonNegative(in S, n);
            AssertOrthoColsLocal(in U, m, n, (float)32 * Consts.floatSqrtEps);
            AssertOrthoColsLocal(in V, n, n, (float)32 * Consts.floatSqrtEps);
            AssertReconstruct(in A, in U, in S, in V, ref arena, svTol);
        }

        void ThinKnownGeometric_30x10()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 30, n = 10;
            var sigma = arena.floatVec(n);
            double s = 1.0; for (int i = 0; i < n; i++) { sigma[i] = (float)s; s *= 0.6; }
            var A = arena.floatMat(m, n);
            var rng = new Unity.Mathematics.Random(0x7A0A0001u);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);
            CheckThinKnown(in A, in sigma, m, n, ref arena);
            arena.Dispose();
        }

        void ThinKnownArithmetic_24x8()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 24, n = 8;
            var sigma = arena.floatVec(n);
            for (int i = 0; i < n; i++) sigma[i] = (float)(10.0 - i);   // 10,9,...,3 (descending)
            var A = arena.floatMat(m, n);
            var rng = new Unity.Mathematics.Random(0x7A0A0002u);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);
            CheckThinKnown(in A, in sigma, m, n, ref arena);
            arena.Dispose();
        }

        void ThinKnownOneSmall_10x10()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 10, n = 10;
            var sigma = arena.floatVec(n);
            for (int i = 0; i < n; i++) sigma[i] = (i == n - 1) ? (float)1E-4f : (float)1;  // κ=1e4
            var A = arena.floatMat(m, n);
            var rng = new Unity.Mathematics.Random(0x7A0A0003u);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);
            CheckThinKnown(in A, in sigma, m, n, ref arena);
            arena.Dispose();
        }

        void ThinKnownClustered_20x8()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 20, n = 8;
            var sigma = arena.floatVec(n);
            // [10,10,10, 3,2,1,0.5,0.25]
            sigma[0]=(float)10; sigma[1]=(float)10; sigma[2]=(float)10; sigma[3]=(float)3;
            sigma[4]=(float)2;  sigma[5]=(float)1;  sigma[6]=(float)0.5f; sigma[7]=(float)0.25f;
            var A = arena.floatMat(m, n);
            var rng = new Unity.Mathematics.Random(0x7A0A0004u);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);
            CheckThinKnown(in A, in sigma, m, n, ref arena);
            arena.Dispose();
        }

        void ThinKnownFlatCliff_40x12()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 40, n = 12;
            var sigma = arena.floatVec(n);
            for (int i = 0; i < n; i++)
                sigma[i] = (i == 0) ? (float)100 : (i == 1) ? (float)80 : (i == 2) ? (float)60 : (float)1E-3f;
            var A = arena.floatMat(m, n);
            var rng = new Unity.Mathematics.Random(0x7A0A0005u);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);
            CheckThinKnown(in A, in sigma, m, n, ref arena);
            arena.Dispose();
        }

        // WIDE aspect: build a tall T (15×6) with known Σ; T = Wᵀ for the wide W = Tᵀ (6×15). thin
        // requires m≥n, so the documented route for a wide matrix is thin(in trans(W)) = thin(T),
        // which must recover W's singular values. Validates the transpose contract for wide inputs.
        void ThinKnownWideViaTranspose_6x15()
        {
            var arena = new Arena(Allocator.Persistent);
            int rows = 6, cols = 15;        // wide W is rows×cols
            int m = cols, n = rows;         // tall T = Wᵀ is cols×rows = 15×6
            var sigma = arena.floatVec(n);
            double s = 1.0; for (int i = 0; i < n; i++) { sigma[i] = (float)s; s *= 0.55; }
            var T = arena.floatMat(m, n);
            var rng = new Unity.Mathematics.Random(0x7A0A0006u);
            BuildRandSvd(ref rng, m, n, in sigma, ref T);   // T (15×6), its transpose is the 6×15 wide W
            CheckThinKnown(in T, in sigma, m, n, ref arena);
            arena.Dispose();
        }

        // Gallery ill-conditioned (Hilbert): assert σ sorted-descending positive, condition number
        // σ_max/σ_min in the expected LARGE ballpark (Hilbert-8 κ≈1.5e10 in double; float resolves
        // ≳1e6 before hitting its own precision floor), and reconstruction holds.
        void ThinGalleryHilbert_8()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 8;
            var A = arena.floatHilbert(n);
            var U = arena.floatMat(n, n);
            var S = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            Assert.IsTrue(SVD.thin(in A, ref U, ref S, ref V));
            Assert.IsFalse(Analysis.isAnyNan(in S));

            AssertDescendingNonNegative(in S, n);
            for (int i = 0; i < n; i++) { if (!(S[i] > (float)0)) Record(S[i], (float)0, S[i]); Assert.IsTrue(S[i] > (float)0); }

            // Large condition number (lenient lower bound so FLOAT, capped by its precision floor, still passes).
            float cond = S[0] / S[n - 1];
            AssertGEf(cond, (float)1E3f);

            AssertReconstruct(in A, in U, in S, in V, ref arena, (float)8 * Consts.floatSqrtEps * (S[0] + (float)1));
            arena.Dispose();
        }

        // Gallery ill-conditioned (Kahan, θ=1.2): upper-triangular classic QRCP counterexample.
        // Assert σ sorted-descending positive, κ large, reconstruction holds.
        void ThinGalleryKahan_12()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 12;
            var A = arena.floatKahan(n, (float)1.2f);
            var U = arena.floatMat(n, n);
            var S = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            Assert.IsTrue(SVD.thin(in A, ref U, ref S, ref V));
            Assert.IsFalse(Analysis.isAnyNan(in S));

            AssertDescendingNonNegative(in S, n);
            for (int i = 0; i < n; i++) { if (!(S[i] > (float)0)) Record(S[i], (float)0, S[i]); Assert.IsTrue(S[i] > (float)0); }

            float cond = S[0] / S[n - 1];
            AssertGEf(cond, (float)10f);   // Kahan is ill-conditioned; lenient bound holds for both types

            AssertReconstruct(in A, in U, in S, in V, ref arena, (float)8 * Consts.floatSqrtEps * (S[0] + (float)1));
            arena.Dispose();
        }

        // Solver API rework (commit 2): SVD.pinvSolve must treat x as OUTPUT ONLY -- prior garbage
        // (here, NaN sentinels) must not survive into the result.
        void PinvSolveUninitXContract()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 8, n = 4;
            var A = arena.floatRandomMat(m, n, -5f, 5f, 434343);
            for (int d = 0; d < n; d++) A[d, d] += (float)10f;
            var xKnown = arena.floatRandomVec(n, -3f, 3f, 545454);
            var b = arena.floatVec(m);
            Blas.dot(in A, in xKnown, ref b);

            var x = arena.floatVec(n);
            for (int i = 0; i < n; i++) x[i] = float.NaN;

            RankInfo pinvInfo = SVD.pinvSolve(ref A, in b, ref x);
            bool converged = pinvInfo;
            int rank = pinvInfo.rank;

            Assert.IsTrue(converged);
            RecordEq(rank, n);
            Assert.IsFalse(Analysis.isAnyNan(in x));

            for (int i = 0; i < n; i++)
            {
                float diff = math.abs(x[i] - xKnown[i]);
                float tol = (float)Consts.floatSqrtEps * (float)10 * (math.abs(xKnown[i]) + (float)1);
                if (!(diff <= tol) && Fail[0] == (float)0)
                {
                    Fail[0] = (float)1; Fail[1] = x[i]; Fail[2] = xKnown[i]; Fail[3] = diff;
                }
                Assert.IsTrue(diff <= tol);
            }

            arena.Dispose();
        }

        // ================================================================================
        // Commit 2.5 SVD coverage restoration (replaces oracle role of the deleted Jacobi SVD).
        // ================================================================================

        // (2b) Independent-algorithm cross-check. The singular values from Golub-Kahan (SVD.values)
        // must satisfy σ_i^2 == λ_i where λ_i are the eigenvalues of the Gram matrix AᵀA obtained
        // from a GENUINELY DIFFERENT algorithm (Householder tridiagonalization + implicit QL in
        // Eigen.valuesSymmetric) -- so agreement is real, independent validation, not circular.
        // ALSO checks the Frobenius identity Σσ_i^2 == ‖A‖_F^2 (free, holds for ANY A).
        void CrossCheckEigenRandom(int m, int n, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatRandomMat(m, n, -5f, 5f, seed);
            CrossCheckEigenCore(ref arena, in A, m, n);
            arena.Dispose();
        }

        // Clustered-σ variant: [10,10,10,3,2,1,0.5,0.25] embedded in a 12x8 A via Haar U/V (randsvd).
        void CrossCheckEigenClustered12x8()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 12, n = 8;
            var sigma = arena.floatVec(n);
            sigma[0]=(float)10; sigma[1]=(float)10; sigma[2]=(float)10; sigma[3]=(float)3;
            sigma[4]=(float)2;  sigma[5]=(float)1;  sigma[6]=(float)0.5f; sigma[7]=(float)0.25f;
            var A = arena.floatMat(m, n);
            var rng = new Unity.Mathematics.Random(0x5EED0013u);
            BuildRandSvd(ref rng, m, n, in sigma, ref A);
            CrossCheckEigenCore(ref arena, in A, m, n);
            arena.Dispose();
        }

        void CrossCheckEigenCore(ref Arena arena, in floatMxN A, int m, int n)
        {
            // (1) singular values via Golub-Kahan; A preserved.
            var S = arena.floatVec(n);
            Assert.IsTrue(SVD.values(in A, ref S));
            Assert.IsFalse(Analysis.isAnyNan(in S));
            AssertDescendingNonNegative(in S, n);

            // (2) Gram matrix AᵀA (n x n, symmetric).
            var At = Blas.trans(A);
            var AtA = Blas.dot(At, A);

            // (3) eigenvalues of AᵀA (DESTROYS AtA; sorted DESCENDING -- same convention as S).
            var lambda = arena.floatVec(n);
            Assert.IsTrue(Eigen.valuesSymmetric(ref AtA, ref lambda));

            // (4) σ_i^2 ≈ λ_i, LOOSE tolerance scaled by σ_0^2 (squaring roughly squares κ, so tiny
            // trailing σ can have large relative error in this comparison -- that's expected). The
            // constant is intentionally generous: this cross-check validates AGREEMENT between two
            // independent algorithms, whose σ^2-vs-λ discrepancy on clustered spectra runs ~1e-7
            // relative in DOUBLE (well above double's sqrtEps≈1.5e-8) yet stays ~1e-3 relative in
            // FLOAT (sqrtEps≈3.4e-4) -- 64·sqrtEps·σ_0^2 covers both with margin while remaining far
            // below the O(σ_0^2) error a genuine algorithm bug would produce.
            float sigma0sq = S[0] * S[0];
            float tol = (float)64 * Consts.floatSqrtEps * (sigma0sq + (float)1);
            for (int i = 0; i < n; i++)
            {
                float si2 = S[i] * S[i];
                float diff = math.abs(si2 - lambda[i]);
                if (!(diff <= tol) && Fail[0] == (float)0)
                {
                    Fail[0] = (float)1; Fail[1] = si2; Fail[2] = lambda[i]; Fail[3] = diff;
                }
                Assert.IsTrue(diff <= tol);
            }

            // (5) Frobenius identity Σσ_i^2 == ‖A‖_F^2.
            AssertFrobeniusIdentity(in A, in S, n, (float)64 * Consts.floatSqrtEps);
        }

        // Frobenius identity: Σ σ_i^2 == ‖A‖_F^2 (== Norms.L2(A)^2). Holds for ANY matrix -- a
        // reconstruction-independent invariant. tolFactor is a RELATIVE bound scaled by the squared
        // norm (per-precision via Consts.floatSqrtEps).
        void AssertFrobeniusIdentity(in floatMxN A, in floatN S, int n, float tolFactor)
        {
            float sumSq = (float)0;
            for (int i = 0; i < n; i++) sumSq += S[i] * S[i];
            float fro = Norms.L2(in A);
            float froSq = fro * fro;
            float tol = tolFactor * (froSq + (float)1);
            float diff = math.abs(sumSq - froSq);
            if (!(diff <= tol) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1; Fail[1] = sumSq; Fail[2] = froSq; Fail[3] = diff;
            }
            Assert.IsTrue(diff <= tol);
        }

        // (2c-i) Known-Σ via Gram-Schmidt on DELIBERATELY COHERENT (overlapping-ramp) vectors --
        // different statistics than BuildRandSvd's Haar-random U/V. Build k=3 orthonormal u_i (m-vec)
        // and v_i (n-vec) by MGS on overlapping ramps, form A = Σ σ_i u_i v_iᵀ with hand-chosen
        // descending σ = [10,4,1]; SVD.thin must recover exactly those (+ zeros).
        void KnownSigmaGramSchmidtCoherent()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 12, n = 10, k = 3;

            // coherent overlapping-ramp raw vectors (independent, so MGS yields a full rank-k basis).
            var Umat = arena.floatMat(m, k);
            var Vmat = arena.floatMat(n, k);
            for (int c = 0; c < k; c++)
            {
                for (int i = 0; i < m; i++) Umat[i, c] = (float)math.max(0, i - 2 * c + 3);
                for (int i = 0; i < n; i++) Vmat[i, c] = (float)math.max(0, i - 3 * c + 4);
            }
            GramSchmidtColumns(ref Umat, m, k);
            GramSchmidtColumns(ref Vmat, n, k);

            var sigma3 = arena.floatVec(k);
            sigma3[0] = (float)10; sigma3[1] = (float)4; sigma3[2] = (float)1;

            var A = arena.floatMat(m, n);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    for (int c = 0; c < k; c++)
                        acc += (double)sigma3[c] * (double)Umat[i, c] * (double)Vmat[j, c];
                    A[i, j] = (float)acc;
                }

            var U = arena.floatMat(m, n);
            var S = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            Assert.IsTrue(SVD.thin(in A, ref U, ref S, ref V));
            Assert.IsFalse(Analysis.isAnyNan(in S));
            AssertDescendingNonNegative(in S, n);

            float svTol = (float)8 * Consts.floatSqrtEps * (sigma3[0] + (float)1);
            AssertClose(S[0], (float)10, svTol);
            AssertClose(S[1], (float)4,  svTol);
            AssertClose(S[2], (float)1,  svTol);
            for (int i = k; i < n; i++) AssertClose(S[i], (float)0, svTol);

            AssertReconstruct(in A, in U, in S, in V, ref arena, svTol);

            arena.Dispose();
        }

        // Modified Gram-Schmidt orthonormalization of the k COLUMNS of M (rows x k), in place.
        // double accumulation internally (like BuildRandSvd) then cast to float.
        void GramSchmidtColumns(ref floatMxN M, int rows, int k)
        {
            for (int c = 0; c < k; c++)
            {
                for (int p = 0; p < c; p++)
                {
                    double dot = 0;
                    for (int i = 0; i < rows; i++) dot += (double)M[i, p] * (double)M[i, c];
                    for (int i = 0; i < rows; i++) M[i, c] = (float)((double)M[i, c] - dot * (double)M[i, p]);
                }
                double nrm = 0;
                for (int i = 0; i < rows; i++) nrm += (double)M[i, c] * (double)M[i, c];
                nrm = math.sqrt(nrm);
                for (int i = 0; i < rows; i++) M[i, c] = (float)((double)M[i, c] / nrm);
            }
        }

        // (2d) Ported from the deleted Jacobi-oracle SVDKnown2x2, now against Golub-Kahan SVD.thin.
        // A = [[3,0],[4,5]] -> singular values sqrt(45)≈6.7082039, sqrt(5)≈2.2360680 (descending).
        void Known2x2GolubKahan()
        {
            var arena = new Arena(Allocator.Persistent);
            int dim = 2;

            var A = arena.floatMat(dim, dim);
            A[0, 0] = 3f; A[0, 1] = 0f;
            A[1, 0] = 4f; A[1, 1] = 5f;

            var U = arena.floatMat(dim, dim);
            var S = arena.floatVec(dim);
            var V = arena.floatMat(dim, dim);
            Assert.IsTrue(SVD.thin(in A, ref U, ref S, ref V));
            Assert.IsFalse(Analysis.isAnyNan(in S));

            AssertClose(S[0], (float)6.7082039f, (float)1E-3f);
            AssertClose(S[1], (float)2.2360680f, (float)1E-3f);
            AssertDescendingNonNegative(in S, dim);
            Assert.IsTrue(Analysis.isOrthogonal(U, (float)1E-4f));
            Assert.IsTrue(Analysis.isOrthogonal(V, (float)1E-4f));
            AssertReconstruct(in A, in U, in S, in V, ref arena, (float)1E-4f);

            arena.Dispose();
        }

        // (2d) Ported from the deleted SVDSingleColumn. 5x1 column [1,2,3,4,5]: single singular value
        // = column 2-norm = sqrt(55)≈7.4161985 (m=5 >= n=1 satisfies thin's requirement).
        void SingleColumn5x1()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 5, n = 1;

            var A = arena.floatMat(m, n);
            A[0, 0] = 1f; A[1, 0] = 2f; A[2, 0] = 3f; A[3, 0] = 4f; A[4, 0] = 5f;

            var U = arena.floatMat(m, n);
            var S = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            Assert.IsTrue(SVD.thin(in A, ref U, ref S, ref V));
            Assert.IsFalse(Analysis.isAnyNan(in S));

            AssertClose(S[0], (float)7.4161985f, (float)1E-3f);

            // U column has unit norm.
            float normSq = (float)0f;
            for (int i = 0; i < m; i++) normSq += U[i, 0] * U[i, 0];
            AssertClose(normSq, (float)1f, (float)1E-4f);

            AssertReconstruct(in A, in U, in S, in V, ref arena, (float)1E-4f);

            arena.Dispose();
        }

        // (2d) Ported from the deleted SVDNonConvergence, adapted to the CURRENT Golub-Kahan
        // implementation. maxIter=1 cannot isolate an 8x8 bidiagonal block in a single per-value
        // iteration, so the bidiagonal QR genuinely returns false (non-convergent). In the current
        // impl, S is written ONLY inside `if (ok)`, so on non-convergence S retains its (zero) pre-fill
        // while U/V still hold the finite orthonormal output of the unconditional Bidiag.decomp. The
        // REAL, checkable regression guard: NO NaN/Inf is EVER written to S/U/V regardless of
        // convergence, and S stays descending & non-negative. The convergence flag itself is NOT
        // hard-asserted (matching the deleted test's own choice).
        void NonConvergenceHilbert8()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 8;
            var A = arena.floatHilbert(n);

            // pre-fill outputs with normal (zero) starting values, NOT a NaN sentinel.
            var U = arena.floatMat(n, n);
            var S = arena.floatVec(n);
            var V = arena.floatMat(n, n);

            SVD.thin(in A, ref U, ref S, ref V, 1);

            Assert.IsFalse(Analysis.isAnyNan(in S));
            Assert.IsFalse(Analysis.isAnyNan(in U));
            Assert.IsFalse(Analysis.isAnyNan(in V));
            Assert.IsFalse(Analysis.isAnyInf(in S));
            Assert.IsFalse(Analysis.isAnyInf(in U));
            Assert.IsFalse(Analysis.isAnyInf(in V));
            AssertDescendingNonNegative(in S, n);

            // Same guarantee for the values-only path.
            var S2 = arena.floatVec(n);
            SVD.values(in A, ref S2, 1, Consts.floatZeroThreshold);
            Assert.IsFalse(Analysis.isAnyNan(in S2));
            Assert.IsFalse(Analysis.isAnyInf(in S2));
            AssertDescendingNonNegative(in S2, n);

            arena.Dispose();
        }

        // (2e) Determinant invariant: for a well-conditioned square A, |det A| == Π σ_i. det via LU
        // (with pivot sign) on a COPY; Π σ via SVD.values on the untouched ORIGINAL. Loose RELATIVE
        // tolerance growing mildly with n (accumulated product of n rounded factors).
        void DetEqualsProductSingularValues()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 7;

            // well-conditioned: diagonal-boosted random (avoids singular/near-singular).
            var A = arena.floatRandomMat(n, n, -3f, 3f, 6543210);
            for (int d = 0; d < n; d++) A[d, d] += (float)(2 * n);

            // |det| via LU on a COPY.
            var Acopy = A.Copy();
            var pivot = new Pivot(n, Allocator.Temp);
            Assert.IsTrue(LU.decompInPlace(ref Acopy, ref pivot));
            float detAbs = math.abs(Analysis.determinant(in Acopy, in pivot));
            pivot.Dispose();

            // Π σ_i via SVD.values on the ORIGINAL untouched A.
            var S = arena.floatVec(n);
            Assert.IsTrue(SVD.values(in A, ref S));
            float prod = (float)1;
            for (int i = 0; i < n; i++) prod *= S[i];

            float relTol = (float)n * (float)8 * Consts.floatSqrtEps;
            float denom = math.max((float)1, math.abs(prod));
            float relDiff = math.abs(detAbs - prod) / denom;
            if (!(relDiff <= relTol) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1; Fail[1] = detAbs; Fail[2] = prod; Fail[3] = relDiff;
            }
            Assert.IsTrue(relDiff <= relTol);

            arena.Dispose();
        }

        void RecordEq(int got, int expected)
        {
            if (got != expected && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = (float)got;
                Fail[2] = (float)expected;
                Fail[3] = (float)(got - expected);
            }
            Assert.AreEqual(expected, got);
        }

        // Fail layout: [1]=val, [2]=limit, [3]=limit-val
        void AssertGEf(float val, float limit)
        {
            if (!(val >= limit) && Fail[0] == (float)0) { Fail[0] = (float)1; Fail[1] = val; Fail[2] = limit; Fail[3] = limit - val; }
            Assert.IsTrue(val >= limit);
        }

        void Record(float got, float expected, float diff)
        {
            if (Fail[0] == (float)0) { Fail[0] = (float)1; Fail[1] = got; Fail[2] = expected; Fail[3] = diff; }
        }


        // GALLERY KNOWN-ANSWER (Gallery.Special): the 4x4 Sylvester-Walsh Hadamard matrix satisfies
        // HᵀH = n·I, so ALL singular values equal √n = √4 = 2 and the condition number is exactly 1.
        // Uses SVD.singularValues (A is not modified, S is sorted descending).
        public void SVDGalleryHadamard()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;

            var A = arena.floatHadamard(n);
            var S = arena.floatVec(n);

            int k = SVD.singularValues(in A, ref S);
            AssertClose((float)k, (float)n, 1E-6f);

            Assert.IsFalse(Analysis.isAnyNan(in S));

            // every singular value == sqrt(4) == 2 (cond = 1)
            for (int i = 0; i < n; i++)
                AssertClose(S[i], (float)2f, 1E-4f);

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

            var A = arena.floatParter(n);
            var S = arena.floatVec(n);

            SVD.singularValues(in A, ref S);

            Assert.IsFalse(Analysis.isAnyNan(in S));

            float pi = (float)Unity.Mathematics.math.PI_DBL;
            // boundary lies within ~1e-10 of π; absorb float SVD error without masking a real overshoot.
            float margin = (float)64 * Consts.floatSqrtEps;

            for (int i = 0; i < n; i++)
            {
                bool below = S[i] <= pi + margin;
                if (!below && Fail[0] == (float)0)
                {
                    Fail[0] = (float)1;
                    Fail[1] = S[i];
                    Fail[2] = pi;
                    Fail[3] = (float)i;
                }
                Assert.IsTrue(below);
            }

            // largest singular value clusters near π (close from below).
            AssertClose(S[0], pi, margin);

            AssertDescendingNonNegative(in S, n);

            arena.Dispose();
        }

        // ---- values (singular VALUES only, A unmodified) ----

        // Identity n=5 -> all singular values 1.
        public void SVValuesIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;

            var A = arena.floatIdentityMat(n);
            var S = arena.floatVec(n);

            bool ok = SVD.values(in A, ref S);
            Assert.IsTrue(ok);

            Assert.IsFalse(Analysis.isAnyNan(in S));

            for (int i = 0; i < n; i++)
                AssertClose(S[i], (float)1f, 1E-4f);

            AssertDescendingNonNegative(in S, n);

            // A must be unchanged (still identity).
            Assert.IsTrue(Analysis.isIdentity(in A, 1E-5f));

            arena.Dispose();
        }

        // Diagonal diag(d) -> singular values = |d_i| sorted descending.
        public void SVValuesDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;

            var A = arena.floatMat(n, n);
            A[0, 0] = 3f;
            A[1, 1] = -2f;
            A[2, 2] = 0.5f;
            A[3, 3] = 5f;
            A[4, 4] = -1f;

            var Apristine = A.Copy();

            var S = arena.floatVec(n);

            bool ok = SVD.values(in A, ref S);
            Assert.IsTrue(ok);

            Assert.IsFalse(Analysis.isAnyNan(in S));

            // |d| sorted descending: 5, 3, 2, 1, 0.5
            AssertClose(S[0], (float)5f, 1E-4f);
            AssertClose(S[1], (float)3f, 1E-4f);
            AssertClose(S[2], (float)2f, 1E-4f);
            AssertClose(S[3], (float)1f, 1E-4f);
            AssertClose(S[4], (float)0.5f, 1E-4f);

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

            var A = arena.floatMat(n, n);
            A[0, 0] = 3f; A[0, 1] = 0f;
            A[1, 0] = 0f; A[1, 1] = -4f;

            var S = arena.floatVec(n);

            bool ok = SVD.values(in A, ref S);
            Assert.IsTrue(ok);

            Assert.IsFalse(Analysis.isAnyNan(in S));

            AssertClose(S[0], (float)4f, 1E-4f);
            AssertClose(S[1], (float)3f, 1E-4f);

            AssertDescendingNonNegative(in S, n);

            arena.Dispose();
        }

        // Rank-1 outer product u*v^T -> exactly one positive singular value (= |u|*|v|), rest ~0.
        public void SVValuesRankDeficient()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;

            var u = arena.floatVec(n);
            u[0] = 1f; u[1] = -2f; u[2] = 3f; u[3] = 0.5f; u[4] = -1.5f;
            var v = arena.floatVec(n);
            v[0] = 2f; v[1] = 1f; v[2] = -1f; v[3] = 4f; v[4] = 0.25f;

            var A = arena.floatOuter(in u, in v);
            var Apristine = A.Copy();

            var S = arena.floatVec(n);

            bool ok = SVD.values(in A, ref S);
            Assert.IsTrue(ok);

            Assert.IsFalse(Analysis.isAnyNan(in S));

            // expected sole singular value = ||u|| * ||v||
            float nu = (float)0f, nv = (float)0f;
            for (int i = 0; i < n; i++) { nu += u[i] * u[i]; nv += v[i] * v[i]; }
            float sigma = math.sqrt(nu) * math.sqrt(nv);

            AssertClose(S[0], sigma, (float)1E-3f + (float)1E-4f * sigma);

            // the rest collapse to ~0
            for (int i = 1; i < n; i++)
                AssertClose(S[i], (float)0f, (float)1E-3f + (float)1E-4f * sigma);

            AssertDescendingNonNegative(in S, n);

            AssertMatrixUnchanged(in A, in Apristine, n, n);

            arena.Dispose();
        }

        // Cross-check values vs the trusted Golub-Kahan full SVD (SVD.thin) for m >= n (square AND
        // tall). thin factors a copy of A; values takes A `in` (must be unmodified).
        public void SVValuesCross(int m, int n, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.floatRandomMat(m, n, -10f, 10f, seed);
            var Apristine = A.Copy();

            // reference path: full thin SVD on a copy of A
            var Aref = A.Copy();
            var U = arena.floatMat(m, n);
            var Sref = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            bool okRef = SVD.thin(in Aref, ref U, ref Sref, ref V);
            Assert.IsTrue(okRef);

            // values-only path on the untouched A
            var S = arena.floatVec(n);
            bool ok = SVD.values(in A, ref S);
            Assert.IsTrue(ok);

            Assert.IsFalse(Analysis.isAnyNan(in S));
            Assert.IsFalse(Analysis.isAnyNan(in Sref));

            AssertDescendingNonNegative(in S, n);
            AssertDescendingNonNegative(in Sref, n);

            // agree element-wise (both descending) with a scale-aware tolerance.
            for (int i = 0; i < n; i++)
            {
                float scale = math.max(math.abs(S[i]), math.abs(Sref[i]));
                float tol = (float)1E-3f + (float)1E-3f * scale;
                AssertClose(S[i], Sref[i], tol);
            }

            // values must NOT have modified A.
            AssertMatrixUnchanged(in A, in Apristine, m, n);

            arena.Dispose();
        }

        // Validates the Golub-Kahan full SVD (SVD.thin) via reconstruction A = U diag(S) Vᵀ and
        // orthonormal U,V, across square and tall shapes (A must be unmodified). Orthogonal U,V plus
        // an accurate reconstruction is sufficient to pin S as the true singular values -- no external
        // oracle needed (see ThinKnown* for the known-Σ variant of this same guarantee).
        public void GolubKahanCross(int m, int n, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.floatRandomMat(m, n, -10f, 10f, seed);
            var Apristine = A.Copy();

            var U = arena.floatMat(m, n);
            var S = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            bool ok = SVD.thin(in A, ref U, ref S, ref V);
            Assert.IsTrue(ok);

            Assert.IsFalse(Analysis.isAnyNan(in S));
            AssertDescendingNonNegative(in S, n);

            float reconTol = (float)1E-3f + (float)1E-4f * math.abs(S[0]);
            AssertReconstruct(in A, in U, in S, in V, ref arena, reconTol);
            Assert.IsTrue(Analysis.isOrthogonal(U, (float)1E-3f));
            Assert.IsTrue(Analysis.isOrthogonal(V, (float)1E-3f));

            AssertMatrixUnchanged(in A, in Apristine, m, n);

            arena.Dispose();
        }

        // Rank-1 (rank-deficient) matrix: one nonzero singular value ||u||*||v||, the rest ~0.
        // Exercises the clustered-zero deflation in the bidiagonal QR.
        public void GolubKahanRankDeficient()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            var u = arena.floatVec(n);
            u[0] = 1f; u[1] = -2f; u[2] = 3f; u[3] = 0.5f; u[4] = -1.5f;
            var v = arena.floatVec(n);
            v[0] = 2f; v[1] = 1f; v[2] = -1f; v[3] = 4f; v[4] = 0.25f;

            var A = arena.floatOuter(in u, in v);
            var Apristine = A.Copy();

            var U = arena.floatMat(n, n);
            var S = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            bool ok = SVD.thin(in A, ref U, ref S, ref V);
            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in S));

            float nu = (float)0f, nv = (float)0f;
            for (int i = 0; i < n; i++) { nu += u[i] * u[i]; nv += v[i] * v[i]; }
            float sigma = math.sqrt(nu) * math.sqrt(nv);

            AssertClose(S[0], sigma, (float)1E-3f + (float)1E-3f * sigma);
            for (int i = 1; i < n; i++)
                AssertClose(S[i], (float)0f, (float)1E-3f + (float)1E-3f * sigma);

            AssertDescendingNonNegative(in S, n);
            AssertReconstruct(in A, in U, in S, in V, ref arena, (float)1E-3f + (float)1E-4f * sigma);
            AssertMatrixUnchanged(in A, in Apristine, n, n);

            arena.Dispose();
        }

        // Fully clustered spectrum: A = 3*I has all singular values equal to 3. Stresses deflation.
        public void GolubKahanClustered()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;
            var A = arena.floatMat(n, n);
            for (int i = 0; i < n; i++) A[i, i] = (float)3f;
            var Apristine = A.Copy();

            var U = arena.floatMat(n, n);
            var S = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            bool ok = SVD.thin(in A, ref U, ref S, ref V);
            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in S));

            for (int i = 0; i < n; i++)
                AssertClose(S[i], (float)3f, (float)1E-3f);

            AssertReconstruct(in A, in U, in S, in V, ref arena, (float)1E-3f);
            Assert.IsTrue(Analysis.isOrthogonal(U, (float)1E-3f));
            Assert.IsTrue(Analysis.isOrthogonal(V, (float)1E-3f));
            AssertMatrixUnchanged(in A, in Apristine, n, n);

            arena.Dispose();
        }

        // Zero matrix: anorm == 0 → deflation threshold 0; must still converge (no NaN), S all 0.
        public void GolubKahanZero()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            var A = arena.floatMat(n, n);   // all zeros

            var U = arena.floatMat(n, n);
            var S = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            bool ok = SVD.thin(in A, ref U, ref S, ref V);
            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in S));
            for (int i = 0; i < n; i++)
                AssertClose(S[i], (float)0f, (float)1E-5f);

            arena.Dispose();
        }

        // Rank-3 6x6 (sum of three independent outer products) → 3 nonzero + 3 zero singular values.
        // The INTERIOR zeros exercise the cancellation branch (|d[nm]| <= thresh) of the bidiagonal QR.
        public void GolubKahanRank3()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;
            var u1 = arena.floatVec(n); u1[0]=1f; u1[1]=0f; u1[2]=0f; u1[3]=1f; u1[4]=0f; u1[5]=0f;
            var v1 = arena.floatVec(n); v1[0]=1f; v1[1]=2f; v1[2]=0f; v1[3]=0f; v1[4]=0f; v1[5]=0f;
            var u2 = arena.floatVec(n); u2[0]=0f; u2[1]=1f; u2[2]=0f; u2[3]=0f; u2[4]=1f; u2[5]=0f;
            var v2 = arena.floatVec(n); v2[0]=0f; v2[1]=0f; v2[2]=1f; v2[3]=3f; v2[4]=0f; v2[5]=0f;
            var u3 = arena.floatVec(n); u3[0]=0f; u3[1]=0f; u3[2]=1f; u3[3]=0f; u3[4]=0f; u3[5]=1f;
            var v3 = arena.floatVec(n); v3[0]=0f; v3[1]=0f; v3[2]=0f; v3[3]=0f; v3[4]=1f; v3[5]=2f;

            var A = arena.floatMat(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = u1[i] * v1[j] + u2[i] * v2[j] + u3[i] * v3[j];
            var Apristine = A.Copy();

            var U = arena.floatMat(n, n);
            var S = arena.floatVec(n);
            var V = arena.floatMat(n, n);
            bool ok = SVD.thin(in A, ref U, ref S, ref V);
            Assert.IsTrue(ok);
            Assert.IsFalse(Analysis.isAnyNan(in S));
            AssertDescendingNonNegative(in S, n);

            // bottom three singular values are the (interior) zeros
            for (int i = 3; i < n; i++)
                AssertClose(S[i], (float)0f, (float)1E-3f + (float)1E-3f * S[0]);

            AssertReconstruct(in A, in U, in S, in V, ref arena, (float)1E-3f + (float)1E-4f * S[0]);
            Assert.IsTrue(Analysis.isOrthogonal(U, (float)1E-3f));
            Assert.IsTrue(Analysis.isOrthogonal(V, (float)1E-3f));
            AssertMatrixUnchanged(in A, in Apristine, n, n);

            // (2c-ii) Explicit DETECTED RANK via the tail: count σ_i above a relative threshold; the
            // matrix is a sum of k=3 rank-1 outer products so exactly n-k = 3 trailing σ are ~0 and
            // the detected rank must be 3.
            int k = 3;
            float rankTol = (float)8 * Consts.floatSqrtEps;
            int detectedRank = 0;
            for (int i = 0; i < n; i++)
                if (S[i] > rankTol * S[0]) detectedRank++;
            if (detectedRank != k && Fail[0] == (float)0)
            {
                Fail[0] = (float)1; Fail[1] = (float)detectedRank; Fail[2] = (float)k; Fail[3] = (float)(detectedRank - k);
            }
            Assert.IsTrue(detectedRank == k);

            // (2c-ii) Frobenius identity Σσ_i^2 == ‖A‖_F^2 (holds for ANY A).
            AssertFrobeniusIdentity(in A, in S, n, (float)64 * Consts.floatSqrtEps);

            arena.Dispose();
        }

        // Fail layout: [1]=A[i,j], [2]=ref[i,j], [3]=diff
        private void AssertMatrixUnchanged(in floatMxN A, in floatMxN B, int m, int n)
        {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float diff = math.abs(A[i, j] - B[i, j]);
                    bool same = diff <= (float)1E-6f;
                    if (!same && Fail[0] == (float)0)
                    {
                        Fail[0] = (float)1;
                        Fail[1] = A[i, j];
                        Fail[2] = B[i, j];
                        Fail[3] = diff;
                    }
                    Assert.IsTrue(same);
                }
        }

        private void AssertReconstruct(in floatMxN A, in floatMxN U, in floatN S, in floatMxN V, ref Arena arena, float precision)
        {
            var diagS = arena.floatDiagonalMat(in S);
            var US = Blas.dot(U, diagS);
            var Vt = Blas.trans(V);
            var recon = Blas.dot(US, Vt);

            floatMxN shouldBeZero = A - recon;

            if (Analysis.isAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            var zeroError = Analysis.MaxZeroError(shouldBeZero);

            // Fail layout: [1]=zeroError, [2]=precision, [3]=diff
            if (!(zeroError <= precision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = zeroError;
                Fail[2] = precision;
                Fail[3] = zeroError - precision;
            }
            Assert.IsTrue(Analysis.isZero(in shouldBeZero, precision));
        }

        // Fail layout: [1]=S[i] (offending element), [2]=bound or S[i-1], [3]=index cast to float
        private void AssertDescendingNonNegative(in floatN S, int n)
        {
            for (int i = 0; i < n; i++)
            {
                bool nonNeg = S[i] >= (float)(-1E-6f);
                if (!nonNeg && Fail[0] == (float)0)
                {
                    Fail[0] = (float)1;
                    Fail[1] = S[i];
                    Fail[2] = (float)(-1E-6f);
                    Fail[3] = (float)i;
                }
                Assert.IsTrue(nonNeg);
            }

            for (int i = 1; i < n; i++)
            {
                bool descending = S[i] <= S[i - 1] + (float)1E-6f;
                if (!descending && Fail[0] == (float)0)
                {
                    Fail[0] = (float)1;
                    Fail[1] = S[i];
                    Fail[2] = S[i - 1];
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

    }

    public static Array GetEnums() {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void SVDDecompTests(TestJob.TestType type)
    {
        var fail = new NativeArray<float>(4, Allocator.TempJob);
        try {
            new TestJob() { Type = type, Fail = fail }.Run();
            // Burst in-job asserts abort without throwing; diagnostics surfaced here too
            // (see floatQRTests.QRDecompTests).
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

    // Managed throw-tests: argument validation runs on the main thread (not in a Burst job).

    [Test]
    public void SVValuesThrowsOnWideMatrix()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.floatMat(2, 3);
        var S = arena.floatVec(3);

        Assert.Catch<ArgumentException>(() => SVD.values(in A, ref S));

        arena.Dispose();
    }

    [Test]
    public void SVValuesThrowsOnWrongSLength()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.floatMat(4, 3);
        var S = arena.floatVec(2);

        Assert.Catch<ArgumentException>(() => SVD.values(in A, ref S));

        arena.Dispose();
    }

}
