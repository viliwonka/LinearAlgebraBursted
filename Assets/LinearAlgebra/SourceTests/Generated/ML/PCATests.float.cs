using System;

using LinearAlgebra;
using LinearAlgebra.ML;        // floatPCA_OP, floatPCAModel, PCAScaling
using LinearAlgebra.Stats;     // floatStats_OP.colMean (oracle side-uses)

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// PCA (LinearAlgebra.ML.floatPCA_OP) — the four fit routes (pcaCovariance / pcaSVD /
// pcaSVDTruncated / pcaRandomized), the floatPCAModel, PCAScaling, and pcaTransform. Tests mirror
// the KMeans / SVDRandomized idiom: a [BurstCompile(FloatPrecision.High)] IJob carries a TestType
// enum, a Fail NativeArray diagnostic channel, and a [TestCaseSource] driver; the managed-throw
// guard paths run as plain [Test]s on the main thread (throws need no Burst context).
//
// Acceptance-criterion coverage (spec docs/spec-pca.md "Tests" + task extras):
//   CrossRouteCovariance / CrossRouteCorrelation  — #1 the cross-route oracle: pcaCovariance ==
//        pcaSVD on explainedVariance (tight) + components (up to sign) for BOTH scalings. Proves the
//        (n-1) denominator convention and the inline correlation handling.
//   KnownSpectrum            — #2 diag(9,4,1) exact-construction data → variances recovered descending,
//        components axis-aligned (identity).
//   VarianceRatioAndOrder    — #3 explainedVarianceRatio sums to ~1 (full routes), ev[0] is the max,
//        ev descending.
//   CorrelationDegenerate    — #4 the fable-caught trap: a constant column ⇒ scale=1, a zero
//        eigen-axis on that feature, totalVariance == #non-degenerate, and BOTH correlation routes
//        still AGREE on explainedVariance (regression guard on the inline-R fix).
//   TopKExactMatchesFull     — #5a pcaSVDTruncated(k) top-k == first k of pcaSVD (tight, up to sign),
//        ratios sum < 1.
//   TopKRandomizedApprox     — #5b pcaRandomized(k) matches full top-k within a LOOSE tol, ratios < 1.
//   TransformScores          — #6 pcaTransform: score-column variance == explainedVariance (Covariance),
//        a hand-computed ((X-mean)/scale)·components matches, scores shape n×k.
//   SignDeterminismNegate    — #7a negate an input column, refit ⇒ identical components (axis-aligned,
//        well-separated spectrum).
//   RandomizedBitwise        — #7b pcaRandomized(arena,X,k) twice ⇒ bitwise-identical (fixed seed).
//   WideCovariance           — #8 pcaCovariance works for p>n; trailing (p−rank) eigenvalues ≈ 0.
//   [Test] guards            — #9 n<2 / pcaSVD,pcaSVDTruncated,pcaRandomized on wide p>n / k out of
//        range / stale-k pcaTransform / mis-sized ref-form model+scores → ArgumentException.
//
// Tolerances scale with Consts.floatSqrtEps (float ≈ 3.45e-4, double ≈ 1.49e-8) so ONE expression is
// loose for float and tight for double. Eigenvalue/variance facts use a (|v|+1)-scaled band; the
// pcaCovariance route squares the condition number (κ²), so its float agreement with the SVD route is
// looser than double — the bands are sized for that. Any spectrum with a small eigen-gap is EXCLUDED
// from vector comparison via a relative-gap guard (rotation ambiguity a sign rule cannot fix, per the
// spec) — value comparisons still run on every component.
public class floatPCATests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            CrossRouteCovariance,
            CrossRouteCorrelation,
            KnownSpectrum,
            VarianceRatioAndOrder,
            CorrelationDegenerate,
            TopKExactMatchesFull,
            TopKRandomizedApprox,
            TransformScores,
            SignDeterminismNegate,
            RandomizedBitwise,
            WideCovariance,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<float> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.CrossRouteCovariance:  CrossRoute(PCAScaling.Covariance);  break;
                case TestType.CrossRouteCorrelation: CrossRoute(PCAScaling.Correlation); break;
                case TestType.KnownSpectrum:         KnownSpectrum();         break;
                case TestType.VarianceRatioAndOrder: VarianceRatioAndOrder(); break;
                case TestType.CorrelationDegenerate: CorrelationDegenerate(); break;
                case TestType.TopKExactMatchesFull:  TopKExactMatchesFull();  break;
                case TestType.TopKRandomizedApprox:  TopKRandomizedApprox();  break;
                case TestType.TransformScores:       TransformScores();       break;
                case TestType.SignDeterminismNegate: SignDeterminismNegate(); break;
                case TestType.RandomizedBitwise:     RandomizedBitwise();     break;
                case TestType.WideCovariance:        WideCovariance();        break;
            }
        }

        // =====================================================================
        // #1 — cross-route oracle. Same correlated data, moderately-decaying but
        //      WELL-SEPARATED spectrum. pcaCovariance and pcaSVD must return the
        //      SAME explainedVariance (tight) + scale, and the SAME components up
        //      to the fixed sign (gap-guarded, magnitude compare). Runs for BOTH
        //      Covariance and Correlation scaling — proving the (n-1) denominator
        //      AND the inline-correlation degenerate handling agree across routes.
        // =====================================================================
        void CrossRoute(PCAScaling scaling)
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 50, p = 6;
            var X = BuildCorrelated(ref arena, n, p);

            var mCov = floatPCA_OP.pcaCovariance(ref arena, in X, scaling);
            var mSvd = floatPCA_OP.pcaSVD(ref arena, in X, scaling);

            AssertTrue(mCov.converged);
            AssertTrue(mSvd.converged);
            RecordEq(mCov.k, p);
            RecordEq(mSvd.k, p);

            // explainedVariance agrees on EVERY component (values are gap-independent).
            for (int i = 0; i < p; i++)
                AssertClose(mCov.explainedVariance[i], mSvd.explainedVariance[i],
                            EvTol(mSvd.explainedVariance[i]));

            // scale agrees (all-ones for Covariance; sample std-dev for Correlation).
            for (int j = 0; j < p; j++)
                AssertClose(mCov.scale[j], mSvd.scale[j], EvTol(mSvd.scale[j]));

            // components agree up to the fixed sign — magnitude compare, ONLY on well-separated
            // columns (degenerate/near-tie columns carry route-dependent rotation the sign rule
            // cannot fix; the spec says keep vector comparisons off those).
            float ctol = (float)100 * Consts.floatSqrtEps;
            for (int c = 0; c < p; c++)
            {
                if (!WellSeparated(in mSvd.explainedVariance, c, p)) continue;
                for (int r = 0; r < p; r++)
                    AssertClose(math.abs(mCov.components[r, c]), math.abs(mSvd.components[r, c]), ctol);
            }

            arena.Dispose();
        }

        // =====================================================================
        // #2 — known spectrum. Orthogonal centered ±1 columns scaled to sample
        //      variances 9,4,1 → covariance is EXACTLY diag(9,4,1). pcaCovariance
        //      must recover [9,4,1] descending and identity components (axis-aligned).
        // =====================================================================
        void KnownSpectrum()
        {
            var arena = new Arena(Allocator.Persistent);

            var X = BuildDiagonal(ref arena);   // 4×3, sample cov == diag(9,4,1)
            int p = 3;

            var m = floatPCA_OP.pcaCovariance(ref arena, in X);   // Covariance
            AssertTrue(m.converged);
            RecordEq(m.k, p);

            // variances recovered, descending.
            AssertClose(m.explainedVariance[0], (float)9, EvTol((float)9));
            AssertClose(m.explainedVariance[1], (float)4, EvTol((float)4));
            AssertClose(m.explainedVariance[2], (float)1, EvTol((float)1));

            // components ≈ identity: |diag| ≈ 1, |off| ≈ 0.
            float ctol = (float)100 * Consts.floatSqrtEps;
            for (int c = 0; c < p; c++)
                for (int r = 0; r < p; r++)
                    AssertClose(math.abs(m.components[r, c]), (r == c) ? (float)1 : (float)0, ctol);

            // scale is all-ones in Covariance mode.
            for (int j = 0; j < p; j++)
                AssertClose(m.scale[j], (float)1, ctol);

            arena.Dispose();
        }

        // =====================================================================
        // #3 — explainedVarianceRatio sums to ~1 for a FULL route, explainedVariance[0]
        //      is the maximum, and explainedVariance is descending.
        // =====================================================================
        void VarianceRatioAndOrder()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 50, p = 6;
            var X = BuildCorrelated(ref arena, n, p);

            var m = floatPCA_OP.pcaCovariance(ref arena, in X);   // full route
            AssertTrue(m.converged);

            float sumRatio = (float)0;
            for (int i = 0; i < p; i++) sumRatio += m.explainedVarianceRatio[i];
            AssertClose(sumRatio, (float)1, (float)200 * Consts.floatSqrtEps);

            // ev[0] is the max, and the sequence is descending.
            float gapTol = (float)50 * Consts.floatSqrtEps;
            for (int i = 0; i < p; i++)
                AssertTrue(m.explainedVariance[0] + gapTol >= m.explainedVariance[i]);
            for (int i = 0; i + 1 < p; i++)
                AssertTrue(m.explainedVariance[i] + gapTol >= m.explainedVariance[i + 1]);

            // each ratio == ev / Σev (full route: totalVariance == Σ explainedVariance up to roundoff).
            float total = (float)0;
            for (int i = 0; i < p; i++) total += m.explainedVariance[i];
            for (int i = 0; i < p; i++)
                AssertClose(m.explainedVarianceRatio[i], m.explainedVariance[i] / total,
                            (float)50 * Consts.floatSqrtEps);

            arena.Dispose();
        }

        // =====================================================================
        // #4 — correlation degenerate feature (the fable-caught trap). A constant
        //      (zero-variance) column: scale==1, an isolated ZERO eigen-axis pinned
        //      to that feature (its component ≈ e_j), totalVariance == #non-degenerate,
        //      and pcaCovariance(Correlation) STILL AGREES with pcaSVD(Correlation) on
        //      explainedVariance. Without the inline-R fix, pcaCovariance would emit a
        //      spurious UNIT eigenvalue there (Σev == 6, not 5) — this catches that.
        // =====================================================================
        void CorrelationDegenerate()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 50, p = 6, deg = 3;   // feature `deg` is constant
            var X = BuildCorrelated(ref arena, n, p);
            for (int r = 0; r < n; r++) X[r, deg] = (float)7;   // zero-variance column

            var mCov = floatPCA_OP.pcaCovariance(ref arena, in X, PCAScaling.Correlation);
            var mSvd = floatPCA_OP.pcaSVD(ref arena, in X, PCAScaling.Correlation);
            AssertTrue(mCov.converged);
            AssertTrue(mSvd.converged);

            // zero-variance feature → scale exactly 1 (never a divide-by-zero).
            AssertExact(mCov.scale[deg], (float)1);
            AssertExact(mSvd.scale[deg], (float)1);

            // totalVariance == #non-degenerate features (5). For the FULL route Σev == totalVariance,
            // so Σ explainedVariance must be 5 (NOT 6 — the regression signature of a spurious unit λ).
            float sumCov = (float)0;
            for (int i = 0; i < p; i++) sumCov += mCov.explainedVariance[i];
            AssertClose(sumCov, (float)5, (float)500 * Consts.floatSqrtEps);

            // Σ explainedVarianceRatio ≈ 1 (== Σev / totalVariance == 5/5).
            float sumRatio = (float)0;
            for (int i = 0; i < p; i++) sumRatio += mCov.explainedVarianceRatio[i];
            AssertClose(sumRatio, (float)1, (float)200 * Consts.floatSqrtEps);

            // the degenerate axis: the smallest (last, descending) eigenvalue ≈ 0, and its ISOLATED
            // eigenvector is pinned to the constant feature (≈ ±e_deg).
            AssertClose(mCov.explainedVariance[p - 1], (float)0, (float)200 * Consts.floatSqrtEps);
            AssertTrue(math.abs(mCov.components[deg, p - 1]) > (float)0.99f);
            for (int r = 0; r < p; r++)
                if (r != deg)
                    AssertTrue(math.abs(mCov.components[r, p - 1]) < (float)0.05f);

            // cross-route AGREEMENT on explainedVariance (the headline correctness point).
            for (int i = 0; i < p; i++)
                AssertClose(mCov.explainedVariance[i], mSvd.explainedVariance[i],
                            EvTol(mSvd.explainedVariance[i]));

            arena.Dispose();
        }

        // =====================================================================
        // #5a — exact top-k. pcaSVDTruncated(k) top-k eigenpairs == first k of the
        //       full pcaSVD (tight, up to sign, gap-guarded); ratios sum < 1.
        // =====================================================================
        void TopKExactMatchesFull()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 50, p = 6, k = 3;
            var X = BuildCorrelated(ref arena, n, p);

            var full = floatPCA_OP.pcaSVD(ref arena, in X);              // k == p
            var trunc = floatPCA_OP.pcaSVDTruncated(ref arena, in X, k); // k == 3
            AssertTrue(full.converged);
            AssertTrue(trunc.converged);
            RecordEq(trunc.k, k);
            RecordEq(trunc.components.N_Cols, k);

            float ctol = (float)200 * Consts.floatSqrtEps;
            for (int i = 0; i < k; i++)
                AssertClose(trunc.explainedVariance[i], full.explainedVariance[i],
                            EvTol(full.explainedVariance[i]));

            for (int c = 0; c < k; c++)
            {
                if (!WellSeparated(in full.explainedVariance, c, p)) continue;
                for (int r = 0; r < p; r++)
                    AssertClose(math.abs(trunc.components[r, c]), math.abs(full.components[r, c]), ctol);
            }

            // top-k ratios sum to strictly less than 1 (the whole point of the ratio).
            float sumRatio = (float)0;
            for (int i = 0; i < k; i++) sumRatio += trunc.explainedVarianceRatio[i];
            AssertTrue(sumRatio < (float)0.99f);
            AssertTrue(sumRatio > (float)0);

            arena.Dispose();
        }

        // =====================================================================
        // #5b — randomized top-k. pcaRandomized(k) recovers the full top-k
        //       eigenvalues within a LOOSE (approximate) tol; ratios sum < 1.
        // =====================================================================
        void TopKRandomizedApprox()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 50, p = 6, k = 3;
            var X = BuildCorrelated(ref arena, n, p);

            var full = floatPCA_OP.pcaSVD(ref arena, in X);
            var rnd = floatPCA_OP.pcaRandomized(ref arena, in X, k);
            AssertTrue(full.converged);
            AssertTrue(rnd.converged);
            RecordEq(rnd.k, k);

            // approximate: recovered eigenvalues within 5% relative of the exact top-k.
            for (int i = 0; i < k; i++)
            {
                float exact = full.explainedVariance[i];
                float rel = math.abs(rnd.explainedVariance[i] - exact) / (math.abs(exact) + (float)1E-6f);
                AssertTrue(rel <= (float)0.05f);
            }

            // leading component recovered (loose, gap-guarded magnitude).
            if (WellSeparated(in full.explainedVariance, 0, p))
                for (int r = 0; r < p; r++)
                    AssertClose(math.abs(rnd.components[r, 0]), math.abs(full.components[r, 0]), (float)0.1f);

            float sumRatio = (float)0;
            for (int i = 0; i < k; i++) sumRatio += rnd.explainedVarianceRatio[i];
            AssertTrue(sumRatio < (float)0.99f);
            AssertTrue(sumRatio > (float)0);

            arena.Dispose();
        }

        // =====================================================================
        // #6 — pcaTransform. Project the training data (Covariance mode): each score
        //      column has sample variance == explainedVariance; a hand-computed
        //      ((X-mean)/scale)·components matches the returned scores; shape is n×k.
        // =====================================================================
        void TransformScores()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 50, p = 6;
            var X = BuildCorrelated(ref arena, n, p);

            var m = floatPCA_OP.pcaCovariance(ref arena, in X);   // Covariance, scale == 1
            AssertTrue(m.converged);

            var scores = floatPCA_OP.pcaTransform(ref arena, in X, in m);

            RecordEq(scores.M_Rows, n);
            RecordEq(scores.N_Cols, m.k);

            // score-column sample variance == explainedVariance (defining property of PCA scores).
            for (int c = 0; c < m.k; c++)
            {
                float mean = (float)0;
                for (int r = 0; r < n; r++) mean += scores[r, c];
                mean /= (float)n;

                float ss = (float)0;
                for (int r = 0; r < n; r++) { float d = scores[r, c] - mean; ss += d * d; }
                float variance = ss / (float)(n - 1);

                AssertClose(variance, m.explainedVariance[c],
                            (math.abs(m.explainedVariance[c]) + (float)1) * (float)200 * Consts.floatSqrtEps);
            }

            // hand-computed ((X-mean)/scale)·components matches the returned scores.
            float mtol = (float)200 * Consts.floatSqrtEps;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < m.k; c++)
                {
                    float acc = (float)0;
                    for (int f = 0; f < p; f++)
                        acc += ((X[r, f] - m.mean[f]) / m.scale[f]) * m.components[f, c];
                    AssertClose(scores[r, c], acc, (math.abs(acc) + (float)1) * mtol);
                }

            arena.Dispose();
        }

        // =====================================================================
        // #7a — sign determinism. On axis-aligned, well-separated data (components ==
        //       identity), negating an input column and refitting yields IDENTICAL
        //       components (the sign rule pins each column deterministically).
        // =====================================================================
        void SignDeterminismNegate()
        {
            var arena = new Arena(Allocator.Persistent);

            var X1 = BuildDiagonal(ref arena);   // 4×3, cov == diag(9,4,1), components == I
            int n = X1.M_Rows, p = X1.N_Cols;

            var X2 = arena.floatMat(n, p);
            for (int r = 0; r < n; r++)
                for (int c = 0; c < p; c++)
                    X2[r, c] = (c == 0) ? -X1[r, c] : X1[r, c];

            var m1 = floatPCA_OP.pcaCovariance(ref arena, in X1);
            var m2 = floatPCA_OP.pcaCovariance(ref arena, in X2);
            AssertTrue(m1.converged);
            AssertTrue(m2.converged);

            float ctol = (float)100 * Consts.floatSqrtEps;
            for (int i = 0; i < p; i++)
                AssertClose(m1.explainedVariance[i], m2.explainedVariance[i], EvTol(m1.explainedVariance[i]));
            for (int c = 0; c < p; c++)
                for (int r = 0; r < p; r++)
                    AssertClose(m1.components[r, c], m2.components[r, c], ctol);

            arena.Dispose();
        }

        // =====================================================================
        // #7b — pcaRandomized(arena, X, k) called twice with the default seed
        //       (0x9E3779B1u) is BITWISE-identical (components + variances + ratios).
        // =====================================================================
        void RandomizedBitwise()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 50, p = 6, k = 3;
            var X = BuildCorrelated(ref arena, n, p);

            var a = floatPCA_OP.pcaRandomized(ref arena, in X, k);
            var b = floatPCA_OP.pcaRandomized(ref arena, in X, k);
            AssertTrue(a.converged);
            AssertTrue(b.converged);

            for (int i = 0; i < k; i++)
            {
                AssertExact(a.explainedVariance[i], b.explainedVariance[i]);
                AssertExact(a.explainedVarianceRatio[i], b.explainedVarianceRatio[i]);
            }
            for (int c = 0; c < k; c++)
                for (int r = 0; r < p; r++)
                    AssertExact(a.components[r, c], b.components[r, c]);

            arena.Dispose();
        }

        // =====================================================================
        // #8 — wide data (p > n). pcaCovariance is the only route that handles it:
        //      k == p, converged, ev[0] > 0, and the trailing (p−rank) eigenvalues
        //      ≈ 0 (centered n×p data has rank ≤ n−1).
        // =====================================================================
        void WideCovariance()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4, p = 6;                            // wide: p > n
            var X = arena.floatRandomMat(n, p, (float)(-2), (float)2, 20240703u);

            var m = floatPCA_OP.pcaCovariance(ref arena, in X);
            AssertTrue(m.converged);
            RecordEq(m.k, p);
            RecordEq(m.components.M_Rows, p);
            RecordEq(m.components.N_Cols, p);

            AssertTrue(m.explainedVariance[0] > (float)0);

            // rank of the centered data ≤ n−1 = 3 ⇒ ev[3..5] ≈ 0 (relative to ev[0]).
            float zeroTol = (float)1E-3f * (m.explainedVariance[0] + (float)1);
            for (int i = n - 1; i < p; i++)
                AssertClose(m.explainedVariance[i], (float)0, zeroTol);

            arena.Dispose();
        }

        // =====================================================================
        // datasets
        // =====================================================================

        // n×p correlated data with a moderately-decaying, WELL-SEPARATED spectrum. X = G·diag(w)·Vᵀ
        // with independent latent columns G (uniform[-1,1]), a fixed orthogonal mixing V, and weights
        // w_k = p−k (6,5,4,3,2,1) → covariance ≈ V diag(w²·varG) Vᵀ with clearly-separated eigenvalues
        // (≈ 12, 8.3, 5.3, 3, 1.3, 0.33) and genuinely correlated features (so the correlation-matrix
        // spectrum is non-trivial too). Deterministic (fixed seeds).
        floatMxN BuildCorrelated(ref Arena arena, int n, int p)
        {
            var G = arena.floatRandomMat(n, p, (float)(-1), (float)1, 20240702u);

            var V = arena.floatMat(p, p);
            var rng = new Unity.Mathematics.Random(0x01234567u);
            floatRandomMatrix_OP.randomOrthogonalInpl(ref rng, ref V);

            var X = arena.floatMat(n, p);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < p; j++)
                {
                    float acc = (float)0;
                    for (int kk = 0; kk < p; kk++)
                    {
                        float wk = (float)(p - kk);
                        acc += G[i, kk] * wk * V[j, kk];
                    }
                    X[i, j] = acc;
                }
            return X;
        }

        // 4×3 data whose sample covariance is EXACTLY diag(9,4,1): three mutually-orthogonal, centered
        // ±1 columns (each orthogonal to the all-ones vector), scaled by a_j = sqrt(3·v_j/4) so that
        // (Σ h² )/(n−1) = 4·a_j²/3 = v_j. Cross-covariances are exactly 0 (integer ±1 dot products) →
        // covariance == diag(9,4,1), eigenvectors == the standard basis.
        floatMxN BuildDiagonal(ref Arena arena)
        {
            var X = arena.floatMat(4, 3);
            float a0 = math.sqrt((float)(3.0 * 9.0 / 4.0));   // variance 9
            float a1 = math.sqrt((float)(3.0 * 4.0 / 4.0));   // variance 4
            float a2 = math.sqrt((float)(3.0 * 1.0 / 4.0));   // variance 1

            X[0, 0] =  a0; X[0, 1] =  a1; X[0, 2] =  a2;
            X[1, 0] =  a0; X[1, 1] = -a1; X[1, 2] = -a2;
            X[2, 0] = -a0; X[2, 1] =  a1; X[2, 2] = -a2;
            X[3, 0] = -a0; X[3, 1] = -a1; X[3, 2] =  a2;
            return X;
        }

        // =====================================================================
        // helpers
        // =====================================================================

        // Value-scaled tolerance for eigenvalue / variance comparisons: absolute floor + a relative
        // band that stays tight for double and stays sane for the κ²-squaring covariance route in float.
        float EvTol(float v) => (math.abs(v) + (float)1) * (float)50 * Consts.floatSqrtEps;

        // Is component c's eigenvalue separated from BOTH neighbors by a comfortable relative gap?
        // Vector comparisons only run where this holds (a near-tie leaves the eigenbasis rotation
        // route-/precision-dependent — no sign rule fixes that, per the spec).
        bool WellSeparated(in floatN ev, int c, int count)
        {
            float vc = math.abs(ev[c]);
            float gap = float.MaxValue;
            if (c > 0)         gap = math.min(gap, math.abs(ev[c] - ev[c - 1]));
            if (c + 1 < count) gap = math.min(gap, math.abs(ev[c] - ev[c + 1]));
            return gap > (float)0.05f * (vc + (float)1E-6f);
        }

        // ---- Fail-array diagnostics ----
        void AssertClose(float a, float b, float precision)
        {
            float diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1; Fail[1] = a; Fail[2] = b; Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertExact(float a, float b)
        {
            if (!(a == b) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1; Fail[1] = a; Fail[2] = b; Fail[3] = a - b;
            }
            Assert.IsTrue(a == b);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (float)0)
            {
                Fail[0] = (float)1; Fail[1] = (float)0; Fail[2] = (float)1; Fail[3] = (float)1;
            }
            Assert.IsTrue(ok);
        }

        void RecordEq(int got, int expected)
        {
            if (got != expected && Fail[0] == (float)0)
            {
                Fail[0] = (float)1; Fail[1] = got; Fail[2] = expected; Fail[3] = got - expected;
            }
            Assert.AreEqual(expected, got);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void PCATests(TestJob.TestType type)
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

    // =====================================================================
    // #9 — managed guard throws (main thread; throw paths need no Burst).
    // =====================================================================

    [Test]
    public void NLessThanTwoThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var X = arena.floatMat(1, 3);   // n == 1 < 2 (variance undefined)
        Assert.Throws<ArgumentException>(() => floatPCA_OP.pcaCovariance(ref arena, in X));
        arena.Dispose();
    }

    [Test]
    public void SvdWideThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var X = arena.floatMat(3, 5);   // wide: p > n
        Assert.Throws<ArgumentException>(() => floatPCA_OP.pcaSVD(ref arena, in X));
        arena.Dispose();
    }

    [Test]
    public void SvdTruncatedWideThrows()
    {
        // The coder confirmed svdTruncated is NOT shape-free; pcaSVDTruncated adds the n>=p guard, so
        // it throws on wide data just like pcaSVD/pcaRandomized (there is deliberately NO "truncated
        // works on wide data" test).
        var arena = new Arena(Allocator.Persistent);
        var X = arena.floatMat(3, 5);
        Assert.Throws<ArgumentException>(() => floatPCA_OP.pcaSVDTruncated(ref arena, in X, 2));
        arena.Dispose();
    }

    [Test]
    public void RandomizedWideThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var X = arena.floatMat(3, 5);
        Assert.Throws<ArgumentException>(() => floatPCA_OP.pcaRandomized(ref arena, in X, 2));
        arena.Dispose();
    }

    [Test]
    public void TruncatedKZeroThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var X = arena.floatMat(10, 4);
        Assert.Throws<ArgumentException>(() => floatPCA_OP.pcaSVDTruncated(ref arena, in X, 0));   // k <= 0
        arena.Dispose();
    }

    [Test]
    public void TruncatedKTooLargeThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var X = arena.floatMat(10, 4);
        Assert.Throws<ArgumentException>(() => floatPCA_OP.pcaSVDTruncated(ref arena, in X, 5));   // k > min(n,p)=4
        arena.Dispose();
    }

    [Test]
    public void RefModelWrongShapeThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var X = arena.floatMat(10, 4);              // p == 4
        var model = arena.floatPCAModel(3, 3);      // sized for p == 3 (wrong)
        Assert.Throws<ArgumentException>(() => floatPCA_OP.pcaCovariance(in X, ref model));
        arena.Dispose();
    }

    [Test]
    public void TransformStaleKThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var X = arena.floatMat(10, 4);
        var model = floatPCA_OP.pcaCovariance(ref arena, in X);   // k == 4 == components.N_Cols
        model.k = model.components.N_Cols + 1;                     // stale / hand-tampered
        Assert.Throws<ArgumentException>(() => floatPCA_OP.pcaTransform(ref arena, in X, in model));
        arena.Dispose();
    }

    [Test]
    public void TransformMisSizedScoresThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var X = arena.floatMat(10, 4);
        var model = floatPCA_OP.pcaCovariance(ref arena, in X);   // k == 4
        var badScores = arena.floatMat(10, 3);                    // wrong column count (should be k == 4)
        Assert.Throws<ArgumentException>(() => floatPCA_OP.pcaTransform(in X, in model, ref badScores));
        arena.Dispose();
    }
}
