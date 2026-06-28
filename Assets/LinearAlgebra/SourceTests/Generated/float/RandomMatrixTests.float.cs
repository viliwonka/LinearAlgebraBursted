using System;
#pragma warning disable 618 // intentionally exercises the deprecated Jacobi svdDecomposition / eigenDecomposition (kept for reference)

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

// Tests for Chunk 3 of the random-generation layer: structured / property matrices
// (floatRandomMatrixOP). Verification is PROPERTY-based — we construct the matrix and then
// independently check the property it is supposed to have, reusing the library's own ops:
//   * randomOrthogonalInpl   -> QᵀQ ≈ I  (dot(Q,Q,transposeA:true)); determinism.
//   * randomSpdInpl          -> symmetry, Cholesky succeeds (PD), eigenvalues ∈ [minEig,maxEig]
//                               (Eigen.eigenDecomposition), trace ∈ [n·minEig, n·maxEig].
//   * randomMatrixWithConditionInpl -> σ_max/σ_min ≈ cond via SVD.singularValues.
//   * randomMatrixWithRankInpl      -> #{σ_i > S[0]·thr} == rank via SVD.singularValues; rank 0 exact zeros.
//   * randomStochasticInpl   -> each row sums to 1, entries in [0,1].
//   * multivariateNormal*    -> empirical mean ≈ mean (cholL = I) and empirical covariance ≈ LLᵀ.
//
// FIXED seeds only (Monte-Carlo statistics must be reproducible). Tolerances are per-precision —
// they scale with Consts.floatSqrtEps (float ≈ 3.45e-4, double ≈ 1.49e-8), so the SAME expression
// is loose for float and tight for double, mirroring the SVDWorkspace/PivotedCholesky tests. The
// statistical (Monte-Carlo) tolerances are deliberately generous so the tests don't flake.
// Throw-tests run on the managed thread (Assert.Throws), like the sibling guard tests.
public class floatRandomMatrixTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            OrthogonalQtQIdentity,
            OrthogonalDeterminism,
            SpdProperties,
            SpdDeterminism,
            ConditionNumberSquare,
            ConditionNumberRect,
            ConditionRankOne,
            RankExact,
            RankZeroAndFull,
            Stochastic,
            MvnIdentityMeanFive,
            MvnIdentityMeanFour,
            MvnCovariance,
            MvnDeterminism,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<float> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.OrthogonalQtQIdentity: OrthogonalQtQIdentity(); break;
                case TestType.OrthogonalDeterminism: OrthogonalDeterminism(); break;
                case TestType.SpdProperties:         SpdProperties();         break;
                case TestType.SpdDeterminism:        SpdDeterminism();        break;
                case TestType.ConditionNumberSquare: ConditionNumberSquare(); break;
                case TestType.ConditionNumberRect:   ConditionNumberRect();   break;
                case TestType.ConditionRankOne:      ConditionRankOne();      break;
                case TestType.RankExact:             RankExact();             break;
                case TestType.RankZeroAndFull:       RankZeroAndFull();       break;
                case TestType.Stochastic:            Stochastic();            break;
                case TestType.MvnIdentityMeanFive:   MvnIdentityMean(5);      break;
                case TestType.MvnIdentityMeanFour:   MvnIdentityMean(4);      break;
                case TestType.MvnCovariance:         MvnCovariance();         break;
                case TestType.MvnDeterminism:        MvnDeterminism();        break;
            }
        }

        // =====================================================================
        // randomOrthogonalInpl
        // =====================================================================

        // QᵀQ ≈ I for a few sizes. Householder-QR + Haar sign fix => Q orthogonal.
        void OrthogonalQtQIdentity()
        {
            CheckOrthogonal(2, 1001u);
            CheckOrthogonal(4, 2002u);
            CheckOrthogonal(1, 3003u);   // 1x1 degenerate orthogonal: Q = [±1]
            CheckOrthogonal(7, 4004u);
        }

        void CheckOrthogonal(int n, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);

            var rng = new Random(seed);
            var Q = arena.floatMat(n, n);
            floatRandomMatrixOP.randomOrthogonalInpl(ref rng, ref Q);

            // QᵀQ
            var QtQ = arena.floatMat(n, n);
            floatOP.dot(in Q, in Q, ref QtQ, transposeA: true);

            // off-diagonal orthonormality error scales with n; this bound is loose for float,
            // tight for double, but still far above the true ~n·eps backward error.
            float tol = (float)30 * Consts.floatSqrtEps;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                {
                    float expected = (r == c) ? (float)1 : (float)0;
                    AssertClose(QtQ[r, c], expected, tol);
                }

            arena.Dispose();
        }

        // Same seed => identical orthogonal matrix, bit-for-bit.
        void OrthogonalDeterminism()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 5;

            var r1 = new Random(424242u);
            var Q1 = arena.floatMat(n, n);
            floatRandomMatrixOP.randomOrthogonalInpl(ref r1, ref Q1);

            var r2 = new Random(424242u);
            var Q2 = arena.floatMat(n, n);
            floatRandomMatrixOP.randomOrthogonalInpl(ref r2, ref Q2);

            for (int i = 0; i < Q1.Length; i++)
                AssertClose(Q1[i], Q2[i], (float)0);

            arena.Dispose();
        }

        // =====================================================================
        // randomSpdInpl
        // =====================================================================

        // (a) symmetry, (b) positive-definite via Cholesky, (c) eigenvalues in [minEig,maxEig]
        //     via symmetric-Jacobi Eigen, plus trace ∈ [n·minEig, n·maxEig].
        void SpdProperties()
        {
            var arena = new Arena(Allocator.Persistent);

            float minEig = (float)0.5, maxEig = (float)8;

            for (uint t = 0; t < 6; t++)
            {
                int n = 4 + (int)t;             // 4..9
                var rng = new Random(5000u + t * 37u);
                var A = arena.floatMat(n, n);
                floatRandomMatrixOP.randomSpdInpl(ref rng, ref A, minEig, maxEig);

                // (a) symmetry — implementation symmetrises exactly, so this is tight.
                float symTol = (float)8 * Consts.floatSqrtEps;
                for (int r = 0; r < n; r++)
                    for (int c = r + 1; c < n; c++)
                        AssertClose(A[r, c], A[c, r], symTol);

                // (b) positive-definite: Cholesky must succeed.
                var L = arena.floatMat(n, n);
                AssertTrue(Cholesky.choleskyDecomposition(in A, ref L));

                // (c) eigenvalues ∈ [minEig, maxEig] (Jacobi destroys its input -> copy).
                var Acopy = arena.floatMat(in A);
                var evals = arena.floatVec(n);
                var V = arena.floatMat(n, n);
                AssertTrue(Eigen.eigenDecomposition(ref Acopy, ref evals, ref V));

                float eigTol = (float)200 * Consts.floatSqrtEps * maxEig;
                float traceLam = (float)0, traceA = (float)0;
                for (int i = 0; i < n; i++)
                {
                    AssertTrue(evals[i] >= minEig - eigTol);
                    AssertTrue(evals[i] <= maxEig + eigTol);
                    traceLam += evals[i];
                    traceA += A[i, i];
                }

                // trace = Σλ ∈ [n·minEig, n·maxEig]; and trace(A) == Σλ.
                AssertTrue(traceA >= (float)n * minEig - eigTol);
                AssertTrue(traceA <= (float)n * maxEig + eigTol);
                AssertClose(traceA, traceLam, (float)10 * Consts.floatSqrtEps * maxEig * (float)n);

                arena.Clear();
            }

            arena.Dispose();
        }

        void SpdDeterminism()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 5;

            var r1 = new Random(96321u);
            var A1 = arena.floatMat(n, n);
            floatRandomMatrixOP.randomSpdInpl(ref r1, ref A1, (float)1, (float)10);

            var r2 = new Random(96321u);
            var A2 = arena.floatMat(n, n);
            floatRandomMatrixOP.randomSpdInpl(ref r2, ref A2, (float)1, (float)10);

            for (int i = 0; i < A1.Length; i++)
                AssertClose(A1[i], A2[i], (float)0);

            arena.Dispose();
        }

        // =====================================================================
        // randomMatrixWithConditionInpl
        // =====================================================================

        void ConditionNumberSquare() => CheckCondition(5, 5, (float)50, 6001u);
        void ConditionNumberRect()   => CheckCondition(5, 3, (float)50, 6002u);

        // σ_max/σ_min ≈ cond, verified through SVD.singularValues (descending order).
        void CheckCondition(int m, int n, float cond, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);

            var rng = new Random(seed);
            var A = arena.floatMat(m, n);
            floatRandomMatrixOP.randomMatrixWithConditionInpl(ref rng, ref A, cond);

            int k = math.min(m, n);
            var S = arena.floatVec(k);
            SVD.singularValues(in A, ref S);

            float got = S[0] / S[k - 1];

            // Generous relative tolerance (float SVD on a reconstructed UΣVᵀ); auto-tightens for double.
            float relTol = (float)60 * Consts.floatSqrtEps;   // float ≈ 2.1e-2, double ≈ 8.9e-7
            AssertClose(got / cond, (float)1, relTol);

            arena.Dispose();
        }

        // k = 1 (1x4) is the documented trivial case: a single singular value => cond = 1.
        void ConditionRankOne()
        {
            var arena = new Arena(Allocator.Persistent);

            var rng = new Random(6003u);
            var A = arena.floatMat(1, 4);
            floatRandomMatrixOP.randomMatrixWithConditionInpl(ref rng, ref A, (float)50);

            var S = arena.floatVec(1);    // k = min(1,4) = 1
            SVD.singularValues(in A, ref S);

            // single sv => σ_max/σ_min = S[0]/S[0] = 1 exactly; assert the sv equals the σ₀ the
            // algorithm sets for k==1, which is 1.
            AssertClose(S[0], (float)1, (float)50 * Consts.floatSqrtEps);

            arena.Dispose();
        }

        // =====================================================================
        // randomMatrixWithRankInpl
        // =====================================================================

        // numerical rank == requested rank for every rank in [1, min(m,n)], over several seeds.
        void RankExact()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 6, n = 4;            // min = 4
            int k = math.min(m, n);

            for (int rank = 1; rank <= k; rank++)
            {
                for (uint t = 0; t < 4; t++)
                {
                    var rng = new Random(7000u + (uint)rank * 101u + t * 13u);
                    var A = arena.floatMat(m, n);
                    floatRandomMatrixOP.randomMatrixWithRankInpl(ref rng, ref A, rank);

                    int got = NumericalRank(in arena, in A);
                    RecordEq(got, rank);

                    arena.Clear();
                }
            }

            arena.Dispose();
        }

        // rank 0 => exact zero matrix; rank == min(m,n) => full numerical rank.
        void RankZeroAndFull()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 5, n = 3;
            int k = math.min(m, n);

            // rank 0: every entry exactly 0.
            var rng0 = new Random(8001u);
            var A0 = arena.floatMat(m, n);
            for (int i = 0; i < A0.Length; i++) A0[i] = (float)999;   // poison
            floatRandomMatrixOP.randomMatrixWithRankInpl(ref rng0, ref A0, 0);
            for (int i = 0; i < A0.Length; i++)
                AssertClose(A0[i], (float)0, (float)0);    // exact
            RecordEq(NumericalRank(in arena, in A0), 0);

            // full rank.
            var rngF = new Random(8002u);
            var AF = arena.floatMat(m, n);
            floatRandomMatrixOP.randomMatrixWithRankInpl(ref rngF, ref AF, k);
            RecordEq(NumericalRank(in arena, in AF), k);

            arena.Dispose();
        }

        // count singular values above S[0]·thr. thr sits just above the machine-zero floor:
        // it scales with eps (float ≈ 7.6e-6, double ≈ 1.4e-14), comfortably ABOVE the ~eps·S[0]
        // numerical-zero singular values of a rank-deficient product yet BELOW the genuine (≳1e-3·S[0])
        // singular values — including the smallest one of a full-rank random Gaussian product, which a
        // larger sqrt(eps)-scaled threshold would wrongly reject in float.
        int NumericalRank(in Arena arena, in floatMxN A)
        {
            int k = math.min(A.M_Rows, A.N_Cols);
            var S = arena.floatVec(k);
            SVD.singularValues(in A, ref S);

            float thr = S[0] * (float)64 * Consts.floatEpsilon;
            int count = 0;
            for (int i = 0; i < k; i++)
                if (S[i] > thr) count++;
            return count;
        }

        // =====================================================================
        // randomStochasticInpl
        // =====================================================================

        void Stochastic()
        {
            CheckStochastic(3, 4, 9001u);
            CheckStochastic(5, 3, 9002u);
            CheckStochastic(1, 6, 9003u);   // single row
            CheckStochastic(8, 2, 9004u);
        }

        void CheckStochastic(int m, int n, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);

            var rng = new Random(seed);
            var A = arena.floatMat(m, n);
            floatRandomMatrixOP.randomStochasticInpl(ref rng, ref A);

            float sumTol = (float)20 * Consts.floatSqrtEps;
            for (int r = 0; r < m; r++)
            {
                float rowSum = (float)0;
                for (int c = 0; c < n; c++)
                {
                    float v = A[r, c];
                    AssertTrue(v >= (float)0);
                    AssertTrue(v <= (float)1);
                    rowSum += v;
                }
                AssertClose(rowSum, (float)1, sumTol);
            }

            arena.Dispose();
        }

        // =====================================================================
        // multivariateNormal*
        // =====================================================================

        // cholL = I, mean = m: a sample = m + z, z ~ N(0,1). Over many samples, empirical mean ≈ m.
        // Exercises both single-sample overloads (5-param scratch and 4-param Temp).
        void MvnIdentityMean(int n)
        {
            var arena = new Arena(Allocator.Persistent);
            const int samples = 8000;

            var I = arena.floatMat(n, n);
            for (int i = 0; i < n; i++) I[i, i] = (float)1;

            var mean = arena.floatVec(n);
            for (int i = 0; i < n; i++) mean[i] = (float)(i + 1);   // 1,2,3,...

            // --- 5-param overload (explicit z scratch) ---
            var rng = new Random(11000u + (uint)n);
            var acc = arena.floatVec(n);
            for (int i = 0; i < n; i++) acc[i] = (float)0;
            var dest = arena.floatVec(n);
            var z = arena.floatVec(n);
            for (int s = 0; s < samples; s++)
            {
                floatRandomMatrixOP.multivariateNormalInpl(ref rng, in I, in mean, ref dest, ref z);
                for (int i = 0; i < n; i++) acc[i] += dest[i];
            }
            float meanTol = (float)0.1;   // std error of mean over 8000 draws ≈ 0.011
            for (int i = 0; i < n; i++)
                AssertClose(acc[i] / (float)samples, mean[i], meanTol);

            // --- 4-param overload (Temp scratch) ---
            var rng2 = new Random(12000u + (uint)n);
            for (int i = 0; i < n; i++) acc[i] = (float)0;
            for (int s = 0; s < samples; s++)
            {
                floatRandomMatrixOP.multivariateNormalInpl(ref rng2, in I, in mean, ref dest);
                for (int i = 0; i < n; i++) acc[i] += dest[i];
            }
            for (int i = 0; i < n; i++)
                AssertClose(acc[i] / (float)samples, mean[i], meanTol);

            arena.Dispose();
        }

        // Known 2x2 cholL => Σ = L·Lᵀ known. Over many rows, empirical column means ≈ mean and
        // empirical covariance ≈ Σ within a loose Monte-Carlo tolerance.
        void MvnCovariance()
        {
            var arena = new Arena(Allocator.Persistent);
            const int rows = 8000;
            int n = 2;

            // L = [[1,0],[0.5,0.8]]  =>  Σ = L Lᵀ = [[1,0.5],[0.5,0.89]]
            var L = arena.floatMat(n, n);
            L[0, 0] = (float)1;   L[0, 1] = (float)0;
            L[1, 0] = (float)0.5; L[1, 1] = (float)0.8;

            float s00 = (float)1;
            float s01 = (float)0.5;
            float s11 = (float)0.89;

            var mean = arena.floatVec(n);
            mean[0] = (float)(-2); mean[1] = (float)3;

            var rng = new Random(13000u);
            var dest = arena.floatMat(rows, n);
            floatRandomMatrixOP.multivariateNormalRowsInpl(ref rng, in L, in mean, ref dest);

            // empirical column means
            float m0 = (float)0, m1 = (float)0;
            for (int r = 0; r < rows; r++) { m0 += dest[r, 0]; m1 += dest[r, 1]; }
            m0 /= (float)rows; m1 /= (float)rows;

            AssertClose(m0, mean[0], (float)0.1);
            AssertClose(m1, mean[1], (float)0.1);

            // empirical covariance (about the empirical mean)
            float c00 = (float)0, c01 = (float)0, c11 = (float)0;
            for (int r = 0; r < rows; r++)
            {
                float d0 = dest[r, 0] - m0;
                float d1 = dest[r, 1] - m1;
                c00 += d0 * d0; c01 += d0 * d1; c11 += d1 * d1;
            }
            c00 /= (float)rows; c01 /= (float)rows; c11 /= (float)rows;

            float covTol = (float)0.12;
            AssertClose(c00, s00, covTol);
            AssertClose(c01, s01, covTol);
            AssertClose(c11, s11, covTol);

            arena.Dispose();
        }

        // Fixed seed => identical rows, bit-for-bit (single deterministic float stream).
        void MvnDeterminism()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 3, rows = 64;

            var L = arena.floatMat(n, n);
            L[0, 0] = (float)1.2; L[1, 0] = (float)(-0.3); L[1, 1] = (float)0.7;
            L[2, 0] = (float)0.1; L[2, 1] = (float)0.4;    L[2, 2] = (float)0.9;

            var mean = arena.floatVec(n);
            mean[0] = (float)1; mean[1] = (float)2; mean[2] = (float)3;

            var r1 = new Random(55667788u);
            var D1 = arena.floatMat(rows, n);
            floatRandomMatrixOP.multivariateNormalRowsInpl(ref r1, in L, in mean, ref D1);

            var r2 = new Random(55667788u);
            var D2 = arena.floatMat(rows, n);
            floatRandomMatrixOP.multivariateNormalRowsInpl(ref r2, in L, in mean, ref D2);

            for (int i = 0; i < D1.Length; i++)
                AssertClose(D1[i], D2[i], (float)0);

            arena.Dispose();
        }

        // =====================================================================
        // helpers (Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff)
        // =====================================================================

        void AssertClose(float a, float b, float precision)
        {
            float diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = (float)(-1);
                Fail[2] = (float)(-1);
                Fail[3] = (float)(-1);
            }
            Assert.IsTrue(ok);
        }

        void RecordEq(int got, int expected)
        {
            if (got != expected && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = got;
                Fail[2] = expected;
                Fail[3] = got - expected;
            }
            Assert.AreEqual(expected, got);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void RandomMatrixTests(TestJob.TestType type)
    {
        var fail = new NativeArray<float>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (float)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (float)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    // ---------------- Managed validation throws (main thread, not in a Burst job) ----------------

    [Test]
    public void OrthogonalNonSquareThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var rng = new Random(1u);
            var dest = arena.floatMat(3, 4);
            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.randomOrthogonalInpl(ref rng, ref dest));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void SpdValidationThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var rng = new Random(1u);

            var nonSquare = arena.floatMat(3, 4);
            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.randomSpdInpl(ref rng, ref nonSquare, (float)1, (float)2));

            var A = arena.floatMat(3, 3);
            // minEig <= 0
            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.randomSpdInpl(ref rng, ref A, (float)0, (float)2));
            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.randomSpdInpl(ref rng, ref A, (float)(-1), (float)2));
            // minEig > maxEig
            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.randomSpdInpl(ref rng, ref A, (float)5, (float)2));
            // non-finite bounds
            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.randomSpdInpl(ref rng, ref A, (float)float.PositiveInfinity, (float)2));
            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.randomSpdInpl(ref rng, ref A, (float)1, (float)float.NaN));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void ConditionValidationThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var rng = new Random(1u);
            var A = arena.floatMat(4, 4);

            // cond < 1
            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.randomMatrixWithConditionInpl(ref rng, ref A, (float)0.5));
            // non-finite cond
            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.randomMatrixWithConditionInpl(ref rng, ref A, (float)float.PositiveInfinity));
            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.randomMatrixWithConditionInpl(ref rng, ref A, (float)float.NaN));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void RankValidationThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var rng = new Random(1u);
            var A = arena.floatMat(5, 3);   // min(m,n) = 3

            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.randomMatrixWithRankInpl(ref rng, ref A, -1));
            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.randomMatrixWithRankInpl(ref rng, ref A, 4)); // > min(m,n)
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void MultivariateNormalDimensionMismatchThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var rng = new Random(1u);

            // cholL not square
            var nonSquare = arena.floatMat(3, 2);
            var mean3 = arena.floatVec(3);
            var dest3 = arena.floatVec(3);
            var z3 = arena.floatVec(3);
            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.multivariateNormalInpl(ref rng, in nonSquare, in mean3, ref dest3, ref z3));
            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.multivariateNormalInpl(ref rng, in nonSquare, in mean3, ref dest3));

            var L = arena.floatMat(3, 3);

            // mean.N mismatch
            var meanBad = arena.floatVec(2);
            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.multivariateNormalInpl(ref rng, in L, in meanBad, ref dest3, ref z3));

            // dest.N mismatch
            var destBad = arena.floatVec(4);
            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.multivariateNormalInpl(ref rng, in L, in mean3, ref destBad, ref z3));

            // zScratch.N mismatch
            var zBad = arena.floatVec(5);
            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.multivariateNormalInpl(ref rng, in L, in mean3, ref dest3, ref zBad));

            // rows overload: cholL not square
            var destRows = arena.floatMat(8, 3);
            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.multivariateNormalRowsInpl(ref rng, in nonSquare, in mean3, ref destRows));
            // rows overload: mean.N mismatch
            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.multivariateNormalRowsInpl(ref rng, in L, in meanBad, ref destRows));
            // rows overload: destRows.N_Cols mismatch
            var destRowsBad = arena.floatMat(8, 4);
            Assert.Throws<ArgumentException>(
                () => floatRandomMatrixOP.multivariateNormalRowsInpl(ref rng, in L, in mean3, ref destRowsBad));
        }
        finally { arena.Dispose(); }
    }
}
