using System;
#pragma warning disable 618 // intentionally exercises the deprecated Jacobi svdDecomposition / Eigen.decompInPlace (kept for reference)

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

// Tests for Chunk 3 of the random-generation layer: structured / property matrices
// (Rand). Verification is PROPERTY-based — we construct the matrix and then
// independently check the property it is supposed to have, reusing the library's own ops:
//   * orthogonalInPlace   -> QᵀQ ≈ I  (dot(Q,Q,transposeA:true)); determinism.
//   * spdInPlace          -> symmetry, Cholesky succeeds (PD), eigenvalues ∈ [minEig,maxEig]
//                               (Eigen.decompInPlace), trace ∈ [n·minEig, n·maxEig].
//   * conditionedInPlace -> σ_max/σ_min ≈ cond via SVD.singularValues.
//   * withRankInPlace      -> #{σ_i > S[0]·thr} == rank via SVD.singularValues; rank 0 exact zeros.
//   * stochasticInPlace   -> each row sums to 1, entries in [0,1].
//   * multivariateNormal*    -> empirical mean ≈ mean (cholL = I) and empirical covariance ≈ LLᵀ.
//
// FIXED seeds only (Monte-Carlo statistics must be reproducible). Tolerances are per-precision —
// they scale with Consts.doubleSqrtEps (float ≈ 3.45e-4, double ≈ 1.49e-8), so the SAME expression
// is loose for float and tight for double, mirroring the SVDWorkspace/PivotedCholesky tests. The
// statistical (Monte-Carlo) tolerances are deliberately generous so the tests don't flake.
// Throw-tests run on the managed thread (Assert.Throws), like the sibling guard tests.
public class doubleRandomMatrixTests
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
        public NativeArray<double> Fail;

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
        // orthogonalInPlace
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
            var Q = arena.doubleMat(n, n);
            Rand.orthogonalInPlace(ref rng, ref Q);

            // QᵀQ
            var QtQ = arena.doubleMat(n, n);
            Blas.dot(in Q, in Q, ref QtQ, transposeA: true);

            // off-diagonal orthonormality error scales with n; this bound is loose for float,
            // tight for double, but still far above the true ~n·eps backward error.
            double tol = (double)30 * Consts.doubleSqrtEps;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < n; c++)
                {
                    double expected = (r == c) ? (double)1 : (double)0;
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
            var Q1 = arena.doubleMat(n, n);
            Rand.orthogonalInPlace(ref r1, ref Q1);

            var r2 = new Random(424242u);
            var Q2 = arena.doubleMat(n, n);
            Rand.orthogonalInPlace(ref r2, ref Q2);

            for (int i = 0; i < Q1.Length; i++)
                AssertClose(Q1[i], Q2[i], (double)0);

            arena.Dispose();
        }

        // =====================================================================
        // spdInPlace
        // =====================================================================

        // (a) symmetry, (b) positive-definite via Cholesky, (c) eigenvalues in [minEig,maxEig]
        //     via symmetric-Jacobi Eigen, plus trace ∈ [n·minEig, n·maxEig].
        void SpdProperties()
        {
            var arena = new Arena(Allocator.Persistent);

            double minEig = (double)0.5, maxEig = (double)8;

            for (uint t = 0; t < 6; t++)
            {
                int n = 4 + (int)t;             // 4..9
                var rng = new Random(5000u + t * 37u);
                var A = arena.doubleMat(n, n);
                Rand.spdInPlace(ref rng, ref A, minEig, maxEig);

                // (a) symmetry — implementation symmetrises exactly, so this is tight.
                double symTol = (double)8 * Consts.doubleSqrtEps;
                for (int r = 0; r < n; r++)
                    for (int c = r + 1; c < n; c++)
                        AssertClose(A[r, c], A[c, r], symTol);

                // (b) positive-definite: Cholesky must succeed.
                var L = arena.doubleMat(n, n);
                AssertTrue(CHO.decomp(in A, ref L));

                // (c) eigenvalues ∈ [minEig, maxEig] (Jacobi destroys its input -> copy).
                var Acopy = arena.doubleMat(in A);
                var evals = arena.doubleVec(n);
                var V = arena.doubleMat(n, n);
                AssertTrue(Eigen.decompInPlace(ref Acopy, ref evals, ref V));

                double eigTol = (double)200 * Consts.doubleSqrtEps * maxEig;
                double traceLam = (double)0, traceA = (double)0;
                for (int i = 0; i < n; i++)
                {
                    AssertTrue(evals[i] >= minEig - eigTol);
                    AssertTrue(evals[i] <= maxEig + eigTol);
                    traceLam += evals[i];
                    traceA += A[i, i];
                }

                // trace = Σλ ∈ [n·minEig, n·maxEig]; and trace(A) == Σλ.
                AssertTrue(traceA >= (double)n * minEig - eigTol);
                AssertTrue(traceA <= (double)n * maxEig + eigTol);
                AssertClose(traceA, traceLam, (double)10 * Consts.doubleSqrtEps * maxEig * (double)n);

                arena.Clear();
            }

            arena.Dispose();
        }

        void SpdDeterminism()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 5;

            var r1 = new Random(96321u);
            var A1 = arena.doubleMat(n, n);
            Rand.spdInPlace(ref r1, ref A1, (double)1, (double)10);

            var r2 = new Random(96321u);
            var A2 = arena.doubleMat(n, n);
            Rand.spdInPlace(ref r2, ref A2, (double)1, (double)10);

            for (int i = 0; i < A1.Length; i++)
                AssertClose(A1[i], A2[i], (double)0);

            arena.Dispose();
        }

        // =====================================================================
        // conditionedInPlace
        // =====================================================================

        void ConditionNumberSquare() => CheckCondition(5, 5, (double)50, 6001u);
        void ConditionNumberRect()   => CheckCondition(5, 3, (double)50, 6002u);

        // σ_max/σ_min ≈ cond, verified through SVD.singularValues (descending order).
        void CheckCondition(int m, int n, double cond, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);

            var rng = new Random(seed);
            var A = arena.doubleMat(m, n);
            Rand.conditionedInPlace(ref rng, ref A, cond);

            int k = math.min(m, n);
            var S = arena.doubleVec(k);
            SVD.singularValues(in A, ref S);

            double got = S[0] / S[k - 1];

            // Generous relative tolerance (float SVD on a reconstructed UΣVᵀ); auto-tightens for double.
            double relTol = (double)60 * Consts.doubleSqrtEps;   // float ≈ 2.1e-2, double ≈ 8.9e-7
            AssertClose(got / cond, (double)1, relTol);

            arena.Dispose();
        }

        // k = 1 (1x4) is the documented trivial case: a single singular value => cond = 1.
        void ConditionRankOne()
        {
            var arena = new Arena(Allocator.Persistent);

            var rng = new Random(6003u);
            var A = arena.doubleMat(1, 4);
            Rand.conditionedInPlace(ref rng, ref A, (double)50);

            var S = arena.doubleVec(1);    // k = min(1,4) = 1
            SVD.singularValues(in A, ref S);

            // single sv => σ_max/σ_min = S[0]/S[0] = 1 exactly; assert the sv equals the σ₀ the
            // algorithm sets for k==1, which is 1.
            AssertClose(S[0], (double)1, (double)50 * Consts.doubleSqrtEps);

            arena.Dispose();
        }

        // =====================================================================
        // withRankInPlace
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
                    var A = arena.doubleMat(m, n);
                    Rand.withRankInPlace(ref rng, ref A, rank);

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
            var A0 = arena.doubleMat(m, n);
            for (int i = 0; i < A0.Length; i++) A0[i] = (double)999;   // poison
            Rand.withRankInPlace(ref rng0, ref A0, 0);
            for (int i = 0; i < A0.Length; i++)
                AssertClose(A0[i], (double)0, (double)0);    // exact
            RecordEq(NumericalRank(in arena, in A0), 0);

            // full rank.
            var rngF = new Random(8002u);
            var AF = arena.doubleMat(m, n);
            Rand.withRankInPlace(ref rngF, ref AF, k);
            RecordEq(NumericalRank(in arena, in AF), k);

            arena.Dispose();
        }

        // count singular values above S[0]·thr. thr sits just above the machine-zero floor:
        // it scales with eps (float ≈ 7.6e-6, double ≈ 1.4e-14), comfortably ABOVE the ~eps·S[0]
        // numerical-zero singular values of a rank-deficient product yet BELOW the genuine (≳1e-3·S[0])
        // singular values — including the smallest one of a full-rank random Gaussian product, which a
        // larger sqrt(eps)-scaled threshold would wrongly reject in float.
        int NumericalRank(in Arena arena, in doubleMxN A)
        {
            int k = math.min(A.M_Rows, A.N_Cols);
            var S = arena.doubleVec(k);
            SVD.singularValues(in A, ref S);

            double thr = S[0] * (double)64 * Consts.doubleEpsilon;
            int count = 0;
            for (int i = 0; i < k; i++)
                if (S[i] > thr) count++;
            return count;
        }

        // =====================================================================
        // stochasticInPlace
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
            var A = arena.doubleMat(m, n);
            Rand.stochasticInPlace(ref rng, ref A);

            double sumTol = (double)20 * Consts.doubleSqrtEps;
            for (int r = 0; r < m; r++)
            {
                double rowSum = (double)0;
                for (int c = 0; c < n; c++)
                {
                    double v = A[r, c];
                    AssertTrue(v >= (double)0);
                    AssertTrue(v <= (double)1);
                    rowSum += v;
                }
                AssertClose(rowSum, (double)1, sumTol);
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

            var I = arena.doubleMat(n, n);
            for (int i = 0; i < n; i++) I[i, i] = (double)1;

            var mean = arena.doubleVec(n);
            for (int i = 0; i < n; i++) mean[i] = (double)(i + 1);   // 1,2,3,...

            // --- 5-param overload (explicit z scratch) ---
            var rng = new Random(11000u + (uint)n);
            var acc = arena.doubleVec(n);
            for (int i = 0; i < n; i++) acc[i] = (double)0;
            var dest = arena.doubleVec(n);
            var z = arena.doubleVec(n);
            for (int s = 0; s < samples; s++)
            {
                Rand.multivariateNormalInPlace(ref rng, in I, in mean, ref dest, ref z);
                for (int i = 0; i < n; i++) acc[i] += dest[i];
            }
            double meanTol = (double)0.1;   // std error of mean over 8000 draws ≈ 0.011
            for (int i = 0; i < n; i++)
                AssertClose(acc[i] / (double)samples, mean[i], meanTol);

            // --- 4-param overload (Temp scratch) ---
            var rng2 = new Random(12000u + (uint)n);
            for (int i = 0; i < n; i++) acc[i] = (double)0;
            for (int s = 0; s < samples; s++)
            {
                Rand.multivariateNormalInPlace(ref rng2, in I, in mean, ref dest);
                for (int i = 0; i < n; i++) acc[i] += dest[i];
            }
            for (int i = 0; i < n; i++)
                AssertClose(acc[i] / (double)samples, mean[i], meanTol);

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
            var L = arena.doubleMat(n, n);
            L[0, 0] = (double)1;   L[0, 1] = (double)0;
            L[1, 0] = (double)0.5; L[1, 1] = (double)0.8;

            double s00 = (double)1;
            double s01 = (double)0.5;
            double s11 = (double)0.89;

            var mean = arena.doubleVec(n);
            mean[0] = (double)(-2); mean[1] = (double)3;

            var rng = new Random(13000u);
            var dest = arena.doubleMat(rows, n);
            Rand.multivariateNormalRowsInPlace(ref rng, in L, in mean, ref dest);

            // empirical column means
            double m0 = (double)0, m1 = (double)0;
            for (int r = 0; r < rows; r++) { m0 += dest[r, 0]; m1 += dest[r, 1]; }
            m0 /= (double)rows; m1 /= (double)rows;

            AssertClose(m0, mean[0], (double)0.1);
            AssertClose(m1, mean[1], (double)0.1);

            // empirical covariance (about the empirical mean)
            double c00 = (double)0, c01 = (double)0, c11 = (double)0;
            for (int r = 0; r < rows; r++)
            {
                double d0 = dest[r, 0] - m0;
                double d1 = dest[r, 1] - m1;
                c00 += d0 * d0; c01 += d0 * d1; c11 += d1 * d1;
            }
            c00 /= (double)rows; c01 /= (double)rows; c11 /= (double)rows;

            double covTol = (double)0.12;
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

            var L = arena.doubleMat(n, n);
            L[0, 0] = (double)1.2; L[1, 0] = (double)(-0.3); L[1, 1] = (double)0.7;
            L[2, 0] = (double)0.1; L[2, 1] = (double)0.4;    L[2, 2] = (double)0.9;

            var mean = arena.doubleVec(n);
            mean[0] = (double)1; mean[1] = (double)2; mean[2] = (double)3;

            var r1 = new Random(55667788u);
            var D1 = arena.doubleMat(rows, n);
            Rand.multivariateNormalRowsInPlace(ref r1, in L, in mean, ref D1);

            var r2 = new Random(55667788u);
            var D2 = arena.doubleMat(rows, n);
            Rand.multivariateNormalRowsInPlace(ref r2, in L, in mean, ref D2);

            for (int i = 0; i < D1.Length; i++)
                AssertClose(D1[i], D2[i], (double)0);

            arena.Dispose();
        }

        // =====================================================================
        // helpers (Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff)
        // =====================================================================

        void AssertClose(double a, double b, double precision)
        {
            double diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = (double)(-1);
                Fail[2] = (double)(-1);
                Fail[3] = (double)(-1);
            }
            Assert.IsTrue(ok);
        }

        void RecordEq(int got, int expected)
        {
            if (got != expected && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
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
        var fail = new NativeArray<double>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (double)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (double)0)
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
            var dest = arena.doubleMat(3, 4);
            Assert.Throws<ArgumentException>(
                () => Rand.orthogonalInPlace(ref rng, ref dest));
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

            var nonSquare = arena.doubleMat(3, 4);
            Assert.Throws<ArgumentException>(
                () => Rand.spdInPlace(ref rng, ref nonSquare, (double)1, (double)2));

            var A = arena.doubleMat(3, 3);
            // minEig <= 0
            Assert.Throws<ArgumentException>(
                () => Rand.spdInPlace(ref rng, ref A, (double)0, (double)2));
            Assert.Throws<ArgumentException>(
                () => Rand.spdInPlace(ref rng, ref A, (double)(-1), (double)2));
            // minEig > maxEig
            Assert.Throws<ArgumentException>(
                () => Rand.spdInPlace(ref rng, ref A, (double)5, (double)2));
            // non-finite bounds
            Assert.Throws<ArgumentException>(
                () => Rand.spdInPlace(ref rng, ref A, (double)double.PositiveInfinity, (double)2));
            Assert.Throws<ArgumentException>(
                () => Rand.spdInPlace(ref rng, ref A, (double)1, (double)float.NaN));
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
            var A = arena.doubleMat(4, 4);

            // cond < 1
            Assert.Throws<ArgumentException>(
                () => Rand.conditionedInPlace(ref rng, ref A, (double)0.5));
            // non-finite cond
            Assert.Throws<ArgumentException>(
                () => Rand.conditionedInPlace(ref rng, ref A, (double)double.PositiveInfinity));
            Assert.Throws<ArgumentException>(
                () => Rand.conditionedInPlace(ref rng, ref A, (double)float.NaN));
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
            var A = arena.doubleMat(5, 3);   // min(m,n) = 3

            Assert.Throws<ArgumentException>(
                () => Rand.withRankInPlace(ref rng, ref A, -1));
            Assert.Throws<ArgumentException>(
                () => Rand.withRankInPlace(ref rng, ref A, 4)); // > min(m,n)
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
            var nonSquare = arena.doubleMat(3, 2);
            var mean3 = arena.doubleVec(3);
            var dest3 = arena.doubleVec(3);
            var z3 = arena.doubleVec(3);
            Assert.Throws<ArgumentException>(
                () => Rand.multivariateNormalInPlace(ref rng, in nonSquare, in mean3, ref dest3, ref z3));
            Assert.Throws<ArgumentException>(
                () => Rand.multivariateNormalInPlace(ref rng, in nonSquare, in mean3, ref dest3));

            var L = arena.doubleMat(3, 3);

            // mean.N mismatch
            var meanBad = arena.doubleVec(2);
            Assert.Throws<ArgumentException>(
                () => Rand.multivariateNormalInPlace(ref rng, in L, in meanBad, ref dest3, ref z3));

            // dest.N mismatch
            var destBad = arena.doubleVec(4);
            Assert.Throws<ArgumentException>(
                () => Rand.multivariateNormalInPlace(ref rng, in L, in mean3, ref destBad, ref z3));

            // zScratch.N mismatch
            var zBad = arena.doubleVec(5);
            Assert.Throws<ArgumentException>(
                () => Rand.multivariateNormalInPlace(ref rng, in L, in mean3, ref dest3, ref zBad));

            // rows overload: cholL not square
            var destRows = arena.doubleMat(8, 3);
            Assert.Throws<ArgumentException>(
                () => Rand.multivariateNormalRowsInPlace(ref rng, in nonSquare, in mean3, ref destRows));
            // rows overload: mean.N mismatch
            Assert.Throws<ArgumentException>(
                () => Rand.multivariateNormalRowsInPlace(ref rng, in L, in meanBad, ref destRows));
            // rows overload: destRows.N_Cols mismatch
            var destRowsBad = arena.doubleMat(8, 4);
            Assert.Throws<ArgumentException>(
                () => Rand.multivariateNormalRowsInPlace(ref rng, in L, in mean3, ref destRowsBad));
        }
        finally { arena.Dispose(); }
    }
}
