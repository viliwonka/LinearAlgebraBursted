using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

// Tests for the random-generation continuous core (Rand + Sampler).
// Two layers:
//   * In-job (Burst) tests: pure ICDF quantiles, empirical moments / support,
//     determinism, stream advance, Gaussian spare bookkeeping, matrix overloads.
//   * Managed throw tests (main thread): constructor / arg validation, mirroring
//     the clampInpl / FFT guard-test convention.
//
// The underlying uniform stream (NextDouble -> (double)NextFloat) is float-valued for
// BOTH expansions, so a fixed seed makes every statistic deterministic; the only
// float/double divergence is in the transcendental ICDF math, absorbed by the loose
// tolerances. Never use a time-based seed.
public class doubleRandomTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            ICDFKnownValues,
            ICDFMonotonic,
            WeibullExponentialIdentity,
            ICDFBoundaryFinite,
            UniformMoments,
            ExponentialMoments,
            GaussianMoments,
            RayleighMoments,
            CauchyMedian,
            ParetoMedian,
            RangeSupport,
            Determinism,
            StreamAdvance,
            GaussianSpareNoAdvance,
            GaussianAdvanceCount,
            MatrixOverloads
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<double> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ICDFKnownValues: ICDFKnownValues(); break;
                case TestType.ICDFMonotonic: ICDFMonotonic(); break;
                case TestType.WeibullExponentialIdentity: WeibullExponentialIdentity(); break;
                case TestType.ICDFBoundaryFinite: ICDFBoundaryFinite(); break;
                case TestType.UniformMoments: UniformMoments(); break;
                case TestType.ExponentialMoments: ExponentialMoments(); break;
                case TestType.GaussianMoments: GaussianMoments(); break;
                case TestType.RayleighMoments: RayleighMoments(); break;
                case TestType.CauchyMedian: CauchyMedian(); break;
                case TestType.ParetoMedian: ParetoMedian(); break;
                case TestType.RangeSupport: RangeSupport(); break;
                case TestType.Determinism: Determinism(); break;
                case TestType.StreamAdvance: StreamAdvance(); break;
                case TestType.GaussianSpareNoAdvance: GaussianSpareNoAdvance(); break;
                case TestType.GaussianAdvanceCount: GaussianAdvanceCount(); break;
                case TestType.MatrixOverloads: MatrixOverloads(); break;
            }
        }

        // ---------------- A. Pure ICDF unit tests ----------------

        // Hand-computed quantiles at u=0, u=0.5, and a mid u for every closed-form sampler.
        void ICDFKnownValues()
        {
            double tol = Consts.doubleSqrtEps;

            // Uniform: ICDF(0)=a, ICDF(0.5)=(a+b)/2, ICDF(u)=a+(b-a)u.
            AssertClose(doubleUniform.UniformICDF((double)0, (double)2, (double)6), (double)2, tol);
            AssertClose(doubleUniform.UniformICDF((double)0.5, (double)2, (double)6), (double)4, tol);
            AssertClose(doubleUniform.UniformICDF((double)0.25, (double)0, (double)4), (double)1, tol);

            // Exponential: ICDF(0,λ)=0; ICDF(u,λ)=-ln(1-u)/λ.
            AssertClose(doubleExponential.ExponentialICDF((double)0, (double)2), (double)0, tol);
            AssertClose(doubleExponential.ExponentialICDF((double)0.5, (double)2),
                        -math.log((double)0.5) / (double)2, tol);
            AssertClose(doubleExponential.ExponentialICDF((double)0.3, (double)1.5),
                        -math.log((double)0.7) / (double)1.5, tol);

            // Rayleigh: ICDF(0,σ)=0; median (u=0.5) = σ·sqrt(ln4).
            AssertClose(doubleRayleigh.RayleighICDF((double)0, (double)2), (double)0, tol);
            AssertClose(doubleRayleigh.RayleighICDF((double)0.5, (double)2),
                        (double)2 * math.sqrt(math.log((double)4)), tol);

            // Pareto: ICDF(0,xm,α)=xm; with α=1, ICDF(0.5)=2·xm.
            AssertClose(doublePareto.ParetoICDF((double)0, (double)2, (double)1.5), (double)2, tol);
            AssertClose(doublePareto.ParetoICDF((double)0.5, (double)2, (double)1), (double)4, tol);

            // Logistic: ICDF(0.5,μ,s)=μ.
            AssertClose(doubleLogistic.LogisticICDF((double)0.5, (double)3, (double)2), (double)3, tol);

            // Cauchy: ICDF(0.5,x0,γ)=x0.
            AssertClose(doubleCauchy.CauchyICDF((double)0.5, (double)5, (double)2), (double)5, tol);

            // Triangular over [low,high]=[-1,5], mode=2 => fc=(2-(-1))/(5-(-1))=0.5.
            // ICDF(0)=low, ICDF(1)=high, ICDF(fc)=mode.
            AssertClose(doubleTriangular.TriangularICDF((double)0, (double)(-1), (double)2, (double)5),
                        (double)(-1), tol);
            AssertClose(doubleTriangular.TriangularICDF((double)1, (double)(-1), (double)2, (double)5),
                        (double)5, tol);
            AssertClose(doubleTriangular.TriangularICDF((double)0.5, (double)(-1), (double)2, (double)5),
                        (double)2, tol);
            // First branch interior value: u=0.15 < fc => a + sqrt(u·(b-a)·(c-a)) = -1 + sqrt(0.15·6·3).
            AssertClose(doubleTriangular.TriangularICDF((double)0.15, (double)(-1), (double)2, (double)5),
                        (double)(-1) + math.sqrt((double)0.15 * (double)6 * (double)3), tol);
        }

        // Every ICDF is non-decreasing in u across a sweep of the open interval.
        void ICDFMonotonic()
        {
            int steps = 200;
            double lo = (double)0.005, hi = (double)0.995;
            double step = (hi - lo) / (double)(steps - 1);

            double pU = doubleUniform.UniformICDF(lo, (double)(-2), (double)4);
            double pE = doubleExponential.ExponentialICDF(lo, (double)2);
            double pR = doubleRayleigh.RayleighICDF(lo, (double)1.5);
            double pW = doubleWeibull.WeibullICDF(lo, (double)1.5, (double)2);
            double pC = doubleCauchy.CauchyICDF(lo, (double)5, (double)2);
            double pL = doubleLogistic.LogisticICDF(lo, (double)3, (double)2);
            double pP = doublePareto.ParetoICDF(lo, (double)2, (double)1.5);
            double pT = doubleTriangular.TriangularICDF(lo, (double)(-1), (double)2, (double)5);

            for (int i = 1; i < steps; i++)
            {
                double u = lo + step * (double)i;
                double cU = doubleUniform.UniformICDF(u, (double)(-2), (double)4);
                double cE = doubleExponential.ExponentialICDF(u, (double)2);
                double cR = doubleRayleigh.RayleighICDF(u, (double)1.5);
                double cW = doubleWeibull.WeibullICDF(u, (double)1.5, (double)2);
                double cC = doubleCauchy.CauchyICDF(u, (double)5, (double)2);
                double cL = doubleLogistic.LogisticICDF(u, (double)3, (double)2);
                double cP = doublePareto.ParetoICDF(u, (double)2, (double)1.5);
                double cT = doubleTriangular.TriangularICDF(u, (double)(-1), (double)2, (double)5);

                NonDecreasing(pU, cU); NonDecreasing(pE, cE); NonDecreasing(pR, cR);
                NonDecreasing(pW, cW); NonDecreasing(pC, cC); NonDecreasing(pL, cL);
                NonDecreasing(pP, cP); NonDecreasing(pT, cT);

                pU = cU; pE = cE; pR = cR; pW = cW; pC = cC; pL = cL; pP = cP; pT = cT;
            }
        }

        // Weibull(k=1, λ) reduces to Exponential with rate 1/λ (scale-vs-rate reciprocal).
        void WeibullExponentialIdentity()
        {
            double lambda = (double)2.5;          // Weibull scale
            double rate = (double)1 / lambda;     // Exponential rate
            double tol = Consts.doubleSqrtEps;

            for (int i = 1; i < 50; i++)
            {
                double u = (double)i / (double)50;   // (0,1)
                double w = doubleWeibull.WeibullICDF(u, (double)1, lambda);
                double e = doubleExponential.ExponentialICDF(u, rate);
                AssertClose(w, e, tol + math.abs(e) * (double)1e-4);
            }
        }

        // Heavy-tailed Cauchy / Logistic ICDF stay finite at the clamped endpoints (u=0, u->1).
        void ICDFBoundaryFinite()
        {
            double c0 = doubleCauchy.CauchyICDF((double)0, (double)5, (double)2);
            double c1 = doubleCauchy.CauchyICDF((double)1, (double)5, (double)2);
            double l0 = doubleLogistic.LogisticICDF((double)0, (double)3, (double)2);
            double l1 = doubleLogistic.LogisticICDF((double)1, (double)3, (double)2);

            AssertTrue(math.isfinite(c0));
            AssertTrue(math.isfinite(c1));
            AssertTrue(math.isfinite(l0));
            AssertTrue(math.isfinite(l1));

            // Symmetric clamp about the median: u=0 below x0, u->1 above x0.
            AssertTrue(c0 < (double)5);
            AssertTrue(c1 > (double)5);
            AssertTrue(l0 < (double)3);
            AssertTrue(l1 > (double)3);
        }

        // ---------------- B. Sampler distribution tests ----------------

        const int StatN = 8192;

        // Uniform[a,b]: mean=(a+b)/2, var=(b-a)^2/12.
        void UniformMoments()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(1234567u);
            var s = new doubleUniform((double)(-2), (double)4);
            var v = arena.doubleVec(StatN);
            Rand.randomInpl(ref rng, ref v, ref s);

            double mean = Mean(in v);
            double var = Variance(in v, mean);
            AssertClose(mean, (double)1, (double)0.1);             // (a+b)/2 = 1
            AssertClose(var, (double)3, (double)0.3);              // (b-a)^2/12 = 3
            arena.Dispose();
        }

        // Exponential(λ): mean=1/λ, var=1/λ^2.
        void ExponentialMoments()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(2468013u);
            double lambda = (double)2;
            var s = new doubleExponential(lambda);
            var v = arena.doubleVec(StatN);
            Rand.randomInpl(ref rng, ref v, ref s);

            double mean = Mean(in v);
            double var = Variance(in v, mean);
            AssertClose(mean, (double)1 / lambda, (double)0.05);          // 0.5
            AssertClose(var, (double)1 / (lambda * lambda), (double)0.08); // 0.25
            arena.Dispose();
        }

        // Gaussian(μ,σ): mean≈μ, var≈σ^2.
        void GaussianMoments()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(13572468u);
            double mu = (double)1.5, sd = (double)2;
            var s = new doubleGaussian(mu, sd);
            var v = arena.doubleVec(StatN);
            Rand.randomInpl(ref rng, ref v, ref s);

            double mean = Mean(in v);
            double var = Variance(in v, mean);
            AssertClose(mean, mu, (double)0.12);
            AssertClose(var, sd * sd, (double)0.4);     // σ^2 = 4
            arena.Dispose();
        }

        // Rayleigh(σ): mean=σ·sqrt(π/2).
        void RayleighMoments()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(97531864u);
            double sigma = (double)1.5;
            var s = new doubleRayleigh(sigma);
            var v = arena.doubleVec(StatN);
            Rand.randomInpl(ref rng, ref v, ref s);

            double mean = Mean(in v);
            double expected = sigma * math.sqrt((double)(System.Math.PI / 2.0));
            AssertClose(mean, expected, (double)0.08);
            arena.Dispose();
        }

        // Cauchy: no finite mean/var. Median = x0 => ~50% of draws below x0.
        void CauchyMedian()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(192837465u);
            double x0 = (double)5, gamma = (double)2;
            var s = new doubleCauchy(x0, gamma);
            var v = arena.doubleVec(StatN);
            Rand.randomInpl(ref rng, ref v, ref s);

            double frac = FractionBelow(in v, x0);
            AssertClose(frac, (double)0.5, (double)0.04);
            arena.Dispose();
        }

        // Pareto α=1.5: variance diverges (α<2). Median = xm·2^(1/α) => ~50% below.
        void ParetoMedian()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(564738291u);
            double xm = (double)2, alpha = (double)1.5;
            var s = new doublePareto(xm, alpha);
            var v = arena.doubleVec(StatN);
            Rand.randomInpl(ref rng, ref v, ref s);

            double median = xm * math.pow((double)2, (double)1 / alpha);
            double frac = FractionBelow(in v, median);
            AssertClose(frac, (double)0.5, (double)0.04);

            // Support: every Pareto draw >= xm.
            for (int i = 0; i < v.N; i++)
                AssertTrue(v[i] >= xm);
            arena.Dispose();
        }

        // Range / support guarantees per distribution.
        void RangeSupport()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(424242u);
            int n = 2048;

            // Uniform[a,b): a <= x < b.
            double a = (double)(-2), b = (double)4;
            var su = new doubleUniform(a, b);
            var vu = arena.doubleVec(n);
            Rand.randomInpl(ref rng, ref vu, ref su);
            for (int i = 0; i < n; i++)
                AssertTrue(vu[i] >= a && vu[i] < b);

            var se = new doubleExponential((double)2);
            var ve = arena.doubleVec(n);
            Rand.randomInpl(ref rng, ref ve, ref se);
            for (int i = 0; i < n; i++)
                AssertTrue(ve[i] >= (double)0);

            var sr = new doubleRayleigh((double)1.5);
            var vr = arena.doubleVec(n);
            Rand.randomInpl(ref rng, ref vr, ref sr);
            for (int i = 0; i < n; i++)
                AssertTrue(vr[i] >= (double)0);

            var sw = new doubleWeibull((double)1.5, (double)2);
            var vw = arena.doubleVec(n);
            Rand.randomInpl(ref rng, ref vw, ref sw);
            for (int i = 0; i < n; i++)
                AssertTrue(vw[i] >= (double)0);

            // Triangular in [low,high].
            double low = (double)(-1), high = (double)5;
            var st = new doubleTriangular(low, (double)2, high);
            var vt = arena.doubleVec(n);
            Rand.randomInpl(ref rng, ref vt, ref st);
            for (int i = 0; i < n; i++)
                AssertTrue(vt[i] >= low && vt[i] <= high);

            arena.Dispose();
        }

        // ---------------- C. Mechanics ----------------

        // Same seed + same sampler => identical fill, element-wise exact.
        void Determinism()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 256;

            var r1 = new Random(55u);
            var s1 = new doubleExponential((double)1.7);
            var v1 = arena.doubleVec(n);
            Rand.randomInpl(ref r1, ref v1, ref s1);

            var r2 = new Random(55u);
            var s2 = new doubleExponential((double)1.7);
            var v2 = arena.doubleVec(n);
            Rand.randomInpl(ref r2, ref v2, ref s2);

            for (int i = 0; i < n; i++)
                AssertClose(v1[i], v2[i], (double)0);   // bit-identical

            arena.Dispose();
        }

        // Two nextUniformInpl calls over the SAME rng advance the stream: buffers differ.
        void StreamAdvance()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 256;
            var rng = new Random(7777u);

            var v1 = arena.doubleVec(n);
            Rand.nextUniformInpl(ref rng, ref v1);
            var v2 = arena.doubleVec(n);
            Rand.nextUniformInpl(ref rng, ref v2);

            bool anyDiff = false;
            for (int i = 0; i < n; i++)
                if (v1[i] != v2[i]) anyDiff = true;
            AssertTrue(anyDiff);

            // Re-seeding resets the stream: a fresh rng reproduces the first buffer.
            var rng3 = new Random(7777u);
            var v3 = arena.doubleVec(n);
            Rand.nextUniformInpl(ref rng3, ref v3);
            for (int i = 0; i < n; i++)
                AssertClose(v1[i], v3[i], (double)0);

            arena.Dispose();
        }

        // Box-Muller spare: the 2nd of a pair is returned without advancing the rng.
        void GaussianSpareNoAdvance()
        {
            var rng = new Random(31415926u);
            var g = new doubleGaussian((double)0, (double)1);

            double a = g.Next(ref rng);
            uint s1 = rng.state;
            double b = g.Next(ref rng);     // spare path: no rng advance
            uint s2 = rng.state;
            double c = g.Next(ref rng);     // new pair: advances
            uint s3 = rng.state;

            AssertTrue(a != b);             // two distinct variates from one pair
            AssertTrue(s2 == s1);           // spare draw left the stream untouched
            AssertTrue(s3 != s2);           // the following draw advanced it again
        }

        // Over an N-element Gaussian fill the rng advances ceil(N/2)*2 NextDouble steps.
        void GaussianAdvanceCount()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 5;                       // odd: exercises the +1 (discarded spare) draw
            uint seed = 99887766u;

            var rngFill = new Random(seed);
            var g = new doubleGaussian((double)0, (double)1);
            var v = arena.doubleVec(n);
            Rand.randomInpl(ref rngFill, ref v, ref g);
            uint stateFill = rngFill.state;

            // Reference: advance an identically-seeded rng by ceil(n/2)*2 uniform draws.
            int draws = ((n + 1) / 2) * 2;   // n=5 -> 6
            var rngRef = new Random(seed);
            for (int i = 0; i < draws; i++)
                rngRef.NextDouble();
            uint stateRef = rngRef.state;

            AssertTrue(stateFill == stateRef);

            // Even N consumes exactly N draws (no discarded spare).
            int nEven = 6;
            var rngFill2 = new Random(seed);
            var g2 = new doubleGaussian((double)0, (double)1);
            var v2 = arena.doubleVec(nEven);
            Rand.randomInpl(ref rngFill2, ref v2, ref g2);

            var rngRef2 = new Random(seed);
            for (int i = 0; i < nEven; i++)
                rngRef2.NextDouble();
            AssertTrue(rngFill2.state == rngRef2.state);

            arena.Dispose();
        }

        // Matrix overloads fill all M*N flat elements, all in range.
        void MatrixOverloads()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(20240626u);

            // nextUniformInpl(min,max) over a 4x5 matrix: poison first, then assert all in [min,max).
            double mn = (double)(-1), mx = (double)2;
            var M = arena.doubleMat(4, 5);
            for (int i = 0; i < M.Length; i++) M[i] = (double)999;
            Rand.nextUniformInpl(ref rng, ref M, mn, mx);
            AssertTrue(M.Length == 20);
            for (int i = 0; i < M.Length; i++)
                AssertTrue(M[i] >= mn && M[i] < mx);

            // nextUniformInpl [0,1) over a 3x7 matrix.
            var M01 = arena.doubleMat(3, 7);
            for (int i = 0; i < M01.Length; i++) M01[i] = (double)999;
            Rand.nextUniformInpl(ref rng, ref M01);
            for (int i = 0; i < M01.Length; i++)
                AssertTrue(M01[i] >= (double)0 && M01[i] < (double)1);

            // randomInpl<S> over a 3x7 matrix with Exponential: all >= 0, all written.
            var g = new doubleExponential((double)2);
            var ME = arena.doubleMat(3, 7);
            for (int i = 0; i < ME.Length; i++) ME[i] = (double)(-999);
            Rand.randomInpl(ref rng, ref ME, ref g);
            AssertTrue(ME.Length == 21);
            for (int i = 0; i < ME.Length; i++)
                AssertTrue(ME[i] >= (double)0);

            arena.Dispose();
        }

        // ---------------- helpers ----------------

        double Mean(in doubleN v)
        {
            double sum = (double)0;
            for (int i = 0; i < v.N; i++) sum += v[i];
            return sum / (double)v.N;
        }

        double Variance(in doubleN v, double mean)
        {
            double sum = (double)0;
            for (int i = 0; i < v.N; i++)
            {
                double d = v[i] - mean;
                sum += d * d;
            }
            return sum / (double)v.N;
        }

        double FractionBelow(in doubleN v, double threshold)
        {
            int count = 0;
            for (int i = 0; i < v.N; i++)
                if (v[i] < threshold) count++;
            return (double)count / (double)v.N;
        }

        // curr must be >= prev (allowing tiny relative float slack).
        void NonDecreasing(double prev, double curr)
        {
            double slack = math.abs(prev) * (double)1e-4 + (double)1e-5;
            if (!(curr >= prev - slack) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = curr;
                Fail[2] = prev;
                Fail[3] = curr - prev;
            }
            Assert.IsTrue(curr >= prev - slack);
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff
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
    }

    void RunJob(TestJob.TestType type)
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

    [Test] public void ICDFKnownValuesTest() => RunJob(TestJob.TestType.ICDFKnownValues);
    [Test] public void ICDFMonotonicTest() => RunJob(TestJob.TestType.ICDFMonotonic);
    [Test] public void WeibullExponentialIdentityTest() => RunJob(TestJob.TestType.WeibullExponentialIdentity);
    [Test] public void ICDFBoundaryFiniteTest() => RunJob(TestJob.TestType.ICDFBoundaryFinite);
    [Test] public void UniformMomentsTest() => RunJob(TestJob.TestType.UniformMoments);
    [Test] public void ExponentialMomentsTest() => RunJob(TestJob.TestType.ExponentialMoments);
    [Test] public void GaussianMomentsTest() => RunJob(TestJob.TestType.GaussianMoments);
    [Test] public void RayleighMomentsTest() => RunJob(TestJob.TestType.RayleighMoments);
    [Test] public void CauchyMedianTest() => RunJob(TestJob.TestType.CauchyMedian);
    [Test] public void ParetoMedianTest() => RunJob(TestJob.TestType.ParetoMedian);
    [Test] public void RangeSupportTest() => RunJob(TestJob.TestType.RangeSupport);
    [Test] public void DeterminismTest() => RunJob(TestJob.TestType.Determinism);
    [Test] public void StreamAdvanceTest() => RunJob(TestJob.TestType.StreamAdvance);
    [Test] public void GaussianSpareNoAdvanceTest() => RunJob(TestJob.TestType.GaussianSpareNoAdvance);
    [Test] public void GaussianAdvanceCountTest() => RunJob(TestJob.TestType.GaussianAdvanceCount);
    [Test] public void MatrixOverloadsTest() => RunJob(TestJob.TestType.MatrixOverloads);

    // ---------------- D. Managed validation throws (main thread, not in a Burst job) ----------------

    [Test]
    public void SamplerConstructorsValidate()
    {
        // Exponential: lambda must be > 0.
        Assert.Throws<ArgumentException>(() => new doubleExponential((double)0));
        Assert.Throws<ArgumentException>(() => new doubleExponential((double)(-1)));

        // Rayleigh: sigma must be > 0.
        Assert.Throws<ArgumentException>(() => new doubleRayleigh((double)(-1)));

        // Weibull: k must be > 0.
        Assert.Throws<ArgumentException>(() => new doubleWeibull((double)0, (double)1));

        // Pareto: alpha must be > 0 (xm=1 valid, alpha=0 invalid).
        Assert.Throws<ArgumentException>(() => new doublePareto((double)1, (double)0));

        // Gaussian: std must be > 0.
        Assert.Throws<ArgumentException>(() => new doubleGaussian((double)0, (double)(-1)));

        // Uniform: min must be <= max.
        Assert.Throws<ArgumentException>(() => new doubleUniform((double)5, (double)1));

        // Triangular: requires low <= mode <= high.
        Assert.Throws<ArgumentException>(() => new doubleTriangular((double)0, (double)15, (double)10)); // mode>high
        Assert.Throws<ArgumentException>(() => new doubleTriangular((double)5, (double)0, (double)10));  // mode<low
    }

    [Test]
    public void NextUniformInplMinGreaterMaxThrows()
    {
        var arena = new Arena(Allocator.Persistent);

        var v = arena.doubleVec(8);
        Random rng = new Random(1u);
        Assert.Throws<ArgumentException>(() => Rand.nextUniformInpl(ref rng, ref v, (double)5, (double)1));

        var M = arena.doubleMat(3, 3);
        Assert.Throws<ArgumentException>(() => Rand.nextUniformInpl(ref rng, ref M, (double)5, (double)1));

        arena.Dispose();
    }
}
