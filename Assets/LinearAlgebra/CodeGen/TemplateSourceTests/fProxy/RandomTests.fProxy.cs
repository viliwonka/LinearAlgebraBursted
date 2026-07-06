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
//     the clampInPlace / FFT guard-test convention.
//
// The underlying uniform stream (NextFProxy -> (fProxy)NextFloat) is float-valued for
// BOTH expansions, so a fixed seed makes every statistic deterministic; the only
// float/double divergence is in the transcendental ICDF math, absorbed by the loose
// tolerances. Never use a time-based seed.
public class fProxyRandomTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
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
        public NativeArray<fProxy> Fail;

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
            fProxy tol = Consts.fProxySqrtEps;

            // Uniform: ICDF(0)=a, ICDF(0.5)=(a+b)/2, ICDF(u)=a+(b-a)u.
            AssertClose(fProxyUniform.UniformICDF((fProxy)0, (fProxy)2, (fProxy)6), (fProxy)2, tol);
            AssertClose(fProxyUniform.UniformICDF((fProxy)0.5, (fProxy)2, (fProxy)6), (fProxy)4, tol);
            AssertClose(fProxyUniform.UniformICDF((fProxy)0.25, (fProxy)0, (fProxy)4), (fProxy)1, tol);

            // Exponential: ICDF(0,λ)=0; ICDF(u,λ)=-ln(1-u)/λ.
            AssertClose(fProxyExponential.ExponentialICDF((fProxy)0, (fProxy)2), (fProxy)0, tol);
            AssertClose(fProxyExponential.ExponentialICDF((fProxy)0.5, (fProxy)2),
                        -math.log((fProxy)0.5) / (fProxy)2, tol);
            AssertClose(fProxyExponential.ExponentialICDF((fProxy)0.3, (fProxy)1.5),
                        -math.log((fProxy)0.7) / (fProxy)1.5, tol);

            // Rayleigh: ICDF(0,σ)=0; median (u=0.5) = σ·sqrt(ln4).
            AssertClose(fProxyRayleigh.RayleighICDF((fProxy)0, (fProxy)2), (fProxy)0, tol);
            AssertClose(fProxyRayleigh.RayleighICDF((fProxy)0.5, (fProxy)2),
                        (fProxy)2 * math.sqrt(math.log((fProxy)4)), tol);

            // Pareto: ICDF(0,xm,α)=xm; with α=1, ICDF(0.5)=2·xm.
            AssertClose(fProxyPareto.ParetoICDF((fProxy)0, (fProxy)2, (fProxy)1.5), (fProxy)2, tol);
            AssertClose(fProxyPareto.ParetoICDF((fProxy)0.5, (fProxy)2, (fProxy)1), (fProxy)4, tol);

            // Logistic: ICDF(0.5,μ,s)=μ.
            AssertClose(fProxyLogistic.LogisticICDF((fProxy)0.5, (fProxy)3, (fProxy)2), (fProxy)3, tol);

            // Cauchy: ICDF(0.5,x0,γ)=x0.
            AssertClose(fProxyCauchy.CauchyICDF((fProxy)0.5, (fProxy)5, (fProxy)2), (fProxy)5, tol);

            // Triangular over [low,high]=[-1,5], mode=2 => fc=(2-(-1))/(5-(-1))=0.5.
            // ICDF(0)=low, ICDF(1)=high, ICDF(fc)=mode.
            AssertClose(fProxyTriangular.TriangularICDF((fProxy)0, (fProxy)(-1), (fProxy)2, (fProxy)5),
                        (fProxy)(-1), tol);
            AssertClose(fProxyTriangular.TriangularICDF((fProxy)1, (fProxy)(-1), (fProxy)2, (fProxy)5),
                        (fProxy)5, tol);
            AssertClose(fProxyTriangular.TriangularICDF((fProxy)0.5, (fProxy)(-1), (fProxy)2, (fProxy)5),
                        (fProxy)2, tol);
            // First branch interior value: u=0.15 < fc => a + sqrt(u·(b-a)·(c-a)) = -1 + sqrt(0.15·6·3).
            AssertClose(fProxyTriangular.TriangularICDF((fProxy)0.15, (fProxy)(-1), (fProxy)2, (fProxy)5),
                        (fProxy)(-1) + math.sqrt((fProxy)0.15 * (fProxy)6 * (fProxy)3), tol);
        }

        // Every ICDF is non-decreasing in u across a sweep of the open interval.
        void ICDFMonotonic()
        {
            int steps = 200;
            fProxy lo = (fProxy)0.005, hi = (fProxy)0.995;
            fProxy step = (hi - lo) / (fProxy)(steps - 1);

            fProxy pU = fProxyUniform.UniformICDF(lo, (fProxy)(-2), (fProxy)4);
            fProxy pE = fProxyExponential.ExponentialICDF(lo, (fProxy)2);
            fProxy pR = fProxyRayleigh.RayleighICDF(lo, (fProxy)1.5);
            fProxy pW = fProxyWeibull.WeibullICDF(lo, (fProxy)1.5, (fProxy)2);
            fProxy pC = fProxyCauchy.CauchyICDF(lo, (fProxy)5, (fProxy)2);
            fProxy pL = fProxyLogistic.LogisticICDF(lo, (fProxy)3, (fProxy)2);
            fProxy pP = fProxyPareto.ParetoICDF(lo, (fProxy)2, (fProxy)1.5);
            fProxy pT = fProxyTriangular.TriangularICDF(lo, (fProxy)(-1), (fProxy)2, (fProxy)5);

            for (int i = 1; i < steps; i++)
            {
                fProxy u = lo + step * (fProxy)i;
                fProxy cU = fProxyUniform.UniformICDF(u, (fProxy)(-2), (fProxy)4);
                fProxy cE = fProxyExponential.ExponentialICDF(u, (fProxy)2);
                fProxy cR = fProxyRayleigh.RayleighICDF(u, (fProxy)1.5);
                fProxy cW = fProxyWeibull.WeibullICDF(u, (fProxy)1.5, (fProxy)2);
                fProxy cC = fProxyCauchy.CauchyICDF(u, (fProxy)5, (fProxy)2);
                fProxy cL = fProxyLogistic.LogisticICDF(u, (fProxy)3, (fProxy)2);
                fProxy cP = fProxyPareto.ParetoICDF(u, (fProxy)2, (fProxy)1.5);
                fProxy cT = fProxyTriangular.TriangularICDF(u, (fProxy)(-1), (fProxy)2, (fProxy)5);

                NonDecreasing(pU, cU); NonDecreasing(pE, cE); NonDecreasing(pR, cR);
                NonDecreasing(pW, cW); NonDecreasing(pC, cC); NonDecreasing(pL, cL);
                NonDecreasing(pP, cP); NonDecreasing(pT, cT);

                pU = cU; pE = cE; pR = cR; pW = cW; pC = cC; pL = cL; pP = cP; pT = cT;
            }
        }

        // Weibull(k=1, λ) reduces to Exponential with rate 1/λ (scale-vs-rate reciprocal).
        void WeibullExponentialIdentity()
        {
            fProxy lambda = (fProxy)2.5;          // Weibull scale
            fProxy rate = (fProxy)1 / lambda;     // Exponential rate
            fProxy tol = Consts.fProxySqrtEps;

            for (int i = 1; i < 50; i++)
            {
                fProxy u = (fProxy)i / (fProxy)50;   // (0,1)
                fProxy w = fProxyWeibull.WeibullICDF(u, (fProxy)1, lambda);
                fProxy e = fProxyExponential.ExponentialICDF(u, rate);
                AssertClose(w, e, tol + math.abs(e) * (fProxy)1e-4);
            }
        }

        // Heavy-tailed Cauchy / Logistic ICDF stay finite at the clamped endpoints (u=0, u->1).
        void ICDFBoundaryFinite()
        {
            fProxy c0 = fProxyCauchy.CauchyICDF((fProxy)0, (fProxy)5, (fProxy)2);
            fProxy c1 = fProxyCauchy.CauchyICDF((fProxy)1, (fProxy)5, (fProxy)2);
            fProxy l0 = fProxyLogistic.LogisticICDF((fProxy)0, (fProxy)3, (fProxy)2);
            fProxy l1 = fProxyLogistic.LogisticICDF((fProxy)1, (fProxy)3, (fProxy)2);

            AssertTrue(math.isfinite(c0));
            AssertTrue(math.isfinite(c1));
            AssertTrue(math.isfinite(l0));
            AssertTrue(math.isfinite(l1));

            // Symmetric clamp about the median: u=0 below x0, u->1 above x0.
            AssertTrue(c0 < (fProxy)5);
            AssertTrue(c1 > (fProxy)5);
            AssertTrue(l0 < (fProxy)3);
            AssertTrue(l1 > (fProxy)3);
        }

        // ---------------- B. Sampler distribution tests ----------------

        const int StatN = 8192;

        // Uniform[a,b]: mean=(a+b)/2, var=(b-a)^2/12.
        void UniformMoments()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(1234567u);
            var s = new fProxyUniform((fProxy)(-2), (fProxy)4);
            var v = arena.fProxyVec(StatN);
            Rand.randomInPlace(ref rng, ref v, ref s);

            fProxy mean = Mean(in v);
            fProxy var = Variance(in v, mean);
            AssertClose(mean, (fProxy)1, (fProxy)0.1);             // (a+b)/2 = 1
            AssertClose(var, (fProxy)3, (fProxy)0.3);              // (b-a)^2/12 = 3
            arena.Dispose();
        }

        // Exponential(λ): mean=1/λ, var=1/λ^2.
        void ExponentialMoments()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(2468013u);
            fProxy lambda = (fProxy)2;
            var s = new fProxyExponential(lambda);
            var v = arena.fProxyVec(StatN);
            Rand.randomInPlace(ref rng, ref v, ref s);

            fProxy mean = Mean(in v);
            fProxy var = Variance(in v, mean);
            AssertClose(mean, (fProxy)1 / lambda, (fProxy)0.05);          // 0.5
            AssertClose(var, (fProxy)1 / (lambda * lambda), (fProxy)0.08); // 0.25
            arena.Dispose();
        }

        // Gaussian(μ,σ): mean≈μ, var≈σ^2.
        void GaussianMoments()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(13572468u);
            fProxy mu = (fProxy)1.5, sd = (fProxy)2;
            var s = new fProxyGaussian(mu, sd);
            var v = arena.fProxyVec(StatN);
            Rand.randomInPlace(ref rng, ref v, ref s);

            fProxy mean = Mean(in v);
            fProxy var = Variance(in v, mean);
            AssertClose(mean, mu, (fProxy)0.12);
            AssertClose(var, sd * sd, (fProxy)0.4);     // σ^2 = 4
            arena.Dispose();
        }

        // Rayleigh(σ): mean=σ·sqrt(π/2).
        void RayleighMoments()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(97531864u);
            fProxy sigma = (fProxy)1.5;
            var s = new fProxyRayleigh(sigma);
            var v = arena.fProxyVec(StatN);
            Rand.randomInPlace(ref rng, ref v, ref s);

            fProxy mean = Mean(in v);
            fProxy expected = sigma * math.sqrt((fProxy)(System.Math.PI / 2.0));
            AssertClose(mean, expected, (fProxy)0.08);
            arena.Dispose();
        }

        // Cauchy: no finite mean/var. Median = x0 => ~50% of draws below x0.
        void CauchyMedian()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(192837465u);
            fProxy x0 = (fProxy)5, gamma = (fProxy)2;
            var s = new fProxyCauchy(x0, gamma);
            var v = arena.fProxyVec(StatN);
            Rand.randomInPlace(ref rng, ref v, ref s);

            fProxy frac = FractionBelow(in v, x0);
            AssertClose(frac, (fProxy)0.5, (fProxy)0.04);
            arena.Dispose();
        }

        // Pareto α=1.5: variance diverges (α<2). Median = xm·2^(1/α) => ~50% below.
        void ParetoMedian()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(564738291u);
            fProxy xm = (fProxy)2, alpha = (fProxy)1.5;
            var s = new fProxyPareto(xm, alpha);
            var v = arena.fProxyVec(StatN);
            Rand.randomInPlace(ref rng, ref v, ref s);

            fProxy median = xm * math.pow((fProxy)2, (fProxy)1 / alpha);
            fProxy frac = FractionBelow(in v, median);
            AssertClose(frac, (fProxy)0.5, (fProxy)0.04);

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
            fProxy a = (fProxy)(-2), b = (fProxy)4;
            var su = new fProxyUniform(a, b);
            var vu = arena.fProxyVec(n);
            Rand.randomInPlace(ref rng, ref vu, ref su);
            for (int i = 0; i < n; i++)
                AssertTrue(vu[i] >= a && vu[i] < b);

            var se = new fProxyExponential((fProxy)2);
            var ve = arena.fProxyVec(n);
            Rand.randomInPlace(ref rng, ref ve, ref se);
            for (int i = 0; i < n; i++)
                AssertTrue(ve[i] >= (fProxy)0);

            var sr = new fProxyRayleigh((fProxy)1.5);
            var vr = arena.fProxyVec(n);
            Rand.randomInPlace(ref rng, ref vr, ref sr);
            for (int i = 0; i < n; i++)
                AssertTrue(vr[i] >= (fProxy)0);

            var sw = new fProxyWeibull((fProxy)1.5, (fProxy)2);
            var vw = arena.fProxyVec(n);
            Rand.randomInPlace(ref rng, ref vw, ref sw);
            for (int i = 0; i < n; i++)
                AssertTrue(vw[i] >= (fProxy)0);

            // Triangular in [low,high].
            fProxy low = (fProxy)(-1), high = (fProxy)5;
            var st = new fProxyTriangular(low, (fProxy)2, high);
            var vt = arena.fProxyVec(n);
            Rand.randomInPlace(ref rng, ref vt, ref st);
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
            var s1 = new fProxyExponential((fProxy)1.7);
            var v1 = arena.fProxyVec(n);
            Rand.randomInPlace(ref r1, ref v1, ref s1);

            var r2 = new Random(55u);
            var s2 = new fProxyExponential((fProxy)1.7);
            var v2 = arena.fProxyVec(n);
            Rand.randomInPlace(ref r2, ref v2, ref s2);

            for (int i = 0; i < n; i++)
                AssertClose(v1[i], v2[i], (fProxy)0);   // bit-identical

            arena.Dispose();
        }

        // Two nextUniformInPlace calls over the SAME rng advance the stream: buffers differ.
        void StreamAdvance()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 256;
            var rng = new Random(7777u);

            var v1 = arena.fProxyVec(n);
            Rand.nextUniformInPlace(ref rng, ref v1);
            var v2 = arena.fProxyVec(n);
            Rand.nextUniformInPlace(ref rng, ref v2);

            bool anyDiff = false;
            for (int i = 0; i < n; i++)
                if (v1[i] != v2[i]) anyDiff = true;
            AssertTrue(anyDiff);

            // Re-seeding resets the stream: a fresh rng reproduces the first buffer.
            var rng3 = new Random(7777u);
            var v3 = arena.fProxyVec(n);
            Rand.nextUniformInPlace(ref rng3, ref v3);
            for (int i = 0; i < n; i++)
                AssertClose(v1[i], v3[i], (fProxy)0);

            arena.Dispose();
        }

        // Box-Muller spare: the 2nd of a pair is returned without advancing the rng.
        void GaussianSpareNoAdvance()
        {
            var rng = new Random(31415926u);
            var g = new fProxyGaussian((fProxy)0, (fProxy)1);

            fProxy a = g.Next(ref rng);
            uint s1 = rng.state;
            fProxy b = g.Next(ref rng);     // spare path: no rng advance
            uint s2 = rng.state;
            fProxy c = g.Next(ref rng);     // new pair: advances
            uint s3 = rng.state;

            AssertTrue(a != b);             // two distinct variates from one pair
            AssertTrue(s2 == s1);           // spare draw left the stream untouched
            AssertTrue(s3 != s2);           // the following draw advanced it again
        }

        // Over an N-element Gaussian fill the rng advances ceil(N/2)*2 NextFProxy steps.
        void GaussianAdvanceCount()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 5;                       // odd: exercises the +1 (discarded spare) draw
            uint seed = 99887766u;

            var rngFill = new Random(seed);
            var g = new fProxyGaussian((fProxy)0, (fProxy)1);
            var v = arena.fProxyVec(n);
            Rand.randomInPlace(ref rngFill, ref v, ref g);
            uint stateFill = rngFill.state;

            // Reference: advance an identically-seeded rng by ceil(n/2)*2 uniform draws.
            int draws = ((n + 1) / 2) * 2;   // n=5 -> 6
            var rngRef = new Random(seed);
            for (int i = 0; i < draws; i++)
                rngRef.NextFProxy();
            uint stateRef = rngRef.state;

            AssertTrue(stateFill == stateRef);

            // Even N consumes exactly N draws (no discarded spare).
            int nEven = 6;
            var rngFill2 = new Random(seed);
            var g2 = new fProxyGaussian((fProxy)0, (fProxy)1);
            var v2 = arena.fProxyVec(nEven);
            Rand.randomInPlace(ref rngFill2, ref v2, ref g2);

            var rngRef2 = new Random(seed);
            for (int i = 0; i < nEven; i++)
                rngRef2.NextFProxy();
            AssertTrue(rngFill2.state == rngRef2.state);

            arena.Dispose();
        }

        // Matrix overloads fill all M*N flat elements, all in range.
        void MatrixOverloads()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(20240626u);

            // nextUniformInPlace(min,max) over a 4x5 matrix: poison first, then assert all in [min,max).
            fProxy mn = (fProxy)(-1), mx = (fProxy)2;
            var M = arena.fProxyMat(4, 5);
            for (int i = 0; i < M.Length; i++) M[i] = (fProxy)999;
            Rand.nextUniformInPlace(ref rng, ref M, mn, mx);
            AssertTrue(M.Length == 20);
            for (int i = 0; i < M.Length; i++)
                AssertTrue(M[i] >= mn && M[i] < mx);

            // nextUniformInPlace [0,1) over a 3x7 matrix.
            var M01 = arena.fProxyMat(3, 7);
            for (int i = 0; i < M01.Length; i++) M01[i] = (fProxy)999;
            Rand.nextUniformInPlace(ref rng, ref M01);
            for (int i = 0; i < M01.Length; i++)
                AssertTrue(M01[i] >= (fProxy)0 && M01[i] < (fProxy)1);

            // randomInPlace<S> over a 3x7 matrix with Exponential: all >= 0, all written.
            var g = new fProxyExponential((fProxy)2);
            var ME = arena.fProxyMat(3, 7);
            for (int i = 0; i < ME.Length; i++) ME[i] = (fProxy)(-999);
            Rand.randomInPlace(ref rng, ref ME, ref g);
            AssertTrue(ME.Length == 21);
            for (int i = 0; i < ME.Length; i++)
                AssertTrue(ME[i] >= (fProxy)0);

            arena.Dispose();
        }

        // ---------------- helpers ----------------

        fProxy Mean(in fProxyN v)
        {
            fProxy sum = (fProxy)0;
            for (int i = 0; i < v.N; i++) sum += v[i];
            return sum / (fProxy)v.N;
        }

        fProxy Variance(in fProxyN v, fProxy mean)
        {
            fProxy sum = (fProxy)0;
            for (int i = 0; i < v.N; i++)
            {
                fProxy d = v[i] - mean;
                sum += d * d;
            }
            return sum / (fProxy)v.N;
        }

        fProxy FractionBelow(in fProxyN v, fProxy threshold)
        {
            int count = 0;
            for (int i = 0; i < v.N; i++)
                if (v[i] < threshold) count++;
            return (fProxy)count / (fProxy)v.N;
        }

        // curr must be >= prev (allowing tiny relative float slack).
        void NonDecreasing(fProxy prev, fProxy curr)
        {
            fProxy slack = math.abs(prev) * (fProxy)1e-4 + (fProxy)1e-5;
            if (!(curr >= prev - slack) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = curr;
                Fail[2] = prev;
                Fail[3] = curr - prev;
            }
            Assert.IsTrue(curr >= prev - slack);
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff
        void AssertClose(fProxy a, fProxy b, fProxy precision)
        {
            fProxy diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = (fProxy)(-1);
                Fail[2] = (fProxy)(-1);
                Fail[3] = (fProxy)(-1);
            }
            Assert.IsTrue(ok);
        }
    }

    void RunJob(TestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (fProxy)0)
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
        Assert.Throws<ArgumentException>(() => new fProxyExponential((fProxy)0));
        Assert.Throws<ArgumentException>(() => new fProxyExponential((fProxy)(-1)));

        // Rayleigh: sigma must be > 0.
        Assert.Throws<ArgumentException>(() => new fProxyRayleigh((fProxy)(-1)));

        // Weibull: k must be > 0.
        Assert.Throws<ArgumentException>(() => new fProxyWeibull((fProxy)0, (fProxy)1));

        // Pareto: alpha must be > 0 (xm=1 valid, alpha=0 invalid).
        Assert.Throws<ArgumentException>(() => new fProxyPareto((fProxy)1, (fProxy)0));

        // Gaussian: std must be > 0.
        Assert.Throws<ArgumentException>(() => new fProxyGaussian((fProxy)0, (fProxy)(-1)));

        // Uniform: min must be <= max.
        Assert.Throws<ArgumentException>(() => new fProxyUniform((fProxy)5, (fProxy)1));

        // Triangular: requires low <= mode <= high.
        Assert.Throws<ArgumentException>(() => new fProxyTriangular((fProxy)0, (fProxy)15, (fProxy)10)); // mode>high
        Assert.Throws<ArgumentException>(() => new fProxyTriangular((fProxy)5, (fProxy)0, (fProxy)10));  // mode<low
    }

    [Test]
    public void NextUniformInPlaceMinGreaterMaxThrows()
    {
        var arena = new Arena(Allocator.Persistent);

        var v = arena.fProxyVec(8);
        Random rng = new Random(1u);
        Assert.Throws<ArgumentException>(() => Rand.nextUniformInPlace(ref rng, ref v, (fProxy)5, (fProxy)1));

        var M = arena.fProxyMat(3, 3);
        Assert.Throws<ArgumentException>(() => Rand.nextUniformInPlace(ref rng, ref M, (fProxy)5, (fProxy)1));

        arena.Dispose();
    }
}
