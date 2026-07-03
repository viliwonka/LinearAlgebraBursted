using System;
using System.Runtime.CompilerServices;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

namespace LinearAlgebra
{
    /// <summary>
    /// Zero-alloc random-fill operations for existing vectors and matrices. Complements the
    /// allocating helpers in <c>ArenaExtensions.double</c> (which create a new buffer with a
    /// fresh internal seed). Use these <em>InPlace</em> forms in per-frame / realtime loops where
    /// the buffer already exists and the caller manages the <see cref="Unity.Mathematics.Random"/>
    /// stream for reproducibility and correlation control.
    ///
    /// Uniform refill: <c>nextUniformInPlace</c> — overwrites a buffer directly from the caller's
    /// evolving RNG stream.
    /// Generic fill: <c>randomInPlace&lt;S&gt;</c> — works with any <see cref="IdoubleSampler"/>
    /// struct-functor. ICDF samplers advance rng once per element. <see cref="doubleGaussian"/>
    /// uses one pair of uniform draws per two samples (Box–Muller), advancing rng by
    /// ceil(N/2)×2 steps: N steps when N is even, N+1 steps when N is odd.
    /// The sampler is passed by <c>ref</c> so that stateful samplers like
    /// <see cref="doubleGaussian"/> accumulate state across elements. double-only.
    /// </summary>
    public static partial class Rand
    {
        // ---- uniform refill (vector) ----

        /// <summary>
        /// Overwrites every element of <paramref name="dest"/> with a uniform draw from [0, 1),
        /// advancing <paramref name="rng"/> by <c>dest.N</c> steps.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void nextUniformInPlace(ref Random rng, ref doubleN dest)
        {
            int len = dest.Data.Length;
            for (int i = 0; i < len; i++)
                dest[i] = rng.NextDouble();
        }

        /// <summary>
        /// Overwrites every element of <paramref name="dest"/> with a uniform draw from
        /// [<paramref name="min"/>, <paramref name="max"/>), advancing <paramref name="rng"/>
        /// by <c>dest.N</c> steps. Throws if <paramref name="min"/> &gt; <paramref name="max"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void nextUniformInPlace(ref Random rng, ref doubleN dest, double min, double max)
        {
            if (!(min <= max))
                throw new ArgumentException("nextUniformInPlace: min must be <= max");
            int len = dest.Data.Length;
            for (int i = 0; i < len; i++)
                dest[i] = rng.NextDouble(min, max);
        }

        // ---- uniform refill (matrix) ----

        /// <summary>
        /// Overwrites every element of <paramref name="dest"/> with a uniform draw from [0, 1),
        /// advancing <paramref name="rng"/> by <c>dest.Length</c> steps.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void nextUniformInPlace(ref Random rng, ref doubleMxN dest)
        {
            int len = dest.Data.Length;
            for (int i = 0; i < len; i++)
                dest[i] = rng.NextDouble();
        }

        /// <summary>
        /// Overwrites every element of <paramref name="dest"/> with a uniform draw from
        /// [<paramref name="min"/>, <paramref name="max"/>), advancing <paramref name="rng"/>
        /// by <c>dest.Length</c> steps. Throws if <paramref name="min"/> &gt; <paramref name="max"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void nextUniformInPlace(ref Random rng, ref doubleMxN dest, double min, double max)
        {
            if (!(min <= max))
                throw new ArgumentException("nextUniformInPlace: min must be <= max");
            int len = dest.Data.Length;
            for (int i = 0; i < len; i++)
                dest[i] = rng.NextDouble(min, max);
        }

        // ---- generic distribution fill (vector) ----

        /// <summary>
        /// Fills every element of <paramref name="dest"/> by calling <c>s.Next(ref rng)</c>
        /// (see class summary for the rng-advance contract). Burst monomorphizes this for
        /// each concrete sampler type.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void randomInPlace<S>(ref Random rng, ref doubleN dest, ref S s)
            where S : struct, IdoubleSampler
        {
            int len = dest.Data.Length;
            for (int i = 0; i < len; i++)
                dest[i] = s.Next(ref rng);
        }

        // ---- generic distribution fill (matrix) ----

        /// <summary>Matrix overload — fills every element by drawing from <paramref name="s"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void randomInPlace<S>(ref Random rng, ref doubleMxN dest, ref S s)
            where S : struct, IdoubleSampler
        {
            int len = dest.Data.Length;
            for (int i = 0; i < len; i++)
                dest[i] = s.Next(ref rng);
        }

        // ---- weighted pick ----

        // Private helper: cumulative-scan pick given a pre-validated total.
        // Called by both weightedPick and weightedPickInPlace so validation + summation
        // run only once per public call (not once per draw in the InPlace case).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static int weightedPickFromTotal(in doubleN weights, double total, ref Random rng)
        {
            double r = rng.NextDouble() * total;
            double acc = (double)0;
            int n = weights.N;
            for (int i = 0; i < n; i++)
            {
                acc += weights[i];
                if (acc > r)
                    return i;
            }
            return n - 1; // clamp to last index for FP edge cases
        }

        // Shared validation + summation used by both public entry points.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static double weightedPickValidateAndSum(in doubleN weights)
        {
            int n = weights.N;
            if (n == 0)
                throw new ArgumentException("weightedPick: weights must be non-empty");
            double total = (double)0;
            for (int i = 0; i < n; i++)
            {
                if (!math.isfinite(weights[i]) || !(weights[i] >= (double)0))
                    throw new ArgumentException("weightedPick: all weights must be finite and >= 0");
                total += weights[i];
            }
            if (!(total > (double)0))
                throw new ArgumentException("weightedPick: total weight must be > 0");
            return total;
        }

        /// <summary>
        /// Picks one index from <c>[0, weights.N)</c> with probability proportional to
        /// <paramref name="weights"/> using a linear scan over cumulative weights.
        /// Algorithm: validate; <c>total = Σweights</c>; <c>r = rng.NextDouble() × total</c>;
        /// walk accumulating until <c>acc &gt; r</c>; return that index.
        /// Clamps to the last index to handle FP edge cases where rounding prevents an early
        /// return. O(N); no allocations.
        /// Throws <see cref="ArgumentException"/> if: weights is empty, any weight is
        /// non-finite or &lt; 0 (+Inf and NaN both throw), or total is 0.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int weightedPick(in doubleN weights, ref Random rng)
        {
            double total = weightedPickValidateAndSum(in weights);
            return weightedPickFromTotal(in weights, total, ref rng);
        }

        /// <summary>
        /// Fills <paramref name="dest"/> with <c>dest.N</c> independent weighted picks
        /// (with replacement) drawn from <paramref name="weights"/>, using the same
        /// validation/throw contract as <see cref="weightedPick"/> (checked once up front,
        /// even when <c>dest.N == 0</c>). Zero-alloc; O(N + k) where k = dest.N.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void weightedPickInPlace(in doubleN weights, ref Indices dest, ref Random rng)
        {
            double total = weightedPickValidateAndSum(in weights);
            int k = dest.N;
            for (int i = 0; i < k; i++)
                dest[i] = weightedPickFromTotal(in weights, total, ref rng);
        }
    }

    // ========================================================================
    // Tier-A inverse-transform samplers
    // Each: struct : IdoubleSampler, public fields for params, a ctor (with
    // validation), a static ICDF (pure quantile, unit-testable in isolation),
    // and Next = ICDF(rng.NextDouble()).
    // No default-valued proxy params (CS1750 in templates).
    // ========================================================================

    /// <summary>
    /// Uniform distribution on [<c>min</c>, <c>max</c>).
    /// ICDF: <c>min + (max−min)·u</c>, u∈[0,1). Requires min ≤ max.
    /// </summary>
    public struct doubleUniform : IdoubleSampler
    {
        public double min;
        public double max;

        public doubleUniform(double min, double max)
        {
            if (!(min <= max))
                throw new ArgumentException("doubleUniform: min must be <= max");
            this.min = min;
            this.max = max;
        }

        /// <summary>Quantile function: <c>min + (max−min)·u</c>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double UniformICDF(double u, double min, double max)
            => min + (max - min) * u;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Next(ref Random rng)
            => UniformICDF(rng.NextDouble(), min, max);
    }

    /// <summary>
    /// Exponential distribution with rate <c>lambda</c> &gt; 0.
    /// ICDF: <c>−log(1−u)/lambda</c>. Draws from [0,∞).
    /// </summary>
    public struct doubleExponential : IdoubleSampler
    {
        public double lambda;

        public doubleExponential(double lambda)
        {
            if (!(lambda > (double)0))
                throw new ArgumentException("doubleExponential: lambda must be > 0");
            this.lambda = lambda;
        }

        /// <summary>
        /// Quantile function: <c>−log(1−u)/lambda</c>.
        /// Uses <c>uc = 1−u</c> (maps [0,1)→(0,1]) to keep log argument positive and finite.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ExponentialICDF(double u, double lambda)
        {
            double uc = (double)1 - u;
            return -math.log(uc) / lambda;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Next(ref Random rng)
            => ExponentialICDF(rng.NextDouble(), lambda);
    }

    /// <summary>
    /// Rayleigh distribution with scale <c>sigma</c> &gt; 0.
    /// ICDF: <c>sigma·sqrt(−2·log(1−u))</c>. Draws from [0,∞).
    /// </summary>
    public struct doubleRayleigh : IdoubleSampler
    {
        public double sigma;

        public doubleRayleigh(double sigma)
        {
            if (!(sigma > (double)0))
                throw new ArgumentException("doubleRayleigh: sigma must be > 0");
            this.sigma = sigma;
        }

        /// <summary>
        /// Quantile function: <c>sigma·sqrt(−2·log(1−u))</c>.
        /// Uses <c>uc = 1−u</c> (maps [0,1)→(0,1]) to keep log argument positive and finite.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double RayleighICDF(double u, double sigma)
        {
            double uc = (double)1 - u;
            return sigma * math.sqrt((double)(-2) * math.log(uc));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Next(ref Random rng)
            => RayleighICDF(rng.NextDouble(), sigma);
    }

    /// <summary>
    /// Weibull distribution with shape <c>k</c> &gt; 0 and scale <c>lambda</c> &gt; 0.
    /// ICDF: <c>lambda·(−log(1−u))^(1/k)</c>. Draws from [0,∞).
    /// </summary>
    public struct doubleWeibull : IdoubleSampler
    {
        public double k;
        public double lambda;

        public doubleWeibull(double k, double lambda)
        {
            if (!(k > (double)0))
                throw new ArgumentException("doubleWeibull: k must be > 0");
            if (!(lambda > (double)0))
                throw new ArgumentException("doubleWeibull: lambda must be > 0");
            this.k = k;
            this.lambda = lambda;
        }

        /// <summary>
        /// Quantile function: <c>lambda·(−log(1−u))^(1/k)</c>.
        /// Uses <c>uc = 1−u</c> (maps [0,1)→(0,1]) to keep log argument positive and finite.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double WeibullICDF(double u, double k, double lambda)
        {
            double uc = (double)1 - u;
            return lambda * math.pow(-math.log(uc), (double)1 / k);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Next(ref Random rng)
            => WeibullICDF(rng.NextDouble(), k, lambda);
    }

    /// <summary>
    /// Cauchy distribution with location <c>x0</c> and scale <c>gamma</c> &gt; 0.
    /// ICDF: <c>x0 + gamma·tan(π·(u−0.5))</c>. Heavy-tailed; no finite mean or variance.
    /// </summary>
    public struct doubleCauchy : IdoubleSampler
    {
        public double x0;
        public double gamma;

        public doubleCauchy(double x0, double gamma)
        {
            if (!(gamma > (double)0))
                throw new ArgumentException("doubleCauchy: gamma must be > 0");
            this.x0 = x0;
            this.gamma = gamma;
        }

        /// <summary>
        /// Quantile function: <c>x0 + gamma·tan(π·(u−0.5))</c>.
        /// <para>Guard: <c>u</c> is clamped to [<c>Consts.doubleEpsilon</c>,
        /// 1−<c>Consts.doubleEpsilon</c>] (~1.19e-7 float / ~2.22e-16 double) to prevent
        /// <c>tan(±π/2)</c> at the endpoints. <c>NextDouble()</c> returns [0,1) so <c>u=1</c>
        /// never occurs in the sampler path, but <c>u=0</c> can occur and would produce
        /// <c>tan(−π/2)</c> → unbounded. Clamping to machine epsilon preserves the full
        /// distribution tail while bounding the output.</para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double CauchyICDF(double u, double x0, double gamma)
        {
            double eps = Consts.doubleEpsilon;
            double uSafe = math.clamp(u, eps, (double)1 - eps);
            return x0 + gamma * math.tan((double)System.Math.PI * (uSafe - (double)0.5));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Next(ref Random rng)
            => CauchyICDF(rng.NextDouble(), x0, gamma);
    }

    /// <summary>
    /// Logistic distribution with location <c>mu</c> and scale <c>s</c> &gt; 0.
    /// ICDF: <c>mu + s·log(u/(1−u))</c>. Draws from (−∞, +∞).
    /// </summary>
    public struct doubleLogistic : IdoubleSampler
    {
        public double mu;
        public double s;

        public doubleLogistic(double mu, double s)
        {
            if (!(s > (double)0))
                throw new ArgumentException("doubleLogistic: s must be > 0");
            this.mu = mu;
            this.s = s;
        }

        /// <summary>
        /// Quantile function (logit): <c>mu + s·log(u/(1−u))</c>.
        /// <para>Guard: same epsilon clamp as <see cref="doubleCauchy.CauchyICDF"/>, here
        /// preventing <c>log(0)</c> at both endpoints (Cauchy's guard covers only <c>u=0</c>).</para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double LogisticICDF(double u, double mu, double s)
        {
            double eps = Consts.doubleEpsilon;
            double uSafe = math.clamp(u, eps, (double)1 - eps);
            double uc = (double)1 - uSafe;
            return mu + s * math.log(uSafe / uc);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Next(ref Random rng)
            => LogisticICDF(rng.NextDouble(), mu, s);
    }

    /// <summary>
    /// Pareto distribution with minimum value <c>xm</c> &gt; 0 and shape <c>alpha</c> &gt; 0.
    /// ICDF: <c>xm / (1−u)^(1/alpha)</c>. Draws from [<c>xm</c>, +∞).
    /// </summary>
    public struct doublePareto : IdoubleSampler
    {
        public double xm;
        public double alpha;

        public doublePareto(double xm, double alpha)
        {
            if (!(xm > (double)0))
                throw new ArgumentException("doublePareto: xm must be > 0");
            if (!(alpha > (double)0))
                throw new ArgumentException("doublePareto: alpha must be > 0");
            this.xm = xm;
            this.alpha = alpha;
        }

        /// <summary>
        /// Quantile function: <c>xm / (1−u)^(1/alpha)</c>.
        /// Uses <c>uc = 1−u</c> (maps [0,1)→(0,1]) to keep the denominator positive and finite.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ParetoICDF(double u, double xm, double alpha)
        {
            double uc = (double)1 - u;
            return xm / math.pow(uc, (double)1 / alpha);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Next(ref Random rng)
            => ParetoICDF(rng.NextDouble(), xm, alpha);
    }

    /// <summary>
    /// Triangular distribution over [<c>a</c>, <c>b</c>] with mode <c>c</c> (a ≤ c ≤ b).
    /// Constructor takes <c>(low, mode, high)</c>. Requires low ≤ mode ≤ high.
    /// </summary>
    public struct doubleTriangular : IdoubleSampler
    {
        /// <summary>Lower limit (low end of support).</summary>
        public double a;
        /// <summary>Upper limit (high end of support).</summary>
        public double b;
        /// <summary>Mode (peak of the distribution; a ≤ c ≤ b required).</summary>
        public double c;

        /// <summary>
        /// Constructs with <paramref name="low"/> = a, <paramref name="mode"/> = c,
        /// <paramref name="high"/> = b. Requires low ≤ mode ≤ high.
        /// </summary>
        public doubleTriangular(double low, double mode, double high)
        {
            if (!(low <= mode && mode <= high))
                throw new ArgumentException("doubleTriangular: requires low <= mode <= high");
            a = low;
            c = mode;
            b = high;
        }

        /// <summary>
        /// Quantile function (piecewise): let <c>fc = (c−a)/(b−a)</c>.
        /// If <c>u &lt; fc</c>: <c>a + sqrt(u·(b−a)·(c−a))</c>.
        /// Else: <c>b − sqrt((1−u)·(b−a)·(b−c))</c>.
        /// Parameters: <paramref name="a"/> = low, <paramref name="c"/> = mode, <paramref name="b"/> = high.
        /// <para>Point-mass fast-path: if b == a (i.e. low == mode == high), returns a directly
        /// to avoid 0/0 in the fc computation.</para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double TriangularICDF(double u, double a, double c, double b)
        {
            if (b == a) return a; // point-mass: low == mode == high, fc = 0/0 = NaN otherwise
            double fc = (c - a) / (b - a);
            if (u < fc)
                return a + math.sqrt(u * (b - a) * (c - a));
            return b - math.sqrt(((double)1 - u) * (b - a) * (b - c));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Next(ref Random rng)
            => TriangularICDF(rng.NextDouble(), a, c, b);
    }

    // ========================================================================
    // Box–Muller Gaussian sampler (stateful — caches one spare variate per pair)
    // ========================================================================

    /// <summary>
    /// Gaussian (normal) distribution with mean <c>mean</c> and standard deviation <c>std</c> &gt; 0,
    /// using the Box–Muller transform. Stateful: each pair of uniform draws produces two
    /// Gaussian variates; the second (fully scaled to the current mean/std) is cached in
    /// <c>spare</c> and returned directly on the next call, halving log/sqrt/sin/cos evaluations
    /// over a long fill.
    ///
    /// <para>Because of the cached spare, the sampler MUST be passed by <c>ref</c> to
    /// <c>Rand.randomInPlace</c> — copying it by value would silently duplicate the
    /// spare state and corrupt the stream.</para>
    ///
    /// <para>No static ICDF is provided: Box–Muller is a two-draw transform, not a simple
    /// closed-form quantile function.</para>
    /// </summary>
    public struct doubleGaussian : IdoubleSampler
    {
        public double mean;
        public double std;
        public bool hasSpare;
        public double spare;

        public doubleGaussian(double mean, double std)
        {
            if (!(std > (double)0))
                throw new ArgumentException("doubleGaussian: std must be > 0");
            this.mean = mean;
            this.std = std;
            hasSpare = false;
            spare = (double)0;
        }

        /// <summary>
        /// Returns one Gaussian variate; the cached-spare path (see class summary) does not
        /// advance rng. <c>math.sincos</c> is not used here because its <c>out</c>-parameter
        /// overload is not available via the type-proxy template mechanism; <c>math.sin</c> and
        /// <c>math.cos</c> are called separately instead.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Next(ref Random rng)
        {
            if (hasSpare)
            {
                hasSpare = false;
                return spare;
            }

            // Draw u1 from (0, 1] to avoid log(0). u2 from [0, 1) is fine for the angle.
            double u1 = (double)1 - rng.NextDouble();
            double u2 = rng.NextDouble();

            double r = math.sqrt((double)(-2) * math.log(u1));
            double angle = (double)(2.0 * System.Math.PI) * u2;

            double sinVal = math.sin(angle);
            double cosVal = math.cos(angle);

            // Store the fully-scaled spare so a mid-fill mean/std change cannot rescale it.
            spare = mean + std * (r * sinVal);
            hasSpare = true;

            return mean + std * (r * cosVal);
        }
    }
}
