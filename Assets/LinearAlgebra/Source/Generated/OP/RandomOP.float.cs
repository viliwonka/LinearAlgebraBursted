using System;
using System.Runtime.CompilerServices;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

namespace LinearAlgebra
{
    /// <summary>
    /// Zero-alloc random-fill operations for existing vectors and matrices. Complements the
    /// allocating helpers in <c>ArenaExtensions.float</c> (which create a new buffer with a
    /// fresh internal seed). Use these <em>Inpl</em> forms in per-frame / realtime loops where
    /// the buffer already exists and the caller manages the <see cref="Unity.Mathematics.Random"/>
    /// stream for reproducibility and correlation control.
    ///
    /// Uniform refill: <c>nextUniformInpl</c> — overwrites a buffer directly from the caller's
    /// evolving RNG stream.
    /// Generic fill: <c>randomInpl&lt;S&gt;</c> — works with any <see cref="IfloatSampler"/>
    /// struct-functor. ICDF samplers advance rng once per element. <see cref="floatGaussian"/>
    /// uses one pair of uniform draws per two samples (Box–Muller), advancing rng by
    /// ceil(N/2)×2 steps: N steps when N is even, N+1 steps when N is odd.
    /// The sampler is passed by <c>ref</c> so that stateful samplers like
    /// <see cref="floatGaussian"/> accumulate state across elements. float-only.
    /// </summary>
    public static partial class floatRandom_OP
    {
        // ---- uniform refill (vector) ----

        /// <summary>
        /// Overwrites every element of <paramref name="dest"/> with a uniform draw from [0, 1),
        /// advancing <paramref name="rng"/> by <c>dest.N</c> steps.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void nextUniformInpl(ref Random rng, ref floatN dest)
        {
            int len = dest.Data.Length;
            for (int i = 0; i < len; i++)
                dest[i] = rng.NextFloat();
        }

        /// <summary>
        /// Overwrites every element of <paramref name="dest"/> with a uniform draw from
        /// [<paramref name="min"/>, <paramref name="max"/>), advancing <paramref name="rng"/>
        /// by <c>dest.N</c> steps. Throws if <paramref name="min"/> &gt; <paramref name="max"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void nextUniformInpl(ref Random rng, ref floatN dest, float min, float max)
        {
            if (!(min <= max))
                throw new ArgumentException("nextUniformInpl: min must be <= max");
            int len = dest.Data.Length;
            for (int i = 0; i < len; i++)
                dest[i] = rng.NextFloat(min, max);
        }

        // ---- uniform refill (matrix) ----

        /// <summary>
        /// Overwrites every element of <paramref name="dest"/> with a uniform draw from [0, 1),
        /// advancing <paramref name="rng"/> by <c>dest.Length</c> steps.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void nextUniformInpl(ref Random rng, ref floatMxN dest)
        {
            int len = dest.Data.Length;
            for (int i = 0; i < len; i++)
                dest[i] = rng.NextFloat();
        }

        /// <summary>
        /// Overwrites every element of <paramref name="dest"/> with a uniform draw from
        /// [<paramref name="min"/>, <paramref name="max"/>), advancing <paramref name="rng"/>
        /// by <c>dest.Length</c> steps. Throws if <paramref name="min"/> &gt; <paramref name="max"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void nextUniformInpl(ref Random rng, ref floatMxN dest, float min, float max)
        {
            if (!(min <= max))
                throw new ArgumentException("nextUniformInpl: min must be <= max");
            int len = dest.Data.Length;
            for (int i = 0; i < len; i++)
                dest[i] = rng.NextFloat(min, max);
        }

        // ---- generic distribution fill (vector) ----

        /// <summary>
        /// Fills every element of <paramref name="dest"/> by calling <c>s.Next(ref rng)</c>
        /// (see class summary for the rng-advance contract). Burst monomorphizes this for
        /// each concrete sampler type.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void randomInpl<S>(ref Random rng, ref floatN dest, ref S s)
            where S : struct, IfloatSampler
        {
            int len = dest.Data.Length;
            for (int i = 0; i < len; i++)
                dest[i] = s.Next(ref rng);
        }

        // ---- generic distribution fill (matrix) ----

        /// <summary>Matrix overload — fills every element by drawing from <paramref name="s"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void randomInpl<S>(ref Random rng, ref floatMxN dest, ref S s)
            where S : struct, IfloatSampler
        {
            int len = dest.Data.Length;
            for (int i = 0; i < len; i++)
                dest[i] = s.Next(ref rng);
        }

        // ---- weighted pick ----

        // Private helper: cumulative-scan pick given a pre-validated total.
        // Called by both weightedPick and weightedPickInpl so validation + summation
        // run only once per public call (not once per draw in the Inpl case).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static int weightedPickFromTotal(in floatN weights, float total, ref Random rng)
        {
            float r = rng.NextFloat() * total;
            float acc = (float)0;
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
        static float weightedPickValidateAndSum(in floatN weights)
        {
            int n = weights.N;
            if (n == 0)
                throw new ArgumentException("weightedPick: weights must be non-empty");
            float total = (float)0;
            for (int i = 0; i < n; i++)
            {
                if (!math.isfinite(weights[i]) || !(weights[i] >= (float)0))
                    throw new ArgumentException("weightedPick: all weights must be finite and >= 0");
                total += weights[i];
            }
            if (!(total > (float)0))
                throw new ArgumentException("weightedPick: total weight must be > 0");
            return total;
        }

        /// <summary>
        /// Picks one index from <c>[0, weights.N)</c> with probability proportional to
        /// <paramref name="weights"/> using a linear scan over cumulative weights.
        /// Algorithm: validate; <c>total = Σweights</c>; <c>r = rng.NextFloat() × total</c>;
        /// walk accumulating until <c>acc &gt; r</c>; return that index.
        /// Clamps to the last index to handle FP edge cases where rounding prevents an early
        /// return. O(N); no allocations.
        /// Throws <see cref="ArgumentException"/> if: weights is empty, any weight is
        /// non-finite or &lt; 0 (+Inf and NaN both throw), or total is 0.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int weightedPick(in floatN weights, ref Random rng)
        {
            float total = weightedPickValidateAndSum(in weights);
            return weightedPickFromTotal(in weights, total, ref rng);
        }

        /// <summary>
        /// Fills <paramref name="dest"/> with <c>dest.N</c> independent weighted picks
        /// (with replacement) drawn from <paramref name="weights"/>, using the same
        /// validation/throw contract as <see cref="weightedPick"/> (checked once up front,
        /// even when <c>dest.N == 0</c>). Zero-alloc; O(N + k) where k = dest.N.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void weightedPickInpl(in floatN weights, ref Indices dest, ref Random rng)
        {
            float total = weightedPickValidateAndSum(in weights);
            int k = dest.N;
            for (int i = 0; i < k; i++)
                dest[i] = weightedPickFromTotal(in weights, total, ref rng);
        }
    }

    // ========================================================================
    // Tier-A inverse-transform samplers
    // Each: struct : IfloatSampler, public fields for params, a ctor (with
    // validation), a static ICDF (pure quantile, unit-testable in isolation),
    // and Next = ICDF(rng.NextFloat()).
    // No default-valued proxy params (CS1750 in templates).
    // ========================================================================

    /// <summary>
    /// Uniform distribution on [<c>min</c>, <c>max</c>).
    /// ICDF: <c>min + (max−min)·u</c>, u∈[0,1). Requires min ≤ max.
    /// </summary>
    public struct floatUniform : IfloatSampler
    {
        public float min;
        public float max;

        public floatUniform(float min, float max)
        {
            if (!(min <= max))
                throw new ArgumentException("floatUniform: min must be <= max");
            this.min = min;
            this.max = max;
        }

        /// <summary>Quantile function: <c>min + (max−min)·u</c>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float UniformICDF(float u, float min, float max)
            => min + (max - min) * u;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Next(ref Random rng)
            => UniformICDF(rng.NextFloat(), min, max);
    }

    /// <summary>
    /// Exponential distribution with rate <c>lambda</c> &gt; 0.
    /// ICDF: <c>−log(1−u)/lambda</c>. Draws from [0,∞).
    /// </summary>
    public struct floatExponential : IfloatSampler
    {
        public float lambda;

        public floatExponential(float lambda)
        {
            if (!(lambda > (float)0))
                throw new ArgumentException("floatExponential: lambda must be > 0");
            this.lambda = lambda;
        }

        /// <summary>
        /// Quantile function: <c>−log(1−u)/lambda</c>.
        /// Uses <c>uc = 1−u</c> (maps [0,1)→(0,1]) to keep log argument positive and finite.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ExponentialICDF(float u, float lambda)
        {
            float uc = (float)1 - u;
            return -math.log(uc) / lambda;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Next(ref Random rng)
            => ExponentialICDF(rng.NextFloat(), lambda);
    }

    /// <summary>
    /// Rayleigh distribution with scale <c>sigma</c> &gt; 0.
    /// ICDF: <c>sigma·sqrt(−2·log(1−u))</c>. Draws from [0,∞).
    /// </summary>
    public struct floatRayleigh : IfloatSampler
    {
        public float sigma;

        public floatRayleigh(float sigma)
        {
            if (!(sigma > (float)0))
                throw new ArgumentException("floatRayleigh: sigma must be > 0");
            this.sigma = sigma;
        }

        /// <summary>
        /// Quantile function: <c>sigma·sqrt(−2·log(1−u))</c>.
        /// Uses <c>uc = 1−u</c> (maps [0,1)→(0,1]) to keep log argument positive and finite.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float RayleighICDF(float u, float sigma)
        {
            float uc = (float)1 - u;
            return sigma * math.sqrt((float)(-2) * math.log(uc));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Next(ref Random rng)
            => RayleighICDF(rng.NextFloat(), sigma);
    }

    /// <summary>
    /// Weibull distribution with shape <c>k</c> &gt; 0 and scale <c>lambda</c> &gt; 0.
    /// ICDF: <c>lambda·(−log(1−u))^(1/k)</c>. Draws from [0,∞).
    /// </summary>
    public struct floatWeibull : IfloatSampler
    {
        public float k;
        public float lambda;

        public floatWeibull(float k, float lambda)
        {
            if (!(k > (float)0))
                throw new ArgumentException("floatWeibull: k must be > 0");
            if (!(lambda > (float)0))
                throw new ArgumentException("floatWeibull: lambda must be > 0");
            this.k = k;
            this.lambda = lambda;
        }

        /// <summary>
        /// Quantile function: <c>lambda·(−log(1−u))^(1/k)</c>.
        /// Uses <c>uc = 1−u</c> (maps [0,1)→(0,1]) to keep log argument positive and finite.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float WeibullICDF(float u, float k, float lambda)
        {
            float uc = (float)1 - u;
            return lambda * math.pow(-math.log(uc), (float)1 / k);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Next(ref Random rng)
            => WeibullICDF(rng.NextFloat(), k, lambda);
    }

    /// <summary>
    /// Cauchy distribution with location <c>x0</c> and scale <c>gamma</c> &gt; 0.
    /// ICDF: <c>x0 + gamma·tan(π·(u−0.5))</c>. Heavy-tailed; no finite mean or variance.
    /// </summary>
    public struct floatCauchy : IfloatSampler
    {
        public float x0;
        public float gamma;

        public floatCauchy(float x0, float gamma)
        {
            if (!(gamma > (float)0))
                throw new ArgumentException("floatCauchy: gamma must be > 0");
            this.x0 = x0;
            this.gamma = gamma;
        }

        /// <summary>
        /// Quantile function: <c>x0 + gamma·tan(π·(u−0.5))</c>.
        /// <para>Guard: <c>u</c> is clamped to [<c>Consts.floatEpsilon</c>,
        /// 1−<c>Consts.floatEpsilon</c>] (~1.19e-7 float / ~2.22e-16 double) to prevent
        /// <c>tan(±π/2)</c> at the endpoints. <c>NextFloat()</c> returns [0,1) so <c>u=1</c>
        /// never occurs in the sampler path, but <c>u=0</c> can occur and would produce
        /// <c>tan(−π/2)</c> → unbounded. Clamping to machine epsilon preserves the full
        /// distribution tail while bounding the output.</para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float CauchyICDF(float u, float x0, float gamma)
        {
            float eps = Consts.floatEpsilon;
            float uSafe = math.clamp(u, eps, (float)1 - eps);
            return x0 + gamma * math.tan((float)System.Math.PI * (uSafe - (float)0.5));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Next(ref Random rng)
            => CauchyICDF(rng.NextFloat(), x0, gamma);
    }

    /// <summary>
    /// Logistic distribution with location <c>mu</c> and scale <c>s</c> &gt; 0.
    /// ICDF: <c>mu + s·log(u/(1−u))</c>. Draws from (−∞, +∞).
    /// </summary>
    public struct floatLogistic : IfloatSampler
    {
        public float mu;
        public float s;

        public floatLogistic(float mu, float s)
        {
            if (!(s > (float)0))
                throw new ArgumentException("floatLogistic: s must be > 0");
            this.mu = mu;
            this.s = s;
        }

        /// <summary>
        /// Quantile function (logit): <c>mu + s·log(u/(1−u))</c>.
        /// <para>Guard: same epsilon clamp as <see cref="floatCauchy.CauchyICDF"/>, here
        /// preventing <c>log(0)</c> at both endpoints (Cauchy's guard covers only <c>u=0</c>).</para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LogisticICDF(float u, float mu, float s)
        {
            float eps = Consts.floatEpsilon;
            float uSafe = math.clamp(u, eps, (float)1 - eps);
            float uc = (float)1 - uSafe;
            return mu + s * math.log(uSafe / uc);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Next(ref Random rng)
            => LogisticICDF(rng.NextFloat(), mu, s);
    }

    /// <summary>
    /// Pareto distribution with minimum value <c>xm</c> &gt; 0 and shape <c>alpha</c> &gt; 0.
    /// ICDF: <c>xm / (1−u)^(1/alpha)</c>. Draws from [<c>xm</c>, +∞).
    /// </summary>
    public struct floatPareto : IfloatSampler
    {
        public float xm;
        public float alpha;

        public floatPareto(float xm, float alpha)
        {
            if (!(xm > (float)0))
                throw new ArgumentException("floatPareto: xm must be > 0");
            if (!(alpha > (float)0))
                throw new ArgumentException("floatPareto: alpha must be > 0");
            this.xm = xm;
            this.alpha = alpha;
        }

        /// <summary>
        /// Quantile function: <c>xm / (1−u)^(1/alpha)</c>.
        /// Uses <c>uc = 1−u</c> (maps [0,1)→(0,1]) to keep the denominator positive and finite.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ParetoICDF(float u, float xm, float alpha)
        {
            float uc = (float)1 - u;
            return xm / math.pow(uc, (float)1 / alpha);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Next(ref Random rng)
            => ParetoICDF(rng.NextFloat(), xm, alpha);
    }

    /// <summary>
    /// Triangular distribution over [<c>a</c>, <c>b</c>] with mode <c>c</c> (a ≤ c ≤ b).
    /// Constructor takes <c>(low, mode, high)</c>. Requires low ≤ mode ≤ high.
    /// </summary>
    public struct floatTriangular : IfloatSampler
    {
        /// <summary>Lower limit (low end of support).</summary>
        public float a;
        /// <summary>Upper limit (high end of support).</summary>
        public float b;
        /// <summary>Mode (peak of the distribution; a ≤ c ≤ b required).</summary>
        public float c;

        /// <summary>
        /// Constructs with <paramref name="low"/> = a, <paramref name="mode"/> = c,
        /// <paramref name="high"/> = b. Requires low ≤ mode ≤ high.
        /// </summary>
        public floatTriangular(float low, float mode, float high)
        {
            if (!(low <= mode && mode <= high))
                throw new ArgumentException("floatTriangular: requires low <= mode <= high");
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
        public static float TriangularICDF(float u, float a, float c, float b)
        {
            if (b == a) return a; // point-mass: low == mode == high, fc = 0/0 = NaN otherwise
            float fc = (c - a) / (b - a);
            if (u < fc)
                return a + math.sqrt(u * (b - a) * (c - a));
            return b - math.sqrt(((float)1 - u) * (b - a) * (b - c));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Next(ref Random rng)
            => TriangularICDF(rng.NextFloat(), a, c, b);
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
    /// <c>floatRandom_OP.randomInpl</c> — copying it by value would silently duplicate the
    /// spare state and corrupt the stream.</para>
    ///
    /// <para>No static ICDF is provided: Box–Muller is a two-draw transform, not a simple
    /// closed-form quantile function.</para>
    /// </summary>
    public struct floatGaussian : IfloatSampler
    {
        public float mean;
        public float std;
        public bool hasSpare;
        public float spare;

        public floatGaussian(float mean, float std)
        {
            if (!(std > (float)0))
                throw new ArgumentException("floatGaussian: std must be > 0");
            this.mean = mean;
            this.std = std;
            hasSpare = false;
            spare = (float)0;
        }

        /// <summary>
        /// Returns one Gaussian variate; the cached-spare path (see class summary) does not
        /// advance rng. <c>math.sincos</c> is not used here because its <c>out</c>-parameter
        /// overload is not available via the type-proxy template mechanism; <c>math.sin</c> and
        /// <c>math.cos</c> are called separately instead.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Next(ref Random rng)
        {
            if (hasSpare)
            {
                hasSpare = false;
                return spare;
            }

            // Draw u1 from (0, 1] to avoid log(0). u2 from [0, 1) is fine for the angle.
            float u1 = (float)1 - rng.NextFloat();
            float u2 = rng.NextFloat();

            float r = math.sqrt((float)(-2) * math.log(u1));
            float angle = (float)(2.0 * System.Math.PI) * u2;

            float sinVal = math.sin(angle);
            float cosVal = math.cos(angle);

            // Store the fully-scaled spare so a mid-fill mean/std change cannot rescale it.
            spare = mean + std * (r * sinVal);
            hasSpare = true;

            return mean + std * (r * cosVal);
        }
    }
}
