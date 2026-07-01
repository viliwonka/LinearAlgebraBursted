using System;
using System.Runtime.CompilerServices;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

namespace LinearAlgebra
{
    /// <summary>
    /// Zero-alloc random-fill operations for existing vectors and matrices. Complements the
    /// allocating helpers in <c>ArenaExtensions.fProxy</c> (which create a new buffer with a
    /// fresh internal seed). Use these <em>Inpl</em> forms in per-frame / realtime loops where
    /// the buffer already exists and the caller manages the <see cref="Unity.Mathematics.Random"/>
    /// stream for reproducibility and correlation control.
    ///
    /// Uniform refill: <c>nextUniformInpl</c> — overwrites a buffer directly from the caller's
    /// evolving RNG stream.
    /// Generic fill: <c>randomInpl&lt;S&gt;</c> — works with any <see cref="IfProxySampler"/>
    /// struct-functor. ICDF samplers advance rng once per element. <see cref="fProxyGaussian"/>
    /// uses one pair of uniform draws per two samples (Box–Muller), advancing rng by
    /// ceil(N/2)×2 steps: N steps when N is even, N+1 steps when N is odd.
    /// The sampler is passed by <c>ref</c> so that stateful samplers like
    /// <see cref="fProxyGaussian"/> accumulate state across elements. fProxy-only.
    /// </summary>
    public static partial class fProxyRandom_OP
    {
        // ---- uniform refill (vector) ----

        /// <summary>
        /// Overwrites every element of <paramref name="dest"/> with a uniform draw from [0, 1),
        /// advancing <paramref name="rng"/> by <c>dest.N</c> steps.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void nextUniformInpl(ref Random rng, ref fProxyN dest)
        {
            int len = dest.Data.Length;
            for (int i = 0; i < len; i++)
                dest[i] = rng.NextFProxy();
        }

        /// <summary>
        /// Overwrites every element of <paramref name="dest"/> with a uniform draw from
        /// [<paramref name="min"/>, <paramref name="max"/>), advancing <paramref name="rng"/>
        /// by <c>dest.N</c> steps. Throws if <paramref name="min"/> &gt; <paramref name="max"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void nextUniformInpl(ref Random rng, ref fProxyN dest, fProxy min, fProxy max)
        {
            if (!(min <= max))
                throw new ArgumentException("nextUniformInpl: min must be <= max");
            int len = dest.Data.Length;
            for (int i = 0; i < len; i++)
                dest[i] = rng.NextFProxy(min, max);
        }

        // ---- uniform refill (matrix) ----

        /// <summary>
        /// Overwrites every element of <paramref name="dest"/> with a uniform draw from [0, 1),
        /// advancing <paramref name="rng"/> by <c>dest.Length</c> steps.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void nextUniformInpl(ref Random rng, ref fProxyMxN dest)
        {
            int len = dest.Data.Length;
            for (int i = 0; i < len; i++)
                dest[i] = rng.NextFProxy();
        }

        /// <summary>
        /// Overwrites every element of <paramref name="dest"/> with a uniform draw from
        /// [<paramref name="min"/>, <paramref name="max"/>), advancing <paramref name="rng"/>
        /// by <c>dest.Length</c> steps. Throws if <paramref name="min"/> &gt; <paramref name="max"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void nextUniformInpl(ref Random rng, ref fProxyMxN dest, fProxy min, fProxy max)
        {
            if (!(min <= max))
                throw new ArgumentException("nextUniformInpl: min must be <= max");
            int len = dest.Data.Length;
            for (int i = 0; i < len; i++)
                dest[i] = rng.NextFProxy(min, max);
        }

        // ---- generic distribution fill (vector) ----

        /// <summary>
        /// Fills every element of <paramref name="dest"/> by calling <c>s.Next(ref rng)</c>.
        /// ICDF samplers advance rng once per element. <see cref="fProxyGaussian"/> uses one
        /// pair of uniform draws per two samples (Box–Muller), advancing rng by ceil(N/2)×2
        /// steps. The sampler is passed by <c>ref</c> so that mutable state (e.g. Box–Muller
        /// spare) persists across elements. Burst monomorphizes this for each concrete sampler type.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void randomInpl<S>(ref Random rng, ref fProxyN dest, ref S s)
            where S : struct, IfProxySampler
        {
            int len = dest.Data.Length;
            for (int i = 0; i < len; i++)
                dest[i] = s.Next(ref rng);
        }

        // ---- generic distribution fill (matrix) ----

        /// <summary>Matrix overload — fills every element by drawing from <paramref name="s"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void randomInpl<S>(ref Random rng, ref fProxyMxN dest, ref S s)
            where S : struct, IfProxySampler
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
        static int weightedPickFromTotal(in fProxyN weights, fProxy total, ref Random rng)
        {
            fProxy r = rng.NextFProxy() * total;
            fProxy acc = (fProxy)0;
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
        static fProxy weightedPickValidateAndSum(in fProxyN weights)
        {
            int n = weights.N;
            if (n == 0)
                throw new ArgumentException("weightedPick: weights must be non-empty");
            fProxy total = (fProxy)0;
            for (int i = 0; i < n; i++)
            {
                if (!math.isfinite(weights[i]) || !(weights[i] >= (fProxy)0))
                    throw new ArgumentException("weightedPick: all weights must be finite and >= 0");
                total += weights[i];
            }
            if (!(total > (fProxy)0))
                throw new ArgumentException("weightedPick: total weight must be > 0");
            return total;
        }

        /// <summary>
        /// Picks one index from <c>[0, weights.N)</c> with probability proportional to
        /// <paramref name="weights"/> using a linear scan over cumulative weights.
        /// Algorithm: validate; <c>total = Σweights</c>; <c>r = rng.NextFProxy() × total</c>;
        /// walk accumulating until <c>acc &gt; r</c>; return that index.
        /// Clamps to the last index to handle FP edge cases where rounding prevents an early
        /// return. O(N); no allocations.
        /// Throws <see cref="ArgumentException"/> if: weights is empty, any weight is
        /// non-finite or &lt; 0 (+Inf and NaN both throw), or total is 0.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int weightedPick(in fProxyN weights, ref Random rng)
        {
            fProxy total = weightedPickValidateAndSum(in weights);
            return weightedPickFromTotal(in weights, total, ref rng);
        }

        /// <summary>
        /// Fills <paramref name="dest"/> with <c>dest.N</c> independent weighted picks
        /// (with replacement) drawn from <paramref name="weights"/>. Validates and computes
        /// the total once before the draw loop (so invalid weights throw even when
        /// <c>dest.N == 0</c>). Zero-alloc; O(N + k) where k = dest.N.
        /// Throws <see cref="ArgumentException"/> if: weights is empty, any weight is
        /// non-finite or &lt; 0 (+Inf and NaN both throw), or total is 0.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void weightedPickInpl(in fProxyN weights, ref Indices dest, ref Random rng)
        {
            // Validate + sum once before the draw loop so bad weights throw
            // even when dest.N == 0 (empty destination).
            fProxy total = weightedPickValidateAndSum(in weights);
            int k = dest.N;
            for (int i = 0; i < k; i++)
                dest[i] = weightedPickFromTotal(in weights, total, ref rng);
        }
    }

    // ========================================================================
    // Tier-A inverse-transform samplers
    // Each: struct : IfProxySampler, public fields for params, a ctor (with
    // validation), a static ICDF (pure quantile, unit-testable in isolation),
    // and Next = ICDF(rng.NextFProxy()).
    // No default-valued proxy params (CS1750 in templates).
    // ========================================================================

    /// <summary>
    /// Uniform distribution on [<c>min</c>, <c>max</c>).
    /// ICDF: <c>min + (max−min)·u</c>, u∈[0,1). Requires min ≤ max.
    /// </summary>
    public struct fProxyUniform : IfProxySampler
    {
        public fProxy min;
        public fProxy max;

        public fProxyUniform(fProxy min, fProxy max)
        {
            if (!(min <= max))
                throw new ArgumentException("fProxyUniform: min must be <= max");
            this.min = min;
            this.max = max;
        }

        /// <summary>Quantile function: <c>min + (max−min)·u</c>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy UniformICDF(fProxy u, fProxy min, fProxy max)
            => min + (max - min) * u;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fProxy Next(ref Random rng)
            => UniformICDF(rng.NextFProxy(), min, max);
    }

    /// <summary>
    /// Exponential distribution with rate <c>lambda</c> &gt; 0.
    /// ICDF: <c>−log(1−u)/lambda</c>. Draws from [0,∞).
    /// </summary>
    public struct fProxyExponential : IfProxySampler
    {
        public fProxy lambda;

        public fProxyExponential(fProxy lambda)
        {
            if (!(lambda > (fProxy)0))
                throw new ArgumentException("fProxyExponential: lambda must be > 0");
            this.lambda = lambda;
        }

        /// <summary>
        /// Quantile function: <c>−log(1−u)/lambda</c>.
        /// Uses <c>uc = 1−u</c> (maps [0,1)→(0,1]) to keep log argument positive and finite.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy ExponentialICDF(fProxy u, fProxy lambda)
        {
            fProxy uc = (fProxy)1 - u;
            return -math.log(uc) / lambda;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fProxy Next(ref Random rng)
            => ExponentialICDF(rng.NextFProxy(), lambda);
    }

    /// <summary>
    /// Rayleigh distribution with scale <c>sigma</c> &gt; 0.
    /// ICDF: <c>sigma·sqrt(−2·log(1−u))</c>. Draws from [0,∞).
    /// </summary>
    public struct fProxyRayleigh : IfProxySampler
    {
        public fProxy sigma;

        public fProxyRayleigh(fProxy sigma)
        {
            if (!(sigma > (fProxy)0))
                throw new ArgumentException("fProxyRayleigh: sigma must be > 0");
            this.sigma = sigma;
        }

        /// <summary>
        /// Quantile function: <c>sigma·sqrt(−2·log(1−u))</c>.
        /// Uses <c>uc = 1−u</c> (maps [0,1)→(0,1]) to keep log argument positive and finite.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy RayleighICDF(fProxy u, fProxy sigma)
        {
            fProxy uc = (fProxy)1 - u;
            return sigma * math.sqrt((fProxy)(-2) * math.log(uc));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fProxy Next(ref Random rng)
            => RayleighICDF(rng.NextFProxy(), sigma);
    }

    /// <summary>
    /// Weibull distribution with shape <c>k</c> &gt; 0 and scale <c>lambda</c> &gt; 0.
    /// ICDF: <c>lambda·(−log(1−u))^(1/k)</c>. Draws from [0,∞).
    /// </summary>
    public struct fProxyWeibull : IfProxySampler
    {
        public fProxy k;
        public fProxy lambda;

        public fProxyWeibull(fProxy k, fProxy lambda)
        {
            if (!(k > (fProxy)0))
                throw new ArgumentException("fProxyWeibull: k must be > 0");
            if (!(lambda > (fProxy)0))
                throw new ArgumentException("fProxyWeibull: lambda must be > 0");
            this.k = k;
            this.lambda = lambda;
        }

        /// <summary>
        /// Quantile function: <c>lambda·(−log(1−u))^(1/k)</c>.
        /// Uses <c>uc = 1−u</c> (maps [0,1)→(0,1]) to keep log argument positive and finite.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy WeibullICDF(fProxy u, fProxy k, fProxy lambda)
        {
            fProxy uc = (fProxy)1 - u;
            return lambda * math.pow(-math.log(uc), (fProxy)1 / k);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fProxy Next(ref Random rng)
            => WeibullICDF(rng.NextFProxy(), k, lambda);
    }

    /// <summary>
    /// Cauchy distribution with location <c>x0</c> and scale <c>gamma</c> &gt; 0.
    /// ICDF: <c>x0 + gamma·tan(π·(u−0.5))</c>. Heavy-tailed; no finite mean or variance.
    /// </summary>
    public struct fProxyCauchy : IfProxySampler
    {
        public fProxy x0;
        public fProxy gamma;

        public fProxyCauchy(fProxy x0, fProxy gamma)
        {
            if (!(gamma > (fProxy)0))
                throw new ArgumentException("fProxyCauchy: gamma must be > 0");
            this.x0 = x0;
            this.gamma = gamma;
        }

        /// <summary>
        /// Quantile function: <c>x0 + gamma·tan(π·(u−0.5))</c>.
        /// <para>Guard: <c>u</c> is clamped to [<c>Consts.fProxyEpsilon</c>,
        /// 1−<c>Consts.fProxyEpsilon</c>] (~1.19e-7 float / ~2.22e-16 double) to prevent
        /// <c>tan(±π/2)</c> at the endpoints. <c>NextFProxy()</c> returns [0,1) so <c>u=1</c>
        /// never occurs in the sampler path, but <c>u=0</c> can occur and would produce
        /// <c>tan(−π/2)</c> → unbounded. Clamping to machine epsilon preserves the full
        /// distribution tail while bounding the output.</para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy CauchyICDF(fProxy u, fProxy x0, fProxy gamma)
        {
            fProxy eps = Consts.fProxyEpsilon;
            fProxy uSafe = math.clamp(u, eps, (fProxy)1 - eps);
            return x0 + gamma * math.tan((fProxy)System.Math.PI * (uSafe - (fProxy)0.5));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fProxy Next(ref Random rng)
            => CauchyICDF(rng.NextFProxy(), x0, gamma);
    }

    /// <summary>
    /// Logistic distribution with location <c>mu</c> and scale <c>s</c> &gt; 0.
    /// ICDF: <c>mu + s·log(u/(1−u))</c>. Draws from (−∞, +∞).
    /// </summary>
    public struct fProxyLogistic : IfProxySampler
    {
        public fProxy mu;
        public fProxy s;

        public fProxyLogistic(fProxy mu, fProxy s)
        {
            if (!(s > (fProxy)0))
                throw new ArgumentException("fProxyLogistic: s must be > 0");
            this.mu = mu;
            this.s = s;
        }

        /// <summary>
        /// Quantile function (logit): <c>mu + s·log(u/(1−u))</c>.
        /// <para>Guard: <c>u</c> is clamped to [<c>Consts.fProxyEpsilon</c>,
        /// 1−<c>Consts.fProxyEpsilon</c>] (~1.19e-7 float / ~2.22e-16 double) to prevent
        /// <c>log(0)</c> at both endpoints. Clamping to machine epsilon preserves the full
        /// distribution tail while bounding the output.</para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy LogisticICDF(fProxy u, fProxy mu, fProxy s)
        {
            fProxy eps = Consts.fProxyEpsilon;
            fProxy uSafe = math.clamp(u, eps, (fProxy)1 - eps);
            fProxy uc = (fProxy)1 - uSafe;
            return mu + s * math.log(uSafe / uc);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fProxy Next(ref Random rng)
            => LogisticICDF(rng.NextFProxy(), mu, s);
    }

    /// <summary>
    /// Pareto distribution with minimum value <c>xm</c> &gt; 0 and shape <c>alpha</c> &gt; 0.
    /// ICDF: <c>xm / (1−u)^(1/alpha)</c>. Draws from [<c>xm</c>, +∞).
    /// </summary>
    public struct fProxyPareto : IfProxySampler
    {
        public fProxy xm;
        public fProxy alpha;

        public fProxyPareto(fProxy xm, fProxy alpha)
        {
            if (!(xm > (fProxy)0))
                throw new ArgumentException("fProxyPareto: xm must be > 0");
            if (!(alpha > (fProxy)0))
                throw new ArgumentException("fProxyPareto: alpha must be > 0");
            this.xm = xm;
            this.alpha = alpha;
        }

        /// <summary>
        /// Quantile function: <c>xm / (1−u)^(1/alpha)</c>.
        /// Uses <c>uc = 1−u</c> (maps [0,1)→(0,1]) to keep the denominator positive and finite.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy ParetoICDF(fProxy u, fProxy xm, fProxy alpha)
        {
            fProxy uc = (fProxy)1 - u;
            return xm / math.pow(uc, (fProxy)1 / alpha);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fProxy Next(ref Random rng)
            => ParetoICDF(rng.NextFProxy(), xm, alpha);
    }

    /// <summary>
    /// Triangular distribution over [<c>a</c>, <c>b</c>] with mode <c>c</c> (a ≤ c ≤ b).
    /// Constructor takes <c>(low, mode, high)</c>. Requires low ≤ mode ≤ high.
    /// </summary>
    public struct fProxyTriangular : IfProxySampler
    {
        /// <summary>Lower limit (low end of support).</summary>
        public fProxy a;
        /// <summary>Upper limit (high end of support).</summary>
        public fProxy b;
        /// <summary>Mode (peak of the distribution; a ≤ c ≤ b required).</summary>
        public fProxy c;

        /// <summary>
        /// Constructs with <paramref name="low"/> = a, <paramref name="mode"/> = c,
        /// <paramref name="high"/> = b. Requires low ≤ mode ≤ high.
        /// </summary>
        public fProxyTriangular(fProxy low, fProxy mode, fProxy high)
        {
            if (!(low <= mode && mode <= high))
                throw new ArgumentException("fProxyTriangular: requires low <= mode <= high");
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
        public static fProxy TriangularICDF(fProxy u, fProxy a, fProxy c, fProxy b)
        {
            if (b == a) return a; // point-mass: low == mode == high, fc = 0/0 = NaN otherwise
            fProxy fc = (c - a) / (b - a);
            if (u < fc)
                return a + math.sqrt(u * (b - a) * (c - a));
            return b - math.sqrt(((fProxy)1 - u) * (b - a) * (b - c));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fProxy Next(ref Random rng)
            => TriangularICDF(rng.NextFProxy(), a, c, b);
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
    /// <c>fProxyRandom_OP.randomInpl</c> — copying it by value would silently duplicate the
    /// spare state and corrupt the stream.</para>
    ///
    /// <para>No static ICDF is provided: Box–Muller is a two-draw transform, not a simple
    /// closed-form quantile function.</para>
    /// </summary>
    public struct fProxyGaussian : IfProxySampler
    {
        public fProxy mean;
        public fProxy std;
        public bool hasSpare;
        public fProxy spare;

        public fProxyGaussian(fProxy mean, fProxy std)
        {
            if (!(std > (fProxy)0))
                throw new ArgumentException("fProxyGaussian: std must be > 0");
            this.mean = mean;
            this.std = std;
            hasSpare = false;
            spare = (fProxy)0;
        }

        /// <summary>
        /// Returns one Gaussian variate. Every other call returns the fully-scaled spare cached
        /// by the previous Box–Muller pair (no RNG advance). The spare is stored fully scaled so
        /// that a mid-fill change to mean/std cannot silently rescale a pending value.
        /// <c>math.sincos</c> is not used here because its <c>out</c>-parameter overload is not
        /// available via the type-proxy template mechanism; <c>math.sin</c> and <c>math.cos</c>
        /// are called separately instead.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public fProxy Next(ref Random rng)
        {
            if (hasSpare)
            {
                hasSpare = false;
                return spare;
            }

            // Draw u1 from (0, 1] to avoid log(0). u2 from [0, 1) is fine for the angle.
            fProxy u1 = (fProxy)1 - rng.NextFProxy();
            fProxy u2 = rng.NextFProxy();

            fProxy r = math.sqrt((fProxy)(-2) * math.log(u1));
            fProxy angle = (fProxy)(2.0 * System.Math.PI) * u2;

            fProxy sinVal = math.sin(angle);
            fProxy cosVal = math.cos(angle);

            // Store the fully-scaled spare so a mid-fill mean/std change cannot rescale it.
            spare = mean + std * (r * sinVal);
            hasSpare = true;

            return mean + std * (r * cosVal);
        }
    }
}
