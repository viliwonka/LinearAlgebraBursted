#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra
{
    /// <summary>
    /// Procedural generators: axes (linspace/arange), curve sampling (sample over any
    /// IfProxyScalarFunction functor), convolution kernels, DSP windows, and rank-1 (1D×1D)
    /// matrix builders (outer / outerSum).
    ///
    /// Every fill comes in two forms — a zero-alloc ref-DESTINATION primitive here
    /// (`fProxyGen_OP.xxx(ref dest, …)`, length taken from dest) and an allocating Arena wrapper
    /// (`arena.fProxyXxx(n, …)`). Use the ref form in per-frame / realtime loops.
    /// fProxy-only. Kernels are normalized to sum 1; easings map t∈[0,1].
    /// </summary>
    public static partial class fProxyGen_OP
    {
        // ---- linspace / arange : the axis ----

        /// <summary>
        /// Fills dest with N evenly spaced values from a to b inclusive: dest[i] = a + (b-a)*i/(N-1).
        /// N==1 yields {a}. This is the canonical input domain for <see cref="sample{F}"/>.
        /// </summary>
        public static void linspace(ref fProxyN dest, fProxy a, fProxy b)
        {
            int N = dest.N;
            if (N == 0)
                throw new ArgumentException("linspace: dest must have length >= 1");

            if (N == 1) { dest[0] = a; return; }

            fProxy scale = (fProxy)1 / (fProxy)(N - 1);
            for (int i = 0; i < N; i++)
                dest[i] = math.lerp(a, b, i * scale);

            // Pin the endpoints exactly: (N-1)*scale is not exactly 1 unless N-1 is a power of two,
            // so the lerp at the last index lands ~1 ulp short of b. Callers rely on dest[N-1] == b.
            dest[0] = a;
            dest[N - 1] = b;
        }

        /// <summary>Fills dest with dest[i] = start + i*step (an arithmetic ramp of length dest.N).</summary>
        public static void arange(ref fProxyN dest, fProxy start, fProxy step)
        {
            int N = dest.N;
            if (N == 0)
                throw new ArgumentException("arange: dest must have length >= 1");

            for (int i = 0; i < N; i++)
                dest[i] = start + i * step;
        }

        // ---- sample<F> : fill from any curve ----

        /// <summary>
        /// Evaluates the functor f at N points evenly spaced over [t0, t1] and writes the results into
        /// dest: dest[i] = f.Eval(t0 + (t1-t0)*i/(N-1)). N==1 yields {f.Eval(t0)}. This is `linspace`
        /// piped through a struct-functor — the Burst-native "lambda" (same pattern as the optimizers).
        /// Works with the built-in <c>fProxyEasing</c>/<c>fProxyWave</c> functors or a caller's own struct.
        /// </summary>
        public static void sample<F>(ref F f, ref fProxyN dest, fProxy t0, fProxy t1)
            where F : struct, IfProxyScalarFunction
        {
            int N = dest.N;
            if (N == 0)
                throw new ArgumentException("sample: dest must have length >= 1");

            if (N == 1) { dest[0] = f.Eval(t0); return; }

            fProxy scale = (fProxy)1 / (fProxy)(N - 1);
            for (int i = 0; i < N; i++)
                dest[i] = f.Eval(math.lerp(t0, t1, i * scale));

            // Evaluate at the exact endpoints (the lerp parameter lands ~1 ulp short of t1 at the
            // last index unless N-1 is a power of two), so the curve hits f(t0) and f(t1) precisely.
            dest[0] = f.Eval(t0);
            dest[N - 1] = f.Eval(t1);
        }

        /// <summary>sample over the default domain [0, 1] (the usual easing / wavetable range).</summary>
        public static void sample<F>(ref F f, ref fProxyN dest)
            where F : struct, IfProxyScalarFunction
            => sample(ref f, ref dest, (fProxy)0, (fProxy)1);

        // ---- convolution kernels (normalized to sum 1, symmetric, centered) ----

        /// <summary>
        /// 1D Gaussian kernel: dest[i] = exp(-(i-c)²/(2σ²)) then divided by its sum, with c=(N-1)/2.
        /// Normalized to sum 1. sigma must be &gt; 0. N==1 yields {1}.
        /// </summary>
        public static void gaussianKernel(ref fProxyN dest, fProxy sigma)
        {
            int N = dest.N;
            if (N == 0)
                throw new ArgumentException("gaussianKernel: dest must have length >= 1");
            if (!(sigma > (fProxy)0))
                throw new ArgumentException("gaussianKernel: sigma must be > 0");

            fProxy c = (fProxy)(N - 1) * (fProxy)0.5;
            fProxy inv2s2 = (fProxy)1 / ((fProxy)2 * sigma * sigma);

            fProxy sum = (fProxy)0;
            for (int i = 0; i < N; i++)
            {
                fProxy d = (fProxy)i - c;
                fProxy w = math.exp(-d * d * inv2s2);
                dest[i] = w;
                sum += w;
            }

            fProxy invSum = (fProxy)1 / sum;
            for (int i = 0; i < N; i++)
                dest[i] *= invSum;
        }

        /// <summary>1D uniform (box) kernel: every weight is 1/N. Normalized to sum 1.</summary>
        public static void boxKernel(ref fProxyN dest)
        {
            int N = dest.N;
            if (N == 0)
                throw new ArgumentException("boxKernel: dest must have length >= 1");

            fProxy w = (fProxy)1 / (fProxy)N;
            for (int i = 0; i < N; i++)
                dest[i] = w;
        }

        /// <summary>
        /// 1D triangular (tent) kernel, peaked at the center and falling off linearly toward the edges:
        /// raw[i] = (c+1) - |i-c| with c=(N-1)/2, then divided by its sum. Normalized to sum 1.
        /// N==1 yields {1}.
        /// </summary>
        public static void tentKernel(ref fProxyN dest)
        {
            int N = dest.N;
            if (N == 0)
                throw new ArgumentException("tentKernel: dest must have length >= 1");

            fProxy c = (fProxy)(N - 1) * (fProxy)0.5;

            fProxy sum = (fProxy)0;
            for (int i = 0; i < N; i++)
            {
                fProxy w = (c + (fProxy)1) - math.abs((fProxy)i - c);
                dest[i] = w;
                sum += w;
            }

            fProxy invSum = (fProxy)1 / sum;
            for (int i = 0; i < N; i++)
                dest[i] *= invSum;
        }

        // ---- DSP window functions ----

        /// <summary>
        /// Fills dest with a DSP window of the given type (index-based, depends on N). Used for tapering
        /// a signal before an FFT or for smoothing. Hann/Hamming/Blackman use the (N-1) denominator;
        /// N==1 yields {1} for every type (the (N-1) formulas are degenerate at a single point).
        /// </summary>
        public static void window(ref fProxyN dest, WindowType type)
        {
            int N = dest.N;
            if (N == 0)
                throw new ArgumentException("window: dest must have length >= 1");

            if (type == WindowType.Box)
            {
                for (int i = 0; i < N; i++)
                    dest[i] = (fProxy)1;
                return;
            }

            if (N == 1) { dest[0] = (fProxy)1; return; }

            fProxy twoPiOverNm1 = (fProxy)(2.0 * System.Math.PI) / (fProxy)(N - 1);
            fProxy fourPiOverNm1 = (fProxy)(4.0 * System.Math.PI) / (fProxy)(N - 1);

            for (int i = 0; i < N; i++)
            {
                fProxy w;
                switch (type)
                {
                    case WindowType.Hann:
                        w = (fProxy)0.5 * ((fProxy)1 - math.cos(twoPiOverNm1 * i));
                        break;
                    case WindowType.Hamming:
                        w = (fProxy)0.54 - (fProxy)0.46 * math.cos(twoPiOverNm1 * i);
                        break;
                    case WindowType.Blackman:
                        w = (fProxy)0.42
                            - (fProxy)0.5 * math.cos(twoPiOverNm1 * i)
                            + (fProxy)0.08 * math.cos(fourPiOverNm1 * i);
                        break;
                    default:
                        w = (fProxy)1;
                        break;
                }
                dest[i] = w;
            }
        }

        // ---- rank-1 (1D × 1D) matrix builders ----

        /// <summary>
        /// Outer product M[i,j] = u[i]*v[j] (a u.N × v.N rank-1 matrix). Forwards to
        /// <see cref="Linear_OP.outerDot(in fProxyN, in fProxyN, ref fProxyMxN)"/>. Use for separable
        /// fields — e.g. a 2D Gaussian is outer(g, g) of a 1D Gaussian.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void outer(in fProxyN u, in fProxyN v, ref fProxyMxN dest)
            => Linear_OP.outerDot(in u, in v, ref dest);

        /// <summary>
        /// Additive outer "sum" M[i,j] = u[i]+v[j] (a u.N × v.N matrix). The separable building block for
        /// additive fields / gradients. No alias guard: the result is a matrix, inputs are vectors, so
        /// they can never share a buffer.
        /// </summary>
        public static void outerSum(in fProxyN u, in fProxyN v, ref fProxyMxN dest)
        {
            if (dest.M_Rows != u.N || dest.N_Cols != v.N)
                throw new ArgumentException("outerSum: dest must be u.N x v.N");

            for (int i = 0; i < u.N; i++)
            {
                fProxy ui = u[i];
                for (int j = 0; j < v.N; j++)
                    dest[i, j] = ui + v[j];
            }
        }

        /// <summary>
        /// N×N separable Gaussian kernel = outer(g, g) of the 1D Gaussian g (normalized to sum 1, so the
        /// 2D kernel also sums to 1). dest must be square (M_Rows == N_Cols). sigma must be &gt; 0.
        /// Allocates one temporary 1D vector internally (Allocator.Temp, disposed before return).
        /// </summary>
        public static void gaussianKernel2D(ref fProxyMxN dest, fProxy sigma)
        {
            if (dest.M_Rows != dest.N_Cols)
                throw new ArgumentException("gaussianKernel2D: dest must be square");

            int N = dest.M_Rows;
            if (N == 0)
                throw new ArgumentException("gaussianKernel2D: dest must be at least 1x1");
            // Validate sigma BEFORE allocating g, so the throw path can't leak the Temp scratch.
            if (!(sigma > (fProxy)0))
                throw new ArgumentException("gaussianKernel2D: sigma must be > 0");

            var g = new fProxyN(N, Allocator.Temp);
            gaussianKernel(ref g, sigma);
            Linear_OP.outerDot(in g, in g, ref dest);
            g.Dispose();
        }
    }
}
