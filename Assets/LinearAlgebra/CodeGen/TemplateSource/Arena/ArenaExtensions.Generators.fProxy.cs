namespace LinearAlgebra
{
    // Allocating (arena) wrappers for the fProxyGenOP generators — each allocates a fresh persistent
    // vector/matrix and delegates to the zero-alloc ref-dest primitive. Use these for one-off /
    // setup-time builds (tween LUTs, kernels, wavetables); use the fProxyGenOP.xxx(ref dest, …) form
    // inside per-frame loops.
    public static partial class ArenaExtensions
    {
        #region AXIS / SAMPLE

        /// <summary>N evenly spaced values over [a, b] inclusive (linspace). N==1 yields {a}.</summary>
        public static fProxyN fProxyLinspace(this ref Arena arena, fProxy a, fProxy b, int N)
        {
            var vec = arena.fProxyVec(N);
            fProxyGenOP.linspace(ref vec, a, b);
            return vec;
        }

        /// <summary>N-element arithmetic ramp: vec[i] = start + i*step.</summary>
        public static fProxyN fProxyArange(this ref Arena arena, fProxy start, fProxy step, int N)
        {
            var vec = arena.fProxyVec(N);
            fProxyGenOP.arange(ref vec, start, step);
            return vec;
        }

        /// <summary>Samples the functor f at N points over [t0, t1] (linspace piped through f).</summary>
        public static fProxyN fProxySample<F>(this ref Arena arena, ref F f, int N, fProxy t0, fProxy t1)
            where F : struct, IfProxyScalarFunction
        {
            var vec = arena.fProxyVec(N);
            fProxyGenOP.sample(ref f, ref vec, t0, t1);
            return vec;
        }

        /// <summary>Samples the functor f at N points over the default domain [0, 1].</summary>
        public static fProxyN fProxySample<F>(this ref Arena arena, ref F f, int N)
            where F : struct, IfProxyScalarFunction
            => arena.fProxySample(ref f, N, (fProxy)0, (fProxy)1);

        /// <summary>Bakes an easing curve into an N-entry LUT over [0, 1] (== fProxySample on [0,1]).</summary>
        public static fProxyN fProxyEasingLUT<F>(this ref Arena arena, ref F ease, int N)
            where F : struct, IfProxyScalarFunction
            => arena.fProxySample(ref ease, N, (fProxy)0, (fProxy)1);

        #endregion

        #region KERNELS / WINDOWS

        /// <summary>1D normalized Gaussian kernel (sum 1). sigma must be &gt; 0.</summary>
        public static fProxyN fProxyGaussianKernel(this ref Arena arena, int N, fProxy sigma)
        {
            var vec = arena.fProxyVec(N);
            fProxyGenOP.gaussianKernel(ref vec, sigma);
            return vec;
        }

        /// <summary>1D uniform (box) kernel: every weight 1/N.</summary>
        public static fProxyN fProxyBoxKernel(this ref Arena arena, int N)
        {
            var vec = arena.fProxyVec(N);
            fProxyGenOP.boxKernel(ref vec);
            return vec;
        }

        /// <summary>1D triangular (tent) kernel, normalized to sum 1.</summary>
        public static fProxyN fProxyTentKernel(this ref Arena arena, int N)
        {
            var vec = arena.fProxyVec(N);
            fProxyGenOP.tentKernel(ref vec);
            return vec;
        }

        /// <summary>N×N separable Gaussian kernel = outer(g, g) of the 1D Gaussian (sum 1). sigma &gt; 0.</summary>
        public static fProxyMxN fProxyGaussianKernel2D(this ref Arena arena, int N, fProxy sigma)
        {
            var mat = arena.fProxyMat(N, N);
            fProxyGenOP.gaussianKernel2D(ref mat, sigma);
            return mat;
        }

        /// <summary>DSP window of length N (Box/Hann/Hamming/Blackman).</summary>
        public static fProxyN fProxyWindow(this ref Arena arena, int N, WindowType type)
        {
            var vec = arena.fProxyVec(N);
            fProxyGenOP.window(ref vec, type);
            return vec;
        }

        #endregion

        #region RANK-1 (1D × 1D) MATRICES

        /// <summary>Outer product matrix M[i,j] = u[i]*v[j] (u.N × v.N), persistent allocation.</summary>
        public static fProxyMxN fProxyOuter(this ref Arena arena, in fProxyN u, in fProxyN v)
        {
            var mat = arena.fProxyMat(u.N, v.N);
            fProxyGenOP.outer(in u, in v, ref mat);
            return mat;
        }

        /// <summary>Additive outer matrix M[i,j] = u[i]+v[j] (u.N × v.N), persistent allocation.</summary>
        public static fProxyMxN fProxyOuterSum(this ref Arena arena, in fProxyN u, in fProxyN v)
        {
            var mat = arena.fProxyMat(u.N, v.N);
            fProxyGenOP.outerSum(in u, in v, ref mat);
            return mat;
        }

        #endregion
    }
}
