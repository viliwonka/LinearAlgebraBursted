namespace LinearAlgebra
{
    // Allocating (arena) wrappers for the floatGen_OP generators — each allocates a fresh persistent
    // vector/matrix and delegates to the zero-alloc ref-dest primitive. Use these for one-off /
    // setup-time builds (tween LUTs, kernels, wavetables); use the floatGen_OP.xxx(ref dest, …) form
    // inside per-frame loops.
    public static partial class ArenaExtensions
    {
        #region AXIS / SAMPLE

        /// <summary>N evenly spaced values over [a, b] inclusive (linspace). N==1 yields {a}.</summary>
        public static floatN floatLinspace(this ref Arena arena, float a, float b, int N)
        {
            var vec = arena.floatVec(N);
            floatGen_OP.linspace(ref vec, a, b);
            return vec;
        }

        /// <summary>N-element arithmetic ramp: vec[i] = start + i*step.</summary>
        public static floatN floatArange(this ref Arena arena, float start, float step, int N)
        {
            var vec = arena.floatVec(N);
            floatGen_OP.arange(ref vec, start, step);
            return vec;
        }

        /// <summary>Samples the functor f at N points over [t0, t1] (linspace piped through f).</summary>
        public static floatN floatSample<F>(this ref Arena arena, ref F f, int N, float t0, float t1)
            where F : struct, IfloatScalarFunction
        {
            var vec = arena.floatVec(N);
            floatGen_OP.sample(ref f, ref vec, t0, t1);
            return vec;
        }

        /// <summary>Samples the functor f at N points over the default domain [0, 1].</summary>
        public static floatN floatSample<F>(this ref Arena arena, ref F f, int N)
            where F : struct, IfloatScalarFunction
            => arena.floatSample(ref f, N, (float)0, (float)1);

        /// <summary>Bakes an easing curve into an N-entry LUT over [0, 1] (== floatSample on [0,1]).</summary>
        public static floatN floatEasingLUT<F>(this ref Arena arena, ref F ease, int N)
            where F : struct, IfloatScalarFunction
            => arena.floatSample(ref ease, N, (float)0, (float)1);

        #endregion

        #region KERNELS / WINDOWS

        /// <summary>1D normalized Gaussian kernel (sum 1). sigma must be &gt; 0.</summary>
        public static floatN floatGaussianKernel(this ref Arena arena, int N, float sigma)
        {
            var vec = arena.floatVec(N);
            floatGen_OP.gaussianKernel(ref vec, sigma);
            return vec;
        }

        /// <summary>1D uniform (box) kernel: every weight 1/N.</summary>
        public static floatN floatBoxKernel(this ref Arena arena, int N)
        {
            var vec = arena.floatVec(N);
            floatGen_OP.boxKernel(ref vec);
            return vec;
        }

        /// <summary>1D triangular (tent) kernel, normalized to sum 1.</summary>
        public static floatN floatTentKernel(this ref Arena arena, int N)
        {
            var vec = arena.floatVec(N);
            floatGen_OP.tentKernel(ref vec);
            return vec;
        }

        /// <summary>N×N separable Gaussian kernel = outer(g, g) of the 1D Gaussian (sum 1). sigma &gt; 0.</summary>
        public static floatMxN floatGaussianKernel2D(this ref Arena arena, int N, float sigma)
        {
            var mat = arena.floatMat(N, N);
            floatGen_OP.gaussianKernel2D(ref mat, sigma);
            return mat;
        }

        /// <summary>DSP window of length N (Box/Hann/Hamming/Blackman).</summary>
        public static floatN floatWindow(this ref Arena arena, int N, WindowType type)
        {
            var vec = arena.floatVec(N);
            floatGen_OP.window(ref vec, type);
            return vec;
        }

        #endregion

        #region RANK-1 (1D × 1D) MATRICES

        /// <summary>Outer product matrix M[i,j] = u[i]*v[j] (u.N × v.N), persistent allocation.</summary>
        public static floatMxN floatOuter(this ref Arena arena, in floatN u, in floatN v)
        {
            var mat = arena.floatMat(u.N, v.N);
            floatGen_OP.outer(in u, in v, ref mat);
            return mat;
        }

        /// <summary>Additive outer matrix M[i,j] = u[i]+v[j] (u.N × v.N), persistent allocation.</summary>
        public static floatMxN floatOuterSum(this ref Arena arena, in floatN u, in floatN v)
        {
            var mat = arena.floatMat(u.N, v.N);
            floatGen_OP.outerSum(in u, in v, ref mat);
            return mat;
        }

        #endregion
    }
}
