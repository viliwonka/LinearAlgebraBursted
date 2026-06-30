namespace LinearAlgebra
{
    // Allocating (arena) wrappers for the single-output fProxyFFT_OP spectrum reductions. The transforms
    // themselves (fft/ifft/rfft/dft/idft) are in-place or caller-provides-arrays, so they have no
    // allocating form; these three reduce a (re, im) spectrum to a fresh real vector.
    public static partial class ArenaExtensions
    {
        /// <summary>Per-bin magnitude sqrt(re² + im²) as a fresh vector.</summary>
        public static fProxyN fProxyMagnitude(this ref Arena arena, in fProxyN re, in fProxyN im)
        {
            var dest = arena.fProxyVec(re.N);
            fProxyFFT_OP.magnitude(in re, in im, ref dest);
            return dest;
        }

        /// <summary>Per-bin power re² + im² as a fresh vector.</summary>
        public static fProxyN fProxyPowerSpectrum(this ref Arena arena, in fProxyN re, in fProxyN im)
        {
            var dest = arena.fProxyVec(re.N);
            fProxyFFT_OP.powerSpectrum(in re, in im, ref dest);
            return dest;
        }

        /// <summary>Per-bin phase atan2(im, re) as a fresh vector.</summary>
        public static fProxyN fProxyPhase(this ref Arena arena, in fProxyN re, in fProxyN im)
        {
            var dest = arena.fProxyVec(re.N);
            fProxyFFT_OP.phase(in re, in im, ref dest);
            return dest;
        }

        /// <summary>
        /// Inverse real FFT: allocates a fresh real output of length N = 2*(re.N-1) and fills it
        /// with <see cref="fProxyFFT_OP.irfft"/>. re.N must be ≥ 2 and N must be a power of two.
        /// </summary>
        public static fProxyN fProxyIrfft(this ref Arena arena, in fProxyN re, in fProxyN im)
        {
            int N = (re.N - 1) << 1;
            var real = arena.fProxyVec(N);
            fProxyFFT_OP.irfft(in re, in im, ref real);
            return real;
        }
    }
}
