namespace LinearAlgebra
{
    // Allocating (arena) wrappers for the single-output FFT spectrum reductions. The transforms
    // themselves (fft/ifft/rfft/dft/idft) are in-place or caller-provides-arrays, so they have no
    // allocating form; these three reduce a (re, im) spectrum to a fresh real vector.
    public static partial class ArenaExtensions
    {
        /// <summary>Per-bin magnitude sqrt(re² + im²) as a fresh vector.</summary>
        public static floatN floatMagnitude(this ref Arena arena, in floatN re, in floatN im)
        {
            var dest = arena.floatVec(re.N);
            FFT.magnitude(in re, in im, ref dest);
            return dest;
        }

        /// <summary>Per-bin power re² + im² as a fresh vector.</summary>
        public static floatN floatPowerSpectrum(this ref Arena arena, in floatN re, in floatN im)
        {
            var dest = arena.floatVec(re.N);
            FFT.powerSpectrum(in re, in im, ref dest);
            return dest;
        }

        /// <summary>Per-bin phase atan2(im, re) as a fresh vector.</summary>
        public static floatN floatPhase(this ref Arena arena, in floatN re, in floatN im)
        {
            var dest = arena.floatVec(re.N);
            FFT.phase(in re, in im, ref dest);
            return dest;
        }

        /// <summary>
        /// Inverse real FFT: allocates a fresh real output of length N = 2*(re.N-1) and fills it
        /// with <see cref="FFT.irfft"/>. re.N must be ≥ 2 and N must be a power of two.
        /// </summary>
        public static floatN floatIrfft(this ref Arena arena, in floatN re, in floatN im)
        {
            int N = (re.N - 1) << 1;
            var real = arena.floatVec(N);
            FFT.irfft(in re, in im, ref real);
            return real;
        }
    }
}
