namespace LinearAlgebra
{
    // Allocating (arena) wrappers for the single-output floatFFT spectrum reductions. The transforms
    // themselves (fft/ifft/rfft/dft/idft) are in-place or caller-provides-arrays, so they have no
    // allocating form; these three reduce a (re, im) spectrum to a fresh real vector.
    public static partial class ArenaExtensions
    {
        /// <summary>Per-bin magnitude sqrt(re² + im²) as a fresh vector.</summary>
        public static floatN floatMagnitude(this ref Arena arena, in floatN re, in floatN im)
        {
            var dest = arena.floatVec(re.N);
            floatFFT.magnitude(in re, in im, ref dest);
            return dest;
        }

        /// <summary>Per-bin power re² + im² as a fresh vector.</summary>
        public static floatN floatPowerSpectrum(this ref Arena arena, in floatN re, in floatN im)
        {
            var dest = arena.floatVec(re.N);
            floatFFT.powerSpectrum(in re, in im, ref dest);
            return dest;
        }

        /// <summary>Per-bin phase atan2(im, re) as a fresh vector.</summary>
        public static floatN floatPhase(this ref Arena arena, in floatN re, in floatN im)
        {
            var dest = arena.floatVec(re.N);
            floatFFT.phase(in re, in im, ref dest);
            return dest;
        }
    }
}
