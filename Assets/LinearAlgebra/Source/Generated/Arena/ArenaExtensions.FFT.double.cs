namespace LinearAlgebra
{
    // Allocating (arena) wrappers for the single-output doubleFFT spectrum reductions. The transforms
    // themselves (fft/ifft/rfft/dft/idft) are in-place or caller-provides-arrays, so they have no
    // allocating form; these three reduce a (re, im) spectrum to a fresh real vector.
    public static partial class ArenaExtensions
    {
        /// <summary>Per-bin magnitude sqrt(re² + im²) as a fresh vector.</summary>
        public static doubleN doubleMagnitude(this ref Arena arena, in doubleN re, in doubleN im)
        {
            var dest = arena.doubleVec(re.N);
            doubleFFT.magnitude(in re, in im, ref dest);
            return dest;
        }

        /// <summary>Per-bin power re² + im² as a fresh vector.</summary>
        public static doubleN doublePowerSpectrum(this ref Arena arena, in doubleN re, in doubleN im)
        {
            var dest = arena.doubleVec(re.N);
            doubleFFT.powerSpectrum(in re, in im, ref dest);
            return dest;
        }

        /// <summary>Per-bin phase atan2(im, re) as a fresh vector.</summary>
        public static doubleN doublePhase(this ref Arena arena, in doubleN re, in doubleN im)
        {
            var dest = arena.doubleVec(re.N);
            doubleFFT.phase(in re, in im, ref dest);
            return dest;
        }
    }
}
