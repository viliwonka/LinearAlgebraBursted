using Unity.Collections;

namespace LinearAlgebra
{
    // Standalone allocating overloads for the single-output FFT spectrum reductions: allocate
    // their own output buffer via allocator.
    public static partial class FFT
    {
        /// <summary>Per-bin magnitude sqrt(re² + im²) as a fresh vector.</summary>
        public static fProxyN fProxyMagnitude(in fProxyN re, in fProxyN im, Allocator allocator = Allocator.Temp)
        {
            var dest = new fProxyN(re.N, allocator);
            magnitude(in re, in im, ref dest);
            return dest;
        }

        /// <summary>Per-bin power re² + im² as a fresh vector.</summary>
        public static fProxyN fProxyPowerSpectrum(in fProxyN re, in fProxyN im, Allocator allocator = Allocator.Temp)
        {
            var dest = new fProxyN(re.N, allocator);
            powerSpectrum(in re, in im, ref dest);
            return dest;
        }

        /// <summary>Per-bin phase atan2(im, re) as a fresh vector.</summary>
        public static fProxyN fProxyPhase(in fProxyN re, in fProxyN im, Allocator allocator = Allocator.Temp)
        {
            var dest = new fProxyN(re.N, allocator);
            phase(in re, in im, ref dest);
            return dest;
        }

        /// <summary>
        /// Inverse real FFT: allocates a fresh real output of length N = 2*(re.N-1) and fills it
        /// with <see cref="irfft"/>. re.N must be ≥ 2 and N must be a power of two; ws must be
        /// built for the signal length N.
        /// </summary>
        public static fProxyN fProxyIrfft(in fProxyN re, in fProxyN im, in fProxyFFTCache ws, Allocator allocator = Allocator.Temp)
        {
            int N = (re.N - 1) << 1;
            var real = new fProxyN(N, allocator);
            irfft(in re, in im, ref real, in ws);
            return real;
        }
    }
}
