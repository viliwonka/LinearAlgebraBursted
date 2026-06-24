#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace LinearAlgebra
{
    /// <summary>
    /// 1D discrete Fourier transform. Like <c>eigenvaluesQR</c>, this avoids a complex TYPE by storing
    /// the real and imaginary parts in two parallel <c>doubleN</c> arrays.
    ///
    /// Two algorithms: <see cref="fft"/>/<see cref="ifft"/> are in-place radix-2 Cooley–Tukey (O(N log N),
    /// length must be a power of two); <see cref="dft"/>/<see cref="idft"/> are the direct O(N²) transform
    /// for any N. Forward sign convention: X[k] = Σ x[n]·exp(-2πi·kn/N); the inverse divides by N.
    /// Helpers <see cref="magnitude"/>/<see cref="phase"/>/<see cref="powerSpectrum"/> reduce a
    /// (re, im) pair to a single real vector. Typical DSP pipeline: window (Hann) → rfft → powerSpectrum.
    /// double-only.
    /// </summary>
    public static partial class doubleFFT
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static bool IsPow2(int n) => n > 0 && (n & (n - 1)) == 0;

        // ---- radix-2 in-place FFT (length must be a power of two) ----

        /// <summary>
        /// In-place forward radix-2 FFT of the complex signal (re, im). Both arrays must have the same
        /// length, which must be a power of two. On return they hold the spectrum X[k].
        /// </summary>
        public static void fft(ref doubleN re, ref doubleN im)
        {
            FftCore(ref re, ref im, false);
        }

        /// <summary>
        /// In-place inverse radix-2 FFT (length a power of two). Divides by N, so ifft(fft(x)) == x.
        /// Implemented via the conjugate trick: conjugate → forward FFT → conjugate → scale by 1/N.
        /// </summary>
        public static void ifft(ref doubleN re, ref doubleN im)
        {
            FftCore(ref re, ref im, true);
        }

        static void FftCore(ref doubleN re, ref doubleN im, bool inverse)
        {
            int n = re.N;
            if (im.N != n)
                throw new ArgumentException("fft: re and im must have the same length");
            if (!IsPow2(n))
                throw new ArgumentException("fft: length must be a power of two (use dft for arbitrary N)");
            if (n == 1)
                return;

            // For the inverse, conjugate the input; we conjugate again and scale at the end.
            if (inverse)
                for (int i = 0; i < n; i++)
                    im[i] = -im[i];

            // Bit-reversal permutation (in place).
            for (int i = 1, j = 0; i < n; i++)
            {
                int bit = n >> 1;
                for (; (j & bit) != 0; bit >>= 1)
                    j &= ~bit;
                j |= bit;

                if (i < j)
                {
                    double tr = re[i]; re[i] = re[j]; re[j] = tr;
                    double ti = im[i]; im[i] = im[j]; im[j] = ti;
                }
            }

            // Butterfly stages. The forward transform uses exp(-2πi/len) per stage.
            for (int len = 2; len <= n; len <<= 1)
            {
                double ang = (double)(-2.0 * System.Math.PI) / (double)len;
                double wRe = math.cos(ang);
                double wIm = math.sin(ang);
                int half = len >> 1;

                for (int i = 0; i < n; i += len)
                {
                    double curRe = (double)1;
                    double curIm = (double)0;
                    for (int k = 0; k < half; k++)
                    {
                        int a = i + k;
                        int b = a + half;

                        double bRe = re[b];
                        double bIm = im[b];
                        // v = cur * b
                        double vRe = curRe * bRe - curIm * bIm;
                        double vIm = curRe * bIm + curIm * bRe;

                        double aRe = re[a];
                        double aIm = im[a];
                        re[a] = aRe + vRe; im[a] = aIm + vIm;
                        re[b] = aRe - vRe; im[b] = aIm - vIm;

                        // cur *= w
                        double nRe = curRe * wRe - curIm * wIm;
                        curIm = curRe * wIm + curIm * wRe;
                        curRe = nRe;
                    }
                }
            }

            if (inverse)
            {
                double invN = (double)1 / (double)n;
                for (int i = 0; i < n; i++)
                {
                    re[i] = re[i] * invN;
                    im[i] = -im[i] * invN;   // undo the input conjugation, with the 1/N scale
                }
            }
        }

        /// <summary>
        /// Real-input forward FFT: fills (re, im) from a real signal (im set to 0) and runs the in-place
        /// radix-2 <see cref="fft"/>. real.N must be a power of two; re/im must match its length. re may
        /// alias real; im must NOT alias real or re.
        /// </summary>
        public static void rfft(in doubleN real, ref doubleN re, ref doubleN im)
        {
            int n = real.N;
            if (re.N != n || im.N != n)
                throw new ArgumentException("rfft: re and im must match real.N");

            unsafe
            {
                if (im.Data.Ptr == real.Data.Ptr || im.Data.Ptr == re.Data.Ptr)
                    throw new ArgumentException("rfft: im must not alias real or re");
            }

            for (int i = 0; i < n; i++)
            {
                re[i] = real[i];
                im[i] = (double)0;
            }

            fft(ref re, ref im);
        }

        // ---- direct O(N²) DFT for arbitrary N ----

        /// <summary>
        /// Forward discrete Fourier transform for ANY length N (O(N²)). outRe/outIm receive the spectrum
        /// and must not alias the inputs (each output bin reads every input sample).
        /// PRECISION NOTE: the twiddle angle is baseAng·k·t; the intermediate product k·t reaches ~N²
        /// before the O(1/N) base angle brings the angle itself to O(N). For the float expansion that
        /// large angle still loses accuracy at big N (its ulp approaches a radian near N≈1e3); prefer
        /// the power-of-two <see cref="fft"/>, or the double expansion, when N is large.
        /// </summary>
        public static void dft(in doubleN inRe, in doubleN inIm, ref doubleN outRe, ref doubleN outIm)
        {
            DftCore(in inRe, in inIm, ref outRe, ref outIm, false);
        }

        /// <summary>
        /// Inverse discrete Fourier transform for ANY length N (O(N²), divides by N). outRe/outIm must not
        /// alias the inputs.
        /// </summary>
        public static void idft(in doubleN inRe, in doubleN inIm, ref doubleN outRe, ref doubleN outIm)
        {
            DftCore(in inRe, in inIm, ref outRe, ref outIm, true);
        }

        static void DftCore(in doubleN inRe, in doubleN inIm, ref doubleN outRe, ref doubleN outIm, bool inverse)
        {
            int n = inRe.N;
            if (inIm.N != n)
                throw new ArgumentException("dft: inRe and inIm must have the same length");
            if (outRe.N != n || outIm.N != n)
                throw new ArgumentException("dft: outRe and outIm must match the input length");

            unsafe
            {
                if (outRe.Data.Ptr == inRe.Data.Ptr || outRe.Data.Ptr == inIm.Data.Ptr ||
                    outIm.Data.Ptr == inRe.Data.Ptr || outIm.Data.Ptr == inIm.Data.Ptr)
                    throw new ArgumentException("dft: output must not alias the input");
            }

            if (n == 0)
                return;

            // Forward: exp(-2πi·kn/N). Inverse: exp(+2πi·kn/N) with a 1/N scale.
            double sign = inverse ? (double)1 : (double)(-1);
            double baseAng = sign * (double)(2.0 * System.Math.PI) / (double)n;

            for (int k = 0; k < n; k++)
            {
                double sumRe = (double)0;
                double sumIm = (double)0;
                for (int t = 0; t < n; t++)
                {
                    double ang = baseAng * (double)k * (double)t;
                    double c = math.cos(ang);
                    double s = math.sin(ang);
                    double xr = inRe[t];
                    double xi = inIm[t];
                    // (xr + i·xi)·(c + i·s)
                    sumRe += xr * c - xi * s;
                    sumIm += xr * s + xi * c;
                }
                outRe[k] = sumRe;
                outIm[k] = sumIm;
            }

            if (inverse)
            {
                double invN = (double)1 / (double)n;
                for (int k = 0; k < n; k++)
                {
                    outRe[k] = outRe[k] * invN;
                    outIm[k] = outIm[k] * invN;
                }
            }
        }

        // ---- spectrum reductions (re, im) -> real vector ----

        /// <summary>Per-bin magnitude sqrt(re² + im²). dest may alias re or im (read-before-write per index).</summary>
        public static void magnitude(in doubleN re, in doubleN im, ref doubleN dest)
        {
            int n = re.N;
            if (im.N != n || dest.N != n)
                throw new ArgumentException("magnitude: re, im and dest must have the same length");

            for (int i = 0; i < n; i++)
                dest[i] = math.sqrt(re[i] * re[i] + im[i] * im[i]);
        }

        /// <summary>Per-bin power |X|² = re² + im² (magnitude squared; cheaper, no sqrt). dest may alias re or im.</summary>
        public static void powerSpectrum(in doubleN re, in doubleN im, ref doubleN dest)
        {
            int n = re.N;
            if (im.N != n || dest.N != n)
                throw new ArgumentException("powerSpectrum: re, im and dest must have the same length");

            for (int i = 0; i < n; i++)
                dest[i] = re[i] * re[i] + im[i] * im[i];
        }

        /// <summary>Per-bin phase angle atan2(im, re) in radians, range (-π, π]. dest may alias re or im.</summary>
        public static void phase(in doubleN re, in doubleN im, ref doubleN dest)
        {
            int n = re.N;
            if (im.N != n || dest.N != n)
                throw new ArgumentException("phase: re, im and dest must have the same length");

            for (int i = 0; i < n; i++)
                dest[i] = math.atan2(im[i], re[i]);
        }
    }
}
