#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
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
        /// Efficient real-input forward FFT using the two-for-one packing trick: the N-point real signal
        /// is packed into one M-point complex FFT (M = N/2), then the M+1 unique half-spectrum bins are
        /// unpacked. This is ~2× faster than a full N-point FFT at large N.
        /// <para><c>real</c> has length N (power of two, N ≥ 1). <c>re</c> and <c>im</c> are OUTPUT and
        /// must have length exactly N/2+1 (the non-redundant bins). The DC bin <c>im[0]</c> and Nyquist
        /// bin <c>im[N/2]</c> are always zero for a real signal. <c>im</c> must not alias <c>real</c>
        /// or <c>re</c>.</para>
        /// </summary>
        public static void rfft(in doubleN real, ref doubleN re, ref doubleN im)
        {
            int n = real.N;
            if (!IsPow2(n))
                throw new ArgumentException("rfft: length must be a power of two");

            int halfSpec = (n >> 1) + 1; // N/2 + 1
            if (re.N != halfSpec || im.N != halfSpec)
                throw new ArgumentException("rfft: re and im must have length N/2+1");

            unsafe
            {
                if (im.Data.Ptr == real.Data.Ptr || im.Data.Ptr == re.Data.Ptr)
                    throw new ArgumentException("rfft: im must not alias real or re");
            }

            // N=1: trivial; half-spectrum has a single bin.
            if (n == 1)
            {
                re[0] = real[0];
                im[0] = (double)0;
                return;
            }

            int M = n >> 1; // N/2

            // Step 1: Pack even and odd samples into a length-M complex sequence.
            var cz = new doubleN(M, Allocator.Temp, false);
            var sz = new doubleN(M, Allocator.Temp, false);
            for (int j = 0; j < M; j++)
            {
                cz[j] = real[2 * j];
                sz[j] = real[2 * j + 1];
            }

            // Step 2: One length-M complex FFT (M is a power of two since N is).
            fft(ref cz, ref sz);

            // Step 3: Unpack. DC and Nyquist are always real for a real input.
            re[0] = cz[0] + sz[0];
            im[0] = (double)0;
            re[M] = cz[0] - sz[0];
            im[M] = (double)0;

            // General bins k = 1 .. M-1.
            // E[k] = (Y[k] + conj(Y[M-k])) / 2  — DFT of the even samples.
            // O[k] = -i*(Y[k] - conj(Y[M-k])) / 2 — DFT of the odd samples (rotated).
            // X[k] = E[k] + W^k * O[k],  W^k = exp(-2πi·k/N).
            // Twiddle W^k = exp(-2πi·k/N), advanced by a block recurrence (one complex multiply per bin)
            // rather than a cos/sin per bin: M per-bin transcendentals would otherwise dominate and erase
            // the half-length-FFT saving (the same per-element-trig trap the direct DFT has). The twiddle
            // is re-seeded with an exact cos/sin every `twiddleBlock` bins so the recurrence drift stays
            // far under the transform's accuracy (block·eps « 1) — only ~M/block transcendentals total.
            const int twiddleBlock = 256;
            double twoPiOverN = (double)(-2.0 * System.Math.PI) / (double)n;
            double wStepRe = math.cos(twoPiOverN);
            double wStepIm = math.sin(twoPiOverN);
            double curRe = (double)1, curIm = (double)0;
            for (int k = 1; k < M; k++)
            {
                if (((k - 1) & (twiddleBlock - 1)) == 0)        // re-seed W^k exactly at each block start
                {
                    double a = twoPiOverN * (double)k;
                    curRe = math.cos(a);
                    curIm = math.sin(a);
                }
                else                                            // cur *= W
                {
                    double nRe = curRe * wStepRe - curIm * wStepIm;
                    curIm = curRe * wStepIm + curIm * wStepRe;
                    curRe = nRe;
                }

                int kr = M - k;
                double E_re = (cz[k] + cz[kr]) * (double)0.5;
                double E_im = (sz[k] - sz[kr]) * (double)0.5;
                double O_re = (sz[k] + sz[kr]) * (double)0.5;
                double O_im = (cz[kr] - cz[k]) * (double)0.5; // -(cz[k]-cz[kr])*0.5

                re[k] = E_re + (curRe * O_re - curIm * O_im);
                im[k] = E_im + (curRe * O_im + curIm * O_re);
            }

            cz.Dispose();
            sz.Dispose();
        }

        /// <summary>
        /// Inverse of <see cref="rfft"/>: reconstructs the length-N real signal from the half-spectrum
        /// (<c>re</c>, <c>im</c>) of length N/2+1. N = 2·(re.N − 1) must be a power of two and re.N ≥ 2.
        /// <c>real</c> receives the output and must have length exactly N; it must not alias <c>re</c>
        /// or <c>im</c>. <c>irfft(rfft(x)) == x</c> to floating-point precision.
        /// </summary>
        public static void irfft(in doubleN re, in doubleN im, ref doubleN real)
        {
            int halfSpec = re.N; // M + 1, where M = N/2
            if (im.N != halfSpec)
                throw new ArgumentException("irfft: re and im must have the same length");
            if (halfSpec < 2)
                throw new ArgumentException("irfft: re.N must be >= 2 (minimum signal length N=2)");

            int M = halfSpec - 1; // N/2
            int N = M << 1;       // signal length

            if (!IsPow2(N))
                throw new ArgumentException("irfft: N = 2*(re.N-1) must be a power of two");
            if (real.N != N)
                throw new ArgumentException("irfft: real.N must equal 2*(re.N-1)");

            unsafe
            {
                if (real.Data.Ptr == re.Data.Ptr || real.Data.Ptr == im.Data.Ptr)
                    throw new ArgumentException("irfft: real must not alias re or im");
            }

            // Reconstruct Y[0..M-1] = E[k] + i·O[k], the M-point complex FFT of the interleaved
            // even/odd real samples, by inverting the rfft unpack step.
            var cz = new doubleN(M, Allocator.Temp, false);
            var sz = new doubleN(M, Allocator.Temp, false);

            // k=0: X[0] = E[0]+O[0] and X[M] = E[0]-O[0] (both are purely real).
            cz[0] = (re[0] + re[M]) * (double)0.5;
            sz[0] = (re[0] - re[M]) * (double)0.5;

            // General bins k = 1 .. M-1.
            // From X[k] = E[k] + W^k·O[k] and X[M-k] = conj(E[k]) - conj(W^k)·O[k]:
            //   E[k] = (X[k] + conj(X[M-k])) / 2
            //   O[k] = (X[k] - conj(X[M-k])) / (2·W^k) = (X[k]-conj(X[M-k]))·conj(W^k)/2
            //   Y[k] = E[k] + i·O[k]
            // Same block-recurrence twiddle as rfft (one complex multiply per bin, exact re-seed every
            // `twiddleBlock` bins) — avoids the per-bin cos/sin that would dominate the inverse unpack.
            const int twiddleBlock = 256;
            double twoPiOverN = (double)(-2.0 * System.Math.PI) / (double)N;
            double wStepRe = math.cos(twoPiOverN);
            double wStepIm = math.sin(twoPiOverN);
            double curRe = (double)1, curIm = (double)0;
            for (int k = 1; k < M; k++)
            {
                if (((k - 1) & (twiddleBlock - 1)) == 0)        // re-seed W^k exactly at each block start
                {
                    double ang = twoPiOverN * (double)k;
                    curRe = math.cos(ang);
                    curIm = math.sin(ang);
                }
                else                                            // cur *= W
                {
                    double nRe = curRe * wStepRe - curIm * wStepIm;
                    curIm = curRe * wStepIm + curIm * wStepRe;
                    curRe = nRe;
                }

                int kr = M - k;
                double xr_k  = re[k],  xi_k  = im[k];
                double xr_kr = re[kr], xi_kr = im[kr];

                // E[k] = (X[k] + conj(X[M-k])) / 2
                double E_re = (xr_k + xr_kr) * (double)0.5;
                double E_im = (xi_k - xi_kr) * (double)0.5;

                // (X[k] - conj(X[M-k])) = a + i·b
                double a = xr_k - xr_kr;
                double b = xi_k + xi_kr;

                // W^k = curRe + i·curIm (|W^k| = 1, so 1/W^k = conj(W^k) = curRe - i·curIm).
                // O[k] = (a + i·b) · conj(W^k) / 2.
                double O_re = (a * curRe + b * curIm) * (double)0.5;
                double O_im = (b * curRe - a * curIm) * (double)0.5;

                // Y[k] = E[k] + i·O[k]
                cz[k] = E_re - O_im;
                sz[k] = E_im + O_re;
            }

            // One M-point inverse FFT recovers the interleaved even/odd real samples.
            ifft(ref cz, ref sz);

            // Deinterleave: real[2j] = even[j], real[2j+1] = odd[j].
            for (int j = 0; j < M; j++)
            {
                real[2 * j]     = cz[j];
                real[2 * j + 1] = sz[j];
            }

            cz.Dispose();
            sz.Dispose();
        }

        // ---- direct O(N²) DFT for arbitrary N ----

        /// <summary>
        /// Forward discrete Fourier transform for ANY length N (O(N²)). outRe/outIm receive the spectrum
        /// and must not alias the inputs (each output bin reads every input sample).
        /// The twiddle angle is reduced as baseAng·((k·t) mod N) — exact, since exp is 2π-periodic and
        /// baseAng = ±2π/N — so the angle stays in (−2π, 2π] and float sin/cos keeps full accuracy even
        /// at large N. The transform is still O(N²); prefer the power-of-two <see cref="fft"/> for speed.
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
                throw new ArgumentException("dft/idft: inRe and inIm must have the same length");
            if (outRe.N != n || outIm.N != n)
                throw new ArgumentException("dft/idft: outRe and outIm must match the input length");

            unsafe
            {
                if (outRe.Data.Ptr == inRe.Data.Ptr || outRe.Data.Ptr == inIm.Data.Ptr ||
                    outIm.Data.Ptr == inRe.Data.Ptr || outIm.Data.Ptr == inIm.Data.Ptr)
                    throw new ArgumentException("dft/idft: output must not alias the input");
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
                    // baseAng = ±2π/n and exp is 2π-periodic, so baseAng·k·t and baseAng·((k·t) mod n)
                    // give the SAME twiddle exactly. Reducing k·t mod n keeps the angle in (−2π, 2π]
                    // instead of letting it grow to ~2π·N: this keeps float sin/cos accurate at large N
                    // (no precision lost reducing a huge argument) and is much faster under
                    // FloatPrecision.High, which otherwise pays extended-precision range reduction on the
                    // big argument. (long) guards the k·t product against int overflow for large N.
                    int kt = (int)(((long)k * t) % n);
                    double ang = baseAng * (double)kt;
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
