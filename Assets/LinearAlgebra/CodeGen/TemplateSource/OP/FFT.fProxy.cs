using System;

using Unity.Mathematics;

namespace LinearAlgebra
{
    /// <summary>
    /// 1D discrete Fourier transform. Like <c>Eigen.valuesQRInPlace</c>, this avoids a complex TYPE by storing
    /// the real and imaginary parts in two parallel <c>fProxyN</c> arrays.
    ///
    /// The fast power-of-two transforms <see cref="fft"/>/<see cref="ifft"/>/<see cref="rfft"/>/<see cref="irfft"/>
    /// require a twiddle-table workspace (built once via <c>arena.fProxyFFTCache(n)</c>); see the workspace
    /// overloads. This file holds the direct O(N²) <see cref="dft"/>/<see cref="idft"/> for arbitrary N and the
    /// spectrum reductions <see cref="magnitude"/>/<see cref="phase"/>/<see cref="powerSpectrum"/> that reduce a
    /// (re, im) pair to a single real vector. Forward sign convention: X[k] = Σ x[n]·exp(-2πi·kn/N); the inverse
    /// divides by N. Typical DSP pipeline: window (Hann) → rfft → powerSpectrum. fProxy-only.
    /// </summary>
    public static partial class FFT
    {
        // ---- direct O(N²) DFT for arbitrary N ----

        /// <summary>
        /// Forward discrete Fourier transform for ANY length N (O(N²)). outRe/outIm receive the spectrum
        /// and must not alias the inputs (each output bin reads every input sample). The twiddle angle
        /// is range-reduced mod N for accuracy at large N — see the comment in DftCore. Still O(N²);
        /// prefer the power-of-two <see cref="fft"/> for speed.
        /// </summary>
        public static void dft(in fProxyN inRe, in fProxyN inIm, ref fProxyN outRe, ref fProxyN outIm)
        {
            DftCore(in inRe, in inIm, ref outRe, ref outIm, false);
        }

        /// <summary>
        /// Inverse discrete Fourier transform for ANY length N (O(N²), divides by N). outRe/outIm must not
        /// alias the inputs.
        /// </summary>
        public static void idft(in fProxyN inRe, in fProxyN inIm, ref fProxyN outRe, ref fProxyN outIm)
        {
            DftCore(in inRe, in inIm, ref outRe, ref outIm, true);
        }

        static void DftCore(in fProxyN inRe, in fProxyN inIm, ref fProxyN outRe, ref fProxyN outIm, bool inverse)
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
            fProxy sign = inverse ? (fProxy)1 : (fProxy)(-1);
            fProxy baseAng = sign * (fProxy)(2.0 * System.Math.PI) / (fProxy)n;

            for (int k = 0; k < n; k++)
            {
                fProxy sumRe = (fProxy)0;
                fProxy sumIm = (fProxy)0;
                for (int t = 0; t < n; t++)
                {
                    // baseAng = ±2π/n and exp is 2π-periodic, so baseAng·k·t and baseAng·((k·t) mod n)
                    // give the SAME twiddle exactly. Reducing k·t mod n keeps the angle in (−2π, 2π]
                    // instead of letting it grow to ~2π·N: this keeps float sin/cos accurate at large N
                    // (no precision lost reducing a huge argument) and is much faster under
                    // FloatPrecision.High, which otherwise pays extended-precision range reduction on the
                    // big argument. (long) guards the k·t product against int overflow for large N.
                    int kt = (int)(((long)k * t) % n);
                    fProxy ang = baseAng * (fProxy)kt;
                    fProxy c = DetMath.Cos(ang);
                    fProxy s = DetMath.Sin(ang);
                    fProxy xr = inRe[t];
                    fProxy xi = inIm[t];
                    // (xr + i·xi)·(c + i·s)
                    sumRe += xr * c - xi * s;
                    sumIm += xr * s + xi * c;
                }
                outRe[k] = sumRe;
                outIm[k] = sumIm;
            }

            if (inverse)
            {
                fProxy invN = (fProxy)1 / (fProxy)n;
                for (int k = 0; k < n; k++)
                {
                    outRe[k] = outRe[k] * invN;
                    outIm[k] = outIm[k] * invN;
                }
            }
        }

        // ---- spectrum reductions (re, im) -> real vector ----

        /// <summary>Per-bin magnitude sqrt(re² + im²). dest may alias re or im (read-before-write per index).</summary>
        public static void magnitude(in fProxyN re, in fProxyN im, ref fProxyN dest)
        {
            int n = re.N;
            if (im.N != n || dest.N != n)
                throw new ArgumentException("magnitude: re, im and dest must have the same length");

            for (int i = 0; i < n; i++)
                dest[i] = math.sqrt(re[i] * re[i] + im[i] * im[i]);
        }

        /// <summary>Per-bin power |X|² = re² + im² (magnitude squared; cheaper, no sqrt). dest may alias re or im.</summary>
        public static void powerSpectrum(in fProxyN re, in fProxyN im, ref fProxyN dest)
        {
            int n = re.N;
            if (im.N != n || dest.N != n)
                throw new ArgumentException("powerSpectrum: re, im and dest must have the same length");

            for (int i = 0; i < n; i++)
                dest[i] = re[i] * re[i] + im[i] * im[i];
        }

        /// <summary>Per-bin phase angle atan2(im, re) in radians, range (-π, π]. dest may alias re or im.</summary>
        public static void phase(in fProxyN re, in fProxyN im, ref fProxyN dest)
        {
            int n = re.N;
            if (im.N != n || dest.N != n)
                throw new ArgumentException("phase: re, im and dest must have the same length");

            for (int i = 0; i < n; i++)
                dest[i] = DetMath.Atan2(im[i], re[i]);
        }
    }
}
