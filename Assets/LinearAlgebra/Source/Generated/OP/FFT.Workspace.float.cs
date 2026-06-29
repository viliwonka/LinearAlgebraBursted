#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace LinearAlgebra
{
    /// <summary>
    /// Precomputed twiddle table for floatFFT. One size-n table serves every radix-2 transform of
    /// length ≤ n: the stage-len butterfly twiddle W_len^k is indexed as twRe[k*(n/len)],
    /// twIm[k*(n/len)], eliminating per-element cos/sin from the hot loop. Build once via
    /// Arena.floatFftWorkspace(n) and reuse across many transforms of the same size.
    ///
    /// The table is computed at double precision and cast to the element type (float or double),
    /// so accuracy is maximized regardless of the transform element type.
    ///
    /// The full-circle table (twReFull/twImFull, length n) extends the half-table for radix-4:
    /// radix-4 twiddles reach index 3n/4, past the half-table boundary. Bandwidth tradeoff:
    /// full table uses ~2× twiddle memory (~16 MB at N=1M for float), offset by halving the
    /// number of full-array passes (log4(N) vs log2(N) passes).
    /// </summary>
    public struct floatFftWorkspace
    {
        public floatN twRe;       // length n/2: cos(-2π·j/n), j = 0..n/2-1  (half circle, radix-2)
        public floatN twIm;       // length n/2: sin(-2π·j/n)
        public floatN twReFull;   // length n:   cos(-2π·m/n), m = 0..n-1    (full circle, radix-4)
        public floatN twImFull;   // length n:   sin(-2π·m/n)
        public int n;              // the FFT size this table is built for (must be a power of two, >= 2)
    }

    public partial struct Arena
    {
        /// <summary>
        /// Allocates a twiddle-table FFT workspace for an n-point transform (n must be a power of two,
        /// n ≥ 2). twRe[j] = cos(-2π·j/n), twIm[j] = sin(-2π·j/n) for j = 0..n/2-1, computed at
        /// double precision so the table is maximally accurate regardless of element type. Entries are
        /// computed with direct per-entry cos/sin (no recurrence) for full accuracy.
        ///
        /// The full-circle table twReFull/twImFull (length n) is also built: needed by fftRadix4 /
        /// ifftRadix4 whose twiddles reach index 3n/4. The half-table suffices for all radix-2 paths.
        ///
        /// The buffers are persistent in this arena (disposed with it), so create the workspace once
        /// outside a hot loop and pass it to the table overloads. One table serves fft/ifft of length
        /// exactly n and rfft/irfft of real signal length exactly n.
        /// </summary>
        public floatFftWorkspace floatFftWorkspace(int n)
        {
            if (n < 2 || (n & (n - 1)) != 0)
                throw new ArgumentException("floatFftWorkspace: n must be a power of two and >= 2");

            int half = n >> 1;
            var twRe     = floatVec(half);
            var twIm     = floatVec(half);
            var twReFull = floatVec(n);
            var twImFull = floatVec(n);

            // Build at double precision for accuracy; direct per-entry cos/sin, no recurrence drift.
            double twoPiOverN = -2.0 * System.Math.PI / n;
            for (int j = 0; j < half; j++)
            {
                double ang = twoPiOverN * j;
                twRe[j] = (float)math.cos(ang);
                twIm[j] = (float)math.sin(ang);
            }

            // Full-circle table: same angle formula, extended to m = 0..n-1.
            for (int m = 0; m < n; m++)
            {
                double ang = twoPiOverN * m;
                twReFull[m] = (float)math.cos(ang);
                twImFull[m] = (float)math.sin(ang);
            }

            return new floatFftWorkspace
            {
                twRe     = twRe,
                twIm     = twIm,
                twReFull = twReFull,
                twImFull = twImFull,
                n        = n,
            };
        }
    }

    public static partial class floatFFT
    {
        // ---- workspace guard ----

        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an n-point FFT. Matches the layout
        /// produced by Arena.floatFftWorkspace(n): ws.n == n and table lengths == n/2.
        /// </summary>
        static void RequireFftWorkspace(in floatFftWorkspace ws, int n, string who)
        {
            if (ws.n != n || ws.twRe.N != n >> 1 || ws.twIm.N != n >> 1)
                throw new ArgumentException(
                    who + ": workspace must be sized for an n-point FFT (use Arena.floatFftWorkspace(n))");
        }

        /// <summary>
        /// Throws if the workspace is missing the full-circle twiddle table required by
        /// fftRadix4 / ifftRadix4. Extends <see cref="RequireFftWorkspace"/>.
        /// </summary>
        static void RequireRadix4Workspace(in floatFftWorkspace ws, int n, string who)
        {
            RequireFftWorkspace(in ws, n, who);
            if (ws.twReFull.N != n || ws.twImFull.N != n)
                throw new ArgumentException(
                    who + ": workspace must have full-circle twiddle table (use Arena.floatFftWorkspace(n))");
        }

        // ---- table-indexed FFT core ----
        // Identical to FftCore (same bit-reversal, same inverse conjugate trick, same 1/N scale) EXCEPT
        // the butterfly twiddle W_len^k is read from the table as twRe[k*(tableN/len)],
        // twIm[k*(tableN/len)] — no per-stage cos/sin, no per-element recurrence.
        //
        // KEY INSIGHT: the size-tableN table T[j] = exp(-2πij/tableN) contains every twiddle any
        // radix-2 stage needs: W_len^k = exp(-2πik/len) = exp(-2πi·k·(tableN/len)/tableN) = T[k*(tableN/len)].
        // The index k*(tableN/len) is always an integer in [0, tableN/2) when len ≤ n ≤ tableN and k < len/2.
        //
        // re.N may be ≤ tableN (rfft passes the inner half-size M = tableN/2 as the data; the table
        // is full-N). re.N must divide tableN (both powers of two, enforced by workspace guard on callers).

        static void FftCoreTable(ref floatN re, ref floatN im,
                                 ref floatN twRe, ref floatN twIm,
                                 int tableN, bool inverse)
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
                    float tr = re[i]; re[i] = re[j]; re[j] = tr;
                    float ti = im[i]; im[i] = im[j]; im[j] = ti;
                }
            }

            // Butterfly stages: W_len^k = T[k*(tableN/len)], reading directly from the table.
            // For the inverse path the data is already conjugated, so the forward twiddle is correct.
            for (int len = 2; len <= n; len <<= 1)
            {
                int step = tableN / len;   // stride into the twiddle table for this stage
                int half = len >> 1;

                for (int i = 0; i < n; i += len)
                {
                    for (int k = 0; k < half; k++)
                    {
                        int a = i + k;
                        int b = a + half;

                        int twIdx = k * step;
                        float wr = twRe[twIdx];
                        float wi = twIm[twIdx];

                        float bRe = re[b];
                        float bIm = im[b];
                        float vRe = wr * bRe - wi * bIm;
                        float vIm = wr * bIm + wi * bRe;

                        float aRe = re[a];
                        float aIm = im[a];
                        re[a] = aRe + vRe; im[a] = aIm + vIm;
                        re[b] = aRe - vRe; im[b] = aIm - vIm;
                    }
                }
            }

            if (inverse)
            {
                float invN = (float)1 / (float)n;
                for (int i = 0; i < n; i++)
                {
                    re[i] = re[i] * invN;
                    im[i] = -im[i] * invN;   // undo the input conjugation, with the 1/N scale
                }
            }
        }

        // ---- table-indexed overloads ----

        /// <summary>
        /// In-place forward radix-2 FFT using a precomputed twiddle table. Eliminates per-element
        /// cos/sin from the hot path. ws must be sized for re.N (build via Arena.floatFftWorkspace(N)).
        /// Both arrays must have the same length, which must be a power of two.
        /// </summary>
        public static void fft(ref floatN re, ref floatN im, in floatFftWorkspace ws)
        {
            RequireFftWorkspace(in ws, re.N, "fft");
            var twRe = ws.twRe;   // copy the struct header (not the data — just a pointer + length)
            var twIm = ws.twIm;
            FftCoreTable(ref re, ref im, ref twRe, ref twIm, ws.n, false);
        }

        /// <summary>
        /// In-place inverse radix-2 FFT using a precomputed twiddle table. Divides by N, so
        /// ifft(fft(x, ws), ws) == x. ws must be sized for re.N.
        /// </summary>
        public static void ifft(ref floatN re, ref floatN im, in floatFftWorkspace ws)
        {
            RequireFftWorkspace(in ws, re.N, "ifft");
            var twRe = ws.twRe;
            var twIm = ws.twIm;
            FftCoreTable(ref re, ref im, ref twRe, ref twIm, ws.n, true);
        }

        /// <summary>
        /// Real-input forward FFT using a precomputed twiddle table. ws must be sized for real.N
        /// (the full real signal length N — not the half-spectrum length N/2+1).
        /// Identical two-for-one packing as the recurrence rfft, but the inner M-point FFT uses
        /// FftCoreTable and the unpack twiddle W_N^k = (ws.twRe[k], ws.twIm[k]) is read directly
        /// — no cos/sin in the hot loop.
        /// </summary>
        public static void rfft(in floatN real, ref floatN re, ref floatN im, in floatFftWorkspace ws)
        {
            int n = real.N;
            RequireFftWorkspace(in ws, n, "rfft");

            if (!IsPow2(n))
                throw new ArgumentException("rfft: length must be a power of two");

            int halfSpec = (n >> 1) + 1;
            if (re.N != halfSpec || im.N != halfSpec)
                throw new ArgumentException("rfft: re and im must have length N/2+1");

            unsafe
            {
                if (im.Data.Ptr == real.Data.Ptr || im.Data.Ptr == re.Data.Ptr)
                    throw new ArgumentException("rfft: im must not alias real or re");
            }

            if (n == 1)
            {
                re[0] = real[0];
                im[0] = (float)0;
                return;
            }

            int M = n >> 1;   // N/2

            // Step 1: Pack even and odd samples into a length-M complex sequence.
            var cz = new floatN(M, Allocator.Temp, false);
            var sz = new floatN(M, Allocator.Temp, false);
            for (int j = 0; j < M; j++)
            {
                cz[j] = real[2 * j];
                sz[j] = real[2 * j + 1];
            }

            // Step 2: Inner M-point FFT via the table. tableN = ws.n = N, data size M = N/2.
            // Stage-len twiddle index k*(N/len) — valid for all len in [2, M] and k < len/2.
            var twRe = ws.twRe;
            var twIm = ws.twIm;
            FftCoreTable(ref cz, ref sz, ref twRe, ref twIm, ws.n, false);

            // Step 3: Unpack. DC and Nyquist are always real for a real input.
            re[0] = cz[0] + sz[0];
            im[0] = (float)0;
            re[M] = cz[0] - sz[0];
            im[M] = (float)0;

            // General bins k = 1..M-1.
            // W_N^k = exp(-2πi·k/N) = (ws.twRe[k], ws.twIm[k]) — read directly, no cos/sin.
            for (int k = 1; k < M; k++)
            {
                int kr = M - k;
                float E_re = (cz[k] + cz[kr]) * (float)0.5;
                float E_im = (sz[k] - sz[kr]) * (float)0.5;
                float O_re = (sz[k] + sz[kr]) * (float)0.5;
                float O_im = (cz[kr] - cz[k]) * (float)0.5;

                float curRe = twRe[k];   // W_N^k real part
                float curIm = twIm[k];   // W_N^k imaginary part

                re[k] = E_re + (curRe * O_re - curIm * O_im);
                im[k] = E_im + (curRe * O_im + curIm * O_re);
            }

            cz.Dispose();
            sz.Dispose();
        }

        /// <summary>
        /// Inverse real FFT using a precomputed twiddle table. ws must be sized for the real signal
        /// length N = 2*(re.N-1) (= real.N). The re-pack conjugate twiddle conj(W_N^k) is read as
        /// (ws.twRe[k], -ws.twIm[k]) — no cos/sin in the hot loop.
        /// irfft(rfft(x, ws), ws) == x to floating-point precision.
        /// </summary>
        public static void irfft(in floatN re, in floatN im, ref floatN real, in floatFftWorkspace ws)
        {
            int halfSpec = re.N;
            if (im.N != halfSpec)
                throw new ArgumentException("irfft: re and im must have the same length");
            if (halfSpec < 2)
                throw new ArgumentException("irfft: re.N must be >= 2 (minimum signal length N=2)");

            int M = halfSpec - 1;   // N/2
            int N = M << 1;         // signal length

            if (!IsPow2(N))
                throw new ArgumentException("irfft: N = 2*(re.N-1) must be a power of two");
            if (real.N != N)
                throw new ArgumentException("irfft: real.N must equal 2*(re.N-1)");

            RequireFftWorkspace(in ws, N, "irfft");

            unsafe
            {
                if (real.Data.Ptr == re.Data.Ptr || real.Data.Ptr == im.Data.Ptr)
                    throw new ArgumentException("irfft: real must not alias re or im");
            }

            var cz = new floatN(M, Allocator.Temp, false);
            var sz = new floatN(M, Allocator.Temp, false);

            // k=0: X[0] = E[0]+O[0] and X[M] = E[0]-O[0] (both purely real).
            cz[0] = (re[0] + re[M]) * (float)0.5;
            sz[0] = (re[0] - re[M]) * (float)0.5;

            // General bins k = 1..M-1.
            // Same algebra as the recurrence irfft, but conj(W_N^k) = (twRe[k], -twIm[k]).
            //   E[k] = (X[k] + conj(X[M-k])) / 2
            //   O[k] = (X[k] - conj(X[M-k])) · conj(W_N^k) / 2
            //   Y[k] = E[k] + i·O[k]
            var twRe = ws.twRe;
            var twIm = ws.twIm;
            for (int k = 1; k < M; k++)
            {
                int kr = M - k;
                float xr_k  = re[k],  xi_k  = im[k];
                float xr_kr = re[kr], xi_kr = im[kr];

                // E[k] = (X[k] + conj(X[M-k])) / 2
                float E_re = (xr_k + xr_kr) * (float)0.5;
                float E_im = (xi_k - xi_kr) * (float)0.5;

                // (X[k] - conj(X[M-k])) = a + i·b
                float a = xr_k - xr_kr;
                float b = xi_k + xi_kr;

                // conj(W_N^k) = (twRe[k], -twIm[k]).
                // O[k] = (a + i·b) · conj(W_N^k) / 2  →  complex multiply (a+ib)·(cRe+icIm):
                //   real = a·cRe - b·cIm,  imag = b·cRe + a·cIm   (with cIm = -twIm[k])
                float cRe = twRe[k];
                float cIm = -twIm[k];   // negate imaginary part for conjugate
                float O_re = (a * cRe - b * cIm) * (float)0.5;
                float O_im = (b * cRe + a * cIm) * (float)0.5;

                // Y[k] = E[k] + i·O[k]
                cz[k] = E_re - O_im;
                sz[k] = E_im + O_re;
            }

            // One M-point inverse FFT recovers the interleaved even/odd real samples.
            FftCoreTable(ref cz, ref sz, ref twRe, ref twIm, ws.n, true);

            // Deinterleave: real[2j] = even[j], real[2j+1] = odd[j].
            for (int j = 0; j < M; j++)
            {
                real[2 * j]     = cz[j];
                real[2 * j + 1] = sz[j];
            }

            cz.Dispose();
            sz.Dispose();
        }

        // ---- radix-4 DIT FFT ----
        //
        // True radix-4 decimation-in-time for lengths that are exact powers of 4 (log4(n) even bits).
        // Radix-4 halves the number of full-array passes vs radix-2: log4(N) vs log2(N) stages,
        // trading twiddle-table memory (~2× for the full-circle table) against half the pass count.
        //
        // Inverse via conjugate trick: conjugate → forward radix-4 → conjugate + scale (1/N).
        // Permutation: base-4 digit reversal (reverse digit order in base-4 = swap bit-pairs).
        // Butterfly: forward DFT sign convention X[k] = Σ x[n]·exp(-2πi·kn/N).

        /// <summary>True if n is a positive power of 4 (n=1,4,16,64,…). The single set bit is at an even bit position.</summary>
        static bool IsPowerOf4(int n) =>
            (n > 0) && ((n & (n - 1)) == 0) && ((n & unchecked((int)0xAAAAAAAA)) == 0);

        // Base-4 digit reversal: reverse the log4n base-4 digits of x.
        // Equivalent to reversing bit-pairs (the 2 bits within each pair stay in order).
        static int ReverseBase4Digits(int x, int log4n)
        {
            int result = 0;
            for (int d = 0; d < log4n; d++)
            {
                result = (result << 2) | (x & 3);
                x >>= 2;
            }
            return result;
        }

        // Inner radix-4 butterfly pointer kernel.
        // Performs log4(n) stages of radix-4 DIT butterflies on already-permuted data.
        // twr/twi are the full-circle twiddle table of length n: T[m] = exp(-2πi·m/n).
        // All four pointer arguments are non-aliasing — [NoAlias] is truthful.
        [MethodImpl(MethodImplOptions.NoInlining)]
        static unsafe void FftCoreRadix4Ptr(
            [NoAlias] float* re, [NoAlias] float* im,
            [NoAlias] float* twr, [NoAlias] float* twi, int n)
        {
            // q = quarter-size per group (stride); starts at 1 and quadruples each stage.
            // sub-transform length = 4q;  step = n/(4q) into the full-circle table.
            for (int q = 1; q < n; q <<= 2)
            {
                int len  = q << 2;   // 4q
                int step = n / len;  // twiddle stride for this stage

                for (int base_ = 0; base_ < n; base_ += len)
                {
                    for (int j = 0; j < q; j++)
                    {
                        int i0 = base_ + j;
                        int i1 = i0 + q;
                        int i2 = i1 + q;
                        int i3 = i2 + q;

                        int tw1 = j * step;
                        int tw2 = tw1 + tw1;     // 2*j*step
                        int tw3 = tw2 + tw1;     // 3*j*step

                        // Load
                        float A_re = re[i0], A_im = im[i0];

                        // B = cmul(W^1, x[i1])
                        float w1r = twr[tw1], w1i = twi[tw1];
                        float B_re = w1r * re[i1] - w1i * im[i1];
                        float B_im = w1r * im[i1] + w1i * re[i1];

                        // C = cmul(W^2, x[i2])
                        float w2r = twr[tw2], w2i = twi[tw2];
                        float C_re = w2r * re[i2] - w2i * im[i2];
                        float C_im = w2r * im[i2] + w2i * re[i2];

                        // D = cmul(W^3, x[i3])
                        float w3r = twr[tw3], w3i = twi[tw3];
                        float D_re = w3r * re[i3] - w3i * im[i3];
                        float D_im = w3r * im[i3] + w3i * re[i3];

                        // 4-point DFT butterfly
                        float T0_re = A_re + C_re, T0_im = A_im + C_im;
                        float T1_re = A_re - C_re, T1_im = A_im - C_im;
                        float T2_re = B_re + D_re, T2_im = B_im + D_im;
                        float T3_re = B_re - D_re, T3_im = B_im - D_im;

                        // Forward sign convention: X[1] = T1 - i*T3, X[3] = T1 + i*T3
                        re[i0] = T0_re + T2_re; im[i0] = T0_im + T2_im;
                        re[i2] = T0_re - T2_re; im[i2] = T0_im - T2_im;
                        re[i1] = T1_re + T3_im; im[i1] = T1_im - T3_re;
                        re[i3] = T1_re - T3_im; im[i3] = T1_im + T3_re;
                    }
                }
            }
        }

        // Outer radix-4 DIT core: permutation + conjugate trick + pointer kernel + inverse scale.
        // n must be a power of 4 (caller guarantees via IsPowerOf4).
        static unsafe void FftCoreRadix4(ref floatN re, ref floatN im,
                                         ref floatN twReFull, ref floatN twImFull,
                                         int n, bool inverse)
        {
            if (n == 1) return;

            // Conjugate trick: conjugate input, run forward, conjugate + scale at the end.
            if (inverse)
                for (int i = 0; i < n; i++)
                    im[i] = -im[i];

            // Base-4 digit reversal permutation.
            int log4n = 0;
            for (int t = n; t > 1; t >>= 2) log4n++;

            for (int i = 0; i < n; i++)
            {
                int j = ReverseBase4Digits(i, log4n);
                if (j > i)
                {
                    float tr = re[i]; re[i] = re[j]; re[j] = tr;
                    float ti = im[i]; im[i] = im[j]; im[j] = ti;
                }
            }

            // Butterfly stages via pointer kernel.
            float* rePtr  = re.Data.Ptr;
            float* imPtr  = im.Data.Ptr;
            float* twrPtr = twReFull.Data.Ptr;
            float* twiPtr = twImFull.Data.Ptr;
            FftCoreRadix4Ptr(rePtr, imPtr, twrPtr, twiPtr, n);

            if (inverse)
            {
                float invN = (float)1 / (float)n;
                for (int i = 0; i < n; i++)
                {
                    re[i] =  re[i] * invN;
                    im[i] = -im[i] * invN;   // undo conjugate, apply 1/N scale
                }
            }
        }

        // Mixed-radix DIT for N = 2·4^k (IsPow2(N) && !IsPowerOf4(N)).
        // One radix-2 level wraps two size-M = N/2 radix-4 sub-FFTs; M is always a power of 4.
        // Inverse via conjugate trick at the OUTER level — sub-FFTs always run forward.
        //
        // Sub-table: T_M[k] = exp(-2πi·k/M) = exp(-2πi·2k/N) = T_N[2k], so we read every other
        // entry from the N-size full-circle table into a Temp M-size array and pass it to FftCoreRadix4.
        static unsafe void FftCoreRadix4Mixed(ref floatN re, ref floatN im,
                                              ref floatN twReFull, ref floatN twImFull,
                                              int n, bool inverse)
        {
            int M = n >> 1;   // N/2, always a power of 4

            // Conjugate trick at the outer level: negate im → forward decomposition → negate+scale.
            if (inverse)
                for (int i = 0; i < n; i++)
                    im[i] = -im[i];

            // Step 1: Deinterleave into even-indexed (E) and odd-indexed (O) halves.
            var ere = new floatN(M, Allocator.Temp, false);
            var eim = new floatN(M, Allocator.Temp, false);
            var ore = new floatN(M, Allocator.Temp, false);
            var oim = new floatN(M, Allocator.Temp, false);

            for (int k = 0; k < M; k++)
            {
                ere[k] = re[2 * k];
                eim[k] = im[2 * k];
                ore[k] = re[2 * k + 1];
                oim[k] = im[2 * k + 1];
            }

            // Step 2: Sub-FFT each half (forward) via the validated radix-4 core.
            // Build an M-size full-circle twiddle table from the N-size one: T_M[k] = T_N[2k].
            var twrM = new floatN(M, Allocator.Temp, false);
            var twiM = new floatN(M, Allocator.Temp, false);
            for (int k = 0; k < M; k++)
            {
                twrM[k] = twReFull[2 * k];
                twiM[k] = twImFull[2 * k];
            }

            FftCoreRadix4(ref ere, ref eim, ref twrM, ref twiM, M, false);
            FftCoreRadix4(ref ore, ref oim, ref twrM, ref twiM, M, false);

            twrM.Dispose();
            twiM.Dispose();

            // Step 3: Radix-2 DIT combine.
            // t = W_N^k * O[k],  W_N^k = T_N[k] = (twReFull[k], twImFull[k]).
            // X[k]   = E[k] + t
            // X[k+M] = E[k] - t
            for (int k = 0; k < M; k++)
            {
                float wr = twReFull[k];
                float wi = twImFull[k];
                float tr = wr * ore[k] - wi * oim[k];
                float ti = wr * oim[k] + wi * ore[k];
                re[k]     = ere[k] + tr;
                im[k]     = eim[k] + ti;
                re[k + M] = ere[k] - tr;
                im[k + M] = eim[k] - ti;
            }

            ere.Dispose();
            eim.Dispose();
            ore.Dispose();
            oim.Dispose();

            // Conjugate and scale for inverse.
            if (inverse)
            {
                float invN = (float)1 / (float)n;
                for (int i = 0; i < n; i++)
                {
                    re[i] =  re[i] * invN;
                    im[i] = -im[i] * invN;   // undo conjugate, apply 1/N scale
                }
            }
        }

        // ---- radix-4 dispatch overloads ----

        /// <summary>
        /// In-place forward FFT using the radix-4 DIT core for power-of-4 lengths, falling back to
        /// the radix-2 table core for other power-of-two lengths. ws must be an n-point workspace
        /// built via Arena.floatFftWorkspace(n) — it must contain the full-circle twiddle table
        /// (twReFull / twImFull) required by the radix-4 path.
        ///
        /// Radix-4 halves the number of full-array passes vs radix-2 (log4(N) vs log2(N) stages),
        /// at the cost of a 2× larger twiddle table. At large N the pass reduction is the goal;
        /// for small N the overhead of the table dominates.
        /// </summary>
        public static void fftRadix4(ref floatN re, ref floatN im, in floatFftWorkspace ws)
        {
            int n = re.N;
            RequireRadix4Workspace(in ws, n, "fftRadix4");

            if (IsPowerOf4(n))
            {
                var twReFull = ws.twReFull;
                var twImFull = ws.twImFull;
                FftCoreRadix4(ref re, ref im, ref twReFull, ref twImFull, n, false);
            }
            else if ((n & (n - 1)) == 0)   // power-of-2, not power-of-4 → 2·4^k mixed-radix path
            {
                var twReFull = ws.twReFull;
                var twImFull = ws.twImFull;
                FftCoreRadix4Mixed(ref re, ref im, ref twReFull, ref twImFull, n, false);
            }
            else
            {
                var twRe = ws.twRe;
                var twIm = ws.twIm;
                FftCoreTable(ref re, ref im, ref twRe, ref twIm, ws.n, false);
            }
        }

        /// <summary>
        /// In-place inverse FFT using the radix-4 DIT core for power-of-4 lengths, falling back to
        /// the radix-2 table core for other power-of-two lengths. Divides by N so that
        /// ifftRadix4(fftRadix4(x, ws), ws) == x. ws must be an n-point workspace.
        /// </summary>
        public static void ifftRadix4(ref floatN re, ref floatN im, in floatFftWorkspace ws)
        {
            int n = re.N;
            RequireRadix4Workspace(in ws, n, "ifftRadix4");

            if (IsPowerOf4(n))
            {
                var twReFull = ws.twReFull;
                var twImFull = ws.twImFull;
                FftCoreRadix4(ref re, ref im, ref twReFull, ref twImFull, n, true);
            }
            else if ((n & (n - 1)) == 0)   // power-of-2, not power-of-4 → 2·4^k mixed-radix path
            {
                var twReFull = ws.twReFull;
                var twImFull = ws.twImFull;
                FftCoreRadix4Mixed(ref re, ref im, ref twReFull, ref twImFull, n, true);
            }
            else
            {
                var twRe = ws.twRe;
                var twIm = ws.twIm;
                FftCoreTable(ref re, ref im, ref twRe, ref twIm, ws.n, true);
            }
        }
    }
}
