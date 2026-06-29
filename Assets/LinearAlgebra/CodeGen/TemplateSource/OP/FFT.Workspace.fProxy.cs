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
    /// Precomputed twiddle table for fProxyFFT. One size-n table serves every radix-2 transform of
    /// length ≤ n: the stage-len butterfly twiddle W_len^k is indexed as twRe[k*(n/len)],
    /// twIm[k*(n/len)], eliminating per-element cos/sin from the hot loop. Build once via
    /// Arena.fProxyFftWorkspace(n) and reuse across many transforms of the same size.
    ///
    /// The table is computed at double precision and cast to the element type (float or double),
    /// so accuracy is maximized regardless of the transform element type.
    ///
    /// The full-circle table (twReFull/twImFull, length n) extends the half-table for radix-4:
    /// radix-4 twiddles reach index 3n/4, past the half-table boundary. Bandwidth tradeoff:
    /// full table uses ~2× twiddle memory (~16 MB at N=1M for float), offset by halving the
    /// number of full-array passes (log4(N) vs log2(N) passes).
    /// </summary>
    public struct fProxyFftWorkspace
    {
        public fProxyN twRe;       // length n/2: cos(-2π·j/n), j = 0..n/2-1  (half circle, radix-2)
        public fProxyN twIm;       // length n/2: sin(-2π·j/n)
        public fProxyN twReFull;   // length n:   cos(-2π·m/n), m = 0..n-1    (full circle, radix-4)
        public fProxyN twImFull;   // length n:   sin(-2π·m/n)
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
        public fProxyFftWorkspace fProxyFftWorkspace(int n)
        {
            if (n < 2 || (n & (n - 1)) != 0)
                throw new ArgumentException("fProxyFftWorkspace: n must be a power of two and >= 2");

            int half = n >> 1;
            var twRe     = fProxyVec(half);
            var twIm     = fProxyVec(half);
            var twReFull = fProxyVec(n);
            var twImFull = fProxyVec(n);

            // Build at double precision for accuracy; direct per-entry cos/sin, no recurrence drift.
            double twoPiOverN = -2.0 * System.Math.PI / n;
            for (int j = 0; j < half; j++)
            {
                double ang = twoPiOverN * j;
                twRe[j] = (fProxy)math.cos(ang);
                twIm[j] = (fProxy)math.sin(ang);
            }

            // Full-circle table: same angle formula, extended to m = 0..n-1.
            for (int m = 0; m < n; m++)
            {
                double ang = twoPiOverN * m;
                twReFull[m] = (fProxy)math.cos(ang);
                twImFull[m] = (fProxy)math.sin(ang);
            }

            return new fProxyFftWorkspace
            {
                twRe     = twRe,
                twIm     = twIm,
                twReFull = twReFull,
                twImFull = twImFull,
                n        = n,
            };
        }
    }

    public static partial class fProxyFFT
    {
        // ---- workspace guard ----

        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an n-point FFT. Matches the layout
        /// produced by Arena.fProxyFftWorkspace(n): ws.n == n and table lengths == n/2.
        /// </summary>
        static void RequireFftWorkspace(in fProxyFftWorkspace ws, int n, string who)
        {
            if (ws.n != n || ws.twRe.N != n >> 1 || ws.twIm.N != n >> 1)
                throw new ArgumentException(
                    who + ": workspace must be sized for an n-point FFT (use Arena.fProxyFftWorkspace(n))");
        }

        /// <summary>
        /// Throws if the workspace is missing the full-circle twiddle table required by
        /// fftRadix4 / ifftRadix4. Extends <see cref="RequireFftWorkspace"/>.
        /// </summary>
        static void RequireRadix4Workspace(in fProxyFftWorkspace ws, int n, string who)
        {
            RequireFftWorkspace(in ws, n, who);
            if (ws.twReFull.N != n || ws.twImFull.N != n)
                throw new ArgumentException(
                    who + ": workspace must have full-circle twiddle table (use Arena.fProxyFftWorkspace(n))");
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

        static void FftCoreTable(ref fProxyN re, ref fProxyN im,
                                 ref fProxyN twRe, ref fProxyN twIm,
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
                    fProxy tr = re[i]; re[i] = re[j]; re[j] = tr;
                    fProxy ti = im[i]; im[i] = im[j]; im[j] = ti;
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
                        fProxy wr = twRe[twIdx];
                        fProxy wi = twIm[twIdx];

                        fProxy bRe = re[b];
                        fProxy bIm = im[b];
                        fProxy vRe = wr * bRe - wi * bIm;
                        fProxy vIm = wr * bIm + wi * bRe;

                        fProxy aRe = re[a];
                        fProxy aIm = im[a];
                        re[a] = aRe + vRe; im[a] = aIm + vIm;
                        re[b] = aRe - vRe; im[b] = aIm - vIm;
                    }
                }
            }

            if (inverse)
            {
                fProxy invN = (fProxy)1 / (fProxy)n;
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
        /// cos/sin from the hot path. ws must be sized for re.N (build via Arena.fProxyFftWorkspace(N)).
        /// Both arrays must have the same length, which must be a power of two.
        /// </summary>
        public static void fft(ref fProxyN re, ref fProxyN im, in fProxyFftWorkspace ws)
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
        public static void ifft(ref fProxyN re, ref fProxyN im, in fProxyFftWorkspace ws)
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
        public static void rfft(in fProxyN real, ref fProxyN re, ref fProxyN im, in fProxyFftWorkspace ws)
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
                im[0] = (fProxy)0;
                return;
            }

            int M = n >> 1;   // N/2

            // Step 1: Pack even and odd samples into a length-M complex sequence.
            var cz = new fProxyN(M, Allocator.Temp, false);
            var sz = new fProxyN(M, Allocator.Temp, false);
            for (int j = 0; j < M; j++)
            {
                cz[j] = real[2 * j];
                sz[j] = real[2 * j + 1];
            }

            // Step 2: Inner M-point FFT via radix-4 (with full-circle table, tableN = ws.n = N).
            // IsPowerOf4(M) → pure radix-4 (M = 4^k); else → mixed 2·4^k path (M = 2·4^k).
            // Both paths index into twReFull/twImFull at step ws.n/len — no sub-table copy.
            var twRe = ws.twRe;
            var twIm = ws.twIm;
            var twReFull = ws.twReFull;
            var twImFull = ws.twImFull;
            if (IsPowerOf4(M))
                FftCoreRadix4(ref cz, ref sz, ref twReFull, ref twImFull, ws.n, false);
            else
                FftCoreRadix4Mixed(ref cz, ref sz, ref twReFull, ref twImFull, ws.n, false);

            // Step 3: Unpack. DC and Nyquist are always real for a real input.
            re[0] = cz[0] + sz[0];
            im[0] = (fProxy)0;
            re[M] = cz[0] - sz[0];
            im[M] = (fProxy)0;

            // General bins k = 1..M-1.
            // W_N^k = exp(-2πi·k/N) = (ws.twRe[k], ws.twIm[k]) — read directly, no cos/sin.
            for (int k = 1; k < M; k++)
            {
                int kr = M - k;
                fProxy E_re = (cz[k] + cz[kr]) * (fProxy)0.5;
                fProxy E_im = (sz[k] - sz[kr]) * (fProxy)0.5;
                fProxy O_re = (sz[k] + sz[kr]) * (fProxy)0.5;
                fProxy O_im = (cz[kr] - cz[k]) * (fProxy)0.5;

                fProxy curRe = twRe[k];   // W_N^k real part (half-table)
                fProxy curIm = twIm[k];   // W_N^k imaginary part (half-table)

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
        public static void irfft(in fProxyN re, in fProxyN im, ref fProxyN real, in fProxyFftWorkspace ws)
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

            var cz = new fProxyN(M, Allocator.Temp, false);
            var sz = new fProxyN(M, Allocator.Temp, false);

            // k=0: X[0] = E[0]+O[0] and X[M] = E[0]-O[0] (both purely real).
            cz[0] = (re[0] + re[M]) * (fProxy)0.5;
            sz[0] = (re[0] - re[M]) * (fProxy)0.5;

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
                fProxy xr_k  = re[k],  xi_k  = im[k];
                fProxy xr_kr = re[kr], xi_kr = im[kr];

                // E[k] = (X[k] + conj(X[M-k])) / 2
                fProxy E_re = (xr_k + xr_kr) * (fProxy)0.5;
                fProxy E_im = (xi_k - xi_kr) * (fProxy)0.5;

                // (X[k] - conj(X[M-k])) = a + i·b
                fProxy a = xr_k - xr_kr;
                fProxy b = xi_k + xi_kr;

                // conj(W_N^k) = (twRe[k], -twIm[k]).
                // O[k] = (a + i·b) · conj(W_N^k) / 2  →  complex multiply (a+ib)·(cRe+icIm):
                //   real = a·cRe - b·cIm,  imag = b·cRe + a·cIm   (with cIm = -twIm[k])
                fProxy cRe = twRe[k];
                fProxy cIm = -twIm[k];   // negate imaginary part for conjugate
                fProxy O_re = (a * cRe - b * cIm) * (fProxy)0.5;
                fProxy O_im = (b * cRe + a * cIm) * (fProxy)0.5;

                // Y[k] = E[k] + i·O[k]
                cz[k] = E_re - O_im;
                sz[k] = E_im + O_re;
            }

            // One M-point inverse FFT via radix-4 (with full-circle table, tableN = ws.n = N).
            var twReFull = ws.twReFull;
            var twImFull = ws.twImFull;
            if (IsPowerOf4(M))
                FftCoreRadix4(ref cz, ref sz, ref twReFull, ref twImFull, ws.n, true);
            else
                FftCoreRadix4Mixed(ref cz, ref sz, ref twReFull, ref twImFull, ws.n, true);

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
        // twr/twi are the full-circle twiddle table of length tableN: T[m] = exp(-2πi·m/tableN).
        // n is the DATA/transform size; tableN is the TABLE size (tableN >= n, both pow-of-4,
        // n divides tableN). Stage-len twiddle W_len^j = T_tableN[j*(tableN/len)] so a single
        // full-size table drives any sub-transform without sub-table copies.
        // All four pointer arguments are non-aliasing — [NoAlias] is truthful.
        [MethodImpl(MethodImplOptions.NoInlining)]
        static unsafe void FftCoreRadix4Ptr(
            [NoAlias] fProxy* re, [NoAlias] fProxy* im,
            [NoAlias] fProxy* twr, [NoAlias] fProxy* twi, int n, int tableN)
        {
            // q = quarter-size per group (stride); starts at 1 and quadruples each stage.
            // sub-transform length = 4q;  step = tableN/(4q) into the full-circle table.
            for (int q = 1; q < n; q <<= 2)
            {
                int len  = q << 2;      // 4q
                int step = tableN / len; // twiddle stride: W_len^j = T_tableN[j*step]

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
                        fProxy A_re = re[i0], A_im = im[i0];

                        // B = cmul(W^1, x[i1])
                        fProxy w1r = twr[tw1], w1i = twi[tw1];
                        fProxy B_re = w1r * re[i1] - w1i * im[i1];
                        fProxy B_im = w1r * im[i1] + w1i * re[i1];

                        // C = cmul(W^2, x[i2])
                        fProxy w2r = twr[tw2], w2i = twi[tw2];
                        fProxy C_re = w2r * re[i2] - w2i * im[i2];
                        fProxy C_im = w2r * im[i2] + w2i * re[i2];

                        // D = cmul(W^3, x[i3])
                        fProxy w3r = twr[tw3], w3i = twi[tw3];
                        fProxy D_re = w3r * re[i3] - w3i * im[i3];
                        fProxy D_im = w3r * im[i3] + w3i * re[i3];

                        // 4-point DFT butterfly
                        fProxy T0_re = A_re + C_re, T0_im = A_im + C_im;
                        fProxy T1_re = A_re - C_re, T1_im = A_im - C_im;
                        fProxy T2_re = B_re + D_re, T2_im = B_im + D_im;
                        fProxy T3_re = B_re - D_re, T3_im = B_im - D_im;

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
        // Transform size is re.N (must be a power of 4, caller guarantees).
        // tableN is the length of the twiddle table (must be a multiple of re.N; == re.N for top-level
        // callers, > re.N when called from FftCoreRadix4Mixed or rfft to drive a sub-transform).
        static unsafe void FftCoreRadix4(ref fProxyN re, ref fProxyN im,
                                         ref fProxyN twReFull, ref fProxyN twImFull,
                                         int tableN, bool inverse)
        {
            int size = re.N;
            if (size == 1) return;

            // Conjugate trick: conjugate input, run forward, conjugate + scale at the end.
            if (inverse)
                for (int i = 0; i < size; i++)
                    im[i] = -im[i];

            // Base-4 digit reversal permutation.
            int log4n = 0;
            for (int t = size; t > 1; t >>= 2) log4n++;

            for (int i = 0; i < size; i++)
            {
                int j = ReverseBase4Digits(i, log4n);
                if (j > i)
                {
                    fProxy tr = re[i]; re[i] = re[j]; re[j] = tr;
                    fProxy ti = im[i]; im[i] = im[j]; im[j] = ti;
                }
            }

            // Butterfly stages via pointer kernel.
            fProxy* rePtr  = re.Data.Ptr;
            fProxy* imPtr  = im.Data.Ptr;
            fProxy* twrPtr = twReFull.Data.Ptr;
            fProxy* twiPtr = twImFull.Data.Ptr;
            FftCoreRadix4Ptr(rePtr, imPtr, twrPtr, twiPtr, size, tableN);

            if (inverse)
            {
                fProxy invN = (fProxy)1 / (fProxy)size;
                for (int i = 0; i < size; i++)
                {
                    re[i] =  re[i] * invN;
                    im[i] = -im[i] * invN;   // undo conjugate, apply 1/N scale
                }
            }
        }

        // Mixed-radix DIT for N = 2·4^k (IsPow2(N) && !IsPowerOf4(N)).
        // Transform size is re.N; tableN is the twiddle table length (>= re.N, multiple of re.N).
        // One radix-2 level wraps two size-M = re.N/2 radix-4 sub-FFTs; M is always a power of 4.
        // Inverse via conjugate trick at the OUTER level — sub-FFTs always run forward.
        //
        // Decoupled twiddle indexing: W_M^j = T_tableN[j*(tableN/M)], computed by FftCoreRadix4Ptr
        // with step = tableN/len — no sub-table copy needed (the Temp twrM/twiM are eliminated).
        // Combine twiddle: W_size^k = T_tableN[k*(tableN/size)].
        static unsafe void FftCoreRadix4Mixed(ref fProxyN re, ref fProxyN im,
                                              ref fProxyN twReFull, ref fProxyN twImFull,
                                              int tableN, bool inverse)
        {
            int size = re.N;
            int M = size >> 1;   // size/2, always a power of 4

            // Conjugate trick at the outer level: negate im → forward decomposition → negate+scale.
            if (inverse)
                for (int i = 0; i < size; i++)
                    im[i] = -im[i];

            // Step 1: Deinterleave into even-indexed (E) and odd-indexed (O) halves.
            var ere = new fProxyN(M, Allocator.Temp, false);
            var eim = new fProxyN(M, Allocator.Temp, false);
            var ore = new fProxyN(M, Allocator.Temp, false);
            var oim = new fProxyN(M, Allocator.Temp, false);

            for (int k = 0; k < M; k++)
            {
                ere[k] = re[2 * k];
                eim[k] = im[2 * k];
                ore[k] = re[2 * k + 1];
                oim[k] = im[2 * k + 1];
            }

            // Step 2: Sub-FFT each half (forward) via the radix-4 core.
            // Pass the full tableN-size table; FftCoreRadix4Ptr computes step = tableN/len so
            // T_M[j] = T_tableN[j*(tableN/M)] is read correctly — no sub-table copy required.
            FftCoreRadix4(ref ere, ref eim, ref twReFull, ref twImFull, tableN, false);
            FftCoreRadix4(ref ore, ref oim, ref twReFull, ref twImFull, tableN, false);

            // Step 3: Radix-2 DIT combine.
            // W_size^k = T_tableN[k*(tableN/size)].
            // X[k]   = E[k] + W_size^k * O[k]
            // X[k+M] = E[k] - W_size^k * O[k]
            int combineStep = tableN / size;
            for (int k = 0; k < M; k++)
            {
                fProxy wr = twReFull[k * combineStep];
                fProxy wi = twImFull[k * combineStep];
                fProxy tr = wr * ore[k] - wi * oim[k];
                fProxy ti = wr * oim[k] + wi * ore[k];
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
                fProxy invN = (fProxy)1 / (fProxy)size;
                for (int i = 0; i < size; i++)
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
        /// built via Arena.fProxyFftWorkspace(n) — it must contain the full-circle twiddle table
        /// (twReFull / twImFull) required by the radix-4 path.
        ///
        /// Radix-4 halves the number of full-array passes vs radix-2 (log4(N) vs log2(N) stages),
        /// at the cost of a 2× larger twiddle table. At large N the pass reduction is the goal;
        /// for small N the overhead of the table dominates.
        /// </summary>
        public static void fftRadix4(ref fProxyN re, ref fProxyN im, in fProxyFftWorkspace ws)
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
        public static void ifftRadix4(ref fProxyN re, ref fProxyN im, in fProxyFftWorkspace ws)
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
