using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra.Internal;   // fProxyW (8-lane AVX helper) for the wide radix-4 butterfly

namespace LinearAlgebra
{
    /// <summary>
    /// Precomputed twiddle table for FFT. One size-n table serves every radix-2 transform of
    /// length ≤ n: the stage-len butterfly twiddle W_len^k is indexed as twRe[k*(n/len)],
    /// twIm[k*(n/len)], eliminating per-element cos/sin from the hot loop. Build once via
    /// Arena.fProxyFFTCache(n) and reuse across many transforms of the same size.
    ///
    /// The table is computed at double precision and cast to the element type (float or double),
    /// so accuracy is maximized regardless of the transform element type.
    ///
    /// The full-circle table (twReFull/twImFull, length n) is required by the auto-dispatch
    /// radix-4 paths inside fft/ifft: radix-4 twiddles reach index 3n/4, past the half-table
    /// boundary.
    /// </summary>
    public struct fProxyFFTCache
    {
        public fProxyN twRe;       // length n/2: cos(-2π·j/n), j = 0..n/2-1  (half circle, radix-2)
        public fProxyN twIm;       // length n/2: sin(-2π·j/n)
        public fProxyN twReFull;   // length n:   cos(-2π·m/n), m = 0..n-1    (full circle, radix-4)
        public fProxyN twImFull;   // length n:   sin(-2π·m/n)
        public int n;              // the FFT size this table is built for (must be a power of two, >= 2)

        // Scratch buffers — allocated once in the factory, reused on every call.
        // Single-use-at-a-time: one workspace per concurrent transform (FFTW-plan semantics).
        public fProxyN cz;         // length n/2: even-sample packing scratch for rfft/irfft
        public fProxyN sz;         // length n/2: odd-sample  packing scratch for rfft/irfft
        public fProxyN visited;    // length n:   cycle-following scratch for FftCoreRadix4Mixed
                                   //             (stores 0/1 flags via fProxy; [0,size) used per call)

        // Contiguous per-stage W^1 twiddle table for the wide (fProxyW) radix-4 butterfly: stages
        // with quarter-stride q >= fProxyW.Width (8 float / 4 double lanes), concatenated in stage
        // order (total length swLen). W^2 and W^3 are derived from W^1 in-register (W^2=W^1·W^1,
        // W^3=W^1·W^2), so only W^1 is stored. Built only for a power-of-4 n; empty otherwise.
        public fProxyN sw1re, sw1im;
        public int swLen;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates a twiddle-table FFT workspace for an n-point transform (n must be a power of two,
        /// n ≥ 2). Entries are computed via direct per-entry cos/sin at double precision (no recurrence
        /// drift); see <see cref="fProxyFFTCache"/> for the table layout and full-circle-table rationale.
        ///
        /// The buffers are persistent in this arena (disposed with it), so create the workspace once
        /// outside a hot loop and pass it to the table overloads. One table serves fft/ifft of length
        /// exactly n and rfft/irfft of real signal length exactly n.
        /// </summary>
        public static unsafe fProxyFFTCache fProxyFFTCache(this ref Arena arena, int n)
        {
            if (n < 2 || (n & (n - 1)) != 0)
                throw new ArgumentException("fProxyFFTCache: n must be a power of two and >= 2");

            int half = n >> 1;
            var twRe     = arena.fProxyVec(half);
            var twIm     = arena.fProxyVec(half);
            var twReFull = arena.fProxyVec(n);
            var twImFull = arena.fProxyVec(n);

            // Twiddle table = Nth roots of unity W_N^m = exp(-2πi·m/n), generated at double precision
            // with only +,-,*,sqrt (cross-arch deterministic under FloatMode.Strict; no sin/cos). The
            // binary generator roots B_k = exp(-2πi·2^k/n) come from stable unit-circle half-angle
            // square roots; each W_N^m is the product of B_k over m's set bits, then cast to fProxy.
            int P = 0;
            for (int t = n; t > 1; t >>= 1) P++;   // log2(n)
            double* bkr = stackalloc double[32];
            double* bki = stackalloc double[32];
            bkr[P - 1] = -1.0; bki[P - 1] = 0.0;                    // B_{P-1} = exp(-πi)
            if (P >= 2) { bkr[P - 2] = 0.0; bki[P - 2] = -1.0; }    // B_{P-2} = exp(-πi/2)
            for (int k = P - 3; k >= 0; k--)
            {
                double a = bkr[k + 1];                    // cos(angle_{k+1})
                double c = math.sqrt((1.0 + a) * 0.5);    // cos(angle_k), angle_k = angle_{k+1}/2
                bkr[k] = c;
                bki[k] = bki[k + 1] / (2.0 * c);          // -sin(angle_k), cancellation-free
            }
            for (int m = 0; m < n; m++)
            {
                double cr = 1.0, ci = 0.0;                // W^0
                int mm = m, k = 0;
                while (mm != 0)
                {
                    if ((mm & 1) != 0)
                    {
                        double nr = cr * bkr[k] - ci * bki[k];
                        ci = cr * bki[k] + ci * bkr[k];
                        cr = nr;
                    }
                    mm >>= 1; k++;
                }
                twReFull[m] = (fProxy)cr;
                twImFull[m] = (fProxy)ci;
                if (m < half) { twRe[m] = (fProxy)cr; twIm[m] = (fProxy)ci; }
            }

            // Scratch buffers — persistent in this arena (disposed with the arena).
            // cz/sz are the two-for-one packing temporaries for rfft/irfft (length n/2 = M).
            // visited is the cycle-following scratch for FftCoreRadix4Mixed (length n; [0,M) used
            // when called from the rfft/irfft inner M-point sub-FFT, still within bounds).
            var cz      = arena.fProxyVec(half, uninit: true);
            var sz      = arena.fProxyVec(half, uninit: true);
            var visited = arena.fProxyVec(n,    uninit: true);

            // Wide-butterfly stage twiddles: only for a power-of-4 n (the wide-dispatched path).
            // n is already a power of two here, so power-of-4 == no odd bit-pair set.
            bool pow4 = (n & unchecked((int)0xAAAAAAAA)) == 0;
            int swLen = 0;
            if (pow4)
                for (int qq = 1; qq < n; qq <<= 2)
                    if (qq >= fProxyW.Width) swLen += qq;
            int swAlloc = swLen > 0 ? swLen : 1;
            var sw1re = arena.fProxyVec(swAlloc, uninit: true);
            var sw1im = arena.fProxyVec(swAlloc, uninit: true);
            if (pow4)
            {
                int off = 0;
                for (int qq = 1; qq < n; qq <<= 2)
                {
                    if (qq < fProxyW.Width) continue;
                    int len  = qq << 2;
                    int step = n / len;
                    for (int j = 0; j < qq; j++)
                    {
                        int t1 = j * step;
                        sw1re[off + j] = twReFull[t1]; sw1im[off + j] = twImFull[t1];
                    }
                    off += qq;
                }
            }

            return new fProxyFFTCache
            {
                twRe     = twRe,
                twIm     = twIm,
                twReFull = twReFull,
                twImFull = twImFull,
                n        = n,
                cz       = cz,
                sz       = sz,
                visited  = visited,
                sw1re = sw1re, sw1im = sw1im,
                swLen = swLen,
            };
        }
    }

    public static partial class FFT
    {
        // ---- workspace guard ----

        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an n-point FFT. Matches the layout
        /// produced by Arena.fProxyFFTCache(n): ws.n == n and table lengths == n/2.
        /// </summary>
        static void RequireFftWorkspace(in fProxyFFTCache ws, int n, string who)
        {
            if (ws.n != n || ws.twRe.N != n >> 1 || ws.twIm.N != n >> 1 ||
                ws.cz.N != n >> 1 || ws.sz.N != n >> 1 || ws.visited.N != n)
                throw new ArgumentException(
                    who + ": workspace must be sized for an n-point FFT (use Arena.fProxyFFTCache(n))");
        }

        /// <summary>
        /// Throws if the workspace is missing the full-circle twiddle table required by the
        /// radix-4 dispatch paths. Extends <see cref="RequireFftWorkspace"/>.
        /// </summary>
        static void RequireRadix4Workspace(in fProxyFFTCache ws, int n, string who)
        {
            RequireFftWorkspace(in ws, n, who);
            if (ws.twReFull.N != n || ws.twImFull.N != n)
                throw new ArgumentException(
                    who + ": workspace must have full-circle twiddle table (use Arena.fProxyFFTCache(n))");
        }

        // ---- table-indexed overloads ----

        /// <summary>
        /// In-place forward FFT for any power-of-two length, using a precomputed twiddle table;
        /// throws for a non-power-of-two length (use dft for arbitrary N). ws must be sized for re.N
        /// (build via Arena.fProxyFFTCache(N)); it must contain the full-circle twiddle table required
        /// by the radix-4 dispatch. Both arrays must have the same length, which must be a power of two.
        /// </summary>
        public static void fft(ref fProxyN re, ref fProxyN im, in fProxyFFTCache ws)
        {
            int n = re.N;
            RequireRadix4Workspace(in ws, n, "fft");

            if (IsPowerOf4(n))
            {
                FftCoreRadix4Wide(ref re, ref im, in ws, false);
            }
            else if ((n & (n - 1)) == 0)   // power-of-2, not power-of-4 → 2·4^k mixed-radix path
            {
                var twReFull    = ws.twReFull;
                var twImFull    = ws.twImFull;
                var visitedScratch = ws.visited;
                FftCoreRadix4Mixed(ref re, ref im, ref twReFull, ref twImFull, visitedScratch, n, false);
            }
            else
            {
                // radix-4 ∪ mixed cover every power of two, so this is reached only by a
                // non-power-of-two length, which Cooley-Tukey cannot handle.
                throw new ArgumentException("fft: length must be a power of two (use dft for arbitrary N)");
            }
        }

        /// <summary>
        /// In-place inverse FFT, same dispatch as fft. Divides by N so that ifft(fft(x, ws), ws) == x.
        /// ws must be sized for re.N.
        /// </summary>
        public static void ifft(ref fProxyN re, ref fProxyN im, in fProxyFFTCache ws)
        {
            int n = re.N;
            RequireRadix4Workspace(in ws, n, "ifft");

            if (IsPowerOf4(n))
            {
                FftCoreRadix4Wide(ref re, ref im, in ws, true);
            }
            else if ((n & (n - 1)) == 0)   // power-of-2, not power-of-4 → 2·4^k mixed-radix path
            {
                var twReFull    = ws.twReFull;
                var twImFull    = ws.twImFull;
                var visitedScratch = ws.visited;
                FftCoreRadix4Mixed(ref re, ref im, ref twReFull, ref twImFull, visitedScratch, n, true);
            }
            else
            {
                // radix-4 ∪ mixed cover every power of two, so this is reached only by a
                // non-power-of-two length, which Cooley-Tukey cannot handle.
                throw new ArgumentException("ifft: length must be a power of two (use idft for arbitrary N)");
            }
        }

        /// <summary>
        /// Real-input forward FFT using a precomputed twiddle table. ws must be sized for real.N
        /// (the full real signal length N — not the half-spectrum length N/2+1).
        /// Identical two-for-one packing as the recurrence rfft, but the inner M-point FFT uses
        /// the radix-4/mixed dispatch and the unpack twiddle W_N^k = (ws.twRe[k], ws.twIm[k]) is
        /// read directly — no cos/sin in the hot loop.
        /// </summary>
        public static void rfft(in fProxyN real, ref fProxyN re, ref fProxyN im, in fProxyFFTCache ws)
        {
            int n = real.N;
            RequireRadix4Workspace(in ws, n, "rfft");

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
            // Use workspace scratch (no per-call allocation).
            var cz = ws.cz;   // fProxyN of length n/2 = M
            var sz = ws.sz;   // fProxyN of length n/2 = M
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
            var visitedScratch = ws.visited;
            if (IsPowerOf4(M))
                FftCoreRadix4(ref cz, ref sz, ref twReFull, ref twImFull, ws.n, false);
            else
                FftCoreRadix4Mixed(ref cz, ref sz, ref twReFull, ref twImFull, visitedScratch, ws.n, false);

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

        }

        /// <summary>
        /// Inverse real FFT using a precomputed twiddle table. ws must be sized for the real signal
        /// length N = 2*(re.N-1) (= real.N). The re-pack conjugate twiddle conj(W_N^k) is read as
        /// (ws.twRe[k], -ws.twIm[k]) — no cos/sin in the hot loop.
        /// irfft(rfft(x, ws), ws) == x to floating-point precision.
        /// </summary>
        public static void irfft(in fProxyN re, in fProxyN im, ref fProxyN real, in fProxyFFTCache ws)
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

            RequireRadix4Workspace(in ws, N, "irfft");

            unsafe
            {
                if (real.Data.Ptr == re.Data.Ptr || real.Data.Ptr == im.Data.Ptr)
                    throw new ArgumentException("irfft: real must not alias re or im");
            }

            // Use workspace scratch (no per-call allocation).
            var cz = ws.cz;   // fProxyN of length n/2 = M
            var sz = ws.sz;   // fProxyN of length n/2 = M

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
            var visitedScratch = ws.visited;
            if (IsPowerOf4(M))
                FftCoreRadix4(ref cz, ref sz, ref twReFull, ref twImFull, ws.n, true);
            else
                FftCoreRadix4Mixed(ref cz, ref sz, ref twReFull, ref twImFull, visitedScratch, ws.n, true);

            // Deinterleave: real[2j] = even[j], real[2j+1] = odd[j].
            for (int j = 0; j < M; j++)
            {
                real[2 * j]     = cz[j];
                real[2 * j + 1] = sz[j];
            }
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

        // IsPowerOf4 and ReverseBase4Digits live in OpHelpers.Shared.cs (type-agnostic, emitted once).

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

        // Wide (fProxyW, 8 float / 4 double lanes) radix-4 DIT for a top-level power-of-4 transform
        // whose size equals the workspace size (ws.n). Digit-reversal, then per-stage butterflies:
        // stages with quarter-stride q >= fProxyW.Width vectorize across j (Width consecutive j give
        // contiguous wide re/im loads and reads of the precomputed contiguous stage twiddles
        // ws.sw*), stages with q < Width run scalar from the full-circle table. Every lane performs
        // the exact scalar butterfly with the same tabulated twiddles, so output matches
        // FftCoreRadix4 to the last bit per element.
        [MethodImpl(MethodImplOptions.NoInlining)]
        static unsafe void FftCoreRadix4Wide(ref fProxyN re, ref fProxyN im, in fProxyFFTCache ws, bool inverse)
        {
            int n = re.N;
            if (n == 1) return;

            fProxy* rp = re.Data.Ptr;
            fProxy* ip = im.Data.Ptr;

            // Conjugate trick for the inverse.
            if (inverse)
                for (int i = 0; i < n; i++) ip[i] = -ip[i];

            // Base-4 digit reversal.
            int log4n = 0;
            for (int t = n; t > 1; t >>= 2) log4n++;
            for (int i = 0; i < n; i++)
            {
                int j = ReverseBase4Digits(i, log4n);
                if (j > i)
                {
                    fProxy tr = rp[i]; rp[i] = rp[j]; rp[j] = tr;
                    fProxy ti = ip[i]; ip[i] = ip[j]; ip[j] = ti;
                }
            }

            fProxy* twr = ws.twReFull.Data.Ptr;
            fProxy* twi = ws.twImFull.Data.Ptr;
            fProxy* s1r = ws.sw1re.Data.Ptr; fProxy* s1i = ws.sw1im.Data.Ptr;

            int W = fProxyW.Width;
            int stageOff = 0;

            for (int q = 1; q < n; q <<= 2)
            {
                int len  = q << 2;
                int step = n / len;   // tableN == n at the top level

                if (q >= W)
                {
                    // Wide stage: q (a power of 4, >= W) is a multiple of W, so no j tail.
                    for (int base_ = 0; base_ < n; base_ += len)
                    {
                        for (int j = 0; j < q; j += W)
                        {
                            int i0 = base_ + j, i1 = i0 + q, i2 = i1 + q, i3 = i2 + q;

                            fProxyW Are = fProxyW.Load(rp + i0, 0), Aim = fProxyW.Load(ip + i0, 0);

                            // W^1 loaded; W^2 = W^1·W^1, W^3 = W^1·W^2 derived in-register.
                            fProxyW w1r = fProxyW.Load(s1r + stageOff + j, 0), w1i = fProxyW.Load(s1i + stageOff + j, 0);
                            fProxyW w2r = w1r * w1r - w1i * w1i, w2i = w1r * w1i + w1i * w1r;
                            fProxyW w3r = w1r * w2r - w1i * w2i, w3i = w1r * w2i + w1i * w2r;

                            fProxyW x1r = fProxyW.Load(rp + i1, 0), x1i = fProxyW.Load(ip + i1, 0);
                            fProxyW Bre = w1r * x1r - w1i * x1i;
                            fProxyW Bim = w1r * x1i + w1i * x1r;

                            fProxyW x2r = fProxyW.Load(rp + i2, 0), x2i = fProxyW.Load(ip + i2, 0);
                            fProxyW Cre = w2r * x2r - w2i * x2i;
                            fProxyW Cim = w2r * x2i + w2i * x2r;

                            fProxyW x3r = fProxyW.Load(rp + i3, 0), x3i = fProxyW.Load(ip + i3, 0);
                            fProxyW Dre = w3r * x3r - w3i * x3i;
                            fProxyW Dim = w3r * x3i + w3i * x3r;

                            fProxyW T0re = Are + Cre, T0im = Aim + Cim;
                            fProxyW T1re = Are - Cre, T1im = Aim - Cim;
                            fProxyW T2re = Bre + Dre, T2im = Bim + Dim;
                            fProxyW T3re = Bre - Dre, T3im = Bim - Dim;

                            fProxyW.Store(rp + i0, 0, T0re + T2re); fProxyW.Store(ip + i0, 0, T0im + T2im);
                            fProxyW.Store(rp + i2, 0, T0re - T2re); fProxyW.Store(ip + i2, 0, T0im - T2im);
                            fProxyW.Store(rp + i1, 0, T1re + T3im); fProxyW.Store(ip + i1, 0, T1im - T3re);
                            fProxyW.Store(rp + i3, 0, T1re - T3im); fProxyW.Store(ip + i3, 0, T1im + T3re);
                        }
                    }
                    stageOff += q;
                }
                else
                {
                    // Small stage (q < Width): scalar, reading the full-circle table directly.
                    for (int base_ = 0; base_ < n; base_ += len)
                    {
                        for (int j = 0; j < q; j++)
                        {
                            int i0 = base_ + j, i1 = i0 + q, i2 = i1 + q, i3 = i2 + q;
                            int tw1 = j * step, tw2 = tw1 + tw1, tw3 = tw2 + tw1;

                            fProxy A_re = rp[i0], A_im = ip[i0];
                            fProxy w1r = twr[tw1], w1i = twi[tw1];
                            fProxy B_re = w1r * rp[i1] - w1i * ip[i1];
                            fProxy B_im = w1r * ip[i1] + w1i * rp[i1];
                            fProxy w2r = twr[tw2], w2i = twi[tw2];
                            fProxy C_re = w2r * rp[i2] - w2i * ip[i2];
                            fProxy C_im = w2r * ip[i2] + w2i * rp[i2];
                            fProxy w3r = twr[tw3], w3i = twi[tw3];
                            fProxy D_re = w3r * rp[i3] - w3i * ip[i3];
                            fProxy D_im = w3r * ip[i3] + w3i * rp[i3];

                            fProxy T0_re = A_re + C_re, T0_im = A_im + C_im;
                            fProxy T1_re = A_re - C_re, T1_im = A_im - C_im;
                            fProxy T2_re = B_re + D_re, T2_im = B_im + D_im;
                            fProxy T3_re = B_re - D_re, T3_im = B_im - D_im;

                            rp[i0] = T0_re + T2_re; ip[i0] = T0_im + T2_im;
                            rp[i2] = T0_re - T2_re; ip[i2] = T0_im - T2_im;
                            rp[i1] = T1_re + T3_im; ip[i1] = T1_im - T3_re;
                            rp[i3] = T1_re - T3_im; ip[i3] = T1_im + T3_re;
                        }
                    }
                }
            }

            // Undo conjugate and apply 1/N for the inverse.
            if (inverse)
            {
                fProxy invN = (fProxy)1 / (fProxy)n;
                for (int i = 0; i < n; i++)
                {
                    rp[i] =  rp[i] * invN;
                    ip[i] = -ip[i] * invN;
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

            // Conjugate trick (see section banner above).
            if (inverse)
                for (int i = 0; i < size; i++)
                    im[i] = -im[i];

            // Digit reversal + butterfly stages via shared slice helper.
            fProxy* rePtr  = re.Data.Ptr;
            fProxy* imPtr  = im.Data.Ptr;
            fProxy* twrPtr = twReFull.Data.Ptr;
            fProxy* twiPtr = twImFull.Data.Ptr;
            FftCoreRadix4Slice(rePtr, imPtr, twrPtr, twiPtr, size, tableN);

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

        // Shared helper: base-4 digit reversal + butterfly on a raw pointer slice of length `size`.
        // re/im are already offset to the start of the sub-array; twr/twi are the full-circle
        // twiddle table (length tableN, shared across calls — read-only inside FftCoreRadix4Ptr).
        // size must be a power of 4; tableN must be a multiple of size.
        static unsafe void FftCoreRadix4Slice(
            fProxy* re, fProxy* im,
            fProxy* twr, fProxy* twi,
            int size, int tableN)
        {
            if (size <= 1) return;

            // Base-4 digit reversal on this slice.
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

            // Butterfly stages.
            FftCoreRadix4Ptr(re, im, twr, twi, size, tableN);
        }

        // Mixed-radix DIT for N = 2·4^k (IsPow2(N) && !IsPowerOf4(N)).
        // Transform size is re.N; tableN is the twiddle table length (>= re.N, multiple of re.N).
        // One radix-2 level wraps two size-M = re.N/2 radix-4 sub-FFTs; M is always a power of 4.
        // Inverse via conjugate trick at the OUTER level — sub-FFTs always run forward.
        //
        // In-place implementation — no fProxyN Temp allocations.
        // Step 1: de-interleave in-place via cycle-following (bool visited scratch, N bytes).
        // Step 2: two radix-4 sub-FFTs on the contiguous even/odd halves via FftCoreRadix4Slice.
        // Step 3: radix-2 DIT combine, writing natural-order output back in-place.
        // Combine twiddle: W_size^k = T_tableN[k*(tableN/size)].
        // visited is a workspace scratch of length >= size (0/1 flags, fProxy-typed).
        // The caller passes ws.visited (length n); only [0, size) is used, cleared at entry.
        static unsafe void FftCoreRadix4Mixed(ref fProxyN re, ref fProxyN im,
                                              ref fProxyN twReFull, ref fProxyN twImFull,
                                              fProxyN visited,
                                              int tableN, bool inverse)
        {
            int size = re.N;
            int M = size >> 1;   // size/2, always a power of 4

            // Conjugate trick at the outer level (see banner above).
            if (inverse)
                for (int i = 0; i < size; i++)
                    im[i] = -im[i];

            // Step 1: In-place de-interleave (unshuffle) via cycle-following.
            // dst(i) = i/2 if i is even; M + i/2 if i is odd.
            // After the permutation: [0,M) holds even-indexed elements, [M,2M) holds odd-indexed,
            // both in natural order — exactly what FftCoreRadix4Slice expects as input.
            // Cycle-following uses the workspace visited scratch (0 = unvisited, 1 = visited).
            // Clear [0, size) — workspace is reused across calls so it must be zeroed each time.
            for (int i = 0; i < size; i++)
                visited[i] = (fProxy)0;

            for (int s = 0; s < size; s++)
            {
                if (visited[s] != (fProxy)0) continue;
                visited[s] = (fProxy)1;

                int j = (s & 1) == 0 ? (s >> 1) : (M + (s >> 1));
                if (j == s) continue;   // fixed point (s==0 or s==size-1)

                fProxy carryRe = re[s], carryIm = im[s];
                while (j != s)
                {
                    fProxy tmpRe = re[j], tmpIm = im[j];
                    re[j] = carryRe;
                    im[j] = carryIm;
                    visited[j] = (fProxy)1;
                    carryRe = tmpRe;
                    carryIm = tmpIm;
                    j = (j & 1) == 0 ? (j >> 1) : (M + (j >> 1));
                }
                re[s] = carryRe;
                im[s] = carryIm;
            }

            // Step 2: Two in-place radix-4 sub-FFTs on the contiguous halves (no temp copy).
            // FftCoreRadix4Ptr twiddle indexing: W_M^j = T_tableN[j*(tableN/M)] via step=tableN/len,
            // so the full-size table drives the sub-transforms correctly — no sub-table copies needed.
            fProxy* rePtr  = re.Data.Ptr;
            fProxy* imPtr  = im.Data.Ptr;
            fProxy* twrPtr = twReFull.Data.Ptr;
            fProxy* twiPtr = twImFull.Data.Ptr;

            FftCoreRadix4Slice(rePtr,     imPtr,     twrPtr, twiPtr, M, tableN);
            FftCoreRadix4Slice(rePtr + M, imPtr + M, twrPtr, twiPtr, M, tableN);

            // Step 3: Radix-2 DIT combine.
            // W_size^k = T_tableN[k*(tableN/size)].
            // X[k]   = E[k] + W_size^k * O[k]
            // X[k+M] = E[k] - W_size^k * O[k]
            int combineStep = tableN / size;
            for (int k = 0; k < M; k++)
            {
                fProxy wr = twReFull[k * combineStep];
                fProxy wi = twImFull[k * combineStep];
                fProxy er  = re[k],     ei  = im[k];
                fProxy or_ = re[M + k], oi_ = im[M + k];
                fProxy tr  = wr * or_ - wi * oi_;
                fProxy ti  = wr * oi_ + wi * or_;
                re[k]     = er + tr;
                im[k]     = ei + ti;
                re[M + k] = er - tr;
                im[M + k] = ei - ti;
            }

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

    }
}
