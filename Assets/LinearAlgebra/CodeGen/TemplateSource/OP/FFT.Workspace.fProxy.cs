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
    /// Precomputed twiddle table for FFT. One size-n workspace drives the radix-4 (4^k) and
    /// mixed-radix (2·4^k = two radix-4 sub-FFTs + one radix-2 combine) dispatch for any power-of-two
    /// length ≤ n, eliminating per-element cos/sin from the hot loop. The butterfly reads its
    /// per-stage W^1 from sw1 and the radix-2 combine reads cw1 — both materialized from this quarter
    /// table via CosQ at build time. Build once via Arena.fProxyFFTCache(n) and reuse across many
    /// transforms of the same size.
    ///
    /// The table is computed at double precision and cast to the element type (float or double),
    /// so accuracy is maximized regardless of the transform element type.
    ///
    /// Only a QUARTER of the cosine circle (twQuarter, length n/4+1) is stored: twQuarter[j] =
    /// cos(2π·j/n) for j = 0..n/4. Both real and imaginary parts of any W^m = exp(-2πi·m/n) are
    /// reconstructed from it by quadrant reflection and a π/2 index shift (see CosQ): Re(W^m) =
    /// CosQ(m), Im(W^m) = CosQ(m + n/4). The reconstruction is ~1 ULP accurate (not bit-exact vs a
    /// full table): reflected entries are independently-built cos values, so a fold's sign flip is
    /// not an exact negation of the direct entry. sw1/cw1 are materialized from CosQ at build time,
    /// so the wide-butterfly and combine hot loops still read contiguous tables; only the scalar
    /// butterfly and the rfft/irfft unpack call CosQ per element.
    /// </summary>
    public struct fProxyFFTCache : IDisposable
    {
        public fProxyN twQuarter;  // length n/4+1: cos(2π·j/n), j = 0..n/4  (first quadrant, incl. 0)
        public int n;              // the FFT size this table is built for (must be a power of two, >= 2)

        // Scratch buffers — allocated once in the factory, reused on every call.
        // Single-use-at-a-time: one workspace per concurrent transform (FFTW-plan semantics).
        public fProxyN cz;         // length n/2: even-sample packing scratch for rfft/irfft
        public fProxyN sz;         // length n/2: odd-sample  packing scratch for rfft/irfft
        public fProxyN visited;    // length n:   cycle-following scratch for FftCoreRadix4Mixed
                                   //             (stores 0/1 flags via fProxy; [0,size) used per call)

        // Contiguous per-stage W^1 twiddle table for the radix-4 butterfly, every stage (quarter-stride
        // qq = 1, 4, 16, …, n/4) concatenated in stage order (total length swLen ≈ n/3). W^2 and W^3
        // are derived from W^1 in-register (W^2=W^1·W^1, W^3=W^1·W^2), so only W^1 is stored, and no
        // twiddle is reconstructed at runtime. Serves the top-level pow-4 butterfly and the pow-4
        // radix-4 sub-transforms of the mixed / rfft / irfft paths (a size-M sub-transform reads the
        // length-log4(M) prefix).
        public fProxyN sw1re, sw1im;
        public int swLen;

        // Contiguous combine-twiddle table for the mixed-radix radix-2 combine (Step 3 of
        // FftCoreRadix4Mixed): cw1re[k] = T_n[k*(n/size)], cw1im[k] = ... for k = 0..size/2-1,
        // where `size` is the single mixed-transform size this n produces (n itself when n = 2·4^k,
        // or n/2 via the rfft/irfft inner FFT when n = 4^k). Gathering the strided combine twiddle
        // into a contiguous run lets the radix-2 combine wide-load twiddles and E/O data together.
        // Both are materialized from CosQ at build time (a quarter table cannot be aliased by the
        // combine's contiguous wide load, so there is no combineStep==1 alias fast-path anymore).
        public fProxyN cw1re, cw1im;

        /// <summary>
        /// Standalone allocation sized and populated identically to <c>Arena.fProxyFFTCache(n)</c>
        /// (same twiddle-table construction, on standalone buffers). Pair with <see cref="Dispose"/>.
        /// </summary>
        public unsafe fProxyFFTCache(int n, Allocator allocator)
        {
            if (n < 2 || (n & (n - 1)) != 0)
                throw new ArgumentException("fProxyFFTCache: n must be a power of two and >= 2");

            this.n = n;
            int half = n >> 1;
            int Q    = n >> 2;
            twQuarter = new fProxyN(Q + 1, allocator);

            int P = 0;
            for (int t = n; t > 1; t >>= 1) P++;
            double* bkr = stackalloc double[32];
            double* bki = stackalloc double[32];
            bkr[P - 1] = -1.0; bki[P - 1] = 0.0;
            if (P >= 2) { bkr[P - 2] = 0.0; bki[P - 2] = -1.0; }
            for (int k = P - 3; k >= 0; k--)
            {
                double a = bkr[k + 1];
                double c = math.sqrt((1.0 + a) * 0.5);
                bkr[k] = c;
                bki[k] = bki[k + 1] / (2.0 * c);
            }
            int Qalloc = Q > 0 ? Q : 1;
            var dre = (double*)UnsafeUtility.Malloc((long)Qalloc * sizeof(double), 16, Allocator.Persistent);
            var dim = (double*)UnsafeUtility.Malloc((long)Qalloc * sizeof(double), 16, Allocator.Persistent);
            dre[0] = 1.0; dim[0] = 0.0;
            for (int k = 0; k < P - 2; k++)
            {
                int block = 1 << k;
                double br = bkr[k], bi = bki[k];
                for (int j = 0; j < block; j++)
                {
                    double ar = dre[j], ai = dim[j];
                    dre[block + j] = ar * br - ai * bi;
                    dim[block + j] = ar * bi + ai * br;
                }
            }
            for (int m = 0; m < Q; m++)
                twQuarter[m] = (fProxy)dre[m];
            twQuarter[Q] = (Q >= 1) ? (fProxy)0 : (fProxy)1;
            UnsafeUtility.Free(dre, Allocator.Persistent);
            UnsafeUtility.Free(dim, Allocator.Persistent);

            cz      = new fProxyN(half, allocator, uninit: true);
            sz      = new fProxyN(half, allocator, uninit: true);
            visited = new fProxyN(n,    allocator, uninit: true);

            fProxy* cq = twQuarter.Data.Ptr;
            swLen = 0;
            for (int qq = 1; 4 * qq <= n; qq <<= 2)
                swLen += qq;
            int swAlloc = swLen > 0 ? swLen : 1;
            sw1re = new fProxyN(swAlloc, allocator, uninit: true);
            sw1im = new fProxyN(swAlloc, allocator, uninit: true);
            {
                int off = 0;
                for (int qq = 1; 4 * qq <= n; qq <<= 2)
                {
                    int len  = qq << 2;
                    int step = n / len;
                    for (int j = 0; j < qq; j++)
                    {
                        int t1 = j * step;
                        FFT.WQ(cq, t1, n, out fProxy wr, out fProxy wi);
                        sw1re[off + j] = wr; sw1im[off + j] = wi;
                    }
                    off += qq;
                }
            }

            int cwLen = (P & 1) == 0 ? (n >> 2) : half;
            int cwStep = (P & 1) == 0 ? 2 : 1;
            cw1re = new fProxyN(cwLen, allocator, uninit: true);
            cw1im = new fProxyN(cwLen, allocator, uninit: true);
            for (int k = 0; k < cwLen; k++)
            {
                FFT.WQ(cq, k * cwStep, n, out fProxy wr, out fProxy wi);
                cw1re[k] = wr; cw1im[k] = wi;
            }
        }

        /// <summary>Dispose only instances built with the Allocator ctor; arena-built instances are arena-owned.</summary>
        public void Dispose()
        {
            twQuarter.Dispose();
            cz.Dispose();
            sz.Dispose();
            visited.Dispose();
            sw1re.Dispose();
            sw1im.Dispose();
            cw1re.Dispose();
            cw1im.Dispose();
        }
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates a twiddle-table FFT workspace for an n-point transform (n must be a power of two,
        /// n ≥ 2). Entries are computed at double precision from sqrt-based roots of unity (no sin/cos,
        /// no recurrence drift); see <see cref="fProxyFFTCache"/> for the quarter-circle table layout.
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
            int Q    = n >> 2;                        // n/4 = quarter-circle span
            var twQuarter = arena.fProxyVec(Q + 1);   // cos(2π·j/n), j = 0..Q

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
            // Recursive-doubling fill: W^0 = 1, then for each bit k, W^(2^k + j) = W^j · B_k for
            // j < 2^k. One complex-mult per entry; each entry is <= log2(n) mults deep in the
            // dependency chain, so error stays O(log n · ε). Kept in a double scratch and cast to
            // fProxy once. Only the first QUADRANT [0, n/4) is built (doubling stops two stages early,
            // at k = P-3); every reader reconstructs the rest via CosQ (quadrant reflection + π/2
            // shift), so the scratch is only n/4 doubles.
            int Qalloc = Q > 0 ? Q : 1;
            var dre = (double*)UnsafeUtility.Malloc((long)Qalloc * sizeof(double), 16, Allocator.Persistent);
            var dim = (double*)UnsafeUtility.Malloc((long)Qalloc * sizeof(double), 16, Allocator.Persistent);
            dre[0] = 1.0; dim[0] = 0.0;
            for (int k = 0; k < P - 2; k++)
            {
                int block = 1 << k;
                double br = bkr[k], bi = bki[k];
                for (int j = 0; j < block; j++)
                {
                    double ar = dre[j], ai = dim[j];
                    dre[block + j] = ar * br - ai * bi;
                    dim[block + j] = ar * bi + ai * br;
                }
            }
            for (int m = 0; m < Q; m++)
                twQuarter[m] = (fProxy)dre[m];
            twQuarter[Q] = (Q >= 1) ? (fProxy)0 : (fProxy)1;   // cos(π/2)=0 (n>=4); cos(0)=1 (n=2)
            UnsafeUtility.Free(dre, Allocator.Persistent);
            UnsafeUtility.Free(dim, Allocator.Persistent);

            // Scratch buffers — persistent in this arena (disposed with the arena).
            // cz/sz are the two-for-one packing temporaries for rfft/irfft (length n/2 = M).
            // visited is the cycle-following scratch for FftCoreRadix4Mixed (length n; [0,M) used
            // when called from the rfft/irfft inner M-point sub-FFT, still within bounds).
            var cz      = arena.fProxyVec(half, uninit: true);
            var sz      = arena.fProxyVec(half, uninit: true);
            var visited = arena.fProxyVec(n,    uninit: true);

            // Per-stage W^1 twiddle table for the radix-4 butterfly. Every stage (both the wide
            // fProxyW stages and the small scalar stages) gets its W^1 = (Re,Im)(W^(j·step)),
            // step = n/(4·qq), materialized here from the quarter table via WQ; the butterfly then
            // derives W^2/W^3 in-register and never reconstructs at runtime. Contiguous in stage order
            // (qq = 1, 4, 16, …, n/4), so a size-M sub-transform reads the length-log4(M) prefix. The
            // scalar stages add only 1+4 = 5 entries, so swLen stays ≈ n/3. Serves the top-level path
            // AND the pow-4 sub-transforms of the mixed / rfft / irfft paths (same tableN).
            fProxy* cq = twQuarter.Data.Ptr;
            int swLen = 0;
            for (int qq = 1; 4 * qq <= n; qq <<= 2)
                swLen += qq;
            int swAlloc = swLen > 0 ? swLen : 1;
            var sw1re = arena.fProxyVec(swAlloc, uninit: true);
            var sw1im = arena.fProxyVec(swAlloc, uninit: true);
            {
                int off = 0;
                for (int qq = 1; 4 * qq <= n; qq <<= 2)
                {
                    int len  = qq << 2;
                    int step = n / len;
                    for (int j = 0; j < qq; j++)
                    {
                        int t1 = j * step;
                        FFT.WQ(cq, t1, n, out fProxy wr, out fProxy wi);
                        sw1re[off + j] = wr; sw1im[off + j] = wi;
                    }
                    off += qq;
                }
            }

            // Combine-twiddle table for the mixed-radix radix-2 combine (Step 3). This n triggers the
            // mixed path at exactly one size/step:
            //   n = 2·4^k (P odd)  → fft/ifft mixed at size = n,   combineStep = 1
            //   n = 4^k    (P even) → rfft/irfft inner mixed at size = n/2, combineStep = 2
            // Both are materialized from CosQ (the quarter table cannot be aliased by a contiguous
            // wide load). Combine twiddle W_size^k = W^(k·combineStep): step 2 gathers even indices
            // 2k (P even, length n/4); step 1 gathers k (P odd, length n/2).
            int cwLen = (P & 1) == 0 ? (n >> 2) : half;
            int cwStep = (P & 1) == 0 ? 2 : 1;
            var cw1re = arena.fProxyVec(cwLen, uninit: true);
            var cw1im = arena.fProxyVec(cwLen, uninit: true);
            for (int k = 0; k < cwLen; k++)
            {
                FFT.WQ(cq, k * cwStep, n, out fProxy wr, out fProxy wi);
                cw1re[k] = wr; cw1im[k] = wi;
            }

            return new fProxyFFTCache
            {
                twQuarter = twQuarter,
                n        = n,
                cz       = cz,
                sz       = sz,
                visited  = visited,
                sw1re = sw1re, sw1im = sw1im,
                swLen = swLen,
                cw1re = cw1re, cw1im = cw1im,
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
            if (ws.n != n ||
                ws.cz.N != n >> 1 || ws.sz.N != n >> 1 || ws.visited.N != n)
                throw new ArgumentException(
                    who + ": workspace must be sized for an n-point FFT (use Arena.fProxyFFTCache(n))");
        }

        /// <summary>
        /// Throws if the workspace is missing the quarter-circle twiddle table required by the
        /// radix-4 dispatch paths. Extends <see cref="RequireFftWorkspace"/>.
        /// </summary>
        static void RequireRadix4Workspace(in fProxyFFTCache ws, int n, string who)
        {
            RequireFftWorkspace(in ws, n, who);
            if (ws.twQuarter.N != (n >> 2) + 1)
                throw new ArgumentException(
                    who + ": workspace must have quarter-circle twiddle table (use Arena.fProxyFFTCache(n))");
        }

        // ---- quarter-table twiddle reconstruction ----
        //
        // The workspace stores only cos over the first quadrant: c[j] = cos(2π·j/tableN), j = 0..tableN/4.
        // CosQ returns cos(2π·idx/tableN) for any idx by quadrant reflection; WQ returns the full
        // twiddle W^idx = exp(-2πi·idx/tableN) = (Re, Im) using Re(W^idx)=CosQ(idx), Im(W^idx)=CosQ(idx+n/4)
        // (since -sin θ = cos(θ+π/2)). tableN is a power of two, so `idx & (tableN-1)` reduces mod tableN.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe fProxy CosQ(fProxy* c, int idx, int tableN)
        {
            int Q = tableN >> 2;
            idx &= tableN - 1;                         // mod tableN
            if (idx <= Q) return c[idx];              // quadrant 1:  +cos
            int h = tableN >> 1;                       // tableN/2
            if (idx <= h) return -c[h - idx];         // quadrant 2:  -cos(π - θ)
            int t3 = h + Q;                            // 3·tableN/4
            if (idx <= t3) return -c[idx - h];        // quadrant 3:  -cos(θ - π)
            return c[tableN - idx];                   // quadrant 4:  +cos(2π - θ)
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe void WQ(fProxy* c, int idx, int tableN, out fProxy wr, out fProxy wi)
        {
            wr = CosQ(c, idx, tableN);
            // Im(W^m) = -sin(2π·idx/tableN) = cos(2π·(idx+tableN/4)/tableN). The π/2 index shift
            // needs tableN/4 >= 1; for the degenerate tableN==2 the imaginary part is identically 0.
            wi = tableN >= 4 ? CosQ(c, idx + (tableN >> 2), tableN) : (fProxy)0;
        }

        // ---- table-indexed overloads ----

        /// <summary>
        /// In-place forward FFT for any power-of-two length, using a precomputed twiddle table;
        /// throws for a non-power-of-two length (use dft for arbitrary N). ws must be sized for re.N
        /// (build via Arena.fProxyFFTCache(N)); it must contain the quarter-circle twiddle table required
        /// by the radix-4 dispatch. Both arrays must have the same length, which must be a power of two.
        /// </summary>
        public static void fft(ref fProxyN re, ref fProxyN im, in fProxyFFTCache ws)
        {
            int n = re.N;
            if (im.N != n)
                throw new ArgumentException("fft: re and im must have the same length");
            RequireRadix4Workspace(in ws, n, "fft");

            var sw1re       = ws.sw1re;
            var sw1im       = ws.sw1im;
            var cw1re       = ws.cw1re;
            var cw1im       = ws.cw1im;
            if (IsPowerOf4(n))
            {
                FftCoreRadix4(ref re, ref im, ref sw1re, ref sw1im, false);
            }
            else if ((n & (n - 1)) == 0)   // power-of-2, not power-of-4 → 2·4^k mixed-radix path
            {
                var visitedScratch = ws.visited;
                FftCoreRadix4Mixed(ref re, ref im, ref sw1re, ref sw1im, ref cw1re, ref cw1im, visitedScratch, false);
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
            if (im.N != n)
                throw new ArgumentException("ifft: re and im must have the same length");
            RequireRadix4Workspace(in ws, n, "ifft");

            var sw1re       = ws.sw1re;
            var sw1im       = ws.sw1im;
            var cw1re       = ws.cw1re;
            var cw1im       = ws.cw1im;
            if (IsPowerOf4(n))
            {
                FftCoreRadix4(ref re, ref im, ref sw1re, ref sw1im, true);
            }
            else if ((n & (n - 1)) == 0)   // power-of-2, not power-of-4 → 2·4^k mixed-radix path
            {
                var visitedScratch = ws.visited;
                FftCoreRadix4Mixed(ref re, ref im, ref sw1re, ref sw1im, ref cw1re, ref cw1im, visitedScratch, true);
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
        /// the radix-4/mixed dispatch and the unpack twiddle W_N^k is reconstructed from the quarter
        /// table via WQ — no cos/sin in the hot loop.
        /// </summary>
        public static unsafe void rfft(in fProxyN real, ref fProxyN re, ref fProxyN im, in fProxyFFTCache ws)
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

            var cz = ws.cz;   // fProxyN of length n/2 = M (packed real parts)
            var sz = ws.sz;   // fProxyN of length n/2 = M (packed imag parts)
            var twQuarter = ws.twQuarter;
            fProxy* cqPtr = twQuarter.Data.Ptr;   // used by the unpack below
            fProxy* czp   = cz.Data.Ptr;
            fProxy* szp   = sz.Data.Ptr;
            fProxy* realp = real.Data.Ptr;
            fProxy* s1rp  = ws.sw1re.Data.Ptr;
            fProxy* s1ip  = ws.sw1im.Data.Ptr;

            // Steps 1+2 FUSED: scatter the even/odd pack straight into the inner M-point FFT's first
            // permutation, out-of-place (real is a separate source), so there is neither a separate
            // pack pass nor an in-place de-interleave. real[2j]/real[2j+1] are the real/imag of packed
            // complex sample j; each lands directly in sample j's post-permutation slot. The compute
            // core then runs the butterflies (and, for the mixed path, the radix-2 combine).
            if (IsPowerOf4(M))
            {
                // Pure radix-4 (M = 4^k): first permutation is the base-4 digit reversal.
                int log4M = 0;
                for (int t = M; t > 1; t >>= 2) log4M++;
                for (int j = 0; j < M; j++)
                {
                    int r = ReverseBase4Digits(j, log4M);
                    czp[r] = realp[2 * j];
                    szp[r] = realp[2 * j + 1];
                }
                FftCoreRadix4Ptr(czp, szp, s1rp, s1ip, M);
            }
            else
            {
                // Mixed 2·4^k: first permutation is the even/odd de-interleave.
                fProxy* cw1rp = ws.cw1re.Data.Ptr;
                fProxy* cw1ip = ws.cw1im.Data.Ptr;
                int deHalf = M >> 1;
                for (int j = 0; j < M; j++)
                {
                    int d = (j & 1) == 0 ? (j >> 1) : (deHalf + (j >> 1));
                    czp[d] = realp[2 * j];
                    szp[d] = realp[2 * j + 1];
                }
                FftCoreRadix4MixedCore(czp, szp, s1rp, s1ip, cw1rp, cw1ip, M, false, null);
            }

            // Step 3: Unpack. DC and Nyquist are always real for a real input.
            re[0] = cz[0] + sz[0];
            im[0] = (fProxy)0;
            re[M] = cz[0] - sz[0];
            im[M] = (fProxy)0;

            // General bins, processed in Hermitian-symmetric pairs (k, M-k): one twiddle and one
            // packed load serve both outputs (halving twiddle work and cz/sz reads). Under k -> M-k
            // the twiddle maps W_N^(M-k) = -conj(W_N^k), so E_im, O_im and Re(W) flip sign, giving
            // re[M-k] = E_re - P, im[M-k] = Q - E_im from the same P, Q.
            // W_N^k = exp(-2πi·k/N), reconstructed from the quarter table via WQ (no cos/sin).
            int half = M >> 1;
            for (int k = 1; k < half; k++)
            {
                int kr = M - k;
                fProxy czk = cz[k], czr = cz[kr], szk = sz[k], szr = sz[kr];
                fProxy E_re = (czk + czr) * (fProxy)0.5;
                fProxy E_im = (szk - szr) * (fProxy)0.5;
                fProxy O_re = (szk + szr) * (fProxy)0.5;
                fProxy O_im = (czr - czk) * (fProxy)0.5;

                WQ(cqPtr, k, n, out fProxy curRe, out fProxy curIm);   // W_N^k
                fProxy P = curRe * O_re - curIm * O_im;
                fProxy Q = curRe * O_im + curIm * O_re;

                re[k]  = E_re + P;   im[k]  = E_im + Q;
                re[kr] = E_re - P;   im[kr] = Q - E_im;
            }
            // Middle bin k = M/2 (self-paired: M-k == k). Skipped for M == 1 (N == 2).
            if (M >= 2)
            {
                int k = half;
                int kr = M - k;
                fProxy E_re = (cz[k] + cz[kr]) * (fProxy)0.5;
                fProxy E_im = (sz[k] - sz[kr]) * (fProxy)0.5;
                fProxy O_re = (sz[k] + sz[kr]) * (fProxy)0.5;
                fProxy O_im = (cz[kr] - cz[k]) * (fProxy)0.5;

                WQ(cqPtr, k, n, out fProxy curRe, out fProxy curIm);

                re[k] = E_re + (curRe * O_re - curIm * O_im);
                im[k] = E_im + (curRe * O_im + curIm * O_re);
            }

        }

        /// <summary>
        /// Inverse real FFT using a precomputed twiddle table. ws must be sized for the real signal
        /// length N = 2*(re.N-1) (= real.N). The re-pack conjugate twiddle conj(W_N^k) is reconstructed
        /// from the quarter table via WQ and its imaginary part negated — no cos/sin in the hot loop.
        /// irfft(rfft(x, ws), ws) == x to floating-point precision.
        /// </summary>
        public static unsafe void irfft(in fProxyN re, in fProxyN im, ref fProxyN real, in fProxyFFTCache ws)
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
            // Same algebra as the recurrence irfft, with conj(W_N^k) = (Re(W_N^k), -Im(W_N^k)).
            //   E[k] = (X[k] + conj(X[M-k])) / 2
            //   O[k] = (X[k] - conj(X[M-k])) · conj(W_N^k) / 2
            //   Y[k] = E[k] + i·O[k]
            var twQuarter    = ws.twQuarter;
            fProxy* cqPtr = twQuarter.Data.Ptr;
            var sw1re    = ws.sw1re;
            var sw1im    = ws.sw1im;
            var cw1re    = ws.cw1re;
            var cw1im    = ws.cw1im;

            // The re-pack writes are FUSED into the inner inverse FFT's first permutation: each
            // re-packed sample k is scattered straight into its post-permutation slot dst(k), so the
            // inner core skips its own de-interleave/reversal. dst is a bijection over [0,M) and the
            // (k, M-k) writes read re/im (a separate buffer) — no collision, no aliasing.
            //   pure radix-4 (M = 4^k): dst = base-4 digit reversal.
            //   mixed 2·4^k:            dst = even/odd de-interleave.
            bool pure = IsPowerOf4(M);
            int log4M = 0;
            if (pure) for (int t = M; t > 1; t >>= 2) log4M++;

            // Processed in Hermitian-symmetric pairs (k, M-k): one twiddle and one load of the
            // (k, M-k) spectrum serve both re-packed outputs. Under k -> M-k the conjugate twiddle
            // maps so E_im, a and Re(W) flip sign, giving O_re unchanged, O_im negated — hence
            // cz[M-k] = E_re + O_im, sz[M-k] = O_re - E_im.
            int half = M >> 1;
            for (int k = 1; k < half; k++)
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

                // conj(W_N^k): reconstruct W_N^k via WQ, then negate its imaginary part.
                // O[k] = (a + i·b) · conj(W_N^k) / 2  →  complex multiply (a+ib)·(cRe+icIm):
                //   real = a·cRe - b·cIm,  imag = b·cRe + a·cIm
                WQ(cqPtr, k, N, out fProxy cRe, out fProxy cImPos);
                fProxy cIm = -cImPos;   // negate imaginary part for conjugate
                fProxy O_re = (a * cRe - b * cIm) * (fProxy)0.5;
                fProxy O_im = (b * cRe + a * cIm) * (fProxy)0.5;

                // Y[k] = E[k] + i·O[k], and its M-k partner — scattered to their permutation slots.
                int dk  = pure ? ReverseBase4Digits(k,  log4M) : ((k  & 1) == 0 ? (k  >> 1) : half + (k  >> 1));
                int dkr = pure ? ReverseBase4Digits(kr, log4M) : ((kr & 1) == 0 ? (kr >> 1) : half + (kr >> 1));
                cz[dk]  = E_re - O_im;   sz[dk]  = E_im + O_re;
                cz[dkr] = E_re + O_im;   sz[dkr] = O_re - E_im;
            }
            // Middle bin k = M/2 (self-paired: M-k == k). Skipped for M == 1 (N == 2).
            if (M >= 2)
            {
                int k = half;
                int kr = M - k;
                fProxy xr_k  = re[k],  xi_k  = im[k];
                fProxy xr_kr = re[kr], xi_kr = im[kr];

                fProxy E_re = (xr_k + xr_kr) * (fProxy)0.5;
                fProxy E_im = (xi_k - xi_kr) * (fProxy)0.5;
                fProxy a = xr_k - xr_kr;
                fProxy b = xi_k + xi_kr;

                WQ(cqPtr, k, N, out fProxy cRe, out fProxy cImPos);
                fProxy cIm = -cImPos;
                fProxy O_re = (a * cRe - b * cIm) * (fProxy)0.5;
                fProxy O_im = (b * cRe + a * cIm) * (fProxy)0.5;

                int dk = pure ? ReverseBase4Digits(k, log4M) : ((k & 1) == 0 ? (k >> 1) : half + (k >> 1));
                cz[dk] = E_re - O_im;
                sz[dk] = E_im + O_re;
            }

            // Inner M-point inverse FFT — first permutation already applied by the scatter above, so
            // call the compute cores directly (skipping their built-in reversal / de-interleave). The
            // output de-interleave (real[2j]=even[j], real[2j+1]=odd[j]) is fused into the core's final
            // 1/N scale pass via the interleaveOut pointer, so there is no separate output pass.
            fProxy* realp = real.Data.Ptr;
            if (pure)
                FftCoreRadix4Core(cz.Data.Ptr, sz.Data.Ptr, sw1re.Data.Ptr, sw1im.Data.Ptr, M, true, realp);
            else
                FftCoreRadix4MixedCore(cz.Data.Ptr, sz.Data.Ptr, sw1re.Data.Ptr, sw1im.Data.Ptr,
                                       cw1re.Data.Ptr, cw1im.Data.Ptr, M, true, realp);
        }

        // ---- radix-4 DIT FFT ----
        //
        // True radix-4 decimation-in-time for lengths that are exact powers of 4 (log4(n) even bits).
        // Radix-4 halves the number of full-array passes vs radix-2: log4(N) vs log2(N) stages,
        // trading twiddle-table memory against half the pass count.
        //
        // Inverse via conjugate trick: conjugate → forward radix-4 → conjugate + scale (1/N).
        // Permutation: base-4 digit reversal (reverse digit order in base-4 = swap bit-pairs).
        // Butterfly: forward DFT sign convention X[k] = Σ x[n]·exp(-2πi·kn/N).

        // IsPowerOf4 and ReverseBase4Digits live in OpHelpers.Shared.cs (type-agnostic, emitted once).

        // Inner radix-4 butterfly pointer kernel (wide + scalar hybrid).
        // Performs log4(n) stages of radix-4 DIT butterflies on already-permuted data.
        // s1r/s1i are the contiguous per-stage W^1 table (ws.sw1re/sw1im), laid out in stage order
        // (qq = 1, 4, 16, …); EVERY stage reads its precomputed W^1 from it and derives W^2/W^3
        // in-register — no runtime twiddle reconstruction. Stages with quarter-stride q >= fProxyW.Width
        // vectorize across j (Width consecutive j give contiguous wide re/im loads and a contiguous
        // read of s1r/s1i); smaller stages run the identical scalar butterfly. stageOff walks the
        // table one stage at a time, so a size-M sub-transform reads the length-log4(M) prefix.
        // n is the DATA/transform size (a power of 4). All pointer arguments are non-aliasing.
        [MethodImpl(MethodImplOptions.NoInlining)]
        static unsafe void FftCoreRadix4Ptr(
            [NoAlias] fProxy* re, [NoAlias] fProxy* im,
            [NoAlias] fProxy* s1r, [NoAlias] fProxy* s1i, int n)
        {
            // q = quarter-size per group (stride); starts at 1 and quadruples each stage.
            int W = fProxyW.Width;
            int stageOff = 0;   // running offset into the contiguous per-stage sw1 table

            for (int q = 1; q < n; q <<= 2)
            {
                int len = q << 2;       // 4q

                if (q >= W)
                {
                    // Wide stage: q (a power of 4, >= W) is a multiple of W, so no j tail.
                    for (int base_ = 0; base_ < n; base_ += len)
                    {
                        for (int j = 0; j < q; j += W)
                        {
                            int i0 = base_ + j, i1 = i0 + q, i2 = i1 + q, i3 = i2 + q;

                            fProxyW Are = fProxyW.Load(re + i0, 0), Aim = fProxyW.Load(im + i0, 0);

                            // W^1 loaded; W^2 = W^1·W^1, W^3 = W^1·W^2 derived in-register.
                            fProxyW w1r = fProxyW.Load(s1r + stageOff + j, 0), w1i = fProxyW.Load(s1i + stageOff + j, 0);
                            fProxyW w2r = w1r * w1r - w1i * w1i, w2i = w1r * w1i + w1i * w1r;
                            fProxyW w3r = w1r * w2r - w1i * w2i, w3i = w1r * w2i + w1i * w2r;

                            fProxyW x1r = fProxyW.Load(re + i1, 0), x1i = fProxyW.Load(im + i1, 0);
                            fProxyW Bre = w1r * x1r - w1i * x1i;
                            fProxyW Bim = w1r * x1i + w1i * x1r;

                            fProxyW x2r = fProxyW.Load(re + i2, 0), x2i = fProxyW.Load(im + i2, 0);
                            fProxyW Cre = w2r * x2r - w2i * x2i;
                            fProxyW Cim = w2r * x2i + w2i * x2r;

                            fProxyW x3r = fProxyW.Load(re + i3, 0), x3i = fProxyW.Load(im + i3, 0);
                            fProxyW Dre = w3r * x3r - w3i * x3i;
                            fProxyW Dim = w3r * x3i + w3i * x3r;

                            fProxyW T0re = Are + Cre, T0im = Aim + Cim;
                            fProxyW T1re = Are - Cre, T1im = Aim - Cim;
                            fProxyW T2re = Bre + Dre, T2im = Bim + Dim;
                            fProxyW T3re = Bre - Dre, T3im = Bim - Dim;

                            fProxyW.Store(re + i0, 0, T0re + T2re); fProxyW.Store(im + i0, 0, T0im + T2im);
                            fProxyW.Store(re + i2, 0, T0re - T2re); fProxyW.Store(im + i2, 0, T0im - T2im);
                            fProxyW.Store(re + i1, 0, T1re + T3im); fProxyW.Store(im + i1, 0, T1im - T3re);
                            fProxyW.Store(re + i3, 0, T1re - T3im); fProxyW.Store(im + i3, 0, T1im + T3re);
                        }
                    }
                }
                else
                {
                    // Small stage (q < Width): identical butterfly, scalar. j-outer so the stage's
                    // q (<= 4) twiddle triples are read from sw1 and W^2/W^3 derived ONCE per j, held
                    // in registers across every group — no per-butterfly reconstruction.
                    for (int j = 0; j < q; j++)
                    {
                        fProxy w1r = s1r[stageOff + j], w1i = s1i[stageOff + j];
                        fProxy w2r = w1r * w1r - w1i * w1i, w2i = w1r * w1i + w1i * w1r;
                        fProxy w3r = w1r * w2r - w1i * w2i, w3i = w1r * w2i + w1i * w2r;
                        for (int base_ = 0; base_ < n; base_ += len)
                        {
                            int i0 = base_ + j, i1 = i0 + q, i2 = i1 + q, i3 = i2 + q;

                            fProxy A_re = re[i0], A_im = im[i0];
                            fProxy B_re = w1r * re[i1] - w1i * im[i1];
                            fProxy B_im = w1r * im[i1] + w1i * re[i1];
                            fProxy C_re = w2r * re[i2] - w2i * im[i2];
                            fProxy C_im = w2r * im[i2] + w2i * re[i2];
                            fProxy D_re = w3r * re[i3] - w3i * im[i3];
                            fProxy D_im = w3r * im[i3] + w3i * re[i3];

                            fProxy T0_re = A_re + C_re, T0_im = A_im + C_im;
                            fProxy T1_re = A_re - C_re, T1_im = A_im - C_im;
                            fProxy T2_re = B_re + D_re, T2_im = B_im + D_im;
                            fProxy T3_re = B_re - D_re, T3_im = B_im - D_im;

                            re[i0] = T0_re + T2_re; im[i0] = T0_im + T2_im;
                            re[i2] = T0_re - T2_re; im[i2] = T0_im - T2_im;
                            re[i1] = T1_re + T3_im; im[i1] = T1_im - T3_re;
                            re[i3] = T1_re - T3_im; im[i3] = T1_im + T3_re;
                        }
                    }
                }
                stageOff += q;
            }
        }

        // Outer radix-4 DIT core: permutation + conjugate trick + pointer kernel + inverse scale.
        // Transform size is re.N (must be a power of 4, caller guarantees).
        // sw1re/sw1im are the workspace's per-stage W^1 table; a size-M sub-transform reads its
        // length-log4(M) prefix (the entries are shared across all sub-transforms of one workspace).
        static unsafe void FftCoreRadix4(ref fProxyN re, ref fProxyN im,
                                         ref fProxyN sw1re, ref fProxyN sw1im,
                                         bool inverse)
        {
            int size = re.N;
            if (size == 1) return;

            fProxy* rePtr  = re.Data.Ptr;
            fProxy* imPtr  = im.Data.Ptr;
            fProxy* s1rPtr = sw1re.Data.Ptr;
            fProxy* s1iPtr = sw1im.Data.Ptr;

            // Base-4 digit reversal, then the compute core.
            int log4n = 0;
            for (int t = size; t > 1; t >>= 2) log4n++;
            for (int i = 0; i < size; i++)
            {
                int j = ReverseBase4Digits(i, log4n);
                if (j > i)
                {
                    fProxy tr = rePtr[i]; rePtr[i] = rePtr[j]; rePtr[j] = tr;
                    fProxy ti = imPtr[i]; imPtr[i] = imPtr[j]; imPtr[j] = ti;
                }
            }
            FftCoreRadix4Core(rePtr, imPtr, s1rPtr, s1iPtr, size, inverse, null);
        }

        // Pure radix-4 compute core: assumes the base-4 digit reversal is already applied. Conjugate
        // trick + butterfly stages + inverse scale. rfft/irfft fuse their pack/re-pack into the
        // reversal and call this directly. The conjugate runs post-reversal here (elementwise negate
        // commutes with the permutation, so this is identical to conjugating first).
        // interleaveOut != null (inverse only): the final 1/N scale writes the result interleaved
        // (interleaveOut[2i]=Re, [2i+1]=Im, length 2*size) instead of in place, fusing irfft's output
        // de-interleave into this pass. Pass null for the in-place complex ifft.
        static unsafe void FftCoreRadix4Core(fProxy* rePtr, fProxy* imPtr,
                                             fProxy* s1rPtr, fProxy* s1iPtr,
                                             int size, bool inverse, fProxy* interleaveOut)
        {
            // Conjugate trick (see section banner above).
            if (inverse)
                for (int i = 0; i < size; i++)
                    imPtr[i] = -imPtr[i];

            FftCoreRadix4Ptr(rePtr, imPtr, s1rPtr, s1iPtr, size);

            if (inverse)
            {
                fProxy invN = (fProxy)1 / (fProxy)size;
                if (interleaveOut != null)
                    for (int i = 0; i < size; i++)
                    {
                        interleaveOut[2 * i]     =  rePtr[i] * invN;
                        interleaveOut[2 * i + 1] = -imPtr[i] * invN;   // undo conjugate, apply 1/N scale
                    }
                else
                    for (int i = 0; i < size; i++)
                    {
                        rePtr[i] =  rePtr[i] * invN;
                        imPtr[i] = -imPtr[i] * invN;   // undo conjugate, apply 1/N scale
                    }
            }
        }

        // Shared helper: base-4 digit reversal + butterfly on a raw pointer slice of length `size`.
        // re/im are already offset to the start of the sub-array; s1r/s1i are the workspace's
        // per-stage W^1 table (shared across sub-transforms). size must be a power of 4.
        static unsafe void FftCoreRadix4Slice(
            fProxy* re, fProxy* im,
            fProxy* s1r, fProxy* s1i,
            int size)
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
            FftCoreRadix4Ptr(re, im, s1r, s1i, size);
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
                                              ref fProxyN sw1re, ref fProxyN sw1im,
                                              ref fProxyN cw1re, ref fProxyN cw1im,
                                              fProxyN visited,
                                              bool inverse)
        {
            int size = re.N;
            MixedDeinterleave(re.Data.Ptr, im.Data.Ptr, visited.Data.Ptr, size);
            FftCoreRadix4MixedCore(re.Data.Ptr, im.Data.Ptr,
                                   sw1re.Data.Ptr, sw1im.Data.Ptr,
                                   cw1re.Data.Ptr, cw1im.Data.Ptr, size, inverse, null);
        }

        // In-place even/odd de-interleave via cycle-following: after it, [0,M) holds the
        // even-indexed elements and [M,2M) holds the odd-indexed (M = size/2), both in natural
        // order — the layout FftCoreRadix4MixedCore expects. visited is a length->=size 0/1
        // scratch, cleared here. rfft/irfft skip this: they scatter their pack straight into the
        // de-interleaved layout out-of-place (real source ≠ dest), so no cycle-following is needed.
        static unsafe void MixedDeinterleave(fProxy* re, fProxy* im, fProxy* visited, int size)
        {
            int M = size >> 1;

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
        }

        // Mixed-radix compute core: assumes the even/odd de-interleave is already done (even-indexed
        // samples in [0,M), odd in [M,2M); M = size/2). Conjugate trick + two radix-4 sub-FFTs +
        // radix-2 combine + inverse scale. The conjugate runs post-de-interleave here (elementwise
        // negate commutes with the permutation, so this is identical to conjugating first).
        // interleaveOut != null (inverse only): the final 1/N scale writes the result interleaved
        // (interleaveOut[2i]=Re, [2i+1]=Im, length 2*size) instead of in place, fusing irfft's output
        // de-interleave into this pass. Pass null for the in-place complex ifft.
        static unsafe void FftCoreRadix4MixedCore(fProxy* rePtr, fProxy* imPtr,
                                                  fProxy* s1rPtr, fProxy* s1iPtr,
                                                  fProxy* cw1r, fProxy* cw1i,
                                                  int size, bool inverse, fProxy* interleaveOut)
        {
            int M = size >> 1;   // size/2, always a power of 4

            // Conjugate trick at the outer level (see banner above).
            if (inverse)
                for (int i = 0; i < size; i++)
                    imPtr[i] = -imPtr[i];

            // Two in-place radix-4 sub-FFTs on the contiguous halves (no temp copy). Each reads the
            // length-log4(M) prefix of the shared per-stage W^1 table (sw1) — no sub-table copies.
            FftCoreRadix4Slice(rePtr,     imPtr,     s1rPtr, s1iPtr, M);
            FftCoreRadix4Slice(rePtr + M, imPtr + M, s1rPtr, s1iPtr, M);

            // Radix-2 DIT combine (wide + scalar hybrid).
            //   X[k]   = E[k] + W_size^k * O[k]
            //   X[k+M] = E[k] - W_size^k * O[k]
            // The combine twiddle W_size^k = cw1[k] (pre-gathered contiguously), so Width consecutive
            // k wide-load the twiddles AND the E/O data together. Scalar tail handles M < Width.
            int Wc = fProxyW.Width;
            int k = 0;
            for (; k + Wc <= M; k += Wc)
            {
                fProxyW wr = fProxyW.Load(cw1r + k, 0), wi = fProxyW.Load(cw1i + k, 0);
                fProxyW er  = fProxyW.Load(rePtr + k, 0),     ei  = fProxyW.Load(imPtr + k, 0);
                fProxyW or_ = fProxyW.Load(rePtr + M + k, 0), oi_ = fProxyW.Load(imPtr + M + k, 0);
                fProxyW tr  = wr * or_ - wi * oi_;
                fProxyW ti  = wr * oi_ + wi * or_;
                fProxyW.Store(rePtr + k, 0,     er + tr); fProxyW.Store(imPtr + k, 0,     ei + ti);
                fProxyW.Store(rePtr + M + k, 0, er - tr); fProxyW.Store(imPtr + M + k, 0, ei - ti);
            }
            for (; k < M; k++)
            {
                fProxy wr = cw1r[k], wi = cw1i[k];
                fProxy er  = rePtr[k],     ei  = imPtr[k];
                fProxy or_ = rePtr[M + k], oi_ = imPtr[M + k];
                fProxy tr  = wr * or_ - wi * oi_;
                fProxy ti  = wr * oi_ + wi * or_;
                rePtr[k]     = er + tr; imPtr[k]     = ei + ti;
                rePtr[M + k] = er - tr; imPtr[M + k] = ei - ti;
            }

            if (inverse)
            {
                fProxy invN = (fProxy)1 / (fProxy)size;
                if (interleaveOut != null)
                    for (int i = 0; i < size; i++)
                    {
                        interleaveOut[2 * i]     =  rePtr[i] * invN;
                        interleaveOut[2 * i + 1] = -imPtr[i] * invN;   // undo conjugate, apply 1/N scale
                    }
                else
                    for (int i = 0; i < size; i++)
                    {
                        rePtr[i] =  rePtr[i] * invN;
                        imPtr[i] = -imPtr[i] * invN;   // undo conjugate, apply 1/N scale
                    }
            }
        }

    }
}
