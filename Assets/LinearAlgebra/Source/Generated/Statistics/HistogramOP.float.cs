#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra.Stats
{
    // HistogramOP: count-based distribution estimation over float/double data.
    //
    // Bin layout: K equal-width bins over [lo, hi), width w = (hi − lo) / K;
    // bin index b = (int)floor((x − lo) / w).
    // Out-of-range policy = DROP (b < 0 or b >= K is skipped).
    // Closed upper edge: x == hi maps to the last bin K−1 (not dropped).
    //
    // All output buffers are zeroed before accumulation (caller buffers may hold garbage).
    // Validate arguments before allocating any scratch so throw paths cannot leak memory.
    //
    // Generic constraint IUnsafefloatArray lets these ops work over both floatN and
    // floatMxN (a matrix histograms its flat row-major data, identical to StatsOP reductions).
    public static partial class floatHistogramOP
    {
        // -------------------------------------------------------------------------
        // histogramInto — explicit range
        // -------------------------------------------------------------------------

        /// <summary>
        /// Bins <paramref name="data"/> into K equal-width bins over [<paramref name="lo"/>, <paramref name="hi"/>]
        /// and writes integer counts into <paramref name="counts"/>.
        /// K = <c>counts.N</c>; K >= 1 and hi > lo required.
        /// <para>Bin index: <c>b = (int)floor((x − lo) / w)</c>, <c>w = (hi − lo) / K</c>.
        /// A sample is kept only when <c>lo &lt;= x &lt;= hi</c>; values outside that range and NaN
        /// are dropped (out-of-range/NaN policy = DROP). The closed upper edge <c>x == hi</c>
        /// maps to the last bin K−1 rather than being dropped. Floating-point rounding at bin
        /// boundaries is clamped so no in-range sample is accidentally discarded.</para>
        /// <para><paramref name="counts"/> is zeroed first; the caller buffer may hold garbage.</para>
        /// </summary>
        public static void histogramInto<T>(in T data, float lo, float hi, ref Indices counts)
            where T : unmanaged, IUnsafefloatArray
        {
            int K = counts.N;
            if (K < 1)
                throw new ArgumentException("histogramInto: counts.N must be >= 1");
            if (!(hi > lo))
                throw new ArgumentException("histogramInto: hi must be > lo");

            // Zero output buffer (caller buffer may hold garbage).
            for (int b = 0; b < K; b++)
                counts[b] = 0;

            float w = (hi - lo) / (float)K;
            int n = data.Data.Length;
            for (int i = 0; i < n; i++)
            {
                float x = data.Data[i];
                // In-range test that also drops NaN (NaN fails both comparisons). Keeps [lo, hi] inclusive;
                // x == hi is the closed upper edge → last bin.
                if (!(x >= lo && x <= hi))
                    continue;
                int b = (x == hi) ? K - 1 : (int)math.floor((x - lo) / w);
                // Guard floating-point rounding at the bin edges; x is in-range here, so b must land in [0, K-1].
                if (b < 0) b = 0;
                else if (b >= K) b = K - 1;
                counts[b]++;
            }
        }

        // -------------------------------------------------------------------------
        // histogramInto — auto-range overload
        // -------------------------------------------------------------------------

        /// <summary>
        /// Auto-range overload: performs one pass over <paramref name="data"/> to find min/max of
        /// finite samples, then bins into K equal-width bins.
        /// K = <c>counts.N</c>; K >= 1 required.
        /// <para>Non-finite samples (NaN, ±Inf) are always dropped.
        /// Empty data (Length == 0) or all-non-finite data → all counts are zero and the method
        /// returns immediately. Constant finite data (max == min): all finite samples are placed in
        /// bin 0 (avoids divide-by-zero). Otherwise forwards to the explicit-range overload with
        /// lo = min, hi = max so that max lands in the last bin via the closed-upper-edge rule.</para>
        /// <para><paramref name="counts"/> is zeroed first; the caller buffer may hold garbage.</para>
        /// </summary>
        public static void histogramInto<T>(in T data, ref Indices counts)
            where T : unmanaged, IUnsafefloatArray
        {
            int K = counts.N;
            if (K < 1)
                throw new ArgumentException("histogramInto: counts.N must be >= 1");

            // Zero output buffer.
            for (int b = 0; b < K; b++)
                counts[b] = 0;

            int n = data.Data.Length;
            if (n == 0)
                return;

            // One pass: find min and max over FINITE samples only.
            // Seed from the first finite element; skip non-finite (NaN, ±Inf).
            float mn = (float)0;
            float mx = (float)0;
            int finiteSeed = -1;
            int finiteCount = 0;
            for (int i = 0; i < n; i++)
            {
                float v = data.Data[i];
                if (!math.isfinite(v)) continue;
                finiteCount++;
                if (finiteSeed < 0)
                {
                    mn = mx = v;
                    finiteSeed = i;
                }
                else
                {
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
            }

            // No finite elements: leave counts all-zero and return.
            if (finiteSeed < 0)
                return;

            if (mn == mx)
            {
                // Constant finite data: all finite samples land in bin 0.
                counts[0] = finiteCount;
                return;
            }

            // Forward to explicit overload (lo=min, hi=max).
            // The explicit overload re-zeroes counts before accumulating (harmless redundancy).
            histogramInto(in data, mn, mx, ref counts);
        }

        // -------------------------------------------------------------------------
        // densityInto
        // -------------------------------------------------------------------------

        /// <summary>
        /// Computes the probability density estimate over [<paramref name="lo"/>, <paramref name="hi"/>)
        /// and writes it into <paramref name="dest"/>.
        /// K = <c>dest.N</c>; K >= 1 and hi > lo and non-empty data required.
        /// <para>Formula: <c>dest[b] = count_b / (N * w)</c>, where N = data.Length and
        /// w = (hi − lo) / K. When all samples are in range, <c>Σ dest[b] * w == 1</c>
        /// (proper probability density integrating to 1). Out-of-range samples (drops) reduce
        /// the integral below 1.</para>
        /// <para>Same bin rule as <see cref="histogramInto{T}(in T, float, float, ref Indices)"/>:
        /// b = (int)floor((x − lo) / w); x == hi → last bin; b outside [0,K) → dropped.</para>
        /// <para>Allocates one <c>Allocator.Temp</c> scratch of K ints; disposed before return.
        /// Arguments are validated before any allocation.</para>
        /// </summary>
        public static void densityInto<T>(in T data, float lo, float hi, ref floatN dest)
            where T : unmanaged, IUnsafefloatArray
        {
            int K = dest.N;
            int N = data.Data.Length;

            // Validate before allocating scratch so throw paths cannot leak.
            if (K < 1)
                throw new ArgumentException("densityInto: dest.N must be >= 1");
            if (!(hi > lo))
                throw new ArgumentException("densityInto: hi must be > lo");
            if (N == 0)
                throw new ArgumentException("densityInto: data must be non-empty (cannot normalize)");

            var scratch = new Indices(K, Allocator.Temp);
            histogramInto(in data, lo, hi, ref scratch);

            float invNW = (float)K / ((float)N * (hi - lo));
            for (int b = 0; b < K; b++)
                dest[b] = (float)scratch[b] * invNW;

            scratch.Dispose();
        }

        // -------------------------------------------------------------------------
        // cdfInto
        // -------------------------------------------------------------------------

        /// <summary>
        /// Computes the empirical cumulative distribution function (CDF) over
        /// [<paramref name="lo"/>, <paramref name="hi"/>), normalized over in-range samples only,
        /// and writes it into <paramref name="dest"/>.
        /// K = <c>dest.N</c>; K >= 1 and hi > lo required.
        /// <para>Formula: <c>dest[b] = (Σ_{i &lt;= b} count_i) / inRangeTotal</c>, monotone
        /// non-decreasing. <c>dest[K−1] == 1</c> if any sample is in range.
        /// If all samples are dropped (none in range), all entries are 0.</para>
        /// <para>Same bin rule as <see cref="histogramInto{T}(in T, float, float, ref Indices)"/>:
        /// b = (int)floor((x − lo) / w); x == hi → last bin; b outside [0,K) → dropped.</para>
        /// <para>Allocates one <c>Allocator.Temp</c> scratch of K ints; disposed before return.
        /// Arguments are validated before any allocation.</para>
        /// </summary>
        public static void cdfInto<T>(in T data, float lo, float hi, ref floatN dest)
            where T : unmanaged, IUnsafefloatArray
        {
            int K = dest.N;

            // Validate before allocating scratch so throw paths cannot leak.
            if (K < 1)
                throw new ArgumentException("cdfInto: dest.N must be >= 1");
            if (!(hi > lo))
                throw new ArgumentException("cdfInto: hi must be > lo");

            var scratch = new Indices(K, Allocator.Temp);
            histogramInto(in data, lo, hi, ref scratch);

            // Sum in-range total.
            int inRangeTotal = 0;
            for (int b = 0; b < K; b++)
                inRangeTotal += scratch[b];

            if (inRangeTotal == 0)
            {
                // All samples dropped (or empty data): CDF is all zeros.
                for (int b = 0; b < K; b++)
                    dest[b] = (float)0;
            }
            else
            {
                float invTotal = (float)1 / (float)inRangeTotal;
                int cum = 0;
                for (int b = 0; b < K; b++)
                {
                    cum += scratch[b];
                    dest[b] = (float)cum * invTotal;
                }
                dest[K - 1] = (float)1;   // pin last bin to bit-exact 1 (cum*invTotal may be 1-ulp short)
            }

            scratch.Dispose();
        }

        // -------------------------------------------------------------------------
        // histogram2DInto
        // -------------------------------------------------------------------------

        /// <summary>
        /// Computes a 2D joint histogram (heatmap) over paired samples and writes float-valued
        /// counts into <paramref name="counts"/> (Kx × Ky matrix, <b>rows = X bins, cols = Y bins</b>).
        /// <para>Kx = <c>counts.M_Rows</c>, Ky = <c>counts.N_Cols</c>; both >= 1 required.
        /// <c>dataX.Data.Length == dataY.Data.Length</c> required (paired points).
        /// hiX > loX and hiY > loY required.</para>
        /// <para>Bin rule applied independently per axis:
        /// b = (int)floor((x − lo) / w); the endpoint (x == hi) lands in the last bin (closed upper
        /// edge). NaN or out-of-range values on either axis cause the pair to be dropped. A point is
        /// counted only when BOTH coords are finite and in range. Floating-point rounding at bin
        /// boundaries is clamped so no in-range sample is accidentally discarded.</para>
        /// <para><paramref name="counts"/> is zeroed first; the caller buffer may hold garbage.</para>
        /// <para>Precision note: float variant counts are exact up to 2^24 (~16.7M) per bin;
        /// the double variant up to 2^53.</para>
        /// </summary>
        public static void histogram2DInto<TX, TY>(
            in TX dataX, in TY dataY,
            float loX, float hiX,
            float loY, float hiY,
            ref floatMxN counts)
            where TX : unmanaged, IUnsafefloatArray
            where TY : unmanaged, IUnsafefloatArray
        {
            if (dataX.Data.Length != dataY.Data.Length)
                throw new ArgumentException("histogram2DInto: dataX and dataY must have the same length");
            if (!(hiX > loX))
                throw new ArgumentException("histogram2DInto: hiX must be > loX");
            if (!(hiY > loY))
                throw new ArgumentException("histogram2DInto: hiY must be > loY");
            if (counts.M_Rows < 1)
                throw new ArgumentException("histogram2DInto: counts.M_Rows must be >= 1");
            if (counts.N_Cols < 1)
                throw new ArgumentException("histogram2DInto: counts.N_Cols must be >= 1");

            int Kx = counts.M_Rows;
            int Ky = counts.N_Cols;

            // Zero output buffer (caller buffer may hold garbage).
            for (int r = 0; r < Kx; r++)
                for (int c = 0; c < Ky; c++)
                    counts[r, c] = (float)0;

            float wX = (hiX - loX) / (float)Kx;
            float wY = (hiY - loY) / (float)Ky;

            int n = dataX.Data.Length;
            for (int i = 0; i < n; i++)
            {
                float x = dataX.Data[i];
                float y = dataY.Data[i];

                // In-range test per axis; also drops NaN (NaN fails both comparisons).
                if (!(x >= loX && x <= hiX)) continue;
                if (!(y >= loY && y <= hiY)) continue;

                int bx = (x == hiX) ? Kx - 1 : (int)math.floor((x - loX) / wX);
                if (bx < 0) bx = 0; else if (bx >= Kx) bx = Kx - 1;
                int by = (y == hiY) ? Ky - 1 : (int)math.floor((y - loY) / wY);
                if (by < 0) by = 0; else if (by >= Ky) by = Ky - 1;
                counts[bx, by] += (float)1;
            }
        }
    }
}
