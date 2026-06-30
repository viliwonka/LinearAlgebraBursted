#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra
{
    /// <summary>
    /// Data-vector interpolation and matrix resizing: evaluate a discrete data vector as a
    /// continuous function at an arbitrary position (<see cref="sampleAt"/>), gather a set of
    /// positions into a destination (<see cref="sampleAtInto"/>), resize a 1-D signal to a new
    /// length (<see cref="resampleInto"/>), and separably resize a matrix
    /// (<see cref="resample2DInto"/>).
    ///
    /// Interpolation modes (<see cref="Interp"/>): Nearest, Linear, Cubic (Catmull-Rom).
    /// Edge modes (<see cref="EdgeMode"/>): Clamp (repeat edge), Wrap (periodic), Mirror
    /// (no-edge-repeat reflection, period 2*(N-1)).
    ///
    /// All methods are Burst-compatible and allocation-free except <see cref="resample2DInto"/>,
    /// which allocates exactly one Allocator.Temp scratch buffer and disposes it before returning.
    /// double-only (float / double generated variants); use the existing integer ops for
    /// index-space resampling.
    /// </summary>
    public static partial class doubleResample_OP
    {
        // ---- private helpers ----

        /// <summary>
        /// Resolves an arbitrary integer index i into the valid range [0, n-1] according to the
        /// given EdgeMode. Clamp is the cheap common-case path (math.clamp); Wrap/Mirror use modulo.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int idx(int i, int n, EdgeMode edge)
        {
            switch (edge)
            {
                case EdgeMode.Clamp:
                    return math.clamp(i, 0, n - 1);
                case EdgeMode.Wrap:
                    return ((i % n) + n) % n;
                default: // EdgeMode.Mirror — no-edge-repeat, period 2*(n-1)
                    if (n == 1) return 0;
                    int p = 2 * (n - 1);
                    int iMod = ((i % p) + p) % p;
                    return iMod < n ? iMod : p - iMod;
            }
        }

        /// <summary>Samples a single row of matrix m at a continuous column position.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double sampleRowAt(in doubleMxN m, int row, double pos, Interp interp, EdgeMode edge)
        {
            int nCols = m.N_Cols;
            switch (interp)
            {
                case Interp.Nearest:
                    return m[row, idx((int)math.round(pos), nCols, edge)];
                case Interp.Linear:
                {
                    int i0 = (int)math.floor(pos);
                    double frac = pos - (double)i0;
                    double v0 = m[row, idx(i0,     nCols, edge)];
                    double v1 = m[row, idx(i0 + 1, nCols, edge)];
                    return math.lerp(v0, v1, frac);
                }
                default: // Interp.Cubic — Catmull-Rom
                {
                    int i0 = (int)math.floor(pos);
                    double t  = pos - (double)i0;
                    double p0 = m[row, idx(i0 - 1, nCols, edge)];
                    double p1 = m[row, idx(i0,     nCols, edge)];
                    double p2 = m[row, idx(i0 + 1, nCols, edge)];
                    double p3 = m[row, idx(i0 + 2, nCols, edge)];
                    return (double)0.5 * (
                        (double)2 * p1
                        + (-p0 + p2) * t
                        + ((double)2 * p0 - (double)5 * p1 + (double)4 * p2 - p3) * t * t
                        + (-p0 + (double)3 * p1 - (double)3 * p2 + p3) * t * t * t);
                }
            }
        }

        /// <summary>Samples a single column of matrix m at a continuous row position.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double sampleColAt(in doubleMxN m, int col, double pos, Interp interp, EdgeMode edge)
        {
            int nRows = m.M_Rows;
            switch (interp)
            {
                case Interp.Nearest:
                    return m[idx((int)math.round(pos), nRows, edge), col];
                case Interp.Linear:
                {
                    int i0 = (int)math.floor(pos);
                    double frac = pos - (double)i0;
                    double v0 = m[idx(i0,     nRows, edge), col];
                    double v1 = m[idx(i0 + 1, nRows, edge), col];
                    return math.lerp(v0, v1, frac);
                }
                default: // Interp.Cubic — Catmull-Rom
                {
                    int i0 = (int)math.floor(pos);
                    double t  = pos - (double)i0;
                    double p0 = m[idx(i0 - 1, nRows, edge), col];
                    double p1 = m[idx(i0,     nRows, edge), col];
                    double p2 = m[idx(i0 + 1, nRows, edge), col];
                    double p3 = m[idx(i0 + 2, nRows, edge), col];
                    return (double)0.5 * (
                        (double)2 * p1
                        + (-p0 + p2) * t
                        + ((double)2 * p0 - (double)5 * p1 + (double)4 * p2 - p3) * t * t
                        + (-p0 + (double)3 * p1 - (double)3 * p2 + p3) * t * t * t);
                }
            }
        }

        // ---- public API ----

        /// <summary>
        /// Evaluates the data vector as a continuous function at position pos.
        /// A length-N vector spans the coordinate range [0, N-1]; pos need not be integral.
        ///
        /// Nearest : rounds pos to the nearest integer index (Unity.Mathematics banker rounding).
        /// Linear  : lerps between data[floor(pos)] and data[floor(pos)+1].
        /// Cubic   : Catmull-Rom over 4 taps i0-1, i0, i0+1, i0+2 (i0=floor(pos)):
        ///             0.5 * ((2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t² + (-p0+3*p1-3*p2+p3)*t³)
        ///           where t = pos - floor(pos).
        ///
        /// Out-of-range tap indices are resolved via the edge mode before lookup.
        /// data.N must be >= 1.
        /// </summary>
        public static double sampleAt(in doubleN data, double pos, Interp interp, EdgeMode edge)
        {
            if (data.N < 1)
                throw new ArgumentException("sampleAt: data must have length >= 1");

            int n = data.N;
            switch (interp)
            {
                case Interp.Nearest:
                    return data[idx((int)math.round(pos), n, edge)];
                case Interp.Linear:
                {
                    int i0 = (int)math.floor(pos);
                    double frac = pos - (double)i0;
                    double v0 = data[idx(i0,     n, edge)];
                    double v1 = data[idx(i0 + 1, n, edge)];
                    return math.lerp(v0, v1, frac);
                }
                default: // Interp.Cubic — Catmull-Rom
                {
                    int i0 = (int)math.floor(pos);
                    double t  = pos - (double)i0;
                    double p0 = data[idx(i0 - 1, n, edge)];
                    double p1 = data[idx(i0,     n, edge)];
                    double p2 = data[idx(i0 + 1, n, edge)];
                    double p3 = data[idx(i0 + 2, n, edge)];
                    return (double)0.5 * (
                        (double)2 * p1
                        + (-p0 + p2) * t
                        + ((double)2 * p0 - (double)5 * p1 + (double)4 * p2 - p3) * t * t
                        + (-p0 + (double)3 * p1 - (double)3 * p2 + p3) * t * t * t);
                }
            }
        }

        /// <summary>
        /// Gathers: dest[j] = sampleAt(data, positions[j], interp, edge) for j in [0, dest.N).
        /// dest.N must equal positions.N.
        ///
        /// Note: dest must not alias data — reads of data must remain stable during the gather.
        /// No alias guard is applied; the caller is responsible for ensuring distinct buffers.
        /// </summary>
        public static void sampleAtInto(in doubleN data, in doubleN positions, ref doubleN dest,
            Interp interp, EdgeMode edge)
        {
            if (dest.N != positions.N)
                throw new ArgumentException("sampleAtInto: dest.N must equal positions.N");
            if (data.N < 1)
                throw new ArgumentException("sampleAtInto: data must have length >= 1");

            int count = dest.N;
            for (int j = 0; j < count; j++)
                dest[j] = sampleAt(in data, positions[j], interp, edge);
        }

        /// <summary>
        /// Resizes src (length src.N) into dst (length dst.N) using point-resampling with the
        /// given interpolation and edge mode.
        ///
        /// Endpoint-preserving: dst[0] maps to src[0] and dst[dst.N-1] maps to src[src.N-1].
        /// Intermediate positions follow pos(j) = j * (src.N-1) / (dst.N-1).
        /// dst.N == 1 → dst[0] = src[0].
        ///
        /// src.N must be >= 1; dst.N must be >= 1.
        ///
        /// Note: dst must not alias src — reads of src must remain stable during the resize.
        /// No alias guard is applied; the caller is responsible for ensuring distinct buffers.
        ///
        /// Downsampling note: this is point-resampling with no anti-alias prefilter — decimating
        /// noisy data may alias. Callers should smooth the source first (e.g. with the existing
        /// gaussianKernel / convolution path) before downsampling.
        /// </summary>
        public static void resampleInto(in doubleN src, ref doubleN dst, Interp interp, EdgeMode edge)
        {
            if (src.N < 1)
                throw new ArgumentException("resampleInto: src must have length >= 1");
            if (dst.N < 1)
                throw new ArgumentException("resampleInto: dst must have length >= 1");

            int srcN = src.N;
            int dstN = dst.N;

            if (dstN == 1)
            {
                dst[0] = src[0];
                return;
            }

            double scale = (double)(srcN - 1) / (double)(dstN - 1);
            for (int j = 0; j < dstN; j++)
            {
                double pos = (double)j * scale;
                dst[j] = sampleAt(in src, pos, interp, edge);
            }

            // Pin the endpoints exactly: the floating-point product j*scale at j==dstN-1 may
            // land ~1 ulp short of srcN-1 for certain combinations of srcN/dstN. Callers rely
            // on dst[0] == src[0] and dst[dstN-1] == src[srcN-1] (endpoint-preserving contract).
            dst[0]        = src[0];
            dst[dstN - 1] = src[srcN - 1];
        }

        /// <summary>
        /// Separably resizes the matrix src (M×N) into dst (M'×N') using the given interpolation
        /// and edge mode. Endpoint-preserving on each axis independently.
        ///
        /// Two-pass separable approach:
        ///   Pass 1 (horizontal): for each source row, resample N columns → N' columns,
        ///           writing into a temporary scratch matrix of size M×N'.
        ///   Pass 2 (vertical):   for each column of the scratch, resample M rows → M' rows,
        ///           writing into dst.
        /// NN / bilinear / bicubic correspond to Interp.Nearest / Linear / Cubic applied
        /// independently on each axis — same kernel, same edge mode.
        ///
        /// src must be at least 1×1.
        /// One Allocator.Temp scratch (M×N') is allocated after validation and disposed before
        /// return. All argument checks run before the allocation so no throw path can leak memory.
        ///
        /// Downsampling note: no anti-alias prefilter — smooth before downsampling if needed
        /// (see <see cref="resampleInto"/>).
        /// </summary>
        public static void resample2DInto(in doubleMxN src, ref doubleMxN dst, Interp interp, EdgeMode edge)
        {
            // Validate BEFORE allocating the scratch so the throw path cannot leak.
            if (src.M_Rows < 1 || src.N_Cols < 1)
                throw new ArgumentException("resample2DInto: src must be at least 1x1");
            if (dst.M_Rows < 1 || dst.N_Cols < 1)
                throw new ArgumentException("resample2DInto: dst must be at least 1x1");

            int srcM = src.M_Rows;
            int srcN = src.N_Cols;
            int dstM = dst.M_Rows;
            int dstN = dst.N_Cols;

            // Scratch: srcM rows × dstN cols — holds the horizontally-resampled intermediate.
            var scratch = new doubleMxN(srcM, dstN, Allocator.Temp);

            // ---- Pass 1: Horizontal (columns within each row, srcN → dstN) ----
            double hScale = dstN > 1 ? (double)(srcN - 1) / (double)(dstN - 1) : (double)0;
            for (int r = 0; r < srcM; r++)
            {
                if (dstN == 1)
                {
                    scratch[r, 0] = src[r, 0];
                }
                else
                {
                    for (int j = 0; j < dstN; j++)
                    {
                        double pos = (double)j * hScale;
                        scratch[r, j] = sampleRowAt(in src, r, pos, interp, edge);
                    }
                    // Pin horizontal endpoints exactly (same reason as resampleInto).
                    scratch[r, 0]        = src[r, 0];
                    scratch[r, dstN - 1] = src[r, srcN - 1];
                }
            }

            // ---- Pass 2: Vertical (rows within each column, srcM → dstM) ----
            double vScale = dstM > 1 ? (double)(srcM - 1) / (double)(dstM - 1) : (double)0;
            for (int c = 0; c < dstN; c++)
            {
                if (dstM == 1)
                {
                    dst[0, c] = scratch[0, c];
                }
                else
                {
                    for (int i = 0; i < dstM; i++)
                    {
                        double pos = (double)i * vScale;
                        dst[i, c] = sampleColAt(in scratch, c, pos, interp, edge);
                    }
                    // Pin vertical endpoints exactly.
                    dst[0,        c] = scratch[0,        c];
                    dst[dstM - 1, c] = scratch[srcM - 1, c];
                }
            }

            scratch.Dispose();
        }
    }
}
