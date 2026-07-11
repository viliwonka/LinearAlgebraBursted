using System.Runtime.CompilerServices;

namespace LinearAlgebra
{
    // Histogram: count-based distribution estimation over float/double data.
    // Bin layout, drop policy, closed-upper-edge rule, and per-method contracts are documented on
    // the generic bodies in fProxyHistogramCore. This is the prefix-free public surface: each op is
    // exposed as concrete overloads over the input shape (fProxyN vector or fProxyMxN whole-matrix,
    // flat row-major) and forwards, inlined, to the core -- so the type-identical
    // `<T> where T:IUnsafefProxyArray` signatures never collide across the merged float/double
    // partial (CS0111).
    public static partial class Histogram
    {
        // ---- histogramInto (explicit range) ----
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void histogramInto(in fProxyN   data, fProxy lo, fProxy hi, ref Indices counts) => fProxyHistogramCore.histogramInto(in data, lo, hi, ref counts);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void histogramInto(in fProxyMxN data, fProxy lo, fProxy hi, ref Indices counts) => fProxyHistogramCore.histogramInto(in data, lo, hi, ref counts);

        // ---- histogramInto (auto-range) ----
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void histogramInto(in fProxyN   data, ref Indices counts) => fProxyHistogramCore.histogramInto(in data, ref counts);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void histogramInto(in fProxyMxN data, ref Indices counts) => fProxyHistogramCore.histogramInto(in data, ref counts);

        // ---- densityInto ----
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void densityInto(in fProxyN   data, fProxy lo, fProxy hi, ref fProxyN dest) => fProxyHistogramCore.densityInto(in data, lo, hi, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void densityInto(in fProxyMxN data, fProxy lo, fProxy hi, ref fProxyN dest) => fProxyHistogramCore.densityInto(in data, lo, hi, ref dest);

        // ---- cdfInto ----
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void cdfInto(in fProxyN   data, fProxy lo, fProxy hi, ref fProxyN dest) => fProxyHistogramCore.cdfInto(in data, lo, hi, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void cdfInto(in fProxyMxN data, fProxy lo, fProxy hi, ref fProxyN dest) => fProxyHistogramCore.cdfInto(in data, lo, hi, ref dest);

        // ---- histogram2DInto (paired samples; all input-shape combinations) ----
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void histogram2DInto(in fProxyN   dataX, in fProxyN   dataY, fProxy loX, fProxy hiX, fProxy loY, fProxy hiY, ref fProxyMxN counts) => fProxyHistogramCore.histogram2DInto(in dataX, in dataY, loX, hiX, loY, hiY, ref counts);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void histogram2DInto(in fProxyN   dataX, in fProxyMxN dataY, fProxy loX, fProxy hiX, fProxy loY, fProxy hiY, ref fProxyMxN counts) => fProxyHistogramCore.histogram2DInto(in dataX, in dataY, loX, hiX, loY, hiY, ref counts);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void histogram2DInto(in fProxyMxN dataX, in fProxyN   dataY, fProxy loX, fProxy hiX, fProxy loY, fProxy hiY, ref fProxyMxN counts) => fProxyHistogramCore.histogram2DInto(in dataX, in dataY, loX, hiX, loY, hiY, ref counts);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void histogram2DInto(in fProxyMxN dataX, in fProxyMxN dataY, fProxy loX, fProxy hiX, fProxy loY, fProxy hiY, ref fProxyMxN counts) => fProxyHistogramCore.histogram2DInto(in dataX, in dataY, loX, hiX, loY, hiY, ref counts);
    }
}
