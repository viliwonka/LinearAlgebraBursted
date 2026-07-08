#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System.Runtime.CompilerServices;

namespace LinearAlgebra
{
    // Histogram: count-based distribution estimation over float/double data.
    // Bin layout, drop policy, closed-upper-edge rule, and per-method contracts are documented on
    // the generic bodies in floatHistogramCore. This is the prefix-free public surface: each op is
    // exposed as concrete overloads over the input shape (floatN vector or floatMxN whole-matrix,
    // flat row-major) and forwards, inlined, to the core -- so the type-identical
    // `<T> where T:IUnsafefloatArray` signatures never collide across the merged float/double
    // partial (CS0111).
    public static partial class Histogram
    {
        // ---- histogramInto (explicit range) ----
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void histogramInto(in floatN   data, float lo, float hi, ref Indices counts) => floatHistogramCore.histogramInto(in data, lo, hi, ref counts);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void histogramInto(in floatMxN data, float lo, float hi, ref Indices counts) => floatHistogramCore.histogramInto(in data, lo, hi, ref counts);

        // ---- histogramInto (auto-range) ----
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void histogramInto(in floatN   data, ref Indices counts) => floatHistogramCore.histogramInto(in data, ref counts);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void histogramInto(in floatMxN data, ref Indices counts) => floatHistogramCore.histogramInto(in data, ref counts);

        // ---- densityInto ----
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void densityInto(in floatN   data, float lo, float hi, ref floatN dest) => floatHistogramCore.densityInto(in data, lo, hi, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void densityInto(in floatMxN data, float lo, float hi, ref floatN dest) => floatHistogramCore.densityInto(in data, lo, hi, ref dest);

        // ---- cdfInto ----
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void cdfInto(in floatN   data, float lo, float hi, ref floatN dest) => floatHistogramCore.cdfInto(in data, lo, hi, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void cdfInto(in floatMxN data, float lo, float hi, ref floatN dest) => floatHistogramCore.cdfInto(in data, lo, hi, ref dest);

        // ---- histogram2DInto (paired samples; all input-shape combinations) ----
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void histogram2DInto(in floatN   dataX, in floatN   dataY, float loX, float hiX, float loY, float hiY, ref floatMxN counts) => floatHistogramCore.histogram2DInto(in dataX, in dataY, loX, hiX, loY, hiY, ref counts);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void histogram2DInto(in floatN   dataX, in floatMxN dataY, float loX, float hiX, float loY, float hiY, ref floatMxN counts) => floatHistogramCore.histogram2DInto(in dataX, in dataY, loX, hiX, loY, hiY, ref counts);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void histogram2DInto(in floatMxN dataX, in floatN   dataY, float loX, float hiX, float loY, float hiY, ref floatMxN counts) => floatHistogramCore.histogram2DInto(in dataX, in dataY, loX, hiX, loY, hiY, ref counts);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void histogram2DInto(in floatMxN dataX, in floatMxN dataY, float loX, float hiX, float loY, float hiY, ref floatMxN counts) => floatHistogramCore.histogram2DInto(in dataX, in dataY, loX, hiX, loY, hiY, ref counts);
    }
}
