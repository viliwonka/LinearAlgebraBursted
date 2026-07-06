#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System.Runtime.CompilerServices;

namespace LinearAlgebra
{
    // Public statistics surface for the SIGNED integer family (int/short/long -- uint is
    // deliberately excluded, see docs/dev/naming-style-guide.md: difference/accumulation code is
    // the unsigned-hostile category). Sibling of StatsOP.fProxy.cs's `Stats` facade -- merges
    // into the SAME bare partial class, forwarding to the distinct intStatsCore/shortStatsCore/
    // longStatsCore generic bodies (see StatsCore.iProxy.cs) so the type-identical generic
    // signatures never collide -- the same reason StatsOP.fProxy.cs's facade forwards to
    // floatStatsCore/doubleStatsCore instead of merging the generic bodies directly (see
    // docs/dev/naming-style-guide.md's "Split vs merge safety").
    //
    // RETURN-TYPE WIDENING (locked convention): sum -> long (widened accumulator, avoids
    // overflow); mean/variance/stdDev/varianceSample/stdDevSample/median -> double (need a
    // fractional result regardless of the source integer type, e.g. mean({1,2}) == 1.5);
    // min/max -> same-type (iProxy); argmin/argmax -> int (an index, not a value).
    //
    // Whole-array forms only (vector iProxyN, or a matrix iProxyMxN treated as one flat
    // distribution) -- no per-axis row/col reductions, no covariance/correlation, no in-place
    // transforms (standardize/rescale/softmax/... don't fit the widened-return convention and
    // are out of scope for this integer surface).
    public static partial class Stats
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long sum(in iProxyN   x) => iProxyStatsCore.sum(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long sum(in iProxyMxN x) => iProxyStatsCore.sum(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double mean(in iProxyN   x) => iProxyStatsCore.mean(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double mean(in iProxyMxN x) => iProxyStatsCore.mean(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double variance(in iProxyN   x) => iProxyStatsCore.variance(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double variance(in iProxyMxN x) => iProxyStatsCore.variance(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double stdDev(in iProxyN   x) => iProxyStatsCore.stdDev(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double stdDev(in iProxyMxN x) => iProxyStatsCore.stdDev(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double varianceSample(in iProxyN   x) => iProxyStatsCore.varianceSample(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double varianceSample(in iProxyMxN x) => iProxyStatsCore.varianceSample(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double stdDevSample(in iProxyN   x) => iProxyStatsCore.stdDevSample(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double stdDevSample(in iProxyMxN x) => iProxyStatsCore.stdDevSample(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double median(in iProxyN   x) => iProxyStatsCore.median(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double median(in iProxyMxN x) => iProxyStatsCore.median(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static iProxy min(in iProxyN   x) => iProxyStatsCore.min(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static iProxy min(in iProxyMxN x) => iProxyStatsCore.min(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static iProxy max(in iProxyN   x) => iProxyStatsCore.max(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static iProxy max(in iProxyMxN x) => iProxyStatsCore.max(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int argmin(in iProxyN   x) => iProxyStatsCore.argmin(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int argmin(in iProxyMxN x) => iProxyStatsCore.argmin(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int argmax(in iProxyN   x) => iProxyStatsCore.argmax(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int argmax(in iProxyMxN x) => iProxyStatsCore.argmax(in x);
    }
}
