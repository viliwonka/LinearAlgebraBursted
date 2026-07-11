#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System.Runtime.CompilerServices;

namespace LinearAlgebra
{
    // Public statistics surface for the SIGNED integer family (int/short/long -- uint is
    // deliberately excluded: difference/accumulation code is the unsigned-hostile category).
    // Sibling of StatsOP.fProxy.cs's `Stats` facade -- merges into the SAME bare partial class,
    // forwarding to the distinct intStatsCore/shortStatsCore/longStatsCore generic bodies (see
    // StatsCore.long.cs) so the type-identical generic signatures never collide -- the same
    // reason StatsOP.fProxy.cs's facade forwards to floatStatsCore/doubleStatsCore instead of
    // merging the generic bodies directly.
    //
    // RETURN-TYPE WIDENING (locked convention): sum -> long (widened accumulator, avoids
    // overflow); mean/variance/stdDev/varianceSample/stdDevSample/median -> double (need a
    // fractional result regardless of the source integer type, e.g. mean({1,2}) == 1.5);
    // min/max -> same-type (long); argmin/argmax -> int (an index, not a value).
    //
    // Whole-array forms only (vector longN, or a matrix longMxN treated as one flat
    // distribution) -- no per-axis row/col reductions, no covariance/correlation, no in-place
    // transforms (standardize/rescale/softmax/... don't fit the widened-return convention and
    // are out of scope for this integer surface).
    public static partial class Stats
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long sum(in longN   x) => longStatsCore.sum(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long sum(in longMxN x) => longStatsCore.sum(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double mean(in longN   x) => longStatsCore.mean(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double mean(in longMxN x) => longStatsCore.mean(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double variance(in longN   x) => longStatsCore.variance(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double variance(in longMxN x) => longStatsCore.variance(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double stdDev(in longN   x) => longStatsCore.stdDev(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double stdDev(in longMxN x) => longStatsCore.stdDev(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double varianceSample(in longN   x) => longStatsCore.varianceSample(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double varianceSample(in longMxN x) => longStatsCore.varianceSample(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double stdDevSample(in longN   x) => longStatsCore.stdDevSample(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double stdDevSample(in longMxN x) => longStatsCore.stdDevSample(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double median(in longN   x) => longStatsCore.median(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double median(in longMxN x) => longStatsCore.median(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long min(in longN   x) => longStatsCore.min(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long min(in longMxN x) => longStatsCore.min(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long max(in longN   x) => longStatsCore.max(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long max(in longMxN x) => longStatsCore.max(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int argmin(in longN   x) => longStatsCore.argmin(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int argmin(in longMxN x) => longStatsCore.argmin(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int argmax(in longN   x) => longStatsCore.argmax(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int argmax(in longMxN x) => longStatsCore.argmax(in x);
    }
}
