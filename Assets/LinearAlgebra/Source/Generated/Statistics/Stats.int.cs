#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System.Runtime.CompilerServices;

namespace LinearAlgebra
{
    // Public statistics surface for the SIGNED integer family (int/short/long -- uint is
    // deliberately excluded: difference/accumulation code is the unsigned-hostile category).
    // Sibling of StatsOP.fProxy.cs's `Stats` facade -- merges into the SAME bare partial class,
    // forwarding to the distinct intStatsCore/shortStatsCore/longStatsCore generic bodies (see
    // StatsCore.int.cs) so the type-identical generic signatures never collide -- the same
    // reason StatsOP.fProxy.cs's facade forwards to floatStatsCore/doubleStatsCore instead of
    // merging the generic bodies directly.
    //
    // RETURN-TYPE WIDENING (locked convention): sum -> long (widened accumulator, avoids
    // overflow); mean/variance/stdDev/varianceSample/stdDevSample/median -> double (need a
    // fractional result regardless of the source integer type, e.g. mean({1,2}) == 1.5);
    // min/max -> same-type (int); argmin/argmax -> int (an index, not a value).
    //
    // Whole-array forms only (vector intN, or a matrix intMxN treated as one flat
    // distribution) -- no per-axis row/col reductions, no covariance/correlation, no in-place
    // transforms (standardize/rescale/softmax/... don't fit the widened-return convention and
    // are out of scope for this integer surface).
    public static partial class Stats
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long sum(in intN   x) => intStatsCore.sum(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static long sum(in intMxN x) => intStatsCore.sum(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double mean(in intN   x) => intStatsCore.mean(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double mean(in intMxN x) => intStatsCore.mean(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double variance(in intN   x) => intStatsCore.variance(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double variance(in intMxN x) => intStatsCore.variance(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double stdDev(in intN   x) => intStatsCore.stdDev(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double stdDev(in intMxN x) => intStatsCore.stdDev(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double varianceSample(in intN   x) => intStatsCore.varianceSample(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double varianceSample(in intMxN x) => intStatsCore.varianceSample(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double stdDevSample(in intN   x) => intStatsCore.stdDevSample(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double stdDevSample(in intMxN x) => intStatsCore.stdDevSample(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double median(in intN   x) => intStatsCore.median(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static double median(in intMxN x) => intStatsCore.median(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int min(in intN   x) => intStatsCore.min(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int min(in intMxN x) => intStatsCore.min(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int max(in intN   x) => intStatsCore.max(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int max(in intMxN x) => intStatsCore.max(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int argmin(in intN   x) => intStatsCore.argmin(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int argmin(in intMxN x) => intStatsCore.argmin(in x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int argmax(in intN   x) => intStatsCore.argmax(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int argmax(in intMxN x) => intStatsCore.argmax(in x);
    }
}
