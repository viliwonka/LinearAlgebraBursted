using System.Runtime.CompilerServices;

namespace BULA
{
    // Integer metric-direction + validation helpers for Query's nearest/farthest/k-selection
    // kernels. Type-agnostic (or return-type-only) signatures would collide on the merged
    // int/short/long `Query` partial (CS0111); here they emit as distinct
    // intQueryCore/shortQueryCore/longQueryCore.
    internal static partial class iProxyQueryCore
    {
        // Integer metrics forbid Euclidean/Cosine (sqrt/division are float-only); call once at entry.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void ValidateIntegerMetric(Metric m)
        {
            if (m == Metric.Euclidean || m == Metric.Cosine)
                throw new System.ArgumentException(
                    "Query: Euclidean and Cosine metrics require sqrt/division and are float-only for integer types. Use Manhattan, Chebyshev, SqEuclidean, or Dot instead.");
        }

        // Dot is similarity (higher = nearer); Manhattan/Chebyshev/SqEuclidean are distance (lower = nearer).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsSimilarityMetric(Metric m) => m == Metric.Dot;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static iProxy WorstScoreForNearest(Metric m)
            => IsSimilarityMetric(m) ? iProxy.MinValue : iProxy.MaxValue;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static iProxy WorstScoreForFarthest(Metric m)
            => IsSimilarityMetric(m) ? iProxy.MaxValue : iProxy.MinValue;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsBetterForNearest(iProxy a, iProxy b, Metric m)
            => IsSimilarityMetric(m) ? a > b : a < b;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsBetterForFarthest(iProxy a, iProxy b, Metric m)
            => IsSimilarityMetric(m) ? a < b : a > b;
    }
}
