#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System.Runtime.CompilerServices;

namespace LinearAlgebra
{
    // Metric-direction helpers for Query's nearest/farthest/k-selection kernels. They take no
    // fProxy parameter (or return fProxy), so hosting them on the merged float+double `Query`
    // partial would collide (CS0111). Here they emit as distinct floatQueryCore/doubleQueryCore.
    // IsSimilarityMetric's rule (Cosine || Dot) is float/double-specific -- the integer variant
    // (Dot only) lives in iProxyQueryCore.
    internal static partial class fProxyQueryCore
    {
        // Similarity metrics (Cosine, Dot): higher score = nearer.
        // Distance metrics (Manhattan, Euclidean, SqEuclidean, Chebyshev): lower score = nearer.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsSimilarityMetric(Metric m) => m == Metric.Cosine || m == Metric.Dot;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static fProxy WorstScoreForNearest(Metric m)
            => IsSimilarityMetric(m) ? fProxy.MinValue : fProxy.MaxValue;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static fProxy WorstScoreForFarthest(Metric m)
            => IsSimilarityMetric(m) ? fProxy.MaxValue : fProxy.MinValue;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsBetterForNearest(fProxy a, fProxy b, Metric m)
            => IsSimilarityMetric(m) ? a > b : a < b;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsBetterForFarthest(fProxy a, fProxy b, Metric m)
            => IsSimilarityMetric(m) ? a < b : a > b;
    }
}
