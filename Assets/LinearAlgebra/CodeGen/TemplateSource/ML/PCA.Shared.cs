using System;
using Unity.Mathematics;

namespace BULA.ML
{
    // Type-agnostic guard factored out of the per-type PCA template so the merged partial
    // class (float+double emit the same `PCA`) holds this int-only signature exactly once.
    public static partial class PCA
    {
        static void RequireTopK(int k, int n, int p, string method)
        {
            if (k <= 0 || k > math.min(n, p))
                throw new ArgumentException(method + ": k must be in (0, min(n, p)]");
        }
    }
}
