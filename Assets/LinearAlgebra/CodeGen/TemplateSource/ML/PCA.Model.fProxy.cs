using LinearAlgebra.ML;

namespace LinearAlgebra.ML
{
    /// <summary>
    /// A fitted PCA model: the axes/variances needed to project new data (<see cref="fProxyPCA_OP.pcaTransform"/>)
    /// or to read off variances for a reduction decision. Every <c>fProxyPCA_OP</c> fit route (pcaCovariance /
    /// pcaSVD / pcaSVDTruncated / pcaRandomized) fills one of these. Allocate via
    /// <c>Arena.fProxyPCAModel(p, k)</c> (p = X.N_Cols features, k = number of components) and reuse across
    /// same-shape fits (realtime pattern: fit each frame into the same model, <c>ClearTemp()</c> reclaims the
    /// internal scratch each fit allocates from the arena's temp pool).
    ///
    /// This is a buffer-carrying (fProxy-prefixed) struct rather than a plain scalar diagnostics struct
    /// (like <c>SolveInfo</c>/<c>EigenSolveInfo</c>) because PCA has a downstream <c>pcaTransform</c> stage
    /// that consumes <c>mean</c>/<c>scale</c>/<c>components</c>/<c>k</c> together as a unit — the same
    /// justification every <c>_WS</c> (<c>fProxyKMeans_WS</c>, <c>fProxySVDThin_WS</c>) already has for
    /// bundling arena buffer handles into one struct.
    /// </summary>
    public struct fProxyPCAModel
    {
        /// <summary>p x k. Column i is the i-th principal axis (unit-norm, sign-fixed — see
        /// <see cref="fProxyPCA_OP"/>'s sign convention). Undefined if <see cref="converged"/> is false.</summary>
        public fProxyMxN components;

        /// <summary>Length k. Variance captured by each component, DESCENDING. Undefined if
        /// <see cref="converged"/> is false.</summary>
        public fProxyN explainedVariance;

        /// <summary>Length k. explainedVariance[i] / totalVariance (totalVariance computed directly from
        /// the fit data, never as the sum of a possibly-truncated explainedVariance). Undefined if
        /// <see cref="converged"/> is false.</summary>
        public fProxyN explainedVarianceRatio;

        /// <summary>Length p. Per-feature mean, needed to center new data in <c>pcaTransform</c>.</summary>
        public fProxyN mean;

        /// <summary>Length p. Per-feature divisor applied before projecting: all ones (PCAScaling.Covariance)
        /// or the per-feature SAMPLE std-dev (PCAScaling.Correlation; a zero-variance feature gets 1, never
        /// a divide-by-zero).</summary>
        public fProxyN scale;

        /// <summary>Number of components: == p (X.N_Cols) for the full routes (pcaCovariance/pcaSVD), or the
        /// requested top-k for pcaSVDTruncated/pcaRandomized.</summary>
        public int k;

        /// <summary>Underlying eigensolve/SVD convergence flag. All other fields (except mean/scale, which
        /// are filled unconditionally) are undefined when this is false.</summary>
        public bool converged;

        /// <summary>Same value as <see cref="converged"/>; use whichever reads better at the call site.</summary>
        public bool Solved => converged;

        /// <summary>Implicit success test, so <c>if (model)</c> reads naturally after a fit.</summary>
        public static implicit operator bool(fProxyPCAModel m) => m.converged;
    }
}

namespace LinearAlgebra
{
    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates a PCA model sized for <paramref name="p"/> features and <paramref name="k"/>
        /// components. All buffers are persistent in this arena (disposed with it).
        /// </summary>
        public static LinearAlgebra.ML.fProxyPCAModel fProxyPCAModel(this ref Arena arena, int p, int k)
        {
            return new LinearAlgebra.ML.fProxyPCAModel
            {
                components             = arena.fProxyMat(p, k),
                explainedVariance       = arena.fProxyVec(k),
                explainedVarianceRatio  = arena.fProxyVec(k),
                mean                    = arena.fProxyVec(p),
                scale                   = arena.fProxyVec(p),
                k                       = k,
                converged               = false
            };
        }
    }
}
