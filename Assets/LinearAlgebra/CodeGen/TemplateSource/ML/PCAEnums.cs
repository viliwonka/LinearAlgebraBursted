//singularFile//
namespace LinearAlgebra.ML
{
    // Scaling mode for the PCA fit routes (fitCov / fitSvd / fitSvdTruncated / fitRandomized).
    // Default (via the forwarding overloads) is Covariance (= sklearn's center-only default).
    // Covariance  = center only (PCA on the covariance matrix). model.scale output = all ones.
    // Correlation = center AND divide by per-feature SAMPLE std-dev (PCA on the correlation matrix).
    //   model.scale output = the sample std-devs; a zero-variance feature gets scale = 1 (never a
    //   divide-by-zero) so it maps to an all-zero standardized column and contributes a zero component.
    public enum PCAScaling { Covariance, Correlation }
}
