using System;
using Unity.Mathematics;
using LinearAlgebra.Stats;

namespace LinearAlgebra.ML
{
    /// <summary>
    /// Principal Component Analysis over a data matrix X (rows = samples n, columns = features p —
    /// matches StatsOP's orientation). Four fit routes, a 2x2 of fast/accurate x full/partial:
    ///
    ///   pcaCovariance    — full, fast. Eigendecomposes the p x p covariance/correlation matrix. The
    ///                      only route that handles WIDE data (p > n). Squares the condition number
    ///                      (kappa^2) -- prefer pcaSVD for near-degenerate data.
    ///   pcaSVD           — full, accurate. SVD of the centered data directly (no Gram, no kappa^2).
    ///                      Requires n >= p.
    ///   pcaSVDTruncated  — partial (top-k), EXACT. Golub-Kahan-Lanczos top-k SVD. Requires n >= p
    ///                      (svdTruncated itself enforces this -- see the method doc below).
    ///   pcaRandomized    — partial (top-k), fast-approximate (Halko-Martinsson-Tropp). Requires n >= p.
    ///                      Pays off only when p is large and k &lt;&lt; p.
    ///
    /// Every fit returns/fills a <see cref="fProxyPCAModel"/> (arena-owned buffers): the caller keeps it
    /// to project new data via <see cref="pcaTransform"/> or to read axes/variances for reduction work.
    ///
    /// THE denominator convention (makes pcaCovariance and pcaSVD agree on explainedVariance): both use
    /// the SAMPLE (n-1) convention, NOT StatsOP.standardizeColumns'/colVariance's population (n) one:
    ///   Covariance mode:  explainedVariance[i] = S[i]^2 / (n-1)  (== the covariance-matrix eigenvalues).
    ///   Correlation mode: standardize with SAMPLE std-dev sqrt(Sigma(x-mean)^2/(n-1)); then
    ///                     S[i]^2/(n-1) == the correlation-matrix eigenvalues.
    ///
    /// THE correlation degenerate-feature trap: pcaCovariance(Correlation) builds its own correlation
    /// matrix R = Cov ./ (sampleStd (x) sampleStd) inline, zeroing the ENTIRE row/column (including the
    /// diagonal) of a zero-variance feature. It deliberately does NOT reuse StatsOP.correlation(), which
    /// puts a spurious 1 on that diagonal -- that would emit a unit eigenvalue the SVD route can't match
    /// (its all-zero standardized column emits 0), breaking cross-route agreement.
    ///
    /// totalVariance (denominator of explainedVarianceRatio) is always computed directly from the data in
    /// one pass -- NEVER as the sum of the (possibly truncated) returned explainedVariance.
    ///
    /// Sign convention: for each component column, the largest-|entry| (first index wins ties) is made
    /// positive; applied once after the solve, SKIPPED when the solve did not converge. This does NOT
    /// resolve degenerate/repeated-eigenvalue rotation ambiguity -- keep cross-route/determinism
    /// expectations to well-separated spectra.
    ///
    /// No fProxyPCA_WS: PCA fits once (no per-frame restart loop like k-means), so each method allocates
    /// its own scratch from X's arena TEMP pool and calls the wrapped kernels' existing non-WS overloads.
    /// Realtime pattern: allocate the model once via Arena.fProxyPCAModel(p, k), call the `ref` fit each
    /// frame, ClearTemp() at end of frame reclaims all internal scratch.
    ///
    /// Deterministic by default: pcaRandomized / pcaSVDTruncated forward the exact default seed
    /// (0x9E3779B1u) svdRandomized/svdTruncated already use, so default calls are bitwise-reproducible.
    /// </summary>
    public static partial class fProxyPCA_OP
    {
        // =====================================================================================
        // Shared guards (managed throws, before any alloc — mirrors the k-means guard-before-alloc rule)
        // =====================================================================================

        static void RequireBasicShape(in fProxyMxN X, string method)
        {
            if (X.M_Rows < 2)
                throw new ArgumentException(method + ": X.M_Rows (n) must be >= 2 (variance is undefined for n<2)");
            if (X.N_Cols < 1)
                throw new ArgumentException(method + ": X.N_Cols (p) must be >= 1");
        }

        // svdThin/svdRandomized/svdTruncated all require m (samples) >= n (features) internally; PCA
        // surfaces that as a guard with a clearer message BEFORE touching the temp pool (rather than
        // letting the kernel throw its own generic message after Xc has already been allocated).
        static void RequireTallShape(in fProxyMxN X, string method)
        {
            if (X.M_Rows < X.N_Cols)
                throw new ArgumentException(method + ": requires samples>=features (X.M_Rows >= X.N_Cols); use pcaCovariance for wide data");
        }

        static void RequireTopK(int k, int n, int p, string method)
        {
            if (k <= 0 || k > math.min(n, p))
                throw new ArgumentException(method + ": k must be in (0, min(n, p)]");
        }

        // model shape guard for the two FULL routes (pcaCovariance/pcaSVD): components p x p,
        // explainedVariance/-Ratio length p, mean/scale length p.
        static void RequireModelShapeFull(in fProxyPCAModel model, int p, string method)
        {
            if (model.components.M_Rows != p || model.components.N_Cols != p)
                throw new ArgumentException(method + ": model.components must be p x p");
            if (model.explainedVariance.N != p)
                throw new ArgumentException(method + ": model.explainedVariance.N must equal p");
            if (model.explainedVarianceRatio.N != p)
                throw new ArgumentException(method + ": model.explainedVarianceRatio.N must equal p");
            if (model.mean.N != p)
                throw new ArgumentException(method + ": model.mean.N must equal p");
            if (model.scale.N != p)
                throw new ArgumentException(method + ": model.scale.N must equal p");
        }

        // model shape guard for the two TOP-K routes (pcaSVDTruncated/pcaRandomized): components p x k,
        // explainedVariance/-Ratio length k, mean/scale length p.
        static void RequireModelShapeTopK(in fProxyPCAModel model, int p, int k, string method)
        {
            if (model.components.M_Rows != p || model.components.N_Cols != k)
                throw new ArgumentException(method + ": model.components must be p x k");
            if (model.explainedVariance.N != k)
                throw new ArgumentException(method + ": model.explainedVariance.N must equal k");
            if (model.explainedVarianceRatio.N != k)
                throw new ArgumentException(method + ": model.explainedVarianceRatio.N must equal k");
            if (model.mean.N != p)
                throw new ArgumentException(method + ": model.mean.N must equal p");
            if (model.scale.N != p)
                throw new ArgumentException(method + ": model.scale.N must equal p");
        }

        // =====================================================================================
        // Shared helpers
        // =====================================================================================

        // Fixes the sign ambiguity of eigen/singular vectors: for each component column, the
        // largest-|entry| (first index wins ties) is made positive. Callers must skip this when the
        // solve did not converge (outputs undefined; NaNs would make the abs-scan meaningless).
        static void ApplySignConvention(ref fProxyMxN components)
        {
            int p = components.M_Rows;
            int k = components.N_Cols;
            for (int c = 0; c < k; c++)
            {
                int bestRow = 0;
                fProxy bestAbs = math.abs(components[0, c]);
                for (int r = 1; r < p; r++)
                {
                    fProxy a = math.abs(components[r, c]);
                    if (a > bestAbs) { bestAbs = a; bestRow = r; }
                }
                if (components[bestRow, c] < (fProxy)0)
                {
                    for (int r = 0; r < p; r++)
                        components[r, c] = -components[r, c];
                }
            }
        }

        // Per-feature SAMPLE std-dev std[c] = sqrt(Sigma_r (X[r,c]-mean[c])^2 / (n-1)), direct
        // row-major accumulation. Shared by pcaCovariance (Correlation) AND BuildWorkingCopy so BOTH
        // routes compute std with bit-identical roundoff -> a borderline-constant column is classified
        // degenerate (std==0) identically by both, keeping the cross-route oracle honest (a Gram-derived
        // sqrt(C[j,j]) could disagree with a direct row-sum on which feature underflows to zero).
        static void ComputeSampleStd(in fProxyMxN X, in fProxyN mean, ref fProxyN std)
        {
            int n = X.M_Rows;
            int p = X.N_Cols;
            for (int c = 0; c < p; c++)
            {
                fProxy ss = (fProxy)0;
                for (int r = 0; r < n; r++) { fProxy d = X[r, c] - mean[c]; ss += d * d; }
                std[c] = math.sqrt(ss / (fProxy)(n - 1));
            }
        }

        // Builds the working copy Xc (n x p) per the denominator trap: center only (Covariance) or
        // center-and-divide by the SAMPLE std-dev (Correlation; degenerate feature -> scale=1, zero
        // column). Fills `scale`. Returns totalVariance computed directly from X (never from the
        // possibly-truncated explainedVariance the caller will fill afterward). `mean` must already be
        // filled (colMean). Used by pcaSVD / pcaSVDTruncated / pcaRandomized (NOT pcaCovariance, which
        // builds its own p x p Cov/R matrix instead of an n x p working copy).
        static fProxy BuildWorkingCopy(in fProxyMxN X, PCAScaling scaling, in fProxyN mean,
                                       ref fProxyN scale, ref fProxyMxN Xc)
        {
            int n = X.M_Rows;
            int p = X.N_Cols;
            fProxy totalVariance = (fProxy)0;

            if (scaling == PCAScaling.Covariance)
            {
                for (int c = 0; c < p; c++) scale[c] = (fProxy)1;

                for (int r = 0; r < n; r++)
                    for (int c = 0; c < p; c++)
                        Xc[r, c] = X[r, c] - mean[c];

                // totalVariance = trace(covariance) = Sigma_j sampleVar(j), computed directly from Xc.
                for (int c = 0; c < p; c++)
                {
                    fProxy ss = (fProxy)0;
                    for (int r = 0; r < n; r++) { fProxy d = Xc[r, c]; ss += d * d; }
                    totalVariance += ss / (fProxy)(n - 1);
                }
            }
            else // Correlation
            {
                var sampleStd = X.tempfProxyVec(p);
                ComputeSampleStd(in X, in mean, ref sampleStd);

                for (int c = 0; c < p; c++)
                    scale[c] = sampleStd[c] > (fProxy)0 ? sampleStd[c] : (fProxy)1;

                for (int r = 0; r < n; r++)
                    for (int c = 0; c < p; c++)
                        Xc[r, c] = sampleStd[c] > (fProxy)0 ? (X[r, c] - mean[c]) / sampleStd[c] : (fProxy)0;

                // totalVariance = # non-degenerate features (each standardized column has sample var 1).
                for (int c = 0; c < p; c++)
                    if (sampleStd[c] > (fProxy)0) totalVariance += (fProxy)1;
            }

            return totalVariance;
        }

        // Converts singular values S[0..k) in-place to variances: S[i] <- S[i]^2 / (n-1) (the
        // denominator-trap scaling shared by pcaSVD / pcaSVDTruncated / pcaRandomized).
        static void SingularValuesToVariances(ref fProxyN S, int k, int n)
        {
            for (int i = 0; i < k; i++)
            {
                fProxy s = S[i];
                S[i] = s * s / (fProxy)(n - 1);
            }
        }

        // Shared tail: set k/converged, fill explainedVarianceRatio, apply the sign convention (skipped
        // when !converged). model.explainedVariance must already hold the final variances.
        static void FinalizeModel(ref fProxyPCAModel model, int k, fProxy totalVariance, bool converged)
        {
            model.k = k;
            model.converged = converged;

            // On !converged the value outputs (explainedVariance, explainedVarianceRatio, components) are
            // left UNDEFINED — matching the wrapped kernels' contract and the model doc; only
            // mean/scale/k/converged are guaranteed. (Earlier this zeroed ONLY the ratio, which produced a
            // self-inconsistent state: ratio 0 beside a garbage/NaN variance.)
            if (converged)
            {
                for (int i = 0; i < k; i++)
                    model.explainedVarianceRatio[i] = totalVariance > (fProxy)0
                        ? model.explainedVariance[i] / totalVariance
                        : (fProxy)0;
                ApplySignConvention(ref model.components);
            }
        }

        // =====================================================================================
        // 1. pcaCovariance — full, fast (covariance/correlation eigensolve; handles wide p>n)
        // =====================================================================================

        /// <summary>
        /// Full PCA via eigendecomposition of the p x p covariance (or correlation) matrix built from X.
        /// The only route that handles WIDE data (p > n): a p x p eigensolve is cheap regardless of n.
        /// Squares the condition number (kappa^2) relative to pcaSVD -- prefer pcaSVD for near-degenerate
        /// data. model.k is set to p (X.N_Cols). Correlation mode builds its own R matrix inline (see
        /// the type doc's degenerate-feature trap) rather than reusing StatsOP.correlation().
        /// </summary>
        public static bool pcaCovariance(in fProxyMxN X, ref fProxyPCAModel model, PCAScaling scaling)
        {
            const string method = "fProxyPCA_OP.pcaCovariance";
            RequireBasicShape(in X, method);

            int p = X.N_Cols;
            RequireModelShapeFull(in model, p, method);

            fProxyStats_OP.colMean(in X, ref model.mean);

            // C is PCA-built scratch each call (temp pool) -- eigenSymmetric destroys it.
            var C = X.tempfProxyMat(p, p);
            fProxyStats_OP.covarianceInto(in X, ref C);

            fProxy totalVariance;

            if (scaling == PCAScaling.Covariance)
            {
                for (int j = 0; j < p; j++) model.scale[j] = (fProxy)1;

                totalVariance = (fProxy)0;
                for (int j = 0; j < p; j++) totalVariance += C[j, j];
            }
            else // Correlation
            {
                // Direct row-sum sampleStd (NOT sqrt(C[j,j])) so degenerate detection is bit-identical
                // to BuildWorkingCopy's — see ComputeSampleStd.
                var sampleStd = X.tempfProxyVec(p);
                ComputeSampleStd(in X, in model.mean, ref sampleStd);

                for (int j = 0; j < p; j++)
                    model.scale[j] = sampleStd[j] > (fProxy)0 ? sampleStd[j] : (fProxy)1;

                totalVariance = (fProxy)0;
                for (int j = 0; j < p; j++)
                    if (sampleStd[j] > (fProxy)0) totalVariance += (fProxy)1;

                // Build R = Cov ./ (sampleStd (x) sampleStd) inline, in place over C. The DIAGONAL of a
                // non-degenerate feature is set to EXACTLY 1 (the correlation-matrix property) rather than
                // C[j,j]/(sj*sj), which is only ~1 now that sampleStd is a direct row-sum, not sqrt(C[j,j]).
                // A zero-variance feature zeroes its ENTIRE row/column, including the diagonal (NOT
                // StatsOP.correlation's convention of 1 there — see the type doc's degenerate-feature trap).
                // Each cell (i,j) is visited exactly once so the in-place overwrite never reads an
                // already-transformed neighbor. Stays exactly symmetric (C[i,j]==C[j,i], si*sj==sj*si) so
                // eigenSymmetric's symmetry check passes.
                for (int i = 0; i < p; i++)
                {
                    fProxy si = sampleStd[i];
                    for (int j = 0; j < p; j++)
                    {
                        fProxy sj = sampleStd[j];
                        if (i == j)
                            C[i, j] = si > (fProxy)0 ? (fProxy)1 : (fProxy)0;
                        else
                            C[i, j] = (si > (fProxy)0 && sj > (fProxy)0) ? C[i, j] / (si * sj) : (fProxy)0;
                    }
                }
            }

            bool converged = Eigen.eigenSymmetric(ref C, ref model.explainedVariance, ref model.components);

            FinalizeModel(ref model, p, totalVariance, converged);
            return converged;
        }

        /// <summary>pcaCovariance with scaling = PCAScaling.Covariance.</summary>
        public static bool pcaCovariance(in fProxyMxN X, ref fProxyPCAModel model)
            => pcaCovariance(in X, ref model, PCAScaling.Covariance);

        /// <summary>
        /// Validates inputs, allocates the model (p x p) from <paramref name="arena"/>, then delegates
        /// to the ref-model overload. Guards fire before any arena allocation.
        /// </summary>
        public static fProxyPCAModel pcaCovariance(ref Arena arena, in fProxyMxN X, PCAScaling scaling)
        {
            const string method = "fProxyPCA_OP.pcaCovariance";
            RequireBasicShape(in X, method);

            int p = X.N_Cols;
            var model = arena.fProxyPCAModel(p, p);
            pcaCovariance(in X, ref model, scaling);
            return model;
        }

        /// <summary>pcaCovariance (allocating) with scaling = PCAScaling.Covariance.</summary>
        public static fProxyPCAModel pcaCovariance(ref Arena arena, in fProxyMxN X)
            => pcaCovariance(ref arena, in X, PCAScaling.Covariance);

        // =====================================================================================
        // 2. pcaSVD — full, accurate (svdThin on a centered/standardized copy; requires n >= p)
        // =====================================================================================

        /// <summary>
        /// Full PCA via SVD of the centered (or standardized) data directly -- no Gram matrix, no
        /// kappa^2 accuracy loss. Requires X.M_Rows (n) >= X.N_Cols (p); throws otherwise (use
        /// pcaCovariance for wide data). model.k is set to p.
        /// </summary>
        public static bool pcaSVD(in fProxyMxN X, ref fProxyPCAModel model, PCAScaling scaling, int maxIter)
        {
            const string method = "fProxyPCA_OP.pcaSVD";
            RequireBasicShape(in X, method);
            RequireTallShape(in X, method);

            int n = X.M_Rows;
            int p = X.N_Cols;
            RequireModelShapeFull(in model, p, method);

            fProxyStats_OP.colMean(in X, ref model.mean);

            var Xc = X.tempfProxyMat(n, p);
            fProxy totalVariance = BuildWorkingCopy(in X, scaling, in model.mean, ref model.scale, ref Xc);

            // svdThin reads A while writing U -- a SEPARATE n x p temp, never Xc itself.
            var U = X.tempfProxyMat(n, p);
            bool converged = SVD.svdThin(in Xc, ref U, ref model.explainedVariance, ref model.components, maxIter);

            if (converged)
                SingularValuesToVariances(ref model.explainedVariance, p, n);

            FinalizeModel(ref model, p, totalVariance, converged);
            return converged;
        }

        /// <summary>pcaSVD with maxIter = svdThin's default (75).</summary>
        public static bool pcaSVD(in fProxyMxN X, ref fProxyPCAModel model, PCAScaling scaling)
            => pcaSVD(in X, ref model, scaling, 75);

        /// <summary>pcaSVD with scaling = PCAScaling.Covariance and maxIter = 75.</summary>
        public static bool pcaSVD(in fProxyMxN X, ref fProxyPCAModel model)
            => pcaSVD(in X, ref model, PCAScaling.Covariance, 75);

        /// <summary>
        /// Validates inputs, allocates the model (p x p) from <paramref name="arena"/>, then delegates
        /// to the ref-model overload. Guards fire before any arena allocation.
        /// </summary>
        public static fProxyPCAModel pcaSVD(ref Arena arena, in fProxyMxN X, PCAScaling scaling)
        {
            const string method = "fProxyPCA_OP.pcaSVD";
            RequireBasicShape(in X, method);
            RequireTallShape(in X, method);

            int p = X.N_Cols;
            var model = arena.fProxyPCAModel(p, p);
            pcaSVD(in X, ref model, scaling, 75);
            return model;
        }

        /// <summary>pcaSVD (allocating) with scaling = PCAScaling.Covariance.</summary>
        public static fProxyPCAModel pcaSVD(ref Arena arena, in fProxyMxN X)
            => pcaSVD(ref arena, in X, PCAScaling.Covariance);

        // =====================================================================================
        // 3. pcaSVDTruncated — exact top-k (Golub-Kahan-Lanczos)
        // =====================================================================================
        //
        // NOTE ON SHAPE: SVD.svdTruncated's core overload enforces A.M_Rows >= A.N_Cols internally
        // (same requirement as svdThin/svdRandomized) -- it is NOT shape-free despite appearances from
        // its `k <= N_Cols` guard alone. PCA therefore applies the same RequireTallShape guard as
        // pcaSVD/pcaRandomized, BEFORE allocating Xc, so invalid (wide) input never touches the temp
        // pool. pcaSVDTruncated does NOT support wide (p>n) data.

        /// <summary>
        /// Exact top-k PCA via Golub-Kahan-Lanczos bidiagonalization (SVD.svdTruncated) on the centered
        /// (or standardized) data -- avoids the full O(n p^2) SVD when only the leading few components
        /// are needed. Requires X.M_Rows (n) >= X.N_Cols (p) (see the shape note above); 0 &lt; k &lt;=
        /// min(n, p). model.k is set to k. explainedVarianceRatio is computed against the FULL
        /// totalVariance, so top-k ratios sum to less than 1.
        /// </summary>
        public static bool pcaSVDTruncated(in fProxyMxN X, ref fProxyPCAModel model, int k,
                                            PCAScaling scaling, int oversample, uint seed, int maxIter)
        {
            const string method = "fProxyPCA_OP.pcaSVDTruncated";
            RequireBasicShape(in X, method);
            RequireTallShape(in X, method);

            int n = X.M_Rows;
            int p = X.N_Cols;
            RequireTopK(k, n, p, method);
            RequireModelShapeTopK(in model, p, k, method);

            fProxyStats_OP.colMean(in X, ref model.mean);

            var Xc = X.tempfProxyMat(n, p);
            fProxy totalVariance = BuildWorkingCopy(in X, scaling, in model.mean, ref model.scale, ref Xc);

            var Uk = X.tempfProxyMat(n, k);
            SVD.svdTruncated(in Xc, ref Uk, ref model.explainedVariance, ref model.components,
                              k, oversample, seed, maxIter, out bool converged);

            if (converged)
                SingularValuesToVariances(ref model.explainedVariance, k, n);

            FinalizeModel(ref model, k, totalVariance, converged);
            return converged;
        }

        /// <summary>
        /// pcaSVDTruncated forwarding svdTruncated's own "generous default Krylov width" formula
        /// verbatim (oversample = max(k, 12), so p = min(n, max(2k, k+12))), its default seed
        /// (0x9E3779B1u) and default maxIter (75).
        /// </summary>
        public static bool pcaSVDTruncated(in fProxyMxN X, ref fProxyPCAModel model, int k, PCAScaling scaling)
            => pcaSVDTruncated(in X, ref model, k, scaling, math.max(k, 12), 0x9E3779B1u, 75);

        /// <summary>pcaSVDTruncated with scaling = PCAScaling.Covariance.</summary>
        public static bool pcaSVDTruncated(in fProxyMxN X, ref fProxyPCAModel model, int k)
            => pcaSVDTruncated(in X, ref model, k, PCAScaling.Covariance);

        /// <summary>
        /// Validates inputs, allocates the model (p x k) from <paramref name="arena"/>, then delegates
        /// to the ref-model overload. Guards fire before any arena allocation.
        /// </summary>
        public static fProxyPCAModel pcaSVDTruncated(ref Arena arena, in fProxyMxN X, int k, PCAScaling scaling)
        {
            const string method = "fProxyPCA_OP.pcaSVDTruncated";
            RequireBasicShape(in X, method);
            RequireTallShape(in X, method);

            int n = X.M_Rows;
            int p = X.N_Cols;
            RequireTopK(k, n, p, method);

            var model = arena.fProxyPCAModel(p, k);
            pcaSVDTruncated(in X, ref model, k, scaling);
            return model;
        }

        /// <summary>pcaSVDTruncated (allocating) with scaling = PCAScaling.Covariance.</summary>
        public static fProxyPCAModel pcaSVDTruncated(ref Arena arena, in fProxyMxN X, int k)
            => pcaSVDTruncated(ref arena, in X, k, PCAScaling.Covariance);

        // =====================================================================================
        // 4. pcaRandomized — approximate top-k (Halko-Martinsson-Tropp; requires n >= p)
        // =====================================================================================

        /// <summary>
        /// Approximate top-k PCA via randomized SVD (SVD.svdRandomized) on the centered (or
        /// standardized) data. Pays off when p is large and k &lt;&lt; p; at small p prefer
        /// pcaCovariance, for exact top-k prefer pcaSVDTruncated. Requires X.M_Rows (n) >= X.N_Cols (p);
        /// throws otherwise (use pcaCovariance for wide data). 0 &lt; k &lt;= min(n, p). model.k is set
        /// to k. Deterministic by default (seed = 0x9E3779B1u, svdRandomized's own default).
        /// </summary>
        public static bool pcaRandomized(in fProxyMxN X, ref fProxyPCAModel model, int k,
                                          PCAScaling scaling, int oversample, int powerIters, uint seed, int maxIter)
        {
            const string method = "fProxyPCA_OP.pcaRandomized";
            RequireBasicShape(in X, method);
            RequireTallShape(in X, method);

            int n = X.M_Rows;
            int p = X.N_Cols;
            RequireTopK(k, n, p, method);
            RequireModelShapeTopK(in model, p, k, method);

            fProxyStats_OP.colMean(in X, ref model.mean);

            var Xc = X.tempfProxyMat(n, p);
            fProxy totalVariance = BuildWorkingCopy(in X, scaling, in model.mean, ref model.scale, ref Xc);

            var Uk = X.tempfProxyMat(n, k);
            bool converged = SVD.svdRandomized(in Xc, ref Uk, ref model.explainedVariance, ref model.components,
                                                k, oversample, powerIters, seed, maxIter);

            if (converged)
                SingularValuesToVariances(ref model.explainedVariance, k, n);

            FinalizeModel(ref model, k, totalVariance, converged);
            return converged;
        }

        /// <summary>pcaRandomized forwarding svdRandomized's own defaults verbatim: oversample=10,
        /// powerIters=2, seed=0x9E3779B1u, maxIter=75.</summary>
        public static bool pcaRandomized(in fProxyMxN X, ref fProxyPCAModel model, int k, PCAScaling scaling)
            => pcaRandomized(in X, ref model, k, scaling, 10, 2, 0x9E3779B1u, 75);

        /// <summary>pcaRandomized with scaling = PCAScaling.Covariance.</summary>
        public static bool pcaRandomized(in fProxyMxN X, ref fProxyPCAModel model, int k)
            => pcaRandomized(in X, ref model, k, PCAScaling.Covariance);

        /// <summary>
        /// Validates inputs, allocates the model (p x k) from <paramref name="arena"/>, then delegates
        /// to the ref-model overload. Guards fire before any arena allocation.
        /// </summary>
        public static fProxyPCAModel pcaRandomized(ref Arena arena, in fProxyMxN X, int k, PCAScaling scaling)
        {
            const string method = "fProxyPCA_OP.pcaRandomized";
            RequireBasicShape(in X, method);
            RequireTallShape(in X, method);

            int n = X.M_Rows;
            int p = X.N_Cols;
            RequireTopK(k, n, p, method);

            var model = arena.fProxyPCAModel(p, k);
            pcaRandomized(in X, ref model, k, scaling);
            return model;
        }

        /// <summary>pcaRandomized (allocating) with scaling = PCAScaling.Covariance.</summary>
        public static fProxyPCAModel pcaRandomized(ref Arena arena, in fProxyMxN X, int k)
            => pcaRandomized(ref arena, in X, k, PCAScaling.Covariance);

        // =====================================================================================
        // Projection
        // =====================================================================================

        /// <summary>
        /// Projects X onto the model's principal axes: scores[i,:] = ((X[i,:] - model.mean) /
        /// model.scale) . model.components. Zero-alloc except for one n_new x p temp scratch matrix.
        /// X.N_Cols must match the model's feature count (model.mean/scale.N and
        /// model.components.M_Rows); model.k must equal model.components.N_Cols (defends a
        /// hand-assembled or stale model); scores must be X.M_Rows x model.k.
        /// </summary>
        public static void pcaTransform(in fProxyMxN X, in fProxyPCAModel model, ref fProxyMxN scores)
        {
            const string method = "fProxyPCA_OP.pcaTransform";
            int p = X.N_Cols;
            int nNew = X.M_Rows;

            if (model.mean.N != p || model.scale.N != p || model.components.M_Rows != p)
                throw new ArgumentException(method + ": model.mean/scale/components must match X.N_Cols (p)");
            if (model.k != model.components.N_Cols)
                throw new ArgumentException(method + ": model.k must equal model.components.N_Cols (stale model)");
            if (scores.M_Rows != nNew || scores.N_Cols != model.k)
                throw new ArgumentException(method + ": scores must be X.M_Rows x model.k");

            var Xs = X.tempfProxyMat(nNew, p);
            for (int r = 0; r < nNew; r++)
                for (int c = 0; c < p; c++)
                    Xs[r, c] = (X[r, c] - model.mean[c]) / model.scale[c];

            Linear_OP.dot(in Xs, in model.components, ref scores);
        }

        /// <summary>Allocating pcaTransform: allocates and returns a fresh X.M_Rows x model.k scores matrix.</summary>
        public static fProxyMxN pcaTransform(ref Arena arena, in fProxyMxN X, in fProxyPCAModel model)
        {
            const string method = "fProxyPCA_OP.pcaTransform";
            int p = X.N_Cols;

            if (model.mean.N != p || model.scale.N != p || model.components.M_Rows != p)
                throw new ArgumentException(method + ": model.mean/scale/components must match X.N_Cols (p)");
            if (model.k != model.components.N_Cols)
                throw new ArgumentException(method + ": model.k must equal model.components.N_Cols (stale model)");

            var scores = arena.fProxyMat(X.M_Rows, model.k);
            pcaTransform(in X, in model, ref scores);
            return scores;
        }
    }
}
