using System;
using Unity.Mathematics;

namespace LinearAlgebra.ML
{
    /// <summary>
    /// Principal Component Analysis over a data matrix X (rows = samples n, columns = features p —
    /// matches StatsOP's orientation). Four fit routes, a 2x2 of fast/accurate x full/partial:
    ///
    ///   fitCov           — full, fast. Eigendecomposes the p x p covariance/correlation matrix. The
    ///                      only route that handles WIDE data (p > n). Squares the condition number
    ///                      (kappa^2) -- prefer fitSvd for near-degenerate data.
    ///   fitSvd           — full, accurate. SVD of the centered data directly (no Gram, no kappa^2).
    ///                      Requires n >= p.
    ///   fitSvdTruncated  — partial (top-k), EXACT. Golub-Kahan-Lanczos top-k SVD. Requires n >= p
    ///                      (SVD.truncated itself enforces this -- see the method doc below).
    ///   fitRandomized    — partial (top-k), fast-approximate (Halko-Martinsson-Tropp). Requires n >= p.
    ///                      Pays off only when p is large and k &lt;&lt; p.
    ///
    /// Every fit returns/fills a <see cref="doublePCAModel"/> (arena-owned buffers): the caller keeps it
    /// to project new data via <see cref="transform"/> or to read axes/variances for reduction work.
    ///
    /// THE denominator convention (makes fitCov and fitSvd agree on explainedVariance): both use
    /// the SAMPLE (n-1) convention, NOT StatsOP.standardizeColumns'/colVariance's population (n) one:
    ///   Covariance mode:  explainedVariance[i] = S[i]^2 / (n-1)  (== the covariance-matrix eigenvalues).
    ///   Correlation mode: standardize with SAMPLE std-dev sqrt(Sigma(x-mean)^2/(n-1)); then
    ///                     S[i]^2/(n-1) == the correlation-matrix eigenvalues.
    ///
    /// THE correlation degenerate-feature trap: fitCov(Correlation) builds its own correlation
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
    /// No doublePCACache: PCA fits once (no per-frame restart loop like k-means), so each method allocates
    /// its own scratch from X's arena TEMP pool and calls the wrapped kernels' existing non-WS overloads.
    /// Realtime pattern: allocate the model once via Arena.doublePCAModel(p, k), call the `ref` fit each
    /// frame, ClearTemp() at end of frame reclaims all internal scratch.
    ///
    /// Deterministic by default: fitRandomized / fitSvdTruncated forward the exact default seed
    /// (0x9E3779B1u) SVD.randomized/SVD.truncated already use, so default calls are bitwise-reproducible.
    /// </summary>
    public static partial class PCA
    {
        // =====================================================================================
        // Shared guards (managed throws, before any alloc — mirrors the k-means guard-before-alloc rule)
        // =====================================================================================

        static void RequireBasicShape(in doubleMxN X, string method)
        {
            if (X.M_Rows < 2)
                throw new ArgumentException(method + ": X.M_Rows (n) must be >= 2 (variance is undefined for n<2)");
            if (X.N_Cols < 1)
                throw new ArgumentException(method + ": X.N_Cols (p) must be >= 1");
        }

        // SVD.thin/randomized/truncated all require m (samples) >= n (features) internally; PCA
        // surfaces that as a guard with a clearer message BEFORE touching the temp pool (rather than
        // letting the kernel throw its own generic message after Xc has already been allocated).
        static void RequireTallShape(in doubleMxN X, string method)
        {
            if (X.M_Rows < X.N_Cols)
                throw new ArgumentException(method + ": requires samples>=features (X.M_Rows >= X.N_Cols); use fitCov for wide data");
        }

        // RequireTopK lives in PCA.Shared.cs (type-agnostic, emitted once).

        // model shape guard for the two FULL routes (fitCov/fitSvd): components p x p,
        // explainedVariance/-Ratio length p, mean/scale length p.
        static void RequireModelShapeFull(in doublePCAModel model, int p, string method)
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

        // model shape guard for the two TOP-K routes (fitSvdTruncated/fitRandomized): components p x k,
        // explainedVariance/-Ratio length k, mean/scale length p.
        static void RequireModelShapeTopK(in doublePCAModel model, int p, int k, string method)
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
        static void ApplySignConvention(ref doubleMxN components)
        {
            int p = components.M_Rows;
            int k = components.N_Cols;
            for (int c = 0; c < k; c++)
            {
                int bestRow = 0;
                double bestAbs = math.abs(components[0, c]);
                for (int r = 1; r < p; r++)
                {
                    double a = math.abs(components[r, c]);
                    if (a > bestAbs) { bestAbs = a; bestRow = r; }
                }
                if (components[bestRow, c] < (double)0)
                {
                    for (int r = 0; r < p; r++)
                        components[r, c] = -components[r, c];
                }
            }
        }

        // Per-feature SAMPLE std-dev std[c] = sqrt(Sigma_r (X[r,c]-mean[c])^2 / (n-1)), direct
        // row-major accumulation. Shared by fitCov (Correlation) AND BuildWorkingCopy so BOTH
        // routes compute std with bit-identical roundoff -> a borderline-constant column is classified
        // degenerate (std==0) identically by both, keeping the cross-route oracle honest (a Gram-derived
        // sqrt(C[j,j]) could disagree with a direct row-sum on which feature underflows to zero).
        static void ComputeSampleStd(in doubleMxN X, in doubleN mean, ref doubleN std)
        {
            int n = X.M_Rows;
            int p = X.N_Cols;
            for (int c = 0; c < p; c++)
            {
                double ss = (double)0;
                for (int r = 0; r < n; r++) { double d = X[r, c] - mean[c]; ss += d * d; }
                std[c] = math.sqrt(ss / (double)(n - 1));
            }
        }

        // Builds the working copy Xc (n x p) per the denominator trap: center only (Covariance) or
        // center-and-divide by the SAMPLE std-dev (Correlation; degenerate feature -> scale=1, zero
        // column). Fills `scale`. Returns totalVariance computed directly from X (never from the
        // possibly-truncated explainedVariance the caller will fill afterward). `mean` must already be
        // filled (colMean). Used by fitSvd / fitSvdTruncated / fitRandomized (NOT fitCov, which
        // builds its own p x p Cov/R matrix instead of an n x p working copy).
        static double BuildWorkingCopy(in doubleMxN X, PCAScaling scaling, in doubleN mean,
                                       ref doubleN scale, ref doubleMxN Xc)
        {
            int n = X.M_Rows;
            int p = X.N_Cols;
            double totalVariance = (double)0;

            if (scaling == PCAScaling.Covariance)
            {
                for (int c = 0; c < p; c++) scale[c] = (double)1;

                for (int r = 0; r < n; r++)
                    for (int c = 0; c < p; c++)
                        Xc[r, c] = X[r, c] - mean[c];

                // totalVariance = trace(covariance) = Sigma_j sampleVar(j), computed directly from Xc.
                for (int c = 0; c < p; c++)
                {
                    double ss = (double)0;
                    for (int r = 0; r < n; r++) { double d = Xc[r, c]; ss += d * d; }
                    totalVariance += ss / (double)(n - 1);
                }
            }
            else // Correlation
            {
                var sampleStd = X.doubleTempVec(p);
                ComputeSampleStd(in X, in mean, ref sampleStd);

                for (int c = 0; c < p; c++)
                    scale[c] = sampleStd[c] > (double)0 ? sampleStd[c] : (double)1;

                for (int r = 0; r < n; r++)
                    for (int c = 0; c < p; c++)
                        Xc[r, c] = sampleStd[c] > (double)0 ? (X[r, c] - mean[c]) / sampleStd[c] : (double)0;

                // totalVariance = # non-degenerate features (each standardized column has sample var 1).
                for (int c = 0; c < p; c++)
                    if (sampleStd[c] > (double)0) totalVariance += (double)1;
            }

            return totalVariance;
        }

        // Converts singular values S[0..k) in-place to variances: S[i] <- S[i]^2 / (n-1) (the
        // denominator-trap scaling shared by fitSvd / fitSvdTruncated / fitRandomized).
        static void SingularValuesToVariances(ref doubleN S, int k, int n)
        {
            for (int i = 0; i < k; i++)
            {
                double s = S[i];
                S[i] = s * s / (double)(n - 1);
            }
        }

        // Shared tail: set k/converged, fill explainedVarianceRatio, apply the sign convention (skipped
        // when !converged). model.explainedVariance must already hold the final variances.
        static void FinalizeModel(ref doublePCAModel model, int k, double totalVariance, bool converged)
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
                    model.explainedVarianceRatio[i] = totalVariance > (double)0
                        ? model.explainedVariance[i] / totalVariance
                        : (double)0;
                ApplySignConvention(ref model.components);
            }
        }

        // =====================================================================================
        // 1. fitCov — full, fast (covariance/correlation eigensolve; handles wide p>n)
        // =====================================================================================

        /// <summary>
        /// Full PCA via eigendecomposition of the p x p covariance (or correlation) matrix built from X.
        /// The only route that handles WIDE data (p > n): a p x p eigensolve is cheap regardless of n.
        /// Squares the condition number (kappa^2) relative to fitSvd -- prefer fitSvd for near-degenerate
        /// data. model.k is set to p (X.N_Cols). Correlation mode builds its own R matrix inline (see
        /// the type doc's degenerate-feature trap) rather than reusing StatsOP.correlation().
        /// </summary>
        public static bool fitCov(in doubleMxN X, ref doublePCAModel model, PCAScaling scaling)
        {
            const string method = "PCA.fitCov";
            RequireBasicShape(in X, method);

            int p = X.N_Cols;
            RequireModelShapeFull(in model, p, method);

            Stats.colMean(in X, ref model.mean);

            // C is PCA-built scratch each call (temp pool) -- symmetric destroys it.
            var C = X.doubleTempMat(p, p);
            Stats.covarianceInto(in X, ref C);

            double totalVariance;

            if (scaling == PCAScaling.Covariance)
            {
                for (int j = 0; j < p; j++) model.scale[j] = (double)1;

                totalVariance = (double)0;
                for (int j = 0; j < p; j++) totalVariance += C[j, j];
            }
            else // Correlation
            {
                // Direct row-sum sampleStd (NOT sqrt(C[j,j])) so degenerate detection is bit-identical
                // to BuildWorkingCopy's — see ComputeSampleStd.
                var sampleStd = X.doubleTempVec(p);
                ComputeSampleStd(in X, in model.mean, ref sampleStd);

                for (int j = 0; j < p; j++)
                    model.scale[j] = sampleStd[j] > (double)0 ? sampleStd[j] : (double)1;

                totalVariance = (double)0;
                for (int j = 0; j < p; j++)
                    if (sampleStd[j] > (double)0) totalVariance += (double)1;

                // Build R = Cov ./ (sampleStd (x) sampleStd) inline, in place over C. The DIAGONAL of a
                // non-degenerate feature is set to EXACTLY 1 (the correlation-matrix property) rather than
                // C[j,j]/(sj*sj), which is only ~1 now that sampleStd is a direct row-sum, not sqrt(C[j,j]).
                // A zero-variance feature zeroes its ENTIRE row/column, including the diagonal (NOT
                // StatsOP.correlation's convention of 1 there — see the type doc's degenerate-feature trap).
                // Each cell (i,j) is visited exactly once so the in-place overwrite never reads an
                // already-transformed neighbor. Stays exactly symmetric (C[i,j]==C[j,i], si*sj==sj*si) so
                // symmetric's symmetry check passes.
                for (int i = 0; i < p; i++)
                {
                    double si = sampleStd[i];
                    for (int j = 0; j < p; j++)
                    {
                        double sj = sampleStd[j];
                        if (i == j)
                            C[i, j] = si > (double)0 ? (double)1 : (double)0;
                        else
                            C[i, j] = (si > (double)0 && sj > (double)0) ? C[i, j] / (si * sj) : (double)0;
                    }
                }
            }

            bool converged = Eigen.symmetric(ref C, ref model.explainedVariance, ref model.components);

            FinalizeModel(ref model, p, totalVariance, converged);
            return converged;
        }

        /// <summary>fitCov with scaling = PCAScaling.Covariance.</summary>
        public static bool fitCov(in doubleMxN X, ref doublePCAModel model)
            => fitCov(in X, ref model, PCAScaling.Covariance);

        /// <summary>Validates inputs, allocates the model (p x p) from <paramref name="arena"/>, then delegates to the ref-model overload. Guards fire before any arena allocation.</summary>
        public static doublePCAModel fitCov(ref Arena arena, in doubleMxN X, PCAScaling scaling)
        {
            const string method = "PCA.fitCov";
            RequireBasicShape(in X, method);

            int p = X.N_Cols;
            var model = arena.doublePCAModel(p, p);
            fitCov(in X, ref model, scaling);
            return model;
        }

        /// <summary>fitCov (allocating) with scaling = PCAScaling.Covariance.</summary>
        public static doublePCAModel fitCov(ref Arena arena, in doubleMxN X)
            => fitCov(ref arena, in X, PCAScaling.Covariance);

        // =====================================================================================
        // 2. fitSvd — full, accurate (SVD.thin on a centered/standardized copy; requires n >= p)
        // =====================================================================================

        /// <summary>
        /// Full PCA via SVD of the centered (or standardized) data directly -- no Gram matrix, no
        /// kappa^2 accuracy loss. Requires X.M_Rows (n) >= X.N_Cols (p); throws otherwise (use
        /// fitCov for wide data). model.k is set to p.
        /// </summary>
        public static bool fitSvd(in doubleMxN X, ref doublePCAModel model, PCAScaling scaling, int maxIter)
        {
            const string method = "PCA.fitSvd";
            RequireBasicShape(in X, method);
            RequireTallShape(in X, method);

            int n = X.M_Rows;
            int p = X.N_Cols;
            RequireModelShapeFull(in model, p, method);

            Stats.colMean(in X, ref model.mean);

            var Xc = X.doubleTempMat(n, p);
            double totalVariance = BuildWorkingCopy(in X, scaling, in model.mean, ref model.scale, ref Xc);

            // SVD.thin reads A while writing U -- a SEPARATE n x p temp, never Xc itself.
            var U = X.doubleTempMat(n, p);
            bool converged = SVD.thin(in Xc, ref U, ref model.explainedVariance, ref model.components, maxIter);

            if (converged)
                SingularValuesToVariances(ref model.explainedVariance, p, n);

            FinalizeModel(ref model, p, totalVariance, converged);
            return converged;
        }

        /// <summary>fitSvd with maxIter = SVD.thin's default (75).</summary>
        public static bool fitSvd(in doubleMxN X, ref doublePCAModel model, PCAScaling scaling)
            => fitSvd(in X, ref model, scaling, 75);

        /// <summary>fitSvd with scaling = PCAScaling.Covariance and maxIter = 75.</summary>
        public static bool fitSvd(in doubleMxN X, ref doublePCAModel model)
            => fitSvd(in X, ref model, PCAScaling.Covariance, 75);

        /// <summary>Validates inputs, allocates the model (p x p) from <paramref name="arena"/>, then delegates to the ref-model overload. Guards fire before any arena allocation.</summary>
        public static doublePCAModel fitSvd(ref Arena arena, in doubleMxN X, PCAScaling scaling)
        {
            const string method = "PCA.fitSvd";
            RequireBasicShape(in X, method);
            RequireTallShape(in X, method);

            int p = X.N_Cols;
            var model = arena.doublePCAModel(p, p);
            fitSvd(in X, ref model, scaling, 75);
            return model;
        }

        /// <summary>fitSvd (allocating) with scaling = PCAScaling.Covariance.</summary>
        public static doublePCAModel fitSvd(ref Arena arena, in doubleMxN X)
            => fitSvd(ref arena, in X, PCAScaling.Covariance);

        // =====================================================================================
        // 3. fitSvdTruncated — exact top-k (Golub-Kahan-Lanczos)
        // =====================================================================================
        //
        // NOTE ON SHAPE: SVD.truncated's core overload enforces A.M_Rows >= A.N_Cols internally
        // (same requirement as SVD.thin/randomized) -- it is NOT shape-free despite appearances from
        // its `k <= N_Cols` guard alone. PCA therefore applies the same RequireTallShape guard as
        // fitSvd/fitRandomized, BEFORE allocating Xc, so invalid (wide) input never touches the temp
        // pool. fitSvdTruncated does NOT support wide (p>n) data.

        /// <summary>
        /// Exact top-k PCA via Golub-Kahan-Lanczos bidiagonalization (SVD.truncated) on the centered
        /// (or standardized) data -- avoids the full O(n p^2) SVD when only the leading few components
        /// are needed. Requires X.M_Rows (n) >= X.N_Cols (p) (see the shape note above); 0 &lt; k &lt;=
        /// min(n, p). model.k is set to k. explainedVarianceRatio is computed against the FULL
        /// totalVariance, so top-k ratios sum to less than 1.
        /// </summary>
        public static bool fitSvdTruncated(in doubleMxN X, ref doublePCAModel model, int k,
                                            PCAScaling scaling, int oversample, uint seed, int maxIter)
        {
            const string method = "PCA.fitSvdTruncated";
            RequireBasicShape(in X, method);
            RequireTallShape(in X, method);

            int n = X.M_Rows;
            int p = X.N_Cols;
            RequireTopK(k, n, p, method);
            RequireModelShapeTopK(in model, p, k, method);

            Stats.colMean(in X, ref model.mean);

            var Xc = X.doubleTempMat(n, p);
            double totalVariance = BuildWorkingCopy(in X, scaling, in model.mean, ref model.scale, ref Xc);

            var Uk = X.doubleTempMat(n, k);
            SVD.truncated(in Xc, ref Uk, ref model.explainedVariance, ref model.components,
                              k, oversample, seed, maxIter, out bool converged);

            if (converged)
                SingularValuesToVariances(ref model.explainedVariance, k, n);

            FinalizeModel(ref model, k, totalVariance, converged);
            return converged;
        }

        /// <summary>
        /// fitSvdTruncated forwarding SVD.truncated's own "generous default Krylov width" formula
        /// verbatim (oversample = max(k, 12), so p = min(n, max(2k, k+12))), its default seed
        /// (0x9E3779B1u) and default maxIter (75).
        /// </summary>
        public static bool fitSvdTruncated(in doubleMxN X, ref doublePCAModel model, int k, PCAScaling scaling)
            => fitSvdTruncated(in X, ref model, k, scaling, math.max(k, 12), 0x9E3779B1u, 75);

        /// <summary>fitSvdTruncated with scaling = PCAScaling.Covariance.</summary>
        public static bool fitSvdTruncated(in doubleMxN X, ref doublePCAModel model, int k)
            => fitSvdTruncated(in X, ref model, k, PCAScaling.Covariance);

        /// <summary>Validates inputs, allocates the model (p x k) from <paramref name="arena"/>, then delegates to the ref-model overload. Guards fire before any arena allocation.</summary>
        public static doublePCAModel fitSvdTruncated(ref Arena arena, in doubleMxN X, int k, PCAScaling scaling)
        {
            const string method = "PCA.fitSvdTruncated";
            RequireBasicShape(in X, method);
            RequireTallShape(in X, method);

            int n = X.M_Rows;
            int p = X.N_Cols;
            RequireTopK(k, n, p, method);

            var model = arena.doublePCAModel(p, k);
            fitSvdTruncated(in X, ref model, k, scaling);
            return model;
        }

        /// <summary>fitSvdTruncated (allocating) with scaling = PCAScaling.Covariance.</summary>
        public static doublePCAModel fitSvdTruncated(ref Arena arena, in doubleMxN X, int k)
            => fitSvdTruncated(ref arena, in X, k, PCAScaling.Covariance);

        // =====================================================================================
        // 4. fitRandomized — approximate top-k (Halko-Martinsson-Tropp; requires n >= p)
        // =====================================================================================

        /// <summary>
        /// Approximate top-k PCA via randomized SVD (SVD.randomized) on the centered (or
        /// standardized) data. Pays off when p is large and k &lt;&lt; p; at small p prefer
        /// fitCov, for exact top-k prefer fitSvdTruncated. Requires X.M_Rows (n) >= X.N_Cols (p);
        /// throws otherwise (use fitCov for wide data). 0 &lt; k &lt;= min(n, p). model.k is set
        /// to k. Deterministic by default (seed = 0x9E3779B1u, SVD.randomized's own default).
        /// </summary>
        public static bool fitRandomized(in doubleMxN X, ref doublePCAModel model, int k,
                                          PCAScaling scaling, int oversample, int powerIters, uint seed, int maxIter)
        {
            const string method = "PCA.fitRandomized";
            RequireBasicShape(in X, method);
            RequireTallShape(in X, method);

            int n = X.M_Rows;
            int p = X.N_Cols;
            RequireTopK(k, n, p, method);
            RequireModelShapeTopK(in model, p, k, method);

            Stats.colMean(in X, ref model.mean);

            var Xc = X.doubleTempMat(n, p);
            double totalVariance = BuildWorkingCopy(in X, scaling, in model.mean, ref model.scale, ref Xc);

            var Uk = X.doubleTempMat(n, k);
            bool converged = SVD.randomized(in Xc, ref Uk, ref model.explainedVariance, ref model.components,
                                                k, oversample, powerIters, seed, maxIter);

            if (converged)
                SingularValuesToVariances(ref model.explainedVariance, k, n);

            FinalizeModel(ref model, k, totalVariance, converged);
            return converged;
        }

        /// <summary>fitRandomized forwarding SVD.randomized's own defaults verbatim: oversample=10,
        /// powerIters=2, seed=0x9E3779B1u, maxIter=75.</summary>
        public static bool fitRandomized(in doubleMxN X, ref doublePCAModel model, int k, PCAScaling scaling)
            => fitRandomized(in X, ref model, k, scaling, 10, 2, 0x9E3779B1u, 75);

        /// <summary>fitRandomized with scaling = PCAScaling.Covariance.</summary>
        public static bool fitRandomized(in doubleMxN X, ref doublePCAModel model, int k)
            => fitRandomized(in X, ref model, k, PCAScaling.Covariance);

        /// <summary>Validates inputs, allocates the model (p x k) from <paramref name="arena"/>, then delegates to the ref-model overload. Guards fire before any arena allocation.</summary>
        public static doublePCAModel fitRandomized(ref Arena arena, in doubleMxN X, int k, PCAScaling scaling)
        {
            const string method = "PCA.fitRandomized";
            RequireBasicShape(in X, method);
            RequireTallShape(in X, method);

            int n = X.M_Rows;
            int p = X.N_Cols;
            RequireTopK(k, n, p, method);

            var model = arena.doublePCAModel(p, k);
            fitRandomized(in X, ref model, k, scaling);
            return model;
        }

        /// <summary>fitRandomized (allocating) with scaling = PCAScaling.Covariance.</summary>
        public static doublePCAModel fitRandomized(ref Arena arena, in doubleMxN X, int k)
            => fitRandomized(ref arena, in X, k, PCAScaling.Covariance);

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
        public static void transform(in doubleMxN X, in doublePCAModel model, ref doubleMxN scores)
        {
            const string method = "PCA.transform";
            int p = X.N_Cols;
            int nNew = X.M_Rows;

            if (model.mean.N != p || model.scale.N != p || model.components.M_Rows != p)
                throw new ArgumentException(method + ": model.mean/scale/components must match X.N_Cols (p)");
            if (model.k != model.components.N_Cols)
                throw new ArgumentException(method + ": model.k must equal model.components.N_Cols (stale model)");
            if (scores.M_Rows != nNew || scores.N_Cols != model.k)
                throw new ArgumentException(method + ": scores must be X.M_Rows x model.k");

            var Xs = X.doubleTempMat(nNew, p);
            for (int r = 0; r < nNew; r++)
                for (int c = 0; c < p; c++)
                    Xs[r, c] = (X[r, c] - model.mean[c]) / model.scale[c];

            Blas.dot(in Xs, in model.components, ref scores);
        }

        /// <summary>Allocating transform: allocates and returns a fresh X.M_Rows x model.k scores matrix.</summary>
        public static doubleMxN transform(ref Arena arena, in doubleMxN X, in doublePCAModel model)
        {
            const string method = "PCA.transform";
            int p = X.N_Cols;

            if (model.mean.N != p || model.scale.N != p || model.components.M_Rows != p)
                throw new ArgumentException(method + ": model.mean/scale/components must match X.N_Cols (p)");
            if (model.k != model.components.N_Cols)
                throw new ArgumentException(method + ": model.k must equal model.components.N_Cols (stale model)");

            var scores = arena.doubleMat(X.M_Rows, model.k);
            transform(in X, in model, ref scores);
            return scores;
        }
    }
}
