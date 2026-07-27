using Unity.Collections;

namespace BULA.Control
{
    /// <summary>
    /// Van der Merwe scaled sigma-point workspace for <see cref="Kalman.ukfPredict{TModel}"/> /
    /// <see cref="Kalman.ukfUpdate{TMeas}"/>, paired ALONGSIDE a <see cref="fProxyKFState"/> (which
    /// still carries <c>x</c>/<c>P</c>) rather than folding sigma-point buffers into that struct --
    /// keeps the linear/EKF/fixed-gain paths free of UKF-only memory. Holds the hyperparameters
    /// (<see cref="alpha"/>/<see cref="beta"/>/<see cref="kappa"/>), their derived weights
    /// (<see cref="Wm"/>/<see cref="Wc"/>, fixed once n/alpha/beta/kappa are known), and every
    /// n-shaped scratch buffer <c>ukfPredict</c>/<c>ukfUpdate</c> need so BOTH are zero-alloc on
    /// their state-shaped work -- only <c>ukfUpdate</c>'s measurement-shaped intermediates (which
    /// vary in size per call) allocate small <c>Allocator.Temp</c> scratch, the same convention
    /// <see cref="Kalman.update"/> uses.
    /// </summary>
    public struct fProxyUKFCache
    {
        /// <summary>Spread parameter (usually small in the classic Van der Merwe write-up, but see
        /// <see cref="fProxyUKFCache(int, Allocator)"/>'s own doc for why this library defaults it
        /// to 1, not 1e-3).</summary>
        public fProxy alpha;

        /// <summary>Distribution-shape parameter; 2 is optimal for a Gaussian prior.</summary>
        public fProxy beta;

        /// <summary>Secondary spread parameter; 0 is a common, n-independent choice.</summary>
        public fProxy kappa;

        /// <summary>lambda = alpha^2 (n+kappa) - n.</summary>
        public fProxy lambda;

        /// <summary>Mean-recombination weights, length 2n+1.</summary>
        public fProxyN Wm;

        /// <summary>Covariance-recombination weights, length 2n+1. <c>Wc[0]</c> can be NEGATIVE
        /// when alpha &lt; 1 -- the classic scaled-UKF pitfall; <c>ukfPredict</c>/<c>ukfUpdate</c>
        /// symmetrize their recombined covariance explicitly to guard against the roundoff this
        /// invites, but a large enough negative weight can still produce an indefinite result (see
        /// their own doc comments).</summary>
        public fProxyN Wc;

        /// <summary>Sigma points, (2n+1) x n, one row per point. Regenerated fresh at the start of
        /// EVERY <c>ukfPredict</c>/<c>ukfUpdate</c> call from that call's current (x, P) -- see
        /// those methods' own doc comments for why this is a deliberate deviation from FilterPy's
        /// reuse-across-calls shortcut.</summary>
        public fProxyMxN X;

        /// <summary>Pivoted-Cholesky factor scratch (n x n) for the sigma-point spread.</summary>
        public fProxyMxN L;

        /// <summary>Pivoted-Cholesky permutation scratch (n).</summary>
        public Pivot Piv;

        /// <summary>Nested <see cref="CHOP"/> workspace (its own n x n symmetric working copy) so
        /// <c>GenerateSigmaPoints</c>'s <see cref="CHOP.decomp(in fProxyMxN, ref fProxyMxN, ref Pivot, ref fProxyCHOPCache)"/>
        /// call never touches <c>Allocator.Temp</c> either -- only <see cref="CHOP.decomp(in fProxyMxN, ref fProxyMxN, ref Pivot)"/>'s
        /// convenience (non-workspace) overload allocates internally, and this cache always uses the
        /// workspace form. <c>bt</c> is left uncreated (unused -- sigma-point generation only ever
        /// calls <c>decomp</c>, never <c>decompSolve</c>).</summary>
        public fProxyCHOPCache chopWs;

        /// <summary>Propagated (through <c>model.F</c>) sigma points, (2n+1) x n -- <c>ukfPredict</c> only.</summary>
        public fProxyMxN Y;

        /// <summary>Single sigma-point row extracted from <see cref="X"/> before an F/H evaluation.</summary>
        public fProxyN rowIn;

        /// <summary>F's output for one sigma point (n) -- <c>ukfPredict</c> only (H's output is m-shaped
        /// and varies per call, so <c>ukfUpdate</c> uses per-call <c>Allocator.Temp</c> for it instead).</summary>
        public fProxyN rowOut;

        /// <summary>Weighted-mean scratch (n) -- <c>ukfPredict</c> only.</summary>
        public fProxyN xPred;

        /// <summary>Per-sigma-point residual scratch (n): the sigma-spread vector during sigma-point
        /// generation, then reused as the state-space residual (X[k]-mean) during covariance
        /// recombination in both <c>ukfPredict</c> and <c>ukfUpdate</c>.</summary>
        public fProxyN diff;

        /// <summary>Covariance-accumulation scratch (n x n) -- <c>ukfPredict</c> only.</summary>
        public fProxyMxN Pacc;

        /// <summary>State dimension this cache was constructed for.</summary>
        public int N;

        /// <summary>
        /// Allocates a cache for an n-dimensional UKF with explicit Van der Merwe hyperparameters.
        /// Throws if alpha/kappa give a non-positive n+lambda (the sigma-point spread's sqrt argument
        /// -- a negative value is always a hyperparameter mistake, never a legitimate input).
        /// </summary>
        public fProxyUKFCache(int n, fProxy alphaP, fProxy betaP, fProxy kappaP, Allocator allocator)
        {
            alpha = alphaP;
            beta = betaP;
            kappa = kappaP;
            lambda = alpha * alpha * ((fProxy)n + kappa) - (fProxy)n;

            if (!((fProxy)n + lambda > (fProxy)0))
                throw new System.ArgumentException("fProxyUKFCache: alpha/kappa must give n+lambda > 0");

            int npts = 2 * n + 1;
            Wm = new fProxyN(npts, allocator);
            Wc = new fProxyN(npts, allocator);
            fProxy c = (fProxy)0.5 / ((fProxy)n + lambda);
            for (int i = 0; i < npts; i++) { Wm[i] = c; Wc[i] = c; }
            Wm[0] = lambda / ((fProxy)n + lambda);
            Wc[0] = Wm[0] + ((fProxy)1 - alpha * alpha + beta);

            X = new fProxyMxN(npts, n, allocator);
            L = new fProxyMxN(n, n, allocator);
            Piv = new Pivot(n, allocator);
            chopWs = new fProxyCHOPCache { W = new fProxyMxN(n, n, allocator), bt = default };
            Y = new fProxyMxN(npts, n, allocator);
            rowIn = new fProxyN(n, allocator);
            rowOut = new fProxyN(n, allocator);
            xPred = new fProxyN(n, allocator);
            diff = new fProxyN(n, allocator);
            Pacc = new fProxyMxN(n, n, allocator);
            N = n;
        }

        /// <summary>
        /// Allocates a cache using this library's default Van der Merwe hyperparameters: alpha=1,
        /// beta=2, kappa=0. alpha=1 makes lambda=kappa for kappa=0, keeping <c>Wc[0]</c>
        /// non-negative for the DEFAULT case (see <see cref="Wc"/>'s own doc for why that matters)
        /// -- a caller who explicitly passes a smaller alpha via the other constructor still gets a
        /// correct, symmetrized result, just with less numerical margin.
        /// </summary>
        public fProxyUKFCache(int n, Allocator allocator)
            : this(n, (fProxy)1, (fProxy)2, (fProxy)0, allocator)
        {
        }

        /// <summary>True once every buffer is allocated (regardless of content validity).</summary>
        public bool IsCreated => X.Data.IsCreated && L.Data.IsCreated;

        /// <summary>True iff created AND sized for exactly n.</summary>
        public bool IsValid(int n) => IsCreated && N == n;

        /// <summary>Releases every buffer. Safe to call on an empty/already-disposed instance (except
        /// <see cref="Piv"/>, which -- like every other <see cref="Pivot"/>-holding scratch in this
        /// library -- has no public created-check and so is disposed unconditionally; do not call
        /// Dispose twice on the same populated cache).</summary>
        public void Dispose()
        {
            if (Wm.Data.IsCreated) Wm.Dispose();
            if (Wc.Data.IsCreated) Wc.Dispose();
            if (X.Data.IsCreated) X.Dispose();
            if (L.Data.IsCreated) L.Dispose();
            Piv.Dispose();
            if (chopWs.W.Data.IsCreated) chopWs.W.Dispose();
            if (Y.Data.IsCreated) Y.Dispose();
            if (rowIn.Data.IsCreated) rowIn.Dispose();
            if (rowOut.Data.IsCreated) rowOut.Dispose();
            if (xPred.Data.IsCreated) xPred.Dispose();
            if (diff.Data.IsCreated) diff.Dispose();
            if (Pacc.Data.IsCreated) Pacc.Dispose();
        }
    }
}
