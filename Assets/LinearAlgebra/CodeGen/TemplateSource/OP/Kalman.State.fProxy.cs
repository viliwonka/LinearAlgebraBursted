using Unity.Collections;

namespace LinearAlgebra
{
    /// <summary>
    /// Warm, buffer-carrying state for the <see cref="Kalman"/> predict/update family: the carried
    /// estimate <see cref="x"/> (n) and covariance <see cref="P"/> (n x n), plus every scratch buffer
    /// <see cref="Kalman.predict(ref fProxyKFState, in fProxyMxN, in fProxyMxN, in fProxyN, in fProxyMxN)"/> /
    /// <see cref="Kalman.ekfPredict{TModel}"/> / <see cref="Kalman.predictFixed(ref fProxyKFState, in fProxyMxN, in fProxyMxN, in fProxyN)"/> /
    /// <see cref="Kalman.updateFixed"/> need -- all pre-allocated here so those calls never touch
    /// <c>Allocator.Temp</c>. The general <see cref="Kalman.update"/> / <see cref="Kalman.ekfUpdate{TMeas}"/>
    /// path DOES allocate small <c>Allocator.Temp</c> scratch per call, sized to that call's own
    /// measurement dimension -- the same per-call-shape convention <see cref="Riccati.dare"/> and the
    /// LQR gain kernel use for their inner solves (not worth pre-allocating for every possible size).
    /// </summary>
    public struct fProxyKFState
    {
        /// <summary>State estimate, length n.</summary>
        public fProxyN x;

        /// <summary>State covariance, n x n.</summary>
        public fProxyMxN P;

        // ---- predict/update-family scratch: n or n x n, reused every call.
        // Public (not internal), matching the house Cache/State convention (fProxyCHOPCache,
        // fProxyLQRState) -- these are workspace buffers, not hidden implementation state. ----
        public fProxyN xNext;
        public fProxyN Bu;
        public fProxyMxN AP;
        public fProxyMxN APAt;
        public fProxyMxN At;
        public fProxyMxN J;

        // ---- updateFixed() scratch: the fixed-gain fast path's only measurement-sized buffer ----
        public fProxyN yFast;

        /// <summary>State dimension this instance was constructed for.</summary>
        public int N;

        /// <summary>Measurement dimension the fixed-gain fast path
        /// (<see cref="Kalman.predictFixed(ref fProxyKFState, in fProxyMxN, in fProxyMxN, in fProxyN)"/> /
        /// <see cref="Kalman.updateFixed"/>) is sized for. The general <see cref="Kalman.update"/> /
        /// <see cref="Kalman.ekfUpdate{TMeas}"/> path is not bound by this -- it accepts any
        /// measurement dimension per call.</summary>
        public int MMax;

        /// <summary>Allocates state sized for an n-dimensional filter whose fixed-gain fast path (if
        /// used) measures mMax dimensions. <see cref="x"/>/<see cref="P"/> start zeroed -- assign
        /// your initial estimate/covariance before the first predict/update call.</summary>
        public fProxyKFState(int n, int mMax, Allocator allocator)
        {
            x = new fProxyN(n, allocator);
            P = new fProxyMxN(n, n, allocator);
            xNext = new fProxyN(n, allocator);
            Bu = new fProxyN(n, allocator);
            AP = new fProxyMxN(n, n, allocator);
            APAt = new fProxyMxN(n, n, allocator);
            At = new fProxyMxN(n, n, allocator);
            J = new fProxyMxN(n, n, allocator);
            yFast = new fProxyN(mMax, allocator);
            N = n;
            MMax = mMax;
        }

        /// <summary>True once every buffer is allocated (regardless of content validity).</summary>
        public bool IsCreated => x.Data.IsCreated && P.Data.IsCreated;

        /// <summary>True iff created AND sized for exactly (n, mMax).</summary>
        public bool IsValid(int n, int mMax) => IsCreated && N == n && MMax == mMax;

        /// <summary>Releases every buffer. Safe to call on an empty/already-disposed instance.</summary>
        public void Dispose()
        {
            if (x.Data.IsCreated) x.Dispose();
            if (P.Data.IsCreated) P.Dispose();
            if (xNext.Data.IsCreated) xNext.Dispose();
            if (Bu.Data.IsCreated) Bu.Dispose();
            if (AP.Data.IsCreated) AP.Dispose();
            if (APAt.Data.IsCreated) APAt.Dispose();
            if (At.Data.IsCreated) At.Dispose();
            if (J.Data.IsCreated) J.Dispose();
            if (yFast.Data.IsCreated) yFast.Dispose();
        }
    }
}
