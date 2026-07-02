namespace LinearAlgebra
{
    /// <summary>
    /// Diagnostics for a least-squares Krylov solve (<c>cgls</c> / <c>lsqr</c> / <c>lsmr</c>),
    /// returned by the opt-in <c>out fProxyLstsqInfo</c> overloads. The three norms are computed
    /// EXACTLY from the final x with a single extra A*x and Aᵀ*r at return -- not tracked
    /// per-iteration estimates -- so they cost roughly one extra iteration and are only paid when
    /// requested; the plain <c>bool</c> overloads are byte-for-byte unaffected.
    ///
    /// The norms are only meaningful when <see cref="converged"/> is true: on a false return the
    /// solver leaves x undefined, so rnorm/Arnorm/xnorm describe that undefined state.
    /// </summary>
    public struct fProxyLstsqInfo
    {
        /// <summary>Residual norm ‖b - A x‖. Nonzero for an inconsistent (over-determined) system
        /// even at the optimum -- it is the least-squares residual, not an error.</summary>
        public fProxy rnorm;

        /// <summary>Normal-equation residual ‖Aᵀ(b - A x)‖ -- or, when solved with Tikhonov damping,
        /// ‖Aᵀ(b - A x) - damp²x‖. This is the true least-squares optimality measure: it goes to
        /// zero at the minimizer regardless of whether the system is consistent.</summary>
        public fProxy Arnorm;

        /// <summary>Solution norm ‖x‖ (useful for tuning Tikhonov damping / monitoring blow-up on
        /// ill-conditioned problems).</summary>
        public fProxy xnorm;

        /// <summary>Iterations actually performed (0 when the solver converged before the first
        /// bidiagonalization/CG step; equals maxIterations when it ran out).</summary>
        public int iterations;

        /// <summary>Whether the solver reached its tolerance -- mirrors the method's bool return.</summary>
        public bool converged;
    }
}
