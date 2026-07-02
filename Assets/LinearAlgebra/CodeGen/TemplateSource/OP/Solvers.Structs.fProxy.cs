namespace LinearAlgebra
{
    /// <summary>
    /// Result of a least-squares Krylov solve (<c>cgls</c> / <c>lsqr</c> / <c>lsmr</c>). Every LS
    /// solver RETURNS this by value; an implicit <c>bool</c> conversion (== <see cref="Solved"/>)
    /// means the old success-test call shapes still compile unchanged:
    /// <code>
    ///   if (Solvers.lsqr(A, b, ref x)) { ... }          // implicit bool -> "did it converge?"
    ///   bool ok = Solvers.cgls(A, b, ref x);            // same
    ///   var info = Solvers.lsmr(A, b, ref x);           // keep the struct for diagnostics
    ///   if (info.Solved) Debug.Log(info.iterations);
    /// </code>
    ///
    /// The three norms are filled from the values the solver ALREADY tracks (or at most a single
    /// dot on a residual it already holds) at the point it returns -- never a fresh A*x/Aᵀ*r, so
    /// the struct costs nothing beyond the plain solve:
    /// <list type="bullet">
    /// <item>cgls -- rnorm from a dot on its live residual r; Arnorm = √gamma (its tracked ‖Aᵀr‖²).</item>
    /// <item>lsqr -- rnorm = phibar, Arnorm = phibar·alpha·|c|, both produced free by the recurrence.</item>
    /// <item>lsmr -- Arnorm = |ζ̄| (free, monotone); rnorm via the Fong-Saunders ‖r‖ recurrence
    ///       (O(1) scalars per iteration, no matvec).</item>
    /// </list>
    /// For an independently-recomputed, certified-exact residual (one extra Apply + ApplyT) call
    /// <see cref="Solvers.lstsqResidual{TOp}"/> on the returned x instead.
    ///
    /// The norms describe the CURRENT x at return; they are only meaningful when <see cref="Solved"/>
    /// (on a Breakdown/MaxIterations return the solver leaves x undefined).
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

        /// <summary>Why the solve stopped -- see <see cref="SolveStatus"/>.</summary>
        public SolveStatus status;

        /// <summary>True iff the solver reached its tolerance (<c>status == SolveStatus.Converged</c>).
        /// Same value as the implicit bool conversion; use whichever reads better.</summary>
        public bool Solved => status == SolveStatus.Converged;

        /// <summary>Implicit success test, so <c>if (solve(...))</c> / <c>bool ok = solve(...)</c>
        /// keep compiling after the return type changed from bool to this struct.</summary>
        public static implicit operator bool(fProxyLstsqInfo info) => info.status == SolveStatus.Converged;
    }

    /// <summary>
    /// Result of a square-system Krylov solve (<c>cg</c> / <c>conjugateGradient</c> / <c>pcg</c> /
    /// <c>minres</c> / <c>biCGStab</c> / <c>cgne</c>). Same contract as <see cref="fProxyLstsqInfo"/>
    /// -- returned by value, implicit <c>bool</c> == <see cref="Solved"/> -- but carries only the
    /// residual norm ‖b - A x‖ (no Aᵀr / xnorm: for a square solve the residual IS the error
    /// measure). Filled from each solver's tracked residual (cg/pcg/cgne: a live ‖r‖; minres:
    /// phibar; biCGStab: its running ‖r‖) -- no extra matvec.
    ///
    /// rnorm is only meaningful when <see cref="Solved"/>; on a Breakdown/MaxIterations return x
    /// is undefined.
    /// </summary>
    public struct fProxySolveInfo
    {
        /// <summary>Residual norm ‖b - A x‖ at the returned x.</summary>
        public fProxy rnorm;

        /// <summary>Iterations actually performed (0 when converged before the first step; equals
        /// maxIterations when it ran out).</summary>
        public int iterations;

        /// <summary>Why the solve stopped -- see <see cref="SolveStatus"/>.</summary>
        public SolveStatus status;

        /// <summary>True iff the solver reached its tolerance (<c>status == SolveStatus.Converged</c>).</summary>
        public bool Solved => status == SolveStatus.Converged;

        /// <summary>Implicit success test so <c>if (solve(...))</c> keeps compiling.</summary>
        public static implicit operator bool(fProxySolveInfo info) => info.status == SolveStatus.Converged;
    }
}
