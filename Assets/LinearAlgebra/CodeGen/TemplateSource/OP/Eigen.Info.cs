//singularFile//
namespace LinearAlgebra
{
    /// <summary>
    /// Result of an iterative dominant/extremal-eigenpair solve (<c>Eigen.powerIteration</c> /
    /// <c>Eigen.inversePowerIteration</c>). Every converted eigensolver RETURNS this by value
    /// (alongside its existing <c>out fProxy lambda</c>); an implicit <c>bool</c> conversion
    /// (== <see cref="Solved"/>) means the old success-test call shapes still compile unchanged:
    /// <code>
    ///   if (Eigen.powerIteration(in A, ref v, ref w, out var lambda, tol, maxIter)) { ... }
    ///   bool ok = Eigen.inversePowerIteration(in A, ref v, out var lambda);
    ///   var info = Eigen.powerIteration(in A, ref v, ref w, out var lambda, tol, maxIter);
    ///   if (info.Solved) Debug.Log(info.iterations);
    /// </code>
    ///
    /// Reuses <see cref="IterativeSolveStatus"/> (the same enum the Krylov solvers use) rather than
    /// a dedicated eigensolver enum -- the three outcomes (Converged / MaxIterations / Breakdown)
    /// mean exactly the same thing here.
    ///
    /// <see cref="residual"/> is reported as <c>double</c> regardless of the solve's precision (a
    /// float solve widens its float residual), matching <c>SolveInfo</c>/<c>LstsqInfo</c>. It is
    /// filled from values the solver already tracks (or, for inversePowerIteration, a single extra
    /// O(n) pass over the A*v it already holds from its last step) -- never a fresh matvec beyond
    /// what the algorithm already performs:
    /// <list type="bullet">
    /// <item>powerIteration -- the infinity-norm residual ‖Av-λv‖ the loop already computes every
    ///       iteration to test convergence.</item>
    /// <item>inversePowerIteration -- ‖Av-λv‖∞ computed once at the return site from the A*v the
    ///       last outer iteration's Rayleigh-quotient step already produced (Ap).</item>
    /// </list>
    ///
    /// On a Converged OR MaxIterations return, (lambda, v) is the last iterate and residual
    /// describes it (so on MaxIterations you can inspect how close it got). On a Breakdown return
    /// (inversePowerIteration only -- powerIteration has no breakdown mode) residual is
    /// <see cref="double.NaN"/> and (lambda, v) are undefined / partially updated.
    /// </summary>
    public struct EigenSolveInfo
    {
        /// <summary>Outer iterations actually performed (a Breakdown return counts only iterations
        /// that ran to completion before the breakdown, so it can be <c>0</c>). Do NOT infer success
        /// from this count alone -- powerIteration's post-loop check can return Converged with
        /// <c>iterations == maxIter</c> (the same value a MaxIterations return carries); always read
        /// <see cref="status"/>.</summary>
        public int iterations;

        /// <summary>Infinity-norm residual ‖A v - lambda v‖ at the returned (lambda, v). Always
        /// <c>double</c> regardless of the solve's precision. <see cref="double.NaN"/> on a
        /// Breakdown return, where (lambda, v) are undefined.</summary>
        public double residual;

        /// <summary>Why the solve stopped -- see <see cref="IterativeSolveStatus"/>.</summary>
        public IterativeSolveStatus status;

        /// <summary>True iff the solver reached its tolerance (<c>status == IterativeSolveStatus.Converged</c>).
        /// Same value as the implicit bool conversion; use whichever reads better.</summary>
        public bool Solved => status == IterativeSolveStatus.Converged;

        /// <summary>Same as <see cref="Solved"/>, so <c>if (solve(...))</c> / <c>bool ok = solve(...)</c>
        /// keep compiling after the return type changed from bool to this struct.</summary>
        public static implicit operator bool(EigenSolveInfo i) => i.status == IterativeSolveStatus.Converged;
    }

    /// <summary>
    /// Result of a symmetric Lanczos tridiagonalization (<c>Eigen.lanczos</c> /
    /// <c>Eigen.lanczosVectors</c>). Every converted overload RETURNS this by value (the
    /// value-returning "allocating" overloads carry it as an <c>out</c> parameter alongside their
    /// <c>fProxyN</c>/<c>fProxyMxN</c> outputs); an implicit <c>bool</c> conversion
    /// (== <see cref="Solved"/>) means the old success-test call shapes still compile unchanged:
    /// <code>
    ///   if (Eigen.lanczos(in A, ref ws, ref eigenvalues, steps)) { ... }
    ///   var eig = Eigen.lanczos(ref arena, in A, steps, out LanczosInfo info);
    ///   if (info.Solved) Debug.Log(info.produced);
    /// </code>
    ///
    /// <see cref="status"/> is Converged iff the inner symmetric tridiagonal eigensolver (QL
    /// iteration on the -- possibly early-breakdown-padded -- tridiagonal T) converged; otherwise
    /// MaxIterations. Lanczos itself has no Breakdown status: an early invariant-subspace
    /// breakdown (see <see cref="produced"/>) is NOT a failure, it just means fewer than
    /// <c>steps</c> Ritz values/vectors were produced -- the ones that WERE produced are exact.
    /// </summary>
    public struct LanczosInfo
    {
        /// <summary>Number of valid Ritz values/vectors produced (&lt;= the requested
        /// <c>steps</c>; strictly less than <c>steps</c> only on early invariant-subspace
        /// breakdown -- see the class doc comment). Entries at index &gt;= <see cref="produced"/>
        /// in the caller's output buffers are padding/meaningless -- ignore them.</summary>
        public int produced;

        /// <summary>Why the tridiagonal eigensolver stopped -- see <see cref="IterativeSolveStatus"/>.
        /// Converged iff the inner QL iteration on T converged.</summary>
        public IterativeSolveStatus status;

        /// <summary>True iff the inner tridiagonal eigensolve converged
        /// (<c>status == IterativeSolveStatus.Converged</c>).</summary>
        public bool Solved => status == IterativeSolveStatus.Converged;

        /// <summary>Same as <see cref="Solved"/>, so <c>if (lanczos(...))</c> keeps compiling after
        /// the return type changed from bool to this struct.</summary>
        public static implicit operator bool(LanczosInfo i) => i.status == IterativeSolveStatus.Converged;
    }
}
