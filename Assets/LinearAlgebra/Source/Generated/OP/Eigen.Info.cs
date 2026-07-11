//singularFile//
using Unity.Collections;

namespace LinearAlgebra
{
    /// <summary>
    /// Result of an iterative dominant/extremal-eigenpair solve (<c>Eigen.powerIteration</c> /
    /// <c>Eigen.inversePowerIteration</c>), returned by value alongside the existing
    /// <c>out fProxy lambda</c>. Implicitly converts to <c>bool</c> (== <see cref="Solved"/>) for use
    /// in <c>if (...)</c>:
    /// <code>
    ///   if (Eigen.powerIteration(in A, ref v, ref w, out var lambda, tolerance, maxIterations)) { ... }
    ///   bool ok = Eigen.inversePowerIteration(in A, ref v, out var lambda);
    ///   var info = Eigen.powerIteration(in A, ref v, ref w, out var lambda, tolerance, maxIterations);
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
        /// <c>iterations == maxIterations</c> (the same value a MaxIterations return carries); always read
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

        /// <summary>Burst-safe compact summary, e.g. <c>EigenSolveInfo(Converged, iters=12,
        /// residual=1.23E-08)</c>. Never allocates managed memory.</summary>
        public FixedString128Bytes ToFixedString()
        {
            FixedString128Bytes str = "EigenSolveInfo(";
            str.Append(status.Name());
            FixedString128Bytes tail = $", iters={iterations}, residual={residual:G3})";
            str.Append(tail);
            return str;
        }

        /// <summary>Managed wrapper -- do not call from inside a [BurstCompile] job.</summary>
        public override string ToString() => ToFixedString().ToString();
    }

    /// <summary>
    /// Result of a dense eigensolve (<c>Eigen.symmetricInPlace</c> / <c>Eigen.valuesSymmetricInPlace</c> /
    /// <c>Eigen.valuesQR</c> / <c>Eigen.decompInPlace</c>), returned by value. Implicitly converts to
    /// <c>bool</c> (== <see cref="Solved"/>) for use in <c>if (...)</c>:
    /// <code>
    ///   if (Eigen.symmetricInPlace(ref A, ref eigenvalues, ref V)) { ... }   // implicit bool
    ///   bool ok = Eigen.valuesSymmetricInPlace(ref A, ref eigenvalues);      // same
    ///   var info = Eigen.symmetricInPlace(ref A, ref eigenvalues, ref V);
    ///   if (info.Solved) Debug.Log(info.sweeps);
    /// </code>
    ///
    /// Reuses <see cref="IterativeSolveStatus"/> (the same enum every other iterative solver in
    /// this library uses) rather than a dedicated Eigen enum: the tridiagonal QL / Hessenberg QR /
    /// cyclic Jacobi iteration either fully converges (Converged) or exhausts its budget
    /// (MaxIterations) -- there is no breakdown mode of its own.
    ///
    /// <see cref="sweeps"/> and <see cref="converged"/> are filled from counters the QL/QR/Jacobi
    /// iteration already tracks while it runs -- never a separate pass. There is no residual field.
    ///
    /// On a MaxIterations return the outputs are NOT usable: eigenvalues/eigenvectors are
    /// unwritten or partial -- always check the returned status before reading them.
    ///
    /// Twin of <see cref="SVDInfo"/> (same shape, deliberately a separate type).
    /// </summary>
    public struct EigenInfo
    {
        /// <summary>Why the eigensolve stopped -- see <see cref="IterativeSolveStatus"/>. Eigen has
        /// no Breakdown mode; only Converged or MaxIterations.</summary>
        public IterativeSolveStatus status;

        /// <summary>Maximum number of QL/QR sweeps (or, for the obsolete cyclic-Jacobi
        /// <c>decompInPlace</c>, full-matrix Jacobi sweeps) consumed by any SINGLE eigenvalue
        /// during this call (the worst-case bottleneck) -- compare against the per-value iteration
        /// budget (<c>Consts.sweepBudget(n)</c>, i.e. <c>max(75, 6*n)</c>, by default) to gauge how
        /// much margin a workload has. 0 is valid (every value deflated immediately). On
        /// MaxIterations this equals the budget that was exhausted.</summary>
        public int sweeps;

        /// <summary>Count of eigenvalues that had already converged when the solve stopped (0..n).
        /// Equals n iff <see cref="status"/> is Converged. For the all-or-nothing cyclic-Jacobi
        /// <c>decompInPlace</c> (which does not resolve individual eigenvalues independently) this
        /// is n on success and 0 on MaxIterations.</summary>
        public int converged;

        /// <summary>True iff every eigenvalue converged (<c>status ==
        /// IterativeSolveStatus.Converged</c>). Same value as the implicit bool conversion; use
        /// whichever reads better.</summary>
        public bool Solved => status == IterativeSolveStatus.Converged;

        /// <summary>Implicit success test, so <c>if (Eigen.symmetricInPlace(...))</c> / <c>bool ok =
        /// Eigen.symmetricInPlace(...)</c> keep compiling after the return type changed from bool to this
        /// struct.</summary>
        public static implicit operator bool(EigenInfo i) => i.status == IterativeSolveStatus.Converged;

        /// <summary>Burst-safe compact summary, e.g. <c>EigenInfo(Converged, sweeps=3, converged=64)</c>.
        /// Never allocates managed memory.</summary>
        public FixedString128Bytes ToFixedString()
        {
            FixedString128Bytes str = "EigenInfo(";
            str.Append(status.Name());
            FixedString128Bytes tail = $", sweeps={sweeps}, converged={converged})";
            str.Append(tail);
            return str;
        }

        /// <summary>Managed wrapper -- do not call from inside a [BurstCompile] job.</summary>
        public override string ToString() => ToFixedString().ToString();
    }

    /// <summary>
    /// Result of a symmetric Lanczos tridiagonalization (<c>Eigen.lanczos</c> /
    /// <c>Eigen.lanczosVectors</c>), returned by value (the value-returning "allocating" overloads
    /// carry it as an <c>out</c> parameter alongside their <c>fProxyN</c>/<c>fProxyMxN</c> outputs).
    /// Implicitly converts to <c>bool</c> (== <see cref="Solved"/>) for use in <c>if (...)</c>:
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

        /// <summary>Burst-safe compact summary, e.g. <c>LanczosInfo(Converged, produced=20)</c>.
        /// Never allocates managed memory.</summary>
        public FixedString128Bytes ToFixedString()
        {
            FixedString128Bytes str = "LanczosInfo(";
            str.Append(status.Name());
            FixedString128Bytes tail = $", produced={produced})";
            str.Append(tail);
            return str;
        }

        /// <summary>Managed wrapper -- do not call from inside a [BurstCompile] job.</summary>
        public override string ToString() => ToFixedString().ToString();
    }
}
