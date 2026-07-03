//singularFile//
using Unity.Collections;

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
    /// The norms are reported as <c>double</c> regardless of the solve's precision (a float solve
    /// widens its float residual -- diagnostics don't need to be precision-typed, which is why this
    /// is a plain, unprefixed struct rather than a float/double-generated one). They are filled from
    /// values the solver ALREADY tracks (or at most a single dot on a residual it already holds) at
    /// the point it returns -- never a fresh A*x/Aᵀ*r, so the struct costs nothing beyond the solve:
    /// <list type="bullet">
    /// <item>cgls -- rnorm from a dot on its live residual r; Arnorm = √gamma (its tracked ‖Aᵀr‖²).</item>
    /// <item>lsqr -- rnorm = phibar, Arnorm = phibar·alpha·|c|, both produced free by the recurrence.</item>
    /// <item>lsmr -- Arnorm = |ζ̄| (free, monotone); rnorm via the Fong-Saunders ‖r‖ recurrence
    ///       (O(1) scalars per iteration, no matvec).</item>
    /// </list>
    /// For an independently-recomputed, certified-exact residual (one extra Apply + ApplyT) call
    /// <see cref="Solvers.lstsqResidual{TOp}"/> on the returned x instead.
    ///
    /// On a Converged OR MaxIterations return, x is the last iterate and the norms describe it. Only
    /// on a Breakdown return is x left partially updated / undefined.
    /// </summary>
    public struct LstsqInfo
    {
        /// <summary>Residual norm ‖b - A x‖. Nonzero for an inconsistent (over-determined) system
        /// even at the optimum -- it is the least-squares residual, not an error.</summary>
        public double rnorm;

        /// <summary>Normal-equation residual ‖Aᵀ(b - A x)‖ -- or, when solved with Tikhonov damping,
        /// ‖Aᵀ(b - A x) - damp²x‖. This is the true least-squares optimality measure: it goes to
        /// zero at the minimizer regardless of whether the system is consistent.</summary>
        public double Arnorm;

        /// <summary>Solution norm ‖x‖ (useful for tuning Tikhonov damping / monitoring blow-up on
        /// ill-conditioned problems).</summary>
        public double xnorm;

        /// <summary>Iterations actually performed (0 when the solver converged before the first
        /// bidiagonalization/CG step; equals maxIterations when it ran out).</summary>
        public int iterations;

        /// <summary>Why the solve stopped -- see <see cref="IterativeSolveStatus"/>.</summary>
        public IterativeSolveStatus status;

        /// <summary>True iff the solver reached its tolerance (<c>status == IterativeSolveStatus.Converged</c>).
        /// Same value as the implicit bool conversion; use whichever reads better.</summary>
        public bool Solved => status == IterativeSolveStatus.Converged;

        /// <summary>Implicit success test, so <c>if (solve(...))</c> / <c>bool ok = solve(...)</c>
        /// keep compiling after the return type changed from bool to this struct.</summary>
        public static implicit operator bool(LstsqInfo info) => info.status == IterativeSolveStatus.Converged;

        /// <summary>Burst-safe compact summary, e.g. <c>LstsqInfo(Converged, iters=17, rnorm=1.23E-08,
        /// Arnorm=4.56E-09, xnorm=2.10E+00)</c>. Never allocates managed memory.</summary>
        public FixedString128Bytes ToFixedString()
        {
            FixedString128Bytes str = "LstsqInfo(";
            str.Append(status.Name());
            FixedString128Bytes tail = $", iters={iterations}, rnorm={rnorm:G3}, Arnorm={Arnorm:G3}, xnorm={xnorm:G3})";
            str.Append(tail);
            return str;
        }

        /// <summary>Managed wrapper -- do not call from inside a [BurstCompile] job.</summary>
        public override string ToString() => ToFixedString().ToString();
    }

    /// <summary>
    /// Result of a square-system Krylov solve (<c>cg</c> / <c>pcg</c> /
    /// <c>minres</c> / <c>biCGStab</c> / <c>cgne</c>). Same contract as <see cref="LstsqInfo"/> --
    /// returned by value, implicit <c>bool</c> == <see cref="Solved"/>, norm reported as <c>double</c>
    /// -- but carries only the residual norm ‖b - A x‖ (no Aᵀr / xnorm: for a square solve the
    /// residual IS the error measure). Filled from each solver's tracked residual (cg/pcg/cgne: a
    /// live ‖r‖; minres: phibar; biCGStab: its running ‖r‖) -- no extra matvec.
    ///
    /// On a Converged OR MaxIterations return, x is the last iterate and rnorm is its true residual
    /// ‖b - A x‖ (so on MaxIterations you can inspect how close it got). Only on a Breakdown return
    /// is x left partially updated / undefined -- there rnorm describes the pre-breakdown iterate,
    /// not a usable solution.
    /// </summary>
    public struct SolveInfo
    {
        /// <summary>Residual norm ‖b - A x‖ at the returned x.</summary>
        public double rnorm;

        /// <summary>Iterations actually performed (0 when converged before the first step; equals
        /// maxIterations when it ran out).</summary>
        public int iterations;

        /// <summary>Why the solve stopped -- see <see cref="IterativeSolveStatus"/>.</summary>
        public IterativeSolveStatus status;

        /// <summary>True iff the solver reached its tolerance (<c>status == IterativeSolveStatus.Converged</c>).</summary>
        public bool Solved => status == IterativeSolveStatus.Converged;

        /// <summary>Implicit success test so <c>if (solve(...))</c> keeps compiling.</summary>
        public static implicit operator bool(SolveInfo info) => info.status == IterativeSolveStatus.Converged;

        /// <summary>Burst-safe compact summary, e.g. <c>SolveInfo(Converged, iters=42, rnorm=1.23E-08)</c>.
        /// Never allocates managed memory.</summary>
        public FixedString128Bytes ToFixedString()
        {
            FixedString128Bytes str = "SolveInfo(";
            str.Append(status.Name());
            FixedString128Bytes tail = $", iters={iterations}, rnorm={rnorm:G3})";
            str.Append(tail);
            return str;
        }

        /// <summary>Managed wrapper -- do not call from inside a [BurstCompile] job.</summary>
        public override string ToString() => ToFixedString().ToString();
    }

    /// <summary>
    /// Result of a DIRECT (non-iterative, factorization-based) solver or decomposition call that
    /// has no notion of "rank" to report -- LU, plain (un-pivoted) Cholesky, un-pivoted QR/LQ, and
    /// the triangular-solve primitives. Every converted direct solver/decomposition RETURNS this by
    /// value; an implicit <c>bool</c> conversion (== <see cref="Solved"/>) means old success-test
    /// call shapes still compile unchanged:
    /// <code>
    ///   if (LU.luDecomposition(ref U, ref L, ref P)) { ... }   // implicit bool -> "did it succeed?"
    ///   bool ok = Cholesky.choleskyDecomposition(in A, ref L); // same
    ///   var info = LU.luDecompositionInPlace(ref LU, ref P);
    ///   if (!info.Solved) { /* singular */ }
    /// </code>
    ///
    /// The <see cref="status"/> field is filled ONLY from what the solver already determined during
    /// its normal control flow (an existing bool return, an existing early-return condition) --
    /// never from a new check. Most direct solves that don't factorize (e.g. luSolve given an
    /// already-valid factor, the triangular-solve primitives) always report
    /// <see cref="DirectSolveStatus.Success"/>: they have no failure mode of their own, and do not
    /// re-verify a factor's validity.
    /// </summary>
    public struct DirectSolveInfo
    {
        /// <summary>Why the solve/decomposition stopped -- see <see cref="DirectSolveStatus"/>.</summary>
        public DirectSolveStatus status;

        /// <summary>True iff the solve/decomposition completed normally
        /// (<c>status == DirectSolveStatus.Success</c>).</summary>
        public bool Solved => status == DirectSolveStatus.Success;

        /// <summary>Implicit success test, so <c>if (solve(...))</c> / <c>bool ok = solve(...)</c>
        /// keep compiling after the return type changed from bool/void to this struct.</summary>
        public static implicit operator bool(DirectSolveInfo i) => i.status == DirectSolveStatus.Success;

        /// <summary>Burst-safe compact summary, e.g. <c>DirectSolveInfo(Success)</c>. Never allocates
        /// managed memory.</summary>
        public FixedString128Bytes ToFixedString()
        {
            FixedString128Bytes str = "DirectSolveInfo(";
            str.Append(status.Name());
            str.Append(')');
            return str;
        }

        /// <summary>Managed wrapper -- do not call from inside a [BurstCompile] job.</summary>
        public override string ToString() => ToFixedString().ToString();
    }

    /// <summary>
    /// Result of a RANK-REVEALING direct solver or decomposition call -- QRCP
    /// (<see cref="QR.qrcpDirectSolve"/>) and pivoted Cholesky
    /// (<see cref="Cholesky.choleskyDecompositionPivot"/> / <see cref="Cholesky.choleskyPivotSolve"/>).
    /// Unlike <see cref="DirectSolveInfo"/>, a rank-deficient input is NOT a hard failure here: both
    /// algorithms still produce a usable (least-squares / minimum-norm) result when the detected
    /// rank is below the full dimension, so <see cref="Solved"/> is true for
    /// <see cref="DirectSolveStatus.Success"/> AND <see cref="DirectSolveStatus.RankDeficient"/> --
    /// only <see cref="DirectSolveStatus.Singular"/> / <see cref="DirectSolveStatus.NotPositiveDefinite"/>
    /// / <see cref="DirectSolveStatus.Indefinite"/> are true failures.
    ///
    /// Both fields are filled ONLY from values the solver already computes (the detected numerical
    /// rank it already counts, the status its existing control flow already determines) -- no new
    /// passes over the factor.
    /// </summary>
    public struct RankRevealingInfo
    {
        /// <summary>Why the solve/decomposition stopped -- see <see cref="DirectSolveStatus"/>.</summary>
        public DirectSolveStatus status;

        /// <summary>Detected numerical rank (0..n). Meaningful whenever <see cref="Solved"/> is
        /// true; undefined on a hard failure (Singular / NotPositiveDefinite / Indefinite).</summary>
        public int rank;

        /// <summary>True iff the result is usable -- either full rank
        /// (<c>status == DirectSolveStatus.Success</c>) or a still-usable rank-deficient result
        /// (<c>status == DirectSolveStatus.RankDeficient</c>).</summary>
        public bool Solved => status == DirectSolveStatus.Success || status == DirectSolveStatus.RankDeficient;

        /// <summary>Implicit success test, so <c>if (solve(...))</c> keeps compiling; true for both
        /// full-rank and (still-usable) rank-deficient results.</summary>
        public static implicit operator bool(RankRevealingInfo i) =>
            i.status == DirectSolveStatus.Success || i.status == DirectSolveStatus.RankDeficient;

        /// <summary>Burst-safe compact summary, e.g. <c>RankRevealingInfo(RankDeficient, rank=3)</c>.
        /// Never allocates managed memory.</summary>
        public FixedString128Bytes ToFixedString()
        {
            FixedString128Bytes str = "RankRevealingInfo(";
            str.Append(status.Name());
            FixedString128Bytes tail = $", rank={rank})";
            str.Append(tail);
            return str;
        }

        /// <summary>Managed wrapper -- do not call from inside a [BurstCompile] job.</summary>
        public override string ToString() => ToFixedString().ToString();
    }
}
