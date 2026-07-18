//singularFile//
using Unity.Collections;

namespace LinearAlgebra
{
    /// <summary>
    /// Result of a least-squares Krylov solve (<c>lsqr</c> / <c>lsmr</c>). Every LS
    /// solver RETURNS this by value; an implicit <c>bool</c> conversion (== <see cref="Solved"/>)
    /// lets old success-test call shapes keep compiling, e.g. <c>if (Krylov.lsqr(A, b, ref x))</c>.
    /// Carries <see cref="rnorm"/> (‖b - A x‖), <see cref="Arnorm"/> (‖Aᵀ(b - A x)‖, or with
    /// Tikhonov damping ‖Aᵀ(b - A x) - damp²x‖), <see cref="xnorm"/> (‖x‖), <see cref="iterations"/>,
    /// and <see cref="status"/>. Norms are always <c>double</c> regardless of the solve's precision,
    /// and are the solver's own running values, not independently recomputed -- for a certified-exact
    /// residual, call <see cref="Krylov.lstsqResidual{TOp}"/> on the returned x instead.
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
        /// bidiagonalization/CG step; equals maxIter when it ran out).</summary>
        public int iterations;

        /// <summary>Why the solve stopped -- see <see cref="IterativeSolveStatus"/>.</summary>
        public IterativeSolveStatus status;

        /// <summary>True iff the solver reached its tolerance (<c>status == IterativeSolveStatus.Converged</c>).
        /// Same value as the implicit bool conversion; use whichever reads better.</summary>
        public bool Solved => status == IterativeSolveStatus.Converged;

        /// <summary>Implicit success test, so <c>if (solve(...))</c> / <c>bool ok = solve(...)</c>
        /// still reads as a success test.</summary>
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
    /// Result of a square-system Krylov solve (<c>cg</c> / <c>cg</c> /
    /// <c>minres</c> / <c>biCGStab</c>). Same contract as <see cref="LstsqInfo"/> --
    /// returned by value, implicit <c>bool</c> == <see cref="Solved"/>, norm reported as <c>double</c>
    /// -- but carries only the residual norm ‖b - A x‖ (no Aᵀr / xnorm: for a square solve the
    /// residual IS the error measure). Filled from each solver's tracked residual (cg: a
    /// live ‖r‖; minres: phibar; biCGStab: its running ‖r‖). cg verify a claimed Converged
    /// exit with one fresh r = b-Ax first; minres/biCGStab are unaffected (no extra
    /// matvec on any of their exits).
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
        /// maxIter when it ran out).</summary>
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
    ///   if (LU.decomp(in A, ref L, ref U, ref P)) { ... }       // implicit bool -> "did it succeed?"
    ///   bool ok = CHO.decomp(in A, ref L);                      // same
    ///   var info = LU.decompInPlace(ref A_to_LU, ref P);
    ///   if (!info.Solved) { /* singular */ }
    /// </code>
    ///
    /// The <see cref="status"/> field is filled ONLY from what the solver already determined during
    /// its normal control flow (an existing bool return, an existing early-return condition) --
    /// never from a new check. Most direct solves that don't factorize (e.g. LU.decompSolve given an
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
        /// still reads as a success test.</summary>
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
    /// Result of an SVD solve (<c>SVD.thin</c> / <c>SVD.values</c> / <c>SVD.truncated</c> /
    /// <c>SVD.randomized</c>). Every converted SVD entry point RETURNS this by value; an implicit
    /// <c>bool</c> conversion (== <see cref="Solved"/>) means the old success-test call shapes
    /// still compile unchanged:
    /// <code>
    ///   if (SVD.thin(in A, ref U, ref S, ref V)) { ... }      // implicit bool -> "did it converge?"
    ///   bool ok = SVD.values(in A, ref S);                    // same
    ///   var info = SVD.thin(in A, ref U, ref S, ref V);
    ///   if (info.Solved) Debug.Log(info.sweeps);
    /// </code>
    ///
    /// Reuses <see cref="IterativeSolveStatus"/> (the same enum every other iterative solver in
    /// this library uses) rather than a dedicated SVD enum: the bidiagonal QR either fully
    /// diagonalizes (Converged) or exhausts its per-value budget on some singular value
    /// (MaxIterations) -- there is no breakdown mode of its own.
    ///
    /// <see cref="sweeps"/> and <see cref="converged"/> are filled from counters the bidiagonal QR
    /// already tracks per singular value while it runs -- never a separate pass. NO residual field
    /// (that is what the test oracles are for, not this struct).
    ///
    /// On a MaxIterations return the outputs are NOT usable: S/U/V are unwritten or partial --
    /// always check the returned status before reading them.
    ///
    /// Twin of <see cref="EigenInfo"/> (same shape, deliberately a SEPARATE type -- SVD and Eigen
    /// results are not interchangeable even though their diagnostics look alike).
    /// </summary>
    public struct SVDInfo
    {
        /// <summary>Why the SVD stopped -- see <see cref="IterativeSolveStatus"/>. SVD has no
        /// Breakdown mode; only Converged or MaxIterations.</summary>
        public IterativeSolveStatus status;

        /// <summary>Maximum number of QR sweeps consumed by any SINGLE singular value during this
        /// call (the worst-case bottleneck) -- compare against the per-value iteration budget
        /// (<c>Consts.sweepBudget(n)</c>, i.e. <c>max(75, 6*n)</c>, by default) to gauge how much
        /// margin a workload has. 0 is valid (every value deflated immediately, no sweep needed).
        /// On MaxIterations this equals the budget that was exhausted.</summary>
        public int sweeps;

        /// <summary>Count of singular values that had already converged when the solve stopped,
        /// relative to the problem the QR pass actually iterated: the full n for
        /// <c>thin</c>/<c>values</c>, the reduced panel size for <c>truncated</c>/<c>randomized</c>.
        /// Equals that problem size iff <see cref="status"/> is Converged.</summary>
        public int converged;

        /// <summary>True iff every singular value converged (<c>status ==
        /// IterativeSolveStatus.Converged</c>). Same value as the implicit bool conversion; use
        /// whichever reads better.</summary>
        public bool Solved => status == IterativeSolveStatus.Converged;

        /// <summary>Implicit success test, so <c>if (SVD.thin(...))</c> / <c>bool ok =
        /// SVD.thin(...)</c> still reads as a success test.</summary>
        public static implicit operator bool(SVDInfo i) => i.status == IterativeSolveStatus.Converged;

        /// <summary>Burst-safe compact summary, e.g. <c>SVDInfo(Converged, sweeps=4, converged=512)</c>.
        /// Never allocates managed memory.</summary>
        public FixedString128Bytes ToFixedString()
        {
            FixedString128Bytes str = "SVDInfo(";
            str.Append(status.Name());
            FixedString128Bytes tail = $", sweeps={sweeps}, converged={converged})";
            str.Append(tail);
            return str;
        }

        /// <summary>Managed wrapper -- do not call from inside a [BurstCompile] job.</summary>
        public override string ToString() => ToFixedString().ToString();
    }

    /// <summary>
    /// Result of a RANK-REVEALING direct solver or decomposition call -- QRCP
    /// (<c>QRCP.solveInPlace</c>) and pivoted Cholesky (<c>CHOP.decomp</c> /
    /// <c>CHOP.decompSolve</c>), AND of the SVD-backed rank-revealing calls
    /// (<c>SVD.pinvSolve</c>, <c>SVD.nullspaceBasis</c>, <c>SVD.rangeBasis</c>).
    /// Unlike <see cref="DirectSolveInfo"/>, a rank-deficient input is NOT a hard failure here: both
    /// algorithms still produce a usable (least-squares / minimum-norm) result when the detected
    /// rank is below the full dimension, so <see cref="Solved"/> is true for
    /// <see cref="DirectSolveStatus.Success"/> AND <see cref="DirectSolveStatus.RankDeficient"/> --
    /// only <see cref="DirectSolveStatus.Singular"/> / <see cref="DirectSolveStatus.NotPositiveDefinite"/>
    /// / <see cref="DirectSolveStatus.Indefinite"/> / <see cref="DirectSolveStatus.NotConverged"/>
    /// (the SVD-backed callers' non-convergence mapping) are true failures.
    ///
    /// Both fields are filled ONLY from values the solver already computes (the detected numerical
    /// rank it already counts, the status its existing control flow already determines) -- no new
    /// passes over the factor.
    /// </summary>
    public struct RankInfo
    {
        /// <summary>Why the solve/decomposition stopped -- see <see cref="DirectSolveStatus"/>.</summary>
        public DirectSolveStatus status;

        /// <summary>Detected numerical rank (0..n). Meaningful whenever <see cref="Solved"/> is
        /// true; undefined on a hard failure (Singular / NotPositiveDefinite / Indefinite /
        /// NotConverged).</summary>
        public int rank;

        /// <summary>True iff the result is usable -- either full rank
        /// (<c>status == DirectSolveStatus.Success</c>) or a still-usable rank-deficient result
        /// (<c>status == DirectSolveStatus.RankDeficient</c>).</summary>
        public bool Solved => status == DirectSolveStatus.Success || status == DirectSolveStatus.RankDeficient;

        /// <summary>Implicit success test, so <c>if (solve(...))</c> keeps compiling; true for both
        /// full-rank and (still-usable) rank-deficient results.</summary>
        public static implicit operator bool(RankInfo i) =>
            i.status == DirectSolveStatus.Success || i.status == DirectSolveStatus.RankDeficient;

        /// <summary>Burst-safe compact summary, e.g. <c>RankInfo(RankDeficient, rank=3)</c>.
        /// Never allocates managed memory.</summary>
        public FixedString128Bytes ToFixedString()
        {
            FixedString128Bytes str = "RankInfo(";
            str.Append(status.Name());
            FixedString128Bytes tail = $", rank={rank})";
            str.Append(tail);
            return str;
        }

        /// <summary>Managed wrapper -- do not call from inside a [BurstCompile] job.</summary>
        public override string ToString() => ToFixedString().ToString();
    }
}
