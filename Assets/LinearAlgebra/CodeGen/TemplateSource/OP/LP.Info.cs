//singularFile//
using Unity.Collections;

namespace LinearAlgebra
{
    /// <summary>
    /// Sense of a single linear-program constraint row: the relation between the row's dot product
    /// <c>Aᵢ·x</c> and its right-hand side <c>bᵢ</c>. Passed as a per-row <see cref="NativeArray{T}"/>
    /// to <see cref="LP.solve"/>. Type-agnostic (no fProxy) on purpose -- it lives in a non-templated
    /// file so codegen does not emit a duplicate definition into both the float and double partials
    /// (CS0102), exactly like <see cref="IterativeSolveStatus"/>.
    ///
    /// The backing values are the sign of the inequality direction (−1 / 0 / +1) so a caller can build
    /// the array arithmetically, and <see cref="ConstraintSense.Equal"/> is the zero default.
    /// </summary>
    public enum ConstraintSense
    {
        /// <summary>Aᵢ·x ≤ bᵢ (add a non-negative slack).</summary>
        LessEqual = -1,

        /// <summary>Aᵢ·x = bᵢ (no slack; needs a phase-1 artificial).</summary>
        Equal = 0,

        /// <summary>Aᵢ·x ≥ bᵢ (subtract a non-negative surplus; needs a phase-1 artificial).</summary>
        GreaterEqual = 1,
    }

    /// <summary>Which backend <see cref="LP.solve"/> uses. Both reach the same optimal vertex on a
    /// bounded, feasible LP; they differ in cost profile and in the kind of point they converge to
    /// (a vertex vs an interior point rounded to one). Type-agnostic (CS0102), same reason as
    /// <see cref="ConstraintSense"/>.</summary>
    public enum LPMethod
    {
        /// <summary>Two-phase dense tableau simplex with Bland's anti-cycling rule. Exact vertex
        /// solution, fully deterministic, robust at modest dense sizes. Default.</summary>
        Simplex = 0,

        /// <summary>Mehrotra primal-dual interior point. Polynomial, scales to larger/denser LPs,
        /// reuses the Cholesky normal-equation stack; converges to an interior optimum.</summary>
        InteriorPoint = 1,

        /// <summary>Bounded-variable PRIMAL revised simplex over an LU-factored basis (HiGHS-lineage,
        /// stage 1 of docs/spec-revised-simplex.md): FTRAN/BTRAN + a product-form-of-the-inverse eta
        /// file replace the tableau's O(mn) per-pivot update. Exact vertex solution like
        /// <see cref="Simplex"/>, with native bounded variables and periodic refactorization instead of
        /// tableau error accumulation.</summary>
        RevisedSimplex = 2,

        /// <summary>Bounded-variable DUAL revised simplex (HiGHS-lineage, stage 2 of
        /// docs/spec-revised-simplex.md): dual steepest-edge pricing (Forrest-Goldfarb) + a long-step
        /// (bound-flipping) Harris ratio test drive an initially dual-feasible basis to primal
        /// feasibility, with artificial-bounds dual phase 1 and HiGHS-style cost perturbation for
        /// degenerate problems; the terminal basis is handed to <see cref="RevisedSimplex"/>'s primal
        /// core as a cleanup pass -- the same composition HiGHS itself uses. Exact vertex solution, same
        /// kernel layer (FTRAN/BTRAN/eta file) as <see cref="RevisedSimplex"/>.</summary>
        DualSimplex = 3,
    }

    /// <summary>
    /// Terminal state of an <see cref="LP.solve"/> / <see cref="LP.lad"/> call, carried by
    /// <see cref="LPInfo"/>. Type-agnostic (CS0102), mirroring <see cref="DirectSolveStatus"/>'s role
    /// for the factorization solvers.
    /// </summary>
    public enum LPStatus
    {
        /// <summary>A finite optimal solution was found; <c>x</c> and <see cref="LPInfo.objective"/>
        /// describe it.</summary>
        Optimal = 0,

        /// <summary>The feasible region is empty (phase 1 could not drive the artificial variables to
        /// zero). No usable <c>x</c>.</summary>
        Infeasible = 1,

        /// <summary>The objective decreases without bound along a feasible ray (an entering column has
        /// no limiting ratio). No finite optimum; <c>x</c> is the last vertex before the unbounded
        /// edge was detected.</summary>
        Unbounded = 2,

        /// <summary>The iteration budget was exhausted before an optimality/unboundedness certificate
        /// was reached. <c>x</c> is the last iterate (feasible, but not proven optimal).</summary>
        MaxIterations = 3,
    }

    /// <summary>
    /// Burst-safe enum-to-name helper for <see cref="LPStatus"/>, used by
    /// <see cref="LPInfo.ToFixedString"/>. A manual <c>switch</c> returning a
    /// <see cref="FixedString32Bytes"/> literal per case -- <c>enum.ToString()</c> is NOT Burst-legal.
    /// </summary>
    public static class LPStatusExtensions
    {
        public static FixedString32Bytes Name(this LPStatus s)
        {
            switch (s)
            {
                case LPStatus.Optimal: return "Optimal";
                case LPStatus.Infeasible: return "Infeasible";
                case LPStatus.Unbounded: return "Unbounded";
                case LPStatus.MaxIterations: return "MaxIterations";
                default: return "Unknown";
            }
        }
    }

    /// <summary>
    /// Result of a linear-program solve (<see cref="LP.solve"/> / <see cref="LP.lad"/>). Returned by
    /// value; an implicit <c>bool</c> conversion (== <see cref="Solved"/>) means the natural success
    /// test reads well:
    /// <code>
    ///   if (LP.solve(A, b, c, senses, ref x, out var obj)) { ... }   // implicit bool -> "optimal?"
    ///   var info = LP.lad(A, b, ref x, out var l1);
    ///   if (info.Solved) Debug.Log(info.iterations);
    /// </code>
    ///
    /// <see cref="objective"/> is reported as <c>double</c> regardless of the solve's precision (a
    /// float solve widens its float objective) -- diagnostics need not be precision-typed, which is
    /// why this is a plain, unprefixed struct rather than a float/double-generated one, matching
    /// <see cref="LstsqInfo"/> / <see cref="SolveInfo"/>. It is the value <c>cᵀx</c> at the returned
    /// <c>x</c> (for <see cref="LP.lad"/>, that is exactly the L1 residual ‖Ax − b‖₁).
    ///
    /// On <see cref="LPStatus.Optimal"/> the outputs are the optimal vertex. On
    /// <see cref="LPStatus.MaxIterations"/> they are the last feasible iterate. On
    /// <see cref="LPStatus.Infeasible"/> / <see cref="LPStatus.Unbounded"/> <c>x</c> is not a usable
    /// optimum -- check the status first.
    /// </summary>
    public struct LPInfo
    {
        /// <summary>Objective value <c>cᵀx</c> at the returned <c>x</c> (the L1 residual for
        /// <see cref="LP.lad"/>). Meaningful on <see cref="LPStatus.Optimal"/> /
        /// <see cref="LPStatus.MaxIterations"/>.</summary>
        public double objective;

        /// <summary>Total simplex pivots (phase 1 + phase 2), or interior-point iterations, actually
        /// performed. Equals the budget on a <see cref="LPStatus.MaxIterations"/> return.</summary>
        public int iterations;

        /// <summary>Why the solve stopped -- see <see cref="LPStatus"/>.</summary>
        public LPStatus status;

        /// <summary>True iff a finite optimum was found (<c>status == LPStatus.Optimal</c>). Same
        /// value as the implicit bool conversion; use whichever reads better.</summary>
        public bool Solved => status == LPStatus.Optimal;

        /// <summary>Implicit success test, so <c>if (LP.solve(...))</c> / <c>bool ok = LP.lad(...)</c>
        /// read as "did it reach an optimum".</summary>
        public static implicit operator bool(LPInfo info) => info.status == LPStatus.Optimal;

        /// <summary>Burst-safe compact summary, e.g. <c>LPInfo(Optimal, iters=23, obj=1.42E+01)</c>.
        /// Never allocates managed memory.</summary>
        public FixedString128Bytes ToFixedString()
        {
            FixedString128Bytes str = "LPInfo(";
            str.Append(status.Name());
            FixedString128Bytes tail = $", iters={iterations}, obj={objective:G4})";
            str.Append(tail);
            return str;
        }

        /// <summary>Managed wrapper -- do not call from inside a [BurstCompile] job.</summary>
        public override string ToString() => ToFixedString().ToString();
    }

    // Shared constants for the revised-simplex kernel layer (LP.RevisedSimplex.fProxy.cs) and the dual
    // simplex built on it (LP.DualSimplex.fProxy.cs). Type-agnostic (plain int/byte, no fProxy token),
    // so they live here in this non-templated (singularFile) file -- exactly like ConstraintSense/
    // LPMethod/LPStatus above -- rather than in either per-dtype template, where an identical
    // declaration would land in BOTH the float and double generated partials of `partial class LP` and
    // collide (CS0102: a member can't be declared twice across a partial class's fragments, even from
    // different source files).
    public static partial class LP
    {
        // Eta-file (product-form-of-the-inverse) refresh cadence: refactorize from scratch after this
        // many pivots instead of growing the eta chain further (bounds FTRAN/BTRAN's per-solve cost and
        // resets accumulated floating-point error).
        internal const int REFACTOR_INTERVAL = 64;

        // Nonbasic-variable status tags (basis[] and status[] together are the revised simplex's whole
        // state): a BASIC variable's value lives in xB; an AT_LOWER/AT_UPPER nonbasic sits exactly on
        // that bound (fixed variables, l==u, are always tagged AT_LOWER).
        internal const byte STATUS_BASIC = 0;
        internal const byte STATUS_AT_LOWER = 1;
        internal const byte STATUS_AT_UPPER = 2;
    }
}
