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

        /// <summary>Bounded-variable PRIMAL revised simplex over an LU-factored basis (HiGHS-lineage):
        /// FTRAN/BTRAN + a product-form-of-the-inverse eta file replace the tableau's O(mn) per-pivot
        /// update. Exact vertex solution like <see cref="Simplex"/>, with native bounded variables and
        /// periodic refactorization instead of tableau error accumulation.</summary>
        RevisedSimplex = 2,

        /// <summary>Bounded-variable DUAL revised simplex (HiGHS-lineage): dual steepest-edge pricing
        /// (Forrest-Goldfarb) + a long-step (bound-flipping) Harris ratio test drive an initially
        /// dual-feasible basis to primal feasibility, with artificial-bounds dual phase 1 and
        /// HiGHS-style cost perturbation for degenerate problems; the terminal basis is handed to
        /// <see cref="RevisedSimplex"/>'s primal core as a cleanup pass -- the same composition HiGHS
        /// itself uses. Exact vertex solution, same kernel layer (FTRAN/BTRAN/eta file) as
        /// <see cref="RevisedSimplex"/>.</summary>
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

    /// <summary>
    /// Captures the revised/dual simplex's COMPLETE basis state -- everything <see cref="LP.solve"/>
    /// needs to seed a re-solve from an existing vertex instead of the all-logical start, via the
    /// <c>ref LPBasis</c> overload of <c>LP.solve</c> (LP.fProxy.cs). Type-agnostic (plain byte/int
    /// buffers, no fProxy field), so it lives here rather than in either per-dtype template.
    ///
    /// Computational-form indexing: a problem with n structural variables and m constraints has
    /// N_total = n + m variables (n structural + m logical/slack); <see cref="status"/> tags each of
    /// the N_total variables one of <c>LP.STATUS_BASIC</c> / <c>LP.STATUS_AT_LOWER</c> /
    /// <c>LP.STATUS_AT_UPPER</c>, and <see cref="basis"/>[i] is the variable index basic in row i.
    ///
    /// RE-SOLVE USE CASE: solve an LP, capture its terminal basis, perturb the problem (a tightened
    /// bound, a changed rhs), then re-solve with the SAME basis via <c>LP.solve(..., ref basis)</c> --
    /// the dual simplex only needs to repair primal feasibility from there (a handful of pivots)
    /// instead of rebuilding a vertex from scratch.
    ///
    /// FACTOR/WEIGHT PERSISTENCE: this struct stays dtype-agnostic, so the per-call fixed costs a
    /// warm re-solve still pays (rebuilding the computational form, refactorizing the basis) are NOT
    /// captured here -- pair it with a per-dtype <c>fProxyLPCache</c> (LP.Cache.fProxy.cs) via
    /// <c>LP.solve(..., ref basis, ref fProxyLPCache cache)</c> for that.
    ///
    /// LIFECYCLE: user-allocated, mirroring <see cref="Pivot"/>'s own <c>(size, Allocator)</c> +
    /// <see cref="Dispose"/> pattern -- no arena requirement, since this needs to persist ACROSS
    /// separate top-level solve calls. Three ways to arrive at <c>LP.solve(..., ref basis)</c>:
    ///   * <c>default(LPBasis)</c> (not <see cref="IsCreated"/>): cold solve, ALLOCATES this struct
    ///     itself (<c>Allocator.Persistent</c>, MANAGED-THREAD ONLY -- a Burst job cannot make this
    ///     allocation) before writing the terminal basis into it. Caller must <see cref="Dispose"/>.
    ///   * <c>new LPBasis(n, m, allocator)</c>, otherwise untouched (<see cref="populated"/> false):
    ///     job-safe (e.g. <c>Allocator.Temp</c>) -- no allocation inside the solve call, which seeds
    ///     the existing buffers with the all-logical start and marks it <see cref="populated"/>.
    ///   * Already <see cref="populated"/> (result of an earlier call, cold or warm, on ANY
    ///     same-shape problem): dimension-validated then used AS THE STARTING POINT.
    /// </summary>
    public struct LPBasis
    {
        /// <summary>Status tag per variable (length N_total = n structural + m logical), one of
        /// <c>LP.STATUS_BASIC</c> / <c>LP.STATUS_AT_LOWER</c> / <c>LP.STATUS_AT_UPPER</c>.</summary>
        public NativeArray<byte> status;

        /// <summary><c>basis[i]</c> = index (into the N_total-wide variable space <see cref="status"/>
        /// indexes) of the variable basic in row i. Length m.</summary>
        public NativeArray<int> basis;

        /// <summary>False on a freshly-constructed instance; <c>LP.solve(..., ref basis)</c> sets this
        /// true once <see cref="status"/>/<see cref="basis"/> hold a real terminal basis (seeded by a
        /// cold solve, or supplied warm by the caller). Distinguishes "just allocated, buffers are
        /// zero-filled garbage" from "has real content" independently of <see cref="IsCreated"/>, since
        /// the job-safe construction path (see the type's own doc comment) allocates well before it has
        /// anything meaningful to put in the buffers.</summary>
        public bool populated;

        /// <summary>
        /// Allocates a basis sized for <paramref name="n"/> structural variables and
        /// <paramref name="m"/> constraints (N_total = n + m). Contents are zero-initialized and
        /// <see cref="populated"/> is false -- NOT a valid warm seed on its own; either pass it straight
        /// to <c>LP.solve(..., ref basis)</c> (which recognizes the unpopulated state and seeds it with
        /// an ordinary cold solve, job-safe), or fill it from a matching-shape problem's own terminal
        /// state (and set <see cref="populated"/> = true) before passing it as a warm seed.
        /// </summary>
        public LPBasis(int n, int m, Allocator allocator)
        {
            status = new NativeArray<byte>(n + m, allocator);
            basis = new NativeArray<int>(m, allocator);
            populated = false;
        }

        /// <summary>True once both buffers are allocated (regardless of content validity).</summary>
        public bool IsCreated => status.IsCreated && basis.IsCreated;

        /// <summary>True for a never-constructed / already-<see cref="Dispose"/>d instance (e.g.
        /// <c>default(LPBasis)</c>), OR a constructed-but-not-yet-<see cref="populated"/> one -- either
        /// way, <c>LP.solve(..., ref basis)</c> treats this as "no warm state, run the cold solve".</summary>
        public bool IsEmpty => !IsCreated || !populated;

        /// <summary>True iff this basis is allocated AND sized for exactly <paramref name="n"/>
        /// structural variables / <paramref name="m"/> constraints -- the dimension check
        /// <c>LP.solve</c>'s <c>ref LPBasis</c> overload runs before trusting a created basis (empty or
        /// populated) as a seed (a mismatch throws rather than silently misreading the buffers).</summary>
        public bool IsValid(int n, int m) => IsCreated && basis.Length == m && status.Length == n + m;

        /// <summary>Releases both buffers. Safe to call on an empty/already-disposed instance.</summary>
        public void Dispose()
        {
            if (status.IsCreated) status.Dispose();
            if (basis.IsCreated) basis.Dispose();
        }

        /// <summary>Burst-safe compact summary, e.g. <c>LPBasis(n_total=16, m=4, created=true)</c>.
        /// Never allocates managed memory.</summary>
        public FixedString128Bytes ToFixedString()
        {
            FixedString128Bytes str = $"LPBasis(n_total={status.Length}, m={basis.Length}, created=";

            if (IsCreated)
            {
                FixedString32Bytes flag = "true)";
                str.Append(flag);
            }
            else
            {
                FixedString32Bytes flag = "false)";
                str.Append(flag);
            }

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

        // NOTE: the LAD hybrid-default routing threshold (BR vs FN in LP.lad's method-less overload)
        // is NOT here -- the crossover is per-dtype, so it lives as an inline /*+choose[..|..]*/
        // literal on the dispatch expression in LP.fProxy.cs, where each generated build gets its own
        // value. A type-agnostic const here cannot express that.

        // LP.ladBR's (LP.BarrodaleRoberts.fProxy.cs) ratio-test candidate-consumption gate: above this
        // many candidates in a single entering-column's ratio test, sort them once (O(nCand log nCand))
        // instead of the original repeated-linear-scan-for-minimum (O(nCand^2)) -- see that file's own
        // comment at the call site for the full rationale. Type-agnostic (plain int), same CS0102
        // reasoning as REFACTOR_INTERVAL above.
        internal const int BR_CAND_SORT_THRESHOLD = 256;
    }
}
