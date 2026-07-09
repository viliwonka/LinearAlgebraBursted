//singularFile//
using Unity.Collections;

namespace LinearAlgebra
{
    /// <summary>
    /// Terminal state of a <see cref="MIP.solve"/> branch-and-bound search, carried by
    /// <see cref="MIPInfo"/>. Stage 3: pseudocost + reliability branching, best-bound node queue with
    /// plunging. Still no propagation/heuristics/gap-limit parameter (stage 4).
    /// <see cref="GapLimit"/> is defined but UNREACHABLE this stage (no relGap/absGap parameter yet).
    /// </summary>
    public enum MIPStatus
    {
        /// <summary>Search tree fully explored: the best incumbent found (if any) is proven optimal.
        /// No incumbent at all means no integer-feasible point exists -- see <see cref="Infeasible"/>.
        /// </summary>
        Optimal = 0,

        /// <summary>Root LP relaxation infeasible, or the tree was exhausted with no integer-feasible
        /// point found. No usable <c>x</c>.</summary>
        Infeasible = 1,

        /// <summary>Root LP relaxation unbounded (only ever detected at the root). No usable <c>x</c>.
        /// </summary>
        Unbounded = 2,

        /// <summary>Reserved for a future gap parameter (stage 4). Still unreachable this stage.</summary>
        GapLimit = 3,

        /// <summary><c>maxNodes</c> exhausted before the tree was fully explored. Best incumbent found
        /// so far (if any) in <c>x</c>, a sound <see cref="MIPInfo.dualBound"/>.</summary>
        NodeLimit = 4,

        /// <summary><c>maxIter</c> cumulative LP-iteration budget exhausted. Same contract as
        /// <see cref="NodeLimit"/>.</summary>
        MaxIterations = 5,
    }

    /// <summary>
    /// Burst-safe enum-to-name helper for <see cref="MIPStatus"/>, used by
    /// <see cref="MIPInfo.ToFixedString"/>.
    /// </summary>
    public static class MIPStatusExtensions
    {
        public static FixedString32Bytes Name(this MIPStatus s)
        {
            switch (s)
            {
                case MIPStatus.Optimal: return "Optimal";
                case MIPStatus.Infeasible: return "Infeasible";
                case MIPStatus.Unbounded: return "Unbounded";
                case MIPStatus.GapLimit: return "GapLimit";
                case MIPStatus.NodeLimit: return "NodeLimit";
                case MIPStatus.MaxIterations: return "MaxIterations";
                default: return "Unknown";
            }
        }
    }

    /// <summary>
    /// Result of a mixed-integer program solve (<see cref="MIP.solve"/>). Implicit <c>bool</c>
    /// conversion == <see cref="Solved"/>, matching <see cref="LPInfo"/>/<see cref="QPInfo"/>.
    ///
    /// On <see cref="MIPStatus.Optimal"/>: proven-optimal incumbent. On
    /// <see cref="MIPStatus.NodeLimit"/>/<see cref="MIPStatus.MaxIterations"/>: best incumbent found so
    /// far, or <see cref="objective"/> == <c>double.PositiveInfinity</c> if none. On
    /// <see cref="MIPStatus.Infeasible"/>/<see cref="MIPStatus.Unbounded"/>: every numeric field except
    /// <see cref="nodes"/>/<see cref="lpIterations"/> is <c>double.NaN</c>.
    /// </summary>
    public struct MIPInfo
    {
        /// <summary>Objective value <c>cᵀx</c> at the returned (incumbent) <c>x</c>.</summary>
        public double objective;

        /// <summary>Proven lower bound on the true MIP optimum. Equals <see cref="objective"/> on
        /// <see cref="MIPStatus.Optimal"/>. On an early stop: <c>min</c> over every still-open node's
        /// own parent-LP bound -- the currently active plunge frontier plus every node still sitting
        /// in the best-bound queue. <c>double.NaN</c> on <see cref="MIPStatus.Infeasible"/>/
        /// <see cref="MIPStatus.Unbounded"/>.</summary>
        public double dualBound;

        /// <summary><c>(objective - dualBound) / max(1, |objective|)</c>. 0 on
        /// <see cref="MIPStatus.Optimal"/>. <c>double.PositiveInfinity</c> with no incumbent yet.
        /// <c>double.NaN</c> on <see cref="MIPStatus.Infeasible"/>/<see cref="MIPStatus.Unbounded"/>.
        /// </summary>
        public double gap;

        /// <summary>Total B&amp;B nodes solved, including the root.</summary>
        public int nodes;

        /// <summary>Cumulative simplex pivots across every node LP solved so far.</summary>
        public int lpIterations;

        /// <summary>Why the solve stopped -- see <see cref="MIPStatus"/>.</summary>
        public MIPStatus status;

        /// <summary>True iff a proven-optimal incumbent was found.</summary>
        public bool Solved => status == MIPStatus.Optimal;

        /// <summary>Implicit success test: <c>if (MIP.solve(...))</c> reads as "proven optimal".
        /// </summary>
        public static implicit operator bool(MIPInfo info) => info.status == MIPStatus.Optimal;

        /// <summary>Burst-safe compact summary, e.g. <c>MIPInfo(Optimal, nodes=7, obj=1.42E+01)</c>.
        /// </summary>
        public FixedString128Bytes ToFixedString()
        {
            FixedString128Bytes str = "MIPInfo(";
            str.Append(status.Name());
            FixedString128Bytes tail = $", nodes={nodes}, obj={objective:G4})";
            str.Append(tail);
            return str;
        }

        /// <summary>Managed wrapper -- do not call from inside a [BurstCompile] job.</summary>
        public override string ToString() => ToFixedString().ToString();
    }

    // Type-agnostic tuning constants for the search (MIP.fProxy.cs); live here (singularFile) to avoid
    // a duplicate-member collision across the float/double generated fragments.
    public static partial class MIP
    {
        // LP-bound pruning tolerance: prune a node whose LP bound is within this of the incumbent.
        internal const double ABS_GAP = 1e-6;

        // Integrality tolerance: |x_j - round(x_j)| <= this * max(1, |x_j|). Fixed for both dtypes.
        internal const double INTEGRALITY_TOL = 1e-6;

        // Pseudocost reliability threshold: observations required, per direction, before a variable's
        // pseudocost is trusted over strong branching (HighsPseudocost::isReliable's minreliable).
        internal const int RELIABILITY = 8;

        // Score-clamp floor for the product-rule branching score, so a zero estimate on one side does
        // not zero out the whole product (HighsPseudocost::getScore uses the same floor).
        internal const double PSEUDOCOST_EPS = 1e-6;

        // Per-call LP-iteration cap for one strong-branching trial solve.
        internal const int STRONG_BRANCH_ITER_CAP = 100;

        // Total strong-branch trial-solve budget = this many calls per integer variable -- enough for
        // every variable to reach RELIABILITY observations in both directions (2*RELIABILITY calls),
        // plus slack for trials that return without a usable observation (non-optimal child).
        internal const int STRONG_BRANCH_CALLS_PER_INT_VAR = 4 * RELIABILITY;
    }
}
