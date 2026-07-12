//singularFile//
using Unity.Collections;

namespace LinearAlgebra
{
    /// <summary>
    /// Terminal state of a <see cref="Control.lqr(in fProxyMxN, in fProxyMxN, in fProxyMxN, in fProxyMxN, ref fProxyMxN, int)"/> /
    /// <c>lqrSchedule</c> call, carried by <see cref="LQRInfo"/>. Type-agnostic (no fProxy) on
    /// purpose -- lives in a non-templated file so codegen does not emit a duplicate definition into
    /// both the float and double partials (CS0102), exactly like <see cref="LPStatus"/>/<see cref="MIPStatus"/>.
    /// </summary>
    public enum LQRStatus
    {
        /// <summary>The Riccati recursion (SDA doubling, plain warm iteration, or a schedule step)
        /// reached its relative-change tolerance; <c>S</c>/<c>K</c> describe the stabilizing
        /// solution.</summary>
        Converged = 0,

        /// <summary>The iteration budget was exhausted before the tolerance was reached. The last
        /// iterate is still returned (a usable, if not fully converged, gain).</summary>
        MaxIterations = 1,

        /// <summary>A data-scaled blowup was detected (or an inner (I+GH)/(I+HG)/(R+BᵀSB) solve
        /// broke down) -- the system is not stabilizable/detectable, or the input itself is
        /// degenerate. Fails fast rather than hanging or returning garbage; outputs are the last
        /// KNOWN-GOOD iterate (not the exploded one), never NaN.</summary>
        Diverged = 2,
    }

    /// <summary>
    /// Burst-safe enum-to-name helper for <see cref="LQRStatus"/>, used by <see cref="LQRInfo.ToFixedString"/>.
    /// </summary>
    public static class LQRStatusExtensions
    {
        public static FixedString32Bytes Name(this LQRStatus s)
        {
            switch (s)
            {
                case LQRStatus.Converged: return "Converged";
                case LQRStatus.MaxIterations: return "MaxIterations";
                case LQRStatus.Diverged: return "Diverged";
                default: return "Unknown";
            }
        }
    }

    /// <summary>
    /// Result of a <see cref="Control"/> LQR solve (cold <c>lqr</c>, warm <c>lqr(..., ref state)</c>,
    /// or <c>lqrSchedule</c>). Implicit <c>bool</c> conversion == <see cref="Solved"/>, matching
    /// <see cref="LPInfo"/>/<see cref="MIPInfo"/>. Every field is filled from numbers the solve
    /// already computes (house diag-struct rule) -- no extra pass.
    /// </summary>
    public struct LQRInfo
    {
        /// <summary>Riccati steps actually performed (SDA doubling steps, warm-recursion steps, or
        /// schedule backward steps). Equals the budget on a <see cref="LQRStatus.MaxIterations"/>
        /// return.</summary>
        public int iterations;

        /// <summary>Relative change ‖S_new − S_old‖_F / max(1, ‖S_new‖_F) at the last completed step
        /// (cold/warm), or at the schedule's last backward step (k=0) for <c>lqrSchedule</c>.
        /// <c>double.PositiveInfinity</c> on <see cref="LQRStatus.Diverged"/> (no meaningful ratio).</summary>
        public double residual;

        /// <summary>Why the solve stopped -- see <see cref="LQRStatus"/>.</summary>
        public LQRStatus status;

        /// <summary>True iff the (R + BᵀSB) solve (or, for the cold SDA path, R's own decomposition
        /// while building G0 = BR⁻¹Bᵀ) reported numerical rank below full -- the optimal control is
        /// non-unique (e.g. a redundant actuator / semidefinite R). The solve still completes via
        /// CHOP's minimum-norm rank-deficient branch; this flag makes that non-uniqueness visible
        /// rather than hiding it.</summary>
        public bool rankDeficientControl;

        /// <summary>True iff a stabilizing solution was reached (<c>status == LQRStatus.Converged</c>).</summary>
        public bool Solved => status == LQRStatus.Converged;

        /// <summary>Implicit success test, so <c>if (Control.lqr(...))</c> reads as "did it converge".</summary>
        public static implicit operator bool(LQRInfo info) => info.status == LQRStatus.Converged;

        /// <summary>Burst-safe compact summary, e.g. <c>LQRInfo(Converged, iters=14, residual=1.23E-09,
        /// rankDeficient=False)</c>. Never allocates managed memory.</summary>
        public FixedString128Bytes ToFixedString()
        {
            FixedString128Bytes str = "LQRInfo(";
            str.Append(status.Name());
            FixedString128Bytes tail = $", iters={iterations}, residual={residual:G3}, rankDeficient={rankDeficientControl})";
            str.Append(tail);
            return str;
        }

        /// <summary>Managed wrapper -- do not call from inside a [BurstCompile] job.</summary>
        public override string ToString() => ToFixedString().ToString();
    }

    /// <summary>
    /// Result of <see cref="Control.lqg"/>: the two independent DARE solves it runs, one per gain.
    /// Type-agnostic (CS0102), same reasoning as <see cref="LQRInfo"/>. Deliberately a thin pair
    /// rather than a merged/averaged diagnostic -- the two Riccati solves are unrelated problems
    /// (control cost vs. process/measurement noise) that only happen to share A and the SDA engine.
    /// </summary>
    public struct LQGInfo
    {
        /// <summary>Terminal state of the LQR (control) DARE solve.</summary>
        public LQRInfo lqrInfo;

        /// <summary>Terminal state of the filter (Kalman) DARE solve.</summary>
        public LQRInfo kfInfo;

        /// <summary>True iff BOTH solves converged.</summary>
        public bool Solved => lqrInfo.Solved && kfInfo.Solved;

        /// <summary>Implicit success test, so <c>if (Control.lqg(...))</c> reads as "did both converge".</summary>
        public static implicit operator bool(LQGInfo info) => info.Solved;

        /// <summary>Burst-safe compact summary, e.g. <c>LQGInfo(lqr=Converged/iters=14, kf=Converged/iters=9)</c>.
        /// Never allocates managed memory.</summary>
        public FixedString128Bytes ToFixedString()
        {
            FixedString128Bytes str = "LQGInfo(lqr=";
            str.Append(lqrInfo.status.Name());
            FixedString128Bytes mid = $"/iters={lqrInfo.iterations}, kf=";
            str.Append(mid);
            str.Append(kfInfo.status.Name());
            FixedString128Bytes tail = $"/iters={kfInfo.iterations})";
            str.Append(tail);
            return str;
        }

        /// <summary>Managed wrapper -- do not call from inside a [BurstCompile] job.</summary>
        public override string ToString() => ToFixedString().ToString();
    }

    // Type-agnostic tuning constants for Control.lqr/.lqrSchedule (Control.fProxy.cs); live here
    // (singularFile) to avoid a duplicate-member collision across the float/double generated
    // fragments, same reasoning as LP.Info.cs's REFACTOR_INTERVAL / MIP.Info.cs's ABS_GAP etc.
    public static partial class Control
    {
        // SDA (cold infinite-horizon) doubling-step cap -- quadratic convergence means legitimate
        // stabilizable/detectable instances reach machine-precision-class residuals in ~10-25 steps
        // (spec estimate); 50 is a generous margin, not a target.
        internal const int SDA_MAX_ITER = 50;

        // Warm-start (plain Riccati recursion, linear-ish convergence from an already-close S) and
        // SDA-oracle iteration cap. A slightly re-linearized A/B (the per-frame warm-start use case)
        // converges in a handful of steps; 500 is a safety net for a materially-changed system that
        // still happens to be stabilizable, not a target.
        internal const int WARM_MAX_ITER = 500;

        // Data-scaled divergence threshold multiplier: ‖S‖_F (or SDA's ‖H_k‖_F) is judged diverged
        // once it exceeds this factor times (1 + ‖Q‖_F + ‖R‖_F). An unstabilizable/undetectable
        // system blows up at the doubling algorithm's own quadratic (or the plain recursion's linear)
        // rate, so this trips within a handful of steps -- fail fast, not a slow leak.
        internal const double BLOWUP_FACTOR = 1e12;
    }
}
