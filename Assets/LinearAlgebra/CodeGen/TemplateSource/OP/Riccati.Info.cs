//singularFile//
using Unity.Collections;

namespace LinearAlgebra
{
    /// <summary>
    /// Terminal state of a <see cref="Riccati.dare"/> solve (also carried by every LQR/Kalman result
    /// that runs it). Type-agnostic (no fProxy) on purpose -- lives in a non-templated file so codegen
    /// does not emit a duplicate definition into both the float and double partials (CS0102), exactly
    /// like <see cref="LPStatus"/>/<see cref="MIPStatus"/>.
    /// </summary>
    public enum RiccatiStatus
    {
        /// <summary>The doubling recursion reached its relative-change tolerance; <c>S</c> is the
        /// stabilizing DARE solution.</summary>
        Converged = 0,

        /// <summary>The iteration budget was exhausted before the tolerance was reached. The last
        /// iterate is still returned (a usable, if not fully converged, solution).</summary>
        MaxIterations = 1,

        /// <summary>A data-scaled blowup was detected (or an inner (I+GH)/(I+HG)/(R+BᵀSB) solve broke
        /// down) -- the system is not stabilizable/detectable, or the input itself is degenerate. Fails
        /// fast rather than hanging or returning garbage; the output is the last KNOWN-GOOD iterate (at
        /// worst H0 = Q), never NaN.</summary>
        Diverged = 2,
    }

    /// <summary>
    /// Burst-safe enum-to-name helper for <see cref="RiccatiStatus"/>, used by <see cref="RiccatiInfo.ToFixedString"/>.
    /// </summary>
    public static class RiccatiStatusExtensions
    {
        public static FixedString32Bytes Name(this RiccatiStatus s)
        {
            switch (s)
            {
                case RiccatiStatus.Converged: return "Converged";
                case RiccatiStatus.MaxIterations: return "MaxIterations";
                case RiccatiStatus.Diverged: return "Diverged";
                default: return "Unknown";
            }
        }
    }

    /// <summary>
    /// Result of a discrete algebraic Riccati equation solve (<see cref="Riccati.dare"/>, and every
    /// LQR/Kalman entry point built on it). Implicit <c>bool</c> conversion == <see cref="Solved"/>,
    /// matching <see cref="LPInfo"/>/<see cref="MIPInfo"/>. Every field is filled from numbers the
    /// solve already computes (house diag-struct rule) -- no extra pass.
    /// </summary>
    public struct RiccatiInfo
    {
        /// <summary>Doubling/recursion steps actually performed. Equals the budget on a
        /// <see cref="RiccatiStatus.MaxIterations"/> return.</summary>
        public int iterations;

        /// <summary>Relative change ‖S_new − S_old‖_F / max(1, ‖S_new‖_F) at the last completed step.
        /// <c>double.PositiveInfinity</c> on <see cref="RiccatiStatus.Diverged"/> (no meaningful ratio).</summary>
        public double residual;

        /// <summary>Why the solve stopped -- see <see cref="RiccatiStatus"/>.</summary>
        public RiccatiStatus status;

        /// <summary>True iff the input-space decomposition (R's own factorization while building
        /// G0 = BR⁻¹Bᵀ, or the R + BᵀSB solve) reported numerical rank below full -- the solution is
        /// non-unique (e.g. a redundant actuator / semidefinite R, or a degenerate measurement in the
        /// filter dual). The solve still completes via CHOP's minimum-norm rank-deficient branch; this
        /// flag makes that non-uniqueness visible rather than hiding it.</summary>
        public bool rankDeficient;

        /// <summary>True iff a stabilizing solution was reached (<c>status == RiccatiStatus.Converged</c>).</summary>
        public bool Solved => status == RiccatiStatus.Converged;

        /// <summary>Implicit success test, so <c>if (Riccati.dare(...))</c> reads as "did it converge".</summary>
        public static implicit operator bool(RiccatiInfo info) => info.status == RiccatiStatus.Converged;

        /// <summary>Burst-safe compact summary, e.g. <c>RiccatiInfo(Converged, iters=14, residual=1.23E-09,
        /// rankDeficient=False)</c>. Never allocates managed memory.</summary>
        public FixedString128Bytes ToFixedString()
        {
            FixedString128Bytes str = "RiccatiInfo(";
            str.Append(status.Name());
            FixedString128Bytes tail = $", iters={iterations}, residual={residual:G3}, rankDeficient={rankDeficient})";
            str.Append(tail);
            return str;
        }

        /// <summary>Managed wrapper -- do not call from inside a [BurstCompile] job.</summary>
        public override string ToString() => ToFixedString().ToString();
    }

    // Type-agnostic tuning constants for Riccati.dare (Riccati.fProxy.cs); live here (singularFile) to
    // avoid a duplicate-member collision across the float/double generated fragments, same reasoning as
    // LP.Info.cs's REFACTOR_INTERVAL / MIP.Info.cs's ABS_GAP etc.
    public static partial class Riccati
    {
        // SDA (cold infinite-horizon) doubling-step cap -- quadratic convergence means legitimate
        // stabilizable/detectable instances reach machine-precision-class residuals in ~10-25 steps;
        // 50 is a generous margin, not a target.
        internal const int SDA_MAX_ITER = 50;

        // Data-scaled divergence threshold multiplier: ‖H_k‖_F is judged diverged once it exceeds this
        // factor times (1 + ‖Q‖_F + ‖R‖_F). An unstabilizable/undetectable system blows up at the
        // doubling algorithm's own quadratic rate, so this trips within a handful of steps -- fail fast,
        // not a slow leak.
        internal const double BLOWUP_FACTOR = 1e12;
    }
}
