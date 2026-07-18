//singularFile//
using Unity.Collections;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Result of building a sparse preconditioner (BlockJacobi / ILU0 / IC0 / FSAI / SPAI, any
    /// precision), returned via the builders' <c>out</c> parameter. Implicitly converts to
    /// <c>bool</c> (== <see cref="Solved"/>) like every other *Info struct in this library.
    ///
    /// Reuses <see cref="DirectSolveStatus"/> (a preconditioner build IS a block/incomplete
    /// factorization): Success; Singular (a BlockJacobi diagonal block could not be inverted, ILU0
    /// broke down at every diagonal shift, or an SPAI row's local least-squares solve broke down
    /// at every Tikhonov shift); NotPositiveDefinite (IC0 broke down at every diagonal shift, or an
    /// FSAI row's local solve broke down at every diagonal shift). On any non-Success status the
    /// preconditioner is unusable — do not Apply.
    /// </summary>
    public struct PreconditionerInfo
    {
        /// <summary>Why the build stopped -- see <see cref="DirectSolveStatus"/>.</summary>
        public DirectSolveStatus status;

        /// <summary>Diagonal (FSAI) or Tikhonov (SPAI) shift that made the factorization/local
        /// solve succeed; 0 for a clean first pass. Always 0 for BlockJacobi (it has no shift
        /// retry). For FSAI/SPAI (independent per-row solves) this is the WORST shift across every
        /// row, not a single global value. Widened to <c>double</c> regardless of the build's
        /// precision, matching the *Info convention.</summary>
        public double shift;

        /// <summary>Factorization attempts consumed (1 = clean first pass; ILU0/IC0 escalate the
        /// diagonal shift for up to 6 attempts; BlockJacobi is always 1; FSAI/SPAI escalate per row
        /// for up to 6 attempts and report the WORST attempts count across every row).</summary>
        public int attempts;

        /// <summary>True iff the build completed (<c>status == DirectSolveStatus.Success</c>).</summary>
        public bool Solved => status == DirectSolveStatus.Success;

        /// <summary>Implicit success test, so <c>if (info)</c> reads the same way every other
        /// *Info struct in this library does.</summary>
        public static implicit operator bool(PreconditionerInfo i) => i.status == DirectSolveStatus.Success;

        /// <summary>Burst-safe compact summary, e.g. <c>PreconditionerInfo(Success, shift=0,
        /// attempts=1)</c>. Never allocates managed memory.</summary>
        public FixedString128Bytes ToFixedString()
        {
            FixedString128Bytes str = "PreconditionerInfo(";
            str.Append(status.Name());
            FixedString128Bytes tail = $", shift={shift:G3}, attempts={attempts})";
            str.Append(tail);
            return str;
        }

        /// <summary>Managed wrapper -- do not call from inside a [BurstCompile] job.</summary>
        public override string ToString() => ToFixedString().ToString();
    }
}
