//singularFile//
using Unity.Collections;

namespace LinearAlgebra
{
    /// <summary>
    /// Result of a block (multi-RHS) iterative solve — the block counterpart of
    /// <see cref="SolveInfo"/>. One struct describes the whole s-column solve. An implicit
    /// <c>bool</c> conversion (== <see cref="Solved"/>) lets <c>if (Krylov.cg(A, B, ref X))</c>
    /// keep compiling.
    /// </summary>
    public struct BlockSolveInfo
    {
        /// <summary>Number of right-hand sides (columns of B / X).</summary>
        public int rhs;

        /// <summary>How many of the <see cref="rhs"/> columns reached the tolerance.</summary>
        public int converged;

        /// <summary>Block iterations actually performed (0 when every column converged before the
        /// first step; equals maxIter when it ran out).</summary>
        public int iterations;

        /// <summary>Worst per-column residual norm ‖B[:,j] - A X[:,j]‖ across all columns at the
        /// returned X.</summary>
        public double maxRnorm;

        /// <summary>Smallest active search-block width reached during the solve — the numerical
        /// row-rank of the (preconditioned) residual block at its most deflated. Equals
        /// <see cref="rhs"/> when nothing deflated; &lt; <see cref="rhs"/> means columns dropped
        /// (converged or linearly dependent).</summary>
        public int minActive;

        /// <summary>Why the solve stopped. <see cref="IterativeSolveStatus.Converged"/> iff ALL
        /// <see cref="rhs"/> columns converged; otherwise MaxIterations or Breakdown.</summary>
        public IterativeSolveStatus status;

        /// <summary>True iff every column reached its tolerance
        /// (<c>status == IterativeSolveStatus.Converged</c>).</summary>
        public bool Solved => status == IterativeSolveStatus.Converged;

        /// <summary>Implicit success test so <c>if (blockSolve(...))</c> keeps compiling.</summary>
        public static implicit operator bool(BlockSolveInfo info) => info.status == IterativeSolveStatus.Converged;

        /// <summary>Burst-safe compact summary. Never allocates managed memory.</summary>
        public FixedString128Bytes ToFixedString()
        {
            FixedString128Bytes str = "BlockSolveInfo(";
            str.Append(status.Name());
            FixedString128Bytes tail = $", rhs={rhs}, converged={converged}, iters={iterations}, minActive={minActive}, maxRnorm={maxRnorm:G3})";
            str.Append(tail);
            return str;
        }

        /// <summary>Managed wrapper -- do not call from inside a [BurstCompile] job.</summary>
        public override string ToString() => ToFixedString().ToString();
    }
}
