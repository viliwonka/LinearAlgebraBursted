//singularFile//
using BULA;

namespace BULA.Sparse
{
    /// <summary>Multigrid cycle shape. V is a fixed linear operator (valid for cg); K adds a 2-step
    /// Flexible-CG acceleration at every level (recovers grid-independence for unsmoothed aggregation)
    /// but is a VARIABLE operator — drive it with <see cref="BULA.Krylov"/>.fcg, not cg.</summary>
    public enum MGCycle { V = 0, K = 1 }

    /// <summary>
    /// Setup knobs for the AMG hierarchy (floatAMG / doubleAMG). Fields are plain int/double (not
    /// proxy-typed), so this file is not float/double-duplicated by codegen — same role as
    /// <see cref="SchwarzOptions"/>.
    /// </summary>
    public struct AMGOptions
    {
        /// <summary>Cycle shape: V (default, cg-safe) or K (Krylov-accelerated, fcg-only).</summary>
        public MGCycle cycle;
        /// <summary>Strength-of-connection threshold (0 keeps all stored off-diagonals).</summary>
        public double theta;
        /// <summary>Pre-smoothing sweeps per level per cycle (&gt;= 0).</summary>
        public int pre;
        /// <summary>Post-smoothing sweeps per level per cycle (&gt;= 0). For cg validity keep == pre.</summary>
        public int post;
        /// <summary>Stop coarsening once a level has &lt;= coarseMax scalar unknowns (direct dense solve there).</summary>
        public int coarseMax;
        /// <summary>Hard cap on level count.</summary>
        public int maxLevels;

        /// <summary>theta=0, pre=post=1, coarseMax=48, maxLevels=20.</summary>
        public static AMGOptions Default => new AMGOptions
        {
            cycle = MGCycle.V,
            theta = 0,
            pre = 1,
            post = 1,
            coarseMax = 48,
            maxLevels = 20,
        };
    }

    /// <summary>Result of an AMG hierarchy build (fields only from already-computed numbers).</summary>
    public struct AMGSetupInfo
    {
        public int levels;
        public int coarseRows;
        public DirectSolveStatus status;
        public bool Solved => status == DirectSolveStatus.Success;
    }
}
