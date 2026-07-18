//singularFile//
using LinearAlgebra;

namespace LinearAlgebra.Sparse
{
    /// <summary>
    /// Setup knobs for <see cref="fProxyAMG"/>. Fields are plain int/double (not proxy-typed), so
    /// this file is not float/double-duplicated by codegen — same role as <see cref="SchwarzOptions"/>.
    /// </summary>
    public struct AMGOptions
    {
        /// <summary>Strength-of-connection threshold (0 keeps all stored off-diagonals).</summary>
        public double theta;
        /// <summary>Pre-smoothing sweeps per level per cycle (&gt;= 0).</summary>
        public int pre;
        /// <summary>Post-smoothing sweeps per level per cycle (&gt;= 0). For pcg validity keep == pre.</summary>
        public int post;
        /// <summary>Stop coarsening once a level has &lt;= coarseMax scalar unknowns (direct dense solve there).</summary>
        public int coarseMax;
        /// <summary>Hard cap on level count.</summary>
        public int maxLevels;

        /// <summary>theta=0, pre=post=1, coarseMax=48, maxLevels=20.</summary>
        public static AMGOptions Default => new AMGOptions
        {
            theta = 0,
            pre = 1,
            post = 1,
            coarseMax = 48,
            maxLevels = 20,
        };
    }

    /// <summary>Result of an <see cref="fProxyAMG"/> build (fields only from already-computed numbers).</summary>
    public struct AMGSetupInfo
    {
        public int levels;
        public int coarseRows;
        public DirectSolveStatus status;
        public bool Solved => status == DirectSolveStatus.Success;
    }
}
