//singularFile//
namespace BULA.Sparse
{
    /// <summary>
    /// Shared build options for <see cref="fProxyFSAI"/> and <see cref="fProxySPAI"/> (both
    /// precisions -- fields are plain double/int, not proxy-typed, so this file is not
    /// float/double-duplicated by codegen; same role as <see cref="PreconditionerInfo"/>).
    /// </summary>
    public struct SaiOptions
    {
        /// <summary>Static pattern power: 1 = pattern(A) (the only implemented value). 2 =
        /// pattern(A^2) is a future extension -- constructors throw if set.</summary>
        public int patternPower;

        /// <summary>Block-norm drop tolerance for FSAI's off-diagonal pattern entries (see
        /// <see cref="fProxyFSAI"/>); 0 (default) keeps the full pattern. Not consulted by
        /// <see cref="fProxySPAI"/>, whose MVP pattern is always A's own row pattern.</summary>
        public double dropTol;

        public static SaiOptions Default => new SaiOptions { patternPower = 1, dropTol = 0 };
    }
}
