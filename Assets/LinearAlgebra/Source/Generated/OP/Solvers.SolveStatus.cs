//singularFile//
namespace LinearAlgebra
{
    /// <summary>
    /// Outcome of a solver call, carried by the solve-result structs (<c>fProxySolveInfo</c> /
    /// <c>fProxyLstsqInfo</c>). Their <c>Solved</c> convenience property (and implicit bool
    /// conversion) is exactly <c>status == SolveStatus.Converged</c>, so <c>if (solver(...))</c>
    /// keeps reading as "did it succeed", while the enum preserves WHY a solve stopped for callers
    /// that want it.
    ///
    /// Type-agnostic (no fProxy) on purpose: it lives in a non-templated file so codegen does not
    /// emit a duplicate definition into both the float and double partials (CS0102).
    /// </summary>
    public enum SolveStatus
    {
        /// <summary>Reached the requested tolerance.</summary>
        Converged = 0,

        /// <summary>Ran the full iteration budget without reaching tolerance.</summary>
        MaxIterations = 1,

        /// <summary>Stopped early on a numerical breakdown (e.g. non-positive curvature, a zero
        /// rotation radius, or a bidiagonalization/Lanczos breakdown) -- the recurrence could make
        /// no further progress. On a breakdown the solution is undefined.</summary>
        Breakdown = 2,
    }
}
