using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    /// <summary>
    /// Multigrid solvers. <see cref="solve(in fProxyAMG, in fProxyN, ref fProxyN, int, fProxy)"/> runs
    /// symmetric V-cycles of an <see cref="fProxyAMG"/> hierarchy to a relative residual tolerance;
    /// the same hierarchy is also usable as a preconditioner via <see cref="fProxyAMGPreconditioner"/>.
    /// </summary>
    public static partial class MG
    {
        /// <summary>
        /// Solves A x = b by V-cycles of the AMG hierarchy. x is a warm-startable initial guess,
        /// overwritten with the solution; tol is relative (converged when ‖b − Ax‖ ≤ tol·‖b‖);
        /// maxIter counts cycles. Returns the shared <see cref="SolveInfo"/>. Every cycle tests a
        /// freshly computed true residual, so no verify-at-exit pass is needed.
        /// </summary>
        public static SolveInfo solve(in fProxyAMG amg, in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
            => amg.Solve(in b, ref x, maxIter, tol);

        /// <summary>V-cycle solve with default maxIter (50 cycles) and tol (Consts.fProxySqrtEps).</summary>
        public static SolveInfo solve(in fProxyAMG amg, in fProxyN b, ref fProxyN x)
            => amg.Solve(in b, ref x, 50, Consts.fProxySqrtEps);
    }
}
