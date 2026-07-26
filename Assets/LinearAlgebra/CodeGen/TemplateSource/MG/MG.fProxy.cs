using Unity.Collections;
using Unity.Mathematics;
using BULA.Sparse;

namespace BULA
{
    /// <summary>
    /// Multigrid solvers. <see cref="solve(in fProxyAMG, in fProxyN, ref fProxyN, int, fProxy)"/> runs
    /// symmetric V-cycles of an <see cref="fProxyAMG"/> hierarchy to a relative residual tolerance;
    /// the same hierarchy is also usable as a preconditioner via <see cref="fProxyAMGPreconditioner"/>.
    /// The <c>in fProxyBSR</c> overloads are the one-shot route: they build a hierarchy from
    /// Allocator.Temp, solve with it, and release it before returning.
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

        /// <summary>
        /// Solves A x = b by cycles of an AMG hierarchy built from A with <paramref name="opts"/> out
        /// of Allocator.Temp and disposed before returning — for callers that solve once; build an
        /// <see cref="fProxyAMG"/> yourself to amortize the setup over several solves or right-hand
        /// sides. A must be a square SPD BSR. The build's own result lands in <paramref name="setup"/>
        /// (levels, coarsest size, status), as with the <see cref="fProxyAMG"/> constructors; pass
        /// <c>out _</c> to ignore it. x is a warm-startable initial guess, overwritten with the
        /// solution; tol is relative (converged when ‖b − Ax‖ ≤ tol·‖b‖); maxIter counts cycles.
        ///
        /// A hierarchy that is not usable (<c>setup.Solved</c> false — the coarsest level is not SPD)
        /// runs no cycle: x is left UNCHANGED and the return is status Breakdown, iterations 0,
        /// rnorm NaN. Throws if A is not square, opts is out of range, or b/x are not A.M_Rows long.
        /// </summary>
        public static SolveInfo solve(in fProxyBSR A, in AMGOptions opts, out AMGSetupInfo setup,
                                      in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
        {
            var amg = new fProxyAMG(in A, in opts, out setup, Allocator.Temp);
            try
            {
                if (!setup.Solved)
                    return new SolveInfo { rnorm = double.NaN, iterations = 0, status = IterativeSolveStatus.Breakdown };
                return amg.Solve(in b, ref x, maxIter, tol);
            }
            finally
            {
                amg.Dispose();
            }
        }

        /// <summary>One-shot AMG solve over a BSR with <see cref="AMGOptions.Default"/> (symmetric
        /// V-cycle). See the explicit-options overload for the build/dispose and Breakdown contract.</summary>
        public static SolveInfo solve(in fProxyBSR A, out AMGSetupInfo setup,
                                      in fProxyN b, ref fProxyN x, int maxIter, fProxy tol)
            => solve(in A, AMGOptions.Default, out setup, in b, ref x, maxIter, tol);

        /// <summary>One-shot AMG solve over a BSR with default maxIter (50 cycles) and tol
        /// (Consts.fProxySqrtEps).</summary>
        public static SolveInfo solve(in fProxyBSR A, in AMGOptions opts, out AMGSetupInfo setup,
                                      in fProxyN b, ref fProxyN x)
            => solve(in A, in opts, out setup, in b, ref x, 50, Consts.fProxySqrtEps);

        /// <summary>One-shot AMG solve over a BSR with <see cref="AMGOptions.Default"/>, default
        /// maxIter (50 cycles) and tol (Consts.fProxySqrtEps).</summary>
        public static SolveInfo solve(in fProxyBSR A, out AMGSetupInfo setup,
                                      in fProxyN b, ref fProxyN x)
            => solve(in A, AMGOptions.Default, out setup, in b, ref x, 50, Consts.fProxySqrtEps);
    }
}
