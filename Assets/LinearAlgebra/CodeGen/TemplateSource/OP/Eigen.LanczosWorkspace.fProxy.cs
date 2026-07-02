using System;

namespace LinearAlgebra
{
    public static partial class Eigen
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an n-dimensional operator run for
        /// <paramref name="steps"/> Lanczos iterations — the layout produced by
        /// <c>Arena.fProxyLanczos_WS(n, steps)</c>. Also validates the nested symmetric-eigenvalue
        /// workspace (sized to <paramref name="steps"/>, since the tridiagonal T is steps x steps).
        /// </summary>
        static void RequireLanczosWorkspace(in fProxyLanczos_WS ws, int n, int steps)
        {
            if (ws.V.M_Rows != steps || ws.V.N_Cols != n)
                throw new ArgumentException("Eigen.lanczos: workspace V must be steps x n (use Arena.fProxyLanczos_WS(n, steps))");

            if (ws.vCur.N != n || ws.w.N != n)
                throw new ArgumentException("Eigen.lanczos: workspace vCur/w must have length n (use Arena.fProxyLanczos_WS(n, steps))");

            if (ws.alpha.N != steps || ws.beta.N != steps)
                throw new ArgumentException("Eigen.lanczos: workspace alpha/beta must have length steps (use Arena.fProxyLanczos_WS(n, steps))");

            if (!ws.T.IsSquare || ws.T.M_Rows != steps)
                throw new ArgumentException("Eigen.lanczos: workspace T must be steps x steps (use Arena.fProxyLanczos_WS(n, steps))");

            RequireEigenSymWorkspace(in ws.symWs, steps);
        }
    }

    /// <summary>
    /// Reusable scratch for <see cref="Eigen.lanczos{TOp}"/> (Lanczos tridiagonalization of a
    /// symmetric operator, with full reorthogonalization, followed by
    /// <see cref="Eigen.eigenvaluesSymmetric(ref fProxyMxN, ref fProxyN, ref fProxyEigenSym_WS)"/>
    /// on the resulting small tridiagonal). Sized for an n-dimensional operator run for
    /// <c>steps</c> Lanczos iterations. Allocate ONCE via <c>Arena.fProxyLanczos_WS(n, steps)</c>
    /// and reuse it across same-shape calls so repeated Lanczos runs are zero-alloc.
    /// </summary>
    public struct fProxyLanczos_WS
    {
        /// <summary>steps x n Krylov basis: row j (0-indexed) holds the unit vector v_(j+1).</summary>
        public fProxyMxN V;

        /// <summary>Length n. Scratch copy of the current Krylov vector — <see cref="IfProxyLinearOperator.Apply"/>
        /// requires distinct input/output buffers, and a row of <see cref="V"/> is not independently
        /// addressable as an <c>fProxyN</c>, so it is copied here before each Apply.</summary>
        public fProxyN vCur;

        /// <summary>Length n. Work vector: holds A*v_j, then the orthogonalized (not yet
        /// normalized) next Krylov vector.</summary>
        public fProxyN w;

        /// <summary>Length steps. Tridiagonal diagonal entries (alpha[i] = alpha_(i+1) in the
        /// algorithm's 1-indexed math notation).</summary>
        public fProxyN alpha;

        /// <summary>Length steps. Tridiagonal off-diagonal entries (beta[i] couples rows i and
        /// i+1 of T). The last entry written by a given call may be left stale/unused — see
        /// <see cref="Eigen.lanczos{TOp}"/>'s doc comment on early breakdown.</summary>
        public fProxyN beta;

        /// <summary>steps x steps. The symmetric tridiagonal assembled from alpha/beta (padded
        /// with a decoupled junk block when the Lanczos process breaks down before `steps`
        /// iterations complete), then destroyed in place by eigenvaluesSymmetric.</summary>
        public fProxyMxN T;

        /// <summary>Nested workspace for eigenvaluesSymmetric's Householder+QL reduction of T,
        /// sized to `steps` (T is always steps x steps regardless of early breakdown).</summary>
        public fProxyEigenSym_WS symWs;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates a Lanczos workspace for an n-dimensional symmetric operator run for `steps`
        /// iterations. The buffers are persistent in this arena (disposed with it), so create the
        /// workspace once outside a hot loop and pass it to <see cref="Eigen.lanczos{TOp}"/>.
        /// </summary>
        public static fProxyLanczos_WS fProxyLanczos_WS(this ref Arena arena, int n, int steps)
        {
            return new fProxyLanczos_WS
            {
                V = arena.fProxyMat(steps, n),
                vCur = arena.fProxyVec(n),
                w = arena.fProxyVec(n),
                alpha = arena.fProxyVec(steps),
                beta = arena.fProxyVec(steps),
                T = arena.fProxyMat(steps, steps),
                symWs = arena.fProxyEigenSym_WS(steps)
            };
        }
    }
}
