using System;

namespace LinearAlgebra
{
    public static partial class Eigen
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an n-dimensional operator run for
        /// <paramref name="steps"/> Lanczos iterations — the layout produced by
        /// <c>Arena.floatLanczosCache(n, steps)</c>. Also validates the nested symmetric-eigenvalue
        /// workspace (sized to <paramref name="steps"/>, since the tridiagonal T is steps x steps).
        /// </summary>
        static void RequireLanczosWorkspace(in floatLanczosCache ws, int n, int steps)
        {
            if (ws.V.M_Rows != steps || ws.V.N_Cols != n)
                throw new ArgumentException("Eigen.lanczos: workspace V must be steps x n (use Arena.floatLanczosCache(n, steps))");

            if (ws.vCur.N != n || ws.w.N != n)
                throw new ArgumentException("Eigen.lanczos: workspace vCur/w must have length n (use Arena.floatLanczosCache(n, steps))");

            if (ws.alpha.N != steps || ws.beta.N != steps)
                throw new ArgumentException("Eigen.lanczos: workspace alpha/beta must have length steps (use Arena.floatLanczosCache(n, steps))");

            if (!ws.T.IsSquare || ws.T.M_Rows != steps)
                throw new ArgumentException("Eigen.lanczos: workspace T must be steps x steps (use Arena.floatLanczosCache(n, steps))");

            RequireEigenSymWorkspace(in ws.symWs, steps);
        }
    }

    /// <summary>
    /// Reusable scratch for <see cref="Eigen.lanczos{TOp}"/> (Lanczos tridiagonalization of a
    /// symmetric operator, with full reorthogonalization, followed by
    /// <see cref="Eigen.eigenvaluesSymmetric(ref floatMxN, ref floatN, ref floatEigenSymCache)"/>
    /// on the resulting small tridiagonal). Sized for an n-dimensional operator run for
    /// <c>steps</c> Lanczos iterations. Allocate ONCE via <c>Arena.floatLanczosCache(n, steps)</c>
    /// and reuse it across same-shape calls so repeated Lanczos runs are zero-alloc.
    /// </summary>
    public struct floatLanczosCache
    {
        /// <summary>steps x n Krylov basis: row j (0-indexed) holds the unit vector v_(j+1).</summary>
        public floatMxN V;

        /// <summary>Length n. Scratch copy of the current Krylov vector — <see cref="IfloatLinearOperator.Apply"/>
        /// requires distinct input/output buffers, and a row of <see cref="V"/> is not independently
        /// addressable as an <c>floatN</c>, so it is copied here before each Apply.</summary>
        public floatN vCur;

        /// <summary>Length n. Work vector: holds A*v_j, then the orthogonalized (not yet
        /// normalized) next Krylov vector.</summary>
        public floatN w;

        /// <summary>Length steps. Tridiagonal diagonal entries (alpha[i] = alpha_(i+1) in the
        /// algorithm's 1-indexed math notation).</summary>
        public floatN alpha;

        /// <summary>Length steps. Tridiagonal off-diagonal entries (beta[i] couples rows i and
        /// i+1 of T). The last entry written by a given call may be left stale/unused — see
        /// <see cref="Eigen.lanczos{TOp}"/>'s doc comment on early breakdown.</summary>
        public floatN beta;

        /// <summary>steps x steps. The symmetric tridiagonal assembled from alpha/beta (padded
        /// with a decoupled junk block when the Lanczos process breaks down before `steps`
        /// iterations complete), then destroyed in place by eigenvaluesSymmetric.</summary>
        public floatMxN T;

        /// <summary>Nested workspace for eigenvaluesSymmetric's Householder+QL reduction of T,
        /// sized to `steps` (T is always steps x steps regardless of early breakdown).</summary>
        public floatEigenSymCache symWs;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates a Lanczos workspace for an n-dimensional symmetric operator run for `steps`
        /// iterations. See <see cref="floatLanczosCache"/> for reuse guidance.
        /// </summary>
        public static floatLanczosCache floatLanczosCache(this ref Arena arena, int n, int steps)
        {
            return new floatLanczosCache
            {
                V = arena.floatMat(steps, n),
                vCur = arena.floatVec(n),
                w = arena.floatVec(n),
                alpha = arena.floatVec(steps),
                beta = arena.floatVec(steps),
                T = arena.floatMat(steps, steps),
                symWs = arena.floatEigenSymCache(steps)
            };
        }
    }
}
