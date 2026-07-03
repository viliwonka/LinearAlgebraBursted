using System;

namespace LinearAlgebra
{
    public static partial class Eigen
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an n x n symmetric eigenvalue problem
        /// (three length-n vectors) — the layout produced by Arena.doubleEigenSymCache(n).
        /// </summary>
        static void RequireEigenSymWorkspace(in doubleEigenSymCache ws, int n)
        {
            if (ws.eVec.N != n || ws.vVec.N != n || ws.pVec.N != n)
                throw new ArgumentException("Eigen.eigenvaluesSymmetric: workspace must be sized for n (use Arena.doubleEigenSymCache(n))");
        }
    }

    /// <summary>
    /// Reusable scratch for Eigen.eigenvaluesSymmetric (Householder tridiagonalization + implicit-shift
    /// QL). The op needs three length-n vectors (the off-diagonal e, the Householder vector v, and the
    /// rank-2-update vector p). Allocate ONCE via Arena.doubleEigenSymCache(n) and reuse it across
    /// same-size calls so repeated symmetric eigenvalue solves are zero-alloc.
    /// </summary>
    public struct doubleEigenSymCache
    {
        public doubleN eVec;
        public doubleN vVec;
        public doubleN pVec;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates a symmetric-eigenvalue workspace for an n x n matrix. See
        /// <see cref="doubleEigenSymCache"/> for reuse guidance.
        /// </summary>
        public static doubleEigenSymCache doubleEigenSymCache(this ref Arena arena, int n)
        {
            return new doubleEigenSymCache
            {
                eVec = arena.doubleVec(n),
                vVec = arena.doubleVec(n),
                pVec = arena.doubleVec(n)
            };
        }
    }
}
