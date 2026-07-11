using System;

namespace LinearAlgebra
{
    public static partial class Eigen
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an n x n symmetric eigenvalue problem
        /// (three length-n vectors) — the layout produced by Arena.floatEigenSymCache(n).
        /// </summary>
        static void RequireEigenSymWorkspace(in floatEigenSymCache ws, int n)
        {
            if (ws.eVec.N != n || ws.vVec.N != n || ws.pVec.N != n)
                throw new ArgumentException("Eigen.valuesSymmetricInPlace: workspace must be sized for n (use Arena.floatEigenSymCache(n))");
        }
    }

    /// <summary>
    /// Reusable scratch for Eigen.valuesSymmetricInPlace (Householder tridiagonalization + implicit-shift
    /// QL). The op needs three length-n vectors (the off-diagonal e, the Householder vector v, and the
    /// rank-2-update vector p). Allocate ONCE via Arena.floatEigenSymCache(n) and reuse it across
    /// same-size calls so repeated symmetric eigenvalue solves are zero-alloc.
    /// </summary>
    public struct floatEigenSymCache
    {
        public floatN eVec;
        public floatN vVec;
        public floatN pVec;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates a symmetric-eigenvalue workspace for an n x n matrix. See
        /// <see cref="floatEigenSymCache"/> for reuse guidance.
        /// </summary>
        public static floatEigenSymCache floatEigenSymCache(this ref Arena arena, int n)
        {
            return new floatEigenSymCache
            {
                eVec = arena.floatVec(n),
                vVec = arena.floatVec(n),
                pVec = arena.floatVec(n)
            };
        }
    }
}
