using System;

namespace LinearAlgebra
{
    public static partial class Eigen
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an n x n symmetric eigenvalue problem
        /// (three length-n vectors) — the layout produced by Arena.fProxyEigenSymWorkspace(n).
        /// </summary>
        static void RequireEigenSymWorkspace(in fProxyEigenSymWorkspace ws, int n, string who = "eigenvaluesSymmetric")
        {
            if (ws.eVec.N != n || ws.vVec.N != n || ws.pVec.N != n)
                throw new ArgumentException(who + ": workspace must be sized for n (use Arena.fProxyEigenSymWorkspace(n))");
        }
    }

    /// <summary>
    /// Reusable scratch for Eigen.eigenvaluesSymmetric (Householder tridiagonalization + implicit-shift
    /// QL). The op needs three length-n vectors (the off-diagonal e, the Householder vector v, and the
    /// rank-2-update vector p). Allocate ONCE via Arena.fProxyEigenSymWorkspace(n) and reuse it across
    /// same-size calls so repeated symmetric eigenvalue solves are zero-alloc.
    /// </summary>
    public struct fProxyEigenSymWorkspace
    {
        public fProxyN eVec;
        public fProxyN vVec;
        public fProxyN pVec;
    }

    public partial struct Arena
    {
        /// <summary>
        /// Allocates a symmetric-eigenvalue workspace for an n x n matrix: three length-n vectors.
        /// The buffers are persistent in this arena (disposed with it), so create the workspace once
        /// outside a hot loop and pass it to the ref-workspace overload of eigenvaluesSymmetric.
        /// </summary>
        public fProxyEigenSymWorkspace fProxyEigenSymWorkspace(int n)
        {
            return new fProxyEigenSymWorkspace
            {
                eVec = fProxyVec(n),
                vVec = fProxyVec(n),
                pVec = fProxyVec(n)
            };
        }
    }
}
