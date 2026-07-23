using System;
using Unity.Collections;

namespace LinearAlgebra
{
    public static partial class Eigen
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an n x n symmetric eigenvalue problem
        /// (three length-n vectors) — the layout produced by the fProxyEigenSymCache(n, allocator) constructor.
        /// </summary>
        static void RequireEigenSymWorkspace(in fProxyEigenSymCache ws, int n)
        {
            if (ws.eVec.N != n || ws.vVec.N != n || ws.pVec.N != n)
                throw new ArgumentException("Eigen.valuesSymmetricInPlace: workspace must be sized for n (use new fProxyEigenSymCache(n, allocator))");
        }
    }

    /// <summary>
    /// Reusable scratch for Eigen.valuesSymmetricInPlace (Householder tridiagonalization + implicit-shift
    /// QL). The op needs three length-n vectors (the off-diagonal e, the Householder vector v, and the
    /// rank-2-update vector p). Allocate ONCE via the Allocator ctor and reuse it across
    /// same-size calls so repeated symmetric eigenvalue solves are zero-alloc.
    /// </summary>
    public struct fProxyEigenSymCache : IDisposable
    {
        public fProxyN eVec;
        public fProxyN vVec;
        public fProxyN pVec;

        /// <summary>Allocates a symmetric-eigenvalue workspace for an n x n matrix. Pair with <see cref="Dispose"/>.</summary>
        public fProxyEigenSymCache(int n, Allocator allocator)
        {
            eVec = new fProxyN(n, allocator);
            vVec = new fProxyN(n, allocator);
            pVec = new fProxyN(n, allocator);
        }

        /// <summary>Disposes the workspace.</summary>
        public void Dispose()
        {
            eVec.Dispose();
            vVec.Dispose();
            pVec.Dispose();
        }
    }
}
