using System;
using Unity.Collections;

namespace LinearAlgebra
{
    public static partial class SVD
    {
        /// <summary>Throws unless <paramref name="ws"/> matches the fProxySVDThinCache(m, n, allocator) constructor sizing (BidiagWs is validated separately, by Bidiag.decomp itself).</summary>
        static void RequireSvdThinWorkspace(in fProxySVDThinCache ws, int m, int n)
        {
            bool ok =
                ws.B.M_Rows == n && ws.B.N_Cols == n &&
                ws.dVec.N == n && ws.eVec.N == n &&
                ws.Ut.M_Rows == n && ws.Ut.N_Cols == m &&
                ws.Vt.M_Rows == n && ws.Vt.N_Cols == n;

            if (!ok)
                throw new ArgumentException("SVD: workspace must be sized for m x n (use new fProxySVDThinCache(m, n, allocator))");
        }
    }

    /// <summary>
    /// Reusable scratch for SVD.thin (Golub-Kahan bidiagonalization + implicit-shift bidiagonal QR).
    /// Allocate ONCE (sized for the matrix shape) via the Allocator ctor and reuse it across
    /// many same-shape calls to avoid the per-call Allocator.Temp allocations thin's allocating
    /// overload makes internally.
    ///
    /// BidiagWs is the nested workspace Bidiag.decomp needs (see fProxyBidiagCache); B (n x n) is
    /// the bidiagonal factor; dVec/eVec (length n) are the diagonal/superdiagonal the bidiagonal QR
    /// diagonalizes; Ut (n x m) / Vt (n x n) are the transposed accumulators the QR sweep rotates
    /// (unit-stride rows, same trick as Eigen.symmetricInPlace).
    /// </summary>
    public struct fProxySVDThinCache : IDisposable
    {
        public fProxyBidiagCache BidiagWs;
        public fProxyMxN B;
        public fProxyN dVec;
        public fProxyN eVec;
        public fProxyMxN Ut;
        public fProxyMxN Vt;

        /// <summary>Allocates a thin workspace for an m x n (m >= n) system. Pair with <see cref="Dispose"/>.</summary>
        public fProxySVDThinCache(int m, int n, Allocator allocator)
        {
            BidiagWs = new fProxyBidiagCache(m, n, allocator);
            B = new fProxyMxN(n, n, allocator);
            dVec = new fProxyN(n, allocator);
            eVec = new fProxyN(n, allocator);
            Ut = new fProxyMxN(n, m, allocator);
            Vt = new fProxyMxN(n, n, allocator);
        }

        /// <summary>Dispose when done.</summary>
        public void Dispose()
        {
            BidiagWs.Dispose();
            B.Dispose();
            dVec.Dispose();
            eVec.Dispose();
            Ut.Dispose();
            Vt.Dispose();
        }
    }
}
