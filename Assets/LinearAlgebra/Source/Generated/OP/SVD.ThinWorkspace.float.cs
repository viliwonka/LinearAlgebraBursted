using System;

namespace LinearAlgebra
{
    public static partial class SVD
    {
        /// <summary>Throws unless <paramref name="ws"/> matches Arena.floatSVDThinCache(m, n) sizing (BidiagWs is validated separately, by Bidiag.decomp itself).</summary>
        static void RequireSvdThinWorkspace(in floatSVDThinCache ws, int m, int n)
        {
            bool ok =
                ws.B.M_Rows == n && ws.B.N_Cols == n &&
                ws.dVec.N == n && ws.eVec.N == n &&
                ws.Ut.M_Rows == n && ws.Ut.N_Cols == m &&
                ws.Vt.M_Rows == n && ws.Vt.N_Cols == n;

            if (!ok)
                throw new ArgumentException("SVD: workspace must be sized for m x n (use Arena.floatSVDThinCache(m, n))");
        }
    }

    /// <summary>
    /// Reusable scratch for SVD.thin (Golub-Kahan bidiagonalization + implicit-shift bidiagonal QR).
    /// Allocate ONCE (sized for the matrix shape) via Arena.floatSVDThinCache(m, n) and reuse it across
    /// many same-shape calls to avoid the per-call Allocator.Temp allocations thin's allocating
    /// overload makes internally.
    ///
    /// BidiagWs is the nested workspace Bidiag.decomp needs (see floatBidiagCache); B (n x n) is
    /// the bidiagonal factor; dVec/eVec (length n) are the diagonal/superdiagonal the bidiagonal QR
    /// diagonalizes; Ut (n x m) / Vt (n x n) are the transposed accumulators the QR sweep rotates
    /// (unit-stride rows, same trick as Eigen.symmetric).
    /// </summary>
    public struct floatSVDThinCache
    {
        public floatBidiagCache BidiagWs;
        public floatMxN B;
        public floatN dVec;
        public floatN eVec;
        public floatMxN Ut;
        public floatMxN Vt;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates an thin workspace for an m x n (m >= n) system — see
        /// <see cref="floatSVDThinCache"/> for layout. Persistent in this arena; create once outside a
        /// hot loop and pass to thin's ref-workspace overload.
        /// </summary>
        public static floatSVDThinCache floatSVDThinCache(this ref Arena arena, int m, int n)
        {
            return new floatSVDThinCache
            {
                BidiagWs = arena.floatBidiagCache(m, n),
                B = arena.floatMat(n, n),
                dVec = arena.floatVec(n),
                eVec = arena.floatVec(n),
                Ut = arena.floatMat(n, m),
                Vt = arena.floatMat(n, n)
            };
        }
    }
}
