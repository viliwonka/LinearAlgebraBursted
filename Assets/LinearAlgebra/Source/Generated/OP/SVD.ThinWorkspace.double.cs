using System;

namespace LinearAlgebra
{
    public static partial class SVD
    {
        /// <summary>Throws unless <paramref name="ws"/> matches Arena.doubleSVDThinCache(m, n) sizing (BidiagWs is validated separately, by Bidiag.decomp itself).</summary>
        static void RequireSvdThinWorkspace(in doubleSVDThinCache ws, int m, int n)
        {
            bool ok =
                ws.B.M_Rows == n && ws.B.N_Cols == n &&
                ws.dVec.N == n && ws.eVec.N == n &&
                ws.Ut.M_Rows == n && ws.Ut.N_Cols == m &&
                ws.Vt.M_Rows == n && ws.Vt.N_Cols == n;

            if (!ok)
                throw new ArgumentException("SVD: workspace must be sized for m x n (use Arena.doubleSVDThinCache(m, n))");
        }
    }

    /// <summary>
    /// Reusable scratch for SVD.thin (Golub-Kahan bidiagonalization + implicit-shift bidiagonal QR).
    /// Allocate ONCE (sized for the matrix shape) via Arena.doubleSVDThinCache(m, n) and reuse it across
    /// many same-shape calls to avoid the per-call Allocator.Temp allocations thin's allocating
    /// overload makes internally.
    ///
    /// BidiagWs is the nested workspace Bidiag.decomp needs (see doubleBidiagCache); B (n x n) is
    /// the bidiagonal factor; dVec/eVec (length n) are the diagonal/superdiagonal the bidiagonal QR
    /// diagonalizes; Ut (n x m) / Vt (n x n) are the transposed accumulators the QR sweep rotates
    /// (unit-stride rows, same trick as Eigen.symmetricInPlace).
    /// </summary>
    public struct doubleSVDThinCache
    {
        public doubleBidiagCache BidiagWs;
        public doubleMxN B;
        public doubleN dVec;
        public doubleN eVec;
        public doubleMxN Ut;
        public doubleMxN Vt;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates an thin workspace for an m x n (m >= n) system — see
        /// <see cref="doubleSVDThinCache"/> for layout. Persistent in this arena; create once outside a
        /// hot loop and pass to thin's ref-workspace overload.
        /// </summary>
        public static doubleSVDThinCache doubleSVDThinCache(this ref Arena arena, int m, int n)
        {
            return new doubleSVDThinCache
            {
                BidiagWs = arena.doubleBidiagCache(m, n),
                B = arena.doubleMat(n, n),
                dVec = arena.doubleVec(n),
                eVec = arena.doubleVec(n),
                Ut = arena.doubleMat(n, m),
                Vt = arena.doubleMat(n, n)
            };
        }
    }
}
