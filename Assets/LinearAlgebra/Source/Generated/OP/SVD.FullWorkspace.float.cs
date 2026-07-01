using System;

namespace LinearAlgebra
{
    public static partial class SVD
    {
        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an m x n full SVD (U m x n, S length n,
        /// V n x n) — the layout produced by Arena.floatSVDFull_WS(m, n).
        /// </summary>
        static void RequireSvdFullWorkspace(in floatSVDFull_WS ws, int m, int n, string who)
        {
            if (ws.U.M_Rows != m || ws.U.N_Cols != n || ws.S.N != n || ws.V.M_Rows != n || ws.V.N_Cols != n)
                throw new ArgumentException(who + ": workspace must be sized for m x n (use Arena.floatSVDFull_WS(m, n))");
        }
    }

    /// <summary>
    /// Reusable scratch storage for the full-SVD-family ops that each compute one Golub-Kahan SVD of
    /// an m x n (m >= n) matrix and slice it: svdTruncated, lowRankApprox, nullspaceBasis, rangeBasis.
    /// Allocate ONCE (sized for the matrix shape) via Arena.floatSVDFull_WS(m, n) and reuse it
    /// across many same-shape calls to avoid the per-call temp allocations.
    ///
    /// Layout matches the (U, S, V) factors svdThin writes: U is m x n (left singular vectors),
    /// S is length n (singular values), V is n x n (right singular vectors). These are exactly the
    /// three scratch buffers the workspace overloads of the family ops expect, bundled so callers
    /// don't size them by hand.
    ///
    /// NOTE: this removes the per-call arena temp-pool allocations of U/S/V; the inner Golub-Kahan SVD
    /// still allocates a little Allocator.Temp scratch of its own each call, so the family ops are
    /// low-alloc rather than strictly zero-alloc.
    /// </summary>
    public struct floatSVDFull_WS
    {
        public floatMxN U;
        public floatN S;
        public floatMxN V;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates a full-SVD-family workspace sized for an m x n (m >= n) system: U is m x n,
        /// S is length n, V is n x n. The buffers are persistent in this arena (disposed with it), so
        /// create the workspace once outside a hot loop and pass it to the workspace overloads of
        /// svdTruncated / lowRankApprox / nullspaceBasis / rangeBasis.
        /// </summary>
        public static floatSVDFull_WS floatSVDFull_WS(this ref Arena arena, int m, int n)
        {
            return new floatSVDFull_WS
            {
                U = arena.floatMat(m, n),
                S = arena.floatVec(n),
                V = arena.floatMat(n, n)
            };
        }
    }
}
