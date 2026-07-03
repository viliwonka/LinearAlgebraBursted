using System;

namespace LinearAlgebra
{
    public static partial class SVD
    {
        /// <summary>Throws unless <paramref name="ws"/> matches Arena.doubleSVDFull_WS(m, n) sizing.</summary>
        static void RequireSvdFullWorkspace(in doubleSVDFull_WS ws, int m, int n)
        {
            if (ws.U.M_Rows != m || ws.U.N_Cols != n || ws.S.N != n || ws.V.M_Rows != n || ws.V.N_Cols != n)
                throw new ArgumentException("SVD: workspace must be sized for m x n (use Arena.doubleSVDFull_WS(m, n))");
        }
    }

    /// <summary>
    /// Reusable scratch for the full-SVD-family ops that each compute one Golub-Kahan SVD of an m x n
    /// (m >= n) matrix and slice it: svdTruncated, lowRankApprox, nullspaceBasis, rangeBasis. Allocate
    /// ONCE via Arena.doubleSVDFull_WS(m, n) and reuse across same-shape calls to avoid per-call temp
    /// allocations.
    ///
    /// Layout matches svdThin's (U, S, V): U is m x n (left singular vectors), S is length n (singular
    /// values), V is n x n (right singular vectors).
    ///
    /// NOTE: removes the per-call U/S/V temp-pool allocations; the inner Golub-Kahan SVD still uses a
    /// little Allocator.Temp scratch of its own, so this is low-alloc rather than strictly zero-alloc.
    /// </summary>
    public struct doubleSVDFull_WS
    {
        public doubleMxN U;
        public doubleN S;
        public doubleMxN V;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates a full-SVD-family workspace for an m x n (m >= n) system — see
        /// <see cref="doubleSVDFull_WS"/> for layout. Persistent in this arena; pass to the workspace
        /// overloads of svdTruncated / lowRankApprox / nullspaceBasis / rangeBasis.
        /// </summary>
        public static doubleSVDFull_WS doubleSVDFull_WS(this ref Arena arena, int m, int n)
        {
            return new doubleSVDFull_WS
            {
                U = arena.doubleMat(m, n),
                S = arena.doubleVec(n),
                V = arena.doubleMat(n, n)
            };
        }
    }
}
