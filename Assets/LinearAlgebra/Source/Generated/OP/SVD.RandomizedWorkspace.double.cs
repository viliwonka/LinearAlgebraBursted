using System;

using Unity.Mathematics;

namespace LinearAlgebra
{
    public static partial class SVD
    {
        /// <summary>
        /// Validates the svdRandomized arguments (shared by the allocating and ref-workspace overloads
        /// so both report the same messages, and the allocating one fails before sizing temps).
        /// </summary>
        static void RequireRandomizedArgs(int m, int n, int k, int oversample, int powerIters,
                                          in doubleMxN Uk, in doubleN Sk, in doubleMxN Vk, int maxIter)
        {
            if (m < n)
                throw new ArgumentException("svdRandomized: A must have m >= n (more rows than columns)");
            if (k < 1 || k > n)
                throw new ArgumentException("svdRandomized: k must be in [1, A.N_Cols]");
            if (oversample < 0)
                throw new ArgumentException("svdRandomized: oversample must be >= 0");
            if (powerIters < 0)
                throw new ArgumentException("svdRandomized: powerIters must be >= 0");
            if (Uk.M_Rows != m || Uk.N_Cols != k)
                throw new ArgumentException("svdRandomized: Uk must be m x k");
            if (Sk.N != k)
                throw new ArgumentException("svdRandomized: Sk must have length k");
            if (Vk.M_Rows != n || Vk.N_Cols != k)
                throw new ArgumentException("svdRandomized: Vk must be n x k");
            if (maxIter < 1)
                throw new ArgumentException("svdRandomized: maxIter must be >= 1");
        }

        /// <summary>
        /// Throws if <paramref name="ws"/> is not sized for an m x n randomized SVD with sketch width
        /// l = min(k + oversample, n) — the layout produced by
        /// Arena.doubleSVDRandomized_WS(m, n, k, oversample).
        /// </summary>
        static void RequireSvdRandomizedWorkspace(in doubleSVDRandomized_WS ws, int m, int n, int l)
        {
            bool ok =
                ws.Omega.M_Rows == n && ws.Omega.N_Cols == l &&
                ws.Y.M_Rows == m && ws.Y.N_Cols == l &&
                ws.R.M_Rows == l && ws.R.N_Cols == l &&
                ws.qu.N == m &&
                ws.qw.N == l &&
                ws.Z.M_Rows == n && ws.Z.N_Cols == l &&
                ws.B.M_Rows == l && ws.B.N_Cols == n &&
                ws.Bt.M_Rows == n && ws.Bt.N_Cols == l &&
                ws.Up.M_Rows == n && ws.Up.N_Cols == l &&
                ws.Sb.N == l &&
                ws.Vp.M_Rows == l && ws.Vp.N_Cols == l &&
                ws.UA.M_Rows == m && ws.UA.N_Cols == l;

            if (!ok)
                throw new ArgumentException(
                    "svdRandomized: workspace must be sized for this (m, n, k, oversample) — use " +
                    "Arena.doubleSVDRandomized_WS(m, n, k, oversample) with the SAME k and oversample");
        }
    }

    /// <summary>
    /// Reusable scratch storage for svdRandomized (Halko-Martinsson-Tropp). The randomized SVD
    /// allocates a dozen intermediate buffers per call; allocate this ONCE via
    /// Arena.doubleSVDRandomized_WS(m, n, k, oversample) and reuse it across same-shape calls
    /// (SAME k and oversample) to make repeated randomized SVDs zero-alloc.
    ///
    /// All buffers are sized by the sketch width l = min(k + oversample, n): Omega (n x l), Y (m x l,
    /// becomes the orthonormal range basis Q), R (l x l), qu (m) / qw (l) QR scratch, Z (n x l),
    /// B (l x n), Bt (n x l), Up (n x l), Sb (l), Vp (l x l), UA (m x l).
    ///
    /// NOTE: this removes svdRandomized's dozen per-call temp-pool allocations; the inner exact
    /// svdThin on the small Bt still uses a little Allocator.Temp scratch of its own, so the op
    /// is low-alloc rather than strictly zero-alloc.
    /// </summary>
    public struct doubleSVDRandomized_WS
    {
        public doubleMxN Omega;
        public doubleMxN Y;
        public doubleMxN R;
        public doubleN qu;
        public doubleN qw;
        public doubleMxN Z;
        public doubleMxN B;
        public doubleMxN Bt;
        public doubleMxN Up;
        public doubleN Sb;
        public doubleMxN Vp;
        public doubleMxN UA;
    }

    public static partial class ArenaExtensions
    {
        /// <summary>
        /// Allocates a randomized-SVD workspace for an m x n (m >= n) matrix with target rank k and
        /// oversampling p, sized by the sketch width l = min(k + oversample, n). Pass the SAME k and
        /// oversample to svdRandomized's ref-workspace overload. The buffers are persistent in this
        /// arena (disposed with it), so create the workspace once outside a hot loop.
        /// </summary>
        public static doubleSVDRandomized_WS doubleSVDRandomized_WS(this ref Arena arena, int m, int n, int k, int oversample)
        {
            int l = math.min(k + oversample, n);
            return new doubleSVDRandomized_WS
            {
                Omega = arena.doubleMat(n, l),
                Y = arena.doubleMat(m, l),
                R = arena.doubleMat(l, l),
                qu = arena.doubleVec(m),
                qw = arena.doubleVec(l),
                Z = arena.doubleMat(n, l),
                B = arena.doubleMat(l, n),
                Bt = arena.doubleMat(n, l),
                Up = arena.doubleMat(n, l),
                Sb = arena.doubleVec(l),
                Vp = arena.doubleMat(l, l),
                UA = arena.doubleMat(m, l)
            };
        }

        /// <summary>
        /// Allocates a randomized-SVD workspace with the default oversample (10) — matches the
        /// svdRandomized convenience overloads (oversample 10, powerIters 2, maxIter 75).
        /// </summary>
        public static doubleSVDRandomized_WS doubleSVDRandomized_WS(this ref Arena arena, int m, int n, int k)
            => arena.doubleSVDRandomized_WS(m, n, k, 10);
    }
}
