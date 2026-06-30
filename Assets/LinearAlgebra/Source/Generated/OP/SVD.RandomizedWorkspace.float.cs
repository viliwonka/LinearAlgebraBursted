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
                                          in floatMxN Uk, in floatN Sk, in floatMxN Vk, int maxIter)
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
        /// Arena.floatSvdRandomizedWorkspace(m, n, k, oversample).
        /// </summary>
        static void RequireSvdRandomizedWorkspace(in floatSvdRandomizedWorkspace ws, int m, int n, int l)
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
                    "Arena.floatSvdRandomizedWorkspace(m, n, k, oversample) with the SAME k and oversample");
        }
    }

    /// <summary>
    /// Reusable scratch storage for svdRandomized (Halko-Martinsson-Tropp). The randomized SVD
    /// allocates a dozen intermediate buffers per call; allocate this ONCE via
    /// Arena.floatSvdRandomizedWorkspace(m, n, k, oversample) and reuse it across same-shape calls
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
    public struct floatSvdRandomizedWorkspace
    {
        public floatMxN Omega;
        public floatMxN Y;
        public floatMxN R;
        public floatN qu;
        public floatN qw;
        public floatMxN Z;
        public floatMxN B;
        public floatMxN Bt;
        public floatMxN Up;
        public floatN Sb;
        public floatMxN Vp;
        public floatMxN UA;
    }

    public partial struct Arena
    {
        /// <summary>
        /// Allocates a randomized-SVD workspace for an m x n (m >= n) matrix with target rank k and
        /// oversampling p, sized by the sketch width l = min(k + oversample, n). Pass the SAME k and
        /// oversample to svdRandomized's ref-workspace overload. The buffers are persistent in this
        /// arena (disposed with it), so create the workspace once outside a hot loop.
        /// </summary>
        public floatSvdRandomizedWorkspace floatSvdRandomizedWorkspace(int m, int n, int k, int oversample)
        {
            int l = math.min(k + oversample, n);
            return new floatSvdRandomizedWorkspace
            {
                Omega = floatMat(n, l),
                Y = floatMat(m, l),
                R = floatMat(l, l),
                qu = floatVec(m),
                qw = floatVec(l),
                Z = floatMat(n, l),
                B = floatMat(l, n),
                Bt = floatMat(n, l),
                Up = floatMat(n, l),
                Sb = floatVec(l),
                Vp = floatMat(l, l),
                UA = floatMat(m, l)
            };
        }

        /// <summary>
        /// Allocates a randomized-SVD workspace with the default oversample (10) — matches the
        /// svdRandomized convenience overloads (oversample 10, powerIters 2, maxIter 75).
        /// </summary>
        public floatSvdRandomizedWorkspace floatSvdRandomizedWorkspace(int m, int n, int k)
            => floatSvdRandomizedWorkspace(m, n, k, 10);
    }
}
