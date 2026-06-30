using Unity.Mathematics;

namespace LinearAlgebra
{
    // Allocating (arena) wrappers for iProxyQuery_OP search operations.
    // Zero-alloc ref-dest primitives (distancesToRow/Column) are in iProxyQuery_OP;
    // these wrappers do count-pass + exact-alloc so callers don't size buffers manually.
    //
    // All Indices buffers use the shared Indices type (arena.Indices(n)) — assembly-shared,
    // no duplication. Score buffers are iProxyN allocated from the arena.
    //
    // Supported integer-exact metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
    // Euclidean and Cosine throw ArgumentException (float-only).
    // Overflow note: element-wise differences must fit the proxy type (for short: ±16383
    // coordinates so differences fit ±32767). SqEuclidean/Dot: maxAbs²×dim must fit.
    // Use float/double for larger ranges.
    public static partial class ArenaExtensions
    {
        // -------------------------------------------------------------------------
        // distancesToRow / distancesToColumn — allocate iProxyN from A's arena
        // -------------------------------------------------------------------------

        /// <summary>
        /// Allocates a fresh iProxyN (length A.M_Rows) from A's arena and fills it
        /// with the distance/similarity from each row of A to query q under metric m.
        /// Supported metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × N_Cols fits the type; otherwise use the float variant.
        /// Manhattan/Chebyshev require element differences to fit the proxy type (e.g. for short, roughly ±16383 so differences fit ±32767). Use float/double for larger ranges.
        /// </para>
        /// </summary>
        public static iProxyN iProxyDistancesToRow(in iProxyMxN A, in iProxyN q, Metric m)
        {
            var dest = A.iProxyVec(A.M_Rows);
            iProxyQuery_OP.distancesToRow(in A, in q, m, ref dest);
            return dest;
        }

        /// <summary>
        /// Allocates a fresh iProxyN (length A.N_Cols) from A's arena and fills it
        /// with the distance/similarity from each column of A to query q under metric m.
        /// Supported metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × M_Rows fits the type; otherwise use the float variant.
        /// Manhattan/Chebyshev require element differences to fit the proxy type (e.g. for short, roughly ±16383 so differences fit ±32767). Use float/double for larger ranges.
        /// </para>
        /// </summary>
        public static iProxyN iProxyDistancesToColumn(in iProxyMxN A, in iProxyN q, Metric m)
        {
            var dest = A.iProxyVec(A.N_Cols);
            iProxyQuery_OP.distancesToColumn(in A, in q, m, ref dest);
            return dest;
        }

        // -------------------------------------------------------------------------
        // nonzeroIndices — count-pass + exact-alloc Indices
        // -------------------------------------------------------------------------

        /// <summary>
        /// Two-pass: count nonzero elements, allocate exact-sized Indices, fill indices.
        /// Returns the allocated Indices (length = count). Arena-owned.
        /// </summary>
        public static Indices iProxyNonzeroIndices<T>(this ref Arena arena, in T x, iProxy tol)
            where T : unmanaged, IUnsafeiProxyArray
        {
            int count = iProxyQuery_OP.countNonzero(in x, tol);
            if (count == 0) return arena.Indices(0);
            var idx = arena.Indices(count);
            int written = 0;
            for (int i = 0; i < x.Data.Length; i++)
            {
                iProxy v = x.Data[i];
                iProxy av = v < (iProxy)0 ? (v == iProxy.MinValue ? iProxy.MaxValue : (iProxy)(-v)) : v;
                if (av > tol) idx[written++] = i;
            }
            return idx;
        }

        // -------------------------------------------------------------------------
        // rowsWithinRadius / columnsWithinRadius — count-pass + exact-alloc Indices
        // -------------------------------------------------------------------------

        /// <summary>
        /// Two-pass: count + exact-alloc Indices of row indices within radius r.
        /// Supported metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): see iProxyDistancesToRow. Manhattan/Chebyshev are overflow-safe.
        /// </para>
        /// </summary>
        public static Indices iProxyRowsWithinRadius(this ref Arena arena, in iProxyMxN A, in iProxyN q, iProxy r, Metric m)
        {
            int count = iProxyQuery_OP.countWithinRadius(in A, in q, r, m);
            if (count == 0) return arena.Indices(0);
            var idx = arena.Indices(count);
            bool sim = m == Metric.Dot;
            int written = 0;
            for (int row = 0; row < A.M_Rows; row++)
            {
                iProxy s = iProxyQuery_OP.RowScore(in A, row, in q, m);
                if (sim ? s >= r : s <= r) idx[written++] = row;
            }
            return idx;
        }

        /// <summary>
        /// Two-pass: count + exact-alloc Indices of column indices within radius r.
        /// Supported metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): see iProxyDistancesToColumn. Manhattan/Chebyshev are overflow-safe.
        /// </para>
        /// </summary>
        public static Indices iProxyColumnsWithinRadius(this ref Arena arena, in iProxyMxN A, in iProxyN q, iProxy r, Metric m)
        {
            int count = iProxyQuery_OP.countWithinColumnRadius(in A, in q, r, m);
            if (count == 0) return arena.Indices(0);
            var idx = arena.Indices(count);
            bool sim = m == Metric.Dot;
            int written = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                iProxy s = iProxyQuery_OP.ColScore(in A, c, in q, m);
                if (sim ? s >= r : s <= r) idx[written++] = c;
            }
            return idx;
        }

        // -------------------------------------------------------------------------
        // kNearestRows / kNearestColumns — arena-alloc Indices + iProxyN scores
        // -------------------------------------------------------------------------

        /// <summary>
        /// Allocates clamped-k Indices + iProxyN from arena, fills via kNearestRows.
        /// Returns idx; scores and count are out params. count = min(k, A.M_Rows).
        /// Supported metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × N_Cols fits the type; otherwise use the float variant.
        /// Manhattan/Chebyshev are the recommended integer metrics; element differences must fit the proxy type (e.g. for short, roughly ±16383 so differences fit ±32767).
        /// </para>
        /// </summary>
        public static Indices iProxyKNearestRows(this ref Arena arena, in iProxyMxN A, in iProxyN q, int k, Metric m, out iProxyN scores, out int count)
        {
            int clampedK = math.min(k, A.M_Rows);
            if (clampedK <= 0) { scores = A.iProxyVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.iProxyVec(clampedK);
            count = iProxyQuery_OP.kNearestRows(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        /// <summary>
        /// Allocates clamped-k Indices + iProxyN from arena, fills via kNearestColumns.
        /// Supported metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × M_Rows fits the type; otherwise use the float variant.
        /// Manhattan and Chebyshev are the recommended integer metrics (see overflow note on iProxyKNearestRows).
        /// </para>
        /// </summary>
        public static Indices iProxyKNearestColumns(this ref Arena arena, in iProxyMxN A, in iProxyN q, int k, Metric m, out iProxyN scores, out int count)
        {
            int clampedK = math.min(k, A.N_Cols);
            if (clampedK <= 0) { scores = A.iProxyVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.iProxyVec(clampedK);
            count = iProxyQuery_OP.kNearestColumns(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        // -------------------------------------------------------------------------
        // kFarthestRows / kFarthestColumns — arena-alloc Indices + iProxyN scores
        // -------------------------------------------------------------------------

        /// <summary>
        /// Allocates clamped-k Indices + iProxyN from arena, fills via kFarthestRows.
        /// Returns idx; scores and count are out params. count = min(k, A.M_Rows).
        /// Supported metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × N_Cols fits the type; otherwise use the float variant.
        /// Manhattan and Chebyshev are the recommended integer metrics (see overflow note on iProxyKNearestRows).
        /// </para>
        /// </summary>
        public static Indices iProxyKFarthestRows(this ref Arena arena, in iProxyMxN A, in iProxyN q, int k, Metric m, out iProxyN scores, out int count)
        {
            int clampedK = math.min(k, A.M_Rows);
            if (clampedK <= 0) { scores = A.iProxyVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.iProxyVec(clampedK);
            count = iProxyQuery_OP.kFarthestRows(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        /// <summary>
        /// Allocates clamped-k Indices + iProxyN from arena, fills via kFarthestColumns.
        /// Returns idx; scores and count are out params. count = min(k, A.N_Cols).
        /// Supported metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × M_Rows fits the type; otherwise use the float variant.
        /// Manhattan and Chebyshev are the recommended integer metrics (see overflow note on iProxyKNearestRows).
        /// </para>
        /// </summary>
        public static Indices iProxyKFarthestColumns(this ref Arena arena, in iProxyMxN A, in iProxyN q, int k, Metric m, out iProxyN scores, out int count)
        {
            int clampedK = math.min(k, A.N_Cols);
            if (clampedK <= 0) { scores = A.iProxyVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.iProxyVec(clampedK);
            count = iProxyQuery_OP.kFarthestColumns(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }
    }
}
