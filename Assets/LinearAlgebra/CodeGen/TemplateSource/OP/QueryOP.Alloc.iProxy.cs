using Unity.Mathematics;
using Unity.Collections;

namespace BULA
{
    // Standalone allocating wrappers for Query search operations over integer types: allocate
    // their own buffers via allocator.
    //
    // Supported integer-exact metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
    // Euclidean and Cosine throw ArgumentException (float-only).
    // Overflow note: element-wise differences must fit the proxy type (for short: ±16383
    // coordinates so differences fit ±32767). SqEuclidean/Dot: maxAbs²×dim must fit.
    // Use float/double for larger ranges.
    public static partial class Query
    {
        // -------------------------------------------------------------------------
        // distancesToRow / distancesToColumn — allocate iProxyN standalone
        // -------------------------------------------------------------------------

        /// <summary>Allocates a fresh iProxyN (length A.M_Rows) with row distances/similarities to q under metric m. See class doc for supported metrics and overflow limits.</summary>
        public static iProxyN iProxyDistancesToRow(in iProxyMxN A, in iProxyN q, Metric m, Allocator allocator = Allocator.Temp)
        {
            var dest = new iProxyN(A.M_Rows, allocator);
            Query.distancesToRow(in A, in q, m, ref dest);
            return dest;
        }

        /// <summary>Allocates a fresh iProxyN (length A.N_Cols) with column distances/similarities to q under metric m. See class doc for supported metrics and overflow limits.</summary>
        public static iProxyN iProxyDistancesToColumn(in iProxyMxN A, in iProxyN q, Metric m, Allocator allocator = Allocator.Temp)
        {
            var dest = new iProxyN(A.N_Cols, allocator);
            Query.distancesToColumn(in A, in q, m, ref dest);
            return dest;
        }

        // -------------------------------------------------------------------------
        // nonzeroIndices — count-pass + exact-alloc Indices
        // -------------------------------------------------------------------------

        /// <summary>
        /// Two-pass: count nonzero elements, allocate exact-sized Indices, fill indices.
        /// Returns the allocated Indices (length = count).
        /// </summary>
        public static Indices iProxyNonzeroIndices<T>(in T x, iProxy tol, Allocator allocator = Allocator.Temp)
            where T : unmanaged, IUnsafeiProxyArray
        {
            int count = Query.countNonzero(in x, tol);
            if (count == 0) return new Indices(0, allocator);
            var idx = new Indices(count, allocator);
            int written = 0;
            for (int i = 0; i < x.Data.Length && written < idx.N; i++)
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

        /// <summary>Two-pass: counts, then exact-allocates Indices of row indices within radius r under metric m.</summary>
        public static Indices iProxyRowsWithinRadius(in iProxyMxN A, in iProxyN q, iProxy r, Metric m, Allocator allocator = Allocator.Temp)
        {
            int count = Query.countWithinRadius(in A, in q, r, m);
            if (count == 0) return new Indices(0, allocator);
            var idx = new Indices(count, allocator);
            bool sim = m == Metric.Dot;
            int written = 0;
            for (int row = 0; row < A.M_Rows && written < idx.N; row++)
            {
                iProxy s = Query.RowScore(in A, row, in q, m);
                if (sim ? s >= r : s <= r) idx[written++] = row;
            }
            return idx;
        }

        /// <summary>Two-pass: counts, then exact-allocates Indices of column indices within radius r under metric m.</summary>
        public static Indices iProxyColumnsWithinRadius(in iProxyMxN A, in iProxyN q, iProxy r, Metric m, Allocator allocator = Allocator.Temp)
        {
            int count = Query.countWithinColumnRadius(in A, in q, r, m);
            if (count == 0) return new Indices(0, allocator);
            var idx = new Indices(count, allocator);
            bool sim = m == Metric.Dot;
            int written = 0;
            for (int c = 0; c < A.N_Cols && written < idx.N; c++)
            {
                iProxy s = Query.ColScore(in A, c, in q, m);
                if (sim ? s >= r : s <= r) idx[written++] = c;
            }
            return idx;
        }

        // -------------------------------------------------------------------------
        // kNearestRows / kNearestColumns — standalone-alloc Indices + iProxyN scores
        // -------------------------------------------------------------------------

        /// <summary>Allocates clamped-k Indices + iProxyN scores, filled via kNearestRows; count = min(k, A.M_Rows).</summary>
        public static Indices iProxyKNearestRows(in iProxyMxN A, in iProxyN q, int k, Metric m, out iProxyN scores, out int count, Allocator allocator = Allocator.Temp)
        {
            int clampedK = math.min(k, A.M_Rows);
            if (clampedK <= 0) { scores = new iProxyN(0, allocator, true); count = 0; return new Indices(0, allocator); }
            var idx = new Indices(clampedK, allocator);
            scores = new iProxyN(clampedK, allocator);
            count = Query.kNearestRows(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        /// <summary>Allocates clamped-k Indices + iProxyN scores, filled via kNearestColumns; count = min(k, A.N_Cols).</summary>
        public static Indices iProxyKNearestColumns(in iProxyMxN A, in iProxyN q, int k, Metric m, out iProxyN scores, out int count, Allocator allocator = Allocator.Temp)
        {
            int clampedK = math.min(k, A.N_Cols);
            if (clampedK <= 0) { scores = new iProxyN(0, allocator, true); count = 0; return new Indices(0, allocator); }
            var idx = new Indices(clampedK, allocator);
            scores = new iProxyN(clampedK, allocator);
            count = Query.kNearestColumns(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        // -------------------------------------------------------------------------
        // kFarthestRows / kFarthestColumns — standalone-alloc Indices + iProxyN scores
        // -------------------------------------------------------------------------

        /// <summary>Allocates clamped-k Indices + iProxyN scores, filled via kFarthestRows; count = min(k, A.M_Rows).</summary>
        public static Indices iProxyKFarthestRows(in iProxyMxN A, in iProxyN q, int k, Metric m, out iProxyN scores, out int count, Allocator allocator = Allocator.Temp)
        {
            int clampedK = math.min(k, A.M_Rows);
            if (clampedK <= 0) { scores = new iProxyN(0, allocator, true); count = 0; return new Indices(0, allocator); }
            var idx = new Indices(clampedK, allocator);
            scores = new iProxyN(clampedK, allocator);
            count = Query.kFarthestRows(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        /// <summary>Allocates clamped-k Indices + iProxyN scores, filled via kFarthestColumns; count = min(k, A.N_Cols).</summary>
        public static Indices iProxyKFarthestColumns(in iProxyMxN A, in iProxyN q, int k, Metric m, out iProxyN scores, out int count, Allocator allocator = Allocator.Temp)
        {
            int clampedK = math.min(k, A.N_Cols);
            if (clampedK <= 0) { scores = new iProxyN(0, allocator, true); count = 0; return new Indices(0, allocator); }
            var idx = new Indices(clampedK, allocator);
            scores = new iProxyN(clampedK, allocator);
            count = Query.kFarthestColumns(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }
    }
}
