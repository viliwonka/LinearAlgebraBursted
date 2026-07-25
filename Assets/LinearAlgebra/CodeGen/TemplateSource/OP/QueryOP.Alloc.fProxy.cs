using Unity.Mathematics;
using Unity.Collections;

namespace BULA
{
    // Standalone allocating wrappers for Query search operations: allocate their own buffers
    // via allocator.
    //
    // Zero-alloc ref-dest primitives (distancesToRow/Column etc.) stay in the rest of QueryOP;
    // this file only adds the allocating convenience surface.
    public static partial class Query
    {
        // -------------------------------------------------------------------------
        // distancesToRow / distancesToColumn — allocate fProxyN standalone
        // -------------------------------------------------------------------------

        /// <summary>
        /// Allocates a fresh fProxyN (length A.M_Rows) and fills it
        /// with the distance/similarity from each row of A to query q under metric m.
        /// </summary>
        public static fProxyN fProxyDistancesToRow(in fProxyMxN A, in fProxyN q, Metric m, Allocator allocator = Allocator.Temp)
        {
            var dest = new fProxyN(A.M_Rows, allocator);
            Query.distancesToRow(in A, in q, m, ref dest);
            return dest;
        }

        /// <summary>
        /// Allocates a fresh fProxyN (length A.N_Cols) and fills it
        /// with the distance/similarity from each column of A to query q under metric m.
        /// </summary>
        public static fProxyN fProxyDistancesToColumn(in fProxyMxN A, in fProxyN q, Metric m, Allocator allocator = Allocator.Temp)
        {
            var dest = new fProxyN(A.N_Cols, allocator);
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
        public static Indices fProxyNonzeroIndices<T>(in T x, fProxy tol, Allocator allocator = Allocator.Temp)
            where T : unmanaged, IUnsafefProxyArray
        {
            int count = Query.countNonzero(in x, tol);
            if (count == 0) return new Indices(0, allocator);
            var idx = new Indices(count, allocator);
            int written = 0;
            for (int i = 0; i < x.Data.Length && written < idx.N; i++)
                if (math.abs(x.Data[i]) > tol) idx[written++] = i;
            return idx;
        }

        // -------------------------------------------------------------------------
        // rowsWithinRadius / columnsWithinRadius — count-pass + exact-alloc Indices
        // -------------------------------------------------------------------------

        /// <summary>
        /// Two-pass: count + exact-alloc Indices of row indices within radius r.
        /// </summary>
        public static Indices fProxyRowsWithinRadius(in fProxyMxN A, in fProxyN q, fProxy r, Metric m, Allocator allocator = Allocator.Temp)
        {
            int count = Query.countWithinRadius(in A, in q, r, m);
            if (count == 0) return new Indices(0, allocator);
            var idx = new Indices(count, allocator);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int written = 0;
            for (int row = 0; row < A.M_Rows && written < idx.N; row++)
            {
                fProxy s = Query.RowScore(in A, row, in q, m);
                if (sim ? s >= r : s <= r) idx[written++] = row;
            }
            return idx;
        }

        /// <summary>
        /// Two-pass: count + exact-alloc Indices of column indices within radius r.
        /// </summary>
        public static Indices fProxyColumnsWithinRadius(in fProxyMxN A, in fProxyN q, fProxy r, Metric m, Allocator allocator = Allocator.Temp)
        {
            int count = Query.countWithinColumnRadius(in A, in q, r, m);
            if (count == 0) return new Indices(0, allocator);
            var idx = new Indices(count, allocator);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int written = 0;
            for (int c = 0; c < A.N_Cols && written < idx.N; c++)
            {
                fProxy s = Query.ColScore(in A, c, in q, m);
                if (sim ? s >= r : s <= r) idx[written++] = c;
            }
            return idx;
        }

        // -------------------------------------------------------------------------
        // kNearestRows / kNearestColumns — standalone-alloc Indices + fProxyN scores
        // -------------------------------------------------------------------------

        /// <summary>
        /// Allocates clamped-k Indices + fProxyN, fills via kNearestRows.
        /// Returns idx; scores and count are out params. count = min(k, A.M_Rows).
        /// </summary>
        public static Indices fProxyKNearestRows(in fProxyMxN A, in fProxyN q, int k, Metric m, out fProxyN scores, out int count, Allocator allocator = Allocator.Temp)
        {
            int clampedK = math.min(k, A.M_Rows);
            if (clampedK <= 0) { scores = new fProxyN(0, allocator, true); count = 0; return new Indices(0, allocator); }
            var idx = new Indices(clampedK, allocator);
            scores = new fProxyN(clampedK, allocator);
            count = Query.kNearestRows(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        /// <summary>
        /// Allocates clamped-k Indices + fProxyN, fills via kNearestColumns.
        /// </summary>
        public static Indices fProxyKNearestColumns(in fProxyMxN A, in fProxyN q, int k, Metric m, out fProxyN scores, out int count, Allocator allocator = Allocator.Temp)
        {
            int clampedK = math.min(k, A.N_Cols);
            if (clampedK <= 0) { scores = new fProxyN(0, allocator, true); count = 0; return new Indices(0, allocator); }
            var idx = new Indices(clampedK, allocator);
            scores = new fProxyN(clampedK, allocator);
            count = Query.kNearestColumns(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        // -------------------------------------------------------------------------
        // kFarthestRows / kFarthestColumns — standalone-alloc Indices + fProxyN scores
        // -------------------------------------------------------------------------

        /// <summary>
        /// Allocates clamped-k Indices + fProxyN, fills via kFarthestRows.
        /// Returns idx; scores and count are out params. count = min(k, A.M_Rows).
        /// </summary>
        public static Indices fProxyKFarthestRows(in fProxyMxN A, in fProxyN q, int k, Metric m, out fProxyN scores, out int count, Allocator allocator = Allocator.Temp)
        {
            int clampedK = math.min(k, A.M_Rows);
            if (clampedK <= 0) { scores = new fProxyN(0, allocator, true); count = 0; return new Indices(0, allocator); }
            var idx = new Indices(clampedK, allocator);
            scores = new fProxyN(clampedK, allocator);
            count = Query.kFarthestRows(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        /// <summary>
        /// Allocates clamped-k Indices + fProxyN, fills via kFarthestColumns.
        /// </summary>
        public static Indices fProxyKFarthestColumns(in fProxyMxN A, in fProxyN q, int k, Metric m, out fProxyN scores, out int count, Allocator allocator = Allocator.Temp)
        {
            int clampedK = math.min(k, A.N_Cols);
            if (clampedK <= 0) { scores = new fProxyN(0, allocator, true); count = 0; return new Indices(0, allocator); }
            var idx = new Indices(clampedK, allocator);
            scores = new fProxyN(clampedK, allocator);
            count = Query.kFarthestColumns(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }
    }
}
