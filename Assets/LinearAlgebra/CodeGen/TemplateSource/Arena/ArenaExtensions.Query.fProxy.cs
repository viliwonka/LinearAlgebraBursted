using Unity.Mathematics;

namespace LinearAlgebra
{
    // Allocating (arena) wrappers for fProxyQueryOP search operations.
    // Zero-alloc ref-dest primitives (distancesToRow/Column) are in QueryOP;
    // these wrappers allocate from the matrix's own arena so callers don't
    // have to size buffers manually for the fProxy-typed outputs.
    //
    // Indices-returning wrappers (nonzero, rowsWithinRadius, kNearestRows, etc.)
    // are in this file using Indices as the shared buffer type, generated via
    // the fProxy template into floatXxx / doubleXxx variants.
    public static partial class ArenaExtensions
    {
        // -------------------------------------------------------------------------
        // distancesToRow / distancesToColumn — allocate fProxyN from A's arena
        // -------------------------------------------------------------------------

        /// <summary>
        /// Allocates a fresh fProxyN (length A.M_Rows) from A's arena and fills it
        /// with the distance/similarity from each row of A to query q under metric m.
        /// </summary>
        public static fProxyN fProxyDistancesToRow(in fProxyMxN A, in fProxyN q, Metric m)
        {
            var dest = A.fProxyVec(A.M_Rows);
            fProxyQueryOP.distancesToRow(in A, in q, m, ref dest);
            return dest;
        }

        /// <summary>
        /// Allocates a fresh fProxyN (length A.N_Cols) from A's arena and fills it
        /// with the distance/similarity from each column of A to query q under metric m.
        /// </summary>
        public static fProxyN fProxyDistancesToColumn(in fProxyMxN A, in fProxyN q, Metric m)
        {
            var dest = A.fProxyVec(A.N_Cols);
            fProxyQueryOP.distancesToColumn(in A, in q, m, ref dest);
            return dest;
        }

        // -------------------------------------------------------------------------
        // nonzeroIndices — count-pass + exact-alloc Indices
        // -------------------------------------------------------------------------

        /// <summary>
        /// Two-pass: count nonzero elements, allocate exact-sized Indices, fill indices.
        /// Returns the allocated Indices (length = count). Arena-owned.
        /// </summary>
        public static Indices fProxyNonzeroIndices<T>(this ref Arena arena, in T x, fProxy tol)
            where T : unmanaged, IUnsafefProxyArray
        {
            int count = fProxyQueryOP.countNonzero(in x, tol);
            if (count == 0) return arena.Indices(0);
            var idx = arena.Indices(count);
            int written = 0;
            for (int i = 0; i < x.Data.Length; i++)
                if (math.abs(x.Data[i]) > tol) idx[written++] = i;
            return idx;
        }

        // -------------------------------------------------------------------------
        // rowsWithinRadius / columnsWithinRadius — count-pass + exact-alloc Indices
        // -------------------------------------------------------------------------

        /// <summary>
        /// Two-pass: count + exact-alloc Indices of row indices within radius r.
        /// </summary>
        public static Indices fProxyRowsWithinRadius(this ref Arena arena, in fProxyMxN A, in fProxyN q, fProxy r, Metric m)
        {
            int count = fProxyQueryOP.countWithinRadius(in A, in q, r, m);
            if (count == 0) return arena.Indices(0);
            var idx = arena.Indices(count);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int written = 0;
            for (int row = 0; row < A.M_Rows; row++)
            {
                fProxy s = fProxyQueryOP.RowScore(in A, row, in q, m);
                if (sim ? s >= r : s <= r) idx[written++] = row;
            }
            return idx;
        }

        /// <summary>
        /// Two-pass: count + exact-alloc Indices of column indices within radius r.
        /// </summary>
        public static Indices fProxyColumnsWithinRadius(this ref Arena arena, in fProxyMxN A, in fProxyN q, fProxy r, Metric m)
        {
            int count = fProxyQueryOP.countWithinColumnRadius(in A, in q, r, m);
            if (count == 0) return arena.Indices(0);
            var idx = arena.Indices(count);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int written = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                fProxy s = fProxyQueryOP.ColScore(in A, c, in q, m);
                if (sim ? s >= r : s <= r) idx[written++] = c;
            }
            return idx;
        }

        // -------------------------------------------------------------------------
        // kNearestRows / kNearestColumns — arena-alloc Indices + fProxyN scores
        // -------------------------------------------------------------------------

        /// <summary>
        /// Allocates clamped-k Indices + fProxyN from arena, fills via kNearestRows.
        /// Returns idx; scores and count are out params. count = min(k, A.M_Rows).
        /// </summary>
        public static Indices fProxyKNearestRows(this ref Arena arena, in fProxyMxN A, in fProxyN q, int k, Metric m, out fProxyN scores, out int count)
        {
            int clampedK = math.min(k, A.M_Rows);
            if (clampedK <= 0) { scores = A.fProxyVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.fProxyVec(clampedK);
            count = fProxyQueryOP.kNearestRows(in A, in q, k, m, ref idx, ref scores);
            return idx;
        }

        /// <summary>
        /// Allocates clamped-k Indices + fProxyN from arena, fills via kNearestColumns.
        /// </summary>
        public static Indices fProxyKNearestColumns(this ref Arena arena, in fProxyMxN A, in fProxyN q, int k, Metric m, out fProxyN scores, out int count)
        {
            int clampedK = math.min(k, A.N_Cols);
            if (clampedK <= 0) { scores = A.fProxyVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.fProxyVec(clampedK);
            count = fProxyQueryOP.kNearestColumns(in A, in q, k, m, ref idx, ref scores);
            return idx;
        }
    }
}
