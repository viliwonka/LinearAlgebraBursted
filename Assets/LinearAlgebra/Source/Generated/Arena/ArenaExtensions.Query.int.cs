using Unity.Mathematics;

namespace LinearAlgebra
{
    // Allocating (arena) wrappers for Query search operations.
    // Zero-alloc ref-dest primitives (distancesToRow/Column) are in Query;
    // these wrappers do count-pass + exact-alloc so callers don't size buffers manually.
    //
    // All Indices buffers use the shared Indices type (arena.Indices(n)) — assembly-shared,
    // no duplication. Score buffers are intN allocated from the arena.
    //
    // Supported integer-exact metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
    // Euclidean and Cosine throw ArgumentException (float-only).
    // Overflow note: element-wise differences must fit the proxy type (for short: ±16383
    // coordinates so differences fit ±32767). SqEuclidean/Dot: maxAbs²×dim must fit.
    // Use float/double for larger ranges.
    public static partial class ArenaExtensions
    {
        // -------------------------------------------------------------------------
        // distancesToRow / distancesToColumn — allocate intN from A's arena
        // -------------------------------------------------------------------------

        /// <summary>Allocates a fresh intN (length A.M_Rows) from A's arena with row distances/similarities to q under metric m. See class doc for supported metrics and overflow limits.</summary>
        public static intN intDistancesToRow(in intMxN A, in intN q, Metric m)
        {
            var dest = A.intVec(A.M_Rows);
            Query.distancesToRow(in A, in q, m, ref dest);
            return dest;
        }

        /// <summary>Allocates a fresh intN (length A.N_Cols) from A's arena with column distances/similarities to q under metric m. See class doc for supported metrics and overflow limits.</summary>
        public static intN intDistancesToColumn(in intMxN A, in intN q, Metric m)
        {
            var dest = A.intVec(A.N_Cols);
            Query.distancesToColumn(in A, in q, m, ref dest);
            return dest;
        }

        // -------------------------------------------------------------------------
        // nonzeroIndices — count-pass + exact-alloc Indices
        // -------------------------------------------------------------------------

        /// <summary>
        /// Two-pass: count nonzero elements, allocate exact-sized Indices, fill indices.
        /// Returns the allocated Indices (length = count). Arena-owned.
        /// </summary>
        public static Indices intNonzeroIndices<T>(this ref Arena arena, in T x, int tol)
            where T : unmanaged, IUnsafeintArray
        {
            int count = Query.countNonzero(in x, tol);
            if (count == 0) return arena.Indices(0);
            var idx = arena.Indices(count);
            int written = 0;
            for (int i = 0; i < x.Data.Length; i++)
            {
                int v = x.Data[i];
                int av = v < (int)0 ? (v == int.MinValue ? int.MaxValue : (int)(-v)) : v;
                if (av > tol) idx[written++] = i;
            }
            return idx;
        }

        // -------------------------------------------------------------------------
        // rowsWithinRadius / columnsWithinRadius — count-pass + exact-alloc Indices
        // -------------------------------------------------------------------------

        /// <summary>Two-pass: counts, then exact-allocates Indices of row indices within radius r under metric m.</summary>
        public static Indices intRowsWithinRadius(this ref Arena arena, in intMxN A, in intN q, int r, Metric m)
        {
            int count = Query.countWithinRadius(in A, in q, r, m);
            if (count == 0) return arena.Indices(0);
            var idx = arena.Indices(count);
            bool sim = m == Metric.Dot;
            int written = 0;
            for (int row = 0; row < A.M_Rows; row++)
            {
                int s = Query.RowScore(in A, row, in q, m);
                if (sim ? s >= r : s <= r) idx[written++] = row;
            }
            return idx;
        }

        /// <summary>Two-pass: counts, then exact-allocates Indices of column indices within radius r under metric m.</summary>
        public static Indices intColumnsWithinRadius(this ref Arena arena, in intMxN A, in intN q, int r, Metric m)
        {
            int count = Query.countWithinColumnRadius(in A, in q, r, m);
            if (count == 0) return arena.Indices(0);
            var idx = arena.Indices(count);
            bool sim = m == Metric.Dot;
            int written = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                int s = Query.ColScore(in A, c, in q, m);
                if (sim ? s >= r : s <= r) idx[written++] = c;
            }
            return idx;
        }

        // -------------------------------------------------------------------------
        // kNearestRows / kNearestColumns — arena-alloc Indices + intN scores
        // -------------------------------------------------------------------------

        /// <summary>Allocates clamped-k Indices + intN scores from arena, filled via kNearestRows; count = min(k, A.M_Rows).</summary>
        public static Indices intKNearestRows(this ref Arena arena, in intMxN A, in intN q, int k, Metric m, out intN scores, out int count)
        {
            int clampedK = math.min(k, A.M_Rows);
            if (clampedK <= 0) { scores = A.intVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.intVec(clampedK);
            count = Query.kNearestRows(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        /// <summary>Allocates clamped-k Indices + intN scores from arena, filled via kNearestColumns; count = min(k, A.N_Cols).</summary>
        public static Indices intKNearestColumns(this ref Arena arena, in intMxN A, in intN q, int k, Metric m, out intN scores, out int count)
        {
            int clampedK = math.min(k, A.N_Cols);
            if (clampedK <= 0) { scores = A.intVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.intVec(clampedK);
            count = Query.kNearestColumns(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        // -------------------------------------------------------------------------
        // kFarthestRows / kFarthestColumns — arena-alloc Indices + intN scores
        // -------------------------------------------------------------------------

        /// <summary>Allocates clamped-k Indices + intN scores from arena, filled via kFarthestRows; count = min(k, A.M_Rows).</summary>
        public static Indices intKFarthestRows(this ref Arena arena, in intMxN A, in intN q, int k, Metric m, out intN scores, out int count)
        {
            int clampedK = math.min(k, A.M_Rows);
            if (clampedK <= 0) { scores = A.intVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.intVec(clampedK);
            count = Query.kFarthestRows(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        /// <summary>Allocates clamped-k Indices + intN scores from arena, filled via kFarthestColumns; count = min(k, A.N_Cols).</summary>
        public static Indices intKFarthestColumns(this ref Arena arena, in intMxN A, in intN q, int k, Metric m, out intN scores, out int count)
        {
            int clampedK = math.min(k, A.N_Cols);
            if (clampedK <= 0) { scores = A.intVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.intVec(clampedK);
            count = Query.kFarthestColumns(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }
    }
}
