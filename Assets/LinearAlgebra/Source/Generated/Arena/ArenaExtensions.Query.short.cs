using Unity.Mathematics;

namespace LinearAlgebra
{
    // Allocating (arena) wrappers for shortQuery_OP search operations.
    // Zero-alloc ref-dest primitives (distancesToRow/Column) are in shortQuery_OP;
    // these wrappers do count-pass + exact-alloc so callers don't size buffers manually.
    //
    // All Indices buffers use the shared Indices type (arena.Indices(n)) — assembly-shared,
    // no duplication. Score buffers are shortN allocated from the arena.
    //
    // Supported integer-exact metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
    // Euclidean and Cosine throw ArgumentException (float-only).
    // Overflow note: element-wise differences must fit the proxy type (for short: ±16383
    // coordinates so differences fit ±32767). SqEuclidean/Dot: maxAbs²×dim must fit.
    // Use float/double for larger ranges.
    public static partial class ArenaExtensions
    {
        // -------------------------------------------------------------------------
        // distancesToRow / distancesToColumn — allocate shortN from A's arena
        // -------------------------------------------------------------------------

        /// <summary>Allocates a fresh shortN (length A.M_Rows) from A's arena with row distances/similarities to q under metric m. See class doc for supported metrics and overflow limits.</summary>
        public static shortN shortDistancesToRow(in shortMxN A, in shortN q, Metric m)
        {
            var dest = A.shortVec(A.M_Rows);
            shortQuery_OP.distancesToRow(in A, in q, m, ref dest);
            return dest;
        }

        /// <summary>Allocates a fresh shortN (length A.N_Cols) from A's arena with column distances/similarities to q under metric m. See class doc for supported metrics and overflow limits.</summary>
        public static shortN shortDistancesToColumn(in shortMxN A, in shortN q, Metric m)
        {
            var dest = A.shortVec(A.N_Cols);
            shortQuery_OP.distancesToColumn(in A, in q, m, ref dest);
            return dest;
        }

        // -------------------------------------------------------------------------
        // nonzeroIndices — count-pass + exact-alloc Indices
        // -------------------------------------------------------------------------

        /// <summary>
        /// Two-pass: count nonzero elements, allocate exact-sized Indices, fill indices.
        /// Returns the allocated Indices (length = count). Arena-owned.
        /// </summary>
        public static Indices shortNonzeroIndices<T>(this ref Arena arena, in T x, short tol)
            where T : unmanaged, IUnsafeshortArray
        {
            int count = shortQuery_OP.countNonzero(in x, tol);
            if (count == 0) return arena.Indices(0);
            var idx = arena.Indices(count);
            int written = 0;
            for (int i = 0; i < x.Data.Length; i++)
            {
                short v = x.Data[i];
                short av = v < (short)0 ? (v == short.MinValue ? short.MaxValue : (short)(-v)) : v;
                if (av > tol) idx[written++] = i;
            }
            return idx;
        }

        // -------------------------------------------------------------------------
        // rowsWithinRadius / columnsWithinRadius — count-pass + exact-alloc Indices
        // -------------------------------------------------------------------------

        /// <summary>Two-pass: counts, then exact-allocates Indices of row indices within radius r under metric m.</summary>
        public static Indices shortRowsWithinRadius(this ref Arena arena, in shortMxN A, in shortN q, short r, Metric m)
        {
            int count = shortQuery_OP.countWithinRadius(in A, in q, r, m);
            if (count == 0) return arena.Indices(0);
            var idx = arena.Indices(count);
            bool sim = m == Metric.Dot;
            int written = 0;
            for (int row = 0; row < A.M_Rows; row++)
            {
                short s = shortQuery_OP.RowScore(in A, row, in q, m);
                if (sim ? s >= r : s <= r) idx[written++] = row;
            }
            return idx;
        }

        /// <summary>Two-pass: counts, then exact-allocates Indices of column indices within radius r under metric m.</summary>
        public static Indices shortColumnsWithinRadius(this ref Arena arena, in shortMxN A, in shortN q, short r, Metric m)
        {
            int count = shortQuery_OP.countWithinColumnRadius(in A, in q, r, m);
            if (count == 0) return arena.Indices(0);
            var idx = arena.Indices(count);
            bool sim = m == Metric.Dot;
            int written = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                short s = shortQuery_OP.ColScore(in A, c, in q, m);
                if (sim ? s >= r : s <= r) idx[written++] = c;
            }
            return idx;
        }

        // -------------------------------------------------------------------------
        // kNearestRows / kNearestColumns — arena-alloc Indices + shortN scores
        // -------------------------------------------------------------------------

        /// <summary>Allocates clamped-k Indices + shortN scores from arena, filled via kNearestRows; count = min(k, A.M_Rows).</summary>
        public static Indices shortKNearestRows(this ref Arena arena, in shortMxN A, in shortN q, int k, Metric m, out shortN scores, out int count)
        {
            int clampedK = math.min(k, A.M_Rows);
            if (clampedK <= 0) { scores = A.shortVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.shortVec(clampedK);
            count = shortQuery_OP.kNearestRows(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        /// <summary>Allocates clamped-k Indices + shortN scores from arena, filled via kNearestColumns; count = min(k, A.N_Cols).</summary>
        public static Indices shortKNearestColumns(this ref Arena arena, in shortMxN A, in shortN q, int k, Metric m, out shortN scores, out int count)
        {
            int clampedK = math.min(k, A.N_Cols);
            if (clampedK <= 0) { scores = A.shortVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.shortVec(clampedK);
            count = shortQuery_OP.kNearestColumns(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        // -------------------------------------------------------------------------
        // kFarthestRows / kFarthestColumns — arena-alloc Indices + shortN scores
        // -------------------------------------------------------------------------

        /// <summary>Allocates clamped-k Indices + shortN scores from arena, filled via kFarthestRows; count = min(k, A.M_Rows).</summary>
        public static Indices shortKFarthestRows(this ref Arena arena, in shortMxN A, in shortN q, int k, Metric m, out shortN scores, out int count)
        {
            int clampedK = math.min(k, A.M_Rows);
            if (clampedK <= 0) { scores = A.shortVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.shortVec(clampedK);
            count = shortQuery_OP.kFarthestRows(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        /// <summary>Allocates clamped-k Indices + shortN scores from arena, filled via kFarthestColumns; count = min(k, A.N_Cols).</summary>
        public static Indices shortKFarthestColumns(this ref Arena arena, in shortMxN A, in shortN q, int k, Metric m, out shortN scores, out int count)
        {
            int clampedK = math.min(k, A.N_Cols);
            if (clampedK <= 0) { scores = A.shortVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.shortVec(clampedK);
            count = shortQuery_OP.kFarthestColumns(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }
    }
}
