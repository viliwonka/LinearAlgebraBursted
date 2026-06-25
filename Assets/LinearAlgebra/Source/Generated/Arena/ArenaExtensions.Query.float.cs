using Unity.Mathematics;

namespace LinearAlgebra
{
    // Allocating (arena) wrappers for floatQueryOP search operations.
    // Zero-alloc ref-dest primitives (distancesToRow/Column) are in QueryOP;
    // these wrappers allocate from the matrix's own arena so callers don't
    // have to size buffers manually for the float-typed outputs.
    //
    // Indices-returning wrappers (nonzero, rowsWithinRadius, kNearestRows, etc.)
    // are in this file using Indices as the shared buffer type, generated via
    // the float template into floatXxx / doubleXxx variants.
    public static partial class ArenaExtensions
    {
        // -------------------------------------------------------------------------
        // distancesToRow / distancesToColumn — allocate floatN from A's arena
        // -------------------------------------------------------------------------

        /// <summary>
        /// Allocates a fresh floatN (length A.M_Rows) from A's arena and fills it
        /// with the distance/similarity from each row of A to query q under metric m.
        /// </summary>
        public static floatN floatDistancesToRow(in floatMxN A, in floatN q, Metric m)
        {
            var dest = A.floatVec(A.M_Rows);
            floatQueryOP.distancesToRow(in A, in q, m, ref dest);
            return dest;
        }

        /// <summary>
        /// Allocates a fresh floatN (length A.N_Cols) from A's arena and fills it
        /// with the distance/similarity from each column of A to query q under metric m.
        /// </summary>
        public static floatN floatDistancesToColumn(in floatMxN A, in floatN q, Metric m)
        {
            var dest = A.floatVec(A.N_Cols);
            floatQueryOP.distancesToColumn(in A, in q, m, ref dest);
            return dest;
        }

        // -------------------------------------------------------------------------
        // nonzeroIndices — count-pass + exact-alloc Indices
        // -------------------------------------------------------------------------

        /// <summary>
        /// Two-pass: count nonzero elements, allocate exact-sized Indices, fill indices.
        /// Returns the allocated Indices (length = count). Arena-owned.
        /// </summary>
        public static Indices floatNonzeroIndices<T>(this ref Arena arena, in T x, float tol)
            where T : unmanaged, IUnsafefloatArray
        {
            int count = floatQueryOP.countNonzero(in x, tol);
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
        public static Indices floatRowsWithinRadius(this ref Arena arena, in floatMxN A, in floatN q, float r, Metric m)
        {
            int count = floatQueryOP.countWithinRadius(in A, in q, r, m);
            if (count == 0) return arena.Indices(0);
            var idx = arena.Indices(count);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int written = 0;
            for (int row = 0; row < A.M_Rows; row++)
            {
                float s = floatQueryOP.RowScore(in A, row, in q, m);
                if (sim ? s >= r : s <= r) idx[written++] = row;
            }
            return idx;
        }

        /// <summary>
        /// Two-pass: count + exact-alloc Indices of column indices within radius r.
        /// </summary>
        public static Indices floatColumnsWithinRadius(this ref Arena arena, in floatMxN A, in floatN q, float r, Metric m)
        {
            int count = floatQueryOP.countWithinColumnRadius(in A, in q, r, m);
            if (count == 0) return arena.Indices(0);
            var idx = arena.Indices(count);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int written = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                float s = floatQueryOP.ColScore(in A, c, in q, m);
                if (sim ? s >= r : s <= r) idx[written++] = c;
            }
            return idx;
        }

        // -------------------------------------------------------------------------
        // kNearestRows / kNearestColumns — arena-alloc Indices + floatN scores
        // -------------------------------------------------------------------------

        /// <summary>
        /// Allocates clamped-k Indices + floatN from arena, fills via kNearestRows.
        /// Returns idx; scores and count are out params. count = min(k, A.M_Rows).
        /// </summary>
        public static Indices floatKNearestRows(this ref Arena arena, in floatMxN A, in floatN q, int k, Metric m, out floatN scores, out int count)
        {
            int clampedK = math.min(k, A.M_Rows);
            if (clampedK <= 0) { scores = A.floatVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.floatVec(clampedK);
            count = floatQueryOP.kNearestRows(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        /// <summary>
        /// Allocates clamped-k Indices + floatN from arena, fills via kNearestColumns.
        /// </summary>
        public static Indices floatKNearestColumns(this ref Arena arena, in floatMxN A, in floatN q, int k, Metric m, out floatN scores, out int count)
        {
            int clampedK = math.min(k, A.N_Cols);
            if (clampedK <= 0) { scores = A.floatVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.floatVec(clampedK);
            count = floatQueryOP.kNearestColumns(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        // -------------------------------------------------------------------------
        // kFarthestRows / kFarthestColumns — arena-alloc Indices + floatN scores
        // -------------------------------------------------------------------------

        /// <summary>
        /// Allocates clamped-k Indices + floatN from arena, fills via kFarthestRows.
        /// Returns idx; scores and count are out params. count = min(k, A.M_Rows).
        /// </summary>
        public static Indices floatKFarthestRows(this ref Arena arena, in floatMxN A, in floatN q, int k, Metric m, out floatN scores, out int count)
        {
            int clampedK = math.min(k, A.M_Rows);
            if (clampedK <= 0) { scores = A.floatVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.floatVec(clampedK);
            count = floatQueryOP.kFarthestRows(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        /// <summary>
        /// Allocates clamped-k Indices + floatN from arena, fills via kFarthestColumns.
        /// Returns idx; scores and count are out params. count = min(k, A.N_Cols).
        /// </summary>
        public static Indices floatKFarthestColumns(this ref Arena arena, in floatMxN A, in floatN q, int k, Metric m, out floatN scores, out int count)
        {
            int clampedK = math.min(k, A.N_Cols);
            if (clampedK <= 0) { scores = A.floatVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.floatVec(clampedK);
            count = floatQueryOP.kFarthestColumns(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }
    }
}
