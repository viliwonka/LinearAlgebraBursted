using Unity.Mathematics;

namespace LinearAlgebra
{
    // Allocating (arena) wrappers for intQueryOP search operations.
    // Zero-alloc ref-dest primitives (distancesToRow/Column) are in intQueryOP;
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

        /// <summary>
        /// Allocates a fresh intN (length A.M_Rows) from A's arena and fills it
        /// with the distance/similarity from each row of A to query q under metric m.
        /// Supported metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × N_Cols fits the type; otherwise use the float variant.
        /// Manhattan/Chebyshev require element differences to fit the proxy type (e.g. for short, roughly ±16383 so differences fit ±32767). Use float/double for larger ranges.
        /// </para>
        /// </summary>
        public static intN intDistancesToRow(in intMxN A, in intN q, Metric m)
        {
            var dest = A.intVec(A.M_Rows);
            intQueryOP.distancesToRow(in A, in q, m, ref dest);
            return dest;
        }

        /// <summary>
        /// Allocates a fresh intN (length A.N_Cols) from A's arena and fills it
        /// with the distance/similarity from each column of A to query q under metric m.
        /// Supported metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × M_Rows fits the type; otherwise use the float variant.
        /// Manhattan/Chebyshev require element differences to fit the proxy type (e.g. for short, roughly ±16383 so differences fit ±32767). Use float/double for larger ranges.
        /// </para>
        /// </summary>
        public static intN intDistancesToColumn(in intMxN A, in intN q, Metric m)
        {
            var dest = A.intVec(A.N_Cols);
            intQueryOP.distancesToColumn(in A, in q, m, ref dest);
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
            int count = intQueryOP.countNonzero(in x, tol);
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

        /// <summary>
        /// Two-pass: count + exact-alloc Indices of row indices within radius r.
        /// Supported metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): see intDistancesToRow. Manhattan/Chebyshev are overflow-safe.
        /// </para>
        /// </summary>
        public static Indices intRowsWithinRadius(this ref Arena arena, in intMxN A, in intN q, int r, Metric m)
        {
            int count = intQueryOP.countWithinRadius(in A, in q, r, m);
            if (count == 0) return arena.Indices(0);
            var idx = arena.Indices(count);
            bool sim = m == Metric.Dot;
            int written = 0;
            for (int row = 0; row < A.M_Rows; row++)
            {
                int s = intQueryOP.RowScore(in A, row, in q, m);
                if (sim ? s >= r : s <= r) idx[written++] = row;
            }
            return idx;
        }

        /// <summary>
        /// Two-pass: count + exact-alloc Indices of column indices within radius r.
        /// Supported metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): see intDistancesToColumn. Manhattan/Chebyshev are overflow-safe.
        /// </para>
        /// </summary>
        public static Indices intColumnsWithinRadius(this ref Arena arena, in intMxN A, in intN q, int r, Metric m)
        {
            int count = intQueryOP.countWithinColumnRadius(in A, in q, r, m);
            if (count == 0) return arena.Indices(0);
            var idx = arena.Indices(count);
            bool sim = m == Metric.Dot;
            int written = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                int s = intQueryOP.ColScore(in A, c, in q, m);
                if (sim ? s >= r : s <= r) idx[written++] = c;
            }
            return idx;
        }

        // -------------------------------------------------------------------------
        // kNearestRows / kNearestColumns — arena-alloc Indices + intN scores
        // -------------------------------------------------------------------------

        /// <summary>
        /// Allocates clamped-k Indices + intN from arena, fills via kNearestRows.
        /// Returns idx; scores and count are out params. count = min(k, A.M_Rows).
        /// Supported metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × N_Cols fits the type; otherwise use the float variant.
        /// Manhattan/Chebyshev are the recommended integer metrics; element differences must fit the proxy type (e.g. for short, roughly ±16383 so differences fit ±32767).
        /// </para>
        /// </summary>
        public static Indices intKNearestRows(this ref Arena arena, in intMxN A, in intN q, int k, Metric m, out intN scores, out int count)
        {
            int clampedK = math.min(k, A.M_Rows);
            if (clampedK <= 0) { scores = A.intVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.intVec(clampedK);
            count = intQueryOP.kNearestRows(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        /// <summary>
        /// Allocates clamped-k Indices + intN from arena, fills via kNearestColumns.
        /// Supported metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × M_Rows fits the type; otherwise use the float variant.
        /// Manhattan and Chebyshev are the recommended integer metrics (see overflow note on intKNearestRows).
        /// </para>
        /// </summary>
        public static Indices intKNearestColumns(this ref Arena arena, in intMxN A, in intN q, int k, Metric m, out intN scores, out int count)
        {
            int clampedK = math.min(k, A.N_Cols);
            if (clampedK <= 0) { scores = A.intVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.intVec(clampedK);
            count = intQueryOP.kNearestColumns(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        // -------------------------------------------------------------------------
        // kFarthestRows / kFarthestColumns — arena-alloc Indices + intN scores
        // -------------------------------------------------------------------------

        /// <summary>
        /// Allocates clamped-k Indices + intN from arena, fills via kFarthestRows.
        /// Returns idx; scores and count are out params. count = min(k, A.M_Rows).
        /// Supported metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × N_Cols fits the type; otherwise use the float variant.
        /// Manhattan and Chebyshev are the recommended integer metrics (see overflow note on intKNearestRows).
        /// </para>
        /// </summary>
        public static Indices intKFarthestRows(this ref Arena arena, in intMxN A, in intN q, int k, Metric m, out intN scores, out int count)
        {
            int clampedK = math.min(k, A.M_Rows);
            if (clampedK <= 0) { scores = A.intVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.intVec(clampedK);
            count = intQueryOP.kFarthestRows(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }

        /// <summary>
        /// Allocates clamped-k Indices + intN from arena, fills via kFarthestColumns.
        /// Returns idx; scores and count are out params. count = min(k, A.N_Cols).
        /// Supported metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × M_Rows fits the type; otherwise use the float variant.
        /// Manhattan and Chebyshev are the recommended integer metrics (see overflow note on intKNearestRows).
        /// </para>
        /// </summary>
        public static Indices intKFarthestColumns(this ref Arena arena, in intMxN A, in intN q, int k, Metric m, out intN scores, out int count)
        {
            int clampedK = math.min(k, A.N_Cols);
            if (clampedK <= 0) { scores = A.intVec(0, true); count = 0; return arena.Indices(0); }
            var idx = arena.Indices(clampedK);
            scores = A.intVec(clampedK);
            count = intQueryOP.kFarthestColumns(in A, in q, clampedK, m, ref idx, ref scores);
            return idx;
        }
    }
}
