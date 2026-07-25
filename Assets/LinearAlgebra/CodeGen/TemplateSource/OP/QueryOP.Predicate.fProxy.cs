using System.Runtime.CompilerServices;
using Unity.Mathematics;

namespace BULA
{
    // Query.Predicate: predicate-filtered and score-based query operators.
    // Extends Query (partial class). Reuses RowScore/ColScore,
    // IsBetterForNearest, and WorstScoreForNearest from Query.fProxy.cs.
    //
    // Groups:
    //   A — Flat / scalar predicate ops (generic T + P): findFirst, count, any, all, findAll.
    //   B — Row / column filter: countRows, whichRows, countColumns, whichColumns.
    //   C — Masked nearest / k-nearest: nearestRowWhere, kNearestRowsWhere + column twins.
    //   D — Score-based selection: argMaxRowBy, argMinRowBy, topKRowsBy + column twins.
    //
    // Empty-result contract (Group C): when zero rows/columns pass the predicate,
    //   nearestRowWhere / nearestColumnWhere set index = -1 and
    //   score = fProxyQueryCore.WorstScoreForNearest(m) (fProxy.MaxValue for distance metrics,
    //   fProxy.MinValue for similarity metrics). Callers must check index == -1 before use.
    //   kNearestRowsWhere / kNearestColumnsWhere return 0.
    public static partial class Query
    {
        // -------------------------------------------------------------------------
        // GROUP A — FLAT / SCALAR PREDICATE OPS
        // -------------------------------------------------------------------------

        /// <summary>
        /// Returns the flat index of the first element in x where pred.Test(x[i]) is true.
        /// Short-circuits on the first match. Returns -1 if none match.
        /// Empty x (length == 0) returns -1 without throwing.
        /// Generic over fProxyN and fProxyMxN (row-major flat index for matrices).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int findFirst<P>(in fProxyN   x, ref P pred) where P : struct, IfProxyPredicate => fProxyQueryCore.findFirst(in x, ref pred);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int findFirst<P>(in fProxyMxN x, ref P pred) where P : struct, IfProxyPredicate => fProxyQueryCore.findFirst(in x, ref pred);

        /// <summary>
        /// Returns the count of elements in x where pred.Test(x[i]) is true.
        /// Full scan. Empty x returns 0.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int count<P>(in fProxyN   x, ref P pred) where P : struct, IfProxyPredicate => fProxyQueryCore.count(in x, ref pred);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int count<P>(in fProxyMxN x, ref P pred) where P : struct, IfProxyPredicate => fProxyQueryCore.count(in x, ref pred);

        /// <summary>
        /// Returns true if at least one element in x satisfies pred.
        /// Short-circuits on the first true. Empty x returns false.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool any<P>(in fProxyN   x, ref P pred) where P : struct, IfProxyPredicate => fProxyQueryCore.any(in x, ref pred);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool any<P>(in fProxyMxN x, ref P pred) where P : struct, IfProxyPredicate => fProxyQueryCore.any(in x, ref pred);

        /// <summary>
        /// Returns true if every element in x satisfies pred.
        /// Short-circuits on the first false. Empty x returns true (vacuous truth).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool all<P>(in fProxyN   x, ref P pred) where P : struct, IfProxyPredicate => fProxyQueryCore.all(in x, ref pred);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool all<P>(in fProxyMxN x, ref P pred) where P : struct, IfProxyPredicate => fProxyQueryCore.all(in x, ref pred);

        /// <summary>
        /// Fills idx[0..count) with flat indices where pred.Test(x[i]) is true,
        /// in ascending scan order. Returns count.
        /// idx must have length >= x.Data.Length (worst case — all elements match).
        /// Empty x returns 0 with no writes.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int findAll<P>(in fProxyN   x, ref P pred, ref Indices idx) where P : struct, IfProxyPredicate => fProxyQueryCore.findAll(in x, ref pred, ref idx);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int findAll<P>(in fProxyMxN x, ref P pred, ref Indices idx) where P : struct, IfProxyPredicate => fProxyQueryCore.findAll(in x, ref pred, ref idx);

        // -------------------------------------------------------------------------
        // GROUP B — ROW / COLUMN FILTER
        // -------------------------------------------------------------------------

        /// <summary>
        /// Returns the count of rows r in A where pred.Test(in A, r) is true.
        /// Empty matrix (0 rows) returns 0.
        /// </summary>
        public static int countRows<P>(in fProxyMxN A, ref P pred)
            where P : struct, IfProxyRowPredicate
        {
            int c = 0;
            for (int r = 0; r < A.M_Rows; r++)
                if (pred.Test(in A, r)) c++;
            return c;
        }

        /// <summary>
        /// Fills idx[0..count) with row indices where pred.Test(in A, r) is true.
        /// Returns count. idx must have length >= A.M_Rows (worst case).
        /// Empty matrix returns 0.
        /// </summary>
        public static int whichRows<P>(in fProxyMxN A, ref P pred, ref Indices idx)
            where P : struct, IfProxyRowPredicate
        {
            if (idx.N < A.M_Rows)
                throw new System.ArgumentException("Query.whichRows: idx.N must be >= A.M_Rows");
            int c = 0;
            for (int r = 0; r < A.M_Rows; r++)
                if (pred.Test(in A, r)) idx[c++] = r;
            return c;
        }

        /// <summary>
        /// Returns the count of columns c in A where pred.Test(in A, c) is true.
        /// Empty matrix (0 columns) returns 0.
        /// </summary>
        public static int countColumns<P>(in fProxyMxN A, ref P pred)
            where P : struct, IfProxyColPredicate
        {
            int c = 0;
            for (int col = 0; col < A.N_Cols; col++)
                if (pred.Test(in A, col)) c++;
            return c;
        }

        /// <summary>
        /// Fills idx[0..count) with column indices where pred.Test(in A, col) is true.
        /// Returns count. idx must have length >= A.N_Cols (worst case).
        /// Empty matrix returns 0.
        /// </summary>
        public static int whichColumns<P>(in fProxyMxN A, ref P pred, ref Indices idx)
            where P : struct, IfProxyColPredicate
        {
            if (idx.N < A.N_Cols)
                throw new System.ArgumentException("Query.whichColumns: idx.N must be >= A.N_Cols");
            int c = 0;
            for (int col = 0; col < A.N_Cols; col++)
                if (pred.Test(in A, col)) idx[c++] = col;
            return c;
        }

        // -------------------------------------------------------------------------
        // GROUP C — MASKED NEAREST / K-NEAREST
        // -------------------------------------------------------------------------

        /// <summary>
        /// Finds the row of A most similar/closest to query q under metric m,
        /// considering only rows where pred.Test(in A, r) returns true.
        /// Empty result (no row passes pred): index = -1,
        ///   score = fProxy.MaxValue for distance metrics, fProxy.MinValue for similarity.
        /// Callers must check index == -1 before use.
        /// Throws InvalidOperationException if A has 0 rows.
        /// Throws ArgumentException if q.N != A.N_Cols.
        /// </summary>
        public static void nearestRowWhere<P>(in fProxyMxN A, in fProxyN q, Metric m, ref P pred,
            out int index, out fProxy score)
            where P : struct, IfProxyRowPredicate
        {
            if (A.M_Rows == 0)
                throw new System.InvalidOperationException("Query.nearestRowWhere: matrix has no rows");
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("Query.nearestRowWhere: q.N must equal A.N_Cols");

            fProxy best = fProxyQueryCore.WorstScoreForNearest(m);
            int bestIdx = -1;
            for (int r = 0; r < A.M_Rows; r++)
            {
                if (!pred.Test(in A, r)) continue;
                fProxy s = RowScore(in A, r, in q, m);
                // bestIdx==-1 guard: adopt the first passing row even if its score equals the
                // worst-case sentinel, so an all-pass predicate matches unmasked nearestRow exactly.
                if (bestIdx == -1 || fProxyQueryCore.IsBetterForNearest(s, best, m)) { best = s; bestIdx = r; }
            }
            index = bestIdx;
            score = best;
        }

        /// <summary>
        /// Finds the k nearest rows to query q among rows where pred.Test(in A, r) is true.
        /// Fills idx[0..count) and scores[0..count) sorted best-first.
        /// Returns actual count of passing rows found (&lt;= min(k, A.M_Rows)).
        /// Returns 0 if A.M_Rows == 0, k &lt;= 0, or no row passes pred.
        /// idx and scores must have length >= k.
        /// </summary>
        public static int kNearestRowsWhere<P>(in fProxyMxN A, in fProxyN q, int k, Metric m,
            ref P pred, ref Indices idx, ref fProxyN scores)
            where P : struct, IfProxyRowPredicate
        {
            if (A.M_Rows == 0 || k <= 0) return 0;
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("Query.kNearestRowsWhere: q.N must equal A.N_Cols");
            if (idx.N < k)
                throw new System.ArgumentException("Query.kNearestRowsWhere: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("Query.kNearestRowsWhere: scores.N must be >= k");

            int clampedK = math.min(k, A.M_Rows);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;

            for (int r = 0; r < A.M_Rows; r++)
            {
                if (!pred.Test(in A, r)) continue;
                fProxy s = RowScore(in A, r, in q, m);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && (sim ? s > scores[ins - 1] : s < scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = r; count++;
                }
                else
                {
                    fProxy kth = scores[clampedK - 1];
                    if (sim ? s <= kth : s >= kth) continue;
                    int ins = clampedK - 1;
                    while (ins > 0 && (sim ? s > scores[ins - 1] : s < scores[ins - 1])) ins--;
                    for (int j = clampedK - 1; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = r;
                }
            }
            return count;
        }

        /// <summary>
        /// Column analog of <see cref="nearestRowWhere{P}"/> (same empty-result contract).
        /// Throws InvalidOperationException if A has 0 columns.
        /// Throws ArgumentException if q.N != A.M_Rows.
        /// </summary>
        public static void nearestColumnWhere<P>(in fProxyMxN A, in fProxyN q, Metric m, ref P pred,
            out int index, out fProxy score)
            where P : struct, IfProxyColPredicate
        {
            if (A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.nearestColumnWhere: matrix has no columns");
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("Query.nearestColumnWhere: q.N must equal A.M_Rows");

            fProxy best = fProxyQueryCore.WorstScoreForNearest(m);
            int bestIdx = -1;
            for (int c = 0; c < A.N_Cols; c++)
            {
                if (!pred.Test(in A, c)) continue;
                fProxy s = ColScore(in A, c, in q, m);
                // bestIdx==-1 guard: adopt the first passing column even at the sentinel score,
                // so an all-pass predicate matches unmasked nearestColumn exactly.
                if (bestIdx == -1 || fProxyQueryCore.IsBetterForNearest(s, best, m)) { best = s; bestIdx = c; }
            }
            index = bestIdx;
            score = best;
        }

        /// <summary>
        /// Finds the k nearest columns to query q among columns where pred.Test(in A, col) is true.
        /// Fills idx[0..count) and scores[0..count) sorted best-first.
        /// Returns actual count of passing columns found (&lt;= min(k, A.N_Cols)).
        /// Returns 0 if A.N_Cols == 0, k &lt;= 0, or no column passes pred.
        /// idx and scores must have length >= k.
        /// </summary>
        public static int kNearestColumnsWhere<P>(in fProxyMxN A, in fProxyN q, int k, Metric m,
            ref P pred, ref Indices idx, ref fProxyN scores)
            where P : struct, IfProxyColPredicate
        {
            if (A.N_Cols == 0 || k <= 0) return 0;
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("Query.kNearestColumnsWhere: q.N must equal A.M_Rows");
            if (idx.N < k)
                throw new System.ArgumentException("Query.kNearestColumnsWhere: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("Query.kNearestColumnsWhere: scores.N must be >= k");

            int clampedK = math.min(k, A.N_Cols);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;

            for (int c = 0; c < A.N_Cols; c++)
            {
                if (!pred.Test(in A, c)) continue;
                fProxy s = ColScore(in A, c, in q, m);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && (sim ? s > scores[ins - 1] : s < scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = c; count++;
                }
                else
                {
                    fProxy kth = scores[clampedK - 1];
                    if (sim ? s <= kth : s >= kth) continue;
                    int ins = clampedK - 1;
                    while (ins > 0 && (sim ? s > scores[ins - 1] : s < scores[ins - 1])) ins--;
                    for (int j = clampedK - 1; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = c;
                }
            }
            return count;
        }

        // -------------------------------------------------------------------------
        // GROUP D — SCORE-BASED ROW / COLUMN SELECTION
        // -------------------------------------------------------------------------

        /// <summary>
        /// Returns the row of A with the highest scorer.Score(in A, r).
        /// On ties the first occurrence wins. Throws if A has 0 rows.
        /// </summary>
        public static void argMaxRowBy<S>(in fProxyMxN A, ref S scorer, out int index, out fProxy score)
            where S : struct, IfProxyRowScore
        {
            if (A.M_Rows == 0)
                throw new System.InvalidOperationException("Query.argMaxRowBy: matrix has no rows");

            fProxy best = fProxy.MinValue;
            int bestIdx = 0;
            for (int r = 0; r < A.M_Rows; r++)
            {
                fProxy s = scorer.Score(in A, r);
                if (s > best) { best = s; bestIdx = r; }
            }
            index = bestIdx;
            score = best;
        }

        /// <summary>
        /// Returns the row of A with the lowest scorer.Score(in A, r).
        /// On ties the first occurrence wins. Throws if A has 0 rows.
        /// </summary>
        public static void argMinRowBy<S>(in fProxyMxN A, ref S scorer, out int index, out fProxy score)
            where S : struct, IfProxyRowScore
        {
            if (A.M_Rows == 0)
                throw new System.InvalidOperationException("Query.argMinRowBy: matrix has no rows");

            fProxy best = fProxy.MaxValue;
            int bestIdx = 0;
            for (int r = 0; r < A.M_Rows; r++)
            {
                fProxy s = scorer.Score(in A, r);
                if (s < best) { best = s; bestIdx = r; }
            }
            index = bestIdx;
            score = best;
        }

        /// <summary>
        /// Fills idx[0..count) and scores[0..count) with the k rows having the highest
        /// scorer.Score, sorted best-first (descending by score).
        /// Returns min(k, A.M_Rows). idx and scores must have length >= k.
        /// Returns 0 if A.M_Rows == 0 or k &lt;= 0.
        /// </summary>
        public static int topKRowsBy<S>(in fProxyMxN A, ref S scorer, int k,
            ref Indices idx, ref fProxyN scores)
            where S : struct, IfProxyRowScore
        {
            if (A.M_Rows == 0 || k <= 0) return 0;
            if (idx.N < k)
                throw new System.ArgumentException("Query.topKRowsBy: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("Query.topKRowsBy: scores.N must be >= k");

            int clampedK = math.min(k, A.M_Rows);
            int count = 0;

            for (int r = 0; r < A.M_Rows; r++)
            {
                fProxy s = scorer.Score(in A, r);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && s > scores[ins - 1]) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = r; count++;
                }
                else
                {
                    fProxy kth = scores[clampedK - 1];
                    if (s <= kth) continue;
                    int ins = clampedK - 1;
                    while (ins > 0 && s > scores[ins - 1]) ins--;
                    for (int j = clampedK - 1; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = r;
                }
            }
            return clampedK;
        }

        /// <summary>
        /// Returns the column of A with the highest scorer.Score(in A, col).
        /// On ties the first occurrence wins. Throws if A has 0 columns.
        /// </summary>
        public static void argMaxColBy<S>(in fProxyMxN A, ref S scorer, out int index, out fProxy score)
            where S : struct, IfProxyColScore
        {
            if (A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.argMaxColBy: matrix has no columns");

            fProxy best = fProxy.MinValue;
            int bestIdx = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                fProxy s = scorer.Score(in A, c);
                if (s > best) { best = s; bestIdx = c; }
            }
            index = bestIdx;
            score = best;
        }

        /// <summary>
        /// Returns the column of A with the lowest scorer.Score(in A, col).
        /// On ties the first occurrence wins. Throws if A has 0 columns.
        /// </summary>
        public static void argMinColBy<S>(in fProxyMxN A, ref S scorer, out int index, out fProxy score)
            where S : struct, IfProxyColScore
        {
            if (A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.argMinColBy: matrix has no columns");

            fProxy best = fProxy.MaxValue;
            int bestIdx = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                fProxy s = scorer.Score(in A, c);
                if (s < best) { best = s; bestIdx = c; }
            }
            index = bestIdx;
            score = best;
        }

        /// <summary>
        /// Fills idx[0..count) and scores[0..count) with the k columns having the highest
        /// scorer.Score, sorted best-first (descending by score).
        /// Returns min(k, A.N_Cols). idx and scores must have length >= k.
        /// Returns 0 if A.N_Cols == 0 or k &lt;= 0.
        /// </summary>
        public static int topKColsBy<S>(in fProxyMxN A, ref S scorer, int k,
            ref Indices idx, ref fProxyN scores)
            where S : struct, IfProxyColScore
        {
            if (A.N_Cols == 0 || k <= 0) return 0;
            if (idx.N < k)
                throw new System.ArgumentException("Query.topKColsBy: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("Query.topKColsBy: scores.N must be >= k");

            int clampedK = math.min(k, A.N_Cols);
            int count = 0;

            for (int c = 0; c < A.N_Cols; c++)
            {
                fProxy s = scorer.Score(in A, c);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && s > scores[ins - 1]) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = c; count++;
                }
                else
                {
                    fProxy kth = scores[clampedK - 1];
                    if (s <= kth) continue;
                    int ins = clampedK - 1;
                    while (ins > 0 && s > scores[ins - 1]) ins--;
                    for (int j = clampedK - 1; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = c;
                }
            }
            return clampedK;
        }
    }
}
