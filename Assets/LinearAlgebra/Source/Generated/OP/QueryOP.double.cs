#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using Unity.Mathematics;
using Unity.Burst;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{
    // QueryOP: search & selection inside vectors / matrices.
    // Rows are contiguous; columns are strided (stride = N_Cols) — column ops loop with stride.
    //
    // Groups:
    //   1 — Extremes: argMaxAbs / argMinAbs (generic, single-value); decodeIndex helper.
    //       Per-axis rowArgMin/Max / colArgMin/Max with intN index buffers live in
    //       Source/OP/QueryOP.Indices.cs (hand-maintained, cross-type).
    //   2 — Norm-selection: argMaxRowNorm / argMaxColNorm (reuses per-row/col norm loops)
    //   3 — Search over a set of vectors: distancesToRow/Column, nearestRow/Column,
    //         farthestRow/Column, kNearestRows/Columns + kFarthest*, rowsWithinRadius/Column,
    //         countWithinRadius/Column.
    //       Methods that return intN index vectors (kNearestRows etc.) are in QueryOP.Indices.cs.
    //   4 — Value / mask search: findValue, nonzero, countNonzero.
    //       nonzero (ref intN) lives in QueryOP.Indices.cs.
    public static partial class doubleQueryOP
    {
        // -------------------------------------------------------------------------
        // GROUP 1 — EXTREMES
        // -------------------------------------------------------------------------

        /// <summary>
        /// Index (flat) and value of the element with the largest absolute value in x.
        /// Generic over vec + matrix flat data; for matrices the index is row-major flat.
        /// On ties the first occurrence wins. Empty input throws.
        /// </summary>
        public static void argMaxAbs<T>(in T x, out double val, out int flatIndex)
            where T : unmanaged, IUnsafedoubleArray
        {
            if (x.Data.Length == 0)
                throw new System.InvalidOperationException("QueryOP.argMaxAbs: empty input");

            double best = math.abs(x.Data[0]);
            int bestIdx = 0;
            for (int i = 1; i < x.Data.Length; i++)
            {
                double a = math.abs(x.Data[i]);
                if (a > best) { best = a; bestIdx = i; }
            }
            val = best;
            flatIndex = bestIdx;
        }

        /// <summary>
        /// Index (flat) and value of the element with the smallest absolute value in x.
        /// On ties the first occurrence wins. Empty input throws.
        /// </summary>
        public static void argMinAbs<T>(in T x, out double val, out int flatIndex)
            where T : unmanaged, IUnsafedoubleArray
        {
            if (x.Data.Length == 0)
                throw new System.InvalidOperationException("QueryOP.argMinAbs: empty input");

            double best = math.abs(x.Data[0]);
            int bestIdx = 0;
            for (int i = 1; i < x.Data.Length; i++)
            {
                double a = math.abs(x.Data[i]);
                if (a < best) { best = a; bestIdx = i; }
            }
            val = best;
            flatIndex = bestIdx;
        }

        /// <summary>
        /// Converts a row-major flat index to (row, col) for a matrix with nCols columns.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void decodeIndex(int flat, int nCols, out int row, out int col)
        {
            row = flat / nCols;
            col = flat % nCols;
        }

        // -------------------------------------------------------------------------
        // GROUP 2 — NORM-SELECTION
        // (Per-axis row/col argMin/Max with intN buffers → QueryOP.Indices.cs)
        // -------------------------------------------------------------------------

        /// <summary>
        /// Returns the row index whose norm (L1/L2/Linf) is largest.
        /// For L2: compares squared norms to avoid a sqrt per row (argmax is monotone under sqrt).
        /// On ties the first occurrence wins.
        /// </summary>
        public static int argMaxRowNorm(in doubleMxN A, Norm n)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.argMaxRowNorm: empty matrix");

            int bestRow = 0;
            double bestNorm = (double)0;

            if (n == Norm.L1)
            {
                for (int r = 0; r < A.M_Rows; r++)
                {
                    double s = (double)0;
                    for (int c = 0; c < A.N_Cols; c++)
                        s += math.abs(A[r, c]);
                    if (s > bestNorm) { bestNorm = s; bestRow = r; }
                }
            }
            else if (n == Norm.L2)
            {
                // Compare squared norms — argmax is monotone under sqrt so no sqrt needed.
                for (int r = 0; r < A.M_Rows; r++)
                {
                    double s = (double)0;
                    for (int c = 0; c < A.N_Cols; c++)
                        s += A[r, c] * A[r, c];
                    if (s > bestNorm) { bestNorm = s; bestRow = r; }
                }
            }
            else // Norm.Linf
            {
                for (int r = 0; r < A.M_Rows; r++)
                {
                    double s = (double)0;
                    for (int c = 0; c < A.N_Cols; c++)
                        s = math.max(s, math.abs(A[r, c]));
                    if (s > bestNorm) { bestNorm = s; bestRow = r; }
                }
            }
            return bestRow;
        }

        /// <summary>
        /// Returns the column index whose norm (L1/L2/Linf) is largest.
        /// Columns are strided (non-contiguous); on ties the first occurrence wins.
        /// </summary>
        public static int argMaxColNorm(in doubleMxN A, Norm n)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.argMaxColNorm: empty matrix");

            int bestCol = 0;
            double bestNorm = (double)0;

            if (n == Norm.L1)
            {
                for (int c = 0; c < A.N_Cols; c++)
                {
                    double s = (double)0;
                    for (int r = 0; r < A.M_Rows; r++)
                        s += math.abs(A[r, c]);
                    if (s > bestNorm) { bestNorm = s; bestCol = c; }
                }
            }
            else if (n == Norm.L2)
            {
                for (int c = 0; c < A.N_Cols; c++)
                {
                    double s = (double)0;
                    for (int r = 0; r < A.M_Rows; r++)
                        s += A[r, c] * A[r, c];
                    if (s > bestNorm) { bestNorm = s; bestCol = c; }
                }
            }
            else // Norm.Linf
            {
                for (int c = 0; c < A.N_Cols; c++)
                {
                    double s = (double)0;
                    for (int r = 0; r < A.M_Rows; r++)
                        s = math.max(s, math.abs(A[r, c]));
                    if (s > bestNorm) { bestNorm = s; bestCol = c; }
                }
            }
            return bestCol;
        }

        // -------------------------------------------------------------------------
        // GROUP 3 — SEARCH OVER A SET OF VECTORS
        // -------------------------------------------------------------------------

        // ---- Metric score kernels -----------------------------------------------
        // Internal: exposed so ArenaExtensions.Query.double.cs can do two-pass alloc
        // without duplicating metric kernels. Row variant (contiguous); Col variant (strided).

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static double RowScore(in doubleMxN A, int r, in doubleN q, Metric m)
        {
            int nCols = A.N_Cols;
            if (m == Metric.Manhattan)
            {
                double s = (double)0;
                for (int c = 0; c < nCols; c++)
                    s += math.abs(A[r, c] - q[c]);
                return s;
            }
            else if (m == Metric.Euclidean)
            {
                double s = (double)0;
                for (int c = 0; c < nCols; c++) { double d = A[r, c] - q[c]; s += d * d; }
                return math.sqrt(s);
            }
            else if (m == Metric.SqEuclidean)
            {
                double s = (double)0;
                for (int c = 0; c < nCols; c++) { double d = A[r, c] - q[c]; s += d * d; }
                return s;
            }
            else if (m == Metric.Chebyshev)
            {
                double s = (double)0;
                for (int c = 0; c < nCols; c++)
                    s = math.max(s, math.abs(A[r, c] - q[c]));
                return s;
            }
            else if (m == Metric.Cosine)
            {
                double dot = (double)0, normA = (double)0, normQ = (double)0;
                for (int c = 0; c < nCols; c++)
                {
                    dot   += A[r, c] * q[c];
                    normA += A[r, c] * A[r, c];
                    normQ += q[c] * q[c];
                }
                double denom = math.sqrt(normA * normQ);
                return denom > (double)0 ? dot / denom : (double)0;
            }
            else // Metric.Dot
            {
                double s = (double)0;
                for (int c = 0; c < nCols; c++)
                    s += A[r, c] * q[c];
                return s;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static double ColScore(in doubleMxN A, int col, in doubleN q, Metric m)
        {
            int mRows = A.M_Rows;
            if (m == Metric.Manhattan)
            {
                double s = (double)0;
                for (int r = 0; r < mRows; r++)
                    s += math.abs(A[r, col] - q[r]);
                return s;
            }
            else if (m == Metric.Euclidean)
            {
                double s = (double)0;
                for (int r = 0; r < mRows; r++) { double d = A[r, col] - q[r]; s += d * d; }
                return math.sqrt(s);
            }
            else if (m == Metric.SqEuclidean)
            {
                double s = (double)0;
                for (int r = 0; r < mRows; r++) { double d = A[r, col] - q[r]; s += d * d; }
                return s;
            }
            else if (m == Metric.Chebyshev)
            {
                double s = (double)0;
                for (int r = 0; r < mRows; r++)
                    s = math.max(s, math.abs(A[r, col] - q[r]));
                return s;
            }
            else if (m == Metric.Cosine)
            {
                double dot = (double)0, normA = (double)0, normQ = (double)0;
                for (int r = 0; r < mRows; r++)
                {
                    dot   += A[r, col] * q[r];
                    normA += A[r, col] * A[r, col];
                    normQ += q[r] * q[r];
                }
                double denom = math.sqrt(normA * normQ);
                return denom > (double)0 ? dot / denom : (double)0;
            }
            else // Metric.Dot
            {
                double s = (double)0;
                for (int r = 0; r < mRows; r++)
                    s += A[r, col] * q[r];
                return s;
            }
        }

        // Metric direction helpers — hoisted outside per-row loops.
        // Similarity metrics (Cosine, Dot): higher score = nearer.
        // Distance metrics (Manhattan, Euclidean, SqEuclidean, Chebyshev): lower score = nearer.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static bool IsSimilarityMetric(Metric m) => m == Metric.Cosine || m == Metric.Dot;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static double WorstScoreForNearest(Metric m)
            => IsSimilarityMetric(m) ? double.MinValue : double.MaxValue;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static double WorstScoreForFarthest(Metric m)
            => IsSimilarityMetric(m) ? double.MaxValue : double.MinValue;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static bool IsBetterForNearest(double a, double b, Metric m)
            => IsSimilarityMetric(m) ? a > b : a < b;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static bool IsBetterForFarthest(double a, double b, Metric m)
            => IsSimilarityMetric(m) ? a < b : a > b;

        // ---- distancesToRow / distancesToColumn ---------------------------------

        /// <summary>
        /// Fills dest[i] with the distance/similarity between row i of A and query q
        /// under metric m. dest must have length A.M_Rows. q.N must equal A.N_Cols.
        /// </summary>
        public static void distancesToRow(in doubleMxN A, in doubleN q, Metric m, ref doubleN dest)
        {
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.distancesToRow: q.N must equal A.N_Cols");
            if (dest.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.distancesToRow: dest.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
                dest[r] = RowScore(in A, r, in q, m);
        }

        /// <summary>
        /// Fills dest[j] with the distance/similarity between column j of A and query q
        /// under metric m. dest must have length A.N_Cols. q.N must equal A.M_Rows.
        /// Columns are strided (non-contiguous).
        /// </summary>
        public static void distancesToColumn(in doubleMxN A, in doubleN q, Metric m, ref doubleN dest)
        {
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.distancesToColumn: q.N must equal A.M_Rows");
            if (dest.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.distancesToColumn: dest.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
                dest[c] = ColScore(in A, c, in q, m);
        }

        // ---- nearestRow / nearestColumn ----------------------------------------

        /// <summary>
        /// Finds the row of A most similar/closest to query q under metric m.
        /// For distance metrics (Manhattan/Euclidean/SqEuclidean/Chebyshev): nearest = min distance.
        /// For similarity metrics (Cosine/Dot): nearest = max similarity.
        /// score is in metric's own units (SqEuclidean → squared). q.N must equal A.N_Cols.
        /// </summary>
        public static void nearestRow(in doubleMxN A, in doubleN q, Metric m, out int index, out double score)
        {
            if (A.M_Rows == 0)
                throw new System.InvalidOperationException("QueryOP.nearestRow: matrix has no rows");
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.nearestRow: q.N must equal A.N_Cols");

            double best = WorstScoreForNearest(m);
            int bestIdx = 0;
            for (int r = 0; r < A.M_Rows; r++)
            {
                double s = RowScore(in A, r, in q, m);
                if (IsBetterForNearest(s, best, m)) { best = s; bestIdx = r; }
            }
            index = bestIdx;
            score = best;
        }

        /// <summary>
        /// Finds the column of A most similar/closest to query q under metric m.
        /// q.N must equal A.M_Rows. Columns are strided.
        /// </summary>
        public static void nearestColumn(in doubleMxN A, in doubleN q, Metric m, out int index, out double score)
        {
            if (A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.nearestColumn: matrix has no columns");
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.nearestColumn: q.N must equal A.M_Rows");

            double best = WorstScoreForNearest(m);
            int bestIdx = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                double s = ColScore(in A, c, in q, m);
                if (IsBetterForNearest(s, best, m)) { best = s; bestIdx = c; }
            }
            index = bestIdx;
            score = best;
        }

        // ---- farthestRow / farthestColumn ---------------------------------------

        /// <summary>
        /// Finds the row of A most dissimilar/farthest from query q under metric m.
        /// For distance metrics: farthest = max distance.
        /// For similarity metrics: farthest = min similarity.
        /// score is in metric's own units.
        /// </summary>
        public static void farthestRow(in doubleMxN A, in doubleN q, Metric m, out int index, out double score)
        {
            if (A.M_Rows == 0)
                throw new System.InvalidOperationException("QueryOP.farthestRow: matrix has no rows");
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.farthestRow: q.N must equal A.N_Cols");

            double worst = WorstScoreForFarthest(m);
            int worstIdx = 0;
            for (int r = 0; r < A.M_Rows; r++)
            {
                double s = RowScore(in A, r, in q, m);
                if (IsBetterForFarthest(s, worst, m)) { worst = s; worstIdx = r; }
            }
            index = worstIdx;
            score = worst;
        }

        /// <summary>
        /// Finds the column of A most dissimilar/farthest from query q under metric m.
        /// q.N must equal A.M_Rows. Columns are strided.
        /// </summary>
        public static void farthestColumn(in doubleMxN A, in doubleN q, Metric m, out int index, out double score)
        {
            if (A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.farthestColumn: matrix has no columns");
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.farthestColumn: q.N must equal A.M_Rows");

            double worst = WorstScoreForFarthest(m);
            int worstIdx = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                double s = ColScore(in A, c, in q, m);
                if (IsBetterForFarthest(s, worst, m)) { worst = s; worstIdx = c; }
            }
            index = worstIdx;
            score = worst;
        }

        // ---- countWithinRadius / countWithinColumnRadius (zero-alloc count) -----

        /// <summary>
        /// Returns the count of rows with distance/similarity to q within radius r.
        /// For distance metrics: count rows with score &lt;= r.
        /// For similarity metrics: count rows with score >= r.
        /// q.N must equal A.N_Cols.
        /// </summary>
        public static int countWithinRadius(in doubleMxN A, in doubleN q, double r, Metric m)
        {
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.countWithinRadius: q.N must equal A.N_Cols");

            bool sim = IsSimilarityMetric(m);
            int count = 0;
            for (int row = 0; row < A.M_Rows; row++)
            {
                double s = RowScore(in A, row, in q, m);
                if (sim ? s >= r : s <= r) count++;
            }
            return count;
        }

        /// <summary>
        /// Returns the count of columns with distance/similarity to q within radius r.
        /// Columns are strided. q.N must equal A.M_Rows.
        /// </summary>
        public static int countWithinColumnRadius(in doubleMxN A, in doubleN q, double r, Metric m)
        {
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.countWithinColumnRadius: q.N must equal A.M_Rows");

            bool sim = IsSimilarityMetric(m);
            int count = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                double s = ColScore(in A, c, in q, m);
                if (sim ? s >= r : s <= r) count++;
            }
            return count;
        }

        // -------------------------------------------------------------------------
        // GROUP 4 — VALUE / MASK SEARCH
        // -------------------------------------------------------------------------

        /// <summary>
        /// Returns the flat index of the first element in x equal to target
        /// (within tolerance: |x[i] - target| &lt;= tol). Returns -1 if not found.
        /// Generic over vec + matrix flat data. (Like Excel MATCH.)
        /// </summary>
        public static int findValue<T>(in T x, double target, double tol)
            where T : unmanaged, IUnsafedoubleArray
        {
            for (int i = 0; i < x.Data.Length; i++)
            {
                if (math.abs(x.Data[i] - target) <= tol)
                    return i;
            }
            return -1;
        }

        /// <summary>
        /// Returns the count of elements in x with absolute value &gt; tol.
        /// Zero-alloc; use with nonzero (QueryOP.Indices) for the full index list.
        /// </summary>
        public static int countNonzero<T>(in T x, double tol)
            where T : unmanaged, IUnsafedoubleArray
        {
            int count = 0;
            for (int i = 0; i < x.Data.Length; i++)
            {
                if (math.abs(x.Data[i]) > tol)
                    count++;
            }
            return count;
        }
    }
}
