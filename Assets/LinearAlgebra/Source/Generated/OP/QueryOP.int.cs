#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using Unity.Mathematics;
using Unity.Burst;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{
    // Query: integer-exact search & selection inside integer vectors / matrices.
    // Only the metrics/norms that are exact for integer types are implemented: Manhattan,
    // Chebyshev, SqEuclidean, Dot (Group 3); L1 and Linf norms (Group 2). Euclidean, Cosine, and
    // L2 are float-only (need sqrt/division) and throw ArgumentException if passed to integer
    // methods.
    //
    // decodeIndex is type-agnostic (int→int) and lives in Query — reuse that,
    // do NOT call or duplicate it here.
    //
    // Overflow note: ALL integer metrics require each element AND each element-wise
    // difference to fit the proxy type. For short: coordinates roughly within ±16383 so
    // differences fit ±32767 — the subtraction A[r,c]-q[c] itself overflows at the boundary.
    // SqEuclidean/Dot additionally require maxAbs²×dimension to fit. Values at MinValue are
    // mapped to MaxValue in abs (off-by-one, correct ordering). Use float/double for larger ranges.
    //
    // Groups implemented:
    //   1 — Extremes: argMaxAbs / argMinAbs (generic); rowArgMin/Max / colArgMin/Max
    //   2 — Norm-selection: argMaxRowNorm / argMaxColNorm (L1 and Linf only; L2 throws)
    //   3 — Search over a set of vectors: distancesToRow/Column, nearestRow/Column,
    //         farthestRow/Column, kNearestRows/Columns + kFarthest*,
    //         rowsWithinRadius/columnsWithinRadius, countWithinRadius/countWithinColumnRadius.
    //       Supported metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
    //       Euclidean and Cosine throw ArgumentException.
    //   4 — Value / mask search: findValue, nonzero, countNonzero.
    public static partial class Query
    {
        // -------------------------------------------------------------------------
        // HELPERS
        // -------------------------------------------------------------------------

        // Saturating absolute value: maps MinValue → MaxValue (off-by-one in magnitude
        // at the extreme, but correct sign/ordering and nonzero classification).
        // Use instead of the raw negation pattern `v < 0 ? -v : v` throughout this file.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static int iAbs(int v) => v < (int)0 ? (v == int.MinValue ? int.MaxValue : (int)(-v)) : v;

        // -------------------------------------------------------------------------
        // GROUP 1 — EXTREMES
        // -------------------------------------------------------------------------

        /// <summary>
        /// Index (flat) and value of the element with the largest absolute value in x.
        /// Generic over vec + matrix flat data; for matrices the index is row-major flat.
        /// On ties the first occurrence wins. Empty input throws.
        /// </summary>
        public static void argMaxAbs<T>(in T x, out int val, out int flatIndex)
            where T : unmanaged, IUnsafeintArray
        {
            if (x.Data.Length == 0)
                throw new System.InvalidOperationException("Query.argMaxAbs: empty input");

            int best = iAbs(x.Data[0]);
            int bestIdx = 0;
            for (int i = 1; i < x.Data.Length; i++)
            {
                int a = iAbs(x.Data[i]);
                if (a > best) { best = a; bestIdx = i; }
            }
            val = best;
            flatIndex = bestIdx;
        }

        /// <summary>
        /// Index (flat) and value of the element with the smallest absolute value in x.
        /// On ties the first occurrence wins. Empty input throws.
        /// </summary>
        public static void argMinAbs<T>(in T x, out int val, out int flatIndex)
            where T : unmanaged, IUnsafeintArray
        {
            if (x.Data.Length == 0)
                throw new System.InvalidOperationException("Query.argMinAbs: empty input");

            int best = iAbs(x.Data[0]);
            int bestIdx = 0;
            for (int i = 1; i < x.Data.Length; i++)
            {
                int a = iAbs(x.Data[i]);
                if (a < best) { best = a; bestIdx = i; }
            }
            val = best;
            flatIndex = bestIdx;
        }

        // ---- Per-axis row/col arg-min/max with Indices buffer ---

        /// <summary>
        /// For each row i of A, writes the column index of the minimum element into
        /// colIndexPerRow[i] and the minimum value into valPerRow[i].
        /// Returns A.M_Rows. Both buffers must have length A.M_Rows.
        /// </summary>
        public static int rowArgMin(in intMxN A, ref Indices colIndexPerRow, ref intN valPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.rowArgMin: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("Query.rowArgMin: colIndexPerRow.N must equal A.M_Rows");
            if (valPerRow.N != A.M_Rows)
                throw new System.ArgumentException("Query.rowArgMin: valPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                int best = A[r, 0];
                int bestC = 0;
                for (int c = 1; c < A.N_Cols; c++)
                    if (A[r, c] < best) { best = A[r, c]; bestC = c; }
                colIndexPerRow[r] = bestC;
                valPerRow[r] = best;
            }
            return A.M_Rows;
        }

        /// <summary>
        /// Index-only form of rowArgMin (no value output).
        /// Returns A.M_Rows. colIndexPerRow must have length A.M_Rows.
        /// </summary>
        public static int rowArgMin(in intMxN A, ref Indices colIndexPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.rowArgMin: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("Query.rowArgMin: colIndexPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                int best = A[r, 0];
                int bestC = 0;
                for (int c = 1; c < A.N_Cols; c++)
                    if (A[r, c] < best) { best = A[r, c]; bestC = c; }
                colIndexPerRow[r] = bestC;
            }
            return A.M_Rows;
        }

        /// <summary>
        /// For each row i of A, writes the column index of the maximum element into
        /// colIndexPerRow[i] and the maximum value into valPerRow[i].
        /// Returns A.M_Rows. Both buffers must have length A.M_Rows.
        /// </summary>
        public static int rowArgMax(in intMxN A, ref Indices colIndexPerRow, ref intN valPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.rowArgMax: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("Query.rowArgMax: colIndexPerRow.N must equal A.M_Rows");
            if (valPerRow.N != A.M_Rows)
                throw new System.ArgumentException("Query.rowArgMax: valPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                int best = A[r, 0];
                int bestC = 0;
                for (int c = 1; c < A.N_Cols; c++)
                    if (A[r, c] > best) { best = A[r, c]; bestC = c; }
                colIndexPerRow[r] = bestC;
                valPerRow[r] = best;
            }
            return A.M_Rows;
        }

        /// <summary>
        /// Index-only form of rowArgMax.
        /// Returns A.M_Rows. colIndexPerRow must have length A.M_Rows.
        /// </summary>
        public static int rowArgMax(in intMxN A, ref Indices colIndexPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.rowArgMax: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("Query.rowArgMax: colIndexPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                int best = A[r, 0];
                int bestC = 0;
                for (int c = 1; c < A.N_Cols; c++)
                    if (A[r, c] > best) { best = A[r, c]; bestC = c; }
                colIndexPerRow[r] = bestC;
            }
            return A.M_Rows;
        }

        /// <summary>
        /// For each column j of A, writes the row index of the minimum element into
        /// rowIndexPerCol[j] and the minimum value into valPerCol[j].
        /// Returns A.N_Cols. Columns are strided (non-contiguous).
        /// </summary>
        public static int colArgMin(in intMxN A, ref Indices rowIndexPerCol, ref intN valPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.colArgMin: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("Query.colArgMin: rowIndexPerCol.N must equal A.N_Cols");
            if (valPerCol.N != A.N_Cols)
                throw new System.ArgumentException("Query.colArgMin: valPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                int best = A[0, c];
                int bestR = 0;
                for (int r = 1; r < A.M_Rows; r++)
                    if (A[r, c] < best) { best = A[r, c]; bestR = r; }
                rowIndexPerCol[c] = bestR;
                valPerCol[c] = best;
            }
            return A.N_Cols;
        }

        /// <summary>Index-only form of colArgMin. Returns A.N_Cols.</summary>
        public static int colArgMin(in intMxN A, ref Indices rowIndexPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.colArgMin: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("Query.colArgMin: rowIndexPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                int best = A[0, c];
                int bestR = 0;
                for (int r = 1; r < A.M_Rows; r++)
                    if (A[r, c] < best) { best = A[r, c]; bestR = r; }
                rowIndexPerCol[c] = bestR;
            }
            return A.N_Cols;
        }

        /// <summary>
        /// For each column j of A, writes the row index of the maximum element into
        /// rowIndexPerCol[j] and the maximum value into valPerCol[j].
        /// Returns A.N_Cols. Columns are strided (non-contiguous).
        /// </summary>
        public static int colArgMax(in intMxN A, ref Indices rowIndexPerCol, ref intN valPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.colArgMax: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("Query.colArgMax: rowIndexPerCol.N must equal A.N_Cols");
            if (valPerCol.N != A.N_Cols)
                throw new System.ArgumentException("Query.colArgMax: valPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                int best = A[0, c];
                int bestR = 0;
                for (int r = 1; r < A.M_Rows; r++)
                    if (A[r, c] > best) { best = A[r, c]; bestR = r; }
                rowIndexPerCol[c] = bestR;
                valPerCol[c] = best;
            }
            return A.N_Cols;
        }

        /// <summary>Index-only form of colArgMax. Returns A.N_Cols.</summary>
        public static int colArgMax(in intMxN A, ref Indices rowIndexPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.colArgMax: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("Query.colArgMax: rowIndexPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                int best = A[0, c];
                int bestR = 0;
                for (int r = 1; r < A.M_Rows; r++)
                    if (A[r, c] > best) { best = A[r, c]; bestR = r; }
                rowIndexPerCol[c] = bestR;
            }
            return A.N_Cols;
        }

        // -------------------------------------------------------------------------
        // GROUP 2 — NORM-SELECTION (L1 and Linf only)
        // -------------------------------------------------------------------------

        /// <summary>
        /// Returns the row index whose norm (L1 or Linf) is largest.
        /// L2 norm is not supported for integer types (requires sqrt) — throws ArgumentException.
        /// On ties the first occurrence wins.
        /// <para>
        /// Overflow note: L1 accumulates |element| values; Linf takes max |element|.
        /// Both are overflow-safe for typical integer ranges.
        /// </para>
        /// </summary>
        public static int argMaxRowNorm(in intMxN A, Norm n)
        {
            if (n == Norm.L2)
                throw new System.ArgumentException("Query.argMaxRowNorm: L2 norm-selection is float-only for integer types");
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.argMaxRowNorm: empty matrix");

            int bestRow = 0;
            int bestNorm = (int)0;

            if (n == Norm.L1)
            {
                for (int r = 0; r < A.M_Rows; r++)
                {
                    int s = (int)0;
                    for (int c = 0; c < A.N_Cols; c++)
                    {
                        int v = A[r, c];
                        s = (int)(s + iAbs(v));
                    }
                    if (s > bestNorm) { bestNorm = s; bestRow = r; }
                }
            }
            else // Norm.Linf
            {
                for (int r = 0; r < A.M_Rows; r++)
                {
                    int s = (int)0;
                    for (int c = 0; c < A.N_Cols; c++)
                    {
                        int v = A[r, c];
                        int av = iAbs(v);
                        if (av > s) s = av;
                    }
                    if (s > bestNorm) { bestNorm = s; bestRow = r; }
                }
            }
            return bestRow;
        }

        /// <summary>
        /// Column analog of <see cref="argMaxRowNorm"/> (same overflow note); columns are
        /// strided (non-contiguous).
        /// </summary>
        public static int argMaxColNorm(in intMxN A, Norm n)
        {
            if (n == Norm.L2)
                throw new System.ArgumentException("Query.argMaxColNorm: L2 norm-selection is float-only for integer types");
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.argMaxColNorm: empty matrix");

            int bestCol = 0;
            int bestNorm = (int)0;

            if (n == Norm.L1)
            {
                for (int c = 0; c < A.N_Cols; c++)
                {
                    int s = (int)0;
                    for (int r = 0; r < A.M_Rows; r++)
                    {
                        int v = A[r, c];
                        s = (int)(s + iAbs(v));
                    }
                    if (s > bestNorm) { bestNorm = s; bestCol = c; }
                }
            }
            else // Norm.Linf
            {
                for (int c = 0; c < A.N_Cols; c++)
                {
                    int s = (int)0;
                    for (int r = 0; r < A.M_Rows; r++)
                    {
                        int v = A[r, c];
                        int av = iAbs(v);
                        if (av > s) s = av;
                    }
                    if (s > bestNorm) { bestNorm = s; bestCol = c; }
                }
            }
            return bestCol;
        }

        // -------------------------------------------------------------------------
        // GROUP 3 — SEARCH OVER A SET OF VECTORS
        // -------------------------------------------------------------------------

        // ---- Metric validation --------------------------------------------------

        // Validates metric is integer-exact; throws ArgumentException if float-only.
        // ValidateIntegerMetric lives in intQueryCore (type-agnostic Metric->void; would collide
        // across the merged int/short/long `Query` partial).

        // ---- Integer metric score kernels ----------------------------------------
        // Internal: Row variant (contiguous elements); Col variant (strided).
        //
        // Overflow: SqEuclidean accumulates (diff*diff) in int; Dot accumulates
        // (A[r,c]*q[c]) in int. Caller must ensure maxAbsValue² × dimension fits
        // the type (int: ~2.1e9, short: ~32767, long: ~9.2e18). Manhattan/Chebyshev
        // are the recommended overflow-safe integer metrics.

        /// <summary>
        /// Score between row r of A and query q under integer-exact metric m.
        /// Supported: Manhattan, Chebyshev, SqEuclidean, Dot. See the overflow note
        /// above (and class header): the "overflow-safe" claim for Manhattan/Chebyshev
        /// covers only the abs/max step, NOT the subtraction (A[r,c] - q[c]) itself.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int RowScore(in intMxN A, int r, in intN q, Metric m)
        {
            int nCols = A.N_Cols;
            if (m == Metric.Manhattan)
            {
                int s = (int)0;
                for (int c = 0; c < nCols; c++)
                {
                    int d = (int)(A[r, c] - q[c]);
                    s = (int)(s + iAbs(d));
                }
                return s;
            }
            else if (m == Metric.SqEuclidean)
            {
                int s = (int)0;
                for (int c = 0; c < nCols; c++) { int d = (int)(A[r, c] - q[c]); s = (int)(s + d * d); }
                return s;
            }
            else if (m == Metric.Chebyshev)
            {
                int s = (int)0;
                for (int c = 0; c < nCols; c++)
                {
                    int d = (int)(A[r, c] - q[c]);
                    int ad = iAbs(d);
                    if (ad > s) s = ad;
                }
                return s;
            }
            else if (m == Metric.Dot)
            {
                int s = (int)0;
                for (int c = 0; c < nCols; c++)
                    s = (int)(s + A[r, c] * q[c]);
                return s;
            }
            else throw new System.ArgumentException("RowScore: unsupported metric for integer types");
        }

        /// <summary>
        /// Column analog of <see cref="RowScore"/> (same overflow note); columns are strided.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static int ColScore(in intMxN A, int col, in intN q, Metric m)
        {
            int mRows = A.M_Rows;
            if (m == Metric.Manhattan)
            {
                int s = (int)0;
                for (int r = 0; r < mRows; r++)
                {
                    int d = (int)(A[r, col] - q[r]);
                    s = (int)(s + iAbs(d));
                }
                return s;
            }
            else if (m == Metric.SqEuclidean)
            {
                int s = (int)0;
                for (int r = 0; r < mRows; r++) { int d = (int)(A[r, col] - q[r]); s = (int)(s + d * d); }
                return s;
            }
            else if (m == Metric.Chebyshev)
            {
                int s = (int)0;
                for (int r = 0; r < mRows; r++)
                {
                    int d = (int)(A[r, col] - q[r]);
                    int ad = iAbs(d);
                    if (ad > s) s = ad;
                }
                return s;
            }
            else if (m == Metric.Dot)
            {
                int s = (int)0;
                for (int r = 0; r < mRows; r++)
                    s = (int)(s + A[r, col] * q[r]);
                return s;
            }
            else throw new System.ArgumentException("ColScore: unsupported metric for integer types");
        }

        // Metric direction helpers (IsSimilarityMetric / WorstScoreForNearest / WorstScoreForFarthest /
        // IsBetterForNearest / IsBetterForFarthest) live in intQueryCore -- see the fProxy note.

        // ---- distancesToRow / distancesToColumn ---------------------------------

        /// <summary>
        /// Fills dest[i] with the distance/similarity between row i of A and query q
        /// under integer-exact metric m. dest must have length A.M_Rows. q.N must equal A.N_Cols.
        /// Supported metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
        /// <para>
        /// Integer distance metrics require each element AND each element-wise difference to fit
        /// the proxy type (e.g. for short, coordinates roughly within ±16383 so differences fit
        /// ±32767); values at the type extreme (MinValue) are not exactly representable in abs.
        /// Use float/double for larger ranges. SqEuclidean/Dot additionally require
        /// maxAbs²·N_Cols to fit the proxy type.
        /// </para>
        /// </summary>
        public static void distancesToRow(in intMxN A, in intN q, Metric m, ref intN dest)
        {
            intQueryCore.ValidateIntegerMetric(m);
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("Query.distancesToRow: q.N must equal A.N_Cols");
            if (dest.N != A.M_Rows)
                throw new System.ArgumentException("Query.distancesToRow: dest.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
                dest[r] = RowScore(in A, r, in q, m);
        }

        /// <summary>
        /// Fills dest[j] with the distance/similarity between column j of A and query q
        /// under integer-exact metric m. dest must have length A.N_Cols. q.N must equal A.M_Rows.
        /// Columns are strided (non-contiguous).
        /// <para>
        /// Integer distance metrics require each element AND each element-wise difference to fit
        /// the proxy type (e.g. for short, coordinates roughly within ±16383 so differences fit
        /// ±32767); values at the type extreme (MinValue) are not exactly representable in abs.
        /// Use float/double for larger ranges. SqEuclidean/Dot additionally require
        /// maxAbs²·M_Rows to fit the proxy type.
        /// </para>
        /// </summary>
        public static void distancesToColumn(in intMxN A, in intN q, Metric m, ref intN dest)
        {
            intQueryCore.ValidateIntegerMetric(m);
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("Query.distancesToColumn: q.N must equal A.M_Rows");
            if (dest.N != A.N_Cols)
                throw new System.ArgumentException("Query.distancesToColumn: dest.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
                dest[c] = ColScore(in A, c, in q, m);
        }

        // ---- nearestRow / nearestColumn ----------------------------------------

        /// <summary>
        /// Finds the row of A most similar/closest to query q under integer-exact metric m.
        /// For distance metrics (Manhattan/Chebyshev/SqEuclidean): nearest = min distance.
        /// For Dot (similarity): nearest = max dot product.
        /// score is in metric's own units (SqEuclidean → squared distance).
        /// q.N must equal A.N_Cols.
        /// <para>
        /// Overflow: see distancesToRow for the full integer overflow contract (element differences must fit the proxy type; SqEuclidean/Dot additionally require maxAbs²·dim to fit).
        /// </para>
        /// </summary>
        public static void nearestRow(in intMxN A, in intN q, Metric m, out int index, out int score)
        {
            intQueryCore.ValidateIntegerMetric(m);
            if (A.M_Rows == 0)
                throw new System.InvalidOperationException("Query.nearestRow: matrix has no rows");
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("Query.nearestRow: q.N must equal A.N_Cols");

            int best = intQueryCore.WorstScoreForNearest(m);
            int bestIdx = 0;
            for (int r = 0; r < A.M_Rows; r++)
            {
                int s = RowScore(in A, r, in q, m);
                if (intQueryCore.IsBetterForNearest(s, best, m)) { best = s; bestIdx = r; }
            }
            index = bestIdx;
            score = best;
        }

        /// <summary>
        /// Finds the column of A most similar/closest to query q under integer-exact metric m.
        /// q.N must equal A.M_Rows. Columns are strided.
        /// <para>
        /// Overflow: see distancesToColumn for the full integer overflow contract (element differences must fit the proxy type; SqEuclidean/Dot additionally require maxAbs²·dim to fit).
        /// </para>
        /// </summary>
        public static void nearestColumn(in intMxN A, in intN q, Metric m, out int index, out int score)
        {
            intQueryCore.ValidateIntegerMetric(m);
            if (A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.nearestColumn: matrix has no columns");
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("Query.nearestColumn: q.N must equal A.M_Rows");

            int best = intQueryCore.WorstScoreForNearest(m);
            int bestIdx = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                int s = ColScore(in A, c, in q, m);
                if (intQueryCore.IsBetterForNearest(s, best, m)) { best = s; bestIdx = c; }
            }
            index = bestIdx;
            score = best;
        }

        // ---- farthestRow / farthestColumn ---------------------------------------

        /// <summary>
        /// Finds the row of A most dissimilar/farthest from query q under integer-exact metric m.
        /// For distance metrics: farthest = max distance. For Dot: farthest = min dot product.
        /// score is in metric's own units.
        /// <para>
        /// Overflow: see distancesToRow for the full integer overflow contract (element differences must fit the proxy type; SqEuclidean/Dot additionally require maxAbs²·dim to fit).
        /// </para>
        /// </summary>
        public static void farthestRow(in intMxN A, in intN q, Metric m, out int index, out int score)
        {
            intQueryCore.ValidateIntegerMetric(m);
            if (A.M_Rows == 0)
                throw new System.InvalidOperationException("Query.farthestRow: matrix has no rows");
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("Query.farthestRow: q.N must equal A.N_Cols");

            int worst = intQueryCore.WorstScoreForFarthest(m);
            int worstIdx = 0;
            for (int r = 0; r < A.M_Rows; r++)
            {
                int s = RowScore(in A, r, in q, m);
                if (intQueryCore.IsBetterForFarthest(s, worst, m)) { worst = s; worstIdx = r; }
            }
            index = worstIdx;
            score = worst;
        }

        /// <summary>
        /// Finds the column of A most dissimilar/farthest from query q under integer-exact metric m.
        /// q.N must equal A.M_Rows. Columns are strided.
        /// <para>
        /// Overflow: see distancesToColumn for the full integer overflow contract (element differences must fit the proxy type; SqEuclidean/Dot additionally require maxAbs²·dim to fit).
        /// </para>
        /// </summary>
        public static void farthestColumn(in intMxN A, in intN q, Metric m, out int index, out int score)
        {
            intQueryCore.ValidateIntegerMetric(m);
            if (A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.farthestColumn: matrix has no columns");
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("Query.farthestColumn: q.N must equal A.M_Rows");

            int worst = intQueryCore.WorstScoreForFarthest(m);
            int worstIdx = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                int s = ColScore(in A, c, in q, m);
                if (intQueryCore.IsBetterForFarthest(s, worst, m)) { worst = s; worstIdx = c; }
            }
            index = worstIdx;
            score = worst;
        }

        // ---- countWithinRadius / countWithinColumnRadius (zero-alloc count) -----

        /// <summary>
        /// Returns the count of rows with distance/similarity to q within radius r.
        /// For distance metrics: count rows with score &lt;= r.
        /// For Dot (similarity): count rows with score >= r.
        /// q.N must equal A.N_Cols.
        /// <para>
        /// Overflow: see distancesToRow for the full integer overflow contract (element differences must fit the proxy type; SqEuclidean/Dot additionally require maxAbs²·dim to fit).
        /// </para>
        /// </summary>
        public static int countWithinRadius(in intMxN A, in intN q, int r, Metric m)
        {
            intQueryCore.ValidateIntegerMetric(m);
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("Query.countWithinRadius: q.N must equal A.N_Cols");

            bool sim = intQueryCore.IsSimilarityMetric(m);
            int count = 0;
            for (int row = 0; row < A.M_Rows; row++)
            {
                int s = RowScore(in A, row, in q, m);
                if (sim ? s >= r : s <= r) count++;
            }
            return count;
        }

        /// <summary>
        /// Returns the count of columns with distance/similarity to q within radius r.
        /// Columns are strided. q.N must equal A.M_Rows.
        /// <para>
        /// Overflow: see distancesToColumn for the full integer overflow contract (element differences must fit the proxy type; SqEuclidean/Dot additionally require maxAbs²·dim to fit).
        /// </para>
        /// </summary>
        public static int countWithinColumnRadius(in intMxN A, in intN q, int r, Metric m)
        {
            intQueryCore.ValidateIntegerMetric(m);
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("Query.countWithinColumnRadius: q.N must equal A.M_Rows");

            bool sim = intQueryCore.IsSimilarityMetric(m);
            int count = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                int s = ColScore(in A, c, in q, m);
                if (sim ? s >= r : s <= r) count++;
            }
            return count;
        }

        // ---- kNearest/kFarthest rows/columns with Indices buffer ----------------

        /// <summary>
        /// Finds the k nearest rows to query q under integer-exact metric m.
        /// idx and scores must both have length >= k.
        /// Fills idx[0..count) and scores[0..count) sorted best-first.
        /// Returns min(k, A.M_Rows). Uses bounded insertion sort (O(M·k)) — optimal for small k.
        /// q.N must equal A.N_Cols.
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × N_Cols fits the type; otherwise use the float variant.
        /// Manhattan/Chebyshev are the recommended integer metrics; see distancesToRow for the full overflow contract (differences must fit the proxy type).
        /// </para>
        /// </summary>
        public static int kNearestRows(in intMxN A, in intN q, int k, Metric m, ref Indices idx, ref intN scores)
        {
            intQueryCore.ValidateIntegerMetric(m);
            if (A.M_Rows == 0 || k <= 0) return 0;
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("Query.kNearestRows: q.N must equal A.N_Cols");
            if (idx.N < k)
                throw new System.ArgumentException("Query.kNearestRows: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("Query.kNearestRows: scores.N must be >= k");

            int clampedK = math.min(k, A.M_Rows);
            bool sim = intQueryCore.IsSimilarityMetric(m);
            int count = 0;

            for (int r = 0; r < A.M_Rows; r++)
            {
                int s = Query.RowScore(in A, r, in q, m);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && (sim ? s > scores[ins - 1] : s < scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = r; count++;
                }
                else
                {
                    int kth = scores[clampedK - 1];
                    if (sim ? s <= kth : s >= kth) continue;
                    int ins = clampedK - 1;
                    while (ins > 0 && (sim ? s > scores[ins - 1] : s < scores[ins - 1])) ins--;
                    for (int j = clampedK - 1; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = r;
                }
            }
            return clampedK;
        }

        /// <summary>
        /// Finds the k nearest columns to query q under integer-exact metric m.
        /// idx and scores must both have length >= k.
        /// Returns min(k, A.N_Cols). q.N must equal A.M_Rows. Columns are strided.
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × M_Rows fits the type; otherwise use the float variant.
        /// Manhattan/Chebyshev are the recommended integer metrics; see distancesToRow for the full overflow contract (differences must fit the proxy type).
        /// </para>
        /// </summary>
        public static int kNearestColumns(in intMxN A, in intN q, int k, Metric m, ref Indices idx, ref intN scores)
        {
            intQueryCore.ValidateIntegerMetric(m);
            if (A.N_Cols == 0 || k <= 0) return 0;
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("Query.kNearestColumns: q.N must equal A.M_Rows");
            if (idx.N < k)
                throw new System.ArgumentException("Query.kNearestColumns: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("Query.kNearestColumns: scores.N must be >= k");

            int clampedK = math.min(k, A.N_Cols);
            bool sim = intQueryCore.IsSimilarityMetric(m);
            int count = 0;

            for (int c = 0; c < A.N_Cols; c++)
            {
                int s = Query.ColScore(in A, c, in q, m);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && (sim ? s > scores[ins - 1] : s < scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = c; count++;
                }
                else
                {
                    int kth = scores[clampedK - 1];
                    if (sim ? s <= kth : s >= kth) continue;
                    int ins = clampedK - 1;
                    while (ins > 0 && (sim ? s > scores[ins - 1] : s < scores[ins - 1])) ins--;
                    for (int j = clampedK - 1; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = c;
                }
            }
            return clampedK;
        }

        /// <summary>
        /// Finds the k farthest rows from query q under integer-exact metric m.
        /// idx and scores must have length >= k.
        /// Returns min(k, A.M_Rows). Sorted worst-first (highest distance / lowest similarity).
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × N_Cols fits the type; otherwise use the float variant.
        /// Manhattan/Chebyshev are the recommended integer metrics; see distancesToRow for the full overflow contract (differences must fit the proxy type).
        /// </para>
        /// </summary>
        public static int kFarthestRows(in intMxN A, in intN q, int k, Metric m, ref Indices idx, ref intN scores)
        {
            intQueryCore.ValidateIntegerMetric(m);
            if (A.M_Rows == 0 || k <= 0) return 0;
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("Query.kFarthestRows: q.N must equal A.N_Cols");
            if (idx.N < k)
                throw new System.ArgumentException("Query.kFarthestRows: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("Query.kFarthestRows: scores.N must be >= k");

            int clampedK = math.min(k, A.M_Rows);
            bool sim = intQueryCore.IsSimilarityMetric(m);
            int count = 0;

            for (int r = 0; r < A.M_Rows; r++)
            {
                int s = Query.RowScore(in A, r, in q, m);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && (sim ? s < scores[ins - 1] : s > scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = r; count++;
                }
                else
                {
                    int kth = scores[clampedK - 1];
                    if (sim ? s >= kth : s <= kth) continue;
                    int ins = clampedK - 1;
                    while (ins > 0 && (sim ? s < scores[ins - 1] : s > scores[ins - 1])) ins--;
                    for (int j = clampedK - 1; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = r;
                }
            }
            return clampedK;
        }

        /// <summary>
        /// Finds the k farthest columns from query q under integer-exact metric m.
        /// Returns min(k, A.N_Cols). q.N must equal A.M_Rows. Columns are strided.
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × M_Rows fits the type; otherwise use the float variant.
        /// Manhattan/Chebyshev are the recommended integer metrics; see distancesToRow for the full overflow contract (differences must fit the proxy type).
        /// </para>
        /// </summary>
        public static int kFarthestColumns(in intMxN A, in intN q, int k, Metric m, ref Indices idx, ref intN scores)
        {
            intQueryCore.ValidateIntegerMetric(m);
            if (A.N_Cols == 0 || k <= 0) return 0;
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("Query.kFarthestColumns: q.N must equal A.M_Rows");
            if (idx.N < k)
                throw new System.ArgumentException("Query.kFarthestColumns: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("Query.kFarthestColumns: scores.N must be >= k");

            int clampedK = math.min(k, A.N_Cols);
            bool sim = intQueryCore.IsSimilarityMetric(m);
            int count = 0;

            for (int c = 0; c < A.N_Cols; c++)
            {
                int s = Query.ColScore(in A, c, in q, m);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && (sim ? s < scores[ins - 1] : s > scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = c; count++;
                }
                else
                {
                    int kth = scores[clampedK - 1];
                    if (sim ? s >= kth : s <= kth) continue;
                    int ins = clampedK - 1;
                    while (ins > 0 && (sim ? s < scores[ins - 1] : s > scores[ins - 1])) ins--;
                    for (int j = clampedK - 1; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = c;
                }
            }
            return clampedK;
        }

        /// <summary>
        /// Fills idx[0..count) with indices of rows within radius r of query q.
        /// For distance metrics: score &lt;= r. For Dot (similarity): score >= r.
        /// Returns count. idx must be sized >= A.M_Rows (worst case).
        /// q.N must equal A.N_Cols.
        /// <para>
        /// Overflow: see distancesToRow for the full integer overflow contract (element differences must fit the proxy type; SqEuclidean/Dot additionally require maxAbs²·dim to fit).
        /// </para>
        /// </summary>
        public static int rowsWithinRadius(in intMxN A, in intN q, int r, Metric m, ref Indices idx)
        {
            intQueryCore.ValidateIntegerMetric(m);
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("Query.rowsWithinRadius: q.N must equal A.N_Cols");
            if (idx.N < A.M_Rows)
                throw new System.ArgumentException("Query.rowsWithinRadius: idx.N must be >= A.M_Rows (worst case)");

            bool sim = intQueryCore.IsSimilarityMetric(m);
            int count = 0;
            for (int row = 0; row < A.M_Rows; row++)
            {
                int s = Query.RowScore(in A, row, in q, m);
                if (sim ? s >= r : s <= r) idx[count++] = row;
            }
            return count;
        }

        /// <summary>
        /// Fills idx[0..count) with indices of columns within radius r of query q.
        /// Returns count. idx must be sized >= A.N_Cols. q.N must equal A.M_Rows.
        /// Columns are strided.
        /// <para>
        /// Overflow: see distancesToColumn for the full integer overflow contract (element differences must fit the proxy type; SqEuclidean/Dot additionally require maxAbs²·dim to fit).
        /// </para>
        /// </summary>
        public static int columnsWithinRadius(in intMxN A, in intN q, int r, Metric m, ref Indices idx)
        {
            intQueryCore.ValidateIntegerMetric(m);
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("Query.columnsWithinRadius: q.N must equal A.M_Rows");
            if (idx.N < A.N_Cols)
                throw new System.ArgumentException("Query.columnsWithinRadius: idx.N must be >= A.N_Cols (worst case)");

            bool sim = intQueryCore.IsSimilarityMetric(m);
            int count = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                int s = Query.ColScore(in A, c, in q, m);
                if (sim ? s >= r : s <= r) idx[count++] = c;
            }
            return count;
        }

        // ---- nonzero with Indices buffer ---

        /// <summary>
        /// Fills idx[0..count) with flat indices of elements in x with |x[i]| > tolerance.
        /// Returns count. idx must be sized >= x.Data.Length (worst case).
        /// Generic over intN and intMxN.
        /// </summary>
        public static int nonzero<T>(in T x, int tolerance, ref Indices idx)
            where T : unmanaged, IUnsafeintArray
        {
            if (idx.N < x.Data.Length)
                throw new System.ArgumentException("Query.nonzero: idx.N must be >= x.Data.Length (worst case)");

            int count = 0;
            for (int i = 0; i < x.Data.Length; i++)
            {
                int v = x.Data[i];
                int av = iAbs(v);
                if (av > tolerance) idx[count++] = i;
            }
            return count;
        }

        // -------------------------------------------------------------------------
        // GROUP 4 — VALUE / MASK SEARCH
        // -------------------------------------------------------------------------

        /// <summary>
        /// Returns the flat index of the first element in x equal to target
        /// (within tolerance: |x[i] - target| &lt;= tolerance). Returns -1 if not found.
        /// Generic over vec + matrix flat data. (Like Excel MATCH.)
        /// <para>
        /// Overflow note: the difference (x[i] - target) is computed in the proxy type.
        /// Ensure |x[i] - target| fits the type range; otherwise use the float/double variant.
        /// For short, coordinates and differences must each be within ±32767.
        /// </para>
        /// </summary>
        public static int findValue<T>(in T x, int target, int tolerance)
            where T : unmanaged, IUnsafeintArray
        {
            for (int i = 0; i < x.Data.Length; i++)
            {
                int d = (int)(x.Data[i] - target);
                int ad = iAbs(d);
                if (ad <= tolerance)
                    return i;
            }
            return -1;
        }

        /// <summary>
        /// Returns the count of elements in x with absolute value &gt; tolerance.
        /// Zero-alloc; use with nonzero (ref Indices) for the full index list.
        /// </summary>
        public static int countNonzero<T>(in T x, int tolerance)
            where T : unmanaged, IUnsafeintArray
        {
            int count = 0;
            for (int i = 0; i < x.Data.Length; i++)
            {
                int v = x.Data[i];
                int av = iAbs(v);
                if (av > tolerance)
                    count++;
            }
            return count;
        }
    }
}
