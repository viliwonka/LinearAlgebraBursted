#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using Unity.Mathematics;
using Unity.Burst;
using System.Runtime.CompilerServices;

namespace LinearAlgebra
{
    // longQuery_OP: integer-exact search & selection inside integer vectors / matrices.
    // This is the P2 subset from spec-query.md — only the metrics/norms that are
    // exact for integer types: Manhattan, Chebyshev, SqEuclidean, Dot (Group 3);
    // L1 and Linf norms (Group 2). Euclidean, Cosine, and L2 are float-only (need
    // sqrt/division) and throw ArgumentException if passed to integer methods.
    //
    // decodeIndex is type-agnostic (int→int) and lives in fProxyQuery_OP — reuse that,
    // do NOT call or duplicate it here.
    //
    // P3 overflow note: ALL integer metrics require each element AND each element-wise
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
    public static partial class longQuery_OP
    {
        // -------------------------------------------------------------------------
        // HELPERS
        // -------------------------------------------------------------------------

        // Saturating absolute value: maps MinValue → MaxValue (off-by-one in magnitude
        // at the extreme, but correct sign/ordering and nonzero classification).
        // Use instead of the raw negation pattern `v < 0 ? -v : v` throughout this file.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static long iAbs(long v) => v < (long)0 ? (v == long.MinValue ? long.MaxValue : (long)(-v)) : v;

        // -------------------------------------------------------------------------
        // GROUP 1 — EXTREMES
        // -------------------------------------------------------------------------

        /// <summary>
        /// Index (flat) and value of the element with the largest absolute value in x.
        /// Generic over vec + matrix flat data; for matrices the index is row-major flat.
        /// On ties the first occurrence wins. Empty input throws.
        /// </summary>
        public static void argMaxAbs<T>(in T x, out long val, out int flatIndex)
            where T : unmanaged, IUnsafelongArray
        {
            if (x.Data.Length == 0)
                throw new System.InvalidOperationException("QueryOP.argMaxAbs: empty input");

            long best = iAbs(x.Data[0]);
            int bestIdx = 0;
            for (int i = 1; i < x.Data.Length; i++)
            {
                long a = iAbs(x.Data[i]);
                if (a > best) { best = a; bestIdx = i; }
            }
            val = best;
            flatIndex = bestIdx;
        }

        /// <summary>
        /// Index (flat) and value of the element with the smallest absolute value in x.
        /// On ties the first occurrence wins. Empty input throws.
        /// </summary>
        public static void argMinAbs<T>(in T x, out long val, out int flatIndex)
            where T : unmanaged, IUnsafelongArray
        {
            if (x.Data.Length == 0)
                throw new System.InvalidOperationException("QueryOP.argMinAbs: empty input");

            long best = iAbs(x.Data[0]);
            int bestIdx = 0;
            for (int i = 1; i < x.Data.Length; i++)
            {
                long a = iAbs(x.Data[i]);
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
        public static int rowArgMin(in longMxN A, ref Indices colIndexPerRow, ref longN valPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.rowArgMin: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowArgMin: colIndexPerRow.N must equal A.M_Rows");
            if (valPerRow.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowArgMin: valPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                long best = A[r, 0];
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
        public static int rowArgMin(in longMxN A, ref Indices colIndexPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.rowArgMin: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowArgMin: colIndexPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                long best = A[r, 0];
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
        public static int rowArgMax(in longMxN A, ref Indices colIndexPerRow, ref longN valPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.rowArgMax: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowArgMax: colIndexPerRow.N must equal A.M_Rows");
            if (valPerRow.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowArgMax: valPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                long best = A[r, 0];
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
        public static int rowArgMax(in longMxN A, ref Indices colIndexPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.rowArgMax: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowArgMax: colIndexPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                long best = A[r, 0];
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
        public static int colArgMin(in longMxN A, ref Indices rowIndexPerCol, ref longN valPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.colArgMin: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.colArgMin: rowIndexPerCol.N must equal A.N_Cols");
            if (valPerCol.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.colArgMin: valPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                long best = A[0, c];
                int bestR = 0;
                for (int r = 1; r < A.M_Rows; r++)
                    if (A[r, c] < best) { best = A[r, c]; bestR = r; }
                rowIndexPerCol[c] = bestR;
                valPerCol[c] = best;
            }
            return A.N_Cols;
        }

        /// <summary>Index-only form of colArgMin. Returns A.N_Cols.</summary>
        public static int colArgMin(in longMxN A, ref Indices rowIndexPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.colArgMin: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.colArgMin: rowIndexPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                long best = A[0, c];
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
        public static int colArgMax(in longMxN A, ref Indices rowIndexPerCol, ref longN valPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.colArgMax: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.colArgMax: rowIndexPerCol.N must equal A.N_Cols");
            if (valPerCol.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.colArgMax: valPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                long best = A[0, c];
                int bestR = 0;
                for (int r = 1; r < A.M_Rows; r++)
                    if (A[r, c] > best) { best = A[r, c]; bestR = r; }
                rowIndexPerCol[c] = bestR;
                valPerCol[c] = best;
            }
            return A.N_Cols;
        }

        /// <summary>Index-only form of colArgMax. Returns A.N_Cols.</summary>
        public static int colArgMax(in longMxN A, ref Indices rowIndexPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.colArgMax: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.colArgMax: rowIndexPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                long best = A[0, c];
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
        public static int argMaxRowNorm(in longMxN A, Norm n)
        {
            if (n == Norm.L2)
                throw new System.ArgumentException("QueryOP.argMaxRowNorm: L2 norm-selection is float-only for integer types");
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.argMaxRowNorm: empty matrix");

            int bestRow = 0;
            long bestNorm = (long)0;

            if (n == Norm.L1)
            {
                for (int r = 0; r < A.M_Rows; r++)
                {
                    long s = (long)0;
                    for (int c = 0; c < A.N_Cols; c++)
                    {
                        long v = A[r, c];
                        s = (long)(s + iAbs(v));
                    }
                    if (s > bestNorm) { bestNorm = s; bestRow = r; }
                }
            }
            else // Norm.Linf
            {
                for (int r = 0; r < A.M_Rows; r++)
                {
                    long s = (long)0;
                    for (int c = 0; c < A.N_Cols; c++)
                    {
                        long v = A[r, c];
                        long av = iAbs(v);
                        if (av > s) s = av;
                    }
                    if (s > bestNorm) { bestNorm = s; bestRow = r; }
                }
            }
            return bestRow;
        }

        /// <summary>
        /// Returns the column index whose norm (L1 or Linf) is largest.
        /// L2 norm is not supported for integer types (requires sqrt) — throws ArgumentException.
        /// Columns are strided (non-contiguous); on ties the first occurrence wins.
        /// <para>
        /// Overflow note: L1 accumulates |element| values; Linf takes max |element|.
        /// Both are overflow-safe for typical integer ranges.
        /// </para>
        /// </summary>
        public static int argMaxColNorm(in longMxN A, Norm n)
        {
            if (n == Norm.L2)
                throw new System.ArgumentException("QueryOP.argMaxColNorm: L2 norm-selection is float-only for integer types");
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.argMaxColNorm: empty matrix");

            int bestCol = 0;
            long bestNorm = (long)0;

            if (n == Norm.L1)
            {
                for (int c = 0; c < A.N_Cols; c++)
                {
                    long s = (long)0;
                    for (int r = 0; r < A.M_Rows; r++)
                    {
                        long v = A[r, c];
                        s = (long)(s + iAbs(v));
                    }
                    if (s > bestNorm) { bestNorm = s; bestCol = c; }
                }
            }
            else // Norm.Linf
            {
                for (int c = 0; c < A.N_Cols; c++)
                {
                    long s = (long)0;
                    for (int r = 0; r < A.M_Rows; r++)
                    {
                        long v = A[r, c];
                        long av = iAbs(v);
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
        // Call once at method entry (hoisted outside per-row loops).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void ValidateIntegerMetric(Metric m)
        {
            if (m == Metric.Euclidean || m == Metric.Cosine)
                throw new System.ArgumentException(
                    "QueryOP: Euclidean and Cosine metrics require sqrt/division and are float-only for integer types. Use Manhattan, Chebyshev, SqEuclidean, or Dot instead.");
        }

        // ---- Integer metric score kernels ----------------------------------------
        // Internal: Row variant (contiguous elements); Col variant (strided).
        //
        // P3 overflow: SqEuclidean accumulates (diff*diff) in long; Dot accumulates
        // (A[r,c]*q[c]) in long. Caller must ensure maxAbsValue² × dimension fits
        // the type (int: ~2.1e9, short: ~32767, long: ~9.2e18). Manhattan/Chebyshev
        // are the recommended overflow-safe integer metrics.

        /// <summary>
        /// Score between row r of A and query q under integer-exact metric m.
        /// Supported: Manhattan, Chebyshev, SqEuclidean, Dot.
        /// <para>
        /// Integer distance metrics require each element AND each element-wise difference
        /// to fit the proxy type (e.g. for short, coordinates roughly within ±16383 so
        /// differences fit ±32767); values at the type extreme (MinValue) are not exactly
        /// representable in abs. Use float/double for larger ranges.
        /// SqEuclidean/Dot additionally require maxAbs²·dim to fit the proxy type.
        /// The "overflow-safe" claim for Manhattan/Chebyshev applies only to the abs/max step,
        /// NOT to the subtraction (A[r,c] - q[c]) which can itself overflow at the type boundary.
        /// </para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static long RowScore(in longMxN A, int r, in longN q, Metric m)
        {
            int nCols = A.N_Cols;
            if (m == Metric.Manhattan)
            {
                long s = (long)0;
                for (int c = 0; c < nCols; c++)
                {
                    long d = (long)(A[r, c] - q[c]);
                    s = (long)(s + iAbs(d));
                }
                return s;
            }
            else if (m == Metric.SqEuclidean)
            {
                long s = (long)0;
                for (int c = 0; c < nCols; c++) { long d = (long)(A[r, c] - q[c]); s = (long)(s + d * d); }
                return s;
            }
            else if (m == Metric.Chebyshev)
            {
                long s = (long)0;
                for (int c = 0; c < nCols; c++)
                {
                    long d = (long)(A[r, c] - q[c]);
                    long ad = iAbs(d);
                    if (ad > s) s = ad;
                }
                return s;
            }
            else if (m == Metric.Dot)
            {
                long s = (long)0;
                for (int c = 0; c < nCols; c++)
                    s = (long)(s + A[r, c] * q[c]);
                return s;
            }
            else throw new System.ArgumentException("RowScore: unsupported metric for integer types");
        }

        /// <summary>
        /// Score between column col of A and query q under integer-exact metric m.
        /// Supported: Manhattan, Chebyshev, SqEuclidean, Dot. Columns are strided.
        /// <para>
        /// Integer distance metrics require each element AND each element-wise difference
        /// to fit the proxy type (e.g. for short, coordinates roughly within ±16383 so
        /// differences fit ±32767); values at the type extreme (MinValue) are not exactly
        /// representable in abs. Use float/double for larger ranges.
        /// SqEuclidean/Dot additionally require maxAbs²·dim to fit the proxy type.
        /// The "overflow-safe" claim for Manhattan/Chebyshev applies only to the abs/max step,
        /// NOT to the subtraction (A[r,col] - q[r]) which can itself overflow at the type boundary.
        /// </para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static long ColScore(in longMxN A, int col, in longN q, Metric m)
        {
            int mRows = A.M_Rows;
            if (m == Metric.Manhattan)
            {
                long s = (long)0;
                for (int r = 0; r < mRows; r++)
                {
                    long d = (long)(A[r, col] - q[r]);
                    s = (long)(s + iAbs(d));
                }
                return s;
            }
            else if (m == Metric.SqEuclidean)
            {
                long s = (long)0;
                for (int r = 0; r < mRows; r++) { long d = (long)(A[r, col] - q[r]); s = (long)(s + d * d); }
                return s;
            }
            else if (m == Metric.Chebyshev)
            {
                long s = (long)0;
                for (int r = 0; r < mRows; r++)
                {
                    long d = (long)(A[r, col] - q[r]);
                    long ad = iAbs(d);
                    if (ad > s) s = ad;
                }
                return s;
            }
            else if (m == Metric.Dot)
            {
                long s = (long)0;
                for (int r = 0; r < mRows; r++)
                    s = (long)(s + A[r, col] * q[r]);
                return s;
            }
            else throw new System.ArgumentException("ColScore: unsupported metric for integer types");
        }

        // Metric direction helpers (hoisted outside per-row loops).
        // Dot is similarity (higher = nearer); Manhattan/Chebyshev/SqEuclidean are distance (lower = nearer).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static bool IsSimilarityMetric(Metric m) => m == Metric.Dot;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static long WorstScoreForNearest(Metric m)
            => IsSimilarityMetric(m) ? long.MinValue : long.MaxValue;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static long WorstScoreForFarthest(Metric m)
            => IsSimilarityMetric(m) ? long.MaxValue : long.MinValue;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static bool IsBetterForNearest(long a, long b, Metric m)
            => IsSimilarityMetric(m) ? a > b : a < b;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static bool IsBetterForFarthest(long a, long b, Metric m)
            => IsSimilarityMetric(m) ? a < b : a > b;

        // ---- distancesToRow / distancesToColumn ---------------------------------

        /// <summary>
        /// Fills dest[i] with the distance/similarity between row i of A and query q
        /// under integer-exact metric m. dest must have length A.M_Rows. q.N must equal A.N_Cols.
        /// Supported metrics: Manhattan, Chebyshev, SqEuclidean, Dot.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Integer distance metrics require each element AND each element-wise difference to fit
        /// the proxy type (e.g. for short, coordinates roughly within ±16383 so differences fit
        /// ±32767); values at the type extreme (MinValue) are not exactly representable in abs.
        /// Use float/double for larger ranges. SqEuclidean/Dot additionally require
        /// maxAbs²·N_Cols to fit the proxy type.
        /// </para>
        /// </summary>
        public static void distancesToRow(in longMxN A, in longN q, Metric m, ref longN dest)
        {
            ValidateIntegerMetric(m);
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.distancesToRow: q.N must equal A.N_Cols");
            if (dest.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.distancesToRow: dest.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
                dest[r] = RowScore(in A, r, in q, m);
        }

        /// <summary>
        /// Fills dest[j] with the distance/similarity between column j of A and query q
        /// under integer-exact metric m. dest must have length A.N_Cols. q.N must equal A.M_Rows.
        /// Columns are strided (non-contiguous).
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Integer distance metrics require each element AND each element-wise difference to fit
        /// the proxy type (e.g. for short, coordinates roughly within ±16383 so differences fit
        /// ±32767); values at the type extreme (MinValue) are not exactly representable in abs.
        /// Use float/double for larger ranges. SqEuclidean/Dot additionally require
        /// maxAbs²·M_Rows to fit the proxy type.
        /// </para>
        /// </summary>
        public static void distancesToColumn(in longMxN A, in longN q, Metric m, ref longN dest)
        {
            ValidateIntegerMetric(m);
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.distancesToColumn: q.N must equal A.M_Rows");
            if (dest.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.distancesToColumn: dest.N must equal A.N_Cols");

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
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow: see distancesToRow for the full integer overflow contract (element differences must fit the proxy type; SqEuclidean/Dot additionally require maxAbs²·dim to fit).
        /// </para>
        /// </summary>
        public static void nearestRow(in longMxN A, in longN q, Metric m, out int index, out long score)
        {
            ValidateIntegerMetric(m);
            if (A.M_Rows == 0)
                throw new System.InvalidOperationException("QueryOP.nearestRow: matrix has no rows");
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.nearestRow: q.N must equal A.N_Cols");

            long best = WorstScoreForNearest(m);
            int bestIdx = 0;
            for (int r = 0; r < A.M_Rows; r++)
            {
                long s = RowScore(in A, r, in q, m);
                if (IsBetterForNearest(s, best, m)) { best = s; bestIdx = r; }
            }
            index = bestIdx;
            score = best;
        }

        /// <summary>
        /// Finds the column of A most similar/closest to query q under integer-exact metric m.
        /// q.N must equal A.M_Rows. Columns are strided.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow: see distancesToColumn for the full integer overflow contract (element differences must fit the proxy type; SqEuclidean/Dot additionally require maxAbs²·dim to fit).
        /// </para>
        /// </summary>
        public static void nearestColumn(in longMxN A, in longN q, Metric m, out int index, out long score)
        {
            ValidateIntegerMetric(m);
            if (A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.nearestColumn: matrix has no columns");
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.nearestColumn: q.N must equal A.M_Rows");

            long best = WorstScoreForNearest(m);
            int bestIdx = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                long s = ColScore(in A, c, in q, m);
                if (IsBetterForNearest(s, best, m)) { best = s; bestIdx = c; }
            }
            index = bestIdx;
            score = best;
        }

        // ---- farthestRow / farthestColumn ---------------------------------------

        /// <summary>
        /// Finds the row of A most dissimilar/farthest from query q under integer-exact metric m.
        /// For distance metrics: farthest = max distance. For Dot: farthest = min dot product.
        /// score is in metric's own units.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow: see distancesToRow for the full integer overflow contract (element differences must fit the proxy type; SqEuclidean/Dot additionally require maxAbs²·dim to fit).
        /// </para>
        /// </summary>
        public static void farthestRow(in longMxN A, in longN q, Metric m, out int index, out long score)
        {
            ValidateIntegerMetric(m);
            if (A.M_Rows == 0)
                throw new System.InvalidOperationException("QueryOP.farthestRow: matrix has no rows");
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.farthestRow: q.N must equal A.N_Cols");

            long worst = WorstScoreForFarthest(m);
            int worstIdx = 0;
            for (int r = 0; r < A.M_Rows; r++)
            {
                long s = RowScore(in A, r, in q, m);
                if (IsBetterForFarthest(s, worst, m)) { worst = s; worstIdx = r; }
            }
            index = worstIdx;
            score = worst;
        }

        /// <summary>
        /// Finds the column of A most dissimilar/farthest from query q under integer-exact metric m.
        /// q.N must equal A.M_Rows. Columns are strided.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow: see distancesToColumn for the full integer overflow contract (element differences must fit the proxy type; SqEuclidean/Dot additionally require maxAbs²·dim to fit).
        /// </para>
        /// </summary>
        public static void farthestColumn(in longMxN A, in longN q, Metric m, out int index, out long score)
        {
            ValidateIntegerMetric(m);
            if (A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.farthestColumn: matrix has no columns");
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.farthestColumn: q.N must equal A.M_Rows");

            long worst = WorstScoreForFarthest(m);
            int worstIdx = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                long s = ColScore(in A, c, in q, m);
                if (IsBetterForFarthest(s, worst, m)) { worst = s; worstIdx = c; }
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
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow: see distancesToRow for the full integer overflow contract (element differences must fit the proxy type; SqEuclidean/Dot additionally require maxAbs²·dim to fit).
        /// </para>
        /// </summary>
        public static int countWithinRadius(in longMxN A, in longN q, long r, Metric m)
        {
            ValidateIntegerMetric(m);
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.countWithinRadius: q.N must equal A.N_Cols");

            bool sim = IsSimilarityMetric(m);
            int count = 0;
            for (int row = 0; row < A.M_Rows; row++)
            {
                long s = RowScore(in A, row, in q, m);
                if (sim ? s >= r : s <= r) count++;
            }
            return count;
        }

        /// <summary>
        /// Returns the count of columns with distance/similarity to q within radius r.
        /// Columns are strided. q.N must equal A.M_Rows.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow: see distancesToColumn for the full integer overflow contract (element differences must fit the proxy type; SqEuclidean/Dot additionally require maxAbs²·dim to fit).
        /// </para>
        /// </summary>
        public static int countWithinColumnRadius(in longMxN A, in longN q, long r, Metric m)
        {
            ValidateIntegerMetric(m);
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.countWithinColumnRadius: q.N must equal A.M_Rows");

            bool sim = IsSimilarityMetric(m);
            int count = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                long s = ColScore(in A, c, in q, m);
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
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × N_Cols fits the type; otherwise use the float variant.
        /// Manhattan/Chebyshev are the recommended integer metrics; see distancesToRow for the full overflow contract (differences must fit the proxy type).
        /// </para>
        /// </summary>
        public static int kNearestRows(in longMxN A, in longN q, int k, Metric m, ref Indices idx, ref longN scores)
        {
            ValidateIntegerMetric(m);
            if (A.M_Rows == 0 || k <= 0) return 0;
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.kNearestRows: q.N must equal A.N_Cols");
            if (idx.N < k)
                throw new System.ArgumentException("QueryOP.kNearestRows: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("QueryOP.kNearestRows: scores.N must be >= k");

            int clampedK = math.min(k, A.M_Rows);
            bool sim = IsSimilarityMetric(m);
            int count = 0;

            for (int r = 0; r < A.M_Rows; r++)
            {
                long s = longQuery_OP.RowScore(in A, r, in q, m);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && (sim ? s > scores[ins - 1] : s < scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = r; count++;
                }
                else
                {
                    long kth = scores[clampedK - 1];
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
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × M_Rows fits the type; otherwise use the float variant.
        /// Manhattan/Chebyshev are the recommended integer metrics; see distancesToRow for the full overflow contract (differences must fit the proxy type).
        /// </para>
        /// </summary>
        public static int kNearestColumns(in longMxN A, in longN q, int k, Metric m, ref Indices idx, ref longN scores)
        {
            ValidateIntegerMetric(m);
            if (A.N_Cols == 0 || k <= 0) return 0;
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.kNearestColumns: q.N must equal A.M_Rows");
            if (idx.N < k)
                throw new System.ArgumentException("QueryOP.kNearestColumns: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("QueryOP.kNearestColumns: scores.N must be >= k");

            int clampedK = math.min(k, A.N_Cols);
            bool sim = IsSimilarityMetric(m);
            int count = 0;

            for (int c = 0; c < A.N_Cols; c++)
            {
                long s = longQuery_OP.ColScore(in A, c, in q, m);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && (sim ? s > scores[ins - 1] : s < scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = c; count++;
                }
                else
                {
                    long kth = scores[clampedK - 1];
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
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × N_Cols fits the type; otherwise use the float variant.
        /// Manhattan/Chebyshev are the recommended integer metrics; see distancesToRow for the full overflow contract (differences must fit the proxy type).
        /// </para>
        /// </summary>
        public static int kFarthestRows(in longMxN A, in longN q, int k, Metric m, ref Indices idx, ref longN scores)
        {
            ValidateIntegerMetric(m);
            if (A.M_Rows == 0 || k <= 0) return 0;
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.kFarthestRows: q.N must equal A.N_Cols");
            if (idx.N < k)
                throw new System.ArgumentException("QueryOP.kFarthestRows: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("QueryOP.kFarthestRows: scores.N must be >= k");

            int clampedK = math.min(k, A.M_Rows);
            bool sim = IsSimilarityMetric(m);
            int count = 0;

            for (int r = 0; r < A.M_Rows; r++)
            {
                long s = longQuery_OP.RowScore(in A, r, in q, m);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && (sim ? s < scores[ins - 1] : s > scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = r; count++;
                }
                else
                {
                    long kth = scores[clampedK - 1];
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
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow (SqEuclidean/Dot): accumulates products in the proxy type.
        /// Ensure maxAbsValue² × M_Rows fits the type; otherwise use the float variant.
        /// Manhattan/Chebyshev are the recommended integer metrics; see distancesToRow for the full overflow contract (differences must fit the proxy type).
        /// </para>
        /// </summary>
        public static int kFarthestColumns(in longMxN A, in longN q, int k, Metric m, ref Indices idx, ref longN scores)
        {
            ValidateIntegerMetric(m);
            if (A.N_Cols == 0 || k <= 0) return 0;
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.kFarthestColumns: q.N must equal A.M_Rows");
            if (idx.N < k)
                throw new System.ArgumentException("QueryOP.kFarthestColumns: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("QueryOP.kFarthestColumns: scores.N must be >= k");

            int clampedK = math.min(k, A.N_Cols);
            bool sim = IsSimilarityMetric(m);
            int count = 0;

            for (int c = 0; c < A.N_Cols; c++)
            {
                long s = longQuery_OP.ColScore(in A, c, in q, m);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && (sim ? s < scores[ins - 1] : s > scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = c; count++;
                }
                else
                {
                    long kth = scores[clampedK - 1];
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
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow: see distancesToRow for the full integer overflow contract (element differences must fit the proxy type; SqEuclidean/Dot additionally require maxAbs²·dim to fit).
        /// </para>
        /// </summary>
        public static int rowsWithinRadius(in longMxN A, in longN q, long r, Metric m, ref Indices idx)
        {
            ValidateIntegerMetric(m);
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.rowsWithinRadius: q.N must equal A.N_Cols");
            if (idx.N < A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowsWithinRadius: idx.N must be >= A.M_Rows (worst case)");

            bool sim = IsSimilarityMetric(m);
            int count = 0;
            for (int row = 0; row < A.M_Rows; row++)
            {
                long s = longQuery_OP.RowScore(in A, row, in q, m);
                if (sim ? s >= r : s <= r) idx[count++] = row;
            }
            return count;
        }

        /// <summary>
        /// Fills idx[0..count) with indices of columns within radius r of query q.
        /// Returns count. idx must be sized >= A.N_Cols. q.N must equal A.M_Rows.
        /// Columns are strided.
        /// Euclidean and Cosine throw ArgumentException (float-only).
        /// <para>
        /// Overflow: see distancesToColumn for the full integer overflow contract (element differences must fit the proxy type; SqEuclidean/Dot additionally require maxAbs²·dim to fit).
        /// </para>
        /// </summary>
        public static int columnsWithinRadius(in longMxN A, in longN q, long r, Metric m, ref Indices idx)
        {
            ValidateIntegerMetric(m);
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.columnsWithinRadius: q.N must equal A.M_Rows");
            if (idx.N < A.N_Cols)
                throw new System.ArgumentException("QueryOP.columnsWithinRadius: idx.N must be >= A.N_Cols (worst case)");

            bool sim = IsSimilarityMetric(m);
            int count = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                long s = longQuery_OP.ColScore(in A, c, in q, m);
                if (sim ? s >= r : s <= r) idx[count++] = c;
            }
            return count;
        }

        // ---- nonzero with Indices buffer ---

        /// <summary>
        /// Fills idx[0..count) with flat indices of elements in x with |x[i]| > tol.
        /// Returns count. idx must be sized >= x.Data.Length (worst case).
        /// Generic over longN and longMxN.
        /// </summary>
        public static int nonzero<T>(in T x, long tol, ref Indices idx)
            where T : unmanaged, IUnsafelongArray
        {
            if (idx.N < x.Data.Length)
                throw new System.ArgumentException("QueryOP.nonzero: idx.N must be >= x.Data.Length (worst case)");

            int count = 0;
            for (int i = 0; i < x.Data.Length; i++)
            {
                long v = x.Data[i];
                long av = iAbs(v);
                if (av > tol) idx[count++] = i;
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
        /// <para>
        /// Overflow note: the difference (x[i] - target) is computed in the proxy type.
        /// Ensure |x[i] - target| fits the type range; otherwise use the float/double variant.
        /// For short, coordinates and differences must each be within ±32767.
        /// </para>
        /// </summary>
        public static int findValue<T>(in T x, long target, long tol)
            where T : unmanaged, IUnsafelongArray
        {
            for (int i = 0; i < x.Data.Length; i++)
            {
                long d = (long)(x.Data[i] - target);
                long ad = iAbs(d);
                if (ad <= tol)
                    return i;
            }
            return -1;
        }

        /// <summary>
        /// Returns the count of elements in x with absolute value &gt; tol.
        /// Zero-alloc; use with nonzero (ref Indices) for the full index list.
        /// </summary>
        public static int countNonzero<T>(in T x, long tol)
            where T : unmanaged, IUnsafelongArray
        {
            int count = 0;
            for (int i = 0; i < x.Data.Length; i++)
            {
                long v = x.Data[i];
                long av = iAbs(v);
                if (av > tol)
                    count++;
            }
            return count;
        }
    }
}
