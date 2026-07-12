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
    //       Per-axis rowArgMin/Max / colArgMin/Max with Indices buffer (filled in this file
    //       using Indices as the shared buffer type).
    //   2 — Norm-selection: argMaxRowNorm / argMaxColNorm (reuses per-row/col norm loops)
    //   3 — Search over a set of vectors: distancesToRow/Column, nearestRow/Column,
    //         farthestRow/Column, kNearestRows/Columns + kFarthest*, rowsWithinRadius/Column,
    //         countWithinRadius/Column.
    //       Methods that return Indices index buffers (kNearestRows etc.) are also in this file.
    //   4 — Value / mask search: findValue, nonzero, countNonzero.
    //       nonzero (ref Indices) is in this file.
    public static partial class Query
    {
        // -------------------------------------------------------------------------
        // GROUP 1 — EXTREMES
        // -------------------------------------------------------------------------

        /// <summary>
        /// Index (flat) and value of the element with the largest absolute value in x.
        /// Generic over vec + matrix flat data; for matrices the index is row-major flat.
        /// On ties the first occurrence wins. Empty input throws.
        /// </summary>
        public static void argMaxAbs<T>(in T x, out fProxy val, out int flatIndex)
            where T : unmanaged, IUnsafefProxyArray
        {
            if (x.Data.Length == 0)
                throw new System.InvalidOperationException("Query.argMaxAbs: empty input");

            fProxy best = math.abs(x.Data[0]);
            int bestIdx = 0;
            for (int i = 1; i < x.Data.Length; i++)
            {
                fProxy a = math.abs(x.Data[i]);
                if (a > best) { best = a; bestIdx = i; }
            }
            val = best;
            flatIndex = bestIdx;
        }

        /// <summary>
        /// Index (flat) and value of the element with the smallest absolute value in x.
        /// On ties the first occurrence wins. Empty input throws.
        /// </summary>
        public static void argMinAbs<T>(in T x, out fProxy val, out int flatIndex)
            where T : unmanaged, IUnsafefProxyArray
        {
            if (x.Data.Length == 0)
                throw new System.InvalidOperationException("Query.argMinAbs: empty input");

            fProxy best = math.abs(x.Data[0]);
            int bestIdx = 0;
            for (int i = 1; i < x.Data.Length; i++)
            {
                fProxy a = math.abs(x.Data[i]);
                if (a < best) { best = a; bestIdx = i; }
            }
            val = best;
            flatIndex = bestIdx;
        }

        // decodeIndex (row-major flat -> (row,col)) is type-agnostic (int-only) and lives in the
        // non-templated Query.Shared.cs so the merged float/double partial emits it exactly once.

        // ---- Per-axis row/col arg-min/max with Indices buffer ---

        /// <summary>
        /// For each row i of A, writes the column index of the minimum element into
        /// colIndexPerRow[i] and the minimum value into valPerRow[i].
        /// Returns A.M_Rows. Both buffers must have length A.M_Rows.
        /// </summary>
        public static int rowArgMin(in fProxyMxN A, ref Indices colIndexPerRow, ref fProxyN valPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.rowArgMin: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("Query.rowArgMin: colIndexPerRow.N must equal A.M_Rows");
            if (valPerRow.N != A.M_Rows)
                throw new System.ArgumentException("Query.rowArgMin: valPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                fProxy best = A[r, 0];
                int bestC = 0;
                for (int c = 1; c < A.N_Cols; c++)
                    if (A[r, c] < best) { best = A[r, c]; bestC = c; }
                colIndexPerRow[r] = bestC;
                valPerRow[r] = best;
            }
            return A.M_Rows;
        }

        /// <summary>Index-only form of rowArgMin. Returns A.M_Rows.</summary>
        public static int rowArgMin(in fProxyMxN A, ref Indices colIndexPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.rowArgMin: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("Query.rowArgMin: colIndexPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                fProxy best = A[r, 0];
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
        public static int rowArgMax(in fProxyMxN A, ref Indices colIndexPerRow, ref fProxyN valPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.rowArgMax: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("Query.rowArgMax: colIndexPerRow.N must equal A.M_Rows");
            if (valPerRow.N != A.M_Rows)
                throw new System.ArgumentException("Query.rowArgMax: valPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                fProxy best = A[r, 0];
                int bestC = 0;
                for (int c = 1; c < A.N_Cols; c++)
                    if (A[r, c] > best) { best = A[r, c]; bestC = c; }
                colIndexPerRow[r] = bestC;
                valPerRow[r] = best;
            }
            return A.M_Rows;
        }

        /// <summary>Index-only form of rowArgMax. Returns A.M_Rows.</summary>
        public static int rowArgMax(in fProxyMxN A, ref Indices colIndexPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.rowArgMax: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("Query.rowArgMax: colIndexPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                fProxy best = A[r, 0];
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
        public static int colArgMin(in fProxyMxN A, ref Indices rowIndexPerCol, ref fProxyN valPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.colArgMin: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("Query.colArgMin: rowIndexPerCol.N must equal A.N_Cols");
            if (valPerCol.N != A.N_Cols)
                throw new System.ArgumentException("Query.colArgMin: valPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                fProxy best = A[0, c];
                int bestR = 0;
                for (int r = 1; r < A.M_Rows; r++)
                    if (A[r, c] < best) { best = A[r, c]; bestR = r; }
                rowIndexPerCol[c] = bestR;
                valPerCol[c] = best;
            }
            return A.N_Cols;
        }

        /// <summary>Index-only form of colArgMin. Returns A.N_Cols.</summary>
        public static int colArgMin(in fProxyMxN A, ref Indices rowIndexPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.colArgMin: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("Query.colArgMin: rowIndexPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                fProxy best = A[0, c];
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
        public static int colArgMax(in fProxyMxN A, ref Indices rowIndexPerCol, ref fProxyN valPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.colArgMax: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("Query.colArgMax: rowIndexPerCol.N must equal A.N_Cols");
            if (valPerCol.N != A.N_Cols)
                throw new System.ArgumentException("Query.colArgMax: valPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                fProxy best = A[0, c];
                int bestR = 0;
                for (int r = 1; r < A.M_Rows; r++)
                    if (A[r, c] > best) { best = A[r, c]; bestR = r; }
                rowIndexPerCol[c] = bestR;
                valPerCol[c] = best;
            }
            return A.N_Cols;
        }

        /// <summary>Index-only form of colArgMax. Returns A.N_Cols.</summary>
        public static int colArgMax(in fProxyMxN A, ref Indices rowIndexPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.colArgMax: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("Query.colArgMax: rowIndexPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                fProxy best = A[0, c];
                int bestR = 0;
                for (int r = 1; r < A.M_Rows; r++)
                    if (A[r, c] > best) { best = A[r, c]; bestR = r; }
                rowIndexPerCol[c] = bestR;
            }
            return A.N_Cols;
        }

        // -------------------------------------------------------------------------
        // GROUP 2 — NORM-SELECTION
        // -------------------------------------------------------------------------

        /// <summary>
        /// Returns the row index whose norm (L1/L2/Linf) is largest.
        /// For L2: compares squared norms to avoid a sqrt per row (argmax is monotone under sqrt).
        /// On ties the first occurrence wins.
        /// </summary>
        public static int argMaxRowNorm(in fProxyMxN A, Norm n)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.argMaxRowNorm: empty matrix");

            int bestRow = 0;
            fProxy bestNorm = (fProxy)0;

            if (n == Norm.L1)
            {
                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy s = (fProxy)0;
                    for (int c = 0; c < A.N_Cols; c++)
                        s += math.abs(A[r, c]);
                    if (s > bestNorm) { bestNorm = s; bestRow = r; }
                }
            }
            else if (n == Norm.L2)
            {
                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy s = (fProxy)0;
                    for (int c = 0; c < A.N_Cols; c++)
                        s += A[r, c] * A[r, c];
                    if (s > bestNorm) { bestNorm = s; bestRow = r; }
                }
            }
            else // Norm.Linf
            {
                for (int r = 0; r < A.M_Rows; r++)
                {
                    fProxy s = (fProxy)0;
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
        public static int argMaxColNorm(in fProxyMxN A, Norm n)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.argMaxColNorm: empty matrix");

            int bestCol = 0;
            fProxy bestNorm = (fProxy)0;

            if (n == Norm.L1)
            {
                for (int c = 0; c < A.N_Cols; c++)
                {
                    fProxy s = (fProxy)0;
                    for (int r = 0; r < A.M_Rows; r++)
                        s += math.abs(A[r, c]);
                    if (s > bestNorm) { bestNorm = s; bestCol = c; }
                }
            }
            else if (n == Norm.L2)
            {
                for (int c = 0; c < A.N_Cols; c++)
                {
                    fProxy s = (fProxy)0;
                    for (int r = 0; r < A.M_Rows; r++)
                        s += A[r, c] * A[r, c];
                    if (s > bestNorm) { bestNorm = s; bestCol = c; }
                }
            }
            else // Norm.Linf
            {
                for (int c = 0; c < A.N_Cols; c++)
                {
                    fProxy s = (fProxy)0;
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
        // Internal: exposed so ArenaExtensions.Query.fProxy.cs can do two-pass alloc
        // without duplicating metric kernels. Row variant (contiguous); Col variant (strided).

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static fProxy RowScore(in fProxyMxN A, int r, in fProxyN q, Metric m)
        {
            int nCols = A.N_Cols;
            if (m == Metric.Manhattan)
            {
                fProxy s = (fProxy)0;
                for (int c = 0; c < nCols; c++)
                    s += math.abs(A[r, c] - q[c]);
                return s;
            }
            else if (m == Metric.Euclidean)
            {
                fProxy s = (fProxy)0;
                for (int c = 0; c < nCols; c++) { fProxy d = A[r, c] - q[c]; s += d * d; }
                return math.sqrt(s);
            }
            else if (m == Metric.SqEuclidean)
            {
                fProxy s = (fProxy)0;
                for (int c = 0; c < nCols; c++) { fProxy d = A[r, c] - q[c]; s += d * d; }
                return s;
            }
            else if (m == Metric.Chebyshev)
            {
                fProxy s = (fProxy)0;
                for (int c = 0; c < nCols; c++)
                    s = math.max(s, math.abs(A[r, c] - q[c]));
                return s;
            }
            else if (m == Metric.Cosine)
            {
                fProxy dot = (fProxy)0, normA = (fProxy)0, normQ = (fProxy)0;
                for (int c = 0; c < nCols; c++)
                {
                    dot   += A[r, c] * q[c];
                    normA += A[r, c] * A[r, c];
                    normQ += q[c] * q[c];
                }
                fProxy denom = math.sqrt(normA * normQ);
                return denom > (fProxy)0 ? dot / denom : (fProxy)0;
            }
            else // Metric.Dot
            {
                fProxy s = (fProxy)0;
                for (int c = 0; c < nCols; c++)
                    s += A[r, c] * q[c];
                return s;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static fProxy ColScore(in fProxyMxN A, int col, in fProxyN q, Metric m)
        {
            int mRows = A.M_Rows;
            if (m == Metric.Manhattan)
            {
                fProxy s = (fProxy)0;
                for (int r = 0; r < mRows; r++)
                    s += math.abs(A[r, col] - q[r]);
                return s;
            }
            else if (m == Metric.Euclidean)
            {
                fProxy s = (fProxy)0;
                for (int r = 0; r < mRows; r++) { fProxy d = A[r, col] - q[r]; s += d * d; }
                return math.sqrt(s);
            }
            else if (m == Metric.SqEuclidean)
            {
                fProxy s = (fProxy)0;
                for (int r = 0; r < mRows; r++) { fProxy d = A[r, col] - q[r]; s += d * d; }
                return s;
            }
            else if (m == Metric.Chebyshev)
            {
                fProxy s = (fProxy)0;
                for (int r = 0; r < mRows; r++)
                    s = math.max(s, math.abs(A[r, col] - q[r]));
                return s;
            }
            else if (m == Metric.Cosine)
            {
                fProxy dot = (fProxy)0, normA = (fProxy)0, normQ = (fProxy)0;
                for (int r = 0; r < mRows; r++)
                {
                    dot   += A[r, col] * q[r];
                    normA += A[r, col] * A[r, col];
                    normQ += q[r] * q[r];
                }
                fProxy denom = math.sqrt(normA * normQ);
                return denom > (fProxy)0 ? dot / denom : (fProxy)0;
            }
            else // Metric.Dot
            {
                fProxy s = (fProxy)0;
                for (int r = 0; r < mRows; r++)
                    s += A[r, col] * q[r];
                return s;
            }
        }

        // ||q||^2, needed by the Cosine branch of RowScore/ColScore. Callers that score q against
        // every row/column of A compute this ONCE (below) and pass it through the normQ overloads,
        // instead of each RowScore/ColScore call re-summing it.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static fProxy QueryNormSq(in fProxyN q)
        {
            fProxy s = (fProxy)0;
            for (int i = 0; i < q.N; i++) s += q[i] * q[i];
            return s;
        }

        /// <summary>Same as <see cref="RowScore"/>, but Cosine uses the caller-supplied ||q||^2
        /// (from <see cref="QueryNormSq"/>) instead of resumming it. normQ is ignored for every
        /// other metric.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static fProxy RowScore(in fProxyMxN A, int r, in fProxyN q, Metric m, fProxy normQ)
        {
            if (m != Metric.Cosine) return RowScore(in A, r, in q, m);

            int nCols = A.N_Cols;
            fProxy dot = (fProxy)0, normA = (fProxy)0;
            for (int c = 0; c < nCols; c++)
            {
                dot   += A[r, c] * q[c];
                normA += A[r, c] * A[r, c];
            }
            fProxy denom = math.sqrt(normA * normQ);
            return denom > (fProxy)0 ? dot / denom : (fProxy)0;
        }

        /// <summary>Column analog of the normQ overload of <see cref="RowScore"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static fProxy ColScore(in fProxyMxN A, int col, in fProxyN q, Metric m, fProxy normQ)
        {
            if (m != Metric.Cosine) return ColScore(in A, col, in q, m);

            int mRows = A.M_Rows;
            fProxy dot = (fProxy)0, normA = (fProxy)0;
            for (int r = 0; r < mRows; r++)
            {
                dot   += A[r, col] * q[r];
                normA += A[r, col] * A[r, col];
            }
            fProxy denom = math.sqrt(normA * normQ);
            return denom > (fProxy)0 ? dot / denom : (fProxy)0;
        }

        // Metric direction helpers (IsSimilarityMetric / WorstScoreForNearest / WorstScoreForFarthest /
        // IsBetterForNearest / IsBetterForFarthest) live in fProxyQueryCore: they take no fProxy
        // parameter (or return fProxy), so in the merged float+double `Query` partial they would
        // collide (CS0111) -- and IsSimilarityMetric's rule even differs from the integer variant.

        // ---- distancesToRow / distancesToColumn ---------------------------------

        /// <summary>
        /// Fills dest[i] with the distance/similarity between row i of A and query q
        /// under metric m. dest must have length A.M_Rows. q.N must equal A.N_Cols.
        /// </summary>
        public static void distancesToRow(in fProxyMxN A, in fProxyN q, Metric m, ref fProxyN dest)
        {
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("Query.distancesToRow: q.N must equal A.N_Cols");
            if (dest.N != A.M_Rows)
                throw new System.ArgumentException("Query.distancesToRow: dest.N must equal A.M_Rows");

            fProxy normQ = m == Metric.Cosine ? QueryNormSq(in q) : (fProxy)0;
            for (int r = 0; r < A.M_Rows; r++)
                dest[r] = RowScore(in A, r, in q, m, normQ);
        }

        /// <summary>
        /// Fills dest[j] with the distance/similarity between column j of A and query q
        /// under metric m. dest must have length A.N_Cols. q.N must equal A.M_Rows.
        /// Columns are strided (non-contiguous).
        /// </summary>
        public static void distancesToColumn(in fProxyMxN A, in fProxyN q, Metric m, ref fProxyN dest)
        {
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("Query.distancesToColumn: q.N must equal A.M_Rows");
            if (dest.N != A.N_Cols)
                throw new System.ArgumentException("Query.distancesToColumn: dest.N must equal A.N_Cols");

            fProxy normQ = m == Metric.Cosine ? QueryNormSq(in q) : (fProxy)0;
            for (int c = 0; c < A.N_Cols; c++)
                dest[c] = ColScore(in A, c, in q, m, normQ);
        }

        // ---- nearestRow / nearestColumn ----------------------------------------

        /// <summary>
        /// Finds the row of A most similar/closest to query q under metric m.
        /// For distance metrics (Manhattan/Euclidean/SqEuclidean/Chebyshev): nearest = min distance.
        /// For similarity metrics (Cosine/Dot): nearest = max similarity.
        /// score is in metric's own units (SqEuclidean → squared). q.N must equal A.N_Cols.
        /// </summary>
        public static void nearestRow(in fProxyMxN A, in fProxyN q, Metric m, out int index, out fProxy score)
        {
            if (A.M_Rows == 0)
                throw new System.InvalidOperationException("Query.nearestRow: matrix has no rows");
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("Query.nearestRow: q.N must equal A.N_Cols");

            fProxy best = fProxyQueryCore.WorstScoreForNearest(m);
            int bestIdx = 0;
            fProxy normQ = m == Metric.Cosine ? QueryNormSq(in q) : (fProxy)0;
            for (int r = 0; r < A.M_Rows; r++)
            {
                fProxy s = RowScore(in A, r, in q, m, normQ);
                if (fProxyQueryCore.IsBetterForNearest(s, best, m)) { best = s; bestIdx = r; }
            }
            index = bestIdx;
            score = best;
        }

        /// <summary>
        /// Finds the column of A most similar/closest to query q under metric m.
        /// q.N must equal A.M_Rows. Columns are strided.
        /// </summary>
        public static void nearestColumn(in fProxyMxN A, in fProxyN q, Metric m, out int index, out fProxy score)
        {
            if (A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.nearestColumn: matrix has no columns");
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("Query.nearestColumn: q.N must equal A.M_Rows");

            fProxy best = fProxyQueryCore.WorstScoreForNearest(m);
            int bestIdx = 0;
            fProxy normQ = m == Metric.Cosine ? QueryNormSq(in q) : (fProxy)0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                fProxy s = ColScore(in A, c, in q, m, normQ);
                if (fProxyQueryCore.IsBetterForNearest(s, best, m)) { best = s; bestIdx = c; }
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
        public static void farthestRow(in fProxyMxN A, in fProxyN q, Metric m, out int index, out fProxy score)
        {
            if (A.M_Rows == 0)
                throw new System.InvalidOperationException("Query.farthestRow: matrix has no rows");
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("Query.farthestRow: q.N must equal A.N_Cols");

            fProxy worst = fProxyQueryCore.WorstScoreForFarthest(m);
            int worstIdx = 0;
            fProxy normQ = m == Metric.Cosine ? QueryNormSq(in q) : (fProxy)0;
            for (int r = 0; r < A.M_Rows; r++)
            {
                fProxy s = RowScore(in A, r, in q, m, normQ);
                if (fProxyQueryCore.IsBetterForFarthest(s, worst, m)) { worst = s; worstIdx = r; }
            }
            index = worstIdx;
            score = worst;
        }

        /// <summary>
        /// Finds the column of A most dissimilar/farthest from query q under metric m.
        /// q.N must equal A.M_Rows. Columns are strided.
        /// </summary>
        public static void farthestColumn(in fProxyMxN A, in fProxyN q, Metric m, out int index, out fProxy score)
        {
            if (A.N_Cols == 0)
                throw new System.InvalidOperationException("Query.farthestColumn: matrix has no columns");
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("Query.farthestColumn: q.N must equal A.M_Rows");

            fProxy worst = fProxyQueryCore.WorstScoreForFarthest(m);
            int worstIdx = 0;
            fProxy normQ = m == Metric.Cosine ? QueryNormSq(in q) : (fProxy)0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                fProxy s = ColScore(in A, c, in q, m, normQ);
                if (fProxyQueryCore.IsBetterForFarthest(s, worst, m)) { worst = s; worstIdx = c; }
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
        public static int countWithinRadius(in fProxyMxN A, in fProxyN q, fProxy r, Metric m)
        {
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("Query.countWithinRadius: q.N must equal A.N_Cols");

            bool sim = fProxyQueryCore.IsSimilarityMetric(m);
            int count = 0;
            fProxy normQ = m == Metric.Cosine ? QueryNormSq(in q) : (fProxy)0;
            for (int row = 0; row < A.M_Rows; row++)
            {
                fProxy s = RowScore(in A, row, in q, m, normQ);
                if (sim ? s >= r : s <= r) count++;
            }
            return count;
        }

        /// <summary>
        /// Returns the count of columns with distance/similarity to q within radius r.
        /// Columns are strided. q.N must equal A.M_Rows.
        /// </summary>
        public static int countWithinColumnRadius(in fProxyMxN A, in fProxyN q, fProxy r, Metric m)
        {
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("Query.countWithinColumnRadius: q.N must equal A.M_Rows");

            bool sim = fProxyQueryCore.IsSimilarityMetric(m);
            int count = 0;
            fProxy normQ = m == Metric.Cosine ? QueryNormSq(in q) : (fProxy)0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                fProxy s = ColScore(in A, c, in q, m, normQ);
                if (sim ? s >= r : s <= r) count++;
            }
            return count;
        }

        // ---- kNearest/kFarthest rows/columns with Indices buffer ---

        /// <summary>
        /// Finds the k nearest rows to query q. idx and scores must both have length >= k.
        /// Fills idx[0..count) and scores[0..count) sorted best-first.
        /// Returns min(k, A.M_Rows). Uses bounded insertion sort (O(M·k)) — optimal for small k.
        /// q.N must equal A.N_Cols.
        /// </summary>
        public static int kNearestRows(in fProxyMxN A, in fProxyN q, int k, Metric m, ref Indices idx, ref fProxyN scores)
        {
            if (A.M_Rows == 0 || k <= 0) return 0;
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("Query.kNearestRows: q.N must equal A.N_Cols");
            if (idx.N < k)
                throw new System.ArgumentException("Query.kNearestRows: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("Query.kNearestRows: scores.N must be >= k");

            int clampedK = math.min(k, A.M_Rows);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;
            fProxy normQ = m == Metric.Cosine ? QueryNormSq(in q) : (fProxy)0;

            for (int r = 0; r < A.M_Rows; r++)
            {
                fProxy s = Query.RowScore(in A, r, in q, m, normQ);
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
            return clampedK;
        }

        /// <summary>
        /// Finds the k nearest columns to query q. idx and scores must both have length >= k.
        /// Returns min(k, A.N_Cols). q.N must equal A.M_Rows. Columns are strided.
        /// </summary>
        public static int kNearestColumns(in fProxyMxN A, in fProxyN q, int k, Metric m, ref Indices idx, ref fProxyN scores)
        {
            if (A.N_Cols == 0 || k <= 0) return 0;
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("Query.kNearestColumns: q.N must equal A.M_Rows");
            if (idx.N < k)
                throw new System.ArgumentException("Query.kNearestColumns: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("Query.kNearestColumns: scores.N must be >= k");

            int clampedK = math.min(k, A.N_Cols);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;
            fProxy normQ = m == Metric.Cosine ? QueryNormSq(in q) : (fProxy)0;

            for (int c = 0; c < A.N_Cols; c++)
            {
                fProxy s = Query.ColScore(in A, c, in q, m, normQ);
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
            return clampedK;
        }

        /// <summary>
        /// Finds the k farthest rows from query q. idx and scores must have length >= k.
        /// Returns min(k, A.M_Rows). Sorted worst-first (highest distance / lowest similarity).
        /// </summary>
        public static int kFarthestRows(in fProxyMxN A, in fProxyN q, int k, Metric m, ref Indices idx, ref fProxyN scores)
        {
            if (A.M_Rows == 0 || k <= 0) return 0;
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("Query.kFarthestRows: q.N must equal A.N_Cols");
            if (idx.N < k)
                throw new System.ArgumentException("Query.kFarthestRows: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("Query.kFarthestRows: scores.N must be >= k");

            int clampedK = math.min(k, A.M_Rows);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;
            fProxy normQ = m == Metric.Cosine ? QueryNormSq(in q) : (fProxy)0;

            for (int r = 0; r < A.M_Rows; r++)
            {
                fProxy s = Query.RowScore(in A, r, in q, m, normQ);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && (sim ? s < scores[ins - 1] : s > scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = r; count++;
                }
                else
                {
                    fProxy kth = scores[clampedK - 1];
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
        /// Finds the k farthest columns from query q. Returns min(k, A.N_Cols).
        /// q.N must equal A.M_Rows. Columns are strided.
        /// </summary>
        public static int kFarthestColumns(in fProxyMxN A, in fProxyN q, int k, Metric m, ref Indices idx, ref fProxyN scores)
        {
            if (A.N_Cols == 0 || k <= 0) return 0;
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("Query.kFarthestColumns: q.N must equal A.M_Rows");
            if (idx.N < k)
                throw new System.ArgumentException("Query.kFarthestColumns: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("Query.kFarthestColumns: scores.N must be >= k");

            int clampedK = math.min(k, A.N_Cols);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;
            fProxy normQ = m == Metric.Cosine ? QueryNormSq(in q) : (fProxy)0;

            for (int c = 0; c < A.N_Cols; c++)
            {
                fProxy s = Query.ColScore(in A, c, in q, m, normQ);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && (sim ? s < scores[ins - 1] : s > scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = c; count++;
                }
                else
                {
                    fProxy kth = scores[clampedK - 1];
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
        /// For distance metrics: score &lt;= r. For similarity metrics: score >= r.
        /// Returns count. idx must be sized >= A.M_Rows (worst case).
        /// q.N must equal A.N_Cols.
        /// </summary>
        public static int rowsWithinRadius(in fProxyMxN A, in fProxyN q, fProxy r, Metric m, ref Indices idx)
        {
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("Query.rowsWithinRadius: q.N must equal A.N_Cols");
            if (idx.N < A.M_Rows)
                throw new System.ArgumentException("Query.rowsWithinRadius: idx.N must be >= A.M_Rows (worst case)");

            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;
            fProxy normQ = m == Metric.Cosine ? QueryNormSq(in q) : (fProxy)0;
            for (int row = 0; row < A.M_Rows; row++)
            {
                fProxy s = Query.RowScore(in A, row, in q, m, normQ);
                if (sim ? s >= r : s <= r) idx[count++] = row;
            }
            return count;
        }

        /// <summary>
        /// Fills idx[0..count) with indices of columns within radius r of query q.
        /// Returns count. idx must be sized >= A.N_Cols. q.N must equal A.M_Rows.
        /// Columns are strided.
        /// </summary>
        public static int columnsWithinRadius(in fProxyMxN A, in fProxyN q, fProxy r, Metric m, ref Indices idx)
        {
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("Query.columnsWithinRadius: q.N must equal A.M_Rows");
            if (idx.N < A.N_Cols)
                throw new System.ArgumentException("Query.columnsWithinRadius: idx.N must be >= A.N_Cols (worst case)");

            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;
            fProxy normQ = m == Metric.Cosine ? QueryNormSq(in q) : (fProxy)0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                fProxy s = Query.ColScore(in A, c, in q, m, normQ);
                if (sim ? s >= r : s <= r) idx[count++] = c;
            }
            return count;
        }

        // ---- nonzero with Indices buffer ---

        /// <summary>
        /// Fills idx[0..count) with flat indices of elements in x with |x[i]| > tolerance.
        /// Returns count. idx must be sized >= x.Data.Length (worst case).
        /// Generic over fProxyN and fProxyMxN.
        /// </summary>
        public static int nonzero<T>(in T x, fProxy tolerance, ref Indices idx)
            where T : unmanaged, IUnsafefProxyArray
        {
            if (idx.N < x.Data.Length)
                throw new System.ArgumentException("Query.nonzero: idx.N must be >= x.Data.Length (worst case)");

            int count = 0;
            for (int i = 0; i < x.Data.Length; i++)
                if (math.abs(x.Data[i]) > tolerance) idx[count++] = i;
            return count;
        }

        // -------------------------------------------------------------------------
        // GROUP 4 — VALUE / MASK SEARCH
        // -------------------------------------------------------------------------

        /// <summary>
        /// Returns the flat index of the first element in x equal to target
        /// (within tolerance: |x[i] - target| &lt;= tolerance). Returns -1 if not found.
        /// Generic over vec + matrix flat data. (Like Excel MATCH.)
        /// </summary>
        public static int findValue<T>(in T x, fProxy target, fProxy tolerance)
            where T : unmanaged, IUnsafefProxyArray
        {
            for (int i = 0; i < x.Data.Length; i++)
            {
                if (math.abs(x.Data[i] - target) <= tolerance)
                    return i;
            }
            return -1;
        }

        /// <summary>
        /// Returns the count of elements in x with absolute value &gt; tolerance.
        /// Zero-alloc; use with nonzero (ref Indices) for the full index list.
        /// </summary>
        public static int countNonzero<T>(in T x, fProxy tolerance)
            where T : unmanaged, IUnsafefProxyArray
        {
            int count = 0;
            for (int i = 0; i < x.Data.Length; i++)
            {
                if (math.abs(x.Data[i]) > tolerance)
                    count++;
            }
            return count;
        }
    }
}
