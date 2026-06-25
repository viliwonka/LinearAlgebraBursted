#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using Unity.Mathematics;
using Unity.Burst;

namespace LinearAlgebra
{
    // Cross-type QueryOP methods: fProxy (float/double) search with intN index buffers.
    // Lives in Source/OP/ (not Generated) because the codegen system cannot express
    // fProxy-typed parameters alongside intN index buffers in a single template file
    // (intN is a generated type not available in the TemplateSource assembly).
    // Hand-maintained: float and double variants are explicit partial class extensions.
    //
    // Covered here:
    //   Group 1 — per-axis rowArgMin/Max / colArgMin/Max (value+index and index-only forms)
    //   Group 3 — rowsWithinRadius/Column (ref intN buffer form)
    //             kNearestRows/Columns + kFarthestRows/Columns (ref intN + ref fProxy buffers)
    //   Group 4 — nonzero<T> (ref intN buffer form)
    //   BoolAnalysis bridge — whichTrue (ref intN, for boolN and boolMxN)
    //   Arena wrappers — count-pass + exact-alloc intN returning forms for:
    //             nonzero, rowsWithinRadius/Column, kNearestRows/Columns, whichTrue

    // ============================================================
    // FLOAT VARIANTS
    // ============================================================

    public static partial class floatQueryOP
    {
        // ---- Group 1: per-axis row/col arg-min/max ---

        /// <summary>
        /// For each row i of A, writes the column index of the minimum element into
        /// colIndexPerRow[i] and the minimum value into valPerRow[i].
        /// Returns A.M_Rows. Both buffers must have length A.M_Rows.
        /// </summary>
        public static int rowArgMin(in floatMxN A, ref intN colIndexPerRow, ref floatN valPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.rowArgMin: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowArgMin: colIndexPerRow.N must equal A.M_Rows");
            if (valPerRow.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowArgMin: valPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                float best = A[r, 0];
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
        public static int rowArgMin(in floatMxN A, ref intN colIndexPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.rowArgMin: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowArgMin: colIndexPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                float best = A[r, 0];
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
        public static int rowArgMax(in floatMxN A, ref intN colIndexPerRow, ref floatN valPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.rowArgMax: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowArgMax: colIndexPerRow.N must equal A.M_Rows");
            if (valPerRow.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowArgMax: valPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                float best = A[r, 0];
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
        public static int rowArgMax(in floatMxN A, ref intN colIndexPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.rowArgMax: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowArgMax: colIndexPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                float best = A[r, 0];
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
        public static int colArgMin(in floatMxN A, ref intN rowIndexPerCol, ref floatN valPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.colArgMin: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.colArgMin: rowIndexPerCol.N must equal A.N_Cols");
            if (valPerCol.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.colArgMin: valPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                float best = A[0, c];
                int bestR = 0;
                for (int r = 1; r < A.M_Rows; r++)
                    if (A[r, c] < best) { best = A[r, c]; bestR = r; }
                rowIndexPerCol[c] = bestR;
                valPerCol[c] = best;
            }
            return A.N_Cols;
        }

        /// <summary>Index-only form of colArgMin. Returns A.N_Cols.</summary>
        public static int colArgMin(in floatMxN A, ref intN rowIndexPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.colArgMin: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.colArgMin: rowIndexPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                float best = A[0, c];
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
        public static int colArgMax(in floatMxN A, ref intN rowIndexPerCol, ref floatN valPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.colArgMax: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.colArgMax: rowIndexPerCol.N must equal A.N_Cols");
            if (valPerCol.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.colArgMax: valPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                float best = A[0, c];
                int bestR = 0;
                for (int r = 1; r < A.M_Rows; r++)
                    if (A[r, c] > best) { best = A[r, c]; bestR = r; }
                rowIndexPerCol[c] = bestR;
                valPerCol[c] = best;
            }
            return A.N_Cols;
        }

        /// <summary>Index-only form of colArgMax. Returns A.N_Cols.</summary>
        public static int colArgMax(in floatMxN A, ref intN rowIndexPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.colArgMax: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.colArgMax: rowIndexPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                float best = A[0, c];
                int bestR = 0;
                for (int r = 1; r < A.M_Rows; r++)
                    if (A[r, c] > best) { best = A[r, c]; bestR = r; }
                rowIndexPerCol[c] = bestR;
            }
            return A.N_Cols;
        }

        // ---- Group 3: kNearest/kFarthest rows/columns with intN buffers ---

        /// <summary>
        /// Finds the k nearest rows to query q. idx and scores must both have length >= k.
        /// Fills idx[0..count) and scores[0..count) sorted best-first.
        /// Returns min(k, A.M_Rows). Uses bounded insertion sort (O(M·k)) — optimal for small k.
        /// q.N must equal A.N_Cols.
        /// </summary>
        public static int kNearestRows(in floatMxN A, in floatN q, int k, Metric m, ref intN idx, ref floatN scores)
        {
            if (A.M_Rows == 0 || k <= 0) return 0;
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.kNearestRows: q.N must equal A.N_Cols");
            if (idx.N < k)
                throw new System.ArgumentException("QueryOP.kNearestRows: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("QueryOP.kNearestRows: scores.N must be >= k");

            int clampedK = math.min(k, A.M_Rows);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;

            for (int r = 0; r < A.M_Rows; r++)
            {
                float s = floatQueryOP.RowScore(in A, r, in q, m);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && (sim ? s > scores[ins - 1] : s < scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = r; count++;
                }
                else
                {
                    float kth = scores[clampedK - 1];
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
        public static int kNearestColumns(in floatMxN A, in floatN q, int k, Metric m, ref intN idx, ref floatN scores)
        {
            if (A.N_Cols == 0 || k <= 0) return 0;
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.kNearestColumns: q.N must equal A.M_Rows");
            if (idx.N < k)
                throw new System.ArgumentException("QueryOP.kNearestColumns: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("QueryOP.kNearestColumns: scores.N must be >= k");

            int clampedK = math.min(k, A.N_Cols);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;

            for (int c = 0; c < A.N_Cols; c++)
            {
                float s = floatQueryOP.ColScore(in A, c, in q, m);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && (sim ? s > scores[ins - 1] : s < scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = c; count++;
                }
                else
                {
                    float kth = scores[clampedK - 1];
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
        public static int kFarthestRows(in floatMxN A, in floatN q, int k, Metric m, ref intN idx, ref floatN scores)
        {
            if (A.M_Rows == 0 || k <= 0) return 0;
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.kFarthestRows: q.N must equal A.N_Cols");
            if (idx.N < k)
                throw new System.ArgumentException("QueryOP.kFarthestRows: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("QueryOP.kFarthestRows: scores.N must be >= k");

            int clampedK = math.min(k, A.M_Rows);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;

            for (int r = 0; r < A.M_Rows; r++)
            {
                float s = floatQueryOP.RowScore(in A, r, in q, m);
                if (count < clampedK)
                {
                    // Sorted descending for distance, ascending for similarity.
                    int ins = count;
                    while (ins > 0 && (sim ? s < scores[ins - 1] : s > scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = r; count++;
                }
                else
                {
                    float kth = scores[clampedK - 1];
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
        public static int kFarthestColumns(in floatMxN A, in floatN q, int k, Metric m, ref intN idx, ref floatN scores)
        {
            if (A.N_Cols == 0 || k <= 0) return 0;
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.kFarthestColumns: q.N must equal A.M_Rows");
            if (idx.N < k)
                throw new System.ArgumentException("QueryOP.kFarthestColumns: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("QueryOP.kFarthestColumns: scores.N must be >= k");

            int clampedK = math.min(k, A.N_Cols);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;

            for (int c = 0; c < A.N_Cols; c++)
            {
                float s = floatQueryOP.ColScore(in A, c, in q, m);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && (sim ? s < scores[ins - 1] : s > scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = c; count++;
                }
                else
                {
                    float kth = scores[clampedK - 1];
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
        public static int rowsWithinRadius(in floatMxN A, in floatN q, float r, Metric m, ref intN idx)
        {
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.rowsWithinRadius: q.N must equal A.N_Cols");
            if (idx.N < A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowsWithinRadius: idx.N must be >= A.M_Rows (worst case)");

            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;
            for (int row = 0; row < A.M_Rows; row++)
            {
                float s = floatQueryOP.RowScore(in A, row, in q, m);
                if (sim ? s >= r : s <= r) idx[count++] = row;
            }
            return count;
        }

        /// <summary>
        /// Fills idx[0..count) with indices of columns within radius r of query q.
        /// Returns count. idx must be sized >= A.N_Cols. q.N must equal A.M_Rows.
        /// Columns are strided.
        /// </summary>
        public static int columnsWithinRadius(in floatMxN A, in floatN q, float r, Metric m, ref intN idx)
        {
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.columnsWithinRadius: q.N must equal A.M_Rows");
            if (idx.N < A.N_Cols)
                throw new System.ArgumentException("QueryOP.columnsWithinRadius: idx.N must be >= A.N_Cols (worst case)");

            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                float s = floatQueryOP.ColScore(in A, c, in q, m);
                if (sim ? s >= r : s <= r) idx[count++] = c;
            }
            return count;
        }

        // ---- Group 4: nonzero with intN buffer ---

        /// <summary>
        /// Fills idx[0..count) with flat indices of elements in x with |x[i]| > tol.
        /// Returns count. idx must be sized >= x.Data.Length (worst case).
        /// Generic over floatN and floatMxN.
        /// </summary>
        public static int nonzero<T>(in T x, float tol, ref intN idx)
            where T : unmanaged, IUnsafefloatArray
        {
            if (idx.N < x.Data.Length)
                throw new System.ArgumentException("QueryOP.nonzero: idx.N must be >= x.Data.Length (worst case)");

            int count = 0;
            for (int i = 0; i < x.Data.Length; i++)
                if (math.abs(x.Data[i]) > tol) idx[count++] = i;
            return count;
        }
    }

    // ============================================================
    // DOUBLE VARIANTS
    // ============================================================

    public static partial class doubleQueryOP
    {
        // ---- Group 1: per-axis row/col arg-min/max ---

        /// <summary>
        /// For each row i of A, writes the column index of the minimum element into
        /// colIndexPerRow[i] and the minimum value into valPerRow[i].
        /// Returns A.M_Rows. Both buffers must have length A.M_Rows.
        /// </summary>
        public static int rowArgMin(in doubleMxN A, ref intN colIndexPerRow, ref doubleN valPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.rowArgMin: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowArgMin: colIndexPerRow.N must equal A.M_Rows");
            if (valPerRow.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowArgMin: valPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                double best = A[r, 0];
                int bestC = 0;
                for (int c = 1; c < A.N_Cols; c++)
                    if (A[r, c] < best) { best = A[r, c]; bestC = c; }
                colIndexPerRow[r] = bestC;
                valPerRow[r] = best;
            }
            return A.M_Rows;
        }

        /// <summary>Index-only form of rowArgMin. Returns A.M_Rows.</summary>
        public static int rowArgMin(in doubleMxN A, ref intN colIndexPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.rowArgMin: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowArgMin: colIndexPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                double best = A[r, 0];
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
        /// Returns A.M_Rows.
        /// </summary>
        public static int rowArgMax(in doubleMxN A, ref intN colIndexPerRow, ref doubleN valPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.rowArgMax: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowArgMax: colIndexPerRow.N must equal A.M_Rows");
            if (valPerRow.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowArgMax: valPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                double best = A[r, 0];
                int bestC = 0;
                for (int c = 1; c < A.N_Cols; c++)
                    if (A[r, c] > best) { best = A[r, c]; bestC = c; }
                colIndexPerRow[r] = bestC;
                valPerRow[r] = best;
            }
            return A.M_Rows;
        }

        /// <summary>Index-only form of rowArgMax. Returns A.M_Rows.</summary>
        public static int rowArgMax(in doubleMxN A, ref intN colIndexPerRow)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.rowArgMax: empty matrix");
            if (colIndexPerRow.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowArgMax: colIndexPerRow.N must equal A.M_Rows");

            for (int r = 0; r < A.M_Rows; r++)
            {
                double best = A[r, 0];
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
        /// Returns A.N_Cols. Columns are strided.
        /// </summary>
        public static int colArgMin(in doubleMxN A, ref intN rowIndexPerCol, ref doubleN valPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.colArgMin: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.colArgMin: rowIndexPerCol.N must equal A.N_Cols");
            if (valPerCol.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.colArgMin: valPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                double best = A[0, c];
                int bestR = 0;
                for (int r = 1; r < A.M_Rows; r++)
                    if (A[r, c] < best) { best = A[r, c]; bestR = r; }
                rowIndexPerCol[c] = bestR;
                valPerCol[c] = best;
            }
            return A.N_Cols;
        }

        /// <summary>Index-only form of colArgMin. Returns A.N_Cols.</summary>
        public static int colArgMin(in doubleMxN A, ref intN rowIndexPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.colArgMin: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.colArgMin: rowIndexPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                double best = A[0, c];
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
        /// Returns A.N_Cols.
        /// </summary>
        public static int colArgMax(in doubleMxN A, ref intN rowIndexPerCol, ref doubleN valPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.colArgMax: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.colArgMax: rowIndexPerCol.N must equal A.N_Cols");
            if (valPerCol.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.colArgMax: valPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                double best = A[0, c];
                int bestR = 0;
                for (int r = 1; r < A.M_Rows; r++)
                    if (A[r, c] > best) { best = A[r, c]; bestR = r; }
                rowIndexPerCol[c] = bestR;
                valPerCol[c] = best;
            }
            return A.N_Cols;
        }

        /// <summary>Index-only form of colArgMax. Returns A.N_Cols.</summary>
        public static int colArgMax(in doubleMxN A, ref intN rowIndexPerCol)
        {
            if (A.M_Rows == 0 || A.N_Cols == 0)
                throw new System.InvalidOperationException("QueryOP.colArgMax: empty matrix");
            if (rowIndexPerCol.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.colArgMax: rowIndexPerCol.N must equal A.N_Cols");

            for (int c = 0; c < A.N_Cols; c++)
            {
                double best = A[0, c];
                int bestR = 0;
                for (int r = 1; r < A.M_Rows; r++)
                    if (A[r, c] > best) { best = A[r, c]; bestR = r; }
                rowIndexPerCol[c] = bestR;
            }
            return A.N_Cols;
        }

        // ---- Group 3: kNearest/kFarthest rows/columns ---

        /// <summary>
        /// Finds the k nearest rows to query q. idx and scores must both have length >= k.
        /// Returns min(k, A.M_Rows). q.N must equal A.N_Cols.
        /// </summary>
        public static int kNearestRows(in doubleMxN A, in doubleN q, int k, Metric m, ref intN idx, ref doubleN scores)
        {
            if (A.M_Rows == 0 || k <= 0) return 0;
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.kNearestRows: q.N must equal A.N_Cols");
            if (idx.N < k)
                throw new System.ArgumentException("QueryOP.kNearestRows: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("QueryOP.kNearestRows: scores.N must be >= k");

            int clampedK = math.min(k, A.M_Rows);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;

            for (int r = 0; r < A.M_Rows; r++)
            {
                double s = doubleQueryOP.RowScore(in A, r, in q, m);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && (sim ? s > scores[ins - 1] : s < scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = r; count++;
                }
                else
                {
                    double kth = scores[clampedK - 1];
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
        /// Finds the k nearest columns to query q. Returns min(k, A.N_Cols).
        /// q.N must equal A.M_Rows. Columns are strided.
        /// </summary>
        public static int kNearestColumns(in doubleMxN A, in doubleN q, int k, Metric m, ref intN idx, ref doubleN scores)
        {
            if (A.N_Cols == 0 || k <= 0) return 0;
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.kNearestColumns: q.N must equal A.M_Rows");
            if (idx.N < k)
                throw new System.ArgumentException("QueryOP.kNearestColumns: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("QueryOP.kNearestColumns: scores.N must be >= k");

            int clampedK = math.min(k, A.N_Cols);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;

            for (int c = 0; c < A.N_Cols; c++)
            {
                double s = doubleQueryOP.ColScore(in A, c, in q, m);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && (sim ? s > scores[ins - 1] : s < scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = c; count++;
                }
                else
                {
                    double kth = scores[clampedK - 1];
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
        /// Finds the k farthest rows from query q. Returns min(k, A.M_Rows).
        /// </summary>
        public static int kFarthestRows(in doubleMxN A, in doubleN q, int k, Metric m, ref intN idx, ref doubleN scores)
        {
            if (A.M_Rows == 0 || k <= 0) return 0;
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.kFarthestRows: q.N must equal A.N_Cols");
            if (idx.N < k)
                throw new System.ArgumentException("QueryOP.kFarthestRows: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("QueryOP.kFarthestRows: scores.N must be >= k");

            int clampedK = math.min(k, A.M_Rows);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;

            for (int r = 0; r < A.M_Rows; r++)
            {
                double s = doubleQueryOP.RowScore(in A, r, in q, m);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && (sim ? s < scores[ins - 1] : s > scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = r; count++;
                }
                else
                {
                    double kth = scores[clampedK - 1];
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
        /// </summary>
        public static int kFarthestColumns(in doubleMxN A, in doubleN q, int k, Metric m, ref intN idx, ref doubleN scores)
        {
            if (A.N_Cols == 0 || k <= 0) return 0;
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.kFarthestColumns: q.N must equal A.M_Rows");
            if (idx.N < k)
                throw new System.ArgumentException("QueryOP.kFarthestColumns: idx.N must be >= k");
            if (scores.N < k)
                throw new System.ArgumentException("QueryOP.kFarthestColumns: scores.N must be >= k");

            int clampedK = math.min(k, A.N_Cols);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;

            for (int c = 0; c < A.N_Cols; c++)
            {
                double s = doubleQueryOP.ColScore(in A, c, in q, m);
                if (count < clampedK)
                {
                    int ins = count;
                    while (ins > 0 && (sim ? s < scores[ins - 1] : s > scores[ins - 1])) ins--;
                    for (int j = count; j > ins; j--) { scores[j] = scores[j - 1]; idx[j] = idx[j - 1]; }
                    scores[ins] = s; idx[ins] = c; count++;
                }
                else
                {
                    double kth = scores[clampedK - 1];
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
        /// Returns count. idx must be sized >= A.M_Rows. q.N must equal A.N_Cols.
        /// </summary>
        public static int rowsWithinRadius(in doubleMxN A, in doubleN q, double r, Metric m, ref intN idx)
        {
            if (q.N != A.N_Cols)
                throw new System.ArgumentException("QueryOP.rowsWithinRadius: q.N must equal A.N_Cols");
            if (idx.N < A.M_Rows)
                throw new System.ArgumentException("QueryOP.rowsWithinRadius: idx.N must be >= A.M_Rows (worst case)");

            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;
            for (int row = 0; row < A.M_Rows; row++)
            {
                double s = doubleQueryOP.RowScore(in A, row, in q, m);
                if (sim ? s >= r : s <= r) idx[count++] = row;
            }
            return count;
        }

        /// <summary>
        /// Fills idx[0..count) with indices of columns within radius r of query q.
        /// Returns count. idx must be sized >= A.N_Cols. q.N must equal A.M_Rows.
        /// </summary>
        public static int columnsWithinRadius(in doubleMxN A, in doubleN q, double r, Metric m, ref intN idx)
        {
            if (q.N != A.M_Rows)
                throw new System.ArgumentException("QueryOP.columnsWithinRadius: q.N must equal A.M_Rows");
            if (idx.N < A.N_Cols)
                throw new System.ArgumentException("QueryOP.columnsWithinRadius: idx.N must be >= A.N_Cols (worst case)");

            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int count = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                double s = doubleQueryOP.ColScore(in A, c, in q, m);
                if (sim ? s >= r : s <= r) idx[count++] = c;
            }
            return count;
        }

        // ---- Group 4: nonzero with intN buffer ---

        /// <summary>
        /// Fills idx[0..count) with flat indices of elements in x with |x[i]| > tol.
        /// Returns count. idx must be sized >= x.Data.Length. Generic over doubleN and doubleMxN.
        /// </summary>
        public static int nonzero<T>(in T x, double tol, ref intN idx)
            where T : unmanaged, IUnsafedoubleArray
        {
            if (idx.N < x.Data.Length)
                throw new System.ArgumentException("QueryOP.nonzero: idx.N must be >= x.Data.Length (worst case)");

            int count = 0;
            for (int i = 0; i < x.Data.Length; i++)
                if (math.abs(x.Data[i]) > tol) idx[count++] = i;
            return count;
        }
    }

    // ============================================================
    // BOOL-ONLY BRIDGE (whichTrue — fills intN from boolN/boolMxN)
    // ============================================================

    public static partial class BoolAnalysis
    {
        /// <summary>
        /// Fills idx[0..count) with the flat indices of true elements in mask.
        /// Returns count. idx must be sized >= mask.N (worst case).
        /// Use countTrue first if you want to allocate an exact-sized buffer.
        /// </summary>
        public static int whichTrue(in boolN mask, ref intN idx)
        {
            if (idx.N < mask.N)
                throw new System.ArgumentException("BoolAnalysis.whichTrue: idx.N must be >= mask.N");
            int count = 0;
            for (int i = 0; i < mask.N; i++)
                if (mask.Data[i]) idx[count++] = i;
            return count;
        }

        /// <summary>
        /// Matrix overload: fills idx[0..count) with flat indices of true elements in mask.
        /// idx must be sized >= mask.M_Rows * mask.N_Cols.
        /// </summary>
        public static int whichTrue(in boolMxN mask, ref intN idx)
        {
            int total = mask.M_Rows * mask.N_Cols;
            if (idx.N < total)
                throw new System.ArgumentException("BoolAnalysis.whichTrue: idx.N must be >= mask total size");
            int count = 0;
            for (int i = 0; i < total; i++)
                if (mask.Data[i]) idx[count++] = i;
            return count;
        }
    }

    // ============================================================
    // ARENA WRAPPERS — count-pass + exact-alloc intN
    // ============================================================

    public static partial class ArenaExtensions
    {
        // ---- nonzero (float) ----

        /// <summary>
        /// Two-pass: count nonzero elements, allocate exact-sized intN, fill indices.
        /// Returns the allocated intN (length = count).
        /// </summary>
        public static intN floatNonzeroIndices<T>(this ref Arena arena, in T x, float tol)
            where T : unmanaged, IUnsafefloatArray
        {
            int count = floatQueryOP.countNonzero(in x, tol);
            if (count == 0) return arena.intVec(0, true);
            var idx = arena.intVec(count);
            int written = 0;
            for (int i = 0; i < x.Data.Length; i++)
                if (math.abs(x.Data[i]) > tol) idx[written++] = i;
            return idx;
        }

        // ---- nonzero (double) ----

        /// <summary>
        /// Two-pass: count nonzero elements, allocate exact-sized intN, fill indices.
        /// Returns the allocated intN (length = count).
        /// </summary>
        public static intN doubleNonzeroIndices<T>(this ref Arena arena, in T x, double tol)
            where T : unmanaged, IUnsafedoubleArray
        {
            int count = doubleQueryOP.countNonzero(in x, tol);
            if (count == 0) return arena.intVec(0, true);
            var idx = arena.intVec(count);
            int written = 0;
            for (int i = 0; i < x.Data.Length; i++)
                if (math.abs(x.Data[i]) > tol) idx[written++] = i;
            return idx;
        }

        // ---- rowsWithinRadius (float) ----

        /// <summary>
        /// Two-pass: count + exact-alloc intN of row indices within radius r (float).
        /// </summary>
        public static intN floatRowsWithinRadius(this ref Arena arena, in floatMxN A, in floatN q, float r, Metric m)
        {
            int count = floatQueryOP.countWithinRadius(in A, in q, r, m);
            if (count == 0) return arena.intVec(0, true);
            var idx = arena.intVec(count);
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
        /// Two-pass: count + exact-alloc intN of column indices within radius r (float).
        /// </summary>
        public static intN floatColumnsWithinRadius(this ref Arena arena, in floatMxN A, in floatN q, float r, Metric m)
        {
            int count = floatQueryOP.countWithinColumnRadius(in A, in q, r, m);
            if (count == 0) return arena.intVec(0, true);
            var idx = arena.intVec(count);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int written = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                float s = floatQueryOP.ColScore(in A, c, in q, m);
                if (sim ? s >= r : s <= r) idx[written++] = c;
            }
            return idx;
        }

        // ---- rowsWithinRadius (double) ----

        /// <summary>
        /// Two-pass: count + exact-alloc intN of row indices within radius r (double).
        /// </summary>
        public static intN doubleRowsWithinRadius(this ref Arena arena, in doubleMxN A, in doubleN q, double r, Metric m)
        {
            int count = doubleQueryOP.countWithinRadius(in A, in q, r, m);
            if (count == 0) return arena.intVec(0, true);
            var idx = arena.intVec(count);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int written = 0;
            for (int row = 0; row < A.M_Rows; row++)
            {
                double s = doubleQueryOP.RowScore(in A, row, in q, m);
                if (sim ? s >= r : s <= r) idx[written++] = row;
            }
            return idx;
        }

        /// <summary>
        /// Two-pass: count + exact-alloc intN of column indices within radius r (double).
        /// </summary>
        public static intN doubleColumnsWithinRadius(this ref Arena arena, in doubleMxN A, in doubleN q, double r, Metric m)
        {
            int count = doubleQueryOP.countWithinColumnRadius(in A, in q, r, m);
            if (count == 0) return arena.intVec(0, true);
            var idx = arena.intVec(count);
            bool sim = m == Metric.Cosine || m == Metric.Dot;
            int written = 0;
            for (int c = 0; c < A.N_Cols; c++)
            {
                double s = doubleQueryOP.ColScore(in A, c, in q, m);
                if (sim ? s >= r : s <= r) idx[written++] = c;
            }
            return idx;
        }

        // ---- kNearestRows / kNearestColumns (float) ----

        /// <summary>
        /// Allocates clamped-k intN + floatN from arena, fills via kNearestRows.
        /// Returns idx; scores and count are out params. count = min(k, A.M_Rows).
        /// </summary>
        public static intN floatKNearestRows(this ref Arena arena, in floatMxN A, in floatN q, int k, Metric m, out floatN scores, out int count)
        {
            int clampedK = math.min(k, A.M_Rows);
            if (clampedK <= 0) { scores = A.floatVec(0, true); count = 0; return arena.intVec(0, true); }
            var idx = arena.intVec(clampedK);
            scores = A.floatVec(clampedK);
            count = floatQueryOP.kNearestRows(in A, in q, k, m, ref idx, ref scores);
            return idx;
        }

        /// <summary>
        /// Allocates clamped-k intN + floatN from arena, fills via kNearestColumns.
        /// </summary>
        public static intN floatKNearestColumns(this ref Arena arena, in floatMxN A, in floatN q, int k, Metric m, out floatN scores, out int count)
        {
            int clampedK = math.min(k, A.N_Cols);
            if (clampedK <= 0) { scores = A.floatVec(0, true); count = 0; return arena.intVec(0, true); }
            var idx = arena.intVec(clampedK);
            scores = A.floatVec(clampedK);
            count = floatQueryOP.kNearestColumns(in A, in q, k, m, ref idx, ref scores);
            return idx;
        }

        // ---- kNearestRows / kNearestColumns (double) ----

        /// <summary>
        /// Allocates clamped-k intN + doubleN from arena, fills via kNearestRows.
        /// </summary>
        public static intN doubleKNearestRows(this ref Arena arena, in doubleMxN A, in doubleN q, int k, Metric m, out doubleN scores, out int count)
        {
            int clampedK = math.min(k, A.M_Rows);
            if (clampedK <= 0) { scores = A.doubleVec(0, true); count = 0; return arena.intVec(0, true); }
            var idx = arena.intVec(clampedK);
            scores = A.doubleVec(clampedK);
            count = doubleQueryOP.kNearestRows(in A, in q, k, m, ref idx, ref scores);
            return idx;
        }

        /// <summary>
        /// Allocates clamped-k intN + doubleN from arena, fills via kNearestColumns.
        /// </summary>
        public static intN doubleKNearestColumns(this ref Arena arena, in doubleMxN A, in doubleN q, int k, Metric m, out doubleN scores, out int count)
        {
            int clampedK = math.min(k, A.N_Cols);
            if (clampedK <= 0) { scores = A.doubleVec(0, true); count = 0; return arena.intVec(0, true); }
            var idx = arena.intVec(clampedK);
            scores = A.doubleVec(clampedK);
            count = doubleQueryOP.kNearestColumns(in A, in q, k, m, ref idx, ref scores);
            return idx;
        }

        // ---- whichTrue (bool → intN) ----

        /// <summary>
        /// Count-pass + exact-alloc: fills exact-sized intN with indices of true elements in mask.
        /// </summary>
        public static intN WhichTrue(this ref Arena arena, in boolN mask)
        {
            int count = BoolAnalysis.countTrue(in mask);
            if (count == 0) return arena.intVec(0, true);
            var idx = arena.intVec(count);
            int written = 0;
            for (int i = 0; i < mask.N; i++)
                if (mask.Data[i]) idx[written++] = i;
            return idx;
        }

        /// <summary>
        /// Matrix overload: count-pass + exact-alloc intN of true element flat indices.
        /// </summary>
        public static intN WhichTrue(this ref Arena arena, in boolMxN mask)
        {
            int count = BoolAnalysis.countTrue(in mask);
            if (count == 0) return arena.intVec(0, true);
            int total = mask.M_Rows * mask.N_Cols;
            var idx = arena.intVec(count);
            int written = 0;
            for (int i = 0; i < total; i++)
                if (mask.Data[i]) idx[written++] = i;
            return idx;
        }
    }
}
