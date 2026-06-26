using System;

using LinearAlgebra;
using LinearAlgebra.Stats;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;

// Feature-scaling tests for StatsOP.normalizeColumns / normalizeRows (in-place MinMax & ZScore).
// MinMax maps each axis to [0,1] via (x-min)/(max-min); ZScore standardises via (x-mean)/stdDev
// using POPULATION std dev (divide by axis length). A constant axis (max==min or stdDev==0) has
// its entries set to EXACTLY 0 (the div-by-zero / NaN guard). Empty matrices throw.
public class floatNormalizeTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            MinMaxColumns,
            MinMaxRows,
            ZScoreColumns,
            ZScoreRows,
            ConstantColumn,
            ConstantRow,
            SingleRowMatrix,
            SingleColumnMatrix,
            InPlaceMutation,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<float> Fail;

        // ZScore reconstruction (mean->0, std->1) accumulates a little roundoff; scale the
        // tolerance with the numeric type. float: ~5.5e-3, double: ~2.4e-7.
        static float ZTol() => (float)16 * (float)Consts.floatSqrtEps;

        // MinMax endpoints (0 / 1) and exact interior fractions are essentially exact.
        const float MinMaxTol = 1E-5f;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.MinMaxColumns:      MinMaxColumns();      break;
                case TestType.MinMaxRows:         MinMaxRows();         break;
                case TestType.ZScoreColumns:      ZScoreColumns();      break;
                case TestType.ZScoreRows:         ZScoreRows();         break;
                case TestType.ConstantColumn:     ConstantColumn();     break;
                case TestType.ConstantRow:        ConstantRow();        break;
                case TestType.SingleRowMatrix:    SingleRowMatrix();    break;
                case TestType.SingleColumnMatrix: SingleColumnMatrix(); break;
                case TestType.InPlaceMutation:    InPlaceMutation();    break;
            }
        }

        // --- MinMax columns: hand-computed oracle on a 3x2 with negatives -----------------------
        // A = {{1,-2},{3,6},{5,2}}
        // col0: min 1, max 5, range 4 -> (0, 0.5, 1)
        // col1: min -2, max 6, range 8 -> (0, 1, 0.5)
        void MinMaxColumns()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.floatMat(3, 2);
            A[0, 0] = 1f;  A[0, 1] = -2f;
            A[1, 0] = 3f;  A[1, 1] = 6f;
            A[2, 0] = 5f;  A[2, 1] = 2f;

            floatStatsOP.normalizeColumns(ref A, NormalizeMode.MinMax);

            // col0: min entry -> 0, max entry -> 1, interior -> exact fraction 0.5
            AssertClose(A[0, 0], (float)0f, MinMaxTol);
            AssertClose(A[1, 0], (float)0.5f, MinMaxTol);
            AssertClose(A[2, 0], (float)1f, MinMaxTol);

            // col1 (negatives): min entry -> 0, max entry -> 1, interior -> 0.5
            AssertClose(A[0, 1], (float)0f, MinMaxTol);
            AssertClose(A[1, 1], (float)1f, MinMaxTol);
            AssertClose(A[2, 1], (float)0.5f, MinMaxTol);

            arena.Dispose();
        }

        // --- MinMax rows: hand-computed oracle on a 2x3 with negatives --------------------------
        // A = {{2,4,10},{-1,0,3}}
        // row0: min 2, max 10, range 8 -> (0, 0.25, 1)
        // row1: min -1, max 3, range 4 -> (0, 0.25, 1)
        void MinMaxRows()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.floatMat(2, 3);
            A[0, 0] = 2f;  A[0, 1] = 4f;  A[0, 2] = 10f;
            A[1, 0] = -1f; A[1, 1] = 0f;  A[1, 2] = 3f;

            floatStatsOP.normalizeRows(ref A, NormalizeMode.MinMax);

            AssertClose(A[0, 0], (float)0f, MinMaxTol);
            AssertClose(A[0, 1], (float)0.25f, MinMaxTol);
            AssertClose(A[0, 2], (float)1f, MinMaxTol);

            AssertClose(A[1, 0], (float)0f, MinMaxTol);
            AssertClose(A[1, 1], (float)0.25f, MinMaxTol);
            AssertClose(A[1, 2], (float)1f, MinMaxTol);

            arena.Dispose();
        }

        // --- ZScore columns: each column -> population mean 0, std 1 ----------------------------
        // A = {{1,10},{3,20},{5,60}}
        // col0: mean 3, popVar 8/3, std sqrt(8/3); col1: mean 30, popVar 1400/3, std sqrt(1400/3).
        void ZScoreColumns()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.floatMat(3, 2);
            A[0, 0] = 1f;  A[0, 1] = 10f;
            A[1, 0] = 3f;  A[1, 1] = 20f;
            A[2, 0] = 5f;  A[2, 1] = 60f;

            floatStatsOP.normalizeColumns(ref A, NormalizeMode.ZScore);

            // Specific entry oracle: [0,0] = (1-3)/sqrt(8/3); [2,1] = (60-30)/sqrt(1400/3).
            float exp00 = ((float)1f - (float)3f) / math.sqrt((float)(8f / 3f));
            float exp21 = ((float)60f - (float)30f) / math.sqrt((float)(1400f / 3f));
            AssertClose(A[0, 0], exp00, ZTol());
            AssertClose(A[2, 1], exp21, ZTol());

            // Property: each normalized column has mean ~ 0 and population std ~ 1.
            var mean = floatStatsOP.colMean(in A);
            var std = floatStatsOP.colStdDev(in A);
            for (int c = 0; c < A.N_Cols; c++)
            {
                AssertClose(mean[c], (float)0f, ZTol());
                AssertClose(std[c], (float)1f, ZTol());
            }

            arena.Dispose();
        }

        // --- ZScore rows: each row -> population mean 0, std 1 ----------------------------------
        // A = {{1,3,5},{10,20,60}}
        // row0: mean 3, std sqrt(8/3); row1: mean 30, std sqrt(1400/3).
        void ZScoreRows()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.floatMat(2, 3);
            A[0, 0] = 1f;  A[0, 1] = 3f;  A[0, 2] = 5f;
            A[1, 0] = 10f; A[1, 1] = 20f; A[1, 2] = 60f;

            floatStatsOP.normalizeRows(ref A, NormalizeMode.ZScore);

            float exp00 = ((float)1f - (float)3f) / math.sqrt((float)(8f / 3f));
            float exp12 = ((float)60f - (float)30f) / math.sqrt((float)(1400f / 3f));
            AssertClose(A[0, 0], exp00, ZTol());
            AssertClose(A[1, 2], exp12, ZTol());

            var mean = floatStatsOP.rowMean(in A);
            var std = floatStatsOP.rowStdDev(in A);
            for (int r = 0; r < A.M_Rows; r++)
            {
                AssertClose(mean[r], (float)0f, ZTol());
                AssertClose(std[r], (float)1f, ZTol());
            }

            arena.Dispose();
        }

        // --- Constant column guard (BOTH modes): constant col -> exactly 0, no NaN --------------
        // A = {{5,1},{5,2},{5,3}} ; col0 constant (==5). col1 is well-behaved and must normalize.
        void ConstantColumn()
        {
            CheckConstantColumn(NormalizeMode.MinMax);
            CheckConstantColumn(NormalizeMode.ZScore);
        }

        void CheckConstantColumn(NormalizeMode mode)
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.floatMat(3, 2);
            A[0, 0] = 5f;  A[0, 1] = 1f;
            A[1, 0] = 5f;  A[1, 1] = 2f;
            A[2, 0] = 5f;  A[2, 1] = 3f;

            floatStatsOP.normalizeColumns(ref A, mode);

            // Constant col0 -> every entry EXACTLY 0 (and not NaN).
            for (int r = 0; r < A.M_Rows; r++)
            {
                AssertNotNaN(A[r, 0]);
                AssertExactZero(A[r, 0]);
            }

            // The non-degenerate col1 must still be normalized (finite, and not all zero).
            float maxAbs = 0f;
            for (int r = 0; r < A.M_Rows; r++)
            {
                AssertNotNaN(A[r, 1]);
                maxAbs = math.max(maxAbs, math.abs(A[r, 1]));
            }
            AssertGreater(maxAbs, (float)0f);

            arena.Dispose();
        }

        // --- Constant row guard (BOTH modes): constant row -> exactly 0, no NaN -----------------
        // A = {{7,7,7},{1,2,3}} ; row0 constant. row1 well-behaved.
        void ConstantRow()
        {
            CheckConstantRow(NormalizeMode.MinMax);
            CheckConstantRow(NormalizeMode.ZScore);
        }

        void CheckConstantRow(NormalizeMode mode)
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.floatMat(2, 3);
            A[0, 0] = 7f;  A[0, 1] = 7f;  A[0, 2] = 7f;
            A[1, 0] = 1f;  A[1, 1] = 2f;  A[1, 2] = 3f;

            floatStatsOP.normalizeRows(ref A, mode);

            for (int c = 0; c < A.N_Cols; c++)
            {
                AssertNotNaN(A[0, c]);
                AssertExactZero(A[0, c]);
            }

            float maxAbs = 0f;
            for (int c = 0; c < A.N_Cols; c++)
            {
                AssertNotNaN(A[1, c]);
                maxAbs = math.max(maxAbs, math.abs(A[1, c]));
            }
            AssertGreater(maxAbs, (float)0f);

            arena.Dispose();
        }

        // --- Edge shape: single-row (1xN). normalizeColumns => every column has length 1 =>
        //     constant axis => 0-fill (both modes). normalizeRows on the single row works normally.
        void SingleRowMatrix()
        {
            // normalizeColumns on 1x3 -> all entries 0 (each column is length-1 / constant).
            for (int m = 0; m < 2; m++)
            {
                var arena = new Arena(Allocator.Persistent);
                var A = arena.floatMat(1, 3);
                A[0, 0] = 4f;  A[0, 1] = 9f;  A[0, 2] = 2f;

                floatStatsOP.normalizeColumns(ref A, m == 0 ? NormalizeMode.MinMax : NormalizeMode.ZScore);

                for (int c = 0; c < A.N_Cols; c++)
                {
                    AssertNotNaN(A[0, c]);
                    AssertExactZero(A[0, c]);
                }
                arena.Dispose();
            }

            // Sanity: normalizeRows over the single (non-degenerate) row normalizes across columns.
            // row {4,9,2}: min 2, max 9, range 7 -> (2/7, 1, 0).
            var arena2 = new Arena(Allocator.Persistent);
            var B = arena2.floatMat(1, 3);
            B[0, 0] = 4f;  B[0, 1] = 9f;  B[0, 2] = 2f;

            floatStatsOP.normalizeRows(ref B, NormalizeMode.MinMax);
            AssertClose(B[0, 0], (float)(2f / 7f), MinMaxTol);
            AssertClose(B[0, 1], (float)1f, MinMaxTol);
            AssertClose(B[0, 2], (float)0f, MinMaxTol);
            arena2.Dispose();
        }

        // --- Edge shape: single-column (Mx1). normalizeRows => every row has length 1 =>
        //     constant axis => 0-fill (both modes). normalizeColumns on the single column works.
        void SingleColumnMatrix()
        {
            for (int m = 0; m < 2; m++)
            {
                var arena = new Arena(Allocator.Persistent);
                var A = arena.floatMat(3, 1);
                A[0, 0] = 4f;  A[1, 0] = 9f;  A[2, 0] = 2f;

                floatStatsOP.normalizeRows(ref A, m == 0 ? NormalizeMode.MinMax : NormalizeMode.ZScore);

                for (int r = 0; r < A.M_Rows; r++)
                {
                    AssertNotNaN(A[r, 0]);
                    AssertExactZero(A[r, 0]);
                }
                arena.Dispose();
            }

            // Sanity: normalizeColumns over the single (non-degenerate) column normalizes across rows.
            // col {4,9,2}: min 2, max 9, range 7 -> (2/7, 1, 0).
            var arena2 = new Arena(Allocator.Persistent);
            var B = arena2.floatMat(3, 1);
            B[0, 0] = 4f;  B[1, 0] = 9f;  B[2, 0] = 2f;

            floatStatsOP.normalizeColumns(ref B, NormalizeMode.MinMax);
            AssertClose(B[0, 0], (float)(2f / 7f), MinMaxTol);
            AssertClose(B[1, 0], (float)1f, MinMaxTol);
            AssertClose(B[2, 0], (float)0f, MinMaxTol);
            arena2.Dispose();
        }

        // --- In-place semantics: the call returns void and mutates the caller's buffer. A shallow
        //     struct alias (sharing the same data buffer) must observe the same normalized values.
        void InPlaceMutation()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.floatMat(3, 2);
            A[0, 0] = 1f;  A[0, 1] = -2f;
            A[1, 0] = 3f;  A[1, 1] = 6f;
            A[2, 0] = 5f;  A[2, 1] = 2f;

            var alias = A; // shallow copy: shares the underlying data buffer

            floatStatsOP.normalizeColumns(ref A, NormalizeMode.MinMax);

            // The original raw value (3) was overwritten in place by the normalized value (0.5).
            AssertClose(A[1, 0], (float)0.5f, MinMaxTol);
            // The alias observes the same in-place mutation (same buffer, not a copy).
            AssertClose(alias[1, 0], (float)0.5f, MinMaxTol);
            AssertClose(alias[2, 0], (float)1f, MinMaxTol);
            AssertClose(alias[0, 1], (float)0f, MinMaxTol);

            arena.Dispose();
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff
        void AssertClose(float a, float b, float precision)
        {
            float diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        // Exact 0 fill for a constant axis (the div-by-zero guard writes 0f, not NaN/Inf).
        void AssertExactZero(float a)
        {
            if (!(a == (float)0f) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = a;
                Fail[2] = (float)0f;
                Fail[3] = a;
            }
            Assert.IsTrue(a == (float)0f);
        }

        void AssertNotNaN(float a)
        {
            bool nan = math.isnan(a);
            if (nan && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = a;
                Fail[2] = (float)(-999f);
                Fail[3] = (float)(-999f);
            }
            Assert.IsFalse(nan);
        }

        void AssertGreater(float a, float limit)
        {
            if (!(a > limit) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = a;
                Fail[2] = limit;
                Fail[3] = (float)0f;
            }
            Assert.IsTrue(a > limit);
        }
    }

    // Helper used by every managed runner to allocate/run/dispose with failure diagnostics.
    private void RunJob(TestJob.TestType type)
    {
        var fail = new NativeArray<float>(4, Allocator.TempJob);
        try {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (float)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e) {
            if (fail[0] != (float)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally {
            fail.Dispose();
        }
    }

    [Test] public void MinMaxColumnsTest()      { RunJob(TestJob.TestType.MinMaxColumns); }
    [Test] public void MinMaxRowsTest()         { RunJob(TestJob.TestType.MinMaxRows); }
    [Test] public void ZScoreColumnsTest()      { RunJob(TestJob.TestType.ZScoreColumns); }
    [Test] public void ZScoreRowsTest()         { RunJob(TestJob.TestType.ZScoreRows); }
    [Test] public void ConstantColumnTest()     { RunJob(TestJob.TestType.ConstantColumn); }
    [Test] public void ConstantRowTest()        { RunJob(TestJob.TestType.ConstantRow); }
    [Test] public void SingleRowMatrixTest()    { RunJob(TestJob.TestType.SingleRowMatrix); }
    [Test] public void SingleColumnMatrixTest() { RunJob(TestJob.TestType.SingleColumnMatrix); }
    [Test] public void InPlaceMutationTest()    { RunJob(TestJob.TestType.InPlaceMutation); }

    // Managed throw-tests: empty matrices (0 rows or 0 cols) must throw InvalidOperationException.
    // These run on the main thread (not inside a Burst job) so the managed exception surfaces.
    [Test]
    public void EmptyMatrixNormalizeColumnsThrows()
    {
        var arena = new Arena(Allocator.Persistent);

        var A0 = arena.floatMat(0, 3); // 0 rows
        var A1 = arena.floatMat(3, 0); // 0 cols

        Assert.Throws<InvalidOperationException>(() => floatStatsOP.normalizeColumns(ref A0, NormalizeMode.MinMax));
        Assert.Throws<InvalidOperationException>(() => floatStatsOP.normalizeColumns(ref A0, NormalizeMode.ZScore));
        Assert.Throws<InvalidOperationException>(() => floatStatsOP.normalizeColumns(ref A1, NormalizeMode.MinMax));
        Assert.Throws<InvalidOperationException>(() => floatStatsOP.normalizeColumns(ref A1, NormalizeMode.ZScore));

        arena.Dispose();
    }

    [Test]
    public void EmptyMatrixNormalizeRowsThrows()
    {
        var arena = new Arena(Allocator.Persistent);

        var A0 = arena.floatMat(0, 3); // 0 rows
        var A1 = arena.floatMat(3, 0); // 0 cols

        Assert.Throws<InvalidOperationException>(() => floatStatsOP.normalizeRows(ref A0, NormalizeMode.MinMax));
        Assert.Throws<InvalidOperationException>(() => floatStatsOP.normalizeRows(ref A0, NormalizeMode.ZScore));
        Assert.Throws<InvalidOperationException>(() => floatStatsOP.normalizeRows(ref A1, NormalizeMode.MinMax));
        Assert.Throws<InvalidOperationException>(() => floatStatsOP.normalizeRows(ref A1, NormalizeMode.ZScore));

        arena.Dispose();
    }
}
