using System;

using LinearAlgebra;
using LinearAlgebra.Stats;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;

public class fProxyStatsTests
{

    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            VectorVarianceStdDev,
            SingleElementVariance,
            ArgMinMaxVector,
            ArgMinMaxAllEqual,
            Matrix2x3,
            Matrix3x2Negatives,
            OneColumnMatrix,
            ArgMinMaxMatrix,
            CovarianceKnown,
            CovarianceDiagEqualsVarianceSample,
            CorrelationKnown,
            CorrelationPerfectAndAnti,
            CorrelationConstantColumn,
            Covariance1Variable
        }

        public TestType Type;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.VectorVarianceStdDev:
                    VectorVarianceStdDev();
                    break;
                case TestType.SingleElementVariance:
                    SingleElementVariance();
                    break;
                case TestType.ArgMinMaxVector:
                    ArgMinMaxVector();
                    break;
                case TestType.ArgMinMaxAllEqual:
                    ArgMinMaxAllEqual();
                    break;
                case TestType.Matrix2x3:
                    Matrix2x3();
                    break;
                case TestType.Matrix3x2Negatives:
                    Matrix3x2Negatives();
                    break;
                case TestType.OneColumnMatrix:
                    OneColumnMatrix();
                    break;
                case TestType.ArgMinMaxMatrix:
                    ArgMinMaxMatrix();
                    break;
                case TestType.CovarianceKnown:
                    CovarianceKnown();
                    break;
                case TestType.CovarianceDiagEqualsVarianceSample:
                    CovarianceDiagEqualsVarianceSample();
                    break;
                case TestType.CorrelationKnown:
                    CorrelationKnown();
                    break;
                case TestType.CorrelationPerfectAndAnti:
                    CorrelationPerfectAndAnti();
                    break;
                case TestType.CorrelationConstantColumn:
                    CorrelationConstantColumn();
                    break;
                case TestType.Covariance1Variable:
                    Covariance1Variable();
                    break;
            }
        }

        // Case 1: Vector {2,4,4,4,5,5,7,9} (n=8, mean 5)
        // variance==4, stdDev==2, varianceSample==32/7, stdDevSample==sqrt(32/7)
        void VectorVarianceStdDev()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.fProxyVec(8);
            v[0] = 2f; v[1] = 4f; v[2] = 4f; v[3] = 4f;
            v[4] = 5f; v[5] = 5f; v[6] = 7f; v[7] = 9f;

            AssertClose(fProxyStatsOP.mean(in v), (fProxy)5f, 1E-5f);
            AssertClose(fProxyStatsOP.variance(in v), (fProxy)4f, 1E-5f);
            AssertClose(fProxyStatsOP.stdDev(in v), (fProxy)2f, 1E-5f);
            AssertClose(fProxyStatsOP.varianceSample(in v), (fProxy)(32f / 7f), 1E-5f);
            AssertClose(fProxyStatsOP.stdDevSample(in v), (fProxy)math.sqrt(32f / 7f), 1E-5f);

            arena.Dispose();
        }

        // Case 2: Single-element vector {3}: variance==0 (bug-fix), stdDev==0
        void SingleElementVariance()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.fProxyVec(1);
            v[0] = 3f;

            AssertClose(fProxyStatsOP.variance(in v), (fProxy)0f, 1E-5f);
            AssertClose(fProxyStatsOP.stdDev(in v), (fProxy)0f, 1E-5f);

            arena.Dispose();
        }

        // Case 3: argmin/argmax on {3,1,4,1,5,9,2,6}: argmin==1 (first tied 1), argmax==5
        void ArgMinMaxVector()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.fProxyVec(8);
            v[0] = 3f; v[1] = 1f; v[2] = 4f; v[3] = 1f;
            v[4] = 5f; v[5] = 9f; v[6] = 2f; v[7] = 6f;

            Assert.AreEqual(1, fProxyStatsOP.argmin(in v));
            Assert.AreEqual(5, fProxyStatsOP.argmax(in v));

            arena.Dispose();
        }

        // Case 4: argmin/argmax on all-equal {7,7,7}: both==0
        void ArgMinMaxAllEqual()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.fProxyVec(3);
            v[0] = 7f; v[1] = 7f; v[2] = 7f;

            Assert.AreEqual(0, fProxyStatsOP.argmin(in v));
            Assert.AreEqual(0, fProxyStatsOP.argmax(in v));

            arena.Dispose();
        }

        // Case 5: 2x3 matrix {{1,2,3},{4,6,8}}
        // rowMin=={1,4}, rowMax=={3,8}, colMin=={1,2,3}, colMax=={4,6,8}
        // rowVariance=={2/3,8/3}, rowStdDev=={sqrt(2/3),sqrt(8/3)}
        // colVariance=={2.25,4,6.25}, colStdDev=={1.5,2,2.5}
        void Matrix2x3()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(2, 3);
            A[0, 0] = 1f; A[0, 1] = 2f; A[0, 2] = 3f;
            A[1, 0] = 4f; A[1, 1] = 6f; A[1, 2] = 8f;

            var rowMin = fProxyStatsOP.rowMin(in A);
            var rowMax = fProxyStatsOP.rowMax(in A);
            var colMin = fProxyStatsOP.colMin(in A);
            var colMax = fProxyStatsOP.colMax(in A);
            var rowVar = fProxyStatsOP.rowVariance(in A);
            var rowStd = fProxyStatsOP.rowStdDev(in A);
            var colVar = fProxyStatsOP.colVariance(in A);
            var colStd = fProxyStatsOP.colStdDev(in A);

            // Result vector lengths
            Assert.AreEqual(2, rowMin.N);
            Assert.AreEqual(2, rowMax.N);
            Assert.AreEqual(3, colMin.N);
            Assert.AreEqual(3, colMax.N);
            Assert.AreEqual(2, rowVar.N);
            Assert.AreEqual(2, rowStd.N);
            Assert.AreEqual(3, colVar.N);
            Assert.AreEqual(3, colStd.N);

            AssertClose(rowMin[0], (fProxy)1f, 1E-5f);
            AssertClose(rowMin[1], (fProxy)4f, 1E-5f);
            AssertClose(rowMax[0], (fProxy)3f, 1E-5f);
            AssertClose(rowMax[1], (fProxy)8f, 1E-5f);

            AssertClose(colMin[0], (fProxy)1f, 1E-5f);
            AssertClose(colMin[1], (fProxy)2f, 1E-5f);
            AssertClose(colMin[2], (fProxy)3f, 1E-5f);
            AssertClose(colMax[0], (fProxy)4f, 1E-5f);
            AssertClose(colMax[1], (fProxy)6f, 1E-5f);
            AssertClose(colMax[2], (fProxy)8f, 1E-5f);

            AssertClose(rowVar[0], (fProxy)(2f / 3f), 1E-5f);
            AssertClose(rowVar[1], (fProxy)(8f / 3f), 1E-5f);
            AssertClose(rowStd[0], (fProxy)math.sqrt(2f / 3f), 1E-5f);
            AssertClose(rowStd[1], (fProxy)math.sqrt(8f / 3f), 1E-5f);

            AssertClose(colVar[0], (fProxy)2.25f, 1E-5f);
            AssertClose(colVar[1], (fProxy)4f, 1E-5f);
            AssertClose(colVar[2], (fProxy)6.25f, 1E-5f);
            AssertClose(colStd[0], (fProxy)1.5f, 1E-5f);
            AssertClose(colStd[1], (fProxy)2f, 1E-5f);
            AssertClose(colStd[2], (fProxy)2.5f, 1E-5f);

            arena.Dispose();
        }

        // Case 6: 3x2 matrix with negatives {{-1,1},{-3,3},{-5,5}}
        // colMin=={-5,1}, colMax=={-1,5}, colVariance=={8/3,8/3}
        void Matrix3x2Negatives()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(3, 2);
            A[0, 0] = -1f; A[0, 1] = 1f;
            A[1, 0] = -3f; A[1, 1] = 3f;
            A[2, 0] = -5f; A[2, 1] = 5f;

            var colMin = fProxyStatsOP.colMin(in A);
            var colMax = fProxyStatsOP.colMax(in A);
            var colVar = fProxyStatsOP.colVariance(in A);

            Assert.AreEqual(2, colMin.N);
            Assert.AreEqual(2, colMax.N);
            Assert.AreEqual(2, colVar.N);

            AssertClose(colMin[0], (fProxy)(-5f), 1E-5f);
            AssertClose(colMin[1], (fProxy)1f, 1E-5f);
            AssertClose(colMax[0], (fProxy)(-1f), 1E-5f);
            AssertClose(colMax[1], (fProxy)5f, 1E-5f);

            AssertClose(colVar[0], (fProxy)(8f / 3f), 1E-5f);
            AssertClose(colVar[1], (fProxy)(8f / 3f), 1E-5f);

            arena.Dispose();
        }

        // Case 7: One-column 3x1 matrix: rowVariance=={0,0,0}
        void OneColumnMatrix()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(3, 1);
            A[0, 0] = 2f;
            A[1, 0] = 7f;
            A[2, 0] = -4f;

            var rowVar = fProxyStatsOP.rowVariance(in A);

            Assert.AreEqual(3, rowVar.N);

            AssertClose(rowVar[0], (fProxy)0f, 1E-5f);
            AssertClose(rowVar[1], (fProxy)0f, 1E-5f);
            AssertClose(rowVar[2], (fProxy)0f, 1E-5f);

            arena.Dispose();
        }

        // Case 8: argmin/argmax on the 2x3 matrix of case 5: argmin==0, argmax==5 (row-major linear index)
        void ArgMinMaxMatrix()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(2, 3);
            A[0, 0] = 1f; A[0, 1] = 2f; A[0, 2] = 3f;
            A[1, 0] = 4f; A[1, 1] = 6f; A[1, 2] = 8f;

            Assert.AreEqual(0, fProxyStatsOP.argmin(in A));
            Assert.AreEqual(5, fProxyStatsOP.argmax(in A));

            arena.Dispose();
        }

        // Case 10: covariance of 3 obs x 2 vars {{1,2},{3,6},{5,4}}.
        // Column means: col0=3, col1=4. Deviations col0=(-2,0,2), col1=(-2,2,0).
        // cov00=8/2=4, cov11=8/2=4, cov01=(4+0+0)/2=2. => {{4,2},{2,4}}.
        void CovarianceKnown()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(3, 2);
            A[0, 0] = 1f; A[0, 1] = 2f;
            A[1, 0] = 3f; A[1, 1] = 6f;
            A[2, 0] = 5f; A[2, 1] = 4f;

            var C = fProxyStatsOP.covariance(in A);

            Assert.AreEqual(2, C.M_Rows);
            Assert.AreEqual(2, C.N_Cols);

            AssertClose(C[0, 0], (fProxy)4f, 1E-5f);
            AssertClose(C[0, 1], (fProxy)2f, 1E-5f);
            AssertClose(C[1, 0], (fProxy)2f, 1E-5f);
            AssertClose(C[1, 1], (fProxy)4f, 1E-5f);

            // Symmetry.
            AssertClose(C[0, 1], C[1, 0], 1E-5f);

            arena.Dispose();
        }

        // Case 11: covariance diagonal equals varianceSample of each column.
        // Same A as CovarianceKnown; C[i,i] must match varianceSample(column i) == 4.
        void CovarianceDiagEqualsVarianceSample()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(3, 2);
            A[0, 0] = 1f; A[0, 1] = 2f;
            A[1, 0] = 3f; A[1, 1] = 6f;
            A[2, 0] = 5f; A[2, 1] = 4f;

            var C = fProxyStatsOP.covariance(in A);

            // Build column vectors and compare with varianceSample.
            var col0 = arena.fProxyVec(3);
            col0[0] = A[0, 0]; col0[1] = A[1, 0]; col0[2] = A[2, 0];
            var col1 = arena.fProxyVec(3);
            col1[0] = A[0, 1]; col1[1] = A[1, 1]; col1[2] = A[2, 1];

            AssertClose(C[0, 0], fProxyStatsOP.varianceSample(in col0), 1E-5f);
            AssertClose(C[1, 1], fProxyStatsOP.varianceSample(in col1), 1E-5f);

            AssertClose(C[0, 0], (fProxy)4f, 1E-5f);
            AssertClose(C[1, 1], (fProxy)4f, 1E-5f);

            arena.Dispose();
        }

        // Case 12: correlation of the same A => {{1,0.5},{0.5,1}}.
        // s0=s1=2; R01 = cov01/(s0*s1) = 2/4 = 0.5. Diagonal exactly 1.
        void CorrelationKnown()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(3, 2);
            A[0, 0] = 1f; A[0, 1] = 2f;
            A[1, 0] = 3f; A[1, 1] = 6f;
            A[2, 0] = 5f; A[2, 1] = 4f;

            var R = fProxyStatsOP.correlation(in A);

            Assert.AreEqual(2, R.M_Rows);
            Assert.AreEqual(2, R.N_Cols);

            AssertClose(R[0, 0], (fProxy)1f, 1E-5f);
            AssertClose(R[1, 1], (fProxy)1f, 1E-5f);
            AssertClose(R[0, 1], (fProxy)0.5f, 1E-5f);
            AssertClose(R[1, 0], (fProxy)0.5f, 1E-5f);

            // Symmetry.
            AssertClose(R[0, 1], R[1, 0], 1E-5f);

            arena.Dispose();
        }

        // Case 13: perfect (+1) and anti (-1) correlation.
        // col = (1,3,5). Identical columns => R01 == 1. Negated column => R01 == -1.
        void CorrelationPerfectAndAnti()
        {
            var arena = new Arena(Allocator.Persistent);

            // Identical columns => +1.
            var Apos = arena.fProxyMat(3, 2);
            Apos[0, 0] = 1f; Apos[0, 1] = 1f;
            Apos[1, 0] = 3f; Apos[1, 1] = 3f;
            Apos[2, 0] = 5f; Apos[2, 1] = 5f;

            var Rpos = fProxyStatsOP.correlation(in Apos);
            AssertClose(Rpos[0, 1], (fProxy)1f, 1E-5f);
            AssertClose(Rpos[1, 0], (fProxy)1f, 1E-5f);

            // Negated column => -1.
            var Aneg = arena.fProxyMat(3, 2);
            Aneg[0, 0] = 1f; Aneg[0, 1] = -1f;
            Aneg[1, 0] = 3f; Aneg[1, 1] = -3f;
            Aneg[2, 0] = 5f; Aneg[2, 1] = -5f;

            var Rneg = fProxyStatsOP.correlation(in Aneg);
            AssertClose(Rneg[0, 1], (fProxy)(-1f), 1E-5f);
            AssertClose(Rneg[1, 0], (fProxy)(-1f), 1E-5f);

            arena.Dispose();
        }

        // Case 14: constant column has zero variance => off-diagonal correlations 0, diagonal 1.
        // col0 = (1,3,5), col1 = (7,7,7). Covariance C[1,1] == 0.
        void CorrelationConstantColumn()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(3, 2);
            A[0, 0] = 1f; A[0, 1] = 7f;
            A[1, 0] = 3f; A[1, 1] = 7f;
            A[2, 0] = 5f; A[2, 1] = 7f;

            var C = fProxyStatsOP.covariance(in A);
            AssertClose(C[1, 1], (fProxy)0f, 1E-5f);

            var R = fProxyStatsOP.correlation(in A);
            AssertClose(R[0, 0], (fProxy)1f, 1E-5f);
            AssertClose(R[1, 1], (fProxy)1f, 1E-5f);
            AssertClose(R[0, 1], (fProxy)0f, 1E-5f);
            AssertClose(R[1, 0], (fProxy)0f, 1E-5f);

            arena.Dispose();
        }

        // Case 15: single-variable matrices. 3x1 col=(2,4,6): mean 4, devs (-2,0,2), varSample 8/2=4.
        // covariance == {{4}}, correlation == {{1}}.
        void Covariance1Variable()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(3, 1);
            A[0, 0] = 2f;
            A[1, 0] = 4f;
            A[2, 0] = 6f;

            var C = fProxyStatsOP.covariance(in A);
            Assert.AreEqual(1, C.M_Rows);
            Assert.AreEqual(1, C.N_Cols);
            AssertClose(C[0, 0], (fProxy)4f, 1E-5f);

            var R = fProxyStatsOP.correlation(in A);
            Assert.AreEqual(1, R.M_Rows);
            Assert.AreEqual(1, R.N_Cols);
            AssertClose(R[0, 0], (fProxy)1f, 1E-5f);

            arena.Dispose();
        }

        private void AssertClose(fProxy a, fProxy b, fProxy precision)
        {
            fProxy diff = math.abs(a - b);
            Assert.IsTrue(diff <= precision, $"Expected {b} got {a} (diff {diff})");
        }
    }

    [Test]
    public void VectorVarianceStdDevTest()
    {
        new TestJob() { Type = TestJob.TestType.VectorVarianceStdDev }.Run();
    }

    [Test]
    public void SingleElementVarianceTest()
    {
        new TestJob() { Type = TestJob.TestType.SingleElementVariance }.Run();
    }

    [Test]
    public void ArgMinMaxVectorTest()
    {
        new TestJob() { Type = TestJob.TestType.ArgMinMaxVector }.Run();
    }

    [Test]
    public void ArgMinMaxAllEqualTest()
    {
        new TestJob() { Type = TestJob.TestType.ArgMinMaxAllEqual }.Run();
    }

    [Test]
    public void Matrix2x3Test()
    {
        new TestJob() { Type = TestJob.TestType.Matrix2x3 }.Run();
    }

    [Test]
    public void Matrix3x2NegativesTest()
    {
        new TestJob() { Type = TestJob.TestType.Matrix3x2Negatives }.Run();
    }

    [Test]
    public void OneColumnMatrixTest()
    {
        new TestJob() { Type = TestJob.TestType.OneColumnMatrix }.Run();
    }

    [Test]
    public void ArgMinMaxMatrixTest()
    {
        new TestJob() { Type = TestJob.TestType.ArgMinMaxMatrix }.Run();
    }

    [Test]
    public void CovarianceKnownTest()
    {
        new TestJob() { Type = TestJob.TestType.CovarianceKnown }.Run();
    }

    [Test]
    public void CovarianceDiagEqualsVarianceSampleTest()
    {
        new TestJob() { Type = TestJob.TestType.CovarianceDiagEqualsVarianceSample }.Run();
    }

    [Test]
    public void CorrelationKnownTest()
    {
        new TestJob() { Type = TestJob.TestType.CorrelationKnown }.Run();
    }

    [Test]
    public void CorrelationPerfectAndAntiTest()
    {
        new TestJob() { Type = TestJob.TestType.CorrelationPerfectAndAnti }.Run();
    }

    [Test]
    public void CorrelationConstantColumnTest()
    {
        new TestJob() { Type = TestJob.TestType.CorrelationConstantColumn }.Run();
    }

    [Test]
    public void Covariance1VariableTest()
    {
        new TestJob() { Type = TestJob.TestType.Covariance1Variable }.Run();
    }

    // Case 9: Managed throw-tests (must run on main thread, not inside a Burst job).

    [Test]
    public void VarianceSampleEmptyVectorThrows()
    {
        var arena = new Arena(Allocator.Persistent);

        var v = arena.fProxyVec(0);

        Assert.Throws<InvalidOperationException>(() => fProxyStatsOP.varianceSample(in v));

        arena.Dispose();
    }

    [Test]
    public void VarianceSampleSingleElementThrows()
    {
        var arena = new Arena(Allocator.Persistent);

        var v = arena.fProxyVec(1);
        v[0] = 5f;

        Assert.Throws<InvalidOperationException>(() => fProxyStatsOP.varianceSample(in v));

        arena.Dispose();
    }

    [Test]
    public void EmptyMatrixStatisticsThrows()
    {
        var arena = new Arena(Allocator.Persistent);

        // 0-row matrix (3 cols) is constructible; row/col stats must throw.
        var A = arena.fProxyMat(0, 3);

        Assert.Throws<InvalidOperationException>(() => fProxyStatsOP.rowMin(in A));
        Assert.Throws<InvalidOperationException>(() => fProxyStatsOP.colMin(in A));
        Assert.Throws<InvalidOperationException>(() => fProxyStatsOP.rowVariance(in A));
        Assert.Throws<InvalidOperationException>(() => fProxyStatsOP.colVariance(in A));

        arena.Dispose();
    }

    // Case 16: covariance/correlation require M_Rows >= 2. A 1x2 matrix must throw.
    [Test]
    public void CovarianceCorrelationTooFewRowsThrows()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.fProxyMat(1, 2);
        A[0, 0] = 1f; A[0, 1] = 2f;

        Assert.Throws<InvalidOperationException>(() => fProxyStatsOP.covariance(in A));
        Assert.Throws<InvalidOperationException>(() => fProxyStatsOP.correlation(in A));

        arena.Dispose();
    }

    // Case 17: covariance/correlation require N_Cols >= 1. A 0-column matrix must throw.
    [Test]
    public void CovarianceCorrelationZeroColumnsThrows()
    {
        var arena = new Arena(Allocator.Persistent);

        // 0-col matrix (3 rows) is constructible; covariance/correlation must throw.
        var A = arena.fProxyMat(3, 0);

        Assert.Throws<InvalidOperationException>(() => fProxyStatsOP.covariance(in A));
        Assert.Throws<InvalidOperationException>(() => fProxyStatsOP.correlation(in A));

        arena.Dispose();
    }
}
