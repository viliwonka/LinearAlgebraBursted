using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;

public class doubleStatsTests
{

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
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
            Covariance1Variable,
            RowColSumMean,
            RowColNorms,
            RefDestMatchesAllocating,
            CovarianceIntoSingleRowZeroFill
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<double> Fail;

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
                case TestType.RowColSumMean:
                    RowColSumMean();
                    break;
                case TestType.RowColNorms:
                    RowColNorms();
                    break;
                case TestType.RefDestMatchesAllocating:
                    RefDestMatchesAllocating();
                    break;
                case TestType.CovarianceIntoSingleRowZeroFill:
                    CovarianceIntoSingleRowZeroFill();
                    break;
            }
        }

        // Guard: covarianceInto must degrade gracefully when M_Rows < 2.
        // Previously 1/(M-1) = 1/0 = Inf and 0*Inf = NaN filled every cell; the guard now
        // zero-fills the N×N output and returns. Build a 1-row matrix (M=1, N=3), poison the
        // 3×3 destination, run covarianceInto, and assert every cell is EXACTLY 0 and not NaN.
        void CovarianceIntoSingleRowZeroFill()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleMat(1, 3);
            A[0, 0] = 5f; A[0, 1] = -2f; A[0, 2] = 9f;

            var C = arena.doubleMat(3, 3);
            // Poison every cell so a non-zeroing / NaN-producing primitive would be caught.
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    C[i, j] = (double)999f;

            Stats.covarianceInto(in A, ref C);

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    AssertNotNaN(C[i, j]);
                    AssertClose(C[i, j], (double)0f, 0f); // exactly zero
                }

            arena.Dispose();
        }

        // Known-value oracle for rowSum/colSum/rowMean/colMean on {{1,2,3},{4,6,8}}:
        // rowSum={6,18}, colSum={5,8,11}, rowMean={2,6}, colMean={2.5,4,5.5}.
        void RowColSumMean()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleMat(2, 3);
            A[0, 0] = 1f; A[0, 1] = 2f; A[0, 2] = 3f;
            A[1, 0] = 4f; A[1, 1] = 6f; A[1, 2] = 8f;

            var rSum = Stats.rowSum(in A);
            var cSum = Stats.colSum(in A);
            var rMean = Stats.rowMean(in A);
            var cMean = Stats.colMean(in A);

            Assert.AreEqual(2, rSum.N); Assert.AreEqual(3, cSum.N);
            Assert.AreEqual(2, rMean.N); Assert.AreEqual(3, cMean.N);

            AssertClose(rSum[0], (double)6f, 1E-5f);
            AssertClose(rSum[1], (double)18f, 1E-5f);
            AssertClose(cSum[0], (double)5f, 1E-5f);
            AssertClose(cSum[1], (double)8f, 1E-5f);
            AssertClose(cSum[2], (double)11f, 1E-5f);

            AssertClose(rMean[0], (double)2f, 1E-5f);
            AssertClose(rMean[1], (double)6f, 1E-5f);
            AssertClose(cMean[0], (double)2.5f, 1E-5f);
            AssertClose(cMean[1], (double)4f, 1E-5f);
            AssertClose(cMean[2], (double)5.5f, 1E-5f);

            arena.Dispose();
        }

        // Per-row / per-col L1 & L2 norms on {{1,-2,3},{-4,6,-8}} (abs handled):
        // rowNormL1=={6,18}, rowNormL2=={sqrt14,sqrt116}, colNormL1=={5,8,11}, colNormL2=={sqrt17,sqrt40,sqrt73}
        void RowColNorms()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleMat(2, 3);
            A[0, 0] = 1f; A[0, 1] = -2f; A[0, 2] = 3f;
            A[1, 0] = -4f; A[1, 1] = 6f; A[1, 2] = -8f;

            var rL1 = Stats.rowNormL1(in A);
            var rL2 = Stats.rowNormL2(in A);
            var cL1 = Stats.colNormL1(in A);
            var cL2 = Stats.colNormL2(in A);

            Assert.AreEqual(2, rL1.N); Assert.AreEqual(2, rL2.N);
            Assert.AreEqual(3, cL1.N); Assert.AreEqual(3, cL2.N);

            AssertClose(rL1[0], (double)6f, 1E-5f);
            AssertClose(rL1[1], (double)18f, 1E-5f);
            AssertClose(rL2[0], (double)math.sqrt(14f), 1E-5f);
            AssertClose(rL2[1], (double)math.sqrt(116f), 1E-5f);

            AssertClose(cL1[0], (double)5f, 1E-5f);
            AssertClose(cL1[1], (double)8f, 1E-5f);
            AssertClose(cL1[2], (double)11f, 1E-5f);
            AssertClose(cL2[0], (double)math.sqrt(17f), 1E-5f);
            AssertClose(cL2[1], (double)math.sqrt(40f), 1E-5f);
            AssertClose(cL2[2], (double)math.sqrt(73f), 1E-5f);

            arena.Dispose();
        }

        // The zero-alloc ref-destination overloads must produce identical results to the allocating
        // wrappers for every row*/col* reduction (covers the whole refactored surface).
        void RefDestMatchesAllocating()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 5, n = 4;
            var A = arena.doubleRandomMat(m, n, -3f, 3f, 778899);

            var rDest = arena.doubleVec(m);
            var cDest = arena.doubleVec(n);

            // Poison dest before each call so an accumulating col op with a missing zeroing loop
            // (which += into garbage) would actually fail — the dest does NOT start zeroed.
            // row ops -> length m
            Poison(in rDest); Stats.rowSum(in A, ref rDest);      EqVec(in rDest, Stats.rowSum(in A), m);
            Poison(in rDest); Stats.rowMean(in A, ref rDest);     EqVec(in rDest, Stats.rowMean(in A), m);
            Poison(in rDest); Stats.rowMin(in A, ref rDest);      EqVec(in rDest, Stats.rowMin(in A), m);
            Poison(in rDest); Stats.rowMax(in A, ref rDest);      EqVec(in rDest, Stats.rowMax(in A), m);
            Poison(in rDest); Stats.rowVariance(in A, ref rDest); EqVec(in rDest, Stats.rowVariance(in A), m);
            Poison(in rDest); Stats.rowStdDev(in A, ref rDest);   EqVec(in rDest, Stats.rowStdDev(in A), m);
            Poison(in rDest); Stats.rowNormL1(in A, ref rDest);   EqVec(in rDest, Stats.rowNormL1(in A), m);
            Poison(in rDest); Stats.rowNormL2(in A, ref rDest);   EqVec(in rDest, Stats.rowNormL2(in A), m);

            // col ops -> length n
            Poison(in cDest); Stats.colSum(in A, ref cDest);      EqVec(in cDest, Stats.colSum(in A), n);
            Poison(in cDest); Stats.colMean(in A, ref cDest);     EqVec(in cDest, Stats.colMean(in A), n);
            Poison(in cDest); Stats.colMin(in A, ref cDest);      EqVec(in cDest, Stats.colMin(in A), n);
            Poison(in cDest); Stats.colMax(in A, ref cDest);      EqVec(in cDest, Stats.colMax(in A), n);
            Poison(in cDest); Stats.colVariance(in A, ref cDest); EqVec(in cDest, Stats.colVariance(in A), n);
            Poison(in cDest); Stats.colStdDev(in A, ref cDest);   EqVec(in cDest, Stats.colStdDev(in A), n);
            Poison(in cDest); Stats.colNormL1(in A, ref cDest);   EqVec(in cDest, Stats.colNormL1(in A), n);
            Poison(in cDest); Stats.colNormL2(in A, ref cDest);   EqVec(in cDest, Stats.colNormL2(in A), n);

            arena.Dispose();
        }

        void Poison(in doubleN v)
        {
            for (int i = 0; i < v.N; i++)
                v[i] = (double)999f;
        }

        void EqVec(in doubleN a, in doubleN b, int len)
        {
            Assert.AreEqual(len, a.N);
            Assert.AreEqual(len, b.N);
            for (int i = 0; i < len; i++)
                AssertClose(a[i], b[i], 1E-5f);
        }

        // Case 1: Vector {2,4,4,4,5,5,7,9} (n=8, mean 5)
        // variance==4, stdDev==2, varianceSample==32/7, stdDevSample==sqrt(32/7)
        void VectorVarianceStdDev()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.doubleVec(8);
            v[0] = 2f; v[1] = 4f; v[2] = 4f; v[3] = 4f;
            v[4] = 5f; v[5] = 5f; v[6] = 7f; v[7] = 9f;

            AssertClose(Stats.mean(in v), (double)5f, 1E-5f);
            AssertClose(Stats.variance(in v), (double)4f, 1E-5f);
            AssertClose(Stats.stdDev(in v), (double)2f, 1E-5f);
            AssertClose(Stats.varianceSample(in v), (double)(32f / 7f), 1E-5f);
            AssertClose(Stats.stdDevSample(in v), (double)math.sqrt(32f / 7f), 1E-5f);

            arena.Dispose();
        }

        // Case 2: Single-element vector {3}: variance==0 (bug-fix), stdDev==0
        void SingleElementVariance()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.doubleVec(1);
            v[0] = 3f;

            AssertClose(Stats.variance(in v), (double)0f, 1E-5f);
            AssertClose(Stats.stdDev(in v), (double)0f, 1E-5f);

            arena.Dispose();
        }

        // Case 3: argmin/argmax on {3,1,4,1,5,9,2,6}: argmin==1 (first tied 1), argmax==5
        void ArgMinMaxVector()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.doubleVec(8);
            v[0] = 3f; v[1] = 1f; v[2] = 4f; v[3] = 1f;
            v[4] = 5f; v[5] = 9f; v[6] = 2f; v[7] = 6f;

            Assert.AreEqual(1, Stats.argmin(in v));
            Assert.AreEqual(5, Stats.argmax(in v));

            arena.Dispose();
        }

        // Case 4: argmin/argmax on all-equal {7,7,7}: both==0
        void ArgMinMaxAllEqual()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.doubleVec(3);
            v[0] = 7f; v[1] = 7f; v[2] = 7f;

            Assert.AreEqual(0, Stats.argmin(in v));
            Assert.AreEqual(0, Stats.argmax(in v));

            arena.Dispose();
        }

        // Case 5: 2x3 matrix {{1,2,3},{4,6,8}}
        // rowMin=={1,4}, rowMax=={3,8}, colMin=={1,2,3}, colMax=={4,6,8}
        // rowVariance=={2/3,8/3}, rowStdDev=={sqrt(2/3),sqrt(8/3)}
        // colVariance=={2.25,4,6.25}, colStdDev=={1.5,2,2.5}
        void Matrix2x3()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleMat(2, 3);
            A[0, 0] = 1f; A[0, 1] = 2f; A[0, 2] = 3f;
            A[1, 0] = 4f; A[1, 1] = 6f; A[1, 2] = 8f;

            var rowMin = Stats.rowMin(in A);
            var rowMax = Stats.rowMax(in A);
            var colMin = Stats.colMin(in A);
            var colMax = Stats.colMax(in A);
            var rowVar = Stats.rowVariance(in A);
            var rowStd = Stats.rowStdDev(in A);
            var colVar = Stats.colVariance(in A);
            var colStd = Stats.colStdDev(in A);

            // Result vector lengths
            Assert.AreEqual(2, rowMin.N);
            Assert.AreEqual(2, rowMax.N);
            Assert.AreEqual(3, colMin.N);
            Assert.AreEqual(3, colMax.N);
            Assert.AreEqual(2, rowVar.N);
            Assert.AreEqual(2, rowStd.N);
            Assert.AreEqual(3, colVar.N);
            Assert.AreEqual(3, colStd.N);

            AssertClose(rowMin[0], (double)1f, 1E-5f);
            AssertClose(rowMin[1], (double)4f, 1E-5f);
            AssertClose(rowMax[0], (double)3f, 1E-5f);
            AssertClose(rowMax[1], (double)8f, 1E-5f);

            AssertClose(colMin[0], (double)1f, 1E-5f);
            AssertClose(colMin[1], (double)2f, 1E-5f);
            AssertClose(colMin[2], (double)3f, 1E-5f);
            AssertClose(colMax[0], (double)4f, 1E-5f);
            AssertClose(colMax[1], (double)6f, 1E-5f);
            AssertClose(colMax[2], (double)8f, 1E-5f);

            AssertClose(rowVar[0], (double)(2f / 3f), 1E-5f);
            AssertClose(rowVar[1], (double)(8f / 3f), 1E-5f);
            AssertClose(rowStd[0], (double)math.sqrt(2f / 3f), 1E-5f);
            AssertClose(rowStd[1], (double)math.sqrt(8f / 3f), 1E-5f);

            AssertClose(colVar[0], (double)2.25f, 1E-5f);
            AssertClose(colVar[1], (double)4f, 1E-5f);
            AssertClose(colVar[2], (double)6.25f, 1E-5f);
            AssertClose(colStd[0], (double)1.5f, 1E-5f);
            AssertClose(colStd[1], (double)2f, 1E-5f);
            AssertClose(colStd[2], (double)2.5f, 1E-5f);

            arena.Dispose();
        }

        // Case 6: 3x2 matrix with negatives {{-1,1},{-3,3},{-5,5}}
        // colMin=={-5,1}, colMax=={-1,5}, colVariance=={8/3,8/3}
        void Matrix3x2Negatives()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleMat(3, 2);
            A[0, 0] = -1f; A[0, 1] = 1f;
            A[1, 0] = -3f; A[1, 1] = 3f;
            A[2, 0] = -5f; A[2, 1] = 5f;

            var colMin = Stats.colMin(in A);
            var colMax = Stats.colMax(in A);
            var colVar = Stats.colVariance(in A);

            Assert.AreEqual(2, colMin.N);
            Assert.AreEqual(2, colMax.N);
            Assert.AreEqual(2, colVar.N);

            AssertClose(colMin[0], (double)(-5f), 1E-5f);
            AssertClose(colMin[1], (double)1f, 1E-5f);
            AssertClose(colMax[0], (double)(-1f), 1E-5f);
            AssertClose(colMax[1], (double)5f, 1E-5f);

            AssertClose(colVar[0], (double)(8f / 3f), 1E-5f);
            AssertClose(colVar[1], (double)(8f / 3f), 1E-5f);

            arena.Dispose();
        }

        // Case 7: One-column 3x1 matrix: rowVariance=={0,0,0}
        void OneColumnMatrix()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleMat(3, 1);
            A[0, 0] = 2f;
            A[1, 0] = 7f;
            A[2, 0] = -4f;

            var rowVar = Stats.rowVariance(in A);

            Assert.AreEqual(3, rowVar.N);

            AssertClose(rowVar[0], (double)0f, 1E-5f);
            AssertClose(rowVar[1], (double)0f, 1E-5f);
            AssertClose(rowVar[2], (double)0f, 1E-5f);

            arena.Dispose();
        }

        // Case 8: argmin/argmax on the 2x3 matrix of case 5: argmin==0, argmax==5 (row-major linear index)
        void ArgMinMaxMatrix()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleMat(2, 3);
            A[0, 0] = 1f; A[0, 1] = 2f; A[0, 2] = 3f;
            A[1, 0] = 4f; A[1, 1] = 6f; A[1, 2] = 8f;

            Assert.AreEqual(0, Stats.argmin(in A));
            Assert.AreEqual(5, Stats.argmax(in A));

            arena.Dispose();
        }

        // Case 10: covariance of 3 obs x 2 vars {{1,2},{3,6},{5,4}}.
        // Column means: col0=3, col1=4. Deviations col0=(-2,0,2), col1=(-2,2,0).
        // cov00=8/2=4, cov11=8/2=4, cov01=(4+0+0)/2=2. => {{4,2},{2,4}}.
        void CovarianceKnown()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleMat(3, 2);
            A[0, 0] = 1f; A[0, 1] = 2f;
            A[1, 0] = 3f; A[1, 1] = 6f;
            A[2, 0] = 5f; A[2, 1] = 4f;

            var C = Stats.covariance(in A);

            Assert.AreEqual(2, C.M_Rows);
            Assert.AreEqual(2, C.N_Cols);

            AssertClose(C[0, 0], (double)4f, 1E-5f);
            AssertClose(C[0, 1], (double)2f, 1E-5f);
            AssertClose(C[1, 0], (double)2f, 1E-5f);
            AssertClose(C[1, 1], (double)4f, 1E-5f);

            // Symmetry.
            AssertClose(C[0, 1], C[1, 0], 1E-5f);

            arena.Dispose();
        }

        // Case 11: covariance diagonal equals varianceSample of each column.
        // Same A as CovarianceKnown; C[i,i] must match varianceSample(column i) == 4.
        void CovarianceDiagEqualsVarianceSample()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleMat(3, 2);
            A[0, 0] = 1f; A[0, 1] = 2f;
            A[1, 0] = 3f; A[1, 1] = 6f;
            A[2, 0] = 5f; A[2, 1] = 4f;

            var C = Stats.covariance(in A);

            // Build column vectors and compare with varianceSample.
            var col0 = arena.doubleVec(3);
            col0[0] = A[0, 0]; col0[1] = A[1, 0]; col0[2] = A[2, 0];
            var col1 = arena.doubleVec(3);
            col1[0] = A[0, 1]; col1[1] = A[1, 1]; col1[2] = A[2, 1];

            AssertClose(C[0, 0], Stats.varianceSample(in col0), 1E-5f);
            AssertClose(C[1, 1], Stats.varianceSample(in col1), 1E-5f);

            AssertClose(C[0, 0], (double)4f, 1E-5f);
            AssertClose(C[1, 1], (double)4f, 1E-5f);

            arena.Dispose();
        }

        // Case 12: correlation of the same A => {{1,0.5},{0.5,1}}.
        // s0=s1=2; R01 = cov01/(s0*s1) = 2/4 = 0.5. Diagonal exactly 1.
        void CorrelationKnown()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleMat(3, 2);
            A[0, 0] = 1f; A[0, 1] = 2f;
            A[1, 0] = 3f; A[1, 1] = 6f;
            A[2, 0] = 5f; A[2, 1] = 4f;

            var R = Stats.correlation(in A);

            Assert.AreEqual(2, R.M_Rows);
            Assert.AreEqual(2, R.N_Cols);

            AssertClose(R[0, 0], (double)1f, 1E-5f);
            AssertClose(R[1, 1], (double)1f, 1E-5f);
            AssertClose(R[0, 1], (double)0.5f, 1E-5f);
            AssertClose(R[1, 0], (double)0.5f, 1E-5f);

            // Symmetry.
            AssertClose(R[0, 1], R[1, 0], 1E-5f);

            arena.Dispose();
        }

        // Case 13: perfect (+1) and anti (-1) correlation.
        // col = (1,3,5). Identical columns => R01 == 1. Negated column => R01 == -1.
        void CorrelationPerfectAndAnti()
        {
            var arena = new Arena(Allocator.Persistent);

            var Apos = arena.doubleMat(3, 2);
            Apos[0, 0] = 1f; Apos[0, 1] = 1f;
            Apos[1, 0] = 3f; Apos[1, 1] = 3f;
            Apos[2, 0] = 5f; Apos[2, 1] = 5f;

            var Rpos = Stats.correlation(in Apos);
            AssertClose(Rpos[0, 1], (double)1f, 1E-5f);
            AssertClose(Rpos[1, 0], (double)1f, 1E-5f);

            var Aneg = arena.doubleMat(3, 2);
            Aneg[0, 0] = 1f; Aneg[0, 1] = -1f;
            Aneg[1, 0] = 3f; Aneg[1, 1] = -3f;
            Aneg[2, 0] = 5f; Aneg[2, 1] = -5f;

            var Rneg = Stats.correlation(in Aneg);
            AssertClose(Rneg[0, 1], (double)(-1f), 1E-5f);
            AssertClose(Rneg[1, 0], (double)(-1f), 1E-5f);

            arena.Dispose();
        }

        // Case 14: constant column has zero variance => off-diagonal correlations 0, diagonal 1.
        // col0 = (1,3,5), col1 = (7,7,7). Covariance C[1,1] == 0.
        void CorrelationConstantColumn()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleMat(3, 2);
            A[0, 0] = 1f; A[0, 1] = 7f;
            A[1, 0] = 3f; A[1, 1] = 7f;
            A[2, 0] = 5f; A[2, 1] = 7f;

            var C = Stats.covariance(in A);
            AssertClose(C[1, 1], (double)0f, 1E-5f);

            var R = Stats.correlation(in A);
            AssertClose(R[0, 0], (double)1f, 1E-5f);
            AssertClose(R[1, 1], (double)1f, 1E-5f);
            AssertClose(R[0, 1], (double)0f, 1E-5f);
            AssertClose(R[1, 0], (double)0f, 1E-5f);

            arena.Dispose();
        }

        // Case 15: single-variable matrices. 3x1 col=(2,4,6): mean 4, devs (-2,0,2), varSample 8/2=4.
        // covariance == {{4}}, correlation == {{1}}.
        void Covariance1Variable()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleMat(3, 1);
            A[0, 0] = 2f;
            A[1, 0] = 4f;
            A[2, 0] = 6f;

            var C = Stats.covariance(in A);
            Assert.AreEqual(1, C.M_Rows);
            Assert.AreEqual(1, C.N_Cols);
            AssertClose(C[0, 0], (double)4f, 1E-5f);

            var R = Stats.correlation(in A);
            Assert.AreEqual(1, R.M_Rows);
            Assert.AreEqual(1, R.N_Cols);
            AssertClose(R[0, 0], (double)1f, 1E-5f);

            arena.Dispose();
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff
        private void AssertClose(double a, double b, double precision)
        {
            double diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        // Fails (records got=NaN-marker) if the value is NaN. Used by the M<2 zero-fill guard test.
        private void AssertNotNaN(double a)
        {
            if (math.isnan(a) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = a;
                Fail[2] = (double)0;
                Fail[3] = a;
            }
            Assert.IsFalse(math.isnan(a));
        }
    }

    // Helper used by every managed runner to allocate/run/dispose with failure diagnostics.
    private void RunJob(TestJob.TestType type)
    {
        var fail = new NativeArray<double>(4, Allocator.TempJob);
        try {
            new TestJob() { Type = type, Fail = fail }.Run();
            // Under Burst a failed in-job assert logs an exception and aborts the job without
            // throwing to the caller - surface the recorded diagnostics here as well.
            if (fail[0] != (double)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");

        }
        catch (Exception e) {
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally {
            fail.Dispose();
        }
    }

    [Test]
    public void VectorVarianceStdDevTest()
    {
        RunJob(TestJob.TestType.VectorVarianceStdDev);
    }

    [Test]
    public void SingleElementVarianceTest()
    {
        RunJob(TestJob.TestType.SingleElementVariance);
    }

    [Test]
    public void ArgMinMaxVectorTest()
    {
        RunJob(TestJob.TestType.ArgMinMaxVector);
    }

    [Test]
    public void ArgMinMaxAllEqualTest()
    {
        RunJob(TestJob.TestType.ArgMinMaxAllEqual);
    }

    [Test]
    public void Matrix2x3Test()
    {
        RunJob(TestJob.TestType.Matrix2x3);
    }

    [Test]
    public void Matrix3x2NegativesTest()
    {
        RunJob(TestJob.TestType.Matrix3x2Negatives);
    }

    [Test]
    public void OneColumnMatrixTest()
    {
        RunJob(TestJob.TestType.OneColumnMatrix);
    }

    [Test]
    public void ArgMinMaxMatrixTest()
    {
        RunJob(TestJob.TestType.ArgMinMaxMatrix);
    }

    [Test]
    public void CovarianceKnownTest()
    {
        RunJob(TestJob.TestType.CovarianceKnown);
    }

    [Test]
    public void CovarianceDiagEqualsVarianceSampleTest()
    {
        RunJob(TestJob.TestType.CovarianceDiagEqualsVarianceSample);
    }

    [Test]
    public void CorrelationKnownTest()
    {
        RunJob(TestJob.TestType.CorrelationKnown);
    }

    [Test]
    public void CorrelationPerfectAndAntiTest()
    {
        RunJob(TestJob.TestType.CorrelationPerfectAndAnti);
    }

    [Test]
    public void CorrelationConstantColumnTest()
    {
        RunJob(TestJob.TestType.CorrelationConstantColumn);
    }

    [Test]
    public void Covariance1VariableTest()
    {
        RunJob(TestJob.TestType.Covariance1Variable);
    }

    [Test]
    public void RowColSumMeanTest()
    {
        RunJob(TestJob.TestType.RowColSumMean);
    }

    [Test]
    public void RowColNormsTest()
    {
        RunJob(TestJob.TestType.RowColNorms);
    }

    [Test]
    public void RefDestMatchesAllocatingTest()
    {
        RunJob(TestJob.TestType.RefDestMatchesAllocating);
    }

    [Test]
    public void CovarianceIntoSingleRowZeroFillTest()
    {
        RunJob(TestJob.TestType.CovarianceIntoSingleRowZeroFill);
    }

    // The zero-alloc primitive covarianceInto degrades to a zero-fill for M<2, but the allocating
    // wrapper covariance(in A) STILL throws — that contract is unchanged. (M=1, N=3.)
    [Test]
    public void CovarianceWrapperSingleRowStillThrows()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.doubleMat(1, 3);
        A[0, 0] = 5f; A[0, 1] = -2f; A[0, 2] = 9f;

        Assert.Throws<InvalidOperationException>(() => Stats.covariance(in A));

        arena.Dispose();
    }

    // Case 9: Managed throw-tests (must run on main thread, not inside a Burst job).

    [Test]
    public void VarianceSampleEmptyVectorThrows()
    {
        var arena = new Arena(Allocator.Persistent);

        var v = arena.doubleVec(0);

        Assert.Throws<InvalidOperationException>(() => Stats.varianceSample(in v));

        arena.Dispose();
    }

    [Test]
    public void VarianceSampleSingleElementThrows()
    {
        var arena = new Arena(Allocator.Persistent);

        var v = arena.doubleVec(1);
        v[0] = 5f;

        Assert.Throws<InvalidOperationException>(() => Stats.varianceSample(in v));

        arena.Dispose();
    }

    [Test]
    public void EmptyMatrixStatisticsThrows()
    {
        var arena = new Arena(Allocator.Persistent);

        // 0-row matrix (3 cols) is constructible; row/col stats must throw.
        var A = arena.doubleMat(0, 3);

        Assert.Throws<InvalidOperationException>(() => Stats.rowMin(in A));
        Assert.Throws<InvalidOperationException>(() => Stats.colMin(in A));
        Assert.Throws<InvalidOperationException>(() => Stats.rowVariance(in A));
        Assert.Throws<InvalidOperationException>(() => Stats.colVariance(in A));

        arena.Dispose();
    }

    // Case 16: covariance/correlation require M_Rows >= 2. A 1x2 matrix must throw.
    [Test]
    public void CovarianceCorrelationTooFewRowsThrows()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.doubleMat(1, 2);
        A[0, 0] = 1f; A[0, 1] = 2f;

        Assert.Throws<InvalidOperationException>(() => Stats.covariance(in A));
        Assert.Throws<InvalidOperationException>(() => Stats.correlation(in A));

        arena.Dispose();
    }

    // Case 17: covariance/correlation require N_Cols >= 1. A 0-column matrix must throw.
    [Test]
    public void CovarianceCorrelationZeroColumnsThrows()
    {
        var arena = new Arena(Allocator.Persistent);

        // 0-col matrix (3 rows) is constructible; covariance/correlation must throw.
        var A = arena.doubleMat(3, 0);

        Assert.Throws<InvalidOperationException>(() => Stats.covariance(in A));
        Assert.Throws<InvalidOperationException>(() => Stats.correlation(in A));

        arena.Dispose();
    }
}
