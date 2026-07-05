using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;

// Tests for the in-place transform family:
//   StatsOP: standardize / rescale / center / maxAbs / softmax  (flat<T>, *Rows, *Columns)
//   NormsOP: Normalize<T>(x, Norm) / NormalizeRows / NormalizeColumns
//   OP.Component: clampInPlace<T> (vec + matrix)
// Verification leans on the existing reduction kernels (mean / stdDev / rowSum / rowNormL2 ...)
// rather than hand-coding loops, so an assertion failure pins the transform, not the oracle.
public class fProxyTransformsTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            // standardize
            StandardizeVector,
            StandardizeConstant,
            StandardizeSingle,
            StandardizeFlatMatrix,
            StandardizeRows,
            StandardizeColumns,
            // rescale
            RescaleVector01,
            RescaleLoHi,
            RescaleConstant,
            RescaleForwardingEquals,
            RescaleRows,
            RescaleColumns,
            // center
            CenterVector,
            CenterRows,
            CenterColumns,
            // maxAbs
            MaxAbsVector,
            MaxAbsAllZero,
            MaxAbsRows,
            MaxAbsColumns,
            // softmax
            SoftmaxVector,
            SoftmaxSingle,
            SoftmaxMonotonic,
            SoftmaxStability,
            SoftmaxFlatMatrixWhole,
            SoftmaxRows,
            SoftmaxColumns,
            // norms
            NormalizeVectorAllNorms,
            NormalizeRowsL2,
            NormalizeColumnsL1,
            NormalizeZeroRowColumn,
            // clamp
            ClampVector,
            ClampMatrix,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<fProxy> Fail;

        // population std/mean of standardized data drifts a bit under float; algebraically-exact
        // endpoints (rescale lo/hi, center mean) are well inside this too.
        const float EPS = 1e-4f;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.StandardizeVector:        StandardizeVector(); break;
                case TestType.StandardizeConstant:      StandardizeConstant(); break;
                case TestType.StandardizeSingle:        StandardizeSingle(); break;
                case TestType.StandardizeFlatMatrix:    StandardizeFlatMatrix(); break;
                case TestType.StandardizeRows:          StandardizeRows(); break;
                case TestType.StandardizeColumns:       StandardizeColumns(); break;

                case TestType.RescaleVector01:          RescaleVector01(); break;
                case TestType.RescaleLoHi:              RescaleLoHi(); break;
                case TestType.RescaleConstant:          RescaleConstant(); break;
                case TestType.RescaleForwardingEquals:  RescaleForwardingEquals(); break;
                case TestType.RescaleRows:              RescaleRows(); break;
                case TestType.RescaleColumns:           RescaleColumns(); break;

                case TestType.CenterVector:             CenterVector(); break;
                case TestType.CenterRows:               CenterRows(); break;
                case TestType.CenterColumns:            CenterColumns(); break;

                case TestType.MaxAbsVector:             MaxAbsVector(); break;
                case TestType.MaxAbsAllZero:            MaxAbsAllZero(); break;
                case TestType.MaxAbsRows:               MaxAbsRows(); break;
                case TestType.MaxAbsColumns:            MaxAbsColumns(); break;

                case TestType.SoftmaxVector:            SoftmaxVector(); break;
                case TestType.SoftmaxSingle:            SoftmaxSingle(); break;
                case TestType.SoftmaxMonotonic:         SoftmaxMonotonic(); break;
                case TestType.SoftmaxStability:         SoftmaxStability(); break;
                case TestType.SoftmaxFlatMatrixWhole:   SoftmaxFlatMatrixWhole(); break;
                case TestType.SoftmaxRows:              SoftmaxRows(); break;
                case TestType.SoftmaxColumns:           SoftmaxColumns(); break;

                case TestType.NormalizeVectorAllNorms:  NormalizeVectorAllNorms(); break;
                case TestType.NormalizeRowsL2:          NormalizeRowsL2(); break;
                case TestType.NormalizeColumnsL1:       NormalizeColumnsL1(); break;
                case TestType.NormalizeZeroRowColumn:   NormalizeZeroRowColumn(); break;

                case TestType.ClampVector:              ClampVector(); break;
                case TestType.ClampMatrix:              ClampMatrix(); break;
            }
        }

        // ---------------- standardize ----------------

        // z-score a vector: result has population mean ~0 and population std ~1.
        void StandardizeVector()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.fProxyVec(8);
            v[0] = 2f; v[1] = 4f; v[2] = 4f; v[3] = 4f;
            v[4] = 5f; v[5] = 5f; v[6] = 7f; v[7] = 9f;

            Stats.standardize(in v);

            AssertClose(Stats.mean(in v), (fProxy)0f, (fProxy)EPS);
            AssertClose(Stats.stdDev(in v), (fProxy)1f, (fProxy)EPS);
            arena.Dispose();
        }

        // Constant axis (stdDev == 0) → zero-fill, no NaN.
        void StandardizeConstant()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.fProxyVec(5);
            for (int i = 0; i < 5; i++) v[i] = 3f;

            Stats.standardize(in v);

            for (int i = 0; i < 5; i++)
            {
                AssertClose(v[i], (fProxy)0f, (fProxy)EPS);
                AssertTrue(math.isfinite(v[i]));
            }
            arena.Dispose();
        }

        // Single element → 0 (mean equals the value, std is 0).
        void StandardizeSingle()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.fProxyVec(1);
            v[0] = 42f;

            Stats.standardize(in v);
            AssertClose(v[0], (fProxy)0f, (fProxy)EPS);
            arena.Dispose();
        }

        // Flat <T> on a matrix = WHOLE-matrix scope: all elements as one distribution.
        // Whole-matrix mean ~0, population std ~1 over every element.
        void StandardizeFlatMatrix()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyMat(2, 3);
            A[0, 0] = 1f; A[0, 1] = 2f; A[0, 2] = 3f;
            A[1, 0] = 4f; A[1, 1] = 6f; A[1, 2] = 8f;

            Stats.standardize(in A);

            AssertClose(Stats.mean(in A), (fProxy)0f, (fProxy)EPS);
            AssertClose(Stats.stdDev(in A), (fProxy)1f, (fProxy)EPS);
            arena.Dispose();
        }

        void StandardizeRows()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomMat(4, 5, -3f, 3f, 12345);
            Stats.standardizeRows(ref A);

            var rMean = Stats.rowMean(in A);
            var rStd = Stats.rowStdDev(in A);
            for (int r = 0; r < 4; r++)
            {
                AssertClose(rMean[r], (fProxy)0f, (fProxy)EPS);
                AssertClose(rStd[r], (fProxy)1f, (fProxy)EPS);
            }
            arena.Dispose();
        }

        void StandardizeColumns()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomMat(5, 4, -3f, 3f, 67890);
            Stats.standardizeColumns(ref A);

            var cMean = Stats.colMean(in A);
            var cStd = Stats.colStdDev(in A);
            for (int c = 0; c < 4; c++)
            {
                AssertClose(cMean[c], (fProxy)0f, (fProxy)EPS);
                AssertClose(cStd[c], (fProxy)1f, (fProxy)EPS);
            }
            arena.Dispose();
        }

        // ---------------- rescale ----------------

        // min→0, max→1, others strictly between.
        void RescaleVector01()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.fProxyVec(4);
            v[0] = 10f; v[1] = 20f; v[2] = 30f; v[3] = 50f;

            Stats.rescale(in v);

            AssertClose(v[0], (fProxy)0f, (fProxy)EPS);
            AssertClose(v[3], (fProxy)1f, (fProxy)EPS);
            // (20-10)/40 = 0.25, (30-10)/40 = 0.5
            AssertClose(v[1], (fProxy)0.25f, (fProxy)EPS);
            AssertClose(v[2], (fProxy)0.5f, (fProxy)EPS);
            AssertTrue(v[1] > (fProxy)0f && v[1] < (fProxy)1f);
            arena.Dispose();
        }

        // (lo,hi) overload: output exactly in [lo,hi], min→lo, max→hi.
        void RescaleLoHi()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.fProxyVec(4);
            v[0] = 10f; v[1] = 20f; v[2] = 30f; v[3] = 50f;
            fProxy lo = -2f, hi = 5f;

            Stats.rescale(in v, lo, hi);

            AssertClose(v[0], lo, (fProxy)EPS);
            AssertClose(v[3], hi, (fProxy)EPS);
            for (int i = 0; i < 4; i++)
                AssertTrue(v[i] >= lo - (fProxy)EPS && v[i] <= hi + (fProxy)EPS);
            // midpoint 0.25 of [lo,hi] = -2 + 0.25*7 = -0.25
            AssertClose(v[1], (fProxy)(-0.25f), (fProxy)EPS);
            arena.Dispose();
        }

        // Constant axis → every element set to lo.
        void RescaleConstant()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.fProxyVec(4);
            for (int i = 0; i < 4; i++) v[i] = 7f;

            Stats.rescale(in v, (fProxy)(-3f), (fProxy)9f);
            for (int i = 0; i < 4; i++)
            {
                AssertClose(v[i], (fProxy)(-3f), (fProxy)EPS);
                AssertTrue(math.isfinite(v[i]));
            }
            arena.Dispose();
        }

        // Forwarding overload rescale(x) must equal explicit rescale(x, 0, 1).
        void RescaleForwardingEquals()
        {
            var arena = new Arena(Allocator.Persistent);
            var a = arena.fProxyVec(5);
            var b = arena.fProxyVec(5);
            a[0] = -4f; a[1] = 1f; a[2] = 0f; a[3] = 9f; a[4] = 2f;
            for (int i = 0; i < 5; i++) b[i] = a[i];

            Stats.rescale(in a);
            Stats.rescale(in b, (fProxy)0f, (fProxy)1f);
            for (int i = 0; i < 5; i++)
                AssertClose(a[i], b[i], (fProxy)EPS);
            arena.Dispose();
        }

        void RescaleRows()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomMat(4, 5, -3f, 3f, 222);
            Stats.rescaleRows(ref A);
            var rMin = Stats.rowMin(in A);
            var rMax = Stats.rowMax(in A);
            for (int r = 0; r < 4; r++)
            {
                AssertClose(rMin[r], (fProxy)0f, (fProxy)EPS);
                AssertClose(rMax[r], (fProxy)1f, (fProxy)EPS);
            }
            arena.Dispose();
        }

        void RescaleColumns()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomMat(5, 4, -3f, 3f, 333);
            fProxy lo = 1f, hi = 3f;
            Stats.rescaleColumns(ref A, lo, hi);
            var cMin = Stats.colMin(in A);
            var cMax = Stats.colMax(in A);
            for (int c = 0; c < 4; c++)
            {
                AssertClose(cMin[c], lo, (fProxy)EPS);
                AssertClose(cMax[c], hi, (fProxy)EPS);
            }
            arena.Dispose();
        }

        // ---------------- center ----------------

        // Known vector {1,2,3} → {-1,0,1}; mean ~0.
        void CenterVector()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.fProxyVec(3);
            v[0] = 1f; v[1] = 2f; v[2] = 3f;

            Stats.center(in v);
            AssertClose(v[0], (fProxy)(-1f), (fProxy)EPS);
            AssertClose(v[1], (fProxy)0f, (fProxy)EPS);
            AssertClose(v[2], (fProxy)1f, (fProxy)EPS);
            AssertClose(Stats.mean(in v), (fProxy)0f, (fProxy)EPS);
            arena.Dispose();
        }

        void CenterRows()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomMat(4, 5, -5f, 5f, 444);
            Stats.centerRows(ref A);
            var rMean = Stats.rowMean(in A);
            for (int r = 0; r < 4; r++)
                AssertClose(rMean[r], (fProxy)0f, (fProxy)EPS);
            arena.Dispose();
        }

        void CenterColumns()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomMat(5, 4, -5f, 5f, 555);
            Stats.centerColumns(ref A);
            var cMean = Stats.colMean(in A);
            for (int c = 0; c < 4; c++)
                AssertClose(cMean[c], (fProxy)0f, (fProxy)EPS);
            arena.Dispose();
        }

        // ---------------- maxAbs ----------------

        // {2,-4,1} / 4 = {0.5,-1,0.25}; max-abs element → -1.
        void MaxAbsVector()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.fProxyVec(3);
            v[0] = 2f; v[1] = -4f; v[2] = 1f;

            Stats.maxAbs(in v);
            AssertClose(v[0], (fProxy)0.5f, (fProxy)EPS);
            AssertClose(v[1], (fProxy)(-1f), (fProxy)EPS);
            AssertClose(v[2], (fProxy)0.25f, (fProxy)EPS);
            AssertTrue(Norms.LInf(in v) <= (fProxy)1f + (fProxy)EPS);
            arena.Dispose();
        }

        // All-zero axis → unchanged (no divide, no NaN).
        void MaxAbsAllZero()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.fProxyVec(4);
            for (int i = 0; i < 4; i++) v[i] = 0f;

            Stats.maxAbs(in v);
            for (int i = 0; i < 4; i++)
            {
                AssertClose(v[i], (fProxy)0f, (fProxy)EPS);
                AssertTrue(math.isfinite(v[i]));
            }
            arena.Dispose();
        }

        void MaxAbsRows()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomMat(4, 5, -3f, 3f, 666);
            Stats.maxAbsRows(ref A);
            for (int r = 0; r < 4; r++)
            {
                fProxy mAbs = (fProxy)0f;
                for (int c = 0; c < 5; c++) mAbs = math.max(mAbs, math.abs(A[r, c]));
                AssertClose(mAbs, (fProxy)1f, (fProxy)EPS);
            }
            arena.Dispose();
        }

        void MaxAbsColumns()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomMat(5, 4, -3f, 3f, 777);
            Stats.maxAbsColumns(ref A);
            for (int c = 0; c < 4; c++)
            {
                fProxy mAbs = (fProxy)0f;
                for (int r = 0; r < 5; r++) mAbs = math.max(mAbs, math.abs(A[r, c]));
                AssertClose(mAbs, (fProxy)1f, (fProxy)EPS);
            }
            arena.Dispose();
        }

        // ---------------- softmax ----------------

        void SoftmaxVector()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.fProxyVec(5);
            v[0] = -1f; v[1] = 0f; v[2] = 2f; v[3] = 1f; v[4] = 3f;

            Stats.softmax(in v);
            AssertClose(Stats.sum(in v), (fProxy)1f, (fProxy)EPS);
            for (int i = 0; i < 5; i++)
                AssertTrue(v[i] > (fProxy)0f && v[i] < (fProxy)1f);
            arena.Dispose();
        }

        void SoftmaxSingle()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.fProxyVec(1);
            v[0] = 17f;
            Stats.softmax(in v);
            AssertClose(v[0], (fProxy)1f, (fProxy)EPS);
            arena.Dispose();
        }

        // Monotonic: strictly increasing input → strictly increasing probabilities.
        void SoftmaxMonotonic()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.fProxyVec(4);
            v[0] = 0.5f; v[1] = 1f; v[2] = 2f; v[3] = 4f;
            Stats.softmax(in v);
            for (int i = 1; i < 4; i++)
                AssertTrue(v[i] > v[i - 1]);
            arena.Dispose();
        }

        // KEY stability test: large magnitudes do NOT overflow. {1000,1001} stays finite, sums to 1.
        void SoftmaxStability()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.fProxyVec(2);
            v[0] = 1000f; v[1] = 1001f;
            Stats.softmax(in v);
            AssertTrue(math.isfinite(v[0]) && math.isfinite(v[1]));
            AssertClose(Stats.sum(in v), (fProxy)1f, (fProxy)EPS);
            AssertTrue(v[1] > v[0]);
            arena.Dispose();
        }

        // Flat <T> on a matrix = whole-matrix softmax: sums to 1 across ALL elements,
        // and NOT per-row (each row's partial sum is strictly below 1).
        void SoftmaxFlatMatrixWhole()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyMat(2, 3);
            A[0, 0] = 1f; A[0, 1] = 2f; A[0, 2] = 3f;
            A[1, 0] = 4f; A[1, 1] = 5f; A[1, 2] = 6f;

            Stats.softmax(in A);

            AssertClose(Stats.sum(in A), (fProxy)1f, (fProxy)EPS);
            var rSum = Stats.rowSum(in A);
            AssertTrue(rSum[0] < (fProxy)1f - (fProxy)EPS);
            AssertTrue(rSum[1] < (fProxy)1f - (fProxy)EPS);
            arena.Dispose();
        }

        void SoftmaxRows()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomMat(4, 5, -2f, 2f, 888);
            Stats.softmaxRows(ref A);
            var rSum = Stats.rowSum(in A);
            for (int r = 0; r < 4; r++)
            {
                AssertClose(rSum[r], (fProxy)1f, (fProxy)EPS);
                for (int c = 0; c < 5; c++)
                    AssertTrue(A[r, c] > (fProxy)0f && A[r, c] < (fProxy)1f);
            }
            arena.Dispose();
        }

        void SoftmaxColumns()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomMat(5, 4, -2f, 2f, 999);
            Stats.softmaxColumns(ref A);
            var cSum = Stats.colSum(in A);
            for (int c = 0; c < 4; c++)
            {
                AssertClose(cSum[c], (fProxy)1f, (fProxy)EPS);
                for (int r = 0; r < 5; r++)
                    AssertTrue(A[r, c] > (fProxy)0f && A[r, c] < (fProxy)1f);
            }
            arena.Dispose();
        }

        // ---------------- NormsOP normalize ----------------

        // Normalize<T>(x, Norm) for L1/L2/Linf → resulting vector has that norm ~1.
        void NormalizeVectorAllNorms()
        {
            var arena = new Arena(Allocator.Persistent);

            var v1 = arena.fProxyVec(4);
            v1[0] = 1f; v1[1] = -2f; v1[2] = 3f; v1[3] = -4f;
            Norms.normalize(in v1, Norm.L1);
            AssertClose(Norms.L1(in v1), (fProxy)1f, (fProxy)EPS);

            var v2 = arena.fProxyVec(4);
            v2[0] = 1f; v2[1] = -2f; v2[2] = 3f; v2[3] = -4f;
            Norms.normalize(in v2, Norm.L2);
            AssertClose(Norms.L2(in v2), (fProxy)1f, (fProxy)EPS);

            var vi = arena.fProxyVec(4);
            vi[0] = 1f; vi[1] = -2f; vi[2] = 3f; vi[3] = -4f;
            Norms.normalize(in vi, Norm.Linf);
            AssertClose(Norms.LInf(in vi), (fProxy)1f, (fProxy)EPS);

            arena.Dispose();
        }

        void NormalizeRowsL2()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomMat(4, 5, -3f, 3f, 1212);
            Norms.normalizeRows(ref A, Norm.L2);
            var rL2 = Stats.rowNormL2(in A);
            for (int r = 0; r < 4; r++)
                AssertClose(rL2[r], (fProxy)1f, (fProxy)EPS);
            arena.Dispose();
        }

        void NormalizeColumnsL1()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomMat(5, 4, -3f, 3f, 3434);
            Norms.normalizeColumns(ref A, Norm.L1);
            var cL1 = Stats.colNormL1(in A);
            for (int c = 0; c < 4; c++)
                AssertClose(cL1[c], (fProxy)1f, (fProxy)EPS);
            arena.Dispose();
        }

        // Zero row / zero column → left at 0 (no NaN / div-by-zero).
        void NormalizeZeroRowColumn()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(3, 3);
            A[0, 0] = 1f; A[0, 1] = 2f; A[0, 2] = 2f;
            A[1, 0] = 0f; A[1, 1] = 0f; A[1, 2] = 0f;
            A[2, 0] = 3f; A[2, 1] = 0f; A[2, 2] = 4f;

            Norms.normalizeRows(ref A, Norm.L2);
            for (int c = 0; c < 3; c++)
            {
                AssertClose(A[1, c], (fProxy)0f, (fProxy)EPS);
                AssertTrue(math.isfinite(A[1, c]));
            }
            var rL2 = Stats.rowNormL2(in A);
            AssertClose(rL2[0], (fProxy)1f, (fProxy)EPS);
            AssertClose(rL2[2], (fProxy)1f, (fProxy)EPS);

            var B = arena.fProxyMat(3, 3);
            B[0, 0] = 1f; B[0, 1] = 0f; B[0, 2] = 2f;
            B[1, 0] = 3f; B[1, 1] = 0f; B[1, 2] = 4f;
            B[2, 0] = 5f; B[2, 1] = 0f; B[2, 2] = 6f;

            Norms.normalizeColumns(ref B, Norm.L1);
            for (int r = 0; r < 3; r++)
            {
                AssertClose(B[r, 1], (fProxy)0f, (fProxy)EPS);
                AssertTrue(math.isfinite(B[r, 1]));
            }
            arena.Dispose();
        }

        // ---------------- clamp (fProxy) ----------------

        void ClampVector()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.fProxyVec(5);
            v[0] = -5f; v[1] = -1f; v[2] = 0f; v[3] = 3f; v[4] = 9f;

            fProxyComp.clampInPlace(v, (fProxy)(-1f), (fProxy)4f);
            AssertClose(v[0], (fProxy)(-1f), (fProxy)EPS);
            AssertClose(v[1], (fProxy)(-1f), (fProxy)EPS);
            AssertClose(v[2], (fProxy)0f, (fProxy)EPS);
            AssertClose(v[3], (fProxy)3f, (fProxy)EPS);
            AssertClose(v[4], (fProxy)4f, (fProxy)EPS);
            arena.Dispose();
        }

        void ClampMatrix()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyMat(2, 2);
            A[0, 0] = -10f; A[0, 1] = 0.5f;
            A[1, 0] = 2f;   A[1, 1] = 100f;

            fProxyComp.clampInPlace(A, (fProxy)0f, (fProxy)1f);
            AssertClose(A[0, 0], (fProxy)0f, (fProxy)EPS);
            AssertClose(A[0, 1], (fProxy)0.5f, (fProxy)EPS);
            AssertClose(A[1, 0], (fProxy)1f, (fProxy)EPS);
            AssertClose(A[1, 1], (fProxy)1f, (fProxy)EPS);
            arena.Dispose();
        }

        // ---------------- assertion helpers ----------------

        void AssertClose(fProxy a, fProxy b, fProxy precision)
        {
            fProxy diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertTrue(bool cond)
        {
            if (!cond && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = (fProxy)(-1);
                Fail[2] = (fProxy)(-1);
                Fail[3] = (fProxy)(-1);
            }
            Assert.IsTrue(cond);
        }
    }

    void RunJob(TestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void TransformCases(TestJob.TestType type) => RunJob(type);

    // lo > hi must throw ArgumentException — called directly on the test thread, not inside a Burst job.
    [Test]
    public void ClampLoGreaterThanHiThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var v = arena.fProxyVec(3);
        v[0] = -5f; v[1] = 0f; v[2] = 5f;
        Assert.Throws<ArgumentException>(() => fProxyComp.clampInPlace(v, (fProxy)4f, (fProxy)(-1f)));
        arena.Dispose();
    }
}
