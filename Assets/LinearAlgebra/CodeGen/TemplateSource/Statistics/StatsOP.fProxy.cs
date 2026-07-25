using System.Runtime.CompilerServices;

namespace BULA
{
    // Public statistics surface. The whole-array reductions/transforms accept a vector (fProxyN)
    // or a matrix (fProxyMxN, whole-matrix scope) and forward, inlined, to the generic bodies in
    // fProxyStatsCore -- so the class is a bare, prefix-free `Stats` while the (type-identical
    // across float/double) generic signatures live in the distinct floatStatsCore/doubleStatsCore
    // and never collide. The per-axis matrix ops (rowMean/colMean/...) forward their concrete bodies.
    public static partial class Stats
    {
        // ---- whole-array reductions (vector or whole-matrix) ----
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy sum(in fProxyN   x) => fProxyStatsCore.sum(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy sum(in fProxyMxN x) => fProxyStatsCore.sum(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy mean(in fProxyN   x) => fProxyStatsCore.mean(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy mean(in fProxyMxN x) => fProxyStatsCore.mean(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy variance(in fProxyN   x) => fProxyStatsCore.variance(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy variance(in fProxyMxN x) => fProxyStatsCore.variance(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy stdDev(in fProxyN   x) => fProxyStatsCore.stdDev(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy stdDev(in fProxyMxN x) => fProxyStatsCore.stdDev(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy varianceSample(in fProxyN   x) => fProxyStatsCore.varianceSample(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy varianceSample(in fProxyMxN x) => fProxyStatsCore.varianceSample(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy stdDevSample(in fProxyN   x) => fProxyStatsCore.stdDevSample(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy stdDevSample(in fProxyMxN x) => fProxyStatsCore.stdDevSample(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int argmin(in fProxyN   x) => fProxyStatsCore.argmin(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int argmin(in fProxyMxN x) => fProxyStatsCore.argmin(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int argmax(in fProxyN   x) => fProxyStatsCore.argmax(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static int argmax(in fProxyMxN x) => fProxyStatsCore.argmax(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy min(in fProxyN   x) => fProxyStatsCore.min(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy min(in fProxyMxN x) => fProxyStatsCore.min(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy max(in fProxyN   x) => fProxyStatsCore.max(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy max(in fProxyMxN x) => fProxyStatsCore.max(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy median(in fProxyN   x) => fProxyStatsCore.median(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy median(in fProxyMxN x) => fProxyStatsCore.median(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy range(in fProxyN   x) => fProxyStatsCore.range(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxy range(in fProxyMxN x) => fProxyStatsCore.range(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyMeanMinMaxRangeStats meanMinMaxRange(in fProxyN   x) => fProxyStatsCore.meanMinMaxRange(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyMeanMinMaxRangeStats meanMinMaxRange(in fProxyMxN x) => fProxyStatsCore.meanMinMaxRange(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyFullStats meanMinMaxRange_medianIQRstdDevVariance(in fProxyN   x) => fProxyStatsCore.meanMinMaxRange_medianIQRstdDevVariance(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyFullStats meanMinMaxRange_medianIQRstdDevVariance(in fProxyMxN x) => fProxyStatsCore.meanMinMaxRange_medianIQRstdDevVariance(in x);

        // ---- whole-array in-place transforms (vector or whole-matrix) ----
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void standardize(in fProxyN   x) => fProxyStatsCore.standardize(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void standardize(in fProxyMxN x) => fProxyStatsCore.standardize(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void rescale(in fProxyN   x) => fProxyStatsCore.rescale(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void rescale(in fProxyMxN x) => fProxyStatsCore.rescale(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void rescale(in fProxyN   x, fProxy lo, fProxy hi) => fProxyStatsCore.rescale(in x, lo, hi);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void rescale(in fProxyMxN x, fProxy lo, fProxy hi) => fProxyStatsCore.rescale(in x, lo, hi);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void center(in fProxyN   x) => fProxyStatsCore.center(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void center(in fProxyMxN x) => fProxyStatsCore.center(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void maxAbs(in fProxyN   x) => fProxyStatsCore.maxAbs(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void maxAbs(in fProxyMxN x) => fProxyStatsCore.maxAbs(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void softmax(in fProxyN   x) => fProxyStatsCore.softmax(in x);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void softmax(in fProxyMxN x) => fProxyStatsCore.softmax(in x);

        // ---- per-axis matrix reductions ----
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void    rowSum(in fProxyMxN A, ref fProxyN dest) => fProxyStatsCore.rowSum(in A, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyN rowSum(in fProxyMxN A) => fProxyStatsCore.rowSum(in A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void    colSum(in fProxyMxN A, ref fProxyN dest) => fProxyStatsCore.colSum(in A, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyN colSum(in fProxyMxN A) => fProxyStatsCore.colSum(in A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void    rowMean(in fProxyMxN A, ref fProxyN dest) => fProxyStatsCore.rowMean(in A, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyN rowMean(in fProxyMxN A) => fProxyStatsCore.rowMean(in A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void    colMean(in fProxyMxN A, ref fProxyN dest) => fProxyStatsCore.colMean(in A, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyN colMean(in fProxyMxN A) => fProxyStatsCore.colMean(in A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void    rowMin(in fProxyMxN A, ref fProxyN dest) => fProxyStatsCore.rowMin(in A, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyN rowMin(in fProxyMxN A) => fProxyStatsCore.rowMin(in A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void    rowMax(in fProxyMxN A, ref fProxyN dest) => fProxyStatsCore.rowMax(in A, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyN rowMax(in fProxyMxN A) => fProxyStatsCore.rowMax(in A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void    colMin(in fProxyMxN A, ref fProxyN dest) => fProxyStatsCore.colMin(in A, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyN colMin(in fProxyMxN A) => fProxyStatsCore.colMin(in A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void    colMax(in fProxyMxN A, ref fProxyN dest) => fProxyStatsCore.colMax(in A, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyN colMax(in fProxyMxN A) => fProxyStatsCore.colMax(in A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void    rowVariance(in fProxyMxN A, ref fProxyN dest) => fProxyStatsCore.rowVariance(in A, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyN rowVariance(in fProxyMxN A) => fProxyStatsCore.rowVariance(in A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void    colVariance(in fProxyMxN A, ref fProxyN dest) => fProxyStatsCore.colVariance(in A, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyN colVariance(in fProxyMxN A) => fProxyStatsCore.colVariance(in A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void    rowStdDev(in fProxyMxN A, ref fProxyN dest) => fProxyStatsCore.rowStdDev(in A, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyN rowStdDev(in fProxyMxN A) => fProxyStatsCore.rowStdDev(in A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void    colStdDev(in fProxyMxN A, ref fProxyN dest) => fProxyStatsCore.colStdDev(in A, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyN colStdDev(in fProxyMxN A) => fProxyStatsCore.colStdDev(in A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void    rowNormL1(in fProxyMxN A, ref fProxyN dest) => fProxyStatsCore.rowNormL1(in A, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyN rowNormL1(in fProxyMxN A) => fProxyStatsCore.rowNormL1(in A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void    rowNormL2(in fProxyMxN A, ref fProxyN dest) => fProxyStatsCore.rowNormL2(in A, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyN rowNormL2(in fProxyMxN A) => fProxyStatsCore.rowNormL2(in A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void    colNormL1(in fProxyMxN A, ref fProxyN dest) => fProxyStatsCore.colNormL1(in A, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyN colNormL1(in fProxyMxN A) => fProxyStatsCore.colNormL1(in A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void    colNormL2(in fProxyMxN A, ref fProxyN dest) => fProxyStatsCore.colNormL2(in A, ref dest);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyN colNormL2(in fProxyMxN A) => fProxyStatsCore.colNormL2(in A);

        // ---- covariance / correlation ----
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void     covarianceInto(in fProxyMxN A, ref fProxyMxN C) => fProxyStatsCore.covarianceInto(in A, ref C);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyMxN covariance(in fProxyMxN A) => fProxyStatsCore.covariance(in A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static fProxyMxN correlation(in fProxyMxN A) => fProxyStatsCore.correlation(in A);

        // ---- per-axis in-place transforms ----
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void standardizeRows(ref fProxyMxN A) => fProxyStatsCore.standardizeRows(ref A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void standardizeColumns(ref fProxyMxN A) => fProxyStatsCore.standardizeColumns(ref A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void rescaleRows(ref fProxyMxN A) => fProxyStatsCore.rescaleRows(ref A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void rescaleRows(ref fProxyMxN A, fProxy lo, fProxy hi) => fProxyStatsCore.rescaleRows(ref A, lo, hi);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void rescaleColumns(ref fProxyMxN A) => fProxyStatsCore.rescaleColumns(ref A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void rescaleColumns(ref fProxyMxN A, fProxy lo, fProxy hi) => fProxyStatsCore.rescaleColumns(ref A, lo, hi);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void centerRows(ref fProxyMxN A) => fProxyStatsCore.centerRows(ref A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void centerColumns(ref fProxyMxN A) => fProxyStatsCore.centerColumns(ref A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void maxAbsRows(ref fProxyMxN A) => fProxyStatsCore.maxAbsRows(ref A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void maxAbsColumns(ref fProxyMxN A) => fProxyStatsCore.maxAbsColumns(ref A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void softmaxRows(ref fProxyMxN A) => fProxyStatsCore.softmaxRows(ref A);
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static void softmaxColumns(ref fProxyMxN A) => fProxyStatsCore.softmaxColumns(ref A);
    }
}
