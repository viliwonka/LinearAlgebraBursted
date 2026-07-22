using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.ML;
using LinearAlgebra.Sparse;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of the determinism conformance harness's stats/FFT/ML/query/gallery
    // groups (stats-core, fft, ml, histogram-resample-query, gallery-analysis). See
    // DeterminismDirect.fProxy.cs's header for the shared job/case-method convention and
    // docs/dev/spec-determinism-conformance-harness.md for the frozen op/group/root hash contract.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetStatsCoreJobFProxy : IJob
    {
        public fProxyN vec;
        public fProxyMxN A;
        public fProxyMxN Cov, Corr;
        public fProxyN rowMeanOut, colStdDevOut;
        public fProxyN standardizeVec, centerVec, rescaleVec;

        public NativeArray<uint> HashOut; // 8 slots

        public void Execute()
        {
            uint h = DetHash.Combine(0u, Stats.sum(in vec));
            h = DetHash.Combine(h, Stats.mean(in vec));
            h = DetHash.Combine(h, Stats.variance(in vec));
            h = DetHash.Combine(h, Stats.stdDev(in vec));
            h = DetHash.Combine(h, Stats.min(in vec));
            h = DetHash.Combine(h, Stats.max(in vec));
            h = DetHash.Combine(h, Stats.median(in vec));
            HashOut[0] = h;

            Stats.covarianceInto(in A, ref Cov);
            HashOut[1] = Hash.hash(in Cov);

            var corr = Stats.correlation(in A);
            HashOut[2] = Hash.hash(in corr);

            Stats.rowMean(in A, ref rowMeanOut);
            HashOut[3] = Hash.hash(in rowMeanOut);

            Stats.colStdDev(in A, ref colStdDevOut);
            HashOut[4] = Hash.hash(in colStdDevOut);

            Stats.standardize(in standardizeVec);
            HashOut[5] = Hash.hash(in standardizeVec);

            Stats.center(in centerVec);
            HashOut[6] = Hash.hash(in centerVec);

            Stats.rescale(in rescaleVec);
            HashOut[7] = Hash.hash(in rescaleVec);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetFftJobFProxy : IJob
    {
        public fProxyN re256, im256;
        public fProxyFFTCache ws256;
        public fProxyN reReal, imReal; // rfft outputs, length n/2+1
        public fProxyN realSignal256;
        public fProxyN realOut256;     // irfft output
        public fProxyN re128, im128;
        public fProxyFFTCache ws128;
        public fProxyN mag, pow;

        public NativeArray<uint> HashOut; // 7 slots

        public void Execute()
        {
            FFT.fft(ref re256, ref im256, in ws256);
            HashOut[0] = Hash.combine(Hash.hash(in re256), Hash.hash(in im256));

            FFT.magnitude(in re256, in im256, ref mag);
            HashOut[1] = Hash.hash(in mag);

            FFT.powerSpectrum(in re256, in im256, ref pow);
            HashOut[2] = Hash.hash(in pow);

            FFT.ifft(ref re256, ref im256, in ws256);
            HashOut[3] = Hash.combine(Hash.hash(in re256), Hash.hash(in im256));

            FFT.rfft(in realSignal256, ref reReal, ref imReal, in ws256);
            HashOut[4] = Hash.combine(Hash.hash(in reReal), Hash.hash(in imReal));

            FFT.irfft(in reReal, in imReal, ref realOut256, in ws256);
            HashOut[5] = Hash.hash(in realOut256);

            FFT.fft(ref re128, ref im128, in ws128);
            HashOut[6] = Hash.combine(Hash.hash(in re128), Hash.hash(in im128));
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetMlJobFProxy : IJob
    {
        public fProxyMxN X;
        public fProxyKMeansCache kmWs;
        public int K;
        public fProxyMxN centroids;
        public Indices assignment;

        public fProxyPCAModel modelCov;
        public fProxyPCAModel modelSvd;
        public fProxyPCAModel modelTrunc;
        public fProxyPCAModel modelRand;
        public fProxyMxN scores;

        public NativeArray<uint> HashOut; // 6 slots

        public void Execute()
        {
            KMeans.fit(in X, K, 12345u, 50, KMeansInit.KMeansPlusPlus, ref centroids, ref assignment, out fProxy inertia, out int iters, ref kmWs);
            uint h = Hash.hash(in centroids);
            h = DetHash.CombineIndices(h, in assignment);
            h = DetHash.Combine(h, inertia);
            h = DetHash.Combine(h, iters);
            HashOut[0] = h;

            bool okCov = PCA.fitCov(in X, ref modelCov, out EigenInfo eigInfo);
            h = Hash.hash(in modelCov.components);
            h = Hash.combine(h, Hash.hash(in modelCov.explainedVariance));
            h = DetHash.Combine(h, okCov);
            HashOut[1] = h;

            bool okSvd = PCA.fitSvd(in X, ref modelSvd, out SVDInfo svdInfo);
            h = Hash.hash(in modelSvd.components);
            h = Hash.combine(h, Hash.hash(in modelSvd.explainedVariance));
            h = DetHash.Combine(h, okSvd);
            HashOut[2] = h;

            bool okTrunc = PCA.fitSvdTruncated(in X, ref modelTrunc, modelTrunc.k, out SVDInfo truncInfo);
            h = Hash.hash(in modelTrunc.components);
            h = Hash.combine(h, Hash.hash(in modelTrunc.explainedVariance));
            h = DetHash.Combine(h, okTrunc);
            HashOut[3] = h;

            bool okRand = PCA.fitRandomized(in X, ref modelRand, modelRand.k, out SVDInfo randInfo);
            h = Hash.hash(in modelRand.components);
            h = Hash.combine(h, Hash.hash(in modelRand.explainedVariance));
            h = DetHash.Combine(h, okRand);
            HashOut[4] = h;

            PCA.transform(in X, in modelCov, ref scores);
            HashOut[5] = Hash.hash(in scores);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetHistQueryJobFProxy : IJob
    {
        public fProxyN data;
        public Indices counts;
        public fProxyN cdf;
        public fProxyN resampleSrc;
        public fProxyN resampleDstNearest, resampleDstLinear, resampleDstCubic;
        public fProxyMxN Arows; public fProxyN q;
        public Indices kIdx; public fProxyN kScores;
        public Indices argMaxIdx; public fProxyN argMaxVal;
        public Indices radiusIdx;
        public fProxyN selA, selB; public boolN selC; public fProxyN selDest;

        public NativeArray<uint> HashOut; // 10 slots

        public void Execute()
        {
            Histogram.histogramInto(in data, -3f, 3f, ref counts);
            HashOut[0] = DetHash.CombineIndices(0u, in counts);

            Histogram.cdfInto(in data, -3f, 3f, ref cdf);
            HashOut[1] = Hash.hash(in cdf);

            Resample.resampleInto(in resampleSrc, ref resampleDstNearest, Interp.Nearest, EdgeMode.Clamp);
            HashOut[2] = Hash.hash(in resampleDstNearest);

            Resample.resampleInto(in resampleSrc, ref resampleDstLinear, Interp.Linear, EdgeMode.Clamp);
            HashOut[3] = Hash.hash(in resampleDstLinear);

            Resample.resampleInto(in resampleSrc, ref resampleDstCubic, Interp.Cubic, EdgeMode.Clamp);
            HashOut[4] = Hash.hash(in resampleDstCubic);

            Query.nearestRow(in Arows, in q, Metric.Euclidean, out int nearestIdx, out fProxy nearestScore);
            uint h = DetHash.Combine(0u, nearestIdx);
            h = DetHash.Combine(h, nearestScore);
            HashOut[5] = h;

            int kFound = Query.kNearestRows(in Arows, in q, kIdx.N, Metric.Euclidean, ref kIdx, ref kScores);
            h = DetHash.CombineIndices(0u, in kIdx);
            h = Hash.combine(h, Hash.hash(in kScores));
            h = DetHash.Combine(h, kFound);
            HashOut[6] = h;

            Query.rowArgMax(in Arows, ref argMaxIdx, ref argMaxVal);
            h = DetHash.CombineIndices(0u, in argMaxIdx);
            h = Hash.combine(h, Hash.hash(in argMaxVal));
            HashOut[7] = h;

            for (int i = 0; i < radiusIdx.N; i++) radiusIdx[i] = 0;
            int radiusCount = Query.rowsWithinRadius(in Arows, in q, (fProxy)5, Metric.Euclidean, ref radiusIdx);
            h = DetHash.CombineIndices(0u, in radiusIdx);
            h = DetHash.Combine(h, radiusCount);
            HashOut[8] = h;

            Select.select(in selA, in selB, in selC, ref selDest);
            HashOut[9] = Hash.hash(in selDest);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetGalleryAnalysisJobFProxy : IJob
    {
        public fProxyMxN hilbert, pascal, lehmer, minij, kms, lap1d;
        public fProxyBSR lap2d;

        public NativeArray<uint> HashOut; // 9 slots

        public unsafe void Execute()
        {
            HashOut[0] = Hash.hash(in hilbert);
            HashOut[1] = Hash.hash(in pascal);
            HashOut[2] = Hash.hash(in lehmer);
            HashOut[3] = Hash.hash(in minij);
            HashOut[4] = Hash.hash(in kms);
            HashOut[5] = Hash.hash(in lap1d);

            uint h = DetHash.Combine(0u, (byte*)lap2d.RowPtr.Ptr, lap2d.RowPtr.Length * sizeof(int));
            h = DetHash.Combine(h, (byte*)lap2d.ColInd.Ptr, lap2d.ColInd.Length * sizeof(int));
            h = DetHash.Combine(h, (byte*)lap2d.Values.Ptr, lap2d.Values.Length * sizeof(fProxy));
            HashOut[6] = h;

            bool sym = Analysis.isSymmetric(in hilbert);
            bool orth = Analysis.isOrthogonal(in hilbert, (fProxy)1e-3);
            bool diag = Analysis.isDiagonal(in kms);
            h = DetHash.Combine(0u, sym);
            HashOut[7] = h;
            h = DetHash.Combine(0u, orth);
            h = DetHash.Combine(h, diag);
            HashOut[8] = h;
        }
    }

    public static partial class DeterminismStatsMl
    {
        public static (string id, uint hash)[] Case_StatsCoreFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(2654435761u ^ 0x0005u);

            const int n = 200, p = 7;
            var vec = arena.fProxyVec(n); for (int i = 0; i < n; i++) vec[i] = rng.NextFProxy(-10f, 10f);
            var A = arena.fProxyMat(n, p);
            for (int r = 0; r < n; r++) for (int c = 0; c < p; c++) A[r, c] = rng.NextFProxy(-5f, 5f);

            var Cov = arena.fProxyMat(p, p);
            var Corr = arena.fProxyMat(p, p);
            var rowMeanOut = arena.fProxyVec(n);
            var colStdDevOut = arena.fProxyVec(p);

            var standardizeVec = arena.fProxyVec(n); for (int i = 0; i < n; i++) standardizeVec[i] = vec[i];
            var centerVec = arena.fProxyVec(n); for (int i = 0; i < n; i++) centerVec[i] = vec[i];
            var rescaleVec = arena.fProxyVec(n); for (int i = 0; i < n; i++) rescaleVec[i] = vec[i];

            var hashOut = new NativeArray<uint>(8, Allocator.Persistent);
            var job = new DetStatsCoreJobFProxy
            {
                vec = vec, A = A, Cov = Cov, Corr = Corr, rowMeanOut = rowMeanOut, colStdDevOut = colStdDevOut,
                standardizeVec = standardizeVec, centerVec = centerVec, rescaleVec = rescaleVec, HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("stats-core/scalars.fProxy.n200", hashOut[0]),
                ("stats-core/covariance.fProxy.200x7", hashOut[1]),
                ("stats-core/correlation.fProxy.200x7", hashOut[2]),
                ("stats-core/rowMean.fProxy.200x7", hashOut[3]),
                ("stats-core/colStdDev.fProxy.200x7", hashOut[4]),
                ("stats-core/standardize.fProxy.n200", hashOut[5]),
                ("stats-core/center.fProxy.n200", hashOut[6]),
                ("stats-core/rescale.fProxy.n200", hashOut[7]),
            };
            hashOut.Dispose();
            arena.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_FftFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(2654435761u ^ 0x000Cu);

            const int n256 = 256, n128 = 128;
            var re256 = arena.fProxyVec(n256); var im256 = arena.fProxyVec(n256);
            for (int i = 0; i < n256; i++) { re256[i] = rng.NextFProxy(-1f, 1f); im256[i] = (fProxy)0; }
            var ws256 = arena.fProxyFFTCache(n256);
            var mag = arena.fProxyVec(n256); var pow = arena.fProxyVec(n256);

            var realSignal256 = arena.fProxyVec(n256); for (int i = 0; i < n256; i++) realSignal256[i] = rng.NextFProxy(-1f, 1f);
            var reReal = arena.fProxyVec(n256 / 2 + 1); var imReal = arena.fProxyVec(n256 / 2 + 1);
            var realOut256 = arena.fProxyVec(n256);

            var re128 = arena.fProxyVec(n128); var im128 = arena.fProxyVec(n128);
            for (int i = 0; i < n128; i++) { re128[i] = rng.NextFProxy(-1f, 1f); im128[i] = (fProxy)0; }
            var ws128 = arena.fProxyFFTCache(n128);

            var hashOut = new NativeArray<uint>(7, Allocator.Persistent);
            var job = new DetFftJobFProxy
            {
                re256 = re256, im256 = im256, ws256 = ws256, mag = mag, pow = pow,
                realSignal256 = realSignal256, reReal = reReal, imReal = imReal, realOut256 = realOut256,
                re128 = re128, im128 = im128, ws128 = ws128, HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("fft/fft.fProxy.n256", hashOut[0]),
                ("fft/magnitude.fProxy.n256", hashOut[1]),
                ("fft/powerSpectrum.fProxy.n256", hashOut[2]),
                ("fft/ifft.fProxy.n256", hashOut[3]),
                ("fft/rfft.fProxy.n256", hashOut[4]),
                ("fft/irfft.fProxy.n256", hashOut[5]),
                ("fft/fft.fProxy.n128.mixed", hashOut[6]),
            };
            hashOut.Dispose();
            arena.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_MlFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(2654435761u ^ 0x0016u);

            const int n = 200, d = 8, k = 5, kPca = 4;
            var X = arena.fProxyMat(n, d);
            for (int r = 0; r < n; r++) for (int c = 0; c < d; c++) X[r, c] = rng.NextFProxy(-3f, 3f);

            var kmWs = arena.fProxyKMeansCache(n, d, k);
            var centroids = arena.fProxyMat(k, d);
            var assignment = new Indices(n, Allocator.Persistent);
            var modelCov = arena.fProxyPCAModel(d, d);
            var modelSvd = arena.fProxyPCAModel(d, d);
            var modelTrunc = arena.fProxyPCAModel(d, kPca);
            var modelRand = arena.fProxyPCAModel(d, kPca);
            var scores = arena.fProxyMat(n, d);

            var hashOut = new NativeArray<uint>(6, Allocator.Persistent);
            var job = new DetMlJobFProxy
            {
                X = X, kmWs = kmWs, K = k, centroids = centroids, assignment = assignment,
                modelCov = modelCov, modelSvd = modelSvd, modelTrunc = modelTrunc, modelRand = modelRand,
                scores = scores, HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("ml/kmeans.fit.fProxy.200x8.k5", hashOut[0]),
                ("ml/pca.fitCov.fProxy.200x8", hashOut[1]),
                ("ml/pca.fitSvd.fProxy.200x8", hashOut[2]),
                ("ml/pca.fitSvdTruncated.fProxy.200x8.k4", hashOut[3]),
                ("ml/pca.fitRandomized.fProxy.200x8.k4", hashOut[4]),
                ("ml/pca.transform.fProxy.200x8", hashOut[5]),
            };
            hashOut.Dispose();
            assignment.Dispose();
            arena.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_HistogramResampleQueryFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(2654435761u ^ 0x0017u);

            const int n = 500;
            var data = arena.fProxyVec(n); for (int i = 0; i < n; i++) data[i] = rng.NextFProxy(-3f, 3f);
            var counts = new Indices(20, Allocator.Persistent);
            var cdf = arena.fProxyVec(n);

            const int srcLen = 40, dstLen = 100;
            var resampleSrc = arena.fProxyVec(srcLen); for (int i = 0; i < srcLen; i++) resampleSrc[i] = rng.NextFProxy(-1f, 1f);
            var resampleDstNearest = arena.fProxyVec(dstLen);
            var resampleDstLinear = arena.fProxyVec(dstLen);
            var resampleDstCubic = arena.fProxyVec(dstLen);

            const int rows = 53, cols = 8;
            var Arows = arena.fProxyMat(rows, cols);
            for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) Arows[r, c] = rng.NextFProxy(-5f, 5f);
            var q = arena.fProxyVec(cols); for (int i = 0; i < cols; i++) q[i] = rng.NextFProxy(-5f, 5f);

            const int k = 5;
            var kIdx = new Indices(k, Allocator.Persistent);
            var kScores = arena.fProxyVec(k);
            var argMaxIdx = new Indices(rows, Allocator.Persistent);
            var argMaxVal = arena.fProxyVec(rows);
            var radiusIdx = new Indices(rows, Allocator.Persistent);

            const int selN = 64;
            var selA = arena.fProxyVec(selN); var selB = arena.fProxyVec(selN);
            var selC = arena.boolVec(selN);
            var selDest = arena.fProxyVec(selN);
            for (int i = 0; i < selN; i++) { selA[i] = rng.NextFProxy(-1f, 1f); selB[i] = rng.NextFProxy(-1f, 1f); selC[i] = (i % 3 == 0); }

            var hashOut = new NativeArray<uint>(10, Allocator.Persistent);
            var job = new DetHistQueryJobFProxy
            {
                data = data, counts = counts, cdf = cdf,
                resampleSrc = resampleSrc, resampleDstNearest = resampleDstNearest, resampleDstLinear = resampleDstLinear, resampleDstCubic = resampleDstCubic,
                Arows = Arows, q = q, kIdx = kIdx, kScores = kScores, argMaxIdx = argMaxIdx, argMaxVal = argMaxVal, radiusIdx = radiusIdx,
                selA = selA, selB = selB, selC = selC, selDest = selDest, HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("histogram-resample-query/histogramInto.fProxy.n500", hashOut[0]),
                ("histogram-resample-query/cdfInto.fProxy.n500", hashOut[1]),
                ("histogram-resample-query/resampleInto.nearest.fProxy.40to100", hashOut[2]),
                ("histogram-resample-query/resampleInto.linear.fProxy.40to100", hashOut[3]),
                ("histogram-resample-query/resampleInto.cubic.fProxy.40to100", hashOut[4]),
                ("histogram-resample-query/nearestRow.fProxy.53x8", hashOut[5]),
                ("histogram-resample-query/kNearestRows.fProxy.53x8.k5", hashOut[6]),
                ("histogram-resample-query/rowArgMax.fProxy.53x8", hashOut[7]),
                ("histogram-resample-query/rowsWithinRadius.fProxy.53x8", hashOut[8]),
                ("histogram-resample-query/select.fProxy.n64", hashOut[9]),
            };
            hashOut.Dispose();
            counts.Dispose(); kIdx.Dispose(); argMaxIdx.Dispose(); radiusIdx.Dispose();
            arena.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_GalleryAnalysisFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            const int n = 32;
            var hilbert = arena.fProxyHilbert(n);
            var pascal = arena.fProxyPascal(n);
            var lehmer = arena.fProxyLehmer(n);
            var minij = arena.fProxyMinIJ(n);
            var kms = arena.fProxyKMS(n, (fProxy)0.5);
            var lap1d = arena.fProxyLaplacian1D(n);
            var lap2d = arena.fProxyLaplacian2D(6, 6);

            var hashOut = new NativeArray<uint>(9, Allocator.Persistent);
            var job = new DetGalleryAnalysisJobFProxy
            {
                hilbert = hilbert, pascal = pascal, lehmer = lehmer, minij = minij, kms = kms, lap1d = lap1d, lap2d = lap2d,
                HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("gallery-analysis/hilbert.fProxy.n32", hashOut[0]),
                ("gallery-analysis/pascal.fProxy.n32", hashOut[1]),
                ("gallery-analysis/lehmer.fProxy.n32", hashOut[2]),
                ("gallery-analysis/minij.fProxy.n32", hashOut[3]),
                ("gallery-analysis/kms.fProxy.n32", hashOut[4]),
                ("gallery-analysis/laplacian1d.fProxy.n32", hashOut[5]),
                ("gallery-analysis/laplacian2d.fProxy.6x6", hashOut[6]),
                ("gallery-analysis/isSymmetric.fProxy.hilbert", hashOut[7]),
                ("gallery-analysis/isOrthogonalIsDiagonal.fProxy", hashOut[8]),
            };
            hashOut.Dispose();
            arena.Dispose();
            return result;
        }
    }
}
