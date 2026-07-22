using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Internal;

namespace LinearAlgebra.Benchmarks
{
    // GENERATED per-dtype half of the determinism conformance harness's SECTION B groups (detmath,
    // elementwise-transcendental, random-samplers, softmax, dft-signal) -- native-math-sensitive: in
    // the default build every op here routes through DetMath (deterministic, cross-arch), but under
    // LINALG_NATIVE_MATH (DetMath.UseNative) these flip to raw math.* and are expected to diverge.
    // Folds into ROOT-B, kept separate from the main ROOT. See
    // docs/dev/spec-determinism-conformance-harness.md section 6 (section B) and DeterminismDirect.
    // fProxy.cs's header for the shared job/case-method convention.

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetMathGridJobFProxy : IJob
    {
        public fProxyN grid, gridPos, gridUnit, gridGeOne;
        public fProxyN outVec, sinOut, cosOut;

        public NativeArray<uint> HashOut; // 20 slots

        public void Execute()
        {
            int n = grid.N;

            for (int i = 0; i < n; i++) outVec[i] = DetMath.Exp(grid[i]);
            HashOut[0] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = DetMath.Exp2(grid[i]);
            HashOut[1] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = DetMath.Exp10(grid[i]);
            HashOut[2] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = DetMath.Log(gridPos[i]);
            HashOut[3] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = DetMath.Log2(gridPos[i]);
            HashOut[4] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = DetMath.Log10(gridPos[i]);
            HashOut[5] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = DetMath.Pow(gridPos[i], (fProxy)2.5);
            HashOut[6] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = DetMath.Pow(grid[i], 3);
            HashOut[7] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = DetMath.Sin(grid[i]);
            HashOut[8] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = DetMath.Cos(grid[i]);
            HashOut[9] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = DetMath.Tan(grid[i]);
            HashOut[10] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) DetMath.SinCos(grid[i], out sinOut[i], out cosOut[i]);
            HashOut[11] = Hash.combine(Hash.hash(in sinOut), Hash.hash(in cosOut));
            for (int i = 0; i < n; i++) outVec[i] = DetMath.Atan(grid[i]);
            HashOut[12] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = DetMath.Atan2(grid[i], gridPos[i]);
            HashOut[13] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = DetMath.Asin(gridUnit[i]);
            HashOut[14] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = DetMath.Acos(gridUnit[i]);
            HashOut[15] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = DetMath.Sinh(grid[i]);
            HashOut[16] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = DetMath.Cosh(grid[i]);
            HashOut[17] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = DetMath.Tanh(grid[i]);
            HashOut[18] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = DetMath.Acosh(gridGeOne[i]);
            HashOut[19] = Hash.hash(in outVec);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetElementwiseTranscendentalJobFProxy : IJob
    {
        public fProxyN general, positive, unit, geOne;
        public fProxyN scratch;

        public NativeArray<uint> HashOut; // 20 slots

        void CopyInto(in fProxyN src)
        {
            for (int i = 0; i < scratch.N; i++) scratch[i] = src[i];
        }

        public unsafe void Execute()
        {
            CopyInto(in general); scratch.expInPlace(); HashOut[0] = Hash.hash(in scratch);
            CopyInto(in general); scratch.exp2InPlace(); HashOut[1] = Hash.hash(in scratch);
            CopyInto(in general); scratch.exp10InPlace(); HashOut[2] = Hash.hash(in scratch);
            CopyInto(in positive); scratch.logInPlace(); HashOut[3] = Hash.hash(in scratch);
            CopyInto(in positive); scratch.log2InPlace(); HashOut[4] = Hash.hash(in scratch);
            CopyInto(in positive); scratch.log10InPlace(); HashOut[5] = Hash.hash(in scratch);
            CopyInto(in general); scratch.sinInPlace(); HashOut[6] = Hash.hash(in scratch);
            CopyInto(in general); scratch.cosInPlace(); HashOut[7] = Hash.hash(in scratch);
            CopyInto(in general); scratch.tanInPlace(); HashOut[8] = Hash.hash(in scratch);
            CopyInto(in unit); scratch.asinInPlace(); HashOut[9] = Hash.hash(in scratch);
            CopyInto(in unit); scratch.acosInPlace(); HashOut[10] = Hash.hash(in scratch);
            CopyInto(in general); scratch.atanInPlace(); HashOut[11] = Hash.hash(in scratch);
            CopyInto(in general); scratch.atan2InPlace(positive); HashOut[12] = Hash.hash(in scratch);
            CopyInto(in general); scratch.sinhInPlace(); HashOut[13] = Hash.hash(in scratch);
            CopyInto(in general); scratch.coshInPlace(); HashOut[14] = Hash.hash(in scratch);
            CopyInto(in general); scratch.tanhInPlace(); HashOut[15] = Hash.hash(in scratch);
            CopyInto(in geOne); scratch.acoshInPlace(); HashOut[16] = Hash.hash(in scratch);
            CopyInto(in general); scratch.powInPlace(3); HashOut[17] = Hash.hash(in scratch);
            CopyInto(in positive); scratch.rsqrtInPlace(); HashOut[18] = Hash.hash(in scratch);
            CopyInto(in general);
            UnsafeMathOP.fmod((fProxy*)scratch.Data.Ptr, (fProxy)2, scratch.N);
            HashOut[19] = Hash.hash(in scratch);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetRandomSamplersJobFProxy : IJob
    {
        public fProxyN uGrid;
        public fProxyN outVec;
        public fProxyMxN cholL; public fProxyN mvnMean; public fProxyN mvnDest, mvnScratch;
        public fProxyMxN orthoMat, spdMat, condMat, rankMat;

        public NativeArray<uint> HashOut; // 13 slots

        public void Execute()
        {
            int n = uGrid.N;

            for (int i = 0; i < n; i++) outVec[i] = fProxyUniform.UniformICDF(uGrid[i], (fProxy)(-2), (fProxy)2);
            HashOut[0] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = fProxyExponential.ExponentialICDF(uGrid[i], (fProxy)1.5);
            HashOut[1] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = fProxyRayleigh.RayleighICDF(uGrid[i], (fProxy)1);
            HashOut[2] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = fProxyWeibull.WeibullICDF(uGrid[i], (fProxy)2, (fProxy)1);
            HashOut[3] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = fProxyCauchy.CauchyICDF(uGrid[i], (fProxy)0, (fProxy)1);
            HashOut[4] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = fProxyLogistic.LogisticICDF(uGrid[i], (fProxy)0, (fProxy)1);
            HashOut[5] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = fProxyPareto.ParetoICDF(uGrid[i], (fProxy)1, (fProxy)2);
            HashOut[6] = Hash.hash(in outVec);
            for (int i = 0; i < n; i++) outVec[i] = fProxyTriangular.TriangularICDF(uGrid[i], (fProxy)0, (fProxy)0.5, (fProxy)1);
            HashOut[7] = Hash.hash(in outVec);

            var rng1 = new Random(0x9E3779B1u);
            Rand.multivariateNormalInPlace(ref rng1, in cholL, in mvnMean, ref mvnDest, ref mvnScratch);
            HashOut[8] = Hash.hash(in mvnDest);

            var rng2 = new Random(0xC0FFEEu);
            Rand.orthogonalInPlace(ref rng2, ref orthoMat);
            HashOut[9] = Hash.hash(in orthoMat);

            var rng3 = new Random(0xDEADBEEFu);
            Rand.spdInPlace(ref rng3, ref spdMat, (fProxy)1, (fProxy)5);
            HashOut[10] = Hash.hash(in spdMat);

            var rng4 = new Random(0xFEEDFACEu);
            Rand.conditionedInPlace(ref rng4, ref condMat, (fProxy)100);
            HashOut[11] = Hash.hash(in condMat);

            var rng5 = new Random(0xB16B00B5u);
            Rand.withRankInPlace(ref rng5, ref rankMat, 4);
            HashOut[12] = Hash.hash(in rankMat);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetSoftmaxJobFProxy : IJob
    {
        public fProxyMxN A, Arows, Acols;

        public NativeArray<uint> HashOut; // 3 slots

        public void Execute()
        {
            Stats.softmax(in A);
            HashOut[0] = Hash.hash(in A);

            Stats.softmaxRows(ref Arows);
            HashOut[1] = Hash.hash(in Arows);

            Stats.softmaxColumns(ref Acols);
            HashOut[2] = Hash.hash(in Acols);
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Strict)]
    public struct DetDftSignalJobFProxy : IJob
    {
        public fProxyN dftInRe, dftInIm, dftOutRe, dftOutIm;
        public fProxyN idftOutRe, idftOutIm;
        public fProxyN phaseRe, phaseIm, phaseDest;
        public fProxyN windowDest;
        public fProxyN gaussKernel;
        public fProxyMxN gaussKernel2D;
        public fProxyMxN prolate;
        public fProxyN waveDest;
        public fProxyWave.Sine waveFn;
        public fProxyN easingDest1, easingDest2, easingDest3;
        public fProxyEasing.EaseInSine easeSine;
        public fProxyEasing.EaseInExpo easeExpo;
        public fProxyEasing.EaseInElastic easeElastic;

        public NativeArray<uint> HashOut; // 9 slots

        public void Execute()
        {
            FFT.dft(in dftInRe, in dftInIm, ref dftOutRe, ref dftOutIm);
            HashOut[0] = Hash.combine(Hash.hash(in dftOutRe), Hash.hash(in dftOutIm));

            FFT.idft(in dftOutRe, in dftOutIm, ref idftOutRe, ref idftOutIm);
            HashOut[1] = Hash.combine(Hash.hash(in idftOutRe), Hash.hash(in idftOutIm));

            FFT.phase(in phaseRe, in phaseIm, ref phaseDest);
            HashOut[2] = Hash.hash(in phaseDest);

            uint h = 0u;
            Generate.window(ref windowDest, WindowType.Box); h = Hash.combine(h, Hash.hash(in windowDest));
            Generate.window(ref windowDest, WindowType.Hann); h = Hash.combine(h, Hash.hash(in windowDest));
            Generate.window(ref windowDest, WindowType.Hamming); h = Hash.combine(h, Hash.hash(in windowDest));
            Generate.window(ref windowDest, WindowType.Blackman); h = Hash.combine(h, Hash.hash(in windowDest));
            HashOut[3] = h;

            Generate.gaussianKernel(ref gaussKernel, (fProxy)1.5);
            HashOut[4] = Hash.hash(in gaussKernel);

            Generate.gaussianKernel2D(ref gaussKernel2D, (fProxy)1.5);
            HashOut[5] = Hash.hash(in gaussKernel2D);

            HashOut[6] = Hash.hash(in prolate);

            Generate.sample(ref waveFn, ref waveDest);
            HashOut[7] = Hash.hash(in waveDest);

            Generate.sample(ref easeSine, ref easingDest1);
            Generate.sample(ref easeExpo, ref easingDest2);
            Generate.sample(ref easeElastic, ref easingDest3);
            h = Hash.hash(in easingDest1);
            h = Hash.combine(h, Hash.hash(in easingDest2));
            h = Hash.combine(h, Hash.hash(in easingDest3));
            HashOut[8] = h;
        }
    }

    public static partial class DeterminismNativeSensitive
    {
        public static (string id, uint hash)[] Case_DetMathFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            const int n = 16;
            var grid = arena.fProxyVec(n); var gridPos = arena.fProxyVec(n);
            var gridUnit = arena.fProxyVec(n); var gridGeOne = arena.fProxyVec(n);
            for (int i = 0; i < n; i++)
            {
                fProxy t = (fProxy)i / (fProxy)(n - 1);           // [0,1]
                fProxy g = (fProxy)6 * t - (fProxy)3;             // [-3,3]
                grid[i] = g;
                gridPos[i] = math.abs(g) + (fProxy)0.01;
                gridUnit[i] = g / (fProxy)3.5;                    // (-1,1)
                gridGeOne[i] = math.abs(g) + (fProxy)1;
            }
            var outVec = arena.fProxyVec(n); var sinOut = arena.fProxyVec(n); var cosOut = arena.fProxyVec(n);

            var hashOut = new NativeArray<uint>(20, Allocator.Persistent);
            var job = new DetMathGridJobFProxy
            {
                grid = grid, gridPos = gridPos, gridUnit = gridUnit, gridGeOne = gridGeOne,
                outVec = outVec, sinOut = sinOut, cosOut = cosOut, HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("detmath/exp.fProxy", hashOut[0]),
                ("detmath/exp2.fProxy", hashOut[1]),
                ("detmath/exp10.fProxy", hashOut[2]),
                ("detmath/log.fProxy", hashOut[3]),
                ("detmath/log2.fProxy", hashOut[4]),
                ("detmath/log10.fProxy", hashOut[5]),
                ("detmath/pow.fProxy", hashOut[6]),
                ("detmath/powInt.fProxy", hashOut[7]),
                ("detmath/sin.fProxy", hashOut[8]),
                ("detmath/cos.fProxy", hashOut[9]),
                ("detmath/tan.fProxy", hashOut[10]),
                ("detmath/sinCos.fProxy", hashOut[11]),
                ("detmath/atan.fProxy", hashOut[12]),
                ("detmath/atan2.fProxy", hashOut[13]),
                ("detmath/asin.fProxy", hashOut[14]),
                ("detmath/acos.fProxy", hashOut[15]),
                ("detmath/sinh.fProxy", hashOut[16]),
                ("detmath/cosh.fProxy", hashOut[17]),
                ("detmath/tanh.fProxy", hashOut[18]),
                ("detmath/acosh.fProxy", hashOut[19]),
            };
            hashOut.Dispose();
            arena.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_ElementwiseTranscendentalFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(2654435761u ^ 0x001Bu);

            const int n = 256;
            var general = arena.fProxyVec(n); var positive = arena.fProxyVec(n);
            var unit = arena.fProxyVec(n); var geOne = arena.fProxyVec(n);
            for (int i = 0; i < n; i++)
            {
                fProxy g = rng.NextFProxy(-3f, 3f);
                general[i] = g;
                positive[i] = math.abs(g) + (fProxy)0.01;
                unit[i] = g / (fProxy)3.5;
                geOne[i] = math.abs(g) + (fProxy)1;
            }
            var scratch = arena.fProxyVec(n);

            var hashOut = new NativeArray<uint>(20, Allocator.Persistent);
            var job = new DetElementwiseTranscendentalJobFProxy
            {
                general = general, positive = positive, unit = unit, geOne = geOne, scratch = scratch, HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("elementwise-transcendental/exp.fProxy.n256", hashOut[0]),
                ("elementwise-transcendental/exp2.fProxy.n256", hashOut[1]),
                ("elementwise-transcendental/exp10.fProxy.n256", hashOut[2]),
                ("elementwise-transcendental/log.fProxy.n256", hashOut[3]),
                ("elementwise-transcendental/log2.fProxy.n256", hashOut[4]),
                ("elementwise-transcendental/log10.fProxy.n256", hashOut[5]),
                ("elementwise-transcendental/sin.fProxy.n256", hashOut[6]),
                ("elementwise-transcendental/cos.fProxy.n256", hashOut[7]),
                ("elementwise-transcendental/tan.fProxy.n256", hashOut[8]),
                ("elementwise-transcendental/asin.fProxy.n256", hashOut[9]),
                ("elementwise-transcendental/acos.fProxy.n256", hashOut[10]),
                ("elementwise-transcendental/atan.fProxy.n256", hashOut[11]),
                ("elementwise-transcendental/atan2.fProxy.n256", hashOut[12]),
                ("elementwise-transcendental/sinh.fProxy.n256", hashOut[13]),
                ("elementwise-transcendental/cosh.fProxy.n256", hashOut[14]),
                ("elementwise-transcendental/tanh.fProxy.n256", hashOut[15]),
                ("elementwise-transcendental/acosh.fProxy.n256", hashOut[16]),
                ("elementwise-transcendental/powInt.fProxy.n256", hashOut[17]),
                ("elementwise-transcendental/rsqrt.fProxy.n256", hashOut[18]),
                ("elementwise-transcendental/fmod.fProxy.n256", hashOut[19]),
            };
            hashOut.Dispose();
            arena.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_RandomSamplersFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            const int n = 16;
            var uGrid = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) uGrid[i] = (fProxy)(i + 1) / (fProxy)(n + 1); // (0,1), avoid exact 0/1
            var outVec = arena.fProxyVec(n);

            const int dim = 5;
            var cholL = arena.fProxyMat(dim, dim);
            for (int i = 0; i < dim; i++) cholL[i, i] = (fProxy)1; // identity Cholesky factor
            var mvnMean = arena.fProxyVec(dim);
            var mvnDest = arena.fProxyVec(dim);
            var mvnScratch = arena.fProxyVec(dim);

            const int md = 8;
            var orthoMat = arena.fProxyMat(md, md);
            var spdMat = arena.fProxyMat(md, md);
            var condMat = arena.fProxyMat(md, md);
            var rankMat = arena.fProxyMat(md, md);

            var hashOut = new NativeArray<uint>(13, Allocator.Persistent);
            var job = new DetRandomSamplersJobFProxy
            {
                uGrid = uGrid, outVec = outVec,
                cholL = cholL, mvnMean = mvnMean, mvnDest = mvnDest, mvnScratch = mvnScratch,
                orthoMat = orthoMat, spdMat = spdMat, condMat = condMat, rankMat = rankMat, HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("random-samplers/uniformICDF.fProxy", hashOut[0]),
                ("random-samplers/exponentialICDF.fProxy", hashOut[1]),
                ("random-samplers/rayleighICDF.fProxy", hashOut[2]),
                ("random-samplers/weibullICDF.fProxy", hashOut[3]),
                ("random-samplers/cauchyICDF.fProxy", hashOut[4]),
                ("random-samplers/logisticICDF.fProxy", hashOut[5]),
                ("random-samplers/paretoICDF.fProxy", hashOut[6]),
                ("random-samplers/triangularICDF.fProxy", hashOut[7]),
                ("random-samplers/multivariateNormalInPlace.fProxy.n5", hashOut[8]),
                ("random-samplers/orthogonalInPlace.fProxy.n8", hashOut[9]),
                ("random-samplers/spdInPlace.fProxy.n8", hashOut[10]),
                ("random-samplers/conditionedInPlace.fProxy.n8", hashOut[11]),
                ("random-samplers/withRankInPlace.fProxy.n8", hashOut[12]),
            };
            hashOut.Dispose();
            arena.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_SoftmaxFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(2654435761u ^ 0x001Du);

            const int m = 53, n = 37;
            var A = arena.fProxyMat(m, n);
            var Arows = arena.fProxyMat(m, n);
            var Acols = arena.fProxyMat(m, n);
            for (int r = 0; r < m; r++) for (int c = 0; c < n; c++)
            {
                fProxy v = rng.NextFProxy(-3f, 3f);
                A[r, c] = v; Arows[r, c] = v; Acols[r, c] = v;
            }

            var hashOut = new NativeArray<uint>(3, Allocator.Persistent);
            var job = new DetSoftmaxJobFProxy { A = A, Arows = Arows, Acols = Acols, HashOut = hashOut };
            job.Run();

            var result = new[]
            {
                ("softmax/softmax.fProxy.53x37", hashOut[0]),
                ("softmax/softmaxRows.fProxy.53x37", hashOut[1]),
                ("softmax/softmaxColumns.fProxy.53x37", hashOut[2]),
            };
            hashOut.Dispose();
            arena.Dispose();
            return result;
        }

        public static (string id, uint hash)[] Case_DftSignalFProxy()
        {
            var arena = new Arena(Allocator.Persistent);
            var rng = new Random(2654435761u ^ 0x001Eu);

            const int n = 32;
            var dftInRe = arena.fProxyVec(n); var dftInIm = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) { dftInRe[i] = rng.NextFProxy(-1f, 1f); dftInIm[i] = (fProxy)0; }
            var dftOutRe = arena.fProxyVec(n); var dftOutIm = arena.fProxyVec(n);
            var idftOutRe = arena.fProxyVec(n); var idftOutIm = arena.fProxyVec(n);

            const int nPhase = 16;
            var phaseRe = arena.fProxyVec(nPhase); var phaseIm = arena.fProxyVec(nPhase);
            for (int i = 0; i < nPhase; i++) { phaseRe[i] = rng.NextFProxy(-1f, 1f); phaseIm[i] = rng.NextFProxy(-1f, 1f); }
            var phaseDest = arena.fProxyVec(nPhase);

            var windowDest = arena.fProxyVec(n);
            var gaussKernel = arena.fProxyVec(15);
            var gaussKernel2D = arena.fProxyMat(7, 7);
            var prolate = arena.fProxyProlate(16, (fProxy)0.25);

            var waveDest = arena.fProxyVec(n);
            var waveFn = new fProxyWave.Sine { Cycles = (fProxy)2, Phase = (fProxy)0 };

            var easingDest1 = arena.fProxyVec(n);
            var easingDest2 = arena.fProxyVec(n);
            var easingDest3 = arena.fProxyVec(n);
            var easeSine = new fProxyEasing.EaseInSine();
            var easeExpo = new fProxyEasing.EaseInExpo();
            var easeElastic = new fProxyEasing.EaseInElastic();

            var hashOut = new NativeArray<uint>(9, Allocator.Persistent);
            var job = new DetDftSignalJobFProxy
            {
                dftInRe = dftInRe, dftInIm = dftInIm, dftOutRe = dftOutRe, dftOutIm = dftOutIm,
                idftOutRe = idftOutRe, idftOutIm = idftOutIm,
                phaseRe = phaseRe, phaseIm = phaseIm, phaseDest = phaseDest,
                windowDest = windowDest, gaussKernel = gaussKernel, gaussKernel2D = gaussKernel2D, prolate = prolate,
                waveDest = waveDest, waveFn = waveFn,
                easingDest1 = easingDest1, easingDest2 = easingDest2, easingDest3 = easingDest3,
                easeSine = easeSine, easeExpo = easeExpo, easeElastic = easeElastic,
                HashOut = hashOut,
            };
            job.Run();

            var result = new[]
            {
                ("dft-signal/dft.fProxy.n32", hashOut[0]),
                ("dft-signal/idft.fProxy.n32", hashOut[1]),
                ("dft-signal/phase.fProxy.n16", hashOut[2]),
                ("dft-signal/window.fProxy.n32.all4", hashOut[3]),
                ("dft-signal/gaussianKernel.fProxy.n15", hashOut[4]),
                ("dft-signal/gaussianKernel2D.fProxy.7x7", hashOut[5]),
                ("dft-signal/gallery.prolate.fProxy.n16", hashOut[6]),
                ("dft-signal/wave.sine.fProxy.n32", hashOut[7]),
                ("dft-signal/easing.sine-expo-elastic.fProxy.n32", hashOut[8]),
            };
            hashOut.Dispose();
            arena.Dispose();
            return result;
        }
    }
}
