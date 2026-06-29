using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;

public class doubleFFTTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            FftConstantDC,
            FftSingleSinusoid,
            FftMatchesDft,
            FftRoundTrip,
            DftRoundTripOddN,
            RfftEqualsFft,
            SpectrumReductions,
            DftConstantOddN,
            IfftKnownValue,
            IdftKnownValue,
            FftSmallSizes,
            RfftRoundTrip,
            RfftKnownSignals,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected, [3] diff
        public NativeArray<double> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.FftConstantDC: FftConstantDC(); break;
                case TestType.FftSingleSinusoid: FftSingleSinusoid(); break;
                case TestType.FftMatchesDft: FftMatchesDft(); break;
                case TestType.FftRoundTrip: FftRoundTrip(); break;
                case TestType.DftRoundTripOddN: DftRoundTripOddN(); break;
                case TestType.RfftEqualsFft: RfftEqualsFft(); break;
                case TestType.SpectrumReductions: SpectrumReductions(); break;
                case TestType.DftConstantOddN: DftConstantOddN(); break;
                case TestType.IfftKnownValue: IfftKnownValue(); break;
                case TestType.IdftKnownValue: IdftKnownValue(); break;
                case TestType.FftSmallSizes: FftSmallSizes(); break;
                case TestType.RfftRoundTrip: RfftRoundTrip(); break;
                case TestType.RfftKnownSignals: RfftKnownSignals(); break;
            }
        }

        // FFT of a constant signal x=[1,1,1,1] -> X[0]=N (DC), all other bins 0.
        void FftConstantDC()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 4;
            var re = arena.doubleVec(N, 1f);     // all ones
            var im = arena.doubleVec(N);         // zeros

            doubleFFT.fft(ref re, ref im);

            AssertClose(re[0], (double)N, 1E-4f);
            AssertClose(im[0], (double)0f, 1E-4f);
            for (int k = 1; k < N; k++)
            {
                AssertClose(re[k], (double)0f, 1E-4f);
                AssertClose(im[k], (double)0f, 1E-4f);
            }
            arena.Dispose();
        }

        // x[n]=cos(2πn/N) -> magnitude peaks N/2 at bins 1 and N-1, ~0 elsewhere.
        void FftSingleSinusoid()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 8;
            var re = arena.doubleVec(N);
            var im = arena.doubleVec(N);
            double w = (double)(2.0 * System.Math.PI) / (double)N;
            for (int n = 0; n < N; n++)
                re[n] = math.cos(w * n);

            doubleFFT.fft(ref re, ref im);

            var mag = arena.doubleVec(N);
            doubleFFT.magnitude(in re, in im, ref mag);

            double half = (double)N * (double)0.5;
            AssertClose(mag[1], half, 1E-3f);
            AssertClose(mag[N - 1], half, 1E-3f);
            // every other bin must be ~0 (no spectral leakage)
            for (int k = 0; k < N; k++)
                if (k != 1 && k != N - 1)
                    AssertClose(mag[k], (double)0f, 1E-3f);
            arena.Dispose();
        }

        // Radix-2 fft must agree bin-for-bin with the direct dft on the same length-8 signal.
        void FftMatchesDft()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 8;
            var sigRe = arena.doubleRandomVector(N, -2f, 2f, 9911);
            var sigIm = arena.doubleRandomVector(N, -2f, 2f, 2244);

            // fft path (in-place on copies)
            var fRe = sigRe.Copy();
            var fIm = sigIm.Copy();
            doubleFFT.fft(ref fRe, ref fIm);

            // dft path
            var dRe = arena.doubleVec(N);
            var dIm = arena.doubleVec(N);
            doubleFFT.dft(in sigRe, in sigIm, ref dRe, ref dIm);

            for (int k = 0; k < N; k++)
            {
                AssertClose(fRe[k], dRe[k], 1E-3f);
                AssertClose(fIm[k], dIm[k], 1E-3f);
            }
            arena.Dispose();
        }

        // ifft(fft(x)) == x for a complex signal (power-of-two length).
        void FftRoundTrip()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 16;
            var re0 = arena.doubleRandomVector(N, -3f, 3f, 5150);
            var im0 = arena.doubleRandomVector(N, -3f, 3f, 6160);

            var re = re0.Copy();
            var im = im0.Copy();
            doubleFFT.fft(ref re, ref im);
            doubleFFT.ifft(ref re, ref im);

            for (int i = 0; i < N; i++)
            {
                AssertClose(re[i], re0[i], 1E-3f);
                AssertClose(im[i], im0[i], 1E-3f);
            }
            arena.Dispose();
        }

        // idft(dft(x)) == x for an arbitrary (non power-of-two) length.
        void DftRoundTripOddN()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 5;
            var re0 = arena.doubleRandomVector(N, -3f, 3f, 7007);
            var im0 = arena.doubleRandomVector(N, -3f, 3f, 8008);

            var fRe = arena.doubleVec(N);
            var fIm = arena.doubleVec(N);
            doubleFFT.dft(in re0, in im0, ref fRe, ref fIm);

            var bRe = arena.doubleVec(N);
            var bIm = arena.doubleVec(N);
            doubleFFT.idft(in fRe, in fIm, ref bRe, ref bIm);

            for (int i = 0; i < N; i++)
            {
                AssertClose(bRe[i], re0[i], 1E-3f);
                AssertClose(bIm[i], im0[i], 1E-3f);
            }
            arena.Dispose();
        }

        // rfft now returns N/2+1 unique bins. Verify each bin matches the corresponding bin of the
        // full N-point FFT (computed by zero-padding im and calling fft). Twiddle factors introduce
        // small rounding, so allow a tight but non-zero tolerance.
        void RfftEqualsFft()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 8;
            int halfSpec = (N >> 1) + 1; // 5
            var real = arena.doubleRandomVector(N, -2f, 2f, 1234);

            // half-spectrum output
            var rRe = arena.doubleVec(halfSpec);
            var rIm = arena.doubleVec(halfSpec);
            doubleFFT.rfft(in real, ref rRe, ref rIm);

            // full N-point FFT oracle
            var fRe = real.Copy();
            var fIm = arena.doubleVec(N); // zeros (real input)
            doubleFFT.fft(ref fRe, ref fIm);

            // Compare only the N/2+1 non-redundant bins (0..N/2).
            for (int k = 0; k <= N / 2; k++)
            {
                AssertClose(rRe[k], fRe[k], (double)1E-4f);
                AssertClose(rIm[k], fIm[k], (double)1E-4f);
            }
            // DC and Nyquist imaginaries are always exactly zero for a real signal.
            AssertClose(rIm[0],     (double)0, 0f);
            AssertClose(rIm[N / 2], (double)0, 0f);
            arena.Dispose();
        }

        // magnitude / powerSpectrum / phase known values; power == magnitude².
        void SpectrumReductions()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 3;
            var re = arena.doubleVec(N);
            var im = arena.doubleVec(N);
            re[0] = 3f;  im[0] = 4f;   // mag 5
            re[1] = -1f; im[1] = 0f;   // mag 1, phase π
            re[2] = 0f;  im[2] = 2f;   // mag 2, phase π/2

            var mag = arena.doubleVec(N);
            var pow = arena.doubleVec(N);
            var ph = arena.doubleVec(N);
            doubleFFT.magnitude(in re, in im, ref mag);
            doubleFFT.powerSpectrum(in re, in im, ref pow);
            doubleFFT.phase(in re, in im, ref ph);

            AssertClose(mag[0], (double)5f, 1E-5f);
            AssertClose(mag[1], (double)1f, 1E-5f);
            AssertClose(mag[2], (double)2f, 1E-5f);

            AssertClose(pow[0], (double)25f, 1E-5f);
            AssertClose(pow[1], (double)1f, 1E-5f);
            AssertClose(pow[2], (double)4f, 1E-5f);

            // power == magnitude²
            for (int i = 0; i < N; i++)
                AssertClose(pow[i], mag[i] * mag[i], 1E-4f);

            AssertClose(ph[0], math.atan2((double)4f, (double)3f), 1E-5f);
            AssertClose(ph[1], (double)System.Math.PI, 1E-5f);
            AssertClose(ph[2], (double)(System.Math.PI * 0.5), 1E-5f);

            // arena wrappers must equal the ref-dest reductions
            var magW = arena.doubleMagnitude(in re, in im);
            var powW = arena.doublePowerSpectrum(in re, in im);
            var phW = arena.doublePhase(in re, in im);
            for (int i = 0; i < N; i++)
            {
                AssertClose(magW[i], mag[i], 0f);
                AssertClose(powW[i], pow[i], 0f);
                AssertClose(phW[i], ph[i], 0f);
            }

            arena.Dispose();
        }

        // dft DC bin of a constant length-3 (non pow2) signal: X[0] = sum = 3.
        void DftConstantOddN()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 3;
            var re = arena.doubleVec(N, 1f);
            var im = arena.doubleVec(N);
            var oRe = arena.doubleVec(N);
            var oIm = arena.doubleVec(N);

            doubleFFT.dft(in re, in im, ref oRe, ref oIm);

            AssertClose(oRe[0], (double)3f, 1E-4f);
            AssertClose(oIm[0], (double)0f, 1E-4f);
            AssertClose(oRe[1], (double)0f, 1E-4f); // sum of cube roots of unity = 0
            AssertClose(oRe[2], (double)0f, 1E-4f);
            AssertClose(oIm[1], (double)0f, 1E-4f); // imag must also vanish (catches a sign-conv bug)
            AssertClose(oIm[2], (double)0f, 1E-4f);
            arena.Dispose();
        }

        // Independent ifft oracle (NOT a forward+inverse round-trip, so it can't hide complementary
        // sign/scale bugs): ifft of a DC spectrum X=[N,0,…,0] is the constant signal x[n]=1. N=8 so the
        // bit-reversal permutation is non-trivial.
        void IfftKnownValue()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 8;
            var re = arena.doubleVec(N);
            var im = arena.doubleVec(N);
            re[0] = (double)N; // X = [8, 0, 0, ...]

            doubleFFT.ifft(ref re, ref im);

            for (int n = 0; n < N; n++)
            {
                AssertClose(re[n], (double)1f, 1E-4f);
                AssertClose(im[n], (double)0f, 1E-4f);
            }
            arena.Dispose();
        }

        // Independent idft oracle: idft of a DC spectrum X=[N,0,0] is the constant x=[1,1,1] (any N).
        void IdftKnownValue()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 3;
            var re = arena.doubleVec(N);
            var im = arena.doubleVec(N);
            re[0] = (double)N; // X = [3, 0, 0]
            var oRe = arena.doubleVec(N);
            var oIm = arena.doubleVec(N);

            doubleFFT.idft(in re, in im, ref oRe, ref oIm);

            for (int n = 0; n < N; n++)
            {
                AssertClose(oRe[n], (double)1f, 1E-4f);
                AssertClose(oIm[n], (double)0f, 1E-4f);
            }
            arena.Dispose();
        }

        // Edge sizes: N=1 fft is the identity (early-return path); N=2 is the smallest real butterfly,
        // fft([1,0]) = [1,1] (re), [0,0] (im).
        void FftSmallSizes()
        {
            var arena = new Arena(Allocator.Persistent);

            var re1 = arena.doubleVec(1);
            var im1 = arena.doubleVec(1);
            re1[0] = (double)7; im1[0] = (double)(-2);
            doubleFFT.fft(ref re1, ref im1);
            AssertClose(re1[0], (double)7f, 1E-5f);   // unchanged
            AssertClose(im1[0], (double)(-2f), 1E-5f);

            var re2 = arena.doubleVec(2);
            var im2 = arena.doubleVec(2);
            re2[0] = (double)1; re2[1] = (double)0;
            doubleFFT.fft(ref re2, ref im2);
            AssertClose(re2[0], (double)1f, 1E-5f);   // X0 = x0+x1
            AssertClose(re2[1], (double)1f, 1E-5f);   // X1 = x0-x1
            AssertClose(im2[0], (double)0f, 1E-5f);
            AssertClose(im2[1], (double)0f, 1E-5f);

            arena.Dispose();
        }

        // irfft(rfft(x)) == x to floating-point precision, at several power-of-two lengths.
        void RfftRoundTrip()
        {
            var arena = new Arena(Allocator.Persistent);

            // N=8
            {
                int N = 8;
                int halfSpec = (N >> 1) + 1;
                var real0 = arena.doubleRandomVector(N, -3f, 3f, 5555);
                var rRe = arena.doubleVec(halfSpec);
                var rIm = arena.doubleVec(halfSpec);
                var real2 = arena.doubleVec(N);
                doubleFFT.rfft(in real0, ref rRe, ref rIm);
                doubleFFT.irfft(in rRe, in rIm, ref real2);
                for (int i = 0; i < N; i++)
                    AssertClose(real2[i], real0[i], (double)1E-4f);
            }

            // N=16
            {
                int N = 16;
                int halfSpec = (N >> 1) + 1;
                var real0 = arena.doubleRandomVector(N, -3f, 3f, 6666);
                var rRe = arena.doubleVec(halfSpec);
                var rIm = arena.doubleVec(halfSpec);
                var real2 = arena.doubleVec(N);
                doubleFFT.rfft(in real0, ref rRe, ref rIm);
                doubleFFT.irfft(in rRe, in rIm, ref real2);
                for (int i = 0; i < N; i++)
                    AssertClose(real2[i], real0[i], (double)1E-4f);
            }

            // N=64
            {
                int N = 64;
                int halfSpec = (N >> 1) + 1;
                var real0 = arena.doubleRandomVector(N, -3f, 3f, 7777);
                var rRe = arena.doubleVec(halfSpec);
                var rIm = arena.doubleVec(halfSpec);
                var real2 = arena.doubleVec(N);
                doubleFFT.rfft(in real0, ref rRe, ref rIm);
                doubleFFT.irfft(in rRe, in rIm, ref real2);
                for (int i = 0; i < N; i++)
                    AssertClose(real2[i], real0[i], (double)1E-4f);
            }

            // irfft arena wrapper round-trip (N=8)
            {
                int N = 8;
                int halfSpec = (N >> 1) + 1;
                var real0 = arena.doubleRandomVector(N, -3f, 3f, 8888);
                var rRe = arena.doubleVec(halfSpec);
                var rIm = arena.doubleVec(halfSpec);
                doubleFFT.rfft(in real0, ref rRe, ref rIm);
                var real2 = arena.doubleIrfft(in rRe, in rIm);
                for (int i = 0; i < N; i++)
                    AssertClose(real2[i], real0[i], (double)1E-4f);
            }

            arena.Dispose();
        }

        // Known-signal oracle tests for rfft (human-readable, catch convention/scale bugs).
        void RfftKnownSignals()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 8;
            int halfSpec = (N >> 1) + 1; // 5

            // --- DC: x[n]=1 for all n ---
            // X[0]=N, all other bins 0; im all 0.
            {
                var dc = arena.doubleVec(N, 1f);
                var dcRe = arena.doubleVec(halfSpec);
                var dcIm = arena.doubleVec(halfSpec);
                doubleFFT.rfft(in dc, ref dcRe, ref dcIm);
                AssertClose(dcRe[0], (double)N, (double)1E-4f);
                AssertClose(dcIm[0], (double)0, 0f);
                for (int k = 1; k <= N / 2; k++)
                {
                    AssertClose(dcRe[k], (double)0, (double)1E-4f);
                    AssertClose(dcIm[k], (double)0, (double)1E-4f);
                }
            }

            // --- Pure cosine at integer frequency f=2: x[n] = cos(2π·2·n/N) ---
            // Full-spectrum DFT has N/2 at bin f and N/2 at bin N-f; the half-spectrum sees bin f.
            // re[f]=N/2, im[f]≈0; all other half-spectrum bins ≈0.
            {
                int f = 2;
                var cosX = arena.doubleVec(N);
                double wf = (double)(2.0 * System.Math.PI * f) / (double)N;
                for (int n = 0; n < N; n++)
                    cosX[n] = math.cos(wf * (double)n);

                var cosRe = arena.doubleVec(halfSpec);
                var cosIm = arena.doubleVec(halfSpec);
                doubleFFT.rfft(in cosX, ref cosRe, ref cosIm);

                AssertClose(cosRe[f], (double)(N / 2), (double)1E-4f);
                AssertClose(cosIm[f], (double)0, (double)1E-4f);
                for (int k = 0; k <= N / 2; k++)
                {
                    if (k == f) continue;
                    AssertClose(cosRe[k], (double)0, (double)1E-4f);
                    AssertClose(cosIm[k], (double)0, (double)1E-4f);
                }
            }

            // --- Nyquist: x[n] = (-1)^n ---
            // X[N/2]=N, all other bins 0; im all 0.
            {
                var nyq = arena.doubleVec(N);
                for (int n = 0; n < N; n++)
                    nyq[n] = (n % 2 == 0) ? (double)1 : (double)(-1);

                var nyqRe = arena.doubleVec(halfSpec);
                var nyqIm = arena.doubleVec(halfSpec);
                doubleFFT.rfft(in nyq, ref nyqRe, ref nyqIm);

                AssertClose(nyqRe[N / 2], (double)N, (double)1E-4f);
                AssertClose(nyqIm[N / 2], (double)0, 0f);
                for (int k = 0; k < N / 2; k++)
                {
                    AssertClose(nyqRe[k], (double)0, (double)1E-4f);
                    AssertClose(nyqIm[k], (double)0, (double)1E-4f);
                }
            }

            // --- N=2 edge case: x=[a,b] -> re=[a+b, a-b], im=[0,0] ---
            {
                var x2 = arena.doubleVec(2);
                x2[0] = (double)3; x2[1] = (double)7;
                var r2 = arena.doubleVec(2);
                var i2 = arena.doubleVec(2);
                doubleFFT.rfft(in x2, ref r2, ref i2);
                AssertClose(r2[0], (double)10, (double)1E-5f);
                AssertClose(r2[1], (double)(-4), (double)1E-5f);
                AssertClose(i2[0], (double)0, 0f);
                AssertClose(i2[1], (double)0, 0f);
            }

            arena.Dispose();
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected, [3]=diff
        void AssertClose(double a, double b, double precision)
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
    }

    void RunJob(TestJob.TestType type)
    {
        var fail = new NativeArray<double>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (double)0)
                Assert.Fail($"got {fail[1]}, expected {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected {fail[2]}, diff {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    [Test] public void FftConstantDCTest() => RunJob(TestJob.TestType.FftConstantDC);
    [Test] public void FftSingleSinusoidTest() => RunJob(TestJob.TestType.FftSingleSinusoid);
    [Test] public void FftMatchesDftTest() => RunJob(TestJob.TestType.FftMatchesDft);
    [Test] public void FftRoundTripTest() => RunJob(TestJob.TestType.FftRoundTrip);
    [Test] public void DftRoundTripOddNTest() => RunJob(TestJob.TestType.DftRoundTripOddN);
    [Test] public void RfftEqualsFftTest() => RunJob(TestJob.TestType.RfftEqualsFft);
    [Test] public void SpectrumReductionsTest() => RunJob(TestJob.TestType.SpectrumReductions);
    [Test] public void DftConstantOddNTest() => RunJob(TestJob.TestType.DftConstantOddN);
    [Test] public void IfftKnownValueTest() => RunJob(TestJob.TestType.IfftKnownValue);
    [Test] public void IdftKnownValueTest() => RunJob(TestJob.TestType.IdftKnownValue);
    [Test] public void FftSmallSizesTest() => RunJob(TestJob.TestType.FftSmallSizes);
    [Test] public void RfftRoundTripTest() => RunJob(TestJob.TestType.RfftRoundTrip);
    [Test] public void RfftKnownSignalsTest() => RunJob(TestJob.TestType.RfftKnownSignals);

    // ---- Managed throw tests (guard paths) ----

    [Test]
    public void FftNonPowerOfTwoThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var re = arena.doubleVec(6); // 6 is not a power of two
        var im = arena.doubleVec(6);
        Assert.Throws<ArgumentException>(() => doubleFFT.fft(ref re, ref im));
        Assert.Throws<ArgumentException>(() => doubleFFT.ifft(ref re, ref im));
        arena.Dispose();
    }

    [Test]
    public void FftMismatchedLengthThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var re = arena.doubleVec(8);
        var im = arena.doubleVec(4);
        Assert.Throws<ArgumentException>(() => doubleFFT.fft(ref re, ref im));
        arena.Dispose();
    }

    [Test]
    public void DftAliasOutputThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var re = arena.doubleVec(4);
        var im = arena.doubleVec(4);
        var oRe = arena.doubleVec(4);
        var oIm = arena.doubleVec(4);
        // every one of the four out-vs-in pointer collisions must throw (each output bin reads all inputs)
        Assert.Throws<ArgumentException>(() => doubleFFT.dft(in re, in im, ref re, ref oIm));   // outRe==inRe
        Assert.Throws<ArgumentException>(() => doubleFFT.dft(in re, in im, ref im, ref oIm));   // outRe==inIm
        Assert.Throws<ArgumentException>(() => doubleFFT.dft(in re, in im, ref oRe, ref re));   // outIm==inRe
        Assert.Throws<ArgumentException>(() => doubleFFT.dft(in re, in im, ref oRe, ref im));   // outIm==inIm
        // idft shares the guard via DftCore
        Assert.Throws<ArgumentException>(() => doubleFFT.idft(in re, in im, ref re, ref oIm));
        arena.Dispose();
    }

    [Test]
    public void DftMismatchedLengthThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var inRe = arena.doubleVec(4);
        var inIm = arena.doubleVec(4);
        var inImShort = arena.doubleVec(3);
        var outRe = arena.doubleVec(4);
        var outIm = arena.doubleVec(4);
        var outShort = arena.doubleVec(3);
        // inRe.N != inIm.N
        Assert.Throws<ArgumentException>(() => doubleFFT.dft(in inRe, in inImShort, ref outRe, ref outIm));
        // output length != input length
        Assert.Throws<ArgumentException>(() => doubleFFT.dft(in inRe, in inIm, ref outRe, ref outShort));
        arena.Dispose();
    }

    [Test]
    public void RfftLengthAndAliasThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var real = arena.doubleVec(8);
        // For N=8, the correct half-spectrum length is N/2+1 = 5.
        var re5  = arena.doubleVec(5);  // correct
        var im5  = arena.doubleVec(5);  // correct
        var re8  = arena.doubleVec(8);  // wrong (full-N, not N/2+1)
        var im4  = arena.doubleVec(4);  // wrong

        // wrong re length
        Assert.Throws<ArgumentException>(() => doubleFFT.rfft(in real, ref re8, ref im5));
        // wrong im length
        Assert.Throws<ArgumentException>(() => doubleFFT.rfft(in real, ref re5, ref im4));
        // non-power-of-two real length
        var real7 = arena.doubleVec(7);
        var re4   = arena.doubleVec(4);
        var im4b  = arena.doubleVec(4);
        Assert.Throws<ArgumentException>(() => doubleFFT.rfft(in real7, ref re4, ref im4b));
        // im aliasing real must throw
        Assert.Throws<ArgumentException>(() => doubleFFT.rfft(in real, ref re5, ref real));
        arena.Dispose();
    }

    [Test]
    public void IrfftGuards()
    {
        var arena = new Arena(Allocator.Persistent);
        // For N=8, half-spectrum has length N/2+1=5.
        var re5   = arena.doubleVec(5);
        var im5   = arena.doubleVec(5);
        var real8 = arena.doubleVec(8);

        // im.N != re.N
        var im4 = arena.doubleVec(4);
        Assert.Throws<ArgumentException>(() => doubleFFT.irfft(in re5, in im4, ref real8));

        // halfSpec < 2 (re.N=1 means N=0; minimum is N=2)
        var re1  = arena.doubleVec(1);
        var im1  = arena.doubleVec(1);
        Assert.Throws<ArgumentException>(() => doubleFFT.irfft(in re1, in im1, ref real8));

        // wrong real output length (real.N=7 but N=8)
        var real7 = arena.doubleVec(7);
        Assert.Throws<ArgumentException>(() => doubleFFT.irfft(in re5, in im5, ref real7));

        // Alias tests: use N=2 (halfSpec=2, real.N=2) so all length guards pass and the alias
        // check is reached. re2.N=2 = N, so real.N matches and the ptr check fires.
        var re2  = arena.doubleVec(2);
        var im2  = arena.doubleVec(2);
        var real2 = arena.doubleVec(2);

        // real aliasing re (correct lengths: halfSpec=2, N=2)
        Assert.Throws<ArgumentException>(() => doubleFFT.irfft(in re2, in im2, ref re2));
        // real aliasing im (correct lengths)
        Assert.Throws<ArgumentException>(() => doubleFFT.irfft(in re2, in im2, ref im2));

        arena.Dispose();
    }

    [Test]
    public void ReductionMismatchedLengthThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var re = arena.doubleVec(4);
        var im = arena.doubleVec(4);
        var shortDest = arena.doubleVec(3);
        Assert.Throws<ArgumentException>(() => doubleFFT.magnitude(in re, in im, ref shortDest));
        Assert.Throws<ArgumentException>(() => doubleFFT.powerSpectrum(in re, in im, ref shortDest));
        Assert.Throws<ArgumentException>(() => doubleFFT.phase(in re, in im, ref shortDest));
        arena.Dispose();
    }
}
