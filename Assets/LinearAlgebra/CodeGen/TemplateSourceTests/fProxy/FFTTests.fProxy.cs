using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;

public class fProxyFFTTests
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
            FftSmallSizes
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected, [3] diff
        public NativeArray<fProxy> Fail;

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
            }
        }

        // FFT of a constant signal x=[1,1,1,1] -> X[0]=N (DC), all other bins 0.
        void FftConstantDC()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 4;
            var re = arena.fProxyVec(N, 1f);     // all ones
            var im = arena.fProxyVec(N);         // zeros

            fProxyFFT.fft(ref re, ref im);

            AssertClose(re[0], (fProxy)N, 1E-4f);
            AssertClose(im[0], (fProxy)0f, 1E-4f);
            for (int k = 1; k < N; k++)
            {
                AssertClose(re[k], (fProxy)0f, 1E-4f);
                AssertClose(im[k], (fProxy)0f, 1E-4f);
            }
            arena.Dispose();
        }

        // x[n]=cos(2πn/N) -> magnitude peaks N/2 at bins 1 and N-1, ~0 elsewhere.
        void FftSingleSinusoid()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 8;
            var re = arena.fProxyVec(N);
            var im = arena.fProxyVec(N);
            fProxy w = (fProxy)(2.0 * System.Math.PI) / (fProxy)N;
            for (int n = 0; n < N; n++)
                re[n] = math.cos(w * n);

            fProxyFFT.fft(ref re, ref im);

            var mag = arena.fProxyVec(N);
            fProxyFFT.magnitude(in re, in im, ref mag);

            fProxy half = (fProxy)N * (fProxy)0.5;
            AssertClose(mag[1], half, 1E-3f);
            AssertClose(mag[N - 1], half, 1E-3f);
            // every other bin must be ~0 (no spectral leakage)
            for (int k = 0; k < N; k++)
                if (k != 1 && k != N - 1)
                    AssertClose(mag[k], (fProxy)0f, 1E-3f);
            arena.Dispose();
        }

        // Radix-2 fft must agree bin-for-bin with the direct dft on the same length-8 signal.
        void FftMatchesDft()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 8;
            var sigRe = arena.fProxyRandomVector(N, -2f, 2f, 9911);
            var sigIm = arena.fProxyRandomVector(N, -2f, 2f, 2244);

            // fft path (in-place on copies)
            var fRe = sigRe.Copy();
            var fIm = sigIm.Copy();
            fProxyFFT.fft(ref fRe, ref fIm);

            // dft path
            var dRe = arena.fProxyVec(N);
            var dIm = arena.fProxyVec(N);
            fProxyFFT.dft(in sigRe, in sigIm, ref dRe, ref dIm);

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
            var re0 = arena.fProxyRandomVector(N, -3f, 3f, 5150);
            var im0 = arena.fProxyRandomVector(N, -3f, 3f, 6160);

            var re = re0.Copy();
            var im = im0.Copy();
            fProxyFFT.fft(ref re, ref im);
            fProxyFFT.ifft(ref re, ref im);

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
            var re0 = arena.fProxyRandomVector(N, -3f, 3f, 7007);
            var im0 = arena.fProxyRandomVector(N, -3f, 3f, 8008);

            var fRe = arena.fProxyVec(N);
            var fIm = arena.fProxyVec(N);
            fProxyFFT.dft(in re0, in im0, ref fRe, ref fIm);

            var bRe = arena.fProxyVec(N);
            var bIm = arena.fProxyVec(N);
            fProxyFFT.idft(in fRe, in fIm, ref bRe, ref bIm);

            for (int i = 0; i < N; i++)
            {
                AssertClose(bRe[i], re0[i], 1E-3f);
                AssertClose(bIm[i], im0[i], 1E-3f);
            }
            arena.Dispose();
        }

        // rfft(real) == fft(real, 0).
        void RfftEqualsFft()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 8;
            var real = arena.fProxyRandomVector(N, -2f, 2f, 1234);

            var rRe = arena.fProxyVec(N);
            var rIm = arena.fProxyVec(N);
            fProxyFFT.rfft(in real, ref rRe, ref rIm);

            var fRe = real.Copy();
            var fIm = arena.fProxyVec(N); // zeros
            fProxyFFT.fft(ref fRe, ref fIm);

            // rfft and the manual (real, 0) -> fft path do bit-identical work: expect EXACT equality.
            for (int k = 0; k < N; k++)
            {
                AssertClose(rRe[k], fRe[k], 0f);
                AssertClose(rIm[k], fIm[k], 0f);
            }
            arena.Dispose();
        }

        // magnitude / powerSpectrum / phase known values; power == magnitude².
        void SpectrumReductions()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 3;
            var re = arena.fProxyVec(N);
            var im = arena.fProxyVec(N);
            re[0] = 3f;  im[0] = 4f;   // mag 5
            re[1] = -1f; im[1] = 0f;   // mag 1, phase π
            re[2] = 0f;  im[2] = 2f;   // mag 2, phase π/2

            var mag = arena.fProxyVec(N);
            var pow = arena.fProxyVec(N);
            var ph = arena.fProxyVec(N);
            fProxyFFT.magnitude(in re, in im, ref mag);
            fProxyFFT.powerSpectrum(in re, in im, ref pow);
            fProxyFFT.phase(in re, in im, ref ph);

            AssertClose(mag[0], (fProxy)5f, 1E-5f);
            AssertClose(mag[1], (fProxy)1f, 1E-5f);
            AssertClose(mag[2], (fProxy)2f, 1E-5f);

            AssertClose(pow[0], (fProxy)25f, 1E-5f);
            AssertClose(pow[1], (fProxy)1f, 1E-5f);
            AssertClose(pow[2], (fProxy)4f, 1E-5f);

            // power == magnitude²
            for (int i = 0; i < N; i++)
                AssertClose(pow[i], mag[i] * mag[i], 1E-4f);

            AssertClose(ph[0], math.atan2((fProxy)4f, (fProxy)3f), 1E-5f);
            AssertClose(ph[1], (fProxy)System.Math.PI, 1E-5f);
            AssertClose(ph[2], (fProxy)(System.Math.PI * 0.5), 1E-5f);

            // arena wrappers must equal the ref-dest reductions
            var magW = arena.fProxyMagnitude(in re, in im);
            var powW = arena.fProxyPowerSpectrum(in re, in im);
            var phW = arena.fProxyPhase(in re, in im);
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
            var re = arena.fProxyVec(N, 1f);
            var im = arena.fProxyVec(N);
            var oRe = arena.fProxyVec(N);
            var oIm = arena.fProxyVec(N);

            fProxyFFT.dft(in re, in im, ref oRe, ref oIm);

            AssertClose(oRe[0], (fProxy)3f, 1E-4f);
            AssertClose(oIm[0], (fProxy)0f, 1E-4f);
            AssertClose(oRe[1], (fProxy)0f, 1E-4f); // sum of cube roots of unity = 0
            AssertClose(oRe[2], (fProxy)0f, 1E-4f);
            AssertClose(oIm[1], (fProxy)0f, 1E-4f); // imag must also vanish (catches a sign-conv bug)
            AssertClose(oIm[2], (fProxy)0f, 1E-4f);
            arena.Dispose();
        }

        // Independent ifft oracle (NOT a forward+inverse round-trip, so it can't hide complementary
        // sign/scale bugs): ifft of a DC spectrum X=[N,0,…,0] is the constant signal x[n]=1. N=8 so the
        // bit-reversal permutation is non-trivial.
        void IfftKnownValue()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 8;
            var re = arena.fProxyVec(N);
            var im = arena.fProxyVec(N);
            re[0] = (fProxy)N; // X = [8, 0, 0, ...]

            fProxyFFT.ifft(ref re, ref im);

            for (int n = 0; n < N; n++)
            {
                AssertClose(re[n], (fProxy)1f, 1E-4f);
                AssertClose(im[n], (fProxy)0f, 1E-4f);
            }
            arena.Dispose();
        }

        // Independent idft oracle: idft of a DC spectrum X=[N,0,0] is the constant x=[1,1,1] (any N).
        void IdftKnownValue()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 3;
            var re = arena.fProxyVec(N);
            var im = arena.fProxyVec(N);
            re[0] = (fProxy)N; // X = [3, 0, 0]
            var oRe = arena.fProxyVec(N);
            var oIm = arena.fProxyVec(N);

            fProxyFFT.idft(in re, in im, ref oRe, ref oIm);

            for (int n = 0; n < N; n++)
            {
                AssertClose(oRe[n], (fProxy)1f, 1E-4f);
                AssertClose(oIm[n], (fProxy)0f, 1E-4f);
            }
            arena.Dispose();
        }

        // Edge sizes: N=1 fft is the identity (early-return path); N=2 is the smallest real butterfly,
        // fft([1,0]) = [1,1] (re), [0,0] (im).
        void FftSmallSizes()
        {
            var arena = new Arena(Allocator.Persistent);

            var re1 = arena.fProxyVec(1);
            var im1 = arena.fProxyVec(1);
            re1[0] = (fProxy)7; im1[0] = (fProxy)(-2);
            fProxyFFT.fft(ref re1, ref im1);
            AssertClose(re1[0], (fProxy)7f, 1E-5f);   // unchanged
            AssertClose(im1[0], (fProxy)(-2f), 1E-5f);

            var re2 = arena.fProxyVec(2);
            var im2 = arena.fProxyVec(2);
            re2[0] = (fProxy)1; re2[1] = (fProxy)0;
            fProxyFFT.fft(ref re2, ref im2);
            AssertClose(re2[0], (fProxy)1f, 1E-5f);   // X0 = x0+x1
            AssertClose(re2[1], (fProxy)1f, 1E-5f);   // X1 = x0-x1
            AssertClose(im2[0], (fProxy)0f, 1E-5f);
            AssertClose(im2[1], (fProxy)0f, 1E-5f);

            arena.Dispose();
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected, [3]=diff
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
    }

    void RunJob(TestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"got {fail[1]}, expected {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (fProxy)0)
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

    // ---- Managed throw tests (guard paths) ----

    [Test]
    public void FftNonPowerOfTwoThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var re = arena.fProxyVec(6); // 6 is not a power of two
        var im = arena.fProxyVec(6);
        Assert.Throws<ArgumentException>(() => fProxyFFT.fft(ref re, ref im));
        Assert.Throws<ArgumentException>(() => fProxyFFT.ifft(ref re, ref im));
        arena.Dispose();
    }

    [Test]
    public void FftMismatchedLengthThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var re = arena.fProxyVec(8);
        var im = arena.fProxyVec(4);
        Assert.Throws<ArgumentException>(() => fProxyFFT.fft(ref re, ref im));
        arena.Dispose();
    }

    [Test]
    public void DftAliasOutputThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var re = arena.fProxyVec(4);
        var im = arena.fProxyVec(4);
        var oRe = arena.fProxyVec(4);
        var oIm = arena.fProxyVec(4);
        // every one of the four out-vs-in pointer collisions must throw (each output bin reads all inputs)
        Assert.Throws<ArgumentException>(() => fProxyFFT.dft(in re, in im, ref re, ref oIm));   // outRe==inRe
        Assert.Throws<ArgumentException>(() => fProxyFFT.dft(in re, in im, ref im, ref oIm));   // outRe==inIm
        Assert.Throws<ArgumentException>(() => fProxyFFT.dft(in re, in im, ref oRe, ref re));   // outIm==inRe
        Assert.Throws<ArgumentException>(() => fProxyFFT.dft(in re, in im, ref oRe, ref im));   // outIm==inIm
        // idft shares the guard via DftCore
        Assert.Throws<ArgumentException>(() => fProxyFFT.idft(in re, in im, ref re, ref oIm));
        arena.Dispose();
    }

    [Test]
    public void DftMismatchedLengthThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var inRe = arena.fProxyVec(4);
        var inIm = arena.fProxyVec(4);
        var inImShort = arena.fProxyVec(3);
        var outRe = arena.fProxyVec(4);
        var outIm = arena.fProxyVec(4);
        var outShort = arena.fProxyVec(3);
        // inRe.N != inIm.N
        Assert.Throws<ArgumentException>(() => fProxyFFT.dft(in inRe, in inImShort, ref outRe, ref outIm));
        // output length != input length
        Assert.Throws<ArgumentException>(() => fProxyFFT.dft(in inRe, in inIm, ref outRe, ref outShort));
        arena.Dispose();
    }

    [Test]
    public void RfftLengthAndAliasThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var real = arena.fProxyVec(8);
        var re = arena.fProxyVec(8);
        var imShort = arena.fProxyVec(4);
        Assert.Throws<ArgumentException>(() => fProxyFFT.rfft(in real, ref re, ref imShort)); // length mismatch
        // im aliasing real must throw
        Assert.Throws<ArgumentException>(() => fProxyFFT.rfft(in real, ref re, ref real));
        arena.Dispose();
    }

    [Test]
    public void ReductionMismatchedLengthThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var re = arena.fProxyVec(4);
        var im = arena.fProxyVec(4);
        var shortDest = arena.fProxyVec(3);
        Assert.Throws<ArgumentException>(() => fProxyFFT.magnitude(in re, in im, ref shortDest));
        Assert.Throws<ArgumentException>(() => fProxyFFT.powerSpectrum(in re, in im, ref shortDest));
        Assert.Throws<ArgumentException>(() => fProxyFFT.phase(in re, in im, ref shortDest));
        arena.Dispose();
    }
}
