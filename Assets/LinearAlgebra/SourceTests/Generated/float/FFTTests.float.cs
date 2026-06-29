using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;

public class floatFFTTests
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
            // twiddle-table workspace tests
            TableFftMatchesRecurrence,
            TableIfftMatchesRecurrence,
            TableRfftMatchesRecurrence,
            TableIrfftMatchesRecurrence,
            TableFftRoundTrip,
            TableRfftRoundTrip,
            // radix-4 oracle validation tests
            Radix4MatchesOracle,
            Radix4RoundTrip,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected, [3] diff
        public NativeArray<float> Fail;

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
                case TestType.TableFftMatchesRecurrence: TableFftMatchesRecurrence(); break;
                case TestType.TableIfftMatchesRecurrence: TableIfftMatchesRecurrence(); break;
                case TestType.TableRfftMatchesRecurrence: TableRfftMatchesRecurrence(); break;
                case TestType.TableIrfftMatchesRecurrence: TableIrfftMatchesRecurrence(); break;
                case TestType.TableFftRoundTrip: TableFftRoundTrip(); break;
                case TestType.TableRfftRoundTrip: TableRfftRoundTrip(); break;
                case TestType.Radix4MatchesOracle: Radix4MatchesOracle(); break;
                case TestType.Radix4RoundTrip: Radix4RoundTrip(); break;
            }
        }

        // FFT of a constant signal x=[1,1,1,1] -> X[0]=N (DC), all other bins 0.
        void FftConstantDC()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 4;
            var re = arena.floatVec(N, 1f);     // all ones
            var im = arena.floatVec(N);         // zeros

            floatFFT.fft(ref re, ref im);

            AssertClose(re[0], (float)N, 1E-4f);
            AssertClose(im[0], (float)0f, 1E-4f);
            for (int k = 1; k < N; k++)
            {
                AssertClose(re[k], (float)0f, 1E-4f);
                AssertClose(im[k], (float)0f, 1E-4f);
            }
            arena.Dispose();
        }

        // x[n]=cos(2πn/N) -> magnitude peaks N/2 at bins 1 and N-1, ~0 elsewhere.
        void FftSingleSinusoid()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 8;
            var re = arena.floatVec(N);
            var im = arena.floatVec(N);
            float w = (float)(2.0 * System.Math.PI) / (float)N;
            for (int n = 0; n < N; n++)
                re[n] = math.cos(w * n);

            floatFFT.fft(ref re, ref im);

            var mag = arena.floatVec(N);
            floatFFT.magnitude(in re, in im, ref mag);

            float half = (float)N * (float)0.5;
            AssertClose(mag[1], half, 1E-3f);
            AssertClose(mag[N - 1], half, 1E-3f);
            // every other bin must be ~0 (no spectral leakage)
            for (int k = 0; k < N; k++)
                if (k != 1 && k != N - 1)
                    AssertClose(mag[k], (float)0f, 1E-3f);
            arena.Dispose();
        }

        // Radix-2 fft must agree bin-for-bin with the direct dft on the same length-8 signal.
        void FftMatchesDft()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 8;
            var sigRe = arena.floatRandomVector(N, -2f, 2f, 9911);
            var sigIm = arena.floatRandomVector(N, -2f, 2f, 2244);

            // fft path (in-place on copies)
            var fRe = sigRe.Copy();
            var fIm = sigIm.Copy();
            floatFFT.fft(ref fRe, ref fIm);

            // dft path
            var dRe = arena.floatVec(N);
            var dIm = arena.floatVec(N);
            floatFFT.dft(in sigRe, in sigIm, ref dRe, ref dIm);

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
            var re0 = arena.floatRandomVector(N, -3f, 3f, 5150);
            var im0 = arena.floatRandomVector(N, -3f, 3f, 6160);

            var re = re0.Copy();
            var im = im0.Copy();
            floatFFT.fft(ref re, ref im);
            floatFFT.ifft(ref re, ref im);

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
            var re0 = arena.floatRandomVector(N, -3f, 3f, 7007);
            var im0 = arena.floatRandomVector(N, -3f, 3f, 8008);

            var fRe = arena.floatVec(N);
            var fIm = arena.floatVec(N);
            floatFFT.dft(in re0, in im0, ref fRe, ref fIm);

            var bRe = arena.floatVec(N);
            var bIm = arena.floatVec(N);
            floatFFT.idft(in fRe, in fIm, ref bRe, ref bIm);

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
            var real = arena.floatRandomVector(N, -2f, 2f, 1234);

            // half-spectrum output
            var rRe = arena.floatVec(halfSpec);
            var rIm = arena.floatVec(halfSpec);
            floatFFT.rfft(in real, ref rRe, ref rIm);

            // full N-point FFT oracle
            var fRe = real.Copy();
            var fIm = arena.floatVec(N); // zeros (real input)
            floatFFT.fft(ref fRe, ref fIm);

            // Compare only the N/2+1 non-redundant bins (0..N/2).
            for (int k = 0; k <= N / 2; k++)
            {
                AssertClose(rRe[k], fRe[k], (float)1E-4f);
                AssertClose(rIm[k], fIm[k], (float)1E-4f);
            }
            // DC and Nyquist imaginaries are always exactly zero for a real signal.
            AssertClose(rIm[0],     (float)0, 0f);
            AssertClose(rIm[N / 2], (float)0, 0f);
            arena.Dispose();
        }

        // magnitude / powerSpectrum / phase known values; power == magnitude².
        void SpectrumReductions()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 3;
            var re = arena.floatVec(N);
            var im = arena.floatVec(N);
            re[0] = 3f;  im[0] = 4f;   // mag 5
            re[1] = -1f; im[1] = 0f;   // mag 1, phase π
            re[2] = 0f;  im[2] = 2f;   // mag 2, phase π/2

            var mag = arena.floatVec(N);
            var pow = arena.floatVec(N);
            var ph = arena.floatVec(N);
            floatFFT.magnitude(in re, in im, ref mag);
            floatFFT.powerSpectrum(in re, in im, ref pow);
            floatFFT.phase(in re, in im, ref ph);

            AssertClose(mag[0], (float)5f, 1E-5f);
            AssertClose(mag[1], (float)1f, 1E-5f);
            AssertClose(mag[2], (float)2f, 1E-5f);

            AssertClose(pow[0], (float)25f, 1E-5f);
            AssertClose(pow[1], (float)1f, 1E-5f);
            AssertClose(pow[2], (float)4f, 1E-5f);

            // power == magnitude²
            for (int i = 0; i < N; i++)
                AssertClose(pow[i], mag[i] * mag[i], 1E-4f);

            AssertClose(ph[0], math.atan2((float)4f, (float)3f), 1E-5f);
            AssertClose(ph[1], (float)System.Math.PI, 1E-5f);
            AssertClose(ph[2], (float)(System.Math.PI * 0.5), 1E-5f);

            // arena wrappers must equal the ref-dest reductions
            var magW = arena.floatMagnitude(in re, in im);
            var powW = arena.floatPowerSpectrum(in re, in im);
            var phW = arena.floatPhase(in re, in im);
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
            var re = arena.floatVec(N, 1f);
            var im = arena.floatVec(N);
            var oRe = arena.floatVec(N);
            var oIm = arena.floatVec(N);

            floatFFT.dft(in re, in im, ref oRe, ref oIm);

            AssertClose(oRe[0], (float)3f, 1E-4f);
            AssertClose(oIm[0], (float)0f, 1E-4f);
            AssertClose(oRe[1], (float)0f, 1E-4f); // sum of cube roots of unity = 0
            AssertClose(oRe[2], (float)0f, 1E-4f);
            AssertClose(oIm[1], (float)0f, 1E-4f); // imag must also vanish (catches a sign-conv bug)
            AssertClose(oIm[2], (float)0f, 1E-4f);
            arena.Dispose();
        }

        // Independent ifft oracle (NOT a forward+inverse round-trip, so it can't hide complementary
        // sign/scale bugs): ifft of a DC spectrum X=[N,0,…,0] is the constant signal x[n]=1. N=8 so the
        // bit-reversal permutation is non-trivial.
        void IfftKnownValue()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 8;
            var re = arena.floatVec(N);
            var im = arena.floatVec(N);
            re[0] = (float)N; // X = [8, 0, 0, ...]

            floatFFT.ifft(ref re, ref im);

            for (int n = 0; n < N; n++)
            {
                AssertClose(re[n], (float)1f, 1E-4f);
                AssertClose(im[n], (float)0f, 1E-4f);
            }
            arena.Dispose();
        }

        // Independent idft oracle: idft of a DC spectrum X=[N,0,0] is the constant x=[1,1,1] (any N).
        void IdftKnownValue()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 3;
            var re = arena.floatVec(N);
            var im = arena.floatVec(N);
            re[0] = (float)N; // X = [3, 0, 0]
            var oRe = arena.floatVec(N);
            var oIm = arena.floatVec(N);

            floatFFT.idft(in re, in im, ref oRe, ref oIm);

            for (int n = 0; n < N; n++)
            {
                AssertClose(oRe[n], (float)1f, 1E-4f);
                AssertClose(oIm[n], (float)0f, 1E-4f);
            }
            arena.Dispose();
        }

        // Edge sizes: N=1 fft is the identity (early-return path); N=2 is the smallest real butterfly,
        // fft([1,0]) = [1,1] (re), [0,0] (im).
        void FftSmallSizes()
        {
            var arena = new Arena(Allocator.Persistent);

            var re1 = arena.floatVec(1);
            var im1 = arena.floatVec(1);
            re1[0] = (float)7; im1[0] = (float)(-2);
            floatFFT.fft(ref re1, ref im1);
            AssertClose(re1[0], (float)7f, 1E-5f);   // unchanged
            AssertClose(im1[0], (float)(-2f), 1E-5f);

            var re2 = arena.floatVec(2);
            var im2 = arena.floatVec(2);
            re2[0] = (float)1; re2[1] = (float)0;
            floatFFT.fft(ref re2, ref im2);
            AssertClose(re2[0], (float)1f, 1E-5f);   // X0 = x0+x1
            AssertClose(re2[1], (float)1f, 1E-5f);   // X1 = x0-x1
            AssertClose(im2[0], (float)0f, 1E-5f);
            AssertClose(im2[1], (float)0f, 1E-5f);

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
                var real0 = arena.floatRandomVector(N, -3f, 3f, 5555);
                var rRe = arena.floatVec(halfSpec);
                var rIm = arena.floatVec(halfSpec);
                var real2 = arena.floatVec(N);
                floatFFT.rfft(in real0, ref rRe, ref rIm);
                floatFFT.irfft(in rRe, in rIm, ref real2);
                for (int i = 0; i < N; i++)
                    AssertClose(real2[i], real0[i], (float)1E-4f);
            }

            // N=16
            {
                int N = 16;
                int halfSpec = (N >> 1) + 1;
                var real0 = arena.floatRandomVector(N, -3f, 3f, 6666);
                var rRe = arena.floatVec(halfSpec);
                var rIm = arena.floatVec(halfSpec);
                var real2 = arena.floatVec(N);
                floatFFT.rfft(in real0, ref rRe, ref rIm);
                floatFFT.irfft(in rRe, in rIm, ref real2);
                for (int i = 0; i < N; i++)
                    AssertClose(real2[i], real0[i], (float)1E-4f);
            }

            // N=64
            {
                int N = 64;
                int halfSpec = (N >> 1) + 1;
                var real0 = arena.floatRandomVector(N, -3f, 3f, 7777);
                var rRe = arena.floatVec(halfSpec);
                var rIm = arena.floatVec(halfSpec);
                var real2 = arena.floatVec(N);
                floatFFT.rfft(in real0, ref rRe, ref rIm);
                floatFFT.irfft(in rRe, in rIm, ref real2);
                for (int i = 0; i < N; i++)
                    AssertClose(real2[i], real0[i], (float)1E-4f);
            }

            // irfft arena wrapper round-trip (N=8)
            {
                int N = 8;
                int halfSpec = (N >> 1) + 1;
                var real0 = arena.floatRandomVector(N, -3f, 3f, 8888);
                var rRe = arena.floatVec(halfSpec);
                var rIm = arena.floatVec(halfSpec);
                floatFFT.rfft(in real0, ref rRe, ref rIm);
                var real2 = arena.floatIrfft(in rRe, in rIm);
                for (int i = 0; i < N; i++)
                    AssertClose(real2[i], real0[i], (float)1E-4f);
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
                var dc = arena.floatVec(N, 1f);
                var dcRe = arena.floatVec(halfSpec);
                var dcIm = arena.floatVec(halfSpec);
                floatFFT.rfft(in dc, ref dcRe, ref dcIm);
                AssertClose(dcRe[0], (float)N, (float)1E-4f);
                AssertClose(dcIm[0], (float)0, 0f);
                for (int k = 1; k <= N / 2; k++)
                {
                    AssertClose(dcRe[k], (float)0, (float)1E-4f);
                    AssertClose(dcIm[k], (float)0, (float)1E-4f);
                }
            }

            // --- Pure cosine at integer frequency f=2: x[n] = cos(2π·2·n/N) ---
            // Full-spectrum DFT has N/2 at bin f and N/2 at bin N-f; the half-spectrum sees bin f.
            // re[f]=N/2, im[f]≈0; all other half-spectrum bins ≈0.
            {
                int f = 2;
                var cosX = arena.floatVec(N);
                float wf = (float)(2.0 * System.Math.PI * f) / (float)N;
                for (int n = 0; n < N; n++)
                    cosX[n] = math.cos(wf * (float)n);

                var cosRe = arena.floatVec(halfSpec);
                var cosIm = arena.floatVec(halfSpec);
                floatFFT.rfft(in cosX, ref cosRe, ref cosIm);

                AssertClose(cosRe[f], (float)(N / 2), (float)1E-4f);
                AssertClose(cosIm[f], (float)0, (float)1E-4f);
                for (int k = 0; k <= N / 2; k++)
                {
                    if (k == f) continue;
                    AssertClose(cosRe[k], (float)0, (float)1E-4f);
                    AssertClose(cosIm[k], (float)0, (float)1E-4f);
                }
            }

            // --- Nyquist: x[n] = (-1)^n ---
            // X[N/2]=N, all other bins 0; im all 0.
            {
                var nyq = arena.floatVec(N);
                for (int n = 0; n < N; n++)
                    nyq[n] = (n % 2 == 0) ? (float)1 : (float)(-1);

                var nyqRe = arena.floatVec(halfSpec);
                var nyqIm = arena.floatVec(halfSpec);
                floatFFT.rfft(in nyq, ref nyqRe, ref nyqIm);

                AssertClose(nyqRe[N / 2], (float)N, (float)1E-4f);
                AssertClose(nyqIm[N / 2], (float)0, 0f);
                for (int k = 0; k < N / 2; k++)
                {
                    AssertClose(nyqRe[k], (float)0, (float)1E-4f);
                    AssertClose(nyqIm[k], (float)0, (float)1E-4f);
                }
            }

            // --- N=2 edge case: x=[a,b] -> re=[a+b, a-b], im=[0,0] ---
            {
                var x2 = arena.floatVec(2);
                x2[0] = (float)3; x2[1] = (float)7;
                var r2 = arena.floatVec(2);
                var i2 = arena.floatVec(2);
                floatFFT.rfft(in x2, ref r2, ref i2);
                AssertClose(r2[0], (float)10, (float)1E-5f);
                AssertClose(r2[1], (float)(-4), (float)1E-5f);
                AssertClose(i2[0], (float)0, 0f);
                AssertClose(i2[1], (float)0, 0f);
            }

            arena.Dispose();
        }

        // ---- twiddle-table workspace tests ----

        // Helper: compare table fft vs recurrence fft for one size.
        void TableFftVsRecurrence(int N, uint seedRe, uint seedIm)
        {
            var arena = new Arena(Allocator.Persistent);
            var ws = arena.floatFftWorkspace(N);
            var re0 = arena.floatRandomVector(N, -2f, 2f, seedRe);
            var im0 = arena.floatRandomVector(N, -2f, 2f, seedIm);

            var reR = re0.Copy(); var imR = im0.Copy();
            floatFFT.fft(ref reR, ref imR);

            var reT = re0.Copy(); var imT = im0.Copy();
            floatFFT.fft(ref reT, ref imT, in ws);

            float tol = (float)1E-4f;
            for (int k = 0; k < N; k++)
            {
                AssertClose(reT[k], reR[k], tol);
                AssertClose(imT[k], imR[k], tol);
            }
            arena.Dispose();
        }

        // Table fft == recurrence fft on random inputs at N=8, 16, 64, 256.
        void TableFftMatchesRecurrence()
        {
            TableFftVsRecurrence(8,   4321u, 4332u);
            TableFftVsRecurrence(16,  4343u, 4354u);
            TableFftVsRecurrence(64,  4365u, 4376u);
            TableFftVsRecurrence(256, 4387u, 4398u);
        }

        // Helper: compare table ifft vs recurrence ifft for one size.
        void TableIfftVsRecurrence(int N, uint seedRe, uint seedIm)
        {
            var arena = new Arena(Allocator.Persistent);
            var ws = arena.floatFftWorkspace(N);
            var re0 = arena.floatRandomVector(N, -2f, 2f, seedRe);
            var im0 = arena.floatRandomVector(N, -2f, 2f, seedIm);

            var reR = re0.Copy(); var imR = im0.Copy();
            floatFFT.ifft(ref reR, ref imR);

            var reT = re0.Copy(); var imT = im0.Copy();
            floatFFT.ifft(ref reT, ref imT, in ws);

            float tol = (float)1E-4f;
            for (int k = 0; k < N; k++)
            {
                AssertClose(reT[k], reR[k], tol);
                AssertClose(imT[k], imR[k], tol);
            }
            arena.Dispose();
        }

        // Table ifft == recurrence ifft on random inputs at N=8, 16, 64, 256.
        void TableIfftMatchesRecurrence()
        {
            TableIfftVsRecurrence(8,   9999u, 10001u);
            TableIfftVsRecurrence(16,  10013u, 10027u);
            TableIfftVsRecurrence(64,  10039u, 10051u);
            TableIfftVsRecurrence(256, 10063u, 10079u);
        }

        // Helper: compare table rfft vs recurrence rfft for one size.
        void TableRfftVsRecurrence(int N, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);
            int halfSpec = (N >> 1) + 1;
            var ws = arena.floatFftWorkspace(N);
            var real = arena.floatRandomVector(N, -2f, 2f, seed);

            var reR = arena.floatVec(halfSpec);
            var imR = arena.floatVec(halfSpec);
            floatFFT.rfft(in real, ref reR, ref imR);

            var reT = arena.floatVec(halfSpec);
            var imT = arena.floatVec(halfSpec);
            floatFFT.rfft(in real, ref reT, ref imT, in ws);

            float tol = (float)1E-4f;
            for (int k = 0; k <= N / 2; k++)
            {
                AssertClose(reT[k], reR[k], tol);
                AssertClose(imT[k], imR[k], tol);
            }
            // DC and Nyquist imaginary parts must be exactly zero
            AssertClose(imT[0],     (float)0, 0f);
            AssertClose(imT[N / 2], (float)0, 0f);
            arena.Dispose();
        }

        // Table rfft == recurrence rfft on random real inputs at N=8, 16, 64, 256.
        void TableRfftMatchesRecurrence()
        {
            TableRfftVsRecurrence(8,   2468u);
            TableRfftVsRecurrence(16,  2475u);
            TableRfftVsRecurrence(64,  2482u);
            TableRfftVsRecurrence(256, 2489u);
        }

        // Helper: compare table irfft vs recurrence irfft for one size.
        void TableIrfftVsRecurrence(int N, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);
            int halfSpec = (N >> 1) + 1;
            var ws = arena.floatFftWorkspace(N);
            var real0 = arena.floatRandomVector(N, -2f, 2f, seed);

            var specRe = arena.floatVec(halfSpec);
            var specIm = arena.floatVec(halfSpec);
            floatFFT.rfft(in real0, ref specRe, ref specIm);   // recurrence rfft as spectrum source

            var realR = arena.floatVec(N);
            floatFFT.irfft(in specRe, in specIm, ref realR);

            var realT = arena.floatVec(N);
            floatFFT.irfft(in specRe, in specIm, ref realT, in ws);

            float tol = (float)1E-4f;
            for (int i = 0; i < N; i++)
                AssertClose(realT[i], realR[i], tol);
            arena.Dispose();
        }

        // Table irfft == recurrence irfft at N=8, 16, 64, 256.
        void TableIrfftMatchesRecurrence()
        {
            TableIrfftVsRecurrence(8,   1357u);
            TableIrfftVsRecurrence(16,  1374u);
            TableIrfftVsRecurrence(64,  1391u);
            TableIrfftVsRecurrence(256, 1408u);
        }

        // ifft(fft(x, ws), ws) == x with a workspace.
        void TableFftRoundTrip()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 64;
            var ws = arena.floatFftWorkspace(N);

            var re0 = arena.floatRandomVector(N, -3f, 3f, 6543u);
            var im0 = arena.floatRandomVector(N, -3f, 3f, 7654u);

            var re = re0.Copy(); var im = im0.Copy();
            floatFFT.fft(ref re, ref im, in ws);
            floatFFT.ifft(ref re, ref im, in ws);

            float tol = (float)1E-3f;
            for (int i = 0; i < N; i++)
            {
                AssertClose(re[i], re0[i], tol);
                AssertClose(im[i], im0[i], tol);
            }
            arena.Dispose();
        }

        // Helper: irfft(rfft(x, ws), ws) == x for one size.
        void TableRfftRoundTripOneSize(int N, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);
            int halfSpec = (N >> 1) + 1;
            var ws = arena.floatFftWorkspace(N);
            var real0 = arena.floatRandomVector(N, -3f, 3f, seed);

            var rRe = arena.floatVec(halfSpec);
            var rIm = arena.floatVec(halfSpec);
            floatFFT.rfft(in real0, ref rRe, ref rIm, in ws);

            var real2 = arena.floatVec(N);
            floatFFT.irfft(in rRe, in rIm, ref real2, in ws);

            float tol = (float)1E-3f;
            for (int i = 0; i < N; i++)
                AssertClose(real2[i], real0[i], tol);
            arena.Dispose();
        }

        // irfft(rfft(x, ws), ws) == x with a workspace at N=8, 16, 64, 256.
        void TableRfftRoundTrip()
        {
            TableRfftRoundTripOneSize(8,   8765u);
            TableRfftRoundTripOneSize(16,  8796u);
            TableRfftRoundTripOneSize(64,  8827u);
            TableRfftRoundTripOneSize(256, 8858u);
        }

        // ---- radix-4 oracle validation tests ----

        // Helper: compare fftRadix4 dispatch against the table oracle fft(ws) for one size.
        // Using fft(ws) as oracle (not the recurrence fft()) eliminates twiddle-source discrepancy:
        // both paths use the same double-precision-cast twiddle table.
        // Power-of-4 sizes exercise FftCoreRadix4 (radix-4 butterfly).
        // Non-power-of-4 power-of-2 sizes (2·4^k) exercise FftCoreRadix4Mixed (one radix-2 stage
        // wrapping two radix-4 sub-FFTs); the oracle uses FftCoreTable, so small differences are
        // expected and covered by the relative tolerance.
        void Radix4VsOracleOneSize(int N, uint seedRe, uint seedIm)
        {
            var arena = new Arena(Allocator.Persistent);
            var ws    = arena.floatFftWorkspace(N);   // has both half-table and full-circle table

            var re0 = arena.floatRandomVector(N, -2f, 2f, seedRe);
            var im0 = arena.floatRandomVector(N, -2f, 2f, seedIm);

            // Oracle: table-based radix-2 fft (same twiddle source as fftRadix4 full table).
            var reRef = re0.Copy(); var imRef = im0.Copy();
            floatFFT.fft(ref reRef, ref imRef, in ws);

            // Radix-4 dispatch: calls FftCoreRadix4 for pow-of-4 sizes, FftCoreTable otherwise.
            // For non-pow-of-4 sizes both paths call FftCoreTable so the error is exactly zero.
            var reR4 = re0.Copy(); var imR4 = im0.Copy();
            floatFFT.fftRadix4(ref reR4, ref imR4, in ws);

            // Relative tolerance: oracle fft() uses per-stage twiddle recurrence; fftRadix4 uses a
            // precomputed full-circle table — two different arithmetic paths. At float precision with
            // N=4096, output magnitudes can reach O(300) and relative differences of ~5e-6 are normal.
            // Absolute tolerance scales with each bin's reference magnitude so small bins stay tight.
            float relTol = (float)1E-3f;
            for (int k = 0; k < N; k++)
            {
                // Floor at 1.0 so near-zero bins still get an absolute tolerance of relTol.
                float absTolRe = relTol * math.max((float)1.0f, math.abs(reRef[k]));
                float absTolIm = relTol * math.max((float)1.0f, math.abs(imRef[k]));
                AssertClose(reR4[k], reRef[k], absTolRe);
                AssertClose(imR4[k], imRef[k], absTolIm);
            }

            arena.Dispose();
        }

        // Compare fftRadix4 vs fft oracle at all sizes in {2,4,8,...,4096,8192,32768}.
        // Power-of-4 sizes {4,16,64,256,1024,4096} exercise the true radix-4 path.
        // Non-power-of-4 sizes {2,8,32,128,512,2048,8192,32768} exercise FftCoreRadix4Mixed.
        void Radix4MatchesOracle()
        {
            Radix4VsOracleOneSize(2,     31001u, 31002u);
            Radix4VsOracleOneSize(4,     31003u, 31004u);
            Radix4VsOracleOneSize(8,     31005u, 31006u);
            Radix4VsOracleOneSize(16,    31007u, 31008u);
            Radix4VsOracleOneSize(32,    31009u, 31010u);
            Radix4VsOracleOneSize(64,    31011u, 31012u);
            Radix4VsOracleOneSize(128,   31013u, 31014u);
            Radix4VsOracleOneSize(256,   31015u, 31016u);
            Radix4VsOracleOneSize(512,   31017u, 31018u);
            Radix4VsOracleOneSize(1024,  31019u, 31020u);
            Radix4VsOracleOneSize(2048,  31021u, 31022u);
            Radix4VsOracleOneSize(4096,  31023u, 31024u);
            Radix4VsOracleOneSize(8192,  31025u, 31026u);   // 2·4^6 mixed-radix
            Radix4VsOracleOneSize(32768, 31027u, 31028u);   // 2·4^7 mixed-radix
        }

        // Helper: ifftRadix4(fftRadix4(x, ws), ws) == x for one size.
        void Radix4RoundTripOneSize(int N, uint seedRe, uint seedIm)
        {
            var arena = new Arena(Allocator.Persistent);
            var ws    = arena.floatFftWorkspace(N);

            var re0 = arena.floatRandomVector(N, -3f, 3f, seedRe);
            var im0 = arena.floatRandomVector(N, -3f, 3f, seedIm);

            var re = re0.Copy(); var im = im0.Copy();
            floatFFT.fftRadix4(ref re, ref im, in ws);
            floatFFT.ifftRadix4(ref re, ref im, in ws);

            float tol = (float)1E-3f;
            for (int i = 0; i < N; i++)
            {
                AssertClose(re[i], re0[i], tol);
                AssertClose(im[i], im0[i], tol);
            }

            arena.Dispose();
        }

        // Round-trip at power-of-4 sizes and mixed-radix (2·4^k) sizes.
        void Radix4RoundTrip()
        {
            Radix4RoundTripOneSize(4,    32001u, 32002u);
            Radix4RoundTripOneSize(16,   32003u, 32004u);
            Radix4RoundTripOneSize(64,   32005u, 32006u);
            Radix4RoundTripOneSize(256,  32007u, 32008u);
            Radix4RoundTripOneSize(1024, 32009u, 32010u);
            Radix4RoundTripOneSize(4096, 32011u, 32012u);
            Radix4RoundTripOneSize(8,    32013u, 32014u);   // 2·4^1 mixed-radix
            Radix4RoundTripOneSize(512,  32015u, 32016u);   // 2·4^4 mixed-radix
            Radix4RoundTripOneSize(2048, 32017u, 32018u);   // 2·4^5 mixed-radix
            Radix4RoundTripOneSize(8192, 32019u, 32020u);   // 2·4^6 mixed-radix
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected, [3]=diff
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
    }

    void RunJob(TestJob.TestType type)
    {
        var fail = new NativeArray<float>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (float)0)
                Assert.Fail($"got {fail[1]}, expected {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (float)0)
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
    // twiddle-table workspace tests
    [Test] public void TableFftMatchesRecurrenceTest() => RunJob(TestJob.TestType.TableFftMatchesRecurrence);
    [Test] public void TableIfftMatchesRecurrenceTest() => RunJob(TestJob.TestType.TableIfftMatchesRecurrence);
    [Test] public void TableRfftMatchesRecurrenceTest() => RunJob(TestJob.TestType.TableRfftMatchesRecurrence);
    [Test] public void TableIrfftMatchesRecurrenceTest() => RunJob(TestJob.TestType.TableIrfftMatchesRecurrence);
    [Test] public void TableFftRoundTripTest() => RunJob(TestJob.TestType.TableFftRoundTrip);
    [Test] public void TableRfftRoundTripTest() => RunJob(TestJob.TestType.TableRfftRoundTrip);
    // radix-4 oracle validation
    [Test] public void Radix4MatchesOracleTest() => RunJob(TestJob.TestType.Radix4MatchesOracle);
    [Test] public void Radix4RoundTripTest() => RunJob(TestJob.TestType.Radix4RoundTrip);

    // ---- Managed throw tests (guard paths) ----

    [Test]
    public void FftNonPowerOfTwoThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var re = arena.floatVec(6); // 6 is not a power of two
        var im = arena.floatVec(6);
        Assert.Throws<ArgumentException>(() => floatFFT.fft(ref re, ref im));
        Assert.Throws<ArgumentException>(() => floatFFT.ifft(ref re, ref im));
        arena.Dispose();
    }

    [Test]
    public void FftMismatchedLengthThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var re = arena.floatVec(8);
        var im = arena.floatVec(4);
        Assert.Throws<ArgumentException>(() => floatFFT.fft(ref re, ref im));
        arena.Dispose();
    }

    [Test]
    public void DftAliasOutputThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var re = arena.floatVec(4);
        var im = arena.floatVec(4);
        var oRe = arena.floatVec(4);
        var oIm = arena.floatVec(4);
        // every one of the four out-vs-in pointer collisions must throw (each output bin reads all inputs)
        Assert.Throws<ArgumentException>(() => floatFFT.dft(in re, in im, ref re, ref oIm));   // outRe==inRe
        Assert.Throws<ArgumentException>(() => floatFFT.dft(in re, in im, ref im, ref oIm));   // outRe==inIm
        Assert.Throws<ArgumentException>(() => floatFFT.dft(in re, in im, ref oRe, ref re));   // outIm==inRe
        Assert.Throws<ArgumentException>(() => floatFFT.dft(in re, in im, ref oRe, ref im));   // outIm==inIm
        // idft shares the guard via DftCore
        Assert.Throws<ArgumentException>(() => floatFFT.idft(in re, in im, ref re, ref oIm));
        arena.Dispose();
    }

    [Test]
    public void DftMismatchedLengthThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var inRe = arena.floatVec(4);
        var inIm = arena.floatVec(4);
        var inImShort = arena.floatVec(3);
        var outRe = arena.floatVec(4);
        var outIm = arena.floatVec(4);
        var outShort = arena.floatVec(3);
        // inRe.N != inIm.N
        Assert.Throws<ArgumentException>(() => floatFFT.dft(in inRe, in inImShort, ref outRe, ref outIm));
        // output length != input length
        Assert.Throws<ArgumentException>(() => floatFFT.dft(in inRe, in inIm, ref outRe, ref outShort));
        arena.Dispose();
    }

    [Test]
    public void RfftLengthAndAliasThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var real = arena.floatVec(8);
        // For N=8, the correct half-spectrum length is N/2+1 = 5.
        var re5  = arena.floatVec(5);  // correct
        var im5  = arena.floatVec(5);  // correct
        var re8  = arena.floatVec(8);  // wrong (full-N, not N/2+1)
        var im4  = arena.floatVec(4);  // wrong

        // wrong re length
        Assert.Throws<ArgumentException>(() => floatFFT.rfft(in real, ref re8, ref im5));
        // wrong im length
        Assert.Throws<ArgumentException>(() => floatFFT.rfft(in real, ref re5, ref im4));
        // non-power-of-two real length
        var real7 = arena.floatVec(7);
        var re4   = arena.floatVec(4);
        var im4b  = arena.floatVec(4);
        Assert.Throws<ArgumentException>(() => floatFFT.rfft(in real7, ref re4, ref im4b));
        // im aliasing real must throw
        Assert.Throws<ArgumentException>(() => floatFFT.rfft(in real, ref re5, ref real));
        arena.Dispose();
    }

    [Test]
    public void IrfftGuards()
    {
        var arena = new Arena(Allocator.Persistent);
        // For N=8, half-spectrum has length N/2+1=5.
        var re5   = arena.floatVec(5);
        var im5   = arena.floatVec(5);
        var real8 = arena.floatVec(8);

        // im.N != re.N
        var im4 = arena.floatVec(4);
        Assert.Throws<ArgumentException>(() => floatFFT.irfft(in re5, in im4, ref real8));

        // halfSpec < 2 (re.N=1 means N=0; minimum is N=2)
        var re1  = arena.floatVec(1);
        var im1  = arena.floatVec(1);
        Assert.Throws<ArgumentException>(() => floatFFT.irfft(in re1, in im1, ref real8));

        // wrong real output length (real.N=7 but N=8)
        var real7 = arena.floatVec(7);
        Assert.Throws<ArgumentException>(() => floatFFT.irfft(in re5, in im5, ref real7));

        // Alias tests: use N=2 (halfSpec=2, real.N=2) so all length guards pass and the alias
        // check is reached. re2.N=2 = N, so real.N matches and the ptr check fires.
        var re2  = arena.floatVec(2);
        var im2  = arena.floatVec(2);
        var real2 = arena.floatVec(2);

        // real aliasing re (correct lengths: halfSpec=2, N=2)
        Assert.Throws<ArgumentException>(() => floatFFT.irfft(in re2, in im2, ref re2));
        // real aliasing im (correct lengths)
        Assert.Throws<ArgumentException>(() => floatFFT.irfft(in re2, in im2, ref im2));

        arena.Dispose();
    }

    [Test]
    public void ReductionMismatchedLengthThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var re = arena.floatVec(4);
        var im = arena.floatVec(4);
        var shortDest = arena.floatVec(3);
        Assert.Throws<ArgumentException>(() => floatFFT.magnitude(in re, in im, ref shortDest));
        Assert.Throws<ArgumentException>(() => floatFFT.powerSpectrum(in re, in im, ref shortDest));
        Assert.Throws<ArgumentException>(() => floatFFT.phase(in re, in im, ref shortDest));
        arena.Dispose();
    }

    // ---- workspace guard tests ----

    [Test]
    public void FftWorkspaceFactoryNonPow2Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        Assert.Throws<ArgumentException>(() => arena.floatFftWorkspace(0));
        Assert.Throws<ArgumentException>(() => arena.floatFftWorkspace(1));
        Assert.Throws<ArgumentException>(() => arena.floatFftWorkspace(3));
        Assert.Throws<ArgumentException>(() => arena.floatFftWorkspace(5));
        Assert.Throws<ArgumentException>(() => arena.floatFftWorkspace(6));
        arena.Dispose();
    }

    [Test]
    public void FftTableWrongWorkspaceSizeThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var re8  = arena.floatVec(8);
        var im8  = arena.floatVec(8);
        var ws16 = arena.floatFftWorkspace(16);   // sized for 16, not 8

        // fft and ifft with mismatched workspace
        Assert.Throws<ArgumentException>(() => floatFFT.fft(ref re8, ref im8, in ws16));
        Assert.Throws<ArgumentException>(() => floatFFT.ifft(ref re8, ref im8, in ws16));

        // rfft: real.N=8 but ws.n=16
        var real8  = arena.floatVec(8);
        var reHalf = arena.floatVec(5);   // 8/2+1=5
        var imHalf = arena.floatVec(5);
        Assert.Throws<ArgumentException>(() => floatFFT.rfft(in real8, ref reHalf, ref imHalf, in ws16));

        // irfft: re.N=5 -> N=8, but ws.n=16
        Assert.Throws<ArgumentException>(() => floatFFT.irfft(in reHalf, in imHalf, ref real8, in ws16));

        arena.Dispose();
    }

    [Test]
    public void RfftTableWrongOutputLengthThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var ws = arena.floatFftWorkspace(8);
        var real = arena.floatVec(8);
        var re5  = arena.floatVec(5);    // correct N/2+1
        var im5  = arena.floatVec(5);    // correct
        var re4  = arena.floatVec(4);    // wrong
        var im4  = arena.floatVec(4);    // wrong

        Assert.Throws<ArgumentException>(() => floatFFT.rfft(in real, ref re4, ref im5, in ws));
        Assert.Throws<ArgumentException>(() => floatFFT.rfft(in real, ref re5, ref im4, in ws));
        arena.Dispose();
    }
}
