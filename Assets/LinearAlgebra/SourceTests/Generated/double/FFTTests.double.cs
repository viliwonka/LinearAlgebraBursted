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
            // comprehensive numerical-stability / correctness suite
            FftVsDftCrossCheck,
            ParsevalEnergy,
            FftLinearity,
            KnownAnalytics,
            RoundTripLargeN,
            WorkspaceReuse,
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
                case TestType.TableFftMatchesRecurrence: TableFftMatchesRecurrence(); break;
                case TestType.TableIfftMatchesRecurrence: TableIfftMatchesRecurrence(); break;
                case TestType.TableRfftMatchesRecurrence: TableRfftMatchesRecurrence(); break;
                case TestType.TableIrfftMatchesRecurrence: TableIrfftMatchesRecurrence(); break;
                case TestType.TableFftRoundTrip: TableFftRoundTrip(); break;
                case TestType.TableRfftRoundTrip: TableRfftRoundTrip(); break;
                case TestType.Radix4MatchesOracle: Radix4MatchesOracle(); break;
                case TestType.Radix4RoundTrip: Radix4RoundTrip(); break;
                case TestType.FftVsDftCrossCheck: FftVsDftCrossCheck(); break;
                case TestType.ParsevalEnergy: ParsevalEnergy(); break;
                case TestType.FftLinearity: FftLinearity(); break;
                case TestType.KnownAnalytics: KnownAnalytics(); break;
                case TestType.RoundTripLargeN: RoundTripLargeN(); break;
                case TestType.WorkspaceReuse: WorkspaceReuse(); break;
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

        // ---- twiddle-table workspace tests ----

        // Helper: compare fft(ws) (auto-dispatch) vs recurrence fft for one size.
        // fft(ws) now dispatches to radix-4 for power-of-4 lengths and mixed for 2·4^k lengths,
        // so the two algorithms accumulate rounding differently. Use relative tolerance (same
        // scheme as Radix4MatchesOracle) — floor at 1.0 so near-zero bins get absolute relTol.
        void TableFftVsRecurrence(int N, uint seedRe, uint seedIm)
        {
            var arena = new Arena(Allocator.Persistent);
            var ws = arena.doubleFftWorkspace(N);
            var re0 = arena.doubleRandomVector(N, -2f, 2f, seedRe);
            var im0 = arena.doubleRandomVector(N, -2f, 2f, seedIm);

            var reR = re0.Copy(); var imR = im0.Copy();
            doubleFFT.fft(ref reR, ref imR);

            var reT = re0.Copy(); var imT = im0.Copy();
            doubleFFT.fft(ref reT, ref imT, in ws);

            double relTol = (double)1E-3f;
            for (int k = 0; k < N; k++)
            {
                double absTolRe = relTol * math.max((double)1.0f, math.abs(reR[k]));
                double absTolIm = relTol * math.max((double)1.0f, math.abs(imR[k]));
                AssertClose(reT[k], reR[k], absTolRe);
                AssertClose(imT[k], imR[k], absTolIm);
            }
            arena.Dispose();
        }

        // fft(ws) auto-dispatch == recurrence fft on random inputs at N=8, 16, 64, 256.
        // N=16,64,256 exercise the radix-4 path; N=8 exercises the mixed-radix path.
        void TableFftMatchesRecurrence()
        {
            TableFftVsRecurrence(8,   4321u, 4332u);
            TableFftVsRecurrence(16,  4343u, 4354u);
            TableFftVsRecurrence(64,  4365u, 4376u);
            TableFftVsRecurrence(256, 4387u, 4398u);
        }

        // Helper: compare ifft(ws) (auto-dispatch) vs recurrence ifft for one size.
        // Same relative-tolerance scheme as TableFftVsRecurrence — radix-4 and mixed paths
        // accumulate rounding differently from the recurrence inverse.
        void TableIfftVsRecurrence(int N, uint seedRe, uint seedIm)
        {
            var arena = new Arena(Allocator.Persistent);
            var ws = arena.doubleFftWorkspace(N);
            var re0 = arena.doubleRandomVector(N, -2f, 2f, seedRe);
            var im0 = arena.doubleRandomVector(N, -2f, 2f, seedIm);

            var reR = re0.Copy(); var imR = im0.Copy();
            doubleFFT.ifft(ref reR, ref imR);

            var reT = re0.Copy(); var imT = im0.Copy();
            doubleFFT.ifft(ref reT, ref imT, in ws);

            double relTol = (double)1E-3f;
            for (int k = 0; k < N; k++)
            {
                double absTolRe = relTol * math.max((double)1.0f, math.abs(reR[k]));
                double absTolIm = relTol * math.max((double)1.0f, math.abs(imR[k]));
                AssertClose(reT[k], reR[k], absTolRe);
                AssertClose(imT[k], imR[k], absTolIm);
            }
            arena.Dispose();
        }

        // ifft(ws) auto-dispatch == recurrence ifft on random inputs at N=8, 16, 64, 256.
        void TableIfftMatchesRecurrence()
        {
            TableIfftVsRecurrence(8,   9999u, 10001u);
            TableIfftVsRecurrence(16,  10013u, 10027u);
            TableIfftVsRecurrence(64,  10039u, 10051u);
            TableIfftVsRecurrence(256, 10063u, 10079u);
        }

        // Helper: compare table rfft(ws) [now radix-4 inner] vs recurrence rfft for one size.
        // Tight 1E-4 absolute tolerance — appropriate for N ≤ 256 where both algorithms agree to
        // that precision. For larger N the cross-algorithm absolute error grows to ~1e-3 (both
        // algorithms are still CORRECT; they just diverge in absolute terms for near-zero bins).
        // Large N correctness is validated by TableRfftRoundTrip (self-consistent irfft·rfft==id).
        void TableRfftVsRecurrence(int N, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);
            int halfSpec = (N >> 1) + 1;
            var ws = arena.doubleFftWorkspace(N);
            var real = arena.doubleRandomVector(N, -2f, 2f, seed);

            var reR = arena.doubleVec(halfSpec);
            var imR = arena.doubleVec(halfSpec);
            doubleFFT.rfft(in real, ref reR, ref imR);

            var reT = arena.doubleVec(halfSpec);
            var imT = arena.doubleVec(halfSpec);
            doubleFFT.rfft(in real, ref reT, ref imT, in ws);

            double tol = (double)1E-4f;
            for (int k = 0; k <= N / 2; k++)
            {
                AssertClose(reT[k], reR[k], tol);
                AssertClose(imT[k], imR[k], tol);
            }
            // DC and Nyquist imaginary parts must be exactly zero (set unconditionally in rfft)
            AssertClose(imT[0],     (double)0, 0f);
            AssertClose(imT[N / 2], (double)0, 0f);
            arena.Dispose();
        }

        // Table rfft(ws) == recurrence rfft at N ≤ 256 (tight cross-algorithm comparison, 1E-4).
        // Covers BOTH inner-M paths:
        //   IsPowerOf4(M)=true  (Radix4 inner): N=2(M=1), N=8(M=4), N=32(M=16), N=128(M=64)
        //   IsPowerOf4(M)=false (Mixed inner):  N=4(M=2), N=16(M=8), N=64(M=32), N=256(M=128)
        // N > 256 cross-algorithm agreement is validated by TableRfftRoundTrip (irfft·rfft==id).
        void TableRfftMatchesRecurrence()
        {
            TableRfftVsRecurrence(2,   3001u);
            TableRfftVsRecurrence(4,   3005u);
            TableRfftVsRecurrence(8,   2468u);
            TableRfftVsRecurrence(16,  2475u);
            TableRfftVsRecurrence(32,  3009u);
            TableRfftVsRecurrence(64,  2482u);
            TableRfftVsRecurrence(128, 3013u);
            TableRfftVsRecurrence(256, 2489u);
        }

        // Helper: compare table irfft vs recurrence irfft for one size.
        void TableIrfftVsRecurrence(int N, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);
            int halfSpec = (N >> 1) + 1;
            var ws = arena.doubleFftWorkspace(N);
            var real0 = arena.doubleRandomVector(N, -2f, 2f, seed);

            var specRe = arena.doubleVec(halfSpec);
            var specIm = arena.doubleVec(halfSpec);
            doubleFFT.rfft(in real0, ref specRe, ref specIm);   // recurrence rfft as spectrum source

            var realR = arena.doubleVec(N);
            doubleFFT.irfft(in specRe, in specIm, ref realR);

            var realT = arena.doubleVec(N);
            doubleFFT.irfft(in specRe, in specIm, ref realT, in ws);

            double tol = (double)1E-4f;
            for (int i = 0; i < N; i++)
                AssertClose(realT[i], realR[i], tol);
            arena.Dispose();
        }

        // Table irfft(ws) == recurrence irfft at N ≤ 256 (tight cross-algorithm comparison, 1E-4).
        // Both inner-M paths covered (same split as TableRfftMatchesRecurrence).
        // Large-N correctness validated by TableRfftRoundTrip (irfft·rfft round-trip).
        void TableIrfftMatchesRecurrence()
        {
            TableIrfftVsRecurrence(2,   4001u);
            TableIrfftVsRecurrence(4,   4005u);
            TableIrfftVsRecurrence(8,   1357u);
            TableIrfftVsRecurrence(16,  1374u);
            TableIrfftVsRecurrence(32,  4009u);
            TableIrfftVsRecurrence(64,  1391u);
            TableIrfftVsRecurrence(128, 4013u);
            TableIrfftVsRecurrence(256, 1408u);
        }

        // ifft(fft(x, ws), ws) == x with a workspace.
        void TableFftRoundTrip()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 64;
            var ws = arena.doubleFftWorkspace(N);

            var re0 = arena.doubleRandomVector(N, -3f, 3f, 6543u);
            var im0 = arena.doubleRandomVector(N, -3f, 3f, 7654u);

            var re = re0.Copy(); var im = im0.Copy();
            doubleFFT.fft(ref re, ref im, in ws);
            doubleFFT.ifft(ref re, ref im, in ws);

            double tol = (double)1E-3f;
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
            var ws = arena.doubleFftWorkspace(N);
            var real0 = arena.doubleRandomVector(N, -3f, 3f, seed);

            var rRe = arena.doubleVec(halfSpec);
            var rIm = arena.doubleVec(halfSpec);
            doubleFFT.rfft(in real0, ref rRe, ref rIm, in ws);

            var real2 = arena.doubleVec(N);
            doubleFFT.irfft(in rRe, in rIm, ref real2, in ws);

            double tol = (double)1E-3f;
            for (int i = 0; i < N; i++)
                AssertClose(real2[i], real0[i], tol);
            arena.Dispose();
        }

        // irfft(rfft(x, ws), ws) == x with a workspace — full size range 2..8192, both inner-M paths.
        void TableRfftRoundTrip()
        {
            TableRfftRoundTripOneSize(2,    5001u);
            TableRfftRoundTripOneSize(4,    5005u);
            TableRfftRoundTripOneSize(8,    8765u);
            TableRfftRoundTripOneSize(16,   8796u);
            TableRfftRoundTripOneSize(32,   5009u);
            TableRfftRoundTripOneSize(64,   8827u);
            TableRfftRoundTripOneSize(128,  5013u);
            TableRfftRoundTripOneSize(256,  8858u);
            TableRfftRoundTripOneSize(512,  5017u);
            TableRfftRoundTripOneSize(1024, 5021u);
            TableRfftRoundTripOneSize(2048, 5025u);
            TableRfftRoundTripOneSize(4096, 5029u);
            TableRfftRoundTripOneSize(8192, 5033u);
        }

        // ---- radix-4 oracle validation tests ----

        // Helper: validate fft(ws) auto-dispatch correctness for one size.
        // fft(ws) dispatches: IsPowerOf4(N) → FftCoreRadix4; 2·4^k → FftCoreRadix4Mixed; else FftCoreTable.
        //
        // Strategy:
        //   N ≤ 256 — cross-algorithm oracle (recurrence fft vs fft(ws)) at relative 1E-3.
        //     The recurrence (cur *= w) accumulates drift O(k · eps_f) per stage inner loop. At N ≤ 256
        //     the last-stage drift reaches ~128 · 1.2e-7 ≈ 1.5e-5 per twiddle — well within 1E-3 relTol.
        //     Covers all auto-dispatch paths: radix-4 (N=4,16,64,256) and mixed (N=2,8,32,128).
        //   N > 256 — round-trip validation (ifft(fft(x,ws),ws) == x) at absolute 1E-3.
        //     At float32 the recurrence last-stage drift reaches O(N/2 · eps_f) ≈ 2e-3 per twiddle at
        //     N=32768; after amplification by intermediate magnitudes O(sqrt(N)), near-zero output bins
        //     accumulate absolute error far exceeding any useful absolute threshold. The round-trip
        //     (forward + inverse errors cancel to machine precision) is the correct large-N correctness
        //     test. Radix4RoundTrip also covers these sizes with different seeds.
        void Radix4VsOracleOneSize(int N, uint seedRe, uint seedIm)
        {
            var arena = new Arena(Allocator.Persistent);
            var ws    = arena.doubleFftWorkspace(N);

            var re0 = arena.doubleRandomVector(N, -2f, 2f, seedRe);
            var im0 = arena.doubleRandomVector(N, -2f, 2f, seedIm);

            if (N <= 256)
            {
                // Cross-algorithm oracle: recurrence fft (per-stage cos/sin) vs auto-dispatch fft(ws).
                var reRef = re0.Copy(); var imRef = im0.Copy();
                doubleFFT.fft(ref reRef, ref imRef);

                var reW = re0.Copy(); var imW = im0.Copy();
                doubleFFT.fft(ref reW, ref imW, in ws);

                double relTol = (double)1E-3f;
                for (int k = 0; k < N; k++)
                {
                    double absTolRe = relTol * math.max((double)1.0f, math.abs(reRef[k]));
                    double absTolIm = relTol * math.max((double)1.0f, math.abs(imRef[k]));
                    AssertClose(reW[k], reRef[k], absTolRe);
                    AssertClose(imW[k], imRef[k], absTolIm);
                }
            }
            else
            {
                // Round-trip: ifft(fft(x,ws),ws) == x — errors cancel, tight 1E-3.
                var re = re0.Copy(); var im = im0.Copy();
                doubleFFT.fft(ref re, ref im, in ws);
                doubleFFT.ifft(ref re, ref im, in ws);

                double tol = (double)1E-3f;
                for (int i = 0; i < N; i++)
                {
                    AssertClose(re[i], re0[i], tol);
                    AssertClose(im[i], im0[i], tol);
                }
            }

            arena.Dispose();
        }

        // Validate fft(ws) auto-dispatch at all sizes in {2,4,8,...,32768}.
        // N ≤ 256: cross-algorithm oracle vs recurrence fft (all dispatch paths exercised).
        // N > 256: round-trip (recurrence oracle not reliable at float32 for large N; see comment above).
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

        // Helper: ifft(fft(x, ws), ws) == x for one size (auto-dispatch round-trip).
        void Radix4RoundTripOneSize(int N, uint seedRe, uint seedIm)
        {
            var arena = new Arena(Allocator.Persistent);
            var ws    = arena.doubleFftWorkspace(N);

            var re0 = arena.doubleRandomVector(N, -3f, 3f, seedRe);
            var im0 = arena.doubleRandomVector(N, -3f, 3f, seedIm);

            var re = re0.Copy(); var im = im0.Copy();
            doubleFFT.fft(ref re, ref im, in ws);
            doubleFFT.ifft(ref re, ref im, in ws);

            double tol = (double)1E-3f;
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

        // ====================================================================================
        // Comprehensive numerical-stability / correctness suite
        // ====================================================================================

        // Relative-tolerance assert: tolerance is relTol scaled by max(1, |reference|). This is the
        // same floor-at-1.0 scheme used by Radix4MatchesOracle — it never hard-asserts near-zero bins
        // (which accumulate twiddle noise at large float N) yet keeps large bins to a relative bound.
        void AssertCloseRel(double got, double reference, double relTol)
        {
            double tol = relTol * math.max((double)1.0f, math.abs(reference));
            AssertClose(got, reference, tol);
        }

        // Total complex energy Σ_i (re[i]² + im[i]²).
        static double Energy(in doubleN re, in doubleN im)
        {
            double s = (double)0;
            for (int i = 0; i < re.N; i++)
                s += re[i] * re[i] + im[i] * im[i];
            return s;
        }

        // ---- 1. fft vs dft cross-check (the anchor) -------------------------------------------
        // dft is the independent O(N²) ground truth. Both the no-workspace recurrence fft and the
        // workspace radix-4/mixed fft must match it bin-by-bin on random complex input. Validates
        // fft (BOTH paths) AND dft simultaneously. Sizes cover power-of-4 (16,64,256) and mixed
        // 2·4^k (8,32) dispatch classes.
        void FftVsDftOneSize(int N, uint seedRe, uint seedIm)
        {
            var arena = new Arena(Allocator.Persistent);
            var ws = arena.doubleFftWorkspace(N);
            var re0 = arena.doubleRandomVector(N, -2f, 2f, seedRe);
            var im0 = arena.doubleRandomVector(N, -2f, 2f, seedIm);

            // Ground truth: direct DFT.
            var dRe = arena.doubleVec(N);
            var dIm = arena.doubleVec(N);
            doubleFFT.dft(in re0, in im0, ref dRe, ref dIm);

            // No-workspace recurrence fft.
            var fRe = re0.Copy(); var fIm = im0.Copy();
            doubleFFT.fft(ref fRe, ref fIm);

            // Workspace radix-4 / mixed fft.
            var wRe = re0.Copy(); var wIm = im0.Copy();
            doubleFFT.fft(ref wRe, ref wIm, in ws);

            double relTol = (double)1E-3f;
            for (int k = 0; k < N; k++)
            {
                AssertCloseRel(fRe[k], dRe[k], relTol);
                AssertCloseRel(fIm[k], dIm[k], relTol);
                AssertCloseRel(wRe[k], dRe[k], relTol);
                AssertCloseRel(wIm[k], dIm[k], relTol);
            }
            arena.Dispose();
        }

        void FftVsDftCrossCheck()
        {
            FftVsDftOneSize(8,   51001u, 51002u);   // 2·4^1 mixed
            FftVsDftOneSize(16,  51003u, 51004u);   // 4^2 radix-4
            FftVsDftOneSize(32,  51005u, 51006u);   // 2·4^2 mixed
            FftVsDftOneSize(64,  51007u, 51008u);   // 4^3 radix-4
            FftVsDftOneSize(256, 51009u, 51010u);   // 4^4 radix-4
        }

        // ---- 2. Parseval / energy conservation ------------------------------------------------
        // Σ|x|² == (1/N)·Σ|X|². Checked as a RELATIVE error on the single total-energy scalar
        // (robust to per-bin twiddle noise). Includes large N (4096, 16384) for fft, dft only to
        // 512 (O(N²)), plus the rfft half-spectrum energy identity.
        void ParsevalFftOneSize(int N, uint seedRe, uint seedIm)
        {
            var arena = new Arena(Allocator.Persistent);
            var ws = arena.doubleFftWorkspace(N);
            var re0 = arena.doubleRandomVector(N, -2f, 2f, seedRe);
            var im0 = arena.doubleRandomVector(N, -2f, 2f, seedIm);

            double timeE = Energy(in re0, in im0);
            double relTol = (double)1E-2f;   // robust scalar-energy bound (float summation at large N)

            // No-workspace fft.
            var fRe = re0.Copy(); var fIm = im0.Copy();
            doubleFFT.fft(ref fRe, ref fIm);
            AssertCloseRel(Energy(in fRe, in fIm) / (double)N, timeE, relTol);

            // Workspace fft.
            var wRe = re0.Copy(); var wIm = im0.Copy();
            doubleFFT.fft(ref wRe, ref wIm, in ws);
            AssertCloseRel(Energy(in wRe, in wIm) / (double)N, timeE, relTol);

            arena.Dispose();
        }

        void ParsevalDftOneSize(int N, uint seedRe, uint seedIm)
        {
            var arena = new Arena(Allocator.Persistent);
            var re0 = arena.doubleRandomVector(N, -2f, 2f, seedRe);
            var im0 = arena.doubleRandomVector(N, -2f, 2f, seedIm);

            double timeE = Energy(in re0, in im0);
            var dRe = arena.doubleVec(N);
            var dIm = arena.doubleVec(N);
            doubleFFT.dft(in re0, in im0, ref dRe, ref dIm);
            AssertCloseRel(Energy(in dRe, in dIm) / (double)N, timeE, (double)5E-3f);
            arena.Dispose();
        }

        // rfft Parseval: real-signal energy Σx² equals (1/N)·full-spectrum energy reconstructed from
        // the half spectrum: |X[0]|² + |X[N/2]|² + 2·Σ_{k=1}^{N/2-1}|X[k]|².
        void ParsevalRfftOneSize(int N, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);
            var ws = arena.doubleFftWorkspace(N);
            int halfSpec = (N >> 1) + 1;
            int M = N >> 1;
            var real = arena.doubleRandomVector(N, -2f, 2f, seed);

            double timeE = (double)0;
            for (int i = 0; i < N; i++) timeE += real[i] * real[i];

            var rRe = arena.doubleVec(halfSpec);
            var rIm = arena.doubleVec(halfSpec);
            doubleFFT.rfft(in real, ref rRe, ref rIm, in ws);

            double specE = rRe[0] * rRe[0] + rIm[0] * rIm[0]
                         + rRe[M] * rRe[M] + rIm[M] * rIm[M];
            for (int k = 1; k < M; k++)
                specE += (double)2 * (rRe[k] * rRe[k] + rIm[k] * rIm[k]);

            AssertCloseRel(specE / (double)N, timeE, (double)1E-2f);
            arena.Dispose();
        }

        void ParsevalEnergy()
        {
            ParsevalFftOneSize(8,     52001u, 52002u);
            ParsevalFftOneSize(64,    52003u, 52004u);
            ParsevalFftOneSize(256,   52005u, 52006u);
            ParsevalFftOneSize(4096,  52007u, 52008u);
            ParsevalFftOneSize(16384, 52009u, 52010u);

            ParsevalDftOneSize(7,   52021u, 52022u);   // non-power-of-two
            ParsevalDftOneSize(64,  52023u, 52024u);
            ParsevalDftOneSize(512, 52025u, 52026u);

            ParsevalRfftOneSize(8,     52031u);
            ParsevalRfftOneSize(64,    52032u);
            ParsevalRfftOneSize(256,   52033u);
            ParsevalRfftOneSize(4096,  52034u);
            ParsevalRfftOneSize(16384, 52035u);
        }

        // ---- 3. Linearity ---------------------------------------------------------------------
        // fft(a·x + b·y) == a·fft(x) + b·fft(y) for random complex scalars a,b and signals x,y.
        // Validated for fft(no-ws), fft(ws) and dft. Per-bin relative tolerance.
        void FftLinearityOneSize(int N, uint sx, uint sy)
        {
            var arena = new Arena(Allocator.Persistent);
            var ws = arena.doubleFftWorkspace(N);

            var xr = arena.doubleRandomVector(N, -2f, 2f, sx);
            var xi = arena.doubleRandomVector(N, -2f, 2f, sx + 17u);
            var yr = arena.doubleRandomVector(N, -2f, 2f, sy);
            var yi = arena.doubleRandomVector(N, -2f, 2f, sy + 17u);

            double aRe = (double)1.5f, aIm = (double)(-0.5f);
            double bRe = (double)(-2.0f), bIm = (double)0.75f;

            // z = a·x + b·y (complex, per sample).
            var zr = arena.doubleVec(N);
            var zi = arena.doubleVec(N);
            for (int n = 0; n < N; n++)
            {
                double axr = aRe * xr[n] - aIm * xi[n];
                double axi = aRe * xi[n] + aIm * xr[n];
                double byr = bRe * yr[n] - bIm * yi[n];
                double byi = bRe * yi[n] + bIm * yr[n];
                zr[n] = axr + byr;
                zi[n] = axi + byi;
            }

            // Forward transforms of x and y for the RHS combination (dft ground truth).
            var Xr = arena.doubleVec(N); var Xi = arena.doubleVec(N);
            var Yr = arena.doubleVec(N); var Yi = arena.doubleVec(N);
            doubleFFT.dft(in xr, in xi, ref Xr, ref Xi);
            doubleFFT.dft(in yr, in yi, ref Yr, ref Yi);

            // LHS via three transforms.
            var Zno_r = zr.Copy(); var Zno_i = zi.Copy(); doubleFFT.fft(ref Zno_r, ref Zno_i);
            var Zws_r = zr.Copy(); var Zws_i = zi.Copy(); doubleFFT.fft(ref Zws_r, ref Zws_i, in ws);
            var Zdf_r = arena.doubleVec(N); var Zdf_i = arena.doubleVec(N);
            doubleFFT.dft(in zr, in zi, ref Zdf_r, ref Zdf_i);

            double relTol = (double)1E-3f;
            for (int k = 0; k < N; k++)
            {
                double rhsRe = (aRe * Xr[k] - aIm * Xi[k]) + (bRe * Yr[k] - bIm * Yi[k]);
                double rhsIm = (aRe * Xi[k] + aIm * Xr[k]) + (bRe * Yi[k] + bIm * Yr[k]);

                AssertCloseRel(Zno_r[k], rhsRe, relTol);
                AssertCloseRel(Zno_i[k], rhsIm, relTol);
                AssertCloseRel(Zws_r[k], rhsRe, relTol);
                AssertCloseRel(Zws_i[k], rhsIm, relTol);
                AssertCloseRel(Zdf_r[k], rhsRe, relTol);
                AssertCloseRel(Zdf_i[k], rhsIm, relTol);
            }
            arena.Dispose();
        }

        void FftLinearity()
        {
            FftLinearityOneSize(16, 53001u, 53101u);
            FftLinearityOneSize(64, 53003u, 53103u);
        }

        // ---- 4. Known analytic transforms -----------------------------------------------------
        // Anchored to closed-form spectra, independently for fft(ws) and dft. Sign convention:
        // X[k] = Σ x[n]·exp(-2πi·kn/N).
        //   impulse δ[0]        -> X[k] = 1 (flat)
        //   shifted impulse δ[m]-> X[k] = exp(-2πi·km/N) = cos(2πkm/N) - i·sin(2πkm/N)
        //   constant c          -> X[0] = c·N, else 0
        //   exp(+2πi·k0·n/N)    -> X[k0] = N, else 0
        // Runs the input through BOTH fft(ws) and dft and compares each to the analytic expectation.
        void KnownRunBoth(in doubleFftWorkspace ws, int N,
                          in doubleN inRe, in doubleN inIm,
                          in doubleN expRe, in doubleN expIm, double relTol)
        {
            var fr = inRe.Copy(); var fi = inIm.Copy();
            doubleFFT.fft(ref fr, ref fi, in ws);
            for (int k = 0; k < N; k++)
            {
                AssertCloseRel(fr[k], expRe[k], relTol);
                AssertCloseRel(fi[k], expIm[k], relTol);
            }
        }

        void KnownAnalytics()
        {
            var arena = new Arena(Allocator.Persistent);
            int N = 16;
            var ws = arena.doubleFftWorkspace(N);
            double twoPi = (double)(2.0 * System.Math.PI);
            double relTol = (double)2E-3f;

            // --- impulse δ[0] -> flat spectrum (all ones) ---
            {
                var inRe = arena.doubleVec(N); var inIm = arena.doubleVec(N);
                inRe[0] = (double)1;
                var expRe = arena.doubleVec(N, 1f); var expIm = arena.doubleVec(N);

                var dRe = arena.doubleVec(N); var dIm = arena.doubleVec(N);
                doubleFFT.dft(in inRe, in inIm, ref dRe, ref dIm);
                for (int k = 0; k < N; k++)
                {
                    AssertCloseRel(dRe[k], expRe[k], relTol);
                    AssertCloseRel(dIm[k], expIm[k], relTol);
                }
                KnownRunBoth(in ws, N, in inRe, in inIm, in expRe, in expIm, relTol);
            }

            // --- shifted impulse δ[m], m=3 -> X[k] = cos(2πkm/N) - i·sin(2πkm/N) ---
            {
                int m = 3;
                var inRe = arena.doubleVec(N); var inIm = arena.doubleVec(N);
                inRe[m] = (double)1;
                var expRe = arena.doubleVec(N); var expIm = arena.doubleVec(N);
                for (int k = 0; k < N; k++)
                {
                    double ang = twoPi * (double)(k * m) / (double)N;
                    expRe[k] = math.cos(ang);
                    expIm[k] = -math.sin(ang);
                }
                var dRe = arena.doubleVec(N); var dIm = arena.doubleVec(N);
                doubleFFT.dft(in inRe, in inIm, ref dRe, ref dIm);
                for (int k = 0; k < N; k++)
                {
                    AssertCloseRel(dRe[k], expRe[k], relTol);
                    AssertCloseRel(dIm[k], expIm[k], relTol);
                }
                KnownRunBoth(in ws, N, in inRe, in inIm, in expRe, in expIm, relTol);
            }

            // --- constant c -> DC spike X[0] = c·N ---
            {
                double c = (double)2.5f;
                var inRe = arena.doubleVec(N, 2.5f); var inIm = arena.doubleVec(N);
                var expRe = arena.doubleVec(N); var expIm = arena.doubleVec(N);
                expRe[0] = c * (double)N;

                var dRe = arena.doubleVec(N); var dIm = arena.doubleVec(N);
                doubleFFT.dft(in inRe, in inIm, ref dRe, ref dIm);
                for (int k = 0; k < N; k++)
                {
                    AssertCloseRel(dRe[k], expRe[k], relTol);
                    AssertCloseRel(dIm[k], expIm[k], relTol);
                }
                KnownRunBoth(in ws, N, in inRe, in inIm, in expRe, in expIm, relTol);
            }

            // --- pure exponential exp(+2πi·k0·n/N), k0=3 -> single bin X[k0] = N ---
            {
                int k0 = 3;
                var inRe = arena.doubleVec(N); var inIm = arena.doubleVec(N);
                double w = twoPi * (double)k0 / (double)N;
                for (int n = 0; n < N; n++)
                {
                    inRe[n] = math.cos(w * (double)n);
                    inIm[n] = math.sin(w * (double)n);
                }
                var expRe = arena.doubleVec(N); var expIm = arena.doubleVec(N);
                expRe[k0] = (double)N;

                var dRe = arena.doubleVec(N); var dIm = arena.doubleVec(N);
                doubleFFT.dft(in inRe, in inIm, ref dRe, ref dIm);
                for (int k = 0; k < N; k++)
                {
                    AssertCloseRel(dRe[k], expRe[k], relTol);
                    AssertCloseRel(dIm[k], expIm[k], relTol);
                }
                KnownRunBoth(in ws, N, in inRe, in inIm, in expRe, in expIm, relTol);
            }

            arena.Dispose();
        }

        // ---- 6. Round-trip accuracy at large N ------------------------------------------------
        // ifft(fft(x,ws),ws)==x and irfft(rfft(x,ws),ws)==x up to N=16384; idft(dft(x))==x to ~512
        // (incl. non-power-of-two). Forward+inverse errors cancel, so a tight absolute/relative
        // bound holds (float ~1e-3, double far tighter).
        void RoundTripFftWsOneSize(int N, uint seedRe, uint seedIm)
        {
            var arena = new Arena(Allocator.Persistent);
            var ws = arena.doubleFftWorkspace(N);
            var re0 = arena.doubleRandomVector(N, -3f, 3f, seedRe);
            var im0 = arena.doubleRandomVector(N, -3f, 3f, seedIm);

            var re = re0.Copy(); var im = im0.Copy();
            doubleFFT.fft(ref re, ref im, in ws);
            doubleFFT.ifft(ref re, ref im, in ws);

            double tol = (double)1E-3f;
            for (int i = 0; i < N; i++)
            {
                AssertClose(re[i], re0[i], tol);
                AssertClose(im[i], im0[i], tol);
            }
            arena.Dispose();
        }

        void RoundTripRfftWsOneSize(int N, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);
            var ws = arena.doubleFftWorkspace(N);
            int halfSpec = (N >> 1) + 1;
            var real0 = arena.doubleRandomVector(N, -3f, 3f, seed);

            var rRe = arena.doubleVec(halfSpec);
            var rIm = arena.doubleVec(halfSpec);
            doubleFFT.rfft(in real0, ref rRe, ref rIm, in ws);

            var real2 = arena.doubleVec(N);
            doubleFFT.irfft(in rRe, in rIm, ref real2, in ws);

            double tol = (double)1E-3f;
            for (int i = 0; i < N; i++)
                AssertClose(real2[i], real0[i], tol);
            arena.Dispose();
        }

        void RoundTripDftOneSize(int N, uint seedRe, uint seedIm)
        {
            var arena = new Arena(Allocator.Persistent);
            var re0 = arena.doubleRandomVector(N, -3f, 3f, seedRe);
            var im0 = arena.doubleRandomVector(N, -3f, 3f, seedIm);

            var fRe = arena.doubleVec(N); var fIm = arena.doubleVec(N);
            doubleFFT.dft(in re0, in im0, ref fRe, ref fIm);
            var bRe = arena.doubleVec(N); var bIm = arena.doubleVec(N);
            doubleFFT.idft(in fRe, in fIm, ref bRe, ref bIm);

            double relTol = (double)5E-3f;
            for (int i = 0; i < N; i++)
            {
                AssertCloseRel(bRe[i], re0[i], relTol);
                AssertCloseRel(bIm[i], im0[i], relTol);
            }
            arena.Dispose();
        }

        void RoundTripLargeN()
        {
            RoundTripFftWsOneSize(1024,  54001u, 54002u);
            RoundTripFftWsOneSize(8192,  54003u, 54004u);   // 2·4^6 mixed
            RoundTripFftWsOneSize(16384, 54005u, 54006u);   // 4^7 radix-4

            RoundTripRfftWsOneSize(1024,  54011u);
            RoundTripRfftWsOneSize(16384, 54012u);

            RoundTripDftOneSize(257, 54021u, 54022u);   // non-power-of-two
            RoundTripDftOneSize(512, 54023u, 54024u);
        }

        // ---- 7. Zero-alloc workspace reuse ----------------------------------------------------
        // Drive fft(ws) -> rfft(ws) -> ifft(ws) on the SAME workspace and confirm each result is
        // still correct. Guards the shared cz/sz/visited scratch against cross-call corruption
        // (the recent zero-alloc change). Covers power-of-4 and mixed sizes (and their inner-M dual).
        void WorkspaceReuseOneSize(int N, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);
            var ws = arena.doubleFftWorkspace(N);
            int halfSpec = (N >> 1) + 1;

            var re0 = arena.doubleRandomVector(N, -2f, 2f, seed);
            var im0 = arena.doubleRandomVector(N, -2f, 2f, seed + 1u);

            // (a) fft(ws) on the fresh workspace, validated against dft.
            var dRe = arena.doubleVec(N); var dIm = arena.doubleVec(N);
            doubleFFT.dft(in re0, in im0, ref dRe, ref dIm);
            var fRe = re0.Copy(); var fIm = im0.Copy();
            doubleFFT.fft(ref fRe, ref fIm, in ws);
            double relTol = (double)1E-3f;
            for (int k = 0; k < N; k++)
            {
                AssertCloseRel(fRe[k], dRe[k], relTol);
                AssertCloseRel(fIm[k], dIm[k], relTol);
            }

            // (b) rfft(ws) on the SAME workspace (touches cz/sz/visited) — compare to no-ws rfft.
            var real = arena.doubleRandomVector(N, -2f, 2f, seed + 2u);
            var rRe = arena.doubleVec(halfSpec); var rIm = arena.doubleVec(halfSpec);
            doubleFFT.rfft(in real, ref rRe, ref rIm, in ws);
            var oRe = arena.doubleVec(halfSpec); var oIm = arena.doubleVec(halfSpec);
            doubleFFT.rfft(in real, ref oRe, ref oIm);   // no-ws oracle
            for (int k = 0; k <= N / 2; k++)
            {
                AssertCloseRel(rRe[k], oRe[k], relTol);
                AssertCloseRel(rIm[k], oIm[k], relTol);
            }

            // (c) ifft(ws) on the SAME workspace, inverting the step-(a) spectrum back to re0/im0.
            // If rfft had corrupted the shared scratch this round-trip would fail.
            doubleFFT.ifft(ref fRe, ref fIm, in ws);
            for (int i = 0; i < N; i++)
            {
                AssertClose(fRe[i], re0[i], (double)1E-3f);
                AssertClose(fIm[i], im0[i], (double)1E-3f);
            }

            arena.Dispose();
        }

        void WorkspaceReuse()
        {
            WorkspaceReuseOneSize(64,  55001u);   // 4^3 radix-4 (inner M=32 mixed for rfft)
            WorkspaceReuseOneSize(128, 55003u);   // 2·4^3 mixed (inner M=64 radix-4 for rfft)
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
    // comprehensive numerical-stability / correctness suite
    [Test] public void FftVsDftCrossCheckTest() => RunJob(TestJob.TestType.FftVsDftCrossCheck);
    [Test] public void ParsevalEnergyTest() => RunJob(TestJob.TestType.ParsevalEnergy);
    [Test] public void FftLinearityTest() => RunJob(TestJob.TestType.FftLinearity);
    [Test] public void KnownAnalyticsTest() => RunJob(TestJob.TestType.KnownAnalytics);
    [Test] public void RoundTripLargeNTest() => RunJob(TestJob.TestType.RoundTripLargeN);
    [Test] public void WorkspaceReuseTest() => RunJob(TestJob.TestType.WorkspaceReuse);

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

    // ---- workspace guard tests ----

    [Test]
    public void FftWorkspaceFactoryNonPow2Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        Assert.Throws<ArgumentException>(() => arena.doubleFftWorkspace(0));
        Assert.Throws<ArgumentException>(() => arena.doubleFftWorkspace(1));
        Assert.Throws<ArgumentException>(() => arena.doubleFftWorkspace(3));
        Assert.Throws<ArgumentException>(() => arena.doubleFftWorkspace(5));
        Assert.Throws<ArgumentException>(() => arena.doubleFftWorkspace(6));
        arena.Dispose();
    }

    [Test]
    public void FftTableWrongWorkspaceSizeThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var re8  = arena.doubleVec(8);
        var im8  = arena.doubleVec(8);
        var ws16 = arena.doubleFftWorkspace(16);   // sized for 16, not 8

        // fft and ifft with mismatched workspace
        Assert.Throws<ArgumentException>(() => doubleFFT.fft(ref re8, ref im8, in ws16));
        Assert.Throws<ArgumentException>(() => doubleFFT.ifft(ref re8, ref im8, in ws16));

        // rfft: real.N=8 but ws.n=16
        var real8  = arena.doubleVec(8);
        var reHalf = arena.doubleVec(5);   // 8/2+1=5
        var imHalf = arena.doubleVec(5);
        Assert.Throws<ArgumentException>(() => doubleFFT.rfft(in real8, ref reHalf, ref imHalf, in ws16));

        // irfft: re.N=5 -> N=8, but ws.n=16
        Assert.Throws<ArgumentException>(() => doubleFFT.irfft(in reHalf, in imHalf, ref real8, in ws16));

        arena.Dispose();
    }

    [Test]
    public void RfftTableWrongOutputLengthThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var ws = arena.doubleFftWorkspace(8);
        var real = arena.doubleVec(8);
        var re5  = arena.doubleVec(5);    // correct N/2+1
        var im5  = arena.doubleVec(5);    // correct
        var re4  = arena.doubleVec(4);    // wrong
        var im4  = arena.doubleVec(4);    // wrong

        Assert.Throws<ArgumentException>(() => doubleFFT.rfft(in real, ref re4, ref im5, in ws));
        Assert.Throws<ArgumentException>(() => doubleFFT.rfft(in real, ref re5, ref im4, in ws));
        arena.Dispose();
    }
}
