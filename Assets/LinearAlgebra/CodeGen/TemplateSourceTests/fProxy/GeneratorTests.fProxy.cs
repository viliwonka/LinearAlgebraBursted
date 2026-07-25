using System;

using BULA;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;

public class fProxyGeneratorTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            LinspaceArange,
            SampleEqualsManual,
            EasingKnownValues,
            EasingOutMirrorsIn,
            GaussianKernel,
            BoxTentKernel,
            Windows,
            OuterAndOuterSum,
            GaussianKernel2D,
            Waves,
            ArenaMatchesRef
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected, [3] diff
        public NativeArray<fProxy> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.LinspaceArange: LinspaceArange(); break;
                case TestType.SampleEqualsManual: SampleEqualsManual(); break;
                case TestType.EasingKnownValues: EasingKnownValues(); break;
                case TestType.EasingOutMirrorsIn: EasingOutMirrorsIn(); break;
                case TestType.GaussianKernel: GaussianKernel(); break;
                case TestType.BoxTentKernel: BoxTentKernel(); break;
                case TestType.Windows: Windows(); break;
                case TestType.OuterAndOuterSum: OuterAndOuterSum(); break;
                case TestType.GaussianKernel2D: GaussianKernel2D(); break;
                case TestType.Waves: Waves(); break;
                case TestType.ArenaMatchesRef: ArenaMatchesRef(); break;
            }
        }

        // linspace(0,1,5) == {0,.25,.5,.75,1}; linspace(a,b,1) == {a}; arange(2,3,4) == {2,5,8,11}.
        void LinspaceArange()
        {
            var v = new fProxyN(5, Allocator.Temp);
            Generate.linspace(ref v, (fProxy)0, (fProxy)1);
            AssertClose(v[0], (fProxy)0f, 1E-5f);
            AssertClose(v[1], (fProxy)0.25f, 1E-5f);
            AssertClose(v[2], (fProxy)0.5f, 1E-5f);
            AssertClose(v[3], (fProxy)0.75f, 1E-5f);
            AssertClose(v[4], (fProxy)1f, 1E-5f);

            // endpoints land EXACTLY (no accumulated lerp error)
            AssertClose(v[0], (fProxy)0f, 0f);
            AssertClose(v[4], (fProxy)1f, 0f);

            var one = new fProxyN(1, Allocator.Temp);
            Generate.linspace(ref one, (fProxy)7, (fProxy)9);
            AssertClose(one[0], (fProxy)7f, 0f);

            var r = new fProxyN(4, Allocator.Temp);
            Generate.arange(ref r, (fProxy)2, (fProxy)3);
            AssertClose(r[0], (fProxy)2f, 1E-5f);
            AssertClose(r[1], (fProxy)5f, 1E-5f);
            AssertClose(r[2], (fProxy)8f, 1E-5f);
            AssertClose(r[3], (fProxy)11f, 1E-5f);
        }

        // sample over [0,1] of EaseInQuad equals the manual loop value (i/(N-1))².
        void SampleEqualsManual()
        {
            int N = 9;
            var f = new fProxyEasing.EaseInQuad();
            var dest = new fProxyN(N, Allocator.Temp);
            Generate.sample(ref f, ref dest);

            fProxy scale = (fProxy)1 / (fProxy)(N - 1);
            for (int i = 0; i < N; i++)
            {
                fProxy t = i * scale;
                AssertClose(dest[i], f.Eval(t), 1E-5f);
                AssertClose(dest[i], t * t, 1E-5f);
            }

            // N==1 -> {f.Eval(t0)}
            var one = new fProxyN(1, Allocator.Temp);
            Generate.sample(ref f, ref one, (fProxy)0.5, (fProxy)0.9);
            AssertClose(one[0], f.Eval((fProxy)0.5), 1E-6f);

            // explicit domain [2,4] hits the endpoints exactly
            var dom = new fProxyN(3, Allocator.Temp);
            var lin = new fProxyEasing.Linear();
            Generate.sample(ref lin, ref dom, (fProxy)2, (fProxy)4);
            AssertClose(dom[0], (fProxy)2f, 1E-5f);
            AssertClose(dom[1], (fProxy)3f, 1E-5f);
            AssertClose(dom[2], (fProxy)4f, 1E-5f);
        }

        // Endpoints and a few interior known values for the core easings.
        void EasingKnownValues()
        {
            // All standard easings pin (0)->0 and (1)->1.
            AssertClose(new fProxyEasing.Linear().Eval((fProxy)0), (fProxy)0f, 1E-6f);
            AssertClose(new fProxyEasing.Linear().Eval((fProxy)1), (fProxy)1f, 1E-6f);
            AssertClose(new fProxyEasing.Linear().Eval((fProxy)0.37), (fProxy)0.37f, 1E-6f);

            AssertClose(new fProxyEasing.SmoothStep().Eval((fProxy)0), (fProxy)0f, 1E-6f);
            AssertClose(new fProxyEasing.SmoothStep().Eval((fProxy)1), (fProxy)1f, 1E-6f);
            AssertClose(new fProxyEasing.SmoothStep().Eval((fProxy)0.5), (fProxy)0.5f, 1E-6f);

            AssertClose(new fProxyEasing.SmootherStep().Eval((fProxy)0), (fProxy)0f, 1E-6f);
            AssertClose(new fProxyEasing.SmootherStep().Eval((fProxy)1), (fProxy)1f, 1E-6f);
            AssertClose(new fProxyEasing.SmootherStep().Eval((fProxy)0.5), (fProxy)0.5f, 1E-6f);

            AssertClose(new fProxyEasing.EaseInQuad().Eval((fProxy)0.5), (fProxy)0.25f, 1E-6f);
            AssertClose(new fProxyEasing.EaseOutQuad().Eval((fProxy)0.5), (fProxy)0.75f, 1E-6f);
            AssertClose(new fProxyEasing.EaseInCubic().Eval((fProxy)0.5), (fProxy)0.125f, 1E-6f);

            // Quart family (was entirely untested)
            AssertClose(new fProxyEasing.EaseInQuart().Eval((fProxy)0.5), (fProxy)0.0625f, 1E-6f);
            AssertClose(new fProxyEasing.EaseOutQuart().Eval((fProxy)0.5), (fProxy)0.9375f, 1E-6f);

            // InOut variants — interior values straddling the t=0.5 branch seam (the most bug-prone spot).
            // InOutQuad: 2t² for t<.5, 1-(-2t+2)²/2 else.
            AssertClose(new fProxyEasing.EaseInOutQuad().Eval((fProxy)0.25), (fProxy)0.125f, 1E-6f);
            AssertClose(new fProxyEasing.EaseInOutQuad().Eval((fProxy)0.5), (fProxy)0.5f, 1E-6f);
            AssertClose(new fProxyEasing.EaseInOutQuad().Eval((fProxy)0.75), (fProxy)0.875f, 1E-6f);
            // InOutCubic: 4t³ / 1-(-2t+2)³/2.
            AssertClose(new fProxyEasing.EaseInOutCubic().Eval((fProxy)0.25), (fProxy)0.0625f, 1E-6f);
            AssertClose(new fProxyEasing.EaseInOutCubic().Eval((fProxy)0.75), (fProxy)0.9375f, 1E-6f);
            // InOutQuart: 8t⁴ / 1-(-2t+2)⁴/2.
            AssertClose(new fProxyEasing.EaseInOutQuart().Eval((fProxy)0.25), (fProxy)0.03125f, 1E-6f);
            AssertClose(new fProxyEasing.EaseInOutQuart().Eval((fProxy)0.75), (fProxy)0.96875f, 1E-6f);

            // Sine variants
            AssertClose(new fProxyEasing.EaseInSine().Eval((fProxy)0), (fProxy)0f, 1E-6f);
            AssertClose(new fProxyEasing.EaseInSine().Eval((fProxy)1), (fProxy)1f, 1E-5f);
            AssertClose(new fProxyEasing.EaseOutSine().Eval((fProxy)0), (fProxy)0f, 1E-5f);
            AssertClose(new fProxyEasing.EaseOutSine().Eval((fProxy)1), (fProxy)1f, 1E-5f);
            AssertClose(new fProxyEasing.EaseInOutSine().Eval((fProxy)0.5), (fProxy)0.5f, 1E-5f);

            // Expo clamps exactly at the ends.
            AssertClose(new fProxyEasing.EaseInExpo().Eval((fProxy)0), (fProxy)0f, 1E-6f);
            AssertClose(new fProxyEasing.EaseOutExpo().Eval((fProxy)1), (fProxy)1f, 1E-6f);
            AssertClose(new fProxyEasing.EaseInOutExpo().Eval((fProxy)0), (fProxy)0f, 1E-6f);
            AssertClose(new fProxyEasing.EaseInOutExpo().Eval((fProxy)1), (fProxy)1f, 1E-6f);
            AssertClose(new fProxyEasing.EaseInOutExpo().Eval((fProxy)0.5), (fProxy)0.5f, 1E-6f);

            // Bounce/Elastic/Back pin the ends (overshoot only in the interior). Endpoint guards mean
            // the *interior* values are what actually exercise the formulas — so check those too.
            AssertClose(new fProxyEasing.EaseOutBounce().Eval((fProxy)0), (fProxy)0f, 1E-6f);
            AssertClose(new fProxyEasing.EaseOutBounce().Eval((fProxy)1), (fProxy)1f, 1E-5f);
            // t=0.5 lands in the 2nd bounce piece: 7.5625*(0.5-1.5/2.75)²+0.75 = 0.765625.
            AssertClose(new fProxyEasing.EaseOutBounce().Eval((fProxy)0.5), (fProxy)0.765625f, 1E-4f);

            AssertClose(new fProxyEasing.EaseInElastic().Eval((fProxy)0), (fProxy)0f, 1E-6f);
            AssertClose(new fProxyEasing.EaseInElastic().Eval((fProxy)1), (fProxy)1f, 1E-6f);
            // t=0.5: -2^-5 * sin(-5.75·2π/3) = -(1/32)·sin(π/6) = -0.015625 (verifies c4 and the 10.75).
            AssertClose(new fProxyEasing.EaseInElastic().Eval((fProxy)0.5), (fProxy)(-0.015625f), 1E-4f);
            AssertClose(new fProxyEasing.EaseOutElastic().Eval((fProxy)0), (fProxy)0f, 1E-6f);
            AssertClose(new fProxyEasing.EaseOutElastic().Eval((fProxy)1), (fProxy)1f, 1E-6f);
            // t=0.5: 2^-5 · sin(4.25·2π/3) + 1 = (1/32)·sin(5π/6) + 1 = 1.015625.
            AssertClose(new fProxyEasing.EaseOutElastic().Eval((fProxy)0.5), (fProxy)1.015625f, 1E-4f);

            AssertClose(new fProxyEasing.EaseInBack().Eval((fProxy)0), (fProxy)0f, 1E-6f);
            AssertClose(new fProxyEasing.EaseInBack().Eval((fProxy)1), (fProxy)1f, 1E-5f);
            // t=0.5: c3·0.125 - c1·0.25 with c1=1.70158, c3=2.70158 → -0.0876975.
            AssertClose(new fProxyEasing.EaseInBack().Eval((fProxy)0.5), (fProxy)(-0.0876975f), 1E-4f);
            AssertClose(new fProxyEasing.EaseOutBack().Eval((fProxy)0), (fProxy)0f, 1E-5f);
            AssertClose(new fProxyEasing.EaseOutBack().Eval((fProxy)1), (fProxy)1f, 1E-5f);
            // t=0.5 mirrors EaseInBack: 1 + c3·(-0.5)³ + c1·(-0.5)² → 1.0876975.
            AssertClose(new fProxyEasing.EaseOutBack().Eval((fProxy)0.5), (fProxy)1.0876975f, 1E-4f);

            // ---- the four family-completing variants (In/InOut Bounce, InOut Elastic, InOut Back) ----
            // InBounce(t) = 1 - OutBounce(1-t): endpoints pin, and (0.5) = 1 - 0.765625 = 0.234375.
            AssertClose(new fProxyEasing.EaseInBounce().Eval((fProxy)0), (fProxy)0f, 1E-5f);
            AssertClose(new fProxyEasing.EaseInBounce().Eval((fProxy)1), (fProxy)1f, 1E-5f);
            AssertClose(new fProxyEasing.EaseInBounce().Eval((fProxy)0.5), (fProxy)0.234375f, 1E-4f);
            // All InOut variants pass through (0.5, 0.5) and pin the ends.
            AssertClose(new fProxyEasing.EaseInOutBounce().Eval((fProxy)0), (fProxy)0f, 1E-5f);
            AssertClose(new fProxyEasing.EaseInOutBounce().Eval((fProxy)1), (fProxy)1f, 1E-5f);
            AssertClose(new fProxyEasing.EaseInOutBounce().Eval((fProxy)0.5), (fProxy)0.5f, 1E-5f);
            AssertClose(new fProxyEasing.EaseInOutElastic().Eval((fProxy)0), (fProxy)0f, 1E-6f);
            AssertClose(new fProxyEasing.EaseInOutElastic().Eval((fProxy)1), (fProxy)1f, 1E-6f);
            AssertClose(new fProxyEasing.EaseInOutElastic().Eval((fProxy)0.5), (fProxy)0.5f, 1E-5f);
            AssertClose(new fProxyEasing.EaseInOutBack().Eval((fProxy)0), (fProxy)0f, 1E-6f);
            AssertClose(new fProxyEasing.EaseInOutBack().Eval((fProxy)1), (fProxy)1f, 1E-5f);
            AssertClose(new fProxyEasing.EaseInOutBack().Eval((fProxy)0.5), (fProxy)0.5f, 1E-5f);
        }

        // The Out variant is the time/value-mirrored In variant: easeOut(t) == 1 - easeIn(1-t).
        void EasingOutMirrorsIn()
        {
            var inQ = new fProxyEasing.EaseInQuad();
            var outQ = new fProxyEasing.EaseOutQuad();
            var inC = new fProxyEasing.EaseInCubic();
            var outC = new fProxyEasing.EaseOutCubic();
            var inK = new fProxyEasing.EaseInQuart();
            var outK = new fProxyEasing.EaseOutQuart();
            var inS = new fProxyEasing.EaseInSine();
            var outS = new fProxyEasing.EaseOutSine();

            for (int k = 0; k <= 10; k++)
            {
                fProxy t = (fProxy)k / (fProxy)10;
                AssertClose(outQ.Eval(t), (fProxy)1 - inQ.Eval((fProxy)1 - t), 1E-5f);
                AssertClose(outC.Eval(t), (fProxy)1 - inC.Eval((fProxy)1 - t), 1E-5f);
                AssertClose(outK.Eval(t), (fProxy)1 - inK.Eval((fProxy)1 - t), 1E-5f);
                AssertClose(outS.Eval(t), (fProxy)1 - inS.Eval((fProxy)1 - t), 1E-5f);
            }
        }

        // 1D Gaussian: sums to 1, symmetric, peak at center.
        void GaussianKernel()
        {
            int N = 5;
            var g = new fProxyN(N, Allocator.Temp);
            Generate.gaussianKernel(ref g, (fProxy)1);

            fProxy sum = (fProxy)0;
            for (int i = 0; i < N; i++) sum += g[i];
            AssertClose(sum, (fProxy)1f, 1E-5f);

            // symmetric
            for (int i = 0; i < N; i++)
                AssertClose(g[i], g[N - 1 - i], 1E-5f);

            // center is the strict max
            AssertTrue(g[2] > g[1]);
            AssertTrue(g[1] > g[0]);

            // N==1 -> {1}
            var one = new fProxyN(1, Allocator.Temp);
            Generate.gaussianKernel(ref one, (fProxy)2);
            AssertClose(one[0], (fProxy)1f, 1E-6f);
        }

        // Box: every weight 1/N (sum 1). Tent: symmetric, sums to 1, peak at center.
        void BoxTentKernel()
        {
            int N = 6;
            var box = new fProxyN(N, Allocator.Temp);
            Generate.boxKernel(ref box);
            fProxy bsum = (fProxy)0;
            for (int i = 0; i < N; i++)
            {
                AssertClose(box[i], (fProxy)1 / (fProxy)N, 1E-6f);
                bsum += box[i];
            }
            AssertClose(bsum, (fProxy)1f, 1E-5f);

            int M = 5;
            var tent = new fProxyN(M, Allocator.Temp);
            Generate.tentKernel(ref tent);
            fProxy tsum = (fProxy)0;
            for (int i = 0; i < M; i++) tsum += tent[i];
            AssertClose(tsum, (fProxy)1f, 1E-5f);
            for (int i = 0; i < M; i++)
                AssertClose(tent[i], tent[M - 1 - i], 1E-5f);
            AssertTrue(tent[2] > tent[1]);
            AssertTrue(tent[1] > tent[0]);
        }

        // Hann/Hamming/Blackman/Box known endpoint and center values (N=5, denom 4).
        void Windows()
        {
            int N = 5;

            var hann = new fProxyN(N, Allocator.Temp);
            Generate.window(ref hann, WindowType.Hann);
            AssertClose(hann[0], (fProxy)0f, 1E-5f);
            AssertClose(hann[4], (fProxy)0f, 1E-5f);
            AssertClose(hann[2], (fProxy)1f, 1E-5f); // 0.5(1-cos π) = 1
            AssertClose(hann[1], hann[3], 1E-5f);    // symmetric

            var hamming = new fProxyN(N, Allocator.Temp);
            Generate.window(ref hamming, WindowType.Hamming);
            AssertClose(hamming[0], (fProxy)0.08f, 1E-5f);
            AssertClose(hamming[4], (fProxy)0.08f, 1E-5f);
            AssertClose(hamming[2], (fProxy)1f, 1E-5f); // 0.54+0.46 = 1

            var black = new fProxyN(N, Allocator.Temp);
            Generate.window(ref black, WindowType.Blackman);
            AssertClose(black[0], (fProxy)0f, 1E-5f);  // 0.42-0.5+0.08 = 0
            AssertClose(black[4], (fProxy)0f, 1E-5f);
            AssertClose(black[2], (fProxy)1f, 1E-5f);  // 0.42+0.5+0.08 = 1 (center)
            AssertClose(black[1], black[3], 1E-5f);    // symmetric

            var box = new fProxyN(N, Allocator.Temp);
            Generate.window(ref box, WindowType.Box);
            for (int i = 0; i < N; i++)
                AssertClose(box[i], (fProxy)1f, 1E-6f);

            // N==1 -> {1} for the (N-1)-denominator windows (no div-by-zero)
            var one = new fProxyN(1, Allocator.Temp);
            Generate.window(ref one, WindowType.Hann);
            AssertClose(one[0], (fProxy)1f, 1E-6f);
        }

        // outer[i,j] == u[i]*v[j]; outerSum[i,j] == u[i]+v[j].
        void OuterAndOuterSum()
        {
            var u = new fProxyN(3, Allocator.Temp);
            u[0] = 1f; u[1] = 2f; u[2] = 3f;
            var v = new fProxyN(2, Allocator.Temp);
            v[0] = 4f; v[1] = 5f;

            var O = new fProxyMxN(3, 2, Allocator.Temp);
            Generate.outer(in u, in v, ref O);

            var S = new fProxyMxN(3, 2, Allocator.Temp);
            Generate.outerSum(in u, in v, ref S);

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 2; j++)
                {
                    AssertClose(O[i, j], u[i] * v[j], 1E-5f);
                    AssertClose(S[i, j], u[i] + v[j], 1E-5f);
                }
        }

        // gaussianKernel2D == outer(g,g) of the 1D Gaussian: separable, sums to 1, symmetric.
        void GaussianKernel2D()
        {
            int N = 5;
            fProxy sigma = (fProxy)1.2;

            var g = new fProxyN(N, Allocator.Temp);
            Generate.gaussianKernel(ref g, sigma);

            var K = new fProxyMxN(N, N, Allocator.Temp);
            Generate.gaussianKernel2D(ref K, sigma);

            fProxy sum = (fProxy)0;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                {
                    AssertClose(K[i, j], g[i] * g[j], 1E-5f); // separable
                    sum += K[i, j];
                }
            AssertClose(sum, (fProxy)1f, 1E-4f); // (Σg)² = 1

            // symmetric both ways
            AssertClose(K[0, 4], K[4, 0], 1E-6f);
            AssertClose(K[1, 3], K[3, 1], 1E-6f);
        }

        // Sine/Saw/Square/Triangle at canonical phase points over one cycle.
        void Waves()
        {
            var sine = new fProxyWave.Sine { Cycles = (fProxy)1 };
            AssertClose(sine.Eval((fProxy)0), (fProxy)0f, 1E-5f);
            AssertClose(sine.Eval((fProxy)0.25), (fProxy)1f, 1E-5f);
            AssertClose(sine.Eval((fProxy)0.5), (fProxy)0f, 1E-5f);
            AssertClose(sine.Eval((fProxy)0.75), (fProxy)(-1f), 1E-5f);

            var saw = new fProxyWave.Saw { Cycles = (fProxy)1 };
            AssertClose(saw.Eval((fProxy)0), (fProxy)(-1f), 1E-5f);
            AssertClose(saw.Eval((fProxy)0.5), (fProxy)0f, 1E-5f);

            var sq = new fProxyWave.Square { Cycles = (fProxy)1, Duty = (fProxy)0.5 };
            AssertClose(sq.Eval((fProxy)0.25), (fProxy)1f, 1E-6f);
            AssertClose(sq.Eval((fProxy)0.75), (fProxy)(-1f), 1E-6f);

            var tri = new fProxyWave.Triangle { Cycles = (fProxy)1 };
            AssertClose(tri.Eval((fProxy)0), (fProxy)(-1f), 1E-5f);
            AssertClose(tri.Eval((fProxy)0.25), (fProxy)0f, 1E-5f);
            AssertClose(tri.Eval((fProxy)0.5), (fProxy)1f, 1E-5f);

            // default-constructed Sine uses Cycles=1 (the 0->1 fallback)
            var def = new fProxyWave.Sine();
            AssertClose(def.Eval((fProxy)0.25), (fProxy)1f, 1E-5f);

            // Cycles != 1: a 2-cycle sine peaks at t=0.125 (quarter of the first of two periods).
            var two = new fProxyWave.Sine { Cycles = (fProxy)2 };
            AssertClose(two.Eval((fProxy)0.125), (fProxy)1f, 1E-5f);
            AssertClose(two.Eval((fProxy)0.25), (fProxy)0f, 1E-5f); // sin(π)

            // Phase shift: a quarter-period phase turns sin into cos, so Eval(0) == 1.
            var ph = new fProxyWave.Sine { Cycles = (fProxy)1, Phase = (fProxy)0.25 };
            AssertClose(ph.Eval((fProxy)0), (fProxy)1f, 1E-5f);
        }

        // Each allocating standalone wrapper equals the zero-alloc ref-dest primitive.
        void ArenaMatchesRef()
        {
            int N = 7;

            var lin = GenerateOP.fProxyLinspace((fProxy)(-2), (fProxy)3, N);
            var linRef = new fProxyN(N, Allocator.Temp);
            Generate.linspace(ref linRef, (fProxy)(-2), (fProxy)3);
            EqVec(in lin, in linRef, N);

            var ar = GenerateOP.fProxyArange((fProxy)5, (fProxy)(-2), N);
            var arRef = new fProxyN(N, Allocator.Temp);
            Generate.arange(ref arRef, (fProxy)5, (fProxy)(-2));
            EqVec(in ar, in arRef, N);

            var quad = new fProxyEasing.EaseInQuad();
            var smp = GenerateOP.fProxySample(ref quad, N, (fProxy)(-1), (fProxy)2);
            var smpRef = new fProxyN(N, Allocator.Temp);
            Generate.sample(ref quad, ref smpRef, (fProxy)(-1), (fProxy)2);
            EqVec(in smp, in smpRef, N);

            var gk = GenerateOP.fProxyGaussianKernel(N, (fProxy)1.5);
            var gkRef = new fProxyN(N, Allocator.Temp);
            Generate.gaussianKernel(ref gkRef, (fProxy)1.5);
            EqVec(in gk, in gkRef, N);

            var bk = GenerateOP.fProxyBoxKernel(N);
            var bkRef = new fProxyN(N, Allocator.Temp);
            Generate.boxKernel(ref bkRef);
            EqVec(in bk, in bkRef, N);

            var tk = GenerateOP.fProxyTentKernel(N);
            var tkRef = new fProxyN(N, Allocator.Temp);
            Generate.tentKernel(ref tkRef);
            EqVec(in tk, in tkRef, N);

            var win = GenerateOP.fProxyWindow(N, WindowType.Blackman);
            var winRef = new fProxyN(N, Allocator.Temp);
            Generate.window(ref winRef, WindowType.Blackman);
            EqVec(in win, in winRef, N);

            var ease = new fProxyEasing.SmoothStep();
            var lut = GenerateOP.fProxyEasingLUT(ref ease, N);
            var lutRef = new fProxyN(N, Allocator.Temp);
            Generate.sample(ref ease, ref lutRef, (fProxy)0, (fProxy)1);
            EqVec(in lut, in lutRef, N);

            // outer wrapper vs primitive
            var u = GenerateOP.fProxyLinspace((fProxy)1, (fProxy)4, 4);
            var v = GenerateOP.fProxyLinspace((fProxy)0, (fProxy)2, 3);
            var O = GenerateOP.fProxyOuter(in u, in v);
            var Oref = new fProxyMxN(4, 3, Allocator.Temp);
            Generate.outer(in u, in v, ref Oref);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++)
                    AssertClose(O[i, j], Oref[i, j], 1E-5f);

            var Sm = GenerateOP.fProxyOuterSum(in u, in v);
            var SmRef = new fProxyMxN(4, 3, Allocator.Temp);
            Generate.outerSum(in u, in v, ref SmRef);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++)
                    AssertClose(Sm[i, j], SmRef[i, j], 1E-5f);

            // gaussianKernel2D wrapper vs primitive
            var K = GenerateOP.fProxyGaussianKernel2D(5, (fProxy)1.3);
            var Kref = new fProxyMxN(5, 5, Allocator.Temp);
            Generate.gaussianKernel2D(ref Kref, (fProxy)1.3);
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 5; j++)
                    AssertClose(K[i, j], Kref[i, j], 1E-5f);
        }

        void EqVec(in fProxyN a, in fProxyN b, int len)
        {
            for (int i = 0; i < len; i++)
                AssertClose(a[i], b[i], 1E-5f);
        }

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

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = (fProxy)(-1);
                Fail[2] = (fProxy)(-1);
                Fail[3] = (fProxy)(-1);
            }
            Assert.IsTrue(ok);
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

    [Test] public void LinspaceArangeTest() => RunJob(TestJob.TestType.LinspaceArange);
    [Test] public void SampleEqualsManualTest() => RunJob(TestJob.TestType.SampleEqualsManual);
    [Test] public void EasingKnownValuesTest() => RunJob(TestJob.TestType.EasingKnownValues);
    [Test] public void EasingOutMirrorsInTest() => RunJob(TestJob.TestType.EasingOutMirrorsIn);
    [Test] public void GaussianKernelTest() => RunJob(TestJob.TestType.GaussianKernel);
    [Test] public void BoxTentKernelTest() => RunJob(TestJob.TestType.BoxTentKernel);
    [Test] public void WindowsTest() => RunJob(TestJob.TestType.Windows);
    [Test] public void OuterAndOuterSumTest() => RunJob(TestJob.TestType.OuterAndOuterSum);
    [Test] public void GaussianKernel2DTest() => RunJob(TestJob.TestType.GaussianKernel2D);
    [Test] public void WavesTest() => RunJob(TestJob.TestType.Waves);
    [Test] public void ArenaMatchesRefTest() => RunJob(TestJob.TestType.ArenaMatchesRef);

    // ---- Managed throw tests (main thread; guard paths) ----

    [Test]
    public void GaussianNonPositiveSigmaThrows()
    {
        var v = new fProxyN(5, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => Generate.gaussianKernel(ref v, (fProxy)0));
        Assert.Throws<ArgumentException>(() => Generate.gaussianKernel(ref v, (fProxy)(-1)));
    }

    [Test]
    public void OuterMisSizedDestThrows()
    {
        var u = new fProxyN(3, Allocator.Temp);
        var w = new fProxyN(2, Allocator.Temp);
        var bad = new fProxyMxN(2, 2, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => Generate.outer(in u, in w, ref bad));
        Assert.Throws<ArgumentException>(() => Generate.outerSum(in u, in w, ref bad));
    }

    [Test]
    public void GaussianKernel2DNonSquareThrows()
    {
        var bad = new fProxyMxN(3, 4, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => Generate.gaussianKernel2D(ref bad, (fProxy)1));
        // sigma guard fires before the internal Temp alloc (no leak on the throw path)
        var sq = new fProxyMxN(4, 4, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => Generate.gaussianKernel2D(ref sq, (fProxy)0));
        Assert.Throws<ArgumentException>(() => Generate.gaussianKernel2D(ref sq, (fProxy)(-2)));
    }

    [Test]
    public void EmptyDestThrows()
    {
        var v0 = new fProxyN(0, Allocator.Temp);
        var quad = new fProxyEasing.EaseInQuad();
        Assert.Throws<ArgumentException>(() => Generate.linspace(ref v0, (fProxy)0, (fProxy)1));
        Assert.Throws<ArgumentException>(() => Generate.arange(ref v0, (fProxy)0, (fProxy)1));
        Assert.Throws<ArgumentException>(() => Generate.sample(ref quad, ref v0));
        Assert.Throws<ArgumentException>(() => Generate.boxKernel(ref v0));
        Assert.Throws<ArgumentException>(() => Generate.tentKernel(ref v0));
        Assert.Throws<ArgumentException>(() => Generate.gaussianKernel(ref v0, (fProxy)1));
        Assert.Throws<ArgumentException>(() => Generate.window(ref v0, WindowType.Hann));
    }
}
