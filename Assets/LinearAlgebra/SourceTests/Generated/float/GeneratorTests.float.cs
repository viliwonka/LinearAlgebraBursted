using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;

public class floatGeneratorTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
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
        public NativeArray<float> Fail;

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
            var arena = new Arena(Allocator.Persistent);

            var v = arena.floatVec(5);
            floatGen_OP.linspace(ref v, (float)0, (float)1);
            AssertClose(v[0], (float)0f, 1E-5f);
            AssertClose(v[1], (float)0.25f, 1E-5f);
            AssertClose(v[2], (float)0.5f, 1E-5f);
            AssertClose(v[3], (float)0.75f, 1E-5f);
            AssertClose(v[4], (float)1f, 1E-5f);

            // endpoints land EXACTLY (no accumulated lerp error)
            AssertClose(v[0], (float)0f, 0f);
            AssertClose(v[4], (float)1f, 0f);

            var one = arena.floatVec(1);
            floatGen_OP.linspace(ref one, (float)7, (float)9);
            AssertClose(one[0], (float)7f, 0f);

            var r = arena.floatVec(4);
            floatGen_OP.arange(ref r, (float)2, (float)3);
            AssertClose(r[0], (float)2f, 1E-5f);
            AssertClose(r[1], (float)5f, 1E-5f);
            AssertClose(r[2], (float)8f, 1E-5f);
            AssertClose(r[3], (float)11f, 1E-5f);

            arena.Dispose();
        }

        // sample over [0,1] of EaseInQuad equals the manual loop value (i/(N-1))².
        void SampleEqualsManual()
        {
            var arena = new Arena(Allocator.Persistent);

            int N = 9;
            var f = new floatEasing.EaseInQuad();
            var dest = arena.floatVec(N);
            floatGen_OP.sample(ref f, ref dest);

            float scale = (float)1 / (float)(N - 1);
            for (int i = 0; i < N; i++)
            {
                float t = i * scale;
                AssertClose(dest[i], f.Eval(t), 1E-5f);
                AssertClose(dest[i], t * t, 1E-5f);
            }

            // N==1 -> {f.Eval(t0)}
            var one = arena.floatVec(1);
            floatGen_OP.sample(ref f, ref one, (float)0.5, (float)0.9);
            AssertClose(one[0], f.Eval((float)0.5), 1E-6f);

            // explicit domain [2,4] hits the endpoints exactly
            var dom = arena.floatVec(3);
            var lin = new floatEasing.Linear();
            floatGen_OP.sample(ref lin, ref dom, (float)2, (float)4);
            AssertClose(dom[0], (float)2f, 1E-5f);
            AssertClose(dom[1], (float)3f, 1E-5f);
            AssertClose(dom[2], (float)4f, 1E-5f);

            arena.Dispose();
        }

        // Endpoints and a few interior known values for the core easings.
        void EasingKnownValues()
        {
            // All standard easings pin (0)->0 and (1)->1.
            AssertClose(new floatEasing.Linear().Eval((float)0), (float)0f, 1E-6f);
            AssertClose(new floatEasing.Linear().Eval((float)1), (float)1f, 1E-6f);
            AssertClose(new floatEasing.Linear().Eval((float)0.37), (float)0.37f, 1E-6f);

            AssertClose(new floatEasing.SmoothStep().Eval((float)0), (float)0f, 1E-6f);
            AssertClose(new floatEasing.SmoothStep().Eval((float)1), (float)1f, 1E-6f);
            AssertClose(new floatEasing.SmoothStep().Eval((float)0.5), (float)0.5f, 1E-6f);

            AssertClose(new floatEasing.SmootherStep().Eval((float)0), (float)0f, 1E-6f);
            AssertClose(new floatEasing.SmootherStep().Eval((float)1), (float)1f, 1E-6f);
            AssertClose(new floatEasing.SmootherStep().Eval((float)0.5), (float)0.5f, 1E-6f);

            AssertClose(new floatEasing.EaseInQuad().Eval((float)0.5), (float)0.25f, 1E-6f);
            AssertClose(new floatEasing.EaseOutQuad().Eval((float)0.5), (float)0.75f, 1E-6f);
            AssertClose(new floatEasing.EaseInCubic().Eval((float)0.5), (float)0.125f, 1E-6f);

            // Quart family (was entirely untested)
            AssertClose(new floatEasing.EaseInQuart().Eval((float)0.5), (float)0.0625f, 1E-6f);
            AssertClose(new floatEasing.EaseOutQuart().Eval((float)0.5), (float)0.9375f, 1E-6f);

            // InOut variants — interior values straddling the t=0.5 branch seam (the most bug-prone spot).
            // InOutQuad: 2t² for t<.5, 1-(-2t+2)²/2 else.
            AssertClose(new floatEasing.EaseInOutQuad().Eval((float)0.25), (float)0.125f, 1E-6f);
            AssertClose(new floatEasing.EaseInOutQuad().Eval((float)0.5), (float)0.5f, 1E-6f);
            AssertClose(new floatEasing.EaseInOutQuad().Eval((float)0.75), (float)0.875f, 1E-6f);
            // InOutCubic: 4t³ / 1-(-2t+2)³/2.
            AssertClose(new floatEasing.EaseInOutCubic().Eval((float)0.25), (float)0.0625f, 1E-6f);
            AssertClose(new floatEasing.EaseInOutCubic().Eval((float)0.75), (float)0.9375f, 1E-6f);
            // InOutQuart: 8t⁴ / 1-(-2t+2)⁴/2.
            AssertClose(new floatEasing.EaseInOutQuart().Eval((float)0.25), (float)0.03125f, 1E-6f);
            AssertClose(new floatEasing.EaseInOutQuart().Eval((float)0.75), (float)0.96875f, 1E-6f);

            // Sine variants
            AssertClose(new floatEasing.EaseInSine().Eval((float)0), (float)0f, 1E-6f);
            AssertClose(new floatEasing.EaseInSine().Eval((float)1), (float)1f, 1E-5f);
            AssertClose(new floatEasing.EaseOutSine().Eval((float)0), (float)0f, 1E-5f);
            AssertClose(new floatEasing.EaseOutSine().Eval((float)1), (float)1f, 1E-5f);
            AssertClose(new floatEasing.EaseInOutSine().Eval((float)0.5), (float)0.5f, 1E-5f);

            // Expo clamps exactly at the ends.
            AssertClose(new floatEasing.EaseInExpo().Eval((float)0), (float)0f, 1E-6f);
            AssertClose(new floatEasing.EaseOutExpo().Eval((float)1), (float)1f, 1E-6f);
            AssertClose(new floatEasing.EaseInOutExpo().Eval((float)0), (float)0f, 1E-6f);
            AssertClose(new floatEasing.EaseInOutExpo().Eval((float)1), (float)1f, 1E-6f);
            AssertClose(new floatEasing.EaseInOutExpo().Eval((float)0.5), (float)0.5f, 1E-6f);

            // Bounce/Elastic/Back pin the ends (overshoot only in the interior). Endpoint guards mean
            // the *interior* values are what actually exercise the formulas — so check those too.
            AssertClose(new floatEasing.EaseOutBounce().Eval((float)0), (float)0f, 1E-6f);
            AssertClose(new floatEasing.EaseOutBounce().Eval((float)1), (float)1f, 1E-5f);
            // t=0.5 lands in the 2nd bounce piece: 7.5625*(0.5-1.5/2.75)²+0.75 = 0.765625.
            AssertClose(new floatEasing.EaseOutBounce().Eval((float)0.5), (float)0.765625f, 1E-4f);

            AssertClose(new floatEasing.EaseInElastic().Eval((float)0), (float)0f, 1E-6f);
            AssertClose(new floatEasing.EaseInElastic().Eval((float)1), (float)1f, 1E-6f);
            // t=0.5: -2^-5 * sin(-5.75·2π/3) = -(1/32)·sin(π/6) = -0.015625 (verifies c4 and the 10.75).
            AssertClose(new floatEasing.EaseInElastic().Eval((float)0.5), (float)(-0.015625f), 1E-4f);
            AssertClose(new floatEasing.EaseOutElastic().Eval((float)0), (float)0f, 1E-6f);
            AssertClose(new floatEasing.EaseOutElastic().Eval((float)1), (float)1f, 1E-6f);
            // t=0.5: 2^-5 · sin(4.25·2π/3) + 1 = (1/32)·sin(5π/6) + 1 = 1.015625.
            AssertClose(new floatEasing.EaseOutElastic().Eval((float)0.5), (float)1.015625f, 1E-4f);

            AssertClose(new floatEasing.EaseInBack().Eval((float)0), (float)0f, 1E-6f);
            AssertClose(new floatEasing.EaseInBack().Eval((float)1), (float)1f, 1E-5f);
            // t=0.5: c3·0.125 - c1·0.25 with c1=1.70158, c3=2.70158 → -0.0876975.
            AssertClose(new floatEasing.EaseInBack().Eval((float)0.5), (float)(-0.0876975f), 1E-4f);
            AssertClose(new floatEasing.EaseOutBack().Eval((float)0), (float)0f, 1E-5f);
            AssertClose(new floatEasing.EaseOutBack().Eval((float)1), (float)1f, 1E-5f);
            // t=0.5 mirrors EaseInBack: 1 + c3·(-0.5)³ + c1·(-0.5)² → 1.0876975.
            AssertClose(new floatEasing.EaseOutBack().Eval((float)0.5), (float)1.0876975f, 1E-4f);

            // ---- the four family-completing variants (In/InOut Bounce, InOut Elastic, InOut Back) ----
            // InBounce(t) = 1 - OutBounce(1-t): endpoints pin, and (0.5) = 1 - 0.765625 = 0.234375.
            AssertClose(new floatEasing.EaseInBounce().Eval((float)0), (float)0f, 1E-5f);
            AssertClose(new floatEasing.EaseInBounce().Eval((float)1), (float)1f, 1E-5f);
            AssertClose(new floatEasing.EaseInBounce().Eval((float)0.5), (float)0.234375f, 1E-4f);
            // All InOut variants pass through (0.5, 0.5) and pin the ends.
            AssertClose(new floatEasing.EaseInOutBounce().Eval((float)0), (float)0f, 1E-5f);
            AssertClose(new floatEasing.EaseInOutBounce().Eval((float)1), (float)1f, 1E-5f);
            AssertClose(new floatEasing.EaseInOutBounce().Eval((float)0.5), (float)0.5f, 1E-5f);
            AssertClose(new floatEasing.EaseInOutElastic().Eval((float)0), (float)0f, 1E-6f);
            AssertClose(new floatEasing.EaseInOutElastic().Eval((float)1), (float)1f, 1E-6f);
            AssertClose(new floatEasing.EaseInOutElastic().Eval((float)0.5), (float)0.5f, 1E-5f);
            AssertClose(new floatEasing.EaseInOutBack().Eval((float)0), (float)0f, 1E-6f);
            AssertClose(new floatEasing.EaseInOutBack().Eval((float)1), (float)1f, 1E-5f);
            AssertClose(new floatEasing.EaseInOutBack().Eval((float)0.5), (float)0.5f, 1E-5f);
        }

        // The Out variant is the time/value-mirrored In variant: easeOut(t) == 1 - easeIn(1-t).
        void EasingOutMirrorsIn()
        {
            var inQ = new floatEasing.EaseInQuad();
            var outQ = new floatEasing.EaseOutQuad();
            var inC = new floatEasing.EaseInCubic();
            var outC = new floatEasing.EaseOutCubic();
            var inK = new floatEasing.EaseInQuart();
            var outK = new floatEasing.EaseOutQuart();
            var inS = new floatEasing.EaseInSine();
            var outS = new floatEasing.EaseOutSine();

            for (int k = 0; k <= 10; k++)
            {
                float t = (float)k / (float)10;
                AssertClose(outQ.Eval(t), (float)1 - inQ.Eval((float)1 - t), 1E-5f);
                AssertClose(outC.Eval(t), (float)1 - inC.Eval((float)1 - t), 1E-5f);
                AssertClose(outK.Eval(t), (float)1 - inK.Eval((float)1 - t), 1E-5f);
                AssertClose(outS.Eval(t), (float)1 - inS.Eval((float)1 - t), 1E-5f);
            }
        }

        // 1D Gaussian: sums to 1, symmetric, peak at center.
        void GaussianKernel()
        {
            var arena = new Arena(Allocator.Persistent);

            int N = 5;
            var g = arena.floatVec(N);
            floatGen_OP.gaussianKernel(ref g, (float)1);

            float sum = (float)0;
            for (int i = 0; i < N; i++) sum += g[i];
            AssertClose(sum, (float)1f, 1E-5f);

            // symmetric
            for (int i = 0; i < N; i++)
                AssertClose(g[i], g[N - 1 - i], 1E-5f);

            // center is the strict max
            AssertTrue(g[2] > g[1]);
            AssertTrue(g[1] > g[0]);

            // N==1 -> {1}
            var one = arena.floatVec(1);
            floatGen_OP.gaussianKernel(ref one, (float)2);
            AssertClose(one[0], (float)1f, 1E-6f);

            arena.Dispose();
        }

        // Box: every weight 1/N (sum 1). Tent: symmetric, sums to 1, peak at center.
        void BoxTentKernel()
        {
            var arena = new Arena(Allocator.Persistent);

            int N = 6;
            var box = arena.floatVec(N);
            floatGen_OP.boxKernel(ref box);
            float bsum = (float)0;
            for (int i = 0; i < N; i++)
            {
                AssertClose(box[i], (float)1 / (float)N, 1E-6f);
                bsum += box[i];
            }
            AssertClose(bsum, (float)1f, 1E-5f);

            int M = 5;
            var tent = arena.floatVec(M);
            floatGen_OP.tentKernel(ref tent);
            float tsum = (float)0;
            for (int i = 0; i < M; i++) tsum += tent[i];
            AssertClose(tsum, (float)1f, 1E-5f);
            for (int i = 0; i < M; i++)
                AssertClose(tent[i], tent[M - 1 - i], 1E-5f);
            AssertTrue(tent[2] > tent[1]);
            AssertTrue(tent[1] > tent[0]);

            arena.Dispose();
        }

        // Hann/Hamming/Blackman/Box known endpoint and center values (N=5, denom 4).
        void Windows()
        {
            var arena = new Arena(Allocator.Persistent);

            int N = 5;

            var hann = arena.floatVec(N);
            floatGen_OP.window(ref hann, WindowType.Hann);
            AssertClose(hann[0], (float)0f, 1E-5f);
            AssertClose(hann[4], (float)0f, 1E-5f);
            AssertClose(hann[2], (float)1f, 1E-5f); // 0.5(1-cos π) = 1
            AssertClose(hann[1], hann[3], 1E-5f);    // symmetric

            var hamming = arena.floatVec(N);
            floatGen_OP.window(ref hamming, WindowType.Hamming);
            AssertClose(hamming[0], (float)0.08f, 1E-5f);
            AssertClose(hamming[4], (float)0.08f, 1E-5f);
            AssertClose(hamming[2], (float)1f, 1E-5f); // 0.54+0.46 = 1

            var black = arena.floatVec(N);
            floatGen_OP.window(ref black, WindowType.Blackman);
            AssertClose(black[0], (float)0f, 1E-5f);  // 0.42-0.5+0.08 = 0
            AssertClose(black[4], (float)0f, 1E-5f);
            AssertClose(black[2], (float)1f, 1E-5f);  // 0.42+0.5+0.08 = 1 (center)
            AssertClose(black[1], black[3], 1E-5f);    // symmetric

            var box = arena.floatVec(N);
            floatGen_OP.window(ref box, WindowType.Box);
            for (int i = 0; i < N; i++)
                AssertClose(box[i], (float)1f, 1E-6f);

            // N==1 -> {1} for the (N-1)-denominator windows (no div-by-zero)
            var one = arena.floatVec(1);
            floatGen_OP.window(ref one, WindowType.Hann);
            AssertClose(one[0], (float)1f, 1E-6f);

            arena.Dispose();
        }

        // outer[i,j] == u[i]*v[j]; outerSum[i,j] == u[i]+v[j].
        void OuterAndOuterSum()
        {
            var arena = new Arena(Allocator.Persistent);

            var u = arena.floatVec(3);
            u[0] = 1f; u[1] = 2f; u[2] = 3f;
            var v = arena.floatVec(2);
            v[0] = 4f; v[1] = 5f;

            var O = arena.floatMat(3, 2);
            floatGen_OP.outer(in u, in v, ref O);

            var S = arena.floatMat(3, 2);
            floatGen_OP.outerSum(in u, in v, ref S);

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 2; j++)
                {
                    AssertClose(O[i, j], u[i] * v[j], 1E-5f);
                    AssertClose(S[i, j], u[i] + v[j], 1E-5f);
                }

            arena.Dispose();
        }

        // gaussianKernel2D == outer(g,g) of the 1D Gaussian: separable, sums to 1, symmetric.
        void GaussianKernel2D()
        {
            var arena = new Arena(Allocator.Persistent);

            int N = 5;
            float sigma = (float)1.2;

            var g = arena.floatVec(N);
            floatGen_OP.gaussianKernel(ref g, sigma);

            var K = arena.floatMat(N, N);
            floatGen_OP.gaussianKernel2D(ref K, sigma);

            float sum = (float)0;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                {
                    AssertClose(K[i, j], g[i] * g[j], 1E-5f); // separable
                    sum += K[i, j];
                }
            AssertClose(sum, (float)1f, 1E-4f); // (Σg)² = 1

            // symmetric both ways
            AssertClose(K[0, 4], K[4, 0], 1E-6f);
            AssertClose(K[1, 3], K[3, 1], 1E-6f);

            arena.Dispose();
        }

        // Sine/Saw/Square/Triangle at canonical phase points over one cycle.
        void Waves()
        {
            var sine = new floatWave.Sine { Cycles = (float)1 };
            AssertClose(sine.Eval((float)0), (float)0f, 1E-5f);
            AssertClose(sine.Eval((float)0.25), (float)1f, 1E-5f);
            AssertClose(sine.Eval((float)0.5), (float)0f, 1E-5f);
            AssertClose(sine.Eval((float)0.75), (float)(-1f), 1E-5f);

            var saw = new floatWave.Saw { Cycles = (float)1 };
            AssertClose(saw.Eval((float)0), (float)(-1f), 1E-5f);
            AssertClose(saw.Eval((float)0.5), (float)0f, 1E-5f);

            var sq = new floatWave.Square { Cycles = (float)1, Duty = (float)0.5 };
            AssertClose(sq.Eval((float)0.25), (float)1f, 1E-6f);
            AssertClose(sq.Eval((float)0.75), (float)(-1f), 1E-6f);

            var tri = new floatWave.Triangle { Cycles = (float)1 };
            AssertClose(tri.Eval((float)0), (float)(-1f), 1E-5f);
            AssertClose(tri.Eval((float)0.25), (float)0f, 1E-5f);
            AssertClose(tri.Eval((float)0.5), (float)1f, 1E-5f);

            // default-constructed Sine uses Cycles=1 (the 0->1 fallback)
            var def = new floatWave.Sine();
            AssertClose(def.Eval((float)0.25), (float)1f, 1E-5f);

            // Cycles != 1: a 2-cycle sine peaks at t=0.125 (quarter of the first of two periods).
            var two = new floatWave.Sine { Cycles = (float)2 };
            AssertClose(two.Eval((float)0.125), (float)1f, 1E-5f);
            AssertClose(two.Eval((float)0.25), (float)0f, 1E-5f); // sin(π)

            // Phase shift: a quarter-period phase turns sin into cos, so Eval(0) == 1.
            var ph = new floatWave.Sine { Cycles = (float)1, Phase = (float)0.25 };
            AssertClose(ph.Eval((float)0), (float)1f, 1E-5f);
        }

        // Each allocating arena wrapper equals the zero-alloc ref-dest primitive.
        void ArenaMatchesRef()
        {
            var arena = new Arena(Allocator.Persistent);

            int N = 7;

            var lin = arena.floatLinspace((float)(-2), (float)3, N);
            var linRef = arena.floatVec(N);
            floatGen_OP.linspace(ref linRef, (float)(-2), (float)3);
            EqVec(in lin, in linRef, N);

            var ar = arena.floatArange((float)5, (float)(-2), N);
            var arRef = arena.floatVec(N);
            floatGen_OP.arange(ref arRef, (float)5, (float)(-2));
            EqVec(in ar, in arRef, N);

            var quad = new floatEasing.EaseInQuad();
            var smp = arena.floatSample(ref quad, N, (float)(-1), (float)2);
            var smpRef = arena.floatVec(N);
            floatGen_OP.sample(ref quad, ref smpRef, (float)(-1), (float)2);
            EqVec(in smp, in smpRef, N);

            var gk = arena.floatGaussianKernel(N, (float)1.5);
            var gkRef = arena.floatVec(N);
            floatGen_OP.gaussianKernel(ref gkRef, (float)1.5);
            EqVec(in gk, in gkRef, N);

            var bk = arena.floatBoxKernel(N);
            var bkRef = arena.floatVec(N);
            floatGen_OP.boxKernel(ref bkRef);
            EqVec(in bk, in bkRef, N);

            var tk = arena.floatTentKernel(N);
            var tkRef = arena.floatVec(N);
            floatGen_OP.tentKernel(ref tkRef);
            EqVec(in tk, in tkRef, N);

            var win = arena.floatWindow(N, WindowType.Blackman);
            var winRef = arena.floatVec(N);
            floatGen_OP.window(ref winRef, WindowType.Blackman);
            EqVec(in win, in winRef, N);

            var ease = new floatEasing.SmoothStep();
            var lut = arena.floatEasingLUT(ref ease, N);
            var lutRef = arena.floatVec(N);
            floatGen_OP.sample(ref ease, ref lutRef, (float)0, (float)1);
            EqVec(in lut, in lutRef, N);

            // outer wrapper vs primitive
            var u = arena.floatLinspace((float)1, (float)4, 4);
            var v = arena.floatLinspace((float)0, (float)2, 3);
            var O = arena.floatOuter(in u, in v);
            var Oref = arena.floatMat(4, 3);
            floatGen_OP.outer(in u, in v, ref Oref);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++)
                    AssertClose(O[i, j], Oref[i, j], 1E-5f);

            var Sm = arena.floatOuterSum(in u, in v);
            var SmRef = arena.floatMat(4, 3);
            floatGen_OP.outerSum(in u, in v, ref SmRef);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++)
                    AssertClose(Sm[i, j], SmRef[i, j], 1E-5f);

            // gaussianKernel2D wrapper vs primitive
            var K = arena.floatGaussianKernel2D(5, (float)1.3);
            var Kref = arena.floatMat(5, 5);
            floatGen_OP.gaussianKernel2D(ref Kref, (float)1.3);
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 5; j++)
                    AssertClose(K[i, j], Kref[i, j], 1E-5f);

            arena.Dispose();
        }

        void EqVec(in floatN a, in floatN b, int len)
        {
            for (int i = 0; i < len; i++)
                AssertClose(a[i], b[i], 1E-5f);
        }

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

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = (float)(-1);
                Fail[2] = (float)(-1);
                Fail[3] = (float)(-1);
            }
            Assert.IsTrue(ok);
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
        var arena = new Arena(Allocator.Persistent);
        var v = arena.floatVec(5);
        Assert.Throws<ArgumentException>(() => floatGen_OP.gaussianKernel(ref v, (float)0));
        Assert.Throws<ArgumentException>(() => floatGen_OP.gaussianKernel(ref v, (float)(-1)));
        arena.Dispose();
    }

    [Test]
    public void OuterMisSizedDestThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var u = arena.floatVec(3);
        var w = arena.floatVec(2);
        var bad = arena.floatMat(2, 2);
        Assert.Throws<ArgumentException>(() => floatGen_OP.outer(in u, in w, ref bad));
        Assert.Throws<ArgumentException>(() => floatGen_OP.outerSum(in u, in w, ref bad));
        arena.Dispose();
    }

    [Test]
    public void GaussianKernel2DNonSquareThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var bad = arena.floatMat(3, 4);
        Assert.Throws<ArgumentException>(() => floatGen_OP.gaussianKernel2D(ref bad, (float)1));
        // sigma guard fires before the internal Temp alloc (no leak on the throw path)
        var sq = arena.floatMat(4, 4);
        Assert.Throws<ArgumentException>(() => floatGen_OP.gaussianKernel2D(ref sq, (float)0));
        Assert.Throws<ArgumentException>(() => floatGen_OP.gaussianKernel2D(ref sq, (float)(-2)));
        arena.Dispose();
    }

    [Test]
    public void EmptyDestThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var v0 = arena.floatVec(0);
        var quad = new floatEasing.EaseInQuad();
        Assert.Throws<ArgumentException>(() => floatGen_OP.linspace(ref v0, (float)0, (float)1));
        Assert.Throws<ArgumentException>(() => floatGen_OP.arange(ref v0, (float)0, (float)1));
        Assert.Throws<ArgumentException>(() => floatGen_OP.sample(ref quad, ref v0));
        Assert.Throws<ArgumentException>(() => floatGen_OP.boxKernel(ref v0));
        Assert.Throws<ArgumentException>(() => floatGen_OP.tentKernel(ref v0));
        Assert.Throws<ArgumentException>(() => floatGen_OP.gaussianKernel(ref v0, (float)1));
        Assert.Throws<ArgumentException>(() => floatGen_OP.window(ref v0, WindowType.Hann));
        arena.Dispose();
    }
}
