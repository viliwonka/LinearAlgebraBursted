using System;

using LinearAlgebra;
using LinearAlgebra.Stats;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using UnityEngine;
using UnityEngine.TestTools;

public class doubleOptimizeTests
{

    // ----- Functor structs (Burst-legal: only numeric fields, no managed state) -----

    // (x - 3)^2 - 4, roots at x = 1 and x = 5.
    public struct ShiftedParabola : IdoubleScalarFunction
    {
        public double Eval(double x) => (x - (double)3) * (x - (double)3) - (double)4;
    }

    // cos(x): root at pi/2, minimum at pi.
    public struct Cos : IdoubleScalarFunction
    {
        public double Eval(double x) => Unity.Mathematics.math.cos(x);
    }

    // x^2 - 2, root at sqrt(2); derivative 2x.
    public struct QuadraticDeriv : IdoubleScalarDerivativeFunction
    {
        public double Eval(double x) => x * x - (double)2;
        public double Derivative(double x) => (double)2 * x;
    }

    // Constant non-zero value with zero derivative everywhere -> Newton must report failure (flat).
    public struct FlatDeriv : IdoubleScalarDerivativeFunction
    {
        public double Eval(double x) => (double)5;
        public double Derivative(double x) => (double)0;
    }

    // (x - 2)^2 + 1, minimum at x = 2.
    public struct ParabolaMin : IdoubleScalarFunction
    {
        public double Eval(double x) => (x - (double)2) * (x - (double)2) + (double)1;
    }

    // 4-D quadratic bowl: f(x) = sum_i (x_i - t_i)^2 with targets (1,2,3,4), c_i = 1.
    // Target components stored as scalar fields (no managed array, Burst-legal).
    public struct Bowl : IdoubleGradientFunction
    {
        public double Target(int i)
        {
            switch (i)
            {
                case 0: return (double)1;
                case 1: return (double)2;
                case 2: return (double)3;
                default: return (double)4;
            }
        }

        public double Eval(in doubleN x)
        {
            double sum = (double)0;
            for (int i = 0; i < x.N; i++)
            {
                double d = x[i] - Target(i);
                sum += d * d;
            }
            return sum;
        }

        public void Gradient(in doubleN x, ref doubleN g)
        {
            for (int i = 0; i < x.N; i++)
                g[i] = (double)2 * (x[i] - Target(i));
        }
    }

    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            BisectionParabola,
            BisectionNotBracketed,
            BisectionCos,
            NewtonSqrt2,
            NewtonFlat,
            GoldenParabolaMin,
            GoldenCosMin,
            GradientDescentBowl,
            GradientDescentAtMinimum
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff/extra
        public NativeArray<double> Fail;

        public void Execute()
        {
            switch(Type)
            {
                case TestType.BisectionParabola:
                    BisectionParabola();
                break;
                case TestType.BisectionNotBracketed:
                    BisectionNotBracketed();
                break;
                case TestType.BisectionCos:
                    BisectionCos();
                break;
                case TestType.NewtonSqrt2:
                    NewtonSqrt2();
                break;
                case TestType.NewtonFlat:
                    NewtonFlat();
                break;
                case TestType.GoldenParabolaMin:
                    GoldenParabolaMin();
                break;
                case TestType.GoldenCosMin:
                    GoldenCosMin();
                break;
                case TestType.GradientDescentBowl:
                    GradientDescentBowl();
                break;
                case TestType.GradientDescentAtMinimum:
                    GradientDescentAtMinimum();
                break;
            }
        }

        // bisection on (x-3)^2 - 4 over [3, 10]: f(3) = -4 < 0, f(10) = 45 > 0, root = 5.
        // xTol = 10 * ZeroThreshold (1e-5 float / 1e-13 double): stays above the ulp
        // near |x| ~ 5 in each precision, so the interval width is actually reachable.
        public void BisectionParabola()
        {
            var fn = new ShiftedParabola();

            bool ok = Optimize.bisection(ref fn, (double)3, (double)10, out double root,
                                         (double)10 * Consts.doubleZeroThreshold);

            Assert.IsTrue(ok);
            AssertFinite(root);
            AssertClose(root, (double)5, (double)100 * Consts.doubleZeroThreshold);
        }

        // Non-bracketing interval [6, 10]: f(6) = 5 > 0, f(10) = 45 > 0, both positive -> false.
        public void BisectionNotBracketed()
        {
            var fn = new ShiftedParabola();

            bool ok = Optimize.bisection(ref fn, (double)6, (double)10, out double root);

            Assert.IsFalse(ok);
            AssertFinite(root);
            // root must be the endpoint with the smaller |f|; |f(6)| = 5 < |f(10)| = 45 -> 6.
            // The endpoint is returned unchanged, so this is exact in both precisions.
            AssertClose(root, (double)6, Consts.doubleZeroThreshold);
        }

        // cos(x) over [0, 3]: f(0) = 1 > 0, f(3) ~ -0.99 < 0, root = pi/2.
        // xTol = 10 * ZeroThreshold (1e-5 float / 1e-13 double): the interval can never
        // shrink below ~2 ulp near pi/2, so xTol must stay above that in each precision.
        public void BisectionCos()
        {
            var fn = new Cos();

            bool ok = Optimize.bisection(ref fn, (double)0, (double)3, out double root,
                                         (double)10 * Consts.doubleZeroThreshold);

            Assert.IsTrue(ok);
            AssertFinite(root);
            // full-precision pi/2 literal: math.PI is a float constant whose error (~4.4e-8)
            // would dominate the double-precision tolerance (1e-12)
            AssertClose(root, (double)1.5707963267948966, (double)100 * Consts.doubleZeroThreshold);
        }

        // Newton on x^2 - 2 from x0 = 1: converges to sqrt(2).
        public void NewtonSqrt2()
        {
            var fn = new QuadraticDeriv();

            bool ok = Optimize.newtonRoot(ref fn, (double)1, out double root);

            Assert.IsTrue(ok);
            AssertFinite(root);
            AssertClose(root, (double)1.4142135623730951, (double)100 * Consts.doubleZeroThreshold);

            // |f(root)| must be within the convergence tolerance (default fTol = ZeroThreshold).
            Assert.IsTrue(Unity.Mathematics.math.abs(fn.Eval(root)) <= Consts.doubleZeroThreshold);
        }

        // Newton with a flat (zero) derivative -> must return false, root finite (no NaN).
        public void NewtonFlat()
        {
            var fn = new FlatDeriv();

            bool ok = Optimize.newtonRoot(ref fn, (double)1, out double root);

            Assert.IsFalse(ok);
            AssertFinite(root);
        }

        // Golden-section on (x-2)^2 + 1 over [-5, 5]: minimum at x = 2.
        // xMin tolerance 3 * SqrtEps: a smooth minimum can only be localized to
        // ~sqrt(machine eps) (~3.5e-4 float / ~1.5e-8 double) because f is constant
        // to rounding within that distance.
        public void GoldenParabolaMin()
        {
            var fn = new ParabolaMin();

            bool ok = Optimize.goldenSection(ref fn, (double)(-5), (double)5, out double xMin,
                                             (double)10 * Consts.doubleZeroThreshold);

            Assert.IsTrue(ok);
            AssertFinite(xMin);
            AssertClose(xMin, (double)2, (double)3 * Consts.doubleSqrtEps);
        }

        // Golden-section on cos(x) over [2, 4]: minimum at x = pi (unimodal on this bracket).
        // xMin tolerance 3 * SqrtEps: see GoldenParabolaMin (sqrt(eps) localization limit).
        public void GoldenCosMin()
        {
            var fn = new Cos();

            bool ok = Optimize.goldenSection(ref fn, (double)2, (double)4, out double xMin,
                                             (double)10 * Consts.doubleZeroThreshold);

            Assert.IsTrue(ok);
            AssertFinite(xMin);
            // full-precision pi literal: math.PI is a float constant whose error (~8.7e-8)
            // would dominate the double-precision tolerance (3 * SqrtEps = 4.5e-8)
            AssertClose(xMin, (double)3.141592653589793, (double)3 * Consts.doubleSqrtEps);
        }

        // Gradient descent on the 4-D bowl from x = 0: converges to targets (1,2,3,4).
        public void GradientDescentBowl()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;

            var x = arena.doubleVec(n);
            var g = arena.doubleVec(n);

            var fn = new Bowl();

            int maxIter = 1000;
            bool ok = Optimize.gradientDescent(ref fn, ref x, ref g,
                                               (double)0.1f, (double)1E-4f, maxIter, out int iterations);

            Assert.IsTrue(ok);
            Assert.IsTrue(iterations < maxIter);

            for (int i = 0; i < n; i++)
            {
                AssertFinite(x[i]);
                AssertClose(x[i], fn.Target(i), 1E-3f);
            }

            arena.Dispose();
        }

        // Starting exactly at the minimum: gradient already below tolerance -> 0 iterations.
        public void GradientDescentAtMinimum()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;

            var x = arena.doubleVec(n);
            var g = arena.doubleVec(n);

            var fn = new Bowl();
            for (int i = 0; i < n; i++)
                x[i] = fn.Target(i);

            int maxIter = 1000;
            bool ok = Optimize.gradientDescent(ref fn, ref x, ref g,
                                               (double)0.1f, (double)1E-4f, maxIter, out int iterations);

            Assert.IsTrue(ok);
            Assert.AreEqual(0, iterations);

            for (int i = 0; i < n; i++)
            {
                AssertFinite(x[i]);
                AssertClose(x[i], fn.Target(i), 1E-6f);
            }

            arena.Dispose();
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff
        private void AssertFinite(double v)
        {
            if (!Unity.Mathematics.math.isfinite(v) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = v;
                Fail[2] = (double)0;
                Fail[3] = (double)0;
            }
            Assert.IsTrue(Unity.Mathematics.math.isfinite(v));
        }

        private void AssertClose(double a, double b, double precision)
        {
            double diff = Unity.Mathematics.math.abs(a - b);
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

    public static Array GetEnums() {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void OptimizeTests(TestJob.TestType type)
    {
        var fail = new NativeArray<double>(4, Allocator.TempJob);
        try {
            new TestJob() { Type = type, Fail = fail }.Run();
            // Under Burst a failed in-job assert logs an exception and aborts the job without
            // throwing to the caller - surface the recorded diagnostics here as well.
            if (fail[0] != (double)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");

        }
        catch (Exception e) {
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally {
            fail.Dispose();
        }
    }

    // Managed throw-test: argument validation runs on the main thread (not in a Burst job).

    [Test]
    public void GradientDescentThrowsOnMismatchedScratch()
    {
        var arena = new Arena(Allocator.Persistent);

        var x = arena.doubleVec(4);
        var g = arena.doubleVec(3);

        var fn = new Bowl();

        Assert.Catch<ArgumentException>(() =>
            Optimize.gradientDescent(ref fn, ref x, ref g,
                                     (double)0.1f, (double)1E-4f, 100, out int iterations));

        arena.Dispose();
    }

}
