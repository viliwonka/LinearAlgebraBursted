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
        public void BisectionParabola()
        {
            var fn = new ShiftedParabola();

            bool ok = Optimize.bisection(ref fn, (double)3, (double)10, out double root, (double)1E-7f);

            Assert.IsTrue(ok);
            AssertFinite(root);
            AssertClose(root, (double)5, 1E-4f);
        }

        // Non-bracketing interval [6, 10]: f(6) = 5 > 0, f(10) = 45 > 0, both positive -> false.
        public void BisectionNotBracketed()
        {
            var fn = new ShiftedParabola();

            bool ok = Optimize.bisection(ref fn, (double)6, (double)10, out double root);

            Assert.IsFalse(ok);
            AssertFinite(root);
            // root must be the endpoint with the smaller |f|; |f(6)| = 5 < |f(10)| = 45 -> 6.
            AssertClose(root, (double)6, 1E-6f);
        }

        // cos(x) over [0, 3]: f(0) = 1 > 0, f(3) ~ -0.99 < 0, root = pi/2.
        public void BisectionCos()
        {
            var fn = new Cos();

            bool ok = Optimize.bisection(ref fn, (double)0, (double)3, out double root, (double)1E-7f);

            Assert.IsTrue(ok);
            AssertFinite(root);
            AssertClose(root, (double)(Unity.Mathematics.math.PI * 0.5), 1E-4f);
        }

        // Newton on x^2 - 2 from x0 = 1: converges to sqrt(2).
        public void NewtonSqrt2()
        {
            var fn = new QuadraticDeriv();

            bool ok = Optimize.newtonRoot(ref fn, (double)1, out double root);

            Assert.IsTrue(ok);
            AssertFinite(root);
            AssertClose(root, (double)1.4142135623730951, 1E-4f);

            // |f(root)| must be within the convergence tolerance (default fTol = ZeroTreshold).
            Assert.IsTrue(Unity.Mathematics.math.abs(fn.Eval(root)) <= Consts.doubleZeroTreshold);
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
        public void GoldenParabolaMin()
        {
            var fn = new ParabolaMin();

            bool ok = Optimize.goldenSection(ref fn, (double)(-5), (double)5, out double xMin, (double)1E-6f);

            Assert.IsTrue(ok);
            AssertFinite(xMin);
            AssertClose(xMin, (double)2, 1E-4f);
        }

        // Golden-section on cos(x) over [2, 4]: minimum at x = pi (unimodal on this bracket).
        public void GoldenCosMin()
        {
            var fn = new Cos();

            bool ok = Optimize.goldenSection(ref fn, (double)2, (double)4, out double xMin, (double)1E-6f);

            Assert.IsTrue(ok);
            AssertFinite(xMin);
            AssertClose(xMin, (double)Unity.Mathematics.math.PI, 1E-4f);
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
            Assert.IsTrue(iterations < maxIter, $"iterations {iterations} should be < {maxIter}");

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

        private void AssertFinite(double v)
        {
            Assert.IsTrue(Unity.Mathematics.math.isfinite(v), $"Expected finite value, got {v}");
        }

        private void AssertClose(double a, double b, double precision)
        {
            double diff = Unity.Mathematics.math.abs(a - b);
            Assert.IsTrue(diff <= precision, $"Expected {b} got {a} (diff {diff})");
        }

    }

    public static Array GetEnums() {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void OptimizeTests(TestJob.TestType type)
    {
        new TestJob() { Type = type }.Run();
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
