using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

public class floatSelectRefTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct SelectRefTestJob : IJob
    {
        public enum TestType
        {
            VecCond,
            MatCond,
            VecScalarTrue,
            VecScalarFalse,
            MatScalarTrue,
            MatScalarFalse,
            VecAliasDest,
        }

        public TestType Type;

        // The ref-dest form runs the SAME kernel as the allocating form, so results are
        // bit-identical in principle; a small per-precision tolerance (~1e-6 float, tighter for
        // double) is kept anyway for robustness across expansions.
        static float Tol() => 256 * Consts.floatSqrtEps;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.VecCond:
                    VecCond();
                    break;
                case TestType.MatCond:
                    MatCond();
                    break;
                case TestType.VecScalarTrue:
                    VecScalarTrue();
                    break;
                case TestType.VecScalarFalse:
                    VecScalarFalse();
                    break;
                case TestType.MatScalarTrue:
                    MatScalarTrue();
                    break;
                case TestType.MatScalarFalse:
                    MatScalarFalse();
                    break;
                case TestType.VecAliasDest:
                    VecAliasDest();
                    break;
            }
        }

        // elementwise select(a, b, c): dest[i] = c[i] ? b[i] : a[i] (vector, boolN cond)
        void VecCond()
        {
            var arena = new Arena(Allocator.Persistent);

            int N = 17;

            var a = arena.floatRandomVec(N, -1f, 1f, 11111);
            var b = arena.floatRandomVec(N, -1f, 1f, 22222);
            var c = arena.boolRandomVec(N, 33333);

            // allocating reference
            var R = Select.select(a, b, c);

            // ref-dest into a preallocated destination
            var D = arena.floatVec(N);
            Select.select(in a, in b, in c, ref D);

            Assert.IsTrue(Analysis.isZero(R - D, Tol()));

            arena.Dispose();
        }

        // Same select(a,b,c) formula as VecCond, for boolMxN cond (matrix).
        void MatCond()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 6;
            int N = 9;

            var a = arena.floatRandomMat(M, N, -1f, 1f, 44444);
            var b = arena.floatRandomMat(M, N, -1f, 1f, 55555);
            var c = arena.boolRandomMat(M, N, 66666);

            var R = Select.select(a, b, c);

            var D = arena.floatMat(M, N);
            Select.select(in a, in b, in c, ref D);

            Assert.IsTrue(Analysis.isZero(R - D, Tol()));

            arena.Dispose();
        }

        // ---- scalar-bool condition: c=true -> dest must equal b; c=false -> dest must equal a
        //      (vector then matrix) ----
        void VecScalarTrue()
        {
            var arena = new Arena(Allocator.Persistent);

            int N = 13;

            var a = arena.floatRandomVec(N, -1f, 1f, 77777);
            var b = arena.floatRandomVec(N, -1f, 1f, 88888);

            var D = arena.floatVec(N);
            Select.select(in a, in b, true, ref D);

            Assert.IsTrue(Analysis.isZero(b - D, Tol()));

            arena.Dispose();
        }

        void VecScalarFalse()
        {
            var arena = new Arena(Allocator.Persistent);

            int N = 13;

            var a = arena.floatRandomVec(N, -1f, 1f, 99999);
            var b = arena.floatRandomVec(N, -1f, 1f, 10101);

            var D = arena.floatVec(N);
            Select.select(in a, in b, false, ref D);

            Assert.IsTrue(Analysis.isZero(a - D, Tol()));

            arena.Dispose();
        }

        void MatScalarTrue()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 5;
            int N = 7;

            var a = arena.floatRandomMat(M, N, -1f, 1f, 20202);
            var b = arena.floatRandomMat(M, N, -1f, 1f, 30303);

            var D = arena.floatMat(M, N);
            Select.select(in a, in b, true, ref D);

            Assert.IsTrue(Analysis.isZero(b - D, Tol()));

            arena.Dispose();
        }

        void MatScalarFalse()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 5;
            int N = 7;

            var a = arena.floatRandomMat(M, N, -1f, 1f, 40404);
            var b = arena.floatRandomMat(M, N, -1f, 1f, 50505);

            var D = arena.floatMat(M, N);
            Select.select(in a, in b, false, ref D);

            Assert.IsTrue(Analysis.isZero(a - D, Tol()));

            arena.Dispose();
        }

        // Elementwise aliasing IS allowed (no guard): select(a, b, c, ref a) must match the
        // allocating select(a, b, c). Compute the reference BEFORE the aliased call (which
        // overwrites a) and confirm aliasing does not corrupt the result.
        void VecAliasDest()
        {
            var arena = new Arena(Allocator.Persistent);

            int N = 21;

            var a = arena.floatRandomVec(N, -1f, 1f, 60606);
            var b = arena.floatRandomVec(N, -1f, 1f, 70707);
            var c = arena.boolRandomVec(N, 80808);

            // Reference into a SEPARATE buffer before a is overwritten.
            var R = arena.floatVec(N);
            Select.select(in a, in b, in c, ref R);

            // Now alias the destination onto input a.
            Select.select(in a, in b, in c, ref a);

            Assert.IsTrue(Analysis.isZero(R - a, Tol()));

            arena.Dispose();
        }
    }

    [Test]
    public void VecCondTest()
    {
        new SelectRefTestJob() { Type = SelectRefTestJob.TestType.VecCond }.Run();
    }

    [Test]
    public void MatCondTest()
    {
        new SelectRefTestJob() { Type = SelectRefTestJob.TestType.MatCond }.Run();
    }

    [Test]
    public void VecScalarTrueTest()
    {
        new SelectRefTestJob() { Type = SelectRefTestJob.TestType.VecScalarTrue }.Run();
    }

    [Test]
    public void VecScalarFalseTest()
    {
        new SelectRefTestJob() { Type = SelectRefTestJob.TestType.VecScalarFalse }.Run();
    }

    [Test]
    public void MatScalarTrueTest()
    {
        new SelectRefTestJob() { Type = SelectRefTestJob.TestType.MatScalarTrue }.Run();
    }

    [Test]
    public void MatScalarFalseTest()
    {
        new SelectRefTestJob() { Type = SelectRefTestJob.TestType.MatScalarFalse }.Run();
    }

    [Test]
    public void VecAliasDestTest()
    {
        new SelectRefTestJob() { Type = SelectRefTestJob.TestType.VecAliasDest }.Run();
    }
}
