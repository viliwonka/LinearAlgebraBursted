using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Integer-family mirror of fProxySelectRefTests (SelectRefTests.fProxy.cs). select() is exact data
// movement (dst[i] = c[i] ? b[i] : a[i]) - no arithmetic, no rounding - so results are compared for
// EXACT equality via the componentwise `==` operator + Analysis.IsAllEqualTo, not a tolerance
// check. (uint is NOT expanded from this template - see SourceTests/UIntTypeTests.cs for a couple
// of hand-written uint select cases alongside its other hand-authored unsigned coverage.)
public class iProxySelectRefTests
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
            int N = 17;

            var a = GenerateOP.iProxyRandomVec(N, (iProxy)(-100), (iProxy)100, 11111);
            var b = GenerateOP.iProxyRandomVec(N, (iProxy)(-100), (iProxy)100, 22222);
            var c = GenerateOP.boolRandomVec(N, 33333);

            // allocating reference
            var R = Select.select(a, b, c);

            // ref-dest into a preallocated destination
            var D = new iProxyN(N, Allocator.Temp);
            Select.select(in a, in b, in c, ref D);

            Assert.IsTrue(Analysis.IsAllEqualTo(R == D, true));
        }

        // Same select(a,b,c) formula as VecCond, for boolMxN cond (matrix).
        void MatCond()
        {
            int M = 6;
            int N = 9;

            var a = GenerateOP.iProxyRandomMat(M, N, (iProxy)(-100), (iProxy)100, 44444);
            var b = GenerateOP.iProxyRandomMat(M, N, (iProxy)(-100), (iProxy)100, 55555);
            var c = GenerateOP.boolRandomMat(M, N, 66666);

            var R = Select.select(a, b, c);

            var D = new iProxyMxN(M, N, Allocator.Temp);
            Select.select(in a, in b, in c, ref D);

            Assert.IsTrue(Analysis.IsAllEqualTo(R == D, true));
        }

        // ---- scalar-bool condition: c=true -> dest must equal b; c=false -> dest must equal a
        //      (vector then matrix) ----
        void VecScalarTrue()
        {
            int N = 13;

            var a = GenerateOP.iProxyRandomVec(N, (iProxy)(-100), (iProxy)100, 77777);
            var b = GenerateOP.iProxyRandomVec(N, (iProxy)(-100), (iProxy)100, 88888);

            var D = new iProxyN(N, Allocator.Temp);
            Select.select(in a, in b, true, ref D);

            Assert.IsTrue(Analysis.IsAllEqualTo(b == D, true));
        }

        void VecScalarFalse()
        {
            int N = 13;

            var a = GenerateOP.iProxyRandomVec(N, (iProxy)(-100), (iProxy)100, 99999);
            var b = GenerateOP.iProxyRandomVec(N, (iProxy)(-100), (iProxy)100, 10101);

            var D = new iProxyN(N, Allocator.Temp);
            Select.select(in a, in b, false, ref D);

            Assert.IsTrue(Analysis.IsAllEqualTo(a == D, true));
        }

        void MatScalarTrue()
        {
            int M = 5;
            int N = 7;

            var a = GenerateOP.iProxyRandomMat(M, N, (iProxy)(-100), (iProxy)100, 20202);
            var b = GenerateOP.iProxyRandomMat(M, N, (iProxy)(-100), (iProxy)100, 30303);

            var D = new iProxyMxN(M, N, Allocator.Temp);
            Select.select(in a, in b, true, ref D);

            Assert.IsTrue(Analysis.IsAllEqualTo(b == D, true));
        }

        void MatScalarFalse()
        {
            int M = 5;
            int N = 7;

            var a = GenerateOP.iProxyRandomMat(M, N, (iProxy)(-100), (iProxy)100, 40404);
            var b = GenerateOP.iProxyRandomMat(M, N, (iProxy)(-100), (iProxy)100, 50505);

            var D = new iProxyMxN(M, N, Allocator.Temp);
            Select.select(in a, in b, false, ref D);

            Assert.IsTrue(Analysis.IsAllEqualTo(a == D, true));
        }

        // Elementwise aliasing IS allowed (no guard): select(a, b, c, ref a) must match the
        // allocating select(a, b, c). Compute the reference BEFORE the aliased call (which
        // overwrites a) and confirm aliasing does not corrupt the result.
        void VecAliasDest()
        {
            int N = 21;

            var a = GenerateOP.iProxyRandomVec(N, (iProxy)(-100), (iProxy)100, 60606);
            var b = GenerateOP.iProxyRandomVec(N, (iProxy)(-100), (iProxy)100, 70707);
            var c = GenerateOP.boolRandomVec(N, 80808);

            // Reference into a SEPARATE buffer before a is overwritten.
            var R = new iProxyN(N, Allocator.Temp);
            Select.select(in a, in b, in c, ref R);

            // Now alias the destination onto input a.
            Select.select(in a, in b, in c, ref a);

            Assert.IsTrue(Analysis.IsAllEqualTo(R == a, true));
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
