using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Tests for the iProxyComp elementwise math surface (abs / min / max / relu / mad) over the SIGNED
// integer family (int / short / long). Oracles are exact integer expressions - no tolerance. Values
// are kept small so the same template is correct for short as well as int/long (no overflow).
// (uint is NOT expanded from this template - abs/relu have no unsigned meaning and are skipFor'd off
// the production kernel; uint's min/max/mad are covered concretely in SourceTests/UIntTypeTests.cs.)
public class iProxyCompMathTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct TestsJob : IJob
    {
        public enum TestType
        {
            Abs, Relu, MinBuf, MaxBuf, Mad,
            SingleElement,
        }

        public TestType Type;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.Abs: AbsTest(); break;
                case TestType.Relu: ReluTest(); break;
                case TestType.MinBuf: MinBufTest(); break;
                case TestType.MaxBuf: MaxBufTest(); break;
                case TestType.Mad: MadTest(); break;
                case TestType.SingleElement: SingleElementTest(); break;
                default: throw new NotImplementedException();
            }
        }

        private void AbsTest()
        {
            int n = 11;
            iProxyN v = new iProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++)
                v[i] = (iProxy)(i - 5); // -5 .. 5, includes 0

            v.absInPlace();

            for (int i = 0; i < n; i++)
            {
                iProxy o = (iProxy)(i - 5);
                iProxy expected = o < 0 ? (iProxy)(-o) : o;
                Assert.IsTrue(v[i] == expected);
            }
        }

        private void ReluTest()
        {
            int n = 11;
            iProxyN v = new iProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++)
                v[i] = (iProxy)(i - 5);

            v.reluInPlace();

            for (int i = 0; i < n; i++)
            {
                iProxy o = (iProxy)(i - 5);
                iProxy expected = o < 0 ? (iProxy)0 : o;
                Assert.IsTrue(v[i] == expected);
            }
        }

        private void MinBufTest()
        {
            int n = 11;
            iProxyN x = new iProxyN(n, Allocator.Temp);
            iProxyN y = new iProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++)
            {
                x[i] = (iProxy)(i - 5);   // -5 .. 5
                y[i] = (iProxy)(5 - i);   // 5 .. -5  (crosses x)
            }
            iProxyN y0 = y.Copy();

            x.minInPlace(y); // x = min(x, y) ; y untouched

            for (int i = 0; i < n; i++)
            {
                iProxy a = (iProxy)(i - 5), b = (iProxy)(5 - i);
                Assert.IsTrue(x[i] == (a < b ? a : b));
                Assert.IsTrue(y[i] == y0[i]);
            }
        }

        private void MaxBufTest()
        {
            int n = 11;
            iProxyN x = new iProxyN(n, Allocator.Temp);
            iProxyN y = new iProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++)
            {
                x[i] = (iProxy)(i - 5);
                y[i] = (iProxy)(5 - i);
            }
            iProxyN y0 = y.Copy();

            x.maxInPlace(y); // x = max(x, y) ; y untouched

            for (int i = 0; i < n; i++)
            {
                iProxy a = (iProxy)(i - 5), b = (iProxy)(5 - i);
                Assert.IsTrue(x[i] == (a > b ? a : b));
                Assert.IsTrue(y[i] == y0[i]);
            }
        }

        private void MadTest()
        {
            int n = 12;
            iProxyN a = new iProxyN(n, Allocator.Temp);
            iProxyN b = new iProxyN(n, Allocator.Temp);
            iProxyN c = new iProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++)
            {
                a[i] = (iProxy)((i % 3) - 1); // -1,0,1
                b[i] = (iProxy)((i % 2) + 1); // 1,2
                c[i] = (iProxy)(i % 3);       // 0,1,2
            }
            iProxyN a0 = a.Copy();
            iProxyN b0 = b.Copy();
            iProxyN c0 = c.Copy();

            a.madInPlace(b, c); // a = a*b + c ; ONLY a mutated

            for (int i = 0; i < n; i++)
            {
                Assert.IsTrue(a[i] == (iProxy)(a0[i] * b0[i] + c0[i]));
                Assert.IsTrue(b[i] == b0[i]);
                Assert.IsTrue(c[i] == c0[i]);
            }
        }

        private void SingleElementTest()
        {
            iProxyN v = new iProxyN(1, Allocator.Temp);
            v[0] = (iProxy)(-4);
            v.absInPlace();
            Assert.IsTrue(v[0] == (iProxy)4);

            iProxyN a = GenerateOP.iProxyVec(1, (iProxy)3);
            iProxyN b = GenerateOP.iProxyVec(1, (iProxy)5);
            iProxyN c = GenerateOP.iProxyVec(1, (iProxy)2);
            a.madInPlace(b, c);
            Assert.IsTrue(a[0] == (iProxy)17); // 3*5 + 2
            Assert.IsTrue(b[0] == (iProxy)5);
            Assert.IsTrue(c[0] == (iProxy)2);
        }
    }

    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(TestsJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void Test(TestsJob.TestType type)
    {
        new TestsJob() { Type = type }.Run();
    }
}
