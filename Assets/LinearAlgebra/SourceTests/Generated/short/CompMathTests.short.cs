using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Tests for the shortComp elementwise math surface (abs / min / max / relu / mad) over the SIGNED
// integer family (int / short / long). Oracles are exact integer expressions - no tolerance. Values
// are kept small so the same template is correct for short as well as int/long (no overflow).
// (uint is NOT expanded from this template - abs/relu have no unsigned meaning and are skipFor'd off
// the production kernel; uint's min/max/mad are covered concretely in SourceTests/UIntTypeTests.cs.)
public class shortCompMathTests
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
            var arena = new Arena(Allocator.Persistent);
            try
            {
                switch (Type)
                {
                    case TestType.Abs: AbsTest(ref arena); break;
                    case TestType.Relu: ReluTest(ref arena); break;
                    case TestType.MinBuf: MinBufTest(ref arena); break;
                    case TestType.MaxBuf: MaxBufTest(ref arena); break;
                    case TestType.Mad: MadTest(ref arena); break;
                    case TestType.SingleElement: SingleElementTest(ref arena); break;
                    default: throw new NotImplementedException();
                }
            }
            finally
            {
                arena.Dispose();
            }
        }

        private void AbsTest(ref Arena arena)
        {
            int n = 11;
            shortN v = arena.shortVec(n);
            for (int i = 0; i < n; i++)
                v[i] = (short)(i - 5); // -5 .. 5, includes 0

            v.absInPlace();

            for (int i = 0; i < n; i++)
            {
                short o = (short)(i - 5);
                short expected = o < 0 ? (short)(-o) : o;
                Assert.IsTrue(v[i] == expected);
            }
        }

        private void ReluTest(ref Arena arena)
        {
            int n = 11;
            shortN v = arena.shortVec(n);
            for (int i = 0; i < n; i++)
                v[i] = (short)(i - 5);

            v.reluInPlace();

            for (int i = 0; i < n; i++)
            {
                short o = (short)(i - 5);
                short expected = o < 0 ? (short)0 : o;
                Assert.IsTrue(v[i] == expected);
            }
        }

        private void MinBufTest(ref Arena arena)
        {
            int n = 11;
            shortN x = arena.shortVec(n);
            shortN y = arena.shortVec(n);
            for (int i = 0; i < n; i++)
            {
                x[i] = (short)(i - 5);   // -5 .. 5
                y[i] = (short)(5 - i);   // 5 .. -5  (crosses x)
            }
            shortN y0 = y.Copy();

            x.minInPlace(y); // x = min(x, y) ; y untouched

            for (int i = 0; i < n; i++)
            {
                short a = (short)(i - 5), b = (short)(5 - i);
                Assert.IsTrue(x[i] == (a < b ? a : b));
                Assert.IsTrue(y[i] == y0[i]);
            }
        }

        private void MaxBufTest(ref Arena arena)
        {
            int n = 11;
            shortN x = arena.shortVec(n);
            shortN y = arena.shortVec(n);
            for (int i = 0; i < n; i++)
            {
                x[i] = (short)(i - 5);
                y[i] = (short)(5 - i);
            }
            shortN y0 = y.Copy();

            x.maxInPlace(y); // x = max(x, y) ; y untouched

            for (int i = 0; i < n; i++)
            {
                short a = (short)(i - 5), b = (short)(5 - i);
                Assert.IsTrue(x[i] == (a > b ? a : b));
                Assert.IsTrue(y[i] == y0[i]);
            }
        }

        private void MadTest(ref Arena arena)
        {
            int n = 12;
            shortN a = arena.shortVec(n);
            shortN b = arena.shortVec(n);
            shortN c = arena.shortVec(n);
            for (int i = 0; i < n; i++)
            {
                a[i] = (short)((i % 3) - 1); // -1,0,1
                b[i] = (short)((i % 2) + 1); // 1,2
                c[i] = (short)(i % 3);       // 0,1,2
            }
            shortN a0 = a.Copy();
            shortN b0 = b.Copy();
            shortN c0 = c.Copy();

            a.madInPlace(b, c); // a = a*b + c ; ONLY a mutated

            for (int i = 0; i < n; i++)
            {
                Assert.IsTrue(a[i] == (short)(a0[i] * b0[i] + c0[i]));
                Assert.IsTrue(b[i] == b0[i]);
                Assert.IsTrue(c[i] == c0[i]);
            }
        }

        private void SingleElementTest(ref Arena arena)
        {
            shortN v = arena.shortVec(1);
            v[0] = (short)(-4);
            v.absInPlace();
            Assert.IsTrue(v[0] == (short)4);

            shortN a = arena.shortVec(1, (short)3);
            shortN b = arena.shortVec(1, (short)5);
            shortN c = arena.shortVec(1, (short)2);
            a.madInPlace(b, c);
            Assert.IsTrue(a[0] == (short)17); // 3*5 + 2
            Assert.IsTrue(b[0] == (short)5);
            Assert.IsTrue(c[0] == (short)2);
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
