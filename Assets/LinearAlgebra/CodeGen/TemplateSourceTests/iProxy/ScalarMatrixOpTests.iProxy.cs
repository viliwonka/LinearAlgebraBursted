using System;

using BULA;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// scalar - matrix must equal s - A[i,j] elementwise (subtraction is not commutative).
public class iProxyScalarMatrixOpTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct TestJob : IJob
    {
        // [0] flag, [1] got, [2] expected
        public NativeArray<iProxy> Fail;

        public void Execute()
        {
            // 5 - [[1,2],[3,4]] must be [[4,3],[2,1]] (NOT [[-4,-3],[-2,-1]]).
            var A = new iProxyMxN(2, 2, Allocator.Temp);
            A[0, 0] = (iProxy)1; A[0, 1] = (iProxy)2;
            A[1, 0] = (iProxy)3; A[1, 1] = (iProxy)4;

            iProxyMxN R = new iProxyMxN(in A, Allocator.Temp);
            iProxyComp.subInPlace((iProxy)5, R);

            AssertEqual(R[0, 0], (iProxy)4);
            AssertEqual(R[0, 1], (iProxy)3);
            AssertEqual(R[1, 0], (iProxy)2);
            AssertEqual(R[1, 1], (iProxy)1);
        }

        void AssertEqual(iProxy got, iProxy expected)
        {
            if (!(got == expected) && Fail[0] == (iProxy)0)
            {
                Fail[0] = (iProxy)1; Fail[1] = got; Fail[2] = expected;
            }
            Assert.IsTrue(got == expected);
        }
    }

    [Test]
    public void ScalarMinusMatrix()
    {
        var fail = new NativeArray<iProxy>(3, Allocator.TempJob);
        try
        {
            new TestJob() { Fail = fail }.Run();
            if (fail[0] != (iProxy)0)
                Assert.Fail($"scalar - matrix: got {fail[1]}, expected {fail[2]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (iProxy)0)
                Assert.Fail($"scalar - matrix: got {fail[1]}, expected {fail[2]} ({e.Message})");
            throw;
        }
        finally { fail.Dispose(); }
    }
}
