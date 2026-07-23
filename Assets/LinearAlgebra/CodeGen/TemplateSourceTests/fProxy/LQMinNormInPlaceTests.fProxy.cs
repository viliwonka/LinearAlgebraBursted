using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// LQ.minNormSolveInPlace (vector, workspace, multi-RHS): must produce the same solution as the
// A-preserving minNormSolve on the same system (same kernel over identical values), destroy A,
// and leave b/B untouched.
public class fProxyLQMinNormInPlaceTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            VectorEquivalence,
            WorkspaceEquivalence,
            MultiRhsEquivalence,
        }

        public TestType Type;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.VectorEquivalence:    VectorEquivalence();    break;
                case TestType.WorkspaceEquivalence: WorkspaceEquivalence(); break;
                case TestType.MultiRhsEquivalence:  MultiRhsEquivalence();  break;
            }
        }

        static fProxy Tol() => /*+choose[1e-5f|1e-12]*/1e-5f/*-choose*/;

        // Wide full-row-rank test matrix: random with a diagonal boost.
        static fProxyMxN BuildA(int m, int n, uint seed)
        {
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, seed);
            for (int d = 0; d < m; d++)
                A[d, d] += (fProxy)10;
            return A;
        }

        void VectorEquivalence()
        {
            int m = 6, n = 12;
            var A = BuildA(m, n, 91001);
            var b = GenerateOP.fProxyRandomVec(m, -5f, 5f, 91002);

            var xRef = new fProxyN(n, Allocator.Temp);
            LQ.minNormSolve(in A, in b, ref xRef);

            var Ainp = new fProxyMxN(in A, Allocator.Temp);
            var x = new fProxyN(n, Allocator.Temp);
            LQ.minNormSolveInPlace(ref Ainp, in b, ref x);

            var dx = new fProxyN(in xRef, Allocator.Temp);
            fProxyComp.subInPlace(dx, x);
            Assert.IsTrue(Analysis.isZero(dx, Tol()));
            // b untouched by the in-place solve.
            var b2 = GenerateOP.fProxyRandomVec(m, -5f, 5f, 91002);
            fProxyComp.subInPlace(b2, b);
            Assert.IsTrue(Analysis.isZero(b2, (fProxy)0));
        }

        void WorkspaceEquivalence()
        {
            int m = 6, n = 12;
            var A = BuildA(m, n, 92001);
            var b = GenerateOP.fProxyRandomVec(m, -5f, 5f, 92002);

            var xRef = new fProxyN(n, Allocator.Temp);
            LQ.minNormSolve(in A, in b, ref xRef);

            var Ainp = new fProxyMxN(in A, Allocator.Temp);
            var x = new fProxyN(n, Allocator.Temp);
            var ws = new fProxyLQMinNormCache(m, n, Allocator.Temp);
            LQ.minNormSolveInPlace(ref Ainp, in b, ref x, ref ws);

            var dx = new fProxyN(in xRef, Allocator.Temp);
            fProxyComp.subInPlace(dx, x);
            Assert.IsTrue(Analysis.isZero(dx, Tol()));
        }

        void MultiRhsEquivalence()
        {
            int m = 5, n = 9, k = 3;
            var A = BuildA(m, n, 93001);
            var B = GenerateOP.fProxyRandomMat(m, k, -3f, 3f, 93002);

            var XRef = new fProxyMxN(n, k, Allocator.Temp);
            LQ.minNormSolve(in A, in B, ref XRef);

            var Ainp = new fProxyMxN(in A, Allocator.Temp);
            var X = new fProxyMxN(n, k, Allocator.Temp);
            LQ.minNormSolveInPlace(ref Ainp, in B, ref X);

            var dX = new fProxyMxN(in XRef, Allocator.Temp);
            fProxyComp.subInPlace(dX, X);
            Assert.IsTrue(Analysis.isZero(dX, Tol()));
            // B untouched by the in-place solve.
            var B2 = GenerateOP.fProxyRandomMat(m, k, -3f, 3f, 93002);
            fProxyComp.subInPlace(B2, B);
            Assert.IsTrue(Analysis.isZero(B2, (fProxy)0));
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void Test(TestJob.TestType type)
    {
        new TestJob() { Type = type }.Run();
    }
}
