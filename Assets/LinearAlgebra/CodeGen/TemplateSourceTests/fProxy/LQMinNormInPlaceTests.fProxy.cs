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
        static fProxyMxN BuildA(ref Arena arena, int m, int n, uint seed)
        {
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, seed);
            for (int d = 0; d < m; d++)
                A[d, d] += (fProxy)10;
            return A;
        }

        void VectorEquivalence()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 6, n = 12;
            var A = BuildA(ref arena, m, n, 91001);
            var b = arena.fProxyRandomVec(m, -5f, 5f, 91002);

            var xRef = arena.fProxyVec(n);
            LQ.minNormSolve(in A, in b, ref xRef);

            var Ainp = A.Copy();
            var x = arena.fProxyVec(n);
            LQ.minNormSolveInPlace(ref Ainp, in b, ref x);

            Assert.IsTrue(Analysis.isZero(xRef - x, Tol()));
            // b untouched by the in-place solve.
            var b2 = arena.fProxyRandomVec(m, -5f, 5f, 91002);
            Assert.IsTrue(Analysis.isZero(b - b2, (fProxy)0));

            arena.Dispose();
        }

        void WorkspaceEquivalence()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 6, n = 12;
            var A = BuildA(ref arena, m, n, 92001);
            var b = arena.fProxyRandomVec(m, -5f, 5f, 92002);

            var xRef = arena.fProxyVec(n);
            LQ.minNormSolve(in A, in b, ref xRef);

            var Ainp = A.Copy();
            var x = arena.fProxyVec(n);
            var ws = arena.fProxyLQMinNormCache(m, n);
            LQ.minNormSolveInPlace(ref Ainp, in b, ref x, ref ws);

            Assert.IsTrue(Analysis.isZero(xRef - x, Tol()));

            arena.Dispose();
        }

        void MultiRhsEquivalence()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 5, n = 9, k = 3;
            var A = BuildA(ref arena, m, n, 93001);
            var B = arena.fProxyRandomMat(m, k, -3f, 3f, 93002);

            var XRef = arena.fProxyMat(n, k);
            LQ.minNormSolve(in A, in B, ref XRef);

            var Ainp = A.Copy();
            var X = arena.fProxyMat(n, k);
            LQ.minNormSolveInPlace(ref Ainp, in B, ref X);

            Assert.IsTrue(Analysis.isZero(XRef - X, Tol()));
            // B untouched by the in-place solve.
            var B2 = arena.fProxyRandomMat(m, k, -3f, 3f, 93002);
            Assert.IsTrue(Analysis.isZero(B - B2, (fProxy)0));

            arena.Dispose();
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void Test(TestJob.TestType type)
    {
        new TestJob() { Type = type }.Run();
    }
}
