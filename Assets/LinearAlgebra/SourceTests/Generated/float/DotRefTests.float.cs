using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

public class floatDotRefTests
{
    [BurstCompile]
    public struct DotRefTestJob : IJob
    {
        public enum TestType
        {
            OuterDot,
            MatVec,
            VecMat,
            MatMat,
            MatMatTransA,
            Trans,
            DirtyDest,
        }

        public TestType Type;

        // The ref-dest form runs the SAME kernel as the allocating form, so results are
        // bit-identical in principle. Use a small per-precision tolerance anyway
        // (1e-6-ish float, tighter for double) to stay robust across expansions.
        static float Tol() => 256 * Consts.floatSqrtEps;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.OuterDot:
                    OuterDot();
                    break;
                case TestType.MatVec:
                    MatVec();
                    break;
                case TestType.VecMat:
                    VecMat();
                    break;
                case TestType.MatMat:
                    MatMat();
                    break;
                case TestType.MatMatTransA:
                    MatMatTransA();
                    break;
                case TestType.Trans:
                    Trans();
                    break;
                case TestType.DirtyDest:
                    DirtyDest();
                    break;
            }
        }

        // The accumulating dot kernels (+=) require a zeroed destination. Feed a
        // pre-filled (reused) destination and confirm the ref form still matches the
        // allocating result, i.e. it zeroes before accumulating. This is the case the
        // ref==alloc equality test misses when the destination is freshly allocated.
        void DirtyDest()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 6;
            int N = 4;
            int K = 5;

            // mat·vec
            {
                var A = arena.floatRandomMatrix(M, N, -1f, 1f, 12321);
                var x = arena.floatRandomVector(N, -1f, 1f, 45654);
                var R = float_OP.dot(A, x);

                var D = arena.floatVec(M);
                float_OP.addInpl(D, (float)999);   // dirty the destination
                float_OP.dot(in A, in x, ref D);
                Assert.IsTrue(Analysis_OP.IsZero(R - D, Tol()));
            }

            // vec·mat
            {
                var y = arena.floatRandomVector(M, -1f, 1f, 11221);
                var A = arena.floatRandomMatrix(M, N, -1f, 1f, 33443);
                var R = float_OP.dot(y, A);

                var D = arena.floatVec(N);
                float_OP.addInpl(D, (float)999);
                float_OP.dot(in y, in A, ref D);
                Assert.IsTrue(Analysis_OP.IsZero(R - D, Tol()));
            }

            // mat·mat
            {
                var a = arena.floatRandomMatrix(M, K, -1f, 1f, 32123);
                var b = arena.floatRandomMatrix(K, N, -1f, 1f, 65456);
                var R = float_OP.dot(a, b, false);

                var D = arena.floatMat(M, N);
                float_OP.addInpl(D, (float)999);
                float_OP.dot(in a, in b, ref D, false);
                Assert.IsTrue(Analysis_OP.IsZero(R - D, Tol()));
            }

            arena.Dispose();
        }

        // outer product: a (col, length M) * b (row, length N) -> M x N
        void OuterDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 5;
            int N = 7;

            var a = arena.floatRandomVector(M, -1f, 1f, 11111);
            var b = arena.floatRandomVector(N, -1f, 1f, 22222);

            // allocating reference
            var R = float_OP.outerDot(a, b);

            // ref-dest into a preallocated M x N destination
            var D = arena.floatMat(M, N);
            float_OP.outerDot(in a, in b, ref D);

            Assert.IsTrue(Analysis_OP.IsZero(R - D, Tol()));

            arena.Dispose();
        }

        // matrix (M x N) * vector (length N) -> vector (length M)
        void MatVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 6;
            int N = 4;

            var A = arena.floatRandomMatrix(M, N, -1f, 1f, 33333);
            var x = arena.floatRandomVector(N, -1f, 1f, 44444);

            var R = float_OP.dot(A, x);

            var D = arena.floatVec(M);
            float_OP.dot(in A, in x, ref D);

            Assert.IsTrue(Analysis_OP.IsZero(R - D, Tol()));

            arena.Dispose();
        }

        // vector (length M) * matrix (M x N) -> vector (length N)
        void VecMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 6;
            int N = 4;

            var y = arena.floatRandomVector(M, -1f, 1f, 55555);
            var A = arena.floatRandomMatrix(M, N, -1f, 1f, 66666);

            var R = float_OP.dot(y, A);

            var D = arena.floatVec(N);
            float_OP.dot(in y, in A, ref D);

            Assert.IsTrue(Analysis_OP.IsZero(R - D, Tol()));

            arena.Dispose();
        }

        // matrix (M x K) * matrix (K x N) -> matrix (M x N), transposeA = false
        void MatMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 5;
            int K = 3;
            int N = 7;

            var a = arena.floatRandomMatrix(M, K, -1f, 1f, 77777);
            var b = arena.floatRandomMatrix(K, N, -1f, 1f, 88888);

            var R = float_OP.dot(a, b, false);

            var D = arena.floatMat(M, N);
            float_OP.dot(in a, in b, ref D, false);

            Assert.IsTrue(Analysis_OP.IsZero(R - D, Tol()));

            arena.Dispose();
        }

        // matrix Aᵀ * matrix b: a is (K x M), b is (K x N) -> (M x N), transposeA = true
        void MatMatTransA()
        {
            var arena = new Arena(Allocator.Persistent);

            int K = 4;
            int M = 5;
            int N = 6;

            var a = arena.floatRandomMatrix(K, M, -1f, 1f, 99999);
            var b = arena.floatRandomMatrix(K, N, -1f, 1f, 10101);

            var R = float_OP.dot(a, b, true);

            // result is M x N (a.N_Cols x b.N_Cols)
            var D = arena.floatMat(M, N);
            float_OP.dot(in a, in b, ref D, true);

            // ref == allocating (delegation check)
            Assert.IsTrue(Analysis_OP.IsZero(R - D, Tol()));

            // Independent oracle: Aᵀ·B computed via an explicit transpose + plain matmul,
            // which exercises a different code path than the fused transposeA kernel.
            var oracle = float_OP.dot(float_OP.trans(a), b);
            Assert.IsTrue(Analysis_OP.IsZero(R - oracle, Tol()));

            arena.Dispose();
        }

        // transpose of a non-square matrix (M x N) -> (N x M)
        void Trans()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 5;
            int N = 8;

            var A = arena.floatRandomMatrix(M, N, -1f, 1f, 20202);

            var R = float_OP.trans(A);

            var D = arena.floatMat(N, M);
            float_OP.trans(in A, ref D);

            Assert.IsTrue(Analysis_OP.IsZero(R - D, Tol()));

            arena.Dispose();
        }
    }

    [Test]
    public void OuterDotTest()
    {
        new DotRefTestJob() { Type = DotRefTestJob.TestType.OuterDot }.Run();
    }

    [Test]
    public void MatVecTest()
    {
        new DotRefTestJob() { Type = DotRefTestJob.TestType.MatVec }.Run();
    }

    [Test]
    public void VecMatTest()
    {
        new DotRefTestJob() { Type = DotRefTestJob.TestType.VecMat }.Run();
    }

    [Test]
    public void MatMatTest()
    {
        new DotRefTestJob() { Type = DotRefTestJob.TestType.MatMat }.Run();
    }

    [Test]
    public void MatMatTransATest()
    {
        new DotRefTestJob() { Type = DotRefTestJob.TestType.MatMatTransA }.Run();
    }

    [Test]
    public void TransTest()
    {
        new DotRefTestJob() { Type = DotRefTestJob.TestType.Trans }.Run();
    }

    [Test]
    public void DirtyDestTest()
    {
        new DotRefTestJob() { Type = DotRefTestJob.TestType.DirtyDest }.Run();
    }
}
