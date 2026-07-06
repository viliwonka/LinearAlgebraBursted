using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

public class shortDotRefTests
{
    [BurstCompile(CompileSynchronously = true)]
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

        // Integer arithmetic is EXACT: the ref-dest form runs the same kernel as the
        // allocating form, so the two results must be bit-for-bit identical. There is no
        // short Analysis.isZero, so assert exact elementwise equality directly.
        static bool ExactEqual(in shortN x, in shortN y)
        {
            if (x.N != y.N)
                return false;
            for (int i = 0; i < x.N; i++)
                if (x[i] != y[i])
                    return false;
            return true;
        }

        static bool ExactEqual(in shortMxN x, in shortMxN y)
        {
            if (x.M_Rows != y.M_Rows || x.N_Cols != y.N_Cols)
                return false;
            for (int i = 0; i < x.Length; i++)
                if (x[i] != y[i])
                    return false;
            return true;
        }

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
                var A = arena.shortRandomMat(M, N, -9, 9, 12321);
                var x = arena.shortRandomVec(N, -9, 9, 45654);
                var R = Blas.dot(A, x);

                var D = arena.shortVec(M);
                shortComp.addInPlace(D, (short)999);   // dirty the destination
                Blas.dot(in A, in x, ref D);
                Assert.IsTrue(ExactEqual(in R, in D));
            }

            // vec·mat
            {
                var y = arena.shortRandomVec(M, -9, 9, 11221);
                var A = arena.shortRandomMat(M, N, -9, 9, 33443);
                var R = Blas.dot(y, A);

                var D = arena.shortVec(N);
                shortComp.addInPlace(D, (short)999);
                Blas.dot(in y, in A, ref D);
                Assert.IsTrue(ExactEqual(in R, in D));
            }

            // mat·mat
            {
                var a = arena.shortRandomMat(M, K, -9, 9, 32123);
                var b = arena.shortRandomMat(K, N, -9, 9, 65456);
                var R = Blas.dot(a, b, false);

                var D = arena.shortMat(M, N);
                shortComp.addInPlace(D, (short)999);
                Blas.dot(in a, in b, ref D, false);
                Assert.IsTrue(ExactEqual(in R, in D));
            }

            arena.Dispose();
        }

        // outer product: a (col, length M) * b (row, length N) -> M x N
        void OuterDot()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 5;
            int N = 7;

            var a = arena.shortRandomVec(M, -9, 9, 11111);
            var b = arena.shortRandomVec(N, -9, 9, 22222);

            // allocating reference
            var R = Blas.outerDot(a, b);

            // ref-dest into a preallocated M x N destination
            var D = arena.shortMat(M, N);
            Blas.outerDot(in a, in b, ref D);

            Assert.IsTrue(ExactEqual(in R, in D));

            arena.Dispose();
        }

        // matrix (M x N) * vector (length N) -> vector (length M)
        void MatVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 6;
            int N = 4;

            var A = arena.shortRandomMat(M, N, -9, 9, 33333);
            var x = arena.shortRandomVec(N, -9, 9, 44444);

            var R = Blas.dot(A, x);

            var D = arena.shortVec(M);
            Blas.dot(in A, in x, ref D);

            Assert.IsTrue(ExactEqual(in R, in D));

            arena.Dispose();
        }

        // vector (length M) * matrix (M x N) -> vector (length N)
        void VecMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 6;
            int N = 4;

            var y = arena.shortRandomVec(M, -9, 9, 55555);
            var A = arena.shortRandomMat(M, N, -9, 9, 66666);

            var R = Blas.dot(y, A);

            var D = arena.shortVec(N);
            Blas.dot(in y, in A, ref D);

            Assert.IsTrue(ExactEqual(in R, in D));

            arena.Dispose();
        }

        // matrix (M x K) * matrix (K x N) -> matrix (M x N), transposeA = false
        void MatMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 5;
            int K = 3;
            int N = 7;

            var a = arena.shortRandomMat(M, K, -9, 9, 77777);
            var b = arena.shortRandomMat(K, N, -9, 9, 88888);

            var R = Blas.dot(a, b, false);

            var D = arena.shortMat(M, N);
            Blas.dot(in a, in b, ref D, false);

            Assert.IsTrue(ExactEqual(in R, in D));

            arena.Dispose();
        }

        // matrix Aᵀ * matrix b: a is (K x M), b is (K x N) -> (M x N), transposeA = true.
        // K is the contracted dim = a.M_Rows = b.M_Rows.
        void MatMatTransA()
        {
            var arena = new Arena(Allocator.Persistent);

            int K = 4;
            int M = 5;
            int N = 6;

            var a = arena.shortRandomMat(K, M, -9, 9, 99999);
            var b = arena.shortRandomMat(K, N, -9, 9, 10101);

            var R = Blas.dot(a, b, true);

            // result is M x N (a.N_Cols x b.N_Cols)
            var D = arena.shortMat(M, N);
            Blas.dot(in a, in b, ref D, true);

            // ref == allocating (delegation check)
            Assert.IsTrue(ExactEqual(in R, in D));

            // Independent oracle: Aᵀ·B computed via an explicit transpose + plain matmul,
            // which exercises a different code path than the fused transposeA kernel.
            var oracle = Blas.dot(Blas.trans(a), b);
            Assert.IsTrue(ExactEqual(in R, in oracle));

            arena.Dispose();
        }

        // transpose of a non-square matrix (M x N) -> (N x M)
        void Trans()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 5;
            int N = 8;

            var A = arena.shortRandomMat(M, N, -9, 9, 20202);

            var R = Blas.trans(A);

            var D = arena.shortMat(N, M);
            Blas.trans(in A, ref D);

            Assert.IsTrue(ExactEqual(in R, in D));

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
