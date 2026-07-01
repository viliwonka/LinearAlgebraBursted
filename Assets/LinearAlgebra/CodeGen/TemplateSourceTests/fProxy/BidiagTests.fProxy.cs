using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

public class fProxyBidiagTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            IdentitySquare,
            DiagonalSquare,
            RandomSquare6x6,
            RandomSquare8x8,
            RandomTall10x6,
            RandomTall12x4,
            RandomTall5x1,
            ValuesMatchFullSquare,
            ValuesMatchFullTall,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<fProxy> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.IdentitySquare:    IdentitySquare();    break;
                case TestType.DiagonalSquare:    DiagonalSquare();    break;
                case TestType.RandomSquare6x6:   RandomSquare6x6();   break;
                case TestType.RandomSquare8x8:   RandomSquare8x8();   break;
                case TestType.RandomTall10x6:    RandomTall10x6();    break;
                case TestType.RandomTall12x4:    RandomTall12x4();    break;
                case TestType.RandomTall5x1:     RandomTall5x1();     break;
                case TestType.ValuesMatchFullSquare: CheckValuesMatchFull(7, 7, (fProxy)(-3f), (fProxy)3f, 161803); break;
                case TestType.ValuesMatchFullTall:   CheckValuesMatchFull(11, 5, (fProxy)(-2f), (fProxy)2f, 424242); break;
            }
        }

        // ---- helpers ----

        // Assert |got - expected| <= tol for a scalar pair
        private void AssertClose(fProxy got, fProxy expected, fProxy tol)
        {
            fProxy diff = Unity.Mathematics.math.abs(got - expected);
            if (!(diff <= tol) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = got;
                Fail[2] = expected;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= tol);
        }

        // All entries of |M| <= tol
        private void AssertNearZero(in fProxyMxN M, fProxy tol, string context)
        {
            fProxy err = Analysis_OP.MaxZeroError(M);
            if (!(err <= tol) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = err;
                Fail[2] = tol;
                Fail[3] = err - tol;
            }
            Assert.IsTrue(Analysis_OP.isZero(in M, tol));
        }

        // Check B is upper bidiagonal: zero everywhere except B[k,k] and B[k,k+1]
        private void AssertUpperBidiagonal(in fProxyMxN B, fProxy tol)
        {
            int n = B.N_Cols;
            for (int r = 0; r < n; r++)
            for (int c = 0; c < n; c++)
            {
                bool allowed = (c == r) || (c == r + 1);
                if (!allowed)
                {
                    fProxy val = Unity.Mathematics.math.abs(B[r, c]);
                    if (!(val <= tol) && Fail[0] == (fProxy)0)
                    {
                        Fail[0] = (fProxy)1;
                        Fail[1] = val;
                        Fail[2] = tol;
                        Fail[3] = val - tol;
                    }
                    Assert.IsTrue(val <= tol);
                }
            }
        }

        // Full suite for a single bidiagonalization result:
        //  1. A ≈ U*B*Vᵀ  (reconstruction)
        //  2. B is upper bidiagonal
        //  3. UᵀU ≈ I_n   (orthonormal columns)
        //  4. VᵀV ≈ I_n   (orthogonal)
        private void AssertBidiag(in fProxyMxN A, in fProxyMxN U, in fProxyMxN B, in fProxyMxN V,
                                   ref Arena arena, fProxy tol)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            // 1. Reconstruction: A ≈ U * B * Vᵀ
            var Vt   = fProxy_OP.trans(V);
            var UB   = fProxy_OP.dot(U, B);
            var UBVt = fProxy_OP.dot(UB, Vt);
            var diff = A - UBVt;

            if (Analysis_OP.isAnyNan(in diff))
                throw new System.Exception("BidiagTests: NaN in reconstruction");

            AssertNearZero(in diff, tol, "reconstruction");

            // 2. B is upper bidiagonal
            AssertUpperBidiagonal(in B, tol);

            // 3. UᵀU ≈ I_n  (Analysis_OP.isOrthogonal handles thin U: computes AᵀA = I_n)
            Assert.IsTrue(Analysis_OP.isOrthogonal(in U, tol));

            // 4. VᵀV ≈ I_n  (V is square)
            Assert.IsTrue(Analysis_OP.isOrthogonal(in V, tol));
        }

        // ---- test cases ----

        void IdentitySquare()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 6;
            var A = arena.fProxyIdentityMat(n);
            var U = arena.fProxyMat(n, n);
            var B = arena.fProxyMat(n, n);
            var V = arena.fProxyMat(n, n);
            Bidiag.bidiagonalize(in A, ref U, ref B, ref V);
            AssertBidiag(in A, in U, in B, in V, ref arena, (fProxy)1E-4f);
            arena.Dispose();
        }

        void DiagonalSquare()
        {
            // Diagonal input: A is already bidiagonal; B should ≈ A (up to signs), U,V ≈ I (up to signs)
            var arena = new Arena(Allocator.Persistent);
            int n = 6;
            var A = arena.fProxyMat(n, n);
            A[0, 0] = (fProxy)3f;
            A[1, 1] = (fProxy)1f;
            A[2, 2] = (fProxy)4f;
            A[3, 3] = (fProxy)1f;
            A[4, 4] = (fProxy)5f;
            A[5, 5] = (fProxy)9f;
            var U = arena.fProxyMat(n, n);
            var B = arena.fProxyMat(n, n);
            var V = arena.fProxyMat(n, n);
            Bidiag.bidiagonalize(in A, ref U, ref B, ref V);
            AssertBidiag(in A, in U, in B, in V, ref arena, (fProxy)1E-4f);
            arena.Dispose();
        }

        void RandomSquare6x6()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 6;
            var A = arena.fProxyRandomMat(n, n, (fProxy)(-2f), (fProxy)2f, 314159);
            var U = arena.fProxyMat(n, n);
            var B = arena.fProxyMat(n, n);
            var V = arena.fProxyMat(n, n);
            Bidiag.bidiagonalize(in A, ref U, ref B, ref V);
            AssertBidiag(in A, in U, in B, in V, ref arena, (fProxy)1E-4f);
            arena.Dispose();
        }

        void RandomSquare8x8()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 8;
            var A = arena.fProxyRandomMat(n, n, (fProxy)(-5f), (fProxy)5f, 271828);
            var U = arena.fProxyMat(n, n);
            var B = arena.fProxyMat(n, n);
            var V = arena.fProxyMat(n, n);
            Bidiag.bidiagonalize(in A, ref U, ref B, ref V);
            AssertBidiag(in A, in U, in B, in V, ref arena, (fProxy)1E-4f);
            arena.Dispose();
        }

        void RandomTall10x6()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 10, n = 6;
            var A = arena.fProxyRandomMat(m, n, (fProxy)(-3f), (fProxy)3f, 112358);
            var U = arena.fProxyMat(m, n);
            var B = arena.fProxyMat(n, n);
            var V = arena.fProxyMat(n, n);
            Bidiag.bidiagonalize(in A, ref U, ref B, ref V);
            AssertBidiag(in A, in U, in B, in V, ref arena, (fProxy)1E-4f);
            arena.Dispose();
        }

        void RandomTall12x4()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 12, n = 4;
            var A = arena.fProxyRandomMat(m, n, (fProxy)(-1f), (fProxy)1f, 999421);
            var U = arena.fProxyMat(m, n);
            var B = arena.fProxyMat(n, n);
            var V = arena.fProxyMat(n, n);
            Bidiag.bidiagonalize(in A, ref U, ref B, ref V);
            AssertBidiag(in A, in U, in B, in V, ref arena, (fProxy)1E-4f);
            arena.Dispose();
        }

        // bidiagonalizeValues must produce EXACTLY the bidiagonal bands of the full bidiagonalize:
        // both use identical reflectors/applies, so d[k]=B[k,k], e[0]=0, e[k]=B[k-1,k].
        void CheckValuesMatchFull(int m, int n, fProxy lo, fProxy hi, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomMat(m, n, lo, hi, seed);
            var U = arena.fProxyMat(m, n);
            var B = arena.fProxyMat(n, n);
            var V = arena.fProxyMat(n, n);
            Bidiag.bidiagonalize(in A, ref U, ref B, ref V);

            var d = arena.fProxyVec(n);
            var e = arena.fProxyVec(n);
            Bidiag.bidiagonalizeValues(in A, ref d, ref e);

            for (int k = 0; k < n; k++)
                AssertClose(d[k], B[k, k], (fProxy)1E-4f);
            AssertClose(e[0], (fProxy)0, (fProxy)1E-4f);
            for (int k = 1; k < n; k++)
                AssertClose(e[k], B[k - 1, k], (fProxy)1E-4f);

            arena.Dispose();
        }

        void RandomTall5x1()
        {
            // Single column: B is 1x1, U is 5x1 unit vector, V is 1x1 = [[±1]]
            var arena = new Arena(Allocator.Persistent);
            int m = 5, n = 1;
            var A = arena.fProxyRandomMat(m, n, (fProxy)(-2f), (fProxy)2f, 77777);
            var U = arena.fProxyMat(m, n);
            var B = arena.fProxyMat(n, n);
            var V = arena.fProxyMat(n, n);
            Bidiag.bidiagonalize(in A, ref U, ref B, ref V);
            AssertBidiag(in A, in U, in B, in V, ref arena, (fProxy)1E-4f);
            arena.Dispose();
        }
    }

    // ---- plumbing ----

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void BidiagTests(TestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    // ---- argument-validation tests (managed, not Burst) ----

    [Test]
    public void BidiagThrowsOnWideMatrix()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyMat(3, 5);
        var U = arena.fProxyMat(3, 5);
        var B = arena.fProxyMat(5, 5);
        var V = arena.fProxyMat(5, 5);
        Assert.Catch<ArgumentException>(() => Bidiag.bidiagonalize(in A, ref U, ref B, ref V));
        arena.Dispose();
    }

    [Test]
    public void BidiagThrowsOnWrongUShape()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyMat(6, 4);
        var U = arena.fProxyMat(6, 3);   // wrong: should be 6x4
        var B = arena.fProxyMat(4, 4);
        var V = arena.fProxyMat(4, 4);
        Assert.Catch<ArgumentException>(() => Bidiag.bidiagonalize(in A, ref U, ref B, ref V));
        arena.Dispose();
    }

    [Test]
    public void BidiagThrowsOnWrongBShape()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyMat(6, 4);
        var U = arena.fProxyMat(6, 4);
        var B = arena.fProxyMat(3, 4);   // wrong: should be 4x4
        var V = arena.fProxyMat(4, 4);
        Assert.Catch<ArgumentException>(() => Bidiag.bidiagonalize(in A, ref U, ref B, ref V));
        arena.Dispose();
    }

    [Test]
    public void BidiagValuesThrowsOnWideMatrix()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyMat(3, 5);
        var d = arena.fProxyVec(5);
        var e = arena.fProxyVec(5);
        Assert.Catch<ArgumentException>(() => Bidiag.bidiagonalizeValues(in A, ref d, ref e));
        arena.Dispose();
    }

    [Test]
    public void BidiagValuesThrowsOnWrongVectorLength()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyMat(6, 4);
        var d = arena.fProxyVec(3);   // wrong: should be length 4
        var e = arena.fProxyVec(4);
        Assert.Catch<ArgumentException>(() => Bidiag.bidiagonalizeValues(in A, ref d, ref e));
        arena.Dispose();
    }

    [Test]
    public void BidiagThrowsOnWrongVShape()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyMat(6, 4);
        var U = arena.fProxyMat(6, 4);
        var B = arena.fProxyMat(4, 4);
        var V = arena.fProxyMat(3, 3);   // wrong: should be 4x4
        Assert.Catch<ArgumentException>(() => Bidiag.bidiagonalize(in A, ref U, ref B, ref V));
        arena.Dispose();
    }
}
