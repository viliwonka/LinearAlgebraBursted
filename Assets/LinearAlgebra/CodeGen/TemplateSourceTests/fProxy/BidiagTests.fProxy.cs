using System;

using BULA;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

public class fProxyBidiagTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default, CompileSynchronously = true)]
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

        private void AssertNearZero(in fProxyMxN M, fProxy tol, string context)
        {
            fProxy err = Analysis.MaxZeroError(M);
            if (!(err <= tol) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = err;
                Fail[2] = tol;
                Fail[3] = err - tol;
            }
            Assert.IsTrue(Analysis.isZero(in M, tol));
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

        // Full suite for a single bidiagonalization result: reconstruction, bidiagonal band,
        // and U/V orthonormality (numbered inline below).
        private void AssertBidiag(in fProxyMxN A, in fProxyMxN U, in fProxyMxN B, in fProxyMxN V,
                                   fProxy tol)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            // 1. Reconstruction: A ≈ U * B * Vᵀ
            var Vt = new fProxyMxN(V.N_Cols, V.M_Rows, Allocator.Temp);
            Blas.trans(in V, ref Vt);
            var UB = new fProxyMxN(U.M_Rows, B.N_Cols, Allocator.Temp);
            Blas.dot(in U, in B, ref UB);
            var UBVt = new fProxyMxN(UB.M_Rows, Vt.N_Cols, Allocator.Temp);
            Blas.dot(in UB, in Vt, ref UBVt);
            var diff = new fProxyMxN(in A, Allocator.Temp);
            fProxyComp.subInPlace(diff, UBVt);

            if (Analysis.isAnyNan(in diff))
                throw new System.Exception("BidiagTests: NaN in reconstruction");

            AssertNearZero(in diff, tol, "reconstruction");

            // 2. B is upper bidiagonal
            AssertUpperBidiagonal(in B, tol);

            // 3. UᵀU ≈ I_n  (Analysis.isOrthogonal handles thin U: computes AᵀA = I_n)
            Assert.IsTrue(Analysis.isOrthogonal(in U, tol));

            // 4. VᵀV ≈ I_n  (V is square)
            Assert.IsTrue(Analysis.isOrthogonal(in V, tol));
        }

        // ---- test cases ----

        void IdentitySquare()
        {
            int n = 6;
            var A = GenerateOP.fProxyIdentityMat(n);
            var U = new fProxyMxN(n, n, Allocator.Temp);
            var B = new fProxyMxN(n, n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            Bidiag.decomp(in A, ref U, ref B, ref V);
            AssertBidiag(in A, in U, in B, in V, (fProxy)1E-4f);
        }

        void DiagonalSquare()
        {
            // Diagonal input: A is already bidiagonal; B should ≈ A (up to signs), U,V ≈ I (up to signs)
            int n = 6;
            var A = new fProxyMxN(n, n, Allocator.Temp);
            A[0, 0] = (fProxy)3f;
            A[1, 1] = (fProxy)1f;
            A[2, 2] = (fProxy)4f;
            A[3, 3] = (fProxy)1f;
            A[4, 4] = (fProxy)5f;
            A[5, 5] = (fProxy)9f;
            var U = new fProxyMxN(n, n, Allocator.Temp);
            var B = new fProxyMxN(n, n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            Bidiag.decomp(in A, ref U, ref B, ref V);
            AssertBidiag(in A, in U, in B, in V, (fProxy)1E-4f);
        }

        void RandomSquare6x6()
        {
            int n = 6;
            var A = GenerateOP.fProxyRandomMat(n, n, (fProxy)(-2f), (fProxy)2f, 314159);
            var U = new fProxyMxN(n, n, Allocator.Temp);
            var B = new fProxyMxN(n, n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            Bidiag.decomp(in A, ref U, ref B, ref V);
            AssertBidiag(in A, in U, in B, in V, (fProxy)1E-4f);
        }

        void RandomSquare8x8()
        {
            int n = 8;
            var A = GenerateOP.fProxyRandomMat(n, n, (fProxy)(-5f), (fProxy)5f, 271828);
            var U = new fProxyMxN(n, n, Allocator.Temp);
            var B = new fProxyMxN(n, n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            Bidiag.decomp(in A, ref U, ref B, ref V);
            AssertBidiag(in A, in U, in B, in V, (fProxy)1E-4f);
        }

        void RandomTall10x6()
        {
            int m = 10, n = 6;
            var A = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-3f), (fProxy)3f, 112358);
            var U = new fProxyMxN(m, n, Allocator.Temp);
            var B = new fProxyMxN(n, n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            Bidiag.decomp(in A, ref U, ref B, ref V);
            AssertBidiag(in A, in U, in B, in V, (fProxy)1E-4f);
        }

        void RandomTall12x4()
        {
            int m = 12, n = 4;
            var A = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-1f), (fProxy)1f, 999421);
            var U = new fProxyMxN(m, n, Allocator.Temp);
            var B = new fProxyMxN(n, n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            Bidiag.decomp(in A, ref U, ref B, ref V);
            AssertBidiag(in A, in U, in B, in V, (fProxy)1E-4f);
        }

        // Bidiag.values must produce EXACTLY the bidiagonal bands of the full Bidiag.decomp:
        // both use identical reflectors/applies, so d[k]=B[k,k], e[0]=0, e[k]=B[k-1,k].
        void CheckValuesMatchFull(int m, int n, fProxy lo, fProxy hi, uint seed)
        {
            var A = GenerateOP.fProxyRandomMat(m, n, lo, hi, seed);
            var U = new fProxyMxN(m, n, Allocator.Temp);
            var B = new fProxyMxN(n, n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            Bidiag.decomp(in A, ref U, ref B, ref V);

            var d = new fProxyN(n, Allocator.Temp);
            var e = new fProxyN(n, Allocator.Temp);
            Bidiag.values(in A, ref d, ref e);

            for (int k = 0; k < n; k++)
                AssertClose(d[k], B[k, k], (fProxy)1E-4f);
            AssertClose(e[0], (fProxy)0, (fProxy)1E-4f);
            for (int k = 1; k < n; k++)
                AssertClose(e[k], B[k - 1, k], (fProxy)1E-4f);
        }

        void RandomTall5x1()
        {
            // Single column: B is 1x1, U is 5x1 unit vector, V is 1x1 = [[±1]]
            int m = 5, n = 1;
            var A = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-2f), (fProxy)2f, 77777);
            var U = new fProxyMxN(m, n, Allocator.Temp);
            var B = new fProxyMxN(n, n, Allocator.Temp);
            var V = new fProxyMxN(n, n, Allocator.Temp);
            Bidiag.decomp(in A, ref U, ref B, ref V);
            AssertBidiag(in A, in U, in B, in V, (fProxy)1E-4f);
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
        var A = new fProxyMxN(3, 5, Allocator.Temp);
        var U = new fProxyMxN(3, 5, Allocator.Temp);
        var B = new fProxyMxN(5, 5, Allocator.Temp);
        var V = new fProxyMxN(5, 5, Allocator.Temp);
        Assert.Catch<ArgumentException>(() => Bidiag.decomp(in A, ref U, ref B, ref V));
    }

    [Test]
    public void BidiagThrowsOnWrongUShape()
    {
        var A = new fProxyMxN(6, 4, Allocator.Temp);
        var U = new fProxyMxN(6, 3, Allocator.Temp);   // wrong: should be 6x4
        var B = new fProxyMxN(4, 4, Allocator.Temp);
        var V = new fProxyMxN(4, 4, Allocator.Temp);
        Assert.Catch<ArgumentException>(() => Bidiag.decomp(in A, ref U, ref B, ref V));
    }

    [Test]
    public void BidiagThrowsOnWrongBShape()
    {
        var A = new fProxyMxN(6, 4, Allocator.Temp);
        var U = new fProxyMxN(6, 4, Allocator.Temp);
        var B = new fProxyMxN(3, 4, Allocator.Temp);   // wrong: should be 4x4
        var V = new fProxyMxN(4, 4, Allocator.Temp);
        Assert.Catch<ArgumentException>(() => Bidiag.decomp(in A, ref U, ref B, ref V));
    }

    [Test]
    public void BidiagValuesThrowsOnWideMatrix()
    {
        var A = new fProxyMxN(3, 5, Allocator.Temp);
        var d = new fProxyN(5, Allocator.Temp);
        var e = new fProxyN(5, Allocator.Temp);
        Assert.Catch<ArgumentException>(() => Bidiag.values(in A, ref d, ref e));
    }

    [Test]
    public void BidiagValuesThrowsOnWrongVectorLength()
    {
        var A = new fProxyMxN(6, 4, Allocator.Temp);
        var d = new fProxyN(3, Allocator.Temp);   // wrong: should be length 4
        var e = new fProxyN(4, Allocator.Temp);
        Assert.Catch<ArgumentException>(() => Bidiag.values(in A, ref d, ref e));
    }

    [Test]
    public void BidiagThrowsOnWrongVShape()
    {
        var A = new fProxyMxN(6, 4, Allocator.Temp);
        var U = new fProxyMxN(6, 4, Allocator.Temp);
        var B = new fProxyMxN(4, 4, Allocator.Temp);
        var V = new fProxyMxN(3, 3, Allocator.Temp);   // wrong: should be 4x4
        Assert.Catch<ArgumentException>(() => Bidiag.decomp(in A, ref U, ref B, ref V));
    }
}
