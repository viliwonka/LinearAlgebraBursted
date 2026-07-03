using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Tests for the numerical-LA "basics": fixed L1 norm, LInf, trace, induced matrix norms
// (1/∞/2), condition number, and numerical rank.
public class fProxyMatrixMetricsTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            VectorL1AndLInf,
            Trace,
            MatrixL1,
            MatrixLInf,
            SpectralNorm,
            Cond,
            CondIdentity,
            CondZero,
            CondSingular,
            CondNonDiagonal,
            OneByOne,
            WideMatrix,
            MatrixNormsNonSquare,
            RankFull,
            RankExplicitTol,
            RankDeficient,
            RankZero,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<fProxy> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.VectorL1AndLInf: VectorL1AndLInf(); break;
                case TestType.Trace:           Trace(); break;
                case TestType.MatrixL1:        MatrixL1(); break;
                case TestType.MatrixLInf:      MatrixLInf(); break;
                case TestType.SpectralNorm:    SpectralNorm(); break;
                case TestType.Cond:            Cond(); break;
                case TestType.CondIdentity:    CondIdentity(); break;
                case TestType.CondZero:        CondZero(); break;
                case TestType.CondSingular:    CondSingular(); break;
                case TestType.CondNonDiagonal: CondNonDiagonal(); break;
                case TestType.OneByOne:        OneByOne(); break;
                case TestType.WideMatrix:      WideMatrix(); break;
                case TestType.MatrixNormsNonSquare: MatrixNormsNonSquare(); break;
                case TestType.RankFull:        RankCase(RankShape.Full); break;
                case TestType.RankExplicitTol: RankExplicitTol(); break;
                case TestType.RankDeficient:   RankCase(RankShape.Deficient); break;
                case TestType.RankZero:        RankCase(RankShape.Zero); break;
            }
        }

        // L1 = Σ|xᵢ| (NOT averaged); LInf = max|xᵢ|. v = [3, -4, 0, 1] -> L1 = 8, LInf = 4.
        void VectorL1AndLInf()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.fProxyVec(4);
            v[0] = (fProxy)3; v[1] = (fProxy)(-4); v[2] = (fProxy)0; v[3] = (fProxy)1;

            AssertClose(Norms.L1(in v), (fProxy)8, (fProxy)1E-5);
            AssertClose(Norms.LInf(in v), (fProxy)4, (fProxy)1E-5);

            arena.Dispose();
        }

        // trace([[1,2],[3,4]]) = 1 + 4 = 5.
        void Trace()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(2, 2);
            A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)2;
            A[1, 0] = (fProxy)3; A[1, 1] = (fProxy)4;

            AssertClose(Analysis.trace(in A), (fProxy)5, (fProxy)1E-5);

            arena.Dispose();
        }

        // ‖A‖₁ = max abs column sum. A = [[1,-2],[-3,4]] -> cols {4, 6} -> 6.
        void MatrixL1()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(2, 2);
            A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)(-2);
            A[1, 0] = (fProxy)(-3); A[1, 1] = (fProxy)4;

            AssertClose(Norms.matrixL1(in A), (fProxy)6, (fProxy)1E-5);

            arena.Dispose();
        }

        // ‖A‖∞ = max abs row sum. Same A -> rows {3, 7} -> 7.
        void MatrixLInf()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(2, 2);
            A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)(-2);
            A[1, 0] = (fProxy)(-3); A[1, 1] = (fProxy)4;

            AssertClose(Norms.matrixLInf(in A), (fProxy)7, (fProxy)1E-5);

            arena.Dispose();
        }

        // ‖A‖₂ = σ_max. diag(5, 2) -> σ_max = 5.
        void SpectralNorm()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(2, 2);
            A[0, 0] = (fProxy)5; A[1, 1] = (fProxy)2;

            AssertClose(Norms.matrixL2(in A), (fProxy)5, (fProxy)1E-4);

            // tall/square branch must NOT modify A (it decomposes a TempCopy, not A itself)
            AssertClose(A[0, 0], (fProxy)5, (fProxy)1E-6);
            AssertClose(A[1, 1], (fProxy)2, (fProxy)1E-6);

            arena.Dispose();
        }

        // Wide matrix (m < n) exercises singularValues' transpose branch. A = [[3,0,0],[0,4,0]]
        // (2x3) has singular values {4, 3}: σ_max = 4, rank 2, cond = 4/3. Also confirms A is
        // left unmodified (the metric calls must decompose a transpose, not A in place).
        void WideMatrix()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(2, 3);
            A[0, 0] = (fProxy)3;
            A[1, 1] = (fProxy)4;

            AssertClose(Norms.matrixL2(in A), (fProxy)4, (fProxy)1E-4);
            AssertIntEqual(Analysis.rank(in A), 2);
            AssertClose(Analysis.cond(in A), (fProxy)4 / (fProxy)3, (fProxy)1E-4);

            AssertClose(A[0, 0], (fProxy)3, (fProxy)1E-6);
            AssertClose(A[1, 1], (fProxy)4, (fProxy)1E-6);

            arena.Dispose();
        }

        // cond(diag(4, 1)) = 4 / 1 = 4.
        void Cond()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(2, 2);
            A[0, 0] = (fProxy)4; A[1, 1] = (fProxy)1;

            AssertClose(Analysis.cond(in A), (fProxy)4, (fProxy)1E-4);

            arena.Dispose();
        }

        // cond(I) = 1 (the canonical perfectly-conditioned case).
        void CondIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyIdentityMat(3);
            AssertClose(Analysis.cond(in A), (fProxy)1, (fProxy)1E-4);

            arena.Dispose();
        }

        // cond(zeros) -> +infinity (σ_max == σ_min == 0; the !(sMin>0) guard). Pins the design
        // choice (this library returns +inf, unlike MATLAB's NaN).
        void CondZero()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(2, 2);   // all zeros
            AssertGreater(Analysis.cond(in A), (fProxy)1E6);

            arena.Dispose();
        }

        // Induced 1/∞-norms on a NON-square matrix. A = [[1,-2],[3,-4],[5,6]] (3x2):
        // columns {9, 12} -> ‖A‖₁ = 12; rows {3, 7, 11} -> ‖A‖∞ = 11.
        void MatrixNormsNonSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(3, 2);
            A[0, 0] = (fProxy)1;  A[0, 1] = (fProxy)(-2);
            A[1, 0] = (fProxy)3;  A[1, 1] = (fProxy)(-4);
            A[2, 0] = (fProxy)5;  A[2, 1] = (fProxy)6;

            AssertClose(Norms.matrixL1(in A), (fProxy)12, (fProxy)1E-5);
            AssertClose(Norms.matrixLInf(in A), (fProxy)11, (fProxy)1E-5);

            arena.Dispose();
        }

        // Singular matrix [[3,0],[4,0]] (rank 1, σ_min = 0) -> cond = +infinity.
        void CondSingular()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(2, 2);
            A[0, 0] = (fProxy)3; A[1, 0] = (fProxy)4;   // column 1 is zero

            fProxy c = Analysis.cond(in A);
            // true value is +inf; accept anything astronomically large (NaN-safe via the record below)
            AssertGreater(c, (fProxy)1E6);

            arena.Dispose();
        }

        enum RankShape { Full, Deficient, Zero }

        void RankCase(RankShape shape)
        {
            var arena = new Arena(Allocator.Persistent);

            int expected;
            fProxyMxN A;

            if (shape == RankShape.Full)
            {
                // diag(2, 3, 5) -> rank 3
                A = arena.fProxyMat(3, 3);
                A[0, 0] = (fProxy)2; A[1, 1] = (fProxy)3; A[2, 2] = (fProxy)5;
                expected = 3;
            }
            else if (shape == RankShape.Deficient)
            {
                // all-ones 4x2 (both columns identical) -> rank 1
                A = arena.fProxyMat(4, 2);
                for (int i = 0; i < 4; i++) { A[i, 0] = (fProxy)1; A[i, 1] = (fProxy)1; }
                expected = 1;
            }
            else
            {
                // zero 3x2 -> rank 0
                A = arena.fProxyMat(3, 2);
                expected = 0;
            }

            int r = Analysis.rank(in A);
            AssertIntEqual(r, expected);

            arena.Dispose();
        }

        // Non-diagonal symmetric A = [[2,1],[1,2]] has eigenvalues {3,1} = singular values (PSD),
        // so cond = 3 and σ_max = 3. This actually exercises the Jacobi rotation path (off-diagonal
        // != 0), unlike the diagonal cond/spectral tests above.
        void CondNonDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(2, 2);
            A[0, 0] = (fProxy)2; A[0, 1] = (fProxy)1;
            A[1, 0] = (fProxy)1; A[1, 1] = (fProxy)2;

            AssertClose(Analysis.cond(in A), (fProxy)3, (fProxy)1E-4);
            AssertClose(Norms.matrixL2(in A), (fProxy)3, (fProxy)1E-4);

            arena.Dispose();
        }

        // 1x1 matrix [[7]] -> trace 7, cond 1, rank 1, σ_max 7. Exercises the n==1 path
        // (Jacobi inner sweep is empty).
        void OneByOne()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(1, 1);
            A[0, 0] = (fProxy)7;

            AssertClose(Analysis.trace(in A), (fProxy)7, (fProxy)1E-5);
            AssertClose(Analysis.cond(in A), (fProxy)1, (fProxy)1E-4);
            AssertIntEqual(Analysis.rank(in A), 1);
            AssertClose(Norms.matrixL2(in A), (fProxy)7, (fProxy)1E-4);

            arena.Dispose();
        }

        // rank with an explicit relTol must change the cutoff. A = diag(1, 1e-5):
        // auto tolerance keeps both (rank 2); a loose relTol = 1e-2 drops the small one (rank 1).
        void RankExplicitTol()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(2, 2);
            A[0, 0] = (fProxy)1; A[1, 1] = (fProxy)1E-5;

            AssertIntEqual(Analysis.rank(in A), 2);                       // auto tol
            AssertIntEqual(Analysis.rank(in A, (fProxy)1E-2), 1);         // loose tol drops σ=1e-5

            arena.Dispose();
        }

        // ---- recording asserts ----

        void AssertClose(fProxy a, fProxy b, fProxy precision)
        {
            fProxy diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = a; Fail[2] = b; Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertGreater(fProxy value, fProxy limit)
        {
            if (!(value > limit) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = value; Fail[2] = limit; Fail[3] = limit - value;
            }
            Assert.IsTrue(value > limit);
        }

        void AssertIntEqual(int got, int expected)
        {
            if (got != expected && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = got; Fail[2] = expected; Fail[3] = got - expected;
            }
            Assert.IsTrue(got == expected);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void MetricsTests(TestJob.TestType type)
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
        finally { fail.Dispose(); }
    }

    // Managed guard: trace requires a square matrix.
    [Test]
    public void Trace_NonSquare_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(2, 3);
            Assert.Throws<ArgumentException>(() => Analysis.trace(in A));
        }
        finally { arena.Dispose(); }
    }
}
