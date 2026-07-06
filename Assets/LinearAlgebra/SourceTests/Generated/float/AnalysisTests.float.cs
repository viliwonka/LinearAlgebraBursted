using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
public class floatAnalysisTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct AnalysisTestJob : IJob
    {
        public enum TestType
        {
            isIdentity,
            IsIdentityEpsilon,
            isSymmetric,
            IsSymmetricEpsilon,
            isDiagonal,
            IsDiagonalEpsilon,
            isUpperTriangular,
            IsUpperTriangularEpsilon,
            isLowerTriangular,
            IsLowerTriangularEpsilon,
            isOrthogonal,
            Determinant,
            LogDeterminant,
        }

        public TestType Type;

        public void Execute()
        {
            switch(Type)
            {
                case TestType.isIdentity:
                    isIdentity();
                    break;
                case TestType.IsIdentityEpsilon:
                    IsIdentityEpsilon();
                    break;
                case TestType.isSymmetric:
                    isSymmetric();
                    break;
                case TestType.IsSymmetricEpsilon:
                    IsSymmetricEpsilon();
                    break;
                case TestType.isDiagonal:
                    isDiagonal();
                    break;
                case TestType.IsDiagonalEpsilon:
                    IsDiagonalEpsilon();
                    break;
                case TestType.isUpperTriangular:
                    isUpperTriangular();
                    break;
                case TestType.IsUpperTriangularEpsilon:
                    IsUpperTriangularEpsilon();
                    break;
                case TestType.isLowerTriangular:
                    isLowerTriangular();
                    break;
                case TestType.IsLowerTriangularEpsilon:
                    IsLowerTriangularEpsilon();
                    break;
                case TestType.isOrthogonal:
                    isOrthogonal();
                    break;
                case TestType.Determinant:
                    Determinant();
                    break;
                case TestType.LogDeterminant:
                    LogDeterminant();
                    break;
            }
        }

        void isIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            
            floatMxN A = arena.floatIdentityMat(dim);

            Assert.IsTrue(Analysis.isIdentity(A));

            arena.Dispose();
        }

        void IsIdentityEpsilon()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            
            floatMxN A = arena.floatIdentityMat(dim);

            Assert.IsTrue(Analysis.isIdentity(A, 0.0001f));

            A += arena.floatRandomMat(dim, dim, -0.001f, 0.001f);

            Assert.IsTrue(Analysis.isIdentity(A, 0.002f));

            arena.Dispose();
        }

        void isSymmetric()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;
            
            floatMxN A = arena.floatIdentityMat(dim);

            Assert.IsTrue(Analysis.isSymmetric(A));

            A = arena.floatRandomMat(dim, dim * 2);

            floatMxN C = Blas.dot(A, A, true);

            Assert.IsTrue(Analysis.isSymmetric(C));

            arena.Dispose();
        }

        void IsSymmetricEpsilon()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            
            floatMxN A = arena.floatIdentityMat(dim);

            Assert.IsTrue(Analysis.isSymmetric(A, 0.000001f));

            A += arena.floatRandomMat(dim, dim, -0.001f, 0.001f);

            Assert.IsTrue(Analysis.isSymmetric(A, 0.002f));

            floatMxN C = Blas.dot(A, A, true);

            C += arena.floatRandomMat(dim, dim, -0.001f, 0.001f);

            Assert.IsTrue(Analysis.isSymmetric(C, 0.002f));

            arena.Dispose();
        }
        
        void isDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            
            floatMxN A = arena.floatIdentityMat(dim);

            Assert.IsTrue(Analysis.isDiagonal(A));

            A += arena.floatRandomMat(dim, dim, -0.001f, 0.001f);

            Assert.IsFalse(Analysis.isDiagonal(A));

            arena.Dispose();
        }

        void IsDiagonalEpsilon()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            
            floatMxN A = arena.floatIdentityMat(dim);

            Assert.IsTrue(Analysis.isDiagonal(A, 0.000001f));

            A += arena.floatRandomMat(dim, dim, -0.001f, 0.001f);

            Assert.IsTrue(Analysis.isDiagonal(A, 0.002f));

            A = arena.floatRandomDiagonalMat(dim, -1f, -1f);

            Assert.IsTrue(Analysis.isDiagonal(A, 0.000001f));

            arena.Dispose();
        }

        void isUpperTriangular()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            
            floatMxN A = arena.floatIdentityMat(dim);

            Assert.IsTrue(Analysis.isUpperTriangular(A));
            
            A += arena.floatRandomMat(dim, dim, -0.001f, 0.001f);

            Assert.IsFalse(Analysis.isUpperTriangular(A));

            A = arena.floatIdentityMat(dim);

            for (int c = 1; c < dim; c++)
            for (int r = 0; r < c; r++)
                A[r, c] = 5f;

            Assert.IsTrue(Analysis.isUpperTriangular(A));

            arena.Dispose();
        }

        void IsUpperTriangularEpsilon()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            
            floatMxN A = arena.floatIdentityMat(dim);

            Assert.IsTrue(Analysis.isUpperTriangular(A, 0.000001f));

            A += arena.floatRandomMat(dim, dim, -0.001f, 0.001f);

            Assert.IsTrue(Analysis.isUpperTriangular(A, 0.002f));

            A = arena.floatIdentityMat(dim);

            for(int c = 1; c < dim; c++)
            for(int r = 0; r < c; r++)
                A[r, c] = 5f;

            A += arena.floatRandomMat(dim, dim, -0.001f, 0.001f);

            Assert.IsTrue(Analysis.isUpperTriangular(A, 0.002f));
                        
            arena.Dispose();   
        }

        void isLowerTriangular()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;

            floatMxN A = arena.floatIdentityMat(dim);

            Assert.IsTrue(Analysis.isLowerTriangular(A));

            A += arena.floatRandomMat(dim, dim, -0.001f, 0.001f);
            Assert.IsFalse(Analysis.isLowerTriangular(A));

            A = arena.floatIdentityMat(dim);

            // Fill elements below the diagonal with a non-zero value; still lower triangular
            for (int r = 1; r < dim; r++)
                for (int c = 0; c < r; c++)
                    A[r, c] = 5f;

            Assert.IsTrue(Analysis.isLowerTriangular(A));

            arena.Dispose();
        }

        void IsLowerTriangularEpsilon()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;

            floatMxN A = arena.floatIdentityMat(dim);

            Assert.IsTrue(Analysis.isLowerTriangular(A, 0.000001f));

            A += arena.floatRandomMat(dim, dim, -0.001f, 0.001f);
            Assert.IsTrue(Analysis.isLowerTriangular(A, 0.002f));

            A = arena.floatIdentityMat(dim);

            // Fill elements below the diagonal with a non-zero value
            for (int r = 1; r < dim; r++)
                for (int c = 0; c < r; c++)
                    A[r, c] = 5f;

            A += arena.floatRandomMat(dim, dim, -0.001f, 0.001f);

            Assert.IsTrue(Analysis.isLowerTriangular(A, 0.002f));

            arena.Dispose();
        }

        void isOrthogonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;

            floatMxN A = arena.floatIdentityMat(dim);

            Assert.IsTrue(Analysis.isOrthogonal(A, 0.00001f));

            A = Blas.dot(arena.floatPermutationMat(dim, 5, 13), A);

            Assert.IsTrue(Analysis.isOrthogonal(A, 0.00001f));

            A = Blas.dot(arena.floatRotationMat(dim, 3, 15, math.PI/4f ), A);

            Assert.IsTrue(Analysis.isOrthogonal(A, 0.00001f));

            floatN reflect = arena.floatRandomVec(dim, -1f, 1f);

            A = Blas.dot(arena.floatHouseholderMat(dim, reflect), A);

            Assert.IsTrue(Analysis.isOrthogonal(A, 0.00001f));

            reflect = arena.floatRandomVec(dim, -1f, 1f, 50301);
            A = Blas.dot(arena.floatHouseholderMat(dim, reflect), A);

            Assert.IsTrue(Analysis.isOrthogonal(A, 0.00001f));

            A = Blas.dot(A, A);

            Assert.IsTrue(Analysis.isOrthogonal(A, 0.00001f));

            A = Blas.dot(A, A, true);

            Assert.IsTrue(Analysis.isIdentity(A, 0.00001f));

            arena.Dispose();
        }

        void Determinant()
        {
            var arena = new Arena(Allocator.Persistent);

            // identity -> det = 1 (matrix-in path; A must be left intact, like cond/rank)
            {
                int dim = 5;
                floatMxN A = arena.floatIdentityMat(dim);
                Assert.IsTrue(math.abs(Analysis.determinant(in A) - (float)1) < (float)1E-4f);
                Assert.IsTrue(Analysis.isIdentity(A, 1E-6f));           // A not modified
            }

            // diagonal [2,-3,0.5,4] -> det = -12
            {
                int dim = 4;
                floatMxN A = arena.floatMat(dim, dim);
                A[0, 0] = 2f; A[1, 1] = -3f; A[2, 2] = 0.5f; A[3, 3] = 4f;
                Assert.IsTrue(math.abs(Analysis.determinant(in A) - (float)(-12f)) < (float)1E-3f);
            }

            // matrix-in path agrees with the zero-alloc from-factor overload on the same A
            {
                int dim = 6;
                floatMxN A = arena.floatRandomMat(dim, dim, -2f, 2f);
                float viaMatrix = Analysis.determinant(in A);

                floatMxN lu = A.Copy();
                var P = new Pivot(dim, Allocator.Temp);
                LU.decompInPlace(ref lu, ref P);
                float viaFactor = Analysis.determinant(in lu, in P);
                P.Dispose();

                Assert.IsTrue(math.abs(viaMatrix - viaFactor) < (float)1E-3f * (math.abs(viaFactor) + (float)1));
            }

            // singular (row 0 == row 1) -> det = 0
            {
                int dim = 3;
                floatMxN A = arena.floatMat(dim, dim);
                A[0, 0] = 1f; A[0, 1] = 2f; A[0, 2] = 3f;
                A[1, 0] = 1f; A[1, 1] = 2f; A[1, 2] = 3f;
                A[2, 0] = 4f; A[2, 1] = 5f; A[2, 2] = 7f;
                Assert.IsTrue(math.abs(Analysis.determinant(in A)) < (float)1E-4f);
            }

            arena.Dispose();
        }

        void LogDeterminant()
        {
            var arena = new Arena(Allocator.Persistent);

            // identity -> log|det| = 0, sign = +1
            {
                int dim = 5;
                floatMxN A = arena.floatIdentityMat(dim);
                float logAbs = Analysis.logDeterminant(in A, out float sign);
                Assert.IsTrue(math.abs(logAbs) < (float)1E-4f);
                Assert.IsTrue(math.abs(sign - (float)1) < (float)1E-6f);
            }

            // diagonal [2,-3,0.5,4] -> det = -12: sign = -1, sign*exp(logAbs) recovers det
            {
                int dim = 4;
                floatMxN A = arena.floatMat(dim, dim);
                A[0, 0] = 2f; A[1, 1] = -3f; A[2, 2] = 0.5f; A[3, 3] = 4f;
                float logAbs = Analysis.logDeterminant(in A, out float sign);
                Assert.IsTrue(math.abs(sign - (float)(-1f)) < (float)1E-6f);
                float recovered = sign * math.exp(logAbs);
                Assert.IsTrue(math.abs(recovered - (float)(-12f)) < (float)1E-3f);
            }

            // singular (row 1 == 2*row 0) -> sign = 0, log|det| = -infinity
            {
                int dim = 3;
                floatMxN A = arena.floatMat(dim, dim);
                A[0, 0] = 1f; A[0, 1] = 2f; A[0, 2] = 3f;
                A[1, 0] = 2f; A[1, 1] = 4f; A[1, 2] = 6f;
                A[2, 0] = 4f; A[2, 1] = 5f; A[2, 2] = 7f;
                float logAbs = Analysis.logDeterminant(in A, out float sign);
                Assert.IsTrue(math.abs(sign) < (float)1E-6f);
                Assert.IsTrue(math.isinf(logAbs) && logAbs < (float)0);
            }

            // the whole point of slogdet: the plain product OVERFLOWS to Inf, log|det| stays finite.
            // 10*I at dim 400 -> det = 10^400, past both float (~1e38) and double (~1e308) range.
            {
                int dim = 400;
                floatMxN A = arena.floatMat(dim, dim);
                for (int i = 0; i < dim; i++)
                    A[i, i] = (float)10f;

                Assert.IsTrue(math.isinf(Analysis.determinant(in A)));           // product overflows

                float logAbs = Analysis.logDeterminant(in A, out float sign);
                Assert.IsFalse(math.isinf(logAbs));                              // log stays finite
                float expected = (float)dim * math.log((float)10f);
                Assert.IsTrue(math.abs(logAbs - expected) < (float)0.5f);
                Assert.IsTrue(math.abs(sign - (float)1) < (float)1E-6f);
            }

            arena.Dispose();
        }

    }

    [Test]
    public void IsIdentityTest()
    {
        new AnalysisTestJob() { Type = AnalysisTestJob.TestType.isIdentity }.Run();
    }

    [Test]
    public void IsIdentityEpsilonTest()
    {
        new AnalysisTestJob() { Type = AnalysisTestJob.TestType.IsIdentityEpsilon }.Run();
    }

    [Test]
    public void IsSymmetricTest()
    {
        new AnalysisTestJob() { Type = AnalysisTestJob.TestType.isSymmetric }.Run();
    }

    [Test]
    public void IsSymmetricEpsilonTest()
    {
        new AnalysisTestJob() { Type = AnalysisTestJob.TestType.IsSymmetricEpsilon }.Run();
    }

    [Test]
    public void IsDiagonalTest()
    {
        new AnalysisTestJob() { Type = AnalysisTestJob.TestType.isDiagonal }.Run();
    }

    [Test]
    public void IsDiagonalEpsilonTest()
    {
        new AnalysisTestJob() { Type = AnalysisTestJob.TestType.IsDiagonalEpsilon }.Run();
    }

    [Test]
    public void IsUpperTriangularTest()
    {
        new AnalysisTestJob() { Type = AnalysisTestJob.TestType.isUpperTriangular }.Run();
    }

    [Test]
    public void IsUpperTriangularEpsilonTest()
    {
        new AnalysisTestJob() { Type = AnalysisTestJob.TestType.IsUpperTriangularEpsilon }.Run();
    }

    [Test]
    public void IsLowerTriangularTest()
    {
        new AnalysisTestJob() { Type = AnalysisTestJob.TestType.isLowerTriangular }.Run();
    }

    [Test]
    public void IsLowerTriangularEpsilonTest()
    {
        new AnalysisTestJob() { Type = AnalysisTestJob.TestType.IsLowerTriangularEpsilon }.Run();
    }

    [Test]
    public void IsOrthogonalTest()
    {
        new AnalysisTestJob() { Type = AnalysisTestJob.TestType.isOrthogonal }.Run();
    }

    [Test]
    public void DeterminantTest()
    {
        new AnalysisTestJob() { Type = AnalysisTestJob.TestType.Determinant }.Run();
    }

    [Test]
    public void LogDeterminantTest()
    {
        new AnalysisTestJob() { Type = AnalysisTestJob.TestType.LogDeterminant }.Run();
    }


}
