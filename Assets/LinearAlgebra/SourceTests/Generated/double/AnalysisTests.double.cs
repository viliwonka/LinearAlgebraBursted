using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
public class doubleAnalysisTests
{
    [BurstCompile]
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
            }
        }

        void isIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            
            doubleMxN A = arena.doubleIdentityMat(dim);

            Assert.IsTrue(Analysis_OP.isIdentity(A));

            arena.Dispose();
        }

        void IsIdentityEpsilon()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            
            doubleMxN A = arena.doubleIdentityMat(dim);

            Assert.IsTrue(Analysis_OP.isIdentity(A, 0.0001f));

            A += arena.doubleRandomMat(dim, dim, -0.001f, 0.001f);

            Assert.IsTrue(Analysis_OP.isIdentity(A, 0.002f));

            arena.Dispose();
        }

        void isSymmetric()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;
            
            doubleMxN A = arena.doubleIdentityMat(dim);

            Assert.IsTrue(Analysis_OP.isSymmetric(A));

            A = arena.doubleRandomMat(dim, dim * 2);

            doubleMxN C = double_OP.dot(A, A, true);

            Assert.IsTrue(Analysis_OP.isSymmetric(C));

            arena.Dispose();
        }

        void IsSymmetricEpsilon()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            
            doubleMxN A = arena.doubleIdentityMat(dim);

            Assert.IsTrue(Analysis_OP.isSymmetric(A, 0.000001f));

            A += arena.doubleRandomMat(dim, dim, -0.001f, 0.001f);

            Assert.IsTrue(Analysis_OP.isSymmetric(A, 0.002f));

            doubleMxN C = double_OP.dot(A, A, true);

            C += arena.doubleRandomMat(dim, dim, -0.001f, 0.001f);

            Assert.IsTrue(Analysis_OP.isSymmetric(C, 0.002f));

            arena.Dispose();
        }
        
        void isDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            
            doubleMxN A = arena.doubleIdentityMat(dim);

            Assert.IsTrue(Analysis_OP.isDiagonal(A));

            A += arena.doubleRandomMat(dim, dim, -0.001f, 0.001f);

            Assert.IsFalse(Analysis_OP.isDiagonal(A));

            arena.Dispose();
        }

        void IsDiagonalEpsilon()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            
            doubleMxN A = arena.doubleIdentityMat(dim);

            Assert.IsTrue(Analysis_OP.isDiagonal(A, 0.000001f));

            A += arena.doubleRandomMat(dim, dim, -0.001f, 0.001f);

            Assert.IsTrue(Analysis_OP.isDiagonal(A, 0.002f));

            A = arena.doubleRandomDiagonalMat(dim, -1f, -1f);

            Assert.IsTrue(Analysis_OP.isDiagonal(A, 0.000001f));

            arena.Dispose();
        }

        void isUpperTriangular()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            
            doubleMxN A = arena.doubleIdentityMat(dim);

            Assert.IsTrue(Analysis_OP.isUpperTriangular(A));
            
            A += arena.doubleRandomMat(dim, dim, -0.001f, 0.001f);

            Assert.IsFalse(Analysis_OP.isUpperTriangular(A));

            A = arena.doubleIdentityMat(dim);

            for (int c = 1; c < dim; c++)
            for (int r = 0; r < c; r++)
                A[r, c] = 5f;

            Assert.IsTrue(Analysis_OP.isUpperTriangular(A));

            arena.Dispose();
        }

        void IsUpperTriangularEpsilon()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            
            doubleMxN A = arena.doubleIdentityMat(dim);

            Assert.IsTrue(Analysis_OP.isUpperTriangular(A, 0.000001f));

            A += arena.doubleRandomMat(dim, dim, -0.001f, 0.001f);

            Assert.IsTrue(Analysis_OP.isUpperTriangular(A, 0.002f));

            A = arena.doubleIdentityMat(dim);

            for(int c = 1; c < dim; c++)
            for(int r = 0; r < c; r++)
                A[r, c] = 5f;

            A += arena.doubleRandomMat(dim, dim, -0.001f, 0.001f);

            Assert.IsTrue(Analysis_OP.isUpperTriangular(A, 0.002f));
                        
            arena.Dispose();   
        }

        void isLowerTriangular()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;

            doubleMxN A = arena.doubleIdentityMat(dim);

            Assert.IsTrue(Analysis_OP.isLowerTriangular(A));

            A += arena.doubleRandomMat(dim, dim, -0.001f, 0.001f);
            Assert.IsFalse(Analysis_OP.isLowerTriangular(A));

            // Reset A to the identity matrix
            A = arena.doubleIdentityMat(dim);

            // Fill elements above the diagonal with a non-zero value and check if it's still lower triangular (it shouldn't be)
            for (int r = 1; r < dim; r++)
                for (int c = 0; c < r; c++)
                    A[r, c] = 5f;

            // The matrix is now lower triangular
            Assert.IsTrue(Analysis_OP.isLowerTriangular(A));

            arena.Dispose();
        }

        void IsLowerTriangularEpsilon()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;

            doubleMxN A = arena.doubleIdentityMat(dim);

            // Test if an identity matrix is lower triangular within the epsilon tolerance
            Assert.IsTrue(Analysis_OP.isLowerTriangular(A, 0.000001f));

            // Add small random values and test if it's still lower triangular within a higher tolerance
            A += arena.doubleRandomMat(dim, dim, -0.001f, 0.001f);
            Assert.IsTrue(Analysis_OP.isLowerTriangular(A, 0.002f));

            // Reset A to the identity matrix
            A = arena.doubleIdentityMat(dim);

            // Fill elements above the diagonal with a non-zero value
            for (int r = 1; r < dim; r++)
                for (int c = 0; c < r; c++)
                    A[r, c] = 5f;

            // Add small random values again
            A += arena.doubleRandomMat(dim, dim, -0.001f, 0.001f);

            // Test if the modified matrix is still lower triangular within the higher epsilon tolerance
            Assert.IsTrue(Analysis_OP.isLowerTriangular(A, 0.002f));

            arena.Dispose();
        }

        void isOrthogonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;

            doubleMxN A = arena.doubleIdentityMat(dim);

            Assert.IsTrue(Analysis_OP.isOrthogonal(A, 0.00001f));

            A = double_OP.dot(arena.doublePermutationMat(dim, 5, 13), A);

            Assert.IsTrue(Analysis_OP.isOrthogonal(A, 0.00001f));

            A = double_OP.dot(arena.doubleRotationMat(dim, 3, 15, math.PI/4f ), A);

            Assert.IsTrue(Analysis_OP.isOrthogonal(A, 0.00001f));

            doubleN reflect = arena.doubleRandomVec(dim, -1f, 1f);

            A = double_OP.dot(arena.doubleHouseholderMat(dim, reflect), A);

            Assert.IsTrue(Analysis_OP.isOrthogonal(A, 0.00001f));

            reflect = arena.doubleRandomVec(dim, -1f, 1f, 50301);
            A = double_OP.dot(arena.doubleHouseholderMat(dim, reflect), A);

            Assert.IsTrue(Analysis_OP.isOrthogonal(A, 0.00001f));

            // self multiply
            A = double_OP.dot(A, A);

            Assert.IsTrue(Analysis_OP.isOrthogonal(A, 0.00001f));

            // testing inverse
            A = double_OP.dot(A, A, true);

            Assert.IsTrue(Analysis_OP.isIdentity(A, 0.00001f));

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


}
