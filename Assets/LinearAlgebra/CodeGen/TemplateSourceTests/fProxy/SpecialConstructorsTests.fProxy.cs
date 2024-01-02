using LinearAlgebra;
using NUnit.Framework;

using System.Diagnostics;

using Unity.Mathematics;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class fProxySpecialConstructorsTests {

    [BurstCompile(FloatPrecision = FloatPrecision.High)] 
    public struct TestJob : IJob
    {
        public enum TestType
        {
            BasisVec,
            IndexZeroVec,
            IndexOneVec,
            RandomUnitVec,
            RandomVec,
            LinVec,

            IdentityMat,
            DiagonalMat,
            RandomDiagonalMat,
            IndexZeroMat,
            IndexOneMat,
            RandomMat,
            RandomRangeMat,
            RotationMat,
            PermutationMat,
            HouseholderMat,

        }

        public TestType TType;

        public void Execute()
        {
            switch(TType)
            {
                case TestType.BasisVec:
                    BasisVec();
                    break;
                case TestType.IndexZeroVec:
                    IndexZeroVec();
                    break;
                case TestType.IndexOneVec:
                    IndexOneVec();
                    break;
                case TestType.RandomUnitVec:
                    RandomUnitVec();
                    break;
                case TestType.RandomVec:
                    RandomVec();
                    break;
                case TestType.LinVec:
                    LinVec();
                    break;

                case TestType.IdentityMat:
                    IdentityMat();
                    break;
                case TestType.DiagonalMat:
                    DiagonalMat();
                    break;
                case TestType.RandomDiagonalMat:
                    RandomDiagonalMat();
                    break;
                case TestType.IndexZeroMat:
                    IndexZeroMat();
                    break;
                case TestType.IndexOneMat:
                    IndexOneMat();
                    break;
                case TestType.RandomMat:
                    RandomMat();
                    break;
                case TestType.RotationMat:
                    RotationMat();
                    break;
                case TestType.PermutationMat:
                    PermutationMat();
                    break;
                case TestType.HouseholderMat:
                    HouseholderMat();
                    break;
            }
        }

        public void BasisVec()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.fProxyBasisVector(10, 0);

            Assert.AreEqual((fProxy)1, v[0]);

            for(int i = 1; i < v.N; i++) {
                Assert.AreEqual((fProxy)0, v[i]);
            }

            v = arena.fProxyBasisVector(10, 9);

            Assert.AreEqual((fProxy)1, v[9]);

            for(int i = 0; i < v.N - 1; i++) {
                Assert.AreEqual((fProxy)0, v[i]);
            }

            arena.Dispose();
        }
        
        public void IndexZeroVec()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.fProxyIndexZeroVector(16);

            for(int i = 0; i < v.N; i++) {
                Assert.AreEqual((fProxy)i, v[i]);
            }

            arena.Dispose();
        }

        public void IndexOneVec()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.fProxyIndexOneVector(16);

            for (int i = 0; i < v.N; i++) {
                Assert.AreEqual((fProxy)i + 1, v[i]);
            }

            arena.Dispose();
        }

        public void RandomUnitVec()
        {
            var arena = new Arena(Allocator.Persistent);

            for (uint seed = 0; seed < 16; seed++)
            {
                var v = arena.fProxyRandomUnitVector(16, 332*seed+17);

                var len = fProxyNormsOP.L2(in v);

                Assert.AreEqual((fProxy)1, len, 0.00001f);
            }

            arena.Dispose();
        }

        public void RandomVec()
        {
            var arena = new Arena(Allocator.Persistent);

            for (uint seed = 0; seed < 16; seed++)
            {
                var v = arena.fProxyRandomVector(16, -3f, 3f, 351*seed+19);

                for (int i = 0; i < v.N; i++)
                    Assert.IsFalse(v[i] < -(fProxy)3 || v[i] > (fProxy)3);
            }

            arena.Dispose();
        }

        public void LinVec()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.fProxyLinVector(16, (fProxy)0, (fProxy)15);

            for (int i = 0; i < v.N; i++)
                Assert.IsTrue(math.abs(i- v[i]) < 0.0001f);

            v = arena.fProxyLinVector(16, 15, 0);

            for (int i = 0; i < v.N; i++)
                Assert.IsTrue(math.abs((15f-i) - v[i]) < 0.0001f);


            arena.Dispose();
        }

        public void IndexZeroMat()
        {
            var arena = new Arena(Allocator.Persistent);
            var m = arena.fProxyIndexZeroMatrix(16, 16);

            for(int i = 0; i < m.Length; i++)
                Assert.AreEqual((fProxy)i, m[i]);
        }

        public void IndexOneMat()
        {
            var arena = new Arena(Allocator.Persistent);
            var m = arena.fProxyIndexOneMatrix(16, 16);

            for (int i = 0; i < m.Length; i++)
                Assert.AreEqual((fProxy)i + 1, m[i]);

        }

        public void IdentityMat()
        {
            var arena = new Arena(Allocator.Persistent);
            var m = arena.fProxyIdentityMatrix(16);

            Assert.IsTrue(Analysis.IsDiagonal(in m));
            Assert.IsTrue(Analysis.IsIdentity(in m));

            for (int i = 0; i < m.M_Rows; i++)
            for(int j = 0; j < m.N_Cols; j++)
            {
                if(i == j) 
                    Assert.AreEqual((fProxy)1, m[i, j]);
                else 
                    Assert.AreEqual((fProxy)0, m[i, j]);
            }

            arena.Dispose();
        }

        public void DiagonalMat()
        {
            var arena = new Arena(Allocator.Persistent);
            var m = arena.fProxyDiagonalMatrix(16, 2f);
            
            Assert.IsTrue(Analysis.IsDiagonal(in m));

            for (int i = 0; i < m.M_Rows; i++)
            for (int j = 0; j < m.N_Cols; j++)
            {
                if (i == j)
                    Assert.AreEqual((fProxy)2, m[i, j]);
                else
                    Assert.AreEqual((fProxy)0, m[i, j]);
            }

            arena.Dispose();
        }

        public void RandomDiagonalMat()
        {
            var arena = new Arena(Allocator.Persistent);
            var m = arena.fProxyRandomDiagonalMatrix(16, -3f, 3f);

            Assert.IsTrue(Analysis.IsDiagonal(in m));

            for (int i = 0; i < m.M_Rows; i++)
            for (int j = 0; j < m.N_Cols; j++)
            { 
                if (i == j)
                    Assert.IsFalse(m[i, j] < -3f || m[i, j] > 3f);
                else
                    Assert.AreEqual((fProxy)0, m[i, j]);  
            }

            arena.Dispose();
        }

        public void RandomMat()
        {
            var arena = new Arena(Allocator.Persistent);
            var m = arena.fProxyRandomMatrix(16, 16);

            for (int i = 0; i < m.M_Rows; i++)
            for (int j = 0; j < m.N_Cols; j++)
                Assert.IsFalse(m[i, j] < -1f || m[i, j] > 1f);

            arena.Dispose();
        }

        public void RandomRangeMat()
        {
            var arena = new Arena(Allocator.Persistent);
            var m = arena.fProxyRandomMatrix(16, 16, -6f, 6f);

            for (int i = 0; i < m.M_Rows; i++)
            for (int j = 0; j < m.N_Cols; j++)
                Assert.IsFalse(m[i, j] < -6f || m[i, j] > 6f);

            arena.Dispose();
        }

        public void RotationMat()
        {
            var arena = new Arena(Allocator.Persistent);
            var m = arena.fProxyRotationMatrix(16, 1, 14, math.PI/4f);

            Assert.IsTrue(Analysis.IsOrthogonal(in m, 0.00001f));
            Assert.IsFalse(Analysis.IsIdentity(in m, 0.00001f));
            
            var mTm = fProxyOP.dot(m, m, true);
            Analysis.IsIdentity(in mTm, 0.00001f);

            m = arena.fProxyRotationMatrix(2, 0, 1, math.PI/4f);

            Assert.IsTrue(math.abs((fProxy)0.70710678118654752440084436210485d - m[0, 0]) < 0.00001f);
            Assert.IsTrue(math.abs((fProxy)0.70710678118654752440084436210485d - m[1, 1]) < 0.00001f);
            Assert.IsTrue(-math.abs((fProxy)0.70710678118654752440084436210485d - m[0, 1]) < 0.00001f);
            Assert.IsTrue(math.abs((fProxy)0.70710678118654752440084436210485d - m[1, 0]) < 0.00001f);

            arena.Dispose();
        }

        public void PermutationMat()
        {
            var arena = new Arena(Allocator.Persistent);
            var m = arena.fProxyPermutationMatrix(16, 1, 14);

            Assert.IsTrue(Analysis.IsOrthogonal(in m, 0.00001f));
            Assert.IsFalse(Analysis.IsIdentity(in m, 0.00001f));

            var mTm = fProxyOP.dot(m, m, true);
            Analysis.IsIdentity(in mTm, 0.00001f);

            m = arena.fProxyPermutationMatrix(2, 0, 1);

            Assert.AreEqual((fProxy)0, m[0, 0]);
            Assert.AreEqual((fProxy)0, m[1, 1]);
            Assert.AreEqual((fProxy)1, m[0, 1]);
            Assert.AreEqual((fProxy)1, m[1, 0]);

            arena.Dispose();
        }

        public void HouseholderMat()
        {
            var arena = new Arena(Allocator.Persistent);
            var v = arena.fProxyRandomUnitVector(16);
            var m = arena.fProxyHouseholderMatrix(16, v);

            Assert.IsTrue(Analysis.IsOrthogonal(in m, 0.00001f));
            Assert.IsFalse(Analysis.IsIdentity(in m, 0.00001f));

            var mTm = fProxyOP.dot(m, m, true);
            Analysis.IsIdentity(in mTm, 0.00001f);

            v = arena.fProxyBasisVector(2, 0);
            m = arena.fProxyHouseholderMatrix(2, v);



            arena.Dispose();
        }
    }

    [Test]
    public void BasisVec()
    {
        new TestJob() { TType = TestJob.TestType.BasisVec }.Run();
    }

    [Test]
    public void IndexZeroVec()
    {
        new TestJob() { TType = TestJob.TestType.IndexZeroVec }.Run();
    }
    [Test]
    public void IndexOneVec()
    {
        new TestJob() { TType = TestJob.TestType.IndexOneVec }.Run();
    }

    [Test]
    public void RandomUnitVec()
    {
        new TestJob() { TType = TestJob.TestType.RandomUnitVec }.Run();
    }

    [Test]
    public void RandomVec()
    {
        new TestJob() { TType = TestJob.TestType.RandomVec }.Run();
    }

    [Test]
    public void LinVec()
    {
        new TestJob() { TType = TestJob.TestType.LinVec }.Run();
    }

    [Test]
    public void IdentityMat()
    {
        new TestJob() { TType = TestJob.TestType.IdentityMat }.Run();
    }

    [Test]
    public void DiagonalMat()
    {
        new TestJob() { TType = TestJob.TestType.DiagonalMat }.Run();
    }

    [Test]
    public void RandomDiagonalMat()
    {
        new TestJob() { TType = TestJob.TestType.RandomDiagonalMat }.Run();
    }

    [Test]
    public void IndexZeroMat()
    {
        new TestJob() { TType = TestJob.TestType.IndexZeroMat }.Run();
    }

    [Test]
    public void IndexOneMat()
    {
        new TestJob() { TType = TestJob.TestType.IndexOneMat }.Run();
    }

    [Test]
    public void RandomMat()
    {
        new TestJob() { TType = TestJob.TestType.RandomMat }.Run();
    }

    [Test]
    public void RandomRangeMat()
    {
        new TestJob() { TType = TestJob.TestType.RandomRangeMat }.Run();
    }

    [Test]
    public void RotationMat()
    {
        new TestJob() { TType = TestJob.TestType.RotationMat }.Run();
    }

    [Test]
    public void PermutationMat()
    {
        new TestJob() { TType = TestJob.TestType.PermutationMat }.Run();
    }

    [Test]
    public void HouseholderMat()
    {
        new TestJob() { TType = TestJob.TestType.HouseholderMat }.Run();
    }

    
}
