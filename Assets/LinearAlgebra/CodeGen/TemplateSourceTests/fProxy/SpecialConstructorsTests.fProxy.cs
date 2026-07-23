using LinearAlgebra;
using NUnit.Framework;

using System.Diagnostics;

using Unity.Mathematics;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class fProxySpecialConstructorsTests {

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High)] 
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
                case TestType.RandomRangeMat:
                    RandomRangeMat();
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
            var v = GenerateOP.fProxyBasisVec(10, 0);

            Assert.IsTrue(v[0] == (fProxy)1);

            for(int i = 1; i < v.N; i++) {
                Assert.IsTrue(v[i] == (fProxy)0);
            }

            v = GenerateOP.fProxyBasisVec(10, 9);

            Assert.IsTrue(v[9] == (fProxy)1);

            for(int i = 0; i < v.N - 1; i++) {
                Assert.IsTrue(v[i] == (fProxy)0);
            }
        }

        public void IndexZeroVec()
        {
            var v = GenerateOP.fProxyIndexZeroVec(16);

            for(int i = 0; i < v.N; i++) {
                Assert.IsTrue(v[i] == (fProxy)i);
            }
        }

        public void IndexOneVec()
        {
            var v = GenerateOP.fProxyIndexOneVec(16);

            for (int i = 0; i < v.N; i++) {
                Assert.IsTrue(v[i] == (fProxy)i + 1);
            }
        }

        public void RandomUnitVec()
        {
            for (uint seed = 0; seed < 16; seed++)
            {
                var v = GenerateOP.fProxyRandomUnitVec(16, 332*seed+17);

                var len = Norms.L2(in v);

                Assert.IsTrue(Unity.Mathematics.math.abs(len - (fProxy)1) <= 0.00001f);
            }
        }

        public void RandomVec()
        {
            for (uint seed = 0; seed < 16; seed++)
            {
                var v = GenerateOP.fProxyRandomVec(16, -3f, 3f, 351*seed+19);

                for (int i = 0; i < v.N; i++)
                    Assert.IsFalse(v[i] < -(fProxy)3 || v[i] > (fProxy)3);
            }
        }

        public void LinVec()
        {
            var v = GenerateOP.fProxyLinVec(16, (fProxy)0, (fProxy)15);

            for (int i = 0; i < v.N; i++)
                Assert.IsTrue(math.abs(i- v[i]) < 0.0001f);

            v = GenerateOP.fProxyLinVec(16, 15, 0);

            for (int i = 0; i < v.N; i++)
                Assert.IsTrue(math.abs((15f-i) - v[i]) < 0.0001f);
        }

        public void IndexZeroMat()
        {
            var m = GenerateOP.fProxyIndexZeroMat(16, 16);

            for(int i = 0; i < m.Length; i++)
                Assert.IsTrue(m[i] == (fProxy)i);
        }

        public void IndexOneMat()
        {
            var m = GenerateOP.fProxyIndexOneMat(16, 16);

            for (int i = 0; i < m.Length; i++)
                Assert.IsTrue(m[i] == (fProxy)i + 1);
        }

        public void IdentityMat()
        {
            var m = GenerateOP.fProxyIdentityMat(16);

            Assert.IsTrue(Analysis.isDiagonal(in m));
            Assert.IsTrue(Analysis.isIdentity(in m));

            for (int i = 0; i < m.M_Rows; i++)
            for(int j = 0; j < m.N_Cols; j++)
            {
                if(i == j)
                    Assert.IsTrue(m[i, j] == (fProxy)1);
                else
                    Assert.IsTrue(m[i, j] == (fProxy)0);
            }
        }

        public void DiagonalMat()
        {
            var m = GenerateOP.fProxyDiagonalMat(16, 2f);

            Assert.IsTrue(Analysis.isDiagonal(in m));

            for (int i = 0; i < m.M_Rows; i++)
            for (int j = 0; j < m.N_Cols; j++)
            {
                if (i == j)
                    Assert.IsTrue(m[i, j] == (fProxy)2);
                else
                    Assert.IsTrue(m[i, j] == (fProxy)0);
            }
        }

        public void RandomDiagonalMat()
        {
            var m = GenerateOP.fProxyRandomDiagonalMat(16, -3f, 3f);

            Assert.IsTrue(Analysis.isDiagonal(in m));

            for (int i = 0; i < m.M_Rows; i++)
            for (int j = 0; j < m.N_Cols; j++)
            {
                if (i == j)
                    Assert.IsFalse(m[i, j] < -3f || m[i, j] > 3f);
                else
                    Assert.IsTrue(m[i, j] == (fProxy)0);
            }
        }

        public void RandomMat()
        {
            var m = GenerateOP.fProxyRandomMat(16, 16);

            for (int i = 0; i < m.M_Rows; i++)
            for (int j = 0; j < m.N_Cols; j++)
                Assert.IsFalse(m[i, j] < -1f || m[i, j] > 1f);
        }

        public void RandomRangeMat()
        {
            var m = GenerateOP.fProxyRandomMat(16, 16, -6f, 6f);

            for (int i = 0; i < m.M_Rows; i++)
            for (int j = 0; j < m.N_Cols; j++)
                Assert.IsFalse(m[i, j] < -6f || m[i, j] > 6f);
        }

        public void RotationMat()
        {
            var m = GenerateOP.fProxyRotationMat(16, 1, 14, math.PI/4f);

            Assert.IsTrue(Analysis.isOrthogonal(in m, 0.00001f));
            Assert.IsFalse(Analysis.isIdentity(in m, 0.00001f));

            var mTm = Blas.dot(m, m, true);
            Assert.IsTrue(Analysis.isIdentity(in mTm, 0.00001f));

            m = GenerateOP.fProxyRotationMat(2, 0, 1, math.PI/4f);

            Assert.IsTrue(math.abs((fProxy)0.70710678118654752440084436210485d - m[0, 0]) < 0.00001f);
            Assert.IsTrue(math.abs((fProxy)0.70710678118654752440084436210485d - m[1, 1]) < 0.00001f);
            Assert.IsTrue(math.abs((fProxy)(-0.70710678118654752440084436210485d) - m[0, 1]) < 0.00001f);
            Assert.IsTrue(math.abs((fProxy)0.70710678118654752440084436210485d - m[1, 0]) < 0.00001f);
        }

        public void PermutationMat()
        {
            var m = GenerateOP.fProxyPermutationMat(16, 1, 14);

            Assert.IsTrue(Analysis.isOrthogonal(in m, 0.00001f));
            Assert.IsFalse(Analysis.isIdentity(in m, 0.00001f));

            var mTm = Blas.dot(m, m, true);
            Assert.IsTrue(Analysis.isIdentity(in mTm, 0.00001f));

            m = GenerateOP.fProxyPermutationMat(2, 0, 1);

            Assert.IsTrue(m[0, 0] == (fProxy)0);
            Assert.IsTrue(m[1, 1] == (fProxy)0);
            Assert.IsTrue(m[0, 1] == (fProxy)1);
            Assert.IsTrue(m[1, 0] == (fProxy)1);
        }

        public void HouseholderMat()
        {
            var v = GenerateOP.fProxyRandomUnitVec(16);
            var m = GenerateOP.fProxyHouseholderMat(16, v);

            Assert.IsTrue(Analysis.isOrthogonal(in m, 0.00001f));
            Assert.IsFalse(Analysis.isIdentity(in m, 0.00001f));

            var mTm = Blas.dot(m, m, true);
            Assert.IsTrue(Analysis.isIdentity(in mTm, 0.00001f));

            v = GenerateOP.fProxyBasisVec(2, 0);
            m = GenerateOP.fProxyHouseholderMat(2, v);
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
