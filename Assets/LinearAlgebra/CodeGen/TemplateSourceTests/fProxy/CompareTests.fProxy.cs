using BULA;
using NUnit.Framework;
using System;

using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class fProxyCompareTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct TestsJob : IJob
    {
        public enum TestType
        {
            VecEquals,
            VecNotEquals,
            VecLess,
            VecLessOrEqual,
            VecGreater,
            VecGreaterOrEqual,

            MatEquals,
            MatNotEquals,
            MatLess,
            MatLessOrEqual,
            MatGreater,
            MatGreaterOrEqual,

            VecRandom,
            MatRandom,

            MatDiagonal,

            VecVecEquals,
            VecVecNotEquals,
            VecVecLess,
            VecVecLessOrEqual,
            VecVecGreater,
            VecVecGreaterOrEqual,
            VecVecRandom,

            MatMatEquals,
            MatMatNotEquals,
            MatMatLess,
            MatMatLessOrEqual,
            MatMatGreater,
            MatMatGreaterOrEqual,
            MatMatRandom,
        }

        public TestType Type;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.VecEquals:
                    VecEquals();
                    break;
                case TestType.VecNotEquals:
                    VecNotEquals();
                    break;
                case TestType.VecLess:
                    VecLess();
                    break;
                case TestType.VecLessOrEqual:
                    VecLessOrEqual();
                    break;
                case TestType.VecGreater:
                    VecGreater();
                    break;
                case TestType.VecGreaterOrEqual:
                    VecGreaterOrEqual();
                    break;
                case TestType.VecRandom:
                    VecRandom();
                    break;

                case TestType.MatEquals:
                    MatEquals();
                    break;
                case TestType.MatNotEquals:
                    MatNotEquals();
                    break;
                case TestType.MatLess:
                    MatLess();
                    break;
                case TestType.MatLessOrEqual:
                    MatLessOrEqual();
                    break;
                case TestType.MatGreater:
                    MatGreater();
                    break;
                case TestType.MatGreaterOrEqual:
                    MatGreaterOrEqual();
                    break;

                case TestType.MatRandom:
                    MatRandom();
                    break;
                case TestType.MatDiagonal:
                    MatDiagonal();
                    break;

                case TestType.VecVecEquals:
                    VecVecEquals();
                    break;
                case TestType.VecVecNotEquals:
                    VecVecNotEquals();
                    break;
                case TestType.VecVecLess:
                    VecVecLess();
                    break;
                case TestType.VecVecLessOrEqual:
                    VecVecLessOrEqual();
                    break;
                case TestType.VecVecGreater:
                    VecVecGreater();
                    break;
                case TestType.VecVecGreaterOrEqual:
                    VecVecGreaterOrEqual();
                    break;
                case TestType.VecVecRandom:
                    VecVecRandom();
                    break;

                case TestType.MatMatEquals:
                    MatMatEquals();
                    break;
                case TestType.MatMatNotEquals:
                    MatMatNotEquals();
                    break;
                case TestType.MatMatLess:
                    MatMatLess();
                    break;
                case TestType.MatMatLessOrEqual:
                    MatMatLessOrEqual();
                    break;
                case TestType.MatMatGreater:
                    MatMatGreater();
                    break;
                case TestType.MatMatGreaterOrEqual:
                    MatMatGreaterOrEqual();
                    break;
                case TestType.MatMatRandom:
                    MatMatRandom();
                    break;



                default:
                    throw new System.NotImplementedException();
            }
        }

        public void VecEquals()
        {
            int dim = 16;

            fProxyN v = new fProxyN(dim, Allocator.Temp);

            var boolVec = v == 0f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, true));

            boolVec = v == 1f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, false));
        }

        public void VecNotEquals()
        {
            int dim = 16;

            fProxyN v = new fProxyN(dim, Allocator.Temp);

            var boolVec = v != 0f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, false));

            boolVec = v != 1f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, true));
        }

        public void VecLess()
        {
            int dim = 16;

            fProxyN v = new fProxyN(dim, Allocator.Temp);

            var boolVec = v < 0f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, false));

            boolVec = v < 1f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, true));
        }

        public void VecLessOrEqual()
        {
            int dim = 16;

            fProxyN v = new fProxyN(dim, Allocator.Temp);

            var boolVec = v <= 0f;

            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, true));
            boolVec = v <= 1f;

            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, true));

            boolVec = v <= -1f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, false));
        }

        public void VecGreater()
        {
            int dim = 16;

            fProxyN v = new fProxyN(dim, Allocator.Temp);

            var boolVec = v > 0f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, false));

            boolVec = v > -1f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, true));
        }

        public void VecGreaterOrEqual()
        {
            int dim = 16;

            fProxyN v = new fProxyN(dim, Allocator.Temp);

            var boolVec = v >= 0f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, true));

            boolVec = v >= -1f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, true));

            boolVec = v >= 1f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, false));
        }

        public void MatEquals()
        {
            int dim = 16;

            fProxyMxN m = new fProxyMxN(dim, dim, Allocator.Temp);

            var boolMat = m == 0f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, true));

            boolMat = m == 1f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, false));
        }

        public void MatNotEquals()
        {
            int dim = 16;

            fProxyMxN m = new fProxyMxN(dim, dim, Allocator.Temp);

            var boolMat = m != 0f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, false));

            boolMat = m != 1f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, true));
        }

        public void MatLess()
        {
            int dim = 16;

            fProxyMxN m = new fProxyMxN(dim, dim, Allocator.Temp);

            var boolMat = m < 0f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, false));

            boolMat = m < 1f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, true));
        }

        public void MatLessOrEqual()
        {
            int dim = 16;

            fProxyMxN m = new fProxyMxN(dim, dim, Allocator.Temp);

            var boolMat = m <= 0f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, true));

            boolMat = m <= 1f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, true));

            boolMat = m <= -1f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, false));
        }

        public void MatGreater()
        {
            int dim = 16;

            fProxyMxN m = new fProxyMxN(dim, dim, Allocator.Temp);

            var boolMat = m > 0f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, false));

            boolMat = m > -1f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, true));
        }

        public void MatGreaterOrEqual()
        {
            int dim = 16;

            fProxyMxN m = new fProxyMxN(dim, dim, Allocator.Temp);

            var boolMat = m >= 0f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, true));

            boolMat = m >= -1f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, true));

            boolMat = m >= 1f;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, false));
        }

        public void VecRandom()
        {
            int dim = 64;

            fProxyN v = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 1451, Allocator.Temp);
            // set first element to zero
            v[0] = 0f;

            var boolVec = v == 0f;

            Assert.IsFalse(Analysis.isAllSame(boolVec));

            boolVec = v != 0f;
            Assert.IsFalse(Analysis.isAllSame(boolVec));

            boolVec = v < 0f;
            Assert.IsFalse(Analysis.isAllSame(boolVec));

            boolVec = v > 0f;
            Assert.IsFalse(Analysis.isAllSame(boolVec));

            boolVec = v <= 0f;
            Assert.IsFalse(Analysis.isAllSame(boolVec));

            boolVec = v >= 0f;
            Assert.IsFalse(Analysis.isAllSame(boolVec));
        }

        public void MatRandom()
        {
            int dim = 32;

            fProxyMxN m = GenerateOP.fProxyRandomMat(dim, dim, -1f, 1f, 1451, Allocator.Temp);
            m[0,0] = 0f;

            var boolMat = m == 0f;

            Assert.IsFalse(Analysis.isAllSame(boolMat));

            boolMat = m != 0f;
            Assert.IsFalse(Analysis.isAllSame(boolMat));

            boolMat = m < 0f;
            Assert.IsFalse(Analysis.isAllSame(boolMat));

            boolMat = m > 0f;
            Assert.IsFalse(Analysis.isAllSame(boolMat));

            boolMat = m <= 0f;
            Assert.IsFalse(Analysis.isAllSame(boolMat));

            boolMat = m >= 0f;
            Assert.IsFalse(Analysis.isAllSame(boolMat));
        }

        public void MatDiagonal()
        {
            int dim = 32;

            fProxyMxN m0 = GenerateOP.fProxyDiagonalMat(dim, 1f, Allocator.Temp);

            var boolMat = m0 == 1f;

            Assert.IsTrue(Analysis.isDiagonal(boolMat));
            Assert.IsFalse(Analysis.isAllEqualTo(boolMat, true));
            Assert.IsFalse(Analysis.isAllEqualTo(boolMat, false));
        }

        public void VecVecEquals()
        {
            int dim = 16;

            fProxyN v0 = GenerateOP.fProxyLinVec(dim, 0f, 1f, Allocator.Temp);
            fProxyN v1 = GenerateOP.fProxyLinVec(dim, 0f, 1f, Allocator.Temp);

            var boolVec = v0 == v1;

            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, true));

            v0[0] = 1f;

            boolVec = v0 == v1;

            Assert.IsFalse(Analysis.isAllSame(boolVec));
        }

        public void VecVecNotEquals()
        {
            int dim = 16;

            fProxyN v0 = GenerateOP.fProxyLinVec(dim, 0f, 1f, Allocator.Temp);
            fProxyN v1 = GenerateOP.fProxyLinVec(dim, 2f, 3f, Allocator.Temp);

            var boolVec = v0 != v1;

            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, true));

            v0 = v1;

            boolVec = v0 != v1;

            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, false));
        }

        public void VecVecLess()
        {

            int dim = 16;

            fProxyN v0 = GenerateOP.fProxyLinVec(dim, 0f, 1f, Allocator.Temp);
            fProxyN v1 = GenerateOP.fProxyLinVec(dim, 2f, 3f, Allocator.Temp);

            var boolVec = v0 < v1;
            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, true));

            v0 = v1;

            boolVec = v0 < v1;
            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, false));
        }

        public void VecVecLessOrEqual()
        {
            int dim = 16;

            fProxyN v0 = GenerateOP.fProxyLinVec(dim, 0f, 1f, Allocator.Temp);
            fProxyN v1 = GenerateOP.fProxyLinVec(dim, 2f, 3f, Allocator.Temp);

            var boolVec = v0 <= v1;

            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, true));

            v0 = v1;

            boolVec = v0 <= v1;

            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, true));

            v0 = GenerateOP.fProxyLinVec(dim, 0f, 1f, Allocator.Temp);
            v1 = GenerateOP.fProxyLinVec(dim, 1f, 0f, Allocator.Temp);

            boolVec = v0 <= v1;

            Assert.IsFalse(Analysis.isAllSame(boolVec));
        }

        public void VecVecGreater()
        {
            int dim = 16;

            fProxyN v0 = GenerateOP.fProxyLinVec(dim, 0f, 1f, Allocator.Temp);
            fProxyN v1 = GenerateOP.fProxyLinVec(dim, 2f, 3f, Allocator.Temp);

            var boolVec = v0 > v1;

            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, false));

            v0 = v1;

            boolVec = v0 > v1;

            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, false));

            v0 = GenerateOP.fProxyLinVec(dim, 1f, 0f, Allocator.Temp);
            v1 = GenerateOP.fProxyLinVec(dim, 0f, 1f, Allocator.Temp);

            boolVec = v0 > v1;
            Assert.IsFalse(Analysis.isAllSame(boolVec));

            boolVec = v1 > v0;
            Assert.IsFalse(Analysis.isAllSame(boolVec));
        }

        public void VecVecGreaterOrEqual()
        {
            int dim = 16;

            fProxyN v0 = GenerateOP.fProxyLinVec(dim, 0f, 1f, Allocator.Temp);
            fProxyN v1 = GenerateOP.fProxyLinVec(dim, 2f, 3f, Allocator.Temp);

            var boolVec = v0 >= v1;

            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, false));
            v0 = v1;

            boolVec = v0 >= v1;
            Assert.IsTrue(Analysis.isAllEqualTo(boolVec, true));

            v0 = GenerateOP.fProxyLinVec(dim, 1f, 0f, Allocator.Temp);

            boolVec = v0 >= v1;
            Assert.IsTrue(Analysis.isAllSame(boolVec));
        }

        public void VecVecRandom()
        {
            int dim = 64;

            fProxyN v0 = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 1451, Allocator.Temp);
            fProxyN v1 = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 6421, Allocator.Temp);

            v0[0] = v1[0];
            v0[1] = 1f-v1[1];
            var boolVec = v0 == v1;

            Assert.IsFalse(Analysis.isAllSame(boolVec));

            boolVec = v0 != v1;
            Assert.IsFalse(Analysis.isAllSame(boolVec));

            boolVec = v0 < v1;
            Assert.IsFalse(Analysis.isAllSame(boolVec));

            boolVec = v0 > v1;
            Assert.IsFalse(Analysis.isAllSame(boolVec));

            boolVec = v0 <= v1;
            Assert.IsFalse(Analysis.isAllSame(boolVec));

            boolVec = v0 >= v1;
            Assert.IsFalse(Analysis.isAllSame(boolVec));
        }

        public void MatMatEquals()
        {
            int dim = 16;

            fProxyMxN m0 = GenerateOP.fProxyRandomMat(dim, dim, 0f, 1f, allocator: Allocator.Temp);
            fProxyMxN m1 = GenerateOP.fProxyRandomMat(dim, dim, 0f, 1f, allocator: Allocator.Temp);

            var boolMat = m0 == m1;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, true));

            m0[0,0] = 1f;
            m0[1,1] = 1f;
            m0[2,2] = 1f;
            m0[3,3] = 1f;

            boolMat = m0 == m1;
            Assert.IsFalse(Analysis.isAllSame(boolMat));
        }

        public void MatMatNotEquals()
        {
            int dim = 16;

            fProxyMxN m0 = GenerateOP.fProxyRandomMat(dim, dim, 0f, 1f, 2131, Allocator.Temp);
            fProxyMxN m1 = GenerateOP.fProxyRandomMat(dim, dim, 0f, 1f, 2131, Allocator.Temp);

            var boolMat = m0 != m1;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, false));

            m1 = GenerateOP.fProxyRandomMat(dim, dim, 2f, 3f, 2131, Allocator.Temp);

            boolMat = m0 != m1;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, true));
        }

        public void MatMatLess()
        {
            int dim = 16;

            fProxyMxN m0 = GenerateOP.fProxyRandomMat(dim, dim, 0f, 1f, 2131, Allocator.Temp);
            fProxyMxN m1 = GenerateOP.fProxyRandomMat(dim, dim, 2f, 3f, 2131, Allocator.Temp);

            var boolMat = m0 < m1;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, true));

            m0 = m1;

            boolMat = m0 < m1;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, false));
        }

        public void MatMatLessOrEqual()
        {
            int dim = 16;

            fProxyMxN m0 = GenerateOP.fProxyRandomMat(dim, dim, 0f, 1f, 2131, Allocator.Temp);
            fProxyMxN m1 = GenerateOP.fProxyRandomMat(dim, dim, 2f, 3f, 2131, Allocator.Temp);

            var boolMat = m0 <= m1;

            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, true));

            m0 = m1;

            boolMat = m0 <= m1;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, true));

            m0 = GenerateOP.fProxyRandomMat(dim, dim, 1f, 0f, 2131, Allocator.Temp);
            m1 = GenerateOP.fProxyRandomMat(dim, dim, 0f, 1f, 2131, Allocator.Temp);

            boolMat = m0 <= m1;

            Assert.IsFalse(Analysis.isAllSame(boolMat));
        }

        public void MatMatGreater()
        {
            int dim = 16;

            fProxyMxN m0 = GenerateOP.fProxyRandomMat(dim, dim, 0f, 1f, 2131, Allocator.Temp);
            fProxyMxN m1 = GenerateOP.fProxyRandomMat(dim, dim, 2f, 3f, 2131, Allocator.Temp);

            var boolMat = m0 > m1;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, false));

            m0 = m1;

            boolMat = m0 > m1;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, false));

            m0 = GenerateOP.fProxyRandomMat(dim, dim, 1f, 0f, 2131, Allocator.Temp);
            m1 = GenerateOP.fProxyRandomMat(dim, dim, 0f, 1f, 2131, Allocator.Temp);

            boolMat = m0 > m1;
            Assert.IsFalse(Analysis.isAllSame(boolMat));

            boolMat = m1 > m0;
            Assert.IsFalse(Analysis.isAllSame(boolMat));
        }

        public void MatMatGreaterOrEqual()
        {
            int dim = 16;

            fProxyMxN m0 = GenerateOP.fProxyRandomMat(dim, dim, 0f, 1f, 2131, Allocator.Temp);
            fProxyMxN m1 = GenerateOP.fProxyRandomMat(dim, dim, 2f, 3f, 2131, Allocator.Temp);

            var boolMat = m0 >= m1;
            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, false));

            m0 = m1;

            boolMat = m0 >= m1;

            Assert.IsTrue(Analysis.isAllEqualTo(boolMat, true));
            m0 = GenerateOP.fProxyRandomMat(dim, dim, 1f, 0f, 2131, Allocator.Temp);

            boolMat = m0 >= m1;
            Assert.IsTrue(Analysis.isAllSame(boolMat));
        }

        public void MatMatRandom()
        {
            int dim = 32;

            fProxyMxN m0 = GenerateOP.fProxyRandomMat(dim, dim, -1f, 1f, 1451, Allocator.Temp);
            fProxyMxN m1 = GenerateOP.fProxyRandomMat(dim, dim, -1f, 1f, 6421, Allocator.Temp);

            m0[0,0] = m1[0,0];
            m0[0,1] = 1f - m1[0,1];
            var boolMat = m0 == m1;

            Assert.IsFalse(Analysis.isAllSame(boolMat));

            boolMat = m0 != m1;
            Assert.IsFalse(Analysis.isAllSame(boolMat));

            boolMat = m0 < m1;
            Assert.IsFalse(Analysis.isAllSame(boolMat));

            boolMat = m0 > m1;
            Assert.IsFalse(Analysis.isAllSame(boolMat));

            boolMat = m0 <= m1;
            Assert.IsFalse(Analysis.isAllSame(boolMat));

            boolMat = m0 >= m1;
            Assert.IsFalse(Analysis.isAllSame(boolMat));
        }
    }

    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(TestsJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void Test(TestsJob.TestType type)
    {
        new TestsJob() { Type = type }.Run();
    }

}
