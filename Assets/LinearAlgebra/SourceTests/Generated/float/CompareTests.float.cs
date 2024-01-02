using LinearAlgebra;
using NUnit.Framework;
using System;

using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class floatCompareTests
{
    [BurstCompile]
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
            var arena = new Arena(Allocator.Persistent);
            try
            {

                switch (Type)
                {
                    case TestType.VecEquals:
                        VecEquals(ref arena);
                        break;
                    case TestType.VecNotEquals:
                        VecNotEquals(ref arena);
                        break;
                    case TestType.VecLess:
                        VecLess(ref arena);
                        break;
                    case TestType.VecLessOrEqual:
                        VecLessOrEqual(ref arena);
                        break;
                    case TestType.VecGreater:
                        VecGreater(ref arena);
                        break;
                    case TestType.VecGreaterOrEqual:
                        VecGreaterOrEqual(ref arena);
                        break;
                    case TestType.VecRandom:
                        VecRandom(ref arena);
                        break;

                    case TestType.MatEquals:
                        MatEquals(ref arena);
                        break;
                    case TestType.MatNotEquals:
                        MatNotEquals(ref arena);
                        break;
                    case TestType.MatLess:
                        MatLess(ref arena);
                        break;
                    case TestType.MatLessOrEqual:
                        MatLessOrEqual(ref arena);
                        break;
                    case TestType.MatGreater:
                        MatGreater(ref arena);
                        break;
                    case TestType.MatGreaterOrEqual:
                        MatGreaterOrEqual(ref arena);
                        break;

                    case TestType.MatRandom:
                        MatRandom(ref arena);
                        break;
                    case TestType.MatDiagonal:
                        MatDiagonal(ref arena);
                        break;

                    case TestType.VecVecEquals:
                        VecVecEquals(ref arena);
                        break;
                    case TestType.VecVecNotEquals:
                        VecVecNotEquals(ref arena);
                        break;
                    case TestType.VecVecLess:
                        VecVecLess(ref arena);
                        break;
                    case TestType.VecVecLessOrEqual:
                        VecVecLessOrEqual(ref arena);
                        break;
                    case TestType.VecVecGreater:
                        VecVecGreater(ref arena);
                        break;
                    case TestType.VecVecGreaterOrEqual:
                        VecVecGreaterOrEqual(ref arena);
                        break;
                    case TestType.VecVecRandom:
                        VecVecRandom(ref arena);
                        break;
                    
                    case TestType.MatMatEquals:
                        MatMatEquals(ref arena);
                        break;
                    case TestType.MatMatNotEquals:
                        MatMatNotEquals(ref arena);
                        break;
                    case TestType.MatMatLess:
                        MatMatLess(ref arena);
                        break;
                    case TestType.MatMatLessOrEqual:
                        MatMatLessOrEqual(ref arena);
                        break;
                    case TestType.MatMatGreater:
                        MatMatGreater(ref arena);
                        break;
                    case TestType.MatMatGreaterOrEqual:
                        MatMatGreaterOrEqual(ref arena);
                        break;
                    case TestType.MatMatRandom:
                        MatMatRandom(ref arena);
                        break;



                    default:
                        throw new System.NotImplementedException();
                }
            }
            finally
            {
                arena.Dispose();
            }
        }

        public void VecEquals(ref Arena arena)
        {
            int dim = 16;
            
            floatN v = arena.floatVec(dim);

            var boolVec = v == 0f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            boolVec = v == 1f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));
        }

        public void VecNotEquals(ref Arena arena)
        {
            int dim = 16;
            
            floatN v = arena.floatVec(dim);

            var boolVec = v != 0f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));

            boolVec = v != 1f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));
        }

        public void VecLess(ref Arena arena)
        {
            int dim = 16;
            
            floatN v = arena.floatVec(dim);

            var boolVec = v < 0f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));

            boolVec = v < 1f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));
        }

        public void VecLessOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            floatN v = arena.floatVec(dim);

            var boolVec = v <= 0f;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));
            boolVec = v <= 1f;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            boolVec = v <= -1f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));
        }

        public void VecGreater(ref Arena arena)
        {
            int dim = 16;
            
            floatN v = arena.floatVec(dim);

            var boolVec = v > 0f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));

            boolVec = v > -1f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));
        }

        public void VecGreaterOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            floatN v = arena.floatVec(dim);

            var boolVec = v >= 0f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            boolVec = v >= -1f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            boolVec = v >= 1f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));
        }

        public void MatEquals(ref Arena arena)
        {
            int dim = 16;
            
            floatMxN m = arena.floatMat(dim, dim);

            var boolMat = m == 0f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));

            boolMat = m == 1f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));
        }

        public void MatNotEquals(ref Arena arena)
        {
            int dim = 16;
            
            floatMxN m = arena.floatMat(dim, dim);

            var boolMat = m != 0f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));

            boolMat = m != 1f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));
        }

        public void MatLess(ref Arena arena)
        {
            int dim = 16;
            
            floatMxN m = arena.floatMat(dim, dim);

            var boolMat = m < 0f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));

            boolMat = m < 1f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));
        }

        public void MatLessOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            floatMxN m = arena.floatMat(dim, dim);

            var boolMat = m <= 0f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));

            boolMat = m <= 1f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));

            boolMat = m <= -1f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));
        }

        public void MatGreater(ref Arena arena)
        {
            int dim = 16;
            
            floatMxN m = arena.floatMat(dim, dim);

            var boolMat = m > 0f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));

            boolMat = m > -1f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));
        }

        public void MatGreaterOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            floatMxN m = arena.floatMat(dim, dim);

            var boolMat = m >= 0f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));

            boolMat = m >= -1f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));

            boolMat = m >= 1f;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));
        }

        public void VecRandom(ref Arena arena)
        {
            int dim = 64;

            floatN v = arena.floatRandomVector(dim, -1f, 1f, 1451);
            // set first element to zero
            v[0] = 0f;

            var boolVec = v == 0f;

            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));

            boolVec = v != 0f;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));

            boolVec = v < 0f;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));

            boolVec = v > 0f;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));

            boolVec = v <= 0f;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));

            boolVec = v >= 0f;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));
        }

        public void MatRandom(ref Arena arena)
        {
            int dim = 32;

            floatMxN m = arena.floatRandomMatrix(dim, dim, -1f, 1f, 1451);
            // set first element to zero
            m[0,0] = 0f;

            var boolMat = m == 0f;

            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));

            boolMat = m != 0f;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));

            boolMat = m < 0f;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));

            boolMat = m > 0f;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));

            boolMat = m <= 0f;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));

            boolMat = m >= 0f;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));
        }

        public void MatDiagonal(ref Arena arena)
        {
            int dim = 32;

            floatMxN m0 = arena.floatDiagonalMatrix(dim, 1f);
            
            var boolMat = m0 == 1f;

            Assert.IsTrue(BoolAnalysis.IsDiagonal(boolMat));
            Assert.IsFalse(BoolAnalysis.IsAllEqualTo(boolMat, true));
            Assert.IsFalse(BoolAnalysis.IsAllEqualTo(boolMat, false));
        }

        public void VecVecEquals(ref Arena arena)
        {
            int dim = 16;
            
            floatN v0 = arena.floatLinVector(dim, 0f, 1f);
            floatN v1 = arena.floatLinVector(dim, 0f, 1f);

            var boolVec = v0 == v1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            v0[0] = 1f;

            boolVec = v0 == v1;

            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));
        }

        public void VecVecNotEquals(ref Arena arena)
        {
            int dim = 16;
            
            floatN v0 = arena.floatLinVector(dim, 0f, 1f);
            floatN v1 = arena.floatLinVector(dim, 2f, 3f);

            var boolVec = v0 != v1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            v0 = v1;

            boolVec = v0 != v1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));
        }

        public void VecVecLess(ref Arena arena)
        {

            int dim = 16;
            
            floatN v0 = arena.floatLinVector(dim, 0f, 1f);
            floatN v1 = arena.floatLinVector(dim, 2f, 3f);

            var boolVec = v0 < v1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            v0 = v1;

            boolVec = v0 < v1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));
        }

        public void VecVecLessOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            floatN v0 = arena.floatLinVector(dim, 0f, 1f);
            floatN v1 = arena.floatLinVector(dim, 2f, 3f);

            var boolVec = v0 <= v1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            v0 = v1;

            boolVec = v0 <= v1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            v0 = arena.floatLinVector(dim, 0f, 1f);
            v1 = arena.floatLinVector(dim, 1f, 0f);

            boolVec = v0 <= v1;

            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));
        }

        public void VecVecGreater(ref Arena arena)
        {
            int dim = 16;
            
            floatN v0 = arena.floatLinVector(dim, 0f, 1f);
            floatN v1 = arena.floatLinVector(dim, 2f, 3f);

            var boolVec = v0 > v1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));

            v0 = v1;

            boolVec = v0 > v1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));

            v0 = arena.floatLinVector(dim, 1f, 0f);
            v1 = arena.floatLinVector(dim, 0f, 1f);

            boolVec = v0 > v1;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));

            boolVec = v1 > v0;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));
        }

        public void VecVecGreaterOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            floatN v0 = arena.floatLinVector(dim, 0f, 1f);
            floatN v1 = arena.floatLinVector(dim, 2f, 3f);

            var boolVec = v0 >= v1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));
            v0 = v1;

            boolVec = v0 >= v1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            v0 = arena.floatLinVector(dim, 1f, 0f);

            boolVec = v0 >= v1;
            Assert.IsTrue(BoolAnalysis.IsAllSame(boolVec));
        }

        public void VecVecRandom(ref Arena arena)
        {
            int dim = 64;

            floatN v0 = arena.floatRandomVector(dim, -1f, 1f, 1451);
            floatN v1 = arena.floatRandomVector(dim, -1f, 1f, 6421);

            v0[0] = v1[0];
            v0[1] = 1f-v1[1];
            var boolVec = v0 == v1;

            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));

            boolVec = v0 != v1;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));

            boolVec = v0 < v1;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));

            boolVec = v0 > v1;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));

            boolVec = v0 <= v1;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));

            boolVec = v0 >= v1;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));
        }

        public void MatMatEquals(ref Arena arena)
        {
            int dim = 16;
            
            floatMxN m0 = arena.floatRandomMatrix(dim, dim, 0f, 1f);
            floatMxN m1 = arena.floatRandomMatrix(dim, dim, 0f, 1f);

            var boolMat = m0 == m1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));

            m0[0,0] = 1f;
            m0[1,1] = 1f;
            m0[2,2] = 1f;
            m0[3,3] = 1f;

            boolMat = m0 == m1;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));
        }

        public void MatMatNotEquals(ref Arena arena)
        {
            int dim = 16;
            
            floatMxN m0 = arena.floatRandomMatrix(dim, dim, 0f, 1f, 2131);
            floatMxN m1 = arena.floatRandomMatrix(dim, dim, 0f, 1f, 2131);

            var boolMat = m0 != m1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));

            m1 = arena.floatRandomMatrix(dim, dim, 2f, 3f, 2131);

            boolMat = m0 != m1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));
        }

        public void MatMatLess(ref Arena arena)
        {
            int dim = 16;
            
            floatMxN m0 = arena.floatRandomMatrix(dim, dim, 0f, 1f, 2131);
            floatMxN m1 = arena.floatRandomMatrix(dim, dim, 2f, 3f, 2131);

            var boolMat = m0 < m1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));

            m0 = m1;

            boolMat = m0 < m1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));
        }

        public void MatMatLessOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            floatMxN m0 = arena.floatRandomMatrix(dim, dim, 0f, 1f, 2131);
            floatMxN m1 = arena.floatRandomMatrix(dim, dim, 2f, 3f, 2131);

            var boolMat = m0 <= m1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));

            m0 = m1;

            boolMat = m0 <= m1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));

            m0 = arena.floatRandomMatrix(dim, dim, 1f, 0f, 2131);
            m1 = arena.floatRandomMatrix(dim, dim, 0f, 1f, 2131);

            boolMat = m0 <= m1;

            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));
        }

        public void MatMatGreater(ref Arena arena)
        {
            int dim = 16;
            
            floatMxN m0 = arena.floatRandomMatrix(dim, dim, 0f, 1f, 2131);
            floatMxN m1 = arena.floatRandomMatrix(dim, dim, 2f, 3f, 2131);

            var boolMat = m0 > m1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));

            m0 = m1;

            boolMat = m0 > m1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));

            m0 = arena.floatRandomMatrix(dim, dim, 1f, 0f, 2131);
            m1 = arena.floatRandomMatrix(dim, dim, 0f, 1f, 2131);

            boolMat = m0 > m1;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));

            boolMat = m1 > m0;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));
        }

        public void MatMatGreaterOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            floatMxN m0 = arena.floatRandomMatrix(dim, dim, 0f, 1f, 2131);
            floatMxN m1 = arena.floatRandomMatrix(dim, dim, 2f, 3f, 2131);

            var boolMat = m0 >= m1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));

            m0 = m1;

            boolMat = m0 >= m1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));
            m0 = arena.floatRandomMatrix(dim, dim, 1f, 0f, 2131);

            boolMat = m0 >= m1;
            Assert.IsTrue(BoolAnalysis.IsAllSame(boolMat));
        }

        public void MatMatRandom(ref Arena arena)
        {
            int dim = 32;

            floatMxN m0 = arena.floatRandomMatrix(dim, dim, -1f, 1f, 1451);
            floatMxN m1 = arena.floatRandomMatrix(dim, dim, -1f, 1f, 6421);

            m0[0,0] = m1[0,0];
            m0[0,1] = 1f - m1[0,1];
            var boolMat = m0 == m1;

            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));

            boolMat = m0 != m1;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));

            boolMat = m0 < m1;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));

            boolMat = m0 > m1;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));

            boolMat = m0 <= m1;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));

            boolMat = m0 >= m1;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));
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
