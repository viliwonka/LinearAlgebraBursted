using LinearAlgebra;
using NUnit.Framework;
using System;

using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class shortCompareTests
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
            
            shortN v = arena.shortVec(dim);

            var boolVec = v == 0;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            boolVec = v == 1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));
        }

        public void VecNotEquals(ref Arena arena)
        {
            int dim = 16;
            
            shortN v = arena.shortVec(dim);

            var boolVec = v != 0;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));

            boolVec = v != 1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));
        }

        public void VecLess(ref Arena arena)
        {
            int dim = 16;
            
            shortN v = arena.shortVec(dim);

            var boolVec = v < 0;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));

            boolVec = v < 1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));
        }

        public void VecLessOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            shortN v = arena.shortVec(dim);

            var boolVec = v <= 0;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));
            boolVec = v <= 1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            boolVec = v <= -1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));
        }

        public void VecGreater(ref Arena arena)
        {
            int dim = 16;
            
            shortN v = arena.shortVec(dim);

            var boolVec = v > 0;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));

            boolVec = v > -1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));
        }

        public void VecGreaterOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            shortN v = arena.shortVec(dim);

            var boolVec = v >= 0;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            boolVec = v >= -1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            boolVec = v >= 1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));
        }

        public void MatEquals(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m = arena.shortMat(dim, dim);

            var boolMat = m == 0;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));

            boolMat = m == 1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));
        }

        public void MatNotEquals(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m = arena.shortMat(dim, dim);

            var boolMat = m != 0;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));

            boolMat = m != 1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));
        }

        public void MatLess(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m = arena.shortMat(dim, dim);

            var boolMat = m < 0;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));

            boolMat = m < 1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));
        }

        public void MatLessOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m = arena.shortMat(dim, dim);

            var boolMat = m <= 0;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));

            boolMat = m <= 1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));

            boolMat = m <= -1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));
        }

        public void MatGreater(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m = arena.shortMat(dim, dim);

            var boolMat = m > 0;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));

            boolMat = m > -1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));
        }

        public void MatGreaterOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m = arena.shortMat(dim, dim);

            var boolMat = m >= 0;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));

            boolMat = m >= -1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));

            boolMat = m >= 1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));
        }

        public void VecRandom(ref Arena arena)
        {
            int dim = 64;

            shortN v = arena.shortRandomVector(dim, -100, 100, 1451);
            // set first element to zero
            v[0] = 0;

            var boolVec = v == 0;

            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));

            boolVec = v != 0;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));

            boolVec = v < 0;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));

            boolVec = v > 0;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));

            boolVec = v <= 0;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));

            boolVec = v >= 0;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));
        }

        public void MatRandom(ref Arena arena)
        {
            int dim = 32;

            shortMxN m = arena.shortRandomMatrix(dim, dim, -100, 100, 1451);
            // set first element to zero
            m[0,0] = 0;

            var boolMat = m == 0;

            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));

            boolMat = m != 0;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));

            boolMat = m < 0;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));

            boolMat = m > 0;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));

            boolMat = m <= 0;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));

            boolMat = m >= 0;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));
        }

        public void MatDiagonal(ref Arena arena)
        {
            int dim = 32;

            shortMxN m0 = arena.shortDiagonalMatrix(dim, 1);
            
            var boolMat = m0 == 1;

            Assert.IsTrue(BoolAnalysis.IsDiagonal(boolMat));
            Assert.IsFalse(BoolAnalysis.IsAllEqualTo(boolMat, true));
            Assert.IsFalse(BoolAnalysis.IsAllEqualTo(boolMat, false));
        }

        public void VecVecEquals(ref Arena arena)
        {
            int dim = 16;
            
            shortN v0 = arena.shortLinVector(dim, 0, 100);
            shortN v1 = arena.shortLinVector(dim, 0, 100);

            var boolVec = v0 == v1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            v0[0] = 1;

            boolVec = v0 == v1;

            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));
        }

        public void VecVecNotEquals(ref Arena arena)
        {
            int dim = 16;
            
            shortN v0 = arena.shortLinVector(dim, 0, 100);
            shortN v1 = arena.shortLinVector(dim, 200, 300);

            var boolVec = v0 != v1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            v0 = v1;

            boolVec = v0 != v1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));
        }

        public void VecVecLess(ref Arena arena)
        {

            int dim = 16;
            
            shortN v0 = arena.shortLinVector(dim, 0, 100);
            shortN v1 = arena.shortLinVector(dim, 200, 300);

            var boolVec = v0 < v1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            v0 = v1;

            boolVec = v0 < v1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));
        }

        public void VecVecLessOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            shortN v0 = arena.shortLinVector(dim, 0, 100);
            shortN v1 = arena.shortLinVector(dim, 200, 300);

            var boolVec = v0 <= v1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            v0 = v1;

            boolVec = v0 <= v1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            v0 = arena.shortLinVector(dim, 0, 100);
            v1 = arena.shortLinVector(dim, 100, 0);

            boolVec = v0 <= v1;

            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));
        }

        public void VecVecGreater(ref Arena arena)
        {
            int dim = 16;
            
            shortN v0 = arena.shortLinVector(dim, 0, 100);
            shortN v1 = arena.shortLinVector(dim, 200, 300);

            var boolVec = v0 > v1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));

            v0 = v1;

            boolVec = v0 > v1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));

            v0 = arena.shortLinVector(dim, 100, 0);
            v1 = arena.shortLinVector(dim, 0, 100);

            boolVec = v0 > v1;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));

            boolVec = v1 > v0;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolVec));
        }

        public void VecVecGreaterOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            shortN v0 = arena.shortLinVector(dim, 0, 100);
            shortN v1 = arena.shortLinVector(dim, 200, 300);

            var boolVec = v0 >= v1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, false));
            v0 = v1;

            boolVec = v0 >= v1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolVec, true));

            v0 = arena.shortLinVector(dim, 1, 0);

            boolVec = v0 >= v1;
            Assert.IsTrue(BoolAnalysis.IsAllSame(boolVec));
        }

        public void VecVecRandom(ref Arena arena)
        {
            int dim = 64;

            shortN v0 = arena.shortRandomVector(dim, -100, 100, 1451);
            shortN v1 = arena.shortRandomVector(dim, -100, 100, 6421);

            v0[0] = v1[0];
            v0[1] = (short)(1-v1[1]);
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
            
            shortMxN m0 = arena.shortRandomMatrix(dim, dim, 0, 100);
            shortMxN m1 = arena.shortRandomMatrix(dim, dim, 0, 100);

            var boolMat = m0 == m1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));

            m0[0,0] = 1;
            m0[1,1] = 1;
            m0[2,2] = 1;
            m0[3,3] = 1;

            boolMat = m0 == m1;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));
        }

        public void MatMatNotEquals(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m0 = arena.shortRandomMatrix(dim, dim, 0, 100, 2131);
            shortMxN m1 = arena.shortRandomMatrix(dim, dim, 0, 100, 2131);

            var boolMat = m0 != m1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));

            m1 = arena.shortRandomMatrix(dim, dim, 200, 300, 2131);

            boolMat = m0 != m1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));
        }

        public void MatMatLess(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m0 = arena.shortRandomMatrix(dim, dim, 000, 100, 2131);
            shortMxN m1 = arena.shortRandomMatrix(dim, dim, 200, 300, 2131);

            var boolMat = m0 < m1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));

            m0 = m1;

            boolMat = m0 < m1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));
        }

        public void MatMatLessOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m0 = arena.shortRandomMatrix(dim, dim, 0, 100, 2131);
            shortMxN m1 = arena.shortRandomMatrix(dim, dim, 200, 300, 2131);

            var boolMat = m0 <= m1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));

            m0 = m1;

            boolMat = m0 <= m1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));

            m0 = arena.shortRandomMatrix(dim, dim, 100, 0, 2131);
            m1 = arena.shortRandomMatrix(dim, dim, 0, 100, 2131);

            boolMat = m0 <= m1;

            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));
        }

        public void MatMatGreater(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m0 = arena.shortRandomMatrix(dim, dim, 0, 100, 2131);
            shortMxN m1 = arena.shortRandomMatrix(dim, dim, 200, 300, 2131);

            var boolMat = m0 > m1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));

            m0 = m1;

            boolMat = m0 > m1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));

            m0 = arena.shortRandomMatrix(dim, dim, 100, 0, 2131);
            m1 = arena.shortRandomMatrix(dim, dim, 0, 100, 2131);

            boolMat = m0 > m1;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));

            boolMat = m1 > m0;
            Assert.IsFalse(BoolAnalysis.IsAllSame(boolMat));
        }

        public void MatMatGreaterOrEqual(ref Arena arena)
        {
            int dim = 16;
            
            shortMxN m0 = arena.shortRandomMatrix(dim, dim, 0, 100, 2131);
            shortMxN m1 = arena.shortRandomMatrix(dim, dim, 200, 300, 2131);

            var boolMat = m0 >= m1;
            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, false));

            m0 = m1;

            boolMat = m0 >= m1;

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(boolMat, true));
            m0 = arena.shortRandomMatrix(dim, dim, 100, 0, 2131);

            boolMat = m0 >= m1;
            Assert.IsTrue(BoolAnalysis.IsAllSame(boolMat));
        }

        public void MatMatRandom(ref Arena arena)
        {
            int dim = 32;

            shortMxN m0 = arena.shortRandomMatrix(dim, dim, -100, 100, 1451);
            shortMxN m1 = arena.shortRandomMatrix(dim, dim, -100, 100, 6421);

            m0[0,0] = m1[0,0];
            m0[0,1] = (short)(1 - m1[0,1]);
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
