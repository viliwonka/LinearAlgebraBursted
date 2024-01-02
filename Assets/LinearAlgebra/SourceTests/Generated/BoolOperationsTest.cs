using System;
using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class BoolOperationsTest
{

    [BurstCompile]
    public struct TestsJob : IJob
    {
        public enum OPType
        {
            NotVec,
            NotMat,
            
            EqualsVec,
            NotEqualsVec,
            AndVec,
            OrVec,
            XorVec,

            EqualsMat,
            NotEqualsMat,
            AndMat,
            OrMat,
            XorMat,

            EqualsVecVec,
            NotEqualsVecVec,   
        }

        public OPType Type;

        public void Execute()
        {
            Arena arena = new Arena(Allocator.Persistent);

            try {

                switch (Type)
                {
                    case OPType.NotVec:
                        NotVec(ref arena);
                        break;
                    case OPType.NotMat:
                        NotMat(ref arena);
                        break;
                    case OPType.EqualsVec:
                        EqualsVec(ref arena);
                        break;
                    case OPType.NotEqualsVec:
                        NotEqualsVec(ref arena);
                        break;
                    case OPType.AndVec:
                        AndVec(ref arena);
                        break;
                    case OPType.OrVec:
                        OrVec(ref arena);
                        break;
                    case OPType.XorVec:
                        XorVec(ref arena);
                        break;
                    case OPType.EqualsMat:
                        EqualsMat(ref arena);
                        break;
                    case OPType.NotEqualsMat:       
                        NotEqualsMat(ref arena);
                        break;
                    case OPType.AndMat:
                        AndMat(ref arena);
                        break;
                    case OPType.OrMat:
                        OrMat(ref arena);
                        break;
                    case OPType.XorMat:
                        XorMat(ref arena);
                        break;
                    case OPType.EqualsVecVec:
                        EqualsVecVec(ref arena);
                    break;
                    case OPType.NotEqualsVecVec:
                        NotEqualsVecVec(ref arena);
                    break;
                    default:
                        throw new NotImplementedException();
                }
            }
            finally {
                arena.Dispose();
            }
        }

        public void NotVec(ref Arena arena)
        {
            int vecLen = 16;

            boolN a = arena.boolRandomVec(vecLen);

            boolN b = !a;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] != b[i]);
        }

        public void NotMat(ref Arena arena)
        {
            int rows = 16;
            int cols = 16;

            boolMxN a = arena.boolRandomMat(rows, cols);

            boolMxN b = !a;

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    Assert.IsTrue(a[i, j] != b[i, j]);
        }

        public void EqualsVec(ref Arena arena)
        {
            int vecLen = 16;

            boolN a = arena.boolRandomVec(vecLen);
            
            boolN b = a == true;
            
            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == a[i]);
        }

        public void NotEqualsVec(ref Arena arena)
        {
            int vecLen = 16;

            boolN a = arena.boolRandomVec(vecLen);
            
            boolN b = a != true;
            
            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == !a[i]);
        }

        public void AndVec(ref Arena arena)
        {
            int vecLen = 16;

            boolN a = arena.boolRandomVec(vecLen);
            
            boolN b = a & true;
            
            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == a[i]);

            b = a & false;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == false);
        }

        public void OrVec(ref Arena arena)
        {
            int vecLen = 16;

            boolN a = arena.boolRandomVec(vecLen);
            
            boolN b = a | true;
            
            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == true);

            b = a | false;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == a[i]);
        }

        public void XorVec(ref Arena arena)
        {
            int vecLen = 16;

            boolN a = arena.boolRandomVec(vecLen);
            
            boolN b = a ^ true;
            
            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == !a[i]);

            b = a ^ false;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == a[i]);
        }

        public void EqualsMat(ref Arena arena)
        {
            int rows = 16;
            int cols = 16;

            boolMxN a = arena.boolRandomMat(rows, cols);
            
            boolMxN b = a == true;
            
            for (int i = 0; i < rows; i++)
                for(int j = 0; j < cols; j++)
                    Assert.IsTrue(b[i, j] == a[i, j]);
        }

        public void NotEqualsMat(ref Arena arena)
        {
            int rows = 16;
            int cols = 16;

            boolMxN a = arena.boolRandomMat(rows, cols);
            
            boolMxN b = a != true;
            
            for (int i = 0; i < rows; i++)
                for(int j = 0; j < cols; j++)
                    Assert.IsTrue(b[i, j] == !a[i, j]);
        }

        public void AndMat(ref Arena arena)
        {
            int rows = 16;
            int cols = 16;

            boolMxN a = arena.boolRandomMat(rows, cols);
            
            boolMxN b = a & true;
            
            for (int i = 0; i < rows; i++)
                for(int j = 0; j < cols; j++)
                    Assert.IsTrue(b[i, j] == a[i, j]);

            b = a & false;

            for (int i = 0; i < rows; i++)
                for(int j = 0; j < cols; j++)
                    Assert.IsTrue(b[i, j] == false);
        }

        public void OrMat(ref Arena arena)
        {
            int rows = 16;
            int cols = 16;

            boolMxN a = arena.boolRandomMat(rows, cols);
            
            boolMxN b = a | true;
            
            for (int i = 0; i < rows; i++)
                for(int j = 0; j < cols; j++)
                    Assert.IsTrue(b[i, j] == true);

            b = a | false;

            for (int i = 0; i < rows; i++)
                for(int j = 0; j < cols; j++)
                    Assert.IsTrue(b[i, j] == a[i, j]);
        }

        public void XorMat(ref Arena arena)
        {
            int rows = 16;
            int cols = 16;

            boolMxN a = arena.boolRandomMat(rows, cols);
            
            boolMxN b = a ^ true;
            
            for (int i = 0; i < rows; i++)
                for(int j = 0; j < cols; j++)
                    Assert.IsTrue(b[i, j] == !a[i, j]);

            b = a ^ false;

            for (int i = 0; i < rows; i++)
                for(int j = 0; j < cols; j++)
                    Assert.IsTrue(b[i, j] == a[i, j]);
        }

        public void EqualsVecVec(ref Arena arena)
        {
            int vecLen = 16;

            boolN a = arena.boolRandomVec(vecLen);
            boolN b = arena.boolRandomVec(vecLen);

            boolN c = a == b;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == b[i] == c[i]);
        }

        public void NotEqualsVecVec(ref Arena arena)
        {
            int vecLen = 16;

            boolN a = arena.boolRandomVec(vecLen);
            boolN b = arena.boolRandomVec(vecLen);
            boolN c = a != b;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == b[i] != c[i]);
        }

        public void AndVecVec(ref Arena arena)
        {
            int vecLen = 16;

            boolN a = arena.boolRandomVec(vecLen);
            boolN b = arena.boolRandomVec(vecLen);
            boolN c = a & b;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] && b[i] == c[i]);
        }

        public void OrVecVec(ref Arena arena)
        {
            int vecLen = 16;

            boolN a = arena.boolRandomVec(vecLen);
            boolN b = arena.boolRandomVec(vecLen);
            boolN c = a | b;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] || b[i] == c[i]);
        }
    }

    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(TestsJob.OPType));
    }

    [TestCaseSource("GetEnums")]
    public void TestCases(TestsJob.OPType type)
    {
        new TestsJob() { Type = type }.Run();
    }

}
