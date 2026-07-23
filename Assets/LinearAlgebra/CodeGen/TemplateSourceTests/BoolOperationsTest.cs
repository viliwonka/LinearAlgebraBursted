using System;
using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class BoolOperationsTest
{

    [BurstCompile(CompileSynchronously = true)]
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
            AndVecVec,
            OrVecVec,
            XorVecVec,
        }

        public OPType Type;

        public void Execute()
        {
            switch (Type)
            {
                case OPType.NotVec:
                    NotVec();
                    break;
                case OPType.NotMat:
                    NotMat();
                    break;
                case OPType.EqualsVec:
                    EqualsVec();
                    break;
                case OPType.NotEqualsVec:
                    NotEqualsVec();
                    break;
                case OPType.AndVec:
                    AndVec();
                    break;
                case OPType.OrVec:
                    OrVec();
                    break;
                case OPType.XorVec:
                    XorVec();
                    break;
                case OPType.EqualsMat:
                    EqualsMat();
                    break;
                case OPType.NotEqualsMat:
                    NotEqualsMat();
                    break;
                case OPType.AndMat:
                    AndMat();
                    break;
                case OPType.OrMat:
                    OrMat();
                    break;
                case OPType.XorMat:
                    XorMat();
                    break;
                case OPType.EqualsVecVec:
                    EqualsVecVec();
                break;
                case OPType.NotEqualsVecVec:
                    NotEqualsVecVec();
                break;
                case OPType.AndVecVec:
                    AndVecVec();
                break;
                case OPType.OrVecVec:
                    OrVecVec();
                break;
                case OPType.XorVecVec:
                    XorVecVec();
                break;
                default:
                    throw new NotImplementedException();
            }
        }

        public void NotVec()
        {
            int vecLen = 16;

            boolN a = GenerateOP.boolRandomVec(vecLen);

            boolN b = !a;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] != b[i]);
        }

        public void NotMat()
        {
            int rows = 16;
            int cols = 16;

            boolMxN a = GenerateOP.boolRandomMat(rows, cols);

            boolMxN b = !a;

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    Assert.IsTrue(a[i, j] != b[i, j]);
        }

        public void EqualsVec()
        {
            int vecLen = 16;

            boolN a = GenerateOP.boolRandomVec(vecLen);
            
            boolN b = a == true;
            
            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == a[i]);
        }

        public void NotEqualsVec()
        {
            int vecLen = 16;

            boolN a = GenerateOP.boolRandomVec(vecLen);
            
            boolN b = a != true;
            
            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == !a[i]);
        }

        public void AndVec()
        {
            int vecLen = 16;

            boolN a = GenerateOP.boolRandomVec(vecLen);
            
            boolN b = a & true;
            
            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == a[i]);

            b = a & false;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == false);
        }

        public void OrVec()
        {
            int vecLen = 16;

            boolN a = GenerateOP.boolRandomVec(vecLen);
            
            boolN b = a | true;
            
            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == true);

            b = a | false;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == a[i]);
        }

        public void XorVec()
        {
            int vecLen = 16;

            boolN a = GenerateOP.boolRandomVec(vecLen);
            
            boolN b = a ^ true;
            
            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == !a[i]);

            b = a ^ false;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(b[i] == a[i]);
        }

        public void EqualsMat()
        {
            int rows = 16;
            int cols = 16;

            boolMxN a = GenerateOP.boolRandomMat(rows, cols);
            
            boolMxN b = a == true;
            
            for (int i = 0; i < rows; i++)
                for(int j = 0; j < cols; j++)
                    Assert.IsTrue(b[i, j] == a[i, j]);
        }

        public void NotEqualsMat()
        {
            int rows = 16;
            int cols = 16;

            boolMxN a = GenerateOP.boolRandomMat(rows, cols);
            
            boolMxN b = a != true;
            
            for (int i = 0; i < rows; i++)
                for(int j = 0; j < cols; j++)
                    Assert.IsTrue(b[i, j] == !a[i, j]);
        }

        public void AndMat()
        {
            int rows = 16;
            int cols = 16;

            boolMxN a = GenerateOP.boolRandomMat(rows, cols);
            
            boolMxN b = a & true;
            
            for (int i = 0; i < rows; i++)
                for(int j = 0; j < cols; j++)
                    Assert.IsTrue(b[i, j] == a[i, j]);

            b = a & false;

            for (int i = 0; i < rows; i++)
                for(int j = 0; j < cols; j++)
                    Assert.IsTrue(b[i, j] == false);
        }

        public void OrMat()
        {
            int rows = 16;
            int cols = 16;

            boolMxN a = GenerateOP.boolRandomMat(rows, cols);
            
            boolMxN b = a | true;
            
            for (int i = 0; i < rows; i++)
                for(int j = 0; j < cols; j++)
                    Assert.IsTrue(b[i, j] == true);

            b = a | false;

            for (int i = 0; i < rows; i++)
                for(int j = 0; j < cols; j++)
                    Assert.IsTrue(b[i, j] == a[i, j]);
        }

        public void XorMat()
        {
            int rows = 16;
            int cols = 16;

            boolMxN a = GenerateOP.boolRandomMat(rows, cols);
            
            boolMxN b = a ^ true;
            
            for (int i = 0; i < rows; i++)
                for(int j = 0; j < cols; j++)
                    Assert.IsTrue(b[i, j] == !a[i, j]);

            b = a ^ false;

            for (int i = 0; i < rows; i++)
                for(int j = 0; j < cols; j++)
                    Assert.IsTrue(b[i, j] == a[i, j]);
        }

        public void EqualsVecVec()
        {
            int vecLen = 16;

            boolN a = GenerateOP.boolRandomVec(vecLen);
            boolN b = GenerateOP.boolRandomVec(vecLen);

            boolN c = a == b;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue((a[i] == b[i]) == c[i]);
        }

        public void NotEqualsVecVec()
        {
            int vecLen = 16;

            boolN a = GenerateOP.boolRandomVec(vecLen);
            boolN b = GenerateOP.boolRandomVec(vecLen);
            boolN c = a != b;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue((a[i] != b[i]) == c[i]);
        }

        public void AndVecVec()
        {
            int vecLen = 16;

            boolN a = GenerateOP.boolRandomVec(vecLen);
            boolN b = GenerateOP.boolRandomVec(vecLen);
            boolN c = a & b;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue((a[i] & b[i]) == c[i]);
        }

        public void OrVecVec()
        {
            int vecLen = 16;

            boolN a = GenerateOP.boolRandomVec(vecLen);
            boolN b = GenerateOP.boolRandomVec(vecLen);
            boolN c = a | b;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue((a[i] | b[i]) == c[i]);
        }

        public void XorVecVec()
        {
            int vecLen = 16;

            boolN a = GenerateOP.boolRandomVec(vecLen);
            boolN b = GenerateOP.boolRandomVec(vecLen);
            boolN c = a ^ b;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue((a[i] ^ b[i]) == c[i]);
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
