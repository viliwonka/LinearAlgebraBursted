using System;
using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class fProxyOperationsTest {

    [BurstCompile(CompileSynchronously = true)]
    public struct BasicVecOpTestJob : IJob
    {
        public void Execute()
        {

            int vecLen = 16;

            fProxy s = 1f;
            fProxyN a = GenerateOP.fProxyVec(vecLen, 10f, Allocator.Temp);


            Assert.AreEqual(vecLen, a.N);

            fProxyN b = GenerateOP.fProxyVec(vecLen, 10f, Allocator.Temp);

            Assert.IsTrue(b[vecLen/2] == a[vecLen/2]);

            fProxyN result = default;

            result = new fProxyN(in a, Allocator.Temp);
            fProxyComp.addInPlace(result, s);

            result = new fProxyN(in a, Allocator.Temp);
            fProxyComp.addInPlace(result, s);

            result = new fProxyN(in a, Allocator.Temp);
            fProxyComp.addInPlace(result, -s);
            result = new fProxyN(in a, Allocator.Temp);
            fProxyComp.subInPlace(s, result);

            result = new fProxyN(in a, Allocator.Temp);
            fProxyComp.mulInPlace(result, s);
            result = new fProxyN(in a, Allocator.Temp);
            fProxyComp.mulInPlace(result, s);

            result = new fProxyN(in a, Allocator.Temp);
            fProxyComp.divInPlace(result, s);
            result = new fProxyN(in a, Allocator.Temp);
            fProxyComp.modInPlace(result, s);
            result = new fProxyN(in a, Allocator.Temp);
            fProxyComp.divInPlace(s, result);
            result = new fProxyN(in a, Allocator.Temp);
            fProxyComp.modInPlace(s, result);

            result = new fProxyN(in a, Allocator.Temp);
            fProxyComp.addInPlace(result, b);
            result = new fProxyN(in a, Allocator.Temp);
            fProxyComp.subInPlace(result, b);
            result = new fProxyN(in a, Allocator.Temp);
            fProxyComp.mulInPlace(result, b);
            result = new fProxyN(in a, Allocator.Temp);
            fProxyComp.divInPlace(result, b);
            result = new fProxyN(in a, Allocator.Temp);
            fProxyComp.modInPlace(result, b);
        }
    }

    [Test]
    public void BasicVecOperationsSimple()
    {
        new BasicVecOpTestJob().Run();
    }

    [BurstCompile(CompileSynchronously = true)]
    public struct BasicMatOpTestJob : IJob
    {
        public void Execute()
        {

            int rows = 8;
            int cols = 8;

            int elements = rows * cols;

            fProxy s = 1f;
            fProxyMxN a = GenerateOP.fProxyMat(rows, cols, 10f, Allocator.Temp);

            fProxyMxN b = GenerateOP.fProxyMat(rows, cols, 10f, Allocator.Temp);

            fProxyMxN result = default;

            result = new fProxyMxN(in a, Allocator.Temp);
            fProxyComp.addInPlace(result, s);

            result = new fProxyMxN(in a, Allocator.Temp);
            fProxyComp.addInPlace(result, s);

            result = new fProxyMxN(in a, Allocator.Temp);
            fProxyComp.addInPlace(result, -s);
            result = new fProxyMxN(in a, Allocator.Temp);
            fProxyComp.subInPlace(s, result);

            result = new fProxyMxN(in a, Allocator.Temp);
            fProxyComp.mulInPlace(result, s);
            result = new fProxyMxN(in a, Allocator.Temp);
            fProxyComp.mulInPlace(result, s);

            result = new fProxyMxN(in a, Allocator.Temp);
            fProxyComp.divInPlace(result, s);
            result = new fProxyMxN(in a, Allocator.Temp);
            fProxyComp.modInPlace(result, s);
            result = new fProxyMxN(in a, Allocator.Temp);
            fProxyComp.divInPlace(s, result);
            result = new fProxyMxN(in a, Allocator.Temp);
            fProxyComp.modInPlace(s, result);

            result = new fProxyMxN(in a, Allocator.Temp);
            fProxyComp.addInPlace(result, b);
            result = new fProxyMxN(in a, Allocator.Temp);
            fProxyComp.subInPlace(result, b);
            result = new fProxyMxN(in a, Allocator.Temp);
            fProxyComp.mulInPlace(result, b);
            result = new fProxyMxN(in a, Allocator.Temp);
            fProxyComp.divInPlace(result, b);
            result = new fProxyMxN(in a, Allocator.Temp);
            fProxyComp.modInPlace(result, b);
        }
    }

    [Test]
    public void BasicMatOperationsSimple()
    {
        new BasicMatOpTestJob().Run();
    }

    [BurstCompile(CompileSynchronously = true)]
    public struct BasicPreciseOPTestJob : IJob
    {
        public enum TestType
        {
            AddVec,
            SubVec,
            MulVec,
            DivVec,
            ModVec,
            SignFlipVec,

            AddMat,
            SubMat,
            MulMat,
            DivMat,
            ModMat,
            SignFlipMat,
        }

        public TestType Type;

        public void Execute()
        {
            switch(Type)
            {
                case TestType.AddVec:
                    AddVec();
                break;

                case TestType.SubVec:
                    SubVec();
                    break;

                case TestType.MulVec:
                    MulVec();
                    break;

                case TestType.DivVec:
                    DivVec();
                    break;

                case TestType.ModVec:
                    ModVec();
                    break;
                case TestType.SignFlipVec:
                    SignFlipVec();
                    break;
                // Matrix operations
                case TestType.AddMat:
                    AddMat();
                    break;
                case TestType.SubMat:
                    SubMat();
                    break;
                case TestType.MulMat:
                    MulMat();
                    break;
                case TestType.DivMat:
                    DivMat();
                    break;
                case TestType.ModMat:
                    ModMat();
                    break;
                case TestType.SignFlipMat:
                    SignFlipMat();
                    break;

            }
        }

        public void SignFlipVec()
        {

            int vecLen = 16;

            fProxyN a = GenerateOP.fProxyVec(vecLen, 10f, Allocator.Temp);

            fProxyComp.signFlipInPlace(a);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == -(fProxy)10f);
        }

        public void AddVec()
        {

            int vecLen = 16;

            fProxyN a = GenerateOP.fProxyVec(vecLen, 10f, Allocator.Temp);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)10d);

            fProxyComp.addInPlace(a, 1f);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)11d);

            fProxyN r = GenerateOP.fProxyVec(vecLen, 5f, Allocator.Temp);

            fProxyComp.addInPlace(a, r);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)16);
        }

        public void SubVec()
        {

            int vecLen = 16;

            fProxyN a = GenerateOP.fProxyVec(vecLen, 10f, Allocator.Temp);

            fProxyComp.addInPlace(a, -1f);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)9f);

            fProxyN r = GenerateOP.fProxyVec(vecLen, 5f, Allocator.Temp);

            fProxyComp.subInPlace(a, r);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)4d);

            a = GenerateOP.fProxyVec(vecLen, 10f, Allocator.Temp);

            fProxyComp.subInPlace(1f, a);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == -(fProxy)9d);
        }

        public void MulVec()
        {

            int vecLen = 16;

            fProxyN a = GenerateOP.fProxyVec(vecLen, 1f, Allocator.Temp);

            fProxyComp.mulInPlace(a, 1f);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)1d);

            fProxyComp.mulInPlace(a, 2f);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)2d);

            a = GenerateOP.fProxyIndexZeroVec(vecLen);

            fProxyComp.mulInPlace(a, 2f);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)(2d*i));

            a = GenerateOP.fProxyIndexZeroVec(vecLen);
            fProxyN b = GenerateOP.fProxyIndexZeroVec(vecLen);

            var c = new fProxyN(in a, Allocator.Temp);
            fProxyComp.mulInPlace(c, b);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(c[i] == (fProxy)(i * i));
        }

        public void DivVec()
        {

            int vecLen = 16;

            fProxyN a = GenerateOP.fProxyVec(vecLen, 1f, Allocator.Temp);

            fProxyComp.divInPlace(a, 1f);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)1f);

            fProxyComp.divInPlace(a, 2f);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)0.5f);

            a = GenerateOP.fProxyIndexZeroVec(vecLen);

            fProxyComp.divInPlace(a, 2f);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)0.5f * i);

            a = GenerateOP.fProxyIndexZeroVec(vecLen);
            fProxyN b = GenerateOP.fProxyIndexZeroVec(vecLen);

            // add 1 so no division by zero
            fProxyComp.addInPlace(a, 1f);
            fProxyComp.addInPlace(b, 1f);

            var c0 = new fProxyN(in a, Allocator.Temp);
            fProxyComp.divInPlace(c0, b);
            var c1 = new fProxyN(in b, Allocator.Temp);
            fProxyComp.divInPlace(c1, a);

            for (int i = 0; i < vecLen; i++)
            {
                Assert.IsTrue(c0[i] == (fProxy)1f);
                Assert.IsTrue(c1[i] == (fProxy)1f);
            }

            a = GenerateOP.fProxyVec(vecLen, 2f, Allocator.Temp);

            fProxyComp.divInPlace(2f, a);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)1f);
        }

        public void ModVec()
        {

            int vecLen = 16;

            fProxyN a = GenerateOP.fProxyVec(vecLen, 10f, Allocator.Temp);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)10f);

            fProxyComp.modInPlace(a, 2f);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)0f);

            a = GenerateOP.fProxyIndexZeroVec(vecLen);

            fProxyComp.modInPlace(a, 2f);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)(i % (fProxy)2d));

            a = GenerateOP.fProxyIndexZeroVec(vecLen);
            fProxyN b = GenerateOP.fProxyIndexZeroVec(vecLen);

            // add 1 so no division by zero
            fProxyComp.addInPlace(a, 1f);
            fProxyComp.addInPlace(b, 1f);

            var c0 = new fProxyN(in a, Allocator.Temp);
            fProxyComp.modInPlace(c0, b);
            var c1 = new fProxyN(in b, Allocator.Temp);
            fProxyComp.modInPlace(c1, a);

            for (int i = 0; i < vecLen; i++)
            {
                Assert.IsTrue(c0[i] == (fProxy)0f);
                Assert.IsTrue(c1[i] == (fProxy)0f);
            }
        }

        public void SignFlipMat()
        {

            int vecLen = 16;
            int totalElements = vecLen * vecLen;
            fProxyMxN a = GenerateOP.fProxyMat(vecLen, vecLen, 10f, Allocator.Temp);

            fProxyComp.signFlipInPlace(a);

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == -(fProxy)10f);
        }

        public void AddMat()
        {

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            fProxyMxN a = GenerateOP.fProxyMat(rows, cols, 10f, Allocator.Temp);

            // Element-wise addition with scalar
            fProxyComp.addInPlace(a, 1f);

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)11f);
        }

        public void SubMat()
        {

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            fProxyMxN a = GenerateOP.fProxyMat(rows, cols, 10f, Allocator.Temp);

            // Element-wise subtraction with scalar
            fProxyComp.addInPlace(a, -5f);

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)5f);
        }

        public void MulMat()
        {

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            fProxyMxN a = GenerateOP.fProxyMat(rows, cols, 2f, Allocator.Temp);

            // Element-wise multiplication with scalar
            fProxyComp.mulInPlace(a, 3f);

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)6f);

            fProxyComp.mulInPlace(a, 3f);

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)18f);

            var b = GenerateOP.fProxyMat(rows, cols, 0.5f, Allocator.Temp);

            fProxyComp.mulInPlace(a, b);

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)9f);
        }

        public void DivMat()
        {

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            fProxyMxN a = GenerateOP.fProxyMat(rows, cols, 10f, Allocator.Temp);

            // Element-wise division with scalar
            fProxyComp.divInPlace(a, 2f);

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)5f);

            fProxyComp.divInPlace(5f, a);

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)1f);

            fProxyMxN b = GenerateOP.fProxyMat(rows, cols, 0.5f, Allocator.Temp);

            var c0 = new fProxyMxN(in a, Allocator.Temp);
            fProxyComp.divInPlace(c0, b);
            var c1 = new fProxyMxN(in b, Allocator.Temp);
            fProxyComp.divInPlace(c1, a);

            for (int i = 0; i < totalElements; i++)
            {
                Assert.IsTrue(c0[i] == (fProxy)2f);
                Assert.IsTrue(c1[i] == (fProxy)0.5f);
            }
        }

        public void ModMat()
        {

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            fProxyMxN a = GenerateOP.fProxyMat(rows, cols, 10f, Allocator.Temp);

            // Element-wise modulo with scalar
            fProxyComp.modInPlace(a, 3f);

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)1f);

            a = GenerateOP.fProxyMat(rows, cols, 4f, Allocator.Temp);

            fProxyComp.modInPlace(4f, a);

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)0f);

            a = GenerateOP.fProxyMat(rows, cols, 3f, Allocator.Temp);
            fProxyMxN b = GenerateOP.fProxyMat(rows, cols, 2f, Allocator.Temp);

            fProxyComp.modInPlace(a, b);

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)1f);
        }
    }

    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(BasicPreciseOPTestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void TestCases(BasicPreciseOPTestJob.TestType type)
    {
        new BasicPreciseOPTestJob() { Type = type }.Run();
    }

}
