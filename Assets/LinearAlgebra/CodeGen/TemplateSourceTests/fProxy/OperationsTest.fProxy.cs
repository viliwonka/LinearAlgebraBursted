using System;
using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class fProxyOperationsTest {

    [BurstCompile]
    public struct BasicVecOpTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            fProxy s = 1f;
            fProxyN a = arena.fProxyVec(vecLen, 10f);


            Assert.AreEqual(vecLen, a.N); 

            fProxyN b = arena.fProxyVec(vecLen, 10f);

            Assert.IsTrue(b[vecLen/2] == a[vecLen/2]);
            
            Assert.AreEqual(2, arena.AllocationsCount);

            fProxyN result = default;

            result = a + s;

            result = s + a;

            result = a - s;
            result = s - a;

            Assert.AreEqual(4, arena.TempAllocationsCount);

            arena.ClearTemp();

            result = a * s;
            result = s * a;

            result = a / s;
            result = a % s;
            result = s / a;
            result = s % a;

            result = a + b;
            result = a - b;
            result = a * b;
            result = a / b;
            result = a % b;

            Assert.AreEqual(11, arena.TempAllocationsCount);

            arena.Dispose();
        }
    }

    [Test]
    public void BasicVecOperationsSimple()
    {
        new BasicVecOpTestJob().Run();
    }

    [BurstCompile]
    public struct BasicMatOpTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;

            int elements = rows * cols;

            fProxy s = 1f;
            fProxyMxN a = arena.fProxyMat(rows, cols, 10f);

            fProxyMxN b = arena.fProxyMat(rows, cols, 10f);

            fProxyMxN result = default;

            result = a + s;

            result = s + a;

            result = a - s;
            result = s - a;

            result = a * s;
            result = s * a;

            result = a / s;
            result = a % s;
            result = s / a;
            result = s % a;

            result = a + b;
            result = a - b;
            result = a * b;
            result = a / b;
            result = a % b;

            arena.Dispose();
        }
    }

    [Test]
    public void BasicMatOperationsSimple()
    {
        new BasicMatOpTestJob().Run();
    }
    
    [BurstCompile]
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
                    SignFlipVec();
                    break;

            }
        }

        public void SignFlipVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            fProxyN a = arena.fProxyVec(vecLen, 10f);

            a = -a;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == -(fProxy)10f);

            arena.Dispose();
        }

        public void AddVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            fProxyN a = arena.fProxyVec(vecLen, 10f);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)10d);

            a += 1f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)11d);

            fProxyN r = arena.fProxyVec(vecLen, 5f);

            a += r;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)16);

            arena.Dispose();
        }

        public void SubVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            fProxyN a = arena.fProxyVec(vecLen, 10f);

            a -= 1f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)9f);

            fProxyN r = arena.fProxyVec(vecLen, 5f);

            a -= r;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)4d);

            a = arena.fProxyVec(vecLen, 10f);

            a = 1f - a;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == -(fProxy)9d);

            arena.Dispose();
        }

        public void MulVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            fProxyN a = arena.fProxyVec(vecLen, 1f);

            a *= 1f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)1d);

            a *= 2f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)2d);

            a = arena.fProxyIndexZeroVector(vecLen);

            a *= 2f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)(2d*i));

            a = arena.fProxyIndexZeroVector(vecLen);
            fProxyN b = arena.fProxyIndexZeroVector(vecLen);

            var c = a * b;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(c[i] == (fProxy)(i * i));

            arena.Dispose();
        }

        public void DivVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            fProxyN a = arena.fProxyVec(vecLen, 1f);

            a /= 1f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)1f);

            a /= 2f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)0.5f);

            a = arena.fProxyIndexZeroVector(vecLen);

            a /= 2f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)0.5f * i);

            a = arena.fProxyIndexZeroVector(vecLen);
            fProxyN b = arena.fProxyIndexZeroVector(vecLen);

            // add 1 so no division by zero
            a += 1f;
            b += 1f;

            var c0 = a / b;
            var c1 = b / a;

            for (int i = 0; i < vecLen; i++)
            {
                Assert.IsTrue(c0[i] == (fProxy)1f);
                Assert.IsTrue(c1[i] == (fProxy)1f);
            }

            a = arena.fProxyVec(vecLen, 2f);

            a = 2f / a;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)1f);

            arena.Dispose();
        }

        public void ModVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            fProxyN a = arena.fProxyVec(vecLen, 10f);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)10f);

            a %= 2f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)0f);

            a = arena.fProxyIndexZeroVector(vecLen);

            a %= 2f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (fProxy)(i % (fProxy)2d));

            a = arena.fProxyIndexZeroVector(vecLen);
            fProxyN b = arena.fProxyIndexZeroVector(vecLen);

            // add 1 so no division by zero
            a += 1f;
            b += 1f;

            var c0 = a % b;
            var c1 = b % a;

            for (int i = 0; i < vecLen; i++)
            {
                Assert.IsTrue(c0[i] == (fProxy)0f);
                Assert.IsTrue(c1[i] == (fProxy)0f);
            }

            arena.Dispose();
        }

        public void SignFlipMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;
            int totalElements = vecLen * vecLen;
            fProxyMxN a = arena.fProxyMat(vecLen, vecLen, 10f);

            a = -a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == -(fProxy)10f);

            arena.Dispose();
        }

        public void AddMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            fProxyMxN a = arena.fProxyMat(rows, cols, 10f);

            // Element-wise addition with scalar
            a += 1f;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)11f);

            arena.Dispose();
        }

        public void SubMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            fProxyMxN a = arena.fProxyMat(rows, cols, 10f);

            // Element-wise subtraction with scalar
            a -= 5f;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)5f);

            arena.Dispose();
        }

        public void MulMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            fProxyMxN a = arena.fProxyMat(rows, cols, 2f);

            // Element-wise multiplication with scalar
            a *= 3f;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)6f);

            a = 3f * a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)18f);

            var b = arena.fProxyMat(rows, cols, 0.5f);

            a = a * b;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)9f);

            arena.Dispose();
        }

        public void DivMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            fProxyMxN a = arena.fProxyMat(rows, cols, 10f);

            // Element-wise division with scalar
            a /= 2f;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)5f);

            a = 5f / a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)1f);

            fProxyMxN b = arena.fProxyMat(rows, cols, 0.5f);

            var c0 = a / b;
            var c1 = b / a;

            for (int i = 0; i < totalElements; i++)
            {
                Assert.IsTrue(c0[i] == (fProxy)2f);
                Assert.IsTrue(c1[i] == (fProxy)0.5f);
            }

            arena.Dispose();
        }

        public void ModMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            fProxyMxN a = arena.fProxyMat(rows, cols, 10f);

            // Element-wise modulo with scalar
            a %= 3f;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)1f);

            a = arena.fProxyMat(rows, cols, 4f);

            a = 4f % a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)0f);

            a = arena.fProxyMat(rows, cols, 3f);
            fProxyMxN b = arena.fProxyMat(rows, cols, 2f);

            a = a % b;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (fProxy)1f);

            arena.Dispose();
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
