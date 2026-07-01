using System;
using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class floatOperationsTest {

    [BurstCompile]
    public struct BasicVecOpTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            float s = 1f;
            floatN a = arena.floatVec(vecLen, 10f);


            Assert.AreEqual(vecLen, a.N); 

            floatN b = arena.floatVec(vecLen, 10f);

            Assert.IsTrue(b[vecLen/2] == a[vecLen/2]);
            
            Assert.AreEqual(2, arena.AllocationsCount);

            floatN result = default;

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

            float s = 1f;
            floatMxN a = arena.floatMat(rows, cols, 10f);

            floatMxN b = arena.floatMat(rows, cols, 10f);

            floatMxN result = default;

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
                    SignFlipMat();
                    break;

            }
        }

        public void SignFlipVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            floatN a = arena.floatVec(vecLen, 10f);

            a = -a;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == -(float)10f);

            arena.Dispose();
        }

        public void AddVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            floatN a = arena.floatVec(vecLen, 10f);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (float)10d);

            a += 1f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (float)11d);

            floatN r = arena.floatVec(vecLen, 5f);

            a += r;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (float)16);

            arena.Dispose();
        }

        public void SubVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            floatN a = arena.floatVec(vecLen, 10f);

            a -= 1f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (float)9f);

            floatN r = arena.floatVec(vecLen, 5f);

            a -= r;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (float)4d);

            a = arena.floatVec(vecLen, 10f);

            a = 1f - a;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == -(float)9d);

            arena.Dispose();
        }

        public void MulVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            floatN a = arena.floatVec(vecLen, 1f);

            a *= 1f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (float)1d);

            a *= 2f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (float)2d);

            a = arena.floatIndexZeroVec(vecLen);

            a *= 2f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (float)(2d*i));

            a = arena.floatIndexZeroVec(vecLen);
            floatN b = arena.floatIndexZeroVec(vecLen);

            var c = a * b;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(c[i] == (float)(i * i));

            arena.Dispose();
        }

        public void DivVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            floatN a = arena.floatVec(vecLen, 1f);

            a /= 1f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (float)1f);

            a /= 2f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (float)0.5f);

            a = arena.floatIndexZeroVec(vecLen);

            a /= 2f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (float)0.5f * i);

            a = arena.floatIndexZeroVec(vecLen);
            floatN b = arena.floatIndexZeroVec(vecLen);

            // add 1 so no division by zero
            a += 1f;
            b += 1f;

            var c0 = a / b;
            var c1 = b / a;

            for (int i = 0; i < vecLen; i++)
            {
                Assert.IsTrue(c0[i] == (float)1f);
                Assert.IsTrue(c1[i] == (float)1f);
            }

            a = arena.floatVec(vecLen, 2f);

            a = 2f / a;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (float)1f);

            arena.Dispose();
        }

        public void ModVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            floatN a = arena.floatVec(vecLen, 10f);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (float)10f);

            a %= 2f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (float)0f);

            a = arena.floatIndexZeroVec(vecLen);

            a %= 2f;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (float)(i % (float)2d));

            a = arena.floatIndexZeroVec(vecLen);
            floatN b = arena.floatIndexZeroVec(vecLen);

            // add 1 so no division by zero
            a += 1f;
            b += 1f;

            var c0 = a % b;
            var c1 = b % a;

            for (int i = 0; i < vecLen; i++)
            {
                Assert.IsTrue(c0[i] == (float)0f);
                Assert.IsTrue(c1[i] == (float)0f);
            }

            arena.Dispose();
        }

        public void SignFlipMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;
            int totalElements = vecLen * vecLen;
            floatMxN a = arena.floatMat(vecLen, vecLen, 10f);

            a = -a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == -(float)10f);

            arena.Dispose();
        }

        public void AddMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            floatMxN a = arena.floatMat(rows, cols, 10f);

            // Element-wise addition with scalar
            a += 1f;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (float)11f);

            arena.Dispose();
        }

        public void SubMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            floatMxN a = arena.floatMat(rows, cols, 10f);

            // Element-wise subtraction with scalar
            a -= 5f;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (float)5f);

            arena.Dispose();
        }

        public void MulMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            floatMxN a = arena.floatMat(rows, cols, 2f);

            // Element-wise multiplication with scalar
            a *= 3f;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (float)6f);

            a = 3f * a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (float)18f);

            var b = arena.floatMat(rows, cols, 0.5f);

            a = a * b;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (float)9f);

            arena.Dispose();
        }

        public void DivMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            floatMxN a = arena.floatMat(rows, cols, 10f);

            // Element-wise division with scalar
            a /= 2f;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (float)5f);

            a = 5f / a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (float)1f);

            floatMxN b = arena.floatMat(rows, cols, 0.5f);

            var c0 = a / b;
            var c1 = b / a;

            for (int i = 0; i < totalElements; i++)
            {
                Assert.IsTrue(c0[i] == (float)2f);
                Assert.IsTrue(c1[i] == (float)0.5f);
            }

            arena.Dispose();
        }

        public void ModMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            floatMxN a = arena.floatMat(rows, cols, 10f);

            // Element-wise modulo with scalar
            a %= 3f;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (float)1f);

            a = arena.floatMat(rows, cols, 4f);

            a = 4f % a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (float)0f);

            a = arena.floatMat(rows, cols, 3f);
            floatMxN b = arena.floatMat(rows, cols, 2f);

            a = a % b;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (float)1f);

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
