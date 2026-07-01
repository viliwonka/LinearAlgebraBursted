using System;
using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class longOperationsTest {

    [BurstCompile]
    public struct BasicVecOpTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            long s = 1;
            longN a = arena.longVec(vecLen, 10);


            Assert.AreEqual(vecLen, a.N); 

            longN b = arena.longVec(vecLen, 10);

            Assert.IsTrue(b[vecLen/2] == a[vecLen/2]);
            
            Assert.AreEqual(2, arena.AllocationsCount);

            longN result = default;

            result = a + s;

            result = s + a;

            result = a - s;
            result = s - a;

            Assert.AreEqual(4, arena.TempAllocationsCount);

            result = ~a;

            arena.ClearTemp();

            result = a * s;
            result = s * a;

            result = a / s;
            result = a % s;
            result = s / a;
            result = s % a;

            result = a | s;
            result = s | a;

            result = a & s;
            result = s & a;

            result = a ^ s;
            result = s ^ a;

            result = result << 5;
            result = result >> 5;

            result = a + b;
            result = a - b;
            result = a * b;
            result = a / b;
            result = a % b;

            result = a | b;
            result = a & b;
            result = a ^ b;

            //Assert.AreEqual(11, arena.TempAllocationsCount);

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

            long s = 1;
            longMxN a = arena.longMat(rows, cols, 10);

            longMxN b = arena.longMat(rows, cols, 10);

            longMxN result = default;

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

            longN a = arena.longVec(vecLen, 10);

            a = -a;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == -(long)10f);

            arena.Dispose();
        }

        public void AddVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            longN a = arena.longVec(vecLen, 10);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (long)10d);

            a += 1;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (long)11d);

            longN r = arena.longVec(vecLen, 5);

            a += r;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (long)16);

            arena.Dispose();
        }

        public void SubVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            longN a = arena.longVec(vecLen, 10);

            a -= 1;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (long)9f);

            longN r = arena.longVec(vecLen, 5);

            a -= r;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (long)4d);

            a = arena.longVec(vecLen, 10);

            a = 1 - a;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == -(long)9d);

            arena.Dispose();
        }

        public void MulVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            longN a = arena.longVec(vecLen, 1);

            a *= 1;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (long)1d);

            a *= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (long)2d);

            a = arena.longIndexZeroVec(vecLen);

            a *= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (long)(2d*i));

            a = arena.longIndexZeroVec(vecLen);
            longN b = arena.longIndexZeroVec(vecLen);

            var c = a * b;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(c[i] == (long)(i * i));

            arena.Dispose();
        }

        public void DivVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            longN a = arena.longVec(vecLen, 2);

            a /= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (long)1);

            a /= 1;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (long)1);

            a = arena.longIndexZeroVec(vecLen);

            a /= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (long)(0.5 * i));

            a = arena.longIndexZeroVec(vecLen);
            longN b = arena.longIndexZeroVec(vecLen);

            // add 1 so no division by zero
            a += 1;
            b += 1;

            var c0 = a / b;
            var c1 = b / a;

            for (int i = 0; i < vecLen; i++)
            {
                Assert.IsTrue(c0[i] == (long)1);
                Assert.IsTrue(c1[i] == (long)1);
            }

            a = arena.longVec(vecLen, 2);

            a = 2 / a;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (long)1);

            arena.Dispose();
        }

        public void ModVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            longN a = arena.longVec(vecLen, 10);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (long)10);

            a %= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (long)0);

            a = arena.longIndexZeroVec(vecLen);

            a %= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (long)(i % (long)2));

            a = arena.longIndexZeroVec(vecLen);
            longN b = arena.longIndexZeroVec(vecLen);

            // add 1 so no division by zero
            a += 1;
            b += 1;

            var c0 = a % b;
            var c1 = b % a;

            for (int i = 0; i < vecLen; i++)
            {
                Assert.IsTrue(c0[i] == (long)0);
                Assert.IsTrue(c1[i] == (long)0);
            }

            arena.Dispose();
        }

        public void SignFlipMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;
            int totalElements = vecLen * vecLen;
            longMxN a = arena.longMat(vecLen, vecLen, 10);

            a = -a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == -(long)10f);

            arena.Dispose();
        }

        public void AddMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            longMxN a = arena.longMat(rows, cols, 10);

            // Element-wise addition with scalar
            a += 1;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (long)11f);

            arena.Dispose();
        }

        public void SubMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            longMxN a = arena.longMat(rows, cols, 10);

            // Element-wise subtraction with scalar
            a -= 5;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (long)5f);

            arena.Dispose();
        }

        public void MulMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            longMxN a = arena.longMat(rows, cols, 2);

            // Element-wise multiplication with scalar
            a *= 3;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (long)6f);

            a = 3 * a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (long)18f);

            arena.Dispose();
        }

        public void DivMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            longMxN a = arena.longMat(rows, cols, 10);

            // Element-wise division with scalar
            a /= 2;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (long)5);

            a = 5 / a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (long)1);

            arena.Dispose();
        }

        public void ModMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            longMxN a = arena.longMat(rows, cols, 10);

            // Element-wise modulo with scalar
            a %= 3;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (long)1f);

            a = arena.longMat(rows, cols, 4);

            a = 4 % a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (long)0f);

            a = arena.longMat(rows, cols, 3);
            longMxN b = arena.longMat(rows, cols, 2);

            a = a % b;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (long)1f);

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
