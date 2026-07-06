using System;
using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class intOperationsTest {

    [BurstCompile(CompileSynchronously = true)]
    public struct BasicVecOpTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            int s = 1;
            intN a = arena.intVec(vecLen, 10);


            Assert.AreEqual(vecLen, a.N); 

            intN b = arena.intVec(vecLen, 10);

            Assert.IsTrue(b[vecLen/2] == a[vecLen/2]);
            
            Assert.AreEqual(2, arena.AllocationsCount);

            intN result = default;

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

            arena.Dispose();
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
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;

            int elements = rows * cols;

            int s = 1;
            intMxN a = arena.intMat(rows, cols, 10);

            intMxN b = arena.intMat(rows, cols, 10);

            intMxN result = default;

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

            intN a = arena.intVec(vecLen, 10);

            a = -a;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == -(int)10f);

            arena.Dispose();
        }

        public void AddVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            intN a = arena.intVec(vecLen, 10);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (int)10d);

            a += 1;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (int)11d);

            intN r = arena.intVec(vecLen, 5);

            a += r;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (int)16);

            arena.Dispose();
        }

        public void SubVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            intN a = arena.intVec(vecLen, 10);

            a -= 1;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (int)9f);

            intN r = arena.intVec(vecLen, 5);

            a -= r;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (int)4d);

            a = arena.intVec(vecLen, 10);

            a = 1 - a;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == -(int)9d);

            arena.Dispose();
        }

        public void MulVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            intN a = arena.intVec(vecLen, 1);

            a *= 1;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (int)1d);

            a *= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (int)2d);

            a = arena.intIndexZeroVec(vecLen);

            a *= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (int)(2d*i));

            a = arena.intIndexZeroVec(vecLen);
            intN b = arena.intIndexZeroVec(vecLen);

            var c = a * b;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(c[i] == (int)(i * i));

            arena.Dispose();
        }

        public void DivVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            intN a = arena.intVec(vecLen, 2);

            a /= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (int)1);

            a /= 1;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (int)1);

            a = arena.intIndexZeroVec(vecLen);

            a /= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (int)(0.5 * i));

            a = arena.intIndexZeroVec(vecLen);
            intN b = arena.intIndexZeroVec(vecLen);

            // add 1 so no division by zero
            a += 1;
            b += 1;

            var c0 = a / b;
            var c1 = b / a;

            for (int i = 0; i < vecLen; i++)
            {
                Assert.IsTrue(c0[i] == (int)1);
                Assert.IsTrue(c1[i] == (int)1);
            }

            a = arena.intVec(vecLen, 2);

            a = 2 / a;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (int)1);

            arena.Dispose();
        }

        public void ModVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            intN a = arena.intVec(vecLen, 10);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (int)10);

            a %= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (int)0);

            a = arena.intIndexZeroVec(vecLen);

            a %= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (int)(i % (int)2));

            a = arena.intIndexZeroVec(vecLen);
            intN b = arena.intIndexZeroVec(vecLen);

            a += 1;
            b += 1;

            var c0 = a % b;
            var c1 = b % a;

            for (int i = 0; i < vecLen; i++)
            {
                Assert.IsTrue(c0[i] == (int)0);
                Assert.IsTrue(c1[i] == (int)0);
            }

            arena.Dispose();
        }

        public void SignFlipMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;
            int totalElements = vecLen * vecLen;
            intMxN a = arena.intMat(vecLen, vecLen, 10);

            a = -a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == -(int)10f);

            arena.Dispose();
        }

        public void AddMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            intMxN a = arena.intMat(rows, cols, 10);

            a += 1;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (int)11f);

            arena.Dispose();
        }

        public void SubMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            intMxN a = arena.intMat(rows, cols, 10);

            a -= 5;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (int)5f);

            arena.Dispose();
        }

        public void MulMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            intMxN a = arena.intMat(rows, cols, 2);

            a *= 3;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (int)6f);

            a = 3 * a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (int)18f);

            arena.Dispose();
        }

        public void DivMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            intMxN a = arena.intMat(rows, cols, 10);

            a /= 2;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (int)5);

            a = 5 / a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (int)1);

            arena.Dispose();
        }

        public void ModMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            intMxN a = arena.intMat(rows, cols, 10);

            a %= 3;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (int)1f);

            a = arena.intMat(rows, cols, 4);

            a = 4 % a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (int)0f);

            a = arena.intMat(rows, cols, 3);
            intMxN b = arena.intMat(rows, cols, 2);

            a = a % b;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (int)1f);

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
