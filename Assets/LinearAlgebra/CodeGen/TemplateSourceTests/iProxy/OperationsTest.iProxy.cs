using System;
using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class iProxyOperationsTest {

    [BurstCompile(CompileSynchronously = true)]
    public struct BasicVecOpTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            iProxy s = 1;
            iProxyN a = arena.iProxyVec(vecLen, 10);


            Assert.AreEqual(vecLen, a.N); 

            iProxyN b = arena.iProxyVec(vecLen, 10);

            Assert.IsTrue(b[vecLen/2] == a[vecLen/2]);
            
            Assert.AreEqual(2, arena.AllocationsCount);

            iProxyN result = default;

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

            iProxy s = 1;
            iProxyMxN a = arena.iProxyMat(rows, cols, 10);

            iProxyMxN b = arena.iProxyMat(rows, cols, 10);

            iProxyMxN result = default;

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

            iProxyN a = arena.iProxyVec(vecLen, 10);

            a = -a;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == -(iProxy)10f);

            arena.Dispose();
        }

        public void AddVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            iProxyN a = arena.iProxyVec(vecLen, 10);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)10d);

            a += 1;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)11d);

            iProxyN r = arena.iProxyVec(vecLen, 5);

            a += r;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)16);

            arena.Dispose();
        }

        public void SubVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            iProxyN a = arena.iProxyVec(vecLen, 10);

            a -= 1;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)9f);

            iProxyN r = arena.iProxyVec(vecLen, 5);

            a -= r;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)4d);

            a = arena.iProxyVec(vecLen, 10);

            a = 1 - a;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == -(iProxy)9d);

            arena.Dispose();
        }

        public void MulVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            iProxyN a = arena.iProxyVec(vecLen, 1);

            a *= 1;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)1d);

            a *= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)2d);

            a = arena.iProxyIndexZeroVec(vecLen);

            a *= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)(2d*i));

            a = arena.iProxyIndexZeroVec(vecLen);
            iProxyN b = arena.iProxyIndexZeroVec(vecLen);

            var c = a * b;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(c[i] == (iProxy)(i * i));

            arena.Dispose();
        }

        public void DivVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            iProxyN a = arena.iProxyVec(vecLen, 2);

            a /= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)1);

            a /= 1;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)1);

            a = arena.iProxyIndexZeroVec(vecLen);

            a /= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)(0.5 * i));

            a = arena.iProxyIndexZeroVec(vecLen);
            iProxyN b = arena.iProxyIndexZeroVec(vecLen);

            // add 1 so no division by zero
            a += 1;
            b += 1;

            var c0 = a / b;
            var c1 = b / a;

            for (int i = 0; i < vecLen; i++)
            {
                Assert.IsTrue(c0[i] == (iProxy)1);
                Assert.IsTrue(c1[i] == (iProxy)1);
            }

            a = arena.iProxyVec(vecLen, 2);

            a = 2 / a;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)1);

            arena.Dispose();
        }

        public void ModVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            iProxyN a = arena.iProxyVec(vecLen, 10);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)10);

            a %= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)0);

            a = arena.iProxyIndexZeroVec(vecLen);

            a %= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)(i % (iProxy)2));

            a = arena.iProxyIndexZeroVec(vecLen);
            iProxyN b = arena.iProxyIndexZeroVec(vecLen);

            a += 1;
            b += 1;

            var c0 = a % b;
            var c1 = b % a;

            for (int i = 0; i < vecLen; i++)
            {
                Assert.IsTrue(c0[i] == (iProxy)0);
                Assert.IsTrue(c1[i] == (iProxy)0);
            }

            arena.Dispose();
        }

        public void SignFlipMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;
            int totalElements = vecLen * vecLen;
            iProxyMxN a = arena.iProxyMat(vecLen, vecLen, 10);

            a = -a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == -(iProxy)10f);

            arena.Dispose();
        }

        public void AddMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            iProxyMxN a = arena.iProxyMat(rows, cols, 10);

            a += 1;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (iProxy)11f);

            arena.Dispose();
        }

        public void SubMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            iProxyMxN a = arena.iProxyMat(rows, cols, 10);

            a -= 5;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (iProxy)5f);

            arena.Dispose();
        }

        public void MulMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            iProxyMxN a = arena.iProxyMat(rows, cols, 2);

            a *= 3;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (iProxy)6f);

            a = 3 * a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (iProxy)18f);

            arena.Dispose();
        }

        public void DivMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            iProxyMxN a = arena.iProxyMat(rows, cols, 10);

            a /= 2;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (iProxy)5);

            a = 5 / a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (iProxy)1);

            arena.Dispose();
        }

        public void ModMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            iProxyMxN a = arena.iProxyMat(rows, cols, 10);

            a %= 3;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (iProxy)1f);

            a = arena.iProxyMat(rows, cols, 4);

            a = 4 % a;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (iProxy)0f);

            a = arena.iProxyMat(rows, cols, 3);
            iProxyMxN b = arena.iProxyMat(rows, cols, 2);

            a = a % b;

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (iProxy)1f);

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
