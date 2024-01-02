using System;
using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

public class shortOperationsTest {

    [BurstCompile]
    public struct BasicVecOpTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            short s = 1;
            shortN a = arena.shortVec(vecLen, 10);


            Assert.AreEqual(vecLen, a.N); 

            shortN b = arena.shortVec(vecLen, 10);

            Assert.AreEqual(a[vecLen/2], b[vecLen/2]);
            
            Assert.AreEqual(2, arena.AllocationsCount);

            shortN result = default;

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

            short s = 1;
            shortMxN a = arena.shortMat(rows, cols, 10);

            shortMxN b = arena.shortMat(rows, cols, 10);

            shortMxN result = default;

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

            shortN a = arena.shortVec(vecLen, 10);

            a = -a;

            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual(-(short)10f, a[i]);

            arena.Dispose();
        }

        public void AddVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            shortN a = arena.shortVec(vecLen, 10);

            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual((short)10d, a[i]);

            a += 1;
            
            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual((short)11d, a[i]);

            shortN r = arena.shortVec(vecLen, 5);

            a += r;

            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual((short)16, a[i]);

            arena.Dispose();
        }

        public void SubVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            shortN a = arena.shortVec(vecLen, 10);

            a -= 1;
            
            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual((short)9f, a[i]);

            shortN r = arena.shortVec(vecLen, 5);

            a -= r;

            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual((short)4d, a[i]);

            a = arena.shortVec(vecLen, 10);
            
            a = 1 - a;

            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual(-(short)9d, a[i]);

            arena.Dispose();
        }

        public void MulVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            shortN a = arena.shortVec(vecLen, 1);

            a *= 1;

            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual((short)1d, a[i]);

            a *= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual((short)2d, a[i]);
                        
            a = arena.shortIndexZeroVector(vecLen);

            a *= 2;
            
            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual((short)(2d*i), a[i]);

            a = arena.shortIndexZeroVector(vecLen);
            shortN b = arena.shortIndexZeroVector(vecLen);

            var c = a * b;

            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual((short)(i * i), c[i]);

            arena.Dispose();
        }

        public void DivVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            shortN a = arena.shortVec(vecLen, 2);

            a /= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual((short)1, a[i]);

            a /= 1;

            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual((short)1, a[i]);

            a = arena.shortIndexZeroVector(vecLen);

            a /= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual((short)(0.5 * i), a[i]);

            a = arena.shortIndexZeroVector(vecLen);
            shortN b = arena.shortIndexZeroVector(vecLen);

            // add 1 so no division by zero
            a += 1;
            b += 1;

            var c0 = a / b;
            var c1 = b / a;

            for (int i = 0; i < vecLen; i++)
            {
                Assert.AreEqual((short)1, c0[i]);
                Assert.AreEqual((short)1, c1[i]);
            }

            a = arena.shortVec(vecLen, 2);

            a = 2 / a;

            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual((short)1, a[i]);   

            arena.Dispose();
        }

        public void ModVec()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;

            shortN a = arena.shortVec(vecLen, 10);

            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual((short)10, a[i]);

            a %= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual((short)0, a[i]);

            a = arena.shortIndexZeroVector(vecLen);

            a %= 2;

            for (int i = 0; i < vecLen; i++)
                Assert.AreEqual((short)(i % (short)2), a[i]);

            a = arena.shortIndexZeroVector(vecLen);
            shortN b = arena.shortIndexZeroVector(vecLen);

            // add 1 so no division by zero
            a += 1;
            b += 1;

            var c0 = a % b;
            var c1 = b % a;

            for (int i = 0; i < vecLen; i++)
            {
                Assert.AreEqual((short)0, c0[i]);
                Assert.AreEqual((short)0, c1[i]);
            }

            arena.Dispose();
        }

        public void SignFlipMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int vecLen = 16;
            int totalElements = vecLen * vecLen;
            shortMxN a = arena.shortMat(vecLen, vecLen, 10);

            a = -a;

            for (int i = 0; i < totalElements; i++)
                Assert.AreEqual(-(short)10f, a[i]);

            arena.Dispose();
        }

        public void AddMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            shortMxN a = arena.shortMat(rows, cols, 10);

            // Element-wise addition with scalar
            a += 1;

            for (int i = 0; i < totalElements; i++)
                Assert.AreEqual((short)11f, a[i]);

            arena.Dispose();
        }

        public void SubMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            shortMxN a = arena.shortMat(rows, cols, 10);

            // Element-wise subtraction with scalar
            a -= 5;

            for (int i = 0; i < totalElements; i++)
                Assert.AreEqual((short)5f, a[i]);

            arena.Dispose();
        }

        public void MulMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            shortMxN a = arena.shortMat(rows, cols, 2);

            // Element-wise multiplication with scalar
            a *= 3;

            for (int i = 0; i < totalElements; i++)
                Assert.AreEqual((short)6f, a[i]);

            a = 3 * a;

            for (int i = 0; i < totalElements; i++)
                Assert.AreEqual((short)18f, a[i]);

            arena.Dispose();
        }

        public void DivMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            shortMxN a = arena.shortMat(rows, cols, 10);

            // Element-wise division with scalar
            a /= 2;

            for (int i = 0; i < totalElements; i++)
                Assert.AreEqual((short)5, a[i]);

            a = 5 / a;

            for (int i = 0; i < totalElements; i++)
                Assert.AreEqual((short)1, a[i]);

            arena.Dispose();
        }

        public void ModMat()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            shortMxN a = arena.shortMat(rows, cols, 10);

            // Element-wise modulo with scalar
            a %= 3;

            for (int i = 0; i < totalElements; i++)
                Assert.AreEqual((short)1f, a[i]);

            a = arena.shortMat(rows, cols, 4);

            a = 4 % a;

            for (int i = 0; i < totalElements; i++)
                Assert.AreEqual((short)0f, a[i]);

            a = arena.shortMat(rows, cols, 3);
            shortMxN b = arena.shortMat(rows, cols, 2);

            a = a % b;

            for (int i = 0; i < totalElements; i++)
                Assert.AreEqual((short)1f, a[i]);

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
