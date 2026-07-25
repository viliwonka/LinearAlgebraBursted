using System;
using BULA;
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
            int vecLen = 16;

            iProxy s = 1;
            iProxyN a = GenerateOP.iProxyVec(vecLen, 10);


            Assert.AreEqual(vecLen, a.N);

            iProxyN b = GenerateOP.iProxyVec(vecLen, 10);

            Assert.IsTrue(b[vecLen/2] == a[vecLen/2]);

            iProxyN result = default;

            result = new iProxyN(in a, Allocator.Temp); iProxyComp.addInPlace(result, s);   // a + s

            result = new iProxyN(in a, Allocator.Temp); iProxyComp.addInPlace(result, s);   // s + a

            result = new iProxyN(in a, Allocator.Temp); iProxyComp.subInPlace(result, s);   // a - s
            result = new iProxyN(in a, Allocator.Temp); iProxyComp.subInPlace(s, result);   // s - a

            result = new iProxyN(in a, Allocator.Temp); iProxyComp.bitwiseComplementInPlace(result);   // ~a

            result = new iProxyN(in a, Allocator.Temp); iProxyComp.mulInPlace(result, s);   // a * s
            result = new iProxyN(in a, Allocator.Temp); iProxyComp.mulInPlace(result, s);   // s * a

            result = new iProxyN(in a, Allocator.Temp); iProxyComp.divInPlace(result, s);   // a / s
            result = new iProxyN(in a, Allocator.Temp); iProxyComp.modInPlace(result, s);   // a % s
            result = new iProxyN(in a, Allocator.Temp); iProxyComp.divInPlace(s, result);   // s / a
            result = new iProxyN(in a, Allocator.Temp); iProxyComp.modInPlace(s, result);   // s % a

            result = new iProxyN(in a, Allocator.Temp); iProxyComp.bitwiseOrInPlace(result, s);    // a | s
            result = new iProxyN(in a, Allocator.Temp); iProxyComp.bitwiseOrInPlace(result, s);    // s | a

            result = new iProxyN(in a, Allocator.Temp); iProxyComp.bitwiseAndInPlace(result, s);   // a & s
            result = new iProxyN(in a, Allocator.Temp); iProxyComp.bitwiseAndInPlace(result, s);   // s & a

            result = new iProxyN(in a, Allocator.Temp); iProxyComp.bitwiseXorInPlace(result, s);   // a ^ s
            result = new iProxyN(in a, Allocator.Temp); iProxyComp.bitwiseXorInPlace(result, s);   // s ^ a

            iProxyComp.bitwiseLeftShiftInPlace(result, 5);    // result <<= 5
            iProxyComp.bitwiseRightShiftInPlace(result, 5);   // result >>= 5

            result = new iProxyN(in a, Allocator.Temp); iProxyComp.addInPlace(result, b);   // a + b
            result = new iProxyN(in a, Allocator.Temp); iProxyComp.subInPlace(result, b);   // a - b
            result = new iProxyN(in a, Allocator.Temp); iProxyComp.mulInPlace(result, b);   // a * b
            result = new iProxyN(in a, Allocator.Temp); iProxyComp.divInPlace(result, b);   // a / b
            result = new iProxyN(in a, Allocator.Temp); iProxyComp.modInPlace(result, b);   // a % b

            result = new iProxyN(in a, Allocator.Temp); iProxyComp.bitwiseOrInPlace(result, b);    // a | b
            result = new iProxyN(in a, Allocator.Temp); iProxyComp.bitwiseAndInPlace(result, b);   // a & b
            result = new iProxyN(in a, Allocator.Temp); iProxyComp.bitwiseXorInPlace(result, b);   // a ^ b
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

            iProxy s = 1;
            iProxyMxN a = GenerateOP.iProxyMat(rows, cols, 10);

            iProxyMxN b = GenerateOP.iProxyMat(rows, cols, 10);

            iProxyMxN result = default;

            result = new iProxyMxN(in a, Allocator.Temp); iProxyComp.addInPlace(result, s);   // a + s

            result = new iProxyMxN(in a, Allocator.Temp); iProxyComp.addInPlace(result, s);   // s + a

            result = new iProxyMxN(in a, Allocator.Temp); iProxyComp.subInPlace(result, s);   // a - s
            result = new iProxyMxN(in a, Allocator.Temp); iProxyComp.subInPlace(s, result);   // s - a

            result = new iProxyMxN(in a, Allocator.Temp); iProxyComp.mulInPlace(result, s);   // a * s
            result = new iProxyMxN(in a, Allocator.Temp); iProxyComp.mulInPlace(result, s);   // s * a

            result = new iProxyMxN(in a, Allocator.Temp); iProxyComp.divInPlace(result, s);   // a / s
            result = new iProxyMxN(in a, Allocator.Temp); iProxyComp.modInPlace(result, s);   // a % s
            result = new iProxyMxN(in a, Allocator.Temp); iProxyComp.divInPlace(s, result);   // s / a
            result = new iProxyMxN(in a, Allocator.Temp); iProxyComp.modInPlace(s, result);   // s % a

            result = new iProxyMxN(in a, Allocator.Temp); iProxyComp.addInPlace(result, b);   // a + b
            result = new iProxyMxN(in a, Allocator.Temp); iProxyComp.subInPlace(result, b);   // a - b
            result = new iProxyMxN(in a, Allocator.Temp); iProxyComp.mulInPlace(result, b);   // a * b
            result = new iProxyMxN(in a, Allocator.Temp); iProxyComp.divInPlace(result, b);   // a / b
            result = new iProxyMxN(in a, Allocator.Temp); iProxyComp.modInPlace(result, b);   // a % b
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
            int vecLen = 16;

            iProxyN a = GenerateOP.iProxyVec(vecLen, 10);

            iProxyComp.signFlipInPlace(a); // a = -a

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == -(iProxy)10f);
        }

        public void AddVec()
        {
            int vecLen = 16;

            iProxyN a = GenerateOP.iProxyVec(vecLen, 10);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)10d);

            iProxyComp.addInPlace(a, 1); // a += 1

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)11d);

            iProxyN r = GenerateOP.iProxyVec(vecLen, 5);

            iProxyComp.addInPlace(a, r); // a += r

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)16);
        }

        public void SubVec()
        {
            int vecLen = 16;

            iProxyN a = GenerateOP.iProxyVec(vecLen, 10);

            iProxyComp.subInPlace(a, 1); // a -= 1

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)9f);

            iProxyN r = GenerateOP.iProxyVec(vecLen, 5);

            iProxyComp.subInPlace(a, r); // a -= r

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)4d);

            a = GenerateOP.iProxyVec(vecLen, 10);

            iProxyComp.subInPlace(1, a); // a = 1 - a

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == -(iProxy)9d);
        }

        public void MulVec()
        {
            int vecLen = 16;

            iProxyN a = GenerateOP.iProxyVec(vecLen, 1);

            iProxyComp.mulInPlace(a, 1); // a *= 1

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)1d);

            iProxyComp.mulInPlace(a, 2); // a *= 2

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)2d);

            a = GenerateOP.iProxyIndexZeroVec(vecLen);

            iProxyComp.mulInPlace(a, 2); // a *= 2

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)(2d*i));

            a = GenerateOP.iProxyIndexZeroVec(vecLen);
            iProxyN b = GenerateOP.iProxyIndexZeroVec(vecLen);

            var c = new iProxyN(in a, Allocator.Temp); iProxyComp.mulInPlace(c, b); // c = a * b

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(c[i] == (iProxy)(i * i));
        }

        public void DivVec()
        {
            int vecLen = 16;

            iProxyN a = GenerateOP.iProxyVec(vecLen, 2);

            iProxyComp.divInPlace(a, 2); // a /= 2

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)1);

            iProxyComp.divInPlace(a, 1); // a /= 1

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)1);

            a = GenerateOP.iProxyIndexZeroVec(vecLen);

            iProxyComp.divInPlace(a, 2); // a /= 2

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)(0.5 * i));

            a = GenerateOP.iProxyIndexZeroVec(vecLen);
            iProxyN b = GenerateOP.iProxyIndexZeroVec(vecLen);

            // add 1 so no division by zero
            iProxyComp.addInPlace(a, 1);
            iProxyComp.addInPlace(b, 1);

            var c0 = new iProxyN(in a, Allocator.Temp); iProxyComp.divInPlace(c0, b); // c0 = a / b
            var c1 = new iProxyN(in b, Allocator.Temp); iProxyComp.divInPlace(c1, a); // c1 = b / a

            for (int i = 0; i < vecLen; i++)
            {
                Assert.IsTrue(c0[i] == (iProxy)1);
                Assert.IsTrue(c1[i] == (iProxy)1);
            }

            a = GenerateOP.iProxyVec(vecLen, 2);

            iProxyComp.divInPlace(2, a); // a = 2 / a

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)1);
        }

        public void ModVec()
        {
            int vecLen = 16;

            iProxyN a = GenerateOP.iProxyVec(vecLen, 10);

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)10);

            iProxyComp.modInPlace(a, 2); // a %= 2

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)0);

            a = GenerateOP.iProxyIndexZeroVec(vecLen);

            iProxyComp.modInPlace(a, 2); // a %= 2

            for (int i = 0; i < vecLen; i++)
                Assert.IsTrue(a[i] == (iProxy)(i % (iProxy)2));

            a = GenerateOP.iProxyIndexZeroVec(vecLen);
            iProxyN b = GenerateOP.iProxyIndexZeroVec(vecLen);

            iProxyComp.addInPlace(a, 1);
            iProxyComp.addInPlace(b, 1);

            var c0 = new iProxyN(in a, Allocator.Temp); iProxyComp.modInPlace(c0, b); // c0 = a % b
            var c1 = new iProxyN(in b, Allocator.Temp); iProxyComp.modInPlace(c1, a); // c1 = b % a

            for (int i = 0; i < vecLen; i++)
            {
                Assert.IsTrue(c0[i] == (iProxy)0);
                Assert.IsTrue(c1[i] == (iProxy)0);
            }
        }

        public void SignFlipMat()
        {
            int vecLen = 16;
            int totalElements = vecLen * vecLen;
            iProxyMxN a = GenerateOP.iProxyMat(vecLen, vecLen, 10);

            iProxyComp.signFlipInPlace(a); // a = -a

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == -(iProxy)10f);
        }

        public void AddMat()
        {
            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            iProxyMxN a = GenerateOP.iProxyMat(rows, cols, 10);

            iProxyComp.addInPlace(a, 1); // a += 1

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (iProxy)11f);
        }

        public void SubMat()
        {
            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            iProxyMxN a = GenerateOP.iProxyMat(rows, cols, 10);

            iProxyComp.subInPlace(a, 5); // a -= 5

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (iProxy)5f);
        }

        public void MulMat()
        {
            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            iProxyMxN a = GenerateOP.iProxyMat(rows, cols, 2);

            iProxyComp.mulInPlace(a, 3); // a *= 3

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (iProxy)6f);

            iProxyComp.mulInPlace(a, 3); // a = 3 * a

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (iProxy)18f);
        }

        public void DivMat()
        {
            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            iProxyMxN a = GenerateOP.iProxyMat(rows, cols, 10);

            iProxyComp.divInPlace(a, 2); // a /= 2

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (iProxy)5);

            iProxyComp.divInPlace(5, a); // a = 5 / a

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (iProxy)1);
        }

        public void ModMat()
        {
            int rows = 8;
            int cols = 8;
            int totalElements = rows * cols;

            iProxyMxN a = GenerateOP.iProxyMat(rows, cols, 10);

            iProxyComp.modInPlace(a, 3); // a %= 3

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (iProxy)1f);

            a = GenerateOP.iProxyMat(rows, cols, 4);

            iProxyComp.modInPlace(4, a); // a = 4 % a

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (iProxy)0f);

            a = GenerateOP.iProxyMat(rows, cols, 3);
            iProxyMxN b = GenerateOP.iProxyMat(rows, cols, 2);

            iProxyComp.modInPlace(a, b); // a = a % b

            for (int i = 0; i < totalElements; i++)
                Assert.IsTrue(a[i] == (iProxy)1f);
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

    // STANDALONE Copy()/TempCopy() contract: both return an independent copy (content-equal;
    // writes to the copy never reach the source).
    [Test]
    public void StandaloneVector_CopyAndTempCopy_ReturnIndependentCopies()
    {
        var v = new iProxyN(4, Allocator.Temp);
        try
        {
            v[1] = 2;
            var c = v.Copy();
            var t = v.TempCopy();
            Assert.IsTrue(c.N == 4);
            Assert.IsTrue(t.N == 4);
            Assert.IsTrue(c[1] == 2);
            Assert.IsTrue(t[1] == 2);
            c[1] = 5;
            t[1] = 7;
            Assert.IsTrue(v[1] == 2);
            c.Dispose();
            t.Dispose();
        }
        finally { v.Dispose(); }
    }

    [Test]
    public void StandaloneMatrix_CopyAndTempCopy_ReturnIndependentCopies()
    {
        var m = new iProxyMxN(3, 3, Allocator.Temp);
        try
        {
            m[1, 2] = 3;
            var c = m.Copy();
            var t = m.TempCopy();
            Assert.IsTrue(c.M_Rows == 3 && c.N_Cols == 3);
            Assert.IsTrue(t.M_Rows == 3 && t.N_Cols == 3);
            Assert.IsTrue(c[1, 2] == 3);
            Assert.IsTrue(t[1, 2] == 3);
            c[1, 2] = 5;
            t[1, 2] = 7;
            Assert.IsTrue(m[1, 2] == 3);
            c.Dispose();
            t.Dispose();
        }
        finally { m.Dispose(); }
    }

}
