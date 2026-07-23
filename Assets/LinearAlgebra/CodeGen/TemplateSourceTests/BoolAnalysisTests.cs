using LinearAlgebra;
using NUnit.Framework;
using System;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

public class BoolAnalysisTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct TestsJob : IJob
    {
        public enum TestType
        {
            isDiagonal,
            IsAllSame,
            IsAllEqualTo,
            IsAnyEqualTo,
            any,
            all,
        }

        public TestType Type;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.isDiagonal:
                    isDiagonal();
                    break;
                case TestType.IsAllSame:
                    IsAllSame();
                    break;
                case TestType.IsAllEqualTo:
                    IsAllEqualTo();
                    break;
                case TestType.IsAnyEqualTo:
                    IsAnyEqualTo();
                    break;
                case TestType.any:
                    any();
                    break;
                case TestType.all:
                    all();
                    break;
                default:
                    throw new NotImplementedException();
            }
        }

        void isDiagonal()
        {
            int dim = 4;
            boolMxN m = new boolMxN(dim, dim, Allocator.Temp);

            // All-false square matrix: no off-diagonal trues -> diagonal.
            Assert.IsTrue(Analysis.isDiagonal(m));

            // Diagonal values are free (any mix of true/false stays diagonal).
            for (int i = 0; i < dim; i++)
                m[i, i] = true;
            Assert.IsTrue(Analysis.isDiagonal(m));

            m[2, 2] = false;
            Assert.IsTrue(Analysis.isDiagonal(m));

            // One off-diagonal true breaks it.
            m[0, 1] = true;
            Assert.IsFalse(Analysis.isDiagonal(m));
            m[0, 1] = false;

            // Non-square is never diagonal, even with an identity-like pattern.
            boolMxN rect = new boolMxN(2, 3, Allocator.Temp);
            rect[0, 0] = true; rect[1, 1] = true;
            Assert.IsFalse(Analysis.isDiagonal(rect));
        }

        void IsAllSame()
        {
            int dim = 64;
            boolN v = GenerateOP.boolRandomVec(dim);

            Assert.IsFalse(Analysis.IsAllSame(v));

            v &= false;

            Assert.IsTrue(Analysis.IsAllSame(v));
        }

        void IsAllEqualTo()
        {
            int dim = 64;
            boolN v = GenerateOP.boolRandomVec(dim);

            Assert.IsFalse(Analysis.IsAllEqualTo(v, true));
            Assert.IsFalse(Analysis.IsAllEqualTo(v, false));

            v |= true;

            Assert.IsTrue(Analysis.IsAllEqualTo(v, true));
        }

        void IsAnyEqualTo()
        {
            int dim = 64;
            boolN v = new boolN(dim, Allocator.Temp);

            Assert.IsFalse(Analysis.IsAnyEqualTo(v, true));

            v[0] = true;

            Assert.IsTrue(Analysis.IsAnyEqualTo(v, true));
        }

        // any/all — thin sugar over IsAnyEqualTo(x,true)/IsAllEqualTo(x,true).
        // Empty semantics (vacuous truth, matching math.any/math.all):
        //   any(empty) == false, all(empty) == true.

        void any()
        {
            int dim = 8;

            // --- vectors ---
            // all-false
            boolN allFalse = new boolN(dim, Allocator.Temp);
            Assert.IsFalse(Analysis.any(allFalse));

            // all-true
            boolN allTrue = new boolN(dim, Allocator.Temp);
            allTrue |= true;
            Assert.IsTrue(Analysis.any(allTrue));

            // mixed (single true element among falses)
            boolN mixed = new boolN(dim, Allocator.Temp);
            mixed[dim - 1] = true;
            Assert.IsTrue(Analysis.any(mixed));

            // single-element
            boolN oneTrue = new boolN(1, Allocator.Temp);
            oneTrue[0] = true;
            Assert.IsTrue(Analysis.any(oneTrue));
            boolN oneFalse = new boolN(1, Allocator.Temp);
            Assert.IsFalse(Analysis.any(oneFalse));

            // empty vector -> false (nothing to short-circuit on)
            boolN emptyVec = new boolN(0, Allocator.Temp);
            Assert.IsFalse(Analysis.any(emptyVec));

            // --- matrices ---
            // all-false
            boolMxN mAllFalse = new boolMxN(dim, dim, Allocator.Temp);
            Assert.IsFalse(Analysis.any(mAllFalse));

            // all-true
            boolMxN mAllTrue = new boolMxN(dim, dim, Allocator.Temp);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    mAllTrue[i, j] = true;
            Assert.IsTrue(Analysis.any(mAllTrue));

            // mixed (single true)
            boolMxN mMixed = new boolMxN(dim, dim, Allocator.Temp);
            mMixed[dim - 1, dim - 1] = true;
            Assert.IsTrue(Analysis.any(mMixed));

            // single-element matrix
            boolMxN mOneTrue = new boolMxN(1, 1, Allocator.Temp);
            mOneTrue[0, 0] = true;
            Assert.IsTrue(Analysis.any(mOneTrue));
            boolMxN mOneFalse = new boolMxN(1, 1, Allocator.Temp);
            Assert.IsFalse(Analysis.any(mOneFalse));

            // empty matrix -> false
            boolMxN emptyMat = new boolMxN(0, 0, Allocator.Temp);
            Assert.IsFalse(Analysis.any(emptyMat));
        }

        void all()
        {
            int dim = 8;

            // --- vectors ---
            // all-true
            boolN allTrue = new boolN(dim, Allocator.Temp);
            allTrue |= true;
            Assert.IsTrue(Analysis.all(allTrue));

            // all-false
            boolN allFalse = new boolN(dim, Allocator.Temp);
            Assert.IsFalse(Analysis.all(allFalse));

            // mixed (all true except one) -> false
            boolN mixed = new boolN(dim, Allocator.Temp);
            mixed |= true;
            mixed[dim - 1] = false;
            Assert.IsFalse(Analysis.all(mixed));

            // single-element
            boolN oneTrue = new boolN(1, Allocator.Temp);
            oneTrue[0] = true;
            Assert.IsTrue(Analysis.all(oneTrue));
            boolN oneFalse = new boolN(1, Allocator.Temp);
            Assert.IsFalse(Analysis.all(oneFalse));

            // empty vector -> true (vacuous truth, no counterexample)
            boolN emptyVec = new boolN(0, Allocator.Temp);
            Assert.IsTrue(Analysis.all(emptyVec));

            // --- matrices ---
            // all-true
            boolMxN mAllTrue = new boolMxN(dim, dim, Allocator.Temp);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    mAllTrue[i, j] = true;
            Assert.IsTrue(Analysis.all(mAllTrue));

            // all-false
            boolMxN mAllFalse = new boolMxN(dim, dim, Allocator.Temp);
            Assert.IsFalse(Analysis.all(mAllFalse));

            // mixed (all true except one) -> false
            boolMxN mMixed = new boolMxN(dim, dim, Allocator.Temp);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    mMixed[i, j] = true;
            mMixed[dim - 1, dim - 1] = false;
            Assert.IsFalse(Analysis.all(mMixed));

            // single-element matrix
            boolMxN mOneTrue = new boolMxN(1, 1, Allocator.Temp);
            mOneTrue[0, 0] = true;
            Assert.IsTrue(Analysis.all(mOneTrue));
            boolMxN mOneFalse = new boolMxN(1, 1, Allocator.Temp);
            Assert.IsFalse(Analysis.all(mOneFalse));

            // empty matrix -> true
            boolMxN emptyMat = new boolMxN(0, 0, Allocator.Temp);
            Assert.IsTrue(Analysis.all(emptyMat));
        }
    }


    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(TestsJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void Tests(TestsJob.TestType testType)
    {
        new TestsJob() { Type = testType }.Run();
    }



}
