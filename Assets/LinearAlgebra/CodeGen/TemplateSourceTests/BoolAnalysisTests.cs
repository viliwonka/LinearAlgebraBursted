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
            Arena arena = new Arena(Allocator.Temp);
            try 
            {
                switch (Type) 
                {
                    case TestType.isDiagonal:
                        isDiagonal(ref arena);
                        break;
                    case TestType.IsAllSame:
                        IsAllSame(ref arena);
                        break;
                    case TestType.IsAllEqualTo:
                        IsAllEqualTo(ref arena);
                        break;
                    case TestType.IsAnyEqualTo:
                        IsAnyEqualTo(ref arena);
                    break;
                    case TestType.any:
                        any(ref arena);
                        break;
                    case TestType.all:
                        all(ref arena);
                        break;
                    default:
                        throw new NotImplementedException();
                }
            }
            finally
            {
                arena.Dispose();
            }
        }

        void isDiagonal(ref Arena arena)
        {
            int dim = 4;
            boolMxN m = arena.boolMat(dim, dim);

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
            boolMxN rect = arena.boolMat(2, 3);
            rect[0, 0] = true; rect[1, 1] = true;
            Assert.IsFalse(Analysis.isDiagonal(rect));
        }

        void IsAllSame(ref Arena arena)
        {
            int dim = 64;
            boolN v = arena.boolRandomVec(dim);

            Assert.IsFalse(Analysis.IsAllSame(v));

            v &= false;

            Assert.IsTrue(Analysis.IsAllSame(v));
        }

        void IsAllEqualTo(ref Arena arena)
        {
            int dim = 64;
            boolN v = arena.boolRandomVec(dim);

            Assert.IsFalse(Analysis.IsAllEqualTo(v, true));
            Assert.IsFalse(Analysis.IsAllEqualTo(v, false));

            v |= true;
            
            Assert.IsTrue(Analysis.IsAllEqualTo(v, true));
        }

        void IsAnyEqualTo(ref Arena arena)
        {
            int dim = 64;
            boolN v = arena.boolVec(dim);

            Assert.IsFalse(Analysis.IsAnyEqualTo(v, true));

            v[0] = true;

            Assert.IsTrue(Analysis.IsAnyEqualTo(v, true));
        }

        // any/all — thin sugar over IsAnyEqualTo(x,true)/IsAllEqualTo(x,true).
        // Empty semantics (vacuous truth, matching math.any/math.all):
        //   any(empty) == false, all(empty) == true.

        void any(ref Arena arena)
        {
            int dim = 8;

            // --- vectors ---
            // all-false
            boolN allFalse = arena.boolVec(dim);
            Assert.IsFalse(Analysis.any(allFalse));

            // all-true
            boolN allTrue = arena.boolVec(dim);
            allTrue |= true;
            Assert.IsTrue(Analysis.any(allTrue));

            // mixed (single true element among falses)
            boolN mixed = arena.boolVec(dim);
            mixed[dim - 1] = true;
            Assert.IsTrue(Analysis.any(mixed));

            // single-element
            boolN oneTrue = arena.boolVec(1);
            oneTrue[0] = true;
            Assert.IsTrue(Analysis.any(oneTrue));
            boolN oneFalse = arena.boolVec(1);
            Assert.IsFalse(Analysis.any(oneFalse));

            // empty vector -> false (nothing to short-circuit on)
            boolN emptyVec = arena.boolVec(0);
            Assert.IsFalse(Analysis.any(emptyVec));

            // --- matrices ---
            // all-false
            boolMxN mAllFalse = arena.boolMat(dim, dim);
            Assert.IsFalse(Analysis.any(mAllFalse));

            // all-true
            boolMxN mAllTrue = arena.boolMat(dim, dim);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    mAllTrue[i, j] = true;
            Assert.IsTrue(Analysis.any(mAllTrue));

            // mixed (single true)
            boolMxN mMixed = arena.boolMat(dim, dim);
            mMixed[dim - 1, dim - 1] = true;
            Assert.IsTrue(Analysis.any(mMixed));

            // single-element matrix
            boolMxN mOneTrue = arena.boolMat(1, 1);
            mOneTrue[0, 0] = true;
            Assert.IsTrue(Analysis.any(mOneTrue));
            boolMxN mOneFalse = arena.boolMat(1, 1);
            Assert.IsFalse(Analysis.any(mOneFalse));

            // empty matrix -> false
            boolMxN emptyMat = arena.boolMat(0, 0);
            Assert.IsFalse(Analysis.any(emptyMat));
        }

        void all(ref Arena arena)
        {
            int dim = 8;

            // --- vectors ---
            // all-true
            boolN allTrue = arena.boolVec(dim);
            allTrue |= true;
            Assert.IsTrue(Analysis.all(allTrue));

            // all-false
            boolN allFalse = arena.boolVec(dim);
            Assert.IsFalse(Analysis.all(allFalse));

            // mixed (all true except one) -> false
            boolN mixed = arena.boolVec(dim);
            mixed |= true;
            mixed[dim - 1] = false;
            Assert.IsFalse(Analysis.all(mixed));

            // single-element
            boolN oneTrue = arena.boolVec(1);
            oneTrue[0] = true;
            Assert.IsTrue(Analysis.all(oneTrue));
            boolN oneFalse = arena.boolVec(1);
            Assert.IsFalse(Analysis.all(oneFalse));

            // empty vector -> true (vacuous truth, no counterexample)
            boolN emptyVec = arena.boolVec(0);
            Assert.IsTrue(Analysis.all(emptyVec));

            // --- matrices ---
            // all-true
            boolMxN mAllTrue = arena.boolMat(dim, dim);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    mAllTrue[i, j] = true;
            Assert.IsTrue(Analysis.all(mAllTrue));

            // all-false
            boolMxN mAllFalse = arena.boolMat(dim, dim);
            Assert.IsFalse(Analysis.all(mAllFalse));

            // mixed (all true except one) -> false
            boolMxN mMixed = arena.boolMat(dim, dim);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    mMixed[i, j] = true;
            mMixed[dim - 1, dim - 1] = false;
            Assert.IsFalse(Analysis.all(mMixed));

            // single-element matrix
            boolMxN mOneTrue = arena.boolMat(1, 1);
            mOneTrue[0, 0] = true;
            Assert.IsTrue(Analysis.all(mOneTrue));
            boolMxN mOneFalse = arena.boolMat(1, 1);
            Assert.IsFalse(Analysis.all(mOneFalse));

            // empty matrix -> true
            boolMxN emptyMat = arena.boolMat(0, 0);
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
