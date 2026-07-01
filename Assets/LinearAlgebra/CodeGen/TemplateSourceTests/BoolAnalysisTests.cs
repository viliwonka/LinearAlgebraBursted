using LinearAlgebra;
using NUnit.Framework;
using System;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

public class BoolAnalysisTests
{
    [BurstCompile]
    public struct TestsJob : IJob
    {
        public enum TestType
        {
            isDiagonal,
            IsAllSame,
            IsAllEqualTo,
            IsAnyEqualTo,
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

            Assert.IsFalse(BoolAnalysis_OP.isDiagonal(m));
            
            for (int i = 0; i < dim; i++)
                m[i, i] = true;

            Assert.IsTrue(BoolAnalysis_OP.isDiagonal(m));
        }

        void IsAllSame(ref Arena arena)
        {
            int dim = 64;
            boolN v = arena.boolRandomVec(dim);

            Assert.IsFalse(BoolAnalysis_OP.IsAllSame(v));

            v &= false;

            Assert.IsTrue(BoolAnalysis_OP.IsAllSame(v));
        }

        void IsAllEqualTo(ref Arena arena)
        {
            int dim = 64;
            boolN v = arena.boolRandomVec(dim);

            Assert.IsFalse(BoolAnalysis_OP.IsAllEqualTo(v, true));
            Assert.IsFalse(BoolAnalysis_OP.IsAllEqualTo(v, false));

            v |= true;
            
            Assert.IsTrue(BoolAnalysis_OP.IsAllEqualTo(v, true));
        }

        void IsAnyEqualTo(ref Arena arena)
        {
            int dim = 64;
            boolN v = arena.boolVec(dim);

            Assert.IsFalse(BoolAnalysis_OP.IsAnyEqualTo(v, true));

            v[0] = true;

            Assert.IsTrue(BoolAnalysis_OP.IsAnyEqualTo(v, true));
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
