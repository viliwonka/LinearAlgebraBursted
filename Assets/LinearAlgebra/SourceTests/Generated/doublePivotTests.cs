using LinearAlgebra;
using NUnit.Framework;
using System;
using System.Diagnostics;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

public class doublePivotTests
{
    [BurstCompile]
    public struct TestsJob : IJob
    {
        public enum TestType
        {
            PivotSimpleTest,
            PivotIdentityMatTest,
            PivotPermutationMatTest,
            PivotVecTest,
        }

        public TestType Type;

        public void Execute()
        {
            Arena arena = new Arena(Allocator.Temp);
            try 
            {
                switch (Type) 
                {
                    case TestType.PivotSimpleTest:
                        Test(ref arena);
                        break;
                    case TestType.PivotIdentityMatTest:
                        IdentityMatTest(ref arena);
                        break;
                    case TestType.PivotPermutationMatTest:
                        PermutationMatTest(ref arena);
                        break;
                    case TestType.PivotVecTest:
                        PivotVecTest(ref arena);
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

        void Test(ref Arena arena)
        {
            Pivot pivot = new Pivot(4, Allocator.Temp);

            Assert.AreEqual(0, pivot[0]);
            Assert.AreEqual(1, pivot[1]);

            pivot.Swap(0, 1);

            Assert.AreEqual(1, pivot[0]);
            Assert.AreEqual(0, pivot[1]);

            pivot.Swap(0, 1);

            Assert.AreEqual(0, pivot[0]);
            Assert.AreEqual(1, pivot[1]);

            pivot.Dispose();
        }

        void IdentityMatTest(ref Arena arena) {

            Pivot pivot = new Pivot(4, Allocator.Temp);

            pivot.Swap(0, 1);
            pivot.Swap(2, 3);

            var identity = arena.doubleIdentityMatrix(4);

            pivot.ApplyRow(ref identity);

            Assert.IsFalse(Analysis.IsIdentity(identity));

            pivot.ApplyInverseRow(ref identity);

            Assert.IsTrue(Analysis.IsIdentity(identity));

            pivot.Reset();

            pivot.ApplyRow(ref identity);

            Assert.IsTrue(Analysis.IsIdentity(identity));

            pivot.Dispose();
        }

        void PermutationMatTest(ref Arena arena) {

            var permutationMatrix = arena.doublePermutationMatrix(4, 2, 3);

            Pivot pivot = new Pivot(4, Allocator.Temp);

            pivot.Swap(2, 3);

            pivot.ApplyInverseRow(ref permutationMatrix);

            Assert.IsTrue(Analysis.IsIdentity(permutationMatrix));

            pivot.Dispose();
        }

        void PivotVecTest(ref Arena arena) {
            
            Pivot pivot = new Pivot(4, Allocator.Temp);

            pivot.Swap(1, 2);

            var vec = arena.doubleBasisVector(4, 0);

            Print.Log(vec);
            
            Assert.AreEqual(1f, vec[0]);
            Assert.AreEqual(0f, vec[1]);
            Assert.AreEqual(0f, vec[2]);
            Assert.AreEqual(0f, vec[3]);

            pivot.ApplyVec(ref vec);

            Assert.AreEqual(1f, vec[0]);
            Assert.AreEqual(0f, vec[1]);
            Assert.AreEqual(0f, vec[2]);
            Assert.AreEqual(0f, vec[3]);

            pivot.Swap(0, 3);

            pivot.ApplyVec(ref vec);

            Assert.AreEqual(0f, vec[0]);
            Assert.AreEqual(0f, vec[1]);
            Assert.AreEqual(0f, vec[2]);
            Assert.AreEqual(1f, vec[3]);

            pivot.Dispose();
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
