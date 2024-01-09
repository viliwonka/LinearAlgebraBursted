using LinearAlgebra;
using NUnit.Framework;
using System;
using System.Diagnostics;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

public class floatPivotTests
{
    //[BurstCompile]
    public struct TestsJob : IJob
    {
        public enum TestType
        {
            PivotSimpleTest,
            RowPivotIdentityMatTest,
            ColumnPivotIdentityMatTest,
            RowPivotLargeIdentityMatTest,
            ColumnPivotLargeIdentityMatTest,
            RowPivotPermutationMatTest,
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
                    case TestType.RowPivotIdentityMatTest:
                        RowIdentityMatTest(ref arena);
                        break;
                    case TestType.ColumnPivotIdentityMatTest:
                        ColumnIdentityMatTest(ref arena);
                        break;
                    case TestType.RowPivotPermutationMatTest:
                        RowPermutationMatTest(ref arena);
                        break;
                    case TestType.ColumnPivotLargeIdentityMatTest:
                        ColumnLargeIdentityMatTest(ref arena);
                        break;
                    case TestType.RowPivotLargeIdentityMatTest:
                        RowLargeIdentityMatTest(ref arena);
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

        void RowIdentityMatTest(ref Arena arena) {

            Pivot pivot = new Pivot(4, Allocator.Temp);

            pivot.Swap(0, 1);
            pivot.Swap(2, 3);

            var identity = arena.floatIdentityMatrix(4);

            pivot.ApplyRow(ref identity);

            Assert.IsFalse(Analysis.IsIdentity(identity));

            pivot.ApplyInverseRow(ref identity);

            Assert.IsTrue(Analysis.IsIdentity(identity));

            pivot.Reset();

            pivot.ApplyRow(ref identity);

            Assert.IsTrue(Analysis.IsIdentity(identity));

            pivot.Dispose();
        }

        void RowLargeIdentityMatTest(ref Arena arena) {

            int dim = 256;

            Pivot pivot = new Pivot(dim, Allocator.Temp);

            Unity.Mathematics.Random rand = new Unity.Mathematics.Random(1232);

            for (int i = 0; i < dim; i++) {
                pivot.Swap(rand.NextInt(0, dim), rand.NextInt(0, dim));

            }

            var identity = arena.floatIdentityMatrix(dim);

            Assert.IsTrue(Analysis.IsIdentity(identity));

            pivot.ApplyRow(ref identity);
            pivot.ApplyRow(ref identity);

            Assert.IsFalse(Analysis.IsIdentity(identity));

            pivot.ApplyInverseRow(ref identity);
            pivot.ApplyInverseRow(ref identity);

            Assert.IsTrue(Analysis.IsIdentity(identity));

            pivot.Reset();

            pivot.ApplyRow(ref identity);

            Assert.IsTrue(Analysis.IsIdentity(identity));

            pivot.Dispose();
        }

        void ColumnIdentityMatTest(ref Arena arena) {

            Pivot pivot = new Pivot(4, Allocator.Temp);

            pivot.Swap(0, 1);
            pivot.Swap(2, 3);

            var identity = arena.floatIdentityMatrix(4);

            pivot.ApplyColumn(ref identity);

            Assert.IsFalse(Analysis.IsIdentity(identity));

            pivot.ApplyInverseColumn(ref identity);

            Assert.IsTrue(Analysis.IsIdentity(identity));

            pivot.Reset();

            pivot.ApplyColumn(ref identity);

            Assert.IsTrue(Analysis.IsIdentity(identity));

            pivot.Dispose();
        }

        void ColumnLargeIdentityMatTest(ref Arena arena) {

            int dim = 256;

            Pivot pivot = new Pivot(dim, Allocator.Temp);

            Unity.Mathematics.Random rand = new Unity.Mathematics.Random(1232);

            for (int i = 0; i < dim; i++) {
                pivot.Swap(rand.NextInt(0, dim), rand.NextInt(0, dim));

            }

            var identity = arena.floatIdentityMatrix(dim);

            Assert.IsTrue(Analysis.IsIdentity(identity));

            pivot.ApplyColumn(ref identity);
            pivot.ApplyColumn(ref identity);

            Assert.IsFalse(Analysis.IsIdentity(identity));

            pivot.ApplyInverseColumn(ref identity);
            pivot.ApplyInverseColumn(ref identity);

            Assert.IsTrue(Analysis.IsIdentity(identity));

            pivot.Reset();

            pivot.ApplyColumn(ref identity);

            Assert.IsTrue(Analysis.IsIdentity(identity));

            pivot.Dispose();
        }

        void RowPermutationMatTest(ref Arena arena) {

            var permutationMatrix = arena.floatPermutationMatrix(4, 2, 3);

            Pivot pivot = new Pivot(4, Allocator.Temp);

            pivot.Swap(2, 3);

            pivot.ApplyInverseRow(ref permutationMatrix);

            Assert.IsTrue(Analysis.IsIdentity(permutationMatrix));

            pivot.Dispose();
        }

        void PivotVecTest(ref Arena arena) {
            
            Pivot pivot = new Pivot(4, Allocator.Temp);

            pivot.Swap(1, 2);

            var vec = arena.floatBasisVector(4, 0);

            Print.Log(vec);

            var vecCopy = vec.Copy();

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(vec == vecCopy, true));
            
            pivot.ApplyVec(ref vec);

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(vec == vecCopy, true));

            pivot.Swap(0, 3);

            pivot.ApplyVec(ref vec);

            Assert.AreEqual((float)0f, vec[0]);
            Assert.AreEqual((float)0f, vec[1]);
            Assert.AreEqual((float)0f, vec[2]);
            Assert.AreEqual((float)1f, vec[3]);

            pivot.ApplyInverseVec(ref vec);

            Assert.IsTrue(BoolAnalysis.IsAllEqualTo(vec == vecCopy, true));

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
