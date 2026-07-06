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
    [BurstCompile(CompileSynchronously = true)]
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
            ColumnPivotPermutationMatTest,
            RowPivotVecTest,
            PivotSignTest,
            PivotArenaTest,
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
                    case TestType.ColumnPivotLargeIdentityMatTest:
                        ColumnLargeIdentityMatTest(ref arena);
                        break;
                    case TestType.RowPivotLargeIdentityMatTest:
                        RowLargeIdentityMatTest(ref arena);
                        break;
                    case TestType.RowPivotPermutationMatTest:
                        RowPermutationMatTest(ref arena);
                        break;
                    case TestType.ColumnPivotPermutationMatTest:
                        ColumnPermutationMatTest(ref arena);
                        break;
                    case TestType.RowPivotVecTest:
                        PivotVecTest(ref arena);
                        break;
                    case TestType.PivotSignTest:
                        SignTest(ref arena);
                        break;
                    case TestType.PivotArenaTest:
                        ArenaPivotTest(ref arena);
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

            var identity = arena.doubleIdentityMat(4);

            pivot.ApplyRow(ref identity);

            Assert.IsFalse(Analysis.isIdentity(identity));

            pivot.ApplyInverseRow(ref identity);

            Assert.IsTrue(Analysis.isIdentity(identity));

            pivot.Reset();

            pivot.ApplyRow(ref identity);

            Assert.IsTrue(Analysis.isIdentity(identity));

            pivot.Dispose();
        }

        void RowLargeIdentityMatTest(ref Arena arena) {

            int dim = 256;

            Pivot pivot = new Pivot(dim, Allocator.Temp);

            Unity.Mathematics.Random rand = new Unity.Mathematics.Random(1232);

            for (int i = 0; i < dim; i++) {
                pivot.Swap(rand.NextInt(0, dim), rand.NextInt(0, dim));
            }

            var identity = arena.doubleIdentityMat(dim);

            Assert.IsTrue(Analysis.isIdentity(identity));

            pivot.ApplyRow(ref identity);
            pivot.ApplyRow(ref identity);

            Assert.IsFalse(Analysis.isIdentity(identity));

            pivot.ApplyInverseRow(ref identity);
            pivot.ApplyInverseRow(ref identity);

            Assert.IsTrue(Analysis.isIdentity(identity));

            pivot.Reset();

            pivot.ApplyRow(ref identity);

            Assert.IsTrue(Analysis.isIdentity(identity));

            pivot.Dispose();
        }

        void ColumnIdentityMatTest(ref Arena arena) {

            Pivot pivot = new Pivot(4, Allocator.Temp);

            pivot.Swap(0, 1);
            pivot.Swap(2, 3);

            var identity = arena.doubleIdentityMat(4);

            pivot.ApplyColumn(ref identity);

            Assert.IsFalse(Analysis.isIdentity(identity));

            pivot.ApplyInverseColumn(ref identity);

            Assert.IsTrue(Analysis.isIdentity(identity));

            pivot.Reset();

            pivot.ApplyColumn(ref identity);

            Assert.IsTrue(Analysis.isIdentity(identity));

            pivot.Dispose();
        }

        void ColumnLargeIdentityMatTest(ref Arena arena) {

            int dim = 256;

            Pivot pivot = new Pivot(dim, Allocator.Temp);

            Unity.Mathematics.Random rand = new Unity.Mathematics.Random(1232);

            for (int i = 0; i < dim; i++) {
                pivot.Swap(rand.NextInt(0, dim), rand.NextInt(0, dim));
            }

            var identity = arena.doubleIdentityMat(dim);

            Assert.IsTrue(Analysis.isIdentity(identity));

            pivot.ApplyColumn(ref identity);
            pivot.ApplyColumn(ref identity);

            Assert.IsFalse(Analysis.isIdentity(identity));

            pivot.ApplyInverseColumn(ref identity);
            pivot.ApplyInverseColumn(ref identity);

            Assert.IsTrue(Analysis.isIdentity(identity));

            pivot.Reset();

            pivot.ApplyColumn(ref identity);

            Assert.IsTrue(Analysis.isIdentity(identity));

            pivot.Dispose();
        }

        void RowPermutationMatTest(ref Arena arena) {

            var permutationMatrix = arena.doublePermutationMat(8, 2, 3);

            permutationMatrix = Blas.dot(permutationMatrix, arena.doublePermutationMat(8, 3, 6));
            permutationMatrix = Blas.dot(permutationMatrix, arena.doublePermutationMat(8, 6, 7));
            permutationMatrix = Blas.dot(permutationMatrix, arena.doublePermutationMat(8, 1, 4));

            Pivot pivot = new Pivot(8, Allocator.Temp);

            pivot.Swap(2, 3);
            pivot.Swap(3, 6);
            pivot.Swap(6, 7);
            pivot.Swap(1, 4);

            // applying inverse pivot operation to permutation matrix should form identity matrix
            pivot.ApplyInverseRow(ref permutationMatrix);

            Assert.IsTrue(Analysis.isIdentity(permutationMatrix));

            pivot.Dispose();
        }

        void ColumnPermutationMatTest(ref Arena arena) {

            var permutationMatrix = arena.doublePermutationMat(8, 2, 3);

            permutationMatrix = Blas.dot(permutationMatrix, arena.doublePermutationMat(8, 3, 6));
            permutationMatrix = Blas.dot(permutationMatrix, arena.doublePermutationMat(8, 6, 7));
            permutationMatrix = Blas.dot(permutationMatrix, arena.doublePermutationMat(8, 1, 4));

            permutationMatrix = Blas.trans(permutationMatrix);

            Pivot pivot = new Pivot(8, Allocator.Temp);

            pivot.Swap(2, 3);
            pivot.Swap(3, 6);
            pivot.Swap(6, 7);
            pivot.Swap(1, 4);

            // column analogue of RowPermutationMatTest above: inverse pivot should form identity.
            pivot.ApplyInverseColumn(ref permutationMatrix);

            Assert.IsTrue(Analysis.isIdentity(permutationMatrix));
              
            pivot.Dispose();
        }

        void PivotVecTest(ref Arena arena) {
            
            Pivot pivot = new Pivot(4, Allocator.Temp);

            pivot.Swap(1, 2);

            // [1, 0, 0, 0]
            var vec = arena.doubleBasisVec(4, 0);

            Print.Log(vec);

            var vecCopy = vec.Copy();

            Assert.IsTrue(Analysis.IsAllEqualTo(vec == vecCopy, true));

            // [1, 0, 0, 0] -> [0, 0, 0, 1]
            pivot.ApplyVec(ref vec);

            Assert.IsTrue(Analysis.IsAllEqualTo(vec == vecCopy, true));

            pivot.Swap(0, 3);

            pivot.ApplyVec(ref vec);

            Assert.IsTrue(vec[0] == (double)0f);
            Assert.IsTrue(vec[1] == (double)0f);
            Assert.IsTrue(vec[2] == (double)0f);
            Assert.IsTrue(vec[3] == (double)1f);

            pivot.ApplyInverseVec(ref vec);

            Assert.IsTrue(Analysis.IsAllEqualTo(vec == vecCopy, true));

            pivot.Dispose();
        }

        void SignTest(ref Arena arena) {

            Pivot pivot = new Pivot(4, Allocator.Temp);

            // fresh pivot is even -> +1
            Assert.AreEqual(1, pivot.Sign);

            // one effective swap -> odd -> -1
            pivot.Swap(0, 1);
            Assert.AreEqual(-1, pivot.Sign);

            // a second distinct swap -> even -> +1
            pivot.Swap(2, 3);
            Assert.AreEqual(1, pivot.Sign);

            // Swap(i,i) is a no-op for parity
            pivot.Swap(2, 2);
            Assert.AreEqual(1, pivot.Sign);

            // another swap -> odd -> -1
            pivot.Swap(1, 3);
            Assert.AreEqual(-1, pivot.Sign);

            // Copy preserves Sign
            var copy = pivot.Copy();
            Assert.AreEqual(pivot.Sign, copy.Sign);

            // InverseCopy preserves Sign (permutation and inverse share parity)
            var inv = pivot.InverseCopy();
            Assert.AreEqual(pivot.Sign, inv.Sign);

            // Reset -> +1
            pivot.Reset();
            Assert.AreEqual(1, pivot.Sign);

            copy.Dispose();
            inv.Dispose();
            pivot.Dispose();
        }

        void ArenaPivotTest(ref Arena arena) {

            // Arena-tracked pivot: do NOT dispose it manually.
            var pivot = arena.Pivot(8);

            Assert.AreEqual(8, pivot.N);

            pivot.Swap(1, 5);
            pivot.Swap(2, 7);

            var identity = arena.doubleIdentityMat(8);

            pivot.ApplyRow(ref identity);

            Assert.IsFalse(Analysis.isIdentity(identity));

            pivot.ApplyInverseRow(ref identity);

            Assert.IsTrue(Analysis.isIdentity(identity));

            // intentionally NOT disposing pivot - arena.Dispose() owns it (in Execute's finally).
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

    // Bounds checks throw managed exceptions, so they must run on the managed
    // thread (NOT inside a BurstCompile job).
    [Test]
    public void PivotBoundsTest()
    {
        Pivot pivot = new Pivot(4, Allocator.Temp);

        // indexer getter out of range
        Assert.Catch<ArgumentOutOfRangeException>(() => { var _ = pivot[-1]; });
        Assert.Catch<ArgumentOutOfRangeException>(() => { var _ = pivot[4]; });

        // Swap out of range on either argument
        Assert.Catch<ArgumentOutOfRangeException>(() => pivot.Swap(-1, 0));
        Assert.Catch<ArgumentOutOfRangeException>(() => pivot.Swap(0, 4));
        Assert.Catch<ArgumentOutOfRangeException>(() => pivot.Swap(4, 4));

        // valid index still works
        Assert.AreEqual(0, pivot[0]);
        Assert.AreEqual(3, pivot[3]);

        pivot.Dispose();
    }
}
