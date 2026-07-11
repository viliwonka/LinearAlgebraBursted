using System;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Entrywise norms (Norms.L1/L2/LInf) and pattern-preserving componentwise ops
// (mulInPlace/signFlipInPlace/absInPlace/addScaledInPlace) over fProxyBSR. Every correctness
// case validates the sparse op against the dense reference on the ToDense expansion — the
// implicit-zero blocks are what make the sparse/dense equivalence non-trivial, so each test
// matrix omits at least one block. Correctness cases run inside a [BurstCompile] IJob; the
// pattern-mismatch guard runs on the managed thread with Assert.Throws (same split as
// SparseBSRTests).
public class fProxySparseCompNormsTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct SparseCompNormsTestJob : IJob
    {
        public enum TestType
        {
            NormsMatchDense,
            ScaleFlipAbs,
            AddScaledSamePattern,
            EmptyMatrixNorms,
        }

        public TestType Type;

        static fProxy Tol() => /*+choose[1e-4f|1e-11]*/1e-4f/*-choose*/;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.NormsMatchDense: NormsMatchDense(); break;
                case TestType.ScaleFlipAbs: ScaleFlipAbs(); break;
                case TestType.AddScaledSamePattern: AddScaledSamePattern(); break;
                case TestType.EmptyMatrixNorms: EmptyMatrixNorms(); break;
            }
        }

        // 2x2 grid of 2x2 blocks with block (1,0) omitted; values include negatives so the
        // abs-based norms are actually exercised.
        static fProxyBSR BuildTestBSR(ref Arena arena, fProxy scale)
        {
            const int BR = 2, BC = 2;
            var builder = arena.fProxyBSRBuilder(2, 2, BR, BC);

            var b00 = arena.fProxyMat(BR, BC);
            var b01 = arena.fProxyMat(BR, BC);
            var b11 = arena.fProxyMat(BR, BC);
            for (int r = 0; r < BR; r++)
                for (int c = 0; c < BC; c++)
                {
                    b00[r, c] = scale * (fProxy)(1 + r * BC + c);      //  1 ..  4
                    b01[r, c] = scale * (fProxy)(-(5 + r * BC + c));   // -5 .. -8
                    b11[r, c] = scale * (fProxy)(9 + r * BC + c);      //  9 .. 12
                }

            builder.AddBlock(0, 0, in b00);
            builder.AddBlock(0, 1, in b01);
            builder.AddBlock(1, 1, in b11);
            return builder.ToBSR(ref arena);
        }

        // Dense entrywise references computed directly on the expansion.
        static void DenseEntrywiseNorms(in fProxyMxN d, out fProxy l1, out fProxy l2, out fProxy lInf)
        {
            l1 = 0; l2 = 0; lInf = 0;
            for (int i = 0; i < d.M_Rows; i++)
                for (int j = 0; j < d.N_Cols; j++)
                {
                    fProxy av = math.abs(d[i, j]);
                    l1 += av;
                    l2 += av * av;
                    if (av > lInf) lInf = av;
                }
            l2 = math.sqrt(l2);
        }

        void NormsMatchDense()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = BuildTestBSR(ref arena, (fProxy)1);
            var dense = A.ToDense(ref arena);
            DenseEntrywiseNorms(in dense, out fProxy l1, out fProxy l2, out fProxy lInf);

            Assert.IsTrue(math.abs(Norms.L1(in A) - l1) < Tol() * l1);
            Assert.IsTrue(math.abs(Norms.L2(in A) - l2) < Tol() * l2);
            Assert.IsTrue(math.abs(Norms.LInf(in A) - lInf) < Tol() * lInf);

            arena.Dispose();
        }

        void ScaleFlipAbs()
        {
            var arena = new Arena(Allocator.Persistent);

            // mulInPlace: A *= -3, compare against dense expansion scaled the same way.
            var A = BuildTestBSR(ref arena, (fProxy)1);
            var reference = A.ToDense(ref arena);
            A.mulInPlace((fProxy)(-3));
            var scaled = A.ToDense(ref arena);
            for (int i = 0; i < reference.M_Rows; i++)
                for (int j = 0; j < reference.N_Cols; j++)
                    Assert.IsTrue(math.abs(scaled[i, j] - (fProxy)(-3) * reference[i, j]) < Tol());

            // signFlipInPlace: back to +3 * reference.
            A.signFlipInPlace();
            var flipped = A.ToDense(ref arena);
            for (int i = 0; i < reference.M_Rows; i++)
                for (int j = 0; j < reference.N_Cols; j++)
                    Assert.IsTrue(math.abs(flipped[i, j] - (fProxy)3 * reference[i, j]) < Tol());

            // absInPlace: |3 * reference|.
            A.absInPlace();
            var abs = A.ToDense(ref arena);
            for (int i = 0; i < reference.M_Rows; i++)
                for (int j = 0; j < reference.N_Cols; j++)
                    Assert.IsTrue(math.abs(abs[i, j] - math.abs((fProxy)3 * reference[i, j])) < Tol());

            arena.Dispose();
        }

        void AddScaledSamePattern()
        {
            var arena = new Arena(Allocator.Persistent);

            // Same builder recipe twice -> identical pattern, different values via scale.
            var y = BuildTestBSR(ref arena, (fProxy)1);
            var x = BuildTestBSR(ref arena, (fProxy)10);
            Assert.IsTrue(BSR.samePattern(in y, in x));

            var yDense = y.ToDense(ref arena);
            var xDense = x.ToDense(ref arena);

            fProxy a = (fProxy)0.5f;
            y.addScaledInPlace(a, in x);

            var result = y.ToDense(ref arena);
            for (int i = 0; i < result.M_Rows; i++)
                for (int j = 0; j < result.N_Cols; j++)
                    Assert.IsTrue(math.abs(result[i, j] - (yDense[i, j] + a * xDense[i, j])) < Tol());

            arena.Dispose();
        }

        void EmptyMatrixNorms()
        {
            var arena = new Arena(Allocator.Persistent);

            // No blocks added: all-zero matrix, all norms exactly 0.
            var builder = arena.fProxyBSRBuilder(2, 2, 2, 2);
            var A = builder.ToBSR(ref arena);

            Assert.IsTrue(Norms.L1(in A) == (fProxy)0);
            Assert.IsTrue(Norms.L2(in A) == (fProxy)0);
            Assert.IsTrue(Norms.LInf(in A) == (fProxy)0);

            arena.Dispose();
        }
    }

    // ---- correctness cases (Burst) -------------------------------------------------------

    [Test]
    public void NormsMatchDenseTest()
        => new SparseCompNormsTestJob { Type = SparseCompNormsTestJob.TestType.NormsMatchDense }.Run();

    [Test]
    public void ScaleFlipAbsTest()
        => new SparseCompNormsTestJob { Type = SparseCompNormsTestJob.TestType.ScaleFlipAbs }.Run();

    [Test]
    public void AddScaledSamePatternTest()
        => new SparseCompNormsTestJob { Type = SparseCompNormsTestJob.TestType.AddScaledSamePattern }.Run();

    [Test]
    public void EmptyMatrixNormsTest()
        => new SparseCompNormsTestJob { Type = SparseCompNormsTestJob.TestType.EmptyMatrixNorms }.Run();

    // ---- guard case (managed thread; Assert.Throws cannot run inside Burst) ---------------

    [Test]
    public void AddScaledPatternMismatchThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            const int BR = 2, BC = 2;
            var by = arena.fProxyBSRBuilder(2, 2, BR, BC);
            var bx = arena.fProxyBSRBuilder(2, 2, BR, BC);
            var block = arena.fProxyMat(BR, BC, (fProxy)1);

            by.AddBlock(0, 0, in block);
            bx.AddBlock(1, 1, in block);   // same count, different placement

            var y = by.ToBSR(ref arena);
            var x = bx.ToBSR(ref arena);

            Assert.IsFalse(BSR.samePattern(in y, in x));
            Assert.Throws<ArgumentException>(() => y.addScaledInPlace((fProxy)1, in x));
        }
        finally
        {
            arena.Dispose();
        }
    }
}
