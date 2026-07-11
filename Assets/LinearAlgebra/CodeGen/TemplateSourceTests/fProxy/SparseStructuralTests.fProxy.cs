using System;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Structural sparse ops: Analysis.trace/diagonal over BSR, BSR.transpose (allocating +
// destination-reuse), and the per-frame reassembly path (builder.Clear +
// BuildAssemblyCache/Refill). Correctness is always established against the dense ToDense
// expansion; test matrices omit blocks (including a DIAGONAL block for trace/diagonal) so the
// implicit-zero handling is exercised. Correctness cases run inside a [BurstCompile] IJob;
// guard cases run on the managed thread with Assert.Throws (same split as SparseBSRTests).
public class fProxySparseStructuralTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct SparseStructuralTestJob : IJob
    {
        public enum TestType
        {
            TraceAndDiagonal,
            DenseDiagonal,
            TransposeRectangular,
            TransposeIntoReuse,
            TransposeSymmetric,
            AssemblyRefill,
        }

        public TestType Type;

        static fProxy Tol() => /*+choose[1e-4f|1e-11]*/1e-4f/*-choose*/;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.TraceAndDiagonal: TraceAndDiagonal(); break;
                case TestType.DenseDiagonal: DenseDiagonal(); break;
                case TestType.TransposeRectangular: TransposeRectangular(); break;
                case TestType.TransposeIntoReuse: TransposeIntoReuse(); break;
                case TestType.TransposeSymmetric: TransposeSymmetric(); break;
                case TestType.AssemblyRefill: AssemblyRefill(); break;
            }
        }

        // Fills a block with distinct values seeded by (br, bc), including negatives.
        static void FillBlock(ref fProxyMxN b, int br, int bc, fProxy scale)
        {
            for (int r = 0; r < b.M_Rows; r++)
                for (int c = 0; c < b.N_Cols; c++)
                    b[r, c] = scale * (fProxy)((1 + br * 31 + bc * 7 + r * b.N_Cols + c) * ((r + c) % 2 == 0 ? 1 : -1));
        }

        void TraceAndDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            // 3x3 grid of 2x2 blocks; diagonal block (0,0) OMITTED — its diag entries must read 0.
            const int BR = 2, BC = 2;
            var builder = arena.fProxyBSRBuilder(3, 3, BR, BC);
            var b = arena.fProxyMat(BR, BC);

            FillBlock(ref b, 0, 1, (fProxy)1); builder.AddBlock(0, 1, in b);
            FillBlock(ref b, 1, 1, (fProxy)1); builder.AddBlock(1, 1, in b);
            FillBlock(ref b, 2, 0, (fProxy)1); builder.AddBlock(2, 0, in b);
            FillBlock(ref b, 2, 2, (fProxy)1); builder.AddBlock(2, 2, in b);

            var A = builder.ToBSR(ref arena);
            var dense = A.ToDense(ref arena);

            // Reference trace + diagonal straight off the dense expansion.
            fProxy refTrace = 0;
            for (int i = 0; i < dense.M_Rows; i++) refTrace += dense[i, i];

            Assert.IsTrue(math.abs(Analysis.trace(in A) - refTrace) < Tol());

            var d = arena.fProxyVec(A.M_Rows);
            Analysis.diagonal(in A, ref d);
            for (int i = 0; i < dense.M_Rows; i++)
                Assert.IsTrue(math.abs(d[i] - dense[i, i]) < Tol());

            // Rows 0..1 fall in the omitted (0,0) diagonal block: exactly zero.
            Assert.IsTrue(d[0] == (fProxy)0);
            Assert.IsTrue(d[1] == (fProxy)0);

            arena.Dispose();
        }

        void DenseDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyMat(4, 6);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 6; j++)
                    A[i, j] = (fProxy)(i * 10 + j);

            var d = arena.fProxyVec(4);   // min(4, 6)
            Analysis.diagonal(in A, ref d);
            for (int i = 0; i < 4; i++)
                Assert.IsTrue(d[i] == A[i, i]);

            arena.Dispose();
        }

        void TransposeRectangular()
        {
            var arena = new Arena(Allocator.Persistent);

            // 2x3 grid of RECTANGULAR 2x3 blocks, blocks (0,1) and (1,0) omitted.
            const int BR = 2, BC = 3;
            var builder = arena.fProxyBSRBuilder(2, 3, BR, BC);
            var b = arena.fProxyMat(BR, BC);

            FillBlock(ref b, 0, 0, (fProxy)1); builder.AddBlock(0, 0, in b);
            FillBlock(ref b, 0, 2, (fProxy)1); builder.AddBlock(0, 2, in b);
            FillBlock(ref b, 1, 1, (fProxy)1); builder.AddBlock(1, 1, in b);
            FillBlock(ref b, 1, 2, (fProxy)1); builder.AddBlock(1, 2, in b);

            var A = builder.ToBSR(ref arena);
            var At = BSR.transpose(in A, ref arena);

            Assert.IsTrue(At.M_Rows == A.N_Cols);
            Assert.IsTrue(At.N_Cols == A.M_Rows);
            Assert.IsTrue(At.Nnzb == A.Nnzb);

            var dense = A.ToDense(ref arena);
            var denseT = At.ToDense(ref arena);
            for (int i = 0; i < dense.M_Rows; i++)
                for (int j = 0; j < dense.N_Cols; j++)
                    Assert.IsTrue(math.abs(denseT[j, i] - dense[i, j]) < Tol());

            arena.Dispose();
        }

        void TransposeIntoReuse()
        {
            var arena = new Arena(Allocator.Persistent);

            const int BR = 2, BC = 2;
            var builder = arena.fProxyBSRBuilder(2, 2, BR, BC);
            var b = arena.fProxyMat(BR, BC);
            FillBlock(ref b, 0, 0, (fProxy)1); builder.AddBlock(0, 0, in b);
            FillBlock(ref b, 0, 1, (fProxy)1); builder.AddBlock(0, 1, in b);
            FillBlock(ref b, 1, 1, (fProxy)1); builder.AddBlock(1, 1, in b);

            var A = builder.ToBSR(ref arena);
            var At = arena.fProxyBSR(A.BlockCols, A.BlockRows, A.BC, A.BR, A.Nnzb, true);

            // First pass.
            BSR.transpose(in A, ref At);
            var dense = A.ToDense(ref arena);
            var denseT = At.ToDense(ref arena);
            for (int i = 0; i < dense.M_Rows; i++)
                for (int j = 0; j < dense.N_Cols; j++)
                    Assert.IsTrue(math.abs(denseT[j, i] - dense[i, j]) < Tol());

            // Mutate A's values, re-transpose into the SAME destination.
            A.mulInPlace((fProxy)(-2));
            BSR.transpose(in A, ref At);
            var dense2 = A.ToDense(ref arena);
            var denseT2 = At.ToDense(ref arena);
            for (int i = 0; i < dense2.M_Rows; i++)
                for (int j = 0; j < dense2.N_Cols; j++)
                    Assert.IsTrue(math.abs(denseT2[j, i] - dense2[i, j]) < Tol());

            arena.Dispose();
        }

        void TransposeSymmetric()
        {
            var arena = new Arena(Allocator.Persistent);

            const int BR = 2;
            var builder = arena.fProxyBSRBuilder(2, 2, BR, BR);
            var diag = arena.fProxyMat(BR, BR);
            var off = arena.fProxyMat(BR, BR);

            // Symmetric diagonal blocks + one upper off-diagonal block.
            diag[0, 0] = (fProxy)4; diag[0, 1] = (fProxy)1;
            diag[1, 0] = (fProxy)1; diag[1, 1] = (fProxy)5;
            FillBlock(ref off, 0, 1, (fProxy)1);

            builder.AddBlock(0, 0, in diag);
            builder.AddBlock(1, 1, in diag);
            builder.AddBlock(0, 1, in off);

            var A = builder.ToBSRSymmetric(ref arena);
            var At = BSR.transpose(in A, ref arena);

            // A symmetric matrix equals its transpose.
            var dense = A.ToDense(ref arena);
            var denseT = At.ToDense(ref arena);
            for (int i = 0; i < dense.M_Rows; i++)
                for (int j = 0; j < dense.N_Cols; j++)
                    Assert.IsTrue(math.abs(denseT[i, j] - dense[i, j]) < Tol());

            arena.Dispose();
        }

        void AssemblyRefill()
        {
            var arena = new Arena(Allocator.Persistent);

            const int BR = 2, BC = 2;
            var builder = arena.fProxyBSRBuilder(2, 2, BR, BC);
            var b = arena.fProxyMat(BR, BC);

            // Frame 0 — includes a DUPLICATE at (1,1) to exercise summed slots through the map.
            FillBlock(ref b, 0, 0, (fProxy)1); builder.AddBlock(0, 0, in b);
            FillBlock(ref b, 1, 1, (fProxy)1); builder.AddBlock(1, 1, in b);
            FillBlock(ref b, 1, 1, (fProxy)2); builder.AddBlock(1, 1, in b);

            var A = builder.ToBSR(ref arena);
            var cache = builder.BuildAssemblyCache(ref arena);

            // Frame 1 — same topology (same positions, same order), new values.
            builder.Clear();
            FillBlock(ref b, 0, 0, (fProxy)5); builder.AddBlock(0, 0, in b);
            FillBlock(ref b, 1, 1, (fProxy)7); builder.AddBlock(1, 1, in b);
            FillBlock(ref b, 1, 1, (fProxy)(-3)); builder.AddBlock(1, 1, in b);

            builder.Refill(in cache, in A);

            // Reference: an independent fresh compression of the frame-1 triplets.
            var reference = builder.ToBSR(ref arena);
            Assert.IsTrue(BSR.samePattern(in A, in reference));

            var got = A.ToDense(ref arena);
            var want = reference.ToDense(ref arena);
            for (int i = 0; i < got.M_Rows; i++)
                for (int j = 0; j < got.N_Cols; j++)
                    Assert.IsTrue(math.abs(got[i, j] - want[i, j]) < Tol());

            arena.Dispose();
        }
    }

    // ---- correctness cases (Burst) -------------------------------------------------------

    [Test]
    public void TraceAndDiagonalTest()
        => new SparseStructuralTestJob { Type = SparseStructuralTestJob.TestType.TraceAndDiagonal }.Run();

    [Test]
    public void DenseDiagonalTest()
        => new SparseStructuralTestJob { Type = SparseStructuralTestJob.TestType.DenseDiagonal }.Run();

    [Test]
    public void TransposeRectangularTest()
        => new SparseStructuralTestJob { Type = SparseStructuralTestJob.TestType.TransposeRectangular }.Run();

    [Test]
    public void TransposeIntoReuseTest()
        => new SparseStructuralTestJob { Type = SparseStructuralTestJob.TestType.TransposeIntoReuse }.Run();

    [Test]
    public void TransposeSymmetricTest()
        => new SparseStructuralTestJob { Type = SparseStructuralTestJob.TestType.TransposeSymmetric }.Run();

    [Test]
    public void AssemblyRefillTest()
        => new SparseStructuralTestJob { Type = SparseStructuralTestJob.TestType.AssemblyRefill }.Run();

    // ---- guard cases (managed thread) ----------------------------------------------------

    [Test]
    public void RefillTopologyChangeThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            const int BR = 2, BC = 2;
            var builder = arena.fProxyBSRBuilder(2, 2, BR, BC);
            var block = arena.fProxyMat(BR, BC, (fProxy)1);

            builder.AddBlock(0, 0, in block);
            var A = builder.ToBSR(ref arena);
            var cache = builder.BuildAssemblyCache(ref arena);

            // Same count, different position -> topology change -> throw.
            builder.Clear();
            builder.AddBlock(1, 1, in block);
            Assert.Throws<ArgumentException>(() => builder.Refill(in cache, in A));

            // Different count -> throw.
            builder.AddBlock(0, 0, in block);
            Assert.Throws<ArgumentException>(() => builder.Refill(in cache, in A));
        }
        finally
        {
            arena.Dispose();
        }
    }

    [Test]
    public void TransposeWrongDestinationThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            const int BR = 2, BC = 3;
            var builder = arena.fProxyBSRBuilder(2, 3, BR, BC);
            var block = arena.fProxyMat(BR, BC, (fProxy)1);
            builder.AddBlock(0, 0, in block);
            var A = builder.ToBSR(ref arena);

            // Same grid as A (NOT transposed) -> throw.
            var wrongGrid = arena.fProxyBSR(2, 3, BR, BC, 1, true);
            Assert.Throws<ArgumentException>(() => BSR.transpose(in A, ref wrongGrid));

            // Transposed grid but wrong nnzb -> throw.
            var wrongNnzb = arena.fProxyBSR(3, 2, BC, BR, 2, true);
            Assert.Throws<ArgumentException>(() => BSR.transpose(in A, ref wrongNnzb));
        }
        finally
        {
            arena.Dispose();
        }
    }
}
