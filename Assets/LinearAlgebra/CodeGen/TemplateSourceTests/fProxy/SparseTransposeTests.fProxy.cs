using System;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Milestone B test suite for the materialized-transpose optimization on fProxyBSR (block-CSR):
//   * Arena.fProxyBSRTranspose(in A)         -- builds A^T as its own compressed BSR
//   * fProxyBSROperator(in A, in AT) two-arg  -- ApplyT forwards to spMV(AT, .) instead of spMVT(A, .)
//
// The whole point of the optimization is that y = A^T x is computed IDENTICALLY, just via a
// cache-friendly forward traversal of a materialized AT rather than the scatter-heavy on-the-fly
// spMVT over A. So every case here is a cross-check: the new path must agree with the old path to
// within a tight (exact-identity, not iterative-convergence) tolerance. The solver-level wiring
// that consumes these (cgls/lsqr allocating BSR overloads) is regression-covered separately by
// fProxySparseSolverTests -- this file targets the transpose primitive + operator ctor directly.
//
// Correctness cases run inside a [BurstCompile] IJob (matches fProxySparseSolverTests /
// fProxySparseSymmetricTests). There are no guard/exception cases here, so no managed-thread
// Assert.Throws methods are needed.
public class fProxySparseTransposeTests
{
    [BurstCompile]
    public struct SparseTransposeTestJob : IJob
    {
        public enum TestType
        {
            // 1. Rectangular transpose matches on-the-fly spMVT, both block-grid orientations.
            TransposeMatchesSpMVT_TallBlockGrid,   // more block-rows than block-cols (dense m > n)
            TransposeMatchesSpMVT_WideBlockGrid,   // more block-cols than block-rows (dense m < n)

            // 2. Operator ApplyT parity between the one-arg (on-the-fly) and two-arg (precomputed AT) ctors.
            OperatorApplyTParity,

            // 3. Symmetric BSR: transpose is a value-identical no-op (returns A unchanged).
            SymmetricNoOp,
        }

        public TestType Type;

        // Exact-identity cross-check tolerance. spMV(AT, x) and spMVT(A, x) compute the SAME
        // products summed in a different block-traversal order; block values live in a bounded
        // range, so the absolute difference stays well below this scaled threshold on both
        // precisions (float needs the looser bound, double is far tighter). Matches the matvec
        // cross-check idiom of fProxySparseSymmetricTests.Tol().
        static fProxy Tol() => /*+choose[1e-4f|1e-11]*/1e-4f/*-choose*/;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.TransposeMatchesSpMVT_TallBlockGrid: TransposeMatchesSpMVT_TallBlockGrid(); break;
                case TestType.TransposeMatchesSpMVT_WideBlockGrid: TransposeMatchesSpMVT_WideBlockGrid(); break;
                case TestType.OperatorApplyTParity: OperatorApplyTParity(); break;
                case TestType.SymmetricNoOp: SymmetricNoOp(); break;
            }
        }

        // ---- helpers ---------------------------------------------------------------------

        static void AssertVecEq(in fProxyN a, in fProxyN b, fProxy tol)
        {
            Assert.IsTrue(Analysis_OP.isZero(a - b, tol));
        }

        // A random rectangular BSR on a 3x2 block grid of 3x2 (BR x BC) blocks -> dense 9x4 (m > n).
        // BR != BC deliberately exercises the block-interior dimension swap (3x2 -> 2x3) in
        // fProxyBSRTranspose. Only 4 of the 6 block positions are filled (leaving (1,0) and (2,1)
        // empty) so genuine block sparsity -- not a dense grid -- is transposed.
        static fProxyBSR BuildTallRectBSR(ref Arena arena)
        {
            const int blockRows = 3, blockCols = 2, BR = 3, BC = 2;
            var b = arena.fProxyBSRBuilder(blockRows, blockCols, BR, BC, 4);
            b.AddBlock(0, 0, arena.fProxyRandomMat(BR, BC, -1f, 1f, 70001));
            b.AddBlock(0, 1, arena.fProxyRandomMat(BR, BC, -1f, 1f, 70002));
            b.AddBlock(1, 1, arena.fProxyRandomMat(BR, BC, -1f, 1f, 70003));
            b.AddBlock(2, 0, arena.fProxyRandomMat(BR, BC, -1f, 1f, 70004));
            return b.ToBSR(ref arena);
        }

        // A random rectangular BSR on a 2x3 block grid of 2x3 (BR x BC) blocks -> dense 4x9 (m < n).
        // Block-interior transpose here is 2x3 -> 3x2. Again 4 of 6 block positions filled.
        static fProxyBSR BuildWideRectBSR(ref Arena arena)
        {
            const int blockRows = 2, blockCols = 3, BR = 2, BC = 3;
            var b = arena.fProxyBSRBuilder(blockRows, blockCols, BR, BC, 4);
            b.AddBlock(0, 0, arena.fProxyRandomMat(BR, BC, -1f, 1f, 71001));
            b.AddBlock(0, 2, arena.fProxyRandomMat(BR, BC, -1f, 1f, 71002));
            b.AddBlock(1, 0, arena.fProxyRandomMat(BR, BC, -1f, 1f, 71003));
            b.AddBlock(1, 1, arena.fProxyRandomMat(BR, BC, -1f, 1f, 71004));
            return b.ToBSR(ref arena);
        }

        // Core cross-check shared by the tall/wide cases: materialize AT, and for a random x of
        // length A.M_Rows (both A^T x paths consume an m-vector: spMVT asserts SameDim(A.M_Rows, x.N)
        // and AT.N_Cols == A.M_Rows) prove spMV(AT, x) == spMVT(A, x). Also pin the transpose's
        // outer dimensions.
        static void CrossCheckTranspose(ref Arena arena, in fProxyBSR A, uint seed)
        {
            var AT = arena.fProxyBSRTranspose(in A);

            // Outer dimensions swap: A is m x n -> AT is n x m.
            Assert.IsTrue(AT.M_Rows == A.N_Cols);
            Assert.IsTrue(AT.N_Cols == A.M_Rows);
            // No blocks lost or gained by the (non-symmetric) transpose rebuild.
            Assert.IsTrue(AT.Nnzb == A.Nnzb);

            var x = arena.fProxyRandomVec(A.M_Rows, -1f, 1f, seed);

            var y1 = Sparse_OP.spMV(in AT, in x);    // new cache-friendly forward traversal of A^T
            var y2 = Sparse_OP.spMVT(in A, in x);    // old on-the-fly scatter traversal

            Assert.IsTrue(y1.N == A.N_Cols);
            Assert.IsTrue(y2.N == A.N_Cols);
            AssertVecEq(in y1, in y2, Tol());
        }

        // ---- 1a. tall block grid (m > n) ----
        void TransposeMatchesSpMVT_TallBlockGrid()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = BuildTallRectBSR(ref arena);
            Assert.IsTrue(A.M_Rows > A.N_Cols);       // genuinely m > n
            CrossCheckTranspose(ref arena, in A, 72001u);
            arena.Dispose();
        }

        // ---- 1b. wide block grid (m < n) ----
        void TransposeMatchesSpMVT_WideBlockGrid()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = BuildWideRectBSR(ref arena);
            Assert.IsTrue(A.M_Rows < A.N_Cols);       // genuinely m < n
            CrossCheckTranspose(ref arena, in A, 72101u);
            arena.Dispose();
        }

        // ---- 2. operator ApplyT parity: one-arg (spMVT) vs two-arg (spMV over AT) ----
        //
        // Both operators must produce numerically identical ApplyT results when AT really is A's
        // transpose -- that is the whole contract of the two-arg ctor. Apply (forward, not ApplyT)
        // is trivially identical (both just spMV(A, .)), but op2 now carries extra AT state, so a
        // one-line Apply cross-check guards against the extra field perturbing the forward path.
        void OperatorApplyTParity()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = BuildTallRectBSR(ref arena);      // 9 x 4
            var AT = arena.fProxyBSRTranspose(in A);

            var op1 = new fProxyBSROperator(in A);        // one-arg: ApplyT == on-the-fly spMVT(A, .)
            var op2 = new fProxyBSROperator(in A, in AT); // two-arg: ApplyT == spMV(AT, .)

            // ApplyT: y = A^T x, x length m = A.M_Rows, y length n = A.N_Cols. y must not alias x,
            // and each operator needs its own destination.
            var xT = arena.fProxyRandomVec(A.M_Rows, -1f, 1f, 73001);
            var yT1 = arena.fProxyVec(A.N_Cols);
            var yT2 = arena.fProxyVec(A.N_Cols);
            op1.ApplyT(in xT, ref yT1);
            op2.ApplyT(in xT, ref yT2);
            AssertVecEq(in yT1, in yT2, Tol());

            // Apply: y = A x, x length n = A.N_Cols, y length m = A.M_Rows. Should be identical
            // between the two operators (both call spMV(A, .)).
            var xF = arena.fProxyRandomVec(A.N_Cols, -1f, 1f, 73002);
            var yF1 = arena.fProxyVec(A.M_Rows);
            var yF2 = arena.fProxyVec(A.M_Rows);
            op1.Apply(in xF, ref yF1);
            op2.Apply(in xF, ref yF2);
            AssertVecEq(in yF1, in yF2, Tol());

            arena.Dispose();
        }

        // ---- 3. symmetric BSR: transpose is a value-identical no-op ----
        //
        // For symmetric upper-block storage A == A^T by construction, so fProxyBSRTranspose returns
        // A itself unchanged (no redundant materialized copy). The primary contract is numerical:
        // spMV(AT, x) == spMV(A, x) for any x. The structural asserts (same Nnzb / BlockRows /
        // Symmetric flag) are a cheap confirmation that no needless rebuild happened.
        void SymmetricNoOp()
        {
            var arena = new Arena(Allocator.Persistent);

            const int BR = 2, nb = 3;                 // 3x3 block grid of 2x2 blocks -> 6x6 dense
            int dim = BR * nb;
            fProxy strong = (fProxy)dim;

            var s = arena.fProxyBSRBuilder(nb, nb, BR, BR, nb + 2);

            // Diagonal blocks: D_i = M_i^T M_i + strong*I is genuinely symmetric, so the assembled
            // matrix is a true symmetric matrix (not merely symmetric STORAGE of an asymmetric one).
            for (int i = 0; i < nb; i++)
            {
                var Mi = arena.fProxyRandomMat(BR, BR, -1f, 1f, (uint)(74000 + i));
                var Di = Linear_OP.dot(Mi, Mi, true);
                for (int d = 0; d < BR; d++)
                    Di[d, d] += strong;
                s.AddBlock(i, i, in Di);
            }

            // Two upper off-diagonal blocks (blockCol > blockRow, as ToBSRSymmetric requires).
            s.AddBlock(0, 1, arena.fProxyRandomMat(BR, BR, -0.2f, 0.2f, 74100));
            s.AddBlock(1, 2, arena.fProxyRandomMat(BR, BR, -0.2f, 0.2f, 74101));

            var A = s.ToBSRSymmetric(ref arena);
            Assert.IsTrue(A.Symmetric);

            var AT = arena.fProxyBSRTranspose(in A);

            // No-op: same structure (returned A unchanged, no rebuild).
            Assert.IsTrue(AT.Symmetric);
            Assert.IsTrue(AT.Nnzb == A.Nnzb);
            Assert.IsTrue(AT.BlockRows == A.BlockRows);
            Assert.IsTrue(AT.M_Rows == A.M_Rows);
            Assert.IsTrue(AT.N_Cols == A.N_Cols);

            // Numerical no-op: transposing a symmetric matrix leaves spMV unchanged (square -> x
            // length dim = N_Cols = M_Rows).
            var x = arena.fProxyRandomVec(dim, -1f, 1f, 74200);
            var yA = Sparse_OP.spMV(in A, in x);
            var yAT = Sparse_OP.spMV(in AT, in x);
            AssertVecEq(in yAT, in yA, Tol());

            arena.Dispose();
        }
    }

    // ---- correctness entry points (Burst) ------------------------------------------------

    [Test]
    public void TransposeMatchesSpMVT_TallBlockGridTest()
        => new SparseTransposeTestJob { Type = SparseTransposeTestJob.TestType.TransposeMatchesSpMVT_TallBlockGrid }.Run();

    [Test]
    public void TransposeMatchesSpMVT_WideBlockGridTest()
        => new SparseTransposeTestJob { Type = SparseTransposeTestJob.TestType.TransposeMatchesSpMVT_WideBlockGrid }.Run();

    [Test]
    public void OperatorApplyTParityTest()
        => new SparseTransposeTestJob { Type = SparseTransposeTestJob.TestType.OperatorApplyTParity }.Run();

    [Test]
    public void SymmetricNoOpTest()
        => new SparseTransposeTestJob { Type = SparseTransposeTestJob.TestType.SymmetricNoOp }.Run();
}
