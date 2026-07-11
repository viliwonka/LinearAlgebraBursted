using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Milestone D: block-size-specialized (unrolled) sparse matvec kernels, b in {1,2,3,4,6}
// (bsrMatVecB{b} / bsrMatVecTB{b} / bsrMatVecSymB{b} in UnsafeOP.Sparse.fProxy.cs), dispatched
// from BSR.spMV / spMVT (SparseOP.fProxy.cs). Every case here proves ONE dispatch branch
// against the dense reference: build a fProxyBSR via the builder, expand with ToDense (already
// independently validated by fProxySparseBSRTests), and assert spMV/spMVT agree with
// Blas.dot on the dense expansion -- exactly the recipe fProxySparseBSRTests.RandomSpMV /
// RandomSpMVT already use, just swept across every specialized block size PLUS the two boundary
// cases that must still fall back to the general kernel: a non-specialized square size (b=5) and
// a rectangular block (BR != BC, which never dispatches to a specialized kernel regardless of
// size). Symmetric-storage cases (bsrMatVecSymB{b}) reuse the same recipe but on a GENUINELY
// symmetric matrix: ToBSRSymmetric now requires symmetric diagonal blocks (the upper triangle is
// stored implicitly as the transpose), so the diagonal blocks are built as M^T M. That makes the
// dense expansion truly symmetric, which lets the spMVT check compare against an INDEPENDENT
// transpose-matvec of the dense (DenseTransMatVec) rather than tautologically re-using spMV's own
// output -- genuinely exercising each bsrMatVecSymB{b} across b in {1,2,3,4,5,6}.
//
// Runs inside a [BurstCompile] IJob, matching every other Sparse test suite.
public class fProxySparseUnrollTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct SparseUnrollTestJob : IJob
    {
        public enum TestType
        {
            SpMV_B1, SpMV_B2, SpMV_B3, SpMV_B4, SpMV_B6, SpMV_B5Fallback,
            SpMVT_B1, SpMVT_B2, SpMVT_B3, SpMVT_B4, SpMVT_B6, SpMVT_B5Fallback,
            SpMV_Rectangular, SpMVT_Rectangular,
            Sym_B1, Sym_B2, Sym_B3, Sym_B4, Sym_B6, Sym_B5Fallback,
        }

        public TestType Type;

        // Matches fProxySparseBSRTests.Tol(): values live in [-1,1], dot products sum a handful
        // of products, so the absolute error stays well below this scaled threshold on both
        // precisions (float needs the looser bound, double is far tighter).
        static fProxy Tol() => /*+choose[1e-4f|1e-11]*/1e-4f/*-choose*/;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.SpMV_B1: CheckSquareSpMV(1, 41000u); break;
                case TestType.SpMV_B2: CheckSquareSpMV(2, 42000u); break;
                case TestType.SpMV_B3: CheckSquareSpMV(3, 43000u); break;
                case TestType.SpMV_B4: CheckSquareSpMV(4, 44000u); break;
                case TestType.SpMV_B6: CheckSquareSpMV(6, 46000u); break;
                case TestType.SpMV_B5Fallback: CheckSquareSpMV(5, 45000u); break;

                case TestType.SpMVT_B1: CheckSquareSpMVT(1, 51000u); break;
                case TestType.SpMVT_B2: CheckSquareSpMVT(2, 52000u); break;
                case TestType.SpMVT_B3: CheckSquareSpMVT(3, 53000u); break;
                case TestType.SpMVT_B4: CheckSquareSpMVT(4, 54000u); break;
                case TestType.SpMVT_B6: CheckSquareSpMVT(6, 56000u); break;
                case TestType.SpMVT_B5Fallback: CheckSquareSpMVT(5, 55000u); break;

                case TestType.SpMV_Rectangular: CheckRectangularSpMV(); break;
                case TestType.SpMVT_Rectangular: CheckRectangularSpMVT(); break;

                case TestType.Sym_B1: CheckSymmetric(1, 61000u); break;
                case TestType.Sym_B2: CheckSymmetric(2, 62000u); break;
                case TestType.Sym_B3: CheckSymmetric(3, 63000u); break;
                case TestType.Sym_B4: CheckSymmetric(4, 64000u); break;
                case TestType.Sym_B6: CheckSymmetric(6, 66000u); break;
                case TestType.Sym_B5Fallback: CheckSymmetric(5, 65000u); break;
            }
        }

        // ---- helpers ---------------------------------------------------------------------

        // Reference y = A^T*x computed directly off the dense expansion (independent of
        // Blas.trans), matching fProxySparseBSRTests.DenseTransMatVec.
        static void DenseTransMatVec(in fProxyMxN dense, in fProxyN x, ref fProxyN y)
        {
            for (int j = 0; j < dense.N_Cols; j++)
            {
                fProxy s = 0;
                for (int i = 0; i < dense.M_Rows; i++)
                    s += dense[i, j] * x[i];
                y[j] = s;
            }
        }

        // 4x4 block grid of b x b blocks (square, non-symmetric storage), six scattered stored
        // blocks (some rows/cols empty) so RowPtr/ColInd traversal isn't trivially degenerate.
        static fProxyBSR BuildRandomSquare(ref Arena arena, int b, uint seedBase)
        {
            var builder = arena.fProxyBSRBuilder(4, 4, b, b);
            builder.AddBlock(0, 0, arena.fProxyRandomMat(b, b, (fProxy)(-1f), (fProxy)1f, seedBase + 1u));
            builder.AddBlock(0, 2, arena.fProxyRandomMat(b, b, (fProxy)(-1f), (fProxy)1f, seedBase + 2u));
            builder.AddBlock(1, 1, arena.fProxyRandomMat(b, b, (fProxy)(-1f), (fProxy)1f, seedBase + 3u));
            builder.AddBlock(1, 3, arena.fProxyRandomMat(b, b, (fProxy)(-1f), (fProxy)1f, seedBase + 4u));
            builder.AddBlock(2, 0, arena.fProxyRandomMat(b, b, (fProxy)(-1f), (fProxy)1f, seedBase + 5u));
            builder.AddBlock(3, 3, arena.fProxyRandomMat(b, b, (fProxy)(-1f), (fProxy)1f, seedBase + 6u));
            return builder.ToBSR(ref arena);
        }

        // Symmetric b x b diagonal block D = M^T M: symmetric by construction (bit-exact, since
        // D[r,c] and D[c,r] sum the identical products in the identical order), so it satisfies
        // ToBSRSymmetric's diagonal-symmetry contract with zero asymmetry.
        static fProxyMxN SymDiagBlock(ref Arena arena, int b, uint seed)
        {
            var M = arena.fProxyRandomMat(b, b, (fProxy)(-1f), (fProxy)1f, seed);
            return Blas.dot(M, M, true);   // M^T M
        }

        // 4x4 block grid of b x b blocks, SYMMETRIC (lower-triangle-only) storage: a SYMMETRIC
        // diagonal block at every grid position (exercises bsrMatVecSymB{b}'s bi==bj branch) plus
        // three off-diagonal pairs, exercising bi!=bj scatter writes from multiple stored blocks
        // sharing a block-row. Diagonal blocks are M^T M so the represented matrix is genuinely
        // symmetric (required by ToBSRSymmetric); off-diagonal blocks stay arbitrary (their
        // transpose fills the upper triangle). dense is derived via A.ToDense (side-agnostic), so
        // only the stored (row, col) POSITION matters here, not which triangle "K" nominally lives in.
        static fProxyBSR BuildRandomSymmetric(ref Arena arena, int b, uint seedBase)
        {
            var builder = arena.fProxyBSRBuilder(4, 4, b, b);
            builder.AddBlock(0, 0, SymDiagBlock(ref arena, b, seedBase + 1u));
            builder.AddBlock(1, 1, SymDiagBlock(ref arena, b, seedBase + 2u));
            builder.AddBlock(2, 2, SymDiagBlock(ref arena, b, seedBase + 3u));
            builder.AddBlock(3, 3, SymDiagBlock(ref arena, b, seedBase + 4u));
            builder.AddBlock(1, 0, arena.fProxyRandomMat(b, b, (fProxy)(-1f), (fProxy)1f, seedBase + 5u));
            builder.AddBlock(3, 1, arena.fProxyRandomMat(b, b, (fProxy)(-1f), (fProxy)1f, seedBase + 6u));
            builder.AddBlock(3, 0, arena.fProxyRandomMat(b, b, (fProxy)(-1f), (fProxy)1f, seedBase + 7u));
            return builder.ToBSRSymmetric(ref arena);
        }

        // ---- square-block spMV / spMVT: covers b in {1,2,3,4,6} (specialized) and b=5
        // (must fall back to the general kernel; not in the specialized set) --------------------

        void CheckSquareSpMV(int b, uint seedBase)
        {
            var arena = new Arena(Allocator.Persistent);

            var A = BuildRandomSquare(ref arena, b, seedBase);
            var dense = A.ToDense(ref arena);
            var x = arena.fProxyRandomVec(A.N_Cols, (fProxy)(-1f), (fProxy)1f, seedBase + 900u);

            var y = arena.fProxyVec(A.M_Rows);
            BSR.spMV(in A, in x, ref y);
            var yRef = Blas.dot(dense, x);
            Assert.IsTrue(Analysis.isZero(y - yRef, Tol()));

            // allocating overload must agree with the ref-dest overload.
            var y2 = BSR.spMV(in A, in x);
            Assert.IsTrue(Analysis.isZero(y2 - yRef, Tol()));

            arena.Dispose();
        }

        void CheckSquareSpMVT(int b, uint seedBase)
        {
            var arena = new Arena(Allocator.Persistent);

            var A = BuildRandomSquare(ref arena, b, seedBase);
            var dense = A.ToDense(ref arena);
            var xt = arena.fProxyRandomVec(A.M_Rows, (fProxy)(-1f), (fProxy)1f, seedBase + 900u);

            var yt = arena.fProxyVec(A.N_Cols);
            BSR.spMVT(in A, in xt, ref yt);

            var ytRef = arena.fProxyVec(A.N_Cols);
            DenseTransMatVec(in dense, in xt, ref ytRef);
            Assert.IsTrue(Analysis.isZero(yt - ytRef, Tol()));

            var yt2 = BSR.spMVT(in A, in xt);
            Assert.IsTrue(Analysis.isZero(yt2 - ytRef, Tol()));

            arena.Dispose();
        }

        // ---- rectangular blocks (BR != BC): must ALWAYS route through the general kernel,
        // regardless of BR/BC individually matching a specialized size -----------------------

        void CheckRectangularSpMV()
        {
            var arena = new Arena(Allocator.Persistent);

            const int BR = 2, BC = 3; // BC would be a specialized size on its own -- BR != BC must still fall back.
            var builder = arena.fProxyBSRBuilder(3, 3, BR, BC);
            builder.AddBlock(0, 0, arena.fProxyRandomMat(BR, BC, (fProxy)(-1f), (fProxy)1f, 71001));
            builder.AddBlock(0, 2, arena.fProxyRandomMat(BR, BC, (fProxy)(-1f), (fProxy)1f, 71002));
            builder.AddBlock(1, 1, arena.fProxyRandomMat(BR, BC, (fProxy)(-1f), (fProxy)1f, 71003));
            builder.AddBlock(2, 0, arena.fProxyRandomMat(BR, BC, (fProxy)(-1f), (fProxy)1f, 71004));
            var A = builder.ToBSR(ref arena);
            var dense = A.ToDense(ref arena);

            var x = arena.fProxyRandomVec(A.N_Cols, (fProxy)(-1f), (fProxy)1f, 71100);
            var y = arena.fProxyVec(A.M_Rows);
            BSR.spMV(in A, in x, ref y);
            Assert.IsTrue(Analysis.isZero(y - Blas.dot(dense, x), Tol()));

            arena.Dispose();
        }

        void CheckRectangularSpMVT()
        {
            var arena = new Arena(Allocator.Persistent);

            const int BR = 2, BC = 3;
            var builder = arena.fProxyBSRBuilder(3, 3, BR, BC);
            builder.AddBlock(0, 0, arena.fProxyRandomMat(BR, BC, (fProxy)(-1f), (fProxy)1f, 72001));
            builder.AddBlock(0, 2, arena.fProxyRandomMat(BR, BC, (fProxy)(-1f), (fProxy)1f, 72002));
            builder.AddBlock(1, 1, arena.fProxyRandomMat(BR, BC, (fProxy)(-1f), (fProxy)1f, 72003));
            builder.AddBlock(2, 0, arena.fProxyRandomMat(BR, BC, (fProxy)(-1f), (fProxy)1f, 72004));
            var A = builder.ToBSR(ref arena);
            var dense = A.ToDense(ref arena);

            var xt = arena.fProxyRandomVec(A.M_Rows, (fProxy)(-1f), (fProxy)1f, 72100);
            var yt = arena.fProxyVec(A.N_Cols);
            BSR.spMVT(in A, in xt, ref yt);

            var ytRef = arena.fProxyVec(A.N_Cols);
            DenseTransMatVec(in dense, in xt, ref ytRef);
            Assert.IsTrue(Analysis.isZero(yt - ytRef, Tol()));

            arena.Dispose();
        }

        // ---- symmetric-storage spMV: covers b in {1,2,3,4,6} (specialized bsrMatVecSymB{b})
        // and b=5 (must fall back to the general bsrMatVecSym) --------------------------------

        void CheckSymmetric(int b, uint seedBase)
        {
            var arena = new Arena(Allocator.Persistent);

            var A = BuildRandomSymmetric(ref arena, b, seedBase);
            var dense = A.ToDense(ref arena);
            var x = arena.fProxyRandomVec(A.N_Cols, (fProxy)(-1f), (fProxy)1f, seedBase + 900u);

            var y = arena.fProxyVec(A.M_Rows);
            BSR.spMV(in A, in x, ref y);
            var yRef = Blas.dot(dense, x);
            Assert.IsTrue(Analysis.isZero(y - yRef, Tol()));

            // spMVT on symmetric storage forwards to spMV (A == A^T). Because the matrix is now
            // GENUINELY symmetric, check spMVT against an INDEPENDENT transpose-matvec of the dense
            // expansion (not spMV's own output): a true cross-check that A^T*x is computed, which
            // for the symmetric A must equal A*x.
            var ytRef = arena.fProxyVec(A.N_Cols);
            DenseTransMatVec(in dense, in x, ref ytRef);
            var yt = arena.fProxyVec(A.N_Cols);
            BSR.spMVT(in A, in x, ref yt);
            Assert.IsTrue(Analysis.isZero(yt - ytRef, Tol()));

            arena.Dispose();
        }
    }

    // ---- square-block spMV (specialized b in {1,2,3,4,6} + b=5 fallback) -----------------

    [Test]
    public void SpMV_B1_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.SpMV_B1 }.Run();

    [Test]
    public void SpMV_B2_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.SpMV_B2 }.Run();

    [Test]
    public void SpMV_B3_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.SpMV_B3 }.Run();

    [Test]
    public void SpMV_B4_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.SpMV_B4 }.Run();

    [Test]
    public void SpMV_B6_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.SpMV_B6 }.Run();

    // b=5 is NOT a specialized size -- proves spMV still routes to the general kernel correctly.
    [Test]
    public void SpMV_B5Fallback_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.SpMV_B5Fallback }.Run();

    // ---- square-block spMVT (specialized b in {1,2,3,4,6} + b=5 fallback) ----------------

    [Test]
    public void SpMVT_B1_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.SpMVT_B1 }.Run();

    [Test]
    public void SpMVT_B2_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.SpMVT_B2 }.Run();

    [Test]
    public void SpMVT_B3_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.SpMVT_B3 }.Run();

    [Test]
    public void SpMVT_B4_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.SpMVT_B4 }.Run();

    [Test]
    public void SpMVT_B6_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.SpMVT_B6 }.Run();

    [Test]
    public void SpMVT_B5Fallback_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.SpMVT_B5Fallback }.Run();

    // ---- rectangular blocks (BR != BC): dispatch boundary must always fall back -----------

    [Test]
    public void SpMV_Rectangular_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.SpMV_Rectangular }.Run();

    [Test]
    public void SpMVT_Rectangular_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.SpMVT_Rectangular }.Run();

    // ---- symmetric-storage spMV (specialized b in {1,2,3,4,6} + b=5 fallback) -------------

    [Test]
    public void Sym_B1_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.Sym_B1 }.Run();

    [Test]
    public void Sym_B2_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.Sym_B2 }.Run();

    [Test]
    public void Sym_B3_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.Sym_B3 }.Run();

    [Test]
    public void Sym_B4_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.Sym_B4 }.Run();

    [Test]
    public void Sym_B6_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.Sym_B6 }.Run();

    [Test]
    public void Sym_B5Fallback_Test() => new SparseUnrollTestJob { Type = SparseUnrollTestJob.TestType.Sym_B5Fallback }.Run();
}
