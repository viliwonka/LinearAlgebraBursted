using System;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Symmetric upper-block-triangle storage test suite for floatBSM (BSR). Milestone A: prove that
// a matrix stored as its upper block-triangle ONLY (floatBSM.Symmetric == true, built via
// floatBSMBuilder.ToBSMSymmetric) behaves identically to the SAME matrix stored FULLY (every
// block, incl. the explicit mirrored lower blocks, built via ToBSM). Every correctness case
// assembles the SAME logical SPD matrix in BOTH storage forms plus a dense reference, and
// cross-checks spMV / spMVT / ToDense / the solvers between them.
//
// The correctness cases run inside a [BurstCompile] IJob (same pattern as floatSparseBSMTests /
// floatSparseSolverTests). Guard / exception cases run on the managed test thread with
// Assert.Throws (NUnit's Assert.Throws cannot execute inside a Burst-compiled job).
public class floatSparseSymmetricTests
{
    [BurstCompile]
    public struct SparseSymmetricTestJob : IJob
    {
        public enum TestType
        {
            // ---- A1 cross-check correctness (full storage == symmetric storage) ----
            CrossCheck_BR1_Sparse,
            CrossCheck_BR1_Dense,
            CrossCheck_BR3_Sparse,
            CrossCheck_BR3_Dense,

            // ---- A1 edge cases ----
            CrossCheck_DiagonalOnly,   // no off-diagonal blocks -> bsmMatVecSym's bi!=bj branch never taken
            CrossCheck_SingleBlock,    // 1x1 block grid: trivially symmetric == full (no lower triangle)
            CrossCheck_Empty,          // zero-triplet symmetric BSM: round-trips to zero dense / zero matvec

            // ---- A2 solver wiring (symmetric BSM feeds cg / minres / block-Jacobi pcg) ----
            CgSymMatchesFull,
            MinresSymMatchesFull,
            PcgBlockJacobiSymMatchesFull,
        }

        public TestType Type;

        // Reconstruction / matvec cross-check tolerance. The symmetric and full storage forms hold
        // byte-identical block values, so ToDense agrees exactly and spMV/spMVT differ only by the
        // block-traversal ORDER of otherwise-identical products -- values live in a bounded range,
        // so the absolute error stays well below this scaled threshold on both precisions (float
        // needs the looser bound, double is far tighter). Matches floatSparseBSMTests.Tol().
        static float Tol() => 1e-4f;

        // Looser threshold for the A2 solver cross-checks: comparing TWO independently-converged
        // iterative solutions (each accurate only to about Consts.floatSqrtEps*scale, not machine
        // epsilon). Matches floatSparseSolverTests.LooseTol().
        static float LooseTol() => 1e-2f;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.CrossCheck_BR1_Sparse: CrossCheck_BR1_Sparse(); break;
                case TestType.CrossCheck_BR1_Dense: CrossCheck_BR1_Dense(); break;
                case TestType.CrossCheck_BR3_Sparse: CrossCheck_BR3_Sparse(); break;
                case TestType.CrossCheck_BR3_Dense: CrossCheck_BR3_Dense(); break;

                case TestType.CrossCheck_DiagonalOnly: CrossCheck_DiagonalOnly(); break;
                case TestType.CrossCheck_SingleBlock: CrossCheck_SingleBlock(); break;
                case TestType.CrossCheck_Empty: CrossCheck_Empty(); break;

                case TestType.CgSymMatchesFull: CgSymMatchesFull(); break;
                case TestType.MinresSymMatchesFull: MinresSymMatchesFull(); break;
                case TestType.PcgBlockJacobiSymMatchesFull: PcgBlockJacobiSymMatchesFull(); break;
            }
        }

        // ================================================================================
        // helpers
        // ================================================================================

        static void AssertVecEq(in floatN a, in floatN b, float tol)
        {
            Assert.IsTrue(Analysis_OP.isZero(a - b, tol));
        }

        static void AssertMatEq(in floatMxN a, in floatMxN b, float tol)
        {
            Assert.IsTrue(a.M_Rows == b.M_Rows);
            Assert.IsTrue(a.N_Cols == b.N_Cols);
            for (int i = 0; i < a.M_Rows; i++)
                for (int j = 0; j < a.N_Cols; j++)
                    Assert.IsTrue(math.abs(a[i, j] - b[i, j]) < tol);
        }

        // A[i,j] == A[j,i] for all i,j (the assembled dense reference must be genuinely symmetric).
        static void AssertSymmetric(in floatMxN A, float tol)
        {
            Assert.IsTrue(A.M_Rows == A.N_Cols);
            for (int i = 0; i < A.M_Rows; i++)
                for (int j = 0; j < A.N_Cols; j++)
                    Assert.IsTrue(math.abs(A[i, j] - A[j, i]) < tol);
        }

        // Diagonal block D_i = M_i^T M_i + strong*I (SPD / diagonally dominant on its own). Added to
        // BOTH the full and symmetric builders (identical block) and scattered into the dense
        // reference. Builders are passed by value: AddBlock mutates through the builder's shared
        // heap _state pointer, so growth/appends are visible to the caller's copy too.
        static void AddDiag(ref Arena arena, int i, int BR, float strong, uint seed,
                            floatBSMBuilder full, floatBSMBuilder sym, ref floatMxN dense)
        {
            var Mi = arena.floatRandomMat(BR, BR, (float)(-1f), (float)1f, seed);
            var Di = Linear_OP.dot(Mi, Mi, true);   // M_i^T M_i, symmetric PSD
            for (int d = 0; d < BR; d++)
                Di[d, d] += strong;

            full.AddBlock(i, i, in Di);
            sym.AddBlock(i, i, in Di);

            int b = i * BR;
            for (int r = 0; r < BR; r++)
                for (int c = 0; c < BR; c++)
                    dense[b + r, b + c] = Di[r, c];
        }

        // Off-diagonal pair (bi < bj), small random block K:
        //   - full storage: AddBlock(bi,bj,K) AND AddBlock(bj,bi,K^T) (explicit mirror)
        //   - symmetric storage: AddBlock(bi,bj,K) ONLY (mirror is implicit)
        //   - dense reference: both dense[bi,bj]=K and dense[bj,bi]=K^T so it stays truly symmetric
        // offScale is kept small so the assembled matrix stays SPD by diagonal dominance.
        static void AddOffDiag(ref Arena arena, int bi, int bj, int BR, uint seed,
                               floatBSMBuilder full, floatBSMBuilder sym, ref floatMxN dense)
        {
            var block = arena.floatRandomMat(BR, BR, (float)(-0.2f), (float)0.2f, seed);

            full.AddBlock(bi, bj, in block);

            var blockT = arena.floatMat(BR, BR);
            for (int r = 0; r < BR; r++)
                for (int c = 0; c < BR; c++)
                    blockT[r, c] = block[c, r];
            full.AddBlock(bj, bi, in blockT);

            sym.AddBlock(bi, bj, in block);

            int ib = bi * BR, jb = bj * BR;
            for (int r = 0; r < BR; r++)
                for (int c = 0; c < BR; c++)
                {
                    dense[ib + r, jb + c] = block[r, c];   // upper
                    dense[jb + c, ib + r] = block[r, c];   // mirror (== blockT read back)
                }
        }

        // The shared A1 assertion battery: given the SAME matrix as a symmetric BSM, a full BSM, and
        // a dense reference, prove every access path agrees.
        static void CrossCheck(ref Arena arena, in floatBSM sym, in floatBSM full,
                               in floatMxN dense, uint seedBase)
        {
            // Storage-mode flags.
            Assert.IsTrue(sym.Symmetric);
            Assert.IsTrue(!full.Symmetric);
            // Symmetric storage never holds MORE blocks than full storage (fewer whenever there is
            // any off-diagonal block; equal for diagonal-only / single-block).
            Assert.IsTrue(sym.Nnzb <= full.Nnzb);

            int n = sym.N_Cols;

            // ---- spMV direction ----
            var x = arena.floatRandomVec(n, (float)(-1f), (float)1f, seedBase);

            var ySym  = Sparse_OP.spMV(in sym, in x);
            var yFull = Sparse_OP.spMV(in full, in x);
            AssertVecEq(in ySym, in yFull, Tol());                 // sym spMV == full spMV

            // full is genuinely symmetric: its transpose traversal equals its forward traversal
            // (independent of the Symmetric-flag shortcut, which full does NOT use).
            var yFullT = Sparse_OP.spMVT(in full, in x);
            AssertVecEq(in yFullT, in yFull, Tol());

            // ---- spMVT direction (separate random x) ----
            var xt = arena.floatRandomVec(n, (float)(-1f), (float)1f, seedBase + 1u);

            var ySymT  = Sparse_OP.spMVT(in sym, in xt);   // sym's spMVT forwards to spMV (A==A^T)
            var yFullT2 = Sparse_OP.spMVT(in full, in xt);  // full's genuine transpose traversal
            AssertVecEq(in ySymT, in yFullT2, Tol());

            // ---- ToDense ----
            var dSym  = sym.ToDense(ref arena);   // mirrors upper blocks into the lower triangle
            var dFull = full.ToDense(ref arena);
            AssertMatEq(in dSym, in dFull, Tol());
            AssertMatEq(in dSym, in dense, Tol());   // both agree with the tracked reference
            AssertSymmetric(in dSym, Tol());         // and the shared dense result is symmetric
        }

        // Assemble the SAME SPD matrix as (symmetric BSM, full BSM, dense reference). The off-diagonal
        // sparsity pattern is written as a straight-line sequence of AddOffDiag calls (NOT a loop over
        // a collection) so the whole method is Burst-compatible.
        static void BuildSpdPair_BR1_Sparse(ref Arena arena, out floatBSM sym, out floatBSM full, out floatMxN dense)
        {
            const int BR = 1, nb = 5;
            int dim = BR * nb;
            float strong = (float)dim;
            dense = arena.floatMat(dim, dim);

            var f = arena.floatBSMBuilder(nb, nb, BR, BR, nb + 2 * 2);
            var s = arena.floatBSMBuilder(nb, nb, BR, BR, nb + 2);

            AddDiag(ref arena, 0, BR, strong, 10000u, f, s, ref dense);
            AddDiag(ref arena, 1, BR, strong, 10001u, f, s, ref dense);
            AddDiag(ref arena, 2, BR, strong, 10002u, f, s, ref dense);
            AddDiag(ref arena, 3, BR, strong, 10003u, f, s, ref dense);
            AddDiag(ref arena, 4, BR, strong, 10004u, f, s, ref dense);

            // sparse-ish: two off-diagonal pairs only
            AddOffDiag(ref arena, 0, 1, BR, 11000u, f, s, ref dense);
            AddOffDiag(ref arena, 1, 3, BR, 11001u, f, s, ref dense);

            sym  = s.ToBSMSymmetric(ref arena);
            full = f.ToBSM(ref arena);
        }

        static void BuildSpdPair_BR1_Dense(ref Arena arena, out floatBSM sym, out floatBSM full, out floatMxN dense)
        {
            const int BR = 1, nb = 5;
            int dim = BR * nb;
            float strong = (float)dim;
            dense = arena.floatMat(dim, dim);

            var f = arena.floatBSMBuilder(nb, nb, BR, BR, nb + 2 * 10);
            var s = arena.floatBSMBuilder(nb, nb, BR, BR, nb + 10);

            AddDiag(ref arena, 0, BR, strong, 12000u, f, s, ref dense);
            AddDiag(ref arena, 1, BR, strong, 12001u, f, s, ref dense);
            AddDiag(ref arena, 2, BR, strong, 12002u, f, s, ref dense);
            AddDiag(ref arena, 3, BR, strong, 12003u, f, s, ref dense);
            AddDiag(ref arena, 4, BR, strong, 12004u, f, s, ref dense);

            // denser: every upper off-diagonal pair on the 5x5 block grid
            AddOffDiag(ref arena, 0, 1, BR, 13000u, f, s, ref dense);
            AddOffDiag(ref arena, 0, 2, BR, 13001u, f, s, ref dense);
            AddOffDiag(ref arena, 0, 3, BR, 13002u, f, s, ref dense);
            AddOffDiag(ref arena, 0, 4, BR, 13003u, f, s, ref dense);
            AddOffDiag(ref arena, 1, 2, BR, 13004u, f, s, ref dense);
            AddOffDiag(ref arena, 1, 3, BR, 13005u, f, s, ref dense);
            AddOffDiag(ref arena, 1, 4, BR, 13006u, f, s, ref dense);
            AddOffDiag(ref arena, 2, 3, BR, 13007u, f, s, ref dense);
            AddOffDiag(ref arena, 2, 4, BR, 13008u, f, s, ref dense);
            AddOffDiag(ref arena, 3, 4, BR, 13009u, f, s, ref dense);

            sym  = s.ToBSMSymmetric(ref arena);
            full = f.ToBSM(ref arena);
        }

        // BR=3 exercises genuine block interior transposes in bsmMatVecSym (the K^T * x_i mirror).
        static void BuildSpdPair_BR3_Sparse(ref Arena arena, out floatBSM sym, out floatBSM full, out floatMxN dense)
        {
            const int BR = 3, nb = 4;
            int dim = BR * nb;
            float strong = (float)dim;
            dense = arena.floatMat(dim, dim);

            var f = arena.floatBSMBuilder(nb, nb, BR, BR, nb + 2 * 3);
            var s = arena.floatBSMBuilder(nb, nb, BR, BR, nb + 3);

            AddDiag(ref arena, 0, BR, strong, 20000u, f, s, ref dense);
            AddDiag(ref arena, 1, BR, strong, 20001u, f, s, ref dense);
            AddDiag(ref arena, 2, BR, strong, 20002u, f, s, ref dense);
            AddDiag(ref arena, 3, BR, strong, 20003u, f, s, ref dense);

            // sparse-ish: block-tridiagonal off-diagonal pattern
            AddOffDiag(ref arena, 0, 1, BR, 21000u, f, s, ref dense);
            AddOffDiag(ref arena, 1, 2, BR, 21001u, f, s, ref dense);
            AddOffDiag(ref arena, 2, 3, BR, 21002u, f, s, ref dense);

            sym  = s.ToBSMSymmetric(ref arena);
            full = f.ToBSM(ref arena);
        }

        static void BuildSpdPair_BR3_Dense(ref Arena arena, out floatBSM sym, out floatBSM full, out floatMxN dense)
        {
            const int BR = 3, nb = 4;
            int dim = BR * nb;
            float strong = (float)dim;
            dense = arena.floatMat(dim, dim);

            var f = arena.floatBSMBuilder(nb, nb, BR, BR, nb + 2 * 6);
            var s = arena.floatBSMBuilder(nb, nb, BR, BR, nb + 6);

            AddDiag(ref arena, 0, BR, strong, 22000u, f, s, ref dense);
            AddDiag(ref arena, 1, BR, strong, 22001u, f, s, ref dense);
            AddDiag(ref arena, 2, BR, strong, 22002u, f, s, ref dense);
            AddDiag(ref arena, 3, BR, strong, 22003u, f, s, ref dense);

            // denser: every upper off-diagonal pair on the 4x4 block grid
            AddOffDiag(ref arena, 0, 1, BR, 23000u, f, s, ref dense);
            AddOffDiag(ref arena, 0, 2, BR, 23001u, f, s, ref dense);
            AddOffDiag(ref arena, 0, 3, BR, 23002u, f, s, ref dense);
            AddOffDiag(ref arena, 1, 2, BR, 23003u, f, s, ref dense);
            AddOffDiag(ref arena, 1, 3, BR, 23004u, f, s, ref dense);
            AddOffDiag(ref arena, 2, 3, BR, 23005u, f, s, ref dense);

            sym  = s.ToBSMSymmetric(ref arena);
            full = f.ToBSM(ref arena);
        }

        // ================================================================================
        // A1 correctness cases
        // ================================================================================

        void CrossCheck_BR1_Sparse()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildSpdPair_BR1_Sparse(ref arena, out var sym, out var full, out var dense);
            // off-diagonals present -> symmetric storage strictly smaller.
            Assert.IsTrue(sym.Nnzb < full.Nnzb);
            CrossCheck(ref arena, in sym, in full, in dense, 14000u);
            arena.Dispose();
        }

        void CrossCheck_BR1_Dense()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildSpdPair_BR1_Dense(ref arena, out var sym, out var full, out var dense);
            Assert.IsTrue(sym.Nnzb < full.Nnzb);
            CrossCheck(ref arena, in sym, in full, in dense, 14100u);
            arena.Dispose();
        }

        void CrossCheck_BR3_Sparse()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildSpdPair_BR3_Sparse(ref arena, out var sym, out var full, out var dense);
            Assert.IsTrue(sym.Nnzb < full.Nnzb);
            CrossCheck(ref arena, in sym, in full, in dense, 24000u);
            arena.Dispose();
        }

        void CrossCheck_BR3_Dense()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildSpdPair_BR3_Dense(ref arena, out var sym, out var full, out var dense);
            Assert.IsTrue(sym.Nnzb < full.Nnzb);
            CrossCheck(ref arena, in sym, in full, in dense, 24100u);
            arena.Dispose();
        }

        // ---- edge: diagonal-only symmetric BSM ----
        //
        // ONLY diagonal blocks are populated -> in bsmMatVecSym the `if (bi != bj)` mirrored-write
        // branch is NEVER taken. This isolates the diagonal path (the branch that would silently
        // double-count a diagonal block if it were mistakenly treated as off-diagonal). sym and full
        // are structurally identical here (no lower triangle to omit), so Nnzb is equal.
        void CrossCheck_DiagonalOnly()
        {
            var arena = new Arena(Allocator.Persistent);

            const int BR = 3, nb = 3;
            int dim = BR * nb;
            float strong = (float)dim;
            var dense = arena.floatMat(dim, dim);

            var f = arena.floatBSMBuilder(nb, nb, BR, BR, nb);
            var s = arena.floatBSMBuilder(nb, nb, BR, BR, nb);

            AddDiag(ref arena, 0, BR, strong, 30000u, f, s, ref dense);
            AddDiag(ref arena, 1, BR, strong, 30001u, f, s, ref dense);
            AddDiag(ref arena, 2, BR, strong, 30002u, f, s, ref dense);

            var sym  = s.ToBSMSymmetric(ref arena);
            var full = f.ToBSM(ref arena);

            Assert.IsTrue(sym.Nnzb == nb);          // exactly the diagonal blocks
            Assert.IsTrue(sym.Nnzb == full.Nnzb);   // no lower triangle to omit

            CrossCheck(ref arena, in sym, in full, in dense, 31000u);

            arena.Dispose();
        }

        // ---- edge: single 1x1 block grid ----
        //
        // BlockRows==BlockCols==1: there is no lower triangle at all, so symmetric == full trivially.
        // A genuine 3x3 diagonal block (D = M^T M + strong*I is symmetric) keeps the case non-degenerate.
        void CrossCheck_SingleBlock()
        {
            var arena = new Arena(Allocator.Persistent);

            const int BR = 3, nb = 1;
            int dim = BR * nb;
            float strong = (float)dim;
            var dense = arena.floatMat(dim, dim);

            var f = arena.floatBSMBuilder(nb, nb, BR, BR, nb);
            var s = arena.floatBSMBuilder(nb, nb, BR, BR, nb);

            AddDiag(ref arena, 0, BR, strong, 40000u, f, s, ref dense);

            var sym  = s.ToBSMSymmetric(ref arena);
            var full = f.ToBSM(ref arena);

            Assert.IsTrue(sym.Nnzb == 1);
            Assert.IsTrue(sym.Nnzb == full.Nnzb);

            CrossCheck(ref arena, in sym, in full, in dense, 41000u);

            arena.Dispose();
        }

        // ---- edge: empty (zero-triplet) symmetric BSM ----
        //
        // A builder with a valid square block-grid shape (BR==BC, BlockRows==BlockCols so
        // ToBSMSymmetric's guard passes) but ZERO triplets ToBSMSymmetric's to a valid empty BSM
        // (Nnzb == 0): every block-row's RowPtr range is empty, so bsmMatVecSym never dereferences
        // the zero-length ColInd/Values buffers. ToDense must produce the all-zero matrix and
        // spMV/spMVT the zero vector for any x. Mirrors floatSparseBSMTests.EmptyBSMRoundTrip, but
        // there is no full-storage twin to compare against -- an empty matrix is checked against
        // zero directly. spMVT specifically exercises the Symmetric-forwarding spMVT->spMV path on
        // an empty matrix (distinct from the full BSM's own empty-transpose loop).
        void CrossCheck_Empty()
        {
            var arena = new Arena(Allocator.Persistent);

            const int BR = 2, nb = 3;                 // 3x3 block grid of 2x2 blocks -> 6x6 dense
            int dim = BR * nb;
            var builder = arena.floatBSMBuilder(nb, nb, BR, BR); // never AddBlock/AddValue
            var sym = builder.ToBSMSymmetric(ref arena);

            Assert.IsTrue(sym.Symmetric);
            Assert.IsTrue(sym.Nnzb == 0);
            Assert.IsTrue(sym.M_Rows == dim);
            Assert.IsTrue(sym.N_Cols == dim);

            // ToDense of an empty symmetric BSM == the all-zero matrix of the right dims.
            var dense = sym.ToDense(ref arena);
            var zero = arena.floatMat(dim, dim);
            AssertMatEq(in dense, in zero, Tol());

            // spMV of an empty BSM == the zero vector, for a random nonzero x.
            var x = arena.floatRandomVec(dim, (float)(-1f), (float)1f, 42000u);
            var y = Sparse_OP.spMV(in sym, in x);
            Assert.IsTrue(Analysis_OP.isZero(y, Tol()));

            // spMVT too -- exercises the Symmetric spMVT->spMV forwarding on an empty matrix.
            var xt = arena.floatRandomVec(dim, (float)(-1f), (float)1f, 42001u);
            var yt = Sparse_OP.spMVT(in sym, in xt);
            Assert.IsTrue(Analysis_OP.isZero(yt, Tol()));

            arena.Dispose();
        }

        // ================================================================================
        // A2 solver-wiring cases (BR=3 dense multi-block SPD system)
        // ================================================================================

        // maxIterations = 4*dim: CG's finite-termination (<=dim iters) only holds in EXACT
        // arithmetic, floating point on an ill-conditioned system can need more (same 4n cap the
        // sibling solver tests use).

        void CgSymMatchesFull()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildSpdPair_BR3_Dense(ref arena, out var sym, out var full, out var dense);
            int dim = sym.M_Rows;
            var b = arena.floatRandomVec(dim, (float)(-1f), (float)1f, 50000u);

            var xSym = arena.floatVec(dim);
            bool okSym = Solvers.cg(in sym, in b, ref xSym, 4 * dim, Consts.floatSqrtEps);
            Assert.IsTrue(okSym);

            var xFull = arena.floatVec(dim);
            bool okFull = Solvers.cg(in full, in b, ref xFull, 4 * dim, Consts.floatSqrtEps);
            Assert.IsTrue(okFull);

            var xDense = arena.floatVec(dim);
            bool okDense = Solvers.cg(in dense, in b, ref xDense, 4 * dim, Consts.floatSqrtEps);
            Assert.IsTrue(okDense);

            AssertVecEq(in xSym, in xFull, LooseTol());
            AssertVecEq(in xSym, in xDense, LooseTol());

            // A*x ~= b for the symmetric solve too.
            var Ax = Sparse_OP.spMV(in sym, in xSym);
            AssertVecEq(in Ax, in b, LooseTol());

            arena.Dispose();
        }

        void MinresSymMatchesFull()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildSpdPair_BR3_Dense(ref arena, out var sym, out var full, out var dense);
            int dim = sym.M_Rows;
            var b = arena.floatRandomVec(dim, (float)(-1f), (float)1f, 51000u);

            var xSym = arena.floatVec(dim);
            bool okSym = Solvers.minres(in sym, in b, ref xSym, 4 * dim, Consts.floatSqrtEps);
            Assert.IsTrue(okSym);

            var xFull = arena.floatVec(dim);
            bool okFull = Solvers.minres(in full, in b, ref xFull, 4 * dim, Consts.floatSqrtEps);
            Assert.IsTrue(okFull);

            var xDense = arena.floatVec(dim);
            bool okDense = Solvers.cg(in dense, in b, ref xDense, 4 * dim, Consts.floatSqrtEps);
            Assert.IsTrue(okDense);

            AssertVecEq(in xSym, in xFull, LooseTol());
            AssertVecEq(in xSym, in xDense, LooseTol());

            arena.Dispose();
        }

        // Block-Jacobi on a symmetric BSM: the diagonal block (col==row) is the FIRST stored entry
        // in a symmetric row range (upper-triangle storage => ColInd ascending, >= row), so the
        // preconditioner's diagonal lookup must succeed WITHOUT special-casing -- this is the
        // specific "not broken by symmetric storage" claim Milestone A asked to verify.
        void PcgBlockJacobiSymMatchesFull()
        {
            var arena = new Arena(Allocator.Persistent);
            BuildSpdPair_BR3_Dense(ref arena, out var sym, out var full, out var dense);
            int dim = sym.M_Rows;
            var b = arena.floatRandomVec(dim, (float)(-1f), (float)1f, 52000u);

            // Both preconditioners must construct without throwing (diagonal blocks ARE present in
            // symmetric upper storage).
            var mSym  = arena.floatBlockJacobi(in sym);
            var mFull = arena.floatBlockJacobi(in full);

            var xPcgSym = arena.floatVec(dim);
            bool okSym = Solvers.pcg(in sym, in mSym, in b, ref xPcgSym, 4 * dim, Consts.floatSqrtEps);
            Assert.IsTrue(okSym);

            var xPcgFull = arena.floatVec(dim);
            bool okFull = Solvers.pcg(in full, in mFull, in b, ref xPcgFull, 4 * dim, Consts.floatSqrtEps);
            Assert.IsTrue(okFull);

            AssertVecEq(in xPcgSym, in xPcgFull, LooseTol());

            // And the preconditioned solve satisfies A*x ~= b.
            var Ax = Sparse_OP.spMV(in sym, in xPcgSym);
            AssertVecEq(in Ax, in b, LooseTol());

            arena.Dispose();
        }
    }

    // ================================================================================
    // A1 correctness entry points (Burst)
    // ================================================================================

    [Test]
    public void CrossCheck_BR1_SparseTest()
        => new SparseSymmetricTestJob { Type = SparseSymmetricTestJob.TestType.CrossCheck_BR1_Sparse }.Run();

    [Test]
    public void CrossCheck_BR1_DenseTest()
        => new SparseSymmetricTestJob { Type = SparseSymmetricTestJob.TestType.CrossCheck_BR1_Dense }.Run();

    [Test]
    public void CrossCheck_BR3_SparseTest()
        => new SparseSymmetricTestJob { Type = SparseSymmetricTestJob.TestType.CrossCheck_BR3_Sparse }.Run();

    [Test]
    public void CrossCheck_BR3_DenseTest()
        => new SparseSymmetricTestJob { Type = SparseSymmetricTestJob.TestType.CrossCheck_BR3_Dense }.Run();

    [Test]
    public void CrossCheck_DiagonalOnlyTest()
        => new SparseSymmetricTestJob { Type = SparseSymmetricTestJob.TestType.CrossCheck_DiagonalOnly }.Run();

    [Test]
    public void CrossCheck_SingleBlockTest()
        => new SparseSymmetricTestJob { Type = SparseSymmetricTestJob.TestType.CrossCheck_SingleBlock }.Run();

    [Test]
    public void CrossCheck_EmptyTest()
        => new SparseSymmetricTestJob { Type = SparseSymmetricTestJob.TestType.CrossCheck_Empty }.Run();

    // ================================================================================
    // A2 solver-wiring entry points (Burst)
    // ================================================================================

    [Test]
    public void CgSymMatchesFullTest()
        => new SparseSymmetricTestJob { Type = SparseSymmetricTestJob.TestType.CgSymMatchesFull }.Run();

    [Test]
    public void MinresSymMatchesFullTest()
        => new SparseSymmetricTestJob { Type = SparseSymmetricTestJob.TestType.MinresSymMatchesFull }.Run();

    [Test]
    public void PcgBlockJacobiSymMatchesFullTest()
        => new SparseSymmetricTestJob { Type = SparseSymmetricTestJob.TestType.PcgBlockJacobiSymMatchesFull }.Run();

    // ================================================================================
    // A1 guard / exception cases (managed thread; Assert.Throws can't run inside Burst)
    // ================================================================================

    // ToBSMSymmetric rejects a lower-triangle triplet (blockCol < blockRow): building a symmetric
    // matrix must only add blocks at (br, bc) with bc >= br, otherwise the caller has (probably)
    // a bug and we refuse to silently fold it into the transpose position.
    [Test]
    public void ToBSMSymmetric_LowerTriangleTriplet_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            const int BR = 2;
            var builder = arena.floatBSMBuilder(2, 2, BR, BR, 1);
            builder.AddBlock(1, 0, arena.floatRandomMat(BR, BR, (float)(-1f), (float)1f, 60001)); // bc < br
            Assert.Throws<ArgumentException>(() => builder.ToBSMSymmetric(ref arena));
        }
        finally { arena.Dispose(); }
    }

    // ToBSMSymmetric rejects a NON-SYMMETRIC diagonal block: upper-block storage represents the
    // implicit lower block (bj,bi) as block(bi,bj)^T, so the matrix is symmetric only if each
    // diagonal block is -- and spMVT forwards to spMV assuming A==A^T, so a non-symmetric diagonal
    // block would silently make spMVT return A*x. The build must refuse it (same stance as the
    // lower-triangle guard). Uses a 1x1 block grid so a single non-symmetric diagonal block is the
    // only thing under test.
    [Test]
    public void ToBSMSymmetric_NonSymmetricDiagonalBlock_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            const int BR = 2;
            var blk = arena.floatMat(BR, BR);
            blk[0, 0] = (float)1; blk[0, 1] = (float)2;
            blk[1, 0] = (float)3; blk[1, 1] = (float)4;   // blk[0,1]=2 != blk[1,0]=3 -> not symmetric
            var builder = arena.floatBSMBuilder(1, 1, BR, BR, 1);
            builder.AddBlock(0, 0, in blk);
            Assert.Throws<ArgumentException>(() => builder.ToBSMSymmetric(ref arena));
        }
        finally { arena.Dispose(); }
    }

    // A SYMMETRIC diagonal block is accepted (the guard's tolerance does not false-positive on a
    // genuinely symmetric block): companion to the throw case above.
    [Test]
    public void ToBSMSymmetric_SymmetricDiagonalBlock_Accepted()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            const int BR = 2;
            var blk = arena.floatMat(BR, BR);
            blk[0, 0] = (float)1; blk[0, 1] = (float)2;
            blk[1, 0] = (float)2; blk[1, 1] = (float)4;   // symmetric (2 == 2)
            var builder = arena.floatBSMBuilder(1, 1, BR, BR, 1);
            builder.AddBlock(0, 0, in blk);
            var sym = builder.ToBSMSymmetric(ref arena);    // must NOT throw
            Assert.IsTrue(sym.Symmetric);
        }
        finally { arena.Dispose(); }
    }

    // ToBSMSymmetric rejects rectangular blocks (BR != BC) -- this guard fires BEFORE the triplet
    // scan, so it throws even with zero triplets.
    [Test]
    public void ToBSMSymmetric_NonSquareBlock_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var builder = arena.floatBSMBuilder(2, 2, 3, 2); // BR=3 != BC=2, but square block grid
            Assert.Throws<ArgumentException>(() => builder.ToBSMSymmetric(ref arena));
        }
        finally { arena.Dispose(); }
    }

    // ToBSMSymmetric rejects a non-square block grid (BlockRows != BlockCols) even with BR == BC.
    [Test]
    public void ToBSMSymmetric_NonSquareGrid_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var builder = arena.floatBSMBuilder(2, 3, 3, 3); // BlockRows=2 != BlockCols=3, BR==BC
            Assert.Throws<ArgumentException>(() => builder.ToBSMSymmetric(ref arena));
        }
        finally { arena.Dispose(); }
    }

    // The floatBSM CONSTRUCTOR's OWN symmetric guard (a different code path than the builder's):
    // rectangular blocks (BR != BC) are rejected up front.
    [Test]
    public void Ctor_SymmetricNonSquareBlock_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            // 2x2 block grid, BR=3 != BC=2 -> symmetric storage forbidden.
            Assert.Throws<ArgumentException>(() => arena.floatBSM(2, 2, 3, 2, 1, symmetric: true));
        }
        finally { arena.Dispose(); }
    }

    // Same constructor guard: a non-square block grid (BlockRows != BlockCols) is rejected even with
    // BR == BC.
    [Test]
    public void Ctor_SymmetricNonSquareGrid_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            // 2x3 block grid, BR==BC==3 -> symmetric storage forbidden (grid not square).
            Assert.Throws<ArgumentException>(() => arena.floatBSM(2, 3, 3, 3, 1, symmetric: true));
        }
        finally { arena.Dispose(); }
    }
}
