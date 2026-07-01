using System;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Phase-1 Block Sparse Matrix (BSM / block-CSR) test suite. Every correctness case validates
// the sparse op against the DENSE reference: build a floatBSM via the builder, expand with
// ToDense, and compare spMV/spMVT against Linear_OP.dot on the dense expansion. Property /
// reconstruction checks are preferred over hard-coded element values, except the small
// hand-computable cases where an exact expected matrix is more convincing.
//
// The correctness cases run inside a [BurstCompile] IJob (same pattern as
// ConjugateGradientTests). The guard / exception cases run on the managed test thread with
// Assert.Throws (same pattern as BidiagWorkspaceTests / ClampTests) -- NUnit's Assert.Throws
// cannot execute inside a Burst-compiled job.
public class floatSparseBSMTests
{
    [BurstCompile]
    public struct SparseBSMTestJob : IJob
    {
        public enum TestType
        {
            HandBuiltSmall,
            RandomSpMV,
            RandomSpMVT,
            RectangularBlocks,
            DuplicateSummation,
            OutOfOrderTriplets,
            OneByOneBlocks,
            GrowthThenDispose,
            ClearThenReallocate,
            EmptyBSMRoundTrip,
        }

        public TestType Type;

        // Reconstruction / matvec error tolerance. Values live in [-1,1] and the dot products
        // sum O(N_Cols) products, so the absolute error stays well below this scaled threshold
        // on both precisions (float needs the looser bound, double is far tighter).
        static float Tol() => 1e-4f;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.HandBuiltSmall: HandBuiltSmall(); break;
                case TestType.RandomSpMV: RandomSpMV(); break;
                case TestType.RandomSpMVT: RandomSpMVT(); break;
                case TestType.RectangularBlocks: RectangularBlocks(); break;
                case TestType.DuplicateSummation: DuplicateSummation(); break;
                case TestType.OutOfOrderTriplets: OutOfOrderTriplets(); break;
                case TestType.OneByOneBlocks: OneByOneBlocks(); break;
                case TestType.GrowthThenDispose: GrowthThenDispose(); break;
                case TestType.ClearThenReallocate: ClearThenReallocate(); break;
                case TestType.EmptyBSMRoundTrip: EmptyBSMRoundTrip(); break;
            }
        }

        // ---- helpers ---------------------------------------------------------------------

        // Independent (does NOT reuse ToDense's indexing) scatter of a BR x BC block into a
        // zero-initialized dense matrix at block position (br, bc).
        static void Scatter(ref floatMxN dense, int br, int bc, in floatMxN block)
        {
            int BR = block.M_Rows, BC = block.N_Cols;
            for (int r = 0; r < BR; r++)
                for (int c = 0; c < BC; c++)
                    dense[br * BR + r, bc * BC + c] = block[r, c];
        }

        // Reference y = Aᵀ·x computed directly from the dense expansion's elements.
        //
        // NOTE: floatBSM.ToDense used to take `in Arena`, which forced a defensive copy of the
        // arena before its internal (mutating) arena.floatMat(...) call -- the returned matrix's
        // _arenaPtr captured the address of that dead temporary, so Linear_OP.trans(dense)
        // (which allocates via dense.tempfloatMat) dereferenced a dangling pointer and threw
        // "allocator handle is not valid" under Burst. ToDense/ToBSM now take `ref Arena`
        // (see ToDense_TransposeReference_Works, formerly the Ignored bug-repro test) so that
        // recipe works again -- this hand-rolled version is kept anyway since it's a cheaper,
        // fully independent reference (no dependence on Linear_OP.trans at all).
        static void DenseTransMatVec(in floatMxN dense, in floatN x, ref floatN y)
        {
            for (int j = 0; j < dense.N_Cols; j++)
            {
                float s = 0;
                for (int i = 0; i < dense.M_Rows; i++)
                    s += dense[i, j] * x[i];
                y[j] = s;
            }
        }

        static void AssertMatEq(in floatMxN a, in floatMxN b, float tol)
        {
            Assert.IsTrue(a.M_Rows == b.M_Rows);
            Assert.IsTrue(a.N_Cols == b.N_Cols);
            for (int i = 0; i < a.M_Rows; i++)
                for (int j = 0; j < a.N_Cols; j++)
                    Assert.IsTrue(math.abs(a[i, j] - b[i, j]) < tol);
        }

        // ---- 1. hand-built small BSM: 2x2 grid of 3x3 blocks, some blocks omitted ---------
        void HandBuiltSmall()
        {
            var arena = new Arena(Allocator.Persistent);

            const int BR = 3, BC = 3;
            var builder = arena.floatBSMBuilder(2, 2, BR, BC);

            // Three distinct blocks placed at (0,0), (0,1), (1,1). Block (1,0) intentionally
            // omitted -> must expand to a zero 3x3 region.
            var b00 = arena.floatMat(BR, BC);
            var b01 = arena.floatMat(BR, BC);
            var b11 = arena.floatMat(BR, BC);
            for (int r = 0; r < BR; r++)
                for (int c = 0; c < BC; c++)
                {
                    b00[r, c] = (float)(10 + r * BC + c);
                    b01[r, c] = (float)(20 + r * BC + c);
                    b11[r, c] = (float)(30 + r * BC + c);
                }

            builder.AddBlock(0, 0, in b00);
            builder.AddBlock(0, 1, in b01);
            builder.AddBlock(1, 1, in b11);

            var A = builder.ToBSM(ref arena);

            // Structural expectations.
            Assert.IsTrue(A.M_Rows == 6);
            Assert.IsTrue(A.N_Cols == 6);
            Assert.IsTrue(A.Nnzb == 3);
            Assert.IsTrue(A.BR == BR);
            Assert.IsTrue(A.BC == BC);
            Assert.IsTrue(A.RowPtr.Length == 3); // BlockRows + 1

            // Independent reference dense (manual scatter, zeros elsewhere).
            var expected = arena.floatMat(6, 6);
            Scatter(ref expected, 0, 0, in b00);
            Scatter(ref expected, 0, 1, in b01);
            Scatter(ref expected, 1, 1, in b11);

            var dense = A.ToDense(ref arena);
            AssertMatEq(in dense, in expected, Tol());

            arena.Dispose();
        }

        // Build a random BSM on a 3x3 block grid of 3x3 blocks (9x9), a handful of blocks.
        static floatBSM BuildRandom(ref Arena arena)
        {
            const int BR = 3, BC = 3;
            var builder = arena.floatBSMBuilder(3, 3, BR, BC);
            builder.AddBlock(0, 0, arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 1001));
            builder.AddBlock(0, 2, arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 1002));
            builder.AddBlock(1, 1, arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 1003));
            builder.AddBlock(2, 0, arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 1004));
            builder.AddBlock(2, 2, arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 1005));
            return builder.ToBSM(ref arena);
        }

        // ---- 2. random BSM: spMV(A,x) == dense(A)*x --------------------------------------
        void RandomSpMV()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = BuildRandom(ref arena);
            var dense = A.ToDense(ref arena);
            var x = arena.floatRandomVec(A.N_Cols, (float)(-1f), (float)1f, 7777);

            // ref-dest overload.
            var y = arena.floatVec(A.M_Rows);
            Sparse_OP.spMV(in A, in x, ref y);

            var yRef = Linear_OP.dot(dense, x);
            Assert.IsTrue(Analysis_OP.isZero(y - yRef, Tol()));

            // allocating overload must agree with the ref-dest overload.
            var y2 = Sparse_OP.spMV(in A, in x);
            Assert.IsTrue(Analysis_OP.isZero(y2 - yRef, Tol()));

            arena.Dispose();
        }

        // ---- 3. random BSM: spMVT(A,x) == dense(A)^T * x ---------------------------------
        void RandomSpMVT()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = BuildRandom(ref arena);
            var dense = A.ToDense(ref arena);
            var x = arena.floatRandomVec(A.M_Rows, (float)(-1f), (float)1f, 8888);

            var y = arena.floatVec(A.N_Cols);
            Sparse_OP.spMVT(in A, in x, ref y);

            var yRef = arena.floatVec(A.N_Cols);
            DenseTransMatVec(in dense, in x, ref yRef);
            Assert.IsTrue(Analysis_OP.isZero(y - yRef, Tol()));

            var y2 = Sparse_OP.spMVT(in A, in x);
            Assert.IsTrue(Analysis_OP.isZero(y2 - yRef, Tol()));

            arena.Dispose();
        }

        // ---- 4. rectangular blocks BR != BC: dims + spMV + spMVT -------------------------
        void RectangularBlocks()
        {
            var arena = new Arena(Allocator.Persistent);

            const int BR = 2, BC = 3;
            var builder = arena.floatBSMBuilder(2, 3, BR, BC); // 4 x 9 dense
            builder.AddBlock(0, 0, arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 2001));
            builder.AddBlock(0, 2, arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 2002));
            builder.AddBlock(1, 1, arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 2003));
            var A = builder.ToBSM(ref arena);

            Assert.IsTrue(A.M_Rows == 4);
            Assert.IsTrue(A.N_Cols == 9);

            var dense = A.ToDense(ref arena);
            Assert.IsTrue(dense.M_Rows == 4);
            Assert.IsTrue(dense.N_Cols == 9);

            // spMV: x has N_Cols, y has M_Rows.
            var x = arena.floatRandomVec(A.N_Cols, (float)(-1f), (float)1f, 2100);
            var y = arena.floatVec(A.M_Rows);
            Sparse_OP.spMV(in A, in x, ref y);
            Assert.IsTrue(Analysis_OP.isZero(y - Linear_OP.dot(dense, x), Tol()));

            // spMVT: x has M_Rows, y has N_Cols.
            var xt = arena.floatRandomVec(A.M_Rows, (float)(-1f), (float)1f, 2200);
            var yt = arena.floatVec(A.N_Cols);
            Sparse_OP.spMVT(in A, in xt, ref yt);
            var ytRef = arena.floatVec(A.N_Cols);
            DenseTransMatVec(in dense, in xt, ref ytRef);
            Assert.IsTrue(Analysis_OP.isZero(yt - ytRef, Tol()));

            arena.Dispose();
        }

        // ---- 5. builder duplicate summation: same (br,bc) twice + AddValue twice ---------
        void DuplicateSummation()
        {
            var arena = new Arena(Allocator.Persistent);

            const int BR = 2, BC = 2;
            var builder = arena.floatBSMBuilder(2, 2, BR, BC); // 4 x 4 dense

            var blkA = arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 3001);
            var blkB = arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 3002);

            builder.AddBlock(0, 0, in blkA);
            builder.AddBlock(0, 0, in blkB);           // duplicate block -> summed

            // Two scalar adds into the SAME global cell (also inside block (0,0)).
            builder.AddValue(1, 0, (float)5);
            builder.AddValue(1, 0, (float)7);          // duplicate scalar -> summed

            Assert.IsTrue(builder.TripletCount == 4);   // pre-compression: 2 blocks + 2 scalars

            var A = builder.ToBSM(ref arena);
            Assert.IsTrue(A.Nnzb == 1);                 // all four triplets collapse to one block

            // Independent reference: sum blkA + blkB, then add 12 to local (1,0) == global (1,0).
            var expected = arena.floatMat(4, 4);
            var sum = arena.floatMat(BR, BC);
            for (int r = 0; r < BR; r++)
                for (int c = 0; c < BC; c++)
                    sum[r, c] = blkA[r, c] + blkB[r, c];
            sum[1, 0] += (float)12;
            Scatter(ref expected, 0, 0, in sum);

            var dense = A.ToDense(ref arena);
            AssertMatEq(in dense, in expected, Tol());

            arena.Dispose();
        }

        // ---- 6. out-of-order triplets: ColInd must come out ascending within each row ----
        void OutOfOrderTriplets()
        {
            var arena = new Arena(Allocator.Persistent);

            const int BR = 2, BC = 2;
            var builder = arena.floatBSMBuilder(2, 3, BR, BC); // 4 x 6 dense

            // Scrambled (br,bc) insertion order across both rows.
            var b_1_2 = arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 4001);
            var b_0_2 = arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 4002);
            var b_1_0 = arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 4003);
            var b_0_0 = arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 4004);
            var b_0_1 = arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 4005);

            builder.AddBlock(1, 2, in b_1_2);
            builder.AddBlock(0, 2, in b_0_2);
            builder.AddBlock(1, 0, in b_1_0);
            builder.AddBlock(0, 0, in b_0_0);
            builder.AddBlock(0, 1, in b_0_1);

            var A = builder.ToBSM(ref arena);

            // ColInd strictly ascending within each block-row.
            for (int row = 0; row < A.BlockRows; row++)
            {
                int s = A.RowPtr[row];
                int e = A.RowPtr[row + 1];
                for (int k = s + 1; k < e; k++)
                    Assert.IsTrue(A.ColInd[k - 1] < A.ColInd[k]);
            }

            // ...and the expansion is correct regardless of insertion order.
            var expected = arena.floatMat(4, 6);
            Scatter(ref expected, 0, 0, in b_0_0);
            Scatter(ref expected, 0, 1, in b_0_1);
            Scatter(ref expected, 0, 2, in b_0_2);
            Scatter(ref expected, 1, 0, in b_1_0);
            Scatter(ref expected, 1, 2, in b_1_2);

            var dense = A.ToDense(ref arena);
            AssertMatEq(in dense, in expected, Tol());

            arena.Dispose();
        }

        // ---- 7. 1x1-block BSM == plain sparse scalar matrix ------------------------------
        void OneByOneBlocks()
        {
            var arena = new Arena(Allocator.Persistent);

            var builder = arena.floatBSMBuilder(4, 4, 1, 1); // 4 x 4 scalar sparse

            // Scatter of scalar entries (some rows empty, one duplicate to also exercise sum).
            builder.AddValue(0, 0, (float)2);
            builder.AddValue(0, 3, (float)(-1));
            builder.AddValue(2, 1, (float)4);
            builder.AddValue(3, 3, (float)5);
            builder.AddValue(0, 0, (float)1); // duplicate at (0,0) -> 2 + 1 = 3

            var A = builder.ToBSM(ref arena);
            Assert.IsTrue(A.M_Rows == 4);
            Assert.IsTrue(A.N_Cols == 4);
            Assert.IsTrue(A.BR == 1);
            Assert.IsTrue(A.BC == 1);

            var expected = arena.floatMat(4, 4);
            expected[0, 0] = (float)3;
            expected[0, 3] = (float)(-1);
            expected[2, 1] = (float)4;
            expected[3, 3] = (float)5;

            var dense = A.ToDense(ref arena);
            AssertMatEq(in dense, in expected, Tol());

            // spMV must match the dense reference too.
            var x = arena.floatRandomVec(4, (float)(-1f), (float)1f, 5050);
            var y = arena.floatVec(4);
            Sparse_OP.spMV(in A, in x, ref y);
            Assert.IsTrue(Analysis_OP.isZero(y - Linear_OP.dot(expected, x), Tol()));

            arena.Dispose();
        }

        // ---- helper: fully-dense NxN 1x1-block builder assembled cell-by-cell via AddValue. --
        //
        // The builder is created with the DEFAULT capacityHint (8) -- deliberately NOT sized to
        // the pattern -- so assembling all N*N cells forces the three internal UnsafeLists to
        // reallocate repeatedly (8 -> 16 -> 32 -> ... , ~doubling each time). For N=15 that is
        // 225 triplets, i.e. five reallocations past the initial capacity. This is exactly the
        // growth path that used to leave the arena's tracked value-copy of the builder pointing
        // at a freed pre-growth buffer (double-free / use-after-free on dispose). Values come
        // from `refDense`; the caller keeps `refDense` as the independent dense reference.
        static floatBSM BuildDenseGrown(ref Arena arena, in floatMxN refDense, out int tripletCount)
        {
            int N = refDense.M_Rows;
            var builder = arena.floatBSMBuilder(N, N, 1, 1); // DEFAULT capacityHint = 8
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    builder.AddValue(r, c, refDense[r, c]);

            tripletCount = builder.TripletCount;
            return builder.ToBSM(ref arena);
        }

        // ---- 8. growth past capacityHint then arena.Dispose() (the crash) + correctness ------
        //
        // Regression for the fixed use-after-free: a builder created with the default
        // capacityHint=8 is grown to 225 triplets (15x15 dense, 1x1 blocks) via AddValue,
        // forcing ~5 UnsafeList reallocations. Pre-fix, arena.Dispose() below double-freed the
        // stale pre-growth buffer held by the arena's tracked copy (native crash, exit code
        // -1073741819). Post-fix this must dispose cleanly. Correctness after all those
        // reallocations is asserted BOTH ways: ToDense == dense reference, and spMV == dense dot.
        void GrowthThenDispose()
        {
            var arena = new Arena(Allocator.Persistent);

            const int N = 15;
            var refDense = arena.floatRandomMat(N, N, (float)(-1f), (float)1f, 74101);

            var A = BuildDenseGrown(ref arena, in refDense, out int tripletCount);

            Assert.IsTrue(tripletCount == N * N);         // 225 triplets -> grew well past 8
            Assert.IsTrue(A.M_Rows == N);
            Assert.IsTrue(A.N_Cols == N);
            Assert.IsTrue(A.Nnzb == N * N);               // every cell distinct -> one block each

            // Correctness #1: dense expansion matches the reference we assembled from.
            var dense = A.ToDense(ref arena);
            AssertMatEq(in dense, in refDense, Tol());

            // Correctness #2: spMV after growth matches the dense matvec.
            var x = arena.floatRandomVec(N, (float)(-1f), (float)1f, 74202);
            var y = arena.floatVec(N);
            Sparse_OP.spMV(in A, in x, ref y);
            Assert.IsTrue(Analysis_OP.isZero(y - Linear_OP.dot(refDense, x), Tol()));

            // The whole point: this must NOT crash / corrupt the heap.
            arena.Dispose();
        }

        // ---- 9. Clear() then reallocate a second grown builder in the SAME arena -------------
        //
        // Proves the arena's builder-tracking list is safely reusable after Clear(): grow a
        // first builder past capacityHint, ToBSM, Clear() (disposes the first builder's shared
        // state and empties the tracking list WITHOUT tearing down the arena), then build and
        // grow a SECOND builder in the same arena and verify its correctness too. Final
        // arena.Dispose() at the very end must also be clean.
        void ClearThenReallocate()
        {
            var arena = new Arena(Allocator.Persistent);

            const int N = 15;

            // First grown builder.
            var refA = arena.floatRandomMat(N, N, (float)(-1f), (float)1f, 75101);
            var A = BuildDenseGrown(ref arena, in refA, out int tripletCountA);
            Assert.IsTrue(tripletCountA == N * N);
            var denseA = A.ToDense(ref arena);
            AssertMatEq(in denseA, in refA, Tol());

            // Reuse the arena: disposes A / refA / builder-A's state, keeps the arena usable.
            arena.Clear();

            // Second grown builder in the SAME arena.
            var refB = arena.floatRandomMat(N, N, (float)(-1f), (float)1f, 75202);
            var B = BuildDenseGrown(ref arena, in refB, out int tripletCountB);
            Assert.IsTrue(tripletCountB == N * N);
            var denseB = B.ToDense(ref arena);
            AssertMatEq(in denseB, in refB, Tol());

            // spMV correctness on the post-Clear builder too.
            var x = arena.floatRandomVec(N, (float)(-1f), (float)1f, 75303);
            var y = arena.floatVec(N);
            Sparse_OP.spMV(in B, in x, ref y);
            Assert.IsTrue(Analysis_OP.isZero(y - Linear_OP.dot(refB, x), Tol()));

            arena.Dispose();
        }

        // ---- 10. empty BSM (zero triplets) + minimal 1x1 single-element BSM round-trip -------
        //
        // A builder with a nonzero block-grid shape but ZERO triplets ToBSM's to a valid empty
        // BSM (Nnzb == 0): every block-row's RowPtr range is empty so bsmMatVec/bsmMatVecT never
        // dereference the (possibly-null-Ptr) zero-length ColInd/Values buffers. ToDense must
        // produce the all-zero matrix and spMV/spMVT the zero vector for any x. Mirrors the
        // codebase's established zero-length-vector pattern (arena.floatVec(0) etc.). Also folds
        // in the smallest non-empty edge: a 1x1-grid, 1x1-block, single-triplet BSM (one scalar).
        void EmptyBSMRoundTrip()
        {
            var arena = new Arena(Allocator.Persistent);

            // --- empty BSM: 3x3 block grid of 2x2 blocks (6x6 dense) with NO triplets ---
            const int BR = 2, BC = 2;
            var builder = arena.floatBSMBuilder(3, 3, BR, BC);
            var A = builder.ToBSM(ref arena);

            Assert.IsTrue(A.Nnzb == 0);
            Assert.IsTrue(A.M_Rows == 6);
            Assert.IsTrue(A.N_Cols == 6);

            // ToDense of an empty BSM == the all-zero matrix of the right dims.
            var dense = A.ToDense(ref arena);
            var zero = arena.floatMat(6, 6);
            AssertMatEq(in dense, in zero, Tol());

            // spMV of an empty BSM == the zero vector, for a random nonzero x.
            var x = arena.floatRandomVec(A.N_Cols, (float)(-1f), (float)1f, 9201);
            var y = arena.floatVec(A.M_Rows);
            Sparse_OP.spMV(in A, in x, ref y);
            Assert.IsTrue(Analysis_OP.isZero(y, Tol()));

            // spMVT too (transpose path's empty-row loop is separate code).
            var xt = arena.floatRandomVec(A.M_Rows, (float)(-1f), (float)1f, 9202);
            var yt = arena.floatVec(A.N_Cols);
            Sparse_OP.spMVT(in A, in xt, ref yt);
            Assert.IsTrue(Analysis_OP.isZero(yt, Tol()));

            // --- minimal non-empty edge: 1x1 grid, 1x1 block, one triplet == a single scalar ---
            var oneBuilder = arena.floatBSMBuilder(1, 1, 1, 1);
            oneBuilder.AddValue(0, 0, (float)7);
            var one = oneBuilder.ToBSM(ref arena);

            Assert.IsTrue(one.Nnzb == 1);
            Assert.IsTrue(one.M_Rows == 1);
            Assert.IsTrue(one.N_Cols == 1);

            var oneDense = one.ToDense(ref arena);
            Assert.IsTrue(math.abs(oneDense[0, 0] - (float)7) < Tol());

            var ox = arena.floatVec(1);
            ox[0] = (float)3;
            var oy = arena.floatVec(1);
            Sparse_OP.spMV(in one, in ox, ref oy);
            Assert.IsTrue(math.abs(oy[0] - (float)21) < Tol()); // 7 * 3

            arena.Dispose();
        }
    }

    // ---- correctness cases (Burst) -------------------------------------------------------

    [Test]
    public void HandBuiltSmallTest()
        => new SparseBSMTestJob { Type = SparseBSMTestJob.TestType.HandBuiltSmall }.Run();

    [Test]
    public void RandomSpMVTest()
        => new SparseBSMTestJob { Type = SparseBSMTestJob.TestType.RandomSpMV }.Run();

    [Test]
    public void RandomSpMVTTest()
        => new SparseBSMTestJob { Type = SparseBSMTestJob.TestType.RandomSpMVT }.Run();

    [Test]
    public void RectangularBlocksTest()
        => new SparseBSMTestJob { Type = SparseBSMTestJob.TestType.RectangularBlocks }.Run();

    [Test]
    public void DuplicateSummationTest()
        => new SparseBSMTestJob { Type = SparseBSMTestJob.TestType.DuplicateSummation }.Run();

    [Test]
    public void OutOfOrderTripletsTest()
        => new SparseBSMTestJob { Type = SparseBSMTestJob.TestType.OutOfOrderTriplets }.Run();

    [Test]
    public void OneByOneBlocksTest()
        => new SparseBSMTestJob { Type = SparseBSMTestJob.TestType.OneByOneBlocks }.Run();

    // Regression: builder grown far past capacityHint (default 8), then arena.Dispose() --
    // used to double-free / use-after-free the arena's stale tracked copy (native crash).
    [Test]
    public void GrowthThenDisposeTest()
        => new SparseBSMTestJob { Type = SparseBSMTestJob.TestType.GrowthThenDispose }.Run();

    // Regression: arena.Clear() then a second grown builder in the same arena.
    [Test]
    public void ClearThenReallocateTest()
        => new SparseBSMTestJob { Type = SparseBSMTestJob.TestType.ClearThenReallocate }.Run();

    // Empty BSM (zero triplets) round-trips to zero dense / zero matvec; + minimal 1x1 scalar BSM.
    [Test]
    public void EmptyBSMRoundTripTest()
        => new SparseBSMTestJob { Type = SparseBSMTestJob.TestType.EmptyBSMRoundTrip }.Run();

    // ---- Regression test: ToDense/ToBSM dangling-arena-pointer bug (fixed) ----------------
    //
    // floatBSM.ToDense and floatBSMBuilder.ToBSM used to take `in Arena arena`, but both call a
    // MUTATING arena allocator method internally (arena.floatMat / arena.floatBSM). Since those
    // Arena methods aren't `readonly`, calling them through an `in Arena` parameter forced the C#
    // compiler to make a defensive copy of the arena, and the allocated result's internal arena
    // pointer captured the address of that dead stack temporary -- a use-after-scope bug. Reading
    // elements off the result was fine (the Values buffer is a real, independent allocation), but
    // any op that allocates through the result's own arena pointer -- e.g.
    // Linear_OP.trans(dense).tempfloatMat -- dereferenced the dangling pointer and threw
    // "allocator handle is not valid" under Burst. This broke the spec's own recommended
    // validation recipe: Sparse_OP.spMVT(A,x) vs Linear_OP.dot(Linear_OP.trans(ToDense(A)), x).
    //
    // Fixed by changing both signatures to `ref Arena arena` (matching how ArenaExtensions
    // factory methods take `this ref Arena`, not `this in Arena`, for the same reason). This test
    // is the regression check that the trans(ToDense(...)) recipe now works end to end.
    [Test]
    public void ToDense_TransposeReference_Works()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquare(ref arena);
            var dense = A.ToDense(ref arena);
            var x = arena.floatRandomVec(A.M_Rows, (float)(-1f), (float)1f, 12321);

            // Used to throw "allocator handle is not valid" (dangling arena pointer); now fixed.
            var yRef = Linear_OP.dot(Linear_OP.trans(dense), x);

            var y = arena.floatVec(A.N_Cols);
            Sparse_OP.spMVT(in A, in x, ref y);
            Assert.IsTrue(Analysis_OP.isZero(y - yRef, (float)1e-4f));
        }
        finally { arena.Dispose(); }
    }

    // ---- growth regression (managed thread): Clear() then Dispose() must not double-dispose --
    //
    // A builder grown past capacityHint (225 AddValue on a 15x15 1x1-block grid, ~5 UnsafeList
    // reallocations) is disposed via arena.Clear() and then arena.Dispose() (whose own trailing
    // Clear() pass runs again). Pre-fix this double-freed the arena's stale tracked copy; post-
    // fix the builder's Dispose() is idempotent via its _state null-guard, so the sequence is
    // safe. Runs on the managed thread so Assert.DoesNotThrow can wrap the dispose sequence.
    [Test]
    public void GrowthClearThenDispose_NoDoubleDispose()
    {
        var arena = new Arena(Allocator.Persistent);

        const int N = 15;
        var builder = arena.floatBSMBuilder(N, N, 1, 1); // DEFAULT capacityHint = 8
        for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++)
                builder.AddValue(r, c, (float)(r * N + c));

        Assert.IsTrue(builder.TripletCount == N * N); // 225 -> grew well past 8
        builder.ToBSM(ref arena);

        // Clear() disposes the builder's shared state once and empties the tracking list;
        // Dispose() then runs Clear() again internally. Neither may double-free / crash.
        Assert.DoesNotThrow(() =>
        {
            arena.Clear();
            arena.Dispose();
        });
    }

    // ---- 8. bounds-check throws (managed thread; Assert.Throws can't run inside Burst) ----

    [Test]
    public void AddBlock_BlockRowOutOfBounds_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var builder = arena.floatBSMBuilder(2, 2, 3, 3);
            var block = arena.floatMat(3, 3);
            Assert.Throws<ArgumentException>(() => builder.AddBlock(2, 0, in block));  // br == BlockRows
            Assert.Throws<ArgumentException>(() => builder.AddBlock(-1, 0, in block));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void AddBlock_BlockColOutOfBounds_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var builder = arena.floatBSMBuilder(2, 2, 3, 3);
            var block = arena.floatMat(3, 3);
            Assert.Throws<ArgumentException>(() => builder.AddBlock(0, 2, in block));  // bc == BlockCols
            Assert.Throws<ArgumentException>(() => builder.AddBlock(0, -1, in block));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void AddBlock_WrongBlockDimensions_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var builder = arena.floatBSMBuilder(2, 2, 3, 3);
            var wrong = arena.floatMat(2, 3); // not 3 x 3
            Assert.Throws<ArgumentException>(() => builder.AddBlock(0, 0, in wrong));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void AddValue_GlobalIndexOutOfBounds_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var builder = arena.floatBSMBuilder(2, 2, 3, 3); // 6 x 6
            Assert.Throws<ArgumentException>(() => builder.AddValue(6, 0, (float)1)); // row == M_Rows
            Assert.Throws<ArgumentException>(() => builder.AddValue(-1, 0, (float)1));
            Assert.Throws<ArgumentException>(() => builder.AddValue(0, 6, (float)1)); // col == N_Cols
            Assert.Throws<ArgumentException>(() => builder.AddValue(0, -1, (float)1));
        }
        finally { arena.Dispose(); }
    }

    // ---- 9. alias guard: y must not alias x ----------------------------------------------

    [Test]
    public void SpMV_AliasingY_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquare(ref arena);          // 4 x 4 (M_Rows == N_Cols)
            var x = arena.floatRandomVec(A.N_Cols, (float)(-1f), (float)1f, 9001);
            var yAlias = x;                          // struct copy shares Data.Ptr with x
            Assert.Throws<ArgumentException>(() => Sparse_OP.spMV(in A, in x, ref yAlias));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void SpMVT_AliasingY_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquare(ref arena);
            var x = arena.floatRandomVec(A.M_Rows, (float)(-1f), (float)1f, 9002);
            var yAlias = x;
            Assert.Throws<ArgumentException>(() => Sparse_OP.spMVT(in A, in x, ref yAlias));
        }
        finally { arena.Dispose(); }
    }

    // floatBlockJacobi.Apply's z-must-not-alias-r guard (each z_i draws on the full r_i block;
    // overwriting r in place mid-block would corrupt later rows of the same block's product).
    [Test]
    public void BlockJacobiApply_AliasingZ_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquare(ref arena);          // square BSM, both diagonal blocks present
            var M = arena.floatBlockJacobi(in A);
            var r = arena.floatRandomVec(A.M_Rows, (float)(-1f), (float)1f, 9003);
            var zAlias = r;                          // struct copy shares Data.Ptr with r
            Assert.Throws<ArgumentException>(() => M.Apply(in r, ref zAlias));
        }
        finally { arena.Dispose(); }
    }

    // ---- 10. dimension-mismatch guard ----------------------------------------------------

    [Test]
    public void SpMV_DimensionMismatch_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquare(ref arena); // 4 x 4
            // wrong x length (must equal N_Cols).
            var badX = arena.floatVec(A.N_Cols + 1);
            var y = arena.floatVec(A.M_Rows);
            Assert.Throws<ArgumentException>(() => Sparse_OP.spMV(in A, in badX, ref y));

            // wrong y length (must equal M_Rows).
            var x = arena.floatVec(A.N_Cols);
            var badY = arena.floatVec(A.M_Rows + 1);
            Assert.Throws<ArgumentException>(() => Sparse_OP.spMV(in A, in x, ref badY));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void SpMVT_DimensionMismatch_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquare(ref arena); // 4 x 4
            // spMVT: x must equal M_Rows, y must equal N_Cols.
            var badX = arena.floatVec(A.M_Rows + 1);
            var y = arena.floatVec(A.N_Cols);
            Assert.Throws<ArgumentException>(() => Sparse_OP.spMVT(in A, in badX, ref y));

            var x = arena.floatVec(A.M_Rows);
            var badY = arena.floatVec(A.N_Cols + 1);
            Assert.Throws<ArgumentException>(() => Sparse_OP.spMVT(in A, in x, ref badY));
        }
        finally { arena.Dispose(); }
    }

    // Small square (4x4) BSM used by the guard tests. Managed helper (no Burst).
    static floatBSM BuildSquare(ref Arena arena)
    {
        const int BR = 2, BC = 2;
        var builder = arena.floatBSMBuilder(2, 2, BR, BC);
        builder.AddBlock(0, 0, arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 6001));
        builder.AddBlock(1, 1, arena.floatRandomMat(BR, BC, (float)(-1f), (float)1f, 6002));
        return builder.ToBSM(ref arena);
    }
}
