using System;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// ARENA-LEVEL wiring tests for the SPARSE family (fProxyBSR / fProxyBlockJacobi) after the
// ChunkedRecordTable migration. Mirrors the dense house set
// (ArenaWiringTests.fProxy.cs) but adapted to the sparse contract, whose seams differ from the
// dense types in three load-bearing ways:
//
//   * fProxyBSR / fProxyBlockJacobi have NO Copy()/TempCopy() -- there is no copy-round-trip case.
//   * fProxyBlockJacobi is a READONLY struct: its Dispose() CANNOT null _rec afterward, so a
//     SAME-COPY double-dispose THROWS from the table's double-Free guard (unlike fProxyN/fProxyBSR
//     whose Dispose() nulls _rec and degrades a same-copy re-dispose to a safe no-op). Both the
//     same-copy and aliased BlockJacobi double-dispose are pinned as throws.
//   * fProxyBSR IS non-readonly and its Dispose() DOES null _rec (fProxyBSR.cs), so a BSR same-copy
//     re-dispose is a safe no-op like fProxyN; only an ALIASED BSR double-dispose throws.
//   * RowPtr/ColInd/Values are get-only dual-mode properties -- an indexer WRITE chained off the
//     property GET does not compile (CS1612). Any hand-populated BSR caches the list into a local
//     first (`var rp = A.RowPtr; rp[0] = ...;`), exactly as fProxyBSRBuilder.ToBSRCore does.
//
// Split, mirroring the house set: Burst-safe assertions (no exceptions expected) run inside a
// [BurstCompile] IJob; guard/throw assertions use NUnit's Assert.Throws, which cannot run inside a
// Burst job, so they are plain managed [Test]s.
public class fProxySparseArenaWiringTests
{
    [BurstCompile(CompileSynchronously = true)]
    public unsafe struct WiringTestJob : IJob
    {
        public enum TestType
        {
            MultiAllocBSRs,
            DisposeThenReallocateRecyclesSlot,
            StandaloneBSRAndBlockJacobi,
            BuilderToBSRSeam,
        }

        public TestType Type;

        // Reconstruction / matvec tolerance. Closed-form values here are small integers, so this
        // is generous on both precisions (float needs the looser bound, double is far tighter).
        static fProxy Tol() => /*+choose[1e-4f|1e-11]*/1e-4f/*-choose*/;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.MultiAllocBSRs: MultiAllocBSRs(); break;
                case TestType.DisposeThenReallocateRecyclesSlot: DisposeThenReallocateRecyclesSlot(); break;
                case TestType.StandaloneBSRAndBlockJacobi: StandaloneBSRAndBlockJacobi(); break;
                case TestType.BuilderToBSRSeam: BuilderToBSRSeam(); break;
                default: throw new NotImplementedException();
            }
        }

        // Hand-populate an arena-tracked (or standalone) 1x1-block-grid BSR carrying a single
        // sz x sz diagonal block scaled by `scale`: block[r,c] = (r==c) ? scale : 0. Caches
        // RowPtr/ColInd/Values into locals before the indexer writes (CS1612 -- see file header).
        static void FillDiagBlock(ref fProxyBSR A, int sz, fProxy scale)
        {
            var rp = A.RowPtr;   // length 2
            rp[0] = 0; rp[1] = 1;

            var ci = A.ColInd;   // length 1
            ci[0] = 0;

            var vals = A.Values; // length sz*sz
            for (int r = 0; r < sz; r++)
                for (int c = 0; c < sz; c++)
                    vals[r * sz + c] = (r == c) ? scale : (fProxy)0;
        }

        // ---- 3. multi-alloc: 30 arena-tracked BSRs of varied block sizes, all independently usable
        // 30 record-table allocations cross the record table's doubling chunk boundaries
        // (8,24,56,...) -> lands in chunk index 2, so several grows happen while earlier BSRs are
        // still live. Each BSR i carries a single (i+1)-scaled sz x sz diagonal block with sz cycling
        // 1..6 (exercising the tiled B1..B4/B6 spMV kernels AND the general BR=5 fallback). After ALL
        // 30 allocations, every BSR's spMV(ones) must return (i+1)*ones -- a relocating/aliasing bug
        // that handed BSR i another's buffer would decode to the wrong scale. Then arena.Dispose()
        // must drain to zero cleanly.
        void MultiAllocBSRs()
        {
            var arena = new Arena(Allocator.Persistent);

            const int n = 30;
            fProxyBSR* mats = stackalloc fProxyBSR[n];

            for (int i = 0; i < n; i++)
            {
                int sz = 1 + (i % 6);                          // block sizes 1..6
                var A = arena.fProxyBSR(1, 1, sz, sz, 1, uninit: true);
                FillDiagBlock(ref A, sz, (fProxy)(i + 1));     // distinct per-alloc scale sentinel
                mats[i] = A;
            }

            Assert.AreEqual(n, arena.AllocationsCount);

            // Read every BSR back AFTER all 30 allocations (and their chunk grows) have happened.
            for (int i = 0; i < n; i++)
            {
                int sz = 1 + (i % 6);
                Assert.AreEqual(sz, mats[i].M_Rows);
                Assert.AreEqual(sz, mats[i].N_Cols);
                Assert.AreEqual(1, mats[i].Nnzb);

                var x = new fProxyN(sz, Allocator.Temp, uninit: true);
                for (int j = 0; j < sz; j++) x[j] = (fProxy)1;
                var y = new fProxyN(sz, Allocator.Temp);       // zero-initialized
                BSR.spMV(in mats[i], in x, ref y);

                // Diagonal (i+1)*I applied to ones -> every entry == i+1.
                for (int j = 0; j < sz; j++)
                    Assert.IsTrue(math.abs(y[j] - (fProxy)(i + 1)) < Tol());

                x.Dispose();
                y.Dispose();
            }

            arena.Dispose();
            Assert.AreEqual(0, arena.AllAllocationsCount);
        }

        // ---- 5. dispose -> reallocate: the freed BSR slot is recycled into a genuinely independent,
        // ---- zero-initialized buffer, not a bleed-through of v1's stale sentinel bytes -----------
        // Mirrors the dense DisposeThenReallocateRecyclesSlot: dispose a sentinel-filled BSR, then
        // allocate a same-shape BSR with uninit:false -- it must be satisfied from the freed slot
        // (AllocationsCount 1 -> 0 -> 1, never 2) and come back all-zero (ClearMemory), with no trace
        // of v1's old Values. Then v2 is populated and spMV'd to prove it's its own writable buffer.
        void DisposeThenReallocateRecyclesSlot()
        {
            var arena = new Arena(Allocator.Persistent);

            const int sz = 2;
            var v1 = arena.fProxyBSR(1, 1, sz, sz, 1, uninit: true);
            FillDiagBlock(ref v1, sz, (fProxy)777);            // sentinel written into v1's Values
            Assert.AreEqual(1, arena.AllocationsCount);

            v1.Dispose();                                       // frees the slot back to the free-list
            Assert.AreEqual(0, arena.AllocationsCount);

            var v2 = arena.fProxyBSR(1, 1, sz, sz, 1, uninit: false); // should recycle v1's freed slot
            Assert.AreEqual(1, arena.AllocationsCount);
            Assert.AreEqual(sz, v2.M_Rows);

            // Zero-init: RowPtr/ColInd/Values all cleared -- no stale 777 sentinel bleed-through.
            var v2vals = v2.Values;
            for (int k = 0; k < sz * sz; k++)
                Assert.IsTrue(math.abs(v2vals[k]) < Tol());

            // And v2 is genuinely its own independent, writable buffer going forward.
            FillDiagBlock(ref v2, sz, (fProxy)3);              // 3*I
            var x = new fProxyN(sz, Allocator.Temp, uninit: true);
            for (int j = 0; j < sz; j++) x[j] = (fProxy)1;
            var y = new fProxyN(sz, Allocator.Temp);
            BSR.spMV(in v2, in x, ref y);
            for (int j = 0; j < sz; j++)
                Assert.IsTrue(math.abs(y[j] - (fProxy)3) < Tol());

            x.Dispose();
            y.Dispose();
            arena.Dispose();
            Assert.AreEqual(0, arena.AllocationsCount);
        }

        // ---- 4. STANDALONE (Allocator-ctor) BSR + BlockJacobi: construct / use / dispose unchanged -
        // The non-arena path (_rec == null, backing stores are the inline UnsafeLists). A
        // block-DIAGONAL A means blockdiag(A) == A, so BlockJacobi is exactly A^-1: applying it to
        // y = A*x recovers x -- a clean round-trip that also pins the row-major block layout (the
        // block is upper-triangular [[2,1],[0,2]], NOT symmetric, so a transposed-layout bug would
        // break both spMV and the round-trip). Everything is Allocator.Temp -- no arena involved.
        void StandaloneBSRAndBlockJacobi()
        {
            const int sz = 2;
            // 1x2-grid? no: 2x2 block grid of 2x2 blocks, diagonal blocks only (nnzb = 2).
            var A = new fProxyBSR(2, 2, sz, sz, 2, Allocator.Temp, uninit: true);

            var rp = A.RowPtr;   // length 3
            rp[0] = 0; rp[1] = 1; rp[2] = 2;
            var ci = A.ColInd;   // length 2
            ci[0] = 0; ci[1] = 1;
            var vals = A.Values; // length nnzb*sz*sz = 8; each block row-major [[2,1],[0,2]]
            for (int b = 0; b < 2; b++)
            {
                int o = b * sz * sz;
                vals[o + 0] = (fProxy)2; vals[o + 1] = (fProxy)1;
                vals[o + 2] = (fProxy)0; vals[o + 3] = (fProxy)2;
            }

            Assert.AreEqual(4, A.M_Rows);
            Assert.AreEqual(4, A.N_Cols);
            Assert.AreEqual(2, A.Nnzb);

            // x = [1,2,3,4]; y = A*x. Block [[2,1],[0,2]] on [1,2] -> [4,4]; on [3,4] -> [10,8].
            var x = new fProxyN(4, Allocator.Temp, uninit: true);
            x[0] = (fProxy)1; x[1] = (fProxy)2; x[2] = (fProxy)3; x[3] = (fProxy)4;
            var y = new fProxyN(4, Allocator.Temp);
            BSR.spMV(in A, in x, ref y);
            Assert.IsTrue(math.abs(y[0] - (fProxy)4) < Tol());
            Assert.IsTrue(math.abs(y[1] - (fProxy)4) < Tol());
            Assert.IsTrue(math.abs(y[2] - (fProxy)10) < Tol());
            Assert.IsTrue(math.abs(y[3] - (fProxy)8) < Tol());

            // Standalone BlockJacobi from the same standalone A. blockdiag(A) == A (A is block-
            // diagonal), so M^-1 (A x) == x exactly.
            var M = new fProxyBlockJacobi(in A, Allocator.Temp);
            Assert.AreEqual(4, M.Rows);
            var z = new fProxyN(4, Allocator.Temp);
            M.Apply(in y, ref z);
            for (int j = 0; j < 4; j++)
                Assert.IsTrue(math.abs(z[j] - x[j]) < Tol());

            // Construct/use/dispose semantics unchanged on the standalone path.
            z.Dispose();
            M.Dispose();
            y.Dispose();
            x.Dispose();
            A.Dispose();
        }

        // ---- 6. builder (old value-copy model) -> ToBSR -> arena-tracked BSR (new record model) ----
        // The seam between the two memory models. Build via the still-value-copy-tracked builder,
        // ToBSR into an arena-tracked BSR record, then dispose the BSR DIRECTLY (new-model record
        // Free). That direct dispose must NOT disturb the old-model builder -- its shared _state is
        // untouched, so builder.TripletCount stays readable (== the seam being clean). arena.Clear()
        // then disposes the tracked builder AND skips the already-freed BSR slot; arena.Dispose()
        // is clean. (Only the arena disposes the tracked builder copy -- callers never do, per the
        // builder's Dispose doc.)
        void BuilderToBSRSeam()
        {
            var arena = new Arena(Allocator.Persistent);

            const int sz = 2;
            var builder = arena.fProxyBSRBuilder(2, 2, sz, sz);

            var d = arena.fProxyMat(sz, sz);
            d[0, 0] = (fProxy)2; d[0, 1] = (fProxy)1;
            d[1, 0] = (fProxy)0; d[1, 1] = (fProxy)2;
            builder.AddBlock(0, 0, in d);
            builder.AddBlock(1, 1, in d);
            Assert.AreEqual(2, builder.TripletCount);

            var A = builder.ToBSR(ref arena);          // arena-tracked (record-backed) BSR
            Assert.AreEqual(4, A.M_Rows);
            Assert.AreEqual(2, A.Nnzb);

            int before = arena.AllocationsCount;        // builder + d mat + A record (+ nothing else)
            A.Dispose();                                // direct new-model record Free
            Assert.AreEqual(before - 1, arena.AllocationsCount);

            // Seam intact: disposing the new-model BSR did not touch the old-model builder's state.
            Assert.AreEqual(2, builder.TripletCount);

            // arena.Clear() disposes the tracked builder AND skips A's already-freed slot -- no crash.
            arena.Clear();
            Assert.AreEqual(0, arena.AllocationsCount);

            arena.Dispose();
            Assert.AreEqual(0, arena.AllAllocationsCount);
        }
    }

    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(WiringTestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void BurstWiring(WiringTestJob.TestType type)
    {
        new WiringTestJob() { Type = type }.Run();
    }

    // ---- Guard / throw tests (managed thread; Assert.Throws can't run inside a Burst job) ----------

    // Small square (4x4) arena-tracked BSR with both diagonal blocks present and invertible
    // ([[2,1],[1,2]] is SPD) -- so fProxyBlockJacobi construction succeeds. Managed helper.
    static fProxyBSR BuildSquareInvertible(ref Arena arena)
    {
        const int BR = 2, BC = 2;
        var builder = arena.fProxyBSRBuilder(2, 2, BR, BC);
        var d = arena.fProxyMat(BR, BC);
        d[0, 0] = (fProxy)2; d[0, 1] = (fProxy)1;
        d[1, 0] = (fProxy)1; d[1, 1] = (fProxy)2;
        builder.AddBlock(0, 0, in d);
        builder.AddBlock(1, 1, in d);
        return builder.ToBSR(ref arena);
    }

    // ---- 1. ALIASED double-dispose of an arena-tracked fProxyBSR ---------------------------------
    // Two struct copies share the SAME fProxyBSRRecord*. Disposing the first frees the slot AND
    // nulls its own _rec; disposing the second (the alias, whose _rec still points at the now-dead
    // record) must throw from the table's double-Free guard, BEFORE touching the freed native
    // buffers a second time. Afterward arena.Clear() must SKIP the freed slot (IsAlive guard) and
    // arena.Dispose() teardown stays clean -- no crash, no leak.
    [Test]
    public void AliasedDoubleDispose_BSR_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareInvertible(ref arena);
            var alias = A;                             // struct copy -- shares A's fProxyBSRRecord*
            int before = arena.AllocationsCount;

            A.Dispose();                               // frees the slot; A._rec -> null
            Assert.AreEqual(before - 1, arena.AllocationsCount);

            // alias still points at the (now dead) record -> second Free throws.
            Assert.Throws<InvalidOperationException>(() => alias.Dispose());

            // The guard rejected the double-Free before any bookkeeping ran twice.
            Assert.AreEqual(before - 1, arena.AllocationsCount);

            // Clear() must skip the freed BSR slot rather than double-free it.
            arena.Clear();
            Assert.AreEqual(0, arena.AllocationsCount);
        }
        finally { arena.Dispose(); }
    }

    // A BSR is NON-readonly and its Dispose() nulls _rec (fProxyBSR.cs), so disposing the SAME
    // variable twice is a safe no-op (the second call takes the standalone branch and disposes
    // default UnsafeLists) -- exactly like fProxyN. Pins that fProxyBSR does NOT share
    // fProxyBlockJacobi's readonly same-copy-throws behavior.
    [Test]
    public void SameInstanceDoubleDispose_BSR_IsNoOp()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareInvertible(ref arena);
            A.Dispose();
            // The builder + its source mat are still arena-tracked; only A's record was freed.
            int afterFirst = arena.AllocationsCount;
            Assert.DoesNotThrow(() => A.Dispose());
            Assert.AreEqual(afterFirst, arena.AllocationsCount);   // no-op: count unchanged
        }
        finally { arena.Dispose(); }
    }

    // ---- 2. fProxyBlockJacobi double-dispose: SAME-COPY throws (readonly struct) + ALIASED throws --
    // fProxyBlockJacobi is a READONLY struct, so Dispose() CANNOT null _rec afterward. Consequence:
    // even a SAME-COPY second Dispose() re-enters the arena-tracked branch (_rec still non-null) and
    // throws from the table's double-Free guard -- it does NOT degrade to the safe no-op that
    // fProxyN/fProxyBSR's mutable Dispose() gives. Pin this divergence distinctly.
    [Test]
    public void SameInstanceDoubleDispose_BlockJacobi_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareInvertible(ref arena);
            var M = arena.fProxyBlockJacobi(in A);
            int before = arena.AllocationsCount;

            M.Dispose();                               // frees the slot; _rec CANNOT be nulled (readonly)
            Assert.AreEqual(before - 1, arena.AllocationsCount);

            // Same variable again -> _rec still non-null -> double-Free guard throws.
            Assert.Throws<InvalidOperationException>(() => M.Dispose());
            Assert.AreEqual(before - 1, arena.AllocationsCount);
        }
        finally { arena.Dispose(); }
    }

    // An ALIASED fProxyBlockJacobi double-dispose throws for the same double-Free-guard reason (this
    // half matches fProxyN/fProxyBSR; the same-copy half above is where they diverge).
    [Test]
    public void AliasedDoubleDispose_BlockJacobi_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareInvertible(ref arena);
            var M = arena.fProxyBlockJacobi(in A);
            var alias = M;                             // struct copy -- shares M's record*

            M.Dispose();
            Assert.Throws<InvalidOperationException>(() => alias.Dispose());
        }
        finally { arena.Dispose(); }
    }

    // ---- Generational-overlay guard tests (ENABLE_UNITY_COLLECTIONS_CHECKS-only) -----------
    // A checks-gated "generational overlay" on the arena-tracked structs' data getters:
    // reading through a STALE handle -- one whose slot was Disposed or arena.Clear()'d, or (option (c)
    // only) freed and then RECYCLED by an unrelated fresh allocation -- throws InvalidOperationException
    // instead of silently returning a dead/garbage buffer. fProxyBSR is option (c): its
    // RowPtr/ColInd/Values getters SHARE one AssertRecordValid() (Alive + a free `_gen` stamp in the
    // struct's internal padding hole), so it ADDITIONALLY catches the recycled-slot case. fProxyBlockJacobi
    // is option (b) (Alive-only: no spare padding), guarding its DInv getter.
    //
    // NO ClearTemp case here for EITHER type: neither fProxyBSR nor fProxyBlockJacobi has a temp-pool
    // counterpart (no fProxyTempBSR / no ClearTemp-tracked BlockJacobi pool -- see Arena.Sparse.fProxy.cs),
    // so the "TempCopy + ClearTemp + stale read" scenario the dense N/MxN families cover simply does not
    // exist on the sparse side. Whole methods are compiled under the same symbol the guard is (so they
    // can't go vacuous when checks are off), mirroring RollingWindowTests' out-of-range throw tests.
#if ENABLE_UNITY_COLLECTIONS_CHECKS

    // (c) BSR read-after-Dispose. fProxyBSR is a mutable struct and Dispose() nulls the DISPOSER's OWN
    // _rec (so the disposed variable itself reads harmlessly standalone -- see SameInstanceDoubleDispose_
    // BSR_IsNoOp), so the stale read must go through an ALIAS whose _rec still points at the freed
    // record. RowPtr/ColInd/Values share one AssertRecordValid(); pin two of the three call sites.
    [Test]
    public void BSR_ReadAfterDispose_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyBSR(1, 1, 2, 2, 1);
            var alias = A;                 // struct copy -- shares A's (about-to-die) record
            A.Dispose();                   // frees the slot; alias._rec now dangles onto a dead slot
            Assert.Throws<InvalidOperationException>(() => { var _ = alias.RowPtr; });
            Assert.Throws<InvalidOperationException>(() => { var _ = alias.ColInd; });
        }
        finally { arena.Dispose(); }
    }

    // (c) BSR read after arena.Clear(). Clear() Frees every live slot but leaves the caller's struct
    // copy alone, so A._rec still points at the now-dead record -> the Values getter throws. (Clear,
    // NOT a full arena.Dispose, which would also free the record table's chunk memory -- a raw-memory
    // use-after-free the guard can't safely observe.)
    [Test]
    public void BSR_ReadAfterArenaClear_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyBSR(1, 1, 2, 2, 1);
            arena.Clear();                 // frees A's slot; A is still a (now stale) handle
            Assert.Throws<InvalidOperationException>(() => { var _ = A.Values; });
        }
        finally { arena.Dispose(); }
    }

    // (c) THE generation-stamp payoff for BSR -- the one case option (c) buys over option (b). A stale
    // handle into a slot freed and then RECYCLED by a fresh, unrelated allocation: Alive alone would
    // read true again for the new occupant, but fProxyBSR's _gen (stamped at v1's construction) no
    // longer matches the table's CURRENT generation for that slot, so the stale read throws -- while v2
    // (on the recycled slot) reads & writes fine. The 1 -> 0 -> 1 count trace pins that v2 reused v1's
    // EXACT freed slot, so it is the GENERATION mismatch (not a dead slot) that trips the guard.
    [Test]
    public void BSR_StaleGenerationAfterSlotRecycle_OldThrows_NewWorks()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var v1 = arena.fProxyBSR(1, 1, 2, 2, 1);
            var alias = v1;                // captures v1's record pointer AND its generation stamp
            Assert.AreEqual(1, arena.AllocationsCount);

            v1.Dispose();                  // frees v1's slot onto the table's LIFO free list
            Assert.AreEqual(0, arena.AllocationsCount);

            var v2 = arena.fProxyBSR(1, 1, 2, 2, 1);   // recycles v1's EXACT slot; generation bumped
            Assert.AreEqual(1, arena.AllocationsCount);

            // alias: slot is alive again (for v2), but its stamped generation is stale -> throws.
            Assert.Throws<InvalidOperationException>(() => { var _ = alias.RowPtr; });

            // v2: same slot, current generation -> reads cleanly and is a usable buffer. RowPtr is
            // get-only, so cache into a local before the indexer write (CS1612 -- see file header).
            Assert.DoesNotThrow(() => { var _ = v2.RowPtr; });
            var rp = v2.RowPtr;            // length blockRows+1 == 2
            rp[0] = 0; rp[1] = 1;
            Assert.AreEqual(1, v2.RowPtr[1]);
        }
        finally { arena.Dispose(); }
    }

    // (b) BLOCKJACOBI read-after-Dispose. fProxyBlockJacobi is a READONLY struct, so Dispose() CANNOT
    // null its _rec afterward -- the SAME variable's _rec still points at the freed slot, so reading its
    // DInv getter throws with NO alias needed (unlike the mutable fProxyN/fProxyBSR families). This is
    // the same readonly asymmetry SameInstanceDoubleDispose_BlockJacobi_Throws pins for Dispose().
    [Test]
    public void BlockJacobi_ReadAfterDispose_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareInvertible(ref arena);
            var M = arena.fProxyBlockJacobi(in A);
            M.Dispose();                   // readonly struct: _rec is NOT nulled
            Assert.Throws<InvalidOperationException>(() => { var _ = M.DInv; });
        }
        finally { arena.Dispose(); }
    }

    // (b) BLOCKJACOBI read after arena.Clear(). Clear() Frees the preconditioner's slot; the stale M
    // handle's DInv getter throws (Alive is false). Same Clear-not-Dispose rationale as the BSR case.
    [Test]
    public void BlockJacobi_ReadAfterArenaClear_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareInvertible(ref arena);
            var M = arena.fProxyBlockJacobi(in A);
            arena.Clear();                 // frees M's slot; M is still a (now stale) handle
            Assert.Throws<InvalidOperationException>(() => { var _ = M.DInv; });
        }
        finally { arena.Dispose(); }
    }
#endif
}
