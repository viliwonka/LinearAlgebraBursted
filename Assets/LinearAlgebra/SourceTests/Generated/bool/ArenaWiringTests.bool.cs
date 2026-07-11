using System;
using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// ARENA-LEVEL wiring tests for the boolN/boolMxN -> ChunkedRecordTable migration. Mirrors the
// floatN/intN wiring tests
// (ArenaWiringTests.float.cs / ArenaWiringTests.int.cs) but adapted to bool's narrower API:
//
//   * bool has NO scalar-fill factory (arena.boolVec(N, s) means "leave uninitialized", not
//     "fill with s"), so buffers are filled element-by-element through the indexer.
//   * bool allocations are NOT tracked in Arena.AllocationsCount / TempAllocationsCount (a
//     PRE-EXISTING gap that predates the record-table migration -- boolVec/boolMat were never
//     counted). So these tests do NOT assert allocation counts; they pin buffer CORRECTNESS across
//     chunk grows / temp recycling / dispose+realloc, and the dispose GUARDS (aliased double-Free
//     throws, Clear() skips a directly-disposed slot without crashing), which is where the
//     migration's real risk lives.
//   * With only two states per cell, a single element is a weak discriminator, so the multi-chunk
//     tests ENCODE each allocation's own index as a bit pattern across its cells (cell j holds
//     bit j of the allocation index). A relocating/aliasing bug that handed allocation i another
//     allocation's buffer would decode to the wrong index and trip the assert.
//
// Split, mirroring ChunkedRecordTableTests.cs: Burst-safe assertions run inside a [BurstCompile]
// IJob; guard/throw assertions (Assert.Throws can't run inside Burst) are plain managed [Test]s.
public class boolArenaWiringTests
{
    [BurstCompile(CompileSynchronously = true)]
    public unsafe struct WiringTestJob : IJob
    {
        public enum TestType
        {
            MultiChunkVectors,
            MultiChunkMatrices,
            TempRecyclingCycles,
            DirectDisposeThenClearVectors,
            DirectDisposeThenClearMatrices,
            DisposeThenReallocateRecyclesSlot,
        }

        public TestType Type;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.MultiChunkVectors: MultiChunkVectors(); break;
                case TestType.MultiChunkMatrices: MultiChunkMatrices(); break;
                case TestType.TempRecyclingCycles: TempRecyclingCycles(); break;
                case TestType.DirectDisposeThenClearVectors: DirectDisposeThenClearVectors(); break;
                case TestType.DirectDisposeThenClearMatrices: DirectDisposeThenClearMatrices(); break;
                case TestType.DisposeThenReallocateRecyclesSlot: DisposeThenReallocateRecyclesSlot(); break;
                default: throw new NotImplementedException();
            }
        }

        // ---- multi-chunk: 300 persistent vectors cross several record-table chunk boundaries ------
        // ChunkedRecordTable chunk capacities double from 8: 8,16,32,64,128,256 (cumulative
        // 8,24,56,120,248,504). 300 allocations land in chunk index 5 -> SIX chunks, crossing FIVE
        // boundaries. Each vector's 10 cells encode its OWN allocation index i as a bit pattern
        // (cell j == bit j of i); 10 bits covers i < 1024 >> 300, so every vector's pattern is
        // globally unique. Reading them all back AFTER every grow catches any scrambled buffer.
        void MultiChunkVectors()
        {
            var arena = new Arena(Allocator.Persistent);

            const int n = 300;
            const int len = 10;   // 10 bits uniquely encodes any i in [0, 300)
            boolN* vecs = stackalloc boolN[n];

            for (int i = 0; i < n; i++)
            {
                boolN v = arena.boolVec(len, uninit: true);
                for (int j = 0; j < len; j++)
                    v[j] = ((i >> j) & 1) == 1;   // cell j holds bit j of the allocation index
                vecs[i] = v;
            }

            // Read every vector back AFTER all 300 allocations (and their chunk grows) have happened.
            for (int i = 0; i < n; i++)
            {
                Assert.AreEqual(len, vecs[i].N);
                for (int j = 0; j < len; j++)
                    Assert.IsTrue(vecs[i][j] == (((i >> j) & 1) == 1));
            }

            arena.Dispose();
        }

        // ---- multi-chunk: 300 persistent matrices, same argument on the boolMatRecords table ------
        // 4x4 = 16 cells encode the allocation index (flat cell k == bit k of i); 16 bits covers all.
        void MultiChunkMatrices()
        {
            var arena = new Arena(Allocator.Persistent);

            const int n = 300;
            const int R = 4, C = 4;   // 16 cells -> 16 bits, uniquely encodes any i in [0, 300)
            boolMxN* mats = stackalloc boolMxN[n];

            for (int i = 0; i < n; i++)
            {
                boolMxN m = arena.boolMat(R, C, uninit: true);
                for (int r = 0; r < R; r++)
                    for (int c = 0; c < C; c++)
                        m[r, c] = ((i >> (r * C + c)) & 1) == 1;
                mats[i] = m;
            }

            for (int i = 0; i < n; i++)
            {
                Assert.AreEqual(R, mats[i].M_Rows);
                Assert.AreEqual(C, mats[i].N_Cols);
                for (int r = 0; r < R; r++)
                    for (int c = 0; c < C; c++)
                        Assert.IsTrue(mats[i][r, c] == (((i >> (r * C + c)) & 1) == 1));
            }

            arena.Dispose();
        }

        // ---- temp recycling: allocate temps, ClearTemp(), repeat -- the per-frame ClearTemp loop ---
        // Three cycles of (N temp vecs + N temp mats via TempCopy, each written+read to prove the
        // recycled slot's buffer is live), then ClearTemp. The persistent seeds must survive every
        // ClearTemp untouched. (Counts aren't asserted -- bool isn't tracked in TempAllocationsCount;
        // ChunkedRecordTableTests already pins the table's drain/recycle invariant directly.)
        void TempRecyclingCycles()
        {
            var arena = new Arena(Allocator.Persistent);

            const int N = 20;
            var seedV = arena.boolVec(4, uninit: true);
            for (int j = 0; j < 4; j++) seedV[j] = (j % 2 == 0);   // known seed pattern
            var seedM = arena.boolMat(3, 3, uninit: true);
            for (int j = 0; j < seedM.Length; j++) seedM[j] = (j % 2 == 0);

            for (int cycle = 0; cycle < 3; cycle++)
            {
                for (int i = 0; i < N; i++)
                {
                    // A fresh temp buffer each call; overwrite element 0 with a per-(cycle,i) value
                    // and read it straight back -- proves the recycled slot's buffer is live & writable.
                    bool bit = ((cycle + i) % 2) == 0;

                    boolN tv = seedV.TempCopy();
                    tv[0] = bit;
                    Assert.IsTrue(tv[0] == bit);

                    boolMxN tm = seedM.TempCopy();
                    tm[0, 0] = bit;
                    Assert.IsTrue(tm[0, 0] == bit);
                }

                arena.ClearTemp();

                // The persistent seeds survive ClearTemp with their patterns intact.
                Assert.AreEqual(4, seedV.N);
                for (int j = 0; j < 4; j++) Assert.IsTrue(seedV[j] == (j % 2 == 0));
                Assert.AreEqual(9, seedM.Length);
                for (int j = 0; j < seedM.Length; j++) Assert.IsTrue(seedM[j] == (j % 2 == 0));
            }

            arena.Dispose();
        }

        // ---- direct Dispose() then Clear() SKIPS the freed slot (IsAlive guard), no double-free ----
        // Bool isn't count-tracked, so this pins the STRUCTURAL half of the contract: a directly
        // disposed middle allocation leaves its survivors readable, and a following Clear()/Dispose()
        // must skip the already-freed slot rather than double-free it (no crash, no corruption).
        void DirectDisposeThenClearVectors()
        {
            var arena = new Arena(Allocator.Persistent);

            var a = arena.boolVec(4, uninit: true);
            var b = arena.boolVec(4, uninit: true);
            var c = arena.boolVec(4, uninit: true);
            for (int j = 0; j < 4; j++) { a[j] = true; b[j] = false; c[j] = (j % 2 == 0); }

            b.Dispose();                              // direct dispose of the middle allocation

            // The survivors are untouched and still readable with their own distinct patterns.
            for (int j = 0; j < 4; j++)
            {
                Assert.IsTrue(a[j]);
                Assert.IsTrue(c[j] == (j % 2 == 0));
            }

            arena.Clear();                            // must SKIP b's dead slot, free a & c -- no crash
            arena.Dispose();                          // clean teardown after a Clear that skipped a slot
        }

        void DirectDisposeThenClearMatrices()
        {
            var arena = new Arena(Allocator.Persistent);

            var a = arena.boolMat(2, 2, uninit: true);
            var b = arena.boolMat(2, 2, uninit: true);
            var c = arena.boolMat(2, 2, uninit: true);
            for (int j = 0; j < 4; j++) { a[j] = true; b[j] = false; c[j] = (j % 2 == 0); }

            b.Dispose();

            Assert.IsTrue(a[0, 0]);
            Assert.IsTrue(c[0, 0]);    // c[0] == (0 % 2 == 0) == true
            Assert.IsFalse(c[0, 1]);   // c[1] == false

            arena.Clear();
            arena.Dispose();
        }

        // ---- dispose -> reallocate: the freed slot is recycled, and the new allocation is a --------
        // ---- genuinely independent zero-initialized buffer, not a bleed-through of stale bytes -----
        // Disposes v1 (all-true sentinel) directly, then allocates v2 with uninit:false. v2 must come
        // back all-false: a bug that skipped re-initializing a recycled record's Data would surface as
        // v1's stale `true` cells leaking through instead of the zero (false) fill.
        void DisposeThenReallocateRecyclesSlot()
        {
            var arena = new Arena(Allocator.Persistent);

            var v1 = arena.boolVec(4, uninit: true);
            for (int j = 0; j < 4; j++) v1[j] = true;   // all-true sentinel written into v1's buffer

            v1.Dispose();                                // frees the slot back to the free-list

            var v2 = arena.boolVec(4, uninit: false);    // should recycle v1's freed slot, zero-init
            Assert.AreEqual(4, v2.N);
            for (int j = 0; j < 4; j++)
                Assert.IsFalse(v2[j]);                    // zero-init: no stale `true` bleed

            // And v2 is genuinely its own independent, writable buffer going forward.
            for (int j = 0; j < 4; j++) v2[j] = (j % 2 == 0);
            for (int j = 0; j < 4; j++) Assert.IsTrue(v2[j] == (j % 2 == 0));

            arena.Dispose();
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

    // DOUBLE-DISPOSE CONTRACT. Two struct copies of one arena-tracked vector share the SAME record
    // pointer. Disposing the first frees the slot; disposing the second must throw from the record
    // table's double-Free guard, BEFORE it touches the (already-freed) native buffer a second time.
    //
    // (An alias is REQUIRED to trip the guard: boolN.Dispose() nulls its own _rec, so calling
    // Dispose() twice through the SAME variable is a safe no-op -- see the IsNoOp test below.)
    [Test]
    public void AliasedDoubleDispose_Vector_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var v = arena.boolVec(8);
            var alias = v;                 // struct copy -- shares v's boolVecRecord*

            v.Dispose();                   // frees the slot; v._rec -> null

            // alias still points at the (now dead) record -> second Free throws.
            Assert.Throws<InvalidOperationException>(() => alias.Dispose());
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void AliasedDoubleDispose_Matrix_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var m = arena.boolMat(3, 4);
            var alias = m;

            m.Dispose();

            Assert.Throws<InvalidOperationException>(() => alias.Dispose());
        }
        finally { arena.Dispose(); }
    }

    // Disposing the SAME variable twice is a safe no-op (Dispose() nulls _rec, so the second call
    // takes the standalone branch and disposes a default(UnsafeList), which is harmless).
    [Test]
    public void SameInstanceDoubleDispose_Vector_IsNoOp()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var v = arena.boolVec(8);
            v.Dispose();
            Assert.DoesNotThrow(() => v.Dispose());
        }
        finally { arena.Dispose(); }
    }

    // STANDALONE Copy()/TempCopy() throw contract. boolN has NO public standalone (Allocator-only)
    // ctor, so a standalone instance is made via the copy ctor `new boolN(in orig, allocator)`, which
    // unconditionally sets _rec = null -> the result is standalone by construction. With a null
    // record, Copy()/TempCopy() have no owning arena to allocate through and must throw
    // InvalidOperationException -- NOT a NullReferenceException from dereferencing a null record/core.
    [Test]
    public void StandaloneVector_CopyAndTempCopy_Throw()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var orig = arena.boolVec(4);              // arena-backed source
            var standalone = new boolN(in orig, Allocator.Temp);   // copy ctor -> _rec == null (standalone)
            try
            {
                Assert.Throws<InvalidOperationException>(() => standalone.Copy());
                Assert.Throws<InvalidOperationException>(() => standalone.TempCopy());
            }
            finally { standalone.Dispose(); }
        }
        finally { arena.Dispose(); }
    }

    // boolMxN DOES have a public standalone (Allocator-only) ctor, so construct it directly.
    [Test]
    public void StandaloneMatrix_CopyAndTempCopy_Throw()
    {
        var m = new boolMxN(3, 3, Allocator.Temp);
        try
        {
            Assert.Throws<InvalidOperationException>(() => m.Copy());
            Assert.Throws<InvalidOperationException>(() => m.TempCopy());
        }
        finally { m.Dispose(); }
    }

    // ---- Generational-overlay guard tests (Stage E; ENABLE_UNITY_COLLECTIONS_CHECKS-only) -----------
    // Stage E added a checks-gated "generational overlay" to the arena-tracked structs' Data getter:
    // reading through a STALE handle -- one whose slot was Disposed / arena.Clear()'d / ClearTemp()'d,
    // or (option (c) only) freed and then RECYCLED by an unrelated fresh allocation -- throws
    // InvalidOperationException instead of silently returning a dead/garbage buffer. boolN is
    // option (b) (Alive-only: the 32B struct has no spare padding for a generation stamp); boolMxN is
    // option (c) (Alive + a free `_gen` stamp riding in its trailing padding hole), so it ADDITIONALLY
    // catches the recycled-slot case that Alive alone cannot. Mirrors ArenaWiringTests.fProxy.cs but
    // with bool's narrower API (no scalar-fill factory -- boolVec(N)/boolMat(R,C) leave contents
    // undefined -- and true/false sentinels). Whole methods are compiled under the same symbol the
    // guard is (so they can't go vacuous when checks are off), like RollingWindowTests' throw tests.
#if ENABLE_UNITY_COLLECTIONS_CHECKS

    // (b) VECTOR read-after-Dispose. boolN.Dispose() nulls the DISPOSER's OWN _rec, so the stale read
    // must go through an ALIAS whose _rec still points at the freed record -> Alive is false -> throws.
    [Test]
    public void Vector_ReadAfterDispose_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var v = arena.boolVec(4);
            var alias = v;                 // struct copy -- shares v's (about-to-die) record
            v.Dispose();                   // frees the slot; alias._rec now dangles onto a dead slot
            Assert.Throws<InvalidOperationException>(() => { var _ = alias.Data; });
        }
        finally { arena.Dispose(); }
    }

    // (b) VECTOR read after arena.Clear(). Clear() Frees every live slot but leaves the caller's struct
    // copy alone, so v._rec still points at the now-dead record -> throws. (Clear, NOT Dispose, which
    // would also free the table's chunk memory -- a raw-memory UAF the Alive guard can't observe.)
    [Test]
    public void Vector_ReadAfterArenaClear_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var v = arena.boolVec(4);
            arena.Clear();                 // frees v's slot; v is still a (now stale) handle
            Assert.Throws<InvalidOperationException>(() => { var _ = v.Data; });
        }
        finally { arena.Dispose(); }
    }

    // (b) VECTOR read after arena.ClearTemp(). A TempCopy() lives in the temp pool; ClearTemp() drains
    // it, Freeing the temp slot. The stale temp handle throws; the persistent seed survives.
    [Test]
    public void Vector_ReadAfterClearTemp_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var seed = arena.boolVec(4);
            var tv = seed.TempCopy();      // temp-pool allocation
            arena.ClearTemp();             // frees the temp slot; tv._rec now dangles
            Assert.Throws<InvalidOperationException>(() => { var _ = tv.Data; });
            Assert.AreEqual(4, seed.N);    // seed (persistent) untouched by ClearTemp
        }
        finally { arena.Dispose(); }
    }

    // (c) MATRIX read-after-Dispose / after-Clear / after-ClearTemp: the vector cases mirrored through
    // boolMxN's Alive+generation guard. Same alias-for-dispose requirement (mutable struct).
    [Test]
    public void Matrix_ReadAfterDispose_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var m = arena.boolMat(3, 4);
            var alias = m;
            m.Dispose();
            Assert.Throws<InvalidOperationException>(() => { var _ = alias.Data; });
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Matrix_ReadAfterArenaClear_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var m = arena.boolMat(3, 4);
            arena.Clear();
            Assert.Throws<InvalidOperationException>(() => { var _ = m.Data; });
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Matrix_ReadAfterClearTemp_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var seed = arena.boolMat(3, 3);
            var tm = seed.TempCopy();
            arena.ClearTemp();
            Assert.Throws<InvalidOperationException>(() => { var _ = tm.Data; });
            Assert.AreEqual(9, seed.Length);   // seed (persistent) untouched by ClearTemp
        }
        finally { arena.Dispose(); }
    }

    // (c) THE generation-stamp payoff -- the one case option (c) buys over option (b). A stale handle
    // into a slot freed and then RECYCLED by a fresh, unrelated allocation: Alive alone would read true
    // again for the new occupant, but boolMxN's _gen (stamped at v1's construction) no longer matches
    // the table's CURRENT generation for that slot, so the stale read throws -- while v2 (on the
    // recycled slot) reads & writes fine. v2 reuses v1's EXACT slot via the table's LIFO free list (v1
    // was just freed with nothing freed after it), so the alias's slot is alive again for v2 and it is
    // the GENERATION mismatch, not a dead slot, that trips the guard. (bool allocations aren't tracked
    // in AllocationsCount -- see this file's header -- so unlike the fProxy/long mirror this can't
    // assert the 1 -> 0 -> 1 recycle count; ChunkedRecordTableTests pins the LIFO recycle directly.)
    [Test]
    public void Matrix_StaleGenerationAfterSlotRecycle_OldThrows_NewWorks()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var v1 = arena.boolMat(2, 3);
            var alias = v1;                // captures v1's record pointer AND its generation stamp
            v1.Dispose();                  // frees v1's slot onto the table's LIFO free list

            var v2 = arena.boolMat(2, 3);  // recycles v1's EXACT slot; generation bumped

            // alias: slot is alive again (for v2), but its stamped generation is stale -> throws.
            Assert.Throws<InvalidOperationException>(() => { var _ = alias.Data; });

            // v2: same slot, current generation -> reads/writes cleanly.
            Assert.DoesNotThrow(() => { var _ = v2.Data; });
            v2[0, 0] = true;
            Assert.IsTrue(v2[0, 0]);
        }
        finally { arena.Dispose(); }
    }
#endif
}
