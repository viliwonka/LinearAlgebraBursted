using System;
using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;


// ARENA-LEVEL wiring tests for the intN/intMxN -> ChunkedRecordTable migration
// (docs/rfc-memory-model.md §4 Option A). The table's OWN primitives (chunk stability, free-list
// recycling, generation bumps, double-Free / out-of-range guards) are already covered directly in
// ChunkedRecordTableTests.cs -- these tests prove the COMPOSITION: that Arena.intVec/intMat,
// intN/intMxN.Dispose()/Copy()/TempCopy(), and Arena.Clear()/ClearTemp()/Dispose() drive
// those tables correctly end to end. Mirrors ArenaWiringTests.fProxy.cs one-for-one.
//
// Split, mirroring ChunkedRecordTableTests.cs:
//   * Burst-safe assertions (no exceptions expected) run inside a [BurstCompile] IJob -- this is
//     also the "the record-backed arena actually works under Burst" acceptance check.
//   * Guard/throw assertions use NUnit's Assert.Throws, which cannot be exercised from inside a
//     [BurstCompile] IJob, so they are plain managed [Test]s on the normal C# thread.
public class intArenaWiringTests
{
    [BurstCompile]
    public unsafe struct WiringTestJob : IJob
    {
        public enum TestType
        {
            MultiChunkVectors,
            MultiChunkMatrices,
            TempRecyclingCycles,
            DirectDisposeCountsThenClearVectors,
            DirectDisposeCountsThenClearMatrices,
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
                case TestType.DirectDisposeCountsThenClearVectors: DirectDisposeCountsThenClearVectors(); break;
                case TestType.DirectDisposeCountsThenClearMatrices: DirectDisposeCountsThenClearMatrices(); break;
                case TestType.DisposeThenReallocateRecyclesSlot: DisposeThenReallocateRecyclesSlot(); break;
                default: throw new NotImplementedException();
            }
        }

        // ---- multi-chunk: 300 persistent vectors cross several record-table chunk boundaries ------
        // ChunkedRecordTable chunk capacities double from 8: 8,16,32,64,128,256 (cumulative
        // 8,24,56,120,248,504). 300 allocations land in chunk index 5 -> SIX chunks, crossing FIVE
        // boundaries. Every earlier vector must stay readable with its own distinct sentinels after
        // all the later grows (a relocating/aliasing bug would scramble earlier buffers), and the
        // whole arena must dispose clean.
        void MultiChunkVectors()
        {
            var arena = new Arena(Allocator.Persistent);

            const int n = 300;
            const int len = 3;
            intN* vecs = stackalloc intN[n];

            for (int i = 0; i < n; i++)
            {
                intN v = arena.intVec(len, uninit: true);
                for (int j = 0; j < len; j++)
                    v[j] = (int)(i * len + j);   // distinct sentinel per (alloc, element)
                vecs[i] = v;
            }

            Assert.AreEqual(n, arena.AllocationsCount);

            // Read every vector back AFTER all 300 allocations (and their chunk grows) have happened.
            for (int i = 0; i < n; i++)
            {
                Assert.AreEqual(len, vecs[i].N);
                // == against a (int)-cast expected, not Assert.AreEqual(int, (int)cell): the
                // `(int)` cast off a narrower int type (short) emits no conversion IL, so Burst's
                // Assert.AreEqual interceptor would see a type mismatch (BC1330). Comparing two
                // same-typed values sidesteps it and matches the established int test idiom.
                for (int j = 0; j < len; j++)
                    Assert.IsTrue(vecs[i][j] == (int)(i * len + j));
            }

            arena.Dispose();
            Assert.AreEqual(0, arena.AllocationsCount);
        }

        // ---- multi-chunk: 300 persistent matrices, same argument on the intMatRecords table ----
        void MultiChunkMatrices()
        {
            var arena = new Arena(Allocator.Persistent);

            const int n = 300;
            const int R = 2, C = 2;
            intMxN* mats = stackalloc intMxN[n];

            for (int i = 0; i < n; i++)
            {
                intMxN m = arena.intMat(R, C, uninit: true);
                for (int r = 0; r < R; r++)
                    for (int c = 0; c < C; c++)
                        m[r, c] = (int)(i * (R * C) + r * C + c);
                mats[i] = m;
            }

            Assert.AreEqual(n, arena.AllocationsCount);

            for (int i = 0; i < n; i++)
            {
                Assert.AreEqual(R, mats[i].M_Rows);
                Assert.AreEqual(C, mats[i].N_Cols);
                for (int r = 0; r < R; r++)
                    for (int c = 0; c < C; c++)
                        Assert.IsTrue(mats[i][r, c] == (int)(i * (R * C) + r * C + c));
            }

            arena.Dispose();
            Assert.AreEqual(0, arena.AllocationsCount);
        }

        // ---- temp recycling: allocate temps, ClearTemp(), repeat -- the per-frame ClearTemp loop ---
        // Three cycles of (allocate N temp vecs + N temp mats via TempCopy, verify each buffer works,
        // then ClearTemp). After every ClearTemp the temp record tables must drain back to 0 alive
        // (TempAllocationsCount == 0) while the persistent seed vec/mat are untouched -- proving the
        // temp pool recycles its slots rather than leaking a fresh one per cycle.
        void TempRecyclingCycles()
        {
            var arena = new Arena(Allocator.Persistent);

            const int N = 20;
            var seedV = arena.intVec(4, (int)0);
            var seedM = arena.intMat(3, 3, (int)0);

            int persistentBaseline = arena.AllocationsCount; // seedV + seedM
            Assert.AreEqual(0, arena.TempAllocationsCount);

            for (int cycle = 0; cycle < 3; cycle++)
            {
                for (int i = 0; i < N; i++)
                {
                    // A fresh temp buffer each call; overwrite element 0 with a per-(cycle,i) sentinel
                    // and read it straight back -- proves the recycled slot's buffer is live & writable.
                    intN tv = seedV.TempCopy();
                    tv[0] = (int)(cycle * 1000 + i);
                    Assert.IsTrue(tv[0] == (int)(cycle * 1000 + i));

                    intMxN tm = seedM.TempCopy();
                    tm[0, 0] = (int)(cycle * 1000 + i);
                    Assert.IsTrue(tm[0, 0] == (int)(cycle * 1000 + i));
                }

                // 2*N temp allocations live; persistent count never moved.
                Assert.AreEqual(2 * N, arena.TempAllocationsCount);
                Assert.AreEqual(persistentBaseline, arena.AllocationsCount);

                arena.ClearTemp();

                // Temp pool fully drained; the persistent seeds survive ClearTemp.
                Assert.AreEqual(0, arena.TempAllocationsCount);
                Assert.AreEqual(persistentBaseline, arena.AllocationsCount);
            }

            // Seeds still readable after all the temp churn.
            Assert.AreEqual(4, seedV.N);
            Assert.AreEqual(9, seedM.Length);

            arena.Dispose();
            Assert.AreEqual(0, arena.AllAllocationsCount);
        }

        // ---- direct Dispose() decrements AllocationsCount immediately; Clear() skips the freed slot -
        // Covers both (a) the new immediate-decrement AllocationsCount semantics after a direct
        // intN.Dispose(), and (b) that a following Clear()/Dispose() SKIPS the already-freed slot
        // (IsAlive guard) instead of double-freeing it -- no crash, no leak.
        void DirectDisposeCountsThenClearVectors()
        {
            var arena = new Arena(Allocator.Persistent);

            var a = arena.intVec(4, (int)1);
            var b = arena.intVec(4, (int)2);
            var c = arena.intVec(4, (int)3);
            Assert.AreEqual(3, arena.AllocationsCount);

            b.Dispose();                              // direct dispose of the middle allocation
            Assert.AreEqual(2, arena.AllocationsCount); // immediate decrement (record AliveCount--)

            // The survivors are untouched and still readable.
            Assert.IsTrue(a[0] == (int)1);
            Assert.IsTrue(c[0] == (int)3);

            arena.Clear();                            // must SKIP b's dead slot, free a & c
            Assert.AreEqual(0, arena.AllocationsCount);

            arena.Dispose();                          // clean teardown after a Clear that skipped a slot
            Assert.AreEqual(0, arena.AllocationsCount);
        }

        void DirectDisposeCountsThenClearMatrices()
        {
            var arena = new Arena(Allocator.Persistent);

            var a = arena.intMat(2, 2, (int)1);
            var b = arena.intMat(2, 2, (int)2);
            var c = arena.intMat(2, 2, (int)3);
            Assert.AreEqual(3, arena.AllocationsCount);

            b.Dispose();
            Assert.AreEqual(2, arena.AllocationsCount);

            Assert.IsTrue(a[0, 0] == (int)1);
            Assert.IsTrue(c[0, 0] == (int)3);

            arena.Clear();
            Assert.AreEqual(0, arena.AllocationsCount);

            arena.Dispose();
            Assert.AreEqual(0, arena.AllocationsCount);
        }

        // ---- dispose -> reallocate: the freed slot is recycled via the free-list, and the new -----
        // ---- allocation is a genuinely independent, correctly-content buffer, not a bleed-through --
        // Disposes v1 directly (pushing its slot onto the table's free list), then allocates v2 --
        // which MUST be satisfied from that free slot (AllocationsCount goes 1 -> 0 -> 1, never 2).
        // v2 is allocated zero-initialized (uninit: false) so a bug that skipped re-initializing a
        // recycled record's Data would surface as v1's old sentinel leaking through instead of zero.
        void DisposeThenReallocateRecyclesSlot()
        {
            var arena = new Arena(Allocator.Persistent);

            var v1 = arena.intVec(4, uninit: true);
            for (int j = 0; j < 4; j++) v1[j] = (int)(100 + j);   // sentinel written into v1's buffer
            Assert.AreEqual(1, arena.AllocationsCount);

            v1.Dispose();                                            // frees the slot back to the free-list
            Assert.AreEqual(0, arena.AllocationsCount);

            var v2 = arena.intVec(4, uninit: false);              // should recycle v1's freed slot
            Assert.AreEqual(1, arena.AllocationsCount);
            Assert.AreEqual(4, v2.N);
            for (int j = 0; j < 4; j++)
                Assert.IsTrue(v2[j] == (int)0);                    // zero-init: no stale sentinel bleed

            // And v2 is genuinely its own independent, writable buffer going forward.
            for (int j = 0; j < 4; j++) v2[j] = (int)(200 + j);
            for (int j = 0; j < 4; j++) Assert.IsTrue(v2[j] == (int)(200 + j));

            arena.Dispose();
            Assert.AreEqual(0, arena.AllocationsCount);
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
    // table's double-Free guard, BEFORE it touches the (already-freed) native buffer a second time --
    // this is exactly the silent double-free the migration set out to make impossible.
    //
    // (An alias is REQUIRED to trip the guard: intN.Dispose() nulls its own _rec, so calling
    // Dispose() twice through the SAME variable is a safe no-op, not a throw -- see
    // SameInstanceDoubleDispose_Vector_IsNoOp below.)
    [Test]
    public void AliasedDoubleDispose_Vector_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var v = arena.intVec(8, (int)0);
            var alias = v;                 // struct copy -- shares v's intVecRecord*
            Assert.AreEqual(1, arena.AllocationsCount);

            v.Dispose();                   // frees the slot; v._rec -> null
            Assert.AreEqual(0, arena.AllocationsCount);

            // alias still points at the (now dead) record -> second Free throws.
            Assert.Throws<InvalidOperationException>(() => alias.Dispose());

            // The guard rejected the double-Free before any bookkeeping ran twice.
            Assert.AreEqual(0, arena.AllocationsCount);
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void AliasedDoubleDispose_Matrix_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var m = arena.intMat(3, 4, (int)0);
            var alias = m;
            Assert.AreEqual(1, arena.AllocationsCount);

            m.Dispose();
            Assert.AreEqual(0, arena.AllocationsCount);

            Assert.Throws<InvalidOperationException>(() => alias.Dispose());
            Assert.AreEqual(0, arena.AllocationsCount);
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
            var v = arena.intVec(8, (int)0);
            v.Dispose();
            Assert.DoesNotThrow(() => v.Dispose());
            Assert.AreEqual(0, arena.AllocationsCount);
        }
        finally { arena.Dispose(); }
    }

    // STANDALONE Copy()/TempCopy() throw contract. A standalone (Allocator-ctor) vector has a null
    // record, so Copy()/TempCopy() have no owning arena to allocate through: they must throw the SAME
    // InvalidOperationException as the pre-migration code did -- NOT a NullReferenceException from
    // dereferencing a null record/core.
    [Test]
    public void StandaloneVector_CopyAndTempCopy_Throw()
    {
        var v = new intN(4, Allocator.Temp);
        try
        {
            Assert.Throws<InvalidOperationException>(() => v.Copy());
            Assert.Throws<InvalidOperationException>(() => v.TempCopy());
        }
        finally { v.Dispose(); }
    }

    [Test]
    public void StandaloneMatrix_CopyAndTempCopy_Throw()
    {
        var m = new intMxN(3, 3, Allocator.Temp);
        try
        {
            Assert.Throws<InvalidOperationException>(() => m.Copy());
            Assert.Throws<InvalidOperationException>(() => m.TempCopy());
        }
        finally { m.Dispose(); }
    }
}
