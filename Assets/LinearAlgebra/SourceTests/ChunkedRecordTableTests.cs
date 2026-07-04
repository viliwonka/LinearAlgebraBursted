using System;
using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Concrete (NOT codegen'd) tests for LinearAlgebra.ChunkedRecordTable<TRecord> -- the Stage-B
// pointer-stable arena record table (docs/rfc-memory-model.md §4 Option A / A1, §6.1, §7 step 2).
// The table is internal, hand-authored, and generic over an `unmanaged` record type, so -- like
// ArenaLayoutTests.cs / UIntTypeTests.cs -- these are hand-written rather than expanded from an
// fProxy/iProxy template. Visibility into the internal type is granted to this test assembly via
// [assembly: InternalsVisibleTo("BurstLinearAlgebra.Tests")] (TemplateSource/AssemblyInfo.cs).
//
// This stage is ADDITIVE ONLY -- nothing in ArenaCore or any math struct uses the table yet, so
// every test here constructs and disposes a table directly, with no Arena involved.
//
// Every test case below (bar the single non-Burst sanity check at the bottom, mirroring
// UIntTypeTests.cs's tail section) runs inside a [BurstCompile] IJob via TestsJob.Run(), which is
// also the "Burst usability" acceptance criterion: the table must actually compile and behave
// correctly under Burst, not just from the managed test thread.
public class ChunkedRecordTableTests
{
    // Minimal unmanaged payload used as TRecord for these tests: a sentinel value (to detect any
    // address/content corruption) plus a tag (independent cross-check, e.g. against the slot index).
    private struct TestRecord
    {
        public long Sentinel;
        public int Tag;
    }

    [BurstCompile]
    public unsafe struct TestsJob : IJob
    {
        public enum TestType
        {
            AddressStabilityAcrossChunkBoundaries,
            FreeListRecyclingDoesNotGrowDirectory,
            GenerationBumpsOnFreeAndPersistsAcrossReuse,
            RecycledSlotContentSurvives,
            InterleavedAllocFreeStress,
            BasicAllocateResolveFreeDispose,
        }

        public TestType Type;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.AddressStabilityAcrossChunkBoundaries: AddressStabilityAcrossChunkBoundaries(); break;
                case TestType.FreeListRecyclingDoesNotGrowDirectory: FreeListRecyclingDoesNotGrowDirectory(); break;
                case TestType.GenerationBumpsOnFreeAndPersistsAcrossReuse: GenerationBumpsOnFreeAndPersistsAcrossReuse(); break;
                case TestType.RecycledSlotContentSurvives: RecycledSlotContentSurvives(); break;
                case TestType.InterleavedAllocFreeStress: InterleavedAllocFreeStress(); break;
                case TestType.BasicAllocateResolveFreeDispose: BasicAllocateResolveFreeDispose(); break;
                default: throw new NotImplementedException();
            }
        }

        // ---- address stability across chunk boundaries ----------------------------------------
        // Chunk capacities double from 8: 8, 16, 32, 64, 128, ... (cumulative 8, 24, 56, 120, 248).
        // Allocating 200 records crosses FOUR chunk boundaries (into chunk index 4, since
        // 120 < 200 <= 248) -- so this exercises multiple chunk grows, not just the first one.
        void AddressStabilityAcrossChunkBoundaries()
        {
            var table = new ChunkedRecordTable<TestRecord>();
            table.Init(Allocator.Persistent);

            const int n = 200;
            int* slots = stackalloc int[n];
            long* addrs = stackalloc long[n];

            for (int i = 0; i < n; i++)
            {
                TestRecord* p = table.Allocate(out int slot);
                p->Sentinel = 1_000_000L + i;
                p->Tag = i;
                slots[i] = slot;
                addrs[i] = (long)p;
            }

            // 8+16+32+64=120 < 200 <= 120+128=248 -> the 5th chunk (index 4) was needed.
            Assert.AreEqual(5, table.ChunkCount);
            Assert.AreEqual(n, table.Count);
            Assert.AreEqual(n, table.AliveCount);

            // Every earlier-returned pointer must resolve to the SAME address, with sentinel
            // content untouched by the later chunk grows (a relocating bug would move the address
            // and/or scramble the content of earlier records).
            for (int i = 0; i < n; i++)
            {
                TestRecord* p2 = table.Resolve(slots[i]);
                Assert.IsTrue((long)p2 == addrs[i]);
                Assert.IsTrue(p2->Sentinel == 1_000_000L + i);
                Assert.IsTrue(p2->Tag == i);
                Assert.IsTrue(table.IsAlive(slots[i]));
            }

            table.Dispose();
        }

        // ---- free-list recycling ---------------------------------------------------------------
        void FreeListRecyclingDoesNotGrowDirectory()
        {
            var table = new ChunkedRecordTable<TestRecord>();
            table.Init(Allocator.Persistent);

            // Fill exactly the first chunk (capacity 8).
            int* slots = stackalloc int[8];
            for (int i = 0; i < 8; i++)
            {
                TestRecord* p = table.Allocate(out int slot);
                p->Tag = i;
                slots[i] = slot;
            }
            Assert.AreEqual(1, table.ChunkCount);
            Assert.AreEqual(8, table.Count);
            Assert.AreEqual(8, table.AliveCount);

            // Free three of them (indices 2, 4, 6 in allocation order).
            table.Free(slots[2]);
            table.Free(slots[4]);
            table.Free(slots[6]);
            Assert.AreEqual(5, table.AliveCount);
            Assert.AreEqual(8, table.Count); // high-water mark is untouched by Free

            int chunkCountBeforeReuse = table.ChunkCount;
            int highWaterBeforeReuse = table.Count;

            // Re-allocate three more: must be satisfied ENTIRELY from the free list -- no new slot
            // carved (Count unchanged) and no new chunk Malloc'd (ChunkCount unchanged).
            TestRecord* r0 = table.Allocate(out int reused0);
            TestRecord* r1 = table.Allocate(out int reused1);
            TestRecord* r2 = table.Allocate(out int reused2);

            Assert.AreEqual(chunkCountBeforeReuse, table.ChunkCount);
            Assert.AreEqual(highWaterBeforeReuse, table.Count);
            Assert.AreEqual(8, table.AliveCount);

            // The free list is a stack (LIFO): the most-recently-freed slot is the first reused.
            Assert.AreEqual(slots[6], reused0);
            Assert.AreEqual(slots[4], reused1);
            Assert.AreEqual(slots[2], reused2);

            table.Dispose();
        }

        // ---- generation semantics ---------------------------------------------------------------
        void GenerationBumpsOnFreeAndPersistsAcrossReuse()
        {
            var table = new ChunkedRecordTable<TestRecord>();
            table.Init(Allocator.Persistent);

            TestRecord* p = table.Allocate(out int slot);
            Assert.AreEqual(0, table.GetGeneration(slot)); // fresh, never-freed slot starts at 0
            Assert.IsTrue(table.IsAlive(slot));

            table.Free(slot);
            Assert.AreEqual(1, table.GetGeneration(slot)); // Free bumps it
            Assert.IsFalse(table.IsAlive(slot));

            // Only one dead slot exists, so the next Allocate MUST recycle it.
            TestRecord* p2 = table.Allocate(out int slot2);
            Assert.AreEqual(slot, slot2);
            Assert.AreEqual(1, table.GetGeneration(slot2)); // Allocate does not reset the generation
            Assert.IsTrue(table.IsAlive(slot2));

            table.Free(slot2);
            Assert.AreEqual(2, table.GetGeneration(slot2)); // bumps again on the second free

            table.Dispose();
        }

        // ---- recycled-slot content contract --------------------------------------------------------
        // Pins the documented "no poisoning on Free" contract (Allocate's XML doc: "a recycled slot
        // retains whatever its previous occupant left behind"). Without this pinned, a future change
        // that starts zeroing/poisoning a freed slot's Record would silently change behavior with no
        // test catching it.
        void RecycledSlotContentSurvives()
        {
            var table = new ChunkedRecordTable<TestRecord>();
            table.Init(Allocator.Persistent);

            TestRecord* p = table.Allocate(out int slot);
            p->Sentinel = 123_456_789L;
            p->Tag = 77;

            table.Free(slot);
            Assert.AreEqual(1, table.GetGeneration(slot)); // bumped by Free

            // Only one dead slot exists, so this MUST recycle the same slot (LIFO free list).
            TestRecord* p2 = table.Allocate(out int slot2);
            Assert.AreEqual(slot, slot2);
            Assert.AreEqual(1, table.GetGeneration(slot2)); // Allocate does not touch Generation

            // The sentinel written before Free is STILL THERE -- Free does not clear/poison Record.
            Assert.AreEqual(123_456_789L, p2->Sentinel);
            Assert.AreEqual(77, p2->Tag);
            Assert.IsTrue(p == p2); // same address too (same physical slot)

            table.Dispose();
        }

        // ---- interleaved alloc/free stress -------------------------------------------------------
        // Deterministic pseudo-random (fixed seed) sequence of 500 alloc/free operations against a
        // parallel "model" (plain stackalloc arrays -- Burst has no managed Dictionary/List) that
        // tracks, per occupied model slot, the table slot index, the expected pointer address, and
        // the expected sentinel content. NOTE on scope: maxAlive=64 caps concurrent occupancy, which
        // in turn caps the table's high-water mark (Count) at 64 -- so this exercises only ~4 chunk
        // grows (capacities 8/16/32/64), NOT "hundreds" of them; chunk-boundary-crossing coverage is
        // AddressStabilityAcrossChunkBoundaries's job. What THIS test actually covers is address/
        // content stability across MANY (hundreds of) free-list recycle cycles under a stochastic,
        // non-LIFO-friendly interleaving -- a much less predictable access pattern than the other
        // tests' hand-scripted sequences. Invariants checked throughout: AliveCount always agrees
        // with the model's live count; and at the end, every still-alive entry's address and content
        // are exactly what was recorded at allocation time (proves no address ever moved and no
        // cross-talk between slots occurred across that churn).
        void InterleavedAllocFreeStress()
        {
            var table = new ChunkedRecordTable<TestRecord>();
            table.Init(Allocator.Persistent);

            const int maxAlive = 64;
            const int ops = 500;

            int* modelTableSlot = stackalloc int[maxAlive];
            long* modelAddr = stackalloc long[maxAlive];
            long* modelSentinel = stackalloc long[maxAlive];
            bool* occupied = stackalloc bool[maxAlive];
            for (int m = 0; m < maxAlive; m++) occupied[m] = false;

            int occupiedCount = 0;
            long nextSentinel = 1;

            var rng = new Unity.Mathematics.Random(0x9E3779B9u); // fixed seed -- deterministic, no Date/time source

            for (int op = 0; op < ops; op++)
            {
                bool doAllocate;
                if (occupiedCount == 0) doAllocate = true;
                else if (occupiedCount >= maxAlive) doAllocate = false;
                else doAllocate = rng.NextFloat() < 0.6f; // mild bias toward growing, still frees a lot

                if (doAllocate)
                {
                    int m = -1;
                    for (int k = 0; k < maxAlive; k++) { if (!occupied[k]) { m = k; break; } }

                    TestRecord* p = table.Allocate(out int tableSlot);
                    long sentinel = nextSentinel++;
                    p->Sentinel = sentinel;
                    p->Tag = tableSlot;

                    modelTableSlot[m] = tableSlot;
                    modelAddr[m] = (long)p;
                    modelSentinel[m] = sentinel;
                    occupied[m] = true;
                    occupiedCount++;
                }
                else
                {
                    int pick = rng.NextInt(0, occupiedCount);
                    int m = -1, seen = 0;
                    for (int k = 0; k < maxAlive; k++)
                    {
                        if (!occupied[k]) continue;
                        if (seen == pick) { m = k; break; }
                        seen++;
                    }

                    table.Free(modelTableSlot[m]);
                    occupied[m] = false;
                    occupiedCount--;
                }

                Assert.AreEqual(occupiedCount, table.AliveCount);
            }

            int finalAliveCheck = 0;
            for (int m = 0; m < maxAlive; m++)
            {
                if (!occupied[m]) continue;
                finalAliveCheck++;

                TestRecord* p = table.Resolve(modelTableSlot[m]);
                Assert.IsTrue((long)p == modelAddr[m]);
                Assert.IsTrue(p->Sentinel == modelSentinel[m]);
                Assert.IsTrue(table.IsAlive(modelTableSlot[m]));
            }
            Assert.AreEqual(occupiedCount, finalAliveCheck);
            Assert.AreEqual(occupiedCount, table.AliveCount);

            table.Dispose();
        }

        // ---- basic smoke (still inside Burst) ---------------------------------------------------
        void BasicAllocateResolveFreeDispose()
        {
            var table = new ChunkedRecordTable<TestRecord>();
            table.Init(Allocator.Persistent);

            Assert.AreEqual(0, table.Count);
            Assert.AreEqual(0, table.AliveCount);
            Assert.AreEqual(0, table.ChunkCount);

            TestRecord* p = table.Allocate(out int slot);
            Assert.AreEqual(0, slot);
            Assert.AreEqual(1, table.Count);
            Assert.AreEqual(1, table.AliveCount);
            Assert.AreEqual(1, table.ChunkCount);

            // A freshly-carved (never-recycled) slot reads back as all-zero.
            Assert.AreEqual(0L, p->Sentinel);
            Assert.AreEqual(0, p->Tag);

            p->Sentinel = 42L;
            Assert.IsTrue(table.Resolve(slot)->Sentinel == 42L);

            table.Free(slot);
            Assert.AreEqual(0, table.AliveCount);
            Assert.AreEqual(1, table.Count); // high-water mark unaffected

            table.Dispose();
        }
    }

    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(TestsJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void Test(TestsJob.TestType type)
    {
        new TestsJob() { Type = type }.Run();
    }

    // ---- Non-Burst sanity outside a job (mirrors ArenaLayoutTests.cs / UIntTypeTests.cs style) ----

    [Test]
    public unsafe void ChunkedRecordTable_AllocateAndDispose_OutsideJob()
    {
        var table = new ChunkedRecordTable<TestRecord>();
        table.Init(Allocator.Persistent);
        try
        {
            TestRecord* p = table.Allocate(out int slot);
            p->Sentinel = 7L;

            Assert.AreEqual(1, table.AliveCount);
            Assert.IsTrue(table.IsAlive(slot));
            Assert.AreEqual(7L, table.Resolve(slot)->Sentinel);

            table.Free(slot);
            Assert.AreEqual(0, table.AliveCount);
            Assert.IsFalse(table.IsAlive(slot));
        }
        finally { table.Dispose(); }
    }

    // ---- Guard / throw tests -------------------------------------------------------------------
    // NUnit's Assert.Throws can't be asserted from inside a [BurstCompile] IJob (see
    // fProxyDotRefGuardTests.cs's header comment: "exceptions can't be asserted there"), so -- like
    // that file -- every guard test below is a plain managed [Test], run on the normal C# thread
    // outside a job, exercising the table directly.

    [Test]
    public unsafe void Free_DoubleFree_Throws()
    {
        var table = new ChunkedRecordTable<TestRecord>();
        table.Init(Allocator.Persistent);
        try
        {
            table.Allocate(out int slot);
            table.Free(slot);
            Assert.Throws<InvalidOperationException>(() => table.Free(slot));

            // AliveCount must NOT go negative -- the guard rejects the double-Free before any of
            // its bookkeeping (free-list push, AliveCount--) runs a second time.
            Assert.AreEqual(0, table.AliveCount);
        }
        finally { table.Dispose(); }
    }

    [Test]
    public unsafe void Free_NeverAllocatedIndex_Throws()
    {
        var table = new ChunkedRecordTable<TestRecord>();
        table.Init(Allocator.Persistent);
        try
        {
            table.Allocate(out _); // Count becomes 1; index 1 was never handed out
            Assert.AreEqual(1, table.Count);
            Assert.Throws<ArgumentOutOfRangeException>(() => table.Free(1));
        }
        finally { table.Dispose(); }
    }

    // Resolve/Free/IsAlive/GetGeneration must all reject idx == Count, idx far past Count, and a
    // negative idx -- the single unsigned bounds check in SlotPtr covers all three shapes.
    [Test]
    public unsafe void OutOfRangeAccess_AllAccessors_Throw()
    {
        var table = new ChunkedRecordTable<TestRecord>();
        table.Init(Allocator.Persistent);
        try
        {
            table.Allocate(out _);
            table.Allocate(out _);
            Assert.AreEqual(2, table.Count);

            int[] badIndices = { table.Count, table.Count + 1000, -1, int.MinValue };
            foreach (int idx in badIndices)
            {
                int captured = idx; // avoid capturing the loop variable across the four lambdas below
                Assert.Throws<ArgumentOutOfRangeException>(() => table.Resolve(captured));
                Assert.Throws<ArgumentOutOfRangeException>(() => table.IsAlive(captured));
                Assert.Throws<ArgumentOutOfRangeException>(() => table.GetGeneration(captured));
                Assert.Throws<ArgumentOutOfRangeException>(() => table.Free(captured));
            }
        }
        finally { table.Dispose(); }
    }

    [Test]
    public void Init_CalledTwice_Throws()
    {
        var table = new ChunkedRecordTable<TestRecord>();
        table.Init(Allocator.Persistent);
        try
        {
            Assert.Throws<InvalidOperationException>(() => table.Init(Allocator.Persistent));
        }
        finally { table.Dispose(); }
    }

    [Test]
    public unsafe void Dispose_IsIdempotent()
    {
        var table = new ChunkedRecordTable<TestRecord>();
        table.Init(Allocator.Persistent);
        table.Allocate(out _);

        table.Dispose();
        Assert.DoesNotThrow(() => table.Dispose()); // second call: safe no-op, not a double-free
        Assert.DoesNotThrow(() => table.Dispose()); // and a third, for good measure
    }

    [Test]
    public unsafe void Operations_AfterDispose_Throw()
    {
        var table = new ChunkedRecordTable<TestRecord>();
        table.Init(Allocator.Persistent);
        table.Allocate(out int slot);
        table.Dispose();

        // Every entry point must reject a disposed table rather than silently reinitializing it
        // (Allocate) or resolving through freed memory (Resolve/Free/IsAlive/GetGeneration).
        Assert.Throws<InvalidOperationException>(() => table.Allocate(out _));
        Assert.Throws<InvalidOperationException>(() => table.Resolve(slot));
        Assert.Throws<InvalidOperationException>(() => table.IsAlive(slot));
        Assert.Throws<InvalidOperationException>(() => table.GetGeneration(slot));
        Assert.Throws<InvalidOperationException>(() => table.Free(slot));
    }

    [Test]
    public unsafe void Operations_OnNeverInitialized_Throw()
    {
        var table = new ChunkedRecordTable<TestRecord>();
        // Init() never called.
        Assert.Throws<InvalidOperationException>(() => table.Allocate(out _));
        Assert.Throws<InvalidOperationException>(() => table.Resolve(0));
        Assert.Throws<InvalidOperationException>(() => table.Free(0));
    }
}
