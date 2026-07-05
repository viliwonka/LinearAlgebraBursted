using System;
using System.Threading;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Collections;

// Concrete (NOT codegen'd) tests for the Arena concurrency/use-after-dispose guards added in the
// arena-concurrency-guards change (ArenaCore.EnterMutation/ExitMutation: an Interlocked `_busy`
// tripwire + an AtomicSafetyHandle, both gated behind ENABLE_UNITY_COLLECTIONS_CHECKS -- see
// Arena.cs's ArenaCore field docs and the "Threading contract" paragraph on Arena's class doc).
//
// Why hand-authored (like ArenaLayoutTests.cs / ChunkedRecordTableTests.cs): Arena is not a
// proxy-typed template, and -- more fundamentally -- these tests spin up managed
// System.Threading.Thread instances and assert exceptions with NUnit's Assert.*, neither of which
// can live inside a [BurstCompile] IJob (Burst has no managed threads and no exception assertions;
// see fProxyDotRefGuardTests.cs / ChunkedRecordTableTests.cs's guard-test headers). So every test
// here is a plain managed [Test] on the normal C# thread.
//
// ---- Why there is deliberately NO "two IJobs scheduled against one Arena is rejected" test -------
// Unity's job system can automatically reject a job that captures a [NativeContainer] already in use
// by another scheduled job -- but ONLY if the container type carries the [NativeContainer] attribute
// and wires its AtomicSafetyHandle into the job-reflection protocol (the handle must be a field on
// the very struct the job captures by value). Arena is DELIBERATELY not a [NativeContainer]: it is a
// bare pointer-sized handle over a heap-resident ArenaCore, and the safety handle lives on the CORE,
// not on the captured Arena struct, precisely so that sizeof(Arena) stays one pointer wide
// (ArenaLayoutTests.Arena_IsPointerSized) and does not grow under ENABLE_UNITY_COLLECTIONS_CHECKS.
// That design decision (documented, pending user review, in Arena.cs's class doc + ArenaCore.Safety's
// field doc) means schedule-time two-jobs rejection is intentionally out of scope -- the guards here
// provide RUN-TIME detection (a loud throw when two mutating calls actually overlap) instead of
// SCHEDULE-TIME prevention. Hence no such test exists; this comment is the record of why.
public class ArenaConcurrencyTests
{
    // ==========================================================================================
    // 1. RACE TRIPWIRE -- two unsynchronized managed threads hammering factory calls on ONE shared
    //    Arena must trip the interlocked `_busy` guard at least once (proving a real overlap was
    //    detected and turned into a throw rather than silently corrupting the record tables).
    //
    //    Constraints honored (per the guard's documented known gaps):
    //      * We race ONLY the factory entry points (floatVec/floatMat) -- both are guarded terminal
    //        factories. We do NOT race per-instance buffer .Dispose() (deliberately unguarded --
    //        see ArenaCore._busy's doc) and we do NOT race arena.Dispose() itself.
    //      * All allocations are left to accumulate and are freed in one shot by the single-threaded
    //        arena.Dispose() AFTER both threads Join() -- nothing is disposed concurrently.
    //
    //    Reliability: the tripwire is an atomic Interlocked.CompareExchange, so whenever two calls
    //    are genuinely in flight at the same instant, exactly one throws -- it is not probabilistic
    //    given an overlap. The only variable is whether overlaps OCCUR, which a shared start-gate
    //    (both threads released together) plus a generous iteration count makes overwhelmingly
    //    likely: two threads each doing tens of thousands of tiny Allocate calls concurrently for
    //    many milliseconds overlap thousands of times. If this ever proves flaky, widen the window
    //    (raise Iterations) rather than deleting it.
    // ==========================================================================================
    const int RaceIterations = 30000;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
    [Test]
    public void ConcurrentFactoryAccess_TripsBusyGuard()
    {
        var arena = new Arena(Allocator.Persistent);

        using (var startGate = new ManualResetEventSlim(false))
        {
            int tripCount = 0;          // times the busy-tripwire fired (Interlocked)
            Exception unexpected = null; // first non-tripwire exception, if any (Interlocked)

            // One thread hammers vectors, the other matrices -- different record tables, but both
            // guarded by the SAME per-core `_busy` flag, so an overlap between them still trips it.
            void Hammer(bool matrices)
            {
                startGate.Wait();
                for (int i = 0; i < RaceIterations; i++)
                {
                    try
                    {
                        if (matrices) arena.floatMat(4);
                        else arena.floatVec(4);
                    }
                    catch (InvalidOperationException)
                    {
                        // The documented tripwire message ("concurrent mutating access detected").
                        Interlocked.Increment(ref tripCount);
                    }
                    catch (Exception e)
                    {
                        // Anything else (e.g. a corrupted table surfacing as a different exception)
                        // is a real failure -- capture the first and stop this thread.
                        Interlocked.CompareExchange(ref unexpected, e, null);
                        break;
                    }
                }
            }

            var tVec = new Thread(() => Hammer(false)) { Name = "ArenaRace-Vec" };
            var tMat = new Thread(() => Hammer(true)) { Name = "ArenaRace-Mat" };

            tVec.Start();
            tMat.Start();
            startGate.Set();    // release both together to maximize overlap
            tVec.Join();
            tMat.Join();

            // Single-threaded teardown, after BOTH threads have joined -- never raced.
            arena.Dispose();

            Assert.IsNull(unexpected,
                $"A non-tripwire exception escaped the race (would indicate the guard failed to " +
                $"serialize and a record table was corrupted): {unexpected}");
            Assert.Greater(tripCount, 0,
                "Expected the Arena concurrency tripwire to fire at least once across " +
                $"{RaceIterations} overlapping iterations per thread, but it never did. Either the " +
                "guard is not arming, or (less likely) the two threads never overlapped -- if the " +
                "latter recurs, raise RaceIterations to widen the contention window.");
        }
    }
#else
    [Test]
    public void ConcurrentFactoryAccess_TripsBusyGuard()
    {
        Assert.Ignore("Arena concurrency tripwire is compiled out without ENABLE_UNITY_COLLECTIONS_CHECKS.");
    }
#endif

    // ==========================================================================================
    // 2. SINGLE-THREADED NO-FALSE-POSITIVE pins -- the guard restructuring split several methods
    //    into "guarded public wrapper -> unguarded core" pairs and added forwarding factory
    //    overloads, specifically so a legitimate single-threaded call never nests EnterMutation()
    //    on itself and trips its OWN tripwire. Most of these nested paths are already exercised
    //    incidentally by the green suite (e.g. floatHilbertMat uses the floatMat(dim) forwarder;
    //    operators use TempCopy; InitTest/ArenaLayoutTests drive Clear/Dispose). These are thin,
    //    explicit pins for the exact restructured entry points, asserting a normal call does NOT
    //    throw -- so a future regression that made a wrapper re-enter its own guard fails HERE with
    //    a clear name, not diffusely across the suite.
    // ==========================================================================================

    // floatMat(int dim) is a pure forwarding wrapper onto floatMat(dim, dim, uninit); only the
    // (rows, cols) terminal overload holds the guard. A single call must not trip anything.
    [Test]
    public void SquareMatForwarder_DoesNotFalseTrip()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            Assert.DoesNotThrow(() => { var m = arena.floatMat(5); m[0, 0] = 1f; });
            Assert.DoesNotThrow(() => { var m = arena.floatMat(5, true); });
        }
        finally { arena.Dispose(); }
    }

    // TempCopy() forwards into the internal floatTempMat(in orig) factory, which is guarded. A
    // single-threaded copy must pass through cleanly and land in the temp pool.
    [Test]
    public void TempCopy_DoesNotFalseTrip()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var m = arena.floatMat(4, 4, 2f);
            floatMxN t = default;
            Assert.DoesNotThrow(() => { t = m.TempCopy(); });
            Assert.AreEqual(1, arena.TempAllocationsCount);
            Assert.IsTrue(arena.isTemp(in t));
        }
        finally { arena.Dispose(); }
    }

    // Clear() acquires the guard ONCE and calls the unguarded ClearCore()/ClearTempCore() directly
    // (routing back through the public guarded ClearTemp() would nest and self-trip). Populate both
    // the persistent and temp pools first so the clear loops actually walk live records.
    [Test]
    public void Clear_WithLiveAndTempAllocations_DoesNotFalseTrip()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var a = arena.floatMat(4, 4, 1f);
            var b = arena.floatVec(8, 3f);
            var t = a.TempCopy();                 // seed the temp pool too
            Assert.Greater(arena.AllAllocationsCount, 0);

            Assert.DoesNotThrow(() => arena.Clear());
            Assert.AreEqual(0, arena.AllAllocationsCount);

            // ...and the guard is properly released -- a follow-up allocate/clear still works.
            Assert.DoesNotThrow(() => { var c = arena.floatVec(2); });
            Assert.DoesNotThrow(() => arena.ClearTemp());
        }
        finally { arena.Dispose(); }
    }

    // ClearTemp() is the sibling guarded wrapper over ClearTempCore(); a normal call on a populated
    // temp pool must clear it without self-tripping and must leave persistent allocations alone.
    [Test]
    public void ClearTemp_DoesNotFalseTrip_AndSparesPersistent()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var persistent = arena.floatMat(3, 3, 1f);
            var temp = persistent.TempCopy();
            Assert.AreEqual(1, arena.AllocationsCount);
            Assert.AreEqual(1, arena.TempAllocationsCount);

            Assert.DoesNotThrow(() => arena.ClearTemp());

            Assert.AreEqual(1, arena.AllocationsCount, "ClearTemp must not touch persistent allocations");
            Assert.AreEqual(0, arena.TempAllocationsCount);
        }
        finally { arena.Dispose(); }
    }

    // Dispose() holds the guard once and calls ClearCore()/ClearTempCore() directly (same reason as
    // Clear()). Exercise it on a fully-populated arena -- persistent + temp + a Pivot/Indices buffer
    // (the value-copy-tracked families) + an ArenaExtensions helper allocation -- to confirm the
    // whole guarded teardown runs without a self-trip.
    [Test]
    public void Dispose_WithMixedAllocations_DoesNotFalseTrip()
    {
        var arena = new Arena(Allocator.Persistent);

        var m = arena.floatIdentityMat(4);        // ArenaExtensions helper -> floatMat(N,N)
        var v = arena.floatVec(6, 1f);
        var t = m.TempCopy();                      // temp pool
        var p = arena.Pivot(4);                    // value-copy-tracked family
        var idx = arena.Indices(4);                // value-copy-tracked family
        Assert.Greater(arena.AllAllocationsCount, 0);

        Assert.DoesNotThrow(() => arena.Dispose());

        // Post-dispose the (now null-cored) handle reports empty, as the accessors contract.
        Assert.AreEqual(0, arena.AllAllocationsCount);
    }

    // An ArenaExtensions convenience method (floatIdentityMat -> floatMat -> guarded terminal
    // factory) called sequentially several times: each fully enters and exits its own guard before
    // the next starts (never nested), so a batch never trips.
    [Test]
    public void ArenaExtensionsHelpers_SequentialCalls_DoNotFalseTrip()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            Assert.DoesNotThrow(() =>
            {
                for (int i = 0; i < 16; i++)
                {
                    var id = arena.floatIdentityMat(3);
                    var rnd = arena.floatRandomVec(4, -1f, 1f, seed: (uint)(i + 1));
                    var lin = arena.floatLinVec(5, 0f, 1f);
                }
            });
            Assert.AreEqual(48, arena.AllocationsCount); // 3 persistent allocations x 16 iterations
        }
        finally { arena.Dispose(); }
    }

    // ==========================================================================================
    // 3. USE-AFTER-DISPOSE of the ARENA -- the AtomicSafetyHandle created in ArenaCore.Init and
    //    released in ArenaCore.Dispose is checked (CheckWriteAndThrow) at the top of every guarded
    //    factory. Using the arena after disposal must surface a clear THROW, not undefined behavior.
    //
    //    IMPORTANT nuance (read Arena.cs): arena.Dispose() nulls the AUTHORITATIVE handle's _core,
    //    so calling a factory on that SAME handle would dereference a null _core -- an uncatchable
    //    access violation, because the factory bodies (unlike Clear/Pivot/Indices) do NOT null-check
    //    _core. The AtomicSafetyHandle guard is designed to catch use-after-dispose through a
    //    surviving ALIASED COPY of the handle (whose _core still points at the freed core) -- exactly
    //    the "a live handle used after Dispose released it throws" case in ArenaCore.Dispose's doc.
    //    So this test disposes via one handle and pokes a copy.
    //
    //    Best-effort caveat (documented in ArenaCore.Dispose): reading Safety through the freed core
    //    block is technically UB and only reliable while those bytes are intact -- which they are
    //    here, since no allocation happens between Dispose() and the poke. We therefore assert that
    //    SOME exception is surfaced (Assert.Catch, not a fixed type) -- empirically Unity's
    //    AtomicSafetyHandle.CheckWriteAndThrow raises InvalidOperationException for a released handle.
    // ==========================================================================================
#if ENABLE_UNITY_COLLECTIONS_CHECKS
    [Test]
    public void FactoryAfterDispose_ThroughAliasedHandle_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        Arena alias = arena;                 // by-value handle copy: shares the same ArenaCore*
        var probe = arena.floatVec(2);       // force Init/Safety to exist

        arena.Dispose();                     // releases Safety + frees core; nulls arena._core only

        // The alias still holds the (now-freed) core pointer, so its factory reaches the released
        // AtomicSafetyHandle and CheckWriteAndThrow fires instead of silently mutating dead state.
        Exception thrown = Assert.Catch(() => { var v = alias.floatVec(4); },
            "Use-after-dispose through an aliased Arena handle must throw via the AtomicSafetyHandle " +
            "guard, not silently operate on the freed core.");
        Assert.IsNotNull(thrown);
    }
#else
    [Test]
    public void FactoryAfterDispose_ThroughAliasedHandle_Throws()
    {
        Assert.Ignore("Arena use-after-dispose safety handle is compiled out without ENABLE_UNITY_COLLECTIONS_CHECKS.");
    }
#endif

    // Unlike the aliased case above, the AUTHORITATIVE handle has its _core nulled by Dispose(),
    // so the factories' unconditional `_core == null` guard fires -- a clean, deterministic throw
    // in EVERY build config (not checks-gated), matching Clear/ClearTemp/Pivot/Indices.
    [Test]
    public void FactoryAfterDispose_AuthoritativeHandle_ThrowsCleanly()
    {
        var arena = new Arena(Allocator.Persistent);
        var probe = arena.floatVec(2);
        arena.Dispose();

        Assert.Throws<InvalidOperationException>(() => { var v = arena.floatVec(4); },
            "A factory call on the disposed authoritative Arena handle must throw the " +
            "not-initialized guard, not null-deref.");
    }
}
