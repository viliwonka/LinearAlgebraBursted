using LinearAlgebra;

using NUnit.Framework;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

// Concrete (NOT codegen'd) tests for the Arena memory-model split (docs/rfc-memory-model.md §4
// Option A / §6.0). Arena itself is not proxy-typed, so these are hand-authored rather than
// generated. They pin the invariants that make failure mode 2 (the dangling-arena-pointer bug)
// structurally impossible: Arena is now a thin single-pointer handle over a heap-Malloc'd
// ArenaCore, so copying an Arena (including compiler-inserted defensive copies of `in Arena`
// parameters) copies only the ArenaCore* value and still resolves to the same live core.
public class ArenaLayoutTests
{
    // Proves the ArenaCore split actually happened (not just a rename): Arena must have shrunk to a
    // single pointer width. If Arena still held its tracking UnsafeLists inline it would be far
    // larger than a pointer, and the old raw-address capture (`Arena* _arenaPtr`) would still be a
    // dangling-pointer risk.
    [Test]
    public unsafe void Arena_IsPointerSized()
    {
        Assert.AreEqual(UnsafeUtility.SizeOf<System.IntPtr>(), UnsafeUtility.SizeOf<Arena>());
    }

    // Copying an Arena handle by value must alias the SAME live core: an allocation tracked through
    // one copy is visible (and disposable) through the other. This is the property that lets
    // defensive copies of `in Arena` params be harmless.
    [Test]
    public void Arena_ValueCopy_SharesSameCore()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            Arena copy = arena;                 // by-value handle copy (copies only the ArenaCore*)

            var v = copy.floatVec(4);           // allocate through the copy
            v[0] = 1; v[1] = 2; v[2] = 3; v[3] = 4;

            // The original handle sees the allocation and reports the buffer as persistent -- both
            // handles resolve to the one live ArenaCore.
            Assert.AreEqual(1, arena.AllocationsCount);
            Assert.IsTrue(arena.isPersistent(in v));
        }
        finally { arena.Dispose(); }
    }

    // A disposed arena's AllocationsCount must read 0 (not crash): Dispose() nulls _core for
    // idempotency, and the count accessors guard on _core != null. Mirrors what the generated
    // init tests rely on when they read AllocationsCount immediately after Dispose().
    [Test]
    public void Arena_AllocationsCount_AfterDispose_IsZero()
    {
        var arena = new Arena(Allocator.Persistent);
        var v = arena.floatVec(3);
        v[0] = 1; v[1] = 2; v[2] = 3;
        Assert.AreEqual(1, arena.AllocationsCount);

        arena.Dispose();

        Assert.AreEqual(0, arena.AllocationsCount);
        Assert.AreEqual(0, arena.TempAllocationsCount);
        Assert.AreEqual(0, arena.AllAllocationsCount);
    }

    // Clear()/ClearTemp() must agree with the accessors above on "disposed/default == empty":
    // a default(Arena) has nothing to clear, so Clear() must be a safe no-op, not a null deref.
    [Test]
    public void Arena_Clear_OnDefault_IsSafeNoOp()
    {
        Assert.DoesNotThrow(() => default(Arena).Clear());
        Assert.DoesNotThrow(() => default(Arena).ClearTemp());
    }

    // Same contract post-Dispose: Dispose() nulls _core, so a subsequent Clear()/ClearTemp() on
    // the same handle must also be a no-op rather than dereferencing the freed core.
    [Test]
    public void Arena_Clear_AfterDispose_IsSafeNoOp()
    {
        var arena = new Arena(Allocator.Persistent);
        var v = arena.floatVec(3);
        v[0] = 1; v[1] = 2; v[2] = 3;

        arena.Dispose();

        Assert.DoesNotThrow(() => arena.Clear());
        Assert.DoesNotThrow(() => arena.ClearTemp());
    }

    // Unlike Clear(), Pivot(int)/Indices(int) MUST allocate -- there is no sensible "empty"
    // result to hand back for a default/disposed arena, so they throw a clear, actionable
    // exception instead of dereferencing a null core.
    [Test]
    public void Arena_Pivot_OnDefault_Throws()
    {
        Assert.Throws<System.InvalidOperationException>(() => default(Arena).Pivot(1));
    }

    [Test]
    public void Arena_Indices_OnDefault_Throws()
    {
        Assert.Throws<System.InvalidOperationException>(() => default(Arena).Indices(1));
    }

    // Same guard, post-Dispose: a torn-down handle must throw rather than deref a freed core.
    [Test]
    public void Arena_Pivot_AfterDispose_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        arena.Dispose();

        Assert.Throws<System.InvalidOperationException>(() => arena.Pivot(1));
    }
}
