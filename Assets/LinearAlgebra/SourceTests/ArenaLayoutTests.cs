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

    // ---- struct-size regression guard for the record-pointer migration -----------------------------
    //
    // The migration's premise is ZERO struct growth: each vector/matrix struct swapped its old 8-byte
    // `Arena _arena` handle for an 8-byte record pointer (_rec), while the backing UnsafeList<T> --
    // once the sole inline Data field -- became the _inlineData field used only on the standalone
    // (non-arena) path. Net: no new fields, no size change. UnsafeList<T> is a fixed-size handle
    // (Ptr + length + capacity + allocator) independent of T, so every N-type is the same size and
    // every MxN-type is the same size across all numeric families AND bool. These pinned byte counts
    // trip immediately if a future edit adds a field to any of these structs. The float/double
    // baselines cross-check the 32B/48B numbers claimed in commit a79c212.
    //
    // Measured empirically (Windows/x64, Unity 6000.3.2f1): every N == 32 bytes, every MxN == 48.

    const int VecStructSize = 32;
    const int MatStructSize = 48;

    [Test]
    public unsafe void VectorStructs_AreExpectedSize()
    {
        Assert.AreEqual(VecStructSize, UnsafeUtility.SizeOf<floatN>(), "floatN");
        Assert.AreEqual(VecStructSize, UnsafeUtility.SizeOf<doubleN>(), "doubleN");
        Assert.AreEqual(VecStructSize, UnsafeUtility.SizeOf<intN>(), "intN");
        Assert.AreEqual(VecStructSize, UnsafeUtility.SizeOf<shortN>(), "shortN");
        Assert.AreEqual(VecStructSize, UnsafeUtility.SizeOf<longN>(), "longN");
        Assert.AreEqual(VecStructSize, UnsafeUtility.SizeOf<uintN>(), "uintN");
        Assert.AreEqual(VecStructSize, UnsafeUtility.SizeOf<boolN>(), "boolN");
    }

    [Test]
    public unsafe void MatrixStructs_AreExpectedSize()
    {
        Assert.AreEqual(MatStructSize, UnsafeUtility.SizeOf<floatMxN>(), "floatMxN");
        Assert.AreEqual(MatStructSize, UnsafeUtility.SizeOf<doubleMxN>(), "doubleMxN");
        Assert.AreEqual(MatStructSize, UnsafeUtility.SizeOf<intMxN>(), "intMxN");
        Assert.AreEqual(MatStructSize, UnsafeUtility.SizeOf<shortMxN>(), "shortMxN");
        Assert.AreEqual(MatStructSize, UnsafeUtility.SizeOf<longMxN>(), "longMxN");
        Assert.AreEqual(MatStructSize, UnsafeUtility.SizeOf<uintMxN>(), "uintMxN");
        Assert.AreEqual(MatStructSize, UnsafeUtility.SizeOf<boolMxN>(), "boolMxN");
    }
}
