using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Collections;

// Regression test: vector Copy() must be PERSISTENT (survives ClearTemp) and TempCopy() must be
// TEMP (cleared by ClearTemp). Previously both routed to the temp pool, so Copy() returned a
// vector that ClearTemp would free out from under the caller (use-after-dispose).
// Managed [Test] (arena on a normal C# thread) so we can read the arena's allocation counters.
public class floatVectorCopyTests
{
    [Test]
    public void Copy_IsPersistent_TempCopy_IsTemp()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var v = arena.floatVec(4);
            v[0] = (float)1; v[1] = (float)2; v[2] = (float)3; v[3] = (float)4;

            int persistBefore = arena.AllocationsCount;       // includes v
            int tempBefore = arena.TempAllocationsCount;

            // Copy() -> persistent pool
            var c = v.Copy();
            Assert.AreEqual(persistBefore + 1, arena.AllocationsCount);
            Assert.AreEqual(tempBefore, arena.TempAllocationsCount);

            // TempCopy() -> temp pool
            var t = v.TempCopy();
            Assert.AreEqual(persistBefore + 1, arena.AllocationsCount);
            Assert.AreEqual(tempBefore + 1, arena.TempAllocationsCount);

            // ClearTemp frees the temp pool but keeps the persistent Copy intact (and readable).
            arena.ClearTemp();
            Assert.AreEqual(tempBefore, arena.TempAllocationsCount);
            Assert.AreEqual(persistBefore + 1, arena.AllocationsCount);
            for (int i = 0; i < 4; i++)
                Assert.AreEqual((double)(i + 1), (double)c[i], 1e-5);
        }
        finally { arena.Dispose(); }
    }
}
