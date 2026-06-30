using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Collections;

// Regression tests (from PR #1's ideas):
//  - addInpl(place, from) must mutate `place` (place += from), NOT `from`. The internal compAdd
//    operands were reversed, so the method used to mutate the wrong operand — masked end-to-end
//    only because the + operators also called it backwards.
//  - the DB_isPersistent / DB_isTemp pool checks, used to assert ops don't move a persistent
//    buffer into the temp pool.
// Managed [Test] (arena on a normal C# thread) — reads the arena's debug pool checks.
public class fProxyInplaceOpTests
{
    [Test]
    public void AddInpl_MutatesPlace_LeavesFromUnchanged()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var a = arena.fProxyVec(3); a[0] = (fProxy)1; a[1] = (fProxy)2; a[2] = (fProxy)3;
            var b = arena.fProxyVec(3); b[0] = (fProxy)10; b[1] = (fProxy)20; b[2] = (fProxy)30;

            fProxy_OP.addInpl(a, b);   // a += b

            // a updated...
            Assert.AreEqual(11.0, (double)a[0], 1e-6);
            Assert.AreEqual(22.0, (double)a[1], 1e-6);
            Assert.AreEqual(33.0, (double)a[2], 1e-6);
            // ...and b left untouched (pre-fix, b would have been mutated instead).
            Assert.AreEqual(10.0, (double)b[0], 1e-6);
            Assert.AreEqual(20.0, (double)b[1], 1e-6);
            Assert.AreEqual(30.0, (double)b[2], 1e-6);
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void SubInpl_MutatesPlace_LeavesFromUnchanged()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var a = arena.fProxyVec(2); a[0] = (fProxy)10; a[1] = (fProxy)20;
            var b = arena.fProxyVec(2); b[0] = (fProxy)3;  b[1] = (fProxy)5;

            fProxy_OP.subInpl(a, b);   // a -= b

            Assert.AreEqual(7.0, (double)a[0], 1e-6);
            Assert.AreEqual(15.0, (double)a[1], 1e-6);
            Assert.AreEqual(3.0, (double)b[0], 1e-6);   // b unchanged
            Assert.AreEqual(5.0, (double)b[1], 1e-6);
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void DB_PoolChecks_And_OpsDoNotStealPersistentBuffers()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var v = arena.fProxyVec(3);
            Assert.IsTrue(arena.DB_isPersistent(in v));
            Assert.IsFalse(arena.DB_isTemp(in v));

            var t = v.TempCopy();
            Assert.IsTrue(arena.DB_isTemp(in t));
            Assert.IsFalse(arena.DB_isPersistent(in t));

            // operator + must leave its operands persistent and return a temp result.
            var a = arena.fProxyVec(2); var b = arena.fProxyVec(2);
            var sum = a + b;
            Assert.IsTrue(arena.DB_isPersistent(in a));
            Assert.IsTrue(arena.DB_isPersistent(in b));
            Assert.IsTrue(arena.DB_isTemp(in sum));
        }
        finally { arena.Dispose(); }
    }
}
