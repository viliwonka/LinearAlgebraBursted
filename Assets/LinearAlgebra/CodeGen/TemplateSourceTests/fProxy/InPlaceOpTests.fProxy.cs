using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Collections;

// Asserts *InPlace(place, from) mutates `place` (place += from etc.), leaves `from` unchanged, and
// never moves a persistent buffer into the temp pool (checked via isPersistent / isTemp).
// Managed [Test] (arena on a normal C# thread) — reads the arena's debug pool checks.
public class fProxyInPlaceOpTests
{
    [Test]
    public void AddInPlace_MutatesPlace_LeavesFromUnchanged()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var a = arena.fProxyVec(3); a[0] = (fProxy)1; a[1] = (fProxy)2; a[2] = (fProxy)3;
            var b = arena.fProxyVec(3); b[0] = (fProxy)10; b[1] = (fProxy)20; b[2] = (fProxy)30;

            fProxyComp.addInPlace(a, b);   // a += b

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
    public void SubInPlace_MutatesPlace_LeavesFromUnchanged()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var a = arena.fProxyVec(2); a[0] = (fProxy)10; a[1] = (fProxy)20;
            var b = arena.fProxyVec(2); b[0] = (fProxy)3;  b[1] = (fProxy)5;

            fProxyComp.subInPlace(a, b);   // a -= b

            Assert.AreEqual(7.0, (double)a[0], 1e-6);
            Assert.AreEqual(15.0, (double)a[1], 1e-6);
            Assert.AreEqual(3.0, (double)b[0], 1e-6);   // b unchanged
            Assert.AreEqual(5.0, (double)b[1], 1e-6);
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void MulInPlace_MutatesPlace_LeavesFromUnchanged()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var a = arena.fProxyVec(3); a[0] = (fProxy)2; a[1] = (fProxy)3; a[2] = (fProxy)4;
            var b = arena.fProxyVec(3); b[0] = (fProxy)10; b[1] = (fProxy)20; b[2] = (fProxy)30;

            fProxyComp.mulInPlace(a, b);   // a *= b

            Assert.AreEqual(20.0, (double)a[0], 1e-6);
            Assert.AreEqual(60.0, (double)a[1], 1e-6);
            Assert.AreEqual(120.0, (double)a[2], 1e-6);
            Assert.AreEqual(10.0, (double)b[0], 1e-6);   // b unchanged
            Assert.AreEqual(20.0, (double)b[1], 1e-6);
            Assert.AreEqual(30.0, (double)b[2], 1e-6);
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void MulInPlace_Matrix_MutatesPlace_LeavesFromUnchanged()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var a = arena.fProxyMat(2, 2);
            a[0, 0] = (fProxy)1; a[0, 1] = (fProxy)2; a[1, 0] = (fProxy)3; a[1, 1] = (fProxy)4;
            var b = arena.fProxyMat(2, 2);
            b[0, 0] = (fProxy)5; b[0, 1] = (fProxy)6; b[1, 0] = (fProxy)7; b[1, 1] = (fProxy)8;

            fProxyComp.mulInPlace(a, b);   // a *= b, component-wise

            Assert.AreEqual(5.0, (double)a[0, 0], 1e-6);
            Assert.AreEqual(12.0, (double)a[0, 1], 1e-6);
            Assert.AreEqual(21.0, (double)a[1, 0], 1e-6);
            Assert.AreEqual(32.0, (double)a[1, 1], 1e-6);
            Assert.AreEqual(5.0, (double)b[0, 0], 1e-6);   // b unchanged
            Assert.AreEqual(8.0, (double)b[1, 1], 1e-6);
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void OperatorMul_ComponentWise_Values_OperandsUntouched()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var a = arena.fProxyVec(3); a[0] = (fProxy)2; a[1] = (fProxy)3; a[2] = (fProxy)4;
            var b = arena.fProxyVec(3); b[0] = (fProxy)10; b[1] = (fProxy)20; b[2] = (fProxy)30;

            var c = a * b;

            Assert.AreEqual(20.0, (double)c[0], 1e-6);
            Assert.AreEqual(60.0, (double)c[1], 1e-6);
            Assert.AreEqual(120.0, (double)c[2], 1e-6);
            // both operands untouched
            Assert.AreEqual(2.0, (double)a[0], 1e-6);
            Assert.AreEqual(4.0, (double)a[2], 1e-6);
            Assert.AreEqual(10.0, (double)b[0], 1e-6);
            Assert.AreEqual(30.0, (double)b[2], 1e-6);

            var am = arena.fProxyMat(2, 2);
            am[0, 0] = (fProxy)1; am[0, 1] = (fProxy)2; am[1, 0] = (fProxy)3; am[1, 1] = (fProxy)4;
            var bm = arena.fProxyMat(2, 2);
            bm[0, 0] = (fProxy)5; bm[0, 1] = (fProxy)6; bm[1, 0] = (fProxy)7; bm[1, 1] = (fProxy)8;

            var cm = am * bm;

            Assert.AreEqual(5.0, (double)cm[0, 0], 1e-6);
            Assert.AreEqual(12.0, (double)cm[0, 1], 1e-6);
            Assert.AreEqual(21.0, (double)cm[1, 0], 1e-6);
            Assert.AreEqual(32.0, (double)cm[1, 1], 1e-6);
            Assert.AreEqual(2.0, (double)am[0, 1], 1e-6);   // operands untouched
            Assert.AreEqual(7.0, (double)bm[1, 0], 1e-6);
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
            Assert.IsTrue(arena.isPersistent(in v));
            Assert.IsFalse(arena.isTemp(in v));

            var t = v.TempCopy();
            Assert.IsTrue(arena.isTemp(in t));
            Assert.IsFalse(arena.isPersistent(in t));

            // operator + must leave its operands persistent and return a temp result.
            var a = arena.fProxyVec(2); var b = arena.fProxyVec(2);
            var sum = a + b;
            Assert.IsTrue(arena.isPersistent(in a));
            Assert.IsTrue(arena.isPersistent(in b));
            Assert.IsTrue(arena.isTemp(in sum));
        }
        finally { arena.Dispose(); }
    }
}
