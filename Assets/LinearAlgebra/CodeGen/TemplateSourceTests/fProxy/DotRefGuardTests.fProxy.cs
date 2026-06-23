using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Collections;

// Guard (safety-check) tests for the ref-dest ops. These assert the ArgumentException
// guards fire, so they use Assert.Throws and therefore run as plain managed [Test]
// methods (NOT inside a Burst IJob - exceptions can't be asserted there). This also
// exercises Arena + vec/mat allocated on a normal C# thread (outside a job).
public class fProxyDotRefGuardTests
{
    // ---- dimension-mismatch guards ----

    [Test]
    public void MatVec_BadDestSize_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(3, 4);
            var x = arena.fProxyVec(4);
            var bad = arena.fProxyVec(2);   // must be length 3 (A.M_Rows)
            Assert.Throws<ArgumentException>(() => fProxyOP.dot(in A, in x, ref bad));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void MatMat_BadDestShape_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var a = arena.fProxyMat(3, 4);
            var b = arena.fProxyMat(4, 5);
            var bad = arena.fProxyMat(3, 4);   // must be 3 x 5
            Assert.Throws<ArgumentException>(() => fProxyOP.dot(in a, in b, ref bad, false));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void MatMat_IncompatibleInputs_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var a = arena.fProxyMat(3, 4);
            var b = arena.fProxyMat(5, 6);     // a.N_Cols (4) != b.M_Rows (5)
            var c = arena.fProxyMat(3, 6);
            Assert.Throws<ArgumentException>(() => fProxyOP.dot(in a, in b, ref c, false));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Trans_BadDestShape_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(3, 5);
            var bad = arena.fProxyMat(3, 5);   // must be 5 x 3
            Assert.Throws<ArgumentException>(() => fProxyOP.trans(in A, ref bad));
        }
        finally { arena.Dispose(); }
    }

    // ---- aliasing guards (destination must not alias an input) ----

    [Test]
    public void MatVec_DestAliasesX_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(3, 3);
            var x = arena.fProxyVec(3);
            var alias = x;   // shares x's buffer
            Assert.Throws<ArgumentException>(() => fProxyOP.dot(in A, in x, ref alias));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void VecMat_DestAliasesY_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var y = arena.fProxyVec(3);
            var A = arena.fProxyMat(3, 3);
            var alias = y;
            Assert.Throws<ArgumentException>(() => fProxyOP.dot(in y, in A, ref alias));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void MatMat_DestAliasesInput_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var a = arena.fProxyMat(3, 3);
            var b = arena.fProxyMat(3, 3);
            var aliasA = a;
            var aliasB = b;
            Assert.Throws<ArgumentException>(() => fProxyOP.dot(in a, in b, ref aliasA, false));
            Assert.Throws<ArgumentException>(() => fProxyOP.dot(in a, in b, ref aliasB, false));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Trans_DestAliasesInput_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(3, 3);
            var alias = A;
            Assert.Throws<ArgumentException>(() => fProxyOP.trans(in A, ref alias));
        }
        finally { arena.Dispose(); }
    }

    // ---- valid same-input aliasing must NOT throw (inputs are read-only) ----

    [Test]
    public void MatMat_SameInputAlias_DoesNotThrow()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var a = arena.fProxyMat(3, 3);
            var c = arena.fProxyMat(3, 3);
            // dot(A, A, ref C): the two inputs alias each other (fine); C is distinct.
            Assert.DoesNotThrow(() => fProxyOP.dot(in a, in a, ref c, false));
        }
        finally { arena.Dispose(); }
    }
}
