using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Collections;

// Guard (safety-check) tests for the ref-dest ops. These assert the ArgumentException
// guards fire, so they use Assert.Throws and therefore run as plain managed [Test]
// methods (NOT inside a Burst IJob - exceptions can't be asserted there). This also
// exercises standalone vec/mat allocation on a normal C# thread (outside a job).
public class fProxyDotRefGuardTests
{
    // ---- dimension-mismatch guards ----

    [Test]
    public void MatVec_BadDestSize_Throws()
    {
        var A = new fProxyMxN(3, 4, Allocator.Temp);
        var x = new fProxyN(4, Allocator.Temp);
        var bad = new fProxyN(2, Allocator.Temp);   // must be length 3 (A.M_Rows)
        Assert.Throws<ArgumentException>(() => Blas.dot(in A, in x, ref bad));
    }

    [Test]
    public void MatMat_BadDestShape_Throws()
    {
        var a = new fProxyMxN(3, 4, Allocator.Temp);
        var b = new fProxyMxN(4, 5, Allocator.Temp);
        var bad = new fProxyMxN(3, 4, Allocator.Temp);   // must be 3 x 5
        Assert.Throws<ArgumentException>(() => Blas.dot(in a, in b, ref bad, false));
    }

    [Test]
    public void MatMat_IncompatibleInputs_Throws()
    {
        var a = new fProxyMxN(3, 4, Allocator.Temp);
        var b = new fProxyMxN(5, 6, Allocator.Temp);     // a.N_Cols (4) != b.M_Rows (5)
        var c = new fProxyMxN(3, 6, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => Blas.dot(in a, in b, ref c, false));
    }

    [Test]
    public void Trans_BadDestShape_Throws()
    {
        var A = new fProxyMxN(3, 5, Allocator.Temp);
        var bad = new fProxyMxN(3, 5, Allocator.Temp);   // must be 5 x 3
        Assert.Throws<ArgumentException>(() => Blas.trans(in A, ref bad));
    }

    // ---- aliasing guards (destination must not alias an input) ----

    [Test]
    public void MatVec_DestAliasesX_Throws()
    {
        var A = new fProxyMxN(3, 3, Allocator.Temp);
        var x = new fProxyN(3, Allocator.Temp);
        var alias = x;   // shares x's buffer
        Assert.Throws<ArgumentException>(() => Blas.dot(in A, in x, ref alias));
    }

    [Test]
    public void VecMat_DestAliasesY_Throws()
    {
        var y = new fProxyN(3, Allocator.Temp);
        var A = new fProxyMxN(3, 3, Allocator.Temp);
        var alias = y;
        Assert.Throws<ArgumentException>(() => Blas.dot(in y, in A, ref alias));
    }

    [Test]
    public void MatMat_DestAliasesInput_Throws()
    {
        var a = new fProxyMxN(3, 3, Allocator.Temp);
        var b = new fProxyMxN(3, 3, Allocator.Temp);
        var aliasA = a;
        var aliasB = b;
        Assert.Throws<ArgumentException>(() => Blas.dot(in a, in b, ref aliasA, false));
        Assert.Throws<ArgumentException>(() => Blas.dot(in a, in b, ref aliasB, false));
    }

    [Test]
    public void Trans_DestAliasesInput_Throws()
    {
        var A = new fProxyMxN(3, 3, Allocator.Temp);
        var alias = A;
        Assert.Throws<ArgumentException>(() => Blas.trans(in A, ref alias));
    }

    // ---- valid same-input aliasing must NOT throw (inputs are read-only) ----

    [Test]
    public void MatMat_SameInputAlias_DoesNotThrow()
    {
        var a = new fProxyMxN(3, 3, Allocator.Temp);
        var c = new fProxyMxN(3, 3, Allocator.Temp);
        // dot(A, A, ref C): the two inputs alias each other (fine); C is distinct.
        Assert.DoesNotThrow(() => Blas.dot(in a, in a, ref c, false));
    }
}
