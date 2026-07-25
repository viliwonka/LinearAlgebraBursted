using System;

using BULA;
using NUnit.Framework;
using Unity.Collections;

// Compact integer-family coverage of the realtime-interop surface (the fProxy sibling
// carries the full matrix/guard/Burst-job coverage; the template lines are shared):
// zeroInPlace/fillInPlace, IsCreated, NativeArray view + CopyTo/CopyFrom.
public class iProxyBridgeFillTests
{
    [Test]
    public void ZeroAndFillInPlace_VectorAndMatrix()
    {
        var v = new iProxyN(4, Allocator.Temp);
        v.fillInPlace((iProxy)3);
        for (int i = 0; i < 4; i++) Assert.IsTrue(v[i] == (iProxy)3);
        v.zeroInPlace();
        for (int i = 0; i < 4; i++) Assert.IsTrue(v[i] == (iProxy)0);

        var m = new iProxyMxN(2, 3, Allocator.Temp);
        m.fillInPlace((iProxy)7);
        Assert.IsTrue(m[1, 2] == (iProxy)7);
        m.zeroInPlace();
        Assert.IsTrue(m[1, 2] == (iProxy)0);
    }

    [Test]
    public void IsCreated_Lifecycle()
    {
        iProxyN defVec = default;
        Assert.IsFalse(defVec.IsCreated);
        var v = new iProxyN(3, Allocator.Persistent);
        Assert.IsTrue(v.IsCreated);
        v.Dispose();
        Assert.IsFalse(v.IsCreated);

        iProxyMxN defMat = default;
        Assert.IsFalse(defMat.IsCreated);
        var m = new iProxyMxN(2, 2, Allocator.Persistent);
        Assert.IsTrue(m.IsCreated);
        m.Dispose();
        Assert.IsFalse(m.IsCreated);
    }

    [Test]
    public void View_And_CopyToFrom_NativeArray()
    {
        var arr = new NativeArray<iProxy>(4, Allocator.Temp);
        arr[2] = (iProxy)5;

        var view = new iProxyN(arr);
        Assert.IsTrue(view.IsCreated);
        Assert.IsTrue(view[2] == (iProxy)5);
        view[1] = (iProxy)9;
        Assert.IsTrue(arr[1] == (iProxy)9);

        var marr = new NativeArray<iProxy>(6, Allocator.Temp);
        var mview = new iProxyMxN(2, 3, marr);
        mview[1, 2] = (iProxy)8;
        Assert.IsTrue(marr[5] == (iProxy)8);
        Assert.Throws<ArgumentException>(() => { var bad = new iProxyMxN(2, 2, marr); });

        var v = new iProxyN(4, Allocator.Temp);
        v.CopyFrom(arr);
        Assert.IsTrue(v[1] == (iProxy)9);
        v[0] = (iProxy)4;
        v.CopyTo(arr);
        Assert.IsTrue(arr[0] == (iProxy)4);

        var wrong = new NativeArray<iProxy>(2, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => v.CopyTo(wrong));
        Assert.Throws<ArgumentException>(() => v.CopyFrom(wrong));
    }
}
