using System;

using BULA;
using NUnit.Framework;
using Unity.Collections;

// bool-family coverage of the realtime-interop surface: IsCreated, NativeArray view,
// CopyTo/CopyFrom(NativeArray), matrix CopyTo/CopyFrom parity.
public class BoolBridgeTests
{
    [Test]
    public void IsCreated_Lifecycle()
    {
        boolN defVec = default;
        Assert.IsFalse(defVec.IsCreated);

        var v = new boolN(3, Allocator.Temp);
        Assert.IsTrue(v.IsCreated);
        v.Dispose();
#if ENABLE_UNITY_COLLECTIONS_CHECKS
        Assert.IsFalse(v.IsCreated);
#endif
        boolMxN defMat = default;
        Assert.IsFalse(defMat.IsCreated);
        var m = new boolMxN(2, 2, Allocator.Temp);
        Assert.IsTrue(m.IsCreated);
    }

    [Test]
    public void View_And_CopyToFrom_NativeArray()
    {
        var arr = new NativeArray<bool>(4, Allocator.Temp);
        arr[2] = true;

        var view = new boolN(arr);
        Assert.IsTrue(view.IsCreated);
        Assert.IsTrue(view[2]);
        view[1] = true;
        Assert.IsTrue(arr[1]);

        var marr = new NativeArray<bool>(6, Allocator.Temp);
        var mview = new boolMxN(2, 3, marr);
        mview[1, 2] = true;
        Assert.IsTrue(marr[5]);
        Assert.Throws<ArgumentException>(() => { var bad = new boolMxN(2, 2, marr); });

        var dst = new NativeArray<bool>(4, Allocator.Temp);
        view.CopyTo(dst);
        Assert.IsTrue(dst[2]);
        var wrong = new NativeArray<bool>(2, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => view.CopyTo(wrong));
        Assert.Throws<ArgumentException>(() => view.CopyFrom(wrong));
    }
}
