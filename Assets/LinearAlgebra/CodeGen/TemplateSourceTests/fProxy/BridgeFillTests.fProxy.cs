using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Coverage for the realtime-interop surface:
//  - fProxyComp.zeroInPlace / fillInPlace (generic over vectors and matrices)
//  - IsCreated on fProxyN / fProxyMxN (standalone and arena-tracked lifecycles)
//  - NativeArray bridge: view constructors + CopyTo/CopyFrom(NativeArray), and the
//    matrix-level CopyTo/CopyFrom(in fProxyMxN) parity members.
// The view path is also exercised INSIDE a Burst job (solving straight into a
// NativeArray through a view) — that is the workflow the bridge exists for.
public class fProxyBridgeFillTests
{
    [Test]
    public void ZeroAndFillInPlace_VectorAndMatrix()
    {
        var v = new fProxyN(4, Allocator.Temp);
        v.fillInPlace((fProxy)3);
        for (int i = 0; i < 4; i++) Assert.AreEqual(3.0, (double)v[i], 1e-6);
        v.zeroInPlace();
        for (int i = 0; i < 4; i++) Assert.AreEqual(0.0, (double)v[i], 0.0);

        var m = new fProxyMxN(3, 2, Allocator.Temp);
        m.fillInPlace((fProxy)7);
        Assert.AreEqual(7.0, (double)m[0, 0], 1e-6);
        Assert.AreEqual(7.0, (double)m[2, 1], 1e-6);
        m.zeroInPlace();
        Assert.AreEqual(0.0, (double)m[0, 0], 0.0);
        Assert.AreEqual(0.0, (double)m[2, 1], 0.0);
    }

    [Test]
    public void IsCreated_Lifecycle_Standalone()
    {
        fProxyN defVec = default;
        Assert.IsFalse(defVec.IsCreated);
        var v = new fProxyN(3, Allocator.Persistent);
        Assert.IsTrue(v.IsCreated);
        v.Dispose();
        Assert.IsFalse(v.IsCreated);

        fProxyMxN defMat = default;
        Assert.IsFalse(defMat.IsCreated);
        var m = new fProxyMxN(2, 2, Allocator.Persistent);
        Assert.IsTrue(m.IsCreated);
        m.Dispose();
        Assert.IsFalse(m.IsCreated);
    }

    [Test]
    public void IsCreated_ArenaTracked_FalseAfterRecordDispose()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var v = arena.fProxyVec(3);
            Assert.IsTrue(v.IsCreated);
            v.Dispose();   // frees the record slot; the table itself stays alive
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            Assert.IsFalse(v.IsCreated);
#endif
            var m = arena.fProxyMat(2, 2);
            Assert.IsTrue(m.IsCreated);
            m.Dispose();
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            Assert.IsFalse(m.IsCreated);
#endif
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void VectorView_AliasesArrayMemory()
    {
        var arr = new NativeArray<fProxy>(4, Allocator.Temp);
        arr[2] = (fProxy)5;

        var view = new fProxyN(arr);
        Assert.IsTrue(view.IsCreated);
        Assert.AreEqual(4, view.N);
        Assert.AreEqual(5.0, (double)view[2], 1e-6);

        view[1] = (fProxy)9;                       // write through the view...
        Assert.AreEqual(9.0, (double)arr[1], 1e-6); // ...lands in the array

        view.Dispose();   // must be a no-op: the array still owns the memory
        Assert.AreEqual(9.0, (double)arr[1], 1e-6);
        arr.Dispose();
    }

    [Test]
    public void MatrixView_AliasesArrayMemory_And_GuardsDims()
    {
        var arr = new NativeArray<fProxy>(6, Allocator.Temp);
        var view = new fProxyMxN(2, 3, arr);
        Assert.IsTrue(view.IsCreated);

        view[1, 2] = (fProxy)8;                     // row-major: index 1*3+2 = 5
        Assert.AreEqual(8.0, (double)arr[5], 1e-6);

        Assert.Throws<ArgumentException>(() => { var bad = new fProxyMxN(2, 2, arr); });
        arr.Dispose();
    }

    [Test]
    public void CopyToFrom_NativeArray_Roundtrip_And_Guards()
    {
        var v = new fProxyN(3, Allocator.Temp);
        v[0] = (fProxy)1; v[1] = (fProxy)2; v[2] = (fProxy)3;

        var arr = new NativeArray<fProxy>(3, Allocator.Temp);
        v.CopyTo(arr);
        Assert.AreEqual(2.0, (double)arr[1], 1e-6);

        arr[1] = (fProxy)20;
        v.CopyFrom(arr);
        Assert.AreEqual(20.0, (double)v[1], 1e-6);

        var wrong = new NativeArray<fProxy>(2, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => v.CopyTo(wrong));
        Assert.Throws<ArgumentException>(() => v.CopyFrom(wrong));

        var m = new fProxyMxN(2, 2, Allocator.Temp);
        m[0, 0] = (fProxy)1; m[1, 1] = (fProxy)4;
        var marr = new NativeArray<fProxy>(4, Allocator.Temp);
        m.CopyTo(marr);
        Assert.AreEqual(4.0, (double)marr[3], 1e-6);
        marr[0] = (fProxy)10;
        m.CopyFrom(marr);
        Assert.AreEqual(10.0, (double)m[0, 0], 1e-6);
        Assert.Throws<ArgumentException>(() => m.CopyTo(wrong));
        Assert.Throws<ArgumentException>(() => m.CopyFrom(wrong));
    }

    [Test]
    public void Matrix_CopyToFrom_Matrix_Parity()
    {
        var a = new fProxyMxN(2, 3, Allocator.Temp);
        a[1, 2] = (fProxy)6;
        var b = new fProxyMxN(2, 3, Allocator.Temp);

        a.CopyTo(b);
        Assert.AreEqual(6.0, (double)b[1, 2], 1e-6);

        b[0, 0] = (fProxy)5;
        a.CopyFrom(b);
        Assert.AreEqual(5.0, (double)a[0, 0], 1e-6);

        var wrong = new fProxyMxN(3, 2, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => a.CopyTo(wrong));
        Assert.Throws<ArgumentException>(() => a.CopyFrom(wrong));
    }

    // Solve straight into a NativeArray through views, inside a Burst job — the
    // bridge's intended workflow: game state stays in NativeArrays, no boundary copy.
    [BurstCompile(CompileSynchronously = true)]
    struct ViewSolveJob : IJob
    {
        [ReadOnly] public NativeArray<fProxy> B;
        public NativeArray<fProxy> X;

        public void Execute()
        {
            int n = B.Length;
            var A = new fProxyMxN(n, n, Allocator.Temp);
            for (int i = 0; i < n; i++) A[i, i] = (fProxy)2;
            A[0, 1] = (fProxy)1; A[1, 0] = (fProxy)1;

            var x = new fProxyN(X);   // view over the output array
            x.CopyFrom(B);            // rhs in, solved in place below
            CHO.solveInPlace(ref A, ref x);
        }
    }

    [Test]
    public void ViewSolveJob_WritesSolutionThroughView()
    {
        // A = [[2,1,0],[1,2,0],[0,0,2]], b = (3,3,2)  ->  x = (1,1,1)
        var b = new NativeArray<fProxy>(3, Allocator.TempJob);
        var x = new NativeArray<fProxy>(3, Allocator.TempJob);
        b[0] = (fProxy)3; b[1] = (fProxy)3; b[2] = (fProxy)2;

        new ViewSolveJob { B = b, X = x }.Run();

        Assert.AreEqual(1.0, (double)x[0], 1e-4);
        Assert.AreEqual(1.0, (double)x[1], 1e-4);
        Assert.AreEqual(1.0, (double)x[2], 1e-4);

        b.Dispose(); x.Dispose();
    }
}
