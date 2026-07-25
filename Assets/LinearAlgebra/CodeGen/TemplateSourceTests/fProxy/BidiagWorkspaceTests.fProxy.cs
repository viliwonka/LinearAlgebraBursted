using System;

using BULA;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Workspace-overload tests for Bidiag.decomp / values and their shared workspace
// fProxyBidiagCache (new fProxyBidiagCache(m, n, Allocator)).
//
// The ws overload is the real body (caller-owned W/leftU/uVec/vVec/wScratch); the allocating overload
// delegates with Temp scratch, so for identical inputs the outputs are bit-identical. Tests:
//   (a) EQUIVALENCE — allocating vs ws on the SAME matrix.
//   (b) REUSE       — ONE workspace reused across two different (same-shape) inputs; the 2nd result
//                     equals a fresh allocating call (guards the leftU buffer-reuse-safety claim).
//   (c) MIS-SIZED   — wrong-dimension workspace throws; plus the needLeftU subtlety: a values-call
//                     with a leftU-less workspace must NOT throw, but a full bidiagonalize must.
public class fProxyBidiagWorkspaceTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default, CompileSynchronously = true)]
    public struct WorkspaceJob : IJob
    {
        public enum TestType
        {
            BidiagEquivSquare,
            BidiagEquivTall,
            ValuesEquivTall,
            ReuseBoth,
        }

        public TestType Type;

        static fProxy Tol() => 256 * Consts.fProxySqrtEps;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.BidiagEquivSquare: BidiagEquiv(6, 6, 1001); break;
                case TestType.BidiagEquivTall:   BidiagEquiv(9, 4, 2002); break;
                case TestType.ValuesEquivTall:   ValuesEquiv(9, 4, 3003); break;
                case TestType.ReuseBoth:         ReuseBoth();             break;
            }
        }

        void BidiagEquiv(int m, int n, uint seed)
        {
            var A = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-3f), (fProxy)3f, seed);

            var Ua = new fProxyMxN(m, n, Allocator.Temp); var Ba = new fProxyMxN(n, n, Allocator.Temp); var Va = new fProxyMxN(n, n, Allocator.Temp);
            Bidiag.decomp(in A, ref Ua, ref Ba, ref Va);

            var ws = new fProxyBidiagCache(m, n, Allocator.Temp);
            var Uw = new fProxyMxN(m, n, Allocator.Temp); var Bw = new fProxyMxN(n, n, Allocator.Temp); var Vw = new fProxyMxN(n, n, Allocator.Temp);
            Bidiag.decomp(in A, ref Uw, ref Bw, ref Vw, ref ws);

            var dU = new fProxyMxN(in Ua, Allocator.Temp); fProxyComp.subInPlace(dU, Uw);
            var dB = new fProxyMxN(in Ba, Allocator.Temp); fProxyComp.subInPlace(dB, Bw);
            var dV = new fProxyMxN(in Va, Allocator.Temp); fProxyComp.subInPlace(dV, Vw);
            Assert.IsTrue(Analysis.isZero(dU, Tol()));
            Assert.IsTrue(Analysis.isZero(dB, Tol()));
            Assert.IsTrue(Analysis.isZero(dV, Tol()));
        }

        void ValuesEquiv(int m, int n, uint seed)
        {
            var A = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-3f), (fProxy)3f, seed);

            var da = new fProxyN(n, Allocator.Temp); var ea = new fProxyN(n, Allocator.Temp);
            Bidiag.values(in A, ref da, ref ea);

            var ws = new fProxyBidiagCache(m, n, Allocator.Temp);
            var dw = new fProxyN(n, Allocator.Temp); var ew = new fProxyN(n, Allocator.Temp);
            Bidiag.values(in A, ref dw, ref ew, ref ws);

            var dd = new fProxyN(in da, Allocator.Temp); fProxyComp.subInPlace(dd, dw);
            var de = new fProxyN(in ea, Allocator.Temp); fProxyComp.subInPlace(de, ew);
            Assert.IsTrue(Analysis.isZero(dd, Tol()));
            Assert.IsTrue(Analysis.isZero(de, Tol()));
        }

        void ReuseBoth()
        {
            int m = 8, n = 5;

            var A1 = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-3f), (fProxy)3f, 4004);
            var A2 = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-2f), (fProxy)2f, 5005);

            var ws = new fProxyBidiagCache(m, n, Allocator.Temp);   // allocated ONCE

            // full bidiagonalize: warm on A1, reuse on A2, compare to fresh allocating on A2.
            var U1 = new fProxyMxN(m, n, Allocator.Temp); var B1 = new fProxyMxN(n, n, Allocator.Temp); var V1 = new fProxyMxN(n, n, Allocator.Temp);
            Bidiag.decomp(in A1, ref U1, ref B1, ref V1, ref ws);
            var Uw = new fProxyMxN(m, n, Allocator.Temp); var Bw = new fProxyMxN(n, n, Allocator.Temp); var Vw = new fProxyMxN(n, n, Allocator.Temp);
            Bidiag.decomp(in A2, ref Uw, ref Bw, ref Vw, ref ws);
            var Ua = new fProxyMxN(m, n, Allocator.Temp); var Ba = new fProxyMxN(n, n, Allocator.Temp); var Va = new fProxyMxN(n, n, Allocator.Temp);
            Bidiag.decomp(in A2, ref Ua, ref Ba, ref Va);
            var dU = new fProxyMxN(in Uw, Allocator.Temp); fProxyComp.subInPlace(dU, Ua);
            var dB = new fProxyMxN(in Bw, Allocator.Temp); fProxyComp.subInPlace(dB, Ba);
            var dV = new fProxyMxN(in Vw, Allocator.Temp); fProxyComp.subInPlace(dV, Va);
            Assert.IsTrue(Analysis.isZero(dU, Tol()));
            Assert.IsTrue(Analysis.isZero(dB, Tol()));
            Assert.IsTrue(Analysis.isZero(dV, Tol()));

            // values: same reused workspace.
            var d1 = new fProxyN(n, Allocator.Temp); var e1 = new fProxyN(n, Allocator.Temp);
            Bidiag.values(in A1, ref d1, ref e1, ref ws);
            var dw = new fProxyN(n, Allocator.Temp); var ew = new fProxyN(n, Allocator.Temp);
            Bidiag.values(in A2, ref dw, ref ew, ref ws);
            var da = new fProxyN(n, Allocator.Temp); var ea = new fProxyN(n, Allocator.Temp);
            Bidiag.values(in A2, ref da, ref ea);
            var dd = new fProxyN(in dw, Allocator.Temp); fProxyComp.subInPlace(dd, da);
            var de = new fProxyN(in ew, Allocator.Temp); fProxyComp.subInPlace(de, ea);
            Assert.IsTrue(Analysis.isZero(dd, Tol()));
            Assert.IsTrue(Analysis.isZero(de, Tol()));
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(WorkspaceJob.TestType));

    [TestCaseSource("GetEnums")]
    public void WorkspaceTests(WorkspaceJob.TestType type)
    {
        new WorkspaceJob() { Type = type }.Run();
    }

    // ---- mis-sized workspace guards (managed [Test]) ----

    [Test]
    public void Bidiagonalize_BadWorkspace_Throws()
    {
        int m = 6, n = 4;
        var A = new fProxyMxN(m, n, Allocator.Temp);
        var U = new fProxyMxN(m, n, Allocator.Temp); var B = new fProxyMxN(n, n, Allocator.Temp); var V = new fProxyMxN(n, n, Allocator.Temp);
        var ws = new fProxyBidiagCache(m + 1, n, Allocator.Temp);   // wrong m
        Assert.Throws<ArgumentException>(
            () => Bidiag.decomp(in A, ref U, ref B, ref V, ref ws));
    }

    [Test]
    public void BidiagonalizeValues_BadWorkspace_Throws()
    {
        int m = 6, n = 4;
        var A = new fProxyMxN(m, n, Allocator.Temp);
        var d = new fProxyN(n, Allocator.Temp); var e = new fProxyN(n, Allocator.Temp);
        var ws = new fProxyBidiagCache(m, n + 1, Allocator.Temp);   // wrong n (W/vVec/wScratch all wrong)
        Assert.Throws<ArgumentException>(
            () => Bidiag.values(in A, ref d, ref e, ref ws));
    }

    // needLeftU subtlety: Bidiag.values never touches leftU, so a leftU-less workspace (common
    // buffers correct, leftU = default) must NOT throw on the leftU check; the full bidiagonalize,
    // which reconstructs U from leftU, MUST throw on the same workspace.
    [Test]
    public void Values_LeftULessWorkspace_DoesNotThrow_ButBidiagonalizeDoes()
    {
        int m = 6, n = 4;
        var A = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-2f), (fProxy)2f, 6006);

        var ws = new fProxyBidiagCache
        {
            W = new fProxyMxN(m, n, Allocator.Temp),
            leftU = default,                 // intentionally absent
            uVec = new fProxyN(m, Allocator.Temp),
            vVec = new fProxyN(n, Allocator.Temp),
            wScratch = new fProxyN(n, Allocator.Temp)
        };

        var d = new fProxyN(n, Allocator.Temp); var e = new fProxyN(n, Allocator.Temp);
        Assert.DoesNotThrow(() => Bidiag.values(in A, ref d, ref e, ref ws));

        // same workspace fails the full bidiagonalize (needs leftU).
        var U = new fProxyMxN(m, n, Allocator.Temp); var B = new fProxyMxN(n, n, Allocator.Temp); var V = new fProxyMxN(n, n, Allocator.Temp);
        Assert.Throws<ArgumentException>(
            () => Bidiag.decomp(in A, ref U, ref B, ref V, ref ws));
    }

    // fProxyBidiagCache(m, n, Allocator): W (m x n), leftU (m x n), uVec (m), vVec (n), wScratch (n).
    [Test]
    public void BidiagWorkspace_Factory_SizesCorrectly()
    {
        var ws = new fProxyBidiagCache(9, 4, Allocator.Temp);
        Assert.AreEqual(9, ws.W.M_Rows);     Assert.AreEqual(4, ws.W.N_Cols);
        Assert.AreEqual(9, ws.leftU.M_Rows); Assert.AreEqual(4, ws.leftU.N_Cols);
        Assert.AreEqual(9, ws.uVec.N);
        Assert.AreEqual(4, ws.vVec.N);
        Assert.AreEqual(4, ws.wScratch.N);
    }
}
