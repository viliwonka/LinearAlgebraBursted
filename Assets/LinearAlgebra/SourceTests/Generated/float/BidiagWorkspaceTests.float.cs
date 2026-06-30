using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Workspace-overload tests for Bidiag.bidiagonalize / bidiagonalizeValues and their shared workspace
// floatBidiag_WS (Arena.floatBidiag_WS(m, n)).
//
// The ws overload is the real body (caller-owned W/leftU/uVec/vVec/wScratch); the allocating overload
// delegates with Temp scratch, so for identical inputs the outputs are bit-identical. Tests:
//   (a) EQUIVALENCE — allocating vs ws on the SAME matrix.
//   (b) REUSE       — ONE workspace reused across two different (same-shape) inputs; the 2nd result
//                     equals a fresh allocating call (guards the leftU buffer-reuse-safety claim).
//   (c) MIS-SIZED   — wrong-dimension workspace throws; plus the needLeftU subtlety: a values-call
//                     with a leftU-less workspace must NOT throw, but a full bidiagonalize must.
public class floatBidiagWorkspaceTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
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

        static float Tol() => 256 * Consts.floatSqrtEps;

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
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatRandomMatrix(m, n, (float)(-3f), (float)3f, seed);

            var Ua = arena.floatMat(m, n); var Ba = arena.floatMat(n, n); var Va = arena.floatMat(n, n);
            Bidiag.bidiagonalize(in A, ref Ua, ref Ba, ref Va);

            var ws = arena.floatBidiag_WS(m, n);
            var Uw = arena.floatMat(m, n); var Bw = arena.floatMat(n, n); var Vw = arena.floatMat(n, n);
            Bidiag.bidiagonalize(in A, ref Uw, ref Bw, ref Vw, ref ws);

            Assert.IsTrue(Analysis_OP.IsZero(Ua - Uw, Tol()));
            Assert.IsTrue(Analysis_OP.IsZero(Ba - Bw, Tol()));
            Assert.IsTrue(Analysis_OP.IsZero(Va - Vw, Tol()));

            arena.Dispose();
        }

        void ValuesEquiv(int m, int n, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatRandomMatrix(m, n, (float)(-3f), (float)3f, seed);

            var da = arena.floatVec(n); var ea = arena.floatVec(n);
            Bidiag.bidiagonalizeValues(in A, ref da, ref ea);

            var ws = arena.floatBidiag_WS(m, n);
            var dw = arena.floatVec(n); var ew = arena.floatVec(n);
            Bidiag.bidiagonalizeValues(in A, ref dw, ref ew, ref ws);

            Assert.IsTrue(Analysis_OP.IsZero(da - dw, Tol()));
            Assert.IsTrue(Analysis_OP.IsZero(ea - ew, Tol()));

            arena.Dispose();
        }

        void ReuseBoth()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 8, n = 5;

            var A1 = arena.floatRandomMatrix(m, n, (float)(-3f), (float)3f, 4004);
            var A2 = arena.floatRandomMatrix(m, n, (float)(-2f), (float)2f, 5005);

            var ws = arena.floatBidiag_WS(m, n);   // allocated ONCE

            // full bidiagonalize: warm on A1, reuse on A2, compare to fresh allocating on A2.
            var U1 = arena.floatMat(m, n); var B1 = arena.floatMat(n, n); var V1 = arena.floatMat(n, n);
            Bidiag.bidiagonalize(in A1, ref U1, ref B1, ref V1, ref ws);
            var Uw = arena.floatMat(m, n); var Bw = arena.floatMat(n, n); var Vw = arena.floatMat(n, n);
            Bidiag.bidiagonalize(in A2, ref Uw, ref Bw, ref Vw, ref ws);
            var Ua = arena.floatMat(m, n); var Ba = arena.floatMat(n, n); var Va = arena.floatMat(n, n);
            Bidiag.bidiagonalize(in A2, ref Ua, ref Ba, ref Va);
            Assert.IsTrue(Analysis_OP.IsZero(Uw - Ua, Tol()));
            Assert.IsTrue(Analysis_OP.IsZero(Bw - Ba, Tol()));
            Assert.IsTrue(Analysis_OP.IsZero(Vw - Va, Tol()));

            // values: same reused workspace.
            var d1 = arena.floatVec(n); var e1 = arena.floatVec(n);
            Bidiag.bidiagonalizeValues(in A1, ref d1, ref e1, ref ws);
            var dw = arena.floatVec(n); var ew = arena.floatVec(n);
            Bidiag.bidiagonalizeValues(in A2, ref dw, ref ew, ref ws);
            var da = arena.floatVec(n); var ea = arena.floatVec(n);
            Bidiag.bidiagonalizeValues(in A2, ref da, ref ea);
            Assert.IsTrue(Analysis_OP.IsZero(dw - da, Tol()));
            Assert.IsTrue(Analysis_OP.IsZero(ew - ea, Tol()));

            arena.Dispose();
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
        var arena = new Arena(Allocator.Persistent);
        try
        {
            int m = 6, n = 4;
            var A = arena.floatMat(m, n);
            var U = arena.floatMat(m, n); var B = arena.floatMat(n, n); var V = arena.floatMat(n, n);
            var ws = arena.floatBidiag_WS(m + 1, n);   // wrong m
            Assert.Throws<ArgumentException>(
                () => Bidiag.bidiagonalize(in A, ref U, ref B, ref V, ref ws));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void BidiagonalizeValues_BadWorkspace_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            int m = 6, n = 4;
            var A = arena.floatMat(m, n);
            var d = arena.floatVec(n); var e = arena.floatVec(n);
            var ws = arena.floatBidiag_WS(m, n + 1);   // wrong n (W/vVec/wScratch all wrong)
            Assert.Throws<ArgumentException>(
                () => Bidiag.bidiagonalizeValues(in A, ref d, ref e, ref ws));
        }
        finally { arena.Dispose(); }
    }

    // needLeftU subtlety: bidiagonalizeValues never touches leftU, so a leftU-less workspace (common
    // buffers correct, leftU = default) must NOT throw on the leftU check; the full bidiagonalize,
    // which reconstructs U from leftU, MUST throw on the same workspace.
    [Test]
    public void Values_LeftULessWorkspace_DoesNotThrow_ButBidiagonalizeDoes()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            int m = 6, n = 4;
            var A = arena.floatRandomMatrix(m, n, (float)(-2f), (float)2f, 6006);

            var ws = new floatBidiag_WS
            {
                W = arena.floatMat(m, n),
                leftU = default,                 // intentionally absent
                uVec = arena.floatVec(m),
                vVec = arena.floatVec(n),
                wScratch = arena.floatVec(n)
            };

            var d = arena.floatVec(n); var e = arena.floatVec(n);
            Assert.DoesNotThrow(() => Bidiag.bidiagonalizeValues(in A, ref d, ref e, ref ws));

            // same workspace fails the full bidiagonalize (needs leftU).
            var U = arena.floatMat(m, n); var B = arena.floatMat(n, n); var V = arena.floatMat(n, n);
            Assert.Throws<ArgumentException>(
                () => Bidiag.bidiagonalize(in A, ref U, ref B, ref V, ref ws));
        }
        finally { arena.Dispose(); }
    }

    // Arena.floatBidiag_WS(m, n): W (m x n), leftU (m x n), uVec (m), vVec (n), wScratch (n).
    [Test]
    public void BidiagWorkspace_Factory_SizesCorrectly()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var ws = arena.floatBidiag_WS(9, 4);
            Assert.AreEqual(9, ws.W.M_Rows);     Assert.AreEqual(4, ws.W.N_Cols);
            Assert.AreEqual(9, ws.leftU.M_Rows); Assert.AreEqual(4, ws.leftU.N_Cols);
            Assert.AreEqual(9, ws.uVec.N);
            Assert.AreEqual(4, ws.vVec.N);
            Assert.AreEqual(4, ws.wScratch.N);
        }
        finally { arena.Dispose(); }
    }
}
