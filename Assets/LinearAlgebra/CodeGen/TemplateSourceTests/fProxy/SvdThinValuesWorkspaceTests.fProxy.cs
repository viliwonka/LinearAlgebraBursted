using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Phase-2 solver-workspace tests for svdThin/svdValues: the caller-provided-scratch overloads
// (svdThin(...,ref fProxySVDThin_WS) / svdValues(...,ref fProxySVDValues_WS)) must produce results
// identical to the allocating wrappers (they run the SAME kernel), and a mis-sized/reused workspace
// must behave correctly (throw on bad size, produce identical results across repeated reuse).
public class fProxySvdThinValuesWorkspaceTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WorkspaceEquivJob : IJob
    {
        public enum TestType
        {
            SvdThinEquivTall,
            SvdThinEquivSquare,
            SvdValuesEquivTall,
            SvdValuesEquivSquare,
            WorkspaceReuse,
        }

        public TestType Type;

        // The scratch overload runs the SAME kernel as the allocating form, so results are
        // bit-identical in principle. Keep a small per-precision tolerance for robustness.
        static fProxy Tol() => 256 * Consts.fProxySqrtEps;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.SvdThinEquivTall:      SvdThinEquiv(10, 4); break;
                case TestType.SvdThinEquivSquare:    SvdThinEquiv(6, 6); break;
                case TestType.SvdValuesEquivTall:    SvdValuesEquiv(10, 4); break;
                case TestType.SvdValuesEquivSquare:  SvdValuesEquiv(6, 6); break;
                case TestType.WorkspaceReuse:         WorkspaceReuse(); break;
            }
        }

        // svdThin scratch overload must match the allocating wrapper.
        void SvdThinEquiv(int m, int n)
        {
            var arena = new Arena(Allocator.Persistent);

            var A0 = arena.fProxyRandomMat(m, n, -5f, 5f, 55123);
            for (int d = 0; d < n; d++)   // boost leading diagonal block for conditioning
                A0[d, d] += (fProxy)10f;

            // allocating reference
            var Aa = A0.Copy();
            var Ua = arena.fProxyMat(m, n);
            var Sa = arena.fProxyVec(n);
            var Va = arena.fProxyMat(n, n);
            bool oka = SVD.svdThin(in Aa, ref Ua, ref Sa, ref Va);

            // workspace-struct form (default maxIter/eps) must match the allocating form
            var Ab = A0.Copy();
            var Ub = arena.fProxyMat(m, n);
            var Sb = arena.fProxyVec(n);
            var Vb = arena.fProxyMat(n, n);
            var ws = arena.fProxySVDThin_WS(m, n);
            bool okb = SVD.svdThin(in Ab, ref Ub, ref Sb, ref Vb, ref ws);

            Assert.IsTrue(oka == okb);
            Assert.IsTrue(Analysis_OP.isZero(Sa - Sb, Tol()));
            Assert.IsTrue(Analysis_OP.isZero(Ua - Ub, Tol()));
            Assert.IsTrue(Analysis_OP.isZero(Va - Vb, Tol()));

            arena.Dispose();
        }

        // svdValues scratch overload must match the allocating wrapper.
        void SvdValuesEquiv(int m, int n)
        {
            var arena = new Arena(Allocator.Persistent);

            var A0 = arena.fProxyRandomMat(m, n, -5f, 5f, 90210);
            for (int d = 0; d < n; d++)
                A0[d, d] += (fProxy)10f;

            var Sa = arena.fProxyVec(n);
            bool oka = SVD.svdValues(in A0, ref Sa);

            var Sb = arena.fProxyVec(n);
            var ws = arena.fProxySVDValues_WS(m, n);
            bool okb = SVD.svdValues(in A0, ref Sb, ref ws);

            Assert.IsTrue(oka == okb);
            Assert.IsTrue(Analysis_OP.isZero(Sa - Sb, Tol()));

            arena.Dispose();
        }

        // Reuse ONE workspace (of each kind) across several consecutive solves: each solve must
        // match a fresh allocating solve, proving no stale state survives reuse.
        void WorkspaceReuse()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 8, n = 4;

            var thinWs = arena.fProxySVDThin_WS(m, n);     // allocated ONCE, reused below
            var valuesWs = arena.fProxySVDValues_WS(m, n); // allocated ONCE, reused below

            for (int t = 0; t < 3; t++)
            {
                var A0 = arena.fProxyRandomMat(m, n, -5f, 5f, (uint)(2000 + t * 11));
                for (int d = 0; d < n; d++)
                    A0[d, d] += (fProxy)10f;

                // svdThin: allocating reference vs reused workspace
                var Aa = A0.Copy();
                var Ua = arena.fProxyMat(m, n);
                var Sa = arena.fProxyVec(n);
                var Va = arena.fProxyMat(n, n);
                SVD.svdThin(in Aa, ref Ua, ref Sa, ref Va);

                var Aw = A0.Copy();
                var Uw = arena.fProxyMat(m, n);
                var Sw = arena.fProxyVec(n);
                var Vw = arena.fProxyMat(n, n);
                SVD.svdThin(in Aw, ref Uw, ref Sw, ref Vw, ref thinWs);

                Assert.IsTrue(Analysis_OP.isZero(Sa - Sw, Tol()));
                Assert.IsTrue(Analysis_OP.isZero(Ua - Uw, Tol()));
                Assert.IsTrue(Analysis_OP.isZero(Va - Vw, Tol()));

                // svdValues: allocating reference vs reused workspace
                var Sva = arena.fProxyVec(n);
                SVD.svdValues(in A0, ref Sva);

                var Svw = arena.fProxyVec(n);
                SVD.svdValues(in A0, ref Svw, ref valuesWs);

                Assert.IsTrue(Analysis_OP.isZero(Sva - Svw, Tol()));
            }

            arena.Dispose();
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(WorkspaceEquivJob.TestType));

    [TestCaseSource("GetEnums")]
    public void WorkspaceEquivTests(WorkspaceEquivJob.TestType type)
    {
        new WorkspaceEquivJob() { Type = type }.Run();
    }

    // ---- mis-sized scratch guards (managed [Test]; run on a normal C# thread, outside a job) ----

    [Test]
    public void SvdThin_BadScratchB_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(6, 4);
            var U = arena.fProxyMat(6, 4);
            var S = arena.fProxyVec(4);
            var V = arena.fProxyMat(4, 4);
            var ws = arena.fProxySVDThin_WS(6, 4);
            ws.B = arena.fProxyMat(3, 3);   // wrong: must be 4 x 4
            Assert.Throws<ArgumentException>(() => SVD.svdThin(in A, ref U, ref S, ref V, ref ws));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void SvdThin_BadScratchUt_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(6, 4);
            var U = arena.fProxyMat(6, 4);
            var S = arena.fProxyVec(4);
            var V = arena.fProxyMat(4, 4);
            var ws = arena.fProxySVDThin_WS(6, 4);
            ws.Ut = arena.fProxyMat(4, 5);  // wrong: must be n x m = 4 x 6
            Assert.Throws<ArgumentException>(() => SVD.svdThin(in A, ref U, ref S, ref V, ref ws));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void SvdValues_BadScratchD_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(6, 4);
            var S = arena.fProxyVec(4);
            var ws = arena.fProxySVDValues_WS(6, 4);
            ws.dVec = arena.fProxyVec(3);   // wrong: must be length 4
            Assert.Throws<ArgumentException>(() => SVD.svdValues(in A, ref S, ref ws));
        }
        finally { arena.Dispose(); }
    }

    // Arena.fProxySVDThin_WS(m, n) / fProxySVDValues_WS(m, n) must size every field for m x n.
    [Test]
    public void SvdThinValuesWorkspace_Factory_SizesCorrectly()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var thinWs = arena.fProxySVDThin_WS(7, 4);
            Assert.AreEqual(4, thinWs.B.M_Rows);
            Assert.AreEqual(4, thinWs.B.N_Cols);
            Assert.AreEqual(4, thinWs.dVec.N);
            Assert.AreEqual(4, thinWs.eVec.N);
            Assert.AreEqual(4, thinWs.Ut.M_Rows);
            Assert.AreEqual(7, thinWs.Ut.N_Cols);
            Assert.AreEqual(4, thinWs.Vt.M_Rows);
            Assert.AreEqual(4, thinWs.Vt.N_Cols);
            Assert.AreEqual(7, thinWs.BidiagWs.W.M_Rows);
            Assert.AreEqual(4, thinWs.BidiagWs.W.N_Cols);

            var valuesWs = arena.fProxySVDValues_WS(7, 4);
            Assert.AreEqual(4, valuesWs.dVec.N);
            Assert.AreEqual(4, valuesWs.eVec.N);
            Assert.AreEqual(7, valuesWs.BidiagWs.W.M_Rows);
            Assert.AreEqual(4, valuesWs.BidiagWs.W.N_Cols);
        }
        finally { arena.Dispose(); }
    }
}
