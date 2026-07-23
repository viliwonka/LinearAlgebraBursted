using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Phase-2 solver-workspace tests for thin/values: the caller-provided-scratch overloads
// (thin(...,ref fProxySVDThinCache) / values(...,ref fProxySVDValuesCache)) must produce results
// identical to the allocating wrappers (they run the SAME kernel), and a mis-sized/reused workspace
// must behave correctly (throw on bad size, produce identical results across repeated reuse).
public class fProxySVDThinValuesWorkspaceTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
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

        // thin scratch overload must match the allocating wrapper.
        void SvdThinEquiv(int m, int n)
        {
            var A0 = GenerateOP.fProxyRandomMat(m, n, -5f, 5f, 55123, allocator: Allocator.Temp);
            for (int d = 0; d < n; d++)   // boost leading diagonal block for conditioning
                A0[d, d] += (fProxy)10f;

            // allocating reference
            var Aa = A0.Copy();
            var Ua = new fProxyMxN(m, n, Allocator.Temp);
            var Sa = new fProxyN(n, Allocator.Temp);
            var Va = new fProxyMxN(n, n, Allocator.Temp);
            bool oka = SVD.thin(in Aa, ref Ua, ref Sa, ref Va);

            // workspace-struct form (default maxIter/eps) must match the allocating form
            var Ab = A0.Copy();
            var Ub = new fProxyMxN(m, n, Allocator.Temp);
            var Sb = new fProxyN(n, Allocator.Temp);
            var Vb = new fProxyMxN(n, n, Allocator.Temp);
            var ws = new fProxySVDThinCache(m, n, Allocator.Temp);
            bool okb = SVD.thin(in Ab, ref Ub, ref Sb, ref Vb, ref ws);

            Assert.IsTrue(oka == okb);
            var SDiff = new fProxyN(in Sa, Allocator.Temp); SDiff.subInPlace(Sb);
            Assert.IsTrue(Analysis.isZero(SDiff, Tol()));
            var UDiff = new fProxyMxN(in Ua, Allocator.Temp); UDiff.subInPlace(Ub);
            Assert.IsTrue(Analysis.isZero(UDiff, Tol()));
            var VDiff = new fProxyMxN(in Va, Allocator.Temp); VDiff.subInPlace(Vb);
            Assert.IsTrue(Analysis.isZero(VDiff, Tol()));
        }

        // values scratch overload must match the allocating wrapper.
        void SvdValuesEquiv(int m, int n)
        {
            var A0 = GenerateOP.fProxyRandomMat(m, n, -5f, 5f, 90210, allocator: Allocator.Temp);
            for (int d = 0; d < n; d++)
                A0[d, d] += (fProxy)10f;

            var Sa = new fProxyN(n, Allocator.Temp);
            bool oka = SVD.values(in A0, ref Sa);

            var Sb = new fProxyN(n, Allocator.Temp);
            var ws = new fProxySVDValuesCache(m, n, Allocator.Temp);
            bool okb = SVD.values(in A0, ref Sb, ref ws);

            Assert.IsTrue(oka == okb);
            var SDiff = new fProxyN(in Sa, Allocator.Temp); SDiff.subInPlace(Sb);
            Assert.IsTrue(Analysis.isZero(SDiff, Tol()));
        }

        // Reuse ONE workspace (of each kind) across several consecutive solves: each solve must
        // match a fresh allocating solve, proving no stale state survives reuse.
        void WorkspaceReuse()
        {
            int m = 8, n = 4;

            var thinWs = new fProxySVDThinCache(m, n, Allocator.Temp);     // allocated ONCE, reused below
            var valuesWs = new fProxySVDValuesCache(m, n, Allocator.Temp); // allocated ONCE, reused below

            for (int t = 0; t < 3; t++)
            {
                var A0 = GenerateOP.fProxyRandomMat(m, n, -5f, 5f, (uint)(2000 + t * 11), allocator: Allocator.Temp);
                for (int d = 0; d < n; d++)
                    A0[d, d] += (fProxy)10f;

                // thin: allocating reference vs reused workspace
                var Aa = A0.Copy();
                var Ua = new fProxyMxN(m, n, Allocator.Temp);
                var Sa = new fProxyN(n, Allocator.Temp);
                var Va = new fProxyMxN(n, n, Allocator.Temp);
                SVD.thin(in Aa, ref Ua, ref Sa, ref Va);

                var Aw = A0.Copy();
                var Uw = new fProxyMxN(m, n, Allocator.Temp);
                var Sw = new fProxyN(n, Allocator.Temp);
                var Vw = new fProxyMxN(n, n, Allocator.Temp);
                SVD.thin(in Aw, ref Uw, ref Sw, ref Vw, ref thinWs);

                var SDiff = new fProxyN(in Sa, Allocator.Temp); SDiff.subInPlace(Sw);
                Assert.IsTrue(Analysis.isZero(SDiff, Tol()));
                var UDiff = new fProxyMxN(in Ua, Allocator.Temp); UDiff.subInPlace(Uw);
                Assert.IsTrue(Analysis.isZero(UDiff, Tol()));
                var VDiff = new fProxyMxN(in Va, Allocator.Temp); VDiff.subInPlace(Vw);
                Assert.IsTrue(Analysis.isZero(VDiff, Tol()));

                // values: allocating reference vs reused workspace
                var Sva = new fProxyN(n, Allocator.Temp);
                SVD.values(in A0, ref Sva);

                var Svw = new fProxyN(n, Allocator.Temp);
                SVD.values(in A0, ref Svw, ref valuesWs);

                var SvDiff = new fProxyN(in Sva, Allocator.Temp); SvDiff.subInPlace(Svw);
                Assert.IsTrue(Analysis.isZero(SvDiff, Tol()));
            }
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
        var A = new fProxyMxN(6, 4, Allocator.Temp);
        var U = new fProxyMxN(6, 4, Allocator.Temp);
        var S = new fProxyN(4, Allocator.Temp);
        var V = new fProxyMxN(4, 4, Allocator.Temp);
        var ws = new fProxySVDThinCache(6, 4, Allocator.Temp);
        ws.B = new fProxyMxN(3, 3, Allocator.Temp);   // wrong: must be 4 x 4
        Assert.Throws<ArgumentException>(() => SVD.thin(in A, ref U, ref S, ref V, ref ws));
    }

    [Test]
    public void SvdThin_BadScratchUt_Throws()
    {
        var A = new fProxyMxN(6, 4, Allocator.Temp);
        var U = new fProxyMxN(6, 4, Allocator.Temp);
        var S = new fProxyN(4, Allocator.Temp);
        var V = new fProxyMxN(4, 4, Allocator.Temp);
        var ws = new fProxySVDThinCache(6, 4, Allocator.Temp);
        ws.Ut = new fProxyMxN(4, 5, Allocator.Temp);  // wrong: must be n x m = 4 x 6
        Assert.Throws<ArgumentException>(() => SVD.thin(in A, ref U, ref S, ref V, ref ws));
    }

    [Test]
    public void SvdValues_BadScratchD_Throws()
    {
        var A = new fProxyMxN(6, 4, Allocator.Temp);
        var S = new fProxyN(4, Allocator.Temp);
        var ws = new fProxySVDValuesCache(6, 4, Allocator.Temp);
        ws.dVec = new fProxyN(3, Allocator.Temp);   // wrong: must be length 4
        Assert.Throws<ArgumentException>(() => SVD.values(in A, ref S, ref ws));
    }

    // Standalone fProxySVDThinCache(m, n, allocator) / fProxySVDValuesCache(m, n, allocator) must
    // size every field for m x n.
    [Test]
    public void SvdThinValuesWorkspace_Factory_SizesCorrectly()
    {
        var thinWs = new fProxySVDThinCache(7, 4, Allocator.Temp);
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

        var valuesWs = new fProxySVDValuesCache(7, 4, Allocator.Temp);
        Assert.AreEqual(4, valuesWs.dVec.N);
        Assert.AreEqual(4, valuesWs.eVec.N);
        Assert.AreEqual(7, valuesWs.BidiagWs.W.M_Rows);
        Assert.AreEqual(4, valuesWs.BidiagWs.W.N_Cols);
    }
}
