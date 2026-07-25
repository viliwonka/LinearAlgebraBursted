using System;

using BULA;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Workspace-overload tests for SVD.randomized (Halko-Martinsson-Tropp) and its workspace
// fProxySVDRandomizedCache (the standalone (m, n, k, oversample, Allocator) ctor and the
// default-oversample (m, n, k, Allocator) ctor).
//
// The allocating overload and the ref-workspace overload run the SAME sketch/QR/exact-SVD pipeline;
// with the SAME seed/oversample/powerIters the Gaussian sketch is regenerated identically, so the
// outputs are bit-identical. Tests:
//   (a) EQUIVALENCE — explicit-args overload and the convenience (seed-only, default oversample 10)
//                     overload each match their allocating twin exactly.
//   (b) REUSE       — ONE workspace reused across two different (same-shape) inputs; the 2nd result
//                     equals a fresh allocating call.
//   (c) MIS-SIZED   — a workspace sized for the wrong dimension throws ArgumentException (managed).
public class fProxySVDRandomizedWorkspaceTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WorkspaceJob : IJob
    {
        public enum TestType
        {
            ExplicitArgsEquiv,
            DefaultFactoryEquiv,
            WorkspaceReuse,
        }

        public TestType Type;

        static fProxy Tol() => 256 * Consts.fProxySqrtEps;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ExplicitArgsEquiv:   ExplicitArgsEquiv();   break;
                case TestType.DefaultFactoryEquiv: DefaultFactoryEquiv(); break;
                case TestType.WorkspaceReuse:      WorkspaceReuse();      break;
            }
        }

        // A = B (m x r) * C (r x n): exactly rank r, so randomized with k >= r is well determined.
        static fProxyMxN RankR(int m, int n, int r, uint seed)
        {
            var B = GenerateOP.fProxyRandomMat(m, r, (fProxy)(-2f), (fProxy)2f, seed, allocator: Allocator.Temp);
            var C = GenerateOP.fProxyRandomMat(r, n, (fProxy)(-2f), (fProxy)2f, seed + 13u, allocator: Allocator.Temp);
            return Blas.dot(B, C);
        }

        // Explicit oversample/powerIters/seed/maxIter: ws overload == allocating overload exactly.
        void ExplicitArgsEquiv()
        {
            int m = 14, n = 6, k = 3, oversample = 4, powerIters = 2, maxIter = 75;
            uint seed = 24681u;
            var A = RankR(m, n, 5, 9001);

            var UkA = new fProxyMxN(m, k, Allocator.Temp); var SkA = new fProxyN(k, Allocator.Temp); var VkA = new fProxyMxN(n, k, Allocator.Temp);
            bool okA = SVD.randomized(in A, ref UkA, ref SkA, ref VkA, k, oversample, powerIters, seed, maxIter);

            var ws = new fProxySVDRandomizedCache(m, n, k, oversample, Allocator.Temp);
            var UkW = new fProxyMxN(m, k, Allocator.Temp); var SkW = new fProxyN(k, Allocator.Temp); var VkW = new fProxyMxN(n, k, Allocator.Temp);
            bool okW = SVD.randomized(in A, ref UkW, ref SkW, ref VkW, k, oversample, powerIters, seed, maxIter, ref ws);

            Assert.IsTrue(okA == okW);
            var SDiff = new fProxyN(in SkA, Allocator.Temp); SDiff.subInPlace(SkW);
            Assert.IsTrue(Analysis.isZero(SDiff, Tol()));
            var UDiff = new fProxyMxN(in UkA, Allocator.Temp); UDiff.subInPlace(UkW);
            Assert.IsTrue(Analysis.isZero(UDiff, Tol()));
            var VDiff = new fProxyMxN(in VkA, Allocator.Temp); VDiff.subInPlace(VkW);
            Assert.IsTrue(Analysis.isZero(VDiff, Tol()));
        }

        // The (m, n, k) factory (default oversample 10) feeds the seed-only convenience overload
        // (oversample 10, powerIters 2, maxIter 75) and must match its allocating twin exactly.
        void DefaultFactoryEquiv()
        {
            int m = 12, n = 6, k = 3;
            uint seed = 13579u;
            var A = RankR(m, n, 4, 9002);

            var UkA = new fProxyMxN(m, k, Allocator.Temp); var SkA = new fProxyN(k, Allocator.Temp); var VkA = new fProxyMxN(n, k, Allocator.Temp);
            bool okA = SVD.randomized(in A, ref UkA, ref SkA, ref VkA, k, seed);

            var ws = new fProxySVDRandomizedCache(m, n, k, Allocator.Temp);   // default oversample 10
            var UkW = new fProxyMxN(m, k, Allocator.Temp); var SkW = new fProxyN(k, Allocator.Temp); var VkW = new fProxyMxN(n, k, Allocator.Temp);
            bool okW = SVD.randomized(in A, ref UkW, ref SkW, ref VkW, k, seed, ref ws);

            Assert.IsTrue(okA == okW);
            var SDiff = new fProxyN(in SkA, Allocator.Temp); SDiff.subInPlace(SkW);
            Assert.IsTrue(Analysis.isZero(SDiff, Tol()));
            var UDiff = new fProxyMxN(in UkA, Allocator.Temp); UDiff.subInPlace(UkW);
            Assert.IsTrue(Analysis.isZero(UDiff, Tol()));
            var VDiff = new fProxyMxN(in VkA, Allocator.Temp); VDiff.subInPlace(VkW);
            Assert.IsTrue(Analysis.isZero(VDiff, Tol()));
        }

        // Reuse ONE workspace across two different (same-shape) inputs; the 2nd result must match a
        // fresh allocating call (proves the dozen internal sketch buffers carry no stale state).
        void WorkspaceReuse()
        {
            int m = 14, n = 6, k = 3, oversample = 4, powerIters = 2, maxIter = 75;
            uint seed = 99173u;

            var A1 = RankR(m, n, 5, 7001);
            var A2 = RankR(m, n, 4, 8002);

            var ws = new fProxySVDRandomizedCache(m, n, k, oversample, Allocator.Temp);   // ONCE

            // warm the workspace on A1
            var U1 = new fProxyMxN(m, k, Allocator.Temp); var S1 = new fProxyN(k, Allocator.Temp); var V1 = new fProxyMxN(n, k, Allocator.Temp);
            SVD.randomized(in A1, ref U1, ref S1, ref V1, k, oversample, powerIters, seed, maxIter, ref ws);

            // reuse on A2
            var UW = new fProxyMxN(m, k, Allocator.Temp); var SW = new fProxyN(k, Allocator.Temp); var VW = new fProxyMxN(n, k, Allocator.Temp);
            bool okW = SVD.randomized(in A2, ref UW, ref SW, ref VW, k, oversample, powerIters, seed, maxIter, ref ws);

            // fresh allocating reference on A2
            var UA = new fProxyMxN(m, k, Allocator.Temp); var SA = new fProxyN(k, Allocator.Temp); var VA = new fProxyMxN(n, k, Allocator.Temp);
            bool okA = SVD.randomized(in A2, ref UA, ref SA, ref VA, k, oversample, powerIters, seed, maxIter);

            Assert.IsTrue(okW == okA);
            var SDiff = new fProxyN(in SW, Allocator.Temp); SDiff.subInPlace(SA);
            Assert.IsTrue(Analysis.isZero(SDiff, Tol()));
            var UDiff = new fProxyMxN(in UW, Allocator.Temp); UDiff.subInPlace(UA);
            Assert.IsTrue(Analysis.isZero(UDiff, Tol()));
            var VDiff = new fProxyMxN(in VW, Allocator.Temp); VDiff.subInPlace(VA);
            Assert.IsTrue(Analysis.isZero(VDiff, Tol()));
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
    public void Randomized_BadWorkspaceM_Throws()
    {
        int m = 14, n = 6, k = 3, oversample = 4;
        var A = new fProxyMxN(m, n, Allocator.Temp);
        var Uk = new fProxyMxN(m, k, Allocator.Temp); var Sk = new fProxyN(k, Allocator.Temp); var Vk = new fProxyMxN(n, k, Allocator.Temp);
        var ws = new fProxySVDRandomizedCache(m + 1, n, k, oversample, Allocator.Temp);   // wrong m
        Assert.Throws<ArgumentException>(
            () => SVD.randomized(in A, ref Uk, ref Sk, ref Vk, k, oversample, 2, 123u, 75, ref ws));
    }

    [Test]
    public void Randomized_BadWorkspaceOversample_Throws()
    {
        int m = 14, n = 6, k = 3;
        var A = new fProxyMxN(m, n, Allocator.Temp);
        var Uk = new fProxyMxN(m, k, Allocator.Temp); var Sk = new fProxyN(k, Allocator.Temp); var Vk = new fProxyMxN(n, k, Allocator.Temp);
        // ws sketch width l = min(3 + 0, 6) = 3, but the call uses oversample 2 -> l = min(5, 6) = 5.
        var ws = new fProxySVDRandomizedCache(m, n, k, 0, Allocator.Temp);
        Assert.Throws<ArgumentException>(
            () => SVD.randomized(in A, ref Uk, ref Sk, ref Vk, k, 2, 2, 123u, 75, ref ws));
    }

    // Standalone fProxySVDRandomizedCache(m, n, k, oversample, allocator) sizes everything by l = min(k+p, n).
    [Test]
    public void SvdRandomizedWorkspace_Factory_SizesCorrectly()
    {
        int m = 14, n = 6, k = 3, oversample = 4;   // l = min(7, 6) = 6
        int l = 6;
        var ws = new fProxySVDRandomizedCache(m, n, k, oversample, Allocator.Temp);
        Assert.AreEqual(n, ws.Omega.M_Rows); Assert.AreEqual(l, ws.Omega.N_Cols);
        Assert.AreEqual(m, ws.Y.M_Rows);     Assert.AreEqual(l, ws.Y.N_Cols);
        Assert.AreEqual(l, ws.R.M_Rows);     Assert.AreEqual(l, ws.R.N_Cols);
        Assert.AreEqual(m, ws.qu.N);
        Assert.AreEqual(l, ws.qw.N);
        Assert.AreEqual(n, ws.B.N_Cols);     Assert.AreEqual(l, ws.B.M_Rows);
        Assert.AreEqual(n, ws.Bt.M_Rows);    Assert.AreEqual(l, ws.Bt.N_Cols);
        Assert.AreEqual(l, ws.Sb.N);
        Assert.AreEqual(m, ws.UA.M_Rows);    Assert.AreEqual(l, ws.UA.N_Cols);

        // default-oversample factory: l = min(3 + 10, 6) = 6 as well
        var wsDef = new fProxySVDRandomizedCache(m, n, k, Allocator.Temp);
        Assert.AreEqual(l, wsDef.Omega.N_Cols);
    }
}
