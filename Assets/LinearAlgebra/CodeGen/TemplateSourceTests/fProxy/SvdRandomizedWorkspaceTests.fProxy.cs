using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Workspace-overload tests for SVD.svdRandomized (Halko-Martinsson-Tropp) and its workspace
// fProxySVDRandomizedCache (Arena.fProxySVDRandomizedCache(m, n, k, oversample) and the
// default-oversample (m, n, k) factory).
//
// The allocating overload and the ref-workspace overload run the SAME sketch/QR/exact-SVD pipeline;
// with the SAME seed/oversample/powerIters the Gaussian sketch is regenerated identically, so the
// outputs are bit-identical. Tests:
//   (a) EQUIVALENCE — explicit-args overload and the convenience (seed-only, default oversample 10)
//                     overload each match their allocating twin exactly.
//   (b) REUSE       — ONE workspace reused across two different (same-shape) inputs; the 2nd result
//                     equals a fresh allocating call.
//   (c) MIS-SIZED   — a workspace sized for the wrong dimension throws ArgumentException (managed).
public class fProxySvdRandomizedWorkspaceTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
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

        // A = B (m x r) * C (r x n): exactly rank r, so svdRandomized with k >= r is well determined.
        static fProxyMxN RankR(ref Arena arena, int m, int n, int r, uint seed)
        {
            var B = arena.fProxyRandomMat(m, r, (fProxy)(-2f), (fProxy)2f, seed);
            var C = arena.fProxyRandomMat(r, n, (fProxy)(-2f), (fProxy)2f, seed + 13u);
            return Linear_OP.dot(B, C);
        }

        // Explicit oversample/powerIters/seed/maxIter: ws overload == allocating overload exactly.
        void ExplicitArgsEquiv()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 14, n = 6, k = 3, oversample = 4, powerIters = 2, maxIter = 75;
            uint seed = 24681u;
            var A = RankR(ref arena, m, n, 5, 9001);

            var UkA = arena.fProxyMat(m, k); var SkA = arena.fProxyVec(k); var VkA = arena.fProxyMat(n, k);
            bool okA = SVD.svdRandomized(in A, ref UkA, ref SkA, ref VkA, k, oversample, powerIters, seed, maxIter);

            var ws = arena.fProxySVDRandomizedCache(m, n, k, oversample);
            var UkW = arena.fProxyMat(m, k); var SkW = arena.fProxyVec(k); var VkW = arena.fProxyMat(n, k);
            bool okW = SVD.svdRandomized(in A, ref UkW, ref SkW, ref VkW, k, oversample, powerIters, seed, maxIter, ref ws);

            Assert.IsTrue(okA == okW);
            Assert.IsTrue(Analysis_OP.isZero(SkA - SkW, Tol()));
            Assert.IsTrue(Analysis_OP.isZero(UkA - UkW, Tol()));
            Assert.IsTrue(Analysis_OP.isZero(VkA - VkW, Tol()));

            arena.Dispose();
        }

        // The (m, n, k) factory (default oversample 10) feeds the seed-only convenience overload
        // (oversample 10, powerIters 2, maxIter 75) and must match its allocating twin exactly.
        void DefaultFactoryEquiv()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 12, n = 6, k = 3;
            uint seed = 13579u;
            var A = RankR(ref arena, m, n, 4, 9002);

            var UkA = arena.fProxyMat(m, k); var SkA = arena.fProxyVec(k); var VkA = arena.fProxyMat(n, k);
            bool okA = SVD.svdRandomized(in A, ref UkA, ref SkA, ref VkA, k, seed);

            var ws = arena.fProxySVDRandomizedCache(m, n, k);   // default oversample 10
            var UkW = arena.fProxyMat(m, k); var SkW = arena.fProxyVec(k); var VkW = arena.fProxyMat(n, k);
            bool okW = SVD.svdRandomized(in A, ref UkW, ref SkW, ref VkW, k, seed, ref ws);

            Assert.IsTrue(okA == okW);
            Assert.IsTrue(Analysis_OP.isZero(SkA - SkW, Tol()));
            Assert.IsTrue(Analysis_OP.isZero(UkA - UkW, Tol()));
            Assert.IsTrue(Analysis_OP.isZero(VkA - VkW, Tol()));

            arena.Dispose();
        }

        // Reuse ONE workspace across two different (same-shape) inputs; the 2nd result must match a
        // fresh allocating call (proves the dozen internal sketch buffers carry no stale state).
        void WorkspaceReuse()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 14, n = 6, k = 3, oversample = 4, powerIters = 2, maxIter = 75;
            uint seed = 99173u;

            var A1 = RankR(ref arena, m, n, 5, 7001);
            var A2 = RankR(ref arena, m, n, 4, 8002);

            var ws = arena.fProxySVDRandomizedCache(m, n, k, oversample);   // ONCE

            // warm the workspace on A1
            var U1 = arena.fProxyMat(m, k); var S1 = arena.fProxyVec(k); var V1 = arena.fProxyMat(n, k);
            SVD.svdRandomized(in A1, ref U1, ref S1, ref V1, k, oversample, powerIters, seed, maxIter, ref ws);

            // reuse on A2
            var UW = arena.fProxyMat(m, k); var SW = arena.fProxyVec(k); var VW = arena.fProxyMat(n, k);
            bool okW = SVD.svdRandomized(in A2, ref UW, ref SW, ref VW, k, oversample, powerIters, seed, maxIter, ref ws);

            // fresh allocating reference on A2
            var UA = arena.fProxyMat(m, k); var SA = arena.fProxyVec(k); var VA = arena.fProxyMat(n, k);
            bool okA = SVD.svdRandomized(in A2, ref UA, ref SA, ref VA, k, oversample, powerIters, seed, maxIter);

            Assert.IsTrue(okW == okA);
            Assert.IsTrue(Analysis_OP.isZero(SW - SA, Tol()));
            Assert.IsTrue(Analysis_OP.isZero(UW - UA, Tol()));
            Assert.IsTrue(Analysis_OP.isZero(VW - VA, Tol()));

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
    public void Randomized_BadWorkspaceM_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            int m = 14, n = 6, k = 3, oversample = 4;
            var A = arena.fProxyMat(m, n);
            var Uk = arena.fProxyMat(m, k); var Sk = arena.fProxyVec(k); var Vk = arena.fProxyMat(n, k);
            var ws = arena.fProxySVDRandomizedCache(m + 1, n, k, oversample);   // wrong m
            Assert.Throws<ArgumentException>(
                () => SVD.svdRandomized(in A, ref Uk, ref Sk, ref Vk, k, oversample, 2, 123u, 75, ref ws));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Randomized_BadWorkspaceOversample_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            int m = 14, n = 6, k = 3;
            var A = arena.fProxyMat(m, n);
            var Uk = arena.fProxyMat(m, k); var Sk = arena.fProxyVec(k); var Vk = arena.fProxyMat(n, k);
            // ws sketch width l = min(3 + 0, 6) = 3, but the call uses oversample 2 -> l = min(5, 6) = 5.
            var ws = arena.fProxySVDRandomizedCache(m, n, k, 0);
            Assert.Throws<ArgumentException>(
                () => SVD.svdRandomized(in A, ref Uk, ref Sk, ref Vk, k, 2, 2, 123u, 75, ref ws));
        }
        finally { arena.Dispose(); }
    }

    // Arena.fProxySVDRandomizedCache(m, n, k, oversample) sizes everything by l = min(k+p, n).
    [Test]
    public void SvdRandomizedWorkspace_Factory_SizesCorrectly()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            int m = 14, n = 6, k = 3, oversample = 4;   // l = min(7, 6) = 6
            int l = 6;
            var ws = arena.fProxySVDRandomizedCache(m, n, k, oversample);
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
            var wsDef = arena.fProxySVDRandomizedCache(m, n, k);
            Assert.AreEqual(l, wsDef.Omega.N_Cols);
        }
        finally { arena.Dispose(); }
    }
}
