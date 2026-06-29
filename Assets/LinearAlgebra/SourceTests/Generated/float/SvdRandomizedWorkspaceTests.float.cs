using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Workspace-overload tests for SVD.svdRandomized (Halko-Martinsson-Tropp) and its workspace
// floatSvdRandomizedWorkspace (Arena.floatSvdRandomizedWorkspace(m, n, k, oversample) and the
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
public class floatSvdRandomizedWorkspaceTests
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

        static float Tol() => 256 * Consts.floatSqrtEps;

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
        static floatMxN RankR(ref Arena arena, int m, int n, int r, uint seed)
        {
            var B = arena.floatRandomMatrix(m, r, (float)(-2f), (float)2f, seed);
            var C = arena.floatRandomMatrix(r, n, (float)(-2f), (float)2f, seed + 13u);
            return floatOP.dot(B, C);
        }

        // Explicit oversample/powerIters/seed/maxIter: ws overload == allocating overload exactly.
        void ExplicitArgsEquiv()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 14, n = 6, k = 3, oversample = 4, powerIters = 2, maxIter = 75;
            uint seed = 24681u;
            var A = RankR(ref arena, m, n, 5, 9001);

            var UkA = arena.floatMat(m, k); var SkA = arena.floatVec(k); var VkA = arena.floatMat(n, k);
            bool okA = SVD.svdRandomized(in A, ref UkA, ref SkA, ref VkA, k, oversample, powerIters, seed, maxIter);

            var ws = arena.floatSvdRandomizedWorkspace(m, n, k, oversample);
            var UkW = arena.floatMat(m, k); var SkW = arena.floatVec(k); var VkW = arena.floatMat(n, k);
            bool okW = SVD.svdRandomized(in A, ref UkW, ref SkW, ref VkW, k, oversample, powerIters, seed, maxIter, ref ws);

            Assert.IsTrue(okA == okW);
            Assert.IsTrue(Analysis.IsZero(SkA - SkW, Tol()));
            Assert.IsTrue(Analysis.IsZero(UkA - UkW, Tol()));
            Assert.IsTrue(Analysis.IsZero(VkA - VkW, Tol()));

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

            var UkA = arena.floatMat(m, k); var SkA = arena.floatVec(k); var VkA = arena.floatMat(n, k);
            bool okA = SVD.svdRandomized(in A, ref UkA, ref SkA, ref VkA, k, seed);

            var ws = arena.floatSvdRandomizedWorkspace(m, n, k);   // default oversample 10
            var UkW = arena.floatMat(m, k); var SkW = arena.floatVec(k); var VkW = arena.floatMat(n, k);
            bool okW = SVD.svdRandomized(in A, ref UkW, ref SkW, ref VkW, k, seed, ref ws);

            Assert.IsTrue(okA == okW);
            Assert.IsTrue(Analysis.IsZero(SkA - SkW, Tol()));
            Assert.IsTrue(Analysis.IsZero(UkA - UkW, Tol()));
            Assert.IsTrue(Analysis.IsZero(VkA - VkW, Tol()));

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

            var ws = arena.floatSvdRandomizedWorkspace(m, n, k, oversample);   // ONCE

            // warm the workspace on A1
            var U1 = arena.floatMat(m, k); var S1 = arena.floatVec(k); var V1 = arena.floatMat(n, k);
            SVD.svdRandomized(in A1, ref U1, ref S1, ref V1, k, oversample, powerIters, seed, maxIter, ref ws);

            // reuse on A2
            var UW = arena.floatMat(m, k); var SW = arena.floatVec(k); var VW = arena.floatMat(n, k);
            bool okW = SVD.svdRandomized(in A2, ref UW, ref SW, ref VW, k, oversample, powerIters, seed, maxIter, ref ws);

            // fresh allocating reference on A2
            var UA = arena.floatMat(m, k); var SA = arena.floatVec(k); var VA = arena.floatMat(n, k);
            bool okA = SVD.svdRandomized(in A2, ref UA, ref SA, ref VA, k, oversample, powerIters, seed, maxIter);

            Assert.IsTrue(okW == okA);
            Assert.IsTrue(Analysis.IsZero(SW - SA, Tol()));
            Assert.IsTrue(Analysis.IsZero(UW - UA, Tol()));
            Assert.IsTrue(Analysis.IsZero(VW - VA, Tol()));

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
            var A = arena.floatMat(m, n);
            var Uk = arena.floatMat(m, k); var Sk = arena.floatVec(k); var Vk = arena.floatMat(n, k);
            var ws = arena.floatSvdRandomizedWorkspace(m + 1, n, k, oversample);   // wrong m
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
            var A = arena.floatMat(m, n);
            var Uk = arena.floatMat(m, k); var Sk = arena.floatVec(k); var Vk = arena.floatMat(n, k);
            // ws sketch width l = min(3 + 0, 6) = 3, but the call uses oversample 2 -> l = min(5, 6) = 5.
            var ws = arena.floatSvdRandomizedWorkspace(m, n, k, 0);
            Assert.Throws<ArgumentException>(
                () => SVD.svdRandomized(in A, ref Uk, ref Sk, ref Vk, k, 2, 2, 123u, 75, ref ws));
        }
        finally { arena.Dispose(); }
    }

    // Arena.floatSvdRandomizedWorkspace(m, n, k, oversample) sizes everything by l = min(k+p, n).
    [Test]
    public void SvdRandomizedWorkspace_Factory_SizesCorrectly()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            int m = 14, n = 6, k = 3, oversample = 4;   // l = min(7, 6) = 6
            int l = 6;
            var ws = arena.floatSvdRandomizedWorkspace(m, n, k, oversample);
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
            var wsDef = arena.floatSvdRandomizedWorkspace(m, n, k);
            Assert.AreEqual(l, wsDef.Omega.N_Cols);
        }
        finally { arena.Dispose(); }
    }
}
