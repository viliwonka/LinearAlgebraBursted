using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Workspace-overload tests for SVD family ops with reusable workspaces.
//
// nullspaceBasis, rangeBasis, and lowRankApprox share fProxySVDFullCache.
// svdTruncated (GKL bidiagonalization) uses its own fProxySVDTruncatedCache.
//
// Each op has a `ref <WorkspaceType> ws` overload (caller-owned scratch) PLUS an allocating
// overload. They run the same kernel with the same seed, so for identical inputs the outputs
// are bit-identical.
//
// Three kinds of test per op (mirrors SVDWorkspaceTests):
//   (a) EQUIVALENCE — allocating vs ws on the SAME matrix, outputs identical.
//   (b) REUSE       — workspace reused across two different (same-shape) inputs; the 2nd result
//                     equals a fresh allocating call (proves no stale data carries over).
//   (c) MIS-SIZED   — a workspace sized for the wrong dimension throws ArgumentException (managed).
public class fProxySvdFullWorkspaceTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WorkspaceJob : IJob
    {
        public enum TestType
        {
            NullspaceEquiv,
            RangeEquiv,
            TruncatedEquiv,
            LowRankEquiv,
            ReuseAllOps,
        }

        public TestType Type;

        // Both overloads run the identical kernel, so equality is bit-exact in principle; keep a tiny
        // per-precision band for robustness (loose for float, tight for double).
        static fProxy Tol() => 256 * Consts.fProxySqrtEps;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.NullspaceEquiv: NullspaceEquiv(); break;
                case TestType.RangeEquiv:     RangeEquiv();     break;
                case TestType.TruncatedEquiv: TruncatedEquiv(); break;
                case TestType.LowRankEquiv:   LowRankEquiv();   break;
                case TestType.ReuseAllOps:    ReuseAllOps();    break;
            }
        }

        // A = B (m x r) * C (r x n) -> m x n of generic rank r (r <= n <= m).
        static fProxyMxN RankDeficient(ref Arena arena, int m, int n, int r, uint seed)
        {
            var B = arena.fProxyRandomMat(m, r, (fProxy)(-2f), (fProxy)2f, seed);
            var C = arena.fProxyRandomMat(r, n, (fProxy)(-2f), (fProxy)2f, seed + 7u);
            return Linear_OP.dot(B, C);
        }

        void NullspaceEquiv()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 6, n = 4;
            var A = RankDeficient(ref arena, m, n, 2, 1001);   // rank 2 -> nullspace dim 2

            var basisA = arena.fProxyMat(n, n);
            int dimA = SVD.nullspaceBasis(in A, ref basisA, out bool cA);

            var ws = arena.fProxySVDFullCache(m, n);
            var basisW = arena.fProxyMat(n, n);
            int dimW = SVD.nullspaceBasis(in A, ref basisW, ref ws, out bool cW);

            Assert.IsTrue(dimA == dimW);
            Assert.IsTrue(cA == cW);
            Assert.IsTrue(Analysis_OP.isZero(basisA - basisW, Tol()));

            arena.Dispose();
        }

        void RangeEquiv()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 6, n = 4;
            var A = RankDeficient(ref arena, m, n, 3, 2002);   // rank 3 -> range rank 3

            var basisA = arena.fProxyMat(m, n);
            int rankA = SVD.rangeBasis(in A, ref basisA, out bool cA);

            var ws = arena.fProxySVDFullCache(m, n);
            var basisW = arena.fProxyMat(m, n);
            int rankW = SVD.rangeBasis(in A, ref basisW, ref ws, out bool cW);

            Assert.IsTrue(rankA == rankW);
            Assert.IsTrue(cA == cW);
            Assert.IsTrue(Analysis_OP.isZero(basisA - basisW, Tol()));

            arena.Dispose();
        }

        void TruncatedEquiv()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 6, n = 4, k = 2;
            var A = arena.fProxyRandomMat(m, n, (fProxy)(-3f), (fProxy)3f, 3003);

            var UkA = arena.fProxyMat(m, k); var SkA = arena.fProxyVec(k); var VkA = arena.fProxyMat(n, k);
            SVD.svdTruncated(in A, ref UkA, ref SkA, ref VkA, k, out bool cA);

            var ws = arena.fProxySVDTruncatedCache(m, n, k);
            var UkW = arena.fProxyMat(m, k); var SkW = arena.fProxyVec(k); var VkW = arena.fProxyMat(n, k);
            SVD.svdTruncated(in A, ref UkW, ref SkW, ref VkW, k, ref ws, out bool cW);

            Assert.IsTrue(cA == cW);
            Assert.IsTrue(Analysis_OP.isZero(SkA - SkW, Tol()));
            Assert.IsTrue(Analysis_OP.isZero(UkA - UkW, Tol()));
            Assert.IsTrue(Analysis_OP.isZero(VkA - VkW, Tol()));

            arena.Dispose();
        }

        void LowRankEquiv()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 6, n = 4, k = 2;
            var A = arena.fProxyRandomMat(m, n, (fProxy)(-3f), (fProxy)3f, 4004);

            var AkA = arena.fProxyMat(m, n);
            SVD.lowRankApprox(in A, ref AkA, k, out bool cA);

            var ws = arena.fProxySVDFullCache(m, n);
            var AkW = arena.fProxyMat(m, n);
            SVD.lowRankApprox(in A, ref AkW, k, ref ws, out bool cW);

            Assert.IsTrue(cA == cW);
            Assert.IsTrue(Analysis_OP.isZero(AkA - AkW, Tol()));

            arena.Dispose();
        }

        // Reuse workspaces across two different (same-shape) inputs and every family op; each
        // second-input result must match a fresh allocating call -> no stale data survives reuse.
        void ReuseAllOps()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 6, n = 4, k = 2;

            var A1 = RankDeficient(ref arena, m, n, 2, 5005);
            var A2 = RankDeficient(ref arena, m, n, 3, 6006);

            var ws = arena.fProxySVDFullCache(m, n);           // for nullspace / range / lowRank
            var wsTrunc = arena.fProxySVDTruncatedCache(m, n, k);  // for svdTruncated (GKL)

            // ---- nullspace ----
            var nb1 = arena.fProxyMat(n, n);
            SVD.nullspaceBasis(in A1, ref nb1, ref ws, out bool _);          // warm the workspace on A1
            var nbW = arena.fProxyMat(n, n);
            int dimW = SVD.nullspaceBasis(in A2, ref nbW, ref ws, out bool _);
            var nbA = arena.fProxyMat(n, n);
            int dimA = SVD.nullspaceBasis(in A2, ref nbA, out bool _);
            Assert.IsTrue(dimW == dimA);
            Assert.IsTrue(Analysis_OP.isZero(nbW - nbA, Tol()));

            // ---- range ----
            var rb1 = arena.fProxyMat(m, n);
            SVD.rangeBasis(in A1, ref rb1, ref ws, out bool _);
            var rbW = arena.fProxyMat(m, n);
            int rkW = SVD.rangeBasis(in A2, ref rbW, ref ws, out bool _);
            var rbA = arena.fProxyMat(m, n);
            int rkA = SVD.rangeBasis(in A2, ref rbA, out bool _);
            Assert.IsTrue(rkW == rkA);
            Assert.IsTrue(Analysis_OP.isZero(rbW - rbA, Tol()));

            // ---- truncated ----
            var U1 = arena.fProxyMat(m, k); var S1 = arena.fProxyVec(k); var V1 = arena.fProxyMat(n, k);
            SVD.svdTruncated(in A1, ref U1, ref S1, ref V1, k, ref wsTrunc, out bool _);
            var UW = arena.fProxyMat(m, k); var SW = arena.fProxyVec(k); var VW = arena.fProxyMat(n, k);
            SVD.svdTruncated(in A2, ref UW, ref SW, ref VW, k, ref wsTrunc, out bool _);
            var UA = arena.fProxyMat(m, k); var SA = arena.fProxyVec(k); var VA = arena.fProxyMat(n, k);
            SVD.svdTruncated(in A2, ref UA, ref SA, ref VA, k, out bool _);
            Assert.IsTrue(Analysis_OP.isZero(SW - SA, Tol()));
            Assert.IsTrue(Analysis_OP.isZero(UW - UA, Tol()));
            Assert.IsTrue(Analysis_OP.isZero(VW - VA, Tol()));

            // ---- low rank ----
            var Ak1 = arena.fProxyMat(m, n);
            SVD.lowRankApprox(in A1, ref Ak1, k, ref ws, out bool _);
            var AkW = arena.fProxyMat(m, n);
            SVD.lowRankApprox(in A2, ref AkW, k, ref ws, out bool _);
            var AkA = arena.fProxyMat(m, n);
            SVD.lowRankApprox(in A2, ref AkA, k, out bool _);
            Assert.IsTrue(Analysis_OP.isZero(AkW - AkA, Tol()));

            arena.Dispose();
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(WorkspaceJob.TestType));

    [TestCaseSource("GetEnums")]
    public void WorkspaceTests(WorkspaceJob.TestType type)
    {
        new WorkspaceJob() { Type = type }.Run();
    }

    // ---- mis-sized workspace guards (managed [Test]; the guard runs outside Burst) ----

    [Test]
    public void Nullspace_BadWorkspace_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(6, 4);
            var basis = arena.fProxyMat(4, 4);
            var ws = arena.fProxySVDFullCache(5, 4);   // wrong m
            Assert.Throws<ArgumentException>(
                () => SVD.nullspaceBasis(in A, ref basis, ref ws, out bool _));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Range_BadWorkspace_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(6, 4);
            var basis = arena.fProxyMat(6, 4);
            var ws = arena.fProxySVDFullCache(6, 3);   // wrong n
            Assert.Throws<ArgumentException>(
                () => SVD.rangeBasis(in A, ref basis, ref ws, out bool _));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Truncated_BadWorkspace_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(6, 4);
            var Uk = arena.fProxyMat(6, 2); var Sk = arena.fProxyVec(2); var Vk = arena.fProxyMat(4, 2);
            var ws = arena.fProxySVDTruncatedCache(7, 4, 2);   // wrong m (7 vs A's 6)
            Assert.Throws<ArgumentException>(
                () => SVD.svdTruncated(in A, ref Uk, ref Sk, ref Vk, 2, ref ws, out bool _));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void LowRank_BadWorkspace_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(6, 4);
            var Ak = arena.fProxyMat(6, 4);
            var ws = arena.fProxySVDFullCache(6, 5);   // wrong n
            Assert.Throws<ArgumentException>(
                () => SVD.lowRankApprox(in A, ref Ak, 2, ref ws, out bool _));
        }
        finally { arena.Dispose(); }
    }

    // Arena.fProxySVDFullCache(m, n) must size U (m x n), S (n), V (n x n).
    [Test]
    public void SvdFullWorkspace_Factory_SizesCorrectly()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var ws = arena.fProxySVDFullCache(7, 4);
            Assert.AreEqual(7, ws.U.M_Rows);
            Assert.AreEqual(4, ws.U.N_Cols);
            Assert.AreEqual(4, ws.S.N);
            Assert.AreEqual(4, ws.V.M_Rows);
            Assert.AreEqual(4, ws.V.N_Cols);
        }
        finally { arena.Dispose(); }
    }
}
