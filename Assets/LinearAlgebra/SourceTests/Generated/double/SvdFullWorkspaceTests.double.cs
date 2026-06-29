using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Workspace-overload tests for the full-SVD-family ops that share doubleSvdFullWorkspace
// (Arena.doubleSvdFullWorkspace(m, n)): nullspaceBasis, rangeBasis, svdTruncated, lowRankApprox.
//
// Each op now has a `ref doubleSvdFullWorkspace ws` overload (the real body, caller-owned U/S/V
// scratch) PLUS the original allocating overload (delegates with temp scratch). They run the SAME
// Golub-Kahan kernel, so for identical inputs the outputs are bit-identical.
//
// Three kinds of test per op (mirrors SVDWorkspaceTests):
//   (a) EQUIVALENCE — allocating vs ws on the SAME matrix, outputs identical.
//   (b) REUSE       — ONE workspace reused across two different (same-shape) inputs; the 2nd result
//                     equals a fresh allocating call (proves no stale U/S/V carries over).
//   (c) MIS-SIZED   — a workspace sized for the wrong dimension throws ArgumentException (managed).
public class doubleSvdFullWorkspaceTests
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
        static double Tol() => 256 * Consts.doubleSqrtEps;

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
        static doubleMxN RankDeficient(ref Arena arena, int m, int n, int r, uint seed)
        {
            var B = arena.doubleRandomMatrix(m, r, (double)(-2f), (double)2f, seed);
            var C = arena.doubleRandomMatrix(r, n, (double)(-2f), (double)2f, seed + 7u);
            return doubleOP.dot(B, C);
        }

        void NullspaceEquiv()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 6, n = 4;
            var A = RankDeficient(ref arena, m, n, 2, 1001);   // rank 2 -> nullspace dim 2

            var basisA = arena.doubleMat(n, n);
            int dimA = SVD.nullspaceBasis(in A, ref basisA, out bool cA);

            var ws = arena.doubleSvdFullWorkspace(m, n);
            var basisW = arena.doubleMat(n, n);
            int dimW = SVD.nullspaceBasis(in A, ref basisW, ref ws, out bool cW);

            Assert.IsTrue(dimA == dimW);
            Assert.IsTrue(cA == cW);
            Assert.IsTrue(Analysis.IsZero(basisA - basisW, Tol()));

            arena.Dispose();
        }

        void RangeEquiv()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 6, n = 4;
            var A = RankDeficient(ref arena, m, n, 3, 2002);   // rank 3 -> range rank 3

            var basisA = arena.doubleMat(m, n);
            int rankA = SVD.rangeBasis(in A, ref basisA, out bool cA);

            var ws = arena.doubleSvdFullWorkspace(m, n);
            var basisW = arena.doubleMat(m, n);
            int rankW = SVD.rangeBasis(in A, ref basisW, ref ws, out bool cW);

            Assert.IsTrue(rankA == rankW);
            Assert.IsTrue(cA == cW);
            Assert.IsTrue(Analysis.IsZero(basisA - basisW, Tol()));

            arena.Dispose();
        }

        void TruncatedEquiv()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 6, n = 4, k = 2;
            var A = arena.doubleRandomMatrix(m, n, (double)(-3f), (double)3f, 3003);

            var UkA = arena.doubleMat(m, k); var SkA = arena.doubleVec(k); var VkA = arena.doubleMat(n, k);
            SVD.svdTruncated(in A, ref UkA, ref SkA, ref VkA, k, out bool cA);

            var ws = arena.doubleSvdFullWorkspace(m, n);
            var UkW = arena.doubleMat(m, k); var SkW = arena.doubleVec(k); var VkW = arena.doubleMat(n, k);
            SVD.svdTruncated(in A, ref UkW, ref SkW, ref VkW, k, ref ws, out bool cW);

            Assert.IsTrue(cA == cW);
            Assert.IsTrue(Analysis.IsZero(SkA - SkW, Tol()));
            Assert.IsTrue(Analysis.IsZero(UkA - UkW, Tol()));
            Assert.IsTrue(Analysis.IsZero(VkA - VkW, Tol()));

            arena.Dispose();
        }

        void LowRankEquiv()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 6, n = 4, k = 2;
            var A = arena.doubleRandomMatrix(m, n, (double)(-3f), (double)3f, 4004);

            var AkA = arena.doubleMat(m, n);
            SVD.lowRankApprox(in A, ref AkA, k, out bool cA);

            var ws = arena.doubleSvdFullWorkspace(m, n);
            var AkW = arena.doubleMat(m, n);
            SVD.lowRankApprox(in A, ref AkW, k, ref ws, out bool cW);

            Assert.IsTrue(cA == cW);
            Assert.IsTrue(Analysis.IsZero(AkA - AkW, Tol()));

            arena.Dispose();
        }

        // Reuse ONE workspace across two different (same-shape) inputs and every family op; each
        // second-input result must match a fresh allocating call -> no stale U/S/V survives reuse.
        void ReuseAllOps()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 6, n = 4, k = 2;

            var A1 = RankDeficient(ref arena, m, n, 2, 5005);
            var A2 = RankDeficient(ref arena, m, n, 3, 6006);

            var ws = arena.doubleSvdFullWorkspace(m, n);   // allocated ONCE

            // ---- nullspace ----
            var nb1 = arena.doubleMat(n, n);
            SVD.nullspaceBasis(in A1, ref nb1, ref ws, out bool _);          // warm the workspace on A1
            var nbW = arena.doubleMat(n, n);
            int dimW = SVD.nullspaceBasis(in A2, ref nbW, ref ws, out bool _);
            var nbA = arena.doubleMat(n, n);
            int dimA = SVD.nullspaceBasis(in A2, ref nbA, out bool _);
            Assert.IsTrue(dimW == dimA);
            Assert.IsTrue(Analysis.IsZero(nbW - nbA, Tol()));

            // ---- range ----
            var rb1 = arena.doubleMat(m, n);
            SVD.rangeBasis(in A1, ref rb1, ref ws, out bool _);
            var rbW = arena.doubleMat(m, n);
            int rkW = SVD.rangeBasis(in A2, ref rbW, ref ws, out bool _);
            var rbA = arena.doubleMat(m, n);
            int rkA = SVD.rangeBasis(in A2, ref rbA, out bool _);
            Assert.IsTrue(rkW == rkA);
            Assert.IsTrue(Analysis.IsZero(rbW - rbA, Tol()));

            // ---- truncated ----
            var U1 = arena.doubleMat(m, k); var S1 = arena.doubleVec(k); var V1 = arena.doubleMat(n, k);
            SVD.svdTruncated(in A1, ref U1, ref S1, ref V1, k, ref ws, out bool _);
            var UW = arena.doubleMat(m, k); var SW = arena.doubleVec(k); var VW = arena.doubleMat(n, k);
            SVD.svdTruncated(in A2, ref UW, ref SW, ref VW, k, ref ws, out bool _);
            var UA = arena.doubleMat(m, k); var SA = arena.doubleVec(k); var VA = arena.doubleMat(n, k);
            SVD.svdTruncated(in A2, ref UA, ref SA, ref VA, k, out bool _);
            Assert.IsTrue(Analysis.IsZero(SW - SA, Tol()));
            Assert.IsTrue(Analysis.IsZero(UW - UA, Tol()));
            Assert.IsTrue(Analysis.IsZero(VW - VA, Tol()));

            // ---- low rank ----
            var Ak1 = arena.doubleMat(m, n);
            SVD.lowRankApprox(in A1, ref Ak1, k, ref ws, out bool _);
            var AkW = arena.doubleMat(m, n);
            SVD.lowRankApprox(in A2, ref AkW, k, ref ws, out bool _);
            var AkA = arena.doubleMat(m, n);
            SVD.lowRankApprox(in A2, ref AkA, k, out bool _);
            Assert.IsTrue(Analysis.IsZero(AkW - AkA, Tol()));

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
            var A = arena.doubleMat(6, 4);
            var basis = arena.doubleMat(4, 4);
            var ws = arena.doubleSvdFullWorkspace(5, 4);   // wrong m
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
            var A = arena.doubleMat(6, 4);
            var basis = arena.doubleMat(6, 4);
            var ws = arena.doubleSvdFullWorkspace(6, 3);   // wrong n
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
            var A = arena.doubleMat(6, 4);
            var Uk = arena.doubleMat(6, 2); var Sk = arena.doubleVec(2); var Vk = arena.doubleMat(4, 2);
            var ws = arena.doubleSvdFullWorkspace(7, 4);   // wrong m
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
            var A = arena.doubleMat(6, 4);
            var Ak = arena.doubleMat(6, 4);
            var ws = arena.doubleSvdFullWorkspace(6, 5);   // wrong n
            Assert.Throws<ArgumentException>(
                () => SVD.lowRankApprox(in A, ref Ak, 2, ref ws, out bool _));
        }
        finally { arena.Dispose(); }
    }

    // Arena.doubleSvdFullWorkspace(m, n) must size U (m x n), S (n), V (n x n).
    [Test]
    public void SvdFullWorkspace_Factory_SizesCorrectly()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var ws = arena.doubleSvdFullWorkspace(7, 4);
            Assert.AreEqual(7, ws.U.M_Rows);
            Assert.AreEqual(4, ws.U.N_Cols);
            Assert.AreEqual(4, ws.S.N);
            Assert.AreEqual(4, ws.V.M_Rows);
            Assert.AreEqual(4, ws.V.N_Cols);
        }
        finally { arena.Dispose(); }
    }
}
