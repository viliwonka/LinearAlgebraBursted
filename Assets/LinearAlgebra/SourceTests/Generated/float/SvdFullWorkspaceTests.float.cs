using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Workspace-overload tests for SVD family ops with reusable workspaces.
//
// nullspaceBasis, rangeBasis, and lowRankApprox share floatSvdFull_WS.
// svdTruncated (GKL bidiagonalization) uses its own floatSvdTruncated_WS.
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
public class floatSvdFullWorkspaceTests
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
        static float Tol() => 256 * Consts.floatSqrtEps;

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
        static floatMxN RankDeficient(ref Arena arena, int m, int n, int r, uint seed)
        {
            var B = arena.floatRandomMatrix(m, r, (float)(-2f), (float)2f, seed);
            var C = arena.floatRandomMatrix(r, n, (float)(-2f), (float)2f, seed + 7u);
            return float_OP.dot(B, C);
        }

        void NullspaceEquiv()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 6, n = 4;
            var A = RankDeficient(ref arena, m, n, 2, 1001);   // rank 2 -> nullspace dim 2

            var basisA = arena.floatMat(n, n);
            int dimA = SVD.nullspaceBasis(in A, ref basisA, out bool cA);

            var ws = arena.floatSvdFull_WS(m, n);
            var basisW = arena.floatMat(n, n);
            int dimW = SVD.nullspaceBasis(in A, ref basisW, ref ws, out bool cW);

            Assert.IsTrue(dimA == dimW);
            Assert.IsTrue(cA == cW);
            Assert.IsTrue(Analysis_OP.IsZero(basisA - basisW, Tol()));

            arena.Dispose();
        }

        void RangeEquiv()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 6, n = 4;
            var A = RankDeficient(ref arena, m, n, 3, 2002);   // rank 3 -> range rank 3

            var basisA = arena.floatMat(m, n);
            int rankA = SVD.rangeBasis(in A, ref basisA, out bool cA);

            var ws = arena.floatSvdFull_WS(m, n);
            var basisW = arena.floatMat(m, n);
            int rankW = SVD.rangeBasis(in A, ref basisW, ref ws, out bool cW);

            Assert.IsTrue(rankA == rankW);
            Assert.IsTrue(cA == cW);
            Assert.IsTrue(Analysis_OP.IsZero(basisA - basisW, Tol()));

            arena.Dispose();
        }

        void TruncatedEquiv()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 6, n = 4, k = 2;
            var A = arena.floatRandomMatrix(m, n, (float)(-3f), (float)3f, 3003);

            var UkA = arena.floatMat(m, k); var SkA = arena.floatVec(k); var VkA = arena.floatMat(n, k);
            SVD.svdTruncated(in A, ref UkA, ref SkA, ref VkA, k, out bool cA);

            var ws = arena.floatSvdTruncated_WS(m, n, k);
            var UkW = arena.floatMat(m, k); var SkW = arena.floatVec(k); var VkW = arena.floatMat(n, k);
            SVD.svdTruncated(in A, ref UkW, ref SkW, ref VkW, k, ref ws, out bool cW);

            Assert.IsTrue(cA == cW);
            Assert.IsTrue(Analysis_OP.IsZero(SkA - SkW, Tol()));
            Assert.IsTrue(Analysis_OP.IsZero(UkA - UkW, Tol()));
            Assert.IsTrue(Analysis_OP.IsZero(VkA - VkW, Tol()));

            arena.Dispose();
        }

        void LowRankEquiv()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 6, n = 4, k = 2;
            var A = arena.floatRandomMatrix(m, n, (float)(-3f), (float)3f, 4004);

            var AkA = arena.floatMat(m, n);
            SVD.lowRankApprox(in A, ref AkA, k, out bool cA);

            var ws = arena.floatSvdFull_WS(m, n);
            var AkW = arena.floatMat(m, n);
            SVD.lowRankApprox(in A, ref AkW, k, ref ws, out bool cW);

            Assert.IsTrue(cA == cW);
            Assert.IsTrue(Analysis_OP.IsZero(AkA - AkW, Tol()));

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

            var ws = arena.floatSvdFull_WS(m, n);           // for nullspace / range / lowRank
            var wsTrunc = arena.floatSvdTruncated_WS(m, n, k);  // for svdTruncated (GKL)

            // ---- nullspace ----
            var nb1 = arena.floatMat(n, n);
            SVD.nullspaceBasis(in A1, ref nb1, ref ws, out bool _);          // warm the workspace on A1
            var nbW = arena.floatMat(n, n);
            int dimW = SVD.nullspaceBasis(in A2, ref nbW, ref ws, out bool _);
            var nbA = arena.floatMat(n, n);
            int dimA = SVD.nullspaceBasis(in A2, ref nbA, out bool _);
            Assert.IsTrue(dimW == dimA);
            Assert.IsTrue(Analysis_OP.IsZero(nbW - nbA, Tol()));

            // ---- range ----
            var rb1 = arena.floatMat(m, n);
            SVD.rangeBasis(in A1, ref rb1, ref ws, out bool _);
            var rbW = arena.floatMat(m, n);
            int rkW = SVD.rangeBasis(in A2, ref rbW, ref ws, out bool _);
            var rbA = arena.floatMat(m, n);
            int rkA = SVD.rangeBasis(in A2, ref rbA, out bool _);
            Assert.IsTrue(rkW == rkA);
            Assert.IsTrue(Analysis_OP.IsZero(rbW - rbA, Tol()));

            // ---- truncated ----
            var U1 = arena.floatMat(m, k); var S1 = arena.floatVec(k); var V1 = arena.floatMat(n, k);
            SVD.svdTruncated(in A1, ref U1, ref S1, ref V1, k, ref wsTrunc, out bool _);
            var UW = arena.floatMat(m, k); var SW = arena.floatVec(k); var VW = arena.floatMat(n, k);
            SVD.svdTruncated(in A2, ref UW, ref SW, ref VW, k, ref wsTrunc, out bool _);
            var UA = arena.floatMat(m, k); var SA = arena.floatVec(k); var VA = arena.floatMat(n, k);
            SVD.svdTruncated(in A2, ref UA, ref SA, ref VA, k, out bool _);
            Assert.IsTrue(Analysis_OP.IsZero(SW - SA, Tol()));
            Assert.IsTrue(Analysis_OP.IsZero(UW - UA, Tol()));
            Assert.IsTrue(Analysis_OP.IsZero(VW - VA, Tol()));

            // ---- low rank ----
            var Ak1 = arena.floatMat(m, n);
            SVD.lowRankApprox(in A1, ref Ak1, k, ref ws, out bool _);
            var AkW = arena.floatMat(m, n);
            SVD.lowRankApprox(in A2, ref AkW, k, ref ws, out bool _);
            var AkA = arena.floatMat(m, n);
            SVD.lowRankApprox(in A2, ref AkA, k, out bool _);
            Assert.IsTrue(Analysis_OP.IsZero(AkW - AkA, Tol()));

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
            var A = arena.floatMat(6, 4);
            var basis = arena.floatMat(4, 4);
            var ws = arena.floatSvdFull_WS(5, 4);   // wrong m
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
            var A = arena.floatMat(6, 4);
            var basis = arena.floatMat(6, 4);
            var ws = arena.floatSvdFull_WS(6, 3);   // wrong n
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
            var A = arena.floatMat(6, 4);
            var Uk = arena.floatMat(6, 2); var Sk = arena.floatVec(2); var Vk = arena.floatMat(4, 2);
            var ws = arena.floatSvdTruncated_WS(7, 4, 2);   // wrong m (7 vs A's 6)
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
            var A = arena.floatMat(6, 4);
            var Ak = arena.floatMat(6, 4);
            var ws = arena.floatSvdFull_WS(6, 5);   // wrong n
            Assert.Throws<ArgumentException>(
                () => SVD.lowRankApprox(in A, ref Ak, 2, ref ws, out bool _));
        }
        finally { arena.Dispose(); }
    }

    // Arena.floatSvdFull_WS(m, n) must size U (m x n), S (n), V (n x n).
    [Test]
    public void SvdFullWorkspace_Factory_SizesCorrectly()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var ws = arena.floatSvdFull_WS(7, 4);
            Assert.AreEqual(7, ws.U.M_Rows);
            Assert.AreEqual(4, ws.U.N_Cols);
            Assert.AreEqual(4, ws.S.N);
            Assert.AreEqual(4, ws.V.M_Rows);
            Assert.AreEqual(4, ws.V.N_Cols);
        }
        finally { arena.Dispose(); }
    }
}
