using System;

using BULA;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Workspace-overload tests for SVD family ops with reusable workspaces.
//
// nullspaceBasis, rangeBasis, and lowRankApprox share fProxySVDFullCache.
// truncated (GKL bidiagonalization) uses its own fProxySVDTruncatedCache.
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
public class fProxySVDFullWorkspaceTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
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
        static fProxyMxN RankDeficient(int m, int n, int r, uint seed)
        {
            var B = GenerateOP.fProxyRandomMat(m, r, (fProxy)(-2f), (fProxy)2f, seed, allocator: Allocator.Temp);
            var C = GenerateOP.fProxyRandomMat(r, n, (fProxy)(-2f), (fProxy)2f, seed + 7u, allocator: Allocator.Temp);
            return Blas.dot(B, C);
        }

        void NullspaceEquiv()
        {
            int m = 6, n = 4;
            var A = RankDeficient(m, n, 2, 1001);   // rank 2 -> nullspace dim 2

            var basisA = new fProxyMxN(n, n, Allocator.Temp);
            RankInfo infoA = SVD.nullspaceBasis(in A, ref basisA);
            bool cA = infoA;
            int dimA = n - infoA.rank;

            var ws = new fProxySVDFullCache(m, n, Allocator.Temp);
            var basisW = new fProxyMxN(n, n, Allocator.Temp);
            RankInfo infoW = SVD.nullspaceBasis(in A, ref basisW, ref ws);
            bool cW = infoW;
            int dimW = n - infoW.rank;

            Assert.IsTrue(dimA == dimW);
            Assert.IsTrue(cA == cW);
            var basisDiff = new fProxyMxN(in basisA, Allocator.Temp);
            basisDiff.subInPlace(basisW);
            Assert.IsTrue(Analysis.isZero(basisDiff, Tol()));
        }

        void RangeEquiv()
        {
            int m = 6, n = 4;
            var A = RankDeficient(m, n, 3, 2002);   // rank 3 -> range rank 3

            var basisA = new fProxyMxN(m, n, Allocator.Temp);
            RankInfo infoA = SVD.rangeBasis(in A, ref basisA);
            bool cA = infoA;
            int rankA = infoA.rank;

            var ws = new fProxySVDFullCache(m, n, Allocator.Temp);
            var basisW = new fProxyMxN(m, n, Allocator.Temp);
            RankInfo infoW = SVD.rangeBasis(in A, ref basisW, ref ws);
            bool cW = infoW;
            int rankW = infoW.rank;

            Assert.IsTrue(rankA == rankW);
            Assert.IsTrue(cA == cW);
            var basisDiff = new fProxyMxN(in basisA, Allocator.Temp);
            basisDiff.subInPlace(basisW);
            Assert.IsTrue(Analysis.isZero(basisDiff, Tol()));
        }

        void TruncatedEquiv()
        {
            int m = 6, n = 4, k = 2;
            var A = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-3f), (fProxy)3f, 3003, allocator: Allocator.Temp);

            var UkA = new fProxyMxN(m, k, Allocator.Temp); var SkA = new fProxyN(k, Allocator.Temp); var VkA = new fProxyMxN(n, k, Allocator.Temp);
            SVDInfo cA = SVD.truncated(in A, ref UkA, ref SkA, ref VkA, k);

            var ws = new fProxySVDTruncatedCache(m, n, k, Allocator.Temp);
            var UkW = new fProxyMxN(m, k, Allocator.Temp); var SkW = new fProxyN(k, Allocator.Temp); var VkW = new fProxyMxN(n, k, Allocator.Temp);
            SVDInfo cW = SVD.truncated(in A, ref UkW, ref SkW, ref VkW, k, ref ws);

            Assert.IsTrue(cA == cW);
            var SDiff = new fProxyN(in SkA, Allocator.Temp); SDiff.subInPlace(SkW);
            Assert.IsTrue(Analysis.isZero(SDiff, Tol()));
            var UDiff = new fProxyMxN(in UkA, Allocator.Temp); UDiff.subInPlace(UkW);
            Assert.IsTrue(Analysis.isZero(UDiff, Tol()));
            var VDiff = new fProxyMxN(in VkA, Allocator.Temp); VDiff.subInPlace(VkW);
            Assert.IsTrue(Analysis.isZero(VDiff, Tol()));
        }

        void LowRankEquiv()
        {
            int m = 6, n = 4, k = 2;
            var A = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-3f), (fProxy)3f, 4004, allocator: Allocator.Temp);

            var AkA = new fProxyMxN(m, n, Allocator.Temp);
            bool cA = SVD.lowRankApprox(in A, ref AkA, k);

            var ws = new fProxySVDFullCache(m, n, Allocator.Temp);
            var AkW = new fProxyMxN(m, n, Allocator.Temp);
            bool cW = SVD.lowRankApprox(in A, ref AkW, k, ref ws);

            Assert.IsTrue(cA == cW);
            var AkDiff = new fProxyMxN(in AkA, Allocator.Temp); AkDiff.subInPlace(AkW);
            Assert.IsTrue(Analysis.isZero(AkDiff, Tol()));
        }

        // Reuse workspaces across two different (same-shape) inputs and every family op; each
        // second-input result must match a fresh allocating call -> no stale data survives reuse.
        void ReuseAllOps()
        {
            int m = 6, n = 4, k = 2;

            var A1 = RankDeficient(m, n, 2, 5005);
            var A2 = RankDeficient(m, n, 3, 6006);

            var ws = new fProxySVDFullCache(m, n, Allocator.Temp);           // for nullspace / range / lowRank
            var wsTrunc = new fProxySVDTruncatedCache(m, n, k, Allocator.Temp);  // for truncated (GKL)

            // ---- nullspace ----
            var nb1 = new fProxyMxN(n, n, Allocator.Temp);
            SVD.nullspaceBasis(in A1, ref nb1, ref ws);          // warm the workspace on A1
            var nbW = new fProxyMxN(n, n, Allocator.Temp);
            RankInfo nbWInfo = SVD.nullspaceBasis(in A2, ref nbW, ref ws);
            int dimW = n - nbWInfo.rank;
            var nbA = new fProxyMxN(n, n, Allocator.Temp);
            RankInfo nbAInfo = SVD.nullspaceBasis(in A2, ref nbA);
            int dimA = n - nbAInfo.rank;
            Assert.IsTrue(dimW == dimA);
            var nbDiff = new fProxyMxN(in nbW, Allocator.Temp); nbDiff.subInPlace(nbA);
            Assert.IsTrue(Analysis.isZero(nbDiff, Tol()));

            // ---- range ----
            var rb1 = new fProxyMxN(m, n, Allocator.Temp);
            SVD.rangeBasis(in A1, ref rb1, ref ws);
            var rbW = new fProxyMxN(m, n, Allocator.Temp);
            int rkW = SVD.rangeBasis(in A2, ref rbW, ref ws).rank;
            var rbA = new fProxyMxN(m, n, Allocator.Temp);
            int rkA = SVD.rangeBasis(in A2, ref rbA).rank;
            Assert.IsTrue(rkW == rkA);
            var rbDiff = new fProxyMxN(in rbW, Allocator.Temp); rbDiff.subInPlace(rbA);
            Assert.IsTrue(Analysis.isZero(rbDiff, Tol()));

            // ---- truncated ----
            var U1 = new fProxyMxN(m, k, Allocator.Temp); var S1 = new fProxyN(k, Allocator.Temp); var V1 = new fProxyMxN(n, k, Allocator.Temp);
            SVD.truncated(in A1, ref U1, ref S1, ref V1, k, ref wsTrunc);
            var UW = new fProxyMxN(m, k, Allocator.Temp); var SW = new fProxyN(k, Allocator.Temp); var VW = new fProxyMxN(n, k, Allocator.Temp);
            SVD.truncated(in A2, ref UW, ref SW, ref VW, k, ref wsTrunc);
            var UA = new fProxyMxN(m, k, Allocator.Temp); var SA = new fProxyN(k, Allocator.Temp); var VA = new fProxyMxN(n, k, Allocator.Temp);
            SVD.truncated(in A2, ref UA, ref SA, ref VA, k);
            var SDiff2 = new fProxyN(in SW, Allocator.Temp); SDiff2.subInPlace(SA);
            Assert.IsTrue(Analysis.isZero(SDiff2, Tol()));
            var UDiff2 = new fProxyMxN(in UW, Allocator.Temp); UDiff2.subInPlace(UA);
            Assert.IsTrue(Analysis.isZero(UDiff2, Tol()));
            var VDiff2 = new fProxyMxN(in VW, Allocator.Temp); VDiff2.subInPlace(VA);
            Assert.IsTrue(Analysis.isZero(VDiff2, Tol()));

            // ---- low rank ----
            var Ak1 = new fProxyMxN(m, n, Allocator.Temp);
            SVD.lowRankApprox(in A1, ref Ak1, k, ref ws);
            var AkW = new fProxyMxN(m, n, Allocator.Temp);
            SVD.lowRankApprox(in A2, ref AkW, k, ref ws);
            var AkA = new fProxyMxN(m, n, Allocator.Temp);
            SVD.lowRankApprox(in A2, ref AkA, k);
            var AkDiff = new fProxyMxN(in AkW, Allocator.Temp); AkDiff.subInPlace(AkA);
            Assert.IsTrue(Analysis.isZero(AkDiff, Tol()));
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
        var A = new fProxyMxN(6, 4, Allocator.Temp);
        var basis = new fProxyMxN(4, 4, Allocator.Temp);
        var ws = new fProxySVDFullCache(5, 4, Allocator.Temp);   // wrong m
        Assert.Throws<ArgumentException>(
            () => SVD.nullspaceBasis(in A, ref basis, ref ws));
    }

    [Test]
    public void Range_BadWorkspace_Throws()
    {
        var A = new fProxyMxN(6, 4, Allocator.Temp);
        var basis = new fProxyMxN(6, 4, Allocator.Temp);
        var ws = new fProxySVDFullCache(6, 3, Allocator.Temp);   // wrong n
        Assert.Throws<ArgumentException>(
            () => SVD.rangeBasis(in A, ref basis, ref ws));
    }

    [Test]
    public void Truncated_BadWorkspace_Throws()
    {
        var A = new fProxyMxN(6, 4, Allocator.Temp);
        var Uk = new fProxyMxN(6, 2, Allocator.Temp); var Sk = new fProxyN(2, Allocator.Temp); var Vk = new fProxyMxN(4, 2, Allocator.Temp);
        var ws = new fProxySVDTruncatedCache(7, 4, 2, Allocator.Temp);   // wrong m (7 vs A's 6)
        Assert.Throws<ArgumentException>(
            () => SVD.truncated(in A, ref Uk, ref Sk, ref Vk, 2, ref ws));
    }

    [Test]
    public void LowRank_BadWorkspace_Throws()
    {
        var A = new fProxyMxN(6, 4, Allocator.Temp);
        var Ak = new fProxyMxN(6, 4, Allocator.Temp);
        var ws = new fProxySVDFullCache(6, 5, Allocator.Temp);   // wrong n
        Assert.Throws<ArgumentException>(
            () => SVD.lowRankApprox(in A, ref Ak, 2, ref ws));
    }

    // Standalone fProxySVDFullCache(m, n, allocator) must size U (m x n), S (n), V (n x n).
    [Test]
    public void SvdFullWorkspace_Factory_SizesCorrectly()
    {
        var ws = new fProxySVDFullCache(7, 4, Allocator.Temp);
        Assert.AreEqual(7, ws.U.M_Rows);
        Assert.AreEqual(4, ws.U.N_Cols);
        Assert.AreEqual(4, ws.S.N);
        Assert.AreEqual(4, ws.V.M_Rows);
        Assert.AreEqual(4, ws.V.N_Cols);
    }
}
