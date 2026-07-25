using System;

using BULA;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Workspace-overload tests for pivoted Cholesky: CHOP.decomp /
// CHOP.decompSolve / CHOP.solveInPlace and their shared workspace fProxyCHOPCache
// (new fProxyCHOPCache(n, allocator)). W (n x n) is the destroyable symmetric working copy the
// decomposition pivots on; bt (n) is the permuted RHS the solve gathers into.
//
// The ws overloads are the real bodies; the allocating overloads delegate with Temp scratch, so for
// the same input the factor / solution are bit-identical. Tests:
//   (a) EQUIVALENCE — allocating vs ws (full-rank SPD AND rank-deficient PSD).
//   (b) REUSE       — ONE workspace reused across two different (same-size) inputs.
//   (c) MIS-SIZED   — wrong-dimension workspace throws; plus the needW/needBt subtlety: a decomp with
//                     a bt-less workspace must NOT throw, and a solve with a W-less workspace must NOT.
public class fProxyCHOPWorkspaceTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WorkspaceJob : IJob
    {
        public enum TestType
        {
            DecompEquivSPD,
            DecompEquivRankDef,
            SolveEquivFullRank,
            FactorSolveEquivSPD,
            FactorSolveEquivRankDef,
            ReuseFactorSolve,
        }

        public TestType Type;

        static fProxy Tol() => 256 * Consts.fProxySqrtEps;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.DecompEquivSPD:          DecompEquivSPD();          break;
                case TestType.DecompEquivRankDef:      DecompEquivRankDef();      break;
                case TestType.SolveEquivFullRank:      SolveEquivFullRank();      break;
                case TestType.FactorSolveEquivSPD:     FactorSolveEquivSPD();     break;
                case TestType.FactorSolveEquivRankDef: FactorSolveEquivRankDef(); break;
                case TestType.ReuseFactorSolve:        ReuseFactorSolve();        break;
            }
        }

        // A = B Bᵀ (B is n x r): symmetric PSD of generic rank min(r, n).
        static fProxyMxN Gram(int n, int r, uint seed)
        {
            var B = GenerateOP.fProxyRandomMat(n, r, (fProxy)(-1f), (fProxy)1f, seed, Allocator.Temp);
            var A = new fProxyMxN(n, n, Allocator.Temp);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    fProxy s = 0;
                    for (int k = 0; k < r; k++) s += B[i, k] * B[j, k];
                    A[i, j] = s;
                }
            return A;
        }

        // well-conditioned full-rank SPD: Gram + diagonal boost.
        static fProxyMxN SPD(int n, uint seed)
        {
            var A = Gram(n, n, seed);
            for (int d = 0; d < n; d++) A[d, d] += (fProxy)n;
            return A;
        }

        void DecompEquivSPD()      => DecompEquiv(6, 6, 1001);
        void DecompEquivRankDef()  => DecompEquiv(7, 4, 2002);

        void DecompEquiv(int n, int r, uint seed)
        {
            var A = (r >= n) ? SPD(n, seed) : Gram(n, r, seed);

            var La = new fProxyMxN(n, n, Allocator.Temp);
            var Pa = new Pivot(n, Allocator.Persistent);
            var infoA = CHOP.decomp(in A, ref La, ref Pa);
            bool okA = infoA.Solved;
            int rankA = infoA.rank;

            var ws = new fProxyCHOPCache(n, Allocator.Temp);
            var Lw = new fProxyMxN(n, n, Allocator.Temp);
            var Pw = new Pivot(n, Allocator.Persistent);
            var infoW = CHOP.decomp(in A, ref Lw, ref Pw, ref ws);
            bool okW = infoW.Solved;
            int rankW = infoW.rank;

            Assert.IsTrue(okA == okW);
            Assert.IsTrue(rankA == rankW);
            for (int i = 0; i < n; i++) Assert.IsTrue(Pa[i] == Pw[i]);
            var LaMinusLw = new fProxyMxN(in La, Allocator.Temp);
            fProxyComp.subInPlace(LaMinusLw, Lw);
            Assert.IsTrue(Analysis.isZero(LaMinusLw, Tol()));

            Pw.Dispose();
            Pa.Dispose();
        }

        // Compare the two CHOP.decompSolve(ref L, in P, rank, ...) overloads on a common factor.
        void SolveEquivFullRank()
        {
            int n = 6;
            var A = SPD(n, 3003);

            // shared factor (allocating decomposition; L/P/rank fed to both solve forms).
            var L = new fProxyMxN(n, n, Allocator.Temp);
            var P = new Pivot(n, Allocator.Persistent);
            int rank = CHOP.decomp(in A, ref L, ref P).rank;

            var b = GenerateOP.fProxyRandomVec(n, (fProxy)(-3f), (fProxy)3f, 4004, Allocator.Temp);

            var ba = new fProxyN(in b, Allocator.Temp);
            CHOP.decompSolve(ref L, in P, rank, ref ba);

            var ws = new fProxyCHOPCache(n, Allocator.Temp);
            var bw = new fProxyN(in b, Allocator.Temp);
            CHOP.decompSolve(ref L, in P, rank, ref bw, ref ws);

            var baMinusBw = new fProxyN(in ba, Allocator.Temp);
            fProxyComp.subInPlace(baMinusBw, bw);
            Assert.IsTrue(Analysis.isZero(baMinusBw, Tol()));

            P.Dispose();
        }

        void FactorSolveEquivSPD()     => FactorSolveEquiv(6, 6, 5005);
        void FactorSolveEquivRankDef() => FactorSolveEquiv(7, 4, 6006);

        void FactorSolveEquiv(int n, int r, uint seed)
        {
            var A = (r >= n) ? SPD(n, seed) : Gram(n, r, seed);
            var b = GenerateOP.fProxyRandomVec(n, (fProxy)(-2f), (fProxy)2f, seed + 100u, Allocator.Temp);

            var Pa = new Pivot(n, Allocator.Persistent);
            var ba = new fProxyN(in b, Allocator.Temp);
            var Aa = new fProxyMxN(in A, Allocator.Temp); // solveInPlace is destructive; each call needs its own copy of A
            bool okA = CHOP.solveInPlace(ref Aa, ref Pa, ref ba);

            var ws = new fProxyCHOPCache(n, Allocator.Temp);
            var Pw = new Pivot(n, Allocator.Persistent);
            var bw = new fProxyN(in b, Allocator.Temp);
            var Aw = new fProxyMxN(in A, Allocator.Temp);
            bool okW = CHOP.solveInPlace(ref Aw, ref Pw, ref bw, ref ws);

            Assert.IsTrue(okA == okW);
            var baMinusBw = new fProxyN(in ba, Allocator.Temp);
            fProxyComp.subInPlace(baMinusBw, bw);
            Assert.IsTrue(Analysis.isZero(baMinusBw, Tol()));

            Pw.Dispose();
            Pa.Dispose();
        }

        // Reuse ONE workspace across two different SPD inputs; the 2nd factor-and-solve matches a
        // fresh allocating call (no stale W/bt carries over).
        void ReuseFactorSolve()
        {
            int n = 6;

            var A1 = SPD(n, 7007);
            var A2 = SPD(n, 8008);
            var b1 = GenerateOP.fProxyRandomVec(n, (fProxy)(-2f), (fProxy)2f, 1111, Allocator.Temp);
            var b2 = GenerateOP.fProxyRandomVec(n, (fProxy)(-2f), (fProxy)2f, 2222, Allocator.Temp);

            var ws = new fProxyCHOPCache(n, Allocator.Temp);   // allocated ONCE

            // warm on (A1, b1)
            var P1 = new Pivot(n, Allocator.Persistent);
            var b1c = new fProxyN(in b1, Allocator.Temp);
            var A1c = new fProxyMxN(in A1, Allocator.Temp); // solveInPlace is destructive; each call needs its own copy of A
            CHOP.solveInPlace(ref A1c, ref P1, ref b1c, ref ws);

            // reuse on (A2, b2)
            var Pw = new Pivot(n, Allocator.Persistent);
            var b2w = new fProxyN(in b2, Allocator.Temp);
            var A2w = new fProxyMxN(in A2, Allocator.Temp);
            bool okW = CHOP.solveInPlace(ref A2w, ref Pw, ref b2w, ref ws);

            // fresh allocating reference on (A2, b2)
            var Pa = new Pivot(n, Allocator.Persistent);
            var b2a = new fProxyN(in b2, Allocator.Temp);
            var A2a = new fProxyMxN(in A2, Allocator.Temp);
            bool okA = CHOP.solveInPlace(ref A2a, ref Pa, ref b2a);

            Assert.IsTrue(okW == okA);
            var b2wMinusB2a = new fProxyN(in b2w, Allocator.Temp);
            fProxyComp.subInPlace(b2wMinusB2a, b2a);
            Assert.IsTrue(Analysis.isZero(b2wMinusB2a, Tol()));

            Pa.Dispose();
            Pw.Dispose();
            P1.Dispose();
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(WorkspaceJob.TestType));

    [TestCaseSource("GetEnums")]
    public void WorkspaceTests(WorkspaceJob.TestType type)
    {
        new WorkspaceJob() { Type = type }.Run();
    }

    // ---- mis-sized workspace guards (managed [Test]) ----

    static fProxyMxN ManagedSPD(int n)
    {
        var A = new fProxyMxN(n, n, Allocator.Temp);
        for (int i = 0; i < n; i++) A[i, i] = (fProxy)(n + 1);   // diagonal SPD, full rank
        return A;
    }

    [Test]
    public void Decomp_BadWorkspaceW_Throws()
    {
        var P = new Pivot(6, Allocator.Persistent);
        try
        {
            int n = 6;
            var A = ManagedSPD(n);
            var L = new fProxyMxN(n, n, Allocator.Temp);
            var ws = new fProxyCHOPCache(n + 1, Allocator.Temp);   // W wrong (needW)
            Assert.Throws<ArgumentException>(
                () => CHOP.decomp(in A, ref L, ref P, ref ws));
        }
        finally { P.Dispose(); }
    }

    [Test]
    public void Solve_BadWorkspaceBt_Throws()
    {
        var P = new Pivot(6, Allocator.Persistent);
        try
        {
            int n = 6;
            var A = ManagedSPD(n);
            var L = new fProxyMxN(n, n, Allocator.Temp);
            int rank = CHOP.decomp(in A, ref L, ref P).rank;

            var b = new fProxyN(n, Allocator.Temp);
            // bt wrong length (needBt) while W is fine.
            var badWs = new fProxyCHOPCache { W = new fProxyMxN(n, n, Allocator.Temp), bt = new fProxyN(n + 1, Allocator.Temp) };
            Assert.Throws<ArgumentException>(
                () => CHOP.decompSolve(ref L, in P, rank, ref b, ref badWs));
        }
        finally { P.Dispose(); }
    }

    // needW/needBt subtlety: the decomposition never reads bt, so a bt-less workspace must NOT throw;
    // the solve never reads W, so a W-less workspace must NOT throw.
    [Test]
    public void Decomp_BtLessWorkspace_DoesNotThrow_AndSolve_WLessWorkspace_DoesNotThrow()
    {
        var P = new Pivot(6, Allocator.Persistent);
        try
        {
            int n = 6;
            var A = ManagedSPD(n);
            var L = new fProxyMxN(n, n, Allocator.Temp);

            // decomposition with W only (bt = default) must succeed.
            var wsNoBt = new fProxyCHOPCache { W = new fProxyMxN(n, n, Allocator.Temp), bt = default };
            Assert.DoesNotThrow(
                () => CHOP.decomp(in A, ref L, ref P, ref wsNoBt));

            int rank = CHOP.decomp(in A, ref L, ref P).rank;

            // solve with bt only (W = default) must succeed.
            var b = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) b[i] = (fProxy)(i + 1);
            var wsNoW = new fProxyCHOPCache { W = default, bt = new fProxyN(n, Allocator.Temp) };
            Assert.DoesNotThrow(
                () => CHOP.decompSolve(ref L, in P, rank, ref b, ref wsNoW));
        }
        finally { P.Dispose(); }
    }

    // fProxyCHOPCache(n, allocator): W (n x n), bt (n).
    [Test]
    public void CHOPWorkspace_Factory_SizesCorrectly()
    {
        var ws = new fProxyCHOPCache(7, Allocator.Temp);
        Assert.AreEqual(7, ws.W.M_Rows);
        Assert.AreEqual(7, ws.W.N_Cols);
        Assert.AreEqual(7, ws.bt.N);
    }
}
