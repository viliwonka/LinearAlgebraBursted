using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Workspace-overload tests for pivoted Cholesky: Cholesky.choleskyDecompositionPivot /
// choleskyPivotSolve and their shared workspace floatCholeskyPivot_WS
// (Arena.floatCholeskyPivot_WS(n)). W (n x n) is the destroyable symmetric working copy the
// decomposition pivots on; bt (n) is the permuted RHS the solve gathers into.
//
// The ws overloads are the real bodies; the allocating overloads delegate with Temp scratch, so for
// the same input the factor / solution are bit-identical. Tests:
//   (a) EQUIVALENCE — allocating vs ws (full-rank SPD AND rank-deficient PSD).
//   (b) REUSE       — ONE workspace reused across two different (same-size) inputs.
//   (c) MIS-SIZED   — wrong-dimension workspace throws; plus the needW/needBt subtlety: a decomp with
//                     a bt-less workspace must NOT throw, and a solve with a W-less workspace must NOT.
public class floatCholeskyPivotWorkspaceTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
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

        static float Tol() => 256 * Consts.floatSqrtEps;

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
        static floatMxN Gram(ref Arena arena, int n, int r, uint seed)
        {
            var B = arena.floatRandomMatrix(n, r, (float)(-1f), (float)1f, seed);
            var A = arena.floatMat(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    float s = 0;
                    for (int k = 0; k < r; k++) s += B[i, k] * B[j, k];
                    A[i, j] = s;
                }
            return A;
        }

        // well-conditioned full-rank SPD: Gram + diagonal boost.
        static floatMxN SPD(ref Arena arena, int n, uint seed)
        {
            var A = Gram(ref arena, n, n, seed);
            for (int d = 0; d < n; d++) A[d, d] += (float)n;
            return A;
        }

        void DecompEquivSPD()      => DecompEquiv(6, 6, 1001);
        void DecompEquivRankDef()  => DecompEquiv(7, 4, 2002);

        void DecompEquiv(int n, int r, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = (r >= n) ? SPD(ref arena, n, seed) : Gram(ref arena, n, r, seed);

            var La = arena.floatMat(n, n);
            var Pa = new Pivot(n, Allocator.Persistent);
            bool okA = Cholesky.choleskyDecompositionPivot(in A, ref La, ref Pa, out int rankA);

            var ws = arena.floatCholeskyPivot_WS(n);
            var Lw = arena.floatMat(n, n);
            var Pw = new Pivot(n, Allocator.Persistent);
            bool okW = Cholesky.choleskyDecompositionPivot(in A, ref Lw, ref Pw, out int rankW, ref ws);

            Assert.IsTrue(okA == okW);
            Assert.IsTrue(rankA == rankW);
            for (int i = 0; i < n; i++) Assert.IsTrue(Pa[i] == Pw[i]);
            Assert.IsTrue(Analysis_OP.IsZero(La - Lw, Tol()));

            Pw.Dispose();
            Pa.Dispose();
            arena.Dispose();
        }

        // Compare the two choleskyPivotSolve(ref L, in P, rank, ...) overloads on a common factor.
        void SolveEquivFullRank()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 6;
            var A = SPD(ref arena, n, 3003);

            // shared factor (allocating decomposition; L/P/rank fed to both solve forms).
            var L = arena.floatMat(n, n);
            var P = new Pivot(n, Allocator.Persistent);
            Cholesky.choleskyDecompositionPivot(in A, ref L, ref P, out int rank);

            var b = arena.floatRandomVector(n, (float)(-3f), (float)3f, 4004);

            var ba = b.Copy();
            Cholesky.choleskyPivotSolve(ref L, in P, rank, ref ba);

            var ws = arena.floatCholeskyPivot_WS(n);
            var bw = b.Copy();
            Cholesky.choleskyPivotSolve(ref L, in P, rank, ref bw, ref ws);

            Assert.IsTrue(Analysis_OP.IsZero(ba - bw, Tol()));

            P.Dispose();
            arena.Dispose();
        }

        void FactorSolveEquivSPD()     => FactorSolveEquiv(6, 6, 5005);
        void FactorSolveEquivRankDef() => FactorSolveEquiv(7, 4, 6006);

        void FactorSolveEquiv(int n, int r, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);
            var A = (r >= n) ? SPD(ref arena, n, seed) : Gram(ref arena, n, r, seed);
            var b = arena.floatRandomVector(n, (float)(-2f), (float)2f, seed + 100u);

            var La = arena.floatMat(n, n);
            var Pa = new Pivot(n, Allocator.Persistent);
            var ba = b.Copy();
            bool okA = Cholesky.choleskyPivotSolve(in A, ref La, ref Pa, ref ba);

            var ws = arena.floatCholeskyPivot_WS(n);
            var Lw = arena.floatMat(n, n);
            var Pw = new Pivot(n, Allocator.Persistent);
            var bw = b.Copy();
            bool okW = Cholesky.choleskyPivotSolve(in A, ref Lw, ref Pw, ref bw, ref ws);

            Assert.IsTrue(okA == okW);
            Assert.IsTrue(Analysis_OP.IsZero(ba - bw, Tol()));

            Pw.Dispose();
            Pa.Dispose();
            arena.Dispose();
        }

        // Reuse ONE workspace across two different SPD inputs; the 2nd factor-and-solve matches a
        // fresh allocating call (no stale W/bt carries over).
        void ReuseFactorSolve()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 6;

            var A1 = SPD(ref arena, n, 7007);
            var A2 = SPD(ref arena, n, 8008);
            var b1 = arena.floatRandomVector(n, (float)(-2f), (float)2f, 1111);
            var b2 = arena.floatRandomVector(n, (float)(-2f), (float)2f, 2222);

            var ws = arena.floatCholeskyPivot_WS(n);   // allocated ONCE

            // warm on (A1, b1)
            var L1 = arena.floatMat(n, n);
            var P1 = new Pivot(n, Allocator.Persistent);
            var b1c = b1.Copy();
            Cholesky.choleskyPivotSolve(in A1, ref L1, ref P1, ref b1c, ref ws);

            // reuse on (A2, b2)
            var Lw = arena.floatMat(n, n);
            var Pw = new Pivot(n, Allocator.Persistent);
            var b2w = b2.Copy();
            bool okW = Cholesky.choleskyPivotSolve(in A2, ref Lw, ref Pw, ref b2w, ref ws);

            // fresh allocating reference on (A2, b2)
            var La = arena.floatMat(n, n);
            var Pa = new Pivot(n, Allocator.Persistent);
            var b2a = b2.Copy();
            bool okA = Cholesky.choleskyPivotSolve(in A2, ref La, ref Pa, ref b2a);

            Assert.IsTrue(okW == okA);
            Assert.IsTrue(Analysis_OP.IsZero(b2w - b2a, Tol()));

            Pa.Dispose();
            Pw.Dispose();
            P1.Dispose();
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

    static floatMxN ManagedSPD(ref Arena arena, int n)
    {
        var A = arena.floatMat(n, n);
        for (int i = 0; i < n; i++) A[i, i] = (float)(n + 1);   // diagonal SPD, full rank
        return A;
    }

    [Test]
    public void Decomp_BadWorkspaceW_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        var P = new Pivot(6, Allocator.Persistent);
        try
        {
            int n = 6;
            var A = ManagedSPD(ref arena, n);
            var L = arena.floatMat(n, n);
            var ws = arena.floatCholeskyPivot_WS(n + 1);   // W wrong (needW)
            Assert.Throws<ArgumentException>(
                () => Cholesky.choleskyDecompositionPivot(in A, ref L, ref P, out int _, ref ws));
        }
        finally { P.Dispose(); arena.Dispose(); }
    }

    [Test]
    public void Solve_BadWorkspaceBt_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        var P = new Pivot(6, Allocator.Persistent);
        try
        {
            int n = 6;
            var A = ManagedSPD(ref arena, n);
            var L = arena.floatMat(n, n);
            Cholesky.choleskyDecompositionPivot(in A, ref L, ref P, out int rank);

            var b = arena.floatVec(n);
            // bt wrong length (needBt) while W is fine.
            var badWs = new floatCholeskyPivot_WS { W = arena.floatMat(n, n), bt = arena.floatVec(n + 1) };
            Assert.Throws<ArgumentException>(
                () => Cholesky.choleskyPivotSolve(ref L, in P, rank, ref b, ref badWs));
        }
        finally { P.Dispose(); arena.Dispose(); }
    }

    // needW/needBt subtlety: the decomposition never reads bt, so a bt-less workspace must NOT throw;
    // the solve never reads W, so a W-less workspace must NOT throw.
    [Test]
    public void Decomp_BtLessWorkspace_DoesNotThrow_AndSolve_WLessWorkspace_DoesNotThrow()
    {
        var arena = new Arena(Allocator.Persistent);
        var P = new Pivot(6, Allocator.Persistent);
        try
        {
            int n = 6;
            var A = ManagedSPD(ref arena, n);
            var L = arena.floatMat(n, n);

            // decomposition with W only (bt = default) must succeed.
            var wsNoBt = new floatCholeskyPivot_WS { W = arena.floatMat(n, n), bt = default };
            Assert.DoesNotThrow(
                () => Cholesky.choleskyDecompositionPivot(in A, ref L, ref P, out int _, ref wsNoBt));

            Cholesky.choleskyDecompositionPivot(in A, ref L, ref P, out int rank);

            // solve with bt only (W = default) must succeed.
            var b = arena.floatVec(n);
            for (int i = 0; i < n; i++) b[i] = (float)(i + 1);
            var wsNoW = new floatCholeskyPivot_WS { W = default, bt = arena.floatVec(n) };
            Assert.DoesNotThrow(
                () => Cholesky.choleskyPivotSolve(ref L, in P, rank, ref b, ref wsNoW));
        }
        finally { P.Dispose(); arena.Dispose(); }
    }

    // Arena.floatCholeskyPivot_WS(n): W (n x n), bt (n).
    [Test]
    public void CholeskyPivotWorkspace_Factory_SizesCorrectly()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var ws = arena.floatCholeskyPivot_WS(7);
            Assert.AreEqual(7, ws.W.M_Rows);
            Assert.AreEqual(7, ws.W.N_Cols);
            Assert.AreEqual(7, ws.bt.N);
        }
        finally { arena.Dispose(); }
    }
}
