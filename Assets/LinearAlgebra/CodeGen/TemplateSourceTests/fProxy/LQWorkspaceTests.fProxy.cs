using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Phase-2 solver-workspace tests for LQ: the caller-provided-scratch overloads
// (lqDecomposition(...,ref fProxyLQ_WS) / lqMinNormSolve(...,ref fProxyLQMinNormSolve_WS)) must
// produce results identical to the allocating wrappers (they run the SAME kernel), and a mis-sized/
// reused workspace must behave correctly (throw on bad size, produce identical results across reuse).
public class fProxyLQWorkspaceTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WorkspaceEquivJob : IJob
    {
        public enum TestType
        {
            LqDecompEquivWide,
            LqDecompEquivSquare,
            LqMinNormSolveEquiv,
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
                case TestType.LqDecompEquivWide:      LqDecompEquiv(4, 10); break;
                case TestType.LqDecompEquivSquare:    LqDecompEquiv(6, 6); break;
                case TestType.LqMinNormSolveEquiv:    LqMinNormSolveEquiv(4, 10); break;
                case TestType.WorkspaceReuse:          WorkspaceReuse(); break;
            }
        }

        // lqDecomposition scratch overload must match the allocating wrapper.
        void LqDecompEquiv(int m, int n)
        {
            var arena = new Arena(Allocator.Persistent);

            var A0 = arena.fProxyRandomMat(m, n, -5f, 5f, 41221);
            for (int d = 0; d < m; d++)   // boost leading diagonal block for conditioning
                A0[d, d] += (fProxy)10f;

            // allocating reference
            var Aa = A0.Copy();
            var La = arena.fProxyMat(m, m);
            var Qa = arena.fProxyMat(m, n);
            LQ.lqDecomposition(ref Aa, ref La, ref Qa);

            // workspace-struct form must match the allocating form
            var Ab = A0.Copy();
            var Lb = arena.fProxyMat(m, m);
            var Qb = arena.fProxyMat(m, n);
            var ws = arena.fProxyLQ_WS(m, n);
            LQ.lqDecomposition(ref Ab, ref Lb, ref Qb, ref ws);

            Assert.IsTrue(Analysis_OP.isZero(La - Lb, Tol()));
            Assert.IsTrue(Analysis_OP.isZero(Qa - Qb, Tol()));

            arena.Dispose();
        }

        // lqMinNormSolve scratch overload must match the allocating wrapper.
        void LqMinNormSolveEquiv(int m, int n)
        {
            var arena = new Arena(Allocator.Persistent);

            var A0 = arena.fProxyRandomMat(m, n, -5f, 5f, 61729);
            for (int d = 0; d < m; d++)
                A0[d, d] += (fProxy)10f;

            var b = arena.fProxyRandomVec(m, -5f, 5f, 8080);

            var Aa = A0.Copy();
            var xa = arena.fProxyVec(n);
            LQ.lqMinNormSolve(ref Aa, ref b, ref xa);

            var Ab = A0.Copy();
            var xb = arena.fProxyVec(n);
            var ws = arena.fProxyLQMinNormSolve_WS(m, n);
            LQ.lqMinNormSolve(ref Ab, ref b, ref xb, ref ws);

            Assert.IsTrue(Analysis_OP.isZero(xa - xb, Tol()));

            arena.Dispose();
        }

        // Reuse ONE workspace (of each kind) across several consecutive solves: each solve must
        // match a fresh allocating solve, proving no stale state survives reuse.
        void WorkspaceReuse()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 4, n = 8;

            var lqWs = arena.fProxyLQ_WS(m, n);                     // allocated ONCE, reused below
            var solveWs = arena.fProxyLQMinNormSolve_WS(m, n);      // allocated ONCE, reused below

            for (int t = 0; t < 3; t++)
            {
                var A0 = arena.fProxyRandomMat(m, n, -5f, 5f, (uint)(3000 + t * 13));
                for (int d = 0; d < m; d++)
                    A0[d, d] += (fProxy)10f;

                // lqDecomposition: allocating reference vs reused workspace
                var Aa = A0.Copy();
                var La = arena.fProxyMat(m, m);
                var Qa = arena.fProxyMat(m, n);
                LQ.lqDecomposition(ref Aa, ref La, ref Qa);

                var Aw = A0.Copy();
                var Lw = arena.fProxyMat(m, m);
                var Qw = arena.fProxyMat(m, n);
                LQ.lqDecomposition(ref Aw, ref Lw, ref Qw, ref lqWs);

                Assert.IsTrue(Analysis_OP.isZero(La - Lw, Tol()));
                Assert.IsTrue(Analysis_OP.isZero(Qa - Qw, Tol()));

                // lqMinNormSolve: allocating reference vs reused workspace
                var b = arena.fProxyRandomVec(m, -5f, 5f, (uint)(4000 + t * 17));

                var Asa = A0.Copy();
                var xa = arena.fProxyVec(n);
                LQ.lqMinNormSolve(ref Asa, ref b, ref xa);

                var Asw = A0.Copy();
                var xw = arena.fProxyVec(n);
                LQ.lqMinNormSolve(ref Asw, ref b, ref xw, ref solveWs);

                Assert.IsTrue(Analysis_OP.isZero(xa - xw, Tol()));
            }

            arena.Dispose();
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
    public void LqDecomp_BadScratchT_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(4, 8);
            var L = arena.fProxyMat(4, 4);
            var Q = arena.fProxyMat(4, 8);
            var ws = arena.fProxyLQ_WS(4, 8);
            ws.T = arena.fProxyMat(8, 3);   // wrong: must be n x m = 8 x 4
            Assert.Throws<ArgumentException>(() => LQ.lqDecomposition(ref A, ref L, ref Q, ref ws));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void LqDecomp_BadScratchQrU_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(4, 8);
            var L = arena.fProxyMat(4, 4);
            var Q = arena.fProxyMat(4, 8);
            var ws = arena.fProxyLQ_WS(4, 8);
            ws.qrU = arena.fProxyVec(5);    // wrong: must be length n = 8
            Assert.Throws<ArgumentException>(() => LQ.lqDecomposition(ref A, ref L, ref Q, ref ws));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void LqMinNormSolve_BadScratchL_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(4, 8);
            var b = arena.fProxyVec(4);
            var x = arena.fProxyVec(8);
            var ws = arena.fProxyLQMinNormSolve_WS(4, 8);
            ws.L = arena.fProxyMat(3, 3);   // wrong: must be m x m = 4 x 4
            Assert.Throws<ArgumentException>(() => LQ.lqMinNormSolve(ref A, ref b, ref x, ref ws));
        }
        finally { arena.Dispose(); }
    }

    // Arena.fProxyLQ_WS(m, n) / fProxyLQMinNormSolve_WS(m, n) must size every field for m x n.
    [Test]
    public void LQWorkspace_Factory_SizesCorrectly()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var lqWs = arena.fProxyLQ_WS(4, 8);
            Assert.AreEqual(8, lqWs.T.M_Rows);
            Assert.AreEqual(4, lqWs.T.N_Cols);
            Assert.AreEqual(4, lqWs.Rqr.M_Rows);
            Assert.AreEqual(4, lqWs.Rqr.N_Cols);
            Assert.AreEqual(8, lqWs.qrU.N);
            Assert.AreEqual(4, lqWs.qrW.N);

            var solveWs = arena.fProxyLQMinNormSolve_WS(4, 8);
            Assert.AreEqual(4, solveWs.L.M_Rows);
            Assert.AreEqual(4, solveWs.L.N_Cols);
            Assert.AreEqual(4, solveWs.Q.M_Rows);
            Assert.AreEqual(8, solveWs.Q.N_Cols);
            Assert.AreEqual(4, solveWs.y.N);
            Assert.AreEqual(8, solveWs.LQWs.T.M_Rows);
            Assert.AreEqual(4, solveWs.LQWs.T.N_Cols);
        }
        finally { arena.Dispose(); }
    }
}
