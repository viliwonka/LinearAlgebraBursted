using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Phase-2 solver-workspace tests for LQ: the caller-provided-scratch overloads
// (decomp(...,ref fProxyLQCache) / minNormSolve(...,ref fProxyLQMinNormCache)) must
// produce results identical to the allocating wrappers (they run the SAME kernel), and a mis-sized/
// reused workspace must behave correctly (throw on bad size, produce identical results across reuse).
public class fProxyLQWorkspaceTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
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
            LQ.decomp(in Aa, ref La, ref Qa);

            // workspace-struct form must match the allocating form
            var Ab = A0.Copy();
            var Lb = arena.fProxyMat(m, m);
            var Qb = arena.fProxyMat(m, n);
            var ws = arena.fProxyLQCache(m, n);
            LQ.decomp(in Ab, ref Lb, ref Qb, ref ws);

            Assert.IsTrue(Analysis.isZero(La - Lb, Tol()));
            Assert.IsTrue(Analysis.isZero(Qa - Qb, Tol()));

            arena.Dispose();
        }

        void LqMinNormSolveEquiv(int m, int n)
        {
            var arena = new Arena(Allocator.Persistent);

            var A0 = arena.fProxyRandomMat(m, n, -5f, 5f, 61729);
            for (int d = 0; d < m; d++)
                A0[d, d] += (fProxy)10f;

            var b = arena.fProxyRandomVec(m, -5f, 5f, 8080);

            var Aa = A0.Copy();
            var xa = arena.fProxyVec(n);
            LQ.minNormSolve(in Aa, in b, ref xa);

            var Ab = A0.Copy();
            var xb = arena.fProxyVec(n);
            var ws = arena.fProxyLQMinNormCache(m, n);
            LQ.minNormSolve(in Ab, in b, ref xb, ref ws);

            Assert.IsTrue(Analysis.isZero(xa - xb, Tol()));

            arena.Dispose();
        }

        // Reuse ONE workspace (of each kind) across several consecutive solves: each solve must
        // match a fresh allocating solve, proving no stale state survives reuse.
        void WorkspaceReuse()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 4, n = 8;

            var lqWs = arena.fProxyLQCache(m, n);                     // allocated ONCE, reused below
            var solveWs = arena.fProxyLQMinNormCache(m, n);      // allocated ONCE, reused below

            for (int t = 0; t < 3; t++)
            {
                var A0 = arena.fProxyRandomMat(m, n, -5f, 5f, (uint)(3000 + t * 13));
                for (int d = 0; d < m; d++)
                    A0[d, d] += (fProxy)10f;

                // LQ.decomp: allocating reference vs reused workspace
                var Aa = A0.Copy();
                var La = arena.fProxyMat(m, m);
                var Qa = arena.fProxyMat(m, n);
                LQ.decomp(in Aa, ref La, ref Qa);

                var Aw = A0.Copy();
                var Lw = arena.fProxyMat(m, m);
                var Qw = arena.fProxyMat(m, n);
                LQ.decomp(in Aw, ref Lw, ref Qw, ref lqWs);

                Assert.IsTrue(Analysis.isZero(La - Lw, Tol()));
                Assert.IsTrue(Analysis.isZero(Qa - Qw, Tol()));

                // LQ.minNormSolve: allocating reference vs reused workspace
                var b = arena.fProxyRandomVec(m, -5f, 5f, (uint)(4000 + t * 17));

                var Asa = A0.Copy();
                var xa = arena.fProxyVec(n);
                LQ.minNormSolve(in Asa, in b, ref xa);

                var Asw = A0.Copy();
                var xw = arena.fProxyVec(n);
                LQ.minNormSolve(in Asw, in b, ref xw, ref solveWs);

                Assert.IsTrue(Analysis.isZero(xa - xw, Tol()));
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
    public void LqDecomp_BadScratchW_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(4, 8);
            var L = arena.fProxyMat(4, 4);
            var Q = arena.fProxyMat(4, 8);
            var ws = arena.fProxyLQCache(4, 8);
            ws.W = arena.fProxyMat(3, 8);   // wrong: must be m x n = 4 x 8
            Assert.Throws<ArgumentException>(() => LQ.decomp(in A, ref L, ref Q, ref ws));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void LqDecomp_BadScratchV_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(4, 8);
            var L = arena.fProxyMat(4, 4);
            var Q = arena.fProxyMat(4, 8);
            var ws = arena.fProxyLQCache(4, 8);
            ws.v = arena.fProxyVec(5);    // wrong: must be length n = 8
            Assert.Throws<ArgumentException>(() => LQ.decomp(in A, ref L, ref Q, ref ws));
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
            var ws = arena.fProxyLQMinNormCache(4, 8);
            ws.L = arena.fProxyMat(3, 3);   // wrong: must be m x m = 4 x 4
            Assert.Throws<ArgumentException>(() => LQ.minNormSolve(in A, in b, ref x, ref ws));
        }
        finally { arena.Dispose(); }
    }

    // Arena.fProxyLQCache(m, n) / fProxyLQMinNormCache(m, n) must size every field for m x n.
    [Test]
    public void LQWorkspace_Factory_SizesCorrectly()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var lqWs = arena.fProxyLQCache(4, 8);
            Assert.AreEqual(4, lqWs.W.M_Rows);
            Assert.AreEqual(8, lqWs.W.N_Cols);
            Assert.AreEqual(8, lqWs.v.N);

            var solveWs = arena.fProxyLQMinNormCache(4, 8);
            Assert.AreEqual(4, solveWs.L.M_Rows);
            Assert.AreEqual(4, solveWs.L.N_Cols);
            Assert.AreEqual(4, solveWs.y.N);
            // No dense-Q buffer any more — the fused solve applies Qᵀ from the reflectors in LQWs.W,
            // which doubles as the factor-only working buffer.
            Assert.AreEqual(4, solveWs.LQWs.W.M_Rows);
            Assert.AreEqual(8, solveWs.LQWs.W.N_Cols);
        }
        finally { arena.Dispose(); }
    }
}
