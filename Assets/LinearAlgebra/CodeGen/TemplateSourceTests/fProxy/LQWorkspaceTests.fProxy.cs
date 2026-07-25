using System;

using BULA;
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
            var A0 = GenerateOP.fProxyRandomMat(m, n, -5f, 5f, 41221);
            for (int d = 0; d < m; d++)   // boost leading diagonal block for conditioning
                A0[d, d] += (fProxy)10f;

            // allocating reference
            var Aa = new fProxyMxN(in A0, Allocator.Temp);
            var La = new fProxyMxN(m, m, Allocator.Temp);
            var Qa = new fProxyMxN(m, n, Allocator.Temp);
            LQ.decomp(in Aa, ref La, ref Qa);

            // workspace-struct form must match the allocating form
            var Ab = new fProxyMxN(in A0, Allocator.Temp);
            var Lb = new fProxyMxN(m, m, Allocator.Temp);
            var Qb = new fProxyMxN(m, n, Allocator.Temp);
            var ws = new fProxyLQCache(m, n, Allocator.Temp);
            LQ.decomp(in Ab, ref Lb, ref Qb, ref ws);

            var dL = new fProxyMxN(in La, Allocator.Temp);
            fProxyComp.subInPlace(dL, Lb);
            Assert.IsTrue(Analysis.isZero(dL, Tol()));
            var dQ = new fProxyMxN(in Qa, Allocator.Temp);
            fProxyComp.subInPlace(dQ, Qb);
            Assert.IsTrue(Analysis.isZero(dQ, Tol()));
        }

        void LqMinNormSolveEquiv(int m, int n)
        {
            var A0 = GenerateOP.fProxyRandomMat(m, n, -5f, 5f, 61729);
            for (int d = 0; d < m; d++)
                A0[d, d] += (fProxy)10f;

            var b = GenerateOP.fProxyRandomVec(m, -5f, 5f, 8080);

            var Aa = new fProxyMxN(in A0, Allocator.Temp);
            var xa = new fProxyN(n, Allocator.Temp);
            LQ.minNormSolve(in Aa, in b, ref xa);

            var Ab = new fProxyMxN(in A0, Allocator.Temp);
            var xb = new fProxyN(n, Allocator.Temp);
            var ws = new fProxyLQMinNormCache(m, n, Allocator.Temp);
            LQ.minNormSolve(in Ab, in b, ref xb, ref ws);

            var dx = new fProxyN(in xa, Allocator.Temp);
            fProxyComp.subInPlace(dx, xb);
            Assert.IsTrue(Analysis.isZero(dx, Tol()));
        }

        // Reuse ONE workspace (of each kind) across several consecutive solves: each solve must
        // match a fresh allocating solve, proving no stale state survives reuse.
        void WorkspaceReuse()
        {
            int m = 4, n = 8;

            var lqWs = new fProxyLQCache(m, n, Allocator.Temp);                     // allocated ONCE, reused below
            var solveWs = new fProxyLQMinNormCache(m, n, Allocator.Temp);      // allocated ONCE, reused below

            for (int t = 0; t < 3; t++)
            {
                var A0 = GenerateOP.fProxyRandomMat(m, n, -5f, 5f, (uint)(3000 + t * 13));
                for (int d = 0; d < m; d++)
                    A0[d, d] += (fProxy)10f;

                // LQ.decomp: allocating reference vs reused workspace
                var Aa = new fProxyMxN(in A0, Allocator.Temp);
                var La = new fProxyMxN(m, m, Allocator.Temp);
                var Qa = new fProxyMxN(m, n, Allocator.Temp);
                LQ.decomp(in Aa, ref La, ref Qa);

                var Aw = new fProxyMxN(in A0, Allocator.Temp);
                var Lw = new fProxyMxN(m, m, Allocator.Temp);
                var Qw = new fProxyMxN(m, n, Allocator.Temp);
                LQ.decomp(in Aw, ref Lw, ref Qw, ref lqWs);

                var dL = new fProxyMxN(in La, Allocator.Temp);
                fProxyComp.subInPlace(dL, Lw);
                Assert.IsTrue(Analysis.isZero(dL, Tol()));
                var dQ = new fProxyMxN(in Qa, Allocator.Temp);
                fProxyComp.subInPlace(dQ, Qw);
                Assert.IsTrue(Analysis.isZero(dQ, Tol()));

                // LQ.minNormSolve: allocating reference vs reused workspace
                var b = GenerateOP.fProxyRandomVec(m, -5f, 5f, (uint)(4000 + t * 17));

                var Asa = new fProxyMxN(in A0, Allocator.Temp);
                var xa = new fProxyN(n, Allocator.Temp);
                LQ.minNormSolve(in Asa, in b, ref xa);

                var Asw = new fProxyMxN(in A0, Allocator.Temp);
                var xw = new fProxyN(n, Allocator.Temp);
                LQ.minNormSolve(in Asw, in b, ref xw, ref solveWs);

                var dx = new fProxyN(in xa, Allocator.Temp);
                fProxyComp.subInPlace(dx, xw);
                Assert.IsTrue(Analysis.isZero(dx, Tol()));
            }
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
        var A = new fProxyMxN(4, 8, Allocator.Temp);
        var L = new fProxyMxN(4, 4, Allocator.Temp);
        var Q = new fProxyMxN(4, 8, Allocator.Temp);
        var ws = new fProxyLQCache(4, 8, Allocator.Temp);
        ws.W = new fProxyMxN(3, 8, Allocator.Temp);   // wrong: must be m x n = 4 x 8
        Assert.Throws<ArgumentException>(() => LQ.decomp(in A, ref L, ref Q, ref ws));
    }

    [Test]
    public void LqDecomp_BadScratchV_Throws()
    {
        var A = new fProxyMxN(4, 8, Allocator.Temp);
        var L = new fProxyMxN(4, 4, Allocator.Temp);
        var Q = new fProxyMxN(4, 8, Allocator.Temp);
        var ws = new fProxyLQCache(4, 8, Allocator.Temp);
        ws.v = new fProxyN(5, Allocator.Temp);    // wrong: must be length n = 8
        Assert.Throws<ArgumentException>(() => LQ.decomp(in A, ref L, ref Q, ref ws));
    }

    [Test]
    public void LqMinNormSolve_BadScratchL_Throws()
    {
        var A = new fProxyMxN(4, 8, Allocator.Temp);
        var b = new fProxyN(4, Allocator.Temp);
        var x = new fProxyN(8, Allocator.Temp);
        var ws = new fProxyLQMinNormCache(4, 8, Allocator.Temp);
        ws.L = new fProxyMxN(3, 3, Allocator.Temp);   // wrong: must be m x m = 4 x 4
        Assert.Throws<ArgumentException>(() => LQ.minNormSolve(in A, in b, ref x, ref ws));
    }

    // fProxyLQCache(m, n, allocator) / fProxyLQMinNormCache(m, n, allocator) must size every field for m x n.
    [Test]
    public void LQWorkspace_Factory_SizesCorrectly()
    {
        var lqWs = new fProxyLQCache(4, 8, Allocator.Temp);
        Assert.AreEqual(4, lqWs.W.M_Rows);
        Assert.AreEqual(8, lqWs.W.N_Cols);
        Assert.AreEqual(8, lqWs.v.N);

        var solveWs = new fProxyLQMinNormCache(4, 8, Allocator.Temp);
        Assert.AreEqual(4, solveWs.L.M_Rows);
        Assert.AreEqual(4, solveWs.L.N_Cols);
        Assert.AreEqual(4, solveWs.y.N);
        // No dense-Q buffer any more — the fused solve applies Qᵀ from the reflectors in LQWs.W,
        // which doubles as the factor-only working buffer.
        Assert.AreEqual(4, solveWs.LQWs.W.M_Rows);
        Assert.AreEqual(8, solveWs.LQWs.W.N_Cols);
    }
}
