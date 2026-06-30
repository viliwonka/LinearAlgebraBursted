using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Phase-2 solver-workspace tests for the SVD solvers: the caller-provided-scratch overloads
// pinvSolve(...,ref S,ref M,ref U,ref At) / pseudoInverse(...,ref S,ref M,ref U,ref At) must produce results
// identical to the allocating wrappers (they run the SAME kernel), for both the tall/square
// (m>=n) and wide (m<n) branches; and a mis-sized scratch must throw.
public class fProxySVDWorkspaceTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WorkspaceEquivJob : IJob
    {
        public enum TestType
        {
            PinvEquivTall,
            PinvEquivWide,
            PseudoEquivSquare,
            PseudoEquivWide,
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
                case TestType.PinvEquivTall:     PinvEquiv(12, 4); break;
                case TestType.PinvEquivWide:     PinvEquiv(4, 12); break;
                case TestType.PseudoEquivSquare: PseudoEquiv(6, 6); break;
                case TestType.PseudoEquivWide:   PseudoEquiv(4, 9); break;
                case TestType.WorkspaceReuse:    WorkspaceReuse(); break;
            }
        }

        // pinvSolve scratch overload must match the allocating wrapper bit-for-bit. k = min(m,n).
        void PinvEquiv(int m, int n)
        {
            var arena = new Arena(Allocator.Persistent);
            int k = m < n ? m : n;

            var A0 = arena.fProxyRandomMatrix(m, n, -5f, 5f, 778231);
            for (int d = 0; d < k; d++)   // boost leading diagonal block for conditioning
                A0[d, d] += (fProxy)10f;

            var b = arena.fProxyRandomVector(m, -5f, 5f, 9090);   // read-only in pinvSolve

            // allocating reference (A is no longer modified, but keep per-call copies for clarity)
            var Aa = A0.Copy();
            var xa = arena.fProxyVec(n);
            int ra = SVD.pinvSolve(ref Aa, in b, ref xa, out bool ca);

            // caller-scratch form (same defaults: relTol = -1 auto, maxSweeps = 30)
            var Ab = A0.Copy();
            var xb = arena.fProxyVec(n);
            var S = arena.fProxyVec(k);
            var M = arena.fProxyMat(k, k);
            var U = arena.fProxyMat(m < n ? n : m, k);
            fProxyMxN At = default;
            if (m < n)
                At = arena.fProxyMat(n, m);
            int rb = SVD.pinvSolve(ref Ab, in b, ref xb, out bool cb, (fProxy)(-1), 30, ref S, ref M, ref U, ref At);

            Assert.IsTrue(ra == rb);
            Assert.IsTrue(ca == cb);
            Assert.IsTrue(Analysis_OP.IsZero(xa - xb, Tol()));

            // workspace-struct form (default relTol/maxSweeps) must match the raw-scratch form
            var Aw = A0.Copy();
            var xw = arena.fProxyVec(n);
            var ws = arena.fProxySvd_WS(m, n);
            int rw = SVD.pinvSolve(ref Aw, in b, ref xw, out bool cw, ref ws);

            Assert.IsTrue(rw == rb);
            Assert.IsTrue(cw == cb);
            Assert.IsTrue(Analysis_OP.IsZero(xw - xb, Tol()));

            arena.Dispose();
        }

        // pseudoInverse scratch overload must match the allocating wrapper bit-for-bit.
        void PseudoEquiv(int m, int n)
        {
            var arena = new Arena(Allocator.Persistent);
            int k = m < n ? m : n;

            var A0 = arena.fProxyRandomMatrix(m, n, -5f, 5f, 314221);
            for (int d = 0; d < k; d++)
                A0[d, d] += (fProxy)10f;

            // allocating reference (Aplus is N_Cols x M_Rows = n x m)
            var Aa = A0.Copy();
            var Pa = arena.fProxyMat(n, m);
            int ra = SVD.pseudoInverse(ref Aa, ref Pa, out bool ca);

            // caller-scratch form
            var Ab = A0.Copy();
            var Pb = arena.fProxyMat(n, m);
            var S = arena.fProxyVec(k);
            var M = arena.fProxyMat(k, k);
            var U = arena.fProxyMat(m < n ? n : m, k);
            fProxyMxN At = default;
            if (m < n)
                At = arena.fProxyMat(n, m);
            int rb = SVD.pseudoInverse(ref Ab, ref Pb, out bool cb, (fProxy)(-1), 30, ref S, ref M, ref U, ref At);

            Assert.IsTrue(ra == rb);
            Assert.IsTrue(ca == cb);
            Assert.IsTrue(Analysis_OP.IsZero(Pa - Pb, Tol()));

            // workspace-struct form (default relTol/maxSweeps) must match the raw-scratch form
            var Aw = A0.Copy();
            var Pw = arena.fProxyMat(n, m);
            var ws = arena.fProxySvd_WS(m, n);
            int rw = SVD.pseudoInverse(ref Aw, ref Pw, out bool cw, ref ws);

            Assert.IsTrue(rw == rb);
            Assert.IsTrue(cw == cb);
            Assert.IsTrue(Analysis_OP.IsZero(Pw - Pb, Tol()));

            arena.Dispose();
        }

        // Reuse ONE workspace across several consecutive solves (the feature's whole purpose):
        // each solve must match a fresh allocating solve, proving no stale state survives reuse.
        void WorkspaceReuse()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 8, n = 4;

            var A0 = arena.fProxyRandomMatrix(m, n, -5f, 5f, 24681);
            for (int d = 0; d < n; d++)
                A0[d, d] += (fProxy)10f;

            var ws = arena.fProxySvd_WS(m, n);   // allocated ONCE, reused below

            for (int t = 0; t < 3; t++)
            {
                var b = arena.fProxyRandomVector(m, -5f, 5f, (uint)(1000 + t * 7));

                // allocating reference (fresh internal scratch each call)
                var Aa = A0.Copy();
                var xa = arena.fProxyVec(n);
                SVD.pinvSolve(ref Aa, in b, ref xa, out bool _);

                // reused workspace
                var Aw = A0.Copy();
                var xw = arena.fProxyVec(n);
                SVD.pinvSolve(ref Aw, in b, ref xw, out bool _, ref ws);

                Assert.IsTrue(Analysis_OP.IsZero(xa - xw, Tol()));
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
    public void Pinv_BadScratchS_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(4, 3);     // tall, k = 3
            var b = arena.fProxyVec(4);
            var x = arena.fProxyVec(3);
            var S = arena.fProxyVec(2);        // must be length 3
            var M = arena.fProxyMat(3, 3);
            var U = arena.fProxyMat(4, 3);
            fProxyMxN At = default;
            Assert.Throws<ArgumentException>(
                () => SVD.pinvSolve(ref A, in b, ref x, out bool c, (fProxy)(-1), 30, ref S, ref M, ref U, ref At));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Pinv_BadScratchM_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(4, 3);     // tall, k = 3
            var b = arena.fProxyVec(4);
            var x = arena.fProxyVec(3);
            var S = arena.fProxyVec(3);
            var M = arena.fProxyMat(3, 2);     // must be 3 x 3
            var U = arena.fProxyMat(4, 3);
            fProxyMxN At = default;
            Assert.Throws<ArgumentException>(
                () => SVD.pinvSolve(ref A, in b, ref x, out bool c, (fProxy)(-1), 30, ref S, ref M, ref U, ref At));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Pinv_BadScratchU_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(4, 3);     // tall, k = 3, big = 4 -> U must be 4 x 3
            var b = arena.fProxyVec(4);
            var x = arena.fProxyVec(3);
            var S = arena.fProxyVec(3);
            var M = arena.fProxyMat(3, 3);
            var U = arena.fProxyMat(3, 3);     // wrong: must be 4 x 3
            fProxyMxN At = default;
            Assert.Throws<ArgumentException>(
                () => SVD.pinvSolve(ref A, in b, ref x, out bool c, (fProxy)(-1), 30, ref S, ref M, ref U, ref At));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Pinv_WideMissingAt_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(3, 5);     // wide, k = 3, needs At = 5 x 3
            var b = arena.fProxyVec(3);
            var x = arena.fProxyVec(5);
            var S = arena.fProxyVec(3);
            var M = arena.fProxyMat(3, 3);
            var U = arena.fProxyMat(5, 3);
            fProxyMxN At = default;            // missing (0 x 0) -> must throw
            Assert.Throws<ArgumentException>(
                () => SVD.pinvSolve(ref A, in b, ref x, out bool c, (fProxy)(-1), 30, ref S, ref M, ref U, ref At));
        }
        finally { arena.Dispose(); }
    }

    // Arena.fProxySvd_WS(m, n) must size S (k), M (k x k), and At (n x m only when wide).
    [Test]
    public void SvdWorkspace_Factory_SizesCorrectly()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            // tall: k = n, big = m; U = big x k = 7 x 4; At unused (left default).
            var wsTall = arena.fProxySvd_WS(7, 4);
            Assert.AreEqual(4, wsTall.S.N);
            Assert.AreEqual(4, wsTall.M.M_Rows);
            Assert.AreEqual(4, wsTall.M.N_Cols);
            Assert.AreEqual(7, wsTall.U.M_Rows);
            Assert.AreEqual(4, wsTall.U.N_Cols);

            // wide: k = m, big = n; U = big x k = 8 x 3; At = n x m
            var wsWide = arena.fProxySvd_WS(3, 8);
            Assert.AreEqual(3, wsWide.S.N);
            Assert.AreEqual(3, wsWide.M.M_Rows);
            Assert.AreEqual(3, wsWide.M.N_Cols);
            Assert.AreEqual(8, wsWide.U.M_Rows);
            Assert.AreEqual(3, wsWide.U.N_Cols);
            Assert.AreEqual(8, wsWide.At.M_Rows);
            Assert.AreEqual(3, wsWide.At.N_Cols);
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void PseudoInverse_WideBadAt_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(3, 5);     // wide, k = 3, needs At = 5 x 3
            var Aplus = arena.fProxyMat(5, 3); // N_Cols x M_Rows
            var S = arena.fProxyVec(3);
            var M = arena.fProxyMat(3, 3);
            var U = arena.fProxyMat(5, 3);
            var At = arena.fProxyMat(3, 5);    // wrong shape (must be 5 x 3) -> must throw
            Assert.Throws<ArgumentException>(
                () => SVD.pseudoInverse(ref A, ref Aplus, out bool c, (fProxy)(-1), 30, ref S, ref M, ref U, ref At));
        }
        finally { arena.Dispose(); }
    }
}
