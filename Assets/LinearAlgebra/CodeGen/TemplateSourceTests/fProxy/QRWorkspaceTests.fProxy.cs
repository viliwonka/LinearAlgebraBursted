using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Phase-2 solver-workspace tests for QR: the caller-provided-scratch QR overloads
// (decompInPlace(...,ref u) / solveInPlace(...,ref u)) must produce results identical
// to the allocating wrappers (they run the SAME kernel), and a mis-sized scratch must throw.
public class fProxyQRWorkspaceTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct WorkspaceEquivJob : IJob
    {
        public enum TestType
        {
            DecompEquiv,
            DecompEquivTall,
            DirectSolveEquiv,
            SolveQRSolve,
            SolveQRSolveTall,
        }

        public TestType Type;

        // The scratch overload runs the SAME kernel as the allocating form, so results are
        // bit-identical in principle. Keep a small per-precision tolerance for robustness.
        static fProxy Tol() => 256 * Consts.fProxySqrtEps;

        // Looser per-solve bound for an actual numeric QR solve (matches QRTests).
        static fProxy SolveTol() => 2000 * Consts.fProxySqrtEps;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.DecompEquiv:      DecompEquiv(8, 8); break;
                case TestType.DecompEquivTall:  DecompEquiv(13, 7); break;
                case TestType.DirectSolveEquiv:   DirectSolveEquiv(); break;
                case TestType.SolveQRSolve:       SolveQRSolve(16, 16); break;
                case TestType.SolveQRSolveTall:   SolveQRSolve(13, 7); break;
            }
        }

        // decompInPlace(ref Q, ref R, ref u) must equal decompInPlace(ref Q, ref R).
        void DecompEquiv(int M, int N)
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyRandomMat(M, N, -1f, 1f, 73101);

            // allocating reference
            var Qa = A.Copy();
            var Ra = arena.fProxyMat(N);
            QR.decompInPlace(ref Qa, ref Ra);

            // caller-scratch form
            var Qb = A.Copy();
            var Rb = arena.fProxyMat(N);
            var u = arena.fProxyVec(M);
            QR.decompInPlace(ref Qb, ref Rb, ref u);

            Assert.IsTrue(Analysis.isZero(Qa - Qb, Tol()));
            Assert.IsTrue(Analysis.isZero(Ra - Rb, Tol()));

            arena.Dispose();
        }

        // solveInPlace(ref A, ref b, ref x, ref u) must equal solveInPlace(ref A, ref b, ref x).
        void DirectSolveEquiv()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            var A0 = arena.fProxyRandomMat(dim, dim, -1f, 1f, 51237);
            // make well-conditioned
            for (int d = 0; d < dim; d++)
                A0[d, d] += 5f;

            var xOrig = arena.fProxyRandomVec(dim, -3f, 3f, 99001);

            // allocating reference (solveInPlace destroys A and b, so use fresh copies)
            var Aa = A0.Copy();
            var ba = Blas.dot(A0, xOrig);
            var xa = arena.fProxyVec(dim);
            QR.solveInPlace(ref Aa, ref ba, ref xa);

            // caller-scratch form
            var Ab = A0.Copy();
            var bb = Blas.dot(A0, xOrig);
            var xb = arena.fProxyVec(dim);
            var u = arena.fProxyVec(dim);
            QR.solveInPlace(ref Ab, ref bb, ref xb, ref u);

            Assert.IsTrue(Analysis.isZero(xa - xb, Tol()));

            arena.Dispose();
        }

        // QR.decompSolve (precomputed-QR path): the ref-dest overload must recover xOrig from a
        // consistent b = A xOrig (square M==N, or tall/overdetermined M>N), and the allocating
        // convenience must agree with it bit-for-bit. xOrig has length N (= Q.N_Cols).
        void SolveQRSolve(int M, int N)
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.fProxyRandomMat(M, N, -1f, 1f, 33417);
            for (int d = 0; d < N; d++)   // well-conditioned columns
                A[d, d] += 5f;

            var xOrig = arena.fProxyRandomVec(N, -3f, 3f, 60221);
            var b = Blas.dot(A, xOrig);   // consistent RHS; read-only in decompSolve, reusable

            // Precompute QR of A (decompInPlace overwrites Q with the orthogonal factor)
            var Q = A.Copy();
            var R = arena.fProxyMat(N);
            QR.decompInPlace(ref Q, ref R);

            // ref-destination overload recovers x (length N)
            var x = arena.fProxyVec(N);
            QR.decompSolve(ref Q, ref R, ref b, ref x);
            Assert.IsTrue(Analysis.isZero(x - xOrig, SolveTol()));

            // allocating convenience must match the ref form exactly (same kernel)
            var xc = QR.decompSolve(ref Q, ref R, ref b);
            Assert.IsTrue(Analysis.isZero(xc - x, Tol()));

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
    public void QrDecomp_BadScratchSize_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var Q = arena.fProxyMat(6, 4);
            var R = arena.fProxyMat(4);
            var badU = arena.fProxyVec(3);   // must be length 6 (Q.M_Rows)
            Assert.Throws<ArgumentException>(() => QR.decompInPlace(ref Q, ref R, ref badU));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void QrDirectSolve_BadScratchSize_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(6, 4);
            var b = arena.fProxyVec(6);
            var x = arena.fProxyVec(4);
            var badU = arena.fProxyVec(4);   // must be length 6 (A.M_Rows)
            Assert.Throws<ArgumentException>(() => QR.solveInPlace(ref A, ref b, ref x, ref badU));
        }
        finally { arena.Dispose(); }
    }

    // decompSolve ref-dest: destination x of the wrong length must throw (Solvers guard x.N != Q.N_Cols).
    [Test]
    public void SolveQR_BadDestSize_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(4, 4);
            for (int d = 0; d < 4; d++) A[d, d] = 2f;   // nonsingular so QR is well-defined
            var Q = A.Copy();
            var R = arena.fProxyMat(4);
            QR.decompInPlace(ref Q, ref R);

            var b = arena.fProxyVec(4);
            var badX = arena.fProxyVec(3);   // must be length 4 (Q.N_Cols)
            Assert.Throws<ArgumentException>(() => QR.decompSolve(ref Q, ref R, ref b, ref badX));
        }
        finally { arena.Dispose(); }
    }

    // decompSolve ref-dest: x must not alias b (the underlying ref-dest vec·mat dot guards this).
    [Test]
    public void SolveQR_DestAliasesB_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(4, 4);
            for (int d = 0; d < 4; d++) A[d, d] = 2f;
            var Q = A.Copy();
            var R = arena.fProxyMat(4);
            QR.decompInPlace(ref Q, ref R);

            var b = arena.fProxyVec(4);
            var aliasB = b;   // shares b's buffer; length 4 == Q.N_Cols so it passes the dim guard
            Assert.Throws<ArgumentException>(() => QR.decompSolve(ref Q, ref R, ref b, ref aliasB));
        }
        finally { arena.Dispose(); }
    }
}
