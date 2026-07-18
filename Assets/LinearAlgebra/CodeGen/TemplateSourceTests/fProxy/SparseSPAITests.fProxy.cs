using System;
using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Row-oriented sparse approximate inverse preconditioner (fProxySPAI), for pbiCGStab ONLY.
// SPAI is NOT symmetric even for symmetric A, so it is intentionally NOT wired to pcg/pminres --
// there is no such overload, and this suite never attempts one (a CG/MINRES call with SPAI would
// not compile). Correctness anchors (spec section 8):
//   (4) RESIDUAL QUALITY: on a nonsymmetric diagonally-dominant BSR, SPAI beats Jacobi scaling:
//       ||M A - I||_F < ||D^-1 A - I||_F; pbiCGStab with SPAI converges to the true solution on
//       the same matrix family the ILU0 test uses.
//   (5) CLEAN BUILD: on a well-conditioned matrix the out-info build reports Solved==true /
//       Success with Shift==0 (a genuine SPAI breakdown cannot be forced: its local normal-equation
//       system N = A_hat A_hat^T is PSD, so the first nonzero Tikhonov shift always rescues it --
//       noted in the report).
//   (6) GUARDS: non-square A, missing diagonal block, and Apply aliasing (z==r) throw
//       ArgumentException. SPAI has no owned Scratch, so it has no z==Scratch/r==Scratch guard.
//   (7) THROUGH-IJOB DETERMINISM: SPAI built once on the main thread, pbiCGStab run inside a Burst
//       IJob.Run() twice, gives bit-identical iteration counts and bit-identical x; the managed
//       path matches to tolerance.
public class fProxySparseSPAITests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct SparseSPAITestJob : IJob
    {
        public enum TestType
        {
            ResidualBeatsJacobi,
            PbiCGStabConverges,
            CleanBuildReportsSuccess,
        }

        public TestType Type;

        static fProxy Tol() => /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;
        static fProxy SolveTol() => /*+choose[1e-3f|1e-7]*/1e-3f/*-choose*/;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ResidualBeatsJacobi: ResidualBeatsJacobi(); break;
                case TestType.PbiCGStabConverges: PbiCGStabConverges(); break;
                case TestType.CleanBuildReportsSuccess: CleanBuildReportsSuccess(); break;
            }
        }

        // ---- helpers ---------------------------------------------------------------------

        // Diagonally dominant NONSYMMETRIC block-tridiagonal system (fill-free pattern) -- the same
        // construction the ILU0 test uses, so SPAI converges where ILU0's testbed converges.
        static fProxyBSR BuildNonsymTridiag(ref Arena arena, int nb, int BR, uint seed)
        {
            var builder = arena.fProxyBSRBuilder(nb, nb, BR, BR);
            var rng = new Unity.Mathematics.Random(seed);
            var blk = arena.fProxyMat(BR, BR);

            for (int i = 0; i < nb; i++)
            {
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        blk[r, c] = (r == c ? (fProxy)(4 * BR) : (fProxy)0) + (fProxy)rng.NextFloat(-0.5f, 0.5f);
                builder.AddBlock(i, i, in blk);

                if (i + 1 < nb)
                {
                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BR; c++)
                            blk[r, c] = (fProxy)rng.NextFloat(-0.6f, 0.6f);
                    builder.AddBlock(i + 1, i, in blk);

                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BR; c++)
                            blk[r, c] = (fProxy)rng.NextFloat(-0.6f, 0.6f);   // NOT the transpose
                    builder.AddBlock(i, i + 1, in blk);
                }
            }
            return builder.ToBSR(ref arena);
        }

        // ||X - I||_F for a square X.
        static fProxy FrobeniusMinusI(in fProxyMxN X)
        {
            fProxy s = 0;
            for (int r = 0; r < X.M_Rows; r++)
                for (int c = 0; c < X.N_Cols; c++)
                {
                    fProxy v = X[r, c] - (r == c ? (fProxy)1 : (fProxy)0);
                    s += v * v;
                }
            return math.sqrt(s);
        }

        // ================================================================================
        // (4) SPAI beats Jacobi in Frobenius residual of the approximate inverse.
        // ================================================================================

        void ResidualBeatsJacobi()
        {
            var arena = new Arena(Allocator.Persistent);
            const int nb = 8, BR = 3;
            var A = BuildNonsymTridiag(ref arena, nb, BR, 851001u);
            int n = A.M_Rows;

            var spai = arena.fProxySPAI(in A);
            Assert.IsTrue(spai.Shift == (fProxy)0);

            var Adense = A.ToDense(ref arena);
            var Mdense = spai.M.ToDense(ref arena);

            // ||M A - I||_F
            var MA = arena.fProxyMat(n, n);
            Blas.dot(in Mdense, in Adense, ref MA);   // plain product M*A
            fProxy froSpai = FrobeniusMinusI(in MA);

            // ||D^-1 A - I||_F, with D^-1 the block-Jacobi (diagonal-block inverse) scaling.
            var jac = arena.fProxyBlockJacobi(in A);
            var DinvA = arena.fProxyMat(n, n);
            var col = arena.fProxyVec(n);
            var outc = arena.fProxyVec(n);
            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < n; i++) col[i] = Adense[i, j];
                jac.Apply(in col, ref outc);
                for (int i = 0; i < n; i++) DinvA[i, j] = outc[i];
            }
            fProxy froJac = FrobeniusMinusI(in DinvA);

            Assert.IsTrue(froSpai < froJac);

            arena.Dispose();
        }

        void PbiCGStabConverges()
        {
            var arena = new Arena(Allocator.Persistent);
            const int nb = 40, BR = 3;
            var A = BuildNonsymTridiag(ref arena, nb, BR, 852001u);
            int n = A.M_Rows;

            var xTrue = arena.fProxyRandomVec(n, 0.5f, 1.5f, 852002u);
            var b = BSR.spMV(in A, in xTrue);
            fProxy tol = Consts.fProxySqrtEps;
            int maxIter = 4 * n;

            var M = arena.fProxySPAI(in A);
            var x = arena.fProxyVec(n);
            var info = Krylov.pbiCGStab(in A, in M, in b, ref x, maxIter, tol);
            Assert.IsTrue(info.Solved);
            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(x[i] - xTrue[i]) < SolveTol() * ((fProxy)1 + math.abs(xTrue[i])));

            arena.Dispose();
        }

        // ================================================================================
        // (5) Well-conditioned build reports Success with no shift (non-throwing twin).
        // ================================================================================

        void CleanBuildReportsSuccess()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomSparse(24, 24, 3, (fProxy)0.4, 853001u);  // square, DD nonsymmetric
            var M = arena.fProxySPAI(in A, out PreconditionerInfo info);
            Assert.IsTrue(info.Solved);
            Assert.IsTrue(info.status == DirectSolveStatus.Success);
            Assert.IsTrue(M.Shift == (fProxy)0);
            Assert.AreEqual(1, info.attempts);
            arena.Dispose();
        }
    }

    // Solve-only job for the through-IJob determinism test: SPAI built ONCE on the main thread.
    [BurstCompile(CompileSynchronously = true)]
    public struct SPAISolveJob : IJob
    {
        public fProxyBSR A;
        public fProxySPAI M;
        public fProxyN b;
        public fProxyN x;
        public NativeArray<int> iters;
        public int maxIter;
        public fProxy tol;

        public void Execute()
        {
            var info = Krylov.pbiCGStab(in A, in M, in b, ref x, maxIter, tol);
            iters[0] = info.iterations;
        }
    }

    // ---- correctness cases (Burst) -------------------------------------------------------

    [Test] public void ResidualBeatsJacobiTest()
        => new SparseSPAITestJob { Type = SparseSPAITestJob.TestType.ResidualBeatsJacobi }.Run();
    [Test] public void PbiCGStabConvergesTest()
        => new SparseSPAITestJob { Type = SparseSPAITestJob.TestType.PbiCGStabConverges }.Run();
    [Test] public void CleanBuildReportsSuccessTest()
        => new SparseSPAITestJob { Type = SparseSPAITestJob.TestType.CleanBuildReportsSuccess }.Run();

    // ---- (7) through-IJob determinism (managed orchestration of Burst jobs) ---------------

    [Test]
    public void ThroughIJobDeterminismTest()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyRandomSparse(20, 20, 3, (fProxy)0.4, 854001u);
        var M = arena.fProxySPAI(in A);
        int n = A.M_Rows;

        var xTrue = arena.fProxyRandomVec(n, 0.5f, 1.5f, 854002u);
        var b = BSR.spMV(in A, in xTrue);
        fProxy tol = Consts.fProxySqrtEps;
        int maxIter = 8 * n;

        var x1 = arena.fProxyVec(n);
        var x2 = arena.fProxyVec(n);
        var it1 = new NativeArray<int>(1, Allocator.Persistent);
        var it2 = new NativeArray<int>(1, Allocator.Persistent);

        new SPAISolveJob { A = A, M = M, b = b, x = x1, iters = it1, maxIter = maxIter, tol = tol }.Run();
        new SPAISolveJob { A = A, M = M, b = b, x = x2, iters = it2, maxIter = maxIter, tol = tol }.Run();

        Assert.AreEqual(it1[0], it2[0]);
        for (int i = 0; i < n; i++)
            Assert.IsTrue(x1[i] == x2[i]);

        var x3 = arena.fProxyVec(n);
        var infoM = Krylov.pbiCGStab(in A, in M, in b, ref x3, maxIter, tol);
        Assert.IsTrue(infoM.Solved);
        fProxy consistencyTol = /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;
        for (int i = 0; i < n; i++)
            Assert.IsTrue(math.abs(x1[i] - x3[i]) < consistencyTol * ((fProxy)1 + math.abs(x3[i])));

        it1.Dispose();
        it2.Dispose();
        arena.Dispose();
    }

    // ---- (6) guard cases (managed thread) ------------------------------------------------

    [Test]
    public void NonSquareThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var builder = arena.fProxyBSRBuilder(2, 3, 2, 2);
            var block = arena.fProxyMat(2, 2, (fProxy)1);
            builder.AddBlock(0, 0, in block);
            var A = builder.ToBSR(ref arena);
            Assert.Throws<ArgumentException>(() => { var m = arena.fProxySPAI(in A); });
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void MissingDiagonalThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var builder = arena.fProxyBSRBuilder(2, 2, 2, 2);
            var block = arena.fProxyMat(2, 2, (fProxy)1);
            builder.AddBlock(0, 0, in block);
            builder.AddBlock(1, 0, in block);   // no (1,1) diagonal block
            var A = builder.ToBSR(ref arena);
            Assert.Throws<ArgumentException>(() => { var m = arena.fProxySPAI(in A); });
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void ApplyAliasThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var builder = arena.fProxyBSRBuilder(2, 2, 2, 2);
            var diag = arena.fProxyMat(2, 2);
            diag[0, 0] = (fProxy)4; diag[1, 1] = (fProxy)4;
            builder.AddBlock(0, 0, in diag);
            builder.AddBlock(1, 1, in diag);
            var A = builder.ToBSR(ref arena);
            var M = arena.fProxySPAI(in A);

            var r = arena.fProxyVec(A.M_Rows, (fProxy)1);
            Assert.Throws<ArgumentException>(() => M.Apply(in r, ref r));
        }
        finally { arena.Dispose(); }
    }
}
