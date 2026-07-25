using System;
using BULA;
using BULA.Gallery;
using BULA.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Row-oriented sparse approximate inverse preconditioner (fProxySPAI), for biCGStab ONLY.
// SPAI is NOT symmetric even for symmetric A, so it is intentionally NOT wired to cg/minres --
// there is no such overload, and this suite never attempts one (a CG/MINRES call with SPAI would
// not compile). Correctness anchors (spec section 8):
//   (4) RESIDUAL QUALITY: on a nonsymmetric diagonally-dominant BSR, SPAI beats Jacobi scaling:
//       ||M A - I||_F < ||D^-1 A - I||_F; biCGStab with SPAI converges to the true solution on
//       the same matrix family the ILU0 test uses.
//   (5) CLEAN BUILD: on a well-conditioned matrix the out-info build reports Solved==true /
//       Success with Shift==0 (a genuine SPAI breakdown cannot be forced: its local normal-equation
//       system N = A_hat A_hat^T is PSD, so the first nonzero Tikhonov shift always rescues it --
//       noted in the report).
//   (6) GUARDS: non-square A, missing diagonal block, and Apply aliasing (z==r) throw
//       ArgumentException. SPAI has no owned Scratch, so it has no z==Scratch/r==Scratch guard.
//   (7) THROUGH-IJOB DETERMINISM: SPAI built once on the main thread, biCGStab run inside a Burst
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
        static fProxyBSR BuildNonsymTridiag(int nb, int BR, uint seed)
        {
            var builder = new fProxyBSRBuilder(nb, nb, BR, BR, Allocator.Temp);
            var rng = new Unity.Mathematics.Random(seed);
            var blk = new fProxyMxN(BR, BR, Allocator.Temp);

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
            return builder.ToBSR(Allocator.Temp);
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
            const int nb = 8, BR = 3;
            var A = BuildNonsymTridiag(nb, BR, 851001u);
            int n = A.M_Rows;

            var spai = new fProxySPAI(in A, Allocator.Temp);
            Assert.IsTrue(spai.Shift == (fProxy)0);

            var Adense = A.ToDense(Allocator.Temp);
            var Mdense = spai.M.ToDense(Allocator.Temp);

            // ||M A - I||_F
            var MA = new fProxyMxN(n, n, Allocator.Temp);
            Blas.dot(in Mdense, in Adense, ref MA);   // plain product M*A
            fProxy froSpai = FrobeniusMinusI(in MA);

            // ||D^-1 A - I||_F, with D^-1 the block-Jacobi (diagonal-block inverse) scaling.
            var jac = new fProxyBlockJacobi(in A, Allocator.Temp);
            var DinvA = new fProxyMxN(n, n, Allocator.Temp);
            var col = new fProxyN(n, Allocator.Temp);
            var outc = new fProxyN(n, Allocator.Temp);
            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < n; i++) col[i] = Adense[i, j];
                jac.Apply(in col, ref outc);
                for (int i = 0; i < n; i++) DinvA[i, j] = outc[i];
            }
            fProxy froJac = FrobeniusMinusI(in DinvA);

            Assert.IsTrue(froSpai < froJac);
        }

        void PbiCGStabConverges()
        {
            const int nb = 40, BR = 3;
            var A = BuildNonsymTridiag(nb, BR, 852001u);
            int n = A.M_Rows;

            var xTrue = GenerateOP.fProxyRandomVec(n, 0.5f, 1.5f, 852002u);
            var b = BSR.spMV(in A, in xTrue);
            fProxy tol = Consts.fProxySqrtEps;
            int maxIter = 4 * n;

            var M = new fProxySPAI(in A, Allocator.Temp);
            var x = new fProxyN(n, Allocator.Temp);
            var info = Krylov.biCGStab(in A, in M, in b, ref x, maxIter, tol);
            Assert.IsTrue(info.Solved);
            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(x[i] - xTrue[i]) < SolveTol() * ((fProxy)1 + math.abs(xTrue[i])));
        }

        // ================================================================================
        // (5) Well-conditioned build reports Success with no shift (non-throwing twin).
        // ================================================================================

        void CleanBuildReportsSuccess()
        {
            var A = fProxyGallery.fProxyRandomSparse(24, 24, 3, (fProxy)0.4, 853001u);  // square, DD nonsymmetric
            var M = new fProxySPAI(in A, Allocator.Temp, out PreconditionerInfo info);
            Assert.IsTrue(info.Solved);
            Assert.IsTrue(info.status == DirectSolveStatus.Success);
            Assert.IsTrue(M.Shift == (fProxy)0);
            Assert.AreEqual(1, info.attempts);
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
            var info = Krylov.biCGStab(in A, in M, in b, ref x, maxIter, tol);
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
        var A = fProxyGallery.fProxyRandomSparse(20, 20, 3, (fProxy)0.4, 854001u);
        var M = new fProxySPAI(in A, Allocator.Temp);
        int n = A.M_Rows;

        var xTrue = GenerateOP.fProxyRandomVec(n, 0.5f, 1.5f, 854002u);
        var b = BSR.spMV(in A, in xTrue);
        fProxy tol = Consts.fProxySqrtEps;
        int maxIter = 8 * n;

        var x1 = new fProxyN(n, Allocator.Temp);
        var x2 = new fProxyN(n, Allocator.Temp);
        var it1 = new NativeArray<int>(1, Allocator.Persistent);
        var it2 = new NativeArray<int>(1, Allocator.Persistent);

        new SPAISolveJob { A = A, M = M, b = b, x = x1, iters = it1, maxIter = maxIter, tol = tol }.Run();
        new SPAISolveJob { A = A, M = M, b = b, x = x2, iters = it2, maxIter = maxIter, tol = tol }.Run();

        Assert.AreEqual(it1[0], it2[0]);
        for (int i = 0; i < n; i++)
            Assert.IsTrue(x1[i] == x2[i]);

        var x3 = new fProxyN(n, Allocator.Temp);
        var infoM = Krylov.biCGStab(in A, in M, in b, ref x3, maxIter, tol);
        Assert.IsTrue(infoM.Solved);
        fProxy consistencyTol = /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;
        for (int i = 0; i < n; i++)
            Assert.IsTrue(math.abs(x1[i] - x3[i]) < consistencyTol * ((fProxy)1 + math.abs(x3[i])));

        it1.Dispose();
        it2.Dispose();
    }

    // ---- (6) guard cases (managed thread) ------------------------------------------------

    [Test]
    public void NonSquareThrows()
    {
        var builder = new fProxyBSRBuilder(2, 3, 2, 2, Allocator.Temp);
        var block = GenerateOP.fProxyMat(2, 2, (fProxy)1);
        builder.AddBlock(0, 0, in block);
        var A = builder.ToBSR(Allocator.Temp);
        Assert.Throws<ArgumentException>(() => { var m = new fProxySPAI(in A, Allocator.Temp); });
    }

    [Test]
    public void MissingDiagonalThrows()
    {
        var builder = new fProxyBSRBuilder(2, 2, 2, 2, Allocator.Temp);
        var block = GenerateOP.fProxyMat(2, 2, (fProxy)1);
        builder.AddBlock(0, 0, in block);
        builder.AddBlock(1, 0, in block);   // no (1,1) diagonal block
        var A = builder.ToBSR(Allocator.Temp);
        Assert.Throws<ArgumentException>(() => { var m = new fProxySPAI(in A, Allocator.Temp); });
    }

    [Test]
    public void ApplyAliasThrows()
    {
        var builder = new fProxyBSRBuilder(2, 2, 2, 2, Allocator.Temp);
        var diag = new fProxyMxN(2, 2, Allocator.Temp);
        diag[0, 0] = (fProxy)4; diag[1, 1] = (fProxy)4;
        builder.AddBlock(0, 0, in diag);
        builder.AddBlock(1, 1, in diag);
        var A = builder.ToBSR(Allocator.Temp);
        var M = new fProxySPAI(in A, Allocator.Temp);

        var r = GenerateOP.fProxyVec(A.M_Rows, (fProxy)1);
        Assert.Throws<ArgumentException>(() => M.Apply(in r, ref r));
    }
}
