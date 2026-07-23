using System;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using LinearAlgebra.Gallery;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// SSOR preconditioner test coverage:
//   (a) BSR.sweepLower/sweepUpper vs a dense LU solve on the EXPANDED (block-diagonal +
//       strictly-block-triangular, diagonal pre-divided by diagScale) matrix -- b in
//       {1,2,3,4,6} (unrolled) plus b=5 (general runtime-BR fallback), both diagScale=1 (plain
//       Gauss-Seidel) and a nontrivial diagScale (the parameter fProxySSOR actually drives).
//   (b) fProxySSOR is M-SPD: a hand-rolled PCG loop (built from the same public primitives
//       Krylov.cg itself uses -- M.Apply/op.ApplyDot/Blas.dot/Blas.updateXR) asserts <r,z> > 0
//       every iteration and that the solve converges to the true solution -- no new production
//       API added just to expose this; the test reads what is already public.
//   (c) fProxySSOR converges in FEWER iterations than fProxyBlockJacobi (>=10% margin) on both
//       fProxyLaplacian2D and fProxyRandomSparseSPD instances.
//   (d) SSOR built from a Symmetric-storage BSR equals SSOR built from its full-storage twin
//       (the one-time mirror path, fProxyBSR.MirrorToFull).
//   (e) fProxySSOR drops into Eigen.lobpcg<TOp,TPre>'s TPre slot with no new overloads.
//
// Value cases run inside a [BurstCompile] IJob (matches every other sparse suite). Guard-throw
// cases (symmetric-storage sweep input, omega out of range) are managed [Test]s with Assert.Throws
// (Burst cannot surface an assertable managed exception).
public class fProxySSORTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct SSORTestJob : IJob
    {
        public enum TestType
        {
            SweepLowerVsDenseOracle,
            SweepUpperVsDenseOracle,
            SSORPositiveDefiniteAndConverges,
            PcgSSORMatchesLUOracle,
            SSORBeatsJacobiOnLaplacian,
            SSORBeatsJacobiOnRandomSparseSPD,
            SSORSymmetricStorageMatchesFullStorage,
            LobpcgAcceptsSSORPreconditioner,
        }

        public TestType Type;

        // sweep block sizes: unrolled {1,2,3,4,6} + general-fallback {5}.
        static readonly int[] SweepBs = { 1, 2, 3, 4, 5, 6 };

        static fProxy Tol() => /*+choose[1e-3f|1e-8]*/1e-3f/*-choose*/;
        static fProxy SolveTol() => /*+choose[1e-3f|1e-7]*/1e-3f/*-choose*/;
        static fProxy TightTol() => /*+choose[1e-4f|1e-10]*/1e-4f/*-choose*/;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.SweepLowerVsDenseOracle: SweepLowerVsDenseOracle(); break;
                case TestType.SweepUpperVsDenseOracle: SweepUpperVsDenseOracle(); break;
                case TestType.SSORPositiveDefiniteAndConverges: SSORPositiveDefiniteAndConverges(); break;
                case TestType.PcgSSORMatchesLUOracle: PcgSSORMatchesLUOracle(); break;
                case TestType.SSORBeatsJacobiOnLaplacian: SSORBeatsJacobiOnLaplacian(); break;
                case TestType.SSORBeatsJacobiOnRandomSparseSPD: SSORBeatsJacobiOnRandomSparseSPD(); break;
                case TestType.SSORSymmetricStorageMatchesFullStorage: SSORSymmetricStorageMatchesFullStorage(); break;
                case TestType.LobpcgAcceptsSSORPreconditioner: LobpcgAcceptsSSORPreconditioner(); break;
            }
        }

        // ---- helpers ---------------------------------------------------------------------

        static void AssertClose(fProxy got, fProxy expected, fProxy tol)
            => Assert.IsTrue(math.abs(got - expected) <= tol * ((fProxy)1 + math.abs(expected)));

        static void AssertVecClose(in fProxyN got, in fProxyN expected, fProxy tol)
        {
            Assert.AreEqual(expected.N, got.N);
            for (int i = 0; i < got.N; i++) AssertClose(got[i], expected[i], tol);
        }

        // SPD b x b block D = M^T M + b*I: well-conditioned, LU-invertible.
        static fProxyMxN SpdBlock(int b, uint seed)
        {
            var M = GenerateOP.fProxyRandomMat(b, b, -1f, 1f, seed, allocator: Allocator.Temp);
            var D = Blas.dot(M, M, true);
            for (int d = 0; d < b; d++) D[d, d] += (fProxy)b;
            return D;
        }

        static fProxyMxN BuildDenseSPD(int dim, uint seed)
        {
            var M = GenerateOP.fProxyRandomMat(dim, dim, -1f, 1f, seed, allocator: Allocator.Temp);
            var A = Blas.dot(M, M, true);
            for (int d = 0; d < dim; d++) A[d, d] += dim;
            return A;
        }

        static fProxyBSR DenseToBSR1x1(in fProxyMxN A, int nnzHint)
        {
            var builder = new fProxyBSRBuilder(A.M_Rows, A.N_Cols, 1, 1, Allocator.Temp, math.max(nnzHint, 1));
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    if (A[r, c] != (fProxy)0) builder.AddValue(r, c, A[r, c]);
            return builder.ToBSR(Allocator.Temp);
        }

        // Full-storage BSR, invertible (SPD) diagonal blocks + a deterministic scatter of small
        // off-diagonal blocks on BOTH sides of the block diagonal (several per row), so
        // sweepLower/sweepUpper's early break/continue and multi-block accumulation are exercised.
        static fProxyBSR BuildFullBSR(int nb, int b, uint seed)
        {
            var builder = new fProxyBSRBuilder(nb, nb, b, b, Allocator.Temp, nb * nb);
            for (int i = 0; i < nb; i++)
                builder.AddBlock(i, i, SpdBlock(b, seed + (uint)i + 1u));
            for (int i = 0; i < nb; i++)
                for (int j = 0; j < nb; j++)
                    if (j != i && ((i + j) % 3 == 0))
                        builder.AddBlock(i, j, GenerateOP.fProxyRandomMat(b, b, -0.2f, 0.2f, seed + (uint)(1000 + i * 100 + j), allocator: Allocator.Temp));
            return builder.ToBSR(Allocator.Temp);
        }

        // Dense expansion of (D/diagScale + L): block-diagonal (scaled) + strictly-block-lower,
        // zero elsewhere -- the "expanded matrix" test point (a) asks for.
        static fProxyMxN BuildLowerExpanded(in fProxyMxN dense, int nb, int b, fProxy diagScale)
        {
            var M = new fProxyMxN(dense.M_Rows, dense.N_Cols, Allocator.Temp);
            for (int bi = 0; bi < nb; bi++)
                for (int bj = 0; bj <= bi; bj++)
                    for (int r = 0; r < b; r++)
                        for (int c = 0; c < b; c++)
                        {
                            fProxy v = dense[bi * b + r, bj * b + c];
                            if (bi == bj) v /= diagScale;
                            M[bi * b + r, bj * b + c] = v;
                        }
            return M;
        }

        // Dense expansion of (D/diagScale + U): block-diagonal (scaled) + strictly-block-upper.
        static fProxyMxN BuildUpperExpanded(in fProxyMxN dense, int nb, int b, fProxy diagScale)
        {
            var M = new fProxyMxN(dense.M_Rows, dense.N_Cols, Allocator.Temp);
            for (int bi = 0; bi < nb; bi++)
                for (int bj = bi; bj < nb; bj++)
                    for (int r = 0; r < b; r++)
                        for (int c = 0; c < b; c++)
                        {
                            fProxy v = dense[bi * b + r, bj * b + c];
                            if (bi == bj) v /= diagScale;
                            M[bi * b + r, bj * b + c] = v;
                        }
            return M;
        }

        // Independent dense oracle: LU-solve M y = rhs (destructive copies, like every other
        // LU-oracle test in this repo).
        static fProxyN DenseSolve(in fProxyMxN M, in fProxyN rhs)
        {
            var LUcopy = M.Copy();
            var pivot = new Pivot(rhs.N, Allocator.Temp);
            bool ok = LU.decompInPlace(ref LUcopy, ref pivot);
            Assert.IsTrue(ok);
            var x = rhs.Copy();
            LU.decompSolve(ref LUcopy, in pivot, ref x);
            pivot.Dispose();
            return x;
        }

        // ==============================================================================
        // (a) sweepLower/sweepUpper vs a dense LU oracle on the expanded matrix.
        // ==============================================================================

        void SweepLowerVsDenseOracle()
        {
            const int nb = 5;

            for (int t = 0; t < SweepBs.Length; t++)
            {
                int b = SweepBs[t];
                var A = BuildFullBSR(nb, b, (uint)(101000 + b * 137));
                var Jacobi = new fProxyBlockJacobi(in A, Allocator.Temp);
                var dense = A.ToDense(Allocator.Temp);
                int n = A.M_Rows;
                var r = GenerateOP.fProxyRandomVec(n, -1f, 1f, (uint)(102000 + b), allocator: Allocator.Temp);

                var yGot1 = new fProxyN(n, Allocator.Temp);
                BSR.sweepLower(in A, in Jacobi, in r, ref yGot1);
                var yRef1 = DenseSolve(BuildLowerExpanded(in dense, nb, b, (fProxy)1), in r);
                AssertVecClose(in yGot1, in yRef1, Tol());

                fProxy ds = (fProxy)0.7;
                var yGot2 = new fProxyN(n, Allocator.Temp);
                BSR.sweepLower(in A, in Jacobi, ds, in r, ref yGot2);
                var yRef2 = DenseSolve(BuildLowerExpanded(in dense, nb, b, ds), in r);
                AssertVecClose(in yGot2, in yRef2, Tol());
            }
        }

        void SweepUpperVsDenseOracle()
        {
            const int nb = 5;

            for (int t = 0; t < SweepBs.Length; t++)
            {
                int b = SweepBs[t];
                var A = BuildFullBSR(nb, b, (uint)(103000 + b * 137));
                var Jacobi = new fProxyBlockJacobi(in A, Allocator.Temp);
                var dense = A.ToDense(Allocator.Temp);
                int n = A.M_Rows;
                var r = GenerateOP.fProxyRandomVec(n, -1f, 1f, (uint)(104000 + b), allocator: Allocator.Temp);

                var yGot1 = new fProxyN(n, Allocator.Temp);
                BSR.sweepUpper(in A, in Jacobi, in r, ref yGot1);
                var yRef1 = DenseSolve(BuildUpperExpanded(in dense, nb, b, (fProxy)1), in r);
                AssertVecClose(in yGot1, in yRef1, Tol());

                fProxy ds = (fProxy)0.7;
                var yGot2 = new fProxyN(n, Allocator.Temp);
                BSR.sweepUpper(in A, in Jacobi, ds, in r, ref yGot2);
                var yRef2 = DenseSolve(BuildUpperExpanded(in dense, nb, b, ds), in r);
                AssertVecClose(in yGot2, in yRef2, Tol());
            }
        }

        // ==============================================================================
        // (b) fProxySSOR is M-SPD: hand-rolled PCG loop (public primitives only) asserts
        //     <r,z> > 0 every iteration and convergence to the true solution.
        // ==============================================================================

        void SSORPositiveDefiniteAndConverges()
        {
            const int nb = 8, b = 3;
            var A = fProxyGallery.fProxyRandomSparseSPD(nb, b, (fProxy)0.3, 811001u, allocator: Allocator.Temp);
            var M = new fProxySSOR(in A, Allocator.Temp);
            var op = new fProxyBSROperator(in A);
            int n = A.M_Rows;

            var xTrue = GenerateOP.fProxyRandomVec(n, -1f, 1f, 811002u, allocator: Allocator.Temp);
            var bRhs = new fProxyN(n, Allocator.Temp);
            op.Apply(in xTrue, ref bRhs);

            var x = new fProxyN(n, Allocator.Temp);
            var r = new fProxyN(n, Allocator.Temp);
            var p = new fProxyN(n, Allocator.Temp);
            var Ap = new fProxyN(n, Allocator.Temp);
            var z = new fProxyN(n, Allocator.Temp);

            r.CopyFrom(bRhs);           // r = b - A*0
            M.Apply(in r, ref z);
            p.CopyFrom(z);
            fProxy rz = Blas.dot(r, z);
            Assert.IsTrue(rz > (fProxy)0);

            fProxy bb = Blas.dot(bRhs, bRhs);
            fProxy tol = Consts.fProxySqrtEps;
            fProxy threshold = tol * tol * bb;

            int maxIter = 6 * n;
            bool converged = false;

            for (int k = 0; k < maxIter; k++)
            {
                fProxy pAp = op.ApplyDot(in p, ref Ap);
                Assert.IsTrue(pAp > (fProxy)0);

                fProxy alpha = rz / pAp;
                fProxy rr = Blas.updateXR(alpha, p, ref x, Ap, ref r);
                if (rr <= threshold) { converged = true; break; }

                M.Apply(in r, ref z);
                fProxy rzNew = Blas.dot(r, z);
                Assert.IsTrue(rzNew > (fProxy)0);   // M-SPD sanity, every iteration

                fProxy beta = rzNew / rz;
                p.scaleAddInPlace(beta, z);
                rz = rzNew;
            }

            Assert.IsTrue(converged);
            for (int i = 0; i < n; i++) AssertClose(x[i], xTrue[i], SolveTol());
        }

        // End-to-end: the production Krylov.cg(in fProxyBSR, in fProxySSOR, ...) three-rung
        // overload matches a dense LU oracle (mirrors PcgBsrMatchesLUOracle for block-Jacobi).
        void PcgSSORMatchesLUOracle()
        {
            int dim = 12;
            var Adense = BuildDenseSPD(dim, 96001);
            var bsm = DenseToBSR1x1(in Adense, dim * dim);
            var M = new fProxySSOR(in bsm, Allocator.Temp);
            var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 96002, allocator: Allocator.Temp);

            var xLU = DenseSolve(in Adense, in b);

            var xPcg = new fProxyN(dim, Allocator.Temp);
            bool okPcg = Krylov.cg(in bsm, in M, in b, ref xPcg, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okPcg);
            AssertVecClose(in xPcg, in xLU, SolveTol());

            var Ax = BSR.spMV(in bsm, in xPcg);
            AssertVecClose(in Ax, in b, SolveTol());
        }

        // ==============================================================================
        // (c) fProxySSOR beats fProxyBlockJacobi's iteration count (>=10% margin).
        // ==============================================================================

        void SSORBeatsJacobiOnLaplacian()
        {
            var A = fProxyGallery.fProxyLaplacian2D(4, 16, allocator: Allocator.Temp);   // BR=4 (unrolled path), 64 dof, spread spectrum
            var bJ = new fProxyBlockJacobi(in A, Allocator.Temp);
            var ssor = new fProxySSOR(in A, Allocator.Temp);
            int n = A.M_Rows;

            var xTrue = GenerateOP.fProxyRandomVec(n, 0.5f, 1.5f, 821001u, allocator: Allocator.Temp);
            var b = BSR.spMV(in A, in xTrue);
            fProxy tol = Consts.fProxySqrtEps;
            int maxIter = 8 * n;

            var xJ = new fProxyN(n, Allocator.Temp);
            var infoJ = Krylov.cg(in A, in bJ, in b, ref xJ, maxIter, tol);
            Assert.IsTrue(infoJ.Solved);

            var xS = new fProxyN(n, Allocator.Temp);
            var infoS = Krylov.cg(in A, in ssor, in b, ref xS, maxIter, tol);
            Assert.IsTrue(infoS.Solved);

            Assert.IsTrue((double)infoS.iterations <= (double)infoJ.iterations * 0.9);
        }

        void SSORBeatsJacobiOnRandomSparseSPD()
        {
            var A = fProxyGallery.fProxyRandomSparseSPD(30, 3, (fProxy)0.35, 822001u, allocator: Allocator.Temp);
            var bJ = new fProxyBlockJacobi(in A, Allocator.Temp);
            var ssor = new fProxySSOR(in A, Allocator.Temp);
            int n = A.M_Rows;

            var xTrue = GenerateOP.fProxyRandomVec(n, 0.5f, 1.5f, 822002u, allocator: Allocator.Temp);
            var b = BSR.spMV(in A, in xTrue);
            fProxy tol = Consts.fProxySqrtEps;
            int maxIter = 8 * n;

            var xJ = new fProxyN(n, Allocator.Temp);
            var infoJ = Krylov.cg(in A, in bJ, in b, ref xJ, maxIter, tol);
            Assert.IsTrue(infoJ.Solved);

            var xS = new fProxyN(n, Allocator.Temp);
            var infoS = Krylov.cg(in A, in ssor, in b, ref xS, maxIter, tol);
            Assert.IsTrue(infoS.Solved);

            Assert.IsTrue((double)infoS.iterations <= (double)infoJ.iterations * 0.9);
        }

        // ==============================================================================
        // (d) SSOR over symmetric-storage BSR == SSOR over its full-storage twin (mirror path).
        // ==============================================================================

        void SSORSymmetricStorageMatchesFullStorage()
        {
            const int nb = 4, b = 3;

            var symBuilder = new fProxyBSRBuilder(nb, nb, b, b, Allocator.Temp, nb * nb);
            var fullBuilder = new fProxyBSRBuilder(nb, nb, b, b, Allocator.Temp, nb * nb);
            for (int i = 0; i < nb; i++)
            {
                var d = SpdBlock(b, (uint)(930000 + i));
                symBuilder.AddBlock(i, i, in d);
                fullBuilder.AddBlock(i, i, in d);
            }
            for (int i = 0; i < nb; i++)
                for (int j = i + 1; j < nb; j++)
                    if ((i + j) % 2 == 0)
                    {
                        var off = GenerateOP.fProxyRandomMat(b, b, -0.2f, 0.2f, (uint)(931000 + i * 10 + j), allocator: Allocator.Temp);
                        fullBuilder.AddBlock(i, j, in off);

                        var offT = new fProxyMxN(b, b, Allocator.Temp);
                        for (int rr = 0; rr < b; rr++)
                            for (int cc = 0; cc < b; cc++)
                                offT[rr, cc] = off[cc, rr];
                        fullBuilder.AddBlock(j, i, in offT);
                        symBuilder.AddBlock(j, i, in offT);   // lower triangle stored now
                    }

            var Asym = symBuilder.ToBSRSymmetric(Allocator.Temp);
            var Afull = fullBuilder.ToBSR(Allocator.Temp);

            var Msym = new fProxySSOR(in Asym, Allocator.Temp);
            var Mfull = new fProxySSOR(in Afull, Allocator.Temp);

            int n = Asym.M_Rows;
            var r = GenerateOP.fProxyRandomVec(n, -1f, 1f, 932001u, allocator: Allocator.Temp);
            var zSym = new fProxyN(n, Allocator.Temp);
            var zFull = new fProxyN(n, Allocator.Temp);
            Msym.Apply(in r, ref zSym);
            Mfull.Apply(in r, ref zFull);

            AssertVecClose(in zSym, in zFull, TightTol());
        }

        // ==============================================================================
        // (e) fProxySSOR drops into Eigen.lobpcg<TOp,TPre>'s TPre slot with no new overloads.
        // ==============================================================================

        void LobpcgAcceptsSSORPreconditioner()
        {
            var A = fProxyGallery.fProxyLaplacian2D(4, 8, allocator: Allocator.Temp);   // 32 dof
            var M = new fProxySSOR(in A, Allocator.Temp);
            var op = new fProxyBSROperator(in A);
            int n = A.M_Rows, k = 3;

            var ws = new fProxyLOBPCGCache(n, k, Allocator.Temp);
            var info = Eigen.lobpcg(in op, in M, ref ws, k, Consts.fProxySqrtEps, 500);

            Assert.IsTrue(info.Solved);
            Assert.AreEqual(k, info.converged);
        }
    }

    [Test] public void SweepLowerVsDenseOracleTest()
        => new SSORTestJob { Type = SSORTestJob.TestType.SweepLowerVsDenseOracle }.Run();
    [Test] public void SweepUpperVsDenseOracleTest()
        => new SSORTestJob { Type = SSORTestJob.TestType.SweepUpperVsDenseOracle }.Run();
    [Test] public void SSORPositiveDefiniteAndConvergesTest()
        => new SSORTestJob { Type = SSORTestJob.TestType.SSORPositiveDefiniteAndConverges }.Run();
    [Test] public void PcgSSORMatchesLUOracleTest()
        => new SSORTestJob { Type = SSORTestJob.TestType.PcgSSORMatchesLUOracle }.Run();
    [Test] public void SSORBeatsJacobiOnLaplacianTest()
        => new SSORTestJob { Type = SSORTestJob.TestType.SSORBeatsJacobiOnLaplacian }.Run();
    [Test] public void SSORBeatsJacobiOnRandomSparseSPDTest()
        => new SSORTestJob { Type = SSORTestJob.TestType.SSORBeatsJacobiOnRandomSparseSPD }.Run();
    [Test] public void SSORSymmetricStorageMatchesFullStorageTest()
        => new SSORTestJob { Type = SSORTestJob.TestType.SSORSymmetricStorageMatchesFullStorage }.Run();
    [Test] public void LobpcgAcceptsSSORPreconditionerTest()
        => new SSORTestJob { Type = SSORTestJob.TestType.LobpcgAcceptsSSORPreconditioner }.Run();

    // ---- managed-thread guard-throw tests (Burst cannot surface an assertable managed exception) ----

    [Test]
    public void SweepLowerThrowsOnSymmetricStorage()
    {
        var b = new fProxyBSRBuilder(3, 3, 2, 2, Allocator.Temp, 3);
        var d = new fProxyMxN(2, 2, Allocator.Temp);
        d[0, 0] = (fProxy)2; d[1, 1] = (fProxy)2;
        b.AddBlock(0, 0, in d); b.AddBlock(1, 1, in d); b.AddBlock(2, 2, in d);
        var A = b.ToBSRSymmetric(Allocator.Temp);
        var Jacobi = new fProxyBlockJacobi(in A, Allocator.Temp);

        var r = new fProxyN(6, Allocator.Temp);
        var y = new fProxyN(6, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => BSR.sweepLower(in A, in Jacobi, in r, ref y));
        Assert.Throws<ArgumentException>(() => BSR.sweepUpper(in A, in Jacobi, in r, ref y));
    }

    [Test]
    public void FProxySSOROmegaOutOfRangeThrows()
    {
        var b = new fProxyBSRBuilder(2, 2, 2, 2, Allocator.Temp, 2);
        var d = new fProxyMxN(2, 2, Allocator.Temp);
        d[0, 0] = (fProxy)2; d[1, 1] = (fProxy)2;
        b.AddBlock(0, 0, in d); b.AddBlock(1, 1, in d);
        var A = b.ToBSR(Allocator.Temp);

        Assert.Throws<ArgumentException>(() => { var m = new fProxySSOR(in A, (fProxy)0, Allocator.Temp); });
        Assert.Throws<ArgumentException>(() => { var m = new fProxySSOR(in A, (fProxy)2, Allocator.Temp); });
        Assert.Throws<ArgumentException>(() => { var m = new fProxySSOR(in A, (fProxy)(-1), Allocator.Temp); });
    }
}
