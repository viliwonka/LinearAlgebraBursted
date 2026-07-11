using System;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using LinearAlgebra.Gallery;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Krylov Round-3 new surfaces:
//   (a) BSR.sweepLower/sweepUpper vs a dense LU solve on the EXPANDED (block-diagonal +
//       strictly-block-triangular, diagonal pre-divided by diagScale) matrix -- b in
//       {1,2,3,4,6} (unrolled) plus b=5 (general runtime-BR fallback), both diagScale=1 (plain
//       Gauss-Seidel) and a nontrivial diagScale (the parameter floatSSOR actually drives).
//   (b) floatSSOR is M-SPD: a hand-rolled PCG loop (built from the same public primitives
//       Krylov.pcg itself uses -- M.Apply/op.ApplyDot/Blas.dot/Blas.updateXR) asserts <r,z> > 0
//       every iteration and that the solve converges to the true solution -- no new production
//       API added just to expose this; the test reads what is already public.
//   (c) floatSSOR converges in FEWER iterations than floatBlockJacobi (>=10% margin) on both
//       floatLaplacian2D and floatRandomSparseSPD instances.
//   (d) SSOR built from a Symmetric-storage BSR equals SSOR built from its full-storage twin
//       (the one-time mirror path, Arena.floatBSRMirrorToFull).
//   (e) floatSSOR drops into Eigen.lobpcg<TOp,TPre>'s TPre slot with no new overloads.
//
// Value cases run inside a [BurstCompile] IJob (matches every other sparse suite). Guard-throw
// cases (symmetric-storage sweep input, omega out of range) are managed [Test]s with Assert.Throws
// (Burst cannot surface an assertable managed exception).
public class floatSSORTests
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

        static float Tol() => 1e-3f;
        static float SolveTol() => 1e-3f;
        static float TightTol() => 1e-4f;

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

        static void AssertClose(float got, float expected, float tol)
            => Assert.IsTrue(math.abs(got - expected) <= tol * ((float)1 + math.abs(expected)));

        static void AssertVecClose(in floatN got, in floatN expected, float tol)
        {
            Assert.AreEqual(expected.N, got.N);
            for (int i = 0; i < got.N; i++) AssertClose(got[i], expected[i], tol);
        }

        // SPD b x b block D = M^T M + b*I: well-conditioned, LU-invertible.
        static floatMxN SpdBlock(ref Arena arena, int b, uint seed)
        {
            var M = arena.floatRandomMat(b, b, -1f, 1f, seed);
            var D = Blas.dot(M, M, true);
            for (int d = 0; d < b; d++) D[d, d] += (float)b;
            return D;
        }

        static floatMxN BuildDenseSPD(ref Arena arena, int dim, uint seed)
        {
            var M = arena.floatRandomMat(dim, dim, -1f, 1f, seed);
            var A = Blas.dot(M, M, true);
            for (int d = 0; d < dim; d++) A[d, d] += dim;
            return A;
        }

        static floatBSR DenseToBSR1x1(ref Arena arena, in floatMxN A, int nnzHint)
        {
            var builder = arena.floatBSRBuilder(A.M_Rows, A.N_Cols, 1, 1, math.max(nnzHint, 1));
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    if (A[r, c] != (float)0) builder.AddValue(r, c, A[r, c]);
            return builder.ToBSR(ref arena);
        }

        // Full-storage BSR, invertible (SPD) diagonal blocks + a deterministic scatter of small
        // off-diagonal blocks on BOTH sides of the block diagonal (several per row), so
        // sweepLower/sweepUpper's early break/continue and multi-block accumulation are exercised.
        static floatBSR BuildFullBSR(ref Arena arena, int nb, int b, uint seed)
        {
            var builder = arena.floatBSRBuilder(nb, nb, b, b, nb * nb);
            for (int i = 0; i < nb; i++)
                builder.AddBlock(i, i, SpdBlock(ref arena, b, seed + (uint)i + 1u));
            for (int i = 0; i < nb; i++)
                for (int j = 0; j < nb; j++)
                    if (j != i && ((i + j) % 3 == 0))
                        builder.AddBlock(i, j, arena.floatRandomMat(b, b, -0.2f, 0.2f, seed + (uint)(1000 + i * 100 + j)));
            return builder.ToBSR(ref arena);
        }

        // Dense expansion of (D/diagScale + L): block-diagonal (scaled) + strictly-block-lower,
        // zero elsewhere -- the "expanded matrix" test point (a) asks for.
        static floatMxN BuildLowerExpanded(ref Arena arena, in floatMxN dense, int nb, int b, float diagScale)
        {
            var M = arena.floatMat(dense.M_Rows, dense.N_Cols);
            for (int bi = 0; bi < nb; bi++)
                for (int bj = 0; bj <= bi; bj++)
                    for (int r = 0; r < b; r++)
                        for (int c = 0; c < b; c++)
                        {
                            float v = dense[bi * b + r, bj * b + c];
                            if (bi == bj) v /= diagScale;
                            M[bi * b + r, bj * b + c] = v;
                        }
            return M;
        }

        // Dense expansion of (D/diagScale + U): block-diagonal (scaled) + strictly-block-upper.
        static floatMxN BuildUpperExpanded(ref Arena arena, in floatMxN dense, int nb, int b, float diagScale)
        {
            var M = arena.floatMat(dense.M_Rows, dense.N_Cols);
            for (int bi = 0; bi < nb; bi++)
                for (int bj = bi; bj < nb; bj++)
                    for (int r = 0; r < b; r++)
                        for (int c = 0; c < b; c++)
                        {
                            float v = dense[bi * b + r, bj * b + c];
                            if (bi == bj) v /= diagScale;
                            M[bi * b + r, bj * b + c] = v;
                        }
            return M;
        }

        // Independent dense oracle: LU-solve M y = rhs (destructive copies, like every other
        // LU-oracle test in this repo).
        static floatN DenseSolve(in floatMxN M, in floatN rhs)
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
            var arena = new Arena(Allocator.Persistent);
            const int nb = 5;

            for (int t = 0; t < SweepBs.Length; t++)
            {
                int b = SweepBs[t];
                var A = BuildFullBSR(ref arena, nb, b, (uint)(101000 + b * 137));
                var Jacobi = arena.floatBlockJacobi(in A);
                var dense = A.ToDense(ref arena);
                int n = A.M_Rows;
                var r = arena.floatRandomVec(n, -1f, 1f, (uint)(102000 + b));

                var yGot1 = arena.floatVec(n);
                BSR.sweepLower(in A, in Jacobi, in r, ref yGot1);
                var yRef1 = DenseSolve(BuildLowerExpanded(ref arena, in dense, nb, b, (float)1), in r);
                AssertVecClose(in yGot1, in yRef1, Tol());

                float ds = (float)0.7;
                var yGot2 = arena.floatVec(n);
                BSR.sweepLower(in A, in Jacobi, ds, in r, ref yGot2);
                var yRef2 = DenseSolve(BuildLowerExpanded(ref arena, in dense, nb, b, ds), in r);
                AssertVecClose(in yGot2, in yRef2, Tol());
            }

            arena.Dispose();
        }

        void SweepUpperVsDenseOracle()
        {
            var arena = new Arena(Allocator.Persistent);
            const int nb = 5;

            for (int t = 0; t < SweepBs.Length; t++)
            {
                int b = SweepBs[t];
                var A = BuildFullBSR(ref arena, nb, b, (uint)(103000 + b * 137));
                var Jacobi = arena.floatBlockJacobi(in A);
                var dense = A.ToDense(ref arena);
                int n = A.M_Rows;
                var r = arena.floatRandomVec(n, -1f, 1f, (uint)(104000 + b));

                var yGot1 = arena.floatVec(n);
                BSR.sweepUpper(in A, in Jacobi, in r, ref yGot1);
                var yRef1 = DenseSolve(BuildUpperExpanded(ref arena, in dense, nb, b, (float)1), in r);
                AssertVecClose(in yGot1, in yRef1, Tol());

                float ds = (float)0.7;
                var yGot2 = arena.floatVec(n);
                BSR.sweepUpper(in A, in Jacobi, ds, in r, ref yGot2);
                var yRef2 = DenseSolve(BuildUpperExpanded(ref arena, in dense, nb, b, ds), in r);
                AssertVecClose(in yGot2, in yRef2, Tol());
            }

            arena.Dispose();
        }

        // ==============================================================================
        // (b) floatSSOR is M-SPD: hand-rolled PCG loop (public primitives only) asserts
        //     <r,z> > 0 every iteration and convergence to the true solution.
        // ==============================================================================

        void SSORPositiveDefiniteAndConverges()
        {
            var arena = new Arena(Allocator.Persistent);
            const int nb = 8, b = 3;
            var A = arena.floatRandomSparseSPD(nb, b, (float)0.3, 811001u);
            var M = arena.floatSSOR(in A);
            var op = new floatBSROperator(in A);
            int n = A.M_Rows;

            var xTrue = arena.floatRandomVec(n, -1f, 1f, 811002u);
            var bRhs = arena.floatVec(n);
            op.Apply(in xTrue, ref bRhs);

            var x = arena.floatVec(n);
            var r = arena.floatVec(n);
            var p = arena.floatVec(n);
            var Ap = arena.floatVec(n);
            var z = arena.floatVec(n);

            r.CopyFrom(bRhs);           // r = b - A*0
            M.Apply(in r, ref z);
            p.CopyFrom(z);
            float rz = Blas.dot(r, z);
            Assert.IsTrue(rz > (float)0);

            float bb = Blas.dot(bRhs, bRhs);
            float tol = Consts.floatSqrtEps;
            float threshold = tol * tol * bb;

            int maxIter = 6 * n;
            bool converged = false;

            for (int k = 0; k < maxIter; k++)
            {
                float pAp = op.ApplyDot(in p, ref Ap);
                Assert.IsTrue(pAp > (float)0);

                float alpha = rz / pAp;
                float rr = Blas.updateXR(alpha, p, ref x, Ap, ref r);
                if (rr <= threshold) { converged = true; break; }

                M.Apply(in r, ref z);
                float rzNew = Blas.dot(r, z);
                Assert.IsTrue(rzNew > (float)0);   // M-SPD sanity, every iteration

                float beta = rzNew / rz;
                p.scaleAddInPlace(beta, z);
                rz = rzNew;
            }

            Assert.IsTrue(converged);
            for (int i = 0; i < n; i++) AssertClose(x[i], xTrue[i], SolveTol());

            arena.Dispose();
        }

        // End-to-end: the production Krylov.pcg(in floatBSR, in floatSSOR, ...) three-rung
        // overload matches a dense LU oracle (mirrors PcgBsrMatchesLUOracle for block-Jacobi).
        void PcgSSORMatchesLUOracle()
        {
            var arena = new Arena(Allocator.Persistent);
            int dim = 12;
            var Adense = BuildDenseSPD(ref arena, dim, 96001);
            var bsm = DenseToBSR1x1(ref arena, in Adense, dim * dim);
            var M = arena.floatSSOR(in bsm);
            var b = arena.floatRandomVec(dim, -1f, 1f, 96002);

            var xLU = DenseSolve(in Adense, in b);

            var xPcg = arena.floatVec(dim);
            bool okPcg = Krylov.pcg(in bsm, in M, in b, ref xPcg, 4 * dim, Consts.floatSqrtEps);
            Assert.IsTrue(okPcg);
            AssertVecClose(in xPcg, in xLU, SolveTol());

            var Ax = BSR.spMV(in bsm, in xPcg);
            AssertVecClose(in Ax, in b, SolveTol());

            arena.Dispose();
        }

        // ==============================================================================
        // (c) floatSSOR beats floatBlockJacobi's iteration count (>=10% margin).
        // ==============================================================================

        void SSORBeatsJacobiOnLaplacian()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatLaplacian2D(4, 16);   // BR=4 (unrolled path), 64 dof, spread spectrum
            var bJ = arena.floatBlockJacobi(in A);
            var ssor = arena.floatSSOR(in A);
            int n = A.M_Rows;

            var xTrue = arena.floatRandomVec(n, 0.5f, 1.5f, 821001u);
            var b = BSR.spMV(in A, in xTrue);
            float tol = Consts.floatSqrtEps;
            int maxIter = 8 * n;

            var xJ = arena.floatVec(n);
            var infoJ = Krylov.pcg(in A, in bJ, in b, ref xJ, maxIter, tol);
            Assert.IsTrue(infoJ.Solved);

            var xS = arena.floatVec(n);
            var infoS = Krylov.pcg(in A, in ssor, in b, ref xS, maxIter, tol);
            Assert.IsTrue(infoS.Solved);

            Assert.IsTrue((double)infoS.iterations <= (double)infoJ.iterations * 0.9);

            arena.Dispose();
        }

        void SSORBeatsJacobiOnRandomSparseSPD()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatRandomSparseSPD(30, 3, (float)0.35, 822001u);
            var bJ = arena.floatBlockJacobi(in A);
            var ssor = arena.floatSSOR(in A);
            int n = A.M_Rows;

            var xTrue = arena.floatRandomVec(n, 0.5f, 1.5f, 822002u);
            var b = BSR.spMV(in A, in xTrue);
            float tol = Consts.floatSqrtEps;
            int maxIter = 8 * n;

            var xJ = arena.floatVec(n);
            var infoJ = Krylov.pcg(in A, in bJ, in b, ref xJ, maxIter, tol);
            Assert.IsTrue(infoJ.Solved);

            var xS = arena.floatVec(n);
            var infoS = Krylov.pcg(in A, in ssor, in b, ref xS, maxIter, tol);
            Assert.IsTrue(infoS.Solved);

            Assert.IsTrue((double)infoS.iterations <= (double)infoJ.iterations * 0.9);

            arena.Dispose();
        }

        // ==============================================================================
        // (d) SSOR over symmetric-storage BSR == SSOR over its full-storage twin (mirror path).
        // ==============================================================================

        void SSORSymmetricStorageMatchesFullStorage()
        {
            var arena = new Arena(Allocator.Persistent);
            const int nb = 4, b = 3;

            var symBuilder = arena.floatBSRBuilder(nb, nb, b, b, nb * nb);
            var fullBuilder = arena.floatBSRBuilder(nb, nb, b, b, nb * nb);
            for (int i = 0; i < nb; i++)
            {
                var d = SpdBlock(ref arena, b, (uint)(930000 + i));
                symBuilder.AddBlock(i, i, in d);
                fullBuilder.AddBlock(i, i, in d);
            }
            for (int i = 0; i < nb; i++)
                for (int j = i + 1; j < nb; j++)
                    if ((i + j) % 2 == 0)
                    {
                        var off = arena.floatRandomMat(b, b, -0.2f, 0.2f, (uint)(931000 + i * 10 + j));
                        symBuilder.AddBlock(i, j, in off);
                        fullBuilder.AddBlock(i, j, in off);

                        var offT = arena.floatMat(b, b);
                        for (int rr = 0; rr < b; rr++)
                            for (int cc = 0; cc < b; cc++)
                                offT[rr, cc] = off[cc, rr];
                        fullBuilder.AddBlock(j, i, in offT);
                    }

            var Asym = symBuilder.ToBSRSymmetric(ref arena);
            var Afull = fullBuilder.ToBSR(ref arena);

            var Msym = arena.floatSSOR(in Asym);
            var Mfull = arena.floatSSOR(in Afull);

            int n = Asym.M_Rows;
            var r = arena.floatRandomVec(n, -1f, 1f, 932001u);
            var zSym = arena.floatVec(n);
            var zFull = arena.floatVec(n);
            Msym.Apply(in r, ref zSym);
            Mfull.Apply(in r, ref zFull);

            AssertVecClose(in zSym, in zFull, TightTol());

            arena.Dispose();
        }

        // ==============================================================================
        // (e) floatSSOR drops into Eigen.lobpcg<TOp,TPre>'s TPre slot with no new overloads.
        // ==============================================================================

        void LobpcgAcceptsSSORPreconditioner()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.floatLaplacian2D(4, 8);   // 32 dof
            var M = arena.floatSSOR(in A);
            var op = new floatBSROperator(in A);
            int n = A.M_Rows, k = 3;

            var ws = arena.floatLOBPCGCache(n, k);
            var info = Eigen.lobpcg(in op, in M, ref ws, k, Consts.floatSqrtEps, 500);

            Assert.IsTrue(info.Solved);
            Assert.AreEqual(k, info.converged);

            arena.Dispose();
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
        var arena = new Arena(Allocator.Persistent);
        var b = arena.floatBSRBuilder(3, 3, 2, 2, 3);
        var d = arena.floatMat(2, 2);
        d[0, 0] = (float)2; d[1, 1] = (float)2;
        b.AddBlock(0, 0, in d); b.AddBlock(1, 1, in d); b.AddBlock(2, 2, in d);
        var A = b.ToBSRSymmetric(ref arena);
        var Jacobi = arena.floatBlockJacobi(in A);

        var r = arena.floatVec(6);
        var y = arena.floatVec(6);
        Assert.Throws<ArgumentException>(() => BSR.sweepLower(in A, in Jacobi, in r, ref y));
        Assert.Throws<ArgumentException>(() => BSR.sweepUpper(in A, in Jacobi, in r, ref y));

        arena.Dispose();
    }

    [Test]
    public void FloatSSOROmegaOutOfRangeThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var b = arena.floatBSRBuilder(2, 2, 2, 2, 2);
        var d = arena.floatMat(2, 2);
        d[0, 0] = (float)2; d[1, 1] = (float)2;
        b.AddBlock(0, 0, in d); b.AddBlock(1, 1, in d);
        var A = b.ToBSR(ref arena);

        Assert.Throws<ArgumentException>(() => { var m = arena.floatSSOR(in A, (float)0); });
        Assert.Throws<ArgumentException>(() => { var m = arena.floatSSOR(in A, (float)2); });
        Assert.Throws<ArgumentException>(() => { var m = arena.floatSSOR(in A, (float)(-1)); });

        arena.Dispose();
    }
}
