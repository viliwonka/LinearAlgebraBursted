using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Block (multi-RHS) IDR(s): Krylov.bidr(in A, in B, ref X) where B/X are m ROWS x n COLS (row = RHS)
// and A is a GENERAL non-symmetric square matrix. A true block Induced Dimension Reduction (one shared
// s-slot shadow space, m x m block coefficients solved via QRCP, ApplyBlock per iteration), NOT m
// scalar idr solves. `s` here is the IDR shadow-space DEPTH (unrelated to the RHS count m) -- it mirrors
// scalar Krylov.idr's own `s`. The seeded shadow space is the only randomness -> same seed = BIT-
// IDENTICAL X. No column locking/deflation, so a rank-deficient RHS block (identical rows) trips a
// defined Breakdown (X stays finite, never NaN, never a throw). Every case runs inside a
// [BurstCompile] IJob (by-value struct copy), so the job-safety criterion -- the caller sees the final
// X written through the ref fProxyMxN -- is exercised by construction.
public class fProxyBlockIDRTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct BlockIdrTestJob : IJob
    {
        public enum TestType
        {
            SolvesDenseNonsym,
            SolvesBSRNonsym,
            MatchesScalarIdrPerColumn,
            KnownSolutionRecovered,
            IdentityFoldBitIdentical,
            RankDeficientRHSBlockBreaksDownGracefully,
            DeterminismExplicitSeed,
            DeterminismDefaultSeed,
            PreconditionedILU0,
            PreconditionedBlockJacobi,
            ZeroRhs,
        }

        public TestType Type;

        // Loose cross-solver / known-solution band. Block IDR(s) has NO monotone residual bound (like
        // block BiCGSTAB/GMRES here), so the block iterate and an independent per-column scalar idr solve
        // reach slightly different iterates of the SAME unique solution -> a wide band, not a tight one.
        static fProxy Tol() => /*+choose[3e-2f|1e-5]*/3e-2f/*-choose*/;

        // Solve tolerance + true-residual convergence check (mirrors scalar IDRTests' own Tol()).
        static fProxy ResTol() => /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;

        // Dense nonsymmetric, diagonally dominant (well-conditioned, nonsingular): random entries + a
        // heavy diagonal. NOT symmetrized (off-diagonals differ across the diagonal) -- do NOT form M^T M.
        static fProxyMxN DenseNonsym(int n, uint seed)
        {
            var A = GenerateOP.fProxyRandomMat(n, n, (fProxy)(-1f), (fProxy)1f, seed, Allocator.Temp);
            for (int i = 0; i < n; i++) A[i, i] += (fProxy)(2 * n);
            return A;
        }

        // Scalar 1D convection-diffusion: diagonal 6, super -1, sub -3 -- nonsymmetric, diagonally
        // dominant. Full storage BSR.
        static fProxyBSR ConvDiff1D(int n)
        {
            var b = new fProxyBSRBuilder(n, n, 1, 1, Allocator.Temp, 3 * n);
            for (int i = 0; i < n; i++)
            {
                b.AddValue(i, i, (fProxy)6);
                if (i > 0) b.AddValue(i, i - 1, (fProxy)(-3));
                if (i < n - 1) b.AddValue(i, i + 1, (fProxy)(-1));
            }
            return b.ToBSR(Allocator.Temp);
        }

        static fProxyN Row(in fProxyMxN B, int j, int n)
        {
            var v = new fProxyN(n, Allocator.Temp);
            for (int c = 0; c < n; c++) v[c] = B[j, c];
            return v;
        }

        // Per-row relative residual ||A X[j] - B[j]|| / ||B[j]|| of a dense block solve, checked <= tol.
        static bool BlockResidualDenseOK(in fProxyMxN A, in fProxyMxN X, in fProxyMxN B, int m, int n, fProxy tol)
        {
            var AX = new fProxyMxN(m, n, Allocator.Temp);
            new fProxyDenseOperatorGeneral(in A).ApplyBlock(in X, ref AX, m);
            for (int j = 0; j < m; j++)
            {
                fProxy num = 0, den = 0;
                for (int c = 0; c < n; c++) { fProxy d = AX[j, c] - B[j, c]; num += d * d; den += B[j, c] * B[j, c]; }
                if (math.sqrt(num) / math.sqrt(math.max(den, (fProxy)1e-30)) > tol) return false;
            }
            return true;
        }

        static bool BlockResidualBSROK(in fProxyBSR A, in fProxyMxN X, in fProxyMxN B, int m, int n, fProxy tol)
        {
            var AX = new fProxyMxN(m, n, Allocator.Temp);
            new fProxyBSROperator(in A).ApplyBlock(in X, ref AX, m);
            for (int j = 0; j < m; j++)
            {
                fProxy num = 0, den = 0;
                for (int c = 0; c < n; c++) { fProxy d = AX[j, c] - B[j, c]; num += d * d; den += B[j, c] * B[j, c]; }
                if (math.sqrt(num) / math.sqrt(math.max(den, (fProxy)1e-30)) > tol) return false;
            }
            return true;
        }

        public void Execute()
        {
            switch (Type)
            {
                case TestType.SolvesDenseNonsym:                        SolvesDenseNonsym();                        break;
                case TestType.SolvesBSRNonsym:                          SolvesBSRNonsym();                          break;
                case TestType.MatchesScalarIdrPerColumn:                MatchesScalarIdrPerColumn();                break;
                case TestType.KnownSolutionRecovered:                   KnownSolutionRecovered();                   break;
                case TestType.IdentityFoldBitIdentical:                 IdentityFoldBitIdentical();                 break;
                case TestType.RankDeficientRHSBlockBreaksDownGracefully: RankDeficientRHSBlockBreaksDownGracefully(); break;
                case TestType.DeterminismExplicitSeed:                  DeterminismExplicitSeed();                  break;
                case TestType.DeterminismDefaultSeed:                   DeterminismDefaultSeed();                   break;
                case TestType.PreconditionedILU0:                       PreconditionedILU0();                       break;
                case TestType.PreconditionedBlockJacobi:                PreconditionedBlockJacobi();                break;
                case TestType.ZeroRhs:                                  ZeroRhs();                                  break;
            }
        }

        // Basic convergence of the block solve on a dense nonsymmetric square system.
        void SolvesDenseNonsym()
        {
            int n = 30, m = 4, sDepth = 4;
            var A = DenseNonsym(n, 0x1B01u);
            var B = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-1f), (fProxy)1f, 0x1B02u, Allocator.Temp);

            var X = new fProxyMxN(m, n, Allocator.Temp);                  // zero initial guess
            var info = Krylov.bidr(in A, in B, ref X, sDepth, 20 * n, ResTol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(info.Solved);
            Assert.AreEqual(m, info.converged);
            Assert.AreEqual(m, info.rhs);
            Assert.IsTrue(BlockResidualDenseOK(in A, in X, in B, m, n, ResTol()));
        }

        // Basic convergence of the block solve over a BSR nonsymmetric A.
        void SolvesBSRNonsym()
        {
            int n = 80, m = 3, sDepth = 4;
            var A = ConvDiff1D(n);
            var B = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-1f), (fProxy)1f, 0x1B12u, Allocator.Temp);

            var X = new fProxyMxN(m, n, Allocator.Temp);
            var info = Krylov.bidr(in A, in B, ref X, sDepth, 20 * n, ResTol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(info.Solved);
            Assert.AreEqual(m, info.converged);
            Assert.IsTrue(BlockResidualBSROK(in A, in X, in B, m, n, ResTol()));
        }

        // Each column of the block solution matches an independent scalar idr solve of that column, and
        // every column reached tolerance. Block IDR(s) is not monotone, so the columns agree only to the
        // wide Tol() band, but both routes converge to the same unique solution.
        void MatchesScalarIdrPerColumn()
        {
            int n = 24, m = 4, sDepth = 4;
            var A = DenseNonsym(n, 0x1B21u);
            var B = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-1f), (fProxy)1f, 0x1B22u, Allocator.Temp);

            int maxIter = 20 * n;
            fProxy tol = ResTol();

            var X = new fProxyMxN(m, n, Allocator.Temp);                  // zero initial guess
            var info = Krylov.bidr(in A, in B, ref X, sDepth, maxIter, tol);

            Assert.IsTrue(info.Solved);
            Assert.AreEqual(m, info.converged);
            Assert.AreEqual(m, info.rhs);

            for (int j = 0; j < m; j++)
            {
                var bj = Row(in B, j, n);
                var xj = new fProxyN(n, Allocator.Temp);
                for (int c = 0; c < n; c++) xj[c] = (fProxy)0;
                Assert.IsTrue(Krylov.idr(in A, in bj, ref xj, sDepth, maxIter, tol).Solved);

                // Block column j matches the scalar solve (both converged to the same unique solution).
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(math.abs((double)X[j, c] - (double)xj[c]) <= Tol() * (1.0 + math.abs((double)xj[c])));
            }
        }

        // Independent of any other solver: pick a KNOWN block solution Xk, form B = A Xk via the GENERAL
        // block apply (fProxyDenseOperatorGeneral -- fProxyDenseOperator.ApplyBlock computes A^T for a
        // non-symmetric A and would build the WRONG B here), solve, and recover Xk.
        void KnownSolutionRecovered()
        {
            int n = 24, m = 5, sDepth = 4;
            var A = DenseNonsym(n, 0x1B31u);
            var Xk = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-1f), (fProxy)1f, 0x1B32u, Allocator.Temp);   // known solution

            var B = new fProxyMxN(m, n, Allocator.Temp);
            new fProxyDenseOperatorGeneral(in A).ApplyBlock(in Xk, ref B, m);           // B[j,:] = A Xk[j,:]

            var X = new fProxyMxN(m, n, Allocator.Temp);
            var info = Krylov.bidr(in A, in B, ref X, sDepth, 20 * n, ResTol());
            Assert.IsTrue(info.Solved);

            for (int j = 0; j < m; j++)
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(math.abs((double)X[j, c] - (double)Xk[j, c]) <= Tol() * (1.0 + math.abs((double)Xk[j, c])));
        }

        // The explicit-identity-preconditioner generic core must fold to EXACTLY the unpreconditioned
        // overload -- bit-identical X, iterations, status. bidr owns its whole workspace (Allocator.Temp),
        // so no external scratch buffers are threaded in.
        void IdentityFoldBitIdentical()
        {
            int n = 20, m = 3, sDepth = 4;
            var A = DenseNonsym(n, 0x1B41u);
            var B = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-1f), (fProxy)1f, 0x1B42u, Allocator.Temp);
            int maxIter = 12 * n;
            fProxy tol = ResTol();
            uint seed = 0x1234ABCDu;

            var opA = new fProxyDenseOperatorGeneral(in A);   // readonly struct -- construct once, reuse

            // Explicit identity preconditioner via the generic core.
            var X1 = new fProxyMxN(m, n, Allocator.Temp);
            var info1 = Krylov.bidr<fProxyDenseOperatorGeneral, fProxyIdentityPreconditioner>(
                in opA, default(fProxyIdentityPreconditioner), in B, ref X1, sDepth, maxIter, tol, seed);

            // Unpreconditioned overload (folds to the identity core internally).
            var X2 = new fProxyMxN(m, n, Allocator.Temp);
            var info2 = Krylov.bidr<fProxyDenseOperatorGeneral>(
                in opA, in B, ref X2, sDepth, maxIter, tol, seed);

            Assert.AreEqual(info1.iterations, info2.iterations);
            Assert.IsTrue(info1.status == info2.status);
            // Bit-identical X (no tolerance).
            for (int i = 0; i < m; i++)
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(X1[i, c] == X2[i, c]);
        }

        // A rank-deficient RHS block (several identical rows) makes an m x m block coefficient singular
        // (there is no column locking/deflation -- the paper leaves that to future work) -> a DEFINED
        // Breakdown. The contract: X stays finite (never NaN/inf), and the solver reports Breakdown
        // rather than throwing or silently claiming Convergence.
        void RankDeficientRHSBlockBreaksDownGracefully()
        {
            int n = 16, m = 5, sDepth = 4;
            var A = DenseNonsym(n, 0x1B51u);
            var B = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-1f), (fProxy)1f, 0x1B52u, Allocator.Temp);
            // Force rows 0, 2, 4 bit-identical -> block rank <= 3; the shadow Gram blocks inherit the
            // duplicated rows -> a genuinely singular m x m block solve.
            for (int c = 0; c < n; c++) { B[2, c] = B[0, c]; B[4, c] = B[0, c]; }

            var X = new fProxyMxN(m, n, Allocator.Temp);
            var info = Krylov.bidr(in A, in B, ref X, sDepth, 8 * n, ResTol());

            // Finite, no NaN -- last committed iterate is returned, never garbage.
            for (int j = 0; j < m; j++)
                for (int c = 0; c < n; c++)
                    Assert.IsFalse(double.IsNaN((double)X[j, c]) || double.IsInfinity((double)X[j, c]));

            // Rank-deficient block coefficient is DEFINED behavior -> Breakdown (not Solved).
            Assert.IsTrue(info.status == IterativeSolveStatus.Breakdown);
            Assert.IsFalse(info.Solved);
        }

        // (3) Determinism with an explicit seed: two independent solves from the same zero initial X must
        // produce a BIT-IDENTICAL X (the seeded shadow space is the only randomness) and equal iterations.
        void DeterminismExplicitSeed()
        {
            int n = 40, m = 3, sDepth = 4;
            var A = ConvDiff1D(n);
            var B = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-1f), (fProxy)1f, 0x1B62u, Allocator.Temp);
            uint seed = 0x1234ABCDu;

            var X1 = new fProxyMxN(m, n, Allocator.Temp);
            var i1 = Krylov.bidr(in A, in B, ref X1, sDepth, 20 * n, ResTol(), seed);

            var X2 = new fProxyMxN(m, n, Allocator.Temp);
            var i2 = Krylov.bidr(in A, in B, ref X2, sDepth, 20 * n, ResTol(), seed);

            Assert.IsTrue(i1.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(i2.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(i1.iterations == i2.iterations);
            for (int i = 0; i < m; i++)
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(X1[i, c] == X2[i, c]);   // EXACT, bit-identical
        }

        // (3) Determinism with the DEFAULT seed (omitted, via the zero-arg-tail overload): two solves must
        // still produce a bit-identical X.
        void DeterminismDefaultSeed()
        {
            int n = 40, m = 3;
            var A = ConvDiff1D(n);
            var B = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-1f), (fProxy)1f, 0x1B72u, Allocator.Temp);

            var X1 = new fProxyMxN(m, n, Allocator.Temp);
            Krylov.bidr(in A, in B, ref X1);

            var X2 = new fProxyMxN(m, n, Allocator.Temp);
            Krylov.bidr(in A, in B, ref X2);

            for (int i = 0; i < m; i++)
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(X1[i, c] == X2[i, c]);   // EXACT, bit-identical
        }

        // (6) ILU0-right-preconditioned BSR block solve converges.
        void PreconditionedILU0()
        {
            int n = 120, m = 3, sDepth = 4;
            var A = ConvDiff1D(n);
            var B = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-1f), (fProxy)1f, 0x1B82u, Allocator.Temp);
            var M = new fProxyILU0(in A, Allocator.Temp);

            var X = new fProxyMxN(m, n, Allocator.Temp);
            var info = Krylov.bidr(in A, in M, in B, ref X, sDepth, 20 * n, ResTol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.AreEqual(m, info.converged);
            Assert.IsTrue(BlockResidualBSROK(in A, in X, in B, m, n, ResTol()));
        }

        // (6) BlockJacobi-right-preconditioned BSR block solve converges.
        void PreconditionedBlockJacobi()
        {
            int n = 120, m = 3, sDepth = 4;
            var A = ConvDiff1D(n);
            var B = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-1f), (fProxy)1f, 0x1B92u, Allocator.Temp);
            var M = new fProxyBlockJacobi(in A, Allocator.Temp);

            var X = new fProxyMxN(m, n, Allocator.Temp);
            var info = Krylov.bidr(in A, in M, in B, ref X, sDepth, 20 * n, ResTol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.AreEqual(m, info.converged);
            Assert.IsTrue(BlockResidualBSROK(in A, in X, in B, m, n, ResTol()));
        }

        // Edge: all-zero B (with a non-zero initial X) -> immediate Converged, iterations == 0, X reset
        // to exactly zero (bit-identical).
        void ZeroRhs()
        {
            int n = 20, m = 3, sDepth = 4;
            var A = ConvDiff1D(n);
            var B = new fProxyMxN(m, n, Allocator.Temp);
            for (int i = 0; i < m; i++)
                for (int c = 0; c < n; c++) B[i, c] = (fProxy)0;

            var X = new fProxyMxN(m, n, Allocator.Temp);
            for (int i = 0; i < m; i++)
                for (int c = 0; c < n; c++) X[i, c] = (fProxy)5;

            var info = Krylov.bidr(in A, in B, ref X, sDepth, 20 * n, ResTol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(info.iterations == 0);
            for (int i = 0; i < m; i++)
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(X[i, c] == (fProxy)0);
        }
    }

    [Test] public void SolvesDenseNonsymTest() => new BlockIdrTestJob { Type = BlockIdrTestJob.TestType.SolvesDenseNonsym }.Run();
    [Test] public void SolvesBSRNonsymTest() => new BlockIdrTestJob { Type = BlockIdrTestJob.TestType.SolvesBSRNonsym }.Run();
    [Test] public void MatchesScalarIdrPerColumnTest() => new BlockIdrTestJob { Type = BlockIdrTestJob.TestType.MatchesScalarIdrPerColumn }.Run();
    [Test] public void KnownSolutionRecoveredTest() => new BlockIdrTestJob { Type = BlockIdrTestJob.TestType.KnownSolutionRecovered }.Run();
    [Test] public void IdentityFoldBitIdenticalTest() => new BlockIdrTestJob { Type = BlockIdrTestJob.TestType.IdentityFoldBitIdentical }.Run();
    [Test] public void RankDeficientRHSBlockBreaksDownGracefullyTest() => new BlockIdrTestJob { Type = BlockIdrTestJob.TestType.RankDeficientRHSBlockBreaksDownGracefully }.Run();
    [Test] public void DeterminismExplicitSeedTest() => new BlockIdrTestJob { Type = BlockIdrTestJob.TestType.DeterminismExplicitSeed }.Run();
    [Test] public void DeterminismDefaultSeedTest() => new BlockIdrTestJob { Type = BlockIdrTestJob.TestType.DeterminismDefaultSeed }.Run();
    [Test] public void PreconditionedILU0Test() => new BlockIdrTestJob { Type = BlockIdrTestJob.TestType.PreconditionedILU0 }.Run();
    [Test] public void PreconditionedBlockJacobiTest() => new BlockIdrTestJob { Type = BlockIdrTestJob.TestType.PreconditionedBlockJacobi }.Run();
    [Test] public void ZeroRhsTest() => new BlockIdrTestJob { Type = BlockIdrTestJob.TestType.ZeroRhs }.Run();
}
