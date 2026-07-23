using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Pseudo-block (multi-RHS) TFQMR: Krylov.btfqmr(in A, in B, ref X) where B/X are s ROWS x n COLS
// (row = RHS) and A is a GENERAL non-symmetric square matrix. NOT a subspace-mixing true block
// method -- s independent scalar-TFQMR recurrences advanced in lockstep, sharing one ApplyBlock call
// per half-step instead of s separate Apply calls (see OP/DEVLOG.md "Krylov.Block.TFQMR" for why a
// true mixing block TFQMR is ill-defined). Because rows never mix, a duplicate RHS row cannot
// singularize any shared coefficient (there is none) -- unlike bbiCGStab/bidr, breakdown here is the
// SAME per-row contract as scalar tfqmr's own (honest Breakdown status, X finite, never NaN, never a
// throw). Every test runs inside a [BurstCompile] IJob (by-value struct copy), so the job-safety
// criterion -- the caller sees the final X written through the ref fProxyMxN -- is exercised by
// construction.
public class fProxyBlockTFQMRTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct BlockTfqmrTestJob : IJob
    {
        public enum TestType
        {
            SolvesDenseNonsym,
            SolvesBSRNonsym,
            MatchesScalarTfqmrPerColumn,
            KnownSolutionRecovered,
            IdentityFoldBitIdentical,
            DuplicateRHSRowsBitIdentical,
            MaxIterBudgetHonestStatus,
            PreconditionedILU0,
            ZeroRhs,
        }

        public TestType Type;

        // Per-column vs. independent-scalar-solve band: both routes converge to the SAME unique
        // solution (A nonsingular), but AU is computed via ApplyBlock in the block path vs. Apply in
        // the scalar path -- not guaranteed bit-identical accumulation order -- so a loose band, not a
        // tight one (mirrors bbiCGStab/bidr's own choice for the identical reason).
        static fProxy Tol() => /*+choose[3e-2f|1e-5]*/3e-2f/*-choose*/;

        static fProxy ResTol() => /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;

        // Diagonally dominant (so nonsingular) but NOT symmetrized -> genuinely non-symmetric. Do NOT
        // form M^T M (that would defeat the point of a transpose-free-nonsymmetric test).
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

        static bool BlockResidualDenseOK(in fProxyMxN A, in fProxyMxN X, in fProxyMxN B, int s, int n, fProxy tol)
        {
            var AX = new fProxyMxN(s, n, Allocator.Temp);
            new fProxyDenseOperatorGeneral(in A).ApplyBlock(in X, ref AX, s);
            for (int j = 0; j < s; j++)
            {
                fProxy num = 0, den = 0;
                for (int c = 0; c < n; c++) { fProxy d = AX[j, c] - B[j, c]; num += d * d; den += B[j, c] * B[j, c]; }
                if (math.sqrt(num) / math.sqrt(math.max(den, (fProxy)1e-30)) > tol) return false;
            }
            return true;
        }

        static bool BlockResidualBSROK(in fProxyBSR A, in fProxyMxN X, in fProxyMxN B, int s, int n, fProxy tol)
        {
            var AX = new fProxyMxN(s, n, Allocator.Temp);
            new fProxyBSROperator(in A).ApplyBlock(in X, ref AX, s);
            for (int j = 0; j < s; j++)
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
                case TestType.SolvesDenseNonsym:              SolvesDenseNonsym();              break;
                case TestType.SolvesBSRNonsym:                SolvesBSRNonsym();                 break;
                case TestType.MatchesScalarTfqmrPerColumn:    MatchesScalarTfqmrPerColumn();     break;
                case TestType.KnownSolutionRecovered:         KnownSolutionRecovered();          break;
                case TestType.IdentityFoldBitIdentical:       IdentityFoldBitIdentical();        break;
                case TestType.DuplicateRHSRowsBitIdentical:   DuplicateRHSRowsBitIdentical();    break;
                case TestType.MaxIterBudgetHonestStatus:      MaxIterBudgetHonestStatus();       break;
                case TestType.PreconditionedILU0:             PreconditionedILU0();              break;
                case TestType.ZeroRhs:                        ZeroRhs();                         break;
            }
        }

        // Basic convergence of the block solve on a dense nonsymmetric square system.
        void SolvesDenseNonsym()
        {
            int n = 30, s = 4;
            var A = DenseNonsym(n, 0x1F01u);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 0x1F02u, Allocator.Temp);

            var X = new fProxyMxN(s, n, Allocator.Temp);                   // zero initial guess
            var info = Krylov.btfqmr(in A, in B, ref X, 40 * n, ResTol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(info.Solved);
            Assert.AreEqual(s, info.converged);
            Assert.AreEqual(s, info.rhs);
            Assert.IsTrue(BlockResidualDenseOK(in A, in X, in B, s, n, ResTol()));
        }

        // Basic convergence of the block solve over a BSR nonsymmetric A.
        void SolvesBSRNonsym()
        {
            int n = 80, s = 3;
            var A = ConvDiff1D(n);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 0x1F12u, Allocator.Temp);

            var X = new fProxyMxN(s, n, Allocator.Temp);
            var info = Krylov.btfqmr(in A, in B, ref X, 40 * n, ResTol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(info.Solved);
            Assert.AreEqual(s, info.converged);
            Assert.IsTrue(BlockResidualBSROK(in A, in X, in B, s, n, ResTol()));
        }

        // Each column of the block solution matches an independent scalar tfqmr solve of that column,
        // and every column reached tolerance.
        void MatchesScalarTfqmrPerColumn()
        {
            int n = 24, s = 4;
            var A = DenseNonsym(n, 0x1F21u);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 0x1F22u, Allocator.Temp);

            int maxIter = 40 * n;
            fProxy tol = ResTol();

            var X = new fProxyMxN(s, n, Allocator.Temp);                   // zero initial guess
            var info = Krylov.btfqmr(in A, in B, ref X, maxIter, tol);

            Assert.IsTrue(info.Solved);
            Assert.AreEqual(s, info.converged);
            Assert.AreEqual(s, info.rhs);

            for (int j = 0; j < s; j++)
            {
                var bj = Row(in B, j, n);
                var xj = new fProxyN(n, Allocator.Temp);
                for (int c = 0; c < n; c++) xj[c] = (fProxy)0;
                Assert.IsTrue(Krylov.tfqmr(in A, in bj, ref xj, maxIter, tol).Solved);

                for (int c = 0; c < n; c++)
                    Assert.IsTrue(math.abs((double)X[j, c] - (double)xj[c]) <= Tol() * (1.0 + math.abs((double)xj[c])));
            }
        }

        // Independent of any other solver: pick a KNOWN block solution Xk, form B = A Xk via the GENERAL
        // block apply (fProxyDenseOperatorGeneral -- fProxyDenseOperator.ApplyBlock computes A^T for a
        // non-symmetric A and would build the WRONG B here), solve, and recover Xk.
        void KnownSolutionRecovered()
        {
            int n = 24, s = 5;
            var A = DenseNonsym(n, 0x1F31u);
            var Xk = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 0x1F32u, Allocator.Temp);   // known solution

            var B = new fProxyMxN(s, n, Allocator.Temp);
            new fProxyDenseOperatorGeneral(in A).ApplyBlock(in Xk, ref B, s);           // B[j,:] = A Xk[j,:]

            var X = new fProxyMxN(s, n, Allocator.Temp);
            var info = Krylov.btfqmr(in A, in B, ref X, 40 * n, ResTol());
            Assert.IsTrue(info.Solved);

            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(math.abs((double)X[j, c] - (double)Xk[j, c]) <= Tol() * (1.0 + math.abs((double)Xk[j, c])));
        }

        // The explicit-identity-preconditioner generic core must fold to EXACTLY the unpreconditioned
        // overload -- bit-identical X, iterations, status.
        void IdentityFoldBitIdentical()
        {
            int n = 20, s = 3;
            var A = DenseNonsym(n, 0x1F41u);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 0x1F42u, Allocator.Temp);
            int maxIter = 30 * n;
            fProxy tol = ResTol();

            var opA = new fProxyDenseOperatorGeneral(in A);   // readonly struct -- construct once, reuse

            var X1     = new fProxyMxN(s, n, Allocator.Temp);
            var Rhat01 = new fProxyMxN(s, n, Allocator.Temp);
            var U1     = new fProxyMxN(s, n, Allocator.Temp);
            var W1     = new fProxyMxN(s, n, Allocator.Temp);
            var V1     = new fProxyMxN(s, n, Allocator.Temp);
            var AU1    = new fProxyMxN(s, n, Allocator.Temp);
            var D1     = new fProxyMxN(s, n, Allocator.Temp);
            var UHat1  = new fProxyMxN(s, n, Allocator.Temp);
            var info1 = Krylov.btfqmr<fProxyDenseOperatorGeneral, fProxyIdentityPreconditioner>(
                in opA, default(fProxyIdentityPreconditioner), in B, ref X1,
                ref Rhat01, ref U1, ref W1, ref V1, ref AU1, ref D1, ref UHat1, maxIter, tol);

            var X2     = new fProxyMxN(s, n, Allocator.Temp);
            var Rhat02 = new fProxyMxN(s, n, Allocator.Temp);
            var U2     = new fProxyMxN(s, n, Allocator.Temp);
            var W2     = new fProxyMxN(s, n, Allocator.Temp);
            var V2     = new fProxyMxN(s, n, Allocator.Temp);
            var AU2    = new fProxyMxN(s, n, Allocator.Temp);
            var D2     = new fProxyMxN(s, n, Allocator.Temp);
            var info2 = Krylov.btfqmr<fProxyDenseOperatorGeneral>(
                in opA, in B, ref X2, ref Rhat02, ref U2, ref W2, ref V2, ref AU2, ref D2, maxIter, tol);

            Assert.AreEqual(info1.iterations, info2.iterations);
            Assert.IsTrue(info1.status == info2.status);
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(X1[i, c] == X2[i, c]);
        }

        // Rows never mix (no shared s x s coefficient -- see OP/DEVLOG.md "Krylov.Block.TFQMR"), so two
        // bit-identical RHS rows must run the identical per-row scalar recurrence and land on
        // bit-identical output rows -- the opposite contract from bbiCGStab/bidr's true block mixing,
        // where a duplicate row singularizes the shared coefficient and trips Breakdown instead.
        void DuplicateRHSRowsBitIdentical()
        {
            int n = 16, s = 5;
            var A = DenseNonsym(n, 0x1F51u);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 0x1F52u, Allocator.Temp);
            for (int c = 0; c < n; c++) { B[2, c] = B[0, c]; B[4, c] = B[0, c]; }   // rows 0,2,4 identical

            var X = new fProxyMxN(s, n, Allocator.Temp);
            var info = Krylov.btfqmr(in A, in B, ref X, 40 * n, ResTol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            for (int c = 0; c < n; c++)
            {
                Assert.IsTrue(X[0, c] == X[2, c]);
                Assert.IsTrue(X[0, c] == X[4, c]);
            }
        }

        // A tiny iteration budget on a system that genuinely needs more must report an HONEST
        // MaxIterations status (not a false Converged), with fewer than all rows converged and X still
        // finite -- no throw.
        void MaxIterBudgetHonestStatus()
        {
            int n = 32, s = 4;
            var A = DenseNonsym(n, 0x1F61u);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 0x1F62u, Allocator.Temp);

            var X = new fProxyMxN(s, n, Allocator.Temp);
            var info = Krylov.btfqmr(in A, in B, ref X, 1, Consts.fProxySqrtEps);   // deliberately tiny budget

            Assert.IsTrue(info.status == IterativeSolveStatus.MaxIterations);
            Assert.IsFalse(info.Solved);
            Assert.IsTrue(info.converged < s);
            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.IsFalse(double.IsNaN((double)X[j, c]) || double.IsInfinity((double)X[j, c]));
        }

        // ILU0-right-preconditioned BSR block solve converges.
        void PreconditionedILU0()
        {
            int n = 120, s = 3;
            var A = ConvDiff1D(n);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 0x1F82u, Allocator.Temp);
            var M = new fProxyILU0(in A, Allocator.Temp);

            var X = new fProxyMxN(s, n, Allocator.Temp);
            var info = Krylov.btfqmr(in A, in M, in B, ref X, 40 * n, ResTol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.AreEqual(s, info.converged);
            Assert.IsTrue(BlockResidualBSROK(in A, in X, in B, s, n, ResTol()));
        }

        // Edge: all-zero B (with a non-zero initial X) -> immediate Converged, iterations == 0, X reset
        // to exactly zero (bit-identical).
        void ZeroRhs()
        {
            int n = 20, s = 3;
            var A = ConvDiff1D(n);
            var B = new fProxyMxN(s, n, Allocator.Temp);
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++) B[i, c] = (fProxy)0;

            var X = new fProxyMxN(s, n, Allocator.Temp);
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++) X[i, c] = (fProxy)5;

            var info = Krylov.btfqmr(in A, in B, ref X, 20 * n, ResTol());

            Assert.IsTrue(info.status == IterativeSolveStatus.Converged);
            Assert.IsTrue(info.iterations == 0);
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(X[i, c] == (fProxy)0);
        }

    }

    [Test] public void SolvesDenseNonsymTest() => new BlockTfqmrTestJob { Type = BlockTfqmrTestJob.TestType.SolvesDenseNonsym }.Run();
    [Test] public void SolvesBSRNonsymTest() => new BlockTfqmrTestJob { Type = BlockTfqmrTestJob.TestType.SolvesBSRNonsym }.Run();
    [Test] public void MatchesScalarTfqmrPerColumnTest() => new BlockTfqmrTestJob { Type = BlockTfqmrTestJob.TestType.MatchesScalarTfqmrPerColumn }.Run();
    [Test] public void KnownSolutionRecoveredTest() => new BlockTfqmrTestJob { Type = BlockTfqmrTestJob.TestType.KnownSolutionRecovered }.Run();
    [Test] public void IdentityFoldBitIdenticalTest() => new BlockTfqmrTestJob { Type = BlockTfqmrTestJob.TestType.IdentityFoldBitIdentical }.Run();
    [Test] public void DuplicateRHSRowsBitIdenticalTest() => new BlockTfqmrTestJob { Type = BlockTfqmrTestJob.TestType.DuplicateRHSRowsBitIdentical }.Run();
    [Test] public void MaxIterBudgetHonestStatusTest() => new BlockTfqmrTestJob { Type = BlockTfqmrTestJob.TestType.MaxIterBudgetHonestStatus }.Run();
    [Test] public void PreconditionedILU0Test() => new BlockTfqmrTestJob { Type = BlockTfqmrTestJob.TestType.PreconditionedILU0 }.Run();
    [Test] public void ZeroRhsTest() => new BlockTfqmrTestJob { Type = BlockTfqmrTestJob.TestType.ZeroRhs }.Run();
}
