using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Block (multi-RHS) BiCGSTAB: Krylov.bbiCGStab(in A, in B, ref X) where B/X are s ROWS x n COLS
// (row = RHS). Unlike bcg (SPD-only), A here is a GENERAL non-symmetric square matrix. A true block
// method (one shared subspace, s x s coefficients solved via QRCP, ApplyBlock per iteration), NOT s
// scalar solves. Breakdown (singular s x s block coefficient, or non-positive/NaN omega denominator)
// reports IterativeSolveStatus.Breakdown with X holding the last committed iterate -- finite, never
// NaN, never a throw. Every test runs inside a [BurstCompile] IJob (by-value struct copy), so the
// job-safety criterion -- the caller sees the final X written through the ref fProxyMxN -- is
// exercised by construction.
public class fProxyBlockBiCGStabTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct BlockBiCGStabTestJob : IJob
    {
        public enum TestType
        {
            MatchesScalarBiCGStabPerColumn,
            KnownSolutionRecovered,
            IdentityFoldBitIdentical,
            RankDeficientRHSBlockBreaksDownGracefully,
            PreconditionedMatchesScalar,
            MaxIterBudgetHonestStatus,
        }

        public TestType Type;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.MatchesScalarBiCGStabPerColumn:            MatchesScalarBiCGStabPerColumn();            break;
                case TestType.KnownSolutionRecovered:                   KnownSolutionRecovered();                    break;
                case TestType.IdentityFoldBitIdentical:                 IdentityFoldBitIdentical();                  break;
                case TestType.RankDeficientRHSBlockBreaksDownGracefully: RankDeficientRHSBlockBreaksDownGracefully(); break;
                case TestType.PreconditionedMatchesScalar:              PreconditionedMatchesScalar();               break;
                case TestType.MaxIterBudgetHonestStatus:                MaxIterBudgetHonestStatus();                 break;
            }
        }

        // BiCGSTAB is not guaranteed monotone (unlike CG), and the block vs per-column scalar solves
        // reach slightly different iterates of the SAME unique solution -- so this is a touch looser
        // than BlockCG's 2e-2f/1e-5. Both routes still converge to residual ~ tol, so the two block
        // columns / scalar columns agree to well within this bound on the well-conditioned systems here.
        static fProxy Tol() => /*+choose[3e-2f|1e-5]*/3e-2f/*-choose*/;

        // Diagonally dominant (so nonsingular) but NOT symmetrized -> genuinely non-symmetric. Do NOT
        // form M^T M (that would make it symmetric, defeating the point of a BiCGSTAB test).
        static fProxyMxN BuildDenseNonSym(int dim, uint seed)
        {
            var A = GenerateOP.fProxyRandomMat(dim, dim, (fProxy)(-1f), (fProxy)1f, seed);
            for (int d = 0; d < dim; d++) A[d, d] += dim;   // diagonally dominant -> nonsingular, nonsymmetric
            return A;
        }

        static fProxyN Row(in fProxyMxN B, int j, int n)
        {
            var v = new fProxyN(n, Allocator.Temp);
            for (int c = 0; c < n; c++) v[c] = B[j, c];
            return v;
        }

        // Each column of the block solution matches an independent scalar biCGStab solve of that
        // column, and every column reached tolerance.
        void MatchesScalarBiCGStabPerColumn()
        {
            int n = 20, s = 4;
            var A = BuildDenseNonSym(n, 81001u);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 81002u);

            int maxIter = 12 * n;                            // generous: BiCGSTAB convergence isn't monotone
            fProxy tol = Consts.fProxySqrtEps;

            var X = new fProxyMxN(s, n, Allocator.Temp);     // zero initial guess
            var info = Krylov.bbiCGStab(in A, in B, ref X, maxIter, tol);

            Assert.IsTrue(info.Solved);
            Assert.AreEqual(s, info.converged);
            Assert.AreEqual(s, info.rhs);

            for (int j = 0; j < s; j++)
            {
                var bj = Row(in B, j, n);
                var xj = new fProxyN(n, Allocator.Temp);
                Assert.IsTrue(Krylov.biCGStab(in A, in bj, ref xj, maxIter, tol).Solved);

                // Block column j matches the scalar solve (both converged to the same unique solution).
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(math.abs((double)X[j, c] - (double)xj[c]) <= Tol() * (1.0 + math.abs((double)xj[c])));
            }
        }

        // Independent of any other solver: pick a KNOWN block solution Xk, form B = A Xk via the
        // GENERAL block apply (fProxyDenseOperatorGeneral -- fProxyDenseOperator.ApplyBlock computes
        // A^T for a non-symmetric A and would build the WRONG B here), solve, and recover Xk.
        void KnownSolutionRecovered()
        {
            int n = 20, s = 5;
            var A = BuildDenseNonSym(n, 82001u);
            var Xk = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 82002u);   // known solution

            var B = new fProxyMxN(s, n, Allocator.Temp);
            new fProxyDenseOperatorGeneral(in A).ApplyBlock(in Xk, ref B, s);          // B[j,:] = A Xk[j,:]

            var X = new fProxyMxN(s, n, Allocator.Temp);
            var info = Krylov.bbiCGStab(in A, in B, ref X, 12 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(info.Solved);

            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(math.abs((double)X[j, c] - (double)Xk[j, c]) <= Tol() * (1.0 + math.abs((double)Xk[j, c])));
        }

        // The explicit-identity-preconditioner generic core must fold to EXACTLY the unpreconditioned
        // overload -- bit-identical X, iterations, status. Allocate every scratch buffer (Phat/Shat
        // included, even though unused under identity, since the preconditioned-shape overload requires
        // them).
        void IdentityFoldBitIdentical()
        {
            int n = 16, s = 3;
            var A = BuildDenseNonSym(n, 83001u);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 83002u);
            int maxIter = 8 * n;
            fProxy tol = Consts.fProxySqrtEps;

            var opA = new fProxyDenseOperatorGeneral(in A);   // readonly struct -- construct once, reuse the value

            // Explicit identity preconditioner (preconditioned-shape overload -> needs Phat/Shat).
            var X1     = new fProxyMxN(s, n, Allocator.Temp);
            var R1     = new fProxyMxN(s, n, Allocator.Temp);
            var Rhat01 = new fProxyMxN(s, n, Allocator.Temp);
            var P1     = new fProxyMxN(s, n, Allocator.Temp);
            var V1     = new fProxyMxN(s, n, Allocator.Temp);
            var T1     = new fProxyMxN(s, n, Allocator.Temp);
            var Phat1  = new fProxyMxN(s, n, Allocator.Temp);
            var Shat1  = new fProxyMxN(s, n, Allocator.Temp);
            var info1 = Krylov.bbiCGStab<fProxyDenseOperatorGeneral, fProxyIdentityPreconditioner>(
                in opA, default(fProxyIdentityPreconditioner), in B, ref X1,
                ref R1, ref Rhat01, ref P1, ref V1, ref T1, ref Phat1, ref Shat1, maxIter, tol);

            // Unpreconditioned overload (no Phat/Shat).
            var X2     = new fProxyMxN(s, n, Allocator.Temp);
            var R2     = new fProxyMxN(s, n, Allocator.Temp);
            var Rhat02 = new fProxyMxN(s, n, Allocator.Temp);
            var P2     = new fProxyMxN(s, n, Allocator.Temp);
            var V2     = new fProxyMxN(s, n, Allocator.Temp);
            var T2     = new fProxyMxN(s, n, Allocator.Temp);
            var info2 = Krylov.bbiCGStab<fProxyDenseOperatorGeneral>(
                in opA, in B, ref X2, ref R2, ref Rhat02, ref P2, ref V2, ref T2, maxIter, tol);

            Assert.AreEqual(info1.iterations, info2.iterations);
            Assert.IsTrue(info1.status == info2.status);
            // Bit-identical X (no tolerance) -- matches the codebase's determinism-check pattern (== via
            // IsTrue rather than the boxing Assert.AreEqual(float,float), which is not Burst-safe).
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(X1[i, c] == X2[i, c]);
        }

        // A rank-deficient RHS block (several identical rows) makes the s x s block coefficient
        // (Rhat0^T V, and Rhat0 inherits the identical rows) singular -> the QRCP solve reports
        // rank-deficiency -> defined Breakdown. The contract: X stays finite (never NaN/inf), and the
        // solver reports Breakdown rather than throwing or silently claiming Convergence.
        void RankDeficientRHSBlockBreaksDownGracefully()
        {
            int n = 16, s = 5;
            var A = BuildDenseNonSym(n, 84001u);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 84002u);
            // Force rows 0, 2, 4 identical -> block rank <= 3, and the shadow residual Rhat0 (= initial
            // R = B, since X0 = 0) inherits those identical rows -> a genuinely singular s x s coeff.
            for (int c = 0; c < n; c++) { B[2, c] = B[0, c]; B[4, c] = B[0, c]; }

            var X = new fProxyMxN(s, n, Allocator.Temp);
            var info = Krylov.bbiCGStab(in A, in B, ref X, 8 * n, Consts.fProxySqrtEps);

            // Finite, no NaN -- last committed iterate (here the zero start) is returned, never garbage.
            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.IsFalse(double.IsNaN((double)X[j, c]) || double.IsInfinity((double)X[j, c]));

            // Rank-deficient block coefficient is DEFINED behavior -> Breakdown (not Solved).
            Assert.IsTrue(info.status == IterativeSolveStatus.Breakdown);
        }

        // BSR non-symmetric diagonally-dominant A + Restricted Additive Schwarz (RAS, non-symmetric,
        // biCGStab's own preconditioner). The block solve matches per-column scalar RAS-preconditioned
        // biCGStab.
        void PreconditionedMatchesScalar()
        {
            var A = fProxyGallery.fProxyRandomSparse(16, 16, 2, (fProxy)0.4, 85001u);   // 32 dof, nonsymmetric diag-dominant
            int n = A.M_Rows;
            int s = 3;
            var opts = new SchwarzOptions { subdomainSize = 12, overlap = 1 };  // multiple subdomains
            var M = new fProxyRestrictedSchwarz(in A, Allocator.Temp, in opts);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 85002u);

            int maxIter = 20 * n;
            fProxy tol = Consts.fProxySqrtEps;

            var X = new fProxyMxN(s, n, Allocator.Temp);
            var info = Krylov.bbiCGStab(in A, in M, in B, ref X, maxIter, tol);
            Assert.IsTrue(info.Solved);
            Assert.AreEqual(s, info.converged);

            for (int j = 0; j < s; j++)
            {
                var bj = Row(in B, j, n);
                var xj = new fProxyN(n, Allocator.Temp);
                Assert.IsTrue(Krylov.biCGStab(in A, in M, in bj, ref xj, maxIter, tol).Solved);
                for (int c = 0; c < n; c++)
                    Assert.IsTrue(math.abs((double)X[j, c] - (double)xj[c]) <= Tol() * (1.0 + math.abs((double)xj[c])));
            }
        }

        // A tiny iteration budget on a system that genuinely needs more must report an HONEST
        // MaxIterations status (not a false Converged), with fewer than all columns converged and X
        // still finite -- no throw.
        void MaxIterBudgetHonestStatus()
        {
            int n = 32, s = 4;
            var A = BuildDenseNonSym(n, 86001u);
            var B = GenerateOP.fProxyRandomMat(s, n, (fProxy)(-1f), (fProxy)1f, 86002u);

            // maxIter = 1 is a degree-2 polynomial in A; the Chebyshev floor over this system's
            // eigenvalue cluster (~2e-2 relative residual) sits ~3000x above float's sqrtEps threshold,
            // so a single iteration provably CANNOT converge -> a robust, non-flaky MaxIterations check.
            var X = new fProxyMxN(s, n, Allocator.Temp);
            var info = Krylov.bbiCGStab(in A, in B, ref X, 1, Consts.fProxySqrtEps);   // deliberately tiny budget

            Assert.IsTrue(info.status == IterativeSolveStatus.MaxIterations);
            Assert.IsFalse(info.Solved);
            Assert.IsTrue(info.converged < s);
            for (int j = 0; j < s; j++)
                for (int c = 0; c < n; c++)
                    Assert.IsFalse(double.IsNaN((double)X[j, c]) || double.IsInfinity((double)X[j, c]));
        }
    }

    [Test]
    public void MatchesScalarBiCGStabPerColumn()
        => new BlockBiCGStabTestJob { Type = BlockBiCGStabTestJob.TestType.MatchesScalarBiCGStabPerColumn }.Run();

    [Test]
    public void KnownSolutionRecovered()
        => new BlockBiCGStabTestJob { Type = BlockBiCGStabTestJob.TestType.KnownSolutionRecovered }.Run();

    [Test]
    public void IdentityFoldBitIdentical()
        => new BlockBiCGStabTestJob { Type = BlockBiCGStabTestJob.TestType.IdentityFoldBitIdentical }.Run();

    [Test]
    public void RankDeficientRHSBlockBreaksDownGracefully()
        => new BlockBiCGStabTestJob { Type = BlockBiCGStabTestJob.TestType.RankDeficientRHSBlockBreaksDownGracefully }.Run();

    [Test]
    public void PreconditionedMatchesScalar()
        => new BlockBiCGStabTestJob { Type = BlockBiCGStabTestJob.TestType.PreconditionedMatchesScalar }.Run();

    [Test]
    public void MaxIterBudgetHonestStatus()
        => new BlockBiCGStabTestJob { Type = BlockBiCGStabTestJob.TestType.MaxIterBudgetHonestStatus }.Run();
}
