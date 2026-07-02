using System;
using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Phase-2 sparse-solver test suite: IfProxyLinearOperator / fProxyDenseOperator /
// fProxyBSMOperator / fProxyBlockJacobi / Solvers.cg&lt;TOp&gt; / Solvers.pcg&lt;TOp,TPre&gt;, plus
// the concrete conjugateGradient/pcg convenience overloads for fProxyMxN and fProxyBSM. Every
// BSM system is cross-checked against the equivalent dense system (same pattern as
// fProxySparseBSMTests: build the SAME system in both forms and compare).
//
// Correctness cases run inside a [BurstCompile] IJob (matches fProxyConjugateGradientTests /
// fProxySparseBSMTests). Guard/exception cases run on the managed test thread with
// Assert.Throws, since NUnit's Assert.Throws cannot execute inside a Burst-compiled job.
public class fProxySparseSolverTests
{
    [BurstCompile]
    public struct SparseSolverTestJob : IJob
    {
        public enum TestType
        {
            Laplacian1DBSMCGMatchesDenseCG,
            ThreeByThreeBlockSPDConverges,
            DenseForwardingUnchanged,
            PCGMatchesCG,
            PCGBeatsCGIllConditioned,
            BlockJacobiApplyHandComputed,
            WarmStart,
            PcgNonSpdPreconditionerBreaksDown,

            // ---- Phase 3: MINRES / BiCGSTAB / CGLS / LSQR ----
            MinresIndefiniteDenseAndBSM,
            MinresSpdMatchesCG,
            BiCGStabNonSymmetricMatchesLU,
            CglsOverdeterminedConsistentDenseAndBSM,
            LsqrOverdeterminedConsistentDenseAndBSM,
            CglsInconsistentMatchesQR,
            LsqrInconsistentMatchesQR,
            CglsLsqrUnderdeterminedConsistent,

            // ---- LSMR (Fong-Saunders): least-squares with monotone ||A^T r|| ----
            LsmrOverdeterminedConsistentDenseAndBSM,
            LsmrInconsistentMatchesQR,
            LsmrUnderdeterminedMatchesLsqr,

            // ---- Phase 3 warm-start plumbing (initial residual r = b - A*x0 from the CALLER's x) ----
            MinresWarmStart,
            BiCGStabWarmStart,
            CglsWarmStart,
            LsqrWarmStart,
            LsmrWarmStart,
        }

        public TestType Type;

        // CG/PCG residuals converge to tolerance^2 * ||b||^2 (see Solvers.cg), so per-component
        // error is well below this scaled comparison threshold on both precisions; loose enough
        // that comparing two INDEPENDENTLY-converged solutions (each accurate only to about
        // Consts.fProxySqrtEps, not machine epsilon) doesn't false-fail.
        static fProxy Tol() => /*+choose[1e-3f|1e-7]*/1e-3f/*-choose*/;

        // Looser threshold for the Phase-3 solvers' cross-checks. These compare TWO
        // independently-converged iterative solutions (or an iterative solution against a direct
        // one) on INDEFINITE / RECTANGULAR / ILL-conditioned systems, whose per-component absolute
        // error can be a few times Consts.fProxySqrtEps*scale -- looser than the SPD-CG cases above.
        // The spec explicitly allows loosening to 1e-2f|1e-5 for exactly these iterative-vs-direct
        // comparisons rather than fighting flaky tolerances.
        static fProxy LooseTol() => /*+choose[1e-2f|1e-5]*/1e-2f/*-choose*/;

        // Least-squares optimality: the NORMAL-EQUATION residual ||A^T(A x - b)|| must vanish
        // relative to the fixed scale ||A^T b|| (mirrors cgls/lsqr's own convergence reference).
        // This -- NOT ||A x - b|| ~= 0 -- is the correct acceptance criterion for an inconsistent
        // (overdetermined) system, whose residual A x - b is left orthogonal to range(A), nonzero.
        static void AssertLeastSquaresOptimal(in fProxyMxN A, in fProxyN x, in fProxyN b, fProxy relTol)
        {
            var Ax  = Linear_OP.dot(A, x);
            var res = Ax - b;                 // r = A x - b     (length m)
            var atr = Linear_OP.dot(res, A);  // A^T r           (length n)  -- vector*matrix == A^T r
            var atb = Linear_OP.dot(b, A);    // A^T b           (scale reference)
            fProxy atrNorm = math.sqrt(Linear_OP.dot(atr, atr));
            fProxy atbNorm = math.sqrt(Linear_OP.dot(atb, atb));
            Assert.IsTrue(atrNorm <= relTol * atbNorm);
        }

        public void Execute()
        {
            switch (Type)
            {
                case TestType.Laplacian1DBSMCGMatchesDenseCG: Laplacian1DBSMCGMatchesDenseCG(); break;
                case TestType.ThreeByThreeBlockSPDConverges: ThreeByThreeBlockSPDConverges(); break;
                case TestType.DenseForwardingUnchanged: DenseForwardingUnchanged(); break;
                case TestType.PCGMatchesCG: PCGMatchesCG(); break;
                case TestType.PCGBeatsCGIllConditioned: PCGBeatsCGIllConditioned(); break;
                case TestType.BlockJacobiApplyHandComputed: BlockJacobiApplyHandComputed(); break;
                case TestType.WarmStart: WarmStart(); break;
                case TestType.PcgNonSpdPreconditionerBreaksDown: PcgNonSpdPreconditionerBreaksDown(); break;

                case TestType.MinresIndefiniteDenseAndBSM: MinresIndefiniteDenseAndBSM(); break;
                case TestType.MinresSpdMatchesCG: MinresSpdMatchesCG(); break;
                case TestType.BiCGStabNonSymmetricMatchesLU: BiCGStabNonSymmetricMatchesLU(); break;
                case TestType.CglsOverdeterminedConsistentDenseAndBSM: CglsOverdeterminedConsistentDenseAndBSM(); break;
                case TestType.LsqrOverdeterminedConsistentDenseAndBSM: LsqrOverdeterminedConsistentDenseAndBSM(); break;
                case TestType.CglsInconsistentMatchesQR: CglsInconsistentMatchesQR(); break;
                case TestType.LsqrInconsistentMatchesQR: LsqrInconsistentMatchesQR(); break;
                case TestType.CglsLsqrUnderdeterminedConsistent: CglsLsqrUnderdeterminedConsistent(); break;

                case TestType.LsmrOverdeterminedConsistentDenseAndBSM: LsmrOverdeterminedConsistentDenseAndBSM(); break;
                case TestType.LsmrInconsistentMatchesQR: LsmrInconsistentMatchesQR(); break;
                case TestType.LsmrUnderdeterminedMatchesLsqr: LsmrUnderdeterminedMatchesLsqr(); break;

                case TestType.MinresWarmStart: MinresWarmStart(); break;
                case TestType.BiCGStabWarmStart: BiCGStabWarmStart(); break;
                case TestType.CglsWarmStart: CglsWarmStart(); break;
                case TestType.LsqrWarmStart: LsqrWarmStart(); break;
                case TestType.LsmrWarmStart: LsmrWarmStart(); break;
            }
        }

        // ---- helpers ---------------------------------------------------------------------

        // Same recipe as fProxyConjugateGradientTests.BuildSPD: A = M^T M + dim*I (strictly SPD,
        // diagonally dominant).
        static fProxyMxN BuildDenseSPD(ref Arena arena, int dim, uint seed)
        {
            var M = arena.fProxyRandomMat(dim, dim, -1f, 1f, seed);
            var A = Linear_OP.dot(M, M, true);
            for (int d = 0; d < dim; d++)
                A[d, d] += dim;
            return A;
        }

        // 1x1-block BSM built from a dense matrix's nonzero entries via AddValue. Triplet count
        // is bounded by the caller-supplied nnzHint (sized to the known nonzero pattern) purely
        // as a perf choice -- it avoids a few reallocations of the builder's internal growable
        // lists, nothing more. Growing the builder's lists past capacityHint (triggering one or
        // more UnsafeList reallocations) is safe: the builder's mutable triplet state lives
        // behind a single heap-allocated pointer shared by every value-copy of the struct
        // (including the arena's own tracked copy), so a reallocation on one copy is visible to
        // all of them. See the growth regression tests in SparseBSMTests.fProxy.cs, which build
        // via many-reallocation growth on purpose to prove this.
        static fProxyBSM DenseToBSM1x1(ref Arena arena, in fProxyMxN A, int nnzHint)
        {
            var builder = arena.fProxyBSMBuilder(A.M_Rows, A.N_Cols, 1, 1, math.max(nnzHint, 1));
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    if (A[r, c] != (fProxy)0)
                        builder.AddValue(r, c, A[r, c]);
            return builder.ToBSM(ref arena);
        }

        static void AssertVecEq(in fProxyN a, in fProxyN b, fProxy tol)
        {
            Assert.IsTrue(Analysis_OP.isZero(a - b, tol));
        }

        // ---- 1. 1D Laplacian tridiagonal as a 1x1-block BSM: CG matches dense CG -----------
        void Laplacian1DBSMCGMatchesDenseCG()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            var A = arena.fProxyLaplacian1D(dim);
            // Tridiagonal: at most 3 nonzeros/row -> 3*dim is a safe upper bound.
            var bsm = DenseToBSM1x1(ref arena, in A, 3 * dim);

            var b = arena.fProxyRandomVec(dim, -1f, 1f, 4242);

            var xDense = arena.fProxyVec(dim);
            bool okDense = Solvers.conjugateGradient(in A, in b, ref xDense, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okDense);

            var xBsm = arena.fProxyVec(dim);
            bool okBsm = Solvers.conjugateGradient(in bsm, in b, ref xBsm, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okBsm);

            AssertVecEq(in xDense, in xBsm, Tol());

            // A*x ~= b for the BSM solve too (spec's explicit acceptance criterion).
            var Ax = Sparse_OP.spMV(in bsm, in xBsm);
            AssertVecEq(in Ax, in b, Tol());

            arena.Dispose();
        }

        // ---- 2. 3x3-block SPD system: CG converges, residual within tol -------------------
        //
        // Build a random block matrix M (BR=3, a handful of blocks on a 3x3 block grid), form
        // the dense A = M^T M + eps*I (guaranteed SPD), then re-encode A as a genuine 3x3-block
        // BSM with every block-row/col pair stored (A^T A is generally dense even when M is
        // sparse) -- so CG genuinely walks a multi-block-per-row BSR structure, not just 1x1
        // scalars.
        void ThreeByThreeBlockSPDConverges()
        {
            var arena = new Arena(Allocator.Persistent);

            const int BR = 3;
            const int nb = 3; // 9x9
            int dim = BR * nb;

            var mb = arena.fProxyBSMBuilder(nb, nb, BR, BR, nb * nb);
            mb.AddBlock(0, 0, arena.fProxyRandomMat(BR, BR, -1f, 1f, 8001));
            mb.AddBlock(0, 1, arena.fProxyRandomMat(BR, BR, -1f, 1f, 8002));
            mb.AddBlock(1, 1, arena.fProxyRandomMat(BR, BR, -1f, 1f, 8003));
            mb.AddBlock(1, 2, arena.fProxyRandomMat(BR, BR, -1f, 1f, 8004));
            mb.AddBlock(2, 2, arena.fProxyRandomMat(BR, BR, -1f, 1f, 8005));
            mb.AddBlock(2, 0, arena.fProxyRandomMat(BR, BR, -1f, 1f, 8006));
            var Mdense = mb.ToBSM(ref arena).ToDense(ref arena);

            var A = Linear_OP.dot(Mdense, Mdense, true);
            for (int i = 0; i < dim; i++)
                A[i, i] += dim;

            var ab = arena.fProxyBSMBuilder(nb, nb, BR, BR, nb * nb);
            for (int bi = 0; bi < nb; bi++)
                for (int bj = 0; bj < nb; bj++)
                {
                    var blk = arena.fProxyMat(BR, BR);
                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BR; c++)
                            blk[r, c] = A[bi * BR + r, bj * BR + c];
                    ab.AddBlock(bi, bj, in blk);
                }
            var Absm = ab.ToBSM(ref arena);

            var b = arena.fProxyRandomVec(dim, -1f, 1f, 8100);
            var x = arena.fProxyVec(dim);
            bool ok = Solvers.conjugateGradient(in Absm, in b, ref x);
            Assert.IsTrue(ok);

            var Ax = Sparse_OP.spMV(in Absm, in x);
            AssertVecEq(in Ax, in b, Tol());

            // Cross-check against the dense reference too.
            var xDense = arena.fProxyVec(dim);
            bool okDense = Solvers.conjugateGradient(in A, in b, ref xDense);
            Assert.IsTrue(okDense);
            AssertVecEq(in x, in xDense, Tol());

            arena.Dispose();
        }

        // ---- 3. Dense forwarding unchanged: guards the conjugateGradient(in fProxyMxN,...) ----
        //         refactor into cg<fProxyDenseOperator> -----------------------------------------
        void DenseForwardingUnchanged()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 12;
            var A = BuildDenseSPD(ref arena, dim, 90125);
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 4242);

            // The (unchanged, public) concrete entry point.
            var xConcrete = arena.fProxyVec(dim);
            bool okConcrete = Solvers.conjugateGradient(in A, in b, ref xConcrete);
            Assert.IsTrue(okConcrete);

            // Calling cg<TOp> directly via fProxyDenseOperator must reproduce the identical
            // result -- this is the single source of truth the concrete overload now forwards
            // into.
            var op = new fProxyDenseOperator(in A);
            var xGeneric = arena.fProxyVec(dim);
            var r = arena.fProxyVec(dim);
            var p = arena.fProxyVec(dim);
            var Ap = arena.fProxyVec(dim);
            bool okGeneric = Solvers.cg(in op, in b, ref xGeneric, ref r, ref p, ref Ap, dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okGeneric);

            AssertVecEq(in xConcrete, in xGeneric, Tol());

            var Ax = Linear_OP.dot(A, xConcrete);
            AssertVecEq(in Ax, in b, Tol());

            // Independent cross-check against a DIRECT solver on a completely different code path
            // (Householder QR, no Krylov/CG involvement). The xConcrete-vs-xGeneric check above is
            // circular now that both funnel through the same cg<TOp> loop; this pins the CG
            // solution to a truly independent reference. qrDirectSolve is DESTRUCTIVE (destroys
            // Q/A and b), so it MUST run on fresh copies, not the A/b the CG calls used.
            var A2 = A.Copy();
            var b2 = b.Copy();
            var xQR = arena.fProxyVec(dim);
            QR.qrDirectSolve(ref A2, ref b2, ref xQR);
            AssertVecEq(in xConcrete, in xQR, Tol());

            arena.Dispose();
        }

        // ---- 4. PCG correctness: matches CG's solution -------------------------------------
        void PCGMatchesCG()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 12;
            var A = BuildDenseSPD(ref arena, dim, 6001);
            var bsm = DenseToBSM1x1(ref arena, in A, dim * dim);
            var M = arena.fProxyBlockJacobi(in bsm);
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 6002);

            var xCG = arena.fProxyVec(dim);
            bool okCG = Solvers.conjugateGradient(in bsm, in b, ref xCG);
            Assert.IsTrue(okCG);

            var xPCG = arena.fProxyVec(dim);
            bool okPCG = Solvers.pcg(in bsm, in M, in b, ref xPCG);
            Assert.IsTrue(okPCG);

            AssertVecEq(in xCG, in xPCG, Tol());

            arena.Dispose();
        }

        // ---- 5. Block-Jacobi PCG needs <= iterations of plain CG on an ill-conditioned,
        //         diagonally-scaled SPD system --------------------------------------------
        //
        // Sym is normalized to a UNIT diagonal ("correlation matrix": Sym = S / sqrt(Sii*Sjj),
        // a diagonal congruence transform, so it stays SPD) before the D*Sym*D rescale. That
        // makes diag(A) EXACTLY d_i^2, so point-Jacobi (BR=1) exactly recovers Sym's own
        // well-conditioned spectrum -- the textbook case where Jacobi preconditioning provably
        // helps. The iteration budget used to find each solver's minimum is 4*dim, not dim:
        // CG's finite-termination property (<=dim iterations) only holds in EXACT arithmetic;
        // floating point on an ill-conditioned system can need more (same 4n cap the
        // GalleryLaplacian1D/MinIJ CG tests use).
        void PCGBeatsCGIllConditioned()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            var S = BuildDenseSPD(ref arena, dim, 7001);

            var Sym = arena.fProxyMat(dim, dim);
            for (int r = 0; r < dim; r++)
                for (int c = 0; c < dim; c++)
                    Sym[r, c] = S[r, c] / math.sqrt(S[r, r] * S[c, c]);

            var A = arena.fProxyMat(dim, dim);
            var d = arena.fProxyVec(dim);
            for (int i = 0; i < dim; i++)
                d[i] = (i % 2 == 0) ? (fProxy)1 : (fProxy)100; // alternating 1,100

            for (int r = 0; r < dim; r++)
                for (int c = 0; c < dim; c++)
                    A[r, c] = d[r] * Sym[r, c] * d[c];

            var bsm = DenseToBSM1x1(ref arena, in A, dim * dim);
            var M = arena.fProxyBlockJacobi(in bsm);
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 7002);

            int maxBudget = 4 * dim;
            int minCG = -1, minPCG = -1;
            for (int budget = 1; budget <= maxBudget && (minCG < 0 || minPCG < 0); budget++)
            {
                if (minCG < 0)
                {
                    var xCG = arena.fProxyVec(dim);
                    if (Solvers.conjugateGradient(in bsm, in b, ref xCG, budget, Consts.fProxySqrtEps))
                        minCG = budget;
                }
                if (minPCG < 0)
                {
                    var xPCG = arena.fProxyVec(dim);
                    if (Solvers.pcg(in bsm, in M, in b, ref xPCG, budget, Consts.fProxySqrtEps))
                        minPCG = budget;
                }
            }

            Assert.IsTrue(minCG > 0);
            Assert.IsTrue(minPCG > 0);
            Assert.IsTrue(minPCG <= minCG);

            arena.Dispose();
        }

        // ---- 6. Block-Jacobi Apply matches a hand-computed block-diagonal inverse ----------
        void BlockJacobiApplyHandComputed()
        {
            var arena = new Arena(Allocator.Persistent);

            const int BR = 2;
            var builder = arena.fProxyBSMBuilder(2, 2, BR, BR, 2);
            var d0 = arena.fProxyMat(BR, BR);
            d0[0, 0] = 4f; d0[0, 1] = 1f;
            d0[1, 0] = 1f; d0[1, 1] = 3f;
            var d1 = arena.fProxyMat(BR, BR);
            d1[0, 0] = 2f; d1[0, 1] = 0f;
            d1[1, 0] = 0f; d1[1, 1] = 5f;
            builder.AddBlock(0, 0, in d0);
            builder.AddBlock(1, 1, in d1);
            var A = builder.ToBSM(ref arena);

            var M = arena.fProxyBlockJacobi(in A);

            var r = arena.fProxyVec(4);
            r[0] = 1f; r[1] = 2f; r[2] = 3f; r[3] = 4f;
            var z = arena.fProxyVec(4);
            M.Apply(in r, ref z);

            // Hand inverse of d0=[[4,1],[1,3]]: det=11, inv = (1/11)*[[3,-1],[-1,4]].
            fProxy det0 = 11f;
            fProxy z0 = (3f * r[0] - 1f * r[1]) / det0;
            fProxy z1 = (-1f * r[0] + 4f * r[1]) / det0;
            // d1 is diagonal: inv = diag(1/2, 1/5).
            fProxy z2 = r[2] / 2f;
            fProxy z3 = r[3] / 5f;

            Assert.IsTrue(math.abs(z[0] - z0) < Tol());
            Assert.IsTrue(math.abs(z[1] - z1) < Tol());
            Assert.IsTrue(math.abs(z[2] - z2) < Tol());
            Assert.IsTrue(math.abs(z[3] - z3) < Tol());

            arena.Dispose();
        }

        // ---- 7. Warm start: seeding x with the exact solution converges immediately --------
        void WarmStart()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 10;
            var A = BuildDenseSPD(ref arena, dim, 9001);
            var bsm = DenseToBSM1x1(ref arena, in A, dim * dim);
            var M = arena.fProxyBlockJacobi(in bsm);
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 9002);

            var x = arena.fProxyVec(dim);
            bool ok = Solvers.pcg(in bsm, in M, in b, ref x);
            Assert.IsTrue(ok);

            // Feed the converged solution back as the initial guess -- a single iteration's
            // worth of budget must still report convergence (matches
            // fProxyConjugateGradientTests.AlreadyConverged's dense-CG counterpart).
            var xWarm = x.Copy();
            bool okWarm = Solvers.pcg(in bsm, in M, in b, ref xWarm, 1, Consts.fProxySqrtEps);
            Assert.IsTrue(okWarm);
            AssertVecEq(in x, in xWarm, Tol());

            // Same check for plain (unpreconditioned) CG.
            var xCG = arena.fProxyVec(dim);
            bool okCG = Solvers.conjugateGradient(in bsm, in b, ref xCG);
            Assert.IsTrue(okCG);
            var xCGWarm = xCG.Copy();
            bool okCGWarm = Solvers.conjugateGradient(in bsm, in b, ref xCGWarm, 1, Consts.fProxySqrtEps);
            Assert.IsTrue(okCGWarm);
            AssertVecEq(in xCG, in xCGWarm, Tol());

            arena.Dispose();
        }

        // ---- 8. pcg breakdown guard: a non-SPD preconditioner bails out (returns false) -------
        //
        // The preconditioned inner product <r,z> must stay positive for PCG to be well-defined.
        // fProxyNegatePreconditioner deliberately returns z = -r, so rzold = <r,-r> = -||r||^2 < 0
        // -- the pcg `if (!(rzold > 0)) return false;` guard (added this pass) must catch it and
        // return false, rather than looping with a wrong-signed alpha/beta (silent divergence /
        // NaN). A itself is a genuine SPD system so the failure is attributable to the
        // preconditioner, not the operator.
        void PcgNonSpdPreconditionerBreaksDown()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;
            var A = BuildDenseSPD(ref arena, dim, 5501);
            var op = new fProxyDenseOperator(in A);
            var pre = new fProxyNegatePreconditioner();
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 5502); // nonzero rhs -> not the b==0 shortcut

            var x = arena.fProxyVec(dim);
            bool ok = Solvers.pcg(in op, in pre, in b, ref x, dim, Consts.fProxySqrtEps);
            Assert.IsFalse(ok);

            arena.Dispose();
        }

        // =================================================================================
        // Phase 3 correctness cases
        // =================================================================================

        // ---- MINRES on a symmetric INDEFINITE system (dense + BSM agree) -----------------
        //
        // Laplacian1D (SPD, diag 2 / off-diag -1) shifted by -2 on the diagonal: eigenvalues
        // become 2-2cos(k*pi/(n+1)) - 2 = -2cos(k*pi/(n+1)) for k=1..n, which straddle 0 -> a
        // genuinely mixed-sign (symmetric indefinite) A. dim=16 -> n+1=17 is odd, so k=(n+1)/2
        // is non-integer and NO eigenvalue is exactly 0 (A stays nonsingular). MINRES handles
        // this cleanly where CG's p.Ap>0 curvature requirement would break down.
        void MinresIndefiniteDenseAndBSM()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            var A = arena.fProxyLaplacian1D(dim);
            for (int i = 0; i < dim; i++)
                A[i, i] -= (fProxy)2;          // shift diag 2 -> 0: mixed-sign spectrum, indefinite

            var b = arena.fProxyRandomVec(dim, -1f, 1f, 31001);

            var xDense = arena.fProxyVec(dim);
            bool okDense = Solvers.minres(in A, in b, ref xDense, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okDense);

            var Ax = Linear_OP.dot(A, xDense);
            AssertVecEq(in Ax, in b, LooseTol());          // A nonsingular -> unique solution, A x ~= b

            // Independent cross-check against a DIRECT LU solve on the SAME indefinite matrix
            // (no Krylov/MINRES involvement) -- pins the iterative solution to a truly independent
            // reference, not just the self-consistent A x ~= b residual. luDecompositionInpl +
            // luSolve are DESTRUCTIVE, so they run on COPIES. The shifted Laplacian above is
            // constructed to be nonsingular (odd n+1 -> no exactly-zero eigenvalue), so LU succeeds.
            var LUcopy = A.Copy();
            var pivot = new Pivot(dim, Allocator.Temp);
            bool okLU = LU.luDecompositionInpl(ref LUcopy, ref pivot);
            Assert.IsTrue(okLU);
            var xLU = b.Copy();
            LU.luSolve(ref LUcopy, in pivot, ref xLU);
            AssertVecEq(in xDense, in xLU, LooseTol());
            pivot.Dispose();

            // Same system as a 1x1-block BSM: minres(BSM) must agree with minres(dense).
            var bsm = DenseToBSM1x1(ref arena, in A, 3 * dim);   // tridiagonal (shifted diag=0 dropped)
            var xBsm = arena.fProxyVec(dim);
            bool okBsm = Solvers.minres(in bsm, in b, ref xBsm, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okBsm);
            AssertVecEq(in xDense, in xBsm, LooseTol());

            var AxBsm = Sparse_OP.spMV(in bsm, in xBsm);
            AssertVecEq(in AxBsm, in b, LooseTol());

            // NOTE (spec nice-to-have, NOT asserted): plain CG on this SAME indefinite A breaks
            // down -- Solvers.conjugateGradient's p.Ap>0 curvature guard fails / returns a much
            // worse residual. MINRES succeeding where CG cannot is the whole point of this case;
            // asserting CG's failure mode is fiddly and left as a documented expectation.

            arena.Dispose();
        }

        // ---- MINRES on a plain SPD system agrees with CG (dense + BSM) --------------------
        void MinresSpdMatchesCG()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 12;
            var A = BuildDenseSPD(ref arena, dim, 32001);
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 32002);

            var xCG = arena.fProxyVec(dim);
            bool okCG = Solvers.conjugateGradient(in A, in b, ref xCG, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okCG);

            var xMin = arena.fProxyVec(dim);
            bool okMin = Solvers.minres(in A, in b, ref xMin, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okMin);
            AssertVecEq(in xCG, in xMin, LooseTol());       // MINRES == CG on an SPD system

            var Ax = Linear_OP.dot(A, xMin);
            AssertVecEq(in Ax, in b, LooseTol());

            // BSM minres agrees with dense minres.
            var bsm = DenseToBSM1x1(ref arena, in A, dim * dim);
            var xMinBsm = arena.fProxyVec(dim);
            bool okMinBsm = Solvers.minres(in bsm, in b, ref xMinBsm, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okMinBsm);
            AssertVecEq(in xMin, in xMinBsm, LooseTol());

            arena.Dispose();
        }

        // ---- BiCGSTAB on a NON-symmetric (diagonally-dominant) system --------------------
        //
        // Random off-diagonals in [-1,1], diagonal boosted to dim+1 so |A_ii| > sum_{j!=i}|A_ij|
        // strictly -> nonsingular, unconditionally BiCGSTAB-friendly, and deliberately NOT
        // symmetrized. Cross-checked against a dense DIRECT LU solve on the SAME matrix.
        void BiCGStabNonSymmetricMatchesLU()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;
            var A = arena.fProxyRandomMat(dim, dim, -1f, 1f, 33001);
            for (int i = 0; i < dim; i++)
                A[i, i] += (fProxy)(dim + 1);   // strict diagonal dominance

            var b = arena.fProxyRandomVec(dim, -1f, 1f, 33002);

            var xBcg = arena.fProxyVec(dim);
            bool okBcg = Solvers.biCGStab(in A, in b, ref xBcg, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okBcg);
            var Ax = Linear_OP.dot(A, xBcg);
            AssertVecEq(in Ax, in b, LooseTol());

            // Direct LU reference on COPIES (luDecompositionInpl + luSolve are DESTRUCTIVE).
            var LUcopy = A.Copy();
            var pivot = new Pivot(dim, Allocator.Temp);
            bool okLU = LU.luDecompositionInpl(ref LUcopy, ref pivot);
            Assert.IsTrue(okLU);
            var xLU = b.Copy();
            LU.luSolve(ref LUcopy, in pivot, ref xLU);
            AssertVecEq(in xBcg, in xLU, LooseTol());
            pivot.Dispose();

            // BSM form agrees with the dense BiCGSTAB solve.
            var bsm = DenseToBSM1x1(ref arena, in A, dim * dim);
            var xBcgBsm = arena.fProxyVec(dim);
            bool okBcgBsm = Solvers.biCGStab(in bsm, in b, ref xBcgBsm, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okBcgBsm);
            AssertVecEq(in xBcg, in xBcgBsm, LooseTol());

            arena.Dispose();
        }

        // ---- CGLS on an overdetermined CONSISTENT least-squares problem (dense + BSM) -----
        //
        // b = A*x_true exactly (b in range(A)) -> the least-squares solution is x_true, recovered
        // exactly (within tolerance). m > n.
        void CglsOverdeterminedConsistentDenseAndBSM()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 10, n = 4;
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, 34001);
            var xTrue = arena.fProxyRandomVec(n, -1f, 1f, 34002);
            var b = Linear_OP.dot(A, xTrue);      // consistent

            var x = arena.fProxyVec(n);
            bool ok = Solvers.cgls(in A, in b, ref x, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);
            AssertVecEq(in x, in xTrue, LooseTol());

            var Ax = Linear_OP.dot(A, x);
            AssertVecEq(in Ax, in b, LooseTol());

            var bsm = DenseToBSM1x1(ref arena, in A, m * n);
            var xBsm = arena.fProxyVec(n);
            bool okBsm = Solvers.cgls(in bsm, in b, ref xBsm, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(okBsm);
            AssertVecEq(in x, in xBsm, LooseTol());

            arena.Dispose();
        }

        // ---- LSQR on an overdetermined CONSISTENT least-squares problem (dense + BSM) ------
        void LsqrOverdeterminedConsistentDenseAndBSM()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 10, n = 4;
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, 35001);
            var xTrue = arena.fProxyRandomVec(n, -1f, 1f, 35002);
            var b = Linear_OP.dot(A, xTrue);      // consistent

            var x = arena.fProxyVec(n);
            bool ok = Solvers.lsqr(in A, in b, ref x, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);
            AssertVecEq(in x, in xTrue, LooseTol());

            var Ax = Linear_OP.dot(A, x);
            AssertVecEq(in Ax, in b, LooseTol());

            var bsm = DenseToBSM1x1(ref arena, in A, m * n);
            var xBsm = arena.fProxyVec(n);
            bool okBsm = Solvers.lsqr(in bsm, in b, ref xBsm, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(okBsm);
            AssertVecEq(in x, in xBsm, LooseTol());

            arena.Dispose();
        }

        // ---- CGLS on an overdetermined INCONSISTENT problem: normal-equation optimality ---
        //
        // Random b generally NOT in range(A) -> ||A x - b|| does NOT go to 0. The correct
        // acceptance criterion is ||A^T(A x - b)|| ~= 0 (residual orthogonal to range(A)),
        // cross-checked against a dense QR least-squares solve on the SAME system.
        void CglsInconsistentMatchesQR()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 10, n = 4;
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, 36001);
            var b = arena.fProxyRandomVec(m, -1f, 1f, 36002);   // inconsistent

            var x = arena.fProxyVec(n);
            bool ok = Solvers.cgls(in A, in b, ref x, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);

            AssertLeastSquaresOptimal(in A, in x, in b, LooseTol());

            // Dense QR least-squares reference on COPIES (qrDirectSolve is DESTRUCTIVE).
            var A2 = A.Copy();
            var b2 = b.Copy();
            var xQR = arena.fProxyVec(n);
            QR.qrDirectSolve(ref A2, ref b2, ref xQR);
            AssertVecEq(in x, in xQR, LooseTol());

            var bsm = DenseToBSM1x1(ref arena, in A, m * n);
            var xBsm = arena.fProxyVec(n);
            bool okBsm = Solvers.cgls(in bsm, in b, ref xBsm, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(okBsm);
            AssertVecEq(in x, in xBsm, LooseTol());

            arena.Dispose();
        }

        // ---- LSQR on an overdetermined INCONSISTENT problem: normal-equation optimality ----
        void LsqrInconsistentMatchesQR()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 10, n = 4;
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, 37001);
            var b = arena.fProxyRandomVec(m, -1f, 1f, 37002);   // inconsistent

            var x = arena.fProxyVec(n);
            bool ok = Solvers.lsqr(in A, in b, ref x, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);

            AssertLeastSquaresOptimal(in A, in x, in b, LooseTol());

            var A2 = A.Copy();
            var b2 = b.Copy();
            var xQR = arena.fProxyVec(n);
            QR.qrDirectSolve(ref A2, ref b2, ref xQR);
            AssertVecEq(in x, in xQR, LooseTol());

            var bsm = DenseToBSM1x1(ref arena, in A, m * n);
            var xBsm = arena.fProxyVec(n);
            bool okBsm = Solvers.lsqr(in bsm, in b, ref xBsm, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(okBsm);
            AssertVecEq(in x, in xBsm, LooseTol());

            arena.Dispose();
        }

        // ---- CGLS & LSQR on an underdetermined (m < n) CONSISTENT problem (nice-to-have) ---
        //
        // Wide A, b = A*x_gen (consistent) -> infinitely many exact solutions. Starting from
        // x0 = 0, both CGLS and LSQR converge to the SAME (minimum-norm) solution. We assert the
        // easy-to-verify properties -- A x ~= b and both solvers agree -- rather than the min-norm
        // optimality itself (that weaker check is all this nice-to-have case claims).
        void CglsLsqrUnderdeterminedConsistent()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 4, n = 10;
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, 38001);
            var xGen = arena.fProxyRandomVec(n, -1f, 1f, 38002);
            var b = Linear_OP.dot(A, xGen);      // consistent

            var xC = arena.fProxyVec(n);
            bool okC = Solvers.cgls(in A, in b, ref xC, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(okC);
            var AxC = Linear_OP.dot(A, xC);
            AssertVecEq(in AxC, in b, LooseTol());

            var xL = arena.fProxyVec(n);
            bool okL = Solvers.lsqr(in A, in b, ref xL, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(okL);
            var AxL = Linear_OP.dot(A, xL);
            AssertVecEq(in AxL, in b, LooseTol());

            // Both start from x0=0 -> both land on the unique minimum-norm solution -> they agree.
            AssertVecEq(in xC, in xL, LooseTol());

            arena.Dispose();
        }

        // =================================================================================
        // Phase 3 warm-start plumbing
        //
        // Every OTHER Phase-3 test starts from a zero-initialized x, so b - A*x0 == b always --
        // a regression that dropped the initial-residual subtraction (r = b - A*x0 silently
        // becoming r = b) would pass all of them. These four seed x with the ALREADY-converged
        // solution and re-solve with maxIterations=1: each solver computes r = b - A*x from the
        // CALLER-supplied x and checks it against tolerance in its PRE-LOOP check (minres ~L595,
        // biCGStab ~L797, cgls ~L999, lsqr ~L1175 of Solvers.fProxy.cs), so an already-converged x
        // must report true WITHOUT spending the single iteration -- and x must come back unchanged.
        // Mirrors the CG/PCG WarmStart test above.
        // =================================================================================

        // ---- MINRES warm start (SPD system, sufficient for the warm-start plumbing) ----
        void MinresWarmStart()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 12;
            var A = BuildDenseSPD(ref arena, dim, 41001);
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 41002);

            var x = arena.fProxyVec(dim);
            bool ok = Solvers.minres(in A, in b, ref x, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);

            // Seed the converged solution back; a single iteration's budget must still report
            // convergence via the pre-loop residual check, and leave x untouched.
            var xWarm = x.Copy();
            bool okWarm = Solvers.minres(in A, in b, ref xWarm, 1, Consts.fProxySqrtEps);
            Assert.IsTrue(okWarm);
            AssertVecEq(in x, in xWarm, LooseTol());

            arena.Dispose();
        }

        // ---- BiCGSTAB warm start (random diagonally-dominant non-symmetric A) ----
        void BiCGStabWarmStart()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;
            var A = arena.fProxyRandomMat(dim, dim, -1f, 1f, 41101);
            for (int i = 0; i < dim; i++)
                A[i, i] += (fProxy)(dim + 1);   // strict diagonal dominance

            var b = arena.fProxyRandomVec(dim, -1f, 1f, 41102);

            var x = arena.fProxyVec(dim);
            bool ok = Solvers.biCGStab(in A, in b, ref x, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);

            var xWarm = x.Copy();
            bool okWarm = Solvers.biCGStab(in A, in b, ref xWarm, 1, Consts.fProxySqrtEps);
            Assert.IsTrue(okWarm);
            AssertVecEq(in x, in xWarm, LooseTol());

            arena.Dispose();
        }

        // ---- CGLS warm start (overdetermined m>n CONSISTENT system, b = A*xTrue) ----
        void CglsWarmStart()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 10, n = 4;
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, 41201);
            var xTrue = arena.fProxyRandomVec(n, -1f, 1f, 41202);
            var b = Linear_OP.dot(A, xTrue);      // consistent

            var x = arena.fProxyVec(n);
            bool ok = Solvers.cgls(in A, in b, ref x, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);

            var xWarm = x.Copy();
            bool okWarm = Solvers.cgls(in A, in b, ref xWarm, 1, Consts.fProxySqrtEps);
            Assert.IsTrue(okWarm);
            AssertVecEq(in x, in xWarm, LooseTol());

            arena.Dispose();
        }

        // ---- LSQR warm start (overdetermined m>n CONSISTENT system, b = A*xTrue) ----
        void LsqrWarmStart()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 10, n = 4;
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, 41301);
            var xTrue = arena.fProxyRandomVec(n, -1f, 1f, 41302);
            var b = Linear_OP.dot(A, xTrue);      // consistent

            var x = arena.fProxyVec(n);
            bool ok = Solvers.lsqr(in A, in b, ref x, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);

            var xWarm = x.Copy();
            bool okWarm = Solvers.lsqr(in A, in b, ref xWarm, 1, Consts.fProxySqrtEps);
            Assert.IsTrue(okWarm);
            AssertVecEq(in x, in xWarm, LooseTol());

            arena.Dispose();
        }

        // ---- LSMR on an overdetermined CONSISTENT least-squares problem (dense + BSM) ------
        //
        // b = A*x_true exactly (b in range(A)) -> the least-squares solution is x_true, recovered
        // exactly (within tolerance). Same acceptance criterion as the LSQR twin.
        void LsmrOverdeterminedConsistentDenseAndBSM()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 10, n = 4;
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, 42001);
            var xTrue = arena.fProxyRandomVec(n, -1f, 1f, 42002);
            var b = Linear_OP.dot(A, xTrue);      // consistent

            var x = arena.fProxyVec(n);
            bool ok = Solvers.lsmr(in A, in b, ref x, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);
            AssertVecEq(in x, in xTrue, LooseTol());

            var Ax = Linear_OP.dot(A, x);
            AssertVecEq(in Ax, in b, LooseTol());

            var bsm = DenseToBSM1x1(ref arena, in A, m * n);
            var xBsm = arena.fProxyVec(n);
            bool okBsm = Solvers.lsmr(in bsm, in b, ref xBsm, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(okBsm);
            AssertVecEq(in x, in xBsm, LooseTol());

            arena.Dispose();
        }

        // ---- LSMR on an overdetermined INCONSISTENT problem: normal-equation optimality ----
        //
        // Random b generally NOT in range(A). Correct acceptance = ||A^T(A x - b)|| ~= 0, plus a
        // cross-check against the dense QR least-squares solution (the unique minimizer) -- the
        // same oracle as the CGLS/LSQR inconsistent tests, so LSMR must land on the same x.
        void LsmrInconsistentMatchesQR()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 10, n = 4;
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, 42101);
            var b = arena.fProxyRandomVec(m, -1f, 1f, 42102);   // inconsistent

            var x = arena.fProxyVec(n);
            bool ok = Solvers.lsmr(in A, in b, ref x, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);

            AssertLeastSquaresOptimal(in A, in x, in b, LooseTol());

            var A2 = A.Copy();
            var b2 = b.Copy();
            var xQR = arena.fProxyVec(n);
            QR.qrDirectSolve(ref A2, ref b2, ref xQR);
            AssertVecEq(in x, in xQR, LooseTol());

            var bsm = DenseToBSM1x1(ref arena, in A, m * n);
            var xBsm = arena.fProxyVec(n);
            bool okBsm = Solvers.lsmr(in bsm, in b, ref xBsm, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(okBsm);
            AssertVecEq(in x, in xBsm, LooseTol());

            arena.Dispose();
        }

        // ---- LSMR on an underdetermined (m < n) CONSISTENT problem: matches LSQR ----
        //
        // Wide A, b = A*x_gen (consistent) -> infinitely many exact solutions. From x0 = 0 both
        // LSMR and LSQR converge to the SAME minimum-norm solution, so assert A x ~= b and that
        // LSMR agrees with the (already-tested) LSQR solution.
        void LsmrUnderdeterminedMatchesLsqr()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 4, n = 10;
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, 42201);
            var xGen = arena.fProxyRandomVec(n, -1f, 1f, 42202);
            var b = Linear_OP.dot(A, xGen);      // consistent

            var xM = arena.fProxyVec(n);
            bool okM = Solvers.lsmr(in A, in b, ref xM, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(okM);
            var AxM = Linear_OP.dot(A, xM);
            AssertVecEq(in AxM, in b, LooseTol());

            var xL = arena.fProxyVec(n);
            bool okL = Solvers.lsqr(in A, in b, ref xL, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(okL);

            AssertVecEq(in xM, in xL, LooseTol());   // both land on the unique minimum-norm solution

            arena.Dispose();
        }

        // ---- LSMR warm start (overdetermined m>n CONSISTENT system, b = A*xTrue) ----
        void LsmrWarmStart()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 10, n = 4;
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, 42301);
            var xTrue = arena.fProxyRandomVec(n, -1f, 1f, 42302);
            var b = Linear_OP.dot(A, xTrue);      // consistent

            var x = arena.fProxyVec(n);
            bool ok = Solvers.lsmr(in A, in b, ref x, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);

            var xWarm = x.Copy();
            bool okWarm = Solvers.lsmr(in A, in b, ref xWarm, 1, Consts.fProxySqrtEps);
            Assert.IsTrue(okWarm);
            AssertVecEq(in x, in xWarm, LooseTol());

            arena.Dispose();
        }
    }

    // Deliberately non-SPD test-double preconditioner: z = M^-1 r := -r, so <r,z> = -||r||^2 <= 0.
    // Used only by PcgNonSpdPreconditionerBreaksDown to exercise pcg's rzold>0 breakdown guard.
    public struct fProxyNegatePreconditioner : IfProxyPreconditioner
    {
        public void Apply(in fProxyN r, ref fProxyN z)
        {
            for (int i = 0; i < r.N; i++)
                z[i] = -r[i];
        }
    }

    // ---- correctness cases (Burst) -------------------------------------------------------

    [Test]
    public void Laplacian1DBSMCGMatchesDenseCGTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.Laplacian1DBSMCGMatchesDenseCG }.Run();

    [Test]
    public void ThreeByThreeBlockSPDConvergesTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.ThreeByThreeBlockSPDConverges }.Run();

    [Test]
    public void DenseForwardingUnchangedTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.DenseForwardingUnchanged }.Run();

    [Test]
    public void PCGMatchesCGTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.PCGMatchesCG }.Run();

    [Test]
    public void PCGBeatsCGIllConditionedTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.PCGBeatsCGIllConditioned }.Run();

    [Test]
    public void BlockJacobiApplyHandComputedTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.BlockJacobiApplyHandComputed }.Run();

    [Test]
    public void WarmStartTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.WarmStart }.Run();

    [Test]
    public void PcgNonSpdPreconditionerBreaksDownTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.PcgNonSpdPreconditionerBreaksDown }.Run();

    // ---- Phase 3 correctness entry points ----

    [Test]
    public void MinresIndefiniteDenseAndBSMTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.MinresIndefiniteDenseAndBSM }.Run();

    [Test]
    public void MinresSpdMatchesCGTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.MinresSpdMatchesCG }.Run();

    [Test]
    public void BiCGStabNonSymmetricMatchesLUTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.BiCGStabNonSymmetricMatchesLU }.Run();

    [Test]
    public void CglsOverdeterminedConsistentDenseAndBSMTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.CglsOverdeterminedConsistentDenseAndBSM }.Run();

    [Test]
    public void LsqrOverdeterminedConsistentDenseAndBSMTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LsqrOverdeterminedConsistentDenseAndBSM }.Run();

    [Test]
    public void CglsInconsistentMatchesQRTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.CglsInconsistentMatchesQR }.Run();

    [Test]
    public void LsqrInconsistentMatchesQRTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LsqrInconsistentMatchesQR }.Run();

    [Test]
    public void CglsLsqrUnderdeterminedConsistentTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.CglsLsqrUnderdeterminedConsistent }.Run();

    // ---- LSMR entry points ----

    [Test]
    public void LsmrOverdeterminedConsistentDenseAndBSMTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LsmrOverdeterminedConsistentDenseAndBSM }.Run();

    [Test]
    public void LsmrInconsistentMatchesQRTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LsmrInconsistentMatchesQR }.Run();

    [Test]
    public void LsmrUnderdeterminedMatchesLsqrTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LsmrUnderdeterminedMatchesLsqr }.Run();

    // ---- Phase 3 warm-start entry points ----

    [Test]
    public void MinresWarmStartTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.MinresWarmStart }.Run();

    [Test]
    public void BiCGStabWarmStartTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.BiCGStabWarmStart }.Run();

    [Test]
    public void CglsWarmStartTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.CglsWarmStart }.Run();

    [Test]
    public void LsqrWarmStartTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LsqrWarmStart }.Run();

    [Test]
    public void LsmrWarmStartTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LsmrWarmStart }.Run();

    // ---- guard / exception cases (managed thread; Assert.Throws can't run inside Burst) ----

    static fProxyBSM BuildSquareBSM(ref Arena arena)
    {
        const int BR = 2, BC = 2;
        var builder = arena.fProxyBSMBuilder(2, 2, BR, BC, 2);
        builder.AddBlock(0, 0, arena.fProxyRandomMat(BR, BC, -1f, 1f, 6101));
        builder.AddBlock(1, 1, arena.fProxyRandomMat(BR, BC, -1f, 1f, 6102));
        return builder.ToBSM(ref arena);
    }

    [Test]
    public void BlockJacobi_MissingDiagonalBlock_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            const int BR = 2;
            // Only an off-diagonal block -- no block at (0,0) or (1,1).
            var builder = arena.fProxyBSMBuilder(2, 2, BR, BR, 1);
            builder.AddBlock(0, 1, arena.fProxyRandomMat(BR, BR, -1f, 1f, 6201));
            var A = builder.ToBSM(ref arena);

            Assert.Throws<ArgumentException>(() => arena.fProxyBlockJacobi(in A));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void BlockJacobi_NonSquareBSM_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            const int BR = 2, BC = 3;
            var builder = arena.fProxyBSMBuilder(2, 2, BR, BC, 1); // BR != BC
            builder.AddBlock(0, 0, arena.fProxyRandomMat(BR, BC, -1f, 1f, 6301));
            var A = builder.ToBSM(ref arena);

            Assert.Throws<ArgumentException>(() => arena.fProxyBlockJacobi(in A));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Cg_NonSquareDenseOperator_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(3, 4); // non-square
            var op = new fProxyDenseOperator(in A);
            var b = arena.fProxyVec(3);
            var x = arena.fProxyVec(4);
            var r = arena.fProxyVec(3);
            var p = arena.fProxyVec(3);
            var Ap = arena.fProxyVec(3);

            Assert.Throws<ArgumentException>(() =>
                Solvers.cg(in op, in b, ref x, ref r, ref p, ref Ap, 4, Consts.fProxySqrtEps));
        }
        finally { arena.Dispose(); }
    }

    // ---- scratch-aliasing guards for cg / pcg --------------------------------------------
    //
    // cg/pcg throw if ANY two of their vector arguments share a Data.Ptr (the elementwise axpy
    // scratch updates silently corrupt on aliasing rather than self-checking). The pairs below
    // are chosen to be ones NOT already caught by a downstream Apply/dot guard, so each proves
    // the up-front distinctness check is doing real work. The guard runs before any computation,
    // so the operator matrix's contents are irrelevant -- a bare square fProxyMat suffices for cg.

    [Test]
    public void Cg_AliasingRAndAp_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            const int dim = 4;
            var A = arena.fProxyMat(dim, dim); // square; guard fires before A is read
            var op = new fProxyDenseOperator(in A);
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 6501);
            var x = arena.fProxyVec(dim);
            var p = arena.fProxyVec(dim);
            var Ap = arena.fProxyVec(dim);
            var rAlias = Ap; // r aliases Ap (would turn r -= Ap into r -= r == 0: false convergence)
            Assert.Throws<ArgumentException>(() =>
                Solvers.cg(in op, in b, ref x, ref rAlias, ref p, ref Ap, dim, Consts.fProxySqrtEps));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Cg_AliasingRAndX_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            const int dim = 4;
            var A = arena.fProxyMat(dim, dim);
            var op = new fProxyDenseOperator(in A);
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 6511);
            var x = arena.fProxyVec(dim);
            var p = arena.fProxyVec(dim);
            var Ap = arena.fProxyVec(dim);
            var rAlias = x; // r aliases x (r.CopyFrom(b) would silently clobber the initial guess)
            Assert.Throws<ArgumentException>(() =>
                Solvers.cg(in op, in b, ref x, ref rAlias, ref p, ref Ap, dim, Consts.fProxySqrtEps));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Pcg_AliasingRAndX_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareBSM(ref arena);       // 4 x 4, both diagonal blocks present
            var M = arena.fProxyBlockJacobi(in A);
            int dim = A.M_Rows;
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 6521);
            var x = arena.fProxyVec(dim);
            var p = arena.fProxyVec(dim);
            var Ap = arena.fProxyVec(dim);
            var z = arena.fProxyVec(dim);
            var rAlias = x; // r aliases x
            Assert.Throws<ArgumentException>(() =>
                Solvers.pcg(in A, in M, in b, ref x, ref rAlias, ref p, ref Ap, ref z, dim, Consts.fProxySqrtEps));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Pcg_AliasingZAndX_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareBSM(ref arena);
            var M = arena.fProxyBlockJacobi(in A);
            int dim = A.M_Rows;
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 6531);
            var x = arena.fProxyVec(dim);
            var r = arena.fProxyVec(dim);
            var p = arena.fProxyVec(dim);
            var Ap = arena.fProxyVec(dim);
            var zAlias = x; // z aliases x (not caught by M.Apply's own r/z guard)
            Assert.Throws<ArgumentException>(() =>
                Solvers.pcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref zAlias, dim, Consts.fProxySqrtEps));
        }
        finally { arena.Dispose(); }
    }

    // x aliasing b: the final pair among {r,p,Ap,x,b} / {r,p,Ap,z,x,b}. Benign in the current loop
    // (b isn't reread after the initial residual), but the guard is documented as ALL-pairs-distinct
    // so it must still throw. xAlias is a struct-copy of b (shares Data.Ptr) -- passing them as two
    // distinct locals keeps b as `in` and x as `ref` without an in/ref same-variable conflict.
    [Test]
    public void Cg_AliasingXAndB_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            const int dim = 4;
            var A = arena.fProxyMat(dim, dim); // square; guard fires before A is read
            var op = new fProxyDenseOperator(in A);
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 6541);
            var xAlias = b; // x aliases b (struct copy shares Data.Ptr)
            var r = arena.fProxyVec(dim);
            var p = arena.fProxyVec(dim);
            var Ap = arena.fProxyVec(dim);
            Assert.Throws<ArgumentException>(() =>
                Solvers.cg(in op, in b, ref xAlias, ref r, ref p, ref Ap, dim, Consts.fProxySqrtEps));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Pcg_AliasingXAndB_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = BuildSquareBSM(ref arena);
            var M = arena.fProxyBlockJacobi(in A);
            int dim = A.M_Rows;
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 6551);
            var xAlias = b; // x aliases b (struct copy shares Data.Ptr)
            var r = arena.fProxyVec(dim);
            var p = arena.fProxyVec(dim);
            var Ap = arena.fProxyVec(dim);
            var z = arena.fProxyVec(dim);
            Assert.Throws<ArgumentException>(() =>
                Solvers.pcg(in A, in M, in b, ref xAlias, ref r, ref p, ref Ap, ref z, dim, Consts.fProxySqrtEps));
        }
        finally { arena.Dispose(); }
    }

    // ---- Phase 3 guard / aliasing cases (managed thread; Assert.Throws can't run in Burst) ----
    //
    // MINRES/BiCGSTAB/CGLS/LSQR share cg/pcg's "every vector argument must be a distinct buffer"
    // contract, enforced up front by RequireDistinctBuffers before any computation -- so the
    // operator matrix's contents are irrelevant and a bare zeroed fProxyMat suffices. 1-2 aliasing
    // cases per solver (matching Phase 2's coverage) prove the guard fires; exhaustive pairwise
    // coverage is not attempted.

    [Test]
    public void Minres_NonSquareDense_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.fProxyMat(3, 4); // non-square -> minres's A.Rows!=A.Cols guard fires first
            var b = arena.fProxyVec(3);
            var x = arena.fProxyVec(3);
            var y  = arena.fProxyVec(3); var r1 = arena.fProxyVec(3); var r2 = arena.fProxyVec(3);
            var v  = arena.fProxyVec(3); var w  = arena.fProxyVec(3);
            var w1 = arena.fProxyVec(3); var w2 = arena.fProxyVec(3);
            Assert.Throws<ArgumentException>(() =>
                Solvers.minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, 3, Consts.fProxySqrtEps));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Minres_AliasingR1AndR2_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            const int dim = 4;
            var A = arena.fProxyMat(dim, dim); // square; guard fires before A is read
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 39001);
            var x = arena.fProxyVec(dim);
            var y  = arena.fProxyVec(dim);
            var r1 = arena.fProxyVec(dim);
            var v  = arena.fProxyVec(dim);
            var w  = arena.fProxyVec(dim);
            var w1 = arena.fProxyVec(dim);
            var w2 = arena.fProxyVec(dim);
            var r2Alias = r1; // r2 aliases r1
            Assert.Throws<ArgumentException>(() =>
                Solvers.minres(in A, in b, ref x, ref y, ref r1, ref r2Alias, ref v, ref w, ref w1, ref w2, dim, Consts.fProxySqrtEps));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void BiCGStab_AliasingXAndB_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            const int dim = 4;
            var A = arena.fProxyMat(dim, dim);
            var b = arena.fProxyRandomVec(dim, -1f, 1f, 39101);
            var xAlias = b; // x aliases b (struct copy shares Data.Ptr)
            var r = arena.fProxyVec(dim); var rHat0 = arena.fProxyVec(dim); var p = arena.fProxyVec(dim);
            var v = arena.fProxyVec(dim); var t = arena.fProxyVec(dim);
            Assert.Throws<ArgumentException>(() =>
                Solvers.biCGStab(in A, in b, ref xAlias, ref r, ref rHat0, ref p, ref v, ref t, dim, Consts.fProxySqrtEps));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Cgls_AliasingRAndQ_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            int m = 5, n = 3;
            var A = arena.fProxyMat(m, n);      // rectangular; guard fires before A is read
            var b = arena.fProxyRandomVec(m, -1f, 1f, 39201);
            var x = arena.fProxyVec(n);
            var r = arena.fProxyVec(m);
            var s = arena.fProxyVec(n);
            var p = arena.fProxyVec(n);
            var qAlias = r; // q aliases r (both length m -> passes the dimension checks, trips the guard)
            Assert.Throws<ArgumentException>(() =>
                Solvers.cgls(in A, in b, ref x, ref r, ref s, ref p, ref qAlias, n, Consts.fProxySqrtEps));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Lsqr_AliasingUAndTmpM_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            int m = 5, n = 3;
            var A = arena.fProxyMat(m, n);
            var b = arena.fProxyRandomVec(m, -1f, 1f, 39301);
            var x = arena.fProxyVec(n);
            var u = arena.fProxyVec(m);
            var v = arena.fProxyVec(n);
            var w = arena.fProxyVec(n);
            var tmpN = arena.fProxyVec(n);
            var tmpMAlias = u; // tmpM aliases u (both length m)
            Assert.Throws<ArgumentException>(() =>
                Solvers.lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpMAlias, ref tmpN, n, Consts.fProxySqrtEps));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Lsmr_AliasingHAndHbar_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            int m = 5, n = 3;
            var A = arena.fProxyMat(m, n);
            var b = arena.fProxyRandomVec(m, -1f, 1f, 39401);
            var x = arena.fProxyVec(n);
            var u = arena.fProxyVec(m);
            var v = arena.fProxyVec(n);
            var h = arena.fProxyVec(n);
            var tmpM = arena.fProxyVec(m);
            var tmpN = arena.fProxyVec(n);
            var hbarAlias = h; // hbar aliases h (both length n)
            Assert.Throws<ArgumentException>(() =>
                Solvers.lsmr(in A, in b, ref x, ref u, ref v, ref h, ref hbarAlias, ref tmpM, ref tmpN, n, Consts.fProxySqrtEps));
        }
        finally { arena.Dispose(); }
    }
}
