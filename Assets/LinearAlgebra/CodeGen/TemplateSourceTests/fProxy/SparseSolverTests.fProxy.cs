using System;
using BULA;
using BULA.Gallery;
using BULA.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Phase-2 sparse-solver test suite: IfProxyLinearOperator / fProxyDenseOperator /
// fProxyBSROperator / fProxyBlockJacobi / Krylov.cg&lt;TOp&gt; / Krylov.cg&lt;TOp,TPre&gt;, plus
// the concrete cg convenience overloads for fProxyMxN and fProxyBSR. Every
// BSR system is cross-checked against the equivalent dense system (same pattern as
// fProxySparseBSRTests: build the SAME system in both forms and compare).
//
// Correctness cases run inside a [BurstCompile] IJob (matches fProxyConjugateGradientTests /
// fProxySparseBSRTests). Guard/exception cases run on the managed test thread with
// Assert.Throws, since NUnit's Assert.Throws cannot execute inside a Burst-compiled job.
public class fProxySparseSolverTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct SparseSolverTestJob : IJob
    {
        public enum TestType
        {
            Laplacian1DBSRCGMatchesDenseCG,
            ThreeByThreeBlockSPDConverges,
            DenseForwardingUnchanged,
            PCGMatchesCG,
            PCGBeatsCGIllConditioned,
            BlockJacobiApplyHandComputed,
            WarmStart,
            MergedCgIdentityMatchesPlainCg,

            // ---- Phase 3: MINRES / BiCGSTAB / LSQR / LSMR ----
            MinresIndefiniteDenseAndBSR,
            MinresSpdMatchesCG,
            BiCGStabNonSymmetricMatchesLU,
            LsqrOverdeterminedConsistentDenseAndBSR,
            LsqrInconsistentMatchesQR,
            LsqrUnderdeterminedConsistent,

            // ---- LSMR (Fong-Saunders): least-squares with monotone ||A^T r|| ----
            LsmrOverdeterminedConsistentDenseAndBSR,
            LsmrInconsistentMatchesQR,
            LsmrUnderdeterminedMatchesLsqr,
            LsmrMonotonicArnorm,

            // ---- Tikhonov damping (LSQR/LSMR): min ||Ax-b||^2 + damp^2||x||^2 ----
            TikhonovDampingMatchesAugmentedQR,

            // ---- LS diagnostics (LstsqInfo: rnorm/Arnorm/xnorm/iterations/status; free tracked estimates) ----
            LstsqInfoMatchesIndependentRecompute,
            LstsqInfoDampedArnorm,
            LsmrRnormMatchesExact,
            LstsqInfoBSRMatchesDense,

            // ---- AᵀA-Jacobi preconditioner: fewer iterations, same solution ----
            JacobiPreconditionerReducesIterations,
            JacobiConvenienceSolversLSOptimalDenseAndBSR,

            // ---- literature ground truth: Strang best-fit-line, EXACT diagnostics ----
            LstsqInfoStrangLineFitExact,

            // ---- square-solver diagnostics (SolveInfo: rnorm/iterations/status; free tracked ‖r‖) ----
            SolveInfoRnormMatchesResidual,

            // ---- Phase 3 warm-start plumbing (initial residual r = b - A*x0 from the CALLER's x) ----
            MinresWarmStart,
            BiCGStabWarmStart,
            LsqrWarmStart,
            LsmrWarmStart,
        }

        public TestType Type;

        // CG/PCG residuals converge to tolerance^2 * ||b||^2 (see Krylov.cg), so per-component
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
        // relative to the fixed scale ||A^T b|| (mirrors lsqr/lsmr's own convergence reference).
        // This -- NOT ||A x - b|| ~= 0 -- is the correct acceptance criterion for an inconsistent
        // (overdetermined) system, whose residual A x - b is left orthogonal to range(A), nonzero.
        static void AssertLeastSquaresOptimal(in fProxyMxN A, in fProxyN x, in fProxyN b, fProxy relTol)
        {
            var Ax  = Blas.dot(A, x);
            var res = new fProxyN(in Ax, Allocator.Temp);
            fProxyComp.subInPlace(res, b);    // r = A x - b     (length m)
            var atr = Blas.dot(res, A);  // A^T r           (length n)  -- vector*matrix == A^T r
            var atb = Blas.dot(b, A);    // A^T b           (scale reference)
            fProxy atrNorm = math.sqrt(Blas.dot(atr, atr));
            fProxy atbNorm = math.sqrt(Blas.dot(atb, atb));
            Assert.IsTrue(atrNorm <= relTol * atbNorm);
        }

        public void Execute()
        {
            switch (Type)
            {
                case TestType.Laplacian1DBSRCGMatchesDenseCG: Laplacian1DBSRCGMatchesDenseCG(); break;
                case TestType.ThreeByThreeBlockSPDConverges: ThreeByThreeBlockSPDConverges(); break;
                case TestType.DenseForwardingUnchanged: DenseForwardingUnchanged(); break;
                case TestType.PCGMatchesCG: PCGMatchesCG(); break;
                case TestType.PCGBeatsCGIllConditioned: PCGBeatsCGIllConditioned(); break;
                case TestType.BlockJacobiApplyHandComputed: BlockJacobiApplyHandComputed(); break;
                case TestType.WarmStart: WarmStart(); break;
                case TestType.MergedCgIdentityMatchesPlainCg: MergedCgIdentityMatchesPlainCg(); break;

                case TestType.MinresIndefiniteDenseAndBSR: MinresIndefiniteDenseAndBSR(); break;
                case TestType.MinresSpdMatchesCG: MinresSpdMatchesCG(); break;
                case TestType.BiCGStabNonSymmetricMatchesLU: BiCGStabNonSymmetricMatchesLU(); break;
                case TestType.LsqrOverdeterminedConsistentDenseAndBSR: LsqrOverdeterminedConsistentDenseAndBSR(); break;
                case TestType.LsqrInconsistentMatchesQR: LsqrInconsistentMatchesQR(); break;
                case TestType.LsqrUnderdeterminedConsistent: LsqrUnderdeterminedConsistent(); break;

                case TestType.LsmrOverdeterminedConsistentDenseAndBSR: LsmrOverdeterminedConsistentDenseAndBSR(); break;
                case TestType.LsmrInconsistentMatchesQR: LsmrInconsistentMatchesQR(); break;
                case TestType.LsmrUnderdeterminedMatchesLsqr: LsmrUnderdeterminedMatchesLsqr(); break;
                case TestType.LsmrMonotonicArnorm: LsmrMonotonicArnorm(); break;

                case TestType.TikhonovDampingMatchesAugmentedQR: TikhonovDampingMatchesAugmentedQR(); break;


                case TestType.LstsqInfoMatchesIndependentRecompute: LstsqInfoMatchesIndependentRecompute(); break;
                case TestType.LstsqInfoDampedArnorm: LstsqInfoDampedArnorm(); break;
                case TestType.LsmrRnormMatchesExact: LsmrRnormMatchesExact(); break;
                case TestType.LstsqInfoBSRMatchesDense: LstsqInfoBSRMatchesDense(); break;
                case TestType.JacobiPreconditionerReducesIterations: JacobiPreconditionerReducesIterations(); break;
                case TestType.JacobiConvenienceSolversLSOptimalDenseAndBSR: JacobiConvenienceSolversLSOptimalDenseAndBSR(); break;
                case TestType.LstsqInfoStrangLineFitExact: LstsqInfoStrangLineFitExact(); break;
                case TestType.SolveInfoRnormMatchesResidual: SolveInfoRnormMatchesResidual(); break;

                case TestType.MinresWarmStart: MinresWarmStart(); break;
                case TestType.BiCGStabWarmStart: BiCGStabWarmStart(); break;
                case TestType.LsqrWarmStart: LsqrWarmStart(); break;
                case TestType.LsmrWarmStart: LsmrWarmStart(); break;
            }
        }

        // ---- helpers ---------------------------------------------------------------------

        // Same recipe as fProxyConjugateGradientTests.BuildSPD: A = M^T M + dim*I (strictly SPD,
        // diagonally dominant).
        static fProxyMxN BuildDenseSPD(int dim, uint seed)
        {
            var M = GenerateOP.fProxyRandomMat(dim, dim, -1f, 1f, seed);
            var A = Blas.dot(M, M, true);
            for (int d = 0; d < dim; d++)
                A[d, d] += dim;
            return A;
        }

        // 1x1-block BSR built from a dense matrix's nonzero entries via AddValue. Triplet count
        // is bounded by the caller-supplied nnzHint (sized to the known nonzero pattern) purely
        // as a perf choice -- it avoids a few reallocations of the builder's internal growable
        // lists, nothing more. Growing the builder's lists past capacityHint (triggering one or
        // more UnsafeList reallocations) is safe: the builder's mutable triplet state lives
        // behind a single heap-allocated pointer shared by every value-copy of the struct, so a
        // reallocation on one copy is visible to all of them. See the growth regression tests in
        // SparseBSRTests.fProxy.cs, which build
        // via many-reallocation growth on purpose to prove this.
        static fProxyBSR DenseToBSR1x1(in fProxyMxN A, int nnzHint)
        {
            var builder = new fProxyBSRBuilder(A.M_Rows, A.N_Cols, 1, 1, Allocator.Temp, math.max(nnzHint, 1));
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    if (A[r, c] != (fProxy)0)
                        builder.AddValue(r, c, A[r, c]);
            return builder.ToBSR(Allocator.Temp);
        }

        static void AssertVecEq(in fProxyN a, in fProxyN b, fProxy tol)
        {
            var diff = new fProxyN(in a, Allocator.Temp);
            fProxyComp.subInPlace(diff, b);
            Assert.IsTrue(Analysis.isZero(diff, tol));
        }

        // got/expected are double so the result-struct norm fields (now double regardless of the
        // solve's precision) pass straight in; fProxy callers just widen. NOT a second fProxy
        // overload -- that would collide with this one in the double generation (CS0111).
        static void AssertClose(double got, double expected, fProxy tol)
            => Assert.IsTrue(math.abs(got - expected) <= tol * ((fProxy)1 + math.abs(expected)));

        // ---- 1. 1D Laplacian tridiagonal as a 1x1-block BSR: CG matches dense CG -----------
        void Laplacian1DBSRCGMatchesDenseCG()
        {
            int dim = 16;
            var A = fProxyGallery.fProxyLaplacian1D(dim);
            // Tridiagonal: at most 3 nonzeros/row -> 3*dim is a safe upper bound.
            var bsm = DenseToBSR1x1(in A, 3 * dim);

            var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 4242);

            var xDense = new fProxyN(dim, Allocator.Temp);
            bool okDense = Krylov.cg(in A, in b, ref xDense, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okDense);

            var xBsr = new fProxyN(dim, Allocator.Temp);
            bool okBsr = Krylov.cg(in bsm, in b, ref xBsr, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okBsr);

            AssertVecEq(in xDense, in xBsr, Tol());

            // A*x ~= b for the BSR solve too (spec's explicit acceptance criterion).
            var Ax = BSR.spMV(in bsm, in xBsr);
            AssertVecEq(in Ax, in b, Tol());
        }

        // ---- 2. 3x3-block SPD system: CG converges, residual within tol -------------------
        //
        // Build a random block matrix M (BR=3, a handful of blocks on a 3x3 block grid), form
        // the dense A = M^T M + eps*I (guaranteed SPD), then re-encode A as a genuine 3x3-block
        // BSR with every block-row/col pair stored (A^T A is generally dense even when M is
        // sparse) -- so CG genuinely walks a multi-block-per-row BSR structure, not just 1x1
        // scalars.
        void ThreeByThreeBlockSPDConverges()
        {
            const int BR = 3;
            const int nb = 3; // 9x9
            int dim = BR * nb;

            var mb = new fProxyBSRBuilder(nb, nb, BR, BR, Allocator.Temp, nb * nb);
            mb.AddBlock(0, 0, GenerateOP.fProxyRandomMat(BR, BR, -1f, 1f, 8001));
            mb.AddBlock(0, 1, GenerateOP.fProxyRandomMat(BR, BR, -1f, 1f, 8002));
            mb.AddBlock(1, 1, GenerateOP.fProxyRandomMat(BR, BR, -1f, 1f, 8003));
            mb.AddBlock(1, 2, GenerateOP.fProxyRandomMat(BR, BR, -1f, 1f, 8004));
            mb.AddBlock(2, 2, GenerateOP.fProxyRandomMat(BR, BR, -1f, 1f, 8005));
            mb.AddBlock(2, 0, GenerateOP.fProxyRandomMat(BR, BR, -1f, 1f, 8006));
            var Mdense = mb.ToBSR(Allocator.Temp).ToDense(Allocator.Temp);

            var A = Blas.dot(Mdense, Mdense, true);
            for (int i = 0; i < dim; i++)
                A[i, i] += dim;

            var ab = new fProxyBSRBuilder(nb, nb, BR, BR, Allocator.Temp, nb * nb);
            for (int bi = 0; bi < nb; bi++)
                for (int bj = 0; bj < nb; bj++)
                {
                    var blk = new fProxyMxN(BR, BR, Allocator.Temp);
                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BR; c++)
                            blk[r, c] = A[bi * BR + r, bj * BR + c];
                    ab.AddBlock(bi, bj, in blk);
                }
            var Absm = ab.ToBSR(Allocator.Temp);

            var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 8100);
            var x = new fProxyN(dim, Allocator.Temp);
            bool ok = Krylov.cg(in Absm, in b, ref x);
            Assert.IsTrue(ok);

            var Ax = BSR.spMV(in Absm, in x);
            AssertVecEq(in Ax, in b, Tol());

            // Cross-check against the dense reference too.
            var xDense = new fProxyN(dim, Allocator.Temp);
            bool okDense = Krylov.cg(in A, in b, ref xDense);
            Assert.IsTrue(okDense);
            AssertVecEq(in x, in xDense, Tol());
        }

        // ---- 3. Dense forwarding unchanged: guards the cg(in fProxyMxN,...) ----
        //         refactor into cg<fProxyDenseOperator> -----------------------------------------
        void DenseForwardingUnchanged()
        {
            int dim = 12;
            var A = BuildDenseSPD(dim, 90125);
            var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 4242);

            // The (unchanged, public) concrete entry point.
            var xConcrete = new fProxyN(dim, Allocator.Temp);
            bool okConcrete = Krylov.cg(in A, in b, ref xConcrete);
            Assert.IsTrue(okConcrete);

            // Calling cg<TOp> directly via fProxyDenseOperator must reproduce the identical
            // result -- this is the single source of truth the concrete overload now forwards
            // into.
            var op = new fProxyDenseOperator(in A);
            var xGeneric = new fProxyN(dim, Allocator.Temp);
            var r = new fProxyN(dim, Allocator.Temp);
            var p = new fProxyN(dim, Allocator.Temp);
            var Ap = new fProxyN(dim, Allocator.Temp);
            bool okGeneric = Krylov.cg(in op, in b, ref xGeneric, ref r, ref p, ref Ap, dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okGeneric);

            AssertVecEq(in xConcrete, in xGeneric, Tol());

            var Ax = Blas.dot(A, xConcrete);
            AssertVecEq(in Ax, in b, Tol());

            // Independent cross-check against a DIRECT solver on a completely different code path
            // (Householder QR, no Krylov/CG involvement). The xConcrete-vs-xGeneric check above is
            // circular now that both funnel through the same cg<TOp> loop; this pins the CG
            // solution to a truly independent reference. QR.solveInPlace is DESTRUCTIVE (destroys
            // A and b), so it MUST run on fresh copies, not the A/b the CG calls used.
            var A2 = A.Copy();
            var b2 = b.Copy();
            var xQR = new fProxyN(dim, Allocator.Temp);
            QR.solveInPlace(ref A2, ref b2, ref xQR);
            AssertVecEq(in xConcrete, in xQR, Tol());
        }

        // ---- 4. PCG correctness: matches CG's solution -------------------------------------
        void PCGMatchesCG()
        {
            int dim = 12;
            var A = BuildDenseSPD(dim, 6001);
            var bsm = DenseToBSR1x1(in A, dim * dim);
            var M = new fProxyBlockJacobi(in bsm, Allocator.Temp);
            var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 6002);

            var xCG = new fProxyN(dim, Allocator.Temp);
            bool okCG = Krylov.cg(in bsm, in b, ref xCG);
            Assert.IsTrue(okCG);

            var xPCG = new fProxyN(dim, Allocator.Temp);
            bool okPCG = Krylov.cg(in bsm, in M, in b, ref xPCG);
            Assert.IsTrue(okPCG);

            AssertVecEq(in xCG, in xPCG, Tol());
        }

        // The merged single-body cg<TOp,TPre> with the identity preconditioner must be BIT-IDENTICAL
        // to the hand-written plain cg<TOp> -- same solution, same iteration count, same status, same
        // rnorm -- on the same operator path (both wrap the BSR in fProxyBSROperator). Proves the
        // IsIdentity fold reproduces plain CG exactly (numerics AND control flow), so collapsing the
        // two bodies is safe. z is passed as `default` on the identity path (never dereferenced).
        void MergedCgIdentityMatchesPlainCg()
        {
            int dim = 12;
            var A = BuildDenseSPD(dim, 6101);
            var bsm = DenseToBSR1x1(in A, dim * dim);
            var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 6102);

            int maxIter = 4 * dim;
            fProxy tol = Consts.fProxySqrtEps;

            // Reference: plain cg<TOp> (explicit scratch).
            var xPlain = new fProxyN(dim, Allocator.Temp);
            var rP = new fProxyN(dim, Allocator.Temp); var pP = new fProxyN(dim, Allocator.Temp); var ApP = new fProxyN(dim, Allocator.Temp);
            var infoPlain = Krylov.cg(in bsm, in b, ref xPlain, ref rP, ref pP, ref ApP, maxIter, tol);

            // Merged: cg<TOp,TPre> with the identity preconditioner; z = default (unused when identity).
            var xMerged = new fProxyN(dim, Allocator.Temp);
            var rM = new fProxyN(dim, Allocator.Temp); var pM = new fProxyN(dim, Allocator.Temp); var ApM = new fProxyN(dim, Allocator.Temp);
            fProxyN zM = default;
            var op = new fProxyBSROperator(in bsm);
            var id = new fProxyIdentityPreconditioner();
            var infoMerged = Krylov.cg(in op, in id, in b, ref xMerged, ref rM, ref pM, ref ApM, ref zM, maxIter, tol);

            // Bit-identical solution.
            Assert.AreEqual(xPlain.N, xMerged.N);
            for (int i = 0; i < xPlain.N; i++)
                Assert.AreEqual((double)xPlain[i], (double)xMerged[i]);

            // Identical control flow + diagnostics.
            Assert.AreEqual(infoPlain.iterations, infoMerged.iterations);
            Assert.AreEqual((int)infoPlain.status, (int)infoMerged.status);
            Assert.AreEqual(infoPlain.rnorm, infoMerged.rnorm);
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
            int dim = 16;
            var S = BuildDenseSPD(dim, 7001);

            var Sym = new fProxyMxN(dim, dim, Allocator.Temp);
            for (int r = 0; r < dim; r++)
                for (int c = 0; c < dim; c++)
                    Sym[r, c] = S[r, c] / math.sqrt(S[r, r] * S[c, c]);

            var A = new fProxyMxN(dim, dim, Allocator.Temp);
            var d = new fProxyN(dim, Allocator.Temp);
            for (int i = 0; i < dim; i++)
                d[i] = (i % 2 == 0) ? (fProxy)1 : (fProxy)100; // alternating 1,100

            for (int r = 0; r < dim; r++)
                for (int c = 0; c < dim; c++)
                    A[r, c] = d[r] * Sym[r, c] * d[c];

            var bsm = DenseToBSR1x1(in A, dim * dim);
            var M = new fProxyBlockJacobi(in bsm, Allocator.Temp);
            var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 7002);

            int maxBudget = 4 * dim;
            int minCG = -1, minPCG = -1;
            for (int budget = 1; budget <= maxBudget && (minCG < 0 || minPCG < 0); budget++)
            {
                if (minCG < 0)
                {
                    var xCG = new fProxyN(dim, Allocator.Temp);
                    if (Krylov.cg(in bsm, in b, ref xCG, budget, Consts.fProxySqrtEps))
                        minCG = budget;
                }
                if (minPCG < 0)
                {
                    var xPCG = new fProxyN(dim, Allocator.Temp);
                    if (Krylov.cg(in bsm, in M, in b, ref xPCG, budget, Consts.fProxySqrtEps))
                        minPCG = budget;
                }
            }

            Assert.IsTrue(minCG > 0);
            Assert.IsTrue(minPCG > 0);
            Assert.IsTrue(minPCG <= minCG);
        }

        // ---- 6. Block-Jacobi Apply matches a hand-computed block-diagonal inverse ----------
        void BlockJacobiApplyHandComputed()
        {
            const int BR = 2;
            var builder = new fProxyBSRBuilder(2, 2, BR, BR, Allocator.Temp, 2);
            var d0 = new fProxyMxN(BR, BR, Allocator.Temp);
            d0[0, 0] = 4f; d0[0, 1] = 1f;
            d0[1, 0] = 1f; d0[1, 1] = 3f;
            var d1 = new fProxyMxN(BR, BR, Allocator.Temp);
            d1[0, 0] = 2f; d1[0, 1] = 0f;
            d1[1, 0] = 0f; d1[1, 1] = 5f;
            builder.AddBlock(0, 0, in d0);
            builder.AddBlock(1, 1, in d1);
            var A = builder.ToBSR(Allocator.Temp);

            var M = new fProxyBlockJacobi(in A, Allocator.Temp);

            var r = new fProxyN(4, Allocator.Temp);
            r[0] = 1f; r[1] = 2f; r[2] = 3f; r[3] = 4f;
            var z = new fProxyN(4, Allocator.Temp);
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
        }

        // ---- 7. Warm start: seeding x with the exact solution converges immediately --------
        void WarmStart()
        {
            int dim = 10;
            var A = BuildDenseSPD(dim, 9001);
            var bsm = DenseToBSR1x1(in A, dim * dim);
            var M = new fProxyBlockJacobi(in bsm, Allocator.Temp);
            var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 9002);

            var x = new fProxyN(dim, Allocator.Temp);
            bool ok = Krylov.cg(in bsm, in M, in b, ref x);
            Assert.IsTrue(ok);

            // Feed the converged solution back as the initial guess -- a single iteration's
            // worth of budget must still report convergence (matches
            // fProxyConjugateGradientTests.AlreadyConverged's dense-CG counterpart).
            var xWarm = x.Copy();
            bool okWarm = Krylov.cg(in bsm, in M, in b, ref xWarm, 1, Consts.fProxySqrtEps);
            Assert.IsTrue(okWarm);
            AssertVecEq(in x, in xWarm, Tol());

            // Same check for plain (unpreconditioned) CG.
            var xCG = new fProxyN(dim, Allocator.Temp);
            bool okCG = Krylov.cg(in bsm, in b, ref xCG);
            Assert.IsTrue(okCG);
            var xCGWarm = xCG.Copy();
            bool okCGWarm = Krylov.cg(in bsm, in b, ref xCGWarm, 1, Consts.fProxySqrtEps);
            Assert.IsTrue(okCGWarm);
            AssertVecEq(in xCG, in xCGWarm, Tol());
        }

        // Note: the old "non-SPD preconditioner breaks down" case was removed when IfProxyPreconditioner
        // gained IsSpd/IsConstant -- cg now REJECTS a non-SPD preconditioner at entry (ArgumentException),
        // covered by fProxyPreconditionerCompatibilityTests.CgRejectsIlu0 on the managed thread. cg's
        // downstream rzold>0 guard remains as a defensive net for a marked-SPD-but-numerically-indefinite M.

        // =================================================================================
        // Phase 3 correctness cases
        // =================================================================================

        // ---- MINRES on a symmetric INDEFINITE system (dense + BSR agree) -----------------
        //
        // Laplacian1D (SPD, diag 2 / off-diag -1) shifted by -2 on the diagonal: eigenvalues
        // become 2-2cos(k*pi/(n+1)) - 2 = -2cos(k*pi/(n+1)) for k=1..n, which straddle 0 -> a
        // genuinely mixed-sign (symmetric indefinite) A. dim=16 -> n+1=17 is odd, so k=(n+1)/2
        // is non-integer and NO eigenvalue is exactly 0 (A stays nonsingular). MINRES handles
        // this cleanly where CG's p.Ap>0 curvature requirement would break down.
        void MinresIndefiniteDenseAndBSR()
        {
            int dim = 16;
            var A = fProxyGallery.fProxyLaplacian1D(dim);
            for (int i = 0; i < dim; i++)
                A[i, i] -= (fProxy)2;          // shift diag 2 -> 0: mixed-sign spectrum, indefinite

            var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 31001);

            var xDense = new fProxyN(dim, Allocator.Temp);
            bool okDense = Krylov.minres(in A, in b, ref xDense, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okDense);

            var Ax = Blas.dot(A, xDense);
            AssertVecEq(in Ax, in b, LooseTol());          // A nonsingular -> unique solution, A x ~= b

            // Independent cross-check against a DIRECT LU solve on the SAME indefinite matrix
            // (no Krylov/MINRES involvement) -- pins the iterative solution to a truly independent
            // reference, not just the self-consistent A x ~= b residual. LU.decompInPlace +
            // LU.decompSolve are DESTRUCTIVE, so they run on COPIES. The shifted Laplacian above is
            // constructed to be nonsingular (odd n+1 -> no exactly-zero eigenvalue), so LU succeeds.
            var LUcopy = A.Copy();
            var pivot = new Pivot(dim, Allocator.Temp);
            bool okLU = LU.decompInPlace(ref LUcopy, ref pivot);
            Assert.IsTrue(okLU);
            var xLU = b.Copy();
            LU.decompSolve(ref LUcopy, in pivot, ref xLU);
            AssertVecEq(in xDense, in xLU, LooseTol());
            pivot.Dispose();

            // Same system as a 1x1-block BSR: minres(BSR) must agree with minres(dense).
            var bsm = DenseToBSR1x1(in A, 3 * dim);   // tridiagonal (shifted diag=0 dropped)
            var xBsr = new fProxyN(dim, Allocator.Temp);
            bool okBsr = Krylov.minres(in bsm, in b, ref xBsr, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okBsr);
            AssertVecEq(in xDense, in xBsr, LooseTol());

            var AxBsr = BSR.spMV(in bsm, in xBsr);
            AssertVecEq(in AxBsr, in b, LooseTol());

            // NOTE (spec nice-to-have, NOT asserted): plain CG on this SAME indefinite A breaks
            // down -- Krylov.cg's p.Ap>0 curvature guard fails / returns a much
            // worse residual. MINRES succeeding where CG cannot is the whole point of this case;
            // asserting CG's failure mode is fiddly and left as a documented expectation.
        }

        // ---- MINRES on a plain SPD system agrees with CG (dense + BSR) --------------------
        void MinresSpdMatchesCG()
        {
            int dim = 12;
            var A = BuildDenseSPD(dim, 32001);
            var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 32002);

            var xCG = new fProxyN(dim, Allocator.Temp);
            bool okCG = Krylov.cg(in A, in b, ref xCG, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okCG);

            var xMin = new fProxyN(dim, Allocator.Temp);
            bool okMin = Krylov.minres(in A, in b, ref xMin, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okMin);
            AssertVecEq(in xCG, in xMin, LooseTol());       // MINRES == CG on an SPD system

            var Ax = Blas.dot(A, xMin);
            AssertVecEq(in Ax, in b, LooseTol());

            // BSR minres agrees with dense minres.
            var bsm = DenseToBSR1x1(in A, dim * dim);
            var xMinBsr = new fProxyN(dim, Allocator.Temp);
            bool okMinBsr = Krylov.minres(in bsm, in b, ref xMinBsr, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okMinBsr);
            AssertVecEq(in xMin, in xMinBsr, LooseTol());
        }

        // ---- BiCGSTAB on a NON-symmetric (diagonally-dominant) system --------------------
        //
        // Random off-diagonals in [-1,1], diagonal boosted to dim+1 so |A_ii| > sum_{j!=i}|A_ij|
        // strictly -> nonsingular, unconditionally BiCGSTAB-friendly, and deliberately NOT
        // symmetrized. Cross-checked against a dense DIRECT LU solve on the SAME matrix.
        void BiCGStabNonSymmetricMatchesLU()
        {
            int dim = 8;
            var A = GenerateOP.fProxyRandomMat(dim, dim, -1f, 1f, 33001);
            for (int i = 0; i < dim; i++)
                A[i, i] += (fProxy)(dim + 1);   // strict diagonal dominance

            var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 33002);

            var xBcg = new fProxyN(dim, Allocator.Temp);
            bool okBcg = Krylov.biCGStab(in A, in b, ref xBcg, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okBcg);
            var Ax = Blas.dot(A, xBcg);
            AssertVecEq(in Ax, in b, LooseTol());

            // Direct LU reference on COPIES (LU.decompInPlace + LU.decompSolve are DESTRUCTIVE).
            var LUcopy = A.Copy();
            var pivot = new Pivot(dim, Allocator.Temp);
            bool okLU = LU.decompInPlace(ref LUcopy, ref pivot);
            Assert.IsTrue(okLU);
            var xLU = b.Copy();
            LU.decompSolve(ref LUcopy, in pivot, ref xLU);
            AssertVecEq(in xBcg, in xLU, LooseTol());
            pivot.Dispose();

            // BSR form agrees with the dense BiCGSTAB solve.
            var bsm = DenseToBSR1x1(in A, dim * dim);
            var xBcgBsr = new fProxyN(dim, Allocator.Temp);
            bool okBcgBsr = Krylov.biCGStab(in bsm, in b, ref xBcgBsr, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(okBcgBsr);
            AssertVecEq(in xBcg, in xBcgBsr, LooseTol());
        }

        // ---- LSQR on an overdetermined CONSISTENT least-squares problem (dense + BSR) ------
        void LsqrOverdeterminedConsistentDenseAndBSR()
        {
            int m = 10, n = 4;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 35001);
            var xTrue = GenerateOP.fProxyRandomVec(n, -1f, 1f, 35002);
            var b = Blas.dot(A, xTrue);      // consistent

            var x = new fProxyN(n, Allocator.Temp);
            bool ok = Krylov.lsqr(in A, in b, ref x, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);
            AssertVecEq(in x, in xTrue, LooseTol());

            var Ax = Blas.dot(A, x);
            AssertVecEq(in Ax, in b, LooseTol());

            var bsm = DenseToBSR1x1(in A, m * n);
            var xBsr = new fProxyN(n, Allocator.Temp);
            bool okBsr = Krylov.lsqr(in bsm, in b, ref xBsr, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(okBsr);
            AssertVecEq(in x, in xBsr, LooseTol());
        }

        // ---- LSQR on an overdetermined INCONSISTENT problem: normal-equation optimality ----
        void LsqrInconsistentMatchesQR()
        {
            int m = 10, n = 4;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 37001);
            var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 37002);   // inconsistent

            var x = new fProxyN(n, Allocator.Temp);
            bool ok = Krylov.lsqr(in A, in b, ref x, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);

            AssertLeastSquaresOptimal(in A, in x, in b, LooseTol());

            var A2 = A.Copy();
            var b2 = b.Copy();
            var xQR = new fProxyN(n, Allocator.Temp);
            QR.solveInPlace(ref A2, ref b2, ref xQR);
            AssertVecEq(in x, in xQR, LooseTol());

            var bsm = DenseToBSR1x1(in A, m * n);
            var xBsr = new fProxyN(n, Allocator.Temp);
            bool okBsr = Krylov.lsqr(in bsm, in b, ref xBsr, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(okBsr);
            AssertVecEq(in x, in xBsr, LooseTol());
        }

        // ---- LSQR on an underdetermined (m < n) CONSISTENT problem (min-norm, nice-to-have) ---
        //
        // Wide A, b = A*x_gen (consistent) -> infinitely many exact solutions. From x0 = 0 LSQR
        // converges to the (minimum-norm) solution; assert the easy-to-verify property A x ~= b.
        void LsqrUnderdeterminedConsistent()
        {
            int m = 4, n = 10;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 38001);
            var xGen = GenerateOP.fProxyRandomVec(n, -1f, 1f, 38002);
            var b = Blas.dot(A, xGen);      // consistent

            var xL = new fProxyN(n, Allocator.Temp);
            bool okL = Krylov.lsqr(in A, in b, ref xL, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(okL);
            var AxL = Blas.dot(A, xL);
            AssertVecEq(in AxL, in b, LooseTol());
        }

        // =================================================================================
        // Phase 3 warm-start plumbing
        //
        // Every OTHER Phase-3 test starts from a zero-initialized x, so b - A*x0 == b always --
        // a regression that dropped the initial-residual subtraction (r = b - A*x0 silently
        // becoming r = b) would pass all of them. These four seed x with the ALREADY-converged
        // solution and re-solve with maxIter=1: each solver computes r = b - A*x from the
        // CALLER-supplied x and checks it against tolerance in its own pre-loop residual check
        // (Krylov.fProxy.cs), so an already-converged x
        // must report true WITHOUT spending the single iteration -- and x must come back unchanged.
        // Mirrors the CG/PCG WarmStart test above.
        // =================================================================================

        // ---- MINRES warm start (SPD system, sufficient for the warm-start plumbing) ----
        void MinresWarmStart()
        {
            int dim = 12;
            var A = BuildDenseSPD(dim, 41001);
            var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 41002);

            var x = new fProxyN(dim, Allocator.Temp);
            bool ok = Krylov.minres(in A, in b, ref x, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);

            // Seed the converged solution back; a single iteration's budget must still report
            // convergence via the pre-loop residual check, and leave x untouched.
            var xWarm = x.Copy();
            bool okWarm = Krylov.minres(in A, in b, ref xWarm, 1, Consts.fProxySqrtEps);
            Assert.IsTrue(okWarm);
            AssertVecEq(in x, in xWarm, LooseTol());
        }

        // ---- BiCGSTAB warm start (random diagonally-dominant non-symmetric A) ----
        void BiCGStabWarmStart()
        {
            int dim = 8;
            var A = GenerateOP.fProxyRandomMat(dim, dim, -1f, 1f, 41101);
            for (int i = 0; i < dim; i++)
                A[i, i] += (fProxy)(dim + 1);   // strict diagonal dominance

            var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 41102);

            var x = new fProxyN(dim, Allocator.Temp);
            bool ok = Krylov.biCGStab(in A, in b, ref x, 4 * dim, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);

            var xWarm = x.Copy();
            bool okWarm = Krylov.biCGStab(in A, in b, ref xWarm, 1, Consts.fProxySqrtEps);
            Assert.IsTrue(okWarm);
            AssertVecEq(in x, in xWarm, LooseTol());
        }

        // ---- LSQR warm start (overdetermined m>n CONSISTENT system, b = A*xTrue) ----
        void LsqrWarmStart()
        {
            int m = 10, n = 4;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 41301);
            var xTrue = GenerateOP.fProxyRandomVec(n, -1f, 1f, 41302);
            var b = Blas.dot(A, xTrue);      // consistent

            var x = new fProxyN(n, Allocator.Temp);
            bool ok = Krylov.lsqr(in A, in b, ref x, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);

            var xWarm = x.Copy();
            bool okWarm = Krylov.lsqr(in A, in b, ref xWarm, 1, Consts.fProxySqrtEps);
            Assert.IsTrue(okWarm);
            AssertVecEq(in x, in xWarm, LooseTol());
        }

        // ---- LSMR on an overdetermined CONSISTENT least-squares problem (dense + BSR) ------
        //
        // Same fixture and acceptance criterion as LsqrOverdeterminedConsistentDenseAndBSR above.
        void LsmrOverdeterminedConsistentDenseAndBSR()
        {
            int m = 10, n = 4;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 42001);
            var xTrue = GenerateOP.fProxyRandomVec(n, -1f, 1f, 42002);
            var b = Blas.dot(A, xTrue);      // consistent

            var x = new fProxyN(n, Allocator.Temp);
            bool ok = Krylov.lsmr(in A, in b, ref x, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);
            AssertVecEq(in x, in xTrue, LooseTol());

            var Ax = Blas.dot(A, x);
            AssertVecEq(in Ax, in b, LooseTol());

            var bsm = DenseToBSR1x1(in A, m * n);
            var xBsr = new fProxyN(n, Allocator.Temp);
            bool okBsr = Krylov.lsmr(in bsm, in b, ref xBsr, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(okBsr);
            AssertVecEq(in x, in xBsr, LooseTol());
        }

        // ---- LSMR on an overdetermined INCONSISTENT problem: normal-equation optimality ----
        //
        // Random b generally NOT in range(A). Correct acceptance = ||A^T(A x - b)|| ~= 0, plus a
        // cross-check against the dense QR least-squares solution (the unique minimizer) -- the
        // same oracle as the LSQR inconsistent test, so LSMR must land on the same x.
        void LsmrInconsistentMatchesQR()
        {
            int m = 10, n = 4;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 42101);
            var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 42102);   // inconsistent

            var x = new fProxyN(n, Allocator.Temp);
            bool ok = Krylov.lsmr(in A, in b, ref x, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);

            AssertLeastSquaresOptimal(in A, in x, in b, LooseTol());

            var A2 = A.Copy();
            var b2 = b.Copy();
            var xQR = new fProxyN(n, Allocator.Temp);
            QR.solveInPlace(ref A2, ref b2, ref xQR);
            AssertVecEq(in x, in xQR, LooseTol());

            var bsm = DenseToBSR1x1(in A, m * n);
            var xBsr = new fProxyN(n, Allocator.Temp);
            bool okBsr = Krylov.lsmr(in bsm, in b, ref xBsr, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(okBsr);
            AssertVecEq(in x, in xBsr, LooseTol());
        }

        // ---- LSMR on an underdetermined (m < n) CONSISTENT problem: matches LSQR ----
        //
        // Same wide-A min-norm setup as LsqrUnderdeterminedConsistent above; here LSMR is
        // cross-checked against the already-tested LSQR solution.
        void LsmrUnderdeterminedMatchesLsqr()
        {
            int m = 4, n = 10;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 42201);
            var xGen = GenerateOP.fProxyRandomVec(n, -1f, 1f, 42202);
            var b = Blas.dot(A, xGen);      // consistent

            var xM = new fProxyN(n, Allocator.Temp);
            bool okM = Krylov.lsmr(in A, in b, ref xM, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(okM);
            var AxM = Blas.dot(A, xM);
            AssertVecEq(in AxM, in b, LooseTol());

            var xL = new fProxyN(n, Allocator.Temp);
            bool okL = Krylov.lsqr(in A, in b, ref xL, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(okL);

            AssertVecEq(in xM, in xL, LooseTol());   // both land on the unique minimum-norm solution
        }

        // ---- LSMR warm start (overdetermined m>n CONSISTENT system, b = A*xTrue) ----
        void LsmrWarmStart()
        {
            int m = 10, n = 4;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 42301);
            var xTrue = GenerateOP.fProxyRandomVec(n, -1f, 1f, 42302);
            var b = Blas.dot(A, xTrue);      // consistent

            var x = new fProxyN(n, Allocator.Temp);
            bool ok = Krylov.lsmr(in A, in b, ref x, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);

            var xWarm = x.Copy();
            bool okWarm = Krylov.lsmr(in A, in b, ref xWarm, 1, Consts.fProxySqrtEps);
            Assert.IsTrue(okWarm);
            AssertVecEq(in x, in xWarm, LooseTol());
        }

        // ---- LSMR monotone ||A^T r|| (its distinguishing property vs LSQR) ----
        //
        // LSMR guarantees ||A^T r_k|| decreases MONOTONICALLY. Run it for k = 1..n iterations (fresh
        // zero start each time, tolerance 0 so it never early-stops and runs EXACTLY k iterations),
        // recompute the true ||A^T(A x_k - b)|| externally, and assert the sequence is non-increasing.
        // A recurrence bug that swapped/mis-scaled the MINRES-layer rotation could still converge to
        // the right final x (self-correcting over enough steps) while violating monotonicity -- this
        // pins the property the QR cross-check cannot see. Generous fp slack: only a GROSS violation
        // (a real bug) exceeds it.
        void LsmrMonotonicArnorm()
        {
            int m = 12, n = 6;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 43101);
            var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 43102);   // inconsistent -> ||A^T r|| > 0 for a while

            fProxy prev = fProxy.MaxValue;
            for (int k = 1; k <= n; k++)
            {
                var x = new fProxyN(n, Allocator.Temp);                     // fresh zero start
                Krylov.lsmr(in A, in b, ref x, k, (fProxy)0);  // tol 0 -> runs exactly k iterations

                var Ax = Blas.dot(A, x);
                var res = new fProxyN(in Ax, Allocator.Temp);
                fProxyComp.subInPlace(res, b);             // res = A x - b   (length m)
                var atr = Blas.dot(res, A);                // A^T res   (length n)
                fProxy nrm = math.sqrt(Blas.dot(atr, atr));

                Assert.IsTrue(nrm <= prev + (fProxy)0.02 * prev + (fProxy)1e-4);   // non-increasing (+ fp slack)
                prev = nrm;
            }
        }

        // ---- Tikhonov damping: LSQR / LSMR both solve min ||Ax-b||^2 + damp^2||x||^2 ----
        //
        // Reference = dense QR least-squares on the AUGMENTED system [A; damp*I] x ~= [b; 0], which
        // IS the damped minimizer x = (A^T A + damp^2 I)^-1 A^T b. All three damped solvers (dense
        // AND 1x1-BSR) must land on it. Uses an INCONSISTENT b so damping actually changes the
        // answer (a wrong/no-op damp term would miss the reference). damp = 0.5 is well inside the
        // regime where the regularization is numerically significant but the system stays solvable.
        void TikhonovDampingMatchesAugmentedQR()
        {
            int m = 10, n = 4;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 43201);
            var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 43202);   // inconsistent
            fProxy damp = (fProxy)0.5;

            var xref = DampedReferenceQR(in A, in b, damp);

            // dense damped solvers
            var xL = new fProxyN(n, Allocator.Temp);
            Assert.IsTrue(Krylov.lsqr(in A, in b, ref xL, 16 * n, Consts.fProxySqrtEps, damp));
            AssertVecEq(in xL, in xref, LooseTol());

            var xM = new fProxyN(n, Allocator.Temp);
            Assert.IsTrue(Krylov.lsmr(in A, in b, ref xM, 16 * n, Consts.fProxySqrtEps, damp));
            AssertVecEq(in xM, in xref, LooseTol());

            // 1x1-BSR damped solvers agree with the same reference
            var bsm = DenseToBSR1x1(in A, m * n);

            var xLb = new fProxyN(n, Allocator.Temp);
            Assert.IsTrue(Krylov.lsqr(in bsm, in b, ref xLb, 16 * n, Consts.fProxySqrtEps, damp));
            AssertVecEq(in xLb, in xref, LooseTol());

            var xMb = new fProxyN(n, Allocator.Temp);
            Assert.IsTrue(Krylov.lsmr(in bsm, in b, ref xMb, 16 * n, Consts.fProxySqrtEps, damp));
            AssertVecEq(in xMb, in xref, LooseTol());
        }

        // ================== LS diagnostics (LstsqInfo) ==================

        // Independently recompute rnorm/Arnorm/xnorm EXACTLY from the returned x (paying a real
        // matvec) and assert the solver's FREE tracked estimates in the info struct match to within
        // LooseTol at convergence. r = b - A x, and A^T(A x - b) = -A^T r so ||A^T(Ax-b)|| == Arnorm;
        // for damp!=0 the reported optimality residual is ||A^T r - damp^2 x|| = ||A^T(Ax-b) + damp^2 x||.
        static void AssertInfoMatches(in fProxyMxN A, in fProxyN b, in fProxyN x, fProxy damp,
                                      in LstsqInfo info, fProxy tol)
        {
            var Ax  = Blas.dot(A, x);
            var res = new fProxyN(in Ax, Allocator.Temp);
            fProxyComp.subInPlace(res, b);            // Ax - b  (= -r), length m
            fProxy rnorm = math.sqrt(Blas.dot(res, res));
            Assert.IsTrue(math.abs(rnorm - info.rnorm) <= tol * ((fProxy)1 + rnorm));

            var g = Blas.dot(res, A);           // A^T(Ax - b), length n  (vector*matrix)
            if (damp != (fProxy)0)
                for (int i = 0; i < g.N; i++) g[i] += (damp * damp) * x[i];   // + damp^2 x
            fProxy arnorm = math.sqrt(Blas.dot(g, g));
            Assert.IsTrue(math.abs(arnorm - info.Arnorm) <= tol * ((fProxy)1 + arnorm));

            fProxy xnorm = math.sqrt(Blas.dot(x, x));
            Assert.IsTrue(math.abs(xnorm - info.xnorm) <= tol * ((fProxy)1 + xnorm));
        }

        // Diagnostics fields match an independent recompute across all three LS solvers, on an
        // INCONSISTENT over-determined system (so rnorm is meaningfully nonzero while Arnorm -> 0).
        void LstsqInfoMatchesIndependentRecompute()
        {
            int m = 12, n = 5;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 51001);
            var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 51002);   // random rhs -> inconsistent
            int maxIter = 8 * n;

            var xL = new fProxyN(n, Allocator.Temp);
            var infoL = Krylov.lsqr(in A, in b, ref xL, maxIter, Consts.fProxySqrtEps, (fProxy)0);
            Assert.IsTrue(infoL.Solved);
            Assert.IsTrue(infoL.iterations >= 1 && infoL.iterations <= maxIter);
            AssertInfoMatches(in A, in b, in xL, (fProxy)0, in infoL, LooseTol());

            var xM = new fProxyN(n, Allocator.Temp);
            var infoM = Krylov.lsmr(in A, in b, ref xM, maxIter, Consts.fProxySqrtEps, (fProxy)0);
            Assert.IsTrue(infoM.Solved);
            Assert.IsTrue(infoM.iterations >= 1 && infoM.iterations <= maxIter);
            AssertInfoMatches(in A, in b, in xM, (fProxy)0, in infoM, LooseTol());

            // Inconsistent system: residual is nonzero but the normal-equation residual vanishes.
            Assert.IsTrue(infoL.rnorm > (fProxy)0.01);
            Assert.IsTrue(infoL.Arnorm <= LooseTol() * ((fProxy)1 + infoL.rnorm));
        }

        // Damped diagnostics: with damp != 0 the reported Arnorm is the DAMPED normal-equation
        // residual ||A^T r - damp^2 x||, which the independent recompute must reproduce.
        void LstsqInfoDampedArnorm()
        {
            int m = 12, n = 5;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 51101);
            var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 51102);
            fProxy damp = (fProxy)0.3;
            int maxIter = 8 * n;

            var xL = new fProxyN(n, Allocator.Temp);
            var infoL = Krylov.lsqr(in A, in b, ref xL, maxIter, Consts.fProxySqrtEps, damp);
            Assert.IsTrue(infoL.Solved);
            AssertInfoMatches(in A, in b, in xL, damp, in infoL, LooseTol());

            var xM = new fProxyN(n, Allocator.Temp);
            var infoM = Krylov.lsmr(in A, in b, ref xM, maxIter, Consts.fProxySqrtEps, damp);
            Assert.IsTrue(infoM.Solved);
            AssertInfoMatches(in A, in b, in xM, damp, in infoM, LooseTol());
        }

        // The LSMR ‖r‖ figure is the one genuinely subtle piece of the free diagnostics: LSMR never
        // holds the residual r = b - A x, so info.rnorm comes from the Fong-Saunders §5.4 scalar
        // recurrence run over the same rotations (O(1)/iteration, no matvec). Pin it against the
        // certified-exact residual (Krylov.lstsqResidual, one real Apply+ApplyT) on BOTH a
        // consistent system (rnorm -> 0) and an inconsistent one (rnorm large) -- the case most
        // likely to expose a recurrence bug.
        void LsmrRnormMatchesExact()
        {
            int m = 16, n = 6;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 51401);

            // (a) consistent: b = A x* for a known x* -> exact recovery, rnorm -> 0.
            var xStar = GenerateOP.fProxyRandomVec(n, -1f, 1f, 51402);
            var bCons = Blas.dot(A, xStar);                // bCons = A x*  (consistent)

            // (b) inconsistent: random rhs -> nonzero least-squares residual.
            var bInc = GenerateOP.fProxyRandomVec(m, -1f, 1f, 51403);

            int maxIter = 8 * n;
            var rS = new fProxyN(m, Allocator.Temp);
            var sS = new fProxyN(n, Allocator.Temp);

            var xc = new fProxyN(n, Allocator.Temp);
            var ic = Krylov.lsmr(in A, in bCons, ref xc, maxIter, Consts.fProxySqrtEps, (fProxy)0);
            Assert.IsTrue(ic.Solved);
            var exC = Krylov.lstsqResidual(new fProxyDenseOperator(in A), in bCons, in xc, (fProxy)0, ref rS, ref sS);
            AssertClose(ic.rnorm, exC.rnorm, LooseTol());

            var xi = new fProxyN(n, Allocator.Temp);
            var ii = Krylov.lsmr(in A, in bInc, ref xi, maxIter, Consts.fProxySqrtEps, (fProxy)0);
            Assert.IsTrue(ii.Solved);
            var exI = Krylov.lstsqResidual(new fProxyDenseOperator(in A), in bInc, in xi, (fProxy)0, ref rS, ref sS);
            Assert.IsTrue(exI.rnorm > (fProxy)0.1);             // genuinely inconsistent
            AssertClose(ii.rnorm, exI.rnorm, LooseTol());

            // (c) mid-flight: force MaxIterations (maxIter=2) so the ‖r‖ recurrence is pinned BEFORE
            // convergence -- where the dnorm += betacheck² accumulation would drift if transcribed
            // wrong. rnorm must still equal the exact residual of the (un-converged) iterate.
            var xf = new fProxyN(n, Allocator.Temp);
            var if2 = Krylov.lsmr(in A, in bInc, ref xf, 2, Consts.fProxySqrtEps, (fProxy)0);
            Assert.IsTrue(!if2.Solved && if2.iterations == 2);   // genuinely stopped mid-flight
            var exF = Krylov.lstsqResidual(new fProxyDenseOperator(in A), in bInc, in xf, (fProxy)0, ref rS, ref sS);
            AssertClose(if2.rnorm, exF.rnorm, LooseTol());
        }

        // The BSR diagnostic overload reports the same diagnostics (up to iterative tolerance) as the
        // dense one for the SAME system -- confirms the BSR info path (with A^T materialization) feeds
        // lstsqInfo identically.
        void LstsqInfoBSRMatchesDense()
        {
            int m = 12, n = 5;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 51301);
            var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 51302);
            var bsm = DenseToBSR1x1(in A, m * n);
            int maxIter = 8 * n;

            var xD = new fProxyN(n, Allocator.Temp);
            var infoD = Krylov.lsqr(in A, in b, ref xD, maxIter, Consts.fProxySqrtEps, (fProxy)0);
            Assert.IsTrue(infoD.Solved);

            var xB = new fProxyN(n, Allocator.Temp);
            var infoB = Krylov.lsqr(in bsm, in b, ref xB, maxIter, Consts.fProxySqrtEps, (fProxy)0);
            Assert.IsTrue(infoB.Solved);

            AssertVecEq(in xD, in xB, LooseTol());
            double sr = (fProxy)1 + infoD.rnorm;
            Assert.IsTrue(math.abs(infoD.rnorm  - infoB.rnorm)  <= LooseTol() * sr);
            Assert.IsTrue(math.abs(infoD.Arnorm - infoB.Arnorm) <= LooseTol() * ((fProxy)1 + infoD.Arnorm));
            Assert.IsTrue(math.abs(infoD.xnorm  - infoB.xnorm)  <= LooseTol() * ((fProxy)1 + infoD.xnorm));
        }

        // ================== AᵀA-Jacobi preconditioner ==================

        // A = Q·diag(s) with Q orthonormal columns (QR of a random tall matrix) and s spanning
        // magnitudes (2^0..2^(n-1)). AᵀA = diag(s²) is badly scaled: the plain solve sees n distinct
        // eigenvalues (needs ~n Krylov steps), while column-Jacobi maps A·D = Q (unit-conditioned),
        // so the preconditioned solve converges almost immediately.
        static fProxyMxN BuildBadlyScaledOrthonormal(int m, int n, uint seed)
        {
            var Q = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, seed);
            var R = new fProxyMxN(n, n, Allocator.Temp);
            QR.decompInPlace(ref Q, ref R);           // Q now has orthonormal columns

            var A = new fProxyMxN(m, n, Allocator.Temp);
            for (int j = 0; j < n; j++)
            {
                fProxy s = math.pow((fProxy)2, (fProxy)j);
                for (int i = 0; i < m; i++) A[i, j] = Q[i, j] * s;
            }
            return A;
        }

        // The preconditioner STRICTLY reduces iterations on the badly-scaled system, both reach the
        // least-squares optimum, and the lsqrJacobi convenience wrapper matches the composable path.
        void JacobiPreconditionerReducesIterations()
        {
            int m = 30, n = 12;
            var A = BuildBadlyScaledOrthonormal(m, n, 52001);
            var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 52002);
            int maxIter = 4 * n;
            fProxy tol = Consts.fProxySqrtEps;

            // plain lsqr with diagnostics
            var xPlain = new fProxyN(n, Allocator.Temp);
            var infoPlain = Krylov.lsqr(in A, in b, ref xPlain, maxIter, tol, (fProxy)0);
            Assert.IsTrue(infoPlain.Solved);

            // preconditioned via the composable diagnostic path (to read the iteration count)
            var d2 = new fProxyN(n, Allocator.Temp); Blas.columnNormsSquared(in A, ref d2);
            var d  = new fProxyN(n, Allocator.Temp); Blas.buildJacobiScale(in d2, ref d);
            var scratch = new fProxyN(n, Allocator.Temp);
            var op = new fProxyColScaledOperator<fProxyDenseOperator>(new fProxyDenseOperator(in A), d, scratch);

            var y    = new fProxyN(n, Allocator.Temp);
            var u    = new fProxyN(m, Allocator.Temp);
            var v    = new fProxyN(n, Allocator.Temp);
            var w    = new fProxyN(n, Allocator.Temp);
            var tmpM = new fProxyN(m, Allocator.Temp);
            var tmpN = new fProxyN(n, Allocator.Temp);
            var infoJac = Krylov.lsqr(op, in b, ref y, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIter, tol, (fProxy)0);
            Assert.IsTrue(infoJac.Solved);
            var xComp = new fProxyN(n, Allocator.Temp);
            for (int j = 0; j < n; j++) xComp[j] = d[j] * y[j];

            Assert.IsTrue(infoJac.iterations < infoPlain.iterations);   // the preconditioner's payoff

            // Both land on a least-squares-optimal x (‖Aᵀr‖ ≈ 0). (Forward error is NOT compared:
            // the plain solve on this ill-conditioned system is much less accurate per-component.)
            AssertLeastSquaresOptimal(in A, in xComp,  in b, LooseTol());
            AssertLeastSquaresOptimal(in A, in xPlain, in b, LooseTol());

            // The convenience wrapper reproduces the composable path exactly.
            var xConv = new fProxyN(n, Allocator.Temp);
            Assert.IsTrue(Krylov.lsqrJacobi(in A, in b, ref xConv, maxIter, tol));
            AssertVecEq(in xConv, in xComp, LooseTol());
        }

        // Every *Jacobi convenience wrapper (lsqr/lsmr, dense AND 1x1-BSR) solves the
        // badly-scaled system to least-squares optimality, and the BSR form matches the dense form.
        void JacobiConvenienceSolversLSOptimalDenseAndBSR()
        {
            int m = 24, n = 8;
            var A = BuildBadlyScaledOrthonormal(m, n, 52101);
            var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 52102);
            var bsm = DenseToBSR1x1(in A, m * n);
            int maxIter = 4 * n;
            fProxy tol = Consts.fProxySqrtEps;

            // lsqr
            var xLd = new fProxyN(n, Allocator.Temp);
            Assert.IsTrue(Krylov.lsqrJacobi(in A, in b, ref xLd, maxIter, tol));
            AssertLeastSquaresOptimal(in A, in xLd, in b, LooseTol());
            var xLb = new fProxyN(n, Allocator.Temp);
            Assert.IsTrue(Krylov.lsqrJacobi(in bsm, in b, ref xLb, maxIter, tol));
            AssertVecEq(in xLd, in xLb, LooseTol());

            // lsmr
            var xMd = new fProxyN(n, Allocator.Temp);
            Assert.IsTrue(Krylov.lsmrJacobi(in A, in b, ref xMd, maxIter, tol));
            AssertLeastSquaresOptimal(in A, in xMd, in b, LooseTol());
            var xMb = new fProxyN(n, Allocator.Temp);
            Assert.IsTrue(Krylov.lsmrJacobi(in bsm, in b, ref xMb, maxIter, tol));
            AssertVecEq(in xMd, in xMb, LooseTol());
        }

        // LITERATURE GROUND TRUTH (Strang, Introduction to Linear Algebra -- best-fit line):
        // fit b = C + D t at t = 0,1,2 with data b = (6,0,0). A = [[1,0],[1,1],[1,2]].
        // Normal equations AᵀA x = Aᵀb -> [[3,3],[3,5]] x = [6,0] -> x = (5, -3) EXACTLY.
        // Residual r = b - A x = (1,-2,1): rnorm = sqrt(6), Aᵀr = (0,0) so Arnorm = 0, xnorm = sqrt(34).
        // Pins the diagnostics AND the LS solution against literal hand-computed constants (external
        // ground truth), not another solver.
        void LstsqInfoStrangLineFitExact()
        {
            var A = new fProxyMxN(3, 2, Allocator.Temp);
            A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)0;
            A[1, 0] = (fProxy)1; A[1, 1] = (fProxy)1;
            A[2, 0] = (fProxy)1; A[2, 1] = (fProxy)2;
            var b = new fProxyN(3, Allocator.Temp);
            b[0] = (fProxy)6; b[1] = (fProxy)0; b[2] = (fProxy)0;

            fProxy tol = Consts.fProxySqrtEps;
            fProxy expRnorm = math.sqrt((fProxy)6);
            fProxy expXnorm = math.sqrt((fProxy)34);

            // Both solvers must reproduce x = (5,-3) and the exact diagnostics.
            for (int which = 0; which < 2; which++)
            {
                var x = new fProxyN(2, Allocator.Temp);
                LstsqInfo info;
                if (which == 0) info = Krylov.lsqr(in A, in b, ref x, 50, tol, (fProxy)0);
                else            info = Krylov.lsmr(in A, in b, ref x, 50, tol, (fProxy)0);

                Assert.IsTrue(info.Solved);
                AssertClose(x[0], (fProxy)5,    LooseTol());
                AssertClose(x[1], (fProxy)(-3), LooseTol());
                AssertClose(info.rnorm, expRnorm, LooseTol());
                Assert.IsTrue(info.Arnorm <= LooseTol() * ((fProxy)1 + expRnorm));   // Aᵀr = 0 exactly
                AssertClose(info.xnorm, expXnorm, LooseTol());
            }
        }

        // Independently recompute ‖b - A x‖ (one real matvec) and check the solver's FREE rnorm
        // (a value it already tracked -- never a fresh A*x) matches it.
        static void AssertResidualNorm(in fProxyMxN A, in fProxyN b, in fProxyN x, double rnorm, fProxy tol)
        {
            var Ax = Blas.dot(A, x);
            fProxy acc = (fProxy)0;
            for (int i = 0; i < b.N; i++) { fProxy e = b[i] - Ax[i]; acc += e * e; }
            AssertClose(rnorm, math.sqrt(acc), tol);
        }

        // BSR counterpart: recompute ‖b - A x‖ with the SAME sparse matvec the solver tracked its
        // residual through (spMV), so the check stays in-arithmetic rather than comparing a BSR-
        // tracked rnorm against a dense recompute (whose summation order differs).
        static void AssertResidualNormBSR(in fProxyBSR A, in fProxyN b, in fProxyN x, double rnorm, fProxy tol)
        {
            var Ax = BSR.spMV(in A, in x);
            fProxy acc = (fProxy)0;
            for (int i = 0; i < b.N; i++) { fProxy e = b[i] - Ax[i]; acc += e * e; }
            AssertClose(rnorm, math.sqrt(acc), tol);
        }

        // The square solvers (cg/minres/biCGStab) RETURN an SolveInfo
        // whose rnorm is filled from each solver's already-tracked residual -- cg a live
        // ‖r‖ (√ of the dot they already form for the convergence test), minres its phibar (the
        // MINRES identity), biCGStab its running ‖r‖ -- never a fresh matvec. Pin that free rnorm
        // against an INDEPENDENTLY recomputed ‖b - A x‖ for every solver + operator shape, plus a
        // forced-MaxIterations mid-flight case where rnorm must still equal the exact residual of
        // the un-converged iterate (that path returns √rsold, the post-update recurrence residual).
        void SolveInfoRnormMatchesResidual()
        {
            fProxy tol = LooseTol();

            // ---- SPD system: cg (plain and block-Jacobi over BSR), minres ----
            int n = 12;
            var Aspd = BuildDenseSPD(n, 52001);
            var bspd = GenerateOP.fProxyRandomVec(n, -1f, 1f, 52002);
            var bsm = DenseToBSR1x1(in Aspd, n * n);
            var M = new fProxyBlockJacobi(in bsm, Allocator.Temp);
            int maxIter = 4 * n;

            var xg = new fProxyN(n, Allocator.Temp);
            var ig = Krylov.cg(in Aspd, in bspd, ref xg, maxIter, Consts.fProxySqrtEps);
            Assert.IsTrue(ig.Solved && ig.iterations >= 1 && ig.iterations <= maxIter);
            AssertResidualNorm(in Aspd, in bspd, in xg, ig.rnorm, tol);

            var xp = new fProxyN(n, Allocator.Temp);
            var ip = Krylov.cg(in bsm, in M, in bspd, ref xp, maxIter, Consts.fProxySqrtEps);
            Assert.IsTrue(ip.Solved && ip.iterations >= 1);
            AssertResidualNormBSR(in bsm, in bspd, in xp, ip.rnorm, tol);   // BSR solve -> BSR recompute

            var xm = new fProxyN(n, Allocator.Temp);
            var im = Krylov.minres(in Aspd, in bspd, ref xm, maxIter, Consts.fProxySqrtEps);
            Assert.IsTrue(im.Solved && im.iterations >= 1);
            AssertResidualNorm(in Aspd, in bspd, in xm, im.rnorm, tol);

            // ---- Non-symmetric diagonally-dominant system: biCGStab ----
            int d = 8;
            var Ansym = GenerateOP.fProxyRandomMat(d, d, -1f, 1f, 52101);
            for (int i = 0; i < d; i++) Ansym[i, i] += (fProxy)(d + 1);   // strict diagonal dominance
            var bns = GenerateOP.fProxyRandomVec(d, -1f, 1f, 52102);
            var xb = new fProxyN(d, Allocator.Temp);
            var ib = Krylov.biCGStab(in Ansym, in bns, ref xb, 4 * d, Consts.fProxySqrtEps);
            Assert.IsTrue(ib.Solved && ib.iterations >= 1);
            AssertResidualNorm(in Ansym, in bns, in xb, ib.rnorm, tol);

            // ---- Mid-flight MaxIterations for EVERY square solver: one (or two) step(s) does NOT
            //      converge, yet rnorm must still equal ‖b - A x‖ of the (updated) un-converged
            //      iterate -- the contract that on MaxIterations x is a valid last iterate, not
            //      undefined. Each returns √(tracked residual) with x fully advanced. ----
            var xh = new fProxyN(n, Allocator.Temp);
            var ih = Krylov.cg(in Aspd, in bspd, ref xh, 1, Consts.fProxySqrtEps);
            Assert.IsTrue(!ih.Solved && ih.status == IterativeSolveStatus.MaxIterations && ih.iterations == 1);
            AssertResidualNorm(in Aspd, in bspd, in xh, ih.rnorm, tol);

            var xhp = new fProxyN(n, Allocator.Temp);
            var ihp = Krylov.cg(in bsm, in M, in bspd, ref xhp, 1, Consts.fProxySqrtEps);
            Assert.IsTrue(ihp.status == IterativeSolveStatus.MaxIterations && ihp.iterations == 1);
            AssertResidualNormBSR(in bsm, in bspd, in xhp, ihp.rnorm, tol);

            var xhm = new fProxyN(n, Allocator.Temp);
            var ihm = Krylov.minres(in Aspd, in bspd, ref xhm, 1, Consts.fProxySqrtEps);
            Assert.IsTrue(ihm.status == IterativeSolveStatus.MaxIterations && ihm.iterations == 1);
            AssertResidualNorm(in Aspd, in bspd, in xhm, ihm.rnorm, tol);

            var xhb = new fProxyN(d, Allocator.Temp);
            var ihb = Krylov.biCGStab(in Ansym, in bns, ref xhb, 1, Consts.fProxySqrtEps);
            Assert.IsTrue(ihb.status == IterativeSolveStatus.MaxIterations && ihb.iterations == 1);
            AssertResidualNorm(in Ansym, in bns, in xhb, ihb.rnorm, tol);

            // ---- Breakdown path: CG on the indefinite A = diag(1,-1) with b = (1,1) and x₀ = 0
            //      hits p·Ap = 1·1 + 1·(-1) = 0 on the very first step -> Breakdown at iterations=0,
            //      x untouched (= 0). rnorm must be the residual of that x: ‖b - A·0‖ = ‖b‖ = √2. ----
            var Aind = new fProxyMxN(2, 2, Allocator.Temp);
            Aind[0, 0] = (fProxy)1; Aind[0, 1] = (fProxy)0;
            Aind[1, 0] = (fProxy)0; Aind[1, 1] = (fProxy)(-1);
            var bind = new fProxyN(2, Allocator.Temp); bind[0] = (fProxy)1; bind[1] = (fProxy)1;
            var xind = new fProxyN(2, Allocator.Temp); xind[0] = (fProxy)0; xind[1] = (fProxy)0;
            var iind = Krylov.cg(in Aind, in bind, ref xind, 10, Consts.fProxySqrtEps);
            Assert.IsTrue(iind.status == IterativeSolveStatus.Breakdown && iind.iterations == 0);
            AssertResidualNorm(in Aind, in bind, in xind, iind.rnorm, tol);
            AssertClose(iind.rnorm, math.sqrt((fProxy)2), tol);

            // ---- b == 0 shortcut: the unique solution is x = 0, reported as Converged at
            //      iterations=0 with rnorm exactly 0 (no matvec, b copied through). ----
            var bzero = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) bzero[i] = (fProxy)0;
            var xzero = new fProxyN(n, Allocator.Temp);
            var izero = Krylov.cg(in Aspd, in bzero, ref xzero, maxIter, Consts.fProxySqrtEps);
            Assert.IsTrue(izero.Solved && izero.iterations == 0);
            AssertClose(izero.rnorm, (fProxy)0, tol);
        }

        // Damped least-squares reference: min ||Ax-b||^2 + damp^2||x||^2 == the plain least-squares
        // solution of the augmented system [A; damp*I] x ~= [b; 0], solved with dense QR. QR.solveInPlace
        // is DESTRUCTIVE, so the augmented matrix/rhs are fresh temporaries.
        static fProxyN DampedReferenceQR(in fProxyMxN A, in fProxyN b, fProxy damp)
        {
            int m = A.M_Rows, n = A.N_Cols;
            var Atil = new fProxyMxN(m + n, n, Allocator.Temp);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    Atil[i, j] = A[i, j];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    Atil[m + i, j] = (i == j) ? damp : (fProxy)0;

            var btil = new fProxyN(m + n, Allocator.Temp);
            for (int i = 0; i < m; i++) btil[i] = b[i];
            for (int i = 0; i < n; i++) btil[m + i] = (fProxy)0;

            var xref = new fProxyN(n, Allocator.Temp);
            QR.solveInPlace(ref Atil, ref btil, ref xref);
            return xref;
        }
    }

    // ---- correctness cases (Burst) -------------------------------------------------------

    [Test]
    public void Laplacian1DBSRCGMatchesDenseCGTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.Laplacian1DBSRCGMatchesDenseCG }.Run();

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
    public void MergedCgIdentityMatchesPlainCgTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.MergedCgIdentityMatchesPlainCg }.Run();

    [Test]
    public void PCGBeatsCGIllConditionedTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.PCGBeatsCGIllConditioned }.Run();

    [Test]
    public void BlockJacobiApplyHandComputedTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.BlockJacobiApplyHandComputed }.Run();

    [Test]
    public void WarmStartTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.WarmStart }.Run();

    // ---- Phase 3 correctness entry points ----

    [Test]
    public void MinresIndefiniteDenseAndBSRTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.MinresIndefiniteDenseAndBSR }.Run();

    [Test]
    public void MinresSpdMatchesCGTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.MinresSpdMatchesCG }.Run();

    [Test]
    public void BiCGStabNonSymmetricMatchesLUTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.BiCGStabNonSymmetricMatchesLU }.Run();

    [Test]
    public void LsqrOverdeterminedConsistentDenseAndBSRTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LsqrOverdeterminedConsistentDenseAndBSR }.Run();

    [Test]
    public void LsqrInconsistentMatchesQRTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LsqrInconsistentMatchesQR }.Run();

    [Test]
    public void LsqrUnderdeterminedConsistentTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LsqrUnderdeterminedConsistent }.Run();

    // ---- LSMR entry points ----

    [Test]
    public void LsmrOverdeterminedConsistentDenseAndBSRTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LsmrOverdeterminedConsistentDenseAndBSR }.Run();

    [Test]
    public void LsmrInconsistentMatchesQRTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LsmrInconsistentMatchesQR }.Run();

    [Test]
    public void LsmrUnderdeterminedMatchesLsqrTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LsmrUnderdeterminedMatchesLsqr }.Run();

    [Test]
    public void LsmrMonotonicArnormTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LsmrMonotonicArnorm }.Run();

    [Test]
    public void TikhonovDampingMatchesAugmentedQRTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.TikhonovDampingMatchesAugmentedQR }.Run();

    // ---- LS diagnostics entry points ----

    [Test]
    public void LstsqInfoMatchesIndependentRecomputeTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LstsqInfoMatchesIndependentRecompute }.Run();

    [Test]
    public void LstsqInfoDampedArnormTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LstsqInfoDampedArnorm }.Run();

    [Test]
    public void LsmrRnormMatchesExactTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LsmrRnormMatchesExact }.Run();

    [Test]
    public void LstsqInfoBSRMatchesDenseTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LstsqInfoBSRMatchesDense }.Run();

    [Test]
    public void JacobiPreconditionerReducesIterationsTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.JacobiPreconditionerReducesIterations }.Run();

    [Test]
    public void JacobiConvenienceSolversLSOptimalDenseAndBSRTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.JacobiConvenienceSolversLSOptimalDenseAndBSR }.Run();

    [Test]
    public void LstsqInfoStrangLineFitExactTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LstsqInfoStrangLineFitExact }.Run();

    [Test]
    public void SolveInfoRnormMatchesResidualTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.SolveInfoRnormMatchesResidual }.Run();

    // ---- Phase 3 warm-start entry points ----

    [Test]
    public void MinresWarmStartTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.MinresWarmStart }.Run();

    [Test]
    public void BiCGStabWarmStartTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.BiCGStabWarmStart }.Run();

    [Test]
    public void LsqrWarmStartTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LsqrWarmStart }.Run();

    [Test]
    public void LsmrWarmStartTest()
        => new SparseSolverTestJob { Type = SparseSolverTestJob.TestType.LsmrWarmStart }.Run();

    // ---- guard / exception cases (managed thread; Assert.Throws can't run inside Burst) ----

    static fProxyBSR BuildSquareBSR()
    {
        const int BR = 2, BC = 2;
        var builder = new fProxyBSRBuilder(2, 2, BR, BC, Allocator.Temp, 2);
        builder.AddBlock(0, 0, GenerateOP.fProxyRandomMat(BR, BC, -1f, 1f, 6101));
        builder.AddBlock(1, 1, GenerateOP.fProxyRandomMat(BR, BC, -1f, 1f, 6102));
        return builder.ToBSR(Allocator.Temp);
    }

    [Test]
    public void BlockJacobi_MissingDiagonalBlock_Throws()
    {
        const int BR = 2;
        // Only an off-diagonal block -- no block at (0,0) or (1,1).
        var builder = new fProxyBSRBuilder(2, 2, BR, BR, Allocator.Temp, 1);
        builder.AddBlock(0, 1, GenerateOP.fProxyRandomMat(BR, BR, -1f, 1f, 6201));
        var A = builder.ToBSR(Allocator.Temp);

        Assert.Throws<ArgumentException>(() => new fProxyBlockJacobi(in A, Allocator.Temp));
    }

    // Numerical breakdown reporting: the out-info overloads return a DirectSolveInfo instead of
    // throwing; the info-less conveniences keep throwing for the same condition.
    [Test]
    public void BlockJacobi_SingularDiagonalBlock_StatusAndThrow()
    {
        const int BR = 2;
        var builder = new fProxyBSRBuilder(2, 2, BR, BR, Allocator.Temp, 2);
        builder.AddBlock(0, 0, GenerateOP.fProxyDiagonalMat(BR, (fProxy)4));
        builder.AddBlock(1, 1, new fProxyMxN(BR, BR, Allocator.Temp));   // all-zero diagonal block: singular
        var A = builder.ToBSR(Allocator.Temp);

        var M = new fProxyBlockJacobi(in A, Allocator.Temp, out PreconditionerInfo info);
        Assert.IsFalse(info.Solved);

        Assert.Throws<ArgumentException>(() => new fProxyBlockJacobi(in A, Allocator.Temp));
    }

    [Test]
    public void Preconditioner_StatusOverloads_SuccessPath()
    {
        const int BR = 2;
        var builder = new fProxyBSRBuilder(2, 2, BR, BR, Allocator.Temp, 2);
        builder.AddBlock(0, 0, GenerateOP.fProxyDiagonalMat(BR, (fProxy)4));
        builder.AddBlock(1, 1, GenerateOP.fProxyDiagonalMat(BR, (fProxy)9));
        var A = builder.ToBSR(Allocator.Temp);

        var mJ = new fProxyBlockJacobi(in A, Allocator.Temp, out PreconditionerInfo infoJ);
        Assert.IsTrue(infoJ.Solved);
        Assert.IsTrue(infoJ.attempts == 1);
        var mI = new fProxyIC0(in A, Allocator.Temp, out PreconditionerInfo infoI);
        Assert.IsTrue(infoI.Solved);
        Assert.IsTrue(infoI.shift == 0.0);   // clean SPD build: no rescue shift
        var mL = new fProxyILU0(in A, Allocator.Temp, out PreconditionerInfo infoL);
        Assert.IsTrue(infoL.Solved);
        Assert.IsTrue(infoL.attempts == 1);

        // The successfully-built Jacobi is usable: z = M^-1 r on the 4x4 system.
        var r = GenerateOP.fProxyVec(2 * BR, (fProxy)1);
        var z = new fProxyN(2 * BR, Allocator.Temp);
        mJ.Apply(in r, ref z);
        Assert.IsTrue(math.abs((float)(z[0] - (fProxy)0.25)) < 1e-6f);
        Assert.IsTrue(math.abs((float)(z[2 * BR - 1] - (fProxy)(1.0 / 9.0))) < 1e-6f);
    }

    [Test]
    public void BlockJacobi_NonSquareBSR_Throws()
    {
        const int BR = 2, BC = 3;
        var builder = new fProxyBSRBuilder(2, 2, BR, BC, Allocator.Temp, 1); // BR != BC
        builder.AddBlock(0, 0, GenerateOP.fProxyRandomMat(BR, BC, -1f, 1f, 6301));
        var A = builder.ToBSR(Allocator.Temp);

        Assert.Throws<ArgumentException>(() => new fProxyBlockJacobi(in A, Allocator.Temp));
    }

    [Test]
    public void Cg_NonSquareDenseOperator_Throws()
    {
        var A = new fProxyMxN(3, 4, Allocator.Temp); // non-square
        var op = new fProxyDenseOperator(in A);
        var b = new fProxyN(3, Allocator.Temp);
        var x = new fProxyN(4, Allocator.Temp);
        var r = new fProxyN(3, Allocator.Temp);
        var p = new fProxyN(3, Allocator.Temp);
        var Ap = new fProxyN(3, Allocator.Temp);

        Assert.Throws<ArgumentException>(() =>
            Krylov.cg(in op, in b, ref x, ref r, ref p, ref Ap, 4, Consts.fProxySqrtEps));
    }

    // ---- scratch-aliasing guards for cg / cg --------------------------------------------
    //
    // cg throw if ANY two of their vector arguments share a Data.Ptr (the elementwise axpy
    // scratch updates silently corrupt on aliasing rather than self-checking). The pairs below
    // are chosen to be ones NOT already caught by a downstream Apply/dot guard, so each proves
    // the up-front distinctness check is doing real work. The guard runs before any computation,
    // so the operator matrix's contents are irrelevant -- a bare square fProxyMat suffices for cg.

    [Test]
    public void Cg_AliasingRAndAp_Throws()
    {
        const int dim = 4;
        var A = new fProxyMxN(dim, dim, Allocator.Temp); // square; guard fires before A is read
        var op = new fProxyDenseOperator(in A);
        var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 6501);
        var x = new fProxyN(dim, Allocator.Temp);
        var p = new fProxyN(dim, Allocator.Temp);
        var Ap = new fProxyN(dim, Allocator.Temp);
        var rAlias = Ap; // r aliases Ap (would turn r -= Ap into r -= r == 0: false convergence)
        Assert.Throws<ArgumentException>(() =>
            Krylov.cg(in op, in b, ref x, ref rAlias, ref p, ref Ap, dim, Consts.fProxySqrtEps));
    }

    [Test]
    public void Cg_AliasingRAndX_Throws()
    {
        const int dim = 4;
        var A = new fProxyMxN(dim, dim, Allocator.Temp);
        var op = new fProxyDenseOperator(in A);
        var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 6511);
        var x = new fProxyN(dim, Allocator.Temp);
        var p = new fProxyN(dim, Allocator.Temp);
        var Ap = new fProxyN(dim, Allocator.Temp);
        var rAlias = x; // r aliases x (r.CopyFrom(b) would silently clobber the initial guess)
        Assert.Throws<ArgumentException>(() =>
            Krylov.cg(in op, in b, ref x, ref rAlias, ref p, ref Ap, dim, Consts.fProxySqrtEps));
    }

    [Test]
    public void Pcg_AliasingRAndX_Throws()
    {
        var A = BuildSquareBSR();       // 4 x 4, both diagonal blocks present
        var M = new fProxyBlockJacobi(in A, Allocator.Temp);
        int dim = A.M_Rows;
        var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 6521);
        var x = new fProxyN(dim, Allocator.Temp);
        var p = new fProxyN(dim, Allocator.Temp);
        var Ap = new fProxyN(dim, Allocator.Temp);
        var z = new fProxyN(dim, Allocator.Temp);
        var rAlias = x; // r aliases x
        Assert.Throws<ArgumentException>(() =>
            Krylov.cg(in A, in M, in b, ref x, ref rAlias, ref p, ref Ap, ref z, dim, Consts.fProxySqrtEps));
    }

    [Test]
    public void Pcg_AliasingZAndX_Throws()
    {
        var A = BuildSquareBSR();
        var M = new fProxyBlockJacobi(in A, Allocator.Temp);
        int dim = A.M_Rows;
        var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 6531);
        var x = new fProxyN(dim, Allocator.Temp);
        var r = new fProxyN(dim, Allocator.Temp);
        var p = new fProxyN(dim, Allocator.Temp);
        var Ap = new fProxyN(dim, Allocator.Temp);
        var zAlias = x; // z aliases x (not caught by M.Apply's own r/z guard)
        Assert.Throws<ArgumentException>(() =>
            Krylov.cg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref zAlias, dim, Consts.fProxySqrtEps));
    }

    // x aliasing b: the final pair among {r,p,Ap,x,b} / {r,p,Ap,z,x,b}. Benign in the current loop
    // (b isn't reread after the initial residual), but the guard is documented as ALL-pairs-distinct
    // so it must still throw. xAlias is a struct-copy of b (shares Data.Ptr) -- passing them as two
    // distinct locals keeps b as `in` and x as `ref` without an in/ref same-variable conflict.
    [Test]
    public void Cg_AliasingXAndB_Throws()
    {
        const int dim = 4;
        var A = new fProxyMxN(dim, dim, Allocator.Temp); // square; guard fires before A is read
        var op = new fProxyDenseOperator(in A);
        var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 6541);
        var xAlias = b; // x aliases b (struct copy shares Data.Ptr)
        var r = new fProxyN(dim, Allocator.Temp);
        var p = new fProxyN(dim, Allocator.Temp);
        var Ap = new fProxyN(dim, Allocator.Temp);
        Assert.Throws<ArgumentException>(() =>
            Krylov.cg(in op, in b, ref xAlias, ref r, ref p, ref Ap, dim, Consts.fProxySqrtEps));
    }

    [Test]
    public void Pcg_AliasingXAndB_Throws()
    {
        var A = BuildSquareBSR();
        var M = new fProxyBlockJacobi(in A, Allocator.Temp);
        int dim = A.M_Rows;
        var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 6551);
        var xAlias = b; // x aliases b (struct copy shares Data.Ptr)
        var r = new fProxyN(dim, Allocator.Temp);
        var p = new fProxyN(dim, Allocator.Temp);
        var Ap = new fProxyN(dim, Allocator.Temp);
        var z = new fProxyN(dim, Allocator.Temp);
        Assert.Throws<ArgumentException>(() =>
            Krylov.cg(in A, in M, in b, ref xAlias, ref r, ref p, ref Ap, ref z, dim, Consts.fProxySqrtEps));
    }

    // ---- Phase 3 guard / aliasing cases (managed thread; Assert.Throws can't run in Burst) ----
    //
    // MINRES/BiCGSTAB/LSQR/LSMR share cg's "every vector argument must be a distinct buffer"
    // contract, enforced up front by RequireDistinctBuffers before any computation -- so the
    // operator matrix's contents are irrelevant and a bare zeroed fProxyMat suffices. 1-2 aliasing
    // cases per solver (matching Phase 2's coverage) prove the guard fires; exhaustive pairwise
    // coverage is not attempted.

    [Test]
    public void Minres_NonSquareDense_Throws()
    {
        var A = new fProxyMxN(3, 4, Allocator.Temp); // non-square -> minres's A.Rows!=A.Cols guard fires first
        var b = new fProxyN(3, Allocator.Temp);
        var x = new fProxyN(3, Allocator.Temp);
        var y  = new fProxyN(3, Allocator.Temp); var r1 = new fProxyN(3, Allocator.Temp); var r2 = new fProxyN(3, Allocator.Temp);
        var v  = new fProxyN(3, Allocator.Temp); var w  = new fProxyN(3, Allocator.Temp);
        var w1 = new fProxyN(3, Allocator.Temp); var w2 = new fProxyN(3, Allocator.Temp);
        Assert.Throws<ArgumentException>(() =>
            Krylov.minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, 3, Consts.fProxySqrtEps));
    }

    [Test]
    public void Minres_AliasingR1AndR2_Throws()
    {
        const int dim = 4;
        var A = new fProxyMxN(dim, dim, Allocator.Temp); // square; guard fires before A is read
        var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 39001);
        var x = new fProxyN(dim, Allocator.Temp);
        var y  = new fProxyN(dim, Allocator.Temp);
        var r1 = new fProxyN(dim, Allocator.Temp);
        var v  = new fProxyN(dim, Allocator.Temp);
        var w  = new fProxyN(dim, Allocator.Temp);
        var w1 = new fProxyN(dim, Allocator.Temp);
        var w2 = new fProxyN(dim, Allocator.Temp);
        var r2Alias = r1; // r2 aliases r1
        Assert.Throws<ArgumentException>(() =>
            Krylov.minres(in A, in b, ref x, ref y, ref r1, ref r2Alias, ref v, ref w, ref w1, ref w2, dim, Consts.fProxySqrtEps));
    }

    [Test]
    public void BiCGStab_AliasingXAndB_Throws()
    {
        const int dim = 4;
        var A = new fProxyMxN(dim, dim, Allocator.Temp);
        var b = GenerateOP.fProxyRandomVec(dim, -1f, 1f, 39101);
        var xAlias = b; // x aliases b (struct copy shares Data.Ptr)
        var r = new fProxyN(dim, Allocator.Temp); var rHat0 = new fProxyN(dim, Allocator.Temp); var p = new fProxyN(dim, Allocator.Temp);
        var v = new fProxyN(dim, Allocator.Temp); var t = new fProxyN(dim, Allocator.Temp);
        Assert.Throws<ArgumentException>(() =>
            Krylov.biCGStab(in A, in b, ref xAlias, ref r, ref rHat0, ref p, ref v, ref t, dim, Consts.fProxySqrtEps));
    }

    [Test]
    public void Lsqr_AliasingUAndTmpM_Throws()
    {
        int m = 5, n = 3;
        var A = new fProxyMxN(m, n, Allocator.Temp);
        var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 39301);
        var x = new fProxyN(n, Allocator.Temp);
        var u = new fProxyN(m, Allocator.Temp);
        var v = new fProxyN(n, Allocator.Temp);
        var w = new fProxyN(n, Allocator.Temp);
        var tmpN = new fProxyN(n, Allocator.Temp);
        var tmpMAlias = u; // tmpM aliases u (both length m)
        Assert.Throws<ArgumentException>(() =>
            Krylov.lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpMAlias, ref tmpN, n, Consts.fProxySqrtEps));
    }

    [Test]
    public void Lsmr_AliasingHAndHbar_Throws()
    {
        int m = 5, n = 3;
        var A = new fProxyMxN(m, n, Allocator.Temp);
        var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 39401);
        var x = new fProxyN(n, Allocator.Temp);
        var u = new fProxyN(m, Allocator.Temp);
        var v = new fProxyN(n, Allocator.Temp);
        var h = new fProxyN(n, Allocator.Temp);
        var tmpM = new fProxyN(m, Allocator.Temp);
        var tmpN = new fProxyN(n, Allocator.Temp);
        var hbarAlias = h; // hbar aliases h (both length n)
        Assert.Throws<ArgumentException>(() =>
            Krylov.lsmr(in A, in b, ref x, ref u, ref v, ref h, ref hbarAlias, ref tmpM, ref tmpN, n, Consts.fProxySqrtEps));
    }

}
