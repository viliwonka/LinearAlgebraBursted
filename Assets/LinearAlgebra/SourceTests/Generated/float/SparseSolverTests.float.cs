using System;
using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Phase-2 sparse-solver test suite: IfloatLinearOperator / floatDenseOperator /
// floatBSMOperator / floatBlockJacobi / Solvers.cg&lt;TOp&gt; / Solvers.pcg&lt;TOp,TPre&gt;, plus
// the concrete conjugateGradient/pcg convenience overloads for floatMxN and floatBSM. Every
// BSM system is cross-checked against the equivalent dense system (same pattern as
// floatSparseBSMTests: build the SAME system in both forms and compare).
//
// Correctness cases run inside a [BurstCompile] IJob (matches floatConjugateGradientTests /
// floatSparseBSMTests). Guard/exception cases run on the managed test thread with
// Assert.Throws, since NUnit's Assert.Throws cannot execute inside a Burst-compiled job.
public class floatSparseSolverTests
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
        }

        public TestType Type;

        // CG/PCG residuals converge to tolerance^2 * ||b||^2 (see Solvers.cg), so per-component
        // error is well below this scaled comparison threshold on both precisions; loose enough
        // that comparing two INDEPENDENTLY-converged solutions (each accurate only to about
        // Consts.floatSqrtEps, not machine epsilon) doesn't false-fail.
        static float Tol() => 1e-3f;

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
            }
        }

        // ---- helpers ---------------------------------------------------------------------

        // Same recipe as floatConjugateGradientTests.BuildSPD: A = M^T M + dim*I (strictly SPD,
        // diagonally dominant).
        static floatMxN BuildDenseSPD(ref Arena arena, int dim, uint seed)
        {
            var M = arena.floatRandomMat(dim, dim, -1f, 1f, seed);
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
        // all of them. See the growth regression tests in SparseBSMTests.float.cs, which build
        // via many-reallocation growth on purpose to prove this.
        static floatBSM DenseToBSM1x1(ref Arena arena, in floatMxN A, int nnzHint)
        {
            var builder = arena.floatBSMBuilder(A.M_Rows, A.N_Cols, 1, 1, math.max(nnzHint, 1));
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    if (A[r, c] != (float)0)
                        builder.AddValue(r, c, A[r, c]);
            return builder.ToBSM(ref arena);
        }

        static void AssertVecEq(in floatN a, in floatN b, float tol)
        {
            Assert.IsTrue(Analysis_OP.isZero(a - b, tol));
        }

        // ---- 1. 1D Laplacian tridiagonal as a 1x1-block BSM: CG matches dense CG -----------
        void Laplacian1DBSMCGMatchesDenseCG()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;
            var A = arena.floatLaplacian1D(dim);
            // Tridiagonal: at most 3 nonzeros/row -> 3*dim is a safe upper bound.
            var bsm = DenseToBSM1x1(ref arena, in A, 3 * dim);

            var b = arena.floatRandomVec(dim, -1f, 1f, 4242);

            var xDense = arena.floatVec(dim);
            bool okDense = Solvers.conjugateGradient(in A, in b, ref xDense, 4 * dim, Consts.floatSqrtEps);
            Assert.IsTrue(okDense);

            var xBsm = arena.floatVec(dim);
            bool okBsm = Solvers.conjugateGradient(in bsm, in b, ref xBsm, 4 * dim, Consts.floatSqrtEps);
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

            var mb = arena.floatBSMBuilder(nb, nb, BR, BR, nb * nb);
            mb.AddBlock(0, 0, arena.floatRandomMat(BR, BR, -1f, 1f, 8001));
            mb.AddBlock(0, 1, arena.floatRandomMat(BR, BR, -1f, 1f, 8002));
            mb.AddBlock(1, 1, arena.floatRandomMat(BR, BR, -1f, 1f, 8003));
            mb.AddBlock(1, 2, arena.floatRandomMat(BR, BR, -1f, 1f, 8004));
            mb.AddBlock(2, 2, arena.floatRandomMat(BR, BR, -1f, 1f, 8005));
            mb.AddBlock(2, 0, arena.floatRandomMat(BR, BR, -1f, 1f, 8006));
            var Mdense = mb.ToBSM(ref arena).ToDense(ref arena);

            var A = Linear_OP.dot(Mdense, Mdense, true);
            for (int i = 0; i < dim; i++)
                A[i, i] += dim;

            var ab = arena.floatBSMBuilder(nb, nb, BR, BR, nb * nb);
            for (int bi = 0; bi < nb; bi++)
                for (int bj = 0; bj < nb; bj++)
                {
                    var blk = arena.floatMat(BR, BR);
                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BR; c++)
                            blk[r, c] = A[bi * BR + r, bj * BR + c];
                    ab.AddBlock(bi, bj, in blk);
                }
            var Absm = ab.ToBSM(ref arena);

            var b = arena.floatRandomVec(dim, -1f, 1f, 8100);
            var x = arena.floatVec(dim);
            bool ok = Solvers.conjugateGradient(in Absm, in b, ref x);
            Assert.IsTrue(ok);

            var Ax = Sparse_OP.spMV(in Absm, in x);
            AssertVecEq(in Ax, in b, Tol());

            // Cross-check against the dense reference too.
            var xDense = arena.floatVec(dim);
            bool okDense = Solvers.conjugateGradient(in A, in b, ref xDense);
            Assert.IsTrue(okDense);
            AssertVecEq(in x, in xDense, Tol());

            arena.Dispose();
        }

        // ---- 3. Dense forwarding unchanged: guards the conjugateGradient(in floatMxN,...) ----
        //         refactor into cg<floatDenseOperator> -----------------------------------------
        void DenseForwardingUnchanged()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 12;
            var A = BuildDenseSPD(ref arena, dim, 90125);
            var b = arena.floatRandomVec(dim, -1f, 1f, 4242);

            // The (unchanged, public) concrete entry point.
            var xConcrete = arena.floatVec(dim);
            bool okConcrete = Solvers.conjugateGradient(in A, in b, ref xConcrete);
            Assert.IsTrue(okConcrete);

            // Calling cg<TOp> directly via floatDenseOperator must reproduce the identical
            // result -- this is the single source of truth the concrete overload now forwards
            // into.
            var op = new floatDenseOperator(in A);
            var xGeneric = arena.floatVec(dim);
            var r = arena.floatVec(dim);
            var p = arena.floatVec(dim);
            var Ap = arena.floatVec(dim);
            bool okGeneric = Solvers.cg(in op, in b, ref xGeneric, ref r, ref p, ref Ap, dim, Consts.floatSqrtEps);
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
            var xQR = arena.floatVec(dim);
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
            var M = arena.floatBlockJacobi(in bsm);
            var b = arena.floatRandomVec(dim, -1f, 1f, 6002);

            var xCG = arena.floatVec(dim);
            bool okCG = Solvers.conjugateGradient(in bsm, in b, ref xCG);
            Assert.IsTrue(okCG);

            var xPCG = arena.floatVec(dim);
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

            var Sym = arena.floatMat(dim, dim);
            for (int r = 0; r < dim; r++)
                for (int c = 0; c < dim; c++)
                    Sym[r, c] = S[r, c] / math.sqrt(S[r, r] * S[c, c]);

            var A = arena.floatMat(dim, dim);
            var d = arena.floatVec(dim);
            for (int i = 0; i < dim; i++)
                d[i] = (i % 2 == 0) ? (float)1 : (float)100; // alternating 1,100

            for (int r = 0; r < dim; r++)
                for (int c = 0; c < dim; c++)
                    A[r, c] = d[r] * Sym[r, c] * d[c];

            var bsm = DenseToBSM1x1(ref arena, in A, dim * dim);
            var M = arena.floatBlockJacobi(in bsm);
            var b = arena.floatRandomVec(dim, -1f, 1f, 7002);

            int maxBudget = 4 * dim;
            int minCG = -1, minPCG = -1;
            for (int budget = 1; budget <= maxBudget && (minCG < 0 || minPCG < 0); budget++)
            {
                if (minCG < 0)
                {
                    var xCG = arena.floatVec(dim);
                    if (Solvers.conjugateGradient(in bsm, in b, ref xCG, budget, Consts.floatSqrtEps))
                        minCG = budget;
                }
                if (minPCG < 0)
                {
                    var xPCG = arena.floatVec(dim);
                    if (Solvers.pcg(in bsm, in M, in b, ref xPCG, budget, Consts.floatSqrtEps))
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
            var builder = arena.floatBSMBuilder(2, 2, BR, BR, 2);
            var d0 = arena.floatMat(BR, BR);
            d0[0, 0] = 4f; d0[0, 1] = 1f;
            d0[1, 0] = 1f; d0[1, 1] = 3f;
            var d1 = arena.floatMat(BR, BR);
            d1[0, 0] = 2f; d1[0, 1] = 0f;
            d1[1, 0] = 0f; d1[1, 1] = 5f;
            builder.AddBlock(0, 0, in d0);
            builder.AddBlock(1, 1, in d1);
            var A = builder.ToBSM(ref arena);

            var M = arena.floatBlockJacobi(in A);

            var r = arena.floatVec(4);
            r[0] = 1f; r[1] = 2f; r[2] = 3f; r[3] = 4f;
            var z = arena.floatVec(4);
            M.Apply(in r, ref z);

            // Hand inverse of d0=[[4,1],[1,3]]: det=11, inv = (1/11)*[[3,-1],[-1,4]].
            float det0 = 11f;
            float z0 = (3f * r[0] - 1f * r[1]) / det0;
            float z1 = (-1f * r[0] + 4f * r[1]) / det0;
            // d1 is diagonal: inv = diag(1/2, 1/5).
            float z2 = r[2] / 2f;
            float z3 = r[3] / 5f;

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
            var M = arena.floatBlockJacobi(in bsm);
            var b = arena.floatRandomVec(dim, -1f, 1f, 9002);

            var x = arena.floatVec(dim);
            bool ok = Solvers.pcg(in bsm, in M, in b, ref x);
            Assert.IsTrue(ok);

            // Feed the converged solution back as the initial guess -- a single iteration's
            // worth of budget must still report convergence (matches
            // floatConjugateGradientTests.AlreadyConverged's dense-CG counterpart).
            var xWarm = x.Copy();
            bool okWarm = Solvers.pcg(in bsm, in M, in b, ref xWarm, 1, Consts.floatSqrtEps);
            Assert.IsTrue(okWarm);
            AssertVecEq(in x, in xWarm, Tol());

            // Same check for plain (unpreconditioned) CG.
            var xCG = arena.floatVec(dim);
            bool okCG = Solvers.conjugateGradient(in bsm, in b, ref xCG);
            Assert.IsTrue(okCG);
            var xCGWarm = xCG.Copy();
            bool okCGWarm = Solvers.conjugateGradient(in bsm, in b, ref xCGWarm, 1, Consts.floatSqrtEps);
            Assert.IsTrue(okCGWarm);
            AssertVecEq(in xCG, in xCGWarm, Tol());

            arena.Dispose();
        }

        // ---- 8. pcg breakdown guard: a non-SPD preconditioner bails out (returns false) -------
        //
        // The preconditioned inner product <r,z> must stay positive for PCG to be well-defined.
        // floatNegatePreconditioner deliberately returns z = -r, so rzold = <r,-r> = -||r||^2 < 0
        // -- the pcg `if (!(rzold > 0)) return false;` guard (added this pass) must catch it and
        // return false, rather than looping with a wrong-signed alpha/beta (silent divergence /
        // NaN). A itself is a genuine SPD system so the failure is attributable to the
        // preconditioner, not the operator.
        void PcgNonSpdPreconditionerBreaksDown()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;
            var A = BuildDenseSPD(ref arena, dim, 5501);
            var op = new floatDenseOperator(in A);
            var pre = new floatNegatePreconditioner();
            var b = arena.floatRandomVec(dim, -1f, 1f, 5502); // nonzero rhs -> not the b==0 shortcut

            var x = arena.floatVec(dim);
            bool ok = Solvers.pcg(in op, in pre, in b, ref x, dim, Consts.floatSqrtEps);
            Assert.IsFalse(ok);

            arena.Dispose();
        }
    }

    // Deliberately non-SPD test-double preconditioner: z = M^-1 r := -r, so <r,z> = -||r||^2 <= 0.
    // Used only by PcgNonSpdPreconditionerBreaksDown to exercise pcg's rzold>0 breakdown guard.
    public struct floatNegatePreconditioner : IfloatPreconditioner
    {
        public void Apply(in floatN r, ref floatN z)
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

    // ---- guard / exception cases (managed thread; Assert.Throws can't run inside Burst) ----

    static floatBSM BuildSquareBSM(ref Arena arena)
    {
        const int BR = 2, BC = 2;
        var builder = arena.floatBSMBuilder(2, 2, BR, BC, 2);
        builder.AddBlock(0, 0, arena.floatRandomMat(BR, BC, -1f, 1f, 6101));
        builder.AddBlock(1, 1, arena.floatRandomMat(BR, BC, -1f, 1f, 6102));
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
            var builder = arena.floatBSMBuilder(2, 2, BR, BR, 1);
            builder.AddBlock(0, 1, arena.floatRandomMat(BR, BR, -1f, 1f, 6201));
            var A = builder.ToBSM(ref arena);

            Assert.Throws<ArgumentException>(() => arena.floatBlockJacobi(in A));
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
            var builder = arena.floatBSMBuilder(2, 2, BR, BC, 1); // BR != BC
            builder.AddBlock(0, 0, arena.floatRandomMat(BR, BC, -1f, 1f, 6301));
            var A = builder.ToBSM(ref arena);

            Assert.Throws<ArgumentException>(() => arena.floatBlockJacobi(in A));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Cg_NonSquareDenseOperator_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.floatMat(3, 4); // non-square
            var op = new floatDenseOperator(in A);
            var b = arena.floatVec(3);
            var x = arena.floatVec(4);
            var r = arena.floatVec(3);
            var p = arena.floatVec(3);
            var Ap = arena.floatVec(3);

            Assert.Throws<ArgumentException>(() =>
                Solvers.cg(in op, in b, ref x, ref r, ref p, ref Ap, 4, Consts.floatSqrtEps));
        }
        finally { arena.Dispose(); }
    }

    // ---- scratch-aliasing guards for cg / pcg --------------------------------------------
    //
    // cg/pcg throw if ANY two of their vector arguments share a Data.Ptr (the elementwise axpy
    // scratch updates silently corrupt on aliasing rather than self-checking). The pairs below
    // are chosen to be ones NOT already caught by a downstream Apply/dot guard, so each proves
    // the up-front distinctness check is doing real work. The guard runs before any computation,
    // so the operator matrix's contents are irrelevant -- a bare square floatMat suffices for cg.

    [Test]
    public void Cg_AliasingRAndAp_Throws()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            const int dim = 4;
            var A = arena.floatMat(dim, dim); // square; guard fires before A is read
            var op = new floatDenseOperator(in A);
            var b = arena.floatRandomVec(dim, -1f, 1f, 6501);
            var x = arena.floatVec(dim);
            var p = arena.floatVec(dim);
            var Ap = arena.floatVec(dim);
            var rAlias = Ap; // r aliases Ap (would turn r -= Ap into r -= r == 0: false convergence)
            Assert.Throws<ArgumentException>(() =>
                Solvers.cg(in op, in b, ref x, ref rAlias, ref p, ref Ap, dim, Consts.floatSqrtEps));
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
            var A = arena.floatMat(dim, dim);
            var op = new floatDenseOperator(in A);
            var b = arena.floatRandomVec(dim, -1f, 1f, 6511);
            var x = arena.floatVec(dim);
            var p = arena.floatVec(dim);
            var Ap = arena.floatVec(dim);
            var rAlias = x; // r aliases x (r.CopyFrom(b) would silently clobber the initial guess)
            Assert.Throws<ArgumentException>(() =>
                Solvers.cg(in op, in b, ref x, ref rAlias, ref p, ref Ap, dim, Consts.floatSqrtEps));
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
            var M = arena.floatBlockJacobi(in A);
            int dim = A.M_Rows;
            var b = arena.floatRandomVec(dim, -1f, 1f, 6521);
            var x = arena.floatVec(dim);
            var p = arena.floatVec(dim);
            var Ap = arena.floatVec(dim);
            var z = arena.floatVec(dim);
            var rAlias = x; // r aliases x
            Assert.Throws<ArgumentException>(() =>
                Solvers.pcg(in A, in M, in b, ref x, ref rAlias, ref p, ref Ap, ref z, dim, Consts.floatSqrtEps));
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
            var M = arena.floatBlockJacobi(in A);
            int dim = A.M_Rows;
            var b = arena.floatRandomVec(dim, -1f, 1f, 6531);
            var x = arena.floatVec(dim);
            var r = arena.floatVec(dim);
            var p = arena.floatVec(dim);
            var Ap = arena.floatVec(dim);
            var zAlias = x; // z aliases x (not caught by M.Apply's own r/z guard)
            Assert.Throws<ArgumentException>(() =>
                Solvers.pcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref zAlias, dim, Consts.floatSqrtEps));
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
            var A = arena.floatMat(dim, dim); // square; guard fires before A is read
            var op = new floatDenseOperator(in A);
            var b = arena.floatRandomVec(dim, -1f, 1f, 6541);
            var xAlias = b; // x aliases b (struct copy shares Data.Ptr)
            var r = arena.floatVec(dim);
            var p = arena.floatVec(dim);
            var Ap = arena.floatVec(dim);
            Assert.Throws<ArgumentException>(() =>
                Solvers.cg(in op, in b, ref xAlias, ref r, ref p, ref Ap, dim, Consts.floatSqrtEps));
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
            var M = arena.floatBlockJacobi(in A);
            int dim = A.M_Rows;
            var b = arena.floatRandomVec(dim, -1f, 1f, 6551);
            var xAlias = b; // x aliases b (struct copy shares Data.Ptr)
            var r = arena.floatVec(dim);
            var p = arena.floatVec(dim);
            var Ap = arena.floatVec(dim);
            var z = arena.floatVec(dim);
            Assert.Throws<ArgumentException>(() =>
                Solvers.pcg(in A, in M, in b, ref xAlias, ref r, ref p, ref Ap, ref z, dim, Consts.floatSqrtEps));
        }
        finally { arena.Dispose(); }
    }
}
