using System;
using LinearAlgebra;
using LinearAlgebra.Gallery;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

// Krylov verify-at-exit: cg recursively
// track their residual; in float that tracked value can drift from the true b-Ax and claim
// convergence early. When the tracked residual FIRST claims convergence, the guarded solver now
// recomputes the true residual fresh and retests, continuing if it
// fails. Only at claimed convergence -- MaxIterations/Breakdown exits are untouched.
//
// (c) drift-firing: an ill-conditioned float instance (Moler matrix, deep tolerance) where the
//     tracked residual claims convergence while the true residual has not actually met tolerance
//     -- proven directly against an in-test replica of the cg loop WITHOUT verify-at-exit (same
//     Blas kernels, just without the verify block). The guarded
//     (production) solver must still report HONEST convergence on the same instance.
// (d) healthy-solve path: on a well-conditioned instance (no drift), verify-at-exit costs exactly
//     one extra Apply and does not change the returned solution (x is never touched by the verify
//     block -- only r/rnorm are refreshed).
// Managed [Test]s (main thread, no Burst job) -- matches fProxyLOBPCGSmokeTests' style for this
// kind of iteration-heavy, algorithm-level comparison.
public class fProxyKrylovVerifyAtExitTests
{
    // ---- shared helpers -------------------------------------------------------------------

    // SPD via M^T M + n*I -- same recipe as fProxyKrylovRound2Tests.BuildDenseSPD.
    static fProxyMxN BuildDenseSPD(ref Arena arena, int dim, uint seed)
    {
        var M = arena.fProxyRandomMat(dim, dim, (fProxy)(-1f), (fProxy)1f, seed);
        var A = Blas.dot(M, M, true);
        for (int d = 0; d < dim; d++) A[d, d] += dim;
        return A;
    }

    // Well-conditioned nonsymmetric: random entries + a heavy diagonal (diagonally dominant,
    // nonsingular) -- same recipe as fProxyIDRTests/fProxyGMRESTests DenseNonsym.
    static fProxyMxN BuildDenseNonsym(ref Arena arena, int dim, uint seed)
    {
        var A = arena.fProxyRandomMat(dim, dim, (fProxy)(-1f), (fProxy)1f, seed);
        for (int i = 0; i < dim; i++) A[i, i] += (fProxy)(2 * dim);
        return A;
    }

    // b - A*x, recomputed fresh (independent of whatever residual the solver tracked internally).
    static fProxy TrueResidualSq<TOp>(in TOp A, in fProxyN b, in fProxyN x, ref fProxyN scratch)
        where TOp : struct, IfProxyLinearOperator
    {
        A.Apply(in x, ref scratch);
        scratch.scaleAddInPlace((fProxy)(-1), b);   // scratch = -Ax + b = b - Ax
        return Blas.dot(scratch, scratch);
    }

    // Counts every Apply/ApplyDot call through a static counter -- used by the "exactly one extra
    // Apply" test. Managed-only (no Burst job in this file), so a plain static field is fine.
    static int s_applyCount;

    readonly struct CountingDenseOperator : IfProxyLinearOperator
    {
        public readonly fProxyMxN A;
        public CountingDenseOperator(in fProxyMxN a) { A = a; }
        public int Rows => A.M_Rows;
        public int Cols => A.N_Cols;
        public void Apply(in fProxyN x, ref fProxyN y) { s_applyCount++; Blas.dot(in A, in x, ref y); }
        public void ApplyT(in fProxyN x, ref fProxyN y) => Blas.dot(in x, in A, ref y);
        public fProxy ApplyDot(in fProxyN x, ref fProxyN y) { s_applyCount++; return Blas.dotSelf(in A, in x, ref y); }
        public void ApplyBlock(in fProxyMxN Vrows, ref fProxyMxN AVrows, int rows) => throw new NotSupportedException();
    }

    // The EXACT cg<TOp> loop minus the verify-at-exit block (same Blas.dot/Blas.updateXR kernels
    // the real cg uses) -- an independent "before" oracle for verify-at-exit's effect.
    static SolveInfo UnguardedCg<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                       ref fProxyN r, ref fProxyN p, ref fProxyN Ap,
                                       int maxIter, fProxy tol)
        where TOp : struct, IfProxyLinearOperator
    {
        fProxy bb = Blas.dot(b, b);
        if (bb == (fProxy)0)
        {
            x.Data.CopyFrom(b.Data);
            return new SolveInfo { rnorm = 0, iterations = 0, status = IterativeSolveStatus.Converged };
        }

        A.Apply(in x, ref Ap);
        r.Data.CopyFrom(b.Data);
        r.addScaledInPlace((fProxy)(-1), Ap);
        p.Data.CopyFrom(r.Data);

        fProxy rsold = Blas.dot(r, r);
        fProxy threshold = tol * tol * bb;

        if (rsold <= threshold)
            return new SolveInfo { rnorm = (double)math.sqrt(rsold), iterations = 0, status = IterativeSolveStatus.Converged };

        for (int k = 0; k < maxIter; k++)
        {
            fProxy pAp = A.ApplyDot(in p, ref Ap);
            if (!(pAp > (fProxy)0))
                return new SolveInfo { rnorm = (double)math.sqrt(rsold), iterations = k, status = IterativeSolveStatus.Breakdown };

            fProxy alpha = rsold / pAp;
            fProxy rsnew = Blas.updateXR(alpha, p, ref x, Ap, ref r);

            if (rsnew <= threshold)   // NO verify -- claims convergence on the tracked residual alone
                return new SolveInfo { rnorm = (double)math.sqrt(rsnew), iterations = k + 1, status = IterativeSolveStatus.Converged };

            fProxy beta = rsnew / rsold;
            p.scaleAddInPlace(beta, r);
            rsold = rsnew;
        }

        return new SolveInfo { rnorm = (double)math.sqrt(rsold), iterations = maxIter, status = IterativeSolveStatus.MaxIterations };
    }

    // ==============================================================================
    // (c) drift-firing: ill-conditioned float Moler instance -- the unguarded oracle claims
    //     convergence while the true residual is still (comfortably) above tolerance; the real,
    //     guarded Krylov.cg must not repeat that mistake.
    // ==============================================================================

    // n/alpha/tol are tuned to this library's actual Blas.dot/Blas.updateXR kernels (a
    // 2x-accumulator SIMD fold, a deterministic summation order). float's ~7-digit precision lets
    // the recursive residual drift past the true one at this tolerance/problem-size combination
    // AND still recover to honest convergence a handful of iterations later; double's ~16 digits
    // don't lie at this size, so requireLie gates the "must have lied" assertion to the float
    // build only -- the guarded solver's honesty check below is unconditional and holds for both
    // dtypes.
    [Test]
    public void VerifyAtExitCatchesOptimisticDriftOnIllConditionedMoler()
    {
        var arena = new Arena(Allocator.Persistent);

        // n/alpha retuned for the fProxyW (8-lane) float reduction tree: at (16, -0.13) the
        // unguarded oracle lies (true residual 1.7x above tolerance) AND the guarded solver
        // still recovers to honest convergence. Any future reduction-tree change re-tunes this
        // pair with the same lie-plus-guarded-recovery sweep.
        int n = 16;
        var Adense = arena.fProxyMoler(n, (fProxy)(-0.13f));
        var op = new fProxyDenseOperator(in Adense);

        var b = arena.fProxyVec(n);
        for (int i = 0; i < n; i++) b[i] = (fProxy)1;

        fProxy tol = /*+choose[1e-6f|1e-6]*/1e-6f/*-choose*/;
        int maxIter = 500;

        var xU = arena.fProxyVec(n);
        var rU = arena.fProxyVec(n);
        var pU = arena.fProxyVec(n);
        var ApU = arena.fProxyVec(n);
        var infoU = UnguardedCg(in op, in b, ref xU, ref rU, ref pU, ref ApU, maxIter, tol);

        var scratchU = arena.fProxyVec(n);
        fProxy trueRsU = TrueResidualSq(in op, in b, in xU, ref scratchU);
        fProxy threshold = tol * tol * Blas.dot(b, b);

        bool requireLie = /*+choose[true|false]*/true/*-choose*/;
        if (requireLie)
        {
            Assert.IsTrue(infoU.Solved, "unguarded oracle should claim convergence on this instance: " + infoU);
            Assert.Greater((double)trueRsU, (double)threshold,
                "unguarded oracle's claimed convergence should be a LIE (true residual still above tolerance)");
        }

        var xG = arena.fProxyVec(n);
        var rG = arena.fProxyVec(n);
        var pG = arena.fProxyVec(n);
        var ApG = arena.fProxyVec(n);
        var infoG = Krylov.cg(in op, in b, ref xG, ref rG, ref pG, ref ApG, maxIter, tol);

        var scratchG = arena.fProxyVec(n);
        fProxy trueRsG = TrueResidualSq(in op, in b, in xG, ref scratchG);

        // The CORE verify-at-exit guarantee, robust to summation-order details: the guarded
        // solver never claims convergence while the true residual is above tolerance.
        Assert.IsFalse(infoG.Solved && (double)trueRsG > (double)threshold,
            "guarded solver claimed convergence but the true residual is above tolerance: " + infoG);

        // On the drift-tuned float instance, eventual recovery is problem- and rounding-
        // dependent (the template-stub build's numerics differ slightly from the generated
        // float build's), so the float side accepts honest non-convergence (MaxIterations /
        // Breakdown) as well as honest convergence — the lie being CAUGHT is the contract.
        // double does not lie at this size and must converge outright.
        if (!requireLie)
            Assert.IsTrue(infoG.Solved, "guarded solver must converge on this instance: " + infoG);

        if (infoG.Solved)
        {
            Assert.LessOrEqual((double)trueRsG, (double)threshold,
                "guarded solver must report HONEST convergence (true residual within tolerance)");

            // The reported rnorm on the Converged path is the VERIFIED true residual
            // (verify-at-exit contract).
            Assert.AreEqual((double)math.sqrt(trueRsG), infoG.rnorm, 1e-6 * (1.0 + infoG.rnorm));
        }

        arena.Dispose();
    }

    // ==============================================================================
    // (d) healthy-solve path: verify-at-exit adds exactly one extra Apply and does not change the
    //     solution vs a plain cg loop with no verify block (x is never touched by the verify block).
    // ==============================================================================

    [Test]
    public void VerifyAtExitAddsExactlyOneApplyAndPreservesSolutionOnHealthySolve()
    {
        var arena = new Arena(Allocator.Persistent);

        int n = 20;
        var Adense = BuildDenseSPD(ref arena, n, 141001);
        var b = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 141002);
        int maxIter = 4 * n;
        fProxy tol = Consts.fProxySqrtEps;

        s_applyCount = 0;
        var opU = new CountingDenseOperator(in Adense);
        var xU = arena.fProxyVec(n);
        var rU = arena.fProxyVec(n);
        var pU = arena.fProxyVec(n);
        var ApU = arena.fProxyVec(n);
        var infoU = UnguardedCg(in opU, in b, ref xU, ref rU, ref pU, ref ApU, maxIter, tol);
        int countU = s_applyCount;

        s_applyCount = 0;
        var opG = new CountingDenseOperator(in Adense);
        var xG = arena.fProxyVec(n);
        var rG = arena.fProxyVec(n);
        var pG = arena.fProxyVec(n);
        var ApG = arena.fProxyVec(n);
        var infoG = Krylov.cg(in opG, in b, ref xG, ref rG, ref pG, ref ApG, maxIter, tol);
        int countG = s_applyCount;

        Assert.IsTrue(infoU.Solved, infoU.ToString());
        Assert.IsTrue(infoG.Solved, infoG.ToString());
        Assert.AreEqual(infoU.iterations, infoG.iterations, "well-conditioned instance: no drift expected, so both should trigger at the same iteration");
        Assert.AreEqual(countU + 1, countG, "verify-at-exit must cost exactly one extra Apply/ApplyDot call");

        for (int i = 0; i < n; i++)
            Assert.AreEqual((double)xU[i], (double)xG[i]);   // x untouched by the verify block -> bit-identical

        arena.Dispose();
    }

    // ==============================================================================
    // Lighter wiring-sanity check for cg: confirm it still converges and its
    // Converged-path rnorm is HONEST (matches an independently-recomputed fresh residual), i.e.
    // the verify block compiles and returns the right value for every verify-at-exit-covered
    // solver, not just cg. Well-conditioned instances (no drift-firing construction needed here --
    // that burden is carried by the cg test above; verify-at-exit's code path is identical in
    // shape across all four solvers).
    // ==============================================================================

    [Test]
    public void PcgConvergedRnormIsHonest()
    {
        var arena = new Arena(Allocator.Persistent);

        int n = 16;
        var A = BuildDenseSPD(ref arena, n, 142001);
        var b = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 142002);

        var op = new fProxyDenseOperator(in A);
        var x = arena.fProxyVec(n);
        var info = Krylov.cg(op, new fProxyIdentityPreconditioner(), in b, ref x, 4 * n, Consts.fProxySqrtEps);
        Assert.IsTrue(info.Solved, info.ToString());

        var scratch = arena.fProxyVec(n);
        fProxy trueRs = TrueResidualSq(in op, in b, in x, ref scratch);
        fProxy threshold = Consts.fProxySqrtEps * Consts.fProxySqrtEps * Blas.dot(b, b);
        Assert.LessOrEqual((double)trueRs, (double)threshold);
        Assert.AreEqual((double)math.sqrt(trueRs), info.rnorm, 1e-6 * (1.0 + info.rnorm));

        arena.Dispose();
    }

    // ==============================================================================
    // minresQLP honesty gate: its QLP stopping metric rnorm/(Anorm*xnorm+beta1) can be
    // deflated below tol by a large Anorm*xnorm on a near-breakdown spectrum, flagging
    // Converged while the true ‖b-Ax‖/‖b‖ is large. The exit gate accepts a Converged only
    // with one of two fresh-residual certificates: compatible (‖r‖ <= 64*tol*‖b‖) or
    // least-squares optimality (‖A·r‖ <= 64*tol*Anorm*‖r‖ -- legitimate on an INCOMPATIBLE
    // singular system, where the LS optimum's raw residual is inherently large). The tests:
    // never a Solved that fails BOTH certificates (Rosser, compatible and incompatible b);
    // a genuine min-length LS optimum on an exactly-singular A IS reported Solved with the
    // oracle x (diag + Householder-conjugated instances); and a genuine compatible
    // convergence is never rejected (well-conditioned).
    // ==============================================================================

    // The gate's invariant, checked from OUTSIDE the solver via fresh recomputes:
    // Solved ⟹ ‖b-Ax‖ <= 64*tol*‖b‖ (compatible) OR ‖A(b-Ax)‖ <= 64*tol*‖A‖F*‖b-Ax‖
    // (least-squares optimality; A symmetric so A·r = Aᵀr). ‖A‖F >= ‖A‖2 >= the solver's
    // internal Anorm estimate, so anything the solver certifies also passes here, while a
    // false Converged (wrong x on a resolvable system) passes neither.
    static void AssertSolvedImpliesCertified(in fProxyMxN A, in fProxyN b, in fProxyN x,
                                             SolveInfo info, fProxy tol, ref Arena arena, string what)
    {
        if (!info.Solved) return;
        var op = new fProxyDenseOperator(in A);
        var r = arena.fProxyVec(b.N);
        double rnorm = math.sqrt((double)TrueResidualSq(in op, in b, in x, ref r));   // r = b - Ax
        double bnorm = math.sqrt((double)Blas.dot(b, b));
        if (rnorm <= (double)((fProxy)64 * tol) * bnorm)
            return;   // compatible certificate
        var Ar = arena.fProxyVec(b.N);
        op.Apply(in r, ref Ar);
        double arnorm = math.sqrt((double)Blas.dot(Ar, Ar));
        Assert.LessOrEqual(arnorm, (double)((fProxy)64 * tol) * (double)Norms.L2(in A) * rnorm,
            what + ": Solved with a large raw residual must be a certified least-squares optimum: " + info);
    }

    [Test]
    public void MinresQLPNeverFalseConvergesOnRosser()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.fProxyRosser();            // 8x8 symmetric, exactly singular, clustered spectrum
        int n = A.M_Rows;
        var b = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 143003);   // INCOMPATIBLE (null-space component)

        fProxy tol = Consts.fProxySqrtEps;
        var x = arena.fProxyVec(n);
        var info = Krylov.minresQLP(in A, in b, ref x, 8 * n, tol);

        // The LS optimum's raw residual is inherently large here (‖b_null‖ ~ ‖b‖/√n), so a raw
        // small-residual demand would forbid every correct answer; the certificate invariant
        // instead allows a genuine LS optimum and still forbids a false Converged.
        AssertSolvedImpliesCertified(in A, in b, in x, info, tol, ref arena, "minresQLP Rosser incompatible");

        arena.Dispose();
    }

    [Test]
    public void MinresQLPNeverFalseConvergesOnCompatibleRosser()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.fProxyRosser();
        int n = A.M_Rows;
        var z = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 143005);
        var b = arena.fProxyVec(n);
        Blas.dot(in A, in z, ref b);             // b = A z: COMPATIBLE, the #53 false-Converged class

        fProxy tol = Consts.fProxySqrtEps;
        var x = arena.fProxyVec(n);
        var info = Krylov.minresQLP(in A, in b, ref x, 8 * n, tol);

        // Compatible system: r = b - Ax stays in range(A), where Rosser's smallest nonzero
        // |eigenvalue| (~0.098) keeps ‖A·r‖/‖r‖ far above the double LS-certificate bound -- so
        // in double this invariant collapses to the original #53 guard (never Solved with a
        // large raw residual), and in float to the documented rank-cutoff semantics.
        AssertSolvedImpliesCertified(in A, in b, in x, info, tol, ref arena, "minresQLP Rosser compatible");

        arena.Dispose();
    }

    // Positive LS oracle: A = diag(1,1,0), b = (1,1,1) -- exactly singular, INCOMPATIBLE, with
    // the min-length least-squares solution known exactly: x* = (1,1,0), r* = (0,0,1),
    // ‖r*‖ = 1 (large: rel residual 1/√3), A·r* = 0. Must be reported Converged via the LS
    // certificate (the raw residual can never pass the compatible one), with the oracle x.
    [Test]
    public void MinresQLPSolvesSingularDiagLeastSquaresToMinLengthOracle()
    {
        var arena = new Arena(Allocator.Persistent);

        int n = 3;
        var A = arena.fProxyMat(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i, j] = (fProxy)0;
        A[0, 0] = (fProxy)1; A[1, 1] = (fProxy)1;   // A[2,2] stays 0
        var b = arena.fProxyVec(n);
        for (int i = 0; i < n; i++) b[i] = (fProxy)1;

        fProxy tol = Consts.fProxySqrtEps;
        var x = arena.fProxyVec(n);
        var info = Krylov.minresQLP(in A, in b, ref x, 50, tol);

        var op = new fProxyDenseOperator(in A);
        var r = arena.fProxyVec(n);
        double rnorm = math.sqrt((double)TrueResidualSq(in op, in b, in x, ref r));
        double bnorm = math.sqrt((double)Blas.dot(b, b));

        // The instance is incompatible for ANY x (every residual has the fixed null-space part
        // b[2] = 1), so a Converged here can only come through the LS certificate, never the
        // compatible one.
        Assert.Greater(rnorm, (double)((fProxy)64 * tol) * bnorm,
            "instance must be incompatible (large raw residual) or it does not exercise the LS certificate");

        // double resolves the exactly-singular terminal Lanczos iteration to oracle precision and
        // must deliver the min-length LS solution outright; float's terminal iteration is
        // rounding-limited on this instance, so the float build accepts an HONEST non-convergence
        // (never a false Converged -- the oracle checks below still bind any Solved it claims).
        bool strictOracle = /*+choose[false|true]*/false/*-choose*/;
        if (strictOracle)
            Assert.IsTrue(info.Solved,
                "minresQLP must report Converged for the exact min-length LS solution of diag(1,1,0): " + info);

        if (info.Solved)
        {
            double xTol = /*+choose[1e-4|1e-10]*/1e-4/*-choose*/;
            Assert.AreEqual(1.0, (double)x[0], xTol, "x[0] must match the min-length oracle (1,1,0)");
            Assert.AreEqual(1.0, (double)x[1], xTol, "x[1] must match the min-length oracle (1,1,0)");
            Assert.AreEqual(0.0, (double)x[2], xTol, "x[2] must match the min-length oracle (1,1,0)");

            Assert.AreEqual(1.0, rnorm, xTol, "‖b-Ax‖ must match the oracle residual ‖r*‖ = 1");
            Assert.AreEqual(rnorm, info.rnorm, xTol, "reported rnorm must be the fresh ‖b-Ax‖");

            var Ar = arena.fProxyVec(n);
            op.Apply(in r, ref Ar);
            double arnorm = math.sqrt((double)Blas.dot(Ar, Ar));
            Assert.AreEqual(0.0, arnorm, xTol, "optimality residual ‖A·r‖ must vanish at the LS optimum");
        }

        arena.Dispose();
    }

    // Positive LS oracle, non-diagonal: A = Q D Qᵀ with D = diag(3,2,1.5,1,0,0) and Q the
    // Householder reflector of v = (1,..,6) -- exactly singular with a genuine multi-step
    // Lanczos run. b = Q·ones so Qᵀb = ones exactly: min-length x* = Q D⁺ Qᵀ b and
    // ‖r*‖ = √2 (the null-space part of b) are hand-computable oracles.
    [Test]
    public void MinresQLPSolvesConjugatedSingularLeastSquaresToMinLengthOracle()
    {
        var arena = new Arena(Allocator.Persistent);

        int n = 6;
        var d = new double[] { 3, 2, 1.5, 1, 0, 0 };
        var v = new double[n];
        double vv = 0;
        for (int i = 0; i < n; i++) { v[i] = i + 1; vv += v[i] * v[i]; }

        // Q = I - 2 v vᵀ / (vᵀv): orthogonal, symmetric. Build Q, A = Q D Qᵀ, b = Q·ones and
        // the pinv oracle x* = Q D⁺ Qᵀ b in double, then cast into the proxy matrix/vectors.
        var Q = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Q[i, j] = (i == j ? 1.0 : 0.0) - 2.0 * v[i] * v[j] / vv;

        var A = arena.fProxyMat(n, n);
        var b = arena.fProxyVec(n);
        var xStar = new double[n];
        for (int i = 0; i < n; i++)
        {
            double bi = 0, xi = 0;
            for (int j = 0; j < n; j++)
            {
                double aij = 0;
                for (int k = 0; k < n; k++) aij += Q[i, k] * d[k] * Q[j, k];
                A[i, j] = (fProxy)aij;
                bi += Q[i, j];                                   // b = Q·(1,..,1)
                xi += d[j] > 0 ? Q[i, j] / d[j] : 0.0;           // x* = Q·D⁺·(1,..,1)
            }
            b[i] = (fProxy)bi;
            xStar[i] = xi;
        }
        double rStarSq = 2.0;   // Qᵀb = ones -> null-space part (last two coords) has ‖·‖² = 2

        fProxy tol = Consts.fProxySqrtEps;
        var x = arena.fProxyVec(n);
        var info = Krylov.minresQLP(in A, in b, ref x, 100, tol);

        var op = new fProxyDenseOperator(in A);
        var r = arena.fProxyVec(n);
        double rnorm = math.sqrt((double)TrueResidualSq(in op, in b, in x, ref r));
        double bnorm = math.sqrt((double)Blas.dot(b, b));

        // Incompatible for ANY x (the null-space part of b is fixed): a Converged here can only
        // come through the LS certificate, never the compatible one.
        Assert.Greater(rnorm, (double)((fProxy)64 * tol) * bnorm,
            "instance must be incompatible (large raw residual) or it does not exercise the LS certificate");

        // Same strict-double / honest-float split as the diag oracle test above.
        bool strictOracle = /*+choose[false|true]*/false/*-choose*/;
        if (strictOracle)
            Assert.IsTrue(info.Solved,
                "minresQLP must report Converged for the min-length LS solution of the conjugated singular A: " + info);

        if (info.Solved)
        {
            double xTol = /*+choose[5e-3|1e-9]*/5e-3/*-choose*/;
            for (int i = 0; i < n; i++)
                Assert.AreEqual(xStar[i], (double)x[i], xTol, "x[" + i + "] must match the pinv min-length oracle");

            Assert.AreEqual(math.sqrt(rStarSq), rnorm, xTol, "‖b-Ax‖ must match the oracle residual √2");
            Assert.AreEqual(rnorm, info.rnorm, xTol, "reported rnorm must be the fresh ‖b-Ax‖");

            var Ar = arena.fProxyVec(n);
            op.Apply(in r, ref Ar);
            double arnorm = math.sqrt((double)Blas.dot(Ar, Ar));
            Assert.LessOrEqual(arnorm, (double)((fProxy)64 * tol) * (double)Norms.L2(in A) * rnorm,
                "optimality residual ‖A·r‖ must pass the LS certificate at the oracle solution");
        }

        arena.Dispose();
    }

    [Test]
    public void MinresQLPStillConvergesHonestlyOnWellConditioned()
    {
        var arena = new Arena(Allocator.Persistent);

        int n = 20;
        var A = BuildDenseSPD(ref arena, n, 143001);   // symmetric, well-conditioned
        var op = new fProxyDenseOperator(in A);
        var b = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 143002);

        fProxy tol = Consts.fProxySqrtEps;
        var x = arena.fProxyVec(n);
        var info = Krylov.minresQLP(in A, in b, ref x, 8 * n, tol);

        // The honesty guard must NOT reject a genuine convergence.
        Assert.IsTrue(info.Solved, "minresQLP must still converge on a well-conditioned system: " + info);

        var scratch = arena.fProxyVec(n);
        fProxy trueRs = TrueResidualSq(in op, in b, in x, ref scratch);
        double bound = (double)((fProxy)64 * tol) * (double)((fProxy)64 * tol) * (double)Blas.dot(b, b);
        Assert.LessOrEqual((double)trueRs, bound, "minresQLP Converged but true residual exceeds the honesty bound: " + info);

        arena.Dispose();
    }

    // ==============================================================================
    // minres (identity / unpreconditioned path) honesty guard: once gamma is floored the Givens
    // recurrence loses unitarity and phibar can decay away from the true ‖b-Ax‖, flagging
    // Converged while the residual is large. The verify-at-exit block must NOT claim convergence
    // with a large true residual (Rosser), and must NOT reject a genuine convergence (SPD).
    // ==============================================================================

    [Test]
    public void MinresNeverFalseConvergesOnRosser()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.fProxyRosser();            // 8x8 symmetric, clustered near-degenerate spectrum
        int n = A.M_Rows;
        var op = new fProxyDenseOperator(in A);
        var b = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 144003);

        fProxy tol = Consts.fProxySqrtEps;
        var x = arena.fProxyVec(n);
        var info = Krylov.minres(in A, in b, ref x, 8 * n, tol);

        var scratch = arena.fProxyVec(n);
        fProxy trueRs = TrueResidualSq(in op, in b, in x, ref scratch);
        double honestBoundSq = (double)((fProxy)64 * tol) * (double)((fProxy)64 * tol) * (double)Blas.dot(b, b);

        // Never Solved while the true residual exceeds the raw bound (the pre-fix identity path
        // trusted phibar unconditionally and could report Solved with a large true residual).
        Assert.IsFalse(info.Solved && (double)trueRs > honestBoundSq,
            "minres claimed convergence on Rosser but the true residual is large: " + info);

        arena.Dispose();
    }

    [Test]
    public void MinresStillConvergesHonestlyOnWellConditioned()
    {
        var arena = new Arena(Allocator.Persistent);

        int n = 20;
        var A = BuildDenseSPD(ref arena, n, 144001);   // symmetric, well-conditioned
        var op = new fProxyDenseOperator(in A);
        var b = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 144002);

        fProxy tol = Consts.fProxySqrtEps;
        var x = arena.fProxyVec(n);
        var info = Krylov.minres(in A, in b, ref x, 8 * n, tol);

        // The verify-at-exit block must NOT over-reject a genuine convergence.
        Assert.IsTrue(info.Solved, "minres must still converge on a well-conditioned system: " + info);

        var scratch = arena.fProxyVec(n);
        fProxy trueRs = TrueResidualSq(in op, in b, in x, ref scratch);
        double bound = (double)((fProxy)64 * tol) * (double)((fProxy)64 * tol) * (double)Blas.dot(b, b);
        Assert.LessOrEqual((double)trueRs, bound, "minres Converged but true residual exceeds the honesty bound: " + info);

        arena.Dispose();
    }

    // ==============================================================================
    // biCGStab honesty guard: r and x are propagated on separate recurrences that only equal
    // b-Ax in exact arithmetic; a near-zero pivot (rho/rv/omega) can decouple them so ss/rr reads
    // small while the true residual is large. Both verify sites (early-exit ss, main rr) must NOT
    // claim convergence with a large true residual (Rosser), and must NOT reject a genuine
    // convergence (well-conditioned nonsymmetric).
    // ==============================================================================

    [Test]
    public void BiCGStabNeverFalseConvergesOnRosser()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.fProxyRosser();
        int n = A.M_Rows;
        var op = new fProxyDenseOperator(in A);
        var b = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 145003);

        fProxy tol = Consts.fProxySqrtEps;
        var x = arena.fProxyVec(n);
        var info = Krylov.biCGStab(in A, in b, ref x, 8 * n, tol);

        var scratch = arena.fProxyVec(n);
        fProxy trueRs = TrueResidualSq(in op, in b, in x, ref scratch);
        double honestBoundSq = (double)((fProxy)64 * tol) * (double)((fProxy)64 * tol) * (double)Blas.dot(b, b);

        Assert.IsFalse(info.Solved && (double)trueRs > honestBoundSq,
            "biCGStab claimed convergence on Rosser but the true residual is large: " + info);

        arena.Dispose();
    }

    [Test]
    public void BiCGStabStillConvergesHonestlyOnWellConditioned()
    {
        var arena = new Arena(Allocator.Persistent);

        int n = 20;
        var A = BuildDenseNonsym(ref arena, n, 145001);   // nonsymmetric, diagonally dominant
        var op = new fProxyDenseOperator(in A);
        var b = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 145002);

        fProxy tol = Consts.fProxySqrtEps;
        var x = arena.fProxyVec(n);
        var info = Krylov.biCGStab(in A, in b, ref x, 8 * n, tol);

        Assert.IsTrue(info.Solved, "biCGStab must still converge on a well-conditioned system: " + info);

        var scratch = arena.fProxyVec(n);
        fProxy trueRs = TrueResidualSq(in op, in b, in x, ref scratch);
        double bound = (double)((fProxy)64 * tol) * (double)((fProxy)64 * tol) * (double)Blas.dot(b, b);
        Assert.LessOrEqual((double)trueRs, bound, "biCGStab Converged but true residual exceeds the honesty bound: " + info);

        arena.Dispose();
    }

    // ==============================================================================
    // idr honesty guard: R and x are propagated on separate recurrences (same shape as biCGStab);
    // a near-zero shadow-space pivot can make rr read small while the true residual is large. Both
    // verify sites (in-sweep, end-of-sweep) must NOT claim convergence with a large true residual
    // (Rosser), and must NOT reject a genuine convergence (well-conditioned nonsymmetric).
    // ==============================================================================

    [Test]
    public void IdrNeverFalseConvergesOnRosser()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = arena.fProxyRosser();
        int n = A.M_Rows;
        var op = new fProxyDenseOperator(in A);
        var b = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 146003);

        fProxy tol = Consts.fProxySqrtEps;
        var x = arena.fProxyVec(n);
        var info = Krylov.idr(in A, in b, ref x, 4, 20 * n, tol);

        var scratch = arena.fProxyVec(n);
        fProxy trueRs = TrueResidualSq(in op, in b, in x, ref scratch);
        double honestBoundSq = (double)((fProxy)64 * tol) * (double)((fProxy)64 * tol) * (double)Blas.dot(b, b);

        Assert.IsFalse(info.Solved && (double)trueRs > honestBoundSq,
            "idr claimed convergence on Rosser but the true residual is large: " + info);

        arena.Dispose();
    }

    [Test]
    public void IdrStillConvergesHonestlyOnWellConditioned()
    {
        var arena = new Arena(Allocator.Persistent);

        int n = 20;
        var A = BuildDenseNonsym(ref arena, n, 146001);   // nonsymmetric, diagonally dominant
        var op = new fProxyDenseOperator(in A);
        var b = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 146002);

        fProxy tol = Consts.fProxySqrtEps;
        var x = arena.fProxyVec(n);
        var info = Krylov.idr(in A, in b, ref x, 4, 20 * n, tol);

        Assert.IsTrue(info.Solved, "idr must still converge on a well-conditioned system: " + info);

        var scratch = arena.fProxyVec(n);
        fProxy trueRs = TrueResidualSq(in op, in b, in x, ref scratch);
        double bound = (double)((fProxy)64 * tol) * (double)((fProxy)64 * tol) * (double)Blas.dot(b, b);
        Assert.LessOrEqual((double)trueRs, bound, "idr Converged but true residual exceeds the honesty bound: " + info);

        arena.Dispose();
    }

    // ==============================================================================
    // biCGStab verify-fail-continue: on a drift-prone ill-conditioned system the recurrence
    // residual can dip under threshold while the true residual has not, so the half-step /
    // end-of-iteration verify fails and iteration continues. The scratch vector v (= A M^-1 p)
    // that the verify overwrites used to be left holding A*x, corrupting the next iteration's
    // p-recurrence and stalling the solve to a wrong answer. A weak-diagonal nonsymmetric A run
    // to a tight tol exercises that continue path; a correct solver still reaches a finite,
    // honestly-converged x. Path-exercising guard (the drift is not guaranteed on any fixed seed),
    // so it asserts the outcome a correct solver must deliver, never a stall to a wrong answer.
    // ==============================================================================
    [Test]
    public void BiCGStabVerifyContinueKeepsConvergingOnDrift()
    {
        var arena = new Arena(Allocator.Persistent);

        int n = 28;
        // Reduced diagonal dominance (0.5*n) -> worse conditioned than the well-conditioned case
        // above, but still nonsingular and solvable within a generous budget.
        var A = arena.fProxyRandomMat(n, n, (fProxy)(-1f), (fProxy)1f, 145011);
        for (int i = 0; i < n; i++) A[i, i] += (fProxy)(0.5f * n);
        var op = new fProxyDenseOperator(in A);

        var xStar = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 145012);
        var b = arena.fProxyVec(n);
        Blas.dot(in A, in xStar, ref b);   // b = A x*

        fProxy tol = Consts.fProxySqrtEps;
        var x = arena.fProxyVec(n);
        var info = Krylov.biCGStab(in A, in b, ref x, 300 * n, tol);   // generous budget

        for (int i = 0; i < n; i++)
            Assert.IsFalse(double.IsNaN((double)x[i]) || double.IsInfinity((double)x[i]),
                "biCGStab produced a non-finite x at " + i);

        // Reaches an honest solution -- pre-fix, the corrupted p-recurrence could stall to a wrong x.
        Assert.IsTrue(info.Solved, "biCGStab failed to reach an honest solution on the drift-prone system: " + info);

        var scratch = arena.fProxyVec(n);
        fProxy trueRs = TrueResidualSq(in op, in b, in x, ref scratch);
        double bound = (double)((fProxy)64 * tol) * (double)((fProxy)64 * tol) * (double)Blas.dot(b, b);
        Assert.LessOrEqual((double)trueRs, bound,
            "biCGStab converged to a wrong answer (true residual exceeds the honesty bound): " + info);

        arena.Dispose();
    }

}
