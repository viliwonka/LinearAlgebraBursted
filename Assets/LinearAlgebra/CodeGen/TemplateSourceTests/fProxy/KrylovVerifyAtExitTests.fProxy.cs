using System;
using LinearAlgebra;
using LinearAlgebra.Gallery;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

// Krylov verify-at-exit: cg/pcg recursively
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
    // Lighter wiring-sanity check for pcg: confirm it still converges and its
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
        var info = Krylov.pcg(op, new fProxyIdentityPreconditioner(), in b, ref x, 4 * n, Consts.fProxySqrtEps);
        Assert.IsTrue(info.Solved, info.ToString());

        var scratch = arena.fProxyVec(n);
        fProxy trueRs = TrueResidualSq(in op, in b, in x, ref scratch);
        fProxy threshold = Consts.fProxySqrtEps * Consts.fProxySqrtEps * Blas.dot(b, b);
        Assert.LessOrEqual((double)trueRs, (double)threshold);
        Assert.AreEqual((double)math.sqrt(trueRs), info.rnorm, 1e-6 * (1.0 + info.rnorm));

        arena.Dispose();
    }

}
