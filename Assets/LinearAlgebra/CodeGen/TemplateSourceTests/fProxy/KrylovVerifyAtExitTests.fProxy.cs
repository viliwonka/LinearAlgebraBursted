using System;
using LinearAlgebra;
using LinearAlgebra.Gallery;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

// Krylov verify-at-exit: cg/pcg/cgls/cgne recursively
// track their residual; in float that tracked value can drift from the true b-Ax and claim
// convergence early. When the tracked residual FIRST claims convergence, the guarded solver now
// recomputes the true residual fresh (+1 Apply, +1 ApplyT for cgls) and retests, continuing if it
// fails. Only at claimed convergence -- MaxIterations/Breakdown exits are untouched.
//
// (c) drift-firing: an ill-conditioned float instance (Moler matrix, deep tolerance) where the
//     tracked residual claims convergence while the true residual has not actually met tolerance
//     -- proven directly against an in-test replica of the PRE-R6a cg loop (same Blas kernels,
//     just without the verify block), so this does not depend on git history. The guarded
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

    // The EXACT pre-R6a cg<TOp> loop (same Blas.dot/Blas.updateXR kernels the real cg uses),
    // minus the verify-at-exit block -- an independent "before" oracle without touching git.
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

            if (rsnew <= threshold)   // NO verify -- the pre-R6a behavior
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

    // n/alpha/tol found by first prototyping the recurrence in a throwaway dotnet console app
    // (naive scalar dot order), then confirmed/retuned directly against this library's actual
    // Blas.dot/Blas.updateXR kernels (a 2x-accumulator SIMD fold -- a DIFFERENT, still
    // deterministic summation order that shifts exactly where the drift crosses the tolerance
    // boundary) via a throwaway diagnostic sweep run through Unity, since the two orders don't
    // land on the same iteration/margin. float's ~7-digit precision lets the recursive residual
    // drift past the true one at this tolerance/problem-size combination AND still recover to
    // honest convergence a handful of iterations later; double's ~16 digits don't lie at this
    // size, so requireLie gates the "must have lied" assertion to the float build only -- the
    // guarded solver's honesty check below is unconditional and holds for both dtypes.
    [Test]
    public void VerifyAtExitCatchesOptimisticDriftOnIllConditionedMoler()
    {
        var arena = new Arena(Allocator.Persistent);

        int n = 20;
        var Adense = arena.fProxyMoler(n, (fProxy)(-0.11f));
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

        Assert.IsTrue(infoG.Solved, "guarded solver must still converge on this instance: " + infoG);

        var scratchG = arena.fProxyVec(n);
        fProxy trueRsG = TrueResidualSq(in op, in b, in xG, ref scratchG);
        Assert.LessOrEqual((double)trueRsG, (double)threshold,
            "guarded solver must report HONEST convergence (true residual within tolerance)");

        // The reported rnorm on the Converged path is the VERIFIED true residual (R6a contract).
        Assert.AreEqual((double)math.sqrt(trueRsG), infoG.rnorm, 1e-6 * (1.0 + infoG.rnorm));

        arena.Dispose();
    }

    // ==============================================================================
    // (d) healthy-solve path: verify-at-exit adds exactly one extra Apply and does not change the
    //     solution vs the pre-R6a behavior (x is never touched by the verify block).
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
    // Lighter wiring-sanity checks for pcg/cgls/cgne: confirm each still converges and its
    // Converged-path rnorm is HONEST (matches an independently-recomputed fresh residual), i.e.
    // the verify block compiles and returns the right value for every R6a-covered solver, not
    // just cg. Well-conditioned instances (no drift-firing construction needed here -- that
    // burden is carried by the cg test above; R6a's code path is identical in shape across all
    // four solvers).
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

    [Test]
    public void CgneConvergedRnormIsHonest()
    {
        var arena = new Arena(Allocator.Persistent);

        int n = 16;
        var A = BuildDenseSPD(ref arena, n, 143001);
        var b = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 143002);

        // cgne has no generic-TOp allocating convenience overload (locked ladder: only the
        // zero-alloc generic core + concrete dense/BSR forwarders) -- solve via the concrete dense
        // overload, then wrap in fProxyDenseOperator just for the independent audit below.
        var x = arena.fProxyVec(n);
        var info = Krylov.cgne(in A, in b, ref x, 4 * n, Consts.fProxySqrtEps);
        Assert.IsTrue(info.Solved, info.ToString());

        var op = new fProxyDenseOperator(in A);
        var scratch = arena.fProxyVec(n);
        fProxy trueRs = TrueResidualSq(in op, in b, in x, ref scratch);
        fProxy threshold = Consts.fProxySqrtEps * Consts.fProxySqrtEps * Blas.dot(b, b);
        Assert.LessOrEqual((double)trueRs, (double)threshold);
        Assert.AreEqual((double)math.sqrt(trueRs), info.rnorm, 1e-6 * (1.0 + info.rnorm));

        arena.Dispose();
    }

    [Test]
    public void CglsConvergedResidualIsHonest()
    {
        var arena = new Arena(Allocator.Persistent);

        // Rectangular over-determined system -- cgls's normal case.
        int m = 24, nCols = 10;
        var A = arena.fProxyRandomMat(m, nCols, (fProxy)(-1f), (fProxy)1f, 144001);
        var b = arena.fProxyRandomVec(m, (fProxy)(-1f), (fProxy)1f, 144002);

        // cgls, like cgne, has no generic-TOp allocating convenience overload (locked ladder) --
        // solve via the concrete dense overload, then wrap in fProxyDenseOperator for the
        // independent (generic) audit call below.
        var x = arena.fProxyVec(nCols);
        var info = Krylov.cgls(in A, in b, ref x, 8 * nCols, Consts.fProxySqrtEps);
        Assert.IsTrue(info.Solved, info.ToString());

        var op = new fProxyDenseOperator(in A);
        var rScratch = arena.fProxyVec(m);
        var sScratch = arena.fProxyVec(nCols);
        var audit = Krylov.lstsqResidual(op, in b, in x, (fProxy)0, ref rScratch, ref sScratch);

        // audit.Arnorm is the certified-exact optimality residual (fresh Apply+ApplyT) -- must be
        // small, and info's own tracked Arnorm must agree with it closely (both describe the same
        // converged x).
        Assert.Less(audit.Arnorm, 1e-2);
        Assert.AreEqual(audit.Arnorm, info.Arnorm, 1e-3 * (1.0 + info.Arnorm));

        arena.Dispose();
    }
}
