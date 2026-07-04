using System;

using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;

using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

// SCRATCH / smoke-test coverage for the new LOBPCG.lobpcg implementation, written by the coder
// agent purely to sanity-check the algorithm while iterating (per the task brief: "author tests
// only for what you need to iterate, mark clearly what needs independent verification"). This is
// NOT the comprehensive suite the spec calls for (analytic Laplacian oracle across k=1..4,
// dense-vs-eigenSymmetric cross-check, preconditioned-vs-unpreconditioned iteration-count
// comparison, rank-deficiency stress with k=n/2, k=1-vs-inversePowerIteration) -- that is left for
// the independent test-writer agent. Managed [Test]s (main thread), matching the simpler
// non-Burst-job test style used elsewhere in this file family (e.g. ArenaConversionsTests).
public class floatLOBPCGSmokeTests
{
    static float Tol() => 1e-3f;

    [Test]
    public void DiagonalSmallestTwoMatchKnownValues()
    {
        var arena = new Arena(Allocator.Persistent);

        int n = 6;
        var A = arena.floatMat(n, n);
        for (int i = 0; i < n; i++) A[i, i] = (float)(i + 1); // eigenvalues 1..6

        var eig = LOBPCG.lobpcg(ref arena, in A, 2, out var vecs, out var info);

        Assert.IsTrue(info.Solved, info.ToString());
        Assert.AreEqual(2, info.converged);
        Assert.AreEqual((float)1, eig[0], Tol());
        Assert.AreEqual((float)2, eig[1], Tol());

        // residual check ||A x_i - lambda_i x_i|| for both returned pairs
        for (int i = 0; i < 2; i++)
        {
            float maxAbs = 0;
            for (int c = 0; c < n; c++)
            {
                float Ax = A[c, c] * vecs[i, c];
                float r = math.abs(Ax - eig[i] * vecs[i, c]);
                if (r > maxAbs) maxAbs = r;
            }
            Assert.LessOrEqual((double)maxAbs, 1e-3);
        }

        arena.Dispose();
    }

    [Test]
    public void Laplacian1DThreeSmallestMatchAnalytic()
    {
        var arena = new Arena(Allocator.Persistent);

        int n = 12;
        var A = arena.floatLaplacian1D(n);

        var eig = LOBPCG.lobpcg(ref arena, in A, 3, out var vecs, out var info);

        Assert.IsTrue(info.Solved, info.ToString());

        for (int j = 1; j <= 3; j++)
        {
            double analytic = 2.0 - 2.0 * math.cos(j * math.PI_DBL / (n + 1));
            Assert.AreEqual(analytic, (double)eig[j - 1], 1e-2);
        }

        arena.Dispose();
    }

    [Test]
    public void KEqualsOneMatchesInversePowerIteration()
    {
        var arena = new Arena(Allocator.Persistent);

        int n = 10;
        var A = arena.floatLaplacian1D(n);

        var eig = LOBPCG.lobpcg(ref arena, in A, 1, out _, out var info);
        Assert.IsTrue(info.Solved);

        var v = arena.floatVec(n);
        var ok = Eigen.inversePowerIteration(in A, ref v, out float lambdaIPI);
        Assert.IsTrue(ok);

        Assert.AreEqual((double)lambdaIPI, (double)eig[0], 1e-2);

        arena.Dispose();
    }

    [Test]
    public void PreconditionedConvergesInFewerIterationsThanUnpreconditioned()
    {
        // 4x4 blocks (BR=4) on a small BSR with a strong diagonal so block-Jacobi is meaningfully
        // better conditioned than the raw system -- built directly as a block-diagonal-dominant
        // BSR via the builder (a tridiagonal-of-blocks Laplacian-like operator).
        var arena = new Arena(Allocator.Persistent);

        int blocks = 6, br = 3;
        int n = blocks * br;
        var builder = arena.floatBSRBuilder(blocks, blocks, br, br);

        for (int b = 0; b < blocks; b++)
        {
            var diag = new floatMxN(br, br, Allocator.Temp);
            for (int r = 0; r < br; r++)
                for (int c = 0; c < br; c++)
                    diag[r, c] = (r == c) ? (float)6 : (float)0.5;
            builder.AddBlock(b, b, in diag);
            diag.Dispose();

            if (b + 1 < blocks)
            {
                var off = new floatMxN(br, br, Allocator.Temp);
                for (int r = 0; r < br; r++)
                    for (int c = 0; c < br; c++)
                        off[r, c] = (r == c) ? (float)(-1) : (float)0;
                builder.AddBlock(b, b + 1, in off);
                builder.AddBlock(b + 1, b, in off);
                off.Dispose();
            }
        }

        var A = builder.ToBSR(ref arena);
        var M = new floatBlockJacobi(in A, Allocator.Persistent);

        int k = 2;

        var wsUnprecond = arena.floatLOBPCGCache(n, k);
        var infoUnprecond = LOBPCG.lobpcg(in A, ref wsUnprecond, k, Consts.floatSqrtEps, 500);

        var wsPrecond = arena.floatLOBPCGCache(n, k);
        var infoPrecond = LOBPCG.lobpcg(in A, in M, ref wsPrecond, k, Consts.floatSqrtEps, 500);

        Assert.IsTrue(infoUnprecond.Solved, infoUnprecond.ToString());
        Assert.IsTrue(infoPrecond.Solved, infoPrecond.ToString());
        Assert.Less(infoPrecond.iterations, infoUnprecond.iterations);

        M.Dispose();
        arena.Dispose();
    }

    // k == n/2 -> 3k = 12 > n = 8, so the combined [X,W,P] Gram is EXACTLY rank-deficient (at
    // most 8 of its 12 rows can be linearly independent) every iteration once P joins the mix --
    // this verifies the solve stays finite (no NaN/divergence) under repeated ridge-retry recovery
    // from that guaranteed degeneracy. It does NOT exercise the drop-P/stall fallback path: per
    // the class doc comment's safeguard 2, FactorGram's Tikhonov ridge absorbs an
    // exactly-singular-but-still-SPD-after-ridge Gram directly (Cholesky succeeds, comfortably
    // clears the pivot-ratio check), so the solver proceeds through a well-defined, heavily
    // regularized reduction rather than ever falling back to 2-block or stalling -- ridge-then-
    // proceed is the adjudicated PRIMARY handler for this kind of degeneracy; drop-P/stall remain
    // a defensive backstop for cases the ridge itself cannot repair (verified separately by the
    // Cholesky-failure/no-op-stall behavior documented on TryRayleighRitz, not by this test).
    [Test]
    public void RankDeficiencyStressDoesNotNaNOrDiverge()
    {
        var arena = new Arena(Allocator.Persistent);

        int n = 8;
        int k = 4;
        var A = arena.floatLaplacian1D(n);

        var eig = LOBPCG.lobpcg(ref arena, in A, k, out var vecs, out var info, Consts.floatSqrtEps, 300);

        Assert.AreNotEqual(IterativeSolveStatus.Breakdown, info.status);

        for (int i = 0; i < k; i++)
        {
            Assert.IsTrue(math.isfinite((float)eig[i]));
            for (int c = 0; c < n; c++)
                Assert.IsTrue(math.isfinite((float)vecs[i, c]));
        }

        arena.Dispose();
    }

    // Regression test for the periodic-seed bug: an earlier default X seed, `(i + c*3 + 1) & 3`,
    // repeats with period 4 in BOTH i and c, so the seeded block had AT MOST 4 distinct rows --
    // EXACTLY rank-deficient for any k > 4, silently absorbed by FactorGram's ridge retry rather
    // than failing loudly. k=6 here would have been unable to find eigenpairs 5/6 correctly under
    // that seed; a non-periodic deterministic (fixed-seed) fill is required to span all 6.
    [Test]
    public void DiagonalTwentySmallestSixMatchAnalyticAscending()
    {
        var arena = new Arena(Allocator.Persistent);

        int n = 20, k = 6;
        var A = arena.floatMat(n, n);
        for (int i = 0; i < n; i++) A[i, i] = (float)(i + 1); // eigenvalues 1..20

        var eig = LOBPCG.lobpcg(ref arena, in A, k, out _, out var info);

        Assert.IsTrue(info.Solved, info.ToString());
        Assert.AreEqual(k, info.converged);

        for (int j = 0; j < k; j++)
            Assert.AreEqual((float)(j + 1), eig[j], Tol());

        arena.Dispose();
    }

    // Cross-check against the full dense solver: LOBPCG's k smallest Ritz values must match
    // eigenSymmetric's k smallest (its last k entries, since it sorts DESCENDING) within tol, on
    // a well-conditioned SPD gallery matrix independent of the Laplacian used elsewhere in this file.
    [Test]
    public void MatchesEigenSymmetricSmallestFour()
    {
        var arena = new Arena(Allocator.Persistent);

        int n = 16, k = 4;
        var A = arena.floatLehmer(n);

        var Afull = A.Copy();
        var eigAll = arena.floatVec(n);
        var Vall = arena.floatMat(n, n);
        Assert.IsTrue(Eigen.eigenSymmetric(ref Afull, ref eigAll, ref Vall));

        var eig = LOBPCG.lobpcg(ref arena, in A, k, out _, out var info);
        Assert.IsTrue(info.Solved, info.ToString());

        for (int j = 0; j < k; j++)
            Assert.AreEqual((double)eigAll[n - 1 - j], (double)eig[j], 1e-2);

        arena.Dispose();
    }

    // Output eigenvectors must be mutually orthonormal after the solve: X X^T ≈ I (unit norm rows,
    // zero cross dot products) within tol.
    [Test]
    public void EigenvectorsAreOrthonormalAfterSolve()
    {
        var arena = new Arena(Allocator.Persistent);

        int n = 12, k = 4;
        var A = arena.floatLaplacian1D(n);

        var eig = LOBPCG.lobpcg(ref arena, in A, k, out var vecs, out var info);
        Assert.IsTrue(info.Solved, info.ToString());

        for (int i = 0; i < k; i++)
            for (int j = i; j < k; j++)
            {
                float dot = (float)0;
                for (int c = 0; c < n; c++) dot += vecs[i, c] * vecs[j, c];
                Assert.AreEqual(i == j ? (double)1 : 0.0, (double)dot, 1e-2);
            }

        arena.Dispose();
    }

    // Multiplicity: a repeated smallest eigenvalue (1,1,2,3,...) with k=2 must return BOTH copies
    // at the repeated value, with mutually orthogonal eigenvectors (their common eigenspace is
    // 2-D, so any orthonormal basis of it is a valid answer -- not necessarily e0/e1 themselves).
    [Test]
    public void MultiplicityBothConvergeToRepeatedEigenvalueOrthogonally()
    {
        var arena = new Arena(Allocator.Persistent);

        int n = 8, k = 2;
        var A = arena.floatMat(n, n);
        A[0, 0] = (float)1;
        A[1, 1] = (float)1;
        for (int i = 2; i < n; i++) A[i, i] = (float)i; // 2,3,4,5,6,7

        var eig = LOBPCG.lobpcg(ref arena, in A, k, out var vecs, out var info);
        Assert.IsTrue(info.Solved, info.ToString());

        Assert.AreEqual((float)1, eig[0], Tol());
        Assert.AreEqual((float)1, eig[1], Tol());

        float dot = (float)0;
        for (int c = 0; c < n; c++) dot += vecs[0, c] * vecs[1, c];
        Assert.AreEqual(0.0, (double)dot, 1e-2);

        arena.Dispose();
    }
}
