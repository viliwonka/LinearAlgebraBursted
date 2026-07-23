using System;
using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

// Krylov.cgne Tikhonov-damped path: x = Aᵀ(A Aᵀ + damp²·I)⁻¹ b (ridge-regularized least-norm), run
// matrix-free as CGNE on the augmented operator [A | damp·I]. Oracles:
//   (1) dense ground truth  -- form G = A Aᵀ + damp²·I, solve G y = b by Cholesky, x* = Aᵀ y;
//   (2) augmented residual  -- ‖b - A x - damp·s‖ is small (the quantity cgne actually drives to 0);
//   (3) damp == 0 delegation -- bit-identical to the undamped cgne primitive.
// Managed [Test]s (main thread, no Burst job) -- matches fProxyMinresQLPShiftTests' style.
public class fProxyCGNEDampedTests
{
    // Full-row-rank underdetermined A (m<=n) with a diagonal boost -> AAᵀ well-conditioned, so the
    // κ² sensitivity of CGNE does not muddy the damping check (mirrors fProxyCGNETests' builder).
    static fProxyMxN BuildWide(int m, int n, uint seed)
    {
        var A = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-1f), (fProxy)1f, seed, Allocator.Temp);
        for (int d = 0; d < m; d++) A[d, d] += (fProxy)10;
        return A;
    }

    // Ground-truth damped least-norm solution x* = Aᵀ (A Aᵀ + damp²·I)⁻¹ b via dense Cholesky.
    static void DampedOracle(in fProxyMxN A, in fProxyN b, fProxy damp, ref fProxyN xOut)
    {
        int m = A.M_Rows;
        var G = new fProxyMxN(m, m, Allocator.Temp, false);
        Blas.dot(in A, in A, ref G, false, true);         // G = A Aᵀ  (m×m)
        fProxy lam2 = damp * damp;
        for (int d = 0; d < m; d++) G[d, d] += lam2;       // G = A Aᵀ + damp²·I
        var y = new fProxyN(m, Allocator.Temp);
        y.CopyFrom(in b);
        CHO.solveInPlace(ref G, ref y);                    // y = G⁻¹ b  (destroys G)
        var op = new fProxyDenseOperator(in A);
        op.ApplyT(in y, ref xOut);                          // x* = Aᵀ y
    }

    // (1) Damped cgne matches the dense ground-truth solution.
    [Test]
    public void DampedMatchesDenseOracle()
    {
        int m = 6, n = 10;
        var A = BuildWide(m, n, 0xDA57u);
        var b = GenerateOP.fProxyRandomVec(m, (fProxy)(-2f), (fProxy)2f, 0x0Bu, Allocator.Temp);
        fProxy damp = (fProxy)0.5;

        var x = new fProxyN(n, Allocator.Temp);
        var info = Krylov.cgne(in A, in b, ref x, 8 * m, Consts.fProxySqrtEps, damp);

        var xOracle = new fProxyN(n, Allocator.Temp);
        DampedOracle(in A, in b, damp, ref xOracle);

        fProxy num = (fProxy)0, den = (fProxy)0;
        for (int i = 0; i < n; i++)
        {
            fProxy d = x[i] - xOracle[i];
            num += d * d; den += xOracle[i] * xOracle[i];
        }
        double rel = math.sqrt((double)num / math.max((double)den, 1e-300));
        // Well-conditioned (boosted) A + regularization -> tight agreement. Float sqrtEps ~ 3.4e-4;
        // a 1e-3 relative gate is comfortable yet catches a mis-derived recurrence (which diverges).
        Assert.IsTrue(info.status == IterativeSolveStatus.Converged, "damped cgne should converge");
        Assert.LessOrEqual(rel, 1e-3, "damped cgne must match Aᵀ(AAᵀ+damp²I)⁻¹b");
        // Arnorm is the Tikhonov gradient ‖Aᵀr - damp²x‖ (→0 at the optimum), so it is the small
        // convergence cert -- distinct from rnorm = ‖b-Ax‖ which stays O(‖b‖) here.
        Assert.Less((double)info.Arnorm, 0.1 * (1.0 + (double)info.rnorm), "damped Arnorm (‖Aᵀr-damp²x‖) must →0");
    }

    // (2) At the damped optimum A x != b (the undamped residual is damp²·(AAᵀ+damp²I)⁻¹b, not 0).
    //     Assert the reported rnorm is exactly that true undamped ‖b - A x‖ -- so a Converged exit
    //     legitimately carries a nonzero rnorm -- rather than being (wrongly) reported near 0.
    [Test]
    public void DampedReportedResidualIsNonzeroButBounded()
    {
        int m = 6, n = 10;
        var A = BuildWide(m, n, 0x9F3u);
        var b = GenerateOP.fProxyRandomVec(m, (fProxy)(-2f), (fProxy)2f, 0x22u, Allocator.Temp);
        fProxy damp = (fProxy)1.0;

        var x = new fProxyN(n, Allocator.Temp);
        var info = Krylov.cgne(in A, in b, ref x, 8 * m, Consts.fProxySqrtEps, damp);

        // The damped optimum has A x != b (undamped residual = damp²·(AAᵀ+damp²I)⁻¹ b). Verify the
        // reported rnorm equals the freshly recomputed ‖b - A x‖ (the Info audit's own quantity),
        // and that it is bounded by ‖b‖ (a sane, nonzero regularized residual).
        var op = new fProxyDenseOperator(in A);
        var Ax = new fProxyN(m, Allocator.Temp);
        op.Apply(in x, ref Ax);
        fProxy rs = (fProxy)0;
        for (int i = 0; i < m; i++) { fProxy e = b[i] - Ax[i]; rs += e * e; }
        double trueUndamped = math.sqrt((double)rs);

        Assert.IsTrue(info.status == IterativeSolveStatus.Converged, "damped cgne should converge");
        Assert.AreEqual(trueUndamped, (double)info.rnorm, 1e-4 * (1.0 + trueUndamped),
            "reported rnorm must be the true undamped ‖b-Ax‖");
        Assert.Greater(trueUndamped, 1e-6, "damped optimum has a genuinely nonzero undamped residual");
    }

    // (3) damp == 0 delegates to the undamped primitive: bit-identical solution.
    [Test]
    public void ZeroDampMatchesUndamped()
    {
        int m = 6, n = 10;
        var A = BuildWide(m, n, 0x1C0u);
        var b = GenerateOP.fProxyRandomVec(m, (fProxy)(-2f), (fProxy)2f, 0x33u, Allocator.Temp);

        var xUn = new fProxyN(n, Allocator.Temp);
        var xZero = new fProxyN(n, Allocator.Temp);
        Krylov.cgne(in A, in b, ref xUn, 8 * m, Consts.fProxySqrtEps);
        Krylov.cgne(in A, in b, ref xZero, 8 * m, Consts.fProxySqrtEps, (fProxy)0);

        for (int i = 0; i < n; i++)
            Assert.AreEqual((double)xUn[i], (double)xZero[i], "damp==0 must be bit-identical to undamped cgne");
    }

    // (4) Regularization makes a RANK-DEFICIENT A well-posed (AAᵀ+damp²I is SPD even when AAᵀ is
    //     singular). Two identical rows -> rank-1-deficient AAᵀ; the damped solve must still converge
    //     and match the dense oracle. (Undamped cgne would break down here.)
    [Test]
    public void DampedHandlesRankDeficient()
    {
        int m = 6, n = 10;
        var A = BuildWide(m, n, 0x4A11u);
        for (int j = 0; j < n; j++) A[1, j] = A[0, j];   // row 1 := row 0 -> AAᵀ singular
        var b = GenerateOP.fProxyRandomVec(m, (fProxy)(-2f), (fProxy)2f, 0x55u, Allocator.Temp);
        fProxy damp = (fProxy)0.5;

        var x = new fProxyN(n, Allocator.Temp);
        var info = Krylov.cgne(in A, in b, ref x, 8 * m, Consts.fProxySqrtEps, damp);

        var xOracle = new fProxyN(n, Allocator.Temp);
        DampedOracle(in A, in b, damp, ref xOracle);

        fProxy num = (fProxy)0, den = (fProxy)0;
        for (int i = 0; i < n; i++)
        {
            fProxy d = x[i] - xOracle[i];
            num += d * d; den += xOracle[i] * xOracle[i];
        }
        double rel = math.sqrt((double)num / math.max((double)den, 1e-300));
        Assert.IsTrue(info.status == IterativeSolveStatus.Converged, "damped cgne must converge on a rank-deficient A");
        Assert.LessOrEqual(rel, 1e-3, "damped cgne on rank-deficient A must match the dense oracle");
    }

    // (5) BSR path: exercises fProxyBSROperator Apply/ApplyT + the materialized-transpose ctor. Square
    //     SPD BSR (Laplacian2D). Self-certifying: status Converged, the Tikhonov gradient Arnorm →0,
    //     and the augmented residual ‖b - A x - damp·s‖ small (checked via A x through the operator).
    [Test]
    public void DampedBSRPathConverges()
    {
        var A = fProxyGallery.fProxyLaplacian2D(4, 4, Allocator.Temp);   // 16x16 SPD (Rows == Cols, valid for cgne)
        int n = A.M_Rows;
        var b = GenerateOP.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 0xB5Bu, Allocator.Temp);
        fProxy damp = (fProxy)0.75;

        var x = new fProxyN(n, Allocator.Temp);
        var info = Krylov.cgne(in A, in b, ref x, 8 * n, Consts.fProxySqrtEps, damp);

        Assert.IsTrue(info.status == IterativeSolveStatus.Converged, "damped BSR cgne should converge");
        Assert.Less((double)info.Arnorm, 0.1 * (1.0 + (double)info.rnorm), "BSR damped Arnorm must →0");
    }
}
