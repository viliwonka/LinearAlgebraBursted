using System;
using LinearAlgebra;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

// Krylov.craig Tikhonov-damped path: x = Aᵀ(A Aᵀ + damp²·I)⁻¹ b (ridge least-norm), obtained by
// running UNDAMPED craig over the augmented operator [A | damp·I]. Oracle = dense Aᵀ(AAᵀ+damp²I)⁻¹b
// via Cholesky (same as the damped-CGNE test). Managed [Test]s.
public class fProxyCraigDampedTests
{
    static fProxyMxN BuildWide(int m, int n, uint seed)
    {
        var A = GenerateOP.fProxyRandomMat(m, n, (fProxy)(-1f), (fProxy)1f, seed, Allocator.Temp);
        for (int d = 0; d < m; d++) A[d, d] += (fProxy)10;   // full row rank, well-conditioned AAᵀ
        return A;
    }

    static void DampedOracle(in fProxyMxN A, in fProxyN b, fProxy damp, ref fProxyN xOut)
    {
        int m = A.M_Rows;
        var G = new fProxyMxN(m, m, Allocator.Temp, false);
        Blas.dot(in A, in A, ref G, false, true);          // G = A Aᵀ
        fProxy lam2 = damp * damp;
        for (int d = 0; d < m; d++) G[d, d] += lam2;        // + damp²·I
        var y = new fProxyN(m, Allocator.Temp); y.CopyFrom(in b);
        CHO.solveInPlace(ref G, ref y);                    // y = G⁻¹ b
        new fProxyDenseOperator(in A).ApplyT(in y, ref xOut); // x = Aᵀ y
    }

    [Test]
    public void DampedMatchesDenseOracle()
    {
        int m = 6, n = 10;
        var A = BuildWide(m, n, 0xC7A16u);
        var b = GenerateOP.fProxyRandomVec(m, (fProxy)(-2f), (fProxy)2f, 0x0Bu, Allocator.Temp);
        fProxy damp = (fProxy)0.5;

        var x = new fProxyN(n, Allocator.Temp);
        var info = Krylov.craig(in A, in b, ref x, 8 * (m + n), Consts.fProxySqrtEps, damp);

        var xOracle = new fProxyN(n, Allocator.Temp);
        DampedOracle(in A, in b, damp, ref xOracle);

        fProxy num = (fProxy)0, den = (fProxy)0;
        for (int i = 0; i < n; i++) { fProxy d = x[i] - xOracle[i]; num += d * d; den += xOracle[i] * xOracle[i]; }
        double rel = math.sqrt((double)num / math.max((double)den, 1e-300));
        Assert.IsTrue(info.status == IterativeSolveStatus.Converged, "damped craig should converge");
        Assert.LessOrEqual(rel, 1e-3, "damped craig must match Aᵀ(AAᵀ+damp²I)⁻¹b");
        // Arnorm = Tikhonov gradient ‖Aᵀr-damp²x‖ → 0 (the real cert); rnorm = ‖b-Ax‖ stays O(‖b‖).
        Assert.Less((double)info.Arnorm, 0.1 * (1.0 + (double)info.rnorm), "damped craig Arnorm must →0");
    }

    // Rank-deficient A: undamped craig would break down, but AAᵀ+damp²I is SPD -> well-posed.
    [Test]
    public void DampedHandlesRankDeficient()
    {
        int m = 6, n = 10;
        var A = BuildWide(m, n, 0x4A11u);
        for (int j = 0; j < n; j++) A[1, j] = A[0, j];   // row 1 := row 0 -> AAᵀ singular
        var b = GenerateOP.fProxyRandomVec(m, (fProxy)(-2f), (fProxy)2f, 0x55u, Allocator.Temp);
        fProxy damp = (fProxy)0.5;

        var x = new fProxyN(n, Allocator.Temp);
        var info = Krylov.craig(in A, in b, ref x, 8 * (m + n), Consts.fProxySqrtEps, damp);

        var xOracle = new fProxyN(n, Allocator.Temp);
        DampedOracle(in A, in b, damp, ref xOracle);
        fProxy num = (fProxy)0, den = (fProxy)0;
        for (int i = 0; i < n; i++) { fProxy d = x[i] - xOracle[i]; num += d * d; den += xOracle[i] * xOracle[i]; }
        double rel = math.sqrt((double)num / math.max((double)den, 1e-300));
        Assert.IsTrue(info.status == IterativeSolveStatus.Converged, "damped craig must converge on rank-deficient A");
        Assert.LessOrEqual(rel, 1e-3, "damped craig on rank-deficient A must match the dense oracle");
    }

    // damp == 0 delegates to plain craig: bit-identical.
    [Test]
    public void ZeroDampMatchesUndamped()
    {
        int m = 6, n = 10;
        var A = BuildWide(m, n, 0x1C0u);
        var b = GenerateOP.fProxyRandomVec(m, (fProxy)(-2f), (fProxy)2f, 0x33u, Allocator.Temp);

        var xUn = new fProxyN(n, Allocator.Temp);
        var xZero = new fProxyN(n, Allocator.Temp);
        Krylov.craig(in A, in b, ref xUn, 4 * (m + n), Consts.fProxySqrtEps);
        Krylov.craig(in A, in b, ref xZero, 4 * (m + n), Consts.fProxySqrtEps, (fProxy)0);

        for (int i = 0; i < n; i++)
            Assert.AreEqual((double)xUn[i], (double)xZero[i], "damp==0 must be bit-identical to undamped craig");
    }

    // craigmr damped: same augmented-operator route, monotonic-residual variant.
    [Test]
    public void CraigmrDampedMatchesDenseOracle()
    {
        int m = 6, n = 10;
        var A = BuildWide(m, n, 0xC7A16u);
        var b = GenerateOP.fProxyRandomVec(m, (fProxy)(-2f), (fProxy)2f, 0x0Bu, Allocator.Temp);
        fProxy damp = (fProxy)0.5;

        var x = new fProxyN(n, Allocator.Temp);
        var info = Krylov.craigmr(in A, in b, ref x, 8 * (m + n), Consts.fProxySqrtEps, damp);

        var xOracle = new fProxyN(n, Allocator.Temp);
        DampedOracle(in A, in b, damp, ref xOracle);
        fProxy num = (fProxy)0, den = (fProxy)0;
        for (int i = 0; i < n; i++) { fProxy d = x[i] - xOracle[i]; num += d * d; den += xOracle[i] * xOracle[i]; }
        double rel = math.sqrt((double)num / math.max((double)den, 1e-300));
        Assert.IsTrue(info.status == IterativeSolveStatus.Converged, "damped craigmr should converge");
        Assert.LessOrEqual(rel, 1e-3, "damped craigmr must match Aᵀ(AAᵀ+damp²I)⁻¹b");
        Assert.Less((double)info.Arnorm, 0.1 * (1.0 + (double)info.rnorm), "damped craigmr Arnorm must →0");
    }

    [Test]
    public void CraigmrZeroDampMatchesUndamped()
    {
        int m = 6, n = 10;
        var A = BuildWide(m, n, 0x1C0u);
        var b = GenerateOP.fProxyRandomVec(m, (fProxy)(-2f), (fProxy)2f, 0x33u, Allocator.Temp);

        var xUn = new fProxyN(n, Allocator.Temp);
        var xZero = new fProxyN(n, Allocator.Temp);
        Krylov.craigmr(in A, in b, ref xUn, 4 * (m + n), Consts.fProxySqrtEps);
        Krylov.craigmr(in A, in b, ref xZero, 4 * (m + n), Consts.fProxySqrtEps, (fProxy)0);

        for (int i = 0; i < n; i++)
            Assert.AreEqual((double)xUn[i], (double)xZero[i], "damp==0 must be bit-identical to undamped craigmr");
    }
}
