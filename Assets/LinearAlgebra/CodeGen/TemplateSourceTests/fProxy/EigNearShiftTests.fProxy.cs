using System;
using LinearAlgebra;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

// Eigen.eigNearShift: shift-and-invert interior eigensolver. Finds the k eigenpairs of a symmetric A
// nearest a target shift via Lanczos over (A - shift*I)⁻¹ (inner minres-qlp solve) + Rayleigh-quotient
// recovery, selecting the modes with largest |Ritz value of the shift-invert operator| (= nearest +
// best-converged). Oracles: a diagonal matrix and a 1D-Laplacian tridiagonal -- both have exact known
// spectra (so "the returned k are the k nearest" is checked as a SET), the Laplacian additionally has
// non-trivial (sine) eigenvectors. Plus an eigenpair-residual check ‖A v - λ v‖. Managed [Test]s.
public class fProxyEigNearShiftTests
{
    static double RelResidual(in fProxyMxN A, in fProxyMxN vectors, int row, fProxy lam)
    {
        int n = A.M_Rows;
        var v = new fProxyN(n, Allocator.Temp);
        for (int i = 0; i < n; i++) v[i] = vectors[row, i];
        var Av = new fProxyN(n, Allocator.Temp);
        new fProxyDenseOperator(in A).Apply(in v, ref Av);
        double num = 0, den = 0;
        for (int i = 0; i < n; i++) { double e = (double)Av[i] - (double)lam * (double)v[i]; num += e * e; den += (double)v[i] * (double)v[i]; }
        return math.sqrt(num / math.max(den, 1e-300));
    }

    // The k values of `spectrum` nearest `shift`, returned SORTED BY VALUE (for a set comparison that
    // is robust to nearness-order ties -- e.g. a spectrum symmetric about shift).
    static double[] KNearestSortedByValue(double[] spectrum, double shift, int k)
    {
        var idx = new int[spectrum.Length];
        for (int i = 0; i < idx.Length; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => math.abs(spectrum[a] - shift).CompareTo(math.abs(spectrum[b] - shift)));
        var outv = new double[k];
        for (int i = 0; i < k; i++) outv[i] = spectrum[idx[i]];
        Array.Sort(outv);
        return outv;
    }

    // Assert the returned eigenvalues, as a SET, equal the k nearest shift.
    static void AssertNearestSet(double[] spectrum, in fProxyN vals, double shift, int k, double tol)
    {
        Assert.AreEqual(k, vals.N, "should return k eigenvalues");
        var want = KNearestSortedByValue(spectrum, shift, k);
        var got = new double[k];
        for (int i = 0; i < k; i++) got[i] = (double)vals[i];
        Array.Sort(got);
        for (int i = 0; i < k; i++)
            Assert.AreEqual(want[i], got[i], tol, $"returned eigenvalue set (sorted) entry {i}");
    }

    // (1) Diagonal A: eigenvalues are the diagonal entries.
    [Test]
    public void EigNearShiftDiagonalExactNearest()
    {
        int n = 20;
        var A = new fProxyMxN(n, n, Allocator.Temp);
        var spectrum = new double[n];
        for (int i = 0; i < n; i++) { A[i, i] = (fProxy)i; spectrum[i] = i; }
        fProxy shift = (fProxy)8.3;
        int k = 3;

        Eigen.eigNearShift(in A, shift, k, out var vals, out var vecs);

        AssertNearestSet(spectrum, in vals, (double)shift, k, 1e-2);
        for (int s = 0; s < k; s++)
            Assert.LessOrEqual(RelResidual(in A, in vecs, s, vals[s]), 3e-2, $"residual of pair #{s}");
    }

    // (2) 1D Laplacian tridiagonal (2 on diag, -1 off): non-trivial (sine) eigenvectors, exact known
    //     spectrum λ_k = 2 - 2cos(kπ/(n+1)) (symmetric about 2 -> the two nearest shift=2 are an exact
    //     tie, which the set comparison handles); non-trivial inner solve.
    [Test]
    public void EigNearShiftLaplacian1DExactNearest()
    {
        int n = 24;
        var A = new fProxyMxN(n, n, Allocator.Temp);
        for (int i = 0; i < n; i++)
        {
            A[i, i] = (fProxy)2;
            if (i > 0) { A[i, i - 1] = (fProxy)(-1); A[i - 1, i] = (fProxy)(-1); }
        }
        var spectrum = new double[n];
        for (int kk = 1; kk <= n; kk++) spectrum[kk - 1] = 2.0 - 2.0 * math.cos(kk * math.PI_DBL / (n + 1));

        fProxy shift = (fProxy)2.0;
        int k = 2;

        Eigen.eigNearShift(in A, shift, k, out var vals, out var vecs);

        AssertNearestSet(spectrum, in vals, (double)shift, k, 1e-2);
        for (int s = 0; s < k; s++)
            Assert.LessOrEqual(RelResidual(in A, in vecs, s, vals[s]), 3e-2, $"residual of pair #{s}");
    }
}
