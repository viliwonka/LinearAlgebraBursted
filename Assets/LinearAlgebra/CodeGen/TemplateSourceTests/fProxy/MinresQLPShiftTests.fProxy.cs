using System;
using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

// Krylov.minresQLP shift: solving (A - shift*I) x = b via the eigenvalue-shifted overload. The
// shift is one extra axpy per iteration (Burst-folded when 0); the Lanczos recurrence is otherwise
// identical because the -shift*v term cancels against +shift in the diagonal. Three oracles:
//   (1) residual certificate   -- ‖(A - shift*I) x - b‖ <= 64*tol*‖b‖ (minresQLP's own bound);
//   (2) explicit-shift match    -- shift overload == unshifted solve of the explicitly formed A-σI;
//   (3) zero-shift bit-identity -- shift == 0 reproduces the plain overload exactly.
// Managed [Test]s (main thread, no Burst job) -- matches fProxyKrylovVerifyAtExitTests' style.
public class fProxyMinresQLPShiftTests
{
    // SPD via M^T M + n*I (same recipe as fProxyKrylovVerifyAtExitTests.BuildDenseSPD). With a
    // NEGATIVE shift, A - shift*I = A + |shift|*I stays SPD and well-conditioned.
    static fProxyMxN BuildDenseSPD(ref Arena arena, int dim, uint seed)
    {
        var M = arena.fProxyRandomMat(dim, dim, (fProxy)(-1f), (fProxy)1f, seed);
        var A = Blas.dot(M, M, true);
        for (int d = 0; d < dim; d++) A[d, d] += dim;
        return A;
    }

    // A - shift*I as a fresh dense matrix (A itself is left intact).
    static fProxyMxN ExplicitShift(ref Arena arena, in fProxyMxN A, fProxy shift)
    {
        var S = A.Copy();
        for (int d = 0; d < S.M_Rows; d++) S[d, d] -= shift;
        return S;
    }

    // ‖(A - shift*I) x - b‖^2, computed fresh through the operator (no explicit shifted matrix).
    static fProxy ShiftedResidualSq<TOp>(in TOp A, fProxy shift, in fProxyN b, in fProxyN x, ref fProxyN scratch)
        where TOp : struct, IfProxyLinearOperator
    {
        A.Apply(in x, ref scratch);                    // scratch = A x
        scratch.addScaledInPlace(-shift, x);           // scratch = (A - shift*I) x
        scratch.addScaledInPlace((fProxy)(-1), b);     // scratch = (A - shift*I) x - b
        return Blas.dot(scratch, scratch);
    }

    // (1) The shifted solve certifies its own residual.
    [Test]
    public void ShiftSolvesShiftedSystemDense()
    {
        var arena = new Arena(Allocator.Persistent);
        int n = 12;
        var A = BuildDenseSPD(ref arena, n, 0xB1A5u);
        var b = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 0x51u);
        var x = arena.fProxyVec(n);
        for (int i = 0; i < n; i++) x[i] = (fProxy)0;

        fProxy shift = (fProxy)(-0.5);
        fProxy tol = Consts.fProxySqrtEps;
        var info = Krylov.minresQLP(in A, in b, ref x, 4 * n, tol, shift);

        var op = new fProxyDenseOperator(in A);
        var scratch = arena.fProxyVec(n);
        fProxy rsq = ShiftedResidualSq(in op, shift, in b, in x, ref scratch);
        fProxy bb = Blas.dot(b, b);
        fProxy bound = (fProxy)64 * tol; bound = bound * bound * bb;

        Assert.AreEqual(IterativeSolveStatus.Converged, info.status, "shifted SPD system should converge");
        Assert.LessOrEqual((double)rsq, (double)bound * 4.0, "‖(A-σI)x - b‖ must meet the minresQLP residual bound");

        arena.Dispose();
    }

    // (2) The shift overload matches an unshifted solve of the explicitly formed A - shift*I.
    [Test]
    public void ShiftMatchesExplicitShiftDense()
    {
        var arena = new Arena(Allocator.Persistent);
        int n = 12;
        var A = BuildDenseSPD(ref arena, n, 0x2C0Du);
        var b = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 0x77u);
        fProxy shift = (fProxy)(-0.75);
        fProxy tol = Consts.fProxySqrtEps;

        var xShift = arena.fProxyVec(n);
        for (int i = 0; i < n; i++) xShift[i] = (fProxy)0;
        Krylov.minresQLP(in A, in b, ref xShift, 4 * n, tol, shift);

        var S = ExplicitShift(ref arena, in A, shift);
        var xExpl = arena.fProxyVec(n);
        for (int i = 0; i < n; i++) xExpl[i] = (fProxy)0;
        Krylov.minresQLP(in S, in b, ref xExpl, 4 * n, tol);

        // Both are well-conditioned SPD solves at the same tol -> agree closely. Compare relative
        // to ‖xExpl‖; float tol is sqrtEps ~ 3.4e-4 so a 1e-2 relative gate is comfortable and
        // still catches a mis-threaded shift (which diverges the two solutions entirely).
        fProxy num = (fProxy)0, den = (fProxy)0;
        for (int i = 0; i < n; i++)
        {
            fProxy d = xShift[i] - xExpl[i];
            num += d * d;
            den += xExpl[i] * xExpl[i];
        }
        double rel = math.sqrt((double)num / math.max((double)den, 1e-300));
        Assert.LessOrEqual(rel, 1e-2, "shift overload must match the explicit A-σI solve");

        arena.Dispose();
    }

    // (3) shift == 0 folds away: bit-identical to the plain overload.
    [Test]
    public void ZeroShiftMatchesUnshifted()
    {
        var arena = new Arena(Allocator.Persistent);
        int n = 10;
        var A = BuildDenseSPD(ref arena, n, 0x0FFu);
        var b = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 0x9Au);

        var xNo = arena.fProxyVec(n);
        var xZero = arena.fProxyVec(n);
        for (int i = 0; i < n; i++) { xNo[i] = (fProxy)0; xZero[i] = (fProxy)0; }

        Krylov.minresQLP(in A, in b, ref xNo, 4 * n, Consts.fProxySqrtEps);
        Krylov.minresQLP(in A, in b, ref xZero, 4 * n, Consts.fProxySqrtEps, (fProxy)0);

        for (int i = 0; i < n; i++)
            Assert.AreEqual((double)xNo[i], (double)xZero[i], "shift==0 must be bit-identical to the unshifted solve");

        arena.Dispose();
    }

    // (4) BSR path: the shifted overload over a block-sparse operator certifies its residual.
    [Test]
    public void ShiftSolvesShiftedSystemBSR()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyLaplacian2D(5, 5);   // SPD, 25x25
        int n = A.M_Rows;
        var b = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 0x1234u);
        var x = arena.fProxyVec(n);
        for (int i = 0; i < n; i++) x[i] = (fProxy)0;

        fProxy shift = (fProxy)(-1.0);   // Laplacian2D is SPD; -1 keeps A-σI SPD
        fProxy tol = Consts.fProxySqrtEps;
        var info = Krylov.minresQLP(in A, in b, ref x, 4 * n, tol, shift);

        var op = new fProxyBSROperator(in A);
        var scratch = arena.fProxyVec(n);
        fProxy rsq = ShiftedResidualSq(in op, shift, in b, in x, ref scratch);
        fProxy bb = Blas.dot(b, b);
        fProxy bound = (fProxy)64 * tol; bound = bound * bound * bb;

        Assert.AreEqual(IterativeSolveStatus.Converged, info.status, "shifted SPD BSR system should converge");
        Assert.LessOrEqual((double)rsq, (double)bound * 4.0, "BSR ‖(A-σI)x - b‖ must meet the residual bound");

        arena.Dispose();
    }

    // (5) Warm start (x0 != 0) exercises the initial-residual shift site r0 = b - (A-σI)x0. A wrong
    //     sign there leaves the solver minimizing the wrong system, so its FINAL residual would be
    //     ~2σ‖x0‖ (nonzero) instead of ~0. Checking the true shifted residual of the warm-started
    //     result catches that directly and avoids the O(tol·κ) noise of comparing two tol-converged
    //     solves (which is float-fragile). x0 is a nonzero guess.
    [Test]
    public void ShiftWarmStartCertifiesResidual()
    {
        var arena = new Arena(Allocator.Persistent);
        int n = 12;
        var A = BuildDenseSPD(ref arena, n, 0x5EEDu);
        var b = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 0x42u);
        fProxy shift = (fProxy)(-0.6);
        fProxy tol = Consts.fProxySqrtEps;

        var x = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 0xA11u);   // nonzero warm start
        var info = Krylov.minresQLP(in A, in b, ref x, 4 * n, tol, shift);

        var op = new fProxyDenseOperator(in A);
        var scratch = arena.fProxyVec(n);
        fProxy rsq = ShiftedResidualSq(in op, shift, in b, in x, ref scratch);
        fProxy bb = Blas.dot(b, b);
        fProxy bound = (fProxy)64 * tol; bound = bound * bound * bb;

        Assert.AreEqual(IterativeSolveStatus.Converged, info.status, "warm-started shifted solve should converge");
        Assert.LessOrEqual((double)rsq, (double)bound * 4.0, "warm-started ‖(A-σI)x - b‖ must be ~0 (wrong x0 sign would leave ~2σ‖x0‖)");

        arena.Dispose();
    }

    // (6) Shift on an INDEFINITE operator (the raison d'être of MINRES-QLP over CG). A positive
    //     shift straddling a symmetric spectrum makes A - shift*I indefinite. Validates the shift
    //     THREADING there via explicit-shift self-consistency: the shift-param solve and an unshifted
    //     solve of the explicitly formed A - shift*I run the identical algorithm on the same
    //     operator, so they must agree closely regardless of how well the (indefinite) system itself
    //     converges -- a wrong shift site/sign would diverge the two entirely. Diagonal A with mixed
    //     signs is a genuine symmetric indefinite operator.
    [Test]
    public void ShiftIndefiniteMatchesExplicitShift()
    {
        var arena = new Arena(Allocator.Persistent);
        int n = 8;
        // Symmetric indefinite: eigenvalues {-4,-3,-2,-1,1,2,3,4}; shift 0.5 keeps it indefinite.
        var d = new fProxy[] { (fProxy)(-4f), (fProxy)(-3f), (fProxy)(-2f), (fProxy)(-1f), (fProxy)1f, (fProxy)2f, (fProxy)3f, (fProxy)4f };
        fProxy shift = (fProxy)0.5;
        fProxy tol = Consts.fProxySqrtEps;

        var A = new fProxyMxN(n, n, Allocator.Persistent, true);   // cleared -> zeros off-diagonal
        for (int i = 0; i < n; i++) A[i, i] = d[i];
        var b = arena.fProxyRandomVec(n, (fProxy)(-1f), (fProxy)1f, 0x7u);

        var xShift = arena.fProxyVec(n);
        for (int i = 0; i < n; i++) xShift[i] = (fProxy)0;
        Krylov.minresQLP(in A, in b, ref xShift, 4 * n, tol, shift);

        // Explicit A - shift*I as a standalone copy (A is not arena-backed, so ExplicitShift's
        // arena Copy() can't be used here).
        var S = new fProxyMxN(in A, Allocator.Persistent);
        for (int i = 0; i < n; i++) S[i, i] -= shift;
        var xExpl = arena.fProxyVec(n);
        for (int i = 0; i < n; i++) xExpl[i] = (fProxy)0;
        Krylov.minresQLP(in S, in b, ref xExpl, 4 * n, tol);

        fProxy num = (fProxy)0, den = (fProxy)0;
        for (int i = 0; i < n; i++)
        {
            fProxy e = xShift[i] - xExpl[i];
            num += e * e; den += xExpl[i] * xExpl[i];
        }
        double rel = math.sqrt((double)num / math.max((double)den, 1e-300));
        Assert.LessOrEqual(rel, 1e-2, "shift-param solve must match the explicit A-σI solve on an indefinite operator");

        S.Dispose();
        A.Dispose();
        arena.Dispose();
    }
}
