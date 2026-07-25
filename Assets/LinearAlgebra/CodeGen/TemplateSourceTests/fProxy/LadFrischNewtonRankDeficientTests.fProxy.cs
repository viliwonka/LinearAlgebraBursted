using LinearAlgebra;
using NUnit.Framework;
using Unity.Collections;

// Rank-deficient and ill-conditioned designs routed through the Frisch-Newton LAD solver. The
// solver seeds itself from an ordinary least-squares fit computed with plain CHO on A^T A; on a
// rank-deficient design that factorization has no unique solution. The reference implementation
// aborts with no fit in that case, this port falls back to y = 0 (a valid strictly-interior start)
// and proceeds -- so the contract under test is that such designs still produce the correct optimum.
public class fProxyLadFrischNewtonRankDeficientTests
{
    // Column 2 duplicates column 1, so A^T A is singular. The coefficients are NOT identifiable --
    // any split of the slope between the two duplicated columns is equally optimal -- but the optimal
    // OBJECTIVE is, and it is the same as the full-rank two-column problem: the outlier line
    // b = t with b[2] = 10 has L1 residual 8.
    [Test]
    public void DuplicateColumnStillReachesTheOptimum()
    {
        int m = 5;
        var A = new fProxyMxN(m, 3, Allocator.Temp);
        var b = new fProxyN(m, Allocator.Temp);
        for (int i = 0; i < m; i++)
        {
            A[i, 0] = (fProxy)1; A[i, 1] = (fProxy)i; A[i, 2] = (fProxy)i;
            b[i] = (fProxy)i;
        }
        b[2] = (fProxy)10;
        var x = new fProxyN(3, Allocator.Temp);

        LP.ladFN(in A, in b, ref x, out double obj);

        for (int j = 0; j < 3; j++)
        {
            double v = (double)x[j];
            Assert.That(double.IsNaN(v) || double.IsInfinity(v), Is.False, $"x[{j}] must be finite, was {v}");
        }
        Assert.That(obj, Is.EqualTo(8.0).Within(/*+choose[1e-1|1e-4]*/1e-1/*-choose*/),
            "L1 residual on a rank-deficient design must match the full-rank equivalent");

        A.Dispose(); b.Dispose(); x.Dispose();
    }

    // A design with an all-zero column: A^T A has a zero row and column. Same contract -- the zero
    // column carries no information, so the optimum is the two-column problem's.
    [Test]
    public void ZeroColumnStillReachesTheOptimum()
    {
        int m = 5;
        var A = new fProxyMxN(m, 3, Allocator.Temp);
        var b = new fProxyN(m, Allocator.Temp);
        for (int i = 0; i < m; i++)
        {
            A[i, 0] = (fProxy)1; A[i, 1] = (fProxy)i; A[i, 2] = (fProxy)0;
            b[i] = (fProxy)i;
        }
        b[2] = (fProxy)10;
        var x = new fProxyN(3, Allocator.Temp);

        LP.ladFN(in A, in b, ref x, out double obj);

        for (int j = 0; j < 3; j++)
        {
            double v = (double)x[j];
            Assert.That(double.IsNaN(v) || double.IsInfinity(v), Is.False, $"x[{j}] must be finite, was {v}");
        }
        Assert.That(obj, Is.EqualTo(8.0).Within(/*+choose[1e-1|1e-4]*/1e-1/*-choose*/),
            "L1 residual with a zero column must match the full-rank equivalent");

        A.Dispose(); b.Dispose(); x.Dispose();
    }

    // Degree-4 Vandermonde: classically ill-conditioned, so the Newton normal matrix is where the
    // regularization, the Jacobi equilibration and CHOP's rank truncation all perturb the arithmetic.
    // That perturbation is exactly what the affine RHS's primal-residual term (bLP - A^T a) corrects
    // -- it is zero in exact arithmetic, so it only earns its keep here. Oracle is ladBR, an
    // independent exact-vertex engine. Measured float relative error on THIS instance: 8.2e-5 with
    // the term, 4.6e-3 without (56x worse). The tolerance sits between the two, so it is a real
    // regression guard, not a formality. Double is unaffected (<= 1e-10 either way).
    [Test]
    public void IllConditionedVandermondeMatchesExactEngine()
    {
        int m = 40, n = 5;
        var A = new fProxyMxN(m, n, Allocator.Temp);
        var b = new fProxyN(m, Allocator.Temp);
        var rng = new Unity.Mathematics.Random(12345u);
        for (int i = 0; i < m; i++)
        {
            double t = 0.1 + 1.9 * i / (m - 1.0);
            double p = 1;
            for (int j = 0; j < n; j++) { A[i, j] = (fProxy)p; p *= t; }
            b[i] = (fProxy)(1.0 + 0.5 * t + rng.NextFProxy(-0.2f, 0.2f));
        }
        b[7] = (fProxy)(b[7] + 20f);   // one gross outlier, so the L1 and L2 fits differ

        var xFN = new fProxyN(n, Allocator.Temp);
        var xBR = new fProxyN(n, Allocator.Temp);
        LP.ladFN(in A, in b, ref xFN, out double objFN);
        LP.ladBR(in A, in b, ref xBR, out double objBR);

        double rel = (objFN - objBR) / objBR;
        Assert.That(rel, Is.LessThan(/*+choose[3e-4|1e-9]*/3e-4/*-choose*/),
            $"FN L1 residual {objFN} exceeds the exact ladBR optimum {objBR} by {rel} relative");

        A.Dispose(); b.Dispose(); xFN.Dispose(); xBR.Dispose();
    }

    // Near-collinear columns are FN's documented weak spot in float: the normal equations square the
    // condition number, and two columns separated by ~1e-6 relative put cond(A^T Q A) past what float
    // can resolve. This is a REGRESSION GUARD on a known limit, not an endorsement of the accuracy --
    // the measured gap is ~8.7e-3 relative in float (double reaches ~1e-10). Prefer ladBR here; lad's
    // own routing already does below the crossover. If this test starts failing, FN got WORSE.
    [Test]
    public void NearCollinearColumnsStayWithinKnownBound()
    {
        int m = 40, n = 3;
        var A = new fProxyMxN(m, n, Allocator.Temp);
        var b = new fProxyN(m, Allocator.Temp);
        var rng = new Unity.Mathematics.Random(4242u);
        for (int i = 0; i < m; i++)
        {
            fProxy t = rng.NextFProxy(0f, 10f);
            A[i, 0] = (fProxy)1; A[i, 1] = t;
            A[i, 2] = (fProxy)(t + 1e-5f * rng.NextFProxy(-1f, 1f));   // ~1e-6 relative separation
            b[i] = (fProxy)(3f + 2f * t + rng.NextFProxy(-1f, 1f));
        }
        b[7] = (fProxy)(b[7] + 50f);

        var xFN = new fProxyN(n, Allocator.Temp);
        var xBR = new fProxyN(n, Allocator.Temp);
        LP.ladFN(in A, in b, ref xFN, out double objFN);
        LP.ladBR(in A, in b, ref xBR, out double objBR);

        double rel = (objFN - objBR) / objBR;
        Assert.That(rel, Is.LessThan(/*+choose[2e-2|1e-8]*/2e-2/*-choose*/),
            $"FN L1 residual {objFN} exceeds the exact ladBR optimum {objBR} by {rel} relative -- " +
            "worse than the known near-collinear bound");

        A.Dispose(); b.Dispose(); xFN.Dispose(); xBR.Dispose();
    }
}
