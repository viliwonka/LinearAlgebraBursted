using LinearAlgebra;
using NUnit.Framework;
using Unity.Collections;

// Rank-deficient designs routed through the Frisch-Newton LAD solver. The solver seeds itself from
// an ordinary least-squares fit computed with plain CHO on A^T A; on a rank-deficient design that
// factorization has no unique solution. The reference implementation aborts with no fit in that
// case, this port falls back to y = 0 (a valid strictly-interior start) and proceeds -- so the
// contract under test is that a rank-deficient design still produces the correct optimum.
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
}
