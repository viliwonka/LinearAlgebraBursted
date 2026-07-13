using LinearAlgebra;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

// Acceptance tests for the tau-sanity behavior of the Frisch-Newton LAD/quantile solver. These call
// the INTERNAL entry point LP.ladFrischNewtonCore(A, b, tau, ...) directly, because tau != 0.5 has no
// public surface yet (a public Stats/ML quantile API comes later). Reached via the InternalsVisibleTo
// grants on both BurstLinearAlgebra.Tests and BurstLinearAlgebra.TemplateSource.Tests-firstpass
// (TemplateSource/AssemblyInfo.cs).
//
// Run on the managed test thread (like LPTests.fProxy.cs's SimplexAndInteriorPointAgree) so NUnit
// asserts are legal and a divergence surfaces with a clear message; the core itself is the same code
// the Burst jobs exercise.
public class fProxyLadFrischNewtonQuantileTests
{
    // ---- LP.ladFN is exactly ladFrischNewtonCore at tau=0.5, and both recover the LAD line.
    // Data: 4 collinear points b=t at t=0..3 plus one gross outlier (t=4 kept, b[2]=10) -- the same
    // BuildLine(outlier:true) set LPTests.fProxy.cs's LadOutlier/DualLad use; LAD line is b=t => (0,1),
    // L1 residual |10-2| = 8.

    [Test]
    public void TauHalfMatchesCore()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyMat(5, 2);
        var b = arena.fProxyVec(5);
        for (int i = 0; i < 5; i++) { A[i, 0] = 1f; A[i, 1] = i; b[i] = i; }
        b[2] = 10f;
        var xFN = arena.fProxyVec(2);
        var xCore = arena.fProxyVec(2);

        LP.ladFN(in A, in b, ref xFN, out double objFN);
        LP.ladFrischNewtonCore(in A, in b, 0.5, ref xCore, out double objCore, 0);

        // ladFN forwards to the core at tau=0.5: identical computation, identical result.
        Assert.That(objFN, Is.EqualTo(objCore).Within(/*+choose[1e-5|1e-9]*/1e-5/*-choose*/), "ladFN vs core objective");
        Assert.That((double)xFN[0], Is.EqualTo((double)xCore[0]).Within(/*+choose[1e-4|1e-9]*/1e-4/*-choose*/));
        Assert.That((double)xFN[1], Is.EqualTo((double)xCore[1]).Within(/*+choose[1e-4|1e-9]*/1e-4/*-choose*/));
        // ...and it lands on the robust LAD line (matching LPTests.fProxy.cs's IpLadOutlier interior-point tolerance).
        Assert.That((double)xFN[0], Is.EqualTo(0.0).Within(/*+choose[5e-2|1e-2]*/5e-2/*-choose*/), "intercept");
        Assert.That((double)xFN[1], Is.EqualTo(1.0).Within(/*+choose[5e-2|1e-2]*/5e-2/*-choose*/), "slope");
        Assert.That(objFN, Is.EqualTo(8.0).Within(/*+choose[1e-1|1e-2]*/1e-1/*-choose*/), "L1 residual");

        arena.Dispose();
    }

    // ---- Quantile-regression semantics at tau=0.25. At the fitted tau line, the fraction of NEGATIVE
    // residuals (b_i strictly below the fit) is ~tau. Synthetic line b = 1 + 2t with SYMMETRIC noise and
    // NO gross outliers (outliers would distort the residual-sign fraction). This is a
    // statistical/finite-sample property (near-exact at the LP optimum up to the ~n interpolated points,
    // but NOT an identity), so the slack is deliberately generous (+/-20% of m; target 0.25*m = 20, so
    // the count may land anywhere in [4, 36]) to avoid flaking.

    [Test]
    public void TauQuarterResidualSign()
    {
        var arena = new Arena(Allocator.Persistent);
        int m = 80;
        var A = arena.fProxyMat(m, 2);
        var b = arena.fProxyVec(m);
        var rng = new Unity.Mathematics.Random(20260709u);
        for (int i = 0; i < m; i++)
        {
            fProxy t = rng.NextFProxy(0f, 10f);
            A[i, 0] = 1f; A[i, 1] = t;
            b[i] = 1f + 2f * t + rng.NextFProxy(-3f, 3f);   // symmetric noise about the true line
        }
        var x = arena.fProxyVec(2);
        LP.ladFrischNewtonCore(in A, in b, 0.25, ref x, out double obj, 0);

        int neg = 0;
        for (int i = 0; i < m; i++)
        {
            double fit = (double)x[0] + (double)x[1] * (double)A[i, 1];
            if ((double)b[i] - fit < 0.0) neg++;
        }
        double target = 0.25 * m;   // 20
        Assert.That(neg, Is.EqualTo(target).Within(0.20 * m),
            $"tau=0.25: {neg}/{m} residuals negative, expected ~{target} (+/-{0.20 * m})");

        arena.Dispose();
    }
}
