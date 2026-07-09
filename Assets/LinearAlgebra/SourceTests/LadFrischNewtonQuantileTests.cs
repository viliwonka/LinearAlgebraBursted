using LinearAlgebra;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

// Concrete (NOT codegen'd) tests for the tau-sanity items of the Frisch-Newton LAD/quantile solver
// (docs/spec-lad-frisch-newton.md, Tests item 3). These call the INTERNAL entry point
// LP.ladFrischNewtonCore(A, b, tau, ...) directly, because tau != 0.5 has no public surface yet (a
// public Stats/ML quantile API "comes later", per the spec). InternalsVisibleTo only reaches this
// GENERATED test assembly ("BurstLinearAlgebra.Tests", granted in TemplateSource/AssemblyInfo.cs) --
// NOT the template-source "-firstpass" compile-check assembly -- so, exactly like
// ChunkedRecordTableTests.cs / the QRCPDowndateTests internal-access note, these live here as
// hand-written tests rather than in the fProxy template (which must compile in BOTH assemblies).
//
// Both dtypes are covered explicitly (float + double), mirroring the template's per-type expansion.
// Run on the managed test thread (like LPTests' SimplexAndInteriorPointAgree) so NUnit asserts are
// legal and a divergence surfaces with a clear message; the core itself is the same code the Burst
// jobs exercise.
public class LadFrischNewtonQuantileTests
{
    // ---- Item 3a: LP.ladFN is exactly ladFrischNewtonCore at tau=0.5, and both recover the LAD line.
    // Data: 4 collinear points b=t at t=0..3 plus one gross outlier (t=4 kept, b[2]=10) -- the same
    // BuildLine(outlier:true) set the template's LadOutlier/DualLad use; LAD line is b=t => (0,1),
    // L1 residual |10-2| = 8.

    [Test]
    public void TauHalfMatchesCore_Float()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.floatMat(5, 2);
        var b = arena.floatVec(5);
        for (int i = 0; i < 5; i++) { A[i, 0] = 1f; A[i, 1] = i; b[i] = i; }
        b[2] = 10f;
        var xFN = arena.floatVec(2);
        var xCore = arena.floatVec(2);

        LP.ladFN(in A, in b, ref xFN, out double objFN);
        LP.ladFrischNewtonCore(in A, in b, 0.5, ref xCore, out double objCore, 0);

        // ladFN forwards to the core at tau=0.5: identical computation, identical result.
        Assert.That(objFN, Is.EqualTo(objCore).Within(1e-5), "ladFN vs core objective");
        Assert.That((double)xFN[0], Is.EqualTo((double)xCore[0]).Within(1e-4));
        Assert.That((double)xFN[1], Is.EqualTo((double)xCore[1]).Within(1e-4));
        // ...and it lands on the robust LAD line (interior-point tolerance, cf. IpLadOutlier's 5e-2).
        Assert.That((double)xFN[0], Is.EqualTo(0.0).Within(5e-2), "intercept");
        Assert.That((double)xFN[1], Is.EqualTo(1.0).Within(5e-2), "slope");
        Assert.That(objFN, Is.EqualTo(8.0).Within(1e-1), "L1 residual");

        arena.Dispose();
    }

    [Test]
    public void TauHalfMatchesCore_Double()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.doubleMat(5, 2);
        var b = arena.doubleVec(5);
        for (int i = 0; i < 5; i++) { A[i, 0] = 1.0; A[i, 1] = i; b[i] = i; }
        b[2] = 10.0;
        var xFN = arena.doubleVec(2);
        var xCore = arena.doubleVec(2);

        LP.ladFN(in A, in b, ref xFN, out double objFN);
        LP.ladFrischNewtonCore(in A, in b, 0.5, ref xCore, out double objCore, 0);

        Assert.That(objFN, Is.EqualTo(objCore).Within(1e-9), "ladFN vs core objective");
        Assert.That(xFN[0], Is.EqualTo(xCore[0]).Within(1e-9));
        Assert.That(xFN[1], Is.EqualTo(xCore[1]).Within(1e-9));
        Assert.That(xFN[0], Is.EqualTo(0.0).Within(1e-2), "intercept");
        Assert.That(xFN[1], Is.EqualTo(1.0).Within(1e-2), "slope");
        Assert.That(objFN, Is.EqualTo(8.0).Within(1e-2), "L1 residual");

        arena.Dispose();
    }

    // ---- Item 3b: quantile-regression semantics at tau=0.25. At the fitted tau line, the fraction of
    // NEGATIVE residuals (b_i strictly below the fit) is ~tau. Synthetic line b = 1 + 2t with SYMMETRIC
    // noise and NO gross outliers (outliers would distort the residual-sign fraction). This is a
    // statistical/finite-sample property (near-exact at the LP optimum up to the ~n interpolated points,
    // but NOT an identity), so the slack is deliberately generous (+/-20% of m; target 0.25*m = 20, so
    // the count may land anywhere in [4, 36]) to avoid flaking.

    [Test]
    public void TauQuarterResidualSign_Float()
    {
        var arena = new Arena(Allocator.Persistent);
        int m = 80;
        var A = arena.floatMat(m, 2);
        var b = arena.floatVec(m);
        var rng = new Unity.Mathematics.Random(20260709u);
        for (int i = 0; i < m; i++)
        {
            float t = rng.NextFloat(0f, 10f);
            A[i, 0] = 1f; A[i, 1] = t;
            b[i] = 1f + 2f * t + rng.NextFloat(-3f, 3f);   // symmetric noise about the true line
        }
        var x = arena.floatVec(2);
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

    [Test]
    public void TauQuarterResidualSign_Double()
    {
        var arena = new Arena(Allocator.Persistent);
        int m = 80;
        var A = arena.doubleMat(m, 2);
        var b = arena.doubleVec(m);
        var rng = new Unity.Mathematics.Random(20260709u);
        for (int i = 0; i < m; i++)
        {
            double t = rng.NextDouble(0.0, 10.0);
            A[i, 0] = 1.0; A[i, 1] = t;
            b[i] = 1.0 + 2.0 * t + rng.NextDouble(-3.0, 3.0);   // symmetric noise about the true line
        }
        var x = arena.doubleVec(2);
        LP.ladFrischNewtonCore(in A, in b, 0.25, ref x, out double obj, 0);

        int neg = 0;
        for (int i = 0; i < m; i++)
        {
            double fit = x[0] + x[1] * A[i, 1];
            if (b[i] - fit < 0.0) neg++;
        }
        double target = 0.25 * m;   // 20
        Assert.That(neg, Is.EqualTo(target).Within(0.20 * m),
            $"tau=0.25: {neg}/{m} residuals negative, expected ~{target} (+/-{0.20 * m})");

        arena.Dispose();
    }
}
