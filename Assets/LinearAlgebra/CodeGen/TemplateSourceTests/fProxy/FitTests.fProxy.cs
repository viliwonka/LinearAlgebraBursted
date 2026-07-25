using System;

using BULA;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

//+deleteThis
using fProxy2 = Unity.Mathematics.float2;
using fProxy3 = Unity.Mathematics.float3;
//-deleteThis

// Acceptance tests for the Fit facade (OP/Fit.*.fProxy.cs): geometric fits (line/plane/sphere) with a
// pluggable metric, and linear-model fits split by which residual is minimized (Fit.linear = vertical,
// Fit.total = orthogonal / errors-in-variables).
//
// Oracles, in order of strength:
//   * Fit.total vs Fit.line on CENTERED data -- two entirely independent code paths (SVD of the
//     augmented [A|b] vs eigendecomposition of the covariance) that must agree on the same quantity.
//     This is the primary correctness anchor for the TLS solve.
//   * Exact-construction recovery (points placed exactly on a line/circle).
//   * Fit.linear vs a direct QR.solveInPlace on the same data.
//   * Defining-property comparisons: robust beats L2 under an outlier; total beats vertical when the
//     PREDICTOR carries noise (the attenuation bias a vertical fit cannot escape).
//
// Robust cases use a MODERATE outlier on purpose. IRLS seeds from the L2 fit and is only locally
// convergent, so an outlier violent enough to flip the L2 fit outright would leave the reweighting
// converging to the wrong fixed point -- that is a property of IRLS, not a defect here, and testing
// in that regime would pin behavior nobody should rely on.
//
// Run on the managed thread (like LadFrischNewtonScaleTests) so a divergence reports a clear value.
public class fProxyFitTests
{
    // Comparison tolerance, in double (every assertion below compares widened values). The cast must
    // stay OUTSIDE a choose marker -- the marker replaces everything between its delimiters, so a
    // `(fProxy)` written inside it is stripped from the generated code.
    static double Tol => /*+choose[2e-3|1e-9]*/2e-3/*-choose*/;

    // ---------------------------------------------------------------- line

    // Points placed exactly on y = 0.5x + 1: the fitted direction must be parallel to (1, 0.5) and
    // the centroid must be the mean. Direction sign is arbitrary, so compare |cos| to 1.
    [Test]
    public void LineExactRecoversDirection()
    {
        var pts = new NativeArray<fProxy2>(6, Allocator.Temp);
        for (int i = 0; i < 6; i++) pts[i] = new fProxy2((fProxy)i, (fProxy)(0.5 * i + 1.0));

        Assert.IsTrue(Fit.line(pts, out fProxy2 c, out fProxy2 d), "Fit.line did not converge");

        fProxy2 want = math.normalize(new fProxy2((fProxy)1, (fProxy)0.5));
        double cos = math.abs(math.dot(math.normalize(d), want));
        Assert.That(cos, Is.EqualTo(1.0).Within(Tol), "direction not parallel to the true line");
        Assert.That((double)c.x, Is.EqualTo(2.5).Within(Tol), "centroid x");

        pts.Dispose();
    }

    // 20 inliers exactly on y = x, plus ONE off-line point. The L2 fit tilts toward the outlier; a
    // robust loss down-weights it and lands closer to the true 45-degree direction. Asserted as a
    // strict comparison of angle error, not an absolute bound, so it measures the metric's effect
    // rather than a tuned tolerance.
    [Test]
    public void LineRobustBeatsL2UnderOutlier()
    {
        const int n = 21;
        var pts = new NativeArray<fProxy2>(n, Allocator.Temp);
        for (int i = 0; i < 20; i++) pts[i] = new fProxy2((fProxy)i, (fProxy)i);
        pts[20] = new fProxy2((fProxy)10, (fProxy)26);          // moderate, off the line

        fProxy2 want = math.normalize(new fProxy2((fProxy)1, (fProxy)1));

        Assert.IsTrue(Fit.line(pts, out _, out fProxy2 dL2));
        double errL2 = AngleError(dL2, want);

        var huber = new fProxyHuberLoss((fProxy)1);
        Assert.IsTrue(Fit.line(pts, in huber, out _, out fProxy2 dH), "robust line fit failed");
        double errH = AngleError(dH, want);

        var l1 = new fProxyL1Loss();
        Assert.IsTrue(Fit.line(pts, in l1, out _, out fProxy2 dL1), "L1 line fit failed");
        double errL1 = AngleError(dL1, want);

        Assert.Less(errH, errL2, $"Huber ({errH}) should beat L2 ({errL2}) under an outlier");
        Assert.Less(errL1, errL2, $"L1 ({errL1}) should beat L2 ({errL2}) under an outlier");

        pts.Dispose();
    }

    // Plane through z = 0 with one lifted point at a CORNER. The corner placement is the whole point:
    // a point lifted at the grid's centroid inflates the z-variance without tilting anything, so the
    // L2 normal stays exactly (0,0,1) and no robust fit can improve on it. Off-centre, the outlier
    // induces genuine xz/yz covariance and tilts the L2 normal, which is what the robust loss undoes.
    [Test]
    public void PlaneRobustBeatsL2UnderOutlier()
    {
        const int n = 17;
        var pts = new NativeArray<fProxy3>(n, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                pts[k++] = new fProxy3((fProxy)i, (fProxy)j, (fProxy)0);
        pts[16] = new fProxy3((fProxy)3, (fProxy)3, (fProxy)4);

        fProxy3 want = new fProxy3((fProxy)0, (fProxy)0, (fProxy)1);

        Assert.IsTrue(Fit.plane(pts, out _, out fProxy3 nL2));
        double errL2 = AngleError(nL2, want);

        var huber = new fProxyHuberLoss((fProxy)0.5);
        Assert.IsTrue(Fit.plane(pts, in huber, out _, out fProxy3 nH), "robust plane fit failed");
        double errH = AngleError(nH, want);

        Assert.Less(errH, errL2, $"Huber normal ({errH}) should beat L2 ({errL2})");

        pts.Dispose();
    }

    // ---------------------------------------------------------------- sphere

    // Points placed exactly on a circle of known center/radius: the algebraic fit is exact there
    // (zero algebraic residual), so this pins the solve rather than its bias.
    [Test]
    public void CircleExactRecoversCenterAndRadius()
    {
        var pts = new NativeArray<fProxy2>(8, Allocator.Temp);
        for (int i = 0; i < 8; i++)
        {
            double t = 2.0 * math.PI_DBL * i / 8.0;
            pts[i] = new fProxy2((fProxy)(2.0 + 5.0 * math.cos(t)), (fProxy)(3.0 + 5.0 * math.sin(t)));
        }

        Assert.IsTrue(Fit.sphere(pts, out fProxy2 c, out fProxy r), "circle fit failed");
        Assert.That((double)c.x, Is.EqualTo(2.0).Within(Tol), "center x");
        Assert.That((double)c.y, Is.EqualTo(3.0).Within(Tol), "center y");
        Assert.That((double)r, Is.EqualTo(5.0).Within(Tol), "radius");

        pts.Dispose();
    }

    // Same circle with one point pulled off it: the robust fit must recover the true radius more
    // closely than the algebraic one.
    [Test]
    public void CircleRobustBeatsAlgebraicUnderOutlier()
    {
        const int n = 13;
        var pts = new NativeArray<fProxy2>(n, Allocator.Temp);
        for (int i = 0; i < 12; i++)
        {
            double t = 2.0 * math.PI_DBL * i / 12.0;
            pts[i] = new fProxy2((fProxy)(5.0 * math.cos(t)), (fProxy)(5.0 * math.sin(t)));
        }
        pts[12] = new fProxy2((fProxy)8, (fProxy)0);            // off the circle, moderate

        Assert.IsTrue(Fit.sphere(pts, out _, out fProxy rA), "algebraic circle fit failed");

        var huber = new fProxyHuberLoss((fProxy)0.5);
        Assert.IsTrue(Fit.sphere(pts, in huber, out _, out fProxy rH), "robust circle fit failed");

        double errA = math.abs((double)rA - 5.0);
        double errH = math.abs((double)rH - 5.0);
        Assert.Less(errH, errA, $"robust radius err ({errH}) should beat algebraic ({errA})");

        pts.Dispose();
    }

    // ---------------------------------------------------------------- linear model

    // Fit.linear is QR least squares; it must reproduce a direct QR.solveInPlace on the same data.
    [Test]
    public void LinearMatchesQrOracle()
    {
        int m = 12, n = 3;
        var A = new fProxyMxN(m, n, Allocator.Temp);
        var b = new fProxyN(m, Allocator.Temp);
        var rng = new Unity.Mathematics.Random(12345u);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++) A[i, j] = (fProxy)rng.NextDouble(-1.0, 1.0);
            b[i] = (fProxy)rng.NextDouble(-1.0, 1.0);
        }

        var x = new fProxyN(n, Allocator.Temp);
        Assert.IsTrue(Fit.linear(in A, in b, ref x), "Fit.linear failed");

        var Ac = new fProxyMxN(m, n, Allocator.Temp);
        var bc = new fProxyN(m, Allocator.Temp);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++) Ac[i, j] = A[i, j];
            bc[i] = b[i];
        }
        var xq = new fProxyN(n, Allocator.Temp);
        Assert.IsTrue(QR.solveInPlace(ref Ac, ref bc, ref xq), "QR oracle failed");

        for (int j = 0; j < n; j++)
            Assert.That((double)x[j], Is.EqualTo((double)xq[j]).Within(Tol), $"coefficient {j}");

        A.Dispose(); b.Dispose(); x.Dispose(); Ac.Dispose(); bc.Dispose(); xq.Dispose();
    }

    // PRIMARY TLS ANCHOR. On centered data, the total-least-squares fit of y on x is exactly the
    // dominant principal direction of the point cloud -- so Fit.total (SVD of the augmented [A|b])
    // and Fit.line (eigendecomposition of the covariance) must agree on the slope, having computed
    // it through completely independent machinery.
    [Test]
    public void TotalMatchesLineDirectionOnCenteredData()
    {
        const int n = 24;
        var raw = new NativeArray<fProxy2>(n, Allocator.Temp);
        var rng = new Unity.Mathematics.Random(777u);
        for (int i = 0; i < n; i++)
        {
            double x = i * 0.5;
            double y = 1.7 * x + rng.NextDouble(-0.4, 0.4);
            raw[i] = new fProxy2((fProxy)x, (fProxy)y);
        }

        // Center, so the through-origin model Fit.total solves matches the through-centroid line.
        double mx = 0, my = 0;
        for (int i = 0; i < n; i++) { mx += (double)raw[i].x; my += (double)raw[i].y; }
        mx /= n; my /= n;

        var pts = new NativeArray<fProxy2>(n, Allocator.Temp);
        var A = new fProxyMxN(n, 1, Allocator.Temp);
        var b = new fProxyN(n, Allocator.Temp);
        for (int i = 0; i < n; i++)
        {
            fProxy cx = (fProxy)((double)raw[i].x - mx), cy = (fProxy)((double)raw[i].y - my);
            pts[i] = new fProxy2(cx, cy);
            A[i, 0] = cx;
            b[i] = cy;
        }

        var x1 = new fProxyN(1, Allocator.Temp);
        Assert.IsTrue(Fit.total(in A, in b, ref x1), "Fit.total failed");

        Assert.IsTrue(Fit.line(pts, out _, out fProxy2 dir), "Fit.line failed");
        double slopeLine = (double)dir.y / (double)dir.x;

        Assert.That((double)x1[0], Is.EqualTo(slopeLine).Within(Tol),
            "TLS slope must equal the dominant principal direction on centered data");

        raw.Dispose(); pts.Dispose(); A.Dispose(); b.Dispose(); x1.Dispose();
    }

    // THE DEFINING PROPERTY of total least squares: when the PREDICTOR carries noise, a vertical fit
    // is biased toward zero slope (regression attenuation) because it charges all error to b. TLS
    // splits the error between A and b and must land closer to the truth. No choice of loss function
    // fixes the vertical fit here -- it is the wrong residual, not the wrong weighting.
    [Test]
    public void TotalBeatsVerticalWhenPredictorIsNoisy()
    {
        const int n = 200;
        const double trueSlope = 2.0;
        var A = new fProxyMxN(n, 1, Allocator.Temp);
        var b = new fProxyN(n, Allocator.Temp);
        var rng = new Unity.Mathematics.Random(2024u);

        double mx = 0, my = 0;
        var xs = new double[n];
        var ys = new double[n];
        for (int i = 0; i < n; i++)
        {
            double xTrue = i * 0.05;                             // spread 0..10
            xs[i] = xTrue + rng.NextDouble(-0.6, 0.6);           // noise in the PREDICTOR
            ys[i] = trueSlope * xTrue + rng.NextDouble(-0.6, 0.6);
            mx += xs[i]; my += ys[i];
        }
        mx /= n; my /= n;
        for (int i = 0; i < n; i++) { A[i, 0] = (fProxy)(xs[i] - mx); b[i] = (fProxy)(ys[i] - my); }

        var xLs = new fProxyN(1, Allocator.Temp);
        var xTls = new fProxyN(1, Allocator.Temp);
        Assert.IsTrue(Fit.linear(in A, in b, ref xLs), "vertical fit failed");
        Assert.IsTrue(Fit.total(in A, in b, ref xTls), "total fit failed");

        double errLs = math.abs((double)xLs[0] - trueSlope);
        double errTls = math.abs((double)xTls[0] - trueSlope);

        Assert.Less((double)xLs[0], trueSlope, "vertical fit should be attenuated below the true slope");
        Assert.Less(errTls, errLs, $"TLS err ({errTls}) should beat vertical err ({errLs}) under x-noise");

        A.Dispose(); b.Dispose(); xLs.Dispose(); xTls.Dispose();
    }

    // Nongeneric TLS: A is all zeros, so the smallest right singular vector of [A|b] is (1,0) -- its
    // b component is exactly zero and no finite x exists. Must report false, not a fabricated x.
    [Test]
    public void TotalNongenericReturnsFalse()
    {
        int m = 4;
        var A = new fProxyMxN(m, 1, Allocator.Temp);
        var b = new fProxyN(m, Allocator.Temp);
        for (int i = 0; i < m; i++) { A[i, 0] = (fProxy)0; b[i] = (fProxy)(i + 1); }

        var x = new fProxyN(1, Allocator.Temp);
        Assert.IsFalse(Fit.total(in A, in b, ref x), "a nongeneric TLS problem must not report success");

        A.Dispose(); b.Dispose(); x.Dispose();
    }

    // ---------------------------------------------------------------- guards

    [Test]
    public void ArgumentGuardsThrow()
    {
        var p1 = new NativeArray<fProxy2>(1, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => Fit.line(p1, out _, out _));
        Assert.Throws<ArgumentException>(() => Fit.sphere(p1, out _, out _));
        p1.Dispose();

        var p2 = new NativeArray<fProxy3>(2, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => Fit.plane(p2, out _, out _));
        p2.Dispose();

        // Fit.total needs the AUGMENTED matrix to stay tall: m >= n + 1.
        var A = new fProxyMxN(2, 2, Allocator.Temp);
        var b = new fProxyN(2, Allocator.Temp);
        var x = new fProxyN(2, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => Fit.total(in A, in b, ref x));
        A.Dispose(); b.Dispose(); x.Dispose();
    }

    static double AngleError(fProxy2 got, fProxy2 want)
    {
        double c = math.abs(math.dot(math.normalize(got), math.normalize(want)));
        return math.acos(math.min(c, 1.0));
    }

    static double AngleError(fProxy3 got, fProxy3 want)
    {
        double c = math.abs(math.dot(math.normalize(got), math.normalize(want)));
        return math.acos(math.min(c, 1.0));
    }
}
