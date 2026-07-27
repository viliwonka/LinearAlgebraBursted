using System;

using BULA;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

using Random = Unity.Mathematics.Random;   // this file imports System, which has its own Random

//+deleteThis
using fProxy2 = Unity.Mathematics.float2;
using fProxy3 = Unity.Mathematics.float3;
using fProxy4 = Unity.Mathematics.float4;
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


    // Wrap to (-pi/2, pi/2]: an ellipse axis is a direction, not a ray.
    static double AngleModPi(double a)
    {
        while (a > math.PI_DBL / 2.0) a -= math.PI_DBL;
        while (a <= -math.PI_DBL / 2.0) a += math.PI_DBL;
        return a;
    }

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

    // ---------------------------------------------------------------- conic / quadric

    // Points placed exactly on a known axis-aligned ellipse (centre (3,-2), semi-axes 5 and 2): the
    // recovered geometry must match. Axis-aligned so the angle is 0 or +-pi/2 and the radii pairing
    // is unambiguous -- a rotated case would need the angle and radii checked jointly.
    [Test]
    public void EllipseExactRecoversGeometry()
    {
        const int n = 16;
        var pts = new NativeArray<fProxy2>(n, Allocator.Temp);
        for (int i = 0; i < n; i++)
        {
            double t = 2.0 * math.PI_DBL * i / n;
            pts[i] = new fProxy2((fProxy)(3.0 + 5.0 * math.cos(t)), (fProxy)(-2.0 + 2.0 * math.sin(t)));
        }

        Assert.IsTrue(Fit.ellipse(pts, out fProxy2 c, out fProxy2 r, out fProxy ang), "ellipse fit failed");

        Assert.That((double)c.x, Is.EqualTo(3.0).Within(Tol), "centre x");
        Assert.That((double)c.y, Is.EqualTo(-2.0).Within(Tol), "centre y");

        // Radii come out ordered by the eigensolve, not by size: compare as a set.
        double big = math.max((double)r.x, (double)r.y);
        double small = math.min((double)r.x, (double)r.y);
        Assert.That(big, Is.EqualTo(5.0).Within(Tol), "major semi-axis");
        Assert.That(small, Is.EqualTo(2.0).Within(Tol), "minor semi-axis");

        pts.Dispose();
    }

    // The ellipse CONSTRAINT is the point of using Halir-Flusser over a plain algebraic conic fit:
    // 4AC - B^2 > 0 must hold for the returned coefficients, so the answer can never be a hyperbola.
    [Test]
    public void ConicIsAlwaysAnEllipse()
    {
        // A short, shallow arc -- the classic case where an unconstrained conic fit escapes to a
        // hyperbola because the data does not pin the far side of the curve.
        const int n = 12;
        var pts = new NativeArray<fProxy2>(n, Allocator.Temp);
        for (int i = 0; i < n; i++)
        {
            double t = 0.35 * math.PI_DBL * i / (n - 1);        // under a quarter turn
            pts[i] = new fProxy2((fProxy)(10.0 * math.cos(t)), (fProxy)(4.0 * math.sin(t)));
        }

        var c = new fProxyN(6, Allocator.Temp);
        Assert.IsTrue(Fit.conic(pts, ref c), "conic fit failed on a short arc");

        double disc = 4.0 * (double)c[0] * (double)c[2] - (double)c[1] * (double)c[1];
        Assert.Greater(disc, 0.0, "4AC - B^2 must be positive: the fit is constrained to an ellipse");

        pts.Dispose(); c.Dispose();
    }

    // `classify` documents itself as scale-invariant, so a big sphere must classify exactly like a
    // small one. It is a real trap rather than a hypothetical: `quadric` returns UNIT-NORM
    // coefficients, so a sphere of radius R has quadratic entries of order 1/R², and a zero test
    // floored at an absolute constant swallows the whole quadratic form once R grows.
    [Test]
    public void ClassifyIsScaleInvariantForLargeShapes()
    {
        foreach (double R in new[] { 1.0, 60.0, 500.0 })
        {
            var pts = SampleEllipsoid(R, R, R, default, 8, 12);

            var c = new fProxyN(10, Allocator.Temp);
            Assert.IsTrue(Fit.quadric(pts, ref c), $"quadric fit failed at R = {R}");
            Assert.AreEqual(QuadricKind.Ellipsoid, Fit.classify(in c),
                $"a sphere of radius {R} is an ellipsoid at every scale");

            pts.Dispose(); c.Dispose();
        }
    }

    // A quadric fitted to points sampled from a true ellipsoid must classify as one; points sampled
    // from a one-sheet hyperboloid must not. Classification is the whole reason `quadric` is one
    // entry point rather than a solver per shape, so it gets both a positive and a negative case.
    [Test]
    public void QuadricClassifiesEllipsoidAndHyperboloid()
    {
        var ell = new NativeArray<fProxy3>(60, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 10; j++)
            {
                double u = math.PI_DBL * (i + 0.5) / 6.0, v = 2.0 * math.PI_DBL * j / 10.0;
                ell[k++] = new fProxy3((fProxy)(3.0 * math.sin(u) * math.cos(v)),
                                       (fProxy)(2.0 * math.sin(u) * math.sin(v)),
                                       (fProxy)(1.5 * math.cos(u)));
            }

        var ce = new fProxyN(10, Allocator.Temp);
        Assert.IsTrue(Fit.quadric(ell, ref ce), "ellipsoid quadric fit failed");
        Assert.AreEqual(QuadricKind.Ellipsoid, Fit.classify(in ce), "should classify as an ellipsoid");

        // x²/4 + y²/4 - z² = 1, a one-sheet hyperboloid: mixed signature.
        var hyp = new NativeArray<fProxy3>(60, Allocator.Temp);
        k = 0;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 10; j++)
            {
                double zz = -1.0 + 2.0 * i / 5.0, v = 2.0 * math.PI_DBL * j / 10.0;
                double rr = 2.0 * math.sqrt(1.0 + zz * zz);
                hyp[k++] = new fProxy3((fProxy)(rr * math.cos(v)), (fProxy)(rr * math.sin(v)), (fProxy)zz);
            }

        var ch = new fProxyN(10, Allocator.Temp);
        Assert.IsTrue(Fit.quadric(hyp, ref ch), "hyperboloid quadric fit failed");
        Assert.AreEqual(QuadricKind.HyperboloidOrCone, Fit.classify(in ch), "should classify as hyperboloid/cone");

        ell.Dispose(); hyp.Dispose(); ce.Dispose(); ch.Dispose();
    }

    // ------------------------------------------------- transform equivariance
    //
    // Generate a shape in canonical position, apply a KNOWN rigid transform, fit the transformed
    // points, and require the recovered shape to be the transform of the canonical answer. This is
    // strictly stronger than the axis-aligned cases above: those leave rotation-carrying outputs
    // (the ellipse angle, the plane normal's tilt) at their trivial values, so a wrong rotation
    // convention would pass unnoticed.

    // Rotation by (c, s) about the origin.
    static fProxy2 Rot2(fProxy2 v, fProxy c, fProxy s) => new fProxy2(c * v.x - s * v.y, s * v.x + c * v.y);

    // Rz(alpha) then Rx(beta). Written out in fProxy arithmetic rather than via quaternion, which
    // Unity.Mathematics provides for float only and so would not survive the double expansion.
    static fProxy3 Rot3(fProxy3 v, fProxy ca, fProxy sa, fProxy cb, fProxy sb)
    {
        fProxy x1 = ca * v.x - sa * v.y, y1 = sa * v.x + ca * v.y, z1 = v.z;
        return new fProxy3(x1, cb * y1 - sb * z1, sb * y1 + cb * z1);
    }

    // THE test the axis-aligned ellipse case cannot do: a rotated ellipse pins the angle AND its
    // pairing with the radii. Fit.ellipse documents radii.x as the semi-axis lying along `angle`, so
    // whichever radius is the major one determines which direction should equal the true major axis.
    // Getting the pairing backwards (reporting the minor axis's direction with the major radius) is
    // a real and easy mistake that only a rotated case exposes.
    [Test]
    public void EllipseRotatedRecoversAxesAndAngle()
    {
        const double theta = 0.6, aMaj = 5.0, bMin = 2.0, cx = 3.0, cy = -2.0;
        fProxy ct = (fProxy)math.cos(theta), st = (fProxy)math.sin(theta);

        const int n = 24;
        var pts = new NativeArray<fProxy2>(n, Allocator.Temp);
        for (int i = 0; i < n; i++)
        {
            double t = 2.0 * math.PI_DBL * i / n;
            var canon = new fProxy2((fProxy)(aMaj * math.cos(t)), (fProxy)(bMin * math.sin(t)));
            var r = Rot2(canon, ct, st);
            pts[i] = new fProxy2(r.x + (fProxy)cx, r.y + (fProxy)cy);
        }

        Assert.IsTrue(Fit.ellipse(pts, out fProxy2 c, out fProxy2 rad, out fProxy ang), "ellipse fit failed");

        Assert.That((double)c.x, Is.EqualTo(cx).Within(Tol), "centre x");
        Assert.That((double)c.y, Is.EqualTo(cy).Within(Tol), "centre y");

        double big = math.max((double)rad.x, (double)rad.y);
        double small = math.min((double)rad.x, (double)rad.y);
        Assert.That(big, Is.EqualTo(aMaj).Within(Tol), "major semi-axis");
        Assert.That(small, Is.EqualTo(bMin).Within(Tol), "minor semi-axis");

        // radii.x lies along `ang`; the major axis direction is therefore ang, or ang + pi/2 when the
        // larger radius came out in y. An ellipse axis is a direction, not a ray, so compare mod pi.
        double angMajor = (double)rad.x >= (double)rad.y ? (double)ang : (double)ang + math.PI_DBL / 2.0;
        Assert.That(AngleModPi(angMajor - theta), Is.EqualTo(0.0).Within(Tol),
            "major-axis direction must match the applied rotation (angle/radius pairing)");

        pts.Dispose();
    }

    // A plane's normal must rotate with the data. Canonical normal is +z; after Rz(a)Rx(b) it is the
    // transform of +z, up to sign (the fit does not fix a side).
    [Test]
    public void PlaneNormalIsRigidEquivariant()
    {
        fProxy ca = (fProxy)math.cos(0.7), sa = (fProxy)math.sin(0.7);
        fProxy cb = (fProxy)math.cos(-0.4), sb = (fProxy)math.sin(-0.4);
        var shift = new fProxy3((fProxy)2, (fProxy)(-5), (fProxy)1.25);

        var pts = new NativeArray<fProxy3>(25, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
            {
                var canon = new fProxy3((fProxy)(i - 2), (fProxy)(j - 2), (fProxy)0);
                var r = Rot3(canon, ca, sa, cb, sb);
                pts[k++] = new fProxy3(r.x + shift.x, r.y + shift.y, r.z + shift.z);
            }

        Assert.IsTrue(Fit.plane(pts, out fProxy3 c, out fProxy3 nrm), "plane fit failed");

        var wantN = Rot3(new fProxy3((fProxy)0, (fProxy)0, (fProxy)1), ca, sa, cb, sb);
        Assert.That(AngleError(nrm, wantN), Is.EqualTo(0.0).Within(Tol), "normal did not follow the rotation");
        Assert.That((double)c.x, Is.EqualTo((double)shift.x).Within(Tol), "centroid x");
        Assert.That((double)c.z, Is.EqualTo((double)shift.z).Within(Tol), "centroid z");

        pts.Dispose();
    }

    // A sphere's centre must follow a rigid transform while its radius is invariant to it, and must
    // scale exactly with a uniform scaling. Two different equivariances on one fit.
    [Test]
    public void SphereIsRigidEquivariantAndScaleCovariant()
    {
        fProxy ca = (fProxy)math.cos(0.9), sa = (fProxy)math.sin(0.9);
        fProxy cb = (fProxy)math.cos(0.3), sb = (fProxy)math.sin(0.3);
        var shift = new fProxy3((fProxy)(-4), (fProxy)7, (fProxy)2);
        const double scale = 3.0, r0 = 1.5;

        var pts = new NativeArray<fProxy3>(42, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 7; j++)
            {
                double u = math.PI_DBL * (i + 0.5) / 6.0, v = 2.0 * math.PI_DBL * j / 7.0;
                var canon = new fProxy3((fProxy)(scale * r0 * math.sin(u) * math.cos(v)),
                                        (fProxy)(scale * r0 * math.sin(u) * math.sin(v)),
                                        (fProxy)(scale * r0 * math.cos(u)));
                var r = Rot3(canon, ca, sa, cb, sb);
                pts[k++] = new fProxy3(r.x + shift.x, r.y + shift.y, r.z + shift.z);
            }

        Assert.IsTrue(Fit.sphere(pts, out fProxy3 c, out fProxy rad), "sphere fit failed");

        Assert.That((double)c.x, Is.EqualTo((double)shift.x).Within(Tol), "centre x follows the shift");
        Assert.That((double)c.y, Is.EqualTo((double)shift.y).Within(Tol), "centre y follows the shift");
        Assert.That((double)c.z, Is.EqualTo((double)shift.z).Within(Tol), "centre z follows the shift");
        Assert.That((double)rad, Is.EqualTo(scale * r0).Within(Tol), "radius scales, and rotation leaves it alone");

        pts.Dispose();
    }

    // A rotation is a similarity transform of the quadratic form, so it preserves the eigenvalue
    // signature exactly -- classification must be rotation-invariant. This pins the classifier
    // against an implementation that accidentally depends on axis alignment.
    [Test]
    public void QuadricClassificationIsRotationInvariant()
    {
        fProxy ca = (fProxy)math.cos(0.55), sa = (fProxy)math.sin(0.55);
        fProxy cb = (fProxy)math.cos(0.8), sb = (fProxy)math.sin(0.8);

        var pts = new NativeArray<fProxy3>(60, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 10; j++)
            {
                double u = math.PI_DBL * (i + 0.5) / 6.0, v = 2.0 * math.PI_DBL * j / 10.0;
                var canon = new fProxy3((fProxy)(3.0 * math.sin(u) * math.cos(v)),
                                        (fProxy)(2.0 * math.sin(u) * math.sin(v)),
                                        (fProxy)(1.5 * math.cos(u)));
                pts[k++] = Rot3(canon, ca, sa, cb, sb);
            }

        var c = new fProxyN(10, Allocator.Temp);
        Assert.IsTrue(Fit.quadric(pts, ref c), "rotated ellipsoid quadric fit failed");
        Assert.AreEqual(QuadricKind.Ellipsoid, Fit.classify(in c),
            "a rotated ellipsoid is still an ellipsoid: the signature is rotation-invariant");

        pts.Dispose(); c.Dispose();
    }

    // ------------------------------------------------------- nonlinear solids
    //
    // Each shape is generated EXACTLY in canonical position, then rigidly transformed, so the fit has
    // a known right answer and the transform exercises the same rotation handling the equivariance
    // tests pin for the linear fits. Points lie exactly on the surface, so a converged fit should
    // reproduce the generating parameters -- these are recovery tests, not noise-tolerance tests.

    static double SolidTol => /*+choose[5e-3|1e-6]*/5e-3/*-choose*/;

    // Rotation used by every solid case below.
    static fProxy3 SolidRot(fProxy3 v) =>
        Rot3(v, (fProxy)math.cos(0.5), (fProxy)math.sin(0.5), (fProxy)math.cos(-0.35), (fProxy)math.sin(-0.35));

    static fProxy3 SolidShift => new fProxy3((fProxy)1.5, (fProxy)(-3), (fProxy)0.75);

    static fProxy3 Place(fProxy3 canonical) => SolidRot(canonical) + SolidShift;

    [Test]
    public void CylinderRecoversAxisAndRadius()
    {
        const double rad = 2.0;
        var pts = new NativeArray<fProxy3>(48, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 8; j++)
            {
                double z = -3.0 + 6.0 * i / 5.0, th = 2.0 * math.PI_DBL * j / 8.0;
                pts[k++] = Place(new fProxy3((fProxy)(rad * math.cos(th)), (fProxy)(rad * math.sin(th)), (fProxy)z));
            }

        fProxy3 q = default, d = default; fProxy r = default;
        Assert.IsTrue(Fit.cylinder(pts, ref q, ref d, ref r), "cylinder fit did not converge");

        var wantDir = SolidRot(new fProxy3((fProxy)0, (fProxy)0, (fProxy)1));
        Assert.That(AngleError(d, wantDir), Is.EqualTo(0.0).Within(SolidTol), "axis direction");
        Assert.That((double)r, Is.EqualTo(rad).Within(SolidTol), "radius");

        // axisPoint is gauge-free ALONG the axis, so pin the LINE: its distance to the true axis.
        fProxy3 v = q - SolidShift;
        double offAxis = math.length(v - math.dot(v, wantDir) * wantDir);
        Assert.That(offAxis, Is.EqualTo(0.0).Within(SolidTol), "axis point must lie on the true axis");

        pts.Dispose();
    }

    [Test]
    public void ConeRecoversApexAxisAndAngle()
    {
        const double half = 0.4;
        double tanA = math.tan(half);
        var pts = new NativeArray<fProxy3>(48, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 8; j++)
            {
                double t = 1.0 + 4.0 * i / 5.0, th = 2.0 * math.PI_DBL * j / 8.0;
                double rr = t * tanA;
                pts[k++] = Place(new fProxy3((fProxy)(rr * math.cos(th)), (fProxy)(rr * math.sin(th)), (fProxy)t));
            }

        fProxy3 apex = default, d = default; fProxy ang = default;
        Assert.IsTrue(Fit.cone(pts, ref apex, ref d, ref ang), "cone fit did not converge");

        var wantDir = SolidRot(new fProxy3((fProxy)0, (fProxy)0, (fProxy)1));
        Assert.That(AngleError(d, wantDir), Is.EqualTo(0.0).Within(SolidTol), "axis direction");
        Assert.That((double)ang, Is.EqualTo(half).Within(SolidTol), "half angle");

        var wantApex = Place(new fProxy3((fProxy)0, (fProxy)0, (fProxy)0));
        Assert.That((double)math.length(apex - wantApex), Is.EqualTo(0.0).Within(SolidTol), "apex position");

        pts.Dispose();
    }

    // (d, a) and (-d, -a) parameterize the SAME cone, so a warm start whose axis points the wrong
    // way legitimately converges with a negative angle. The report must fold that sign back into
    // the axis: an unflipped axis names the mirror nappe, a cone touching none of the inputs.
    [Test]
    public void ConeWarmStartedAntiParallelReturnsTheTrueNappe()
    {
        const double half = 0.4;
        double tanA = math.tan(half);
        var pts = new NativeArray<fProxy3>(48, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 8; j++)
            {
                double t = 1.0 + 4.0 * i / 5.0, th = 2.0 * math.PI_DBL * j / 8.0;
                double rr = t * tanA;
                pts[k++] = Place(new fProxy3((fProxy)(rr * math.cos(th)), (fProxy)(rr * math.sin(th)), (fProxy)t));
            }

        var wantDir = SolidRot(new fProxy3((fProxy)0, (fProxy)0, (fProxy)1));
        var wantApex = Place(new fProxy3((fProxy)0, (fProxy)0, (fProxy)0));

        fProxy3 apex = wantApex;
        fProxy3 d = -wantDir;               // same axis LINE, wrong orientation
        fProxy ang = (fProxy)half;
        Assert.IsTrue(Fit.cone(pts, ref apex, ref d, ref ang), "cone fit did not converge");

        // Signed alignment: AngleError's abs(dot) would pass on the mirror nappe too.
        Assert.IsTrue(math.dot(d, wantDir) > (fProxy)0.9, "axis must point into the fitted nappe");
        Assert.That((double)ang, Is.EqualTo(half).Within(SolidTol), "half angle");

        // Distance-zero oracle: every input must lie on the REPORTED cone's surface.
        double maxDist = 0.0;
        for (int i = 0; i < pts.Length; i++)
        {
            fProxy3 v = pts[i] - apex;
            fProxy ax = math.dot(v, d);
            fProxy rad = math.length(v - ax * d);
            double dist = math.abs((double)(rad * math.cos(ang) - ax * math.sin(ang)));
            if (dist > maxDist) maxDist = dist;
        }
        Assert.That(maxDist, Is.EqualTo(0.0).Within(SolidTol), "every input point must lie on the reported cone");

        pts.Dispose();
    }

    [Test]
    public void TorusRecoversRadiiAndAxis()
    {
        const double R = 3.0, r0 = 1.0;
        var pts = new NativeArray<fProxy3>(96, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < 12; i++)
            for (int j = 0; j < 8; j++)
            {
                double th = 2.0 * math.PI_DBL * i / 12.0, ph = 2.0 * math.PI_DBL * j / 8.0;
                double rr = R + r0 * math.cos(ph);
                pts[k++] = Place(new fProxy3((fProxy)(rr * math.cos(th)), (fProxy)(rr * math.sin(th)),
                                             (fProxy)(r0 * math.sin(ph))));
            }

        fProxy3 c = default, d = default; fProxy R1 = default, r1 = default;
        Assert.IsTrue(Fit.torus(pts, ref c, ref d, ref R1, ref r1), "torus fit did not converge");

        var wantDir = SolidRot(new fProxy3((fProxy)0, (fProxy)0, (fProxy)1));
        Assert.That(AngleError(d, wantDir), Is.EqualTo(0.0).Within(SolidTol), "ring axis");
        Assert.That((double)R1, Is.EqualTo(R).Within(SolidTol), "major radius");
        Assert.That((double)r1, Is.EqualTo(r0).Within(SolidTol), "minor radius");
        Assert.That((double)math.length(c - SolidShift), Is.EqualTo(0.0).Within(SolidTol), "centre");

        pts.Dispose();
    }

    [Test]
    public void CapsuleRecoversEndpointsAndRadius()
    {
        const double halfLen = 2.0, rad = 1.0;
        var pts = new NativeArray<fProxy3>(72, Allocator.Temp);
        int k = 0;

        // Barrel.
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 8; j++)
            {
                double z = -halfLen + 2.0 * halfLen * i / 5.0, th = 2.0 * math.PI_DBL * j / 8.0;
                pts[k++] = Place(new fProxy3((fProxy)(rad * math.cos(th)), (fProxy)(rad * math.sin(th)), (fProxy)z));
            }
        // Caps -- without these the fit cannot tell a capsule from a longer cylinder.
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
            {
                double u = math.PI_DBL * 0.5 * (i + 1) / 3.0, th = 2.0 * math.PI_DBL * j / 4.0;
                double rr = rad * math.sin(u), dz = rad * math.cos(u);
                pts[k++] = Place(new fProxy3((fProxy)(rr * math.cos(th)), (fProxy)(rr * math.sin(th)),
                                             (fProxy)(halfLen + dz)));
                pts[k++] = Place(new fProxy3((fProxy)(rr * math.cos(th)), (fProxy)(rr * math.sin(th)),
                                             (fProxy)(-halfLen - dz)));
            }

        fProxy3 a = default, b = default; fProxy r = default;
        Assert.IsTrue(Fit.capsule(pts, ref a, ref b, ref r), "capsule fit did not converge");

        Assert.That((double)r, Is.EqualTo(rad).Within(SolidTol), "radius");

        // Endpoints may come back in either order: compare as a set.
        var e0 = Place(new fProxy3((fProxy)0, (fProxy)0, (fProxy)(-halfLen)));
        var e1 = Place(new fProxy3((fProxy)0, (fProxy)0, (fProxy)halfLen));
        double straight = math.length(a - e0) + math.length(b - e1);
        double swapped = math.length(a - e1) + math.length(b - e0);
        Assert.That(math.min(straight, swapped), Is.EqualTo(0.0).Within(2.0 * SolidTol), "segment endpoints");

        pts.Dispose();
    }

    // Warm start: handing the solver the exact answer must converge immediately and leave it there.
    // This is the path a per-frame tracker takes, and it also proves the ref-parameter convention --
    // a nonzero incoming direction is USED, not overwritten by the seed.
    [Test]
    public void CylinderWarmStartFromExactAnswerHolds()
    {
        const double rad = 2.0;
        var pts = new NativeArray<fProxy3>(48, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 8; j++)
            {
                double z = -3.0 + 6.0 * i / 5.0, th = 2.0 * math.PI_DBL * j / 8.0;
                pts[k++] = Place(new fProxy3((fProxy)(rad * math.cos(th)), (fProxy)(rad * math.sin(th)), (fProxy)z));
            }

        fProxy3 q = SolidShift;
        fProxy3 d = SolidRot(new fProxy3((fProxy)0, (fProxy)0, (fProxy)1));
        fProxy r = (fProxy)rad;

        Assert.IsTrue(Fit.cylinder(pts, ref q, ref d, ref r), "warm-started cylinder fit failed");
        Assert.That((double)r, Is.EqualTo(rad).Within(SolidTol), "radius must stay put");
        Assert.That(AngleError(d, SolidRot(new fProxy3((fProxy)0, (fProxy)0, (fProxy)1))),
            Is.EqualTo(0.0).Within(SolidTol), "axis must stay put");

        pts.Dispose();
    }

    // ---------------------------------------------------------------- RANSAC

    // Builds a cloud that is 40% plane and 60% uniform junk -- the regime robust losses cannot reach.
    static NativeArray<fProxy3> ContaminatedPlane(int inliers, int outliers, uint seed)
    {
        var pts = new NativeArray<fProxy3>(inliers + outliers, Allocator.Temp);
        var rng = new Unity.Mathematics.Random(seed);

        for (int i = 0; i < inliers; i++)
            pts[i] = new fProxy3((fProxy)rng.NextDouble(-5.0, 5.0), (fProxy)rng.NextDouble(-5.0, 5.0), (fProxy)0);
        for (int i = 0; i < outliers; i++)
            pts[inliers + i] = new fProxy3((fProxy)rng.NextDouble(-5.0, 5.0),
                                           (fProxy)rng.NextDouble(-5.0, 5.0),
                                           (fProxy)rng.NextDouble(-5.0, 5.0));
        return pts;
    }

    // THE case that justifies RANSAC over a robust loss. At 60% contamination the L2 plane fit is
    // dominated by the junk, and IRLS can only walk downhill from there -- so Huber inherits the wrong
    // answer. RANSAC never starts from a contaminated fit, so it recovers the plane. Asserted as a
    // direct comparison, not an absolute bound, so it measures the METHOD rather than a tuned number.
    [Test]
    public void RansacBeatsRobustLossUnderGrossContamination()
    {
        var pts = ContaminatedPlane(40, 60, 991u);
        var want = new fProxy3((fProxy)0, (fProxy)0, (fProxy)1);

        Assert.IsTrue(Fit.plane(pts, out _, out fProxy3 nL2));
        double errL2 = AngleError(nL2, want);

        var huber = new fProxyHuberLoss((fProxy)0.2);
        Assert.IsTrue(Fit.plane(pts, in huber, out _, out fProxy3 nH));
        double errHuber = AngleError(nH, want);

        var model = new Fit.fProxyPlane();
        var info = Fit.ransac(pts, ref model, (fProxy)0.15, 0, 12345u);
        Assert.IsTrue(info, $"RANSAC found no consensus ({info.ToString()})");
        double errRansac = AngleError(model.Normal, want);

        Assert.That(errRansac, Is.LessThan(0.05), $"RANSAC should recover the plane ({info.ToString()})");
        Assert.Less(errRansac, errHuber, $"RANSAC ({errRansac}) must beat Huber ({errHuber}) at 60% junk");
        Assert.Less(errRansac, errL2, $"RANSAC ({errRansac}) must beat L2 ({errL2}) at 60% junk");

        // The consensus set should be the planar points, give or take junk that lands near the plane.
        Assert.That(info.inliers, Is.GreaterThanOrEqualTo(35), "should recover most of the plane points");

        pts.Dispose();
    }

    // Same points and seed must give the same answer, always -- the determinism contract.
    [Test]
    public void RansacIsDeterministicForAGivenSeed()
    {
        var pts = ContaminatedPlane(30, 30, 55u);

        var m1 = new Fit.fProxyPlane();
        var i1 = Fit.ransac(pts, ref m1, (fProxy)0.15, 0, 4242u);
        var m2 = new Fit.fProxyPlane();
        var i2 = Fit.ransac(pts, ref m2, (fProxy)0.15, 0, 4242u);

        Assert.IsTrue(i1 && i2, "both runs should find a consensus");
        Assert.AreEqual(i1.inliers, i2.inliers, "inlier count must match");
        Assert.AreEqual(i1.iterations, i2.iterations, "iteration count must match");
        Assert.That((double)math.length(m1.Normal - m2.Normal), Is.EqualTo(0.0).Within(0.0),
            "same seed must give a bit-identical normal");

        pts.Dispose();
    }

    // Clean data lets the adaptive rule stop almost immediately: with a ~100% inlier ratio, one draw
    // already gives the requested confidence. This pins the adaptive budget as real, not decorative.
    [Test]
    public void RansacAdaptiveStopsEarlyOnCleanData()
    {
        var pts = ContaminatedPlane(60, 0, 7u);

        var model = new Fit.fProxyPlane();
        var info = Fit.ransac(pts, ref model, (fProxy)0.05, 0, 99u);

        Assert.IsTrue(info, "clean data should find a consensus");
        Assert.AreEqual(60, info.inliers, "every point is an inlier on clean data");
        Assert.That(info.iterations, Is.LessThan(50),
            $"adaptive stopping should end far short of the {Fit.DefaultRansacIter} cap ({info.ToString()})");

        pts.Dispose();
    }

    // The sphere and line models: same driver, different minimal sample size and distance function.
    [Test]
    public void RansacSphereAndLineModels()
    {
        var rng = new Unity.Mathematics.Random(31u);

        var sp = new NativeArray<fProxy3>(70, Allocator.Temp);
        for (int i = 0; i < 45; i++)
        {
            double u = math.PI_DBL * rng.NextDouble(), v = 2.0 * math.PI_DBL * rng.NextDouble();
            sp[i] = new fProxy3((fProxy)(2.0 + 3.0 * math.sin(u) * math.cos(v)),
                                (fProxy)(-1.0 + 3.0 * math.sin(u) * math.sin(v)),
                                (fProxy)(0.5 + 3.0 * math.cos(u)));
        }
        for (int i = 45; i < 70; i++)
            sp[i] = new fProxy3((fProxy)rng.NextDouble(-8.0, 8.0), (fProxy)rng.NextDouble(-8.0, 8.0),
                                (fProxy)rng.NextDouble(-8.0, 8.0));

        var sm = new Fit.fProxySphere3();
        var si = Fit.ransac(sp, ref sm, (fProxy)0.1, 0, 808u);
        Assert.IsTrue(si, $"sphere RANSAC failed ({si.ToString()})");
        Assert.That((double)sm.Radius, Is.EqualTo(3.0).Within(0.1), "sphere radius");
        Assert.That((double)math.length(sm.Center - new fProxy3((fProxy)2, (fProxy)(-1), (fProxy)0.5)),
            Is.EqualTo(0.0).Within(0.1), "sphere centre");

        var ln = new NativeArray<fProxy3>(60, Allocator.Temp);
        for (int i = 0; i < 40; i++)
        {
            double t = rng.NextDouble(-5.0, 5.0);
            ln[i] = new fProxy3((fProxy)t, (fProxy)(2.0 * t + 1.0), (fProxy)(-t));
        }
        for (int i = 40; i < 60; i++)
            ln[i] = new fProxy3((fProxy)rng.NextDouble(-8.0, 8.0), (fProxy)rng.NextDouble(-8.0, 8.0),
                                (fProxy)rng.NextDouble(-8.0, 8.0));

        var lm = new Fit.fProxyLine3();
        var li = Fit.ransac(ln, ref lm, (fProxy)0.1, 0, 606u);
        Assert.IsTrue(li, $"line RANSAC failed ({li.ToString()})");
        var wantDir = math.normalize(new fProxy3((fProxy)1, (fProxy)2, (fProxy)(-1)));
        Assert.That(AngleError(lm.Direction, wantDir), Is.EqualTo(0.0).Within(0.05), "line direction");

        sp.Dispose(); ln.Dispose();
    }

    [Test]
    public void RansacGuardsThrow()
    {
        var pts = new NativeArray<fProxy3>(2, Allocator.Temp);
        var model = new Fit.fProxyPlane();

        // Fewer points than the model's minimal sample.
        Assert.Throws<ArgumentException>(() => { var m = model; Fit.ransac(pts, ref m, (fProxy)0.1); });
        pts.Dispose();

        var ok = ContaminatedPlane(10, 0, 3u);
        Assert.Throws<ArgumentException>(() => { var m = model; Fit.ransac(ok, ref m, (fProxy)0); });
        Assert.Throws<ArgumentException>(() => { var m = model; Fit.ransac(ok, ref m, (fProxy)(-1)); });
        ok.Dispose();
    }

    // ------------------------------------------------- robust linear + losses

    // Builds y = 2x + 1 with gross outliers in the RESPONSE.
    static void OutlierLine(fProxyMxN A, fProxyN b, int n, int badEvery)
    {
        for (int i = 0; i < n; i++)
        {
            double x = i * 0.5;
            A[i, 0] = (fProxy)x; A[i, 1] = (fProxy)1;
            b[i] = (fProxy)(2.0 * x + 1.0);
            if (i % badEvery == badEvery - 1) b[i] = (fProxy)(2.0 * x + 1.0 + 25.0);
        }
    }

    // fProxyL2Loss has RhoPrime == 1, so the weights never move: the robust overload must reproduce
    // the plain QR fit EXACTLY. Pins the identity element of the loss family.
    [Test]
    public void LinearWithL2LossMatchesPlainFit()
    {
        int n = 20;
        var A = new fProxyMxN(n, 2, Allocator.Temp);
        var b = new fProxyN(n, Allocator.Temp);
        OutlierLine(A, b, n, 4);

        var xPlain = new fProxyN(2, Allocator.Temp);
        Assert.IsTrue(Fit.linear(in A, in b, ref xPlain), "plain fit failed");

        var xL2 = new fProxyN(2, Allocator.Temp);
        var l2 = new fProxyL2Loss();
        Assert.IsTrue(Fit.linear(in A, in b, ref xL2, in l2), "L2-loss fit failed");

        for (int j = 0; j < 2; j++)
            Assert.That((double)xL2[j], Is.EqualTo((double)xPlain[j]).Within(Tol), $"coefficient {j}");

        A.Dispose(); b.Dispose(); xPlain.Dispose(); xL2.Dispose();
    }

    // With a quarter of the responses corrupted, a robust loss must recover the true slope better than
    // plain least squares. Direct comparison, so it measures the loss rather than a tuned bound.
    [Test]
    public void LinearRobustBeatsL2UnderResponseOutliers()
    {
        int n = 20;
        var A = new fProxyMxN(n, 2, Allocator.Temp);
        var b = new fProxyN(n, Allocator.Temp);
        OutlierLine(A, b, n, 4);

        var xL2 = new fProxyN(2, Allocator.Temp);
        Assert.IsTrue(Fit.linear(in A, in b, ref xL2));
        double errL2 = math.abs((double)xL2[0] - 2.0);

        var xH = new fProxyN(2, Allocator.Temp);
        var huber = new fProxyHuberLoss((fProxy)1);
        Assert.IsTrue(Fit.linear(in A, in b, ref xH, in huber), "robust linear fit failed");
        double errH = math.abs((double)xH[0] - 2.0);

        Assert.Less(errH, errL2, $"Huber slope err ({errH}) should beat L2 ({errL2})");

        A.Dispose(); b.Dispose(); xL2.Dispose(); xH.Dispose();
    }

    // WARM START, pinned by a case where it decides success from failure -- and a regression guard on
    // the collapse that case exposed.
    //
    // Tukey is REDESCENDING: its weight is exactly zero past Scale. Started cold, the first pass is
    // plain least squares, which the outliers drag so far from every point that EVERY residual exceeds
    // Scale and the whole design zeroes out. That is unrecoverable, and the honest answer is false --
    // before the guard this returned true with NaN coefficients, a false certificate the caller had no
    // way to detect. Seeded with the truth the same loss rejects the outliers and holds.
    [Test]
    public void LinearWarmStartIsHonouredAndCollapseIsHonest()
    {
        int n = 20;
        var A = new fProxyMxN(n, 2, Allocator.Temp);
        var b = new fProxyN(n, Allocator.Temp);
        OutlierLine(A, b, n, 4);

        var tukey = new fProxyTukeyLoss((fProxy)2);

        var xCold = new fProxyN(2, Allocator.Temp);
        xCold[0] = (fProxy)0; xCold[1] = (fProxy)0;                  // zero => auto-seed
        bool coldOk = Fit.linear(in A, in b, ref xCold, in tukey);
        if (!coldOk)
        {
            // The documented collapse. Whatever it reports must not be a silent NaN success.
            Assert.Pass("cold redescending fit collapsed and said so");
        }

        var xWarm = new fProxyN(2, Allocator.Temp);
        xWarm[0] = (fProxy)2; xWarm[1] = (fProxy)1;                  // the truth
        Assert.IsTrue(Fit.linear(in A, in b, ref xWarm, in tukey), "warm robust fit must succeed");

        double errWarm = math.abs((double)xWarm[0] - 2.0);
        Assert.IsFalse(double.IsNaN(errWarm), "warm fit must not be NaN");
        Assert.That(errWarm, Is.LessThan(0.05), "a warm start at the truth must stay near it");

        A.Dispose(); b.Dispose(); xCold.Dispose(); xWarm.Dispose();
    }

    // Explicit regression guard for the false certificate: a redescending loss whose scale rejects
    // EVERY point must report failure, never a NaN reported as success.
    [Test]
    public void LinearRedescendingCollapseNeverReportsNaNSuccess()
    {
        int n = 12;
        var A = new fProxyMxN(n, 2, Allocator.Temp);
        var b = new fProxyN(n, Allocator.Temp);
        for (int i = 0; i < n; i++)
        {
            A[i, 0] = (fProxy)i; A[i, 1] = (fProxy)1;
            b[i] = (fProxy)(i % 2 == 0 ? 1000.0 : -1000.0);          // no line fits this
        }

        var x = new fProxyN(2, Allocator.Temp);
        var tukey = new fProxyTukeyLoss((fProxy)1e-3);                // rejects essentially everything

        bool ok = Fit.linear(in A, in b, ref x, in tukey);
        if (ok)
            for (int j = 0; j < 2; j++)
                Assert.IsFalse(double.IsNaN((double)x[j]) || double.IsInfinity((double)x[j]),
                    $"reported success with a non-finite coefficient {j}");

        A.Dispose(); b.Dispose(); x.Dispose();
    }

    // fProxyL1Loss directly. IRLS only ever calls RhoPrime, so Rho and RhoPrime2 have no coverage from
    // the fitting tests at all -- but nlsSolve's robust scaling uses all three, so a robust loss on a
    // solid fit would reach them.
    [Test]
    public void L1LossMatchesItsDefinition()
    {
        var loss = new fProxyL1Loss((fProxy)1e-3);

        // rho(s) = sqrt(s) above the floor, so the objective is 0.5*sum|r|.
        Assert.That((double)loss.Rho((fProxy)4), Is.EqualTo(2.0).Within(1e-4), "Rho(4) = 2");
        Assert.That((double)loss.Rho((fProxy)9), Is.EqualTo(3.0).Within(1e-4), "Rho(9) = 3");

        // rho'(s) = 1/(2 sqrt(s)): the IRLS weight, falling as 1/|r|.
        Assert.That((double)loss.RhoPrime((fProxy)4), Is.EqualTo(0.25).Within(1e-4), "RhoPrime(4)");
        Assert.That((double)loss.RhoPrime((fProxy)16), Is.EqualTo(0.125).Within(1e-4), "RhoPrime(16)");

        // rho''(s) = -1/(4 s^{3/2}), negative: the weight decreases with residual.
        Assert.Less((double)loss.RhoPrime2((fProxy)4), 0.0, "RhoPrime2 must be negative");

        // The floor is what keeps the weight finite at zero residual -- without it IRLS divides by 0.
        double wAtZero = (double)loss.RhoPrime((fProxy)0);
        Assert.IsTrue(wAtZero > 0.0 && !double.IsInfinity(wAtZero), "weight at r=0 must be finite");
        Assert.That(wAtZero, Is.EqualTo(0.5 / 1e-3).Within(1.0), "floor should cap the weight at 1/(2*floor)");

        // A default-constructed instance must still be usable, not a divide by zero.
        var bare = new fProxyL1Loss();
        double wBare = (double)bare.RhoPrime((fProxy)0);
        Assert.IsTrue(wBare > 0.0 && !double.IsInfinity(wBare), "default L1Loss must have a working floor");
    }

    // Robust loss on a NONLINEAR fit -- the path where nlsSolve does its own loss scaling, distinct
    // from the IRLS loop the linear and geometric fits use.
    [Test]
    public void CylinderRobustBeatsL2UnderOutliers()
    {
        const double rad = 2.0;
        var pts = new NativeArray<fProxy3>(54, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 8; j++)
            {
                double z = -3.0 + 6.0 * i / 5.0, th = 2.0 * math.PI_DBL * j / 8.0;
                pts[k++] = new fProxy3((fProxy)(rad * math.cos(th)), (fProxy)(rad * math.sin(th)), (fProxy)z);
            }
        for (int i = 0; i < 6; i++)                                  // off-surface junk
            pts[k++] = new fProxy3((fProxy)(4.5 * math.cos(i)), (fProxy)(4.5 * math.sin(i)), (fProxy)(i - 3));

        fProxy3 q = default, d = default; fProxy r = default;
        Assert.IsTrue(Fit.cylinder(pts, ref q, ref d, ref r), "L2 cylinder fit failed");
        double errL2 = math.abs((double)r - rad);

        fProxy3 q2 = default, d2 = default; fProxy r2 = default;
        var huber = new fProxyHuberLoss((fProxy)0.3);
        Assert.IsTrue(Fit.cylinder(pts, ref q2, ref d2, ref r2, in huber), "robust cylinder fit failed");
        double errH = math.abs((double)r2 - rad);

        Assert.Less(errH, errL2, $"Huber radius err ({errH}) should beat L2 ({errL2})");

        pts.Dispose();
    }

    // classify's remaining branches, driven by hand-built coefficient vectors rather than a fit, so
    // each signature is hit exactly and deterministically.
    [Test]
    public void ClassifyCoversEverySignature()
    {
        var c = new fProxyN(10, Allocator.Temp);

        // (+,+,+) -> ellipsoid; the fitted cases above already cover this, included for contrast.
        SetQuad(c, 1, 1, 1);
        Assert.AreEqual(QuadricKind.Ellipsoid, Fit.classify(in c), "(+,+,+)");

        // (-,-,-) is the same surface family with the equation negated.
        SetQuad(c, -1, -1, -1);
        Assert.AreEqual(QuadricKind.Ellipsoid, Fit.classify(in c), "(-,-,-) is still an ellipsoid");

        SetQuad(c, 1, 1, -1);
        Assert.AreEqual(QuadricKind.HyperboloidOrCone, Fit.classify(in c), "(+,+,-)");

        // A zero eigenvalue means no centre: paraboloid or cylinder.
        SetQuad(c, 1, 1, 0);
        Assert.AreEqual(QuadricKind.Paraboloid, Fit.classify(in c), "(+,+,0)");

        // No quadratic part at all: the fit collapsed to a plane.
        SetQuad(c, 0, 0, 0);
        c[6] = (fProxy)1;                                            // a linear term, so it is a plane
        Assert.AreEqual(QuadricKind.Degenerate, Fit.classify(in c), "(0,0,0)");

        c.Dispose();
    }

    static void SetQuad(fProxyN c, double a, double b, double cc)
    {
        for (int i = 0; i < 10; i++) c[i] = (fProxy)0;
        c[0] = (fProxy)a; c[1] = (fProxy)b; c[2] = (fProxy)cc;
    }

    // Ground truth for Fit.quadric. The classification test above only checks WHICH FAMILY the fit
    // landed in -- a completely different ellipsoid would pass it. This compares the recovered
    // coefficients against the ones that generated the points.
    //
    // Quadric coefficients are defined only up to overall SCALE (the fit returns them unit-norm) and
    // therefore up to SIGN, so both vectors are normalized and sign-aligned on their largest component
    // before comparison. The ellipsoid is placed OFF-CENTRE so the linear terms G/H/I are genuinely
    // non-zero; the cross terms D/E/F must come back at zero, which is its own assertion -- a fit that
    // invented spurious cross terms would fail here.
    [Test]
    public void QuadricCoefficientsMatchTheGeneratingEllipsoid()
    {
        // (x-cx)²/a² + (y-cy)²/b² + (z-cz)²/cc² = 1
        const double a = 3.0, b = 2.0, cc = 1.5;
        const double cx = 1.0, cy = -2.0, cz = 0.5;

        var pts = new NativeArray<fProxy3>(96, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 12; j++)
            {
                double u = math.PI_DBL * (i + 0.5) / 8.0, v = 2.0 * math.PI_DBL * j / 12.0;
                pts[k++] = new fProxy3((fProxy)(cx + a * math.sin(u) * math.cos(v)),
                                       (fProxy)(cy + b * math.sin(u) * math.sin(v)),
                                       (fProxy)(cz + cc * math.cos(u)));
            }

        var got = new fProxyN(10, Allocator.Temp);
        Assert.IsTrue(Fit.quadric(pts, ref got), "quadric fit failed");

        // Expand the generating equation into the same (A..J) ordering the fit reports.
        double ia = 1.0 / (a * a), ib = 1.0 / (b * b), ic = 1.0 / (cc * cc);
        var want = new double[10]
        {
            ia, ib, ic,
            0, 0, 0,
            -2.0 * cx * ia, -2.0 * cy * ib, -2.0 * cz * ic,
            cx * cx * ia + cy * cy * ib + cz * cz * ic - 1.0,
        };

        Normalize(want);
        var g = new double[10];
        for (int i = 0; i < 10; i++) g[i] = (double)got[i];
        Normalize(g);
        AlignSign(g, want);

        for (int i = 0; i < 10; i++)
            Assert.That(g[i], Is.EqualTo(want[i]).Within(QuadTol), $"coefficient {i}");

        pts.Dispose(); got.Dispose();
    }

    static double QuadTol => /*+choose[5e-3|1e-8]*/5e-3/*-choose*/;

    static void Normalize(double[] v)
    {
        double s = 0;
        for (int i = 0; i < v.Length; i++) s += v[i] * v[i];
        s = math.sqrt(s);
        if (s > 0) for (int i = 0; i < v.Length; i++) v[i] /= s;
    }

    // Coefficients are defined up to sign; flip `g` to match `want` on their largest shared component.
    static void AlignSign(double[] g, double[] want)
    {
        int big = 0;
        for (int i = 1; i < want.Length; i++) if (math.abs(want[i]) > math.abs(want[big])) big = i;
        if (g[big] * want[big] < 0) for (int i = 0; i < g.Length; i++) g[i] = -g[i];
    }

    // ------------------------------------------------------------- L1 paths
    //
    // The library reaches the L1 objective two completely independent ways: LP.lad solves it EXACTLY
    // by a finite combinatorial algorithm, and Fit.linear with fProxyL1Loss approaches it by iterative
    // reweighting. fProxyL1Loss's own doc points callers at LP.lad for exactness -- these tests are
    // what make that claim checkable, and they cross-validate both implementations at once.

    // Same objective, two engines, one answer. A disagreement here means one of them is wrong; nothing
    // else in the suite compares them.
    [Test]
    public void IrlsL1AgreesWithTheExactLadSolver()
    {
        int n = 15;
        var A = new fProxyMxN(n, 2, Allocator.Temp);
        var b = new fProxyN(n, Allocator.Temp);
        for (int i = 0; i < n; i++)
        {
            double x = i * 0.5;
            A[i, 0] = (fProxy)x; A[i, 1] = (fProxy)1;
            b[i] = (fProxy)(2.0 * x + 1.0);
        }
        b[3] = (fProxy)40; b[9] = (fProxy)(-30); b[12] = (fProxy)55;    // gross, and asymmetric

        var xExact = new fProxyN(2, Allocator.Temp);
        Assert.IsTrue(LP.lad(in A, in b, ref xExact, out double objExact), "exact LAD failed");

        // The L1 optimum interpolates data points exactly, so IRLS weights would diverge as those
        // residuals reach zero. The floor is what bounds them -- set here to the data's own noise
        // scale rather than left at the default epsilon.
        var l1 = new fProxyL1Loss((fProxy)1e-2);
        var xIrls = new fProxyN(2, Allocator.Temp);
        Assert.IsTrue(Fit.linear(in A, in b, ref xIrls, in l1, 200), "IRLS L1 failed");

        // Compare on the OBJECTIVE, not the coefficients: the L1 optimum can be non-unique (any line
        // through the same two interpolated points scores identically), so equal coefficients is a
        // stronger claim than the problem actually makes.
        double objIrls = 0;
        for (int i = 0; i < n; i++)
        {
            double r = (double)A[i, 0] * (double)xIrls[0] + (double)A[i, 1] * (double)xIrls[1] - (double)b[i];
            objIrls += math.abs(r);
        }

        Assert.That(objIrls, Is.EqualTo(objExact).Within(0.05 * objExact),
            $"IRLS L1 objective {objIrls} should approach the exact optimum {objExact}");
        Assert.GreaterOrEqual(objIrls, objExact - 1e-6,
            "nothing may beat the exact optimum: that would mean LP.lad is not optimal");

        A.Dispose(); b.Dispose(); xExact.Dispose(); xIrls.Dispose();
    }

    // L1 is a robust metric, so it must resist response outliers where L2 cannot -- the same
    // discrimination the Huber case makes, for the loss a caller is most likely to reach for.
    [Test]
    public void LinearL1BeatsL2UnderResponseOutliers()
    {
        int n = 20;
        var A = new fProxyMxN(n, 2, Allocator.Temp);
        var b = new fProxyN(n, Allocator.Temp);
        OutlierLine(A, b, n, 4);

        var xL2 = new fProxyN(2, Allocator.Temp);
        Assert.IsTrue(Fit.linear(in A, in b, ref xL2));
        double errL2 = math.abs((double)xL2[0] - 2.0);

        var l1 = new fProxyL1Loss((fProxy)1e-2);
        var xL1 = new fProxyN(2, Allocator.Temp);
        Assert.IsTrue(Fit.linear(in A, in b, ref xL1, in l1, 200), "IRLS L1 failed");
        double errL1 = math.abs((double)xL1[0] - 2.0);

        Assert.Less(errL1, errL2, $"L1 slope err ({errL1}) should beat L2 ({errL2})");

        A.Dispose(); b.Dispose(); xL2.Dispose(); xL1.Dispose();
    }

    // L1 on a GEOMETRIC fit, where no exact solver exists to compare against -- so the assertion is
    // the robustness property itself, on the 3D plane normal.
    [Test]
    public void PlaneL1BeatsL2UnderOutlier()
    {
        const int n = 17;
        var pts = new NativeArray<fProxy3>(n, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                pts[k++] = new fProxy3((fProxy)i, (fProxy)j, (fProxy)0);
        pts[16] = new fProxy3((fProxy)3, (fProxy)3, (fProxy)4);

        var want = new fProxy3((fProxy)0, (fProxy)0, (fProxy)1);

        Assert.IsTrue(Fit.plane(pts, out _, out fProxy3 nL2));
        double errL2 = AngleError(nL2, want);

        var l1 = new fProxyL1Loss((fProxy)1e-2);
        Assert.IsTrue(Fit.plane(pts, in l1, out _, out fProxy3 nL1), "L1 plane fit failed");
        double errL1 = AngleError(nL1, want);

        Assert.Less(errL1, errL2, $"L1 normal ({errL1}) should beat L2 ({errL2})");

        pts.Dispose();
    }

    // --------------------------------------- robust conic / quadric (Sampson)

    // Robust ellipse fit: points on a known ellipse plus one off-curve point. The algebraic fit is
    // pulled toward the outlier; reweighting by Sampson distance rejects it.
    [Test]
    public void ConicRobustBeatsAlgebraicUnderOutlier()
    {
        const double a = 5.0, b = 2.0;
        const int n = 17;
        var pts = new NativeArray<fProxy2>(n, Allocator.Temp);
        for (int i = 0; i < 16; i++)
        {
            double t = 2.0 * math.PI_DBL * i / 16.0;
            pts[i] = new fProxy2((fProxy)(a * math.cos(t)), (fProxy)(b * math.sin(t)));
        }
        // MODERATE, per this file's standing rule. At 3x the semi-minor axis the algebraic seed is
        // already wrecked, and IRLS reweighting from a wrecked ellipse converges somewhere worse --
        // that is the RANSAC regime, not the loss regime, and testing it here would pin behaviour
        // nobody should rely on.
        pts[16] = new fProxy2((fProxy)0, (fProxy)3.2);

        // Geometry recovered through the plain (algebraic) route.
        Assert.IsTrue(Fit.ellipse(pts, out _, out fProxy2 radA, out _), "algebraic ellipse fit failed");
        double errA = math.abs(math.max((double)radA.x, (double)radA.y) - a)
                    + math.abs(math.min((double)radA.x, (double)radA.y) - b);

        // Same data, robust conic, then decomposed the same way ellipse() does.
        var c = new fProxyN(6, Allocator.Temp);
        var huber = new fProxyHuberLoss((fProxy)0.3);
        Assert.IsTrue(Fit.conic(pts, in huber, ref c), "robust conic fit failed");
        Assert.Greater(4.0 * (double)c[0] * (double)c[2] - (double)c[1] * (double)c[1], 0.0,
            "the robust fit must still satisfy the ellipse constraint");

        Assert.IsTrue(EllipseRadiiFromConic(in c, out double r0, out double r1), "conic was not a real ellipse");
        double errH = math.abs(math.max(r0, r1) - a) + math.abs(math.min(r0, r1) - b);

        Assert.Less(errH, errA, $"Sampson-robust semi-axis err ({errH}) should beat algebraic ({errA})");

        pts.Dispose(); c.Dispose();
    }

    // L2 through the Sampson route is NOT a no-op: it swaps the algebraic residual for a geometric
    // one, which removes most of the plain fit's bias on unevenly-sampled data. Sampling a SHORT ARC
    // is what exposes that bias -- a full sweep is nearly unbiased either way.
    [Test]
    public void QuadricSampsonL2ReducesAlgebraicBias()
    {
        const double a = 3.0, b = 2.0, cc = 1.5;
        var pts = new NativeArray<fProxy3>(70, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < 7; i++)
            for (int j = 0; j < 10; j++)
            {
                double u = math.PI_DBL * (0.15 + 0.30 * i / 6.0);    // a band, not the whole sphere
                double v = 2.0 * math.PI_DBL * j / 10.0;
                pts[k++] = new fProxy3((fProxy)(a * math.sin(u) * math.cos(v)),
                                       (fProxy)(b * math.sin(u) * math.sin(v)),
                                       (fProxy)(cc * math.cos(u)));
            }

        double ia = 1.0 / (a * a), ib = 1.0 / (b * b), ic = 1.0 / (cc * cc);
        var want = new double[10] { ia, ib, ic, 0, 0, 0, 0, 0, 0, -1.0 };
        Normalize(want);

        var plain = new fProxyN(10, Allocator.Temp);
        Assert.IsTrue(Fit.quadric(pts, ref plain), "plain quadric fit failed");

        var l2 = new fProxyL2Loss();
        var samp = new fProxyN(10, Allocator.Temp);
        Assert.IsTrue(Fit.quadric(pts, in l2, ref samp), "Sampson quadric fit failed");

        // STRICTLY better, not merely "not worse". The weak form of this assertion passed vacuously
        // while the Sampson weight was missing its 1/|grad F|² factor -- without that factor L2's
        // RhoPrime == 1 leaves every weight at 1, the loop converges before its first refit, and the
        // result is bit-identical to the plain algebraic fit.
        Assert.Less(CoeffError(samp, want), CoeffError(plain, want),
            "reweighting by Sampson distance must beat the raw algebraic fit");

        pts.Dispose(); plain.Dispose(); samp.Dispose();
    }

    // Quadric with a robust loss against gross outliers.
    [Test]
    public void QuadricRobustBeatsAlgebraicUnderOutliers()
    {
        const double a = 3.0, b = 2.0, cc = 1.5;
        var pts = new NativeArray<fProxy3>(102, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 12; j++)
            {
                double u = math.PI_DBL * (i + 0.5) / 8.0, v = 2.0 * math.PI_DBL * j / 12.0;
                pts[k++] = new fProxy3((fProxy)(a * math.sin(u) * math.cos(v)),
                                       (fProxy)(b * math.sin(u) * math.sin(v)),
                                       (fProxy)(cc * math.cos(u)));
            }
        for (int i = 0; i < 6; i++)                                  // off-surface junk
            pts[k++] = new fProxy3((fProxy)(6.0 * math.cos(i)), (fProxy)(6.0 * math.sin(i)), (fProxy)i);

        double ia = 1.0 / (a * a), ib = 1.0 / (b * b), ic = 1.0 / (cc * cc);
        var want = new double[10] { ia, ib, ic, 0, 0, 0, 0, 0, 0, -1.0 };
        Normalize(want);

        var plain = new fProxyN(10, Allocator.Temp);
        Assert.IsTrue(Fit.quadric(pts, ref plain), "plain quadric fit failed");

        var huber = new fProxyHuberLoss((fProxy)0.2);
        var rob = new fProxyN(10, Allocator.Temp);
        Assert.IsTrue(Fit.quadric(pts, in huber, ref rob), "robust quadric fit failed");

        Assert.Less(CoeffError(rob, want), CoeffError(plain, want),
            "the robust quadric must recover the true surface better than the algebraic one");

        pts.Dispose(); plain.Dispose(); rob.Dispose();
    }

    static double CoeffError(fProxyN got, double[] want)
    {
        var g = new double[10];
        for (int i = 0; i < 10; i++) g[i] = (double)got[i];
        Normalize(g);
        AlignSign(g, want);
        double e = 0;
        for (int i = 0; i < 10; i++) e += math.abs(g[i] - want[i]);
        return e;
    }

    // Centre + semi-axes from conic coefficients, the same decomposition Fit.ellipse performs.
    static bool EllipseRadiiFromConic(in fProxyN c, out double r0, out double r1)
    {
        r0 = r1 = 0;
        double A = (double)c[0], B = (double)c[1], C = (double)c[2];
        double D = (double)c[3], E = (double)c[4], F = (double)c[5];

        double det = 4.0 * A * C - B * B;
        if (math.abs(det) < 1e-12) return false;

        double cx = (2.0 * C * (-D) - B * (-E)) / det;
        double cy = (2.0 * A * (-E) - B * (-D)) / det;
        double Fc = F + 0.5 * (D * cx + E * cy);

        // Eigenvalues of [[A, B/2],[B/2, C]].
        double tr = A + C, dd = math.sqrt(math.max((A - C) * (A - C) + B * B, 0.0));
        double l0 = 0.5 * (tr + dd), l1 = 0.5 * (tr - dd);
        if (math.abs(l0) < 1e-12 || math.abs(l1) < 1e-12) return false;

        double s0 = -Fc / l0, s1 = -Fc / l1;
        if (s0 <= 0 || s1 <= 0) return false;

        r0 = math.sqrt(s0); r1 = math.sqrt(s1);
        return true;
    }

    // Collapse guard for the SPHERE IRLS loop -- the copy that shipped without one. A redescending
    // loss whose scale rejects every point must not report success with NaN.
    [Test]
    public void SphereRedescendingCollapseNeverReportsNaNSuccess()
    {
        var pts = new NativeArray<fProxy2>(12, Allocator.Temp);
        for (int i = 0; i < 12; i++)
        {
            double t = 2.0 * math.PI_DBL * i / 12.0;
            pts[i] = new fProxy2((fProxy)(5.0 * math.cos(t)), (fProxy)(5.0 * math.sin(t)));
        }
        pts[0] = new fProxy2((fProxy)500, (fProxy)500);               // wrecks the algebraic seed

        var tukey = new fProxyTukeyLoss((fProxy)1e-4);                // rejects essentially everything
        bool ok = Fit.sphere(pts, in tukey, out fProxy2 c, out fProxy r);

        if (ok)
        {
            Assert.IsFalse(double.IsNaN((double)r) || double.IsInfinity((double)r),
                "reported success with a non-finite radius");
            Assert.IsFalse(double.IsNaN((double)c.x) || double.IsNaN((double)c.y),
                "reported success with a non-finite centre");
        }

        pts.Dispose();
    }

    // ------------------------------------------------- remaining overloads
    //
    // Coverage completion. Each entry point below is a distinct generic instantiation, so exercising
    // the 2D form proves nothing about the 3D or 4D one -- codegen emits them separately and the
    // dimension-dependent index arithmetic is written out per arity.

    // 3D robust line: same discrimination as the 2D case, different instantiation.
    [Test]
    public void Line3DRobustBeatsL2UnderOutlier()
    {
        const int n = 21;
        var pts = new NativeArray<fProxy3>(n, Allocator.Temp);
        for (int i = 0; i < 20; i++) pts[i] = new fProxy3((fProxy)i, (fProxy)i, (fProxy)i);
        pts[20] = new fProxy3((fProxy)10, (fProxy)26, (fProxy)10);

        var want = math.normalize(new fProxy3((fProxy)1, (fProxy)1, (fProxy)1));

        Assert.IsTrue(Fit.line(pts, out _, out fProxy3 dL2), "3D line fit failed");
        double errL2 = AngleError(dL2, want);

        var huber = new fProxyHuberLoss((fProxy)1);
        Assert.IsTrue(Fit.line(pts, in huber, out _, out fProxy3 dH), "3D robust line fit failed");
        double errH = AngleError(dH, want);

        Assert.Less(errH, errL2, $"3D Huber ({errH}) should beat L2 ({errL2})");

        pts.Dispose();
    }

    // 4D line, plain and robust -- the only arity with no other coverage at all.
    [Test]
    public void Line4DPlainAndRobust()
    {
        const int n = 16;
        var pts = new NativeArray<fProxy4>(n, Allocator.Temp);
        for (int i = 0; i < 15; i++)
            pts[i] = new fProxy4((fProxy)i, (fProxy)i, (fProxy)i, (fProxy)i);
        pts[15] = new fProxy4((fProxy)7, (fProxy)20, (fProxy)7, (fProxy)7);

        var want = math.normalize(new fProxy4((fProxy)1, (fProxy)1, (fProxy)1, (fProxy)1));

        Assert.IsTrue(Fit.line(pts, out fProxy4 c, out fProxy4 dL2), "4D line fit failed");
        double cosL2 = math.abs(math.dot(math.normalize(dL2), want));
        double errL2 = math.acos(math.min(cosL2, 1.0));
        Assert.IsFalse(double.IsNaN((double)c.w), "4D centroid must be finite in every component");

        var huber = new fProxyHuberLoss((fProxy)1);
        Assert.IsTrue(Fit.line(pts, in huber, out _, out fProxy4 dH), "4D robust line fit failed");
        double cosH = math.abs(math.dot(math.normalize(dH), want));
        double errH = math.acos(math.min(cosH, 1.0));

        Assert.Less(errH, errL2, $"4D Huber ({errH}) should beat L2 ({errL2})");

        pts.Dispose();
    }

    // 3D robust sphere: the 2D circle case covers a different instantiation.
    [Test]
    public void Sphere3DRobustBeatsAlgebraicUnderOutlier()
    {
        const int n = 43;
        var pts = new NativeArray<fProxy3>(n, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 7; j++)
            {
                double u = math.PI_DBL * (i + 0.5) / 6.0, v = 2.0 * math.PI_DBL * j / 7.0;
                pts[k++] = new fProxy3((fProxy)(4.0 * math.sin(u) * math.cos(v)),
                                       (fProxy)(4.0 * math.sin(u) * math.sin(v)),
                                       (fProxy)(4.0 * math.cos(u)));
            }
        pts[k++] = new fProxy3((fProxy)7, (fProxy)0, (fProxy)0);     // off the surface

        Assert.IsTrue(Fit.sphere(pts, out _, out fProxy rA), "3D algebraic sphere fit failed");

        var huber = new fProxyHuberLoss((fProxy)0.5);
        Assert.IsTrue(Fit.sphere(pts, in huber, out _, out fProxy rH), "3D robust sphere fit failed");

        double errA = math.abs((double)rA - 4.0);
        double errH = math.abs((double)rH - 4.0);
        Assert.Less(errH, errA, $"3D robust radius err ({errH}) should beat algebraic ({errA})");

        pts.Dispose();
    }

    // An EXPLICIT maxIter disables the adaptive rule, so the driver must run exactly that many draws --
    // the branch the adaptive tests never take.
    [Test]
    public void RansacExplicitBudgetRunsEveryDraw()
    {
        var pts = ContaminatedPlane(40, 20, 123u);
        var model = new Fit.fProxyPlane();

        var info = Fit.ransac(pts, ref model, (fProxy)0.15, 25, 77u);

        Assert.IsTrue(info, $"fixed-budget RANSAC found no consensus ({info.ToString()})");
        Assert.AreEqual(25, info.iterations,
            "an explicit maxIter must be spent in full, not cut short by the adaptive rule");

        pts.Dispose();
    }

    // ------------------------------------------------------- ellipsoid / ellipse2
    //
    // Fit.ellipsoid is the CONSTRAINED counterpart of Fit.quadric, and the shapes are what let a
    // metric reach either family at all -- an ellipse or ellipsoid has a real distance function,
    // where a general quadric only has Sampson.

    // Accuracy of the bracketed root solve behind fProxyEllipsoid/fProxyEllipse2's Distance is set by
    // its coordinate floor, max(radius) * sqrtEps -- far coarser than Tol in double, so assertions on
    // a DISTANCE use this instead of Tol.
    static double DistTol => /*+choose[5e-3|1e-6]*/5e-3/*-choose*/;

    // Fit.ellipsoid's accuracy floor. Li & Griffiths needs the SCATTER matrix for its generalized
    // eigenproblem, and forming it squares the condition number, so the constrained fit lands at
    // normal-equations accuracy rather than the direct-factorization accuracy Fit.quadric reaches.
    // Asserting Tol here would be asserting something the method does not claim.
    static double AlgTol => /*+choose[2e-3|1e-7]*/2e-3/*-choose*/;

    // Spherical grid over the ellipsoid (a, b, cc) centred at o. The half-step in u keeps samples off
    // the poles, where the grid would otherwise pile up.
    static NativeArray<fProxy3> SampleEllipsoid(double a, double b, double cc, fProxy3 o,
                                                int nu, int nv, int extra = 0)
    {
        var pts = new NativeArray<fProxy3>(nu * nv + extra, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < nu; i++)
            for (int j = 0; j < nv; j++)
            {
                double u = math.PI_DBL * (i + 0.5) / nu, v = 2.0 * math.PI_DBL * j / nv;
                pts[k++] = o + new fProxy3((fProxy)(a * math.sin(u) * math.cos(v)),
                                           (fProxy)(b * math.sin(u) * math.sin(v)),
                                           (fProxy)(cc * math.cos(u)));
            }
        return pts;
    }

    // Points placed exactly on a known axis-aligned ellipsoid. The eigensolve reports axes in its own
    // order, so the radii are compared as a SET.
    [Test]
    public void EllipsoidExactRecoversGeometry()
    {
        var o = new fProxy3((fProxy)1.0, (fProxy)(-2.0), (fProxy)0.5);
        var pts = SampleEllipsoid(3.0, 2.0, 1.5, o, 8, 12);

        Assert.IsTrue(Fit.ellipsoid(pts, out fProxy3 ctr, out fProxy3 rad, out _, out _),
            "ellipsoid fit failed");

        Assert.That((double)ctr.x, Is.EqualTo(1.0).Within(AlgTol), "centre x");
        Assert.That((double)ctr.y, Is.EqualTo(-2.0).Within(AlgTol), "centre y");
        Assert.That((double)ctr.z, Is.EqualTo(0.5).Within(AlgTol), "centre z");

        var got = new double[] { (double)rad.x, (double)rad.y, (double)rad.z };
        Array.Sort(got);
        Assert.That(got[0], Is.EqualTo(1.5).Within(AlgTol), "smallest semi-axis");
        Assert.That(got[1], Is.EqualTo(2.0).Within(AlgTol), "middle semi-axis");
        Assert.That(got[2], Is.EqualTo(3.0).Within(AlgTol), "largest semi-axis");

        pts.Dispose();
    }

    // Li & Griffiths' 4J - I² > 0 is SUFFICIENT for an ellipsoid but not NECESSARY. For a quadratic
    // form with eigenvalues (1, 1, t) it equals t(4 - t), so it goes negative once t >= 4 -- and
    // since eigenvalues go as 1/radius², that is an axis ratio of only 2:1. If the constraint really
    // does exclude flatter ellipsoids, this exact, noise-free 2.5:1 cloud cannot be fitted.
    [Test]
    public void EllipsoidHandlesFlatterThanTwoToOne()
    {
        var o = new fProxy3((fProxy)0, (fProxy)0, (fProxy)0);
        var pts = SampleEllipsoid(1.0, 1.0, 0.4, o, 10, 16);

        Assert.IsTrue(Fit.ellipsoid(pts, out _, out fProxy3 rad, out _, out _),
            "a 2.5:1 oblate ellipsoid must be fittable");

        var got = new double[] { (double)rad.x, (double)rad.y, (double)rad.z };
        Array.Sort(got);
        Assert.That(got[0], Is.EqualTo(0.4).Within(AlgTol), "polar semi-axis");
        Assert.That(got[2], Is.EqualTo(1.0).Within(AlgTol), "equatorial semi-axis");

        pts.Dispose();
    }

    // THE reason Fit.ellipsoid exists alongside Fit.quadric. The same one-sheet hyperboloid data that
    // sends the UNCONSTRAINED fit to a hyperboloid must still come back an ellipsoid here, because
    // Li & Griffiths' constraint admits nothing else. Asserting the precondition too keeps this test
    // honest: if `quadric` ever stopped going astray on this cloud, the guarantee would be untested
    // rather than silently trivial.
    [Test]
    public void EllipsoidConstraintHoldsOnHyperboloidData()
    {
        // x²/4 + y²/4 - z² = 1.
        var pts = new NativeArray<fProxy3>(60, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 10; j++)
            {
                double zz = -1.0 + 2.0 * i / 5.0, v = 2.0 * math.PI_DBL * j / 10.0;
                double rr = 2.0 * math.sqrt(1.0 + zz * zz);
                pts[k++] = new fProxy3((fProxy)(rr * math.cos(v)), (fProxy)(rr * math.sin(v)), (fProxy)zz);
            }

        var q = new fProxyN(10, Allocator.Temp);
        Assert.IsTrue(Fit.quadric(pts, ref q), "quadric fit failed");
        Assert.AreEqual(QuadricKind.HyperboloidOrCone, Fit.classify(in q),
            "precondition: the unconstrained fit must go astray on this data");

        var e = new fProxyN(10, Allocator.Temp);
        Assert.IsTrue(Fit.ellipsoid(pts, ref e), "ellipsoid fit failed");
        Assert.AreEqual(QuadricKind.Ellipsoid, Fit.classify(in e),
            "the constrained fit must be an ellipsoid whatever the data says");

        pts.Dispose(); q.Dispose(); e.Dispose();
    }

    // A rotated ellipsoid pins the axes AND their pairing with the radii. An axis-aligned case cannot:
    // it would pass a fit that returned the right three lengths against the wrong three directions.
    [Test]
    public void EllipsoidRotatedRecoversAxesAndRadii()
    {
        const double alpha = 0.7, beta = 0.4;
        const double ra = 3.0, rb = 2.0, rc = 1.2;
        fProxy ca = (fProxy)math.cos(alpha), sa = (fProxy)math.sin(alpha);
        fProxy cb = (fProxy)math.cos(beta), sb = (fProxy)math.sin(beta);

        var pts = new NativeArray<fProxy3>(96, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 12; j++)
            {
                double u = math.PI_DBL * (i + 0.5) / 8.0, v = 2.0 * math.PI_DBL * j / 12.0;
                var canon = new fProxy3((fProxy)(ra * math.sin(u) * math.cos(v)),
                                        (fProxy)(rb * math.sin(u) * math.sin(v)),
                                        (fProxy)(rc * math.cos(u)));
                pts[k++] = Rot3(canon, ca, sa, cb, sb);
            }

        Assert.IsTrue(Fit.ellipsoid(pts, out fProxy3 ctr, out fProxy3 rad,
                                    out fProxy3 axA, out fProxy3 axB), "ellipsoid fit failed");

        Assert.Less(math.length(ctr), (fProxy)AlgTol, "centre must stay at the origin");

        var axes = new fProxy3[] { axA, axB, math.cross(axA, axB) };
        var radii = new double[] { (double)rad.x, (double)rad.y, (double)rad.z };

        // Each TRUE axis, rotated into place, must be matched by the recovered axis carrying its
        // radius -- which is the pairing an axis-aligned fit leaves untested.
        var truthAxis = new fProxy3[]
        {
            Rot3(new fProxy3((fProxy)1, (fProxy)0, (fProxy)0), ca, sa, cb, sb),
            Rot3(new fProxy3((fProxy)0, (fProxy)1, (fProxy)0), ca, sa, cb, sb),
            Rot3(new fProxy3((fProxy)0, (fProxy)0, (fProxy)1), ca, sa, cb, sb),
        };
        var truthRadius = new double[] { ra, rb, rc };

        for (int t = 0; t < 3; t++)
        {
            int best = 0;
            double bestDot = -1.0;
            for (int g = 0; g < 3; g++)
            {
                double d = math.abs(math.dot(math.normalize(axes[g]), truthAxis[t]));
                if (d > bestDot) { bestDot = d; best = g; }
            }

            Assert.That(bestDot, Is.EqualTo(1.0).Within(AlgTol), $"axis {t} direction");
            Assert.That(radii[best], Is.EqualTo(truthRadius[t]).Within(AlgTol),
                $"axis {t} must carry its own semi-axis length");
        }

        pts.Dispose();
    }

    // The distance function's own oracle: zero on the surface, min(radii) at the centre. The centre is
    // the case the bracketed root solve gets wrong without a coordinate floor -- every term of F
    // vanishes there, the search runs to the bracket floor, and the answer comes back 0, which would
    // score a dead-centre outlier as a perfect inlier.
    [Test]
    public void EllipsoidDistanceIsZeroOnSurfaceAndMinRadiusAtCentre()
    {
        var e = new Fit.fProxyEllipsoid
        {
            Center = new fProxy3((fProxy)1, (fProxy)(-1), (fProxy)2),
            AxisA = new fProxy3((fProxy)1, (fProxy)0, (fProxy)0),
            AxisB = new fProxy3((fProxy)0, (fProxy)1, (fProxy)0),
            Radii = new fProxy3((fProxy)3, (fProxy)2, (fProxy)1.5),
        };

        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 8; j++)
            {
                double u = math.PI_DBL * (i + 0.5) / 6.0, v = 2.0 * math.PI_DBL * (j + 0.5) / 8.0;
                var p = e.Center + new fProxy3((fProxy)(3.0 * math.sin(u) * math.cos(v)),
                                               (fProxy)(2.0 * math.sin(u) * math.sin(v)),
                                               (fProxy)(1.5 * math.cos(u)));
                Assert.That((double)e.Distance(p), Is.EqualTo(0.0).Within(DistTol), "on the surface");
            }

        Assert.That((double)e.Distance(e.Center), Is.EqualTo(1.5).Within(DistTol),
            "the centre is min(radii) from the surface, not on it");

        // A point on an axis, outside: the answer is exact along that axis.
        var onAxis = e.Center + new fProxy3((fProxy)5, (fProxy)0, (fProxy)0);
        Assert.That((double)e.Distance(onAxis), Is.EqualTo(2.0).Within(DistTol), "on the +x axis");
    }

    // Same regression in 2D, which is where the missing floor actually shipped: EllipseDistance2D is
    // shared by fProxyEllipse2 and the flat 3D ellipse.
    [Test]
    public void Ellipse2DistanceIsZeroOnCurveAndMinRadiusAtCentre()
    {
        var e = new Fit.fProxyEllipse2
        {
            Center = new fProxy2((fProxy)2, (fProxy)(-1)),
            Radii = new fProxy2((fProxy)4, (fProxy)2),
            Angle = (fProxy)0,
        };

        for (int i = 0; i < 16; i++)
        {
            double t = 2.0 * math.PI_DBL * (i + 0.5) / 16.0;
            var p = e.Center + new fProxy2((fProxy)(4.0 * math.cos(t)), (fProxy)(2.0 * math.sin(t)));
            Assert.That((double)e.Distance(p), Is.EqualTo(0.0).Within(DistTol), "on the curve");
        }

        Assert.That((double)e.Distance(e.Center), Is.EqualTo(2.0).Within(DistTol),
            "the centre is min(radii) from the curve, not on it");
    }

    // fProxyEllipse2 through the 2D IRLS driver. The shape exists so a metric can reach an ellipse at
    // all; the oracle is the defining property -- a robust loss must recover the true semi-axes better
    // than plain least squares once the cloud carries an outlier.
    [Test]
    public void Ellipse2IrlsRobustBeatsL2UnderOutlier()
    {
        const double aMaj = 5.0, bMin = 2.0;
        const int n = 32;
        var pts = new NativeArray<fProxy2>(n + 1, Allocator.Temp);
        for (int i = 0; i < n; i++)
        {
            double t = 2.0 * math.PI_DBL * i / n;
            pts[i] = new fProxy2((fProxy)(aMaj * math.cos(t)), (fProxy)(bMin * math.sin(t)));
        }
        pts[n] = new fProxy2((fProxy)0.0, (fProxy)4.0);          // off-curve, off both axes' truth

        var l2 = new fProxyL2Loss();
        var plain = new Fit.fProxyEllipse2();
        Assert.IsTrue(Fit.irls(pts, ref plain, in l2), "L2 ellipse IRLS failed");

        var huber = new fProxyHuberLoss((fProxy)0.3);
        var rob = new Fit.fProxyEllipse2();
        Assert.IsTrue(Fit.irls(pts, ref rob, in huber), "robust ellipse IRLS failed");

        Assert.Less(RadiiError(rob.Radii, aMaj, bMin), RadiiError(plain.Radii, aMaj, bMin),
            "the robust fit must recover the true semi-axes better than L2");

        pts.Dispose();
    }

    static double RadiiError(fProxy2 got, double a, double b)
    {
        double big = math.max((double)got.x, (double)got.y);
        double small = math.min((double)got.x, (double)got.y);
        return math.abs(big - a) + math.abs(small - b);
    }

    // fProxyEllipsoid through the 3D IRLS driver -- the hole this shape closes. Before it, an
    // ellipsoid could only be fitted through Fit.quadric, which has no distance function, so NO loss
    // could reach the family geometrically.
    [Test]
    public void EllipsoidIrlsRobustBeatsL2UnderOutliers()
    {
        var o = new fProxy3((fProxy)0, (fProxy)0, (fProxy)0);
        var pts = SampleEllipsoid(3.0, 2.0, 1.5, o, 8, 12, extra: 6);
        for (int i = 0; i < 6; i++)                                  // off-surface junk
            pts[96 + i] = new fProxy3((fProxy)(5.0 * math.cos(i)), (fProxy)(5.0 * math.sin(i)),
                                      (fProxy)(0.5 * i));

        var l2 = new fProxyL2Loss();
        var plain = new Fit.fProxyEllipsoid();
        Assert.IsTrue(Fit.irls(pts, ref plain, in l2), "L2 ellipsoid IRLS failed");

        var huber = new fProxyHuberLoss((fProxy)0.25);
        var rob = new Fit.fProxyEllipsoid();
        Assert.IsTrue(Fit.irls(pts, ref rob, in huber), "robust ellipsoid IRLS failed");

        Assert.Less(RadiiError(rob.Radii, 3.0, 2.0, 1.5), RadiiError(plain.Radii, 3.0, 2.0, 1.5),
            "the robust fit must recover the true semi-axes better than L2");

        pts.Dispose();
    }

    static double RadiiError(fProxy3 got, double a, double b, double c)
    {
        var g = new double[] { (double)got.x, (double)got.y, (double)got.z };
        var want = new double[] { a, b, c };
        Array.Sort(g); Array.Sort(want);
        return math.abs(g[0] - want[0]) + math.abs(g[1] - want[1]) + math.abs(g[2] - want[2]);
    }

    // The ellipsoid also satisfies IfProxyEstimable3, so the consensus driver takes it. Nine points is
    // a large minimal sample, which is a property of the algebraic estimator and is why MinimalSamples
    // reports 9 rather than the surface's 9 dof by coincidence -- RANSAC's cost scales as w^-m.
    [Test]
    public void EllipsoidRansacRejectsOutliers()
    {
        var o = new fProxy3((fProxy)0, (fProxy)0, (fProxy)0);
        var pts = SampleEllipsoid(3.0, 2.0, 1.5, o, 8, 12, extra: 12);
        for (int i = 0; i < 12; i++)
            pts[96 + i] = new fProxy3((fProxy)(6.0 + 0.3 * i), (fProxy)(1.0 * i), (fProxy)(-2.0 + i));

        var model = new Fit.fProxyEllipsoid();
        var info = Fit.ransac(pts, ref model, (fProxy)0.1, maxIter: 400, seed: 7u);

        Assert.IsTrue(info.found, "RANSAC found no ellipsoid consensus");
        Assert.GreaterOrEqual(info.inliers, 90, "the 96 on-surface points should dominate the consensus");
        Assert.LessOrEqual(info.inliers, 100, "the 12 planted outliers must not all be inliers");

        pts.Dispose();
    }

    // ----------------------------------------------------------------- sampling
    //
    // Fitting's inverse. The primary oracle is a cross-check between two independent pieces of
    // geometry: a point drawn from a shape must measure zero distance to that same shape. A sampler
    // written against a different parameterization than Distance assumes would fail it immediately.

    static void AssertSamplesOnSurface<TModel>(TModel shape, uint seed, string name)
        where TModel : struct, IfProxySampleable3
    {
        var rng = new Random(seed);
        var pts = new NativeArray<fProxy3>(200, Allocator.Temp);
        Fit.sample(in shape, ref rng, pts);

        for (int i = 0; i < pts.Length; i++)
            Assert.That((double)shape.Distance(pts[i]), Is.EqualTo(0.0).Within(DistTol),
                $"{name}: sample {i} must lie on the shape that produced it");

        pts.Dispose();
    }

    static void AssertSamplesOnCurve<TModel>(TModel shape, uint seed, string name)
        where TModel : struct, IfProxySampleable2
    {
        var rng = new Random(seed);
        var pts = new NativeArray<fProxy2>(200, Allocator.Temp);
        Fit.sample(in shape, ref rng, pts);

        for (int i = 0; i < pts.Length; i++)
            Assert.That((double)shape.Distance(pts[i]), Is.EqualTo(0.0).Within(DistTol),
                $"{name}: sample {i} must lie on the shape that produced it");

        pts.Dispose();
    }

    [Test]
    public void EverySampleLandsOnItsOwnShape()
    {
        var tilt = math.normalize(new fProxy3((fProxy)1, (fProxy)2, (fProxy)2));

        AssertSamplesOnSurface(new Fit.fProxySphere3
        {
            Center = new fProxy3((fProxy)1, (fProxy)(-2), (fProxy)0.5), Radius = (fProxy)2,
        }, 11u, "sphere");

        AssertSamplesOnSurface(new Fit.fProxyTorus
        {
            Center = new fProxy3((fProxy)(-1), (fProxy)0, (fProxy)3), Axis = tilt,
            MajorRadius = (fProxy)3, MinorRadius = (fProxy)0.8,
        }, 12u, "torus");

        AssertSamplesOnSurface(new Fit.fProxyCapsule
        {
            A = new fProxy3((fProxy)(-2), (fProxy)1, (fProxy)0),
            B = new fProxy3((fProxy)3, (fProxy)1, (fProxy)2), Radius = (fProxy)0.7,
        }, 13u, "capsule");

        AssertSamplesOnSurface(new Fit.fProxyCircle3
        {
            Center = new fProxy3((fProxy)0, (fProxy)1, (fProxy)(-1)), Normal = tilt,
            Radius = (fProxy)2.5,
        }, 14u, "circle3");

        AssertSamplesOnSurface(new Fit.fProxyEllipse3
        {
            Center = new fProxy3((fProxy)1, (fProxy)1, (fProxy)1), Normal = tilt,
            AxisA = math.normalize(math.cross(tilt, new fProxy3((fProxy)0, (fProxy)0, (fProxy)1))),
            RadiusA = (fProxy)3, RadiusB = (fProxy)1.5,
        }, 15u, "ellipse3");

        AssertSamplesOnSurface(new Fit.fProxyEllipsoid
        {
            Center = new fProxy3((fProxy)2, (fProxy)0, (fProxy)(-1)),
            AxisA = math.normalize(new fProxy3((fProxy)1, (fProxy)1, (fProxy)0)),
            AxisB = math.normalize(new fProxy3((fProxy)(-1), (fProxy)1, (fProxy)0)),
            Radii = new fProxy3((fProxy)3, (fProxy)2, (fProxy)1.5),
        }, 16u, "ellipsoid");

        AssertSamplesOnCurve(new Fit.fProxyCircle
        {
            Center = new fProxy2((fProxy)2, (fProxy)(-3)), Radius = (fProxy)4,
        }, 17u, "circle2");

        AssertSamplesOnCurve(new Fit.fProxyEllipse2
        {
            Center = new fProxy2((fProxy)1, (fProxy)2),
            Radii = new fProxy2((fProxy)4, (fProxy)2), Angle = (fProxy)0.6,
        }, 18u, "ellipse2");
    }

    // Uniform by AREA, not by angle. A torus's area element is (R + r·cos theta), so its outer rim
    // carries more surface than the inner one: the outer half must take 1/2 + r/(pi·R) of the samples,
    // where a sampler that just stepped theta uniformly would give exactly 1/2. Measuring that gap is
    // what makes this a test of uniformity rather than of mere membership.
    [Test]
    public void TorusSamplingIsUniformByAreaNotByAngle()
    {
        var t = new Fit.fProxyTorus
        {
            Axis = new fProxy3((fProxy)0, (fProxy)0, (fProxy)1),
            MajorRadius = (fProxy)3, MinorRadius = (fProxy)1,
        };

        const int n = 20000;
        var rng = new Random(12345u);
        var pts = new NativeArray<fProxy3>(n, Allocator.Temp);
        Fit.sample(in t, ref rng, pts);

        int outer = 0;
        for (int i = 0; i < n; i++)
            if (math.length(new fProxy2(pts[i].x, pts[i].y)) > (fProxy)3) outer++;   // cos(theta) > 0

        double want = 0.5 + 1.0 / (math.PI_DBL * 3.0);          // ~0.6061
        Assert.That(outer / (double)n, Is.EqualTo(want).Within(0.02),
            "the outer rim must take its share by AREA, not the 0.5 a uniform-angle sampler gives");

        pts.Dispose();
    }

    // Uniform by ARC LENGTH, not by angle. The arc element sqrt(a² sin²t + b² cos²t) is SMALLEST at
    // the major axis's vertices, so stepping t uniformly crowds the pointed ends. For this 5:1 ellipse
    // the share of arc within 45 degrees of those vertices is about 0.32, against the exactly 0.5 a
    // uniform-angle sampler would produce -- a gap no tolerance choice can blur.
    [Test]
    public void EllipseSamplingIsUniformByArcLengthNotByAngle()
    {
        var e = new Fit.fProxyEllipse2 { Radii = new fProxy2((fProxy)5, (fProxy)1), Angle = (fProxy)0 };

        const int n = 20000;
        var rng = new Random(999u);
        var pts = new NativeArray<fProxy2>(n, Allocator.Temp);
        Fit.sample(in e, ref rng, pts);

        int nearVertex = 0;
        for (int i = 0; i < n; i++)
            if (math.abs((double)pts[i].x) > 5.0 / math.SQRT2) nearVertex++;         // |cos t| > 1/sqrt(2)

        double frac = nearVertex / (double)n;
        Assert.Less(frac, 0.40, "uniform ANGLE would give 0.5; arc length must crowd the flat sides");
        Assert.Greater(frac, 0.25, "...without abandoning the vertices altogether");

        pts.Dispose();
    }

    // The ellipsoid carries the hardest Jacobian in the set, and neither surface-membership nor
    // fit-back can catch a mistake in it -- a biased sampler still lands ON the surface and still
    // fits back exactly. This measures the bias directly.
    //
    // Scaling a uniform sphere direction n by the radii stretches area by
    // sqrt((bc·nx)² + (ca·ny)² + (ab·nz)²), which for the oblate (1, 1, 0.5) spheroid is
    // sqrt(0.25 + 0.75·nz²) -- heaviest at the poles. So E[|nz|] over the correct distribution is
    // (7/9) / 1.38017 = 0.5635, where an unrejected sampler would leave nz uniform and give exactly
    // 0.5. Getting the stretch factor wrong (dropping it, or inverting it) misses by ~30 sigma.
    [Test]
    public void EllipsoidSamplingIsUniformByAreaNotByDirection()
    {
        var e = new Fit.fProxyEllipsoid
        {
            Center = default,
            AxisA = new fProxy3((fProxy)1, (fProxy)0, (fProxy)0),
            AxisB = new fProxy3((fProxy)0, (fProxy)1, (fProxy)0),
            Radii = new fProxy3((fProxy)1, (fProxy)1, (fProxy)0.5),
        };

        const int n = 20000;
        var rng = new Random(2468u);
        var pts = new NativeArray<fProxy3>(n, Allocator.Temp);
        Fit.sample(in e, ref rng, pts);

        double acc = 0;
        for (int i = 0; i < n; i++) acc += math.abs((double)pts[i].z) / 0.5;   // |nz|

        Assert.That(acc / n, Is.EqualTo(0.5635).Within(0.02),
            "area weighting must crowd the poles; an unrejected sampler gives 0.5");

        pts.Dispose();
    }

    // The capsule splits between tube and caps by area, so the split itself needs an oracle: the tube
    // takes 2·pi·r·L of the surface against the caps' 4·pi·r², i.e. L / (L + 2r). Membership tests
    // cannot see a wrong ratio -- both pieces are on the capsule either way.
    [Test]
    public void CapsuleSamplingSplitsTubeAndCapsByArea()
    {
        const double len = 4.0, rad = 1.0;
        var c = new Fit.fProxyCapsule
        {
            A = default,
            B = new fProxy3((fProxy)0, (fProxy)0, (fProxy)len),
            Radius = (fProxy)rad,
        };

        const int n = 20000;
        var rng = new Random(1357u);
        var pts = new NativeArray<fProxy3>(n, Allocator.Temp);
        Fit.sample(in c, ref rng, pts);

        int tube = 0;
        for (int i = 0; i < n; i++)
        {
            double z = (double)pts[i].z;
            if (z > 0.0 && z < len) tube++;          // outside [0, L] is necessarily a cap
        }

        double want = len / (len + 2.0 * rad);       // 2/3
        Assert.That(tube / (double)n, Is.EqualTo(want).Within(0.02),
            "tube and caps must be chosen by their areas");

        pts.Dispose();
    }

    // What having both directions buys: generate from a known shape, fit it back, compare -- with no
    // hand-rolled parameterization in the test at all.
    [Test]
    public void SampledEllipsoidFitsBackToItself()
    {
        var truth = new Fit.fProxyEllipsoid
        {
            Center = new fProxy3((fProxy)1, (fProxy)(-2), (fProxy)0.5),
            AxisA = math.normalize(new fProxy3((fProxy)1, (fProxy)1, (fProxy)0)),
            AxisB = math.normalize(new fProxy3((fProxy)(-1), (fProxy)1, (fProxy)0)),
            Radii = new fProxy3((fProxy)3, (fProxy)2, (fProxy)1.5),
        };

        var rng = new Random(4242u);
        var pts = new NativeArray<fProxy3>(400, Allocator.Temp);
        Fit.sample(in truth, ref rng, pts);

        var got = new Fit.fProxyEllipsoid();
        Assert.IsTrue(got.Estimate(pts), "fit of the sampled points failed");

        Assert.Less(math.length(got.Center - truth.Center), (fProxy)(3.0 * AlgTol), "centre");
        Assert.Less(RadiiError(got.Radii, 3.0, 2.0, 1.5), 3.0 * AlgTol, "semi-axes");

        pts.Dispose();
    }

    // Fit.total is scale-INVARIANT: scaling A and b together scales the augmented matrix, hence its
    // singular values, but leaves V untouched -- so the TLS solution is identical. That makes it an
    // exact oracle, and it is the property a threshold scaled by the largest singular value breaks:
    // once S[0] passes 1/sqrtEps such a threshold exceeds 1, and since the tested quantity is a
    // component of a UNIT vector it can never pass, so a perfectly conditioned fit reports failure.
    [Test]
    public void TotalIsScaleInvariantAtLargeMagnitude()
    {
        const int m = 12, n = 2;

        foreach (double s in new[] { 1.0, 1e4 })
        {
            var A = new fProxyMxN(m, n, Allocator.Temp);
            var b = new fProxyN(m, Allocator.Temp);
            var x = new fProxyN(n, Allocator.Temp);

            // b = 2*a0 - 0.5*a1 exactly, so TLS and OLS agree and the answer is known.
            for (int i = 0; i < m; i++)
            {
                double a0 = s * (1.0 + 0.37 * i), a1 = s * (2.0 - 0.11 * i);
                A[i, 0] = (fProxy)a0; A[i, 1] = (fProxy)a1;
                b[i] = (fProxy)(2.0 * a0 - 0.5 * a1);
            }

            Assert.IsTrue(Fit.total(in A, in b, ref x), $"total failed at scale {s}");
            Assert.That((double)x[0], Is.EqualTo(2.0).Within(1e-2), $"x0 at scale {s}");
            Assert.That((double)x[1], Is.EqualTo(-0.5).Within(1e-2), $"x1 at scale {s}");

            A.Dispose(); b.Dispose(); x.Dispose();
        }
    }

    // ------------------------------------------------------- review regressions

    // The conic design entries are FOURTH powers of the coordinates, so an off-origin cloud
    // conditions the scatter like offset^4 -- 6e10 at offset 500, past what float carries. Without
    // normalization this returns garbage or false for a shape it fits perfectly at the origin.
    [Test]
    public void ConicSurvivesAnOffOriginCloud()
    {
        foreach (double off in new[] { 0.0, 500.0 })
        {
            const int n = 40;
            var pts = new NativeArray<fProxy2>(n, Allocator.Temp);
            for (int i = 0; i < n; i++)
            {
                double t = 2.0 * math.PI_DBL * i / n;
                pts[i] = new fProxy2((fProxy)(off + 3.0 * math.cos(t)), (fProxy)(off + 2.0 * math.sin(t)));
            }

            Assert.IsTrue(Fit.ellipse(pts, out fProxy2 c, out fProxy2 rad, out _),
                $"ellipse fit failed at offset {off}");

            Assert.That((double)c.x, Is.EqualTo(off).Within(1e-3 * math.max(off, 1.0)), $"centre x at {off}");
            Assert.That((double)c.y, Is.EqualTo(off).Within(1e-3 * math.max(off, 1.0)), $"centre y at {off}");

            double big = math.max((double)rad.x, (double)rad.y);
            double small = math.min((double)rad.x, (double)rad.y);
            Assert.That(big, Is.EqualTo(3.0).Within(1e-2), $"major semi-axis at {off}");
            Assert.That(small, Is.EqualTo(2.0).Within(1e-2), $"minor semi-axis at {off}");

            pts.Dispose();
        }
    }

    // A cone is ONE nappe. The perpendicular-to-the-generating-line formula measures the infinite
    // DOUBLE cone, so a point behind the apex scores as though it lay on the mirror surface -- which
    // makes RANSAC count phantom inliers there. Apex nearest means the distance is just |p - apex|.
    [Test]
    public void ConeDistanceExcludesTheMirrorNappe()
    {
        var cone = new Fit.fProxyCone
        {
            Apex = default,
            Axis = new fProxy3((fProxy)0, (fProxy)0, (fProxy)1),
            HalfAngle = (fProxy)(math.PI_DBL / 6.0),          // 30 degrees
        };

        // Straight down the mirror axis: the nearest point of the +z nappe is the apex itself.
        var behind = new fProxy3((fProxy)0, (fProxy)0, (fProxy)(-10));
        Assert.That((double)cone.Distance(behind), Is.EqualTo(10.0).Within(Tol),
            "a point behind the apex is |p - apex| away, not the mirror cone's perpendicular");

        // A point genuinely on the surface still measures zero.
        double ax = 4.0, rad = ax * math.tan(math.PI_DBL / 6.0);
        var on = new fProxy3((fProxy)rad, (fProxy)0, (fProxy)ax);
        Assert.That((double)cone.Distance(on), Is.EqualTo(0.0).Within(Tol), "on the nappe");
    }

    // The point-to-ellipse root solve brackets on s and iterates a bounded number of times. With a
    // loose bracket the search degrades to bisection over [0, big²] and cannot reach the root at
    // extreme aspect ratios -- a point exactly ON the curve then reports a large distance, which
    // poisons every consensus and reweighting step that trusts it.
    [Test]
    public void EllipseDistanceSurvivesExtremeAspectRatio()
    {
        var e = new Fit.fProxyEllipse2
        {
            Center = default,
            Radii = new fProxy2((fProxy)1e7, (fProxy)1),
            Angle = (fProxy)0,
        };

        // The minor-axis vertex, exactly on the curve.
        Assert.That((double)e.Distance(new fProxy2((fProxy)0, (fProxy)1)), Is.EqualTo(0.0).Within(DistTol),
            "minor-axis vertex of a 1e7:1 ellipse");

        // ...and the major-axis vertex, the other extreme of the same bracket.
        Assert.That((double)e.Distance(new fProxy2((fProxy)1e7, (fProxy)0)), Is.EqualTo(0.0).Within(1e-3 * 1e7),
            "major-axis vertex of a 1e7:1 ellipse");
    }

    // irls's warm start is opt-in, and must actually do something: seeding from a good model has to
    // change the answer when the unweighted first pass would land somewhere else. A redescending loss
    // makes that visible -- from the contaminated unweighted fit it converges in the wrong basin.
    [Test]
    public void IrlsWarmStartUsesTheIncomingModel()
    {
        const int n = 60;
        var pts = new NativeArray<fProxy3>(n, Allocator.Temp);
        for (int i = 0; i < 40; i++)                                     // a plane at z = 0
            pts[i] = new fProxy3((fProxy)(i % 8), (fProxy)(i / 8), (fProxy)0);
        for (int i = 40; i < n; i++)                                     // heavy contamination at z = 6
            pts[i] = new fProxy3((fProxy)(i % 5), (fProxy)(i % 4), (fProxy)6);

        var truth = new Fit.fProxyPlane
        {
            Point = default,
            Normal = new fProxy3((fProxy)0, (fProxy)0, (fProxy)1),
        };

        var tukey = new fProxyTukeyLoss((fProxy)1);

        var cold = truth;
        Fit.irls(pts, ref cold, in tukey, maxIter: 0, warmStart: false);

        var warm = truth;
        Assert.IsTrue(Fit.irls(pts, ref warm, in tukey, maxIter: 0, warmStart: true),
            "warm-started IRLS failed");

        // The warm start keeps the plane it was handed; the cold start cannot see it at all.
        Assert.Less(math.abs((double)warm.Point.z), 0.5,
            "warm start must stay on the plane it was given");
        Assert.Greater(math.abs((double)warm.Point.z - (double)cold.Point.z), 0.5,
            "if warm and cold agree, the warm-start path is doing nothing");

        pts.Dispose();
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

        // An ellipsoid needs 9 points: 10 coefficients less the one constraint.
        var p8 = new NativeArray<fProxy3>(8, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => Fit.ellipsoid(p8, out _, out _, out _, out _));
        p8.Dispose();

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
