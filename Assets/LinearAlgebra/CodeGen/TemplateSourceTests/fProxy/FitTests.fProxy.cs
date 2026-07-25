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

        var model = new Fit.fProxyPlaneModel();
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

        var m1 = new Fit.fProxyPlaneModel();
        var i1 = Fit.ransac(pts, ref m1, (fProxy)0.15, 0, 4242u);
        var m2 = new Fit.fProxyPlaneModel();
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

        var model = new Fit.fProxyPlaneModel();
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

        var sm = new Fit.fProxySphereModel();
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

        var lm = new Fit.fProxyLine3Model();
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
        var model = new Fit.fProxyPlaneModel();

        // Fewer points than the model's minimal sample.
        Assert.Throws<ArgumentException>(() => { var m = model; Fit.ransac(pts, ref m, (fProxy)0.1); });
        pts.Dispose();

        var ok = ContaminatedPlane(10, 0, 3u);
        Assert.Throws<ArgumentException>(() => { var m = model; Fit.ransac(ok, ref m, (fProxy)0); });
        Assert.Throws<ArgumentException>(() => { var m = model; Fit.ransac(ok, ref m, (fProxy)(-1)); });
        ok.Dispose();
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
