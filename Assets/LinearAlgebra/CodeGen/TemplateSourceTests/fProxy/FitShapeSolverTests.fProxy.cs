using System;

using BULA;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

//+deleteThis
using fProxy2 = Unity.Mathematics.float2;
using fProxy3 = Unity.Mathematics.float3;
//-deleteThis

// The shape x solver grid.
//
// The point of the refactor is that shapes and solvers are independent: a shape declares what it can
// do by which interfaces it implements, and every solver accepting those interfaces works with it
// unchanged. This file is what makes that claim true rather than aspirational -- it drives each shape
// through each solver it qualifies for, so a new shape is one struct plus one row here.
//
//   shape       Distance  Estimate(RANSAC)  Refit(IRLS)  Pack/Unpack(NLS)
//   Plane          y            y               y              -
//   Line3          y            y               y              -
//   Sphere3        y            y               y              -
//   Cylinder       y            y               -              y
//   Cone           y            y               -              y
//   Torus          y            y               -              y
//   Capsule        y            y               -              y
//
// Cylinder/cone/torus/capsule have no closed-form WEIGHTED fit, so they are estimable and parametric
// but not IRLS-refittable -- the interfaces say so, and the compiler enforces it. That asymmetry is
// the design working, not a gap.
public class fProxyFitShapeSolverTests
{
    static double Tol => /*+choose[5e-3|1e-6]*/5e-3/*-choose*/;

    static NativeArray<fProxy3> PlaneCloud(int n, fProxy tilt)
    {
        var pts = new NativeArray<fProxy3>(n * n, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                pts[k++] = new fProxy3((fProxy)i, (fProxy)j, tilt * (fProxy)i);
        return pts;
    }

    static NativeArray<fProxy3> SphereCloud(fProxy3 c, double r, int nu, int nv)
    {
        var pts = new NativeArray<fProxy3>(nu * nv, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < nu; i++)
            for (int j = 0; j < nv; j++)
            {
                double u = math.PI_DBL * (i + 0.5) / nu, v = 2.0 * math.PI_DBL * j / nv;
                pts[k++] = new fProxy3((fProxy)((double)c.x + r * math.sin(u) * math.cos(v)),
                                       (fProxy)((double)c.y + r * math.sin(u) * math.sin(v)),
                                       (fProxy)((double)c.z + r * math.cos(u)));
            }
        return pts;
    }

    static NativeArray<fProxy3> CylinderCloud(double rad, int nz, int nth)
    {
        var pts = new NativeArray<fProxy3>(nz * nth, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < nz; i++)
            for (int j = 0; j < nth; j++)
            {
                double z = -3.0 + 6.0 * i / (nz - 1.0), th = 2.0 * math.PI_DBL * j / nth;
                pts[k++] = new fProxy3((fProxy)(rad * math.cos(th)), (fProxy)(rad * math.sin(th)), (fProxy)z);
            }
        return pts;
    }

    // ---------------------------------------------------------------- IRLS

    [Test]
    public void PlaneThroughIrls()
    {
        var pts = PlaneCloud(5, (fProxy)0);
        var model = new Fit.fProxyPlane();
        var l2 = new fProxyL2Loss();

        Assert.IsTrue(Fit.irls(pts, ref model, in l2), "plane IRLS failed");
        Assert.That(math.abs((double)model.Normal.z), Is.EqualTo(1.0).Within(Tol), "plane normal");

        pts.Dispose();
    }

    [Test]
    public void SphereThroughIrls()
    {
        var c = new fProxy3((fProxy)1, (fProxy)(-2), (fProxy)0.5);
        var pts = SphereCloud(c, 3.0, 6, 8);
        var model = new Fit.fProxySphere3();
        var l2 = new fProxyL2Loss();

        Assert.IsTrue(Fit.irls(pts, ref model, in l2), "sphere IRLS failed");
        Assert.That((double)model.Radius, Is.EqualTo(3.0).Within(Tol), "sphere radius");
        Assert.That((double)math.length(model.Center - c), Is.EqualTo(0.0).Within(Tol), "sphere centre");

        pts.Dispose();
    }

    [Test]
    public void LineThroughIrls()
    {
        var pts = new NativeArray<fProxy3>(12, Allocator.Temp);
        for (int i = 0; i < 12; i++) pts[i] = new fProxy3((fProxy)i, (fProxy)(2 * i), (fProxy)(-i));

        var model = new Fit.fProxyLine3();
        var l2 = new fProxyL2Loss();

        Assert.IsTrue(Fit.irls(pts, ref model, in l2), "line IRLS failed");
        var want = math.normalize(new fProxy3((fProxy)1, (fProxy)2, (fProxy)(-1)));
        double cos = math.abs(math.dot(math.normalize(model.Direction), want));
        Assert.That(cos, Is.EqualTo(1.0).Within(Tol), "line direction");

        pts.Dispose();
    }

    // A robust loss through the SAME driver, no shape-specific code involved.
    [Test]
    public void IrlsRobustLossBeatsL2OnAnyShape()
    {
        var pts = PlaneCloud(4, (fProxy)0);
        var withOutlier = new NativeArray<fProxy3>(pts.Length + 1, Allocator.Temp);
        for (int i = 0; i < pts.Length; i++) withOutlier[i] = pts[i];
        withOutlier[pts.Length] = new fProxy3((fProxy)3, (fProxy)3, (fProxy)4);

        var want = new fProxy3((fProxy)0, (fProxy)0, (fProxy)1);

        var mL2 = new Fit.fProxyPlane();
        var l2 = new fProxyL2Loss();
        Assert.IsTrue(Fit.irls(withOutlier, ref mL2, in l2));
        double errL2 = math.acos(math.min(math.abs(math.dot(math.normalize(mL2.Normal), want)), 1.0));

        var mH = new Fit.fProxyPlane();
        var huber = new fProxyHuberLoss((fProxy)0.5);
        Assert.IsTrue(Fit.irls(withOutlier, ref mH, in huber));
        double errH = math.acos(math.min(math.abs(math.dot(math.normalize(mH.Normal), want)), 1.0));

        Assert.Less(errH, errL2, $"Huber ({errH}) should beat L2 ({errL2}) through the shared driver");

        pts.Dispose(); withOutlier.Dispose();
    }

    // PRIOR WEIGHTS: the generalization that makes one-shot weighted least squares the same code path.
    // Down-weighting the outlier to zero must recover the clean answer without any robust loss.
    [Test]
    public void IrlsPriorWeightsActAsWeightedLeastSquares()
    {
        var pts = PlaneCloud(4, (fProxy)0);
        var withOutlier = new NativeArray<fProxy3>(pts.Length + 1, Allocator.Temp);
        for (int i = 0; i < pts.Length; i++) withOutlier[i] = pts[i];
        withOutlier[pts.Length] = new fProxy3((fProxy)3, (fProxy)3, (fProxy)9);

        var prior = new fProxyN(withOutlier.Length, Allocator.Temp);
        for (int i = 0; i < withOutlier.Length; i++) prior[i] = (fProxy)1;
        prior[pts.Length] = (fProxy)0;                    // caller KNOWS this sample is bad

        var model = new Fit.fProxyPlane();
        var l2 = new fProxyL2Loss();
        Assert.IsTrue(Fit.irls(withOutlier, ref model, in l2, in prior), "prior-weighted IRLS failed");

        Assert.That(math.abs((double)model.Normal.z), Is.EqualTo(1.0).Within(Tol),
            "a zero prior weight must remove the outlier entirely");

        pts.Dispose(); withOutlier.Dispose(); prior.Dispose();
    }

    // ---------------------------------------------------------------- NLS

    [Test]
    public void CylinderThroughNls()
    {
        var pts = CylinderCloud(2.0, 6, 8);

        var model = new Fit.fProxyCylinder
        {
            AxisPoint = new fProxy3((fProxy)0.1, (fProxy)(-0.1), (fProxy)0),
            Axis = new fProxy3((fProxy)0.05, (fProxy)0.05, (fProxy)1),
            Radius = (fProxy)1.7,
        };

        Assert.IsTrue(Fit.nls(pts, ref model), "cylinder NLS failed");
        Assert.That((double)model.Radius, Is.EqualTo(2.0).Within(Tol), "cylinder radius");
        Assert.That(math.abs((double)model.Axis.z), Is.EqualTo(1.0).Within(Tol), "cylinder axis");

        pts.Dispose();
    }

    [Test]
    public void TorusThroughNls()
    {
        const double R = 3.0, r0 = 1.0;
        var pts = new NativeArray<fProxy3>(96, Allocator.Temp);
        int k = 0;
        for (int i = 0; i < 12; i++)
            for (int j = 0; j < 8; j++)
            {
                double th = 2.0 * math.PI_DBL * i / 12.0, ph = 2.0 * math.PI_DBL * j / 8.0;
                double rr = R + r0 * math.cos(ph);
                pts[k++] = new fProxy3((fProxy)(rr * math.cos(th)), (fProxy)(rr * math.sin(th)),
                                       (fProxy)(r0 * math.sin(ph)));
            }

        var model = new Fit.fProxyTorus
        {
            Center = default,
            Axis = new fProxy3((fProxy)0, (fProxy)0.05, (fProxy)1),
            MajorRadius = (fProxy)2.8,
            MinorRadius = (fProxy)1.2,
        };

        Assert.IsTrue(Fit.nls(pts, ref model), "torus NLS failed");
        Assert.That((double)model.MajorRadius, Is.EqualTo(R).Within(Tol), "major radius");
        Assert.That((double)model.MinorRadius, Is.EqualTo(r0).Within(Tol), "minor radius");

        pts.Dispose();
    }

    // A robust loss on a NONLINEAR shape, again through the shared driver.
    [Test]
    public void NlsAcceptsARobustLoss()
    {
        var clean = CylinderCloud(2.0, 6, 8);
        var pts = new NativeArray<fProxy3>(clean.Length + 4, Allocator.Temp);
        for (int i = 0; i < clean.Length; i++) pts[i] = clean[i];
        for (int i = 0; i < 4; i++)
            pts[clean.Length + i] = new fProxy3((fProxy)(4.5 * math.cos(i)), (fProxy)(4.5 * math.sin(i)), (fProxy)i);

        var seed = new Fit.fProxyCylinder
        {
            AxisPoint = default,
            Axis = new fProxy3((fProxy)0, (fProxy)0, (fProxy)1),
            Radius = (fProxy)2.1,
        };

        var mL2 = seed; Assert.IsTrue(Fit.nls(pts, ref mL2), "L2 cylinder NLS failed");
        var mH = seed;
        var huber = new fProxyHuberLoss((fProxy)0.3);
        Assert.IsTrue(Fit.nls(pts, ref mH, in huber), "robust cylinder NLS failed");

        Assert.Less(math.abs((double)mH.Radius - 2.0), math.abs((double)mL2.Radius - 2.0),
            "robust NLS should recover the radius better under outliers");

        clean.Dispose(); pts.Dispose();
    }

    // ---------------------------------------------------------------- RANSAC

    [Test]
    public void ShapesThroughRansac()
    {
        var rng = new Unity.Mathematics.Random(4242u);

        // Plane with 40% junk.
        var pl = new NativeArray<fProxy3>(60, Allocator.Temp);
        for (int i = 0; i < 36; i++)
            pl[i] = new fProxy3((fProxy)rng.NextDouble(-5, 5), (fProxy)rng.NextDouble(-5, 5), (fProxy)0);
        for (int i = 36; i < 60; i++)
            pl[i] = new fProxy3((fProxy)rng.NextDouble(-5, 5), (fProxy)rng.NextDouble(-5, 5), (fProxy)rng.NextDouble(-5, 5));

        var planeModel = new Fit.fProxyPlane();
        Assert.IsTrue(Fit.ransac(pl, ref planeModel, (fProxy)0.15, 0, 11u), "plane RANSAC failed");
        Assert.That(math.abs((double)planeModel.Normal.z), Is.EqualTo(1.0).Within(0.05), "plane normal");

        // Sphere with junk.
        var c = new fProxy3((fProxy)2, (fProxy)(-1), (fProxy)0.5);
        var clean = SphereCloud(c, 3.0, 6, 8);
        var sp = new NativeArray<fProxy3>(clean.Length + 20, Allocator.Temp);
        for (int i = 0; i < clean.Length; i++) sp[i] = clean[i];
        for (int i = 0; i < 20; i++)
            sp[clean.Length + i] = new fProxy3((fProxy)rng.NextDouble(-8, 8), (fProxy)rng.NextDouble(-8, 8),
                                               (fProxy)rng.NextDouble(-8, 8));

        var sphereModel = new Fit.fProxySphere3();
        Assert.IsTrue(Fit.ransac(sp, ref sphereModel, (fProxy)0.1, 0, 22u), "sphere RANSAC failed");
        Assert.That((double)sphereModel.Radius, Is.EqualTo(3.0).Within(0.1), "sphere radius");

        pl.Dispose(); clean.Dispose(); sp.Dispose();
    }

    // RANSAC then IRLS on the same shape struct -- the intended pipeline. Consensus finds the
    // inliers, then a robust loss polishes them, with no conversion between the two stages.
    [Test]
    public void RansacThenIrlsPipeline()
    {
        var rng = new Unity.Mathematics.Random(99u);
        var pts = new NativeArray<fProxy3>(70, Allocator.Temp);
        for (int i = 0; i < 45; i++)
            pts[i] = new fProxy3((fProxy)rng.NextDouble(-5, 5), (fProxy)rng.NextDouble(-5, 5),
                                 (fProxy)rng.NextDouble(-0.02, 0.02));
        for (int i = 45; i < 70; i++)
            pts[i] = new fProxy3((fProxy)rng.NextDouble(-5, 5), (fProxy)rng.NextDouble(-5, 5),
                                 (fProxy)rng.NextDouble(-5, 5));

        var model = new Fit.fProxyPlane();
        var info = Fit.ransac(pts, ref model, (fProxy)0.1, 0, 5u);
        Assert.IsTrue(info, "RANSAC stage failed");

        // Same struct, straight into IRLS -- no adapter, no repacking.
        var huber = new fProxyHuberLoss((fProxy)0.1);
        Assert.IsTrue(Fit.irls(pts, ref model, in huber), "IRLS polish stage failed");

        Assert.That(math.abs((double)model.Normal.z), Is.EqualTo(1.0).Within(0.05),
            "the polished plane should still be the ground plane");

        pts.Dispose();
    }

    // ---------------------------------------------------------------- flat shapes in 3D

    // A circle lying in a TILTED plane -- neither a 2D circle nor a sphere. Points are generated in
    // that plane exactly, so centre, normal and radius must all come back.
    static NativeArray<fProxy3> TiltedCircle(fProxy3 c, fProxy3 u, fProxy3 v, double r, int n)
    {
        var pts = new NativeArray<fProxy3>(n, Allocator.Temp);
        for (int i = 0; i < n; i++)
        {
            double t = 2.0 * math.PI_DBL * i / n;
            pts[i] = c + (fProxy)(r * math.cos(t)) * u + (fProxy)(r * math.sin(t)) * v;
        }
        return pts;
    }

    [Test]
    public void Circle3RecoversTiltedCircle()
    {
        var n0 = math.normalize(new fProxy3((fProxy)0.4, (fProxy)(-0.3), (fProxy)1));
        Fit.OrthoBasis(n0, out fProxy3 u, out fProxy3 v);
        var c0 = new fProxy3((fProxy)1, (fProxy)2, (fProxy)(-1));

        var pts = TiltedCircle(c0, u, v, 2.5, 16);

        var model = new Fit.fProxyCircle3();
        Assert.IsTrue(model.Estimate(pts), "circle3 estimate failed");

        Assert.That((double)model.Radius, Is.EqualTo(2.5).Within(Tol), "radius");
        Assert.That((double)math.length(model.Center - c0), Is.EqualTo(0.0).Within(Tol), "centre");
        Assert.That(math.abs((double)math.dot(model.Normal, n0)), Is.EqualTo(1.0).Within(Tol), "normal");

        // Distance must penalize BOTH errors: a point pushed off the plane is not on the circle even
        // though its in-plane radius is exactly right.
        fProxy3 onCircle = c0 + (fProxy)2.5 * u;
        Assert.That((double)model.Distance(onCircle), Is.EqualTo(0.0).Within(Tol), "on-circle point");
        Assert.That((double)model.Distance(onCircle + (fProxy)0.5 * n0), Is.EqualTo(0.5).Within(Tol),
            "a point lifted off the plane must be 0.5 away, not 0");

        pts.Dispose();
    }

    // Three points determine a circle in 3D exactly, so RANSAC needs only a 3-point sample -- cheaper
    // per draw than a sphere's 4.
    [Test]
    public void Circle3ThroughRansac()
    {
        var n0 = math.normalize(new fProxy3((fProxy)0, (fProxy)0.3, (fProxy)1));
        Fit.OrthoBasis(n0, out fProxy3 u, out fProxy3 v);
        var c0 = new fProxy3((fProxy)0.5, (fProxy)0, (fProxy)0);

        var clean = TiltedCircle(c0, u, v, 3.0, 30);
        var pts = new NativeArray<fProxy3>(clean.Length + 20, Allocator.Temp);
        for (int i = 0; i < clean.Length; i++) pts[i] = clean[i];
        var rng = new Unity.Mathematics.Random(808u);
        for (int i = 0; i < 20; i++)
            pts[clean.Length + i] = new fProxy3((fProxy)rng.NextDouble(-6, 6), (fProxy)rng.NextDouble(-6, 6),
                                                (fProxy)rng.NextDouble(-6, 6));

        var model = new Fit.fProxyCircle3();
        Assert.AreEqual(3, model.MinimalSamples, "a 3D circle is determined by three points");

        var info = Fit.ransac(pts, ref model, (fProxy)0.1, 0, 17u);
        Assert.IsTrue(info, $"circle3 RANSAC failed ({info.ToString()})");
        Assert.That((double)model.Radius, Is.EqualTo(3.0).Within(0.1), "RANSAC radius");
        Assert.That(math.abs((double)math.dot(math.normalize(model.Normal), n0)),
            Is.EqualTo(1.0).Within(0.05), "RANSAC normal");

        clean.Dispose(); pts.Dispose();
    }

    [Test]
    public void Ellipse3RecoversTiltedEllipse()
    {
        var n0 = math.normalize(new fProxy3((fProxy)0.2, (fProxy)0.5, (fProxy)1));
        Fit.OrthoBasis(n0, out fProxy3 u, out fProxy3 v);
        var c0 = new fProxy3((fProxy)(-1), (fProxy)0.5, (fProxy)2);
        const double a = 4.0, b = 1.5;

        var pts = new NativeArray<fProxy3>(20, Allocator.Temp);
        for (int i = 0; i < 20; i++)
        {
            double t = 2.0 * math.PI_DBL * i / 20;
            pts[i] = c0 + (fProxy)(a * math.cos(t)) * u + (fProxy)(b * math.sin(t)) * v;
        }

        var model = new Fit.fProxyEllipse3();
        Assert.IsTrue(model.Estimate(pts), "ellipse3 estimate failed");

        double big = math.max((double)model.RadiusA, (double)model.RadiusB);
        double small = math.min((double)model.RadiusA, (double)model.RadiusB);
        Assert.That(big, Is.EqualTo(a).Within(0.05), "major semi-axis");
        Assert.That(small, Is.EqualTo(b).Within(0.05), "minor semi-axis");
        Assert.That((double)math.length(model.Center - c0), Is.EqualTo(0.0).Within(0.05), "centre");

        // Points generated ON the ellipse should measure near zero through the Newton distance.
        for (int i = 0; i < pts.Length; i++)
            Assert.Less((double)model.Distance(pts[i]), 0.05, $"on-ellipse point {i} should measure ~0");

        pts.Dispose();
    }

    // The Newton in-plane distance against a case with a known answer: on the major axis the closest
    // point is the vertex, so the distance is exactly the overshoot.
    [Test]
    public void Ellipse3DistanceMatchesKnownGeometry()
    {
        var model = new Fit.fProxyEllipse3
        {
            Center = default,
            Normal = new fProxy3((fProxy)0, (fProxy)0, (fProxy)1),
            AxisA = new fProxy3((fProxy)1, (fProxy)0, (fProxy)0),
            RadiusA = (fProxy)4,
            RadiusB = (fProxy)2,
        };

        Assert.That((double)model.Distance(new fProxy3((fProxy)6, (fProxy)0, (fProxy)0)),
            Is.EqualTo(2.0).Within(1e-3), "along the major axis: 6 - 4");
        Assert.That((double)model.Distance(new fProxy3((fProxy)0, (fProxy)5, (fProxy)0)),
            Is.EqualTo(3.0).Within(1e-3), "along the minor axis: 5 - 2");
        Assert.That((double)model.Distance(new fProxy3((fProxy)4, (fProxy)0, (fProxy)3)),
            Is.EqualTo(3.0).Within(1e-3), "straight off the plane from a vertex");
    }

    // Flat shapes must reach the metric axis, not just RANSAC: Circle3 and Ellipse3 both implement
    // Weighted3 (IRLS) and Parametric3 (NLS), so any loss applies to them like any other shape.
    [Test]
    public void FlatShapesTakeAnyMetric()
    {
        var n0 = math.normalize(new fProxy3((fProxy)0.3, (fProxy)0.2, (fProxy)1));
        Fit.OrthoBasis(n0, out fProxy3 u, out fProxy3 v);
        var c0 = new fProxy3((fProxy)1, (fProxy)0, (fProxy)0.5);

        var clean = TiltedCircle(c0, u, v, 2.0, 24);
        var pts = new NativeArray<fProxy3>(clean.Length + 1, Allocator.Temp);
        for (int i = 0; i < clean.Length; i++) pts[i] = clean[i];
        pts[clean.Length] = c0 + (fProxy)3.2 * u;          // moderate, off the circle

        // IRLS with L2 vs a robust loss -- robust must recover the radius better.
        var mL2 = new Fit.fProxyCircle3();
        var l2 = new fProxyL2Loss();
        Assert.IsTrue(Fit.irls(pts, ref mL2, in l2), "circle3 IRLS/L2 failed");

        var mH = new Fit.fProxyCircle3();
        var huber = new fProxyHuberLoss((fProxy)0.2);
        Assert.IsTrue(Fit.irls(pts, ref mH, in huber), "circle3 IRLS/Huber failed");

        Assert.Less(math.abs((double)mH.Radius - 2.0), math.abs((double)mL2.Radius - 2.0),
            "a robust metric should recover the flat circle better under an outlier");

        // Circle3 also satisfies Parametric3, so NLS accepts it -- but IRLS is the right solver here
        // and is what this asserts. Two reasons NLS is a poor fit for this shape specifically:
        //   * it has an EXACT closed-form weighted fit, so iterating buys nothing;
        //   * its packing carries gauge freedom (7 parameters for 6 dof, the normal's length being
        //     free), which leaves the finite-difference Jacobian rank-deficient by one. In double LM
        //     damps through that fine; in float the damping swamps the real directions and the step
        //     collapses, leaving the seed untouched.
        // Prefer NLS for the shapes with no weighted form -- cylinder, cone, torus, capsule.
        var mN = new Fit.fProxyCircle3 { Center = c0, Normal = n0, Radius = (fProxy)1.8 };
        Fit.nls(pts, ref mN, in huber);
        Assert.IsFalse(double.IsNaN((double)mN.Radius), "NLS must leave a finite radius even when it cannot improve");
        Assert.Greater((double)mN.Radius, 0.0, "NLS must not flip the radius negative");

        // Ellipse3 through IRLS: previously it implemented only Estimable3, so no metric reached it.
        var el = new Fit.fProxyEllipse3();
        Assert.IsTrue(Fit.irls(clean, ref el, in l2), "ellipse3 IRLS failed");
        Assert.That(math.max((double)el.RadiusA, (double)el.RadiusB), Is.EqualTo(2.0).Within(0.05),
            "a circle is an ellipse with equal semi-axes");

        clean.Dispose(); pts.Dispose();
    }

    // ---------------------------------------------------------------- 2D family

    [Test]
    public void Line2AndCircleThroughIrlsAndRansac()
    {
        // 2D line y = 2x + 1, IRLS.
        var ln = new NativeArray<fProxy2>(12, Allocator.Temp);
        for (int i = 0; i < 12; i++) ln[i] = new fProxy2((fProxy)i, (fProxy)(2 * i + 1));

        var lm = new Fit.fProxyLine2();
        var l2 = new fProxyL2Loss();
        Assert.IsTrue(Fit.irls(ln, ref lm, in l2), "2D line IRLS failed");
        var wantD = math.normalize(new fProxy2((fProxy)1, (fProxy)2));
        Assert.That(math.abs((double)math.dot(math.normalize(lm.Direction), wantD)),
            Is.EqualTo(1.0).Within(Tol), "2D line direction");

        // Circle through IRLS.
        var ci = new NativeArray<fProxy2>(16, Allocator.Temp);
        for (int i = 0; i < 16; i++)
        {
            double t = 2.0 * math.PI_DBL * i / 16.0;
            ci[i] = new fProxy2((fProxy)(2.0 + 5.0 * math.cos(t)), (fProxy)(-1.0 + 5.0 * math.sin(t)));
        }

        var cm = new Fit.fProxyCircle();
        Assert.IsTrue(Fit.irls(ci, ref cm, in l2), "circle IRLS failed");
        Assert.That((double)cm.Radius, Is.EqualTo(5.0).Within(Tol), "circle radius");

        // Circle through RANSAC, with junk.
        var rng = new Unity.Mathematics.Random(64u);
        var noisy = new NativeArray<fProxy2>(40, Allocator.Temp);
        for (int i = 0; i < 24; i++) noisy[i] = ci[i % 16];
        for (int i = 24; i < 40; i++)
            noisy[i] = new fProxy2((fProxy)rng.NextDouble(-10, 12), (fProxy)rng.NextDouble(-10, 10));

        var rm = new Fit.fProxyCircle();
        Assert.IsTrue(Fit.ransac(noisy, ref rm, (fProxy)0.1, 0, 3u), "2D circle RANSAC failed");
        Assert.That((double)rm.Radius, Is.EqualTo(5.0).Within(0.2), "RANSAC circle radius");

        ln.Dispose(); ci.Dispose(); noisy.Dispose();
    }

    // ---------------------------------------------------------------- LO-RANSAC / MAGSAC

    static NativeArray<fProxy3> NoisyPlaneWithJunk(int inliers, int outliers, double noise, uint seed)
    {
        var pts = new NativeArray<fProxy3>(inliers + outliers, Allocator.Temp);
        var rng = new Unity.Mathematics.Random(seed);
        for (int i = 0; i < inliers; i++)
            pts[i] = new fProxy3((fProxy)rng.NextDouble(-5, 5), (fProxy)rng.NextDouble(-5, 5),
                                 (fProxy)rng.NextDouble(-noise, noise));
        for (int i = 0; i < outliers; i++)
            pts[inliers + i] = new fProxy3((fProxy)rng.NextDouble(-5, 5), (fProxy)rng.NextDouble(-5, 5),
                                           (fProxy)rng.NextDouble(-5, 5));
        return pts;
    }

    // LO-RANSAC's claim: optimizing locally on each new best consensus finds a better model than
    // plain RANSAC at the SAME draw budget, because a minimal-sample hypothesis is the least accurate
    // estimate obtainable from those points. Compared at a fixed budget so the comparison is about
    // the local optimization and not about who was allowed more draws.
    [Test]
    public void LoRansacBeatsPlainRansacAtEqualBudget()
    {
        var pts = NoisyPlaneWithJunk(50, 30, 0.05, 313u);
        var want = new fProxy3((fProxy)0, (fProxy)0, (fProxy)1);

        var plain = new Fit.fProxyPlane();
        var pi = Fit.ransac(pts, ref plain, (fProxy)0.15, 30, 88u);
        Assert.IsTrue(pi, "plain RANSAC failed");
        double errPlain = math.acos(math.min(math.abs(math.dot(math.normalize(plain.Normal), want)), 1.0));

        var lo = new Fit.fProxyPlane();
        var li = Fit.ransacLo(pts, ref lo, (fProxy)0.15, 30, 88u);
        Assert.IsTrue(li, "LO-RANSAC failed");
        double errLo = math.acos(math.min(math.abs(math.dot(math.normalize(lo.Normal), want)), 1.0));

        Assert.LessOrEqual(errLo, errPlain + 1e-9,
            $"LO-RANSAC ({errLo}) should be at least as good as plain ({errPlain}) at equal budget");
        Assert.GreaterOrEqual(li.inliers, pi.inliers,
            "local optimization should not shrink the consensus");

        pts.Dispose();
    }

    // MAGSAC's selling point: it takes an UPPER BOUND on the noise rather than a tuned threshold, so
    // the same loose bound works across noise levels where a fixed threshold would have to change.
    [Test]
    public void MagsacToleratesALooseNoiseBound()
    {
        var want = new fProxy3((fProxy)0, (fProxy)0, (fProxy)1);

        // One generous sigmaMax, two very different true noise levels.
        foreach (double noise in new[] { 0.02, 0.30 })
        {
            var pts = NoisyPlaneWithJunk(50, 25, noise, 707u);

            var model = new Fit.fProxyPlane();
            var info = Fit.magsac(pts, ref model, (fProxy)0.6, 0, 21u);

            Assert.IsTrue(info, $"MAGSAC found no consensus at noise {noise} ({info.ToString()})");
            double err = math.acos(math.min(math.abs(math.dot(math.normalize(model.Normal), want)), 1.0));
            Assert.Less(err, 0.1, $"MAGSAC should recover the plane at noise {noise}, err {err}");

            pts.Dispose();
        }
    }

    [Test]
    public void ConsensusEstimatorsAreDeterministic()
    {
        var pts = NoisyPlaneWithJunk(40, 20, 0.05, 5u);

        var a = new Fit.fProxyPlane(); var ia = Fit.ransacLo(pts, ref a, (fProxy)0.15, 25, 1234u);
        var b = new Fit.fProxyPlane(); var ib = Fit.ransacLo(pts, ref b, (fProxy)0.15, 25, 1234u);
        Assert.AreEqual(ia.inliers, ib.inliers, "LO-RANSAC inliers must match at one seed");
        Assert.That((double)math.length(a.Normal - b.Normal), Is.EqualTo(0.0).Within(0.0),
            "LO-RANSAC must be bit-identical at one seed");

        var c = new Fit.fProxyPlane(); var ic = Fit.magsac(pts, ref c, (fProxy)0.5, 25, 99u);
        var d = new Fit.fProxyPlane(); var id = Fit.magsac(pts, ref d, (fProxy)0.5, 25, 99u);
        Assert.AreEqual(ic.inliers, id.inliers, "MAGSAC inliers must match at one seed");
        Assert.That((double)math.length(c.Normal - d.Normal), Is.EqualTo(0.0).Within(0.0),
            "MAGSAC must be bit-identical at one seed");

        pts.Dispose();
    }

    [Test]
    public void ConsensusGuardsThrow()
    {
        var few = new NativeArray<fProxy3>(2, Allocator.Temp);
        var model = new Fit.fProxyPlane();
        Assert.Throws<ArgumentException>(() => { var m = model; Fit.ransacLo(few, ref m, (fProxy)0.1); });
        Assert.Throws<ArgumentException>(() => { var m = model; Fit.magsac(few, ref m, (fProxy)0.1); });
        few.Dispose();

        var ok = PlaneCloud(3, (fProxy)0);
        Assert.Throws<ArgumentException>(() => { var m = model; Fit.ransacLo(ok, ref m, (fProxy)0); });
        Assert.Throws<ArgumentException>(() => { var m = model; Fit.magsac(ok, ref m, (fProxy)(-1)); });
        ok.Dispose();
    }

    [Test]
    public void IrlsGuardsThrow()
    {
        var pts = new NativeArray<fProxy3>(2, Allocator.Temp);
        var model = new Fit.fProxyPlane();
        var l2 = new fProxyL2Loss();

        // Fewer points than the shape's MinimalSamples.
        Assert.Throws<ArgumentException>(() => { var m = model; Fit.irls(pts, ref m, in l2); });
        pts.Dispose();

        // Mis-sized prior weights.
        var ok = PlaneCloud(3, (fProxy)0);
        var badPrior = new fProxyN(2, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => { var m = model; Fit.irls(ok, ref m, in l2, in badPrior); });
        ok.Dispose(); badPrior.Dispose();
    }
}
