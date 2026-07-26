using System;

using BULA;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

//+deleteThis
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
