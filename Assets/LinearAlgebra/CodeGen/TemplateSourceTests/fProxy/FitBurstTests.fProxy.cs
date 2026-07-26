using BULA;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

//+deleteThis
using fProxy2 = Unity.Mathematics.float2;
using fProxy3 = Unity.Mathematics.float3;
//-deleteThis

// Fit under Burst. A managed pass proves the math, not that the code is Burst-legal.
//
// One job with an enum switch (house pattern). That saves job scaffolding, not specializations --
// Burst still compiles everything reachable from the switch.
//
// The shape x solver grid stays managed in FitShapeSolverTests; this Bursts representative
// combinations only. Compiling the whole grid would cost build time for little extra signal.
//
// Clouds are [ReadOnly] as a regression guard: the geometric fits used to wrap the reinterpreted
// array in a MUTABLE view, which aborts the job on a read-only array and silently zeroes the output.
public class fProxyFitBurstTests
{
    public enum Case { Geometry = 0, Robust = 1, Ransac = 2, Solid = 3, ShapeSolvers = 4 }

    [BurstCompile(CompileSynchronously = true)]
    struct FitJob : IJob
    {
        public Case Which;
        [ReadOnly] public NativeArray<fProxy3> Points;
        public NativeArray<fProxy> Out;
        public NativeArray<int> Ints;

        public void Execute()
        {
            switch (Which)
            {
                case Case.Geometry:
                {
                    Fit.plane(Points, out fProxy3 c, out fProxy3 n);
                    Out[0] = n.x; Out[1] = n.y; Out[2] = n.z;

                    Fit.sphere(Points, out fProxy3 sc, out fProxy r);
                    Out[3] = r;

                    Fit.line(Points, out fProxy3 lc, out fProxy3 ld);
                    Out[4] = ld.x;
                    break;
                }

                case Case.Robust:
                {
                    var huber = new fProxyHuberLoss((fProxy)0.5);
                    Fit.plane(Points, in huber, out fProxy3 c, out fProxy3 n);
                    Out[0] = n.z;

                    var l1 = new fProxyL1Loss((fProxy)1e-2);
                    Fit.line(Points, in l1, out fProxy3 lc, out fProxy3 ld);
                    Out[1] = math.length(ld);
                    break;
                }

                case Case.Ransac:
                {
                    var model = new Fit.fProxyPlane();
                    var info = Fit.ransac(Points, ref model, (fProxy)0.1, 40, 7u);
                    Out[0] = model.Normal.z;
                    Ints[0] = info.inliers;
                    break;
                }

                case Case.Solid:
                {
                    fProxy3 q = default, d = default; fProxy rad = default;
                    Fit.cylinder(Points, ref q, ref d, ref rad);
                    Out[0] = rad;
                    break;
                }

                // The new generic core: a shape through IRLS, and through both consensus variants.
                // Every other test of these is managed, so without this the drivers are unverified
                // under Burst -- and they are the pieces most likely to trip it, being generic over
                // both the shape and the loss.
                case Case.ShapeSolvers:
                {
                    var pl = new Fit.fProxyPlane();
                    var l2 = new fProxyL2Loss();
                    if (Fit.irls(Points, ref pl, in l2)) Out[0] = pl.Normal.z;

                    var lo = new Fit.fProxyPlane();
                    var loInfo = Fit.ransacLo(Points, ref lo, (fProxy)0.1, 30, 3u);
                    Out[1] = lo.Normal.z;
                    Ints[0] = loInfo.inliers;

                    var mg = new Fit.fProxyPlane();
                    Fit.magsac(Points, ref mg, (fProxy)0.3, 30, 4u);
                    Out[2] = mg.Normal.z;
                    break;
                }
            }
        }
    }

    static NativeArray<fProxy3> PlanarCloud()
    {
        var pts = new NativeArray<fProxy3>(25, Allocator.TempJob);
        int k = 0;
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                pts[k++] = new fProxy3((fProxy)(i - 2), (fProxy)(j - 2), (fProxy)0);
        return pts;
    }

    static NativeArray<fProxy3> CylinderCloud()
    {
        var pts = new NativeArray<fProxy3>(48, Allocator.TempJob);
        int k = 0;
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 8; j++)
            {
                double z = -3.0 + 6.0 * i / 5.0, th = 2.0 * math.PI_DBL * j / 8.0;
                pts[k++] = new fProxy3((fProxy)(2.0 * math.cos(th)), (fProxy)(2.0 * math.sin(th)), (fProxy)z);
            }
        return pts;
    }

    static void RunCase(Case which, NativeArray<fProxy3> pts, int outN,
                        out NativeArray<fProxy> outp, out NativeArray<int> ints)
    {
        outp = new NativeArray<fProxy>(outN, Allocator.TempJob);
        ints = new NativeArray<int>(1, Allocator.TempJob);
        new FitJob { Which = which, Points = pts, Out = outp, Ints = ints }.Run();
    }

    [Test]
    public void GeometricFitsRunUnderBurst()
    {
        var pts = PlanarCloud();
        RunCase(Case.Geometry, pts, 5, out var outp, out var ints);

        Assert.That(math.abs((double)outp[2]), Is.EqualTo(1.0).Within(1e-3),
            "plane normal from inside a Burst job should be +-z");
        Assert.IsFalse(double.IsNaN((double)outp[3]), "sphere radius must be finite");
        Assert.IsFalse(double.IsNaN((double)outp[4]), "line direction must be finite");

        pts.Dispose(); outp.Dispose(); ints.Dispose();
    }

    [Test]
    public void RobustFitsRunUnderBurst()
    {
        var pts = PlanarCloud();
        RunCase(Case.Robust, pts, 2, out var outp, out var ints);

        Assert.That(math.abs((double)outp[0]), Is.EqualTo(1.0).Within(1e-3), "robust plane normal");
        Assert.That((double)outp[1], Is.EqualTo(1.0).Within(1e-3), "line direction must stay unit length");

        pts.Dispose(); outp.Dispose(); ints.Dispose();
    }

    [Test]
    public void RansacRunsUnderBurst()
    {
        var pts = PlanarCloud();
        RunCase(Case.Ransac, pts, 1, out var outp, out var ints);

        Assert.That(math.abs((double)outp[0]), Is.EqualTo(1.0).Within(1e-3), "RANSAC plane normal");
        Assert.AreEqual(25, ints[0], "every point of a clean planar cloud is an inlier");

        pts.Dispose(); outp.Dispose(); ints.Dispose();
    }

    [Test]
    public void NonlinearFitRunsUnderBurst()
    {
        var pts = CylinderCloud();
        RunCase(Case.Solid, pts, 1, out var outp, out var ints);

        Assert.That((double)outp[0], Is.EqualTo(2.0).Within(1e-2),
            "cylinder radius recovered from inside a Burst job");

        pts.Dispose(); outp.Dispose(); ints.Dispose();
    }

    [Test]
    public void ShapeSolversRunUnderBurst()
    {
        var pts = PlanarCloud();
        RunCase(Case.ShapeSolvers, pts, 3, out var outp, out var ints);

        Assert.That(math.abs((double)outp[0]), Is.EqualTo(1.0).Within(1e-3), "irls plane normal");
        Assert.That(math.abs((double)outp[1]), Is.EqualTo(1.0).Within(1e-3), "ransacLo plane normal");
        Assert.That(math.abs((double)outp[2]), Is.EqualTo(1.0).Within(1e-3), "magsac plane normal");
        Assert.AreEqual(25, ints[0], "clean cloud: every point an inlier under LO-RANSAC");

        pts.Dispose(); outp.Dispose(); ints.Dispose();
    }
}
