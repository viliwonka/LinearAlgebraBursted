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

// Does the Fit facade actually run under Burst?
//
// Every other Fit test is a managed [Test], so until now nothing in the suite had ever compiled these
// entry points through Burst at all -- in a Burst library that is a real gap, not a formality. A
// managed-only pass proves the math and nothing about whether the code is Burst-legal: throwing
// argument guards, NativeArray.Reinterpret, Allocator.Temp scratch and the struct-functor generics are
// each capable of failing to compile while the managed build stays perfectly green.
//
// It also re-runs the whole fit inside the job and reads results back through a NativeArray, which is
// the shape that caught the LOBPCG IJob struct-copy bug: a solver whose state is reseated by a swap
// loses that reseat when Burst copies the job struct by value.
//
// CompileSynchronously forces the compile to happen (and fail) here rather than silently falling back
// to Mono, which would make this test pass while proving nothing.
//
// The clouds below are deliberately [ReadOnly] -- that is how a caller would idiomatically pass input
// to a job, and it is a REGRESSION GUARD. This file first went red on exactly that: the geometric
// entry points used to wrap the reinterpreted array in a MUTABLE fProxyMxN view, which trips the
// safety system on a read-only array, aborting the job so every output stayed zero -- silently, with
// no error reported. They now index the flat reinterpreted array directly, the way Fit.sphere always
// did. Reinterpret itself preserves read-only-ness perfectly well; the mutable VIEW was the problem.
public class fProxyFitBurstTests
{
    [BurstCompile(CompileSynchronously = true)]
    struct GeometryJob : IJob
    {
        [ReadOnly] public NativeArray<fProxy3> Points;
        public NativeArray<fProxy> Out;          // 0..2 normal, 3 radius, 4 line dir x

        public void Execute()
        {
            Fit.plane(Points, out fProxy3 c, out fProxy3 n);
            Out[0] = n.x; Out[1] = n.y; Out[2] = n.z;

            Fit.sphere(Points, out fProxy3 sc, out fProxy r);
            Out[3] = r;

            Fit.line(Points, out fProxy3 lc, out fProxy3 ld);
            Out[4] = ld.x;
        }
    }

    [BurstCompile(CompileSynchronously = true)]
    struct RobustJob : IJob
    {
        [ReadOnly] public NativeArray<fProxy3> Points;
        public NativeArray<fProxy> Out;

        public void Execute()
        {
            var huber = new fProxyHuberLoss((fProxy)0.5);
            Fit.plane(Points, in huber, out fProxy3 c, out fProxy3 n);
            Out[0] = n.z;

            var l1 = new fProxyL1Loss((fProxy)1e-2);
            Fit.line(Points, in l1, out fProxy3 lc, out fProxy3 ld);
            Out[1] = math.length(ld);
        }
    }

    [BurstCompile(CompileSynchronously = true)]
    struct RansacJob : IJob
    {
        [ReadOnly] public NativeArray<fProxy3> Points;
        public NativeArray<fProxy> Out;
        public NativeArray<int> Inliers;

        public void Execute()
        {
            var model = new Fit.fProxyPlaneModel();
            var info = Fit.ransac(Points, ref model, (fProxy)0.1, 40, 7u);
            Out[0] = model.Normal.z;
            Inliers[0] = info.inliers;
        }
    }

    [BurstCompile(CompileSynchronously = true)]
    struct SolidJob : IJob
    {
        [ReadOnly] public NativeArray<fProxy3> Points;
        public NativeArray<fProxy> Out;

        public void Execute()
        {
            fProxy3 q = default, d = default; fProxy rad = default;
            Fit.cylinder(Points, ref q, ref d, ref rad);
            Out[0] = rad;
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

    [Test]
    public void GeometricFitsRunUnderBurst()
    {
        var pts = PlanarCloud();
        var outp = new NativeArray<fProxy>(5, Allocator.TempJob);

        new GeometryJob { Points = pts, Out = outp }.Run();

        Assert.That(math.abs((double)outp[2]), Is.EqualTo(1.0).Within(1e-3),
            "plane normal from inside a Burst job should be +-z");
        Assert.IsFalse(double.IsNaN((double)outp[3]), "sphere radius must be finite");
        Assert.IsFalse(double.IsNaN((double)outp[4]), "line direction must be finite");

        pts.Dispose(); outp.Dispose();
    }

    [Test]
    public void RobustFitsRunUnderBurst()
    {
        var pts = PlanarCloud();
        var outp = new NativeArray<fProxy>(2, Allocator.TempJob);

        new RobustJob { Points = pts, Out = outp }.Run();

        Assert.That(math.abs((double)outp[0]), Is.EqualTo(1.0).Within(1e-3), "robust plane normal");
        Assert.That((double)outp[1], Is.EqualTo(1.0).Within(1e-3), "line direction must stay unit length");

        pts.Dispose(); outp.Dispose();
    }

    [Test]
    public void RansacRunsUnderBurst()
    {
        var pts = PlanarCloud();
        var outp = new NativeArray<fProxy>(1, Allocator.TempJob);
        var inl = new NativeArray<int>(1, Allocator.TempJob);

        new RansacJob { Points = pts, Out = outp, Inliers = inl }.Run();

        Assert.That(math.abs((double)outp[0]), Is.EqualTo(1.0).Within(1e-3), "RANSAC plane normal");
        Assert.AreEqual(25, inl[0], "every point of a clean planar cloud is an inlier");

        pts.Dispose(); outp.Dispose(); inl.Dispose();
    }

    [Test]
    public void NonlinearFitRunsUnderBurst()
    {
        var pts = CylinderCloud();
        var outp = new NativeArray<fProxy>(1, Allocator.TempJob);

        new SolidJob { Points = pts, Out = outp }.Run();

        Assert.That((double)outp[0], Is.EqualTo(2.0).Within(1e-2),
            "cylinder radius recovered from inside a Burst job");

        pts.Dispose(); outp.Dispose();
    }
}
