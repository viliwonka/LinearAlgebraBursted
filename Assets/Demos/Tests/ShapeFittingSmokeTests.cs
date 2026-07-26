using NUnit.Framework;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LinearAlgebraDemos.Tests
{
    /// <summary>
    /// Headless smoke test for <see cref="ShapeFittingDemo.SolveJob"/>, the Burst job the demo's
    /// whole fit runs inside.
    ///
    /// RUNNING the job is the point. A plain C# compile proves nothing about Burst: a job that is
    /// never executed is never Burst-compiled, so a constraint violation would surface only in play
    /// mode. With CompileSynchronously the first Run forces the compile here instead.
    /// </summary>
    public class ShapeFittingSmokeTests
    {
        [Test]
        public void SolveJob_FitsAContaminatedPlane_InBurst()
        {
            const int n = 240, junk = 80;
            var pts3 = new NativeArray<float3>(n, Allocator.TempJob);
            var rng = new Unity.Mathematics.Random(12345u);
            for (int i = 0; i < n; i++)
                pts3[i] = i < n - junk
                    ? new float3(rng.NextFloat(-4f, 4f), rng.NextFloat(-4f, 4f), rng.NextFloat(-0.02f, 0.02f))
                    : new float3(rng.NextFloat(-4f, 4f), rng.NextFloat(-4f, 4f), rng.NextFloat(2f, 5f));

            var pts2 = new NativeArray<float2>(n, Allocator.TempJob);
            var outBuf = new NativeArray<ShapeFittingDemo.FitOut>(1, Allocator.TempJob);

            new ShapeFittingDemo.SolveJob
            {
                Pts3 = pts3, Pts2 = pts2, Out = outBuf,
                Is2D = false,
                ModelSel = ShapeFittingDemo.Model.Plane,
                SolverSel = ShapeFittingDemo.Solver.RansacLo,
                MetricSel = ShapeFittingDemo.Metric.Huber,
                Threshold = 0.12f, LossScale = 0.2f,
            }.Run();

            var o = outBuf[0];
            Assert.IsTrue(o.Ok != 0, "plane fit failed");
            Assert.GreaterOrEqual(o.Inliers, 140, "the 160 on-plane points should dominate the consensus");
            Assert.Greater(math.abs(o.Plane.Normal.z), 0.99f, "the normal must come back along z");

            pts3.Dispose(); pts2.Dispose(); outBuf.Dispose();
        }

        // The 2D branch reaches different generic instantiations (IfloatWeighted2), so Burst compiles
        // code the 3D case never touches.
        [Test]
        public void SolveJob_FitsA2DCircle_InBurst()
        {
            const int n = 200, junk = 50;
            var pts2 = new NativeArray<float2>(n, Allocator.TempJob);
            var rng = new Unity.Mathematics.Random(999u);
            for (int i = 0; i < n; i++)
            {
                if (i < n - junk)
                {
                    float t = 2f * math.PI * i / (n - junk);
                    pts2[i] = new float2(1f + 3f * math.cos(t), -2f + 3f * math.sin(t));
                }
                else pts2[i] = rng.NextFloat2(-8f, 8f);
            }

            var pts3 = new NativeArray<float3>(n, Allocator.TempJob);
            var outBuf = new NativeArray<ShapeFittingDemo.FitOut>(1, Allocator.TempJob);

            new ShapeFittingDemo.SolveJob
            {
                Pts3 = pts3, Pts2 = pts2, Out = outBuf,
                Is2D = true,
                ModelSel = ShapeFittingDemo.Model.Circle,
                SolverSel = ShapeFittingDemo.Solver.Ransac,
                MetricSel = ShapeFittingDemo.Metric.Huber,
                Threshold = 0.15f, LossScale = 0.2f,
            }.Run();

            var o = outBuf[0];
            Assert.IsTrue(o.Ok != 0, "circle fit failed");
            Assert.That((double)o.Circle.Radius, Is.EqualTo(3.0).Within(0.1), "radius");
            Assert.Less(math.length(o.Circle.Center - new float2(1f, -2f)), 0.1f, "centre");

            pts3.Dispose(); pts2.Dispose(); outBuf.Dispose();
        }
    }
}
