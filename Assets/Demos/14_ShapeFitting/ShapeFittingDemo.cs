using BULA;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using Random = Unity.Mathematics.Random;

namespace LinearAlgebraDemos
{
    /// <summary>
    /// Shape fitting sandbox. Generates a noisy, outlier-contaminated cloud from one shape and fits
    /// another to it with a chosen solver and metric, re-solving every frame. Drop on an empty
    /// GameObject and enter play mode.
    ///
    /// The three dropdowns are independent on purpose: the shape you GENERATE, the shape you FIT and
    /// the SOLVER are separate choices, so you can fit the wrong shape or the wrong solver and watch
    /// what that looks like.
    /// </summary>
    public class ShapeFittingDemo : MonoBehaviour
    {
        public enum Dim { TwoD = 0, ThreeD = 1 }
        public enum Truth { Line = 0, Circle = 1, Plane = 2, Sphere = 3, Cylinder = 4, Torus = 5 }
        public enum Model { Line = 0, Circle = 1, Plane = 2, Sphere = 3, Cylinder = 4, Torus = 5 }
        public enum Solver { Irls = 0, Ransac = 1, RansacLo = 2, Magsac = 3, Nls = 4 }
        public enum Metric { L2 = 0, L1 = 1, Huber = 2, Cauchy = 3, Tukey = 4 }

        [Header("Scene")]
        public Dim dimension = Dim.ThreeD;
        public Truth truth = Truth.Plane;
        [Range(32, 2048)] public int pointCount = 400;
        [Range(0f, 0.5f)] public float noise = 0.04f;
        [Range(0f, 0.8f)] public float outlierFraction = 0.35f;
        public bool animate = true;

        [Header("Truth drift")]
        [Tooltip("Drift the GENERATING shape over time. IRLS and LM are local solvers, so a slowly " +
                 "moving target is where warm-starting from the previous frame earns its keep.")]
        public bool driftTruth = true;
        [Range(0f, 1f)] public float driftRate = 0.15f;
        [Range(0f, 2f)] public float driftAmount = 0.8f;

        [Header("Fit")]
        public Model model = Model.Plane;
        public Solver solver = Solver.RansacLo;
        public Metric metric = Metric.Huber;
        [Range(0.01f, 1f)] public float threshold = 0.12f;
        [Range(0.01f, 2f)] public float lossScale = 0.2f;

        NativeArray<float3> pts3;
        NativeArray<float2> pts2;
        NativeArray<FitOut> result;
        int count;
        uint frame;

        bool ok;
        string status = "";
        int inliers;

        // Whatever the fit produced, in a form Draw and the inlier colouring can both consume.
        Fit.floatPlane fPlane;
        Fit.floatSphere3 fSphere;
        Fit.floatLine3 fLine3;
        Fit.floatCylinder fCyl;
        Fit.floatTorus fTorus;
        Fit.floatLine2 fLine2;
        Fit.floatCircle fCircle;

        // The fitted cylinder is INFINITE and its AxisPoint is gauge-free along the axis, so the
        // drawn tube takes its centre and half-length from the inliers instead.
        float3 cylMid;
        float cylHalfLen = 4f;

        bool Is2D => dimension == Dim.TwoD;

        void OnEnable() => Allocate();
        void OnDisable() => Release();

        void Allocate()
        {
            Release();
            count = pointCount;
            pts3 = new NativeArray<float3>(count, Allocator.Persistent);
            pts2 = new NativeArray<float2>(count, Allocator.Persistent);
            result = new NativeArray<FitOut>(1, Allocator.Persistent);
        }

        void Release()
        {
            if (pts3.IsCreated) pts3.Dispose();
            if (pts2.IsCreated) pts2.Dispose();
            if (result.IsCreated) result.Dispose();
        }

        void Update()
        {
            if (count != pointCount) Allocate();
            if (animate || frame == 0) frame++;

            Generate();
            Solve();
        }

        // ---- cloud ---------------------------------------------------------------------------------

        // Truth shape parameters for THIS frame. Hoisted out of the point loop deliberately: drifting
        // per point would draw every sample from a slightly different shape, which is not a shape.
        float3 truthCenter;
        float3 truthAxis;
        float truthR1, truthR2;

        // Simplex noise rather than a sine: separate channels stay uncorrelated, so the shape wanders
        // instead of pulsing in lockstep, and it is smooth, so the fit never sees a jump it would
        // report as divergence.
        // Fully qualified: this class has its own `noise` field (the sample jitter), which shadows
        // Unity.Mathematics.noise.
        static float Wobble(float t, int channel)
            => Unity.Mathematics.noise.snoise(new float2(t, channel * 17.3f));

        void UpdateTruth()
        {
            float t = driftTruth ? Time.time * driftRate : 0f;
            float a = driftTruth ? driftAmount : 0f;

            truthCenter = a * new float3(Wobble(t, 0), Wobble(t, 1), Wobble(t, 2));
            truthAxis = math.normalize(new float3(1f, 0.5f, 0.3f)
                                     + 0.4f * a * new float3(Wobble(t, 3), Wobble(t, 4), Wobble(t, 5)));
            truthR1 = 3f + 0.5f * a * Wobble(t, 6);
            truthR2 = 1f + 0.25f * a * Wobble(t, 7);
        }

        void Generate()
        {
            UpdateTruth();

            var rng = new Random(animate ? frame * 747796405u + 1u : 1u);
            int outliers = (int)(count * outlierFraction);

            for (int i = 0; i < count; i++)
            {
                bool junk = i >= count - outliers;

                if (Is2D)
                {
                    float2 p = junk
                        ? new float2(rng.NextFloat(-6f, 6f), rng.NextFloat(-6f, 6f))
                        : TruthPoint2(ref rng) + rng.NextFloat2Direction() * rng.NextFloat(0f, noise);
                    pts2[i] = p;
                    pts3[i] = new float3(p.x, p.y, 0f);          // so 3D draw/consensus still work
                }
                else
                {
                    float3 p = junk
                        ? new float3(rng.NextFloat(-6f, 6f), rng.NextFloat(-6f, 6f), rng.NextFloat(-6f, 6f))
                        : TruthPoint3(ref rng) + rng.NextFloat3Direction() * rng.NextFloat(0f, noise);
                    pts3[i] = p;
                    pts2[i] = new float2(p.x, p.y);
                }
            }
        }

        float2 TruthPoint2(ref Random rng)
        {
            switch (truth)
            {
                case Truth.Circle:
                {
                    float t = rng.NextFloat(0f, 2f * math.PI);
                    return new float2(1f, -0.5f) + truthCenter.xy
                         + truthR1 * new float2(math.cos(t), math.sin(t));
                }
                default:
                {
                    float s = rng.NextFloat(-5f, 5f);
                    return new float2(1f, 0.5f) + truthCenter.xy + s * math.normalize(truthAxis.xy);
                }
            }
        }

        float3 TruthPoint3(ref Random rng)
        {
            switch (truth)
            {
                case Truth.Sphere:
                {
                    float3 d = rng.NextFloat3Direction();
                    return new float3(0.5f, 0f, 0.5f) + truthCenter + truthR1 * d;
                }
                case Truth.Cylinder:
                {
                    float t = rng.NextFloat(0f, 2f * math.PI), z = rng.NextFloat(-3f, 3f);
                    float r = 2f + (truthR1 - 3f);                // drift the radius, keep it near 2
                    return truthCenter + new float3(r * math.cos(t), r * math.sin(t), z);
                }
                case Truth.Torus:
                {
                    float a = rng.NextFloat(0f, 2f * math.PI), b = rng.NextFloat(0f, 2f * math.PI);
                    float rr = truthR1 + truthR2 * math.cos(b);
                    return truthCenter
                         + new float3(rr * math.cos(a), rr * math.sin(a), truthR2 * math.sin(b));
                }
                case Truth.Line:
                {
                    float s = rng.NextFloat(-5f, 5f);
                    return truthCenter + s * truthAxis;
                }
                default:                                          // plane, tilted by the drifting axis
                {
                    float2 uv = rng.NextFloat2(-5f, 5f);
                    float3 n = truthAxis;
                    float3 u = math.normalize(math.cross(n, math.abs(n.x) < 0.9f
                                                            ? new float3(1f, 0f, 0f)
                                                            : new float3(0f, 0f, 1f)));
                    return truthCenter + uv.x * u + uv.y * math.cross(n, u);
                }
            }
        }

        // ---- solve ---------------------------------------------------------------------------------

        void Solve()
        {
            var job = new SolveJob
            {
                Pts3 = pts3, Pts2 = pts2, Out = result,
                Is2D = Is2D, ModelSel = model, SolverSel = solver, MetricSel = metric,
                Threshold = threshold, LossScale = lossScale,
            };
            job.Run();

            var o = result[0];
            ok = o.Ok != 0;
            inliers = o.Inliers;
            fPlane = o.Plane; fSphere = o.Sphere; fLine3 = o.Line3;
            fCyl = o.Cyl; fTorus = o.Torus; fLine2 = o.Line2; fCircle = o.Circle;

            if (!Is2D && model == Model.Cylinder && ok) MeasureCylinderExtent();

            status = $"{dimension} {truth} -> {model} via {solver}/{metric}   " +
                     (ok ? $"ok, inliers {inliers}/{count}" : "FAILED");
        }

        // Projects the inliers onto the fitted axis and spans their range. Falls back to the fixed
        // half-length when nothing classifies as an inlier.
        void MeasureCylinderExtent()
        {
            float tMin = float.PositiveInfinity, tMax = float.NegativeInfinity;
            for (int i = 0; i < count; i++)
            {
                if (fCyl.Distance(pts3[i]) > threshold) continue;
                float t = math.dot(pts3[i] - fCyl.AxisPoint, fCyl.Axis);
                tMin = math.min(tMin, t);
                tMax = math.max(tMax, t);
            }

            bool any = tMin <= tMax;
            cylMid = any ? fCyl.AxisPoint + 0.5f * (tMin + tMax) * fCyl.Axis : fCyl.AxisPoint;
            cylHalfLen = any ? 0.5f * (tMax - tMin) : 4f;
        }

        // Every shape the job can produce, so one blittable struct carries the result out whichever
        // branch ran. Only the one matching ModelSel is populated.
        public struct FitOut
        {
            public Fit.floatPlane Plane;
            public Fit.floatSphere3 Sphere;
            public Fit.floatLine3 Line3;
            public Fit.floatCylinder Cyl;
            public Fit.floatTorus Torus;
            public Fit.floatLine2 Line2;
            public Fit.floatCircle Circle;
            public int Inliers;
            public byte Ok;
        }

        // ONE job that switches on the enums, not a generic job per combination: 6 models x 5 solvers
        // x 5 metrics would ask Burst to compile ~150 variants of the same code. Drawing stays on the
        // managed side -- Debug.DrawLine is an engine ICall Burst cannot emit (see Draw's own header).
        [BurstCompile(CompileSynchronously = true)]
        public struct SolveJob : IJob
        {
            // The cloud is never written. [ReadOnly] survives NativeArray.Reinterpret, so it also
            // pins that no fit on this path builds a WRITABLE matrix view over the caller's points --
            // the mistake that made PCA.fitLine unusable from inside a job.
            [ReadOnly] public NativeArray<float3> Pts3;
            [ReadOnly] public NativeArray<float2> Pts2;
            public NativeArray<FitOut> Out;

            public bool Is2D;
            public Model ModelSel;
            public Solver SolverSel;
            public Metric MetricSel;
            public float Threshold;
            public float LossScale;

            public void Execute()
            {
                var o = default(FitOut);
                bool r;

                if (Is2D)
                {
                    if (ModelSel == Model.Circle) r = Solve2(ref o.Circle, out o.Inliers);
                    else                          r = Solve2(ref o.Line2,  out o.Inliers);
                }
                else
                {
                    switch (ModelSel)
                    {
                        case Model.Sphere: r = Solve3(ref o.Sphere, out o.Inliers); break;
                        case Model.Line:   r = Solve3(ref o.Line3,  out o.Inliers); break;

                        // Cylinder and torus have no closed-form weighted fit, so they are NLS-only.
                        // The interfaces already say so; this routes accordingly rather than offering
                        // a choice that would not compile. Both need a seed -- these solves are local.
                        case Model.Cylinder:
                            o.Cyl = new Fit.floatCylinder { Axis = new float3(0f, 0f, 1f), Radius = 1.5f };
                            r = RunNls(ref o.Cyl);
                            o.Inliers = CountInliers(in o.Cyl);
                            break;

                        case Model.Torus:
                            o.Torus = new Fit.floatTorus
                            {
                                Axis = new float3(0f, 0f, 1f), MajorRadius = 2.5f, MinorRadius = 0.8f,
                            };
                            r = RunNls(ref o.Torus);
                            o.Inliers = CountInliers(in o.Torus);
                            break;

                        default: r = Solve3(ref o.Plane, out o.Inliers); break;
                    }
                }

                o.Ok = r ? (byte)1 : (byte)0;
                Out[0] = o;
            }

            bool RunNls<T>(ref T m) where T : struct, IfloatParametric3
            {
                switch (MetricSel)
                {
                    case Metric.L1:     { var l = new floatL1Loss(LossScale);     return Fit.nls(Pts3, ref m, in l); }
                    case Metric.Huber:  { var l = new floatHuberLoss(LossScale);  return Fit.nls(Pts3, ref m, in l); }
                    case Metric.Cauchy: { var l = new floatCauchyLoss(LossScale); return Fit.nls(Pts3, ref m, in l); }
                    case Metric.Tukey:  { var l = new floatTukeyLoss(LossScale);  return Fit.nls(Pts3, ref m, in l); }
                    default:            { var l = new floatL2Loss();              return Fit.nls(Pts3, ref m, in l); }
                }
            }

            bool Solve3<T>(ref T m, out int inliers) where T : struct, IfloatWeighted3
            {
                switch (SolverSel)
                {
                    case Solver.Ransac:   { var i = Fit.ransac(Pts3, ref m, Threshold);      inliers = i.inliers; return i; }
                    case Solver.RansacLo: { var i = Fit.ransacLo(Pts3, ref m, Threshold);    inliers = i.inliers; return i; }
                    case Solver.Magsac:   { var i = Fit.magsac(Pts3, ref m, Threshold * 3f); inliers = i.inliers; return i; }
                    default:              { bool r = RunIrls3(ref m); inliers = CountInliers(in m); return r; }
                }
            }

            bool RunIrls3<T>(ref T m) where T : struct, IfloatWeighted3
            {
                switch (MetricSel)
                {
                    case Metric.L1:     { var l = new floatL1Loss(LossScale);     return Fit.irls(Pts3, ref m, in l); }
                    case Metric.Huber:  { var l = new floatHuberLoss(LossScale);  return Fit.irls(Pts3, ref m, in l); }
                    case Metric.Cauchy: { var l = new floatCauchyLoss(LossScale); return Fit.irls(Pts3, ref m, in l); }
                    case Metric.Tukey:  { var l = new floatTukeyLoss(LossScale);  return Fit.irls(Pts3, ref m, in l); }
                    default:            { var l = new floatL2Loss();              return Fit.irls(Pts3, ref m, in l); }
                }
            }

            bool Solve2<T>(ref T m, out int inliers) where T : struct, IfloatWeighted2
            {
                // The consensus estimators are 3D-only for now; 2D gets RANSAC or IRLS.
                if (SolverSel == Solver.Ransac || SolverSel == Solver.RansacLo || SolverSel == Solver.Magsac)
                {
                    var i = Fit.ransac(Pts2, ref m, Threshold);
                    inliers = i.inliers;
                    return i;
                }

                bool r;
                switch (MetricSel)
                {
                    case Metric.L1:     { var l = new floatL1Loss(LossScale);     r = Fit.irls(Pts2, ref m, in l); break; }
                    case Metric.Huber:  { var l = new floatHuberLoss(LossScale);  r = Fit.irls(Pts2, ref m, in l); break; }
                    case Metric.Cauchy: { var l = new floatCauchyLoss(LossScale); r = Fit.irls(Pts2, ref m, in l); break; }
                    case Metric.Tukey:  { var l = new floatTukeyLoss(LossScale);  r = Fit.irls(Pts2, ref m, in l); break; }
                    default:            { var l = new floatL2Loss();              r = Fit.irls(Pts2, ref m, in l); break; }
                }

                int c = 0;
                for (int i = 0; i < Pts2.Length; i++) if (m.Distance(Pts2[i]) <= Threshold) c++;
                inliers = c;
                return r;
            }

            int CountInliers<T>(in T m) where T : struct, IfloatShape3
            {
                var s = m;
                int c = 0;
                for (int i = 0; i < Pts3.Length; i++) if (s.Distance(Pts3[i]) <= Threshold) c++;
                return c;
            }
        }

        // ---- draw ----------------------------------------------------------------------------------

        void OnDrawGizmos()
        {
            if (!pts3.IsCreated || !ok) return;

            switch (Is2D ? -1 : (int)model)
            {
                case (int)Model.Plane:    Draw.consensus(pts3, in fPlane, threshold);  break;
                case (int)Model.Sphere:   Draw.consensus(pts3, in fSphere, threshold); break;
                case (int)Model.Line:     Draw.consensus(pts3, in fLine3, threshold);  break;
                case (int)Model.Cylinder: Draw.consensus(pts3, in fCyl, threshold);    break;
                case (int)Model.Torus:    Draw.consensus(pts3, in fTorus, threshold);  break;
                default:                  Draw.points(pts3, Color.grey);               break;
            }

            if (Is2D)
            {
                if (model == Model.Circle)
                    Draw.circle(new float3(fCircle.Center.x, fCircle.Center.y, 0f),
                                new float3(0f, 0f, 1f), fCircle.Radius, Color.cyan);
                else
                    Draw.line(new float3(fLine2.Point.x, fLine2.Point.y, 0f),
                              new float3(fLine2.Direction.x, fLine2.Direction.y, 0f), 12f, Color.cyan);
                return;
            }

            switch (model)
            {
                case Model.Plane:    Draw.plane(fPlane.Point, fPlane.Normal, 8f, Color.cyan); break;
                case Model.Sphere:   Draw.sphere(fSphere.Center, fSphere.Radius, Color.cyan); break;
                case Model.Line:     Draw.line(fLine3.Point, fLine3.Direction, 12f, Color.cyan); break;
                case Model.Cylinder: Draw.cylinder(cylMid, fCyl.Axis, fCyl.Radius, cylHalfLen, Color.cyan); break;
                case Model.Torus:    Draw.torus(fTorus.Center, fTorus.Axis, fTorus.MajorRadius,
                                                fTorus.MinorRadius, Color.cyan); break;
            }
        }

        void OnGUI()
        {
            GUI.Label(new Rect(12, 12, 900, 22), status);
            GUI.Label(new Rect(12, 32, 900, 22),
                "green = inlier, red = outlier, cyan = fitted shape");
        }
    }
}
