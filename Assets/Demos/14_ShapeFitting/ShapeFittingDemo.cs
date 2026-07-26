using BULA;
using Unity.Collections;
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

        [Header("Fit")]
        public Model model = Model.Plane;
        public Solver solver = Solver.RansacLo;
        public Metric metric = Metric.Huber;
        [Range(0.01f, 1f)] public float threshold = 0.12f;
        [Range(0.01f, 2f)] public float lossScale = 0.2f;

        NativeArray<float3> pts3;
        NativeArray<float2> pts2;
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

        bool Is2D => dimension == Dim.TwoD;

        void OnEnable() => Allocate();
        void OnDisable() => Release();

        void Allocate()
        {
            Release();
            count = pointCount;
            pts3 = new NativeArray<float3>(count, Allocator.Persistent);
            pts2 = new NativeArray<float2>(count, Allocator.Persistent);
        }

        void Release()
        {
            if (pts3.IsCreated) pts3.Dispose();
            if (pts2.IsCreated) pts2.Dispose();
        }

        void Update()
        {
            if (count != pointCount) Allocate();
            if (animate || frame == 0) frame++;

            Generate();
            Solve();
        }

        // ---- cloud ---------------------------------------------------------------------------------

        void Generate()
        {
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
                    return new float2(1f, -0.5f) + 3f * new float2(math.cos(t), math.sin(t));
                }
                default:
                {
                    float s = rng.NextFloat(-5f, 5f);
                    return new float2(1f, 0.5f) + s * math.normalize(new float2(1f, 0.6f));
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
                    return new float3(0.5f, 0f, 0.5f) + 3f * d;
                }
                case Truth.Cylinder:
                {
                    float t = rng.NextFloat(0f, 2f * math.PI), z = rng.NextFloat(-3f, 3f);
                    return new float3(2f * math.cos(t), 2f * math.sin(t), z);
                }
                case Truth.Torus:
                {
                    float a = rng.NextFloat(0f, 2f * math.PI), b = rng.NextFloat(0f, 2f * math.PI);
                    float rr = 3f + 1f * math.cos(b);
                    return new float3(rr * math.cos(a), rr * math.sin(a), math.sin(b));
                }
                case Truth.Line:
                {
                    float s = rng.NextFloat(-5f, 5f);
                    return s * math.normalize(new float3(1f, 0.5f, 0.3f));
                }
                default:                                          // plane
                {
                    float2 uv = rng.NextFloat2(-5f, 5f);
                    return new float3(uv.x, uv.y, 0f);
                }
            }
        }

        // ---- solve ---------------------------------------------------------------------------------

        void Solve()
        {
            ok = false; inliers = 0;

            if (Is2D)
            {
                if (model == Model.Circle) { fCircle = default; ok = Solve2(ref fCircle); }
                else { fLine2 = default; ok = Solve2(ref fLine2); }
            }
            else
            {
                switch (model)
                {
                    case Model.Sphere:   fSphere = default; ok = Solve3(ref fSphere); break;
                    case Model.Line:     fLine3 = default;  ok = Solve3(ref fLine3);  break;
                    case Model.Cylinder: ok = SolveCylinder(); break;
                    case Model.Torus:    ok = SolveTorus();    break;
                    default:             fPlane = default;  ok = Solve3(ref fPlane);  break;
                }
            }

            status = $"{dimension} {truth} -> {model} via {solver}/{metric}   " +
                     (ok ? $"ok, inliers {inliers}/{count}" : "FAILED");
        }

        // Cylinder and torus have no closed-form weighted fit, so they are NLS-only here. The
        // interfaces already say that; this just routes accordingly instead of offering a choice
        // that would not compile.
        bool SolveCylinder()
        {
            fCyl = new Fit.floatCylinder { Axis = new float3(0f, 0f, 1f), Radius = 1.5f };
            bool r = RunNls(ref fCyl);
            inliers = CountInliers(in fCyl);
            return r;
        }

        bool SolveTorus()
        {
            fTorus = new Fit.floatTorus
            {
                Axis = new float3(0f, 0f, 1f), MajorRadius = 2.5f, MinorRadius = 0.8f,
            };
            bool r = RunNls(ref fTorus);
            inliers = CountInliers(in fTorus);
            return r;
        }

        bool RunNls<T>(ref T m) where T : struct, IfloatParametric3
        {
            switch (metric)
            {
                case Metric.L1:     { var l = new floatL1Loss(lossScale);     return Fit.nls(pts3, ref m, in l); }
                case Metric.Huber:  { var l = new floatHuberLoss(lossScale);  return Fit.nls(pts3, ref m, in l); }
                case Metric.Cauchy: { var l = new floatCauchyLoss(lossScale); return Fit.nls(pts3, ref m, in l); }
                case Metric.Tukey:  { var l = new floatTukeyLoss(lossScale);  return Fit.nls(pts3, ref m, in l); }
                default:            { var l = new floatL2Loss();              return Fit.nls(pts3, ref m, in l); }
            }
        }

        bool Solve3<T>(ref T m) where T : struct, IfloatWeighted3
        {
            bool r;
            switch (solver)
            {
                case Solver.Ransac:   { var i = Fit.ransac(pts3, ref m, threshold);   r = i; inliers = i.inliers; break; }
                case Solver.RansacLo: { var i = Fit.ransacLo(pts3, ref m, threshold); r = i; inliers = i.inliers; break; }
                case Solver.Magsac:   { var i = Fit.magsac(pts3, ref m, threshold * 3f); r = i; inliers = i.inliers; break; }
                default:              { r = RunIrls3(ref m); inliers = CountInliers(in m); break; }
            }
            return r;
        }

        bool RunIrls3<T>(ref T m) where T : struct, IfloatWeighted3
        {
            switch (metric)
            {
                case Metric.L1:     { var l = new floatL1Loss(lossScale);     return Fit.irls(pts3, ref m, in l); }
                case Metric.Huber:  { var l = new floatHuberLoss(lossScale);  return Fit.irls(pts3, ref m, in l); }
                case Metric.Cauchy: { var l = new floatCauchyLoss(lossScale); return Fit.irls(pts3, ref m, in l); }
                case Metric.Tukey:  { var l = new floatTukeyLoss(lossScale);  return Fit.irls(pts3, ref m, in l); }
                default:            { var l = new floatL2Loss();              return Fit.irls(pts3, ref m, in l); }
            }
        }

        bool Solve2<T>(ref T m) where T : struct, IfloatWeighted2
        {
            // The consensus estimators are 3D-only for now; 2D gets RANSAC or IRLS.
            if (solver == Solver.Ransac || solver == Solver.RansacLo || solver == Solver.Magsac)
            {
                var i = Fit.ransac(pts2, ref m, threshold);
                inliers = i.inliers;
                return i;
            }

            bool r;
            switch (metric)
            {
                case Metric.L1:     { var l = new floatL1Loss(lossScale);     r = Fit.irls(pts2, ref m, in l); break; }
                case Metric.Huber:  { var l = new floatHuberLoss(lossScale);  r = Fit.irls(pts2, ref m, in l); break; }
                case Metric.Cauchy: { var l = new floatCauchyLoss(lossScale); r = Fit.irls(pts2, ref m, in l); break; }
                case Metric.Tukey:  { var l = new floatTukeyLoss(lossScale);  r = Fit.irls(pts2, ref m, in l); break; }
                default:            { var l = new floatL2Loss();              r = Fit.irls(pts2, ref m, in l); break; }
            }

            int c = 0;
            for (int i = 0; i < count; i++) if (m.Distance(pts2[i]) <= threshold) c++;
            inliers = c;
            return r;
        }

        int CountInliers<T>(in T m) where T : struct, IfloatShape3
        {
            int c = 0;
            for (int i = 0; i < count; i++) if (m.Distance(pts3[i]) <= threshold) c++;
            return c;
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
                case Model.Cylinder: Draw.cylinder(fCyl.AxisPoint, fCyl.Axis, fCyl.Radius, 4f, Color.cyan); break;
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
