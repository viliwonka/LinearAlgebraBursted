using Unity.Collections;
using Unity.Mathematics;

using UnityEngine;

//+deleteThis
// TEMPLATE-ONLY alias: codegen rewrites each fProxy* token -> float*/double* (real Unity.Mathematics
// types), so the field access below (.x/.y/.z) and constructors resolve natively.
using fProxy2 = Unity.Mathematics.float2;
using fProxy3 = Unity.Mathematics.float3;
//-deleteThis

namespace BULA
{
    // ================================================================================================
    // Wireframe debug drawing for the shapes Fit produces, over UnityEngine.Debug.DrawLine.
    //
    // MANAGED ONLY -- never call from inside a Burst job, same restriction as Print.ToText/SaveCsv.
    // Works from Update (lines persist for `duration`) and from OnDrawGizmos alike, and Unity strips
    // Debug.DrawLine from release builds, so these calls cost nothing there.
    //
    // Everything is drawn as line segments in world space. `color` defaults to white: Color is not a
    // compile-time constant so it cannot be a default parameter value, and a zero-alpha Color -- what
    // `default` gives -- would draw nothing, so it is remapped rather than silently invisible.
    //
    // Coordinates are narrowed to Vector3 (float) on the way out. The double instantiation therefore
    // draws at float precision while fitting at double, which is the intended split: the numbers are
    // the result, the picture is a check on them.
    // ================================================================================================
    public static partial class Draw
    {
        /// <summary>Points as small axis-aligned crosses.</summary>
        public static void points(NativeArray<fProxy3> pts, Color color = default,
                                  fProxy size = default, float duration = 0f)
        {
            Color c = Resolve(color);
            fProxy s = size > (fProxy)0 ? size : (fProxy)0.05;
            for (int i = 0; i < pts.Length; i++) Cross(pts[i], s, c, duration);
        }

        /// <summary>
        /// Points coloured by whether <paramref name="model"/> accepts them -- the companion to
        /// <see cref="Fit.ransac"/>, and the fastest way to see whether a threshold is right.
        /// </summary>
        public static void consensus<TModel>(NativeArray<fProxy3> pts, in TModel model, fProxy threshold,
                                             Color inlier = default, Color outlier = default,
                                             fProxy size = default, float duration = 0f)
            where TModel : struct, IfProxyShape3
        {
            Color ci = inlier.a > 0f ? inlier : Color.green;
            Color co = outlier.a > 0f ? outlier : Color.red;
            fProxy s = size > (fProxy)0 ? size : (fProxy)0.05;

            for (int i = 0; i < pts.Length; i++)
                Cross(pts[i], s, model.Distance(pts[i]) <= threshold ? ci : co, duration);
        }

        /// <summary>Line segment of <paramref name="length"/> centred on <paramref name="point"/>.</summary>
        public static void line(fProxy3 point, fProxy3 dir, fProxy length, Color color = default,
                                float duration = 0f)
        {
            fProxy3 d = math.normalizesafe(dir);
            fProxy h = (fProxy)0.5 * length;
            Seg(point - h * d, point + h * d, Resolve(color), duration);
        }

        /// <summary>Square patch of side <paramref name="size"/> centred on the plane, plus a normal stub.</summary>
        public static void plane(fProxy3 center, fProxy3 normal, fProxy size, Color color = default,
                                 float duration = 0f)
        {
            Color c = Resolve(color);
            fProxy3 n = math.normalizesafe(normal);
            Basis(n, out fProxy3 u, out fProxy3 v);
            fProxy h = (fProxy)0.5 * size;

            fProxy3 p00 = center - h * u - h * v, p10 = center + h * u - h * v;
            fProxy3 p11 = center + h * u + h * v, p01 = center - h * u + h * v;
            Seg(p00, p10, c, duration); Seg(p10, p11, c, duration);
            Seg(p11, p01, c, duration); Seg(p01, p00, c, duration);
            Seg(center, center + h * (fProxy)0.5 * n, c, duration);
        }

        /// <summary>Circle of <paramref name="radius"/> about <paramref name="normal"/>.</summary>
        public static void circle(fProxy3 center, fProxy3 normal, fProxy radius, Color color = default,
                                  int segments = 32, float duration = 0f)
        {
            Basis(math.normalizesafe(normal), out fProxy3 u, out fProxy3 v);
            Ring(center, u, v, radius, Resolve(color), segments, duration);
        }

        /// <summary>Sphere as three orthogonal great circles.</summary>
        public static void sphere(fProxy3 center, fProxy radius, Color color = default,
                                  int segments = 32, float duration = 0f)
        {
            Color c = Resolve(color);
            var ex = new fProxy3((fProxy)1, (fProxy)0, (fProxy)0);
            var ey = new fProxy3((fProxy)0, (fProxy)1, (fProxy)0);
            var ez = new fProxy3((fProxy)0, (fProxy)0, (fProxy)1);
            Ring(center, ex, ey, radius, c, segments, duration);
            Ring(center, ey, ez, radius, c, segments, duration);
            Ring(center, ez, ex, radius, c, segments, duration);
        }

        /// <summary>
        /// Cylinder as two end rings joined by rails. Infinite cylinders have no natural extent, so
        /// <paramref name="halfLength"/> is a drawing choice, not a fitted quantity.
        /// </summary>
        public static void cylinder(fProxy3 axisPoint, fProxy3 axisDir, fProxy radius, fProxy halfLength,
                                    Color color = default, int segments = 32, float duration = 0f)
        {
            Color c = Resolve(color);
            fProxy3 d = math.normalizesafe(axisDir);
            Basis(d, out fProxy3 u, out fProxy3 v);
            fProxy3 a = axisPoint - halfLength * d, b = axisPoint + halfLength * d;

            Ring(a, u, v, radius, c, segments, duration);
            Ring(b, u, v, radius, c, segments, duration);
            for (int k = 0; k < 4; k++)
            {
                double t = 2.0 * math.PI_DBL * k / 4.0;
                fProxy3 off = radius * ((fProxy)math.cos(t) * u + (fProxy)math.sin(t) * v);
                Seg(a + off, b + off, c, duration);
            }
        }

        /// <summary>Cone as a base ring plus rails to the apex, drawn out to <paramref name="length"/> along the axis.</summary>
        public static void cone(fProxy3 apex, fProxy3 axisDir, fProxy halfAngle, fProxy length,
                                Color color = default, int segments = 32, float duration = 0f)
        {
            Color c = Resolve(color);
            fProxy3 d = math.normalizesafe(axisDir);
            Basis(d, out fProxy3 u, out fProxy3 v);

            fProxy3 baseC = apex + length * d;
            fProxy r = length * (fProxy)math.tan(halfAngle);
            Ring(baseC, u, v, r, c, segments, duration);

            for (int k = 0; k < 4; k++)
            {
                double t = 2.0 * math.PI_DBL * k / 4.0;
                Seg(apex, baseC + r * ((fProxy)math.cos(t) * u + (fProxy)math.sin(t) * v), c, duration);
            }
        }

        /// <summary>Torus as the major ring plus minor rings spaced around it.</summary>
        public static void torus(fProxy3 center, fProxy3 axisDir, fProxy majorRadius, fProxy minorRadius,
                                 Color color = default, int majorSegments = 32, int minorSegments = 12,
                                 float duration = 0f)
        {
            Color c = Resolve(color);
            fProxy3 d = math.normalizesafe(axisDir);
            Basis(d, out fProxy3 u, out fProxy3 v);

            Ring(center, u, v, majorRadius, c, majorSegments, duration);

            int rings = math.max(minorSegments, 3);
            for (int k = 0; k < rings; k++)
            {
                double t = 2.0 * math.PI_DBL * k / rings;
                fProxy3 radial = (fProxy)math.cos(t) * u + (fProxy)math.sin(t) * v;
                Ring(center + majorRadius * radial, radial, d, minorRadius, c, 16, duration);
            }
        }

        /// <summary>Capsule as two end rings, a ring at each cap's pole, and rails along the barrel.</summary>
        public static void capsule(fProxy3 a, fProxy3 b, fProxy radius, Color color = default,
                                   int segments = 32, float duration = 0f)
        {
            Color c = Resolve(color);
            fProxy3 d = math.normalizesafe(b - a);
            if (math.lengthsq(d) <= (fProxy)0) { sphere(a, radius, c, segments, duration); return; }
            Basis(d, out fProxy3 u, out fProxy3 v);

            Ring(a, u, v, radius, c, segments, duration);
            Ring(b, u, v, radius, c, segments, duration);
            Ring(a - radius * d, u, v, radius * (fProxy)0.5, c, segments, duration);
            Ring(b + radius * d, u, v, radius * (fProxy)0.5, c, segments, duration);

            for (int k = 0; k < 4; k++)
            {
                double t = 2.0 * math.PI_DBL * k / 4.0;
                fProxy3 off = radius * ((fProxy)math.cos(t) * u + (fProxy)math.sin(t) * v);
                Seg(a + off, b + off, c, duration);
            }
        }

        /// <summary>2D ellipse in the XY plane, matching Fit.ellipse's output convention.</summary>
        public static void ellipse(fProxy2 center, fProxy2 radii, fProxy angle, Color color = default,
                                   int segments = 48, float duration = 0f)
        {
            Color c = Resolve(color);
            int seg = math.max(segments, 3);
            fProxy ca = (fProxy)math.cos(angle), sa = (fProxy)math.sin(angle);

            fProxy3 prev = default;
            for (int k = 0; k <= seg; k++)
            {
                double t = 2.0 * math.PI_DBL * k / seg;
                fProxy x = radii.x * (fProxy)math.cos(t), y = radii.y * (fProxy)math.sin(t);
                var p = new fProxy3(center.x + ca * x - sa * y, center.y + sa * x + ca * y, (fProxy)0);
                if (k > 0) Seg(prev, p, c, duration);
                prev = p;
            }
        }

        // ---- primitives ----------------------------------------------------------------------------

        static void Ring(fProxy3 center, fProxy3 u, fProxy3 v, fProxy radius, Color c,
                         int segments, float duration)
        {
            int seg = math.max(segments, 3);
            fProxy3 uu = math.normalizesafe(u), vv = math.normalizesafe(v);
            fProxy3 prev = center + radius * uu;
            for (int k = 1; k <= seg; k++)
            {
                double t = 2.0 * math.PI_DBL * k / seg;
                fProxy3 p = center + radius * ((fProxy)math.cos(t) * uu + (fProxy)math.sin(t) * vv);
                Seg(prev, p, c, duration);
                prev = p;
            }
        }

        static void Cross(fProxy3 p, fProxy s, Color c, float duration)
        {
            Seg(p - new fProxy3(s, (fProxy)0, (fProxy)0), p + new fProxy3(s, (fProxy)0, (fProxy)0), c, duration);
            Seg(p - new fProxy3((fProxy)0, s, (fProxy)0), p + new fProxy3((fProxy)0, s, (fProxy)0), c, duration);
            Seg(p - new fProxy3((fProxy)0, (fProxy)0, s), p + new fProxy3((fProxy)0, (fProxy)0, s), c, duration);
        }

        // Any unit vector perpendicular to d, plus their cross product. The axis picked to seed it is
        // whichever of x/z is least aligned with d, so the cross never collapses toward zero.
        static void Basis(fProxy3 d, out fProxy3 u, out fProxy3 v)
        {
            fProxy3 seed = math.abs(d.x) < (fProxy)0.9
                ? new fProxy3((fProxy)1, (fProxy)0, (fProxy)0)
                : new fProxy3((fProxy)0, (fProxy)0, (fProxy)1);
            u = math.normalizesafe(math.cross(d, seed));
            v = math.normalizesafe(math.cross(d, u));
        }

        static void Seg(fProxy3 a, fProxy3 b, Color c, float duration)
            => UnityEngine.Debug.DrawLine(V(a), V(b), c, duration);

        static Vector3 V(fProxy3 p) => new Vector3((float)p.x, (float)p.y, (float)p.z);

        // Resolve(Color) is element-type-agnostic and lives in Draw.Shared.cs -- declared here it would
        // be emitted identically into both generated dtype files and collide on the partial-class
        // merge, since `Draw` carries no per-type token.
    }
}
