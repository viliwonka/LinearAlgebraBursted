using System;

using Unity.Collections;
using Unity.Mathematics;

using Random = Unity.Mathematics.Random;   // this file imports System, which has its own Random

//+deleteThis
using fProxy2 = Unity.Mathematics.float2;
//-deleteThis

namespace BULA
{
    // ================================================================================================
    // The 2D shape vocabulary -- a parallel family to IfProxyShape3, not a shared one.
    //
    // C# has no numeric-vector abstraction that survives Burst (no usable INumberBase over fProxy2 vs
    // fProxy3), so being generic over dimension is not available and the interfaces are duplicated per
    // dimension. That is a language limit, and it costs exactly two copies of each SOLVER; the shapes
    // themselves do not duplicate, which is where the count would actually have hurt.
    // ================================================================================================

    /// <summary>A 2D shape that can measure its distance to a point. Distance must be non-negative.</summary>
    public interface IfProxyShape2
    {
        fProxy Distance(in fProxy2 p);
    }

    /// <summary>A 2D shape estimable from a point sample. See <see cref="IfProxyEstimable3"/>.</summary>
    public interface IfProxyEstimable2 : IfProxyShape2
    {
        int MinimalSamples { get; }
        bool Estimate(NativeArray<fProxy2> sample);
    }

    /// <summary>A 2D shape re-estimable from per-point weights, one weighted fit and no iteration.</summary>
    public interface IfProxyWeighted2 : IfProxyEstimable2
    {
        bool Refit(NativeArray<fProxy2> points, in fProxyN w);
    }

    public static partial class Fit
    {
        /// <summary>Infinite 2D line through <see cref="Point"/> along unit <see cref="Direction"/>.</summary>
        public struct fProxyLine2 : IfProxyWeighted2
        {
            public fProxy2 Point;
            public fProxy2 Direction;

            public int MinimalSamples => 2;

            // In 2D the perpendicular distance is the magnitude of the 2D cross product, which avoids
            // forming the rejected component at all.
            public fProxy Distance(in fProxy2 p)
            {
                fProxy2 v = p - Point;
                return math.abs(v.x * Direction.y - v.y * Direction.x);
            }

            public bool Estimate(NativeArray<fProxy2> sample)
            {
                bool ok = line(sample, out fProxy2 c, out fProxy2 d);
                if (ok) { Point = c; Direction = d; }
                return ok;
            }

            public bool Refit(NativeArray<fProxy2> points, in fProxyN w)
                => Subspace2Refit(points, in w, out Point, out Direction);
        }

        /// <summary>Circle of <see cref="Radius"/> about <see cref="Center"/>.</summary>
        public struct fProxyCircle : IfProxyWeighted2, IfProxySampleable2
        {
            public fProxy2 Center;
            public fProxy Radius;

            public int MinimalSamples => 3;

            public fProxy Distance(in fProxy2 p) => math.abs(math.length(p - Center) - Radius);

            public fProxy2 Sample(ref Random rng)
            {
                fProxy t = rng.NextFProxy((fProxy)0, (fProxy)(2.0 * math.PI_DBL));
                return Center + Radius * new fProxy2(math.cos(t), math.sin(t));
            }

            public bool Estimate(NativeArray<fProxy2> sample)
            {
                bool ok = sphere(sample, out fProxy2 c, out fProxy r);
                if (ok) { Center = c; Radius = r; }
                return ok;
            }

            public bool Refit(NativeArray<fProxy2> points, in fProxyN w)
            {
                if (points.Length < MinimalSamples) return false;   // see fProxySphere3.Refit

                var c = new fProxyN(2, Allocator.Temp);
                bool ok = SphereAlgebraic(points.Reinterpret<fProxy2, fProxy>(), points.Length, 2,
                                          in w, ref c, out fProxy r);
                if (ok) { Center = new fProxy2(c[0], c[1]); Radius = r; }
                c.Dispose();
                return ok;
            }
        }

        /// <summary>
        /// Ellipse with semi-axes <see cref="Radii"/> about <see cref="Center"/>, the .x axis rotated
        /// <see cref="Angle"/> radians from +x. Fitting it is CONSTRAINED to an ellipse (see
        /// <see cref="conic"/>), so a noisy near-parabolic cloud cannot come back a hyperbola.
        ///
        /// <see cref="Distance"/> is APPROXIMATE, by bracketed Newton -- see
        /// <see cref="fProxyEllipse3"/> for where and by how much.
        /// </summary>
        public struct fProxyEllipse2 : IfProxyWeighted2, IfProxySampleable2
        {
            public fProxy2 Center;
            public fProxy2 Radii;
            public fProxy Angle;

            public int MinimalSamples => 5;      // a conic has 5 degrees of freedom

            public fProxy2 Sample(ref Random rng)
            {
                fProxy t = EllipseAngle(ref rng, Radii.x, Radii.y);
                fProxy cs = math.cos(Angle), sn = math.sin(Angle);
                fProxy x = Radii.x * math.cos(t), y = Radii.y * math.sin(t);
                return Center + new fProxy2(x * cs - y * sn, x * sn + y * cs);
            }

            public fProxy Distance(in fProxy2 p)
            {
                fProxy cs = math.cos(Angle), sn = math.sin(Angle);
                fProxy2 v = p - Center;
                return EllipseDistance2D(math.abs(v.x * cs + v.y * sn),
                                         math.abs(-v.x * sn + v.y * cs), Radii.x, Radii.y);
            }

            public bool Estimate(NativeArray<fProxy2> sample)
            {
                bool ok = ellipse(sample, out fProxy2 c, out fProxy2 r, out fProxy a);
                if (ok) { Center = c; Radii = r; Angle = a; }
                return ok;
            }

            public bool Refit(NativeArray<fProxy2> points, in fProxyN w)
            {
                if (points.Length < MinimalSamples) return false;   // see fProxySphere3.Refit

                var coeffs = new fProxyN(6, Allocator.Temp);
                fProxy2 c = default, r = default;
                fProxy a = default;

                bool ok = ConicWeighted(points, in w, ref coeffs);
                if (ok) ok = EllipseFromConic(in coeffs, out c, out r, out a);
                if (ok) { Center = c; Radii = r; Angle = a; }

                coeffs.Dispose();
                return ok;
            }
        }

        // Weighted centroid + covariance eigendecomposition in 2D, reporting the largest-variance
        // axis. Unlike its 3D sibling this takes no axis selector: 2D has no plane-equivalent, so a
        // line's direction is the only thing anything here asks for.
        static bool Subspace2Refit(NativeArray<fProxy2> points, in fProxyN w,
                                   out fProxy2 origin, out fProxy2 axis)
        {
            var flat = points.Reinterpret<fProxy2, fProxy>();
            var mean = new fProxyN(2, Allocator.Temp);
            var C = new fProxyMxN(2, 2, Allocator.Temp);
            var eig = new fProxyN(2, Allocator.Temp);
            var V = new fProxyMxN(2, 2, Allocator.Temp);

            bool ok = SubspaceFitWeighted(flat, points.Length, 2, in w, ref mean, ref C, ref eig, ref V);

            origin = new fProxy2(mean[0], mean[1]);
            axis = ok ? new fProxy2(V[0, 0], V[1, 0]) : default;

            mean.Dispose(); C.Dispose(); eig.Dispose(); V.Dispose();
            return ok;
        }
    }
}
