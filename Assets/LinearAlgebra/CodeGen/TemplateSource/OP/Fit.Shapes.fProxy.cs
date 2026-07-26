using System;

using Unity.Collections;
using Unity.Mathematics;

using Random = Unity.Mathematics.Random;   // this file imports System, which has its own Random

//+deleteThis
// TEMPLATE-ONLY alias: codegen rewrites each fProxy* token -> float*/double* (real Unity.Mathematics
// types), so the field access below (.x/.y/.z) and constructors resolve natively.
using fProxy3 = Unity.Mathematics.float3;
//-deleteThis

namespace BULA
{
    // ================================================================================================
    // The shape vocabulary the solvers speak.
    //
    // A shape is a small struct holding ONLY its parameters, plus the operations a solver needs from
    // it. Solvers (Fit.irls / Fit.ransac / Fit.nls) are generic over the shape, so adding a shape is
    // one struct and it immediately works with every solver whose interface it satisfies -- N + M
    // instead of N x M. This is the split PCL's sample_consensus uses: models and algorithms as
    // independent pieces, combined freely.
    //
    // The interfaces are opt-in, because not every shape can honestly do every operation:
    //   IfProxyShape3       every shape: distance from a point
    //   IfProxyEstimable3   + can be estimated from a small sample      -> RANSAC
    //   IfProxyWeighted3    + can be re-estimated from per-point weights -> IRLS
    //   IfProxyParametric3  + packs into a parameter vector              -> NLS
    //
    // A quadric implements none of these: its exact orthogonal distance has no closed form, which is
    // precisely why Sampson distance exists, so it keeps its own routes in Fit.Quadric.
    //
    // DIMENSION is baked into the interface name. C# has no numeric-vector abstraction that survives
    // Burst, so being generic over fProxy2 vs fProxy3 is not available; 2D shapes get a parallel
    // family rather than a shared one.
    // ================================================================================================

    /// <summary>A shape that can measure its distance to a point. Distance must be non-negative.</summary>
    public interface IfProxyShape3
    {
        fProxy Distance(in fProxy3 p);
    }

    /// <summary>
    /// A shape estimable from a point sample. <see cref="MinimalSamples"/> is what the estimator
    /// actually NEEDS, not the shape's degrees of freedom -- a cylinder has 5 dof but this library
    /// estimates it by least squares from 7 points, and RANSAC's cost scales as w^-m, so reporting
    /// the honest larger number matters.
    /// </summary>
    public interface IfProxyEstimable3 : IfProxyShape3
    {
        int MinimalSamples { get; }
        bool Estimate(NativeArray<fProxy3> sample);
    }

    /// <summary>
    /// A shape re-estimable from per-point weights: ONE weighted fit, no iteration. The reweighting
    /// loop belongs to <see cref="Fit.irls{TModel,TLoss}"/>, not to the shape.
    /// </summary>
    public interface IfProxyWeighted3 : IfProxyEstimable3
    {
        bool Refit(NativeArray<fProxy3> points, in fProxyN w);
    }

    /// <summary>
    /// A shape that packs into a parameter vector, for the nonlinear solver. Pack/Unpack must round
    /// trip. The packing may be redundant (a unit axis stored as three components), which leaves a
    /// flat manifold of equivalent minima -- Levenberg-Marquardt's damping handles that, and the
    /// SHAPE is still determined even though its parameters are not unique.
    /// </summary>
    public interface IfProxyParametric3 : IfProxyShape3
    {
        int ParamCount { get; }
        void Pack(ref fProxyN p);
        void Unpack(in fProxyN p);
    }

    public static partial class Fit
    {
        // ---- plane ---------------------------------------------------------------------------------

        /// <summary>Plane through <see cref="Point"/> with unit <see cref="Normal"/>.</summary>
        public struct fProxyPlane : IfProxyWeighted3
        {
            public fProxy3 Point;
            public fProxy3 Normal;

            public int MinimalSamples => 3;

            public fProxy Distance(in fProxy3 p) => math.abs(math.dot(p - Point, Normal));

            public bool Estimate(NativeArray<fProxy3> sample)
            {
                bool ok = plane(sample, out fProxy3 c, out fProxy3 n);
                if (ok) { Point = c; Normal = n; }
                return ok;
            }

            public bool Refit(NativeArray<fProxy3> points, in fProxyN w)
                => SubspaceRefit(points, in w, 3, out Point, out Normal);
        }

        // ---- line ----------------------------------------------------------------------------------

        /// <summary>Infinite line through <see cref="Point"/> along unit <see cref="Direction"/>.</summary>
        public struct fProxyLine3 : IfProxyWeighted3
        {
            public fProxy3 Point;
            public fProxy3 Direction;

            public int MinimalSamples => 2;

            public fProxy Distance(in fProxy3 p)
            {
                fProxy3 v = p - Point;
                return math.length(v - math.dot(v, Direction) * Direction);
            }

            public bool Estimate(NativeArray<fProxy3> sample)
            {
                bool ok = line(sample, out fProxy3 c, out fProxy3 d);
                if (ok) { Point = c; Direction = d; }
                return ok;
            }

            public bool Refit(NativeArray<fProxy3> points, in fProxyN w)
                => SubspaceRefit(points, in w, 1, out Point, out Direction);
        }

        // ---- sphere --------------------------------------------------------------------------------

        /// <summary>Sphere of <see cref="Radius"/> about <see cref="Center"/>.</summary>
        public struct fProxySphere3 : IfProxyWeighted3, IfProxySampleable3
        {
            public fProxy3 Center;
            public fProxy Radius;

            public int MinimalSamples => 4;

            public fProxy Distance(in fProxy3 p) => math.abs(math.length(p - Center) - Radius);

            public fProxy3 Sample(ref Random rng)
            {
                UniformDirection(ref rng, out fProxy3 d);
                return Center + Radius * d;
            }

            public bool Estimate(NativeArray<fProxy3> sample)
            {
                bool ok = sphere(sample, out fProxy3 c, out fProxy r);
                if (ok) { Center = c; Radius = r; }
                return ok;
            }

            public bool Refit(NativeArray<fProxy3> points, in fProxyN w)
            {
                var c = new fProxyN(3, Allocator.Temp);
                bool ok = SphereAlgebraic(points.Reinterpret<fProxy3, fProxy>(), points.Length, 3,
                                          in w, ref c, out fProxy r);
                if (ok) { Center = new fProxy3(c[0], c[1], c[2]); Radius = r; }
                c.Dispose();
                return ok;
            }
        }

        // ---- cylinder ------------------------------------------------------------------------------

        /// <summary>
        /// Infinite cylinder: unit <see cref="Axis"/> through <see cref="AxisPoint"/>, given
        /// <see cref="Radius"/>. Position ALONG the axis is a gauge freedom -- any point on the axis
        /// describes the same cylinder.
        /// </summary>
        public struct fProxyCylinder : IfProxyEstimable3, IfProxyParametric3
        {
            public fProxy3 AxisPoint;
            public fProxy3 Axis;
            public fProxy Radius;

            public int MinimalSamples => 7;      // the least-squares estimator's need, not the 5 dof
            public int ParamCount => 7;

            public fProxy Distance(in fProxy3 p)
            {
                fProxy3 v = p - AxisPoint;
                return math.abs(math.length(v - math.dot(v, Axis) * Axis) - Radius);
            }

            public bool Estimate(NativeArray<fProxy3> sample)
            {
                fProxy3 q = AxisPoint, d = Axis; fProxy r = Radius;
                bool ok = cylinder(sample, ref q, ref d, ref r);
                if (ok) { AxisPoint = q; Axis = d; Radius = r; }
                return ok;
            }

            public void Pack(ref fProxyN p)
            {
                p[0] = AxisPoint.x; p[1] = AxisPoint.y; p[2] = AxisPoint.z;
                p[3] = Axis.x;      p[4] = Axis.y;      p[5] = Axis.z;
                p[6] = Radius;
            }

            public void Unpack(in fProxyN p)
            {
                AxisPoint = new fProxy3(p[0], p[1], p[2]);
                Axis = math.normalizesafe(new fProxy3(p[3], p[4], p[5]));
                Radius = math.abs(p[6]);
            }
        }

        // ---- cone ----------------------------------------------------------------------------------

        /// <summary>Cone surface: <see cref="Apex"/>, unit <see cref="Axis"/>, <see cref="HalfAngle"/> in radians.</summary>
        public struct fProxyCone : IfProxyEstimable3, IfProxyParametric3
        {
            public fProxy3 Apex;
            public fProxy3 Axis;
            public fProxy HalfAngle;

            public int MinimalSamples => 7;
            public int ParamCount => 7;

            public fProxy Distance(in fProxy3 p)
            {
                fProxy3 v = p - Apex;
                fProxy ax = math.dot(v, Axis);
                fProxy rad = math.length(v - ax * Axis);

                fProxy s = math.sin(HalfAngle), c = math.cos(HalfAngle);

                // Distance to this NAPPE, not to the generating line. Projecting onto the generator
                // gives arclength t from the apex; where that is negative the point sits beyond the
                // apex, the nearest surface point IS the apex, and the perpendicular formula would
                // otherwise measure the mirror cone -- scoring points behind the apex as inliers.
                fProxy t = rad * s + ax * c;
                return t >= (fProxy)0 ? math.abs(rad * c - ax * s) : math.length(v);
            }

            public bool Estimate(NativeArray<fProxy3> sample)
            {
                fProxy3 a = Apex, d = Axis; fProxy h = HalfAngle;
                bool ok = cone(sample, ref a, ref d, ref h);
                if (ok) { Apex = a; Axis = d; HalfAngle = h; }
                return ok;
            }

            public void Pack(ref fProxyN p)
            {
                p[0] = Apex.x; p[1] = Apex.y; p[2] = Apex.z;
                p[3] = Axis.x; p[4] = Axis.y; p[5] = Axis.z;
                p[6] = HalfAngle;
            }

            public void Unpack(in fProxyN p)
            {
                Apex = new fProxy3(p[0], p[1], p[2]);
                Axis = math.normalizesafe(new fProxy3(p[3], p[4], p[5]));
                HalfAngle = math.abs(p[6]);
            }
        }

        // ---- torus ---------------------------------------------------------------------------------

        /// <summary>Torus about unit <see cref="Axis"/> through <see cref="Center"/>.</summary>
        public struct fProxyTorus : IfProxyEstimable3, IfProxyParametric3, IfProxySampleable3
        {
            public fProxy3 Center;
            public fProxy3 Axis;
            public fProxy MajorRadius;
            public fProxy MinorRadius;

            public int MinimalSamples => 8;
            public int ParamCount => 8;

            public fProxy Distance(in fProxy3 p)
            {
                fProxy3 v = p - Center;
                fProxy ax = math.dot(v, Axis);
                fProxy dr = math.length(v - ax * Axis) - MajorRadius;
                return math.abs(math.sqrt(dr * dr + ax * ax) - MinorRadius);
            }

            public bool Estimate(NativeArray<fProxy3> sample)
            {
                fProxy3 c = Center, d = Axis; fProxy R = MajorRadius, r = MinorRadius;
                bool ok = torus(sample, ref c, ref d, ref R, ref r);
                if (ok) { Center = c; Axis = d; MajorRadius = R; MinorRadius = r; }
                return ok;
            }

            // The area element is (MajorRadius + MinorRadius·cos theta), so the OUTER rim carries more
            // area than the inner one and a uniform theta would oversample the hole. Rejection against
            // the element's maximum corrects it; the azimuth is uniform as it stands.
            public fProxy3 Sample(ref Random rng)
            {
                OrthoBasis(Axis, out fProxy3 u, out fProxy3 v);

                fProxy theta = (fProxy)0, ct = (fProxy)1;
                fProxy bound = MajorRadius + MinorRadius;
                for (int i = 0; i < SampleTries; i++)
                {
                    theta = rng.NextFProxy((fProxy)0, (fProxy)(2.0 * math.PI_DBL));
                    ct = math.cos(theta);
                    if (rng.NextFProxy() * bound <= MajorRadius + MinorRadius * ct) break;
                }

                fProxy phi = rng.NextFProxy((fProxy)0, (fProxy)(2.0 * math.PI_DBL));
                fProxy3 radial = math.cos(phi) * u + math.sin(phi) * v;
                return Center + (MajorRadius + MinorRadius * ct) * radial
                              + MinorRadius * math.sin(theta) * Axis;
            }

            public void Pack(ref fProxyN p)
            {
                p[0] = Center.x; p[1] = Center.y; p[2] = Center.z;
                p[3] = Axis.x;   p[4] = Axis.y;   p[5] = Axis.z;
                p[6] = MajorRadius; p[7] = MinorRadius;
            }

            public void Unpack(in fProxyN p)
            {
                Center = new fProxy3(p[0], p[1], p[2]);
                Axis = math.normalizesafe(new fProxy3(p[3], p[4], p[5]));
                MajorRadius = math.abs(p[6]);
                MinorRadius = math.abs(p[7]);
            }
        }

        // ---- capsule -------------------------------------------------------------------------------

        /// <summary>Capsule: the segment <see cref="A"/>..<see cref="B"/> swept by <see cref="Radius"/>.</summary>
        public struct fProxyCapsule : IfProxyEstimable3, IfProxyParametric3, IfProxySampleable3
        {
            public fProxy3 A;
            public fProxy3 B;
            public fProxy Radius;

            public int MinimalSamples => 7;
            public int ParamCount => 7;

            public fProxy Distance(in fProxy3 p)
            {
                fProxy3 seg = B - A, v = p - A;
                fProxy len2 = math.dot(seg, seg);
                fProxy t = len2 > (fProxy)0 ? math.saturate(math.dot(v, seg) / len2) : (fProxy)0;
                return math.abs(math.length(v - t * seg) - Radius);
            }

            public bool Estimate(NativeArray<fProxy3> sample)
            {
                fProxy3 a = A, b = B; fProxy r = Radius;
                bool ok = capsule(sample, ref a, ref b, ref r);
                if (ok) { A = a; B = b; Radius = r; }
                return ok;
            }

            // Two pieces of constant density -- the tube (2·pi·r·L) and the two caps, which together
            // are one whole sphere (4·pi·r²) -- so picking between them by area and then sampling each
            // uniformly needs no rejection. A zero-length capsule degenerates to that sphere on its own.
            public fProxy3 Sample(ref Random rng)
            {
                fProxy3 seg = B - A;
                fProxy len = math.length(seg);
                fProxy tube = (fProxy)2 * len, caps = (fProxy)4 * Radius;   // 2·pi·r dropped from both

                if (rng.NextFProxy() * (tube + caps) < tube)
                {
                    fProxy3 axis = seg / len;
                    OrthoBasis(axis, out fProxy3 u, out fProxy3 v);
                    fProxy phi = rng.NextFProxy((fProxy)0, (fProxy)(2.0 * math.PI_DBL));
                    return A + rng.NextFProxy() * seg
                             + Radius * (math.cos(phi) * u + math.sin(phi) * v);
                }

                // Which cap a direction belongs to is decided by the direction itself, so the two
                // hemispheres together consume one uniform sphere sample.
                UniformDirection(ref rng, out fProxy3 n);
                return (math.dot(n, seg) >= (fProxy)0 ? B : A) + Radius * n;
            }

            public void Pack(ref fProxyN p)
            {
                p[0] = A.x; p[1] = A.y; p[2] = A.z;
                p[3] = B.x; p[4] = B.y; p[5] = B.z;
                p[6] = Radius;
            }

            public void Unpack(in fProxyN p)
            {
                A = new fProxy3(p[0], p[1], p[2]);
                B = new fProxy3(p[3], p[4], p[5]);
                Radius = math.abs(p[6]);
            }
        }

        // One weighted subspace fit, shared by the plane and line shapes. `k` is the subspace
        // dimension: 1 for a line (report the dominant axis), 3-1 = 2 for a plane (report the
        // least-variance axis as its normal). Reported axis is column k-1 for a line, column d-1 for
        // a plane -- expressed here as "dominant" vs "minor" by the caller's choice of k.
        static bool SubspaceRefit(NativeArray<fProxy3> points, in fProxyN w, int k,
                                  out fProxy3 origin, out fProxy3 axis)
        {
            var flat = points.Reinterpret<fProxy3, fProxy>();
            var mean = new fProxyN(3, Allocator.Temp);
            var C = new fProxyMxN(3, 3, Allocator.Temp);
            var eig = new fProxyN(3, Allocator.Temp);
            var V = new fProxyMxN(3, 3, Allocator.Temp);

            bool ok = SubspaceFitWeighted(flat, points.Length, 3, in w, ref mean, ref C, ref eig, ref V);

            origin = new fProxy3(mean[0], mean[1], mean[2]);
            // k == 1 -> a line, whose direction is the DOMINANT axis (column 0).
            // k == 3 -> a plane, whose normal is the LEAST-variance axis (column 2).
            int col = k == 1 ? 0 : 2;
            axis = ok ? new fProxy3(V[0, col], V[1, col], V[2, col]) : default;

            mean.Dispose(); C.Dispose(); eig.Dispose(); V.Dispose();
            return ok;
        }
    }
}
