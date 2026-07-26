using System;

using Unity.Collections;
using Unity.Mathematics;

//+deleteThis
using fProxy2 = Unity.Mathematics.float2;
using fProxy3 = Unity.Mathematics.float3;
//-deleteThis

namespace BULA
{
    // ================================================================================================
    // FLAT shapes in 3D -- a circle or ellipse lying in an arbitrary plane. Distinct from both the 2D
    // versions (which live in the xy plane) and from a sphere.
    //
    // Both fit the same way: fit the plane, project onto its basis, fit the 2D shape there, lift the
    // result back. That reuses the existing plane and circle/conic fitters instead of deriving new
    // normal equations, and it is exact for the minimal case -- three points give an exact plane and
    // then an exact circumcircle.
    // ================================================================================================
    public static partial class Fit
    {
        /// <summary>
        /// Circle in 3D: <see cref="Radius"/> about <see cref="Center"/>, lying in the plane with unit
        /// <see cref="Normal"/>. Distance combines the out-of-plane and in-plane errors, so a point
        /// off the plane is penalized even when its in-plane radius is right.
        /// </summary>
        public struct fProxyCircle3 : IfProxyWeighted3, IfProxyParametric3
        {
            public fProxy3 Center;
            public fProxy3 Normal;
            public fProxy Radius;

            public int MinimalSamples => 3;      // three points fix the plane and the circumcircle
            public int ParamCount => 7;

            public fProxy Distance(in fProxy3 p)
            {
                fProxy3 q = p - Center;
                fProxy axial = math.dot(q, Normal);
                fProxy radial = math.length(q - axial * Normal) - Radius;
                return math.sqrt(axial * axial + radial * radial);
            }

            public bool Estimate(NativeArray<fProxy3> sample)
            {
                if (!plane(sample, out fProxy3 c0, out fProxy3 n)) return false;
                OrthoBasis(n, out fProxy3 u, out fProxy3 v);

                var flat = Project(sample, c0, u, v);
                bool ok = sphere(flat, out fProxy2 c2, out fProxy r);
                flat.Dispose();
                if (!ok) return false;

                Center = c0 + c2.x * u + c2.y * v;
                Normal = n;
                Radius = r;
                return true;
            }

            // Weighted plane, then a weighted circle in that plane -- the same two stages as Estimate,
            // with the caller's weights carried into both.
            public bool Refit(NativeArray<fProxy3> points, in fProxyN w)
            {
                if (!WeightedPlane(points, in w, out fProxy3 c0, out fProxy3 n)) return false;
                OrthoBasis(n, out fProxy3 u, out fProxy3 v);

                var flat = Project(points, c0, u, v);
                var c2 = new fProxyN(2, Allocator.Temp);
                bool ok = SphereAlgebraic(flat.Reinterpret<fProxy2, fProxy>(), points.Length, 2,
                                          in w, ref c2, out fProxy r);
                if (ok)
                {
                    Center = c0 + c2[0] * u + c2[1] * v;
                    Normal = n;
                    Radius = r;
                }

                flat.Dispose(); c2.Dispose();
                return ok;
            }

            public void Pack(ref fProxyN p)
            {
                p[0] = Center.x; p[1] = Center.y; p[2] = Center.z;
                p[3] = Normal.x; p[4] = Normal.y; p[5] = Normal.z;
                p[6] = Radius;
            }

            public void Unpack(in fProxyN p)
            {
                Center = new fProxy3(p[0], p[1], p[2]);
                Normal = math.normalizesafe(new fProxy3(p[3], p[4], p[5]));
                Radius = math.abs(p[6]);
            }
        }

        /// <summary>
        /// Ellipse in 3D: centred at <see cref="Center"/> in the plane with unit <see cref="Normal"/>,
        /// with semi-axes <see cref="RadiusA"/> along unit <see cref="AxisA"/> and <see cref="RadiusB"/>
        /// along Normal x AxisA.
        ///
        /// <see cref="Distance"/> is APPROXIMATE in the plane: the exact point-to-ellipse distance has
        /// no closed form (it needs the root of a quartic), so the in-plane part is Newton-refined from
        /// the radial guess. A few iterations converge tightly except very near the centre, where the
        /// closest point is genuinely ambiguous.
        /// </summary>
        public struct fProxyEllipse3 : IfProxyWeighted3, IfProxyParametric3
        {
            public fProxy3 Center;
            public fProxy3 Normal;
            public fProxy3 AxisA;
            public fProxy RadiusA;
            public fProxy RadiusB;

            public int MinimalSamples => 5;      // a conic has 5 dof once the plane is fixed

            public fProxy Distance(in fProxy3 p)
            {
                fProxy3 q = p - Center;
                fProxy axial = math.dot(q, Normal);
                fProxy3 inPlane = q - axial * Normal;

                fProxy3 b = math.cross(Normal, AxisA);
                fProxy x = math.dot(inPlane, AxisA), y = math.dot(inPlane, b);
                fProxy radial = EllipseDistance2D(math.abs(x), math.abs(y), RadiusA, RadiusB);

                return math.sqrt(axial * axial + radial * radial);
            }

            public bool Estimate(NativeArray<fProxy3> sample)
            {
                if (!plane(sample, out fProxy3 c0, out fProxy3 n)) return false;
                OrthoBasis(n, out fProxy3 u, out fProxy3 v);

                var flat = Project(sample, c0, u, v);
                bool ok = ellipse(flat, out fProxy2 c2, out fProxy2 rad, out fProxy ang);
                flat.Dispose();
                if (!ok) return false;

                Center = c0 + c2.x * u + c2.y * v;
                Normal = n;
                AxisA = math.cos(ang) * u + math.sin(ang) * v;
                RadiusA = rad.x;
                RadiusB = rad.y;
                return true;
            }

            public bool Refit(NativeArray<fProxy3> points, in fProxyN w)
            {
                if (!WeightedPlane(points, in w, out fProxy3 c0, out fProxy3 n)) return false;
                OrthoBasis(n, out fProxy3 u, out fProxy3 v);

                var flat = Project(points, c0, u, v);
                var coeffs = new fProxyN(6, Allocator.Temp);

                fProxy2 c2 = default, rad = default;
                fProxy ang = default;
                bool ok = ConicWeighted(flat, in w, ref coeffs);
                if (ok) ok = EllipseFromConic(in coeffs, out c2, out rad, out ang);
                if (ok)
                {
                    Center = c0 + c2.x * u + c2.y * v;
                    Normal = n;
                    AxisA = math.cos(ang) * u + math.sin(ang) * v;
                    RadiusA = rad.x;
                    RadiusB = rad.y;
                }

                flat.Dispose(); coeffs.Dispose();
                return ok;
            }

            // 11 parameters for 8 degrees of freedom: Normal and AxisA are stored as free vectors and
            // re-orthonormalized on Unpack, so the solver never has to respect the constraint itself.
            public int ParamCount => 11;

            public void Pack(ref fProxyN p)
            {
                p[0] = Center.x; p[1] = Center.y; p[2] = Center.z;
                p[3] = Normal.x; p[4] = Normal.y; p[5] = Normal.z;
                p[6] = AxisA.x;  p[7] = AxisA.y;  p[8] = AxisA.z;
                p[9] = RadiusA;  p[10] = RadiusB;
            }

            public void Unpack(in fProxyN p)
            {
                Center = new fProxy3(p[0], p[1], p[2]);
                Normal = math.normalizesafe(new fProxy3(p[3], p[4], p[5]),
                                            new fProxy3((fProxy)0, (fProxy)0, (fProxy)1));

                // Gram-Schmidt AxisA against Normal so the pair stays orthonormal whatever the solver
                // proposed; a component along Normal would otherwise skew both radii.
                fProxy3 a = new fProxy3(p[6], p[7], p[8]);
                a -= math.dot(a, Normal) * Normal;
                if (math.lengthsq(a) <= Consts.fProxyEpsilon) OrthoBasis(Normal, out a, out _);
                AxisA = math.normalizesafe(a);

                RadiusA = math.abs(p[9]);
                RadiusB = math.abs(p[10]);
            }
        }

        // Weighted best-fit plane: centroid and least-variance axis of the weighted covariance.
        static bool WeightedPlane(NativeArray<fProxy3> points, in fProxyN w,
                                  out fProxy3 origin, out fProxy3 normal)
        {
            var flat = points.Reinterpret<fProxy3, fProxy>();
            var mean = new fProxyN(3, Allocator.Temp);
            var C = new fProxyMxN(3, 3, Allocator.Temp);
            var eig = new fProxyN(3, Allocator.Temp);
            var V = new fProxyMxN(3, 3, Allocator.Temp);

            bool ok = SubspaceFitWeighted(flat, points.Length, 3, in w, ref mean, ref C, ref eig, ref V);

            origin = new fProxy3(mean[0], mean[1], mean[2]);
            normal = ok ? new fProxy3(V[0, 2], V[1, 2], V[2, 2]) : default;

            mean.Dispose(); C.Dispose(); eig.Dispose(); V.Dispose();
            return ok;
        }

        // Distance from (x, y), both non-negative, to the axis-aligned ellipse (a, b).
        //
        // The closest point satisfies (a²x/(t+a²), b²y/(t+b²)) for the root of
        // F(t) = (ax/(t+a²))² + (by/(t+b²))² - 1. F is strictly decreasing on t > -min(a,b)², from
        // +infinity down to -1, so the root is unique and bracketable.
        //
        // BRACKETED Newton, not plain Newton: from a seed above the root, an unguarded step can jump
        // below -min(a,b)² where F is not even defined. Keeping the bracket and bisecting whenever a
        // step leaves it makes that impossible, at the cost of a few more iterations.
        static fProxy EllipseDistance2D(fProxy x, fProxy y, fProxy a, fProxy b)
        {
            if (!(a > (fProxy)0) || !(b > (fProxy)0)) return math.sqrt(x * x + y * y);

            fProxy a2 = a * a, b2 = b * b;
            fProxy small = math.min(a2, b2);

            fProxy lo = -small + math.max(small, (fProxy)1) * Consts.fProxySqrtEps;   // F(lo) > 0
            fProxy hi = math.max(a, b) * math.sqrt(x * x + y * y) + math.max(a2, b2); // F(hi) < 0
            fProxy t = (fProxy)0.5 * (lo + hi);

            for (int i = 0; i < 40; i++)
            {
                fProxy ta = t + a2, tb = t + b2;
                fProxy fa = a * x / ta, fb = b * y / tb;
                fProxy f = fa * fa + fb * fb - (fProxy)1;

                if (f > (fProxy)0) lo = t; else hi = t;

                fProxy df = (fProxy)(-2) * (fa * fa / ta + fb * fb / tb);
                fProxy next = math.abs(df) > Consts.fProxyEpsilon ? t - f / df : (fProxy)0.5 * (lo + hi);
                if (!(next > lo) || !(next < hi)) next = (fProxy)0.5 * (lo + hi);

                fProxy step = math.abs(next - t);
                t = next;
                if (step <= Consts.fProxySqrtEps * math.max(math.abs(t), (fProxy)1)) break;
            }

            fProxy cx = a2 * x / (t + a2), cy = b2 * y / (t + b2);
            return math.sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
        }

        // Any orthonormal pair spanning the plane with this normal. Seeded from whichever axis is
        // least aligned with n, so the cross product never collapses.
        internal static void OrthoBasis(fProxy3 n, out fProxy3 u, out fProxy3 v)
        {
            fProxy3 seed = math.abs(n.x) < (fProxy)0.9
                ? new fProxy3((fProxy)1, (fProxy)0, (fProxy)0)
                : new fProxy3((fProxy)0, (fProxy)0, (fProxy)1);
            u = math.normalizesafe(math.cross(n, seed));
            v = math.normalizesafe(math.cross(n, u));
        }

        static NativeArray<fProxy2> Project(NativeArray<fProxy3> pts, fProxy3 origin, fProxy3 u, fProxy3 v)
        {
            var flat = new NativeArray<fProxy2>(pts.Length, Allocator.Temp);
            for (int i = 0; i < pts.Length; i++)
            {
                fProxy3 q = pts[i] - origin;
                flat[i] = new fProxy2(math.dot(q, u), math.dot(q, v));
            }
            return flat;
        }
    }
}
