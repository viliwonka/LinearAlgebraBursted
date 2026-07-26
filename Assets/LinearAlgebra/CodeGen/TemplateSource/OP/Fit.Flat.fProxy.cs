using System;

using Unity.Collections;
using Unity.Mathematics;

using Random = Unity.Mathematics.Random;   // this file imports System, which has its own Random

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
        public struct fProxyCircle3 : IfProxyWeighted3, IfProxyParametric3, IfProxySampleable3
        {
            public fProxy3 Center;
            public fProxy3 Normal;
            public fProxy Radius;

            public int MinimalSamples => 3;      // three points fix the plane and the circumcircle
            public int ParamCount => 7;

            // A circle's arc length is proportional to its angle, so a uniform angle is already
            // uniform arc length -- the one curve here that needs no rejection.
            public fProxy3 Sample(ref Random rng)
            {
                OrthoBasis(Normal, out fProxy3 u, out fProxy3 v);
                fProxy t = rng.NextFProxy((fProxy)0, (fProxy)(2.0 * math.PI_DBL));
                return Center + Radius * (math.cos(t) * u + math.sin(t) * v);
            }

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
                if (points.Length < MinimalSamples) return false;   // see fProxySphere3.Refit
                if (!SubspaceRefit(points, in w, 3, out fProxy3 c0, out fProxy3 n)) return false;
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
        /// no closed form (it needs the root of a quartic), so the in-plane part is found by bracketed
        /// Newton. It is tight everywhere except for points lying almost exactly ON an axis inside the
        /// evolute, where the true closest point is off-axis and the answer errs by roughly the
        /// coordinate floor.
        /// </summary>
        public struct fProxyEllipse3 : IfProxyWeighted3, IfProxyParametric3, IfProxySampleable3
        {
            public fProxy3 Center;
            public fProxy3 Normal;
            public fProxy3 AxisA;
            public fProxy RadiusA;
            public fProxy RadiusB;

            public int MinimalSamples => 5;      // a conic has 5 dof once the plane is fixed

            public fProxy3 Sample(ref Random rng)
            {
                fProxy t = EllipseAngle(ref rng, RadiusA, RadiusB);
                fProxy3 b = math.cross(Normal, AxisA);
                return Center + RadiusA * math.cos(t) * AxisA + RadiusB * math.sin(t) * b;
            }

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
                if (points.Length < MinimalSamples) return false;   // see fProxySphere3.Refit
                if (!SubspaceRefit(points, in w, 3, out fProxy3 c0, out fProxy3 n)) return false;
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

        // Distance from (x, y), both non-negative, to the axis-aligned ellipse (a, b).
        //
        // The closest point is (a²x/(s+da), b²y/(s+db)) for the root of
        // F(s) = (a·x/(s+da))² + (b·y/(s+db))² - 1, where d_i = r_i² - min(a,b)². F falls strictly
        // from +infinity to -1 on s > 0, so the root is unique and bracketable.
        //
        // The classic form of this solve searches t = s - min(a,b)², and the shift is NOT cosmetic:
        // every quantity that matters is (t + min²), a difference of two nearly-equal numbers, so in
        // t the divisors lose most of their significant digits near the ellipse's own centre AND a
        // convergence test scaled by |t| is orders of magnitude looser than the root needs. Searching
        // s directly makes both the divisors and the relative tolerance exact.
        //
        // BRACKETED Newton, not plain Newton: an unguarded step can leave the bracket entirely, so
        // one that does is replaced by a bisection.
        static fProxy EllipseDistance2D(fProxy x, fProxy y, fProxy a, fProxy b)
        {
            if (!(a > (fProxy)0) || !(b > (fProxy)0)) return math.sqrt(x * x + y * y);

            fProxy a2 = a * a, b2 = b * b;
            fProxy small = math.min(a2, b2), big = math.max(a, b);

            // A coordinate of exactly zero kills its term, and with it the +infinity end of the
            // bracket: the search would then run to s = 0 and report the CENTRE as lying ON the
            // ellipse -- distance zero, so a dead-centre outlier would score as a perfect inlier.
            // Flooring each coordinate keeps the root real, at a cost of about the floor for points
            // sitting on an axis.
            // PER-AXIS floor, each scaled by its own radius. One shared floor taken from the largest
            // radius would raise a legitimate small coordinate on the OTHER axis: on a 1e7:1 ellipse
            // it lifts the minor-axis vertex y = 1 up to 3452, and the vertex then measures thousands
            // of units from the curve it lies on.
            x = math.max(x, a * Consts.fProxySqrtEps);
            y = math.max(y, b * Consts.fProxySqrtEps);

            fProxy da = a2 - small, db = b2 - small;

            // Bracket both ends from the root condition itself, rather than 0 and a crude upper bound.
            // The two terms of F sum to 1 at the root, so each is <= 1 (giving s + d_i >= r_i·q_i) and
            // the larger is >= 1/2 (giving s + d_i <= sqrt(2)·r_i·q_i for that axis). Between them the
            // bracket is a factor of sqrt(2) wide instead of spanning to big².
            //
            // This is not a micro-optimization. Far above the root F is flat at -1, so every Newton
            // step overshoots the bracket and degrades to bisection; a bracket spanning big² then
            // needs log2(big²/s*) halvings, which past roughly a 1e6 aspect ratio exceeds the
            // iteration cap and returns a large distance for a point sitting ON the ellipse.
            fProxy lo = math.max((fProxy)0, math.max(a * x - da, b * y - db));
            fProxy hi = math.max(Consts.fProxySqrt2 * a * x - da, Consts.fProxySqrt2 * b * y - db);
            fProxy s = (fProxy)0.5 * hi;

            for (int i = 0; i < 40; i++)
            {
                fProxy sa = s + da, sb = s + db;
                fProxy fa = a * x / sa, fb = b * y / sb;
                fProxy f = fa * fa + fb * fb - (fProxy)1;

                if (f > (fProxy)0) lo = s; else hi = s;

                fProxy df = (fProxy)(-2) * (fa * fa / sa + fb * fb / sb);
                fProxy next = math.abs(df) > Consts.fProxyEpsilon ? s - f / df : (fProxy)0.5 * (lo + hi);
                if (!(next > lo) || !(next < hi)) next = (fProxy)0.5 * (lo + hi);

                fProxy step = math.abs(next - s);
                s = next;
                if (step <= Consts.fProxySqrtEps * s) break;
            }

            fProxy cx = a2 * x / (s + da), cy = b2 * y / (s + db);
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
