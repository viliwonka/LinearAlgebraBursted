using System;

using Unity.Collections;
using Unity.Mathematics;

//+deleteThis
// TEMPLATE-ONLY alias: codegen rewrites each fProxy* token -> float*/double* (real Unity.Mathematics
// types), so the field access below (.x/.y/.z/.w) and constructors resolve natively.
using fProxy2 = Unity.Mathematics.float2;
using fProxy3 = Unity.Mathematics.float3;
using fProxy4 = Unity.Mathematics.float4;
//-deleteThis

namespace BULA
{
    // ================================================================================================
    // Circle / sphere / hypersphere fitting.
    //
    // The plain overloads are ALGEBRAIC (Kasa): |p-c|² = r² is linear in (c, u) for u = |c|²-r², so
    // one least-squares solve gives the fit outright, in any dimension. That minimizes the algebraic
    // residual |p-c|²-r², NOT the geometric distance ||p-c|-r|, so it is slightly biased when the
    // points cover only a short arc -- points far from the surface are weighted by roughly 2r.
    //
    // The loss overloads run IRLS on the TRUE geometric residual, which removes that bias as a side
    // effect of reweighting; pass fProxyL2Loss to get geometric (unbiased) least squares, or a robust
    // loss to also reject outliers.
    //
    // Job-safe: scratch is Allocator.Temp, disposed before returning.
    // ================================================================================================
    public static partial class Fit
    {
        /// <summary>
        /// Best-fit circle through a 2D point cloud (algebraic). Requires points.Length &gt;= 3.
        /// False means the least-squares solve failed or the fit implies a negative squared radius
        /// (collinear or degenerate input); center/radius are then undefined.
        /// </summary>
        public static bool sphere(NativeArray<fProxy2> points, out fProxy2 center, out fProxy radius)
        {
            if (points.Length < 3) throw new ArgumentException("Fit.sphere: points.Length must be >= 3 in 2D");

            var c = new fProxyN(2, Allocator.Temp);
            bool ok = SphereAlgebraic(points.Reinterpret<fProxy2, fProxy>(), points.Length, 2, default(fProxyN), ref c, out radius);
            center = ok ? new fProxy2(c[0], c[1]) : default;
            c.Dispose();
            return ok;
        }

        /// <summary>
        /// Best-fit sphere through a 3D point cloud (algebraic). Requires points.Length &gt;= 4.
        /// See the 2D overload for the failure contract.
        /// </summary>
        public static bool sphere(NativeArray<fProxy3> points, out fProxy3 center, out fProxy radius)
        {
            if (points.Length < 4) throw new ArgumentException("Fit.sphere: points.Length must be >= 4 in 3D");

            var c = new fProxyN(3, Allocator.Temp);
            bool ok = SphereAlgebraic(points.Reinterpret<fProxy3, fProxy>(), points.Length, 3, default(fProxyN), ref c, out radius);
            center = ok ? new fProxy3(c[0], c[1], c[2]) : default;
            c.Dispose();
            return ok;
        }

        /// <summary>
        /// Best-fit circle through a 2D point cloud under <paramref name="loss"/>, by IRLS on the
        /// geometric residual ||p-c| - r|, started from the algebraic fit. Requires points.Length
        /// &gt;= 3. <paramref name="maxIter"/> &lt;= 0 picks a default budget.
        /// </summary>
        public static bool sphere<TLoss>(NativeArray<fProxy2> points, in TLoss loss,
                                         out fProxy2 center, out fProxy radius, int maxIter = 0)
            where TLoss : struct, IfProxyRobustLoss
        {
            if (points.Length < 3) throw new ArgumentException("Fit.sphere: points.Length must be >= 3 in 2D");

            var c = new fProxyN(2, Allocator.Temp);
            bool ok = SphereIrls(points.Reinterpret<fProxy2, fProxy>(), points.Length, 2, in loss, maxIter, ref c, out radius);
            center = ok ? new fProxy2(c[0], c[1]) : default;
            c.Dispose();
            return ok;
        }

        /// <summary>Robust best-fit sphere through a 3D point cloud. See the 2D overload.</summary>
        public static bool sphere<TLoss>(NativeArray<fProxy3> points, in TLoss loss,
                                         out fProxy3 center, out fProxy radius, int maxIter = 0)
            where TLoss : struct, IfProxyRobustLoss
        {
            if (points.Length < 4) throw new ArgumentException("Fit.sphere: points.Length must be >= 4 in 3D");

            var c = new fProxyN(3, Allocator.Temp);
            bool ok = SphereIrls(points.Reinterpret<fProxy3, fProxy>(), points.Length, 3, in loss, maxIter, ref c, out radius);
            center = ok ? new fProxy3(c[0], c[1], c[2]) : default;
            c.Dispose();
            return ok;
        }

        // ============================================================================================
        // One weighted algebraic solve: rows 2·p_i scaled by sqrt(w_i) against |p_i|², solving for
        // (c, u) with u = |c|² - r². `w` may be default/uncreated, meaning unit weights.
        // ============================================================================================
        static bool SphereAlgebraic(NativeArray<fProxy> flat, int n, int d,
                                    in fProxyN w, ref fProxyN center, out fProxy radius)
        {
            radius = (fProxy)0;
            bool weighted = w.IsCreated;

            var A = new fProxyMxN(n, d + 1, Allocator.Temp);
            var b = new fProxyN(n, Allocator.Temp);
            var x = new fProxyN(d + 1, Allocator.Temp);

            for (int i = 0; i < n; i++)
            {
                fProxy s = weighted ? math.sqrt(math.max(w[i], (fProxy)0)) : (fProxy)1;
                fProxy sq = (fProxy)0;
                for (int j = 0; j < d; j++)
                {
                    fProxy v = flat[i * d + j];
                    A[i, j] = s * (fProxy)2 * v;
                    sq += v * v;
                }
                A[i, d] = -s;
                b[i] = s * sq;
            }

            bool ok = QR.solveInPlace(ref A, ref b, ref x);   // DESTROYS A and b
            if (ok)
            {
                fProxy c2 = (fProxy)0;
                for (int j = 0; j < d; j++) { center[j] = x[j]; c2 += x[j] * x[j]; }
                fProxy r2 = c2 - x[d];
                if (r2 >= (fProxy)0) radius = math.sqrt(r2);
                else ok = false;                              // no real radius: degenerate input
            }

            A.Dispose(); b.Dispose(); x.Dispose();
            return ok;
        }

        // IRLS on the geometric residual, seeded from the algebraic fit.
        static bool SphereIrls<TLoss>(NativeArray<fProxy> flat, int n, int d, in TLoss loss,
                                      int maxIter, ref fProxyN center, out fProxy radius)
            where TLoss : struct, IfProxyRobustLoss
        {
            if (maxIter <= 0) maxIter = DefaultIrlsIter;

            if (!SphereAlgebraic(flat, n, d, default(fProxyN), ref center, out radius))
                return false;

            var w = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) w[i] = (fProxy)1;

            bool ok = true;
            for (int it = 0; it < maxIter; it++)
            {
                fProxy maxDelta = (fProxy)0, sw = (fProxy)0;
                for (int i = 0; i < n; i++)
                {
                    fProxy dist2 = (fProxy)0;
                    for (int j = 0; j < d; j++)
                    {
                        fProxy q = flat[i * d + j] - center[j];
                        dist2 += q * q;
                    }
                    fProxy res = math.sqrt(dist2) - radius;
                    fProxy wNew = loss.RhoPrime(res * res);
                    maxDelta = math.max(maxDelta, math.abs(wNew - w[i]));
                    w[i] = wNew;
                    sw += wNew;
                }

                // A redescending loss can zero every weight, leaving an all-zero design whose solve
                // returns NaN. Reporting that as success would be a false certificate -- the same
                // collapse the other IRLS loops guard, which this one was missing.
                if (!(sw > (fProxy)0)) { ok = false; break; }
                if (maxDelta <= Consts.fProxySqrtEps) break;

                ok = SphereAlgebraic(flat, n, d, in w, ref center, out radius);
                if (!ok) break;
            }

            w.Dispose();
            return ok;
        }
    }
}
