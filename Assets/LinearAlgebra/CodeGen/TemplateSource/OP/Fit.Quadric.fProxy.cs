using System;

using Unity.Collections;
using Unity.Mathematics;

//+deleteThis
// TEMPLATE-ONLY alias: codegen rewrites each fProxy* token -> float*/double* (real Unity.Mathematics
// types), so the field access below (.x/.y/.z) and constructors resolve natively.
using fProxy2 = Unity.Mathematics.float2;
using fProxy3 = Unity.Mathematics.float3;
//-deleteThis

namespace BULA
{
    // ================================================================================================
    // Conic (2D) and quadric (3D) fitting -- the shape families whose ALGEBRAIC fit is linear in the
    // coefficients, so one factorization answers them all. A cone, a hyperboloid, an ellipsoid and a
    // paraboloid are the same fit with different coefficient signatures, which is why there is one
    // `quadric` entry point plus a classifier rather than a solver per shape.
    //
    // Both minimize the ALGEBRAIC residual (how far the implicit equation is from zero), not the
    // geometric distance to the surface -- points far from the surface carry more weight than their
    // true distance warrants. For a geometric fit, seed Optimize.nlsSolve from these coefficients.
    //
    // Job-safe: scratch is Allocator.Temp, disposed before returning.
    // ================================================================================================
    public static partial class Fit
    {
        /// <summary>
        /// Fits a general conic A·x² + B·x·y + C·y² + D·x + E·y + F = 0 to a 2D point cloud,
        /// CONSTRAINED to an ellipse (Halir &amp; Flusser 1998, the numerically stable form of
        /// Fitzgibbon's direct ellipse fit). The constraint 4AC − B² = 1 makes an ellipse the only
        /// possible answer, so a noisy near-parabolic cloud returns a valid (if elongated) ellipse
        /// rather than silently fitting a hyperbola.
        ///
        /// <paramref name="coeffs"/> receives (A, B, C, D, E, F), length 6, scale-normalized.
        /// Requires points.Length &gt;= 5 (a conic has 5 degrees of freedom). False means the
        /// factorization failed or no constraint-satisfying solution was found -- typically a
        /// degenerate cloud (all points collinear).
        /// </summary>
        public static bool conic(NativeArray<fProxy2> points, ref fProxyN coeffs)
            => ConicWeighted(points, default(fProxyN), ref coeffs);

        /// <summary>
        /// Ellipse fit under <paramref name="loss"/>, by IRLS on the SAMPSON distance -- the
        /// first-order approximation |F(p)| / ‖∇F(p)‖ to true geometric distance. Reweighting by the
        /// raw algebraic residual instead would be wrong in a way that matters: that residual scales
        /// with the local gradient, so it is larger in high-curvature regions for the SAME true
        /// distance, and a loss keyed on it would reject points partly for where they sit on the curve
        /// rather than for being outliers.
        ///
        /// Because of that, passing <see cref="fProxyL2Loss"/> here is NOT a no-op: it still trades
        /// the algebraic residual for a geometric one and removes most of the plain fit's bias, the
        /// same way <see cref="sphere{TLoss}(NativeArray{fProxy2}, in TLoss, out fProxy2, out fProxy, int)"/>
        /// does. Requires points.Length &gt;= 5.
        /// </summary>
        public static bool conic<TLoss>(NativeArray<fProxy2> points, in TLoss loss, ref fProxyN coeffs,
                                        int maxIter = 0)
            where TLoss : struct, IfProxyRobustLoss
        {
            int n = points.Length;
            if (maxIter <= 0) maxIter = DefaultIrlsIter;
            if (!conic(points, ref coeffs)) return false;

            var w = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) w[i] = (fProxy)1;

            bool ok = true;
            for (int it = 0; it < maxIter; it++)
            {
                fProxy maxDelta = (fProxy)0, maxW = (fProxy)0, sw = (fProxy)0;
                for (int i = 0; i < n; i++)
                {
                    fProxy d = SampsonConic(in coeffs, points[i], out fProxy g2);
                    fProxy wNew = loss.RhoPrime(d * d) / g2;
                    maxDelta = math.max(maxDelta, math.abs(wNew - w[i]));
                    maxW = math.max(maxW, wNew);
                    w[i] = wNew;
                    sw += wNew;
                }

                if (!(sw > (fProxy)0)) { ok = false; break; }   // redescending loss rejected everything
                if (maxDelta <= Consts.fProxySqrtEps * math.max(maxW, (fProxy)1)) break;

                ok = ConicWeighted(points, in w, ref coeffs);
                if (!ok) break;
            }

            w.Dispose();
            return ok;
        }

        static bool ConicWeighted(NativeArray<fProxy2> points, in fProxyN w, ref fProxyN coeffs)
        {
            int n = points.Length;
            if (n < 5) throw new ArgumentException("Fit.conic: points.Length must be >= 5");
            if (coeffs.N != 6) throw new ArgumentException("Fit.conic: coeffs.N must be 6");
            bool weighted = w.IsCreated;

            // Halir-Flusser splits the design into the quadratic part D1 = [x², xy, y²] and the linear
            // part D2 = [x, y, 1], so the 3x3 blocks below stay well scaled where the raw 6x6 scatter
            // matrix of Fitzgibbon's original form does not.
            var S1 = new fProxyMxN(3, 3, Allocator.Temp);
            var S2 = new fProxyMxN(3, 3, Allocator.Temp);
            var S3 = new fProxyMxN(3, 3, Allocator.Temp);
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++) { S1[a, b] = (fProxy)0; S2[a, b] = (fProxy)0; S3[a, b] = (fProxy)0; }

            // Hartley normalization, for the same reason the 3D ellipsoid route has it: the design
            // entries are FOURTH powers of the coordinates, so a cloud sitting even a few units off
            // the origin conditions the scatter blocks past what float can carry. A unit circle
            // centred at (500, 500) faces a scatter conditioned like offset^4. Weighted, to match the
            // fit underneath -- an unweighted centroid would let outliers set the scale under IRLS.
            fProxy2 org = default;
            fProxy wsum = (fProxy)0;
            for (int i = 0; i < n; i++)
            {
                fProxy wi = weighted ? math.max(w[i], (fProxy)0) : (fProxy)1;
                org += wi * points[i];
                wsum += wi;
            }
            if (!(wsum > (fProxy)0)) { S1.Dispose(); S2.Dispose(); S3.Dispose(); return false; }
            org /= wsum;

            fProxy acc = (fProxy)0;
            for (int i = 0; i < n; i++)
            {
                fProxy wi = weighted ? math.max(w[i], (fProxy)0) : (fProxy)1;
                acc += wi * math.lengthsq(points[i] - org);
            }
            fProxy scale = math.sqrt(acc / wsum);
            if (!(scale > (fProxy)0)) { S1.Dispose(); S2.Dispose(); S3.Dispose(); return false; }
            fProxy invScale = (fProxy)1 / scale;

            var q = new fProxyN(3, Allocator.Temp);
            var l = new fProxyN(3, Allocator.Temp);
            for (int i = 0; i < n; i++)
            {
                fProxy2 p = (points[i] - org) * invScale;
                fProxy x = p.x, y = p.y;
                q[0] = x * x; q[1] = x * y; q[2] = y * y;
                l[0] = x;     l[1] = y;     l[2] = (fProxy)1;

                fProxy wi = weighted ? math.max(w[i], (fProxy)0) : (fProxy)1;
                for (int a = 0; a < 3; a++)
                {
                    for (int b = 0; b < 3; b++)
                    {
                        S1[a, b] += wi * q[a] * q[b];
                        S2[a, b] += wi * q[a] * l[b];
                        S3[a, b] += wi * l[a] * l[b];
                    }
                }
            }

            // T = -S3^-1 S2^T, M = S1 + S2 T, then premultiply by the inverse of the constraint block
            // C1 = [[0,0,2],[0,-1,0],[2,0,0]], whose inverse is [[0,0,0.5],[0,-1,0],[0.5,0,0]].
            var T = new fProxyMxN(3, 3, Allocator.Temp);
            bool ok = SolveNegS3(in S3, in S2, ref T);

            var M = new fProxyMxN(3, 3, Allocator.Temp);
            var Mp = new fProxyMxN(3, 3, Allocator.Temp);
            if (ok)
            {
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++)
                    {
                        fProxy s = S1[a, b];
                        for (int k = 0; k < 3; k++) s += S2[a, k] * T[k, b];
                        M[a, b] = s;
                    }

                for (int b = 0; b < 3; b++)
                {
                    Mp[0, b] = (fProxy)0.5 * M[2, b];
                    Mp[1, b] = -M[1, b];
                    Mp[2, b] = (fProxy)0.5 * M[0, b];
                }

                ok = ConicEigenvector(in Mp, ref coeffs, in T);
                if (ok) DenormalizeConic(ref coeffs, org, scale);
            }

            q.Dispose(); l.Dispose();
            S1.Dispose(); S2.Dispose(); S3.Dispose(); T.Dispose(); M.Dispose(); Mp.Dispose();
            return ok;
        }

        // Undoes the fit's x' = (x - org) / scale substitution. Multiplying the conic through by
        // scale² clears every denominator and coefficients are defined only up to overall scale, so
        // no divisions are needed. The QUADRATIC block comes through untouched, which is why the
        // ellipse constraint 4AC - B² survives the map.
        static void DenormalizeConic(ref fProxyN c, fProxy2 org, fProxy scale)
        {
            fProxy A = c[0], B = c[1], C = c[2];
            fProxy tx = org.x, ty = org.y;

            fProxy qx = A * tx + (fProxy)0.5 * B * ty;          // (Q·org)_x
            fProxy qy = (fProxy)0.5 * B * tx + C * ty;          // (Q·org)_y
            fProxy orgQorg = tx * qx + ty * qy;
            fProxy gt = c[3] * tx + c[4] * ty;

            c[3] = (fProxy)(-2) * qx + scale * c[3];
            c[4] = (fProxy)(-2) * qy + scale * c[4];
            c[5] = orgQorg - scale * gt + scale * scale * c[5];
        }

        /// <summary>
        /// Fits an ellipse to a 2D point cloud and reports it geometrically: <paramref name="center"/>,
        /// <paramref name="radii"/> (semi-axis lengths, x = the axis along <paramref name="angle"/>)
        /// and <paramref name="angle"/> in radians. Wraps <see cref="conic"/>, so the same
        /// ellipse-only constraint and point-count requirement apply. False means the conic fit failed
        /// or its coefficients do not describe a real ellipse.
        /// </summary>
        public static bool ellipse(NativeArray<fProxy2> points, out fProxy2 center,
                                   out fProxy2 radii, out fProxy angle)
        {
            center = default; radii = default; angle = (fProxy)0;

            var c = new fProxyN(6, Allocator.Temp);
            bool ok = conic(points, ref c) && EllipseFromConic(in c, out center, out radii, out angle);
            c.Dispose();
            return ok;
        }

        /// <summary>
        /// Geometry of the ellipse described by conic coefficients (A..F): centre, semi-axes, and the
        /// rotation of the axis whose length is radii.x. False when they do not describe a real
        /// ellipse. Shared by <see cref="ellipse"/> and the flat 3D ellipse's weighted refit.
        /// </summary>
        internal static bool EllipseFromConic(in fProxyN c, out fProxy2 center,
                                              out fProxy2 radii, out fProxy angle)
        {
            center = default; radii = default; angle = (fProxy)0;

            fProxy A = c[0], B = c[1], C = c[2], D = c[3], E = c[4], F = c[5];

            fProxy det = (fProxy)4 * A * C - B * B;
            if (math.abs(det) <= Consts.fProxyEpsilon) return false;

            fProxy cx = ((fProxy)2 * C * (-D) - B * (-E)) / det;
            fProxy cy = ((fProxy)2 * A * (-E) - B * (-D)) / det;
            center = new fProxy2(cx, cy);

            fProxy Fc = F + (fProxy)0.5 * (D * cx + E * cy);

            var Q = new fProxyMxN(2, 2, Allocator.Temp);
            var ev = new fProxyN(2, Allocator.Temp);
            var V = new fProxyMxN(2, 2, Allocator.Temp);
            Q[0, 0] = A; Q[0, 1] = (fProxy)0.5 * B;
            Q[1, 0] = (fProxy)0.5 * B; Q[1, 1] = C;

            bool ok = Eigen.symmetricInPlace(ref Q, ref ev, ref V);
            if (ok)
            {
                fProxy r0 = -Fc / ev[0], r1 = -Fc / ev[1];
                if (r0 > (fProxy)0 && r1 > (fProxy)0)
                {
                    radii = new fProxy2(math.sqrt(r0), math.sqrt(r1));
                    angle = math.atan2(V[1, 0], V[0, 0]);
                }
                else ok = false;
            }

            Q.Dispose(); ev.Dispose(); V.Dispose();
            return ok;
        }

        /// <summary>
        /// Fits a general quadric surface to a 3D point cloud:
        /// A·x² + B·y² + C·z² + D·xy + E·xz + F·yz + G·x + H·y + I·z + J = 0, by minimizing the
        /// algebraic residual subject to ‖coefficients‖ = 1 (the smallest right singular vector of the
        /// 10-column design). UNCONSTRAINED as to type -- the result may be an ellipsoid, hyperboloid,
        /// cone, paraboloid or a degenerate form; pass it to <see cref="classify"/> to find out which.
        ///
        /// <paramref name="coeffs"/> receives (A..J), length 10, unit norm. Requires points.Length
        /// &gt;= 9. False means the SVD did not converge.
        /// </summary>
        public static bool quadric(NativeArray<fProxy3> points, ref fProxyN coeffs)
            => QuadricWeighted(points, default(fProxyN), ref coeffs);

        /// <summary>
        /// Quadric fit under <paramref name="loss"/>, by IRLS on the SAMPSON distance
        /// |F(p)| / ‖∇F(p)‖. See <see cref="conic{TLoss}"/> for why the raw algebraic residual is the
        /// wrong thing to reweight by, and why <see cref="fProxyL2Loss"/> is not a no-op here.
        /// Requires points.Length &gt;= 9.
        /// </summary>
        public static bool quadric<TLoss>(NativeArray<fProxy3> points, in TLoss loss, ref fProxyN coeffs,
                                          int maxIter = 0)
            where TLoss : struct, IfProxyRobustLoss
            => QuadricIrls(points, in loss, ref coeffs, maxIter, ellipsoidOnly: false);

        // Sampson IRLS over the 3D coefficient families, shared by `quadric` and `ellipsoid`: their
        // coefficients and their Sampson distance are the same, only the weighted fit underneath
        // differs. One loop, so the collapse guard has exactly one home.
        static bool QuadricIrls<TLoss>(NativeArray<fProxy3> points, in TLoss loss, ref fProxyN coeffs,
                                       int maxIter, bool ellipsoidOnly)
            where TLoss : struct, IfProxyRobustLoss
        {
            int n = points.Length;
            if (maxIter <= 0) maxIter = DefaultIrlsIter;

            bool seeded = ellipsoidOnly ? ellipsoid(points, ref coeffs) : quadric(points, ref coeffs);
            if (!seeded) return false;

            var w = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) w[i] = (fProxy)1;

            bool ok = true;
            for (int it = 0; it < maxIter; it++)
            {
                fProxy maxDelta = (fProxy)0, maxW = (fProxy)0, sw = (fProxy)0;
                for (int i = 0; i < n; i++)
                {
                    fProxy d = SampsonQuadric(in coeffs, points[i], out fProxy g2);
                    fProxy wNew = loss.RhoPrime(d * d) / g2;
                    maxDelta = math.max(maxDelta, math.abs(wNew - w[i]));
                    maxW = math.max(maxW, wNew);
                    w[i] = wNew;
                    sw += wNew;
                }

                if (!(sw > (fProxy)0)) { ok = false; break; }   // redescending loss rejected everything
                if (maxDelta <= Consts.fProxySqrtEps * math.max(maxW, (fProxy)1)) break;

                ok = ellipsoidOnly ? EllipsoidWeighted(points, in w, ref coeffs)
                                   : QuadricWeighted(points, in w, ref coeffs);
                if (!ok) break;
            }

            w.Dispose();
            return ok;
        }

        static bool QuadricWeighted(NativeArray<fProxy3> points, in fProxyN w, ref fProxyN coeffs)
        {
            int n = points.Length;
            // 10, not 9: the design has one column per coefficient and SVD.thin needs at least as many
            // rows as columns. A 9-point call used to reach the SVD and throw from inside it.
            if (n < 10) throw new ArgumentException("Fit.quadric: points.Length must be >= 10");
            if (coeffs.N != 10) throw new ArgumentException("Fit.quadric: coeffs.N must be 10");
            bool weighted = w.IsCreated;

            var Dm = new fProxyMxN(n, 10, Allocator.Temp);
            for (int i = 0; i < n; i++)
            {
                fProxy x = points[i].x, y = points[i].y, z = points[i].z;
                fProxy s = weighted ? math.sqrt(math.max(w[i], (fProxy)0)) : (fProxy)1;
                Dm[i, 0] = s * x * x; Dm[i, 1] = s * y * y; Dm[i, 2] = s * z * z;
                Dm[i, 3] = s * x * y; Dm[i, 4] = s * x * z; Dm[i, 5] = s * y * z;
                Dm[i, 6] = s * x;     Dm[i, 7] = s * y;     Dm[i, 8] = s * z;
                Dm[i, 9] = s;
            }

            var U = new fProxyMxN(n, 10, Allocator.Temp);
            var S = new fProxyN(10, Allocator.Temp);
            var V = new fProxyMxN(10, 10, Allocator.Temp);

            bool ok = SVD.thin(in Dm, ref U, ref S, ref V);
            if (ok)
                for (int j = 0; j < 10; j++) coeffs[j] = V[j, 9];   // smallest singular value

            Dm.Dispose(); U.Dispose(); S.Dispose(); V.Dispose();
            return ok;
        }

        /// <summary>
        /// Classifies quadric coefficients from <see cref="quadric"/> by the eigenvalue signature of
        /// their 3x3 quadratic form. Returns <see cref="QuadricKind.Unknown"/> if the eigensolve fails.
        /// The zero test is relative to the largest |eigenvalue|, so classification is scale-invariant;
        /// a surface near a type boundary (a very flat ellipsoid, a nearly-degenerate cone) can still
        /// land either side of it, which is a property of the data rather than of this test.
        /// </summary>
        public static QuadricKind classify(in fProxyN coeffs)
        {
            if (coeffs.N != 10) throw new ArgumentException("Fit.classify: coeffs.N must be 10");

            var Q = new fProxyMxN(3, 3, Allocator.Temp);
            var ev = new fProxyN(3, Allocator.Temp);
            var V = new fProxyMxN(3, 3, Allocator.Temp);

            Q[0, 0] = coeffs[0];                    Q[1, 1] = coeffs[1];                    Q[2, 2] = coeffs[2];
            Q[0, 1] = (fProxy)0.5 * coeffs[3];      Q[1, 0] = Q[0, 1];
            Q[0, 2] = (fProxy)0.5 * coeffs[4];      Q[2, 0] = Q[0, 2];
            Q[1, 2] = (fProxy)0.5 * coeffs[5];      Q[2, 1] = Q[1, 2];

            QuadricKind kind = QuadricKind.Unknown;
            if (Eigen.symmetricInPlace(ref Q, ref ev, ref V))
            {
                fProxy scale = math.max(math.max(math.abs(ev[0]), math.abs(ev[1])), math.abs(ev[2]));

                // Purely RELATIVE to the largest eigenvalue. Flooring the scale at 1 would make this
                // an absolute test whenever every eigenvalue is below 1 -- which is the normal case,
                // since `quadric` returns unit-norm coefficients and a sphere of radius R has
                // quadratic entries of order 1/R². A large sphere would then classify as Degenerate.
                fProxy zero = scale * Consts.fProxySqrtEps;

                int pos = 0, neg = 0, nul = 0;
                for (int i = 0; i < 3; i++)
                {
                    if (ev[i] > zero) pos++;
                    else if (ev[i] < -zero) neg++;
                    else nul++;
                }

                if (nul == 3) kind = QuadricKind.Degenerate;          // no quadratic part left: a plane
                else if (nul > 0) kind = QuadricKind.Paraboloid;      // a zero eigenvalue: no centre
                else if (pos == 3 || neg == 3) kind = QuadricKind.Ellipsoid;
                else kind = QuadricKind.HyperboloidOrCone;            // mixed signature
            }

            Q.Dispose(); ev.Dispose(); V.Dispose();
            return kind;
        }

        // ---- Sampson distance ----------------------------------------------------------------------
        //
        // |F(p)| / ‖∇F(p)‖ -- the algebraic residual divided by how fast F changes there, which is a
        // first-order estimate of the true orthogonal distance to the surface. Exact for a plane,
        // and accurate wherever the point is close relative to the local curvature. The gradient is
        // linear in the point for both families, so this costs a handful of multiplies.
        //
        // The guard matters: ∇F vanishes at a quadric's centre (and along a degenerate one's axis),
        // where no first-order distance exists. Returning the raw |F| there keeps the weight finite
        // and errs toward KEEPING the point, which is the safe direction -- discarding points because
        // the model happens to be singular near them would let a bad fit defend itself.

        static fProxy SampsonConic(in fProxyN c, fProxy2 p, out fProxy grad2)
        {
            fProxy A = c[0], B = c[1], C = c[2], D = c[3], E = c[4], F = c[5];
            fProxy x = p.x, y = p.y;

            fProxy val = A * x * x + B * x * y + C * y * y + D * x + E * y + F;
            fProxy gx = (fProxy)2 * A * x + B * y + D;
            fProxy gy = B * x + (fProxy)2 * C * y + E;

            fProxy g2 = gx * gx + gy * gy;
            grad2 = math.max(g2, Consts.fProxyEpsilon);

            fProxy g = math.sqrt(g2);
            return g > Consts.fProxySqrtEps ? math.abs(val) / g : math.abs(val);
        }

        static fProxy SampsonQuadric(in fProxyN c, fProxy3 p, out fProxy grad2)
        {
            fProxy A = c[0], B = c[1], C = c[2], D = c[3], E = c[4], F = c[5];
            fProxy G = c[6], H = c[7], I = c[8], J = c[9];
            fProxy x = p.x, y = p.y, z = p.z;

            fProxy val = A * x * x + B * y * y + C * z * z
                       + D * x * y + E * x * z + F * y * z
                       + G * x + H * y + I * z + J;

            fProxy gx = (fProxy)2 * A * x + D * y + E * z + G;
            fProxy gy = (fProxy)2 * B * y + D * x + F * z + H;
            fProxy gz = (fProxy)2 * C * z + E * x + F * y + I;

            fProxy g2 = gx * gx + gy * gy + gz * gz;
            grad2 = math.max(g2, Consts.fProxyEpsilon);

            fProxy g = math.sqrt(g2);
            return g > Consts.fProxySqrtEps ? math.abs(val) / g : math.abs(val);
        }

        // T = -S3^-1 S2^T, by solving S3 T = -S2^T column by column.
        static bool SolveNegS3(in fProxyMxN S3, in fProxyMxN S2, ref fProxyMxN T)
        {
            var Lu = new fProxyMxN(3, 3, Allocator.Temp);
            var rhs = new fProxyN(3, Allocator.Temp);
            var sol = new fProxyN(3, Allocator.Temp);

            bool ok = true;
            for (int col = 0; col < 3 && ok; col++)
            {
                for (int a = 0; a < 3; a++)
                {
                    for (int b = 0; b < 3; b++) Lu[a, b] = S3[a, b];
                    rhs[a] = -S2[col, a];                    // row `col` of S2 is column `col` of S2^T
                }

                // Rank-revealing: S3 is the Gram of (x, y, 1) and goes exactly singular on a collinear
                // cloud. An un-pivoted solve would report success and hand back Inf/NaN.
                ok = QRCP.solveInPlace(ref Lu, ref rhs, ref sol).rank == 3;
                if (ok) for (int a = 0; a < 3; a++) T[a, col] = sol[a];
            }

            Lu.Dispose(); rhs.Dispose(); sol.Dispose();
            return ok;
        }

        // Picks the eigenvector of the 3x3 (NON-SYMMETRIC) Mp satisfying 4·a1·a3 - a2² > 0 -- the one
        // that is an ellipse -- and expands it to the full 6 conic coefficients via a2 = T·a1.
        //
        // This library has no general nonsymmetric eigenVECTOR solver (see the GCRO-DR note in
        // OP/DEVLOG.md), so the vector for each eigenvalue is recovered as the null space of
        // (Mp - lambda·I): its smallest right singular vector. At 3x3 that is cheap and, unlike an
        // inverse-iteration solve, needs no guard against the exactly-singular matrix it is handed
        // BY CONSTRUCTION.
        static bool ConicEigenvector(in fProxyMxN Mp, ref fProxyN coeffs, in fProxyMxN T)
        {
            var work = new fProxyMxN(3, 3, Allocator.Temp);
            var re = new fProxyN(3, Allocator.Temp);
            var im = new fProxyN(3, Allocator.Temp);

            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++) work[a, b] = Mp[a, b];

            bool ok = Eigen.valuesQRInPlace(ref work, ref re, ref im);

            var shifted = new fProxyMxN(3, 3, Allocator.Temp);
            var U = new fProxyMxN(3, 3, Allocator.Temp);
            var S = new fProxyN(3, Allocator.Temp);
            var V = new fProxyMxN(3, 3, Allocator.Temp);

            bool found = false;
            if (ok)
            {
                for (int e = 0; e < 3 && !found; e++)
                {
                    if (math.abs(im[e]) > Consts.fProxySqrtEps) continue;   // complex pair: not our root

                    for (int a = 0; a < 3; a++)
                        for (int b = 0; b < 3; b++)
                            shifted[a, b] = Mp[a, b] - (a == b ? re[e] : (fProxy)0);

                    if (!SVD.thin(in shifted, ref U, ref S, ref V)) continue;

                    fProxy a1 = V[0, 2], a2 = V[1, 2], a3 = V[2, 2];        // null vector
                    if ((fProxy)4 * a1 * a3 - a2 * a2 > (fProxy)0)
                    {
                        coeffs[0] = a1; coeffs[1] = a2; coeffs[2] = a3;
                        for (int r = 0; r < 3; r++)
                            coeffs[3 + r] = T[r, 0] * a1 + T[r, 1] * a2 + T[r, 2] * a3;
                        found = true;
                    }
                }
            }

            work.Dispose(); re.Dispose(); im.Dispose();
            shifted.Dispose(); U.Dispose(); S.Dispose(); V.Dispose();
            return found;
        }
    }
}
