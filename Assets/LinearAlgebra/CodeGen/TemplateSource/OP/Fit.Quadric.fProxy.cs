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
        {
            int n = points.Length;
            if (n < 5) throw new ArgumentException("Fit.conic: points.Length must be >= 5");
            if (coeffs.N != 6) throw new ArgumentException("Fit.conic: coeffs.N must be 6");

            // Halir-Flusser splits the design into the quadratic part D1 = [x², xy, y²] and the linear
            // part D2 = [x, y, 1], so the 3x3 blocks below stay well scaled where the raw 6x6 scatter
            // matrix of Fitzgibbon's original form does not.
            var S1 = new fProxyMxN(3, 3, Allocator.Temp);
            var S2 = new fProxyMxN(3, 3, Allocator.Temp);
            var S3 = new fProxyMxN(3, 3, Allocator.Temp);
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++) { S1[a, b] = (fProxy)0; S2[a, b] = (fProxy)0; S3[a, b] = (fProxy)0; }

            var q = new fProxyN(3, Allocator.Temp);
            var l = new fProxyN(3, Allocator.Temp);
            for (int i = 0; i < n; i++)
            {
                fProxy x = points[i].x, y = points[i].y;
                q[0] = x * x; q[1] = x * y; q[2] = y * y;
                l[0] = x;     l[1] = y;     l[2] = (fProxy)1;

                for (int a = 0; a < 3; a++)
                {
                    for (int b = 0; b < 3; b++)
                    {
                        S1[a, b] += q[a] * q[b];
                        S2[a, b] += q[a] * l[b];
                        S3[a, b] += l[a] * l[b];
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
            }

            q.Dispose(); l.Dispose();
            S1.Dispose(); S2.Dispose(); S3.Dispose(); T.Dispose(); M.Dispose(); Mp.Dispose();
            return ok;
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
            bool ok = conic(points, ref c);

            if (ok)
            {
                fProxy A = c[0], B = c[1], C = c[2], D = c[3], E = c[4], F = c[5];

                // Centre solves [[2A, B],[B, 2C]] c = [-D, -E].
                fProxy det = (fProxy)4 * A * C - B * B;
                if (math.abs(det) <= Consts.fProxyEpsilon) ok = false;
                else
                {
                    fProxy cx = ((fProxy)2 * C * (-D) - B * (-E)) / det;
                    fProxy cy = ((fProxy)2 * A * (-E) - B * (-D)) / det;
                    center = new fProxy2(cx, cy);

                    // Translate the constant term to the centre, then the semi-axes follow from the
                    // eigenvalues of the quadratic form [[A, B/2],[B/2, C]].
                    fProxy Fc = F + (fProxy)0.5 * (D * cx + E * cy);

                    var Q = new fProxyMxN(2, 2, Allocator.Temp);
                    var ev = new fProxyN(2, Allocator.Temp);
                    var V = new fProxyMxN(2, 2, Allocator.Temp);
                    Q[0, 0] = A; Q[0, 1] = (fProxy)0.5 * B;
                    Q[1, 0] = (fProxy)0.5 * B; Q[1, 1] = C;

                    if (Eigen.symmetricInPlace(ref Q, ref ev, ref V))
                    {
                        fProxy r0 = -Fc / ev[0], r1 = -Fc / ev[1];
                        if (r0 > (fProxy)0 && r1 > (fProxy)0)
                        {
                            radii = new fProxy2(math.sqrt(r0), math.sqrt(r1));
                            angle = math.atan2(V[1, 0], V[0, 0]);
                        }
                        else ok = false;                       // not a real ellipse
                    }
                    else ok = false;

                    Q.Dispose(); ev.Dispose(); V.Dispose();
                }
            }

            c.Dispose();
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
        {
            int n = points.Length;
            if (n < 9) throw new ArgumentException("Fit.quadric: points.Length must be >= 9");
            if (coeffs.N != 10) throw new ArgumentException("Fit.quadric: coeffs.N must be 10");

            var Dm = new fProxyMxN(n, 10, Allocator.Temp);
            for (int i = 0; i < n; i++)
            {
                fProxy x = points[i].x, y = points[i].y, z = points[i].z;
                Dm[i, 0] = x * x; Dm[i, 1] = y * y; Dm[i, 2] = z * z;
                Dm[i, 3] = x * y; Dm[i, 4] = x * z; Dm[i, 5] = y * z;
                Dm[i, 6] = x;     Dm[i, 7] = y;     Dm[i, 8] = z;
                Dm[i, 9] = (fProxy)1;
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
                fProxy zero = math.max(scale, (fProxy)1) * Consts.fProxySqrtEps;

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

                ok = QR.solveInPlace(ref Lu, ref rhs, ref sol);
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
