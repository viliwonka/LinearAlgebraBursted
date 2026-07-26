using System;

using Unity.Collections;
using Unity.Mathematics;

using BULA.ML;

//+deleteThis
// TEMPLATE-ONLY alias: codegen rewrites each fProxy* token -> float*/double* (real Unity.Mathematics
// types), so the field access below (.x/.y/.z/.w) and constructors resolve natively -- no proxy-struct
// shim needed. See ConvertOP.fProxy.cs for the same pattern.
using fProxy2 = Unity.Mathematics.float2;
using fProxy3 = Unity.Mathematics.float3;
using fProxy4 = Unity.Mathematics.float4;
//-deleteThis

namespace BULA
{
    // ================================================================================================
    // Fitting geometry to point clouds, parameterized by the METRIC.
    //
    // Every entry point comes in two forms: a plain overload minimizing squared orthogonal distance
    // (closed form, no iteration) and a generic overload taking an <see cref="IfProxyRobustLoss"/>
    // that reweights by residual -- fProxyL1Loss, fProxyHuberLoss, fProxyCauchyLoss, fProxyTukeyLoss.
    // The robust form is IRLS: reweight, refit, repeat. Each iteration is one weighted covariance
    // (O(n·d²)) plus one d x d eigensolve, d in {2,3,4}, so cost is dominated by the point count.
    //
    // Job-safe: scratch is Allocator.Temp, disposed before returning.
    // ================================================================================================
    public static partial class Fit
    {
        // ============================================================================================
        // LINE -- the best-fit 1-D affine subspace (minimizes distance PERPENDICULAR to the line, not
        // vertical residual: this is orthogonal / total-least-squares regression, not y-on-x).
        // ============================================================================================

        /// <summary>
        /// Best-fit line through a 2D point cloud, minimizing squared perpendicular distance.
        /// centroid = mean of points, direction = unit, sign arbitrary. Requires
        /// points.Length &gt;= 2. False means the underlying eigensolve did not converge; direction is
        /// then undefined.
        /// </summary>
        public static bool line(NativeArray<fProxy2> points, out fProxy2 centroid, out fProxy2 direction)
        {
            var l2 = new fProxyL2Loss();
            return line(points, in l2, out centroid, out direction);
        }

        /// <summary>Best-fit line through a 3D point cloud. See the 2D overload.</summary>
        public static bool line(NativeArray<fProxy3> points, out fProxy3 centroid, out fProxy3 direction)
        {
            var l2 = new fProxyL2Loss();
            return line(points, in l2, out centroid, out direction);
        }

        /// <summary>Best-fit line through a 4D point cloud. See the 2D overload.</summary>
        public static bool line(NativeArray<fProxy4> points, out fProxy4 centroid, out fProxy4 direction)
        {
            var l2 = new fProxyL2Loss();
            return line(points, in l2, out centroid, out direction);
        }

        /// <summary>
        /// Best-fit line through a 2D point cloud under <paramref name="loss"/>, by IRLS on the
        /// perpendicular residual. centroid is the WEIGHTED mean, direction the dominant weighted
        /// principal axis (unit, sign arbitrary). Requires points.Length &gt;= 2.
        /// <paramref name="maxIter"/> &lt;= 0 picks a default budget; a redescending loss
        /// (<see cref="fProxyTukeyLoss"/>) can leave every point zero-weighted from a bad start, which
        /// returns false rather than a fabricated line.
        /// </summary>
        public static bool line<TLoss>(NativeArray<fProxy2> points, in TLoss loss,
                                       out fProxy2 centroid, out fProxy2 direction, int maxIter = 0)
            where TLoss : struct, IfProxyRobustLoss
        {
            if (points.Length < 2) throw new ArgumentException("Fit.line: points.Length must be >= 2");

            var flat = points.Reinterpret<fProxy2, fProxy>();
            var mean = new fProxyN(2, Allocator.Temp);
            var V = new fProxyMxN(2, 2, Allocator.Temp);

            bool ok = SubspaceIrls(flat, points.Length, 2, 1, in loss, maxIter, ref mean, ref V);
            centroid = new fProxy2(mean[0], mean[1]);
            direction = ok ? new fProxy2(V[0, 0], V[1, 0]) : default;

            mean.Dispose(); V.Dispose();
            return ok;
        }

        /// <summary>Robust best-fit line through a 3D point cloud. See the 2D overload.</summary>
        public static bool line<TLoss>(NativeArray<fProxy3> points, in TLoss loss,
                                       out fProxy3 centroid, out fProxy3 direction, int maxIter = 0)
            where TLoss : struct, IfProxyRobustLoss
        {
            if (points.Length < 2) throw new ArgumentException("Fit.line: points.Length must be >= 2");

            var flat = points.Reinterpret<fProxy3, fProxy>();
            var mean = new fProxyN(3, Allocator.Temp);
            var V = new fProxyMxN(3, 3, Allocator.Temp);

            bool ok = SubspaceIrls(flat, points.Length, 3, 1, in loss, maxIter, ref mean, ref V);
            centroid = new fProxy3(mean[0], mean[1], mean[2]);
            direction = ok ? new fProxy3(V[0, 0], V[1, 0], V[2, 0]) : default;

            mean.Dispose(); V.Dispose();
            return ok;
        }

        /// <summary>Robust best-fit line through a 4D point cloud. See the 2D overload.</summary>
        public static bool line<TLoss>(NativeArray<fProxy4> points, in TLoss loss,
                                       out fProxy4 centroid, out fProxy4 direction, int maxIter = 0)
            where TLoss : struct, IfProxyRobustLoss
        {
            if (points.Length < 2) throw new ArgumentException("Fit.line: points.Length must be >= 2");

            var flat = points.Reinterpret<fProxy4, fProxy>();
            var mean = new fProxyN(4, Allocator.Temp);
            var V = new fProxyMxN(4, 4, Allocator.Temp);

            bool ok = SubspaceIrls(flat, points.Length, 4, 1, in loss, maxIter, ref mean, ref V);
            centroid = new fProxy4(mean[0], mean[1], mean[2], mean[3]);
            direction = ok ? new fProxy4(V[0, 0], V[1, 0], V[2, 0], V[3, 0]) : default;

            mean.Dispose(); V.Dispose();
            return ok;
        }

        // ============================================================================================
        // PLANE -- the best-fit (d-1)-dimensional affine subspace, reported by its unit normal.
        // ============================================================================================

        /// <summary>
        /// Best-fit plane through a 3D point cloud, minimizing squared perpendicular distance.
        /// centroid = mean of points, normal = unit, sign arbitrary. Requires points.Length &gt;= 3.
        /// False means the underlying eigensolve did not converge; normal is then undefined.
        /// </summary>
        public static bool plane(NativeArray<fProxy3> points, out fProxy3 centroid, out fProxy3 normal)
        {
            var l2 = new fProxyL2Loss();
            return plane(points, in l2, out centroid, out normal);
        }

        /// <summary>
        /// Best-fit plane through a 3D point cloud under <paramref name="loss"/>, by IRLS on the
        /// perpendicular residual. centroid is the WEIGHTED mean, normal the least-variance weighted
        /// principal axis (unit, sign arbitrary). Requires points.Length &gt;= 3. See
        /// <see cref="line{TLoss}(NativeArray{fProxy2}, in TLoss, out fProxy2, out fProxy2, int)"/>
        /// for the shared iteration contract.
        /// </summary>
        public static bool plane<TLoss>(NativeArray<fProxy3> points, in TLoss loss,
                                        out fProxy3 centroid, out fProxy3 normal, int maxIter = 0)
            where TLoss : struct, IfProxyRobustLoss
        {
            if (points.Length < 3) throw new ArgumentException("Fit.plane: points.Length must be >= 3");

            var flat = points.Reinterpret<fProxy3, fProxy>();
            var mean = new fProxyN(3, Allocator.Temp);
            var V = new fProxyMxN(3, 3, Allocator.Temp);

            bool ok = SubspaceIrls(flat, points.Length, 3, 2, in loss, maxIter, ref mean, ref V);
            centroid = new fProxy3(mean[0], mean[1], mean[2]);
            normal = ok ? new fProxy3(V[0, 2], V[1, 2], V[2, 2]) : default;

            mean.Dispose(); V.Dispose();
            return ok;
        }

        // ============================================================================================
        // The shared IRLS core: weighted centroid + weighted covariance eigendecomposition.
        //
        // Fits the affine subspace of dimension `k` through the n x d cloud X (row i = point i).
        // On return `mean` is the weighted centroid and V's COLUMNS are the weighted principal axes,
        // eigenvalues descending -- so the subspace is span(V[:,0..k-1]) and its orthogonal complement
        // is span(V[:,k..d-1]). The residual driving the reweighting is the distance from a point to
        // that subspace: r² = ‖q‖² − Σ_{j<k} (q·v_j)², q = x_i − mean.
        //
        // fProxyL2Loss has RhoPrime == 1, so the weights never move and this exits after one pass --
        // identical to plain PCA, which is why the non-generic overloads forward there directly.
        // ============================================================================================
        // ONE weighted pass: weighted centroid, weighted covariance, eigendecomposition. Factored out
        // of the IRLS loop so the generic Fit.irls driver and the shape structs can call a single
        // weighted fit without inheriting a loop -- the loop belongs to the solver, not the shape.
        //
        // Returns false when the total weight is zero, which a redescending loss can produce: solving
        // from an all-zero weighting yields NaN, and reporting that as success would be a false
        // certificate. C is DESTROYED (symmetricInPlace), so callers rebuild it every pass.
        internal static bool SubspaceFitWeighted(NativeArray<fProxy> X, int n, int d, in fProxyN w,
                                                 ref fProxyN mean, ref fProxyMxN C,
                                                 ref fProxyN eig, ref fProxyMxN V)
        {
            fProxy sw = (fProxy)0;
            for (int j = 0; j < d; j++) mean[j] = (fProxy)0;
            for (int i = 0; i < n; i++)
            {
                fProxy wi = w[i];
                sw += wi;
                for (int j = 0; j < d; j++) mean[j] += wi * X[i * d + j];
            }
            if (!(sw > (fProxy)0)) return false;
            for (int j = 0; j < d; j++) mean[j] /= sw;

            for (int a = 0; a < d; a++)
                for (int b = 0; b < d; b++) C[a, b] = (fProxy)0;
            for (int i = 0; i < n; i++)
            {
                fProxy wi = w[i];
                for (int a = 0; a < d; a++)
                {
                    fProxy qa = X[i * d + a] - mean[a];
                    for (int b = 0; b < d; b++) C[a, b] += wi * qa * (X[i * d + b] - mean[b]);
                }
            }
            for (int a = 0; a < d; a++)
                for (int b = 0; b < d; b++) C[a, b] /= sw;

            return Eigen.symmetricInPlace(ref C, ref eig, ref V);
        }

        static bool SubspaceIrls<TLoss>(NativeArray<fProxy> X, int n, int d, int k,
                                        in TLoss loss, int maxIter,
                                        ref fProxyN mean, ref fProxyMxN V)
            where TLoss : struct, IfProxyRobustLoss
        {
            if (maxIter <= 0) maxIter = DefaultIrlsIter;

            var w = new fProxyN(n, Allocator.Temp);
            var C = new fProxyMxN(d, d, Allocator.Temp);
            var eig = new fProxyN(d, Allocator.Temp);
            for (int i = 0; i < n; i++) w[i] = (fProxy)1;

            bool ok = false;
            for (int it = 0; it < maxIter; it++)
            {
                ok = SubspaceFitWeighted(X, n, d, in w, ref mean, ref C, ref eig, ref V);
                if (!ok) break;

                fProxy maxDelta = (fProxy)0;
                for (int i = 0; i < n; i++)
                {
                    fProxy q2 = (fProxy)0;
                    for (int a = 0; a < d; a++)
                    {
                        fProxy qa = X[i * d + a] - mean[a];
                        q2 += qa * qa;
                    }

                    fProxy inSub = (fProxy)0;
                    for (int j = 0; j < k; j++)
                    {
                        fProxy dp = (fProxy)0;
                        for (int a = 0; a < d; a++) dp += (X[i * d + a] - mean[a]) * V[a, j];
                        inSub += dp * dp;
                    }

                    fProxy wNew = loss.RhoPrime(math.max(q2 - inSub, (fProxy)0));
                    maxDelta = math.max(maxDelta, math.abs(wNew - w[i]));
                    w[i] = wNew;
                }

                if (maxDelta <= Consts.fProxySqrtEps) break;
            }

            w.Dispose(); C.Dispose(); eig.Dispose();
            return ok;
        }
    }
}
