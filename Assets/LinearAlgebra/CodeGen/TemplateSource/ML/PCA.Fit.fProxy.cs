using System;
using Unity.Collections;
using Unity.Mathematics;

//+deleteThis
// TEMPLATE-ONLY alias: codegen rewrites each fProxy* token -> float*/double* (real Unity.Mathematics
// types), so the field access below (.x/.y/.z/.w) and constructors resolve natively -- no proxy-struct
// shim needed. See ConvertOP.fProxy.cs for the same pattern.
using fProxy2 = Unity.Mathematics.float2;
using fProxy3 = Unity.Mathematics.float3;
using fProxy4 = Unity.Mathematics.float4;
//-deleteThis

namespace LinearAlgebra.ML
{
    // Game-facing convenience wrappers over PCA.fitCov: fit a line/plane through a small point cloud
    // without assembling a fProxyPCAModel yourself. Always uses PCAScaling.Covariance (never
    // Correlation, which would rescale axes independently and break the perpendicular-distance
    // meaning of a geometric fit). Zero-copy: each NativeArray<fProxy2/3/4> point cloud is reinterpreted
    // as a flat NativeArray<fProxy> and viewed as an n x {2,3,4} fProxyMxN (row-major AoS == the
    // point cloud's own memory layout, so no transpose/copy is needed).
    public static partial class PCA
    {
        /// <summary>
        /// Best-fit line through a 2D point cloud (minimizes perpendicular distance): centroid = mean
        /// of points; direction = the dominant principal component (largest-variance axis), unit
        /// length. Direction's sign is arbitrary (eigenvector sign convention -- see PCA's class doc).
        /// Requires points.Length &gt;= 2. Returns the underlying eigensolve's convergence flag (see
        /// <see cref="fitCov(in fProxyMxN, ref fProxyPCAModel)"/>); direction is undefined when false.
        /// </summary>
        public static bool fitLine(NativeArray<fProxy2> points, out fProxy2 centroid, out fProxy2 direction)
        {
            const string method = "PCA.fitLine";
            if (points.Length < 2)
                throw new ArgumentException(method + ": points.Length must be >= 2");

            var flat = points.Reinterpret<fProxy2, fProxy>();
            var X = new fProxyMxN(points.Length, 2, flat);

            var model = new fProxyPCAModel(2, 2, Allocator.Temp);
            bool converged = fitCov(in X, ref model);

            centroid = new fProxy2(model.mean[0], model.mean[1]);
            direction = converged ? new fProxy2(model.components[0, 0], model.components[1, 0]) : default;

            model.Dispose();
            return converged;
        }

        /// <summary>
        /// Best-fit line through a 3D point cloud (minimizes perpendicular distance): centroid = mean
        /// of points; direction = the dominant principal component (largest-variance axis), unit
        /// length. Direction's sign is arbitrary (eigenvector sign convention -- see PCA's class doc).
        /// Requires points.Length &gt;= 2. Returns the underlying eigensolve's convergence flag (see
        /// <see cref="fitCov(in fProxyMxN, ref fProxyPCAModel)"/>); direction is undefined when false.
        /// </summary>
        public static bool fitLine(NativeArray<fProxy3> points, out fProxy3 centroid, out fProxy3 direction)
        {
            const string method = "PCA.fitLine";
            if (points.Length < 2)
                throw new ArgumentException(method + ": points.Length must be >= 2");

            var flat = points.Reinterpret<fProxy3, fProxy>();
            var X = new fProxyMxN(points.Length, 3, flat);

            var model = new fProxyPCAModel(3, 3, Allocator.Temp);
            bool converged = fitCov(in X, ref model);

            centroid = new fProxy3(model.mean[0], model.mean[1], model.mean[2]);
            direction = converged
                ? new fProxy3(model.components[0, 0], model.components[1, 0], model.components[2, 0])
                : default;

            model.Dispose();
            return converged;
        }

        /// <summary>
        /// Best-fit line through a 4D point cloud: centroid = mean of points; direction = the dominant
        /// principal component (largest-variance axis), unit length. Direction's sign is arbitrary
        /// (eigenvector sign convention -- see PCA's class doc). Requires points.Length &gt;= 2.
        /// Returns the underlying eigensolve's convergence flag (see
        /// <see cref="fitCov(in fProxyMxN, ref fProxyPCAModel)"/>); direction is undefined when false.
        /// </summary>
        public static bool fitLine(NativeArray<fProxy4> points, out fProxy4 centroid, out fProxy4 direction)
        {
            const string method = "PCA.fitLine";
            if (points.Length < 2)
                throw new ArgumentException(method + ": points.Length must be >= 2");

            var flat = points.Reinterpret<fProxy4, fProxy>();
            var X = new fProxyMxN(points.Length, 4, flat);

            var model = new fProxyPCAModel(4, 4, Allocator.Temp);
            bool converged = fitCov(in X, ref model);

            centroid = new fProxy4(model.mean[0], model.mean[1], model.mean[2], model.mean[3]);
            direction = converged
                ? new fProxy4(model.components[0, 0], model.components[1, 0], model.components[2, 0], model.components[3, 0])
                : default;

            model.Dispose();
            return converged;
        }

        /// <summary>
        /// Best-fit plane through a 3D point cloud (minimizes perpendicular distance): centroid = mean
        /// of points; normal = the least-variance principal component, unit length. Normal's sign is
        /// arbitrary (eigenvector sign convention -- see PCA's class doc). Requires points.Length &gt;=
        /// 3 (a plane's normal is not well-defined from fewer points). Returns the underlying
        /// eigensolve's convergence flag (see <see cref="fitCov(in fProxyMxN, ref fProxyPCAModel)"/>);
        /// normal is undefined when false.
        /// </summary>
        public static bool fitPlane(NativeArray<fProxy3> points, out fProxy3 centroid, out fProxy3 normal)
        {
            const string method = "PCA.fitPlane";
            if (points.Length < 3)
                throw new ArgumentException(method + ": points.Length must be >= 3");

            var flat = points.Reinterpret<fProxy3, fProxy>();
            var X = new fProxyMxN(points.Length, 3, flat);

            var model = new fProxyPCAModel(3, 3, Allocator.Temp);
            bool converged = fitCov(in X, ref model);

            centroid = new fProxy3(model.mean[0], model.mean[1], model.mean[2]);
            normal = converged
                ? new fProxy3(model.components[0, 2], model.components[1, 2], model.components[2, 2])
                : default;

            model.Dispose();
            return converged;
        }
    }
}
