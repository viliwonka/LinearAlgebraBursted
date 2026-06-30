#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

namespace LinearAlgebra.ML
{
    /// <summary>
    /// Lloyd k-means clustering with GEMM-accelerated assignment and k-means++ seeding.
    ///
    /// Distance metric: squared Euclidean only (L2²). The GEMM assignment trick exploits
    ///   ‖xₙ − cⱼ‖² = ‖xₙ‖² − 2·(xₙ·cⱼ) + ‖cⱼ‖²
    /// to reduce each assignment step to a single matrix multiply (X·Cᵀ) followed by an
    /// in-place patch and a row-argmin sweep — O(N·D·k) with cache-friendly access.
    ///
    /// Generated for float and double; no integer variant.
    /// Opt in with <c>using LinearAlgebra.ML;</c>.
    ///
    /// Multiple restarts for best inertia: call the workspace overload <c>n_init</c> times,
    /// compare the returned <c>inertia</c> values, and keep the best assignment + centroids.
    /// </summary>
    public static partial class fProxyKMeans_OP
    {
        // =========================================================================
        // PRIMARY — zero-alloc workspace-taking overload (explicit init)
        // =========================================================================

        /// <summary>
        /// Lloyd k-means on X (N×D). Seeding is controlled by <paramref name="init"/>;
        /// both strategies are deterministic for a fixed <paramref name="seed"/>.
        /// Early-exit when zero assignments change between iterations.
        /// Empty clusters are reseeded to the point farthest from its current centroid.
        /// Inertia = total SSE = Σₙ ‖xₙ − c_{assignment[n]}‖².
        ///
        /// <paramref name="X"/>          N×D point matrix (one row per point).
        /// <paramref name="k"/>          Number of clusters; clamped to min(k, N) internally.
        /// <paramref name="seed"/>       Deterministic RNG seed; mapped to 1u if 0 (Unity.Mathematics.Random guard).
        /// <paramref name="maxIter"/>    Max Lloyd iterations; must be >= 1.
        /// <paramref name="init"/>       Seeding strategy: KMeansPlusPlus or Uniform.
        /// <paramref name="centroids"/>  k×D output centroids (caller pre-allocated; exactly k×D after clamp).
        /// <paramref name="assignment"/> N output cluster labels in [0, k); always consistent with returned centroids.
        /// <paramref name="inertia"/>    Final total SSE; always consistent with returned centroids and assignment.
        /// <paramref name="iters"/>      Actual iteration count in [1, maxIter].
        /// <paramref name="ws"/>         Pre-allocated workspace — Arena.fProxyKMeans_WS(N, D, k).
        /// </summary>
        public static void kmeans(
            in fProxyMxN X,
            int k,
            uint seed,
            int maxIter,
            KMeansInit init,
            ref fProxyMxN centroids,
            ref Indices assignment,
            out fProxy inertia,
            out int iters,
            ref fProxyKMeans_WS ws)
        {
            // ---- input guards ----
            if (X.M_Rows == 0 || X.N_Cols == 0)
                throw new InvalidOperationException("fProxyKMeans_OP.kmeans: X is empty");
            if (k <= 0)
                throw new ArgumentException("fProxyKMeans_OP.kmeans: k must be >= 1");
            if (maxIter < 1)
                throw new ArgumentException("fProxyKMeans_OP.kmeans: maxIter must be >= 1");

            int N = X.M_Rows;
            int D = X.N_Cols;
            k = math.min(k, N);  // documented clamp — not an error

            // ---- shape checks for caller-supplied outputs ----
            if (centroids.M_Rows != k || centroids.N_Cols != D)
                throw new ArgumentException(
                    "fProxyKMeans_OP.kmeans: centroids must be k×D after clamping k to min(k,N)");
            if (assignment.N != N)
                throw new ArgumentException(
                    "fProxyKMeans_OP.kmeans: assignment.N must equal X.M_Rows (N)");

            // ---- workspace shape checks ----
            if (ws.Gram.M_Rows != N || ws.Gram.N_Cols != k)
                throw new ArgumentException(
                    "fProxyKMeans_OP.kmeans: ws.Gram must be N×k");
            if (ws.Ct.M_Rows != D || ws.Ct.N_Cols != k)
                throw new ArgumentException(
                    "fProxyKMeans_OP.kmeans: ws.Ct must be D×k");
            if (ws.PointNormSq.N != N)
                throw new ArgumentException(
                    "fProxyKMeans_OP.kmeans: ws.PointNormSq.N must equal N");
            if (ws.CentNormSq.N != k)
                throw new ArgumentException(
                    "fProxyKMeans_OP.kmeans: ws.CentNormSq.N must equal k");
            if (ws.PrevAssignment.N != N)
                throw new ArgumentException(
                    "fProxyKMeans_OP.kmeans: ws.PrevAssignment.N must equal N");
            if (ws.NewCentroids.M_Rows != k || ws.NewCentroids.N_Cols != D)
                throw new ArgumentException(
                    "fProxyKMeans_OP.kmeans: ws.NewCentroids must be k×D");
            if (ws.ClusterCounts.N != k)
                throw new ArgumentException(
                    "fProxyKMeans_OP.kmeans: ws.ClusterCounts.N must equal k");
            if (ws.D2Weights.N != N)
                throw new ArgumentException(
                    "fProxyKMeans_OP.kmeans: ws.D2Weights.N must equal N");

            // ---- precompute point squared norms (once, before the Lloyd loop) ----
            for (int n = 0; n < N; n++)
            {
                fProxy s = (fProxy)0;
                for (int f = 0; f < D; f++) { fProxy v = X[n, f]; s += v * v; }
                ws.PointNormSq[n] = s;
            }

            // FIX 1: initialise assignment (not PrevAssignment) to -1 so that the first
            // iteration's "PrevAssignment = assignment" step copies -1 into PrevAssignment,
            // guaranteeing all N points register as changed on iter 0. Initialising
            // PrevAssignment instead read from an uninitialized assignment buffer and could
            // produce zero changes on the first iteration (assignment buffer often all-zeros
            // from UninitializedMemory), causing k=1 to return the seeded point rather than
            // the global mean.
            for (int n = 0; n < N; n++)
                assignment[n] = -1;

            // ---- seeding ----
            var rng = new Random(seed == 0u ? 1u : seed);

            if (init == KMeansInit.KMeansPlusPlus)
                SeedKMeansPlusPlus(in X, N, D, k, ref rng, ref centroids, ref ws);
            else
                SeedUniform(in X, N, D, k, ref rng, ref centroids);

            // ---- Lloyd iteration loop ----
            // iters is set at the top of each iteration so it reflects the count on exit
            // (break or natural loop end). Initial value satisfies C# definite-assignment.
            iters = 0;
            inertia = (fProxy)0; // overwritten in the converged branch (5.4.6) or in the final sync
            bool converged = false;

            for (int iter = 0; iter < maxIter; iter++)
            {
                iters = iter + 1;

                // 5.4.1  Centroid squared norms
                for (int j = 0; j < k; j++)
                {
                    fProxy s = (fProxy)0;
                    for (int f = 0; f < D; f++) { fProxy v = centroids[j, f]; s += v * v; }
                    ws.CentNormSq[j] = s;
                }

                // 5.4.2  Transpose centroids (k×D) -> ws.Ct (D×k)
                fProxy_OP.trans(in centroids, ref ws.Ct);

                // 5.4.3  GEMM: ws.Gram = X * ws.Ct  (N×k); dot zero-clears before accumulating.
                fProxy_OP.dot(in X, in ws.Ct, ref ws.Gram);

                // 5.4.4  Patch Gram in-place: score[n,j] = cn[j] - 2*G[n,j]
                //   pn[n] omitted (constant over j — no effect on argmin).
                for (int n = 0; n < N; n++)
                    for (int j = 0; j < k; j++)
                        ws.Gram[n, j] = ws.CentNormSq[j] - (fProxy)2 * ws.Gram[n, j];

                // 5.4.5  Save previous assignment; compute new assignment; count changes.
                for (int n = 0; n < N; n++)
                    ws.PrevAssignment[n] = assignment[n];

                fProxyQuery_OP.rowArgMin(in ws.Gram, ref assignment);

                int changes = 0;
                for (int n = 0; n < N; n++)
                    if (assignment[n] != ws.PrevAssignment[n]) changes++;

                // 5.4.6  Convergence: break before updating centroids so that returned
                //   centroids are the ones used for this iteration's Gram + assignment.
                //   Gram is already valid for the current centroids (computed in 5.4.3-5.4.4),
                //   so inertia can be computed here from the existing Gram — no extra GEMM.
                if (changes == 0)
                {
                    fProxy sseCvg = (fProxy)0;
                    for (int n = 0; n < N; n++)
                        sseCvg += ws.PointNormSq[n] + ws.Gram[n, assignment[n]];
                    inertia = math.max(sseCvg, (fProxy)0);
                    converged = true;
                    break;
                }

                // 5.4.7  Zero centroid accumulators and cluster counts
                for (int j = 0; j < k; j++)
                {
                    for (int f = 0; f < D; f++) ws.NewCentroids[j, f] = (fProxy)0;
                    ws.ClusterCounts[j] = 0;
                }

                // 5.4.8  Accumulate points into cluster sums
                for (int n = 0; n < N; n++)
                {
                    int j = assignment[n];
                    ws.ClusterCounts[j]++;
                    for (int f = 0; f < D; f++) ws.NewCentroids[j, f] += X[n, f];
                }

                // 5.4.9  Empty-cluster reseed.
                // FIX 2: reuse ws.D2Weights as a "remaining distance" scratch to prevent
                // multiple empty clusters from picking the same farthest point. Pre-fill
                // with squared distances, then set each chosen point's entry to -1 to
                // exclude it from subsequent empty-cluster scans.
                // ws.D2Weights was used for k-means++ seeding and is free at this point.
                for (int n = 0; n < N; n++)
                    ws.D2Weights[n] = ws.PointNormSq[n] + ws.Gram[n, assignment[n]];

                for (int j = 0; j < k; j++)
                {
                    if (ws.ClusterCounts[j] != 0) continue;

                    // Find the point with the largest squared distance to its centroid
                    // that has not already been claimed by a previous empty-cluster reseed.
                    fProxy maxDist = (fProxy)(-1);
                    int farthestPt = 0;
                    for (int n = 0; n < N; n++)
                    {
                        if (ws.D2Weights[n] > maxDist)
                        {
                            maxDist = ws.D2Weights[n];
                            farthestPt = n;
                        }
                    }
                    for (int f = 0; f < D; f++) ws.NewCentroids[j, f] = X[farthestPt, f];
                    ws.ClusterCounts[j] = 1;         // sentinel: avoid divide-by-zero below
                    ws.D2Weights[farthestPt] = (fProxy)(-1); // exclude from subsequent scans
                }

                // 5.4.10  Divide accumulators -> new centroids
                for (int j = 0; j < k; j++)
                {
                    fProxy invN = (fProxy)1 / (fProxy)ws.ClusterCounts[j];
                    for (int f = 0; f < D; f++) centroids[j, f] = ws.NewCentroids[j, f] * invN;
                }
            }

            // Final sync: only executed on the MaxIter-exhaustion path.
            //
            // MaxIter-exhaustion path (converged == false): centroids were updated in 5.4.10
            //   of the last iteration, but assignment/Gram still reflect the pre-update
            //   centroids. Recompute Gram, assignment, and inertia so all outputs are
            //   mutually consistent with the returned centroids.
            //
            // Convergence path (converged == true): Gram is already valid (computed in
            //   5.4.3-5.4.4 from unchanged centroids), inertia was computed in 5.4.6.
            //   Skipped entirely — avoids a redundant O(N·D·k) GEMM + transpose.
            if (!converged)
            {
                for (int j = 0; j < k; j++)
                {
                    fProxy s = (fProxy)0;
                    for (int f = 0; f < D; f++) { fProxy v = centroids[j, f]; s += v * v; }
                    ws.CentNormSq[j] = s;
                }
                fProxy_OP.trans(in centroids, ref ws.Ct);
                fProxy_OP.dot(in X, in ws.Ct, ref ws.Gram);
                for (int n = 0; n < N; n++)
                    for (int j = 0; j < k; j++)
                        ws.Gram[n, j] = ws.CentNormSq[j] - (fProxy)2 * ws.Gram[n, j];
                fProxyQuery_OP.rowArgMin(in ws.Gram, ref assignment);

                fProxy sse = (fProxy)0;
                for (int n = 0; n < N; n++)
                    sse += ws.PointNormSq[n] + ws.Gram[n, assignment[n]];
                // FIX 4: clamp to >= 0 to guard against tiny negative values from FP
                // cancellation when a point sits exactly on its centroid.
                inertia = math.max(sse, (fProxy)0);
            }
        }

        // =========================================================================
        // PRIMARY — forwarding overload defaulting to KMeansPlusPlus (FIX 5)
        // =========================================================================

        /// <summary>
        /// Calls <see cref="kmeans(in fProxyMxN,int,uint,int,KMeansInit,ref fProxyMxN,ref Indices,out fProxy,out int,ref fProxyKMeans_WS)"/>
        /// with <c>init = KMeansInit.KMeansPlusPlus</c>.
        /// </summary>
        public static void kmeans(
            in fProxyMxN X,
            int k,
            uint seed,
            int maxIter,
            ref fProxyMxN centroids,
            ref Indices assignment,
            out fProxy inertia,
            out int iters,
            ref fProxyKMeans_WS ws)
            => kmeans(in X, k, seed, maxIter, KMeansInit.KMeansPlusPlus,
                      ref centroids, ref assignment, out inertia, out iters, ref ws);

        // =========================================================================
        // ALLOCATING CONVENIENCE WRAPPER — explicit init (FIX 6: guards before alloc)
        // =========================================================================

        /// <summary>
        /// Validates inputs, then allocates centroids (k×D), assignment (N), and workspace
        /// from <paramref name="arena"/> and delegates to the workspace overload.
        /// All outputs are arena-owned. Guards fire before any arena allocation so that
        /// no memory is orphaned on invalid input.
        ///
        /// For multiple restarts: call the workspace overload directly so scratch can be
        /// reused across calls. Compare <paramref name="inertia"/> values and keep the best.
        /// </summary>
        public static void kmeans(
            ref Arena arena,
            in fProxyMxN X,
            int k,
            uint seed,
            int maxIter,
            KMeansInit init,
            out fProxyMxN centroids,
            out Indices assignment,
            out fProxy inertia,
            out int iters)
        {
            // FIX 6: validate before allocating so invalid args cannot orphan arena memory.
            if (X.M_Rows == 0 || X.N_Cols == 0)
                throw new InvalidOperationException("fProxyKMeans_OP.kmeans: X is empty");
            if (k <= 0)
                throw new ArgumentException("fProxyKMeans_OP.kmeans: k must be >= 1");
            if (maxIter < 1)
                throw new ArgumentException("fProxyKMeans_OP.kmeans: maxIter must be >= 1");

            int N = X.M_Rows;
            int D = X.N_Cols;
            int kk = math.min(k, N);  // match the primary overload's clamp
            centroids  = arena.fProxyMat(kk, D);
            assignment = arena.Indices(N);
            var ws     = arena.fProxyKMeans_WS(N, D, kk);
            kmeans(in X, k, seed, maxIter, init, ref centroids, ref assignment,
                   out inertia, out iters, ref ws);
        }

        // =========================================================================
        // ALLOCATING CONVENIENCE WRAPPER — defaults to KMeansPlusPlus (FIX 5)
        // =========================================================================

        /// <summary>
        /// Calls <see cref="kmeans(ref Arena,in fProxyMxN,int,uint,int,KMeansInit,out fProxyMxN,out Indices,out fProxy,out int)"/>
        /// with <c>init = KMeansInit.KMeansPlusPlus</c>.
        /// </summary>
        public static void kmeans(
            ref Arena arena,
            in fProxyMxN X,
            int k,
            uint seed,
            int maxIter,
            out fProxyMxN centroids,
            out Indices assignment,
            out fProxy inertia,
            out int iters)
            => kmeans(ref arena, in X, k, seed, maxIter, KMeansInit.KMeansPlusPlus,
                      out centroids, out assignment, out inertia, out iters);

        // =========================================================================
        // PRIVATE — seeding helpers
        // =========================================================================

        // k-means++ seeding.
        // FIX 8: incremental D2Weights (O(k·N·D)) — after adding centroid ci, update
        // D2Weights[n] = min(D2Weights[n], dist(xn, ci)) instead of recomputing from
        // scratch (which was O(k²·N·D)).
        // All-identical-point fallback: uniform random when total weight == 0.
        static void SeedKMeansPlusPlus(
            in fProxyMxN X, int N, int D, int k,
            ref Random rng,
            ref fProxyMxN centroids,
            ref fProxyKMeans_WS ws)
        {
            // First centroid: uniform random point.
            int firstIdx = math.min((int)(rng.NextFProxy() * (fProxy)N), N - 1);
            for (int f = 0; f < D; f++) centroids[0, f] = X[firstIdx, f];

            if (k == 1) return; // nothing more to seed

            // Initialise D2Weights with squared distances to c_0.
            for (int n = 0; n < N; n++)
            {
                fProxy d2 = (fProxy)0;
                for (int f = 0; f < D; f++)
                {
                    fProxy diff = X[n, f] - centroids[0, f];
                    d2 += diff * diff;
                }
                ws.D2Weights[n] = d2;
            }

            // Centroids 1..k-1 via D² weighting (incremental).
            for (int ci = 1; ci < k; ci++)
            {
                // At this point D2Weights[n] = min dist² to c_0..c_{ci-1}.

                // Sum D² weights; if all zero (duplicate points) fall back to uniform.
                fProxy total = (fProxy)0;
                for (int n = 0; n < N; n++) total += ws.D2Weights[n];

                int nextIdx;
                if (!(total > (fProxy)0))
                {
                    // All remaining distances are zero (all points identical) — uniform fallback.
                    nextIdx = math.min((int)(rng.NextFProxy() * (fProxy)N), N - 1);
                }
                else
                {
                    // fProxyRandom_OP.weightedPick validates + draws from D² distribution.
                    nextIdx = fProxyRandom_OP.weightedPick(in ws.D2Weights, ref rng);
                }

                for (int f = 0; f < D; f++) centroids[ci, f] = X[nextIdx, f];

                // Update D2Weights incrementally: take the min with distance to the new centroid.
                // Skip on the last centroid (ci == k-1) since D2Weights won't be read again.
                if (ci < k - 1)
                {
                    for (int n = 0; n < N; n++)
                    {
                        fProxy d2 = (fProxy)0;
                        for (int f = 0; f < D; f++)
                        {
                            fProxy diff = X[n, f] - centroids[ci, f];
                            d2 += diff * diff;
                        }
                        if (d2 < ws.D2Weights[n]) ws.D2Weights[n] = d2;
                    }
                }
            }
        }

        // Uniform seeding: select k distinct points via reservoir selection (Algorithm S / Knuth).
        // O(N) time, zero allocation. Works for any k <= N.
        static void SeedUniform(
            in fProxyMxN X, int N, int D, int k,
            ref Random rng,
            ref fProxyMxN centroids)
        {
            int needed = k;
            int ci = 0;
            for (int n = 0; n < N && needed > 0; n++)
            {
                // Include point n with probability needed / (N - n).
                // Draw r uniformly from [0, N-n); include iff r < needed.
                fProxy r = rng.NextFProxy() * (fProxy)(N - n);
                if (r < (fProxy)needed)
                {
                    for (int f = 0; f < D; f++) centroids[ci, f] = X[n, f];
                    ci++;
                    needed--;
                }
            }
            // Fallback: unreachable after the k = min(k, N) clamp in the entry point,
            // but guards against any residual gap by duplicating the last centroid.
            for (; ci < k; ci++)
                for (int f = 0; f < D; f++) centroids[ci, f] = centroids[ci > 0 ? ci - 1 : 0, f];
        }
    }
}
