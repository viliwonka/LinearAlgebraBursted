# Spec: K-Means Clustering -- LinearAlgebra.ML

Status: **UNBUILT** (2026-06-28). fProxy-only (float / double). New sub-namespace + folder.

---

## 1. Placement and naming

### Namespace: LinearAlgebra.ML

Rationale over .Data:
- k-means is a machine learning algorithm, not a data-structure or preprocessing primitive. .ML is
  unambiguous and future-proof (PCA, Kalman, k-NN classifiers would also live here).
- Matches the existing sub-namespace pattern: LinearAlgebra.Gallery, LinearAlgebra.Realtime,
  LinearAlgebra.Stats. Each is a focused layer, opt-in via a using directive.
- .Data would invite confusion with Unity Collections data-structures; .ML does not.

### New folder: Assets/LinearAlgebra/CodeGen/TemplateSource/ML/

Two template files:
- ML/KMeans.fProxy.cs         -- static class fProxyKMeans_OP (algorithm + Lloyd loop)
- ML/KMeans.Workspace.fProxy.cs -- workspace struct + Arena factory

Codegen produces (fProxy -> float + double):
- Assets/LinearAlgebra/Source/Generated/ML/KMeans.float.cs
- Assets/LinearAlgebra/Source/Generated/ML/KMeans.double.cs
- Assets/LinearAlgebra/Source/Generated/ML/KMeans.Workspace.float.cs
- Assets/LinearAlgebra/Source/Generated/ML/KMeans.Workspace.double.cs

Static class: public static partial class fProxyKMeans_OP -- consistent with fProxyQuery_OP,
fProxyRandom_OP, fProxyStats_OP. No iProxy variant (integer k-means is not meaningful).

Test template: Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/KMeansTests.fProxy.cs

---

## 2. Why GEMM-based assignment is the centerpiece

The naive assignment loop is O(N*k*D) with scatter that defeats the CPU memory hierarchy. The GEMM
path uses one dense matrix multiply -- already Burst-accelerated -- and reduces k-means assignment
to a post-GEMM argmin sweep.

The identity that enables it:

    ||x_n - c_j||^2 = ||x_n||^2 - 2*(x_n . c_j) + ||c_j||^2

Let:
  G = X * C^T  (N x k Gram matrix, one GEMM)  -- G[n,j] = x_n . c_j
  pn[n] = ||x_n||^2  (N-vector, constant across iterations, computed ONCE before the loop)
  cn[j] = ||c_j||^2  (k-vector, computed once per iteration)

Then:

    ||x_n - c_j||^2 = pn[n] - 2*G[n,j] + cn[j]

The term pn[n] is SAME for all j at row n, so:
    argmin_j ||x_n - c_j||^2  ==  argmin_j (cn[j] - 2*G[n,j])

Assignment step:
1. Patch Gram in-place: Gram[n,j] <- cn[j] - 2*G[n,j]  (no allocation, one loop)
2. Call fProxyQuery_OP.rowArgMin(in ws.Gram, ref assignment)
   (QueryOP.fProxy.cs, index-only overload at line 115)

pn[n] IS needed for true inertia:
    ||x_n - c_{assignment[n]}||^2  =  pn[n] + Gram[n, assignment[n]]  (after the patch)
It does NOT affect argmin and need not enter the score matrix.

### Implementing G = X * C^T using existing GEMM

OP.Dot.fProxy.cs exposes:
- fProxy_OP.dot(in fProxyMxN a, in fProxyMxN b, ref fProxyMxN c, bool transposeA = false)
  (line 128)  -- C = A*B; transposeA=true gives A^T*B
- fProxy_OP.trans(in fProxyMxN A, ref fProxyMxN T) (line 183) -- explicit transpose

The existing dot API has only transposeA (not transposeB). To compute G = X * C^T
where X is N x D and centroids C is k x D:

  Step 1: fProxy_OP.trans(in centroids, ref ws.Ct)
          -- transposes C (k x D) into ws.Ct (D x k)
  Step 2: fProxy_OP.dot(in X, in ws.Ct, ref ws.Gram)
          -- X (N x D) * Ct (D x k) = Gram (N x k)

Both use ref-dest primitives. Gram must not alias X or Ct -- guaranteed by distinct workspace
fields. The dot kernel zero-clears its destination before accumulating (OP.Dot.fProxy.cs line 156).

---

## 3. Workspace struct and Arena factory

Pattern mirrors fProxySvd_WS / Arena.fProxySvd_WS in SVD.Workspace.fProxy.cs.

### struct fProxyKMeans_WS  (file: ML/KMeans.Workspace.fProxy.cs)

`csharp
namespace LinearAlgebra.ML
{
    // Reusable scratch for zero-alloc Lloyd k-means. Allocate once per (N, D, k) shape via
    // Arena.fProxyKMeans_WS(N, D, k). All buffers are arena-owned (disposed with arena).
    public struct fProxyKMeans_WS
    {
        public fProxyMxN Gram;           // N x k  GEMM output X*C^T, patched to scores in-place
        public fProxyMxN Ct;             // D x k  transposed centroids (refreshed each iteration)
        public fProxyN   PointNormSq;    // N      ||x_n||^2 constant (computed once before loop)
        public fProxyN   CentNormSq;     // k      ||c_j||^2 (recomputed each iteration)
        public Indices   PrevAssignment; // N      cluster labels from previous iter (early-exit)
        public fProxyMxN NewCentroids;   // k x D  centroid accumulator (zeroed each iteration)
        public Indices   ClusterCounts;  // k      per-cluster point count (zeroed each iteration)
        public fProxyN   D2Weights;      // N      D^2 distances for k-means++ seeding only
    }
}
`

All fields are unmanaged value types; the struct is Burst-compatible.
Memory: 2*N*k + 2*D*k + 2*N + k  fProxy scalars, plus N + k ints (Indices).

### Arena.fProxyKMeans_WS factory  (partial Arena in same file)

`csharp
public partial struct Arena
{
    // Allocates workspace for N points, D features, k clusters.
    // Persistent arena allocations -- disposed with the arena.
    // Create once outside hot loops; reuse for same-shape calls.
    public fProxyKMeans_WS fProxyKMeans_WS(int N, int D, int k)
    {
        return new fProxyKMeans_WS
        {
            Gram           = fProxyMat(N, k),
            Ct             = fProxyMat(D, k),
            PointNormSq    = fProxyVec(N),
            CentNormSq     = fProxyVec(k),
            PrevAssignment = Indices(N),      // Arena.Indices -- Arena.cs line 83
            NewCentroids   = fProxyMat(k, D),
            ClusterCounts  = Indices(k),      // Arena.Indices -- Arena.cs line 83
            D2Weights      = fProxyVec(N)
        };
    }
}
`

---

## 4. API signatures

File: Assets/LinearAlgebra/CodeGen/TemplateSource/ML/KMeans.fProxy.cs

`csharp
namespace LinearAlgebra.ML
{
    public static partial class fProxyKMeans_OP
    {
        // PRIMARY -- zero-alloc workspace-taking overload.
        // Lloyd k-means on X (N x D). k-means++ seeding (deterministic for fixed seed).
        // Early-exit when zero assignments change between iterations.
        // Empty clusters reseeded to the point farthest from its current centroid.
        // Inertia = total SSE = sum_n ||x_n - c_{assignment[n]}||^2.
        //
        // X          N x D point matrix (one row per point)
        // k          number of clusters; clamped to min(k, N) internally
        // seed       uint RNG seed; mapped to 1u if 0 (Unity.Mathematics.Random requirement)
        // maxIter    max Lloyd iterations (must be >= 1)
        // centroids  k x D output centroids (caller pre-allocated, exactly k x D after clamp)
        // assignment N output labels in [0, k) (caller pre-allocated, length N)
        // inertia    final total SSE (out)
        // iters      actual iteration count in [1, maxIter] (out)
        // ws         pre-allocated workspace -- Arena.fProxyKMeans_WS(N, D, k)
        public static void kmeans(
            in fProxyMxN X,
            int k,
            uint seed,
            int maxIter,
            ref fProxyMxN centroids,
            ref Indices assignment,
            out fProxy inertia,
            out int iters,
            ref fProxyKMeans_WS ws
        )

        // ALLOCATING CONVENIENCE WRAPPER.
        // Allocates centroids (k x D), assignment (N), and workspace from arena,
        // then delegates to the workspace overload above.
        public static void kmeans(
            ref Arena arena,
            in fProxyMxN X,
            int k,
            uint seed,
            int maxIter,
            out fProxyMxN centroids,
            out Indices assignment,
            out fProxy inertia,
            out int iters
        )
    }
}
`

No default-valued fProxy parameters (CS1750 in templates). These signatures have no fProxy defaults.

---

## 5. Algorithm in detail

### 5.1 Input guards (both overloads, before seeding)

- X.M_Rows == 0 || X.N_Cols == 0  ->  InvalidOperationException("kmeans: X is empty")
- k <= 0  ->  ArgumentException("kmeans: k must be >= 1")
- maxIter < 1  ->  ArgumentException("kmeans: maxIter must be >= 1")
- Internal clamp: k = math.min(k, N)  (documented; not thrown)
- centroids.M_Rows != k || centroids.N_Cols != D  ->  ArgumentException (shape mismatch)
- assignment.N != N  ->  ArgumentException (size mismatch)
- Each ws field checked against expected dimension; throw ArgumentException with field name.

### 5.2 Precompute point norms (once, before loop)

    for n = 0..N-1:
        fProxy s = (fProxy)0
        for f = 0..D-1: fProxy v = X[n, f]; s += v * v
        ws.PointNormSq[n] = s

Direct loop. StatsOP.rowNormL2 (StatsOP.fProxy.cs line 562) computes sqrt and is not usable here.
Initialize all ws.PrevAssignment[n] = -1 so the first iteration counts all N as changed.

### 5.3 K-means++ seeding

    var rng = new Random(seed == 0u ? 1u : seed)   // Unity.Mathematics.Random, unmanaged

    // First centroid: uniform random point
    int firstIdx = (int)(rng.NextFProxy() * N)
    for f = 0..D-1: centroids[0, f] = X[firstIdx, f]

    // Centroids 1..k-1 via D^2 weighting
    for j = 1 to k-1:
        // D^2[n] = min over chosen centroids [0..j-1] of ||x_n - c_i||^2
        for n = 0..N-1:
            fProxy best = fProxy.MaxValue
            for i = 0..j-1:
                fProxy d2 = (fProxy)0
                for f = 0..D-1:
                    fProxy diff = X[n, f] - centroids[i, f]
                    d2 += diff * diff
                if d2 < best: best = d2
            ws.D2Weights[n] = best

        // Fallback if all D^2 == 0 (all points identical)
        fProxy total = (fProxy)0
        for n = 0..N-1: total += ws.D2Weights[n]
        int nextIdx
        if !(total > (fProxy)0):
            nextIdx = (int)(rng.NextFProxy() * N)   // uniform fallback
        else:
            // fProxyRandom_OP.weightedPick -- Random_OP.fProxy.cs line 164
            // Validates weights finite + non-negative + total > 0 before drawing
            nextIdx = fProxyRandom_OP.weightedPick(in ws.D2Weights, ref rng)
        for f = 0..D-1: centroids[j, f] = X[nextIdx, f]

Seeding cost: O(k^2 * N * D). Acceptable for k << N. See OQ2 for uniform-random alternative.

### 5.4 Lloyd iteration loop

    iters = 0
    inertia = fProxy.MaxValue

    for iter = 0 to maxIter-1:

        // 5.4.1  Centroid squared norms
        for j = 0..k-1:
            fProxy s = (fProxy)0
            for f = 0..D-1: fProxy v = centroids[j, f]; s += v * v
            ws.CentNormSq[j] = s

        // 5.4.2  Transpose centroids (k x D) -> ws.Ct (D x k)
        fProxy_OP.trans(in centroids, ref ws.Ct)
        // OP.Dot.fProxy.cs line 183; Ct must not alias centroids (guaranteed by workspace)

        // 5.4.3  GEMM: ws.Gram = X * ws.Ct  (N x k)
        fProxy_OP.dot(in X, in ws.Ct, ref ws.Gram)
        // OP.Dot.fProxy.cs line 128, transposeA=false
        // dot zero-clears Gram before accumulating (OP.Dot.fProxy.cs line 156)

        // 5.4.4  Patch Gram in-place: score[n,j] = cn[j] - 2*G[n,j]
        for n = 0..N-1:
            for j = 0..k-1:
                ws.Gram[n, j] = ws.CentNormSq[j] - 2 * ws.Gram[n, j]
        // pn[n] omitted from score matrix (constant over j -- no effect on argmin)

        // 5.4.5  Assignment and change count
        for n = 0..N-1: ws.PrevAssignment[n] = assignment[n]
        fProxyQuery_OP.rowArgMin(in ws.Gram, ref assignment)  // QueryOP.fProxy.cs line 115
        int changes = 0
        for n = 0..N-1: if assignment[n] != ws.PrevAssignment[n]: changes++

        // 5.4.6  Inertia from score matrix + PointNormSq
        // ||x_n - c_{assignment[n]}||^2 = PointNormSq[n] + Gram[n, assignment[n]]
        fProxy sse = (fProxy)0
        for n = 0..N-1: sse += ws.PointNormSq[n] + ws.Gram[n, assignment[n]]
        inertia = sse
        iters = iter + 1

        // 5.4.7  Convergence check (after inertia so it reflects this assignment)
        if changes == 0: break

        // 5.4.8  Zero accumulator and counts
        for j = 0..k-1:
            for f = 0..D-1: ws.NewCentroids[j, f] = (fProxy)0
            ws.ClusterCounts[j] = 0

        // 5.4.9  Accumulate points into cluster sums
        for n = 0..N-1:
            int j = assignment[n]
            ws.ClusterCounts[j]++
            for f = 0..D-1: ws.NewCentroids[j, f] += X[n, f]

        // 5.4.10  Empty-cluster reseed (before dividing accumulators)
        // ws.Gram still holds step-5.4.4 scores; ws.PointNormSq is constant.
        for j = 0..k-1:
            if ws.ClusterCounts[j] != 0: continue
            fProxy maxDist = (fProxy)(-1)
            int farthestPt = 0
            for n = 0..N-1:
                fProxy dist = ws.PointNormSq[n] + ws.Gram[n, assignment[n]]
                if dist > maxDist: maxDist = dist; farthestPt = n
            for f = 0..D-1: ws.NewCentroids[j, f] = X[farthestPt, f]
            ws.ClusterCounts[j] = 1   // sentinel to avoid divide-by-zero in 5.4.11

        // 5.4.11  Divide accumulators -> new centroids
        for j = 0..k-1:
            fProxy invN = (fProxy)1 / (fProxy)ws.ClusterCounts[j]
            for f = 0..D-1: centroids[j, f] = ws.NewCentroids[j, f] * invN

### 5.5 Allocating wrapper body

    public static void kmeans(ref Arena arena, in fProxyMxN X, int k, uint seed, int maxIter,
        out fProxyMxN centroids, out Indices assignment, out fProxy inertia, out int iters)
    {
        int N = X.M_Rows, D = X.N_Cols;
        int kk = math.min(math.max(k, 1), N);
        centroids  = arena.fProxyMat(kk, D);
        assignment = arena.Indices(N);
        var ws     = arena.fProxyKMeans_WS(N, D, kk);
        kmeans(in X, k, seed, maxIter, ref centroids, ref assignment, out inertia, out iters, ref ws);
    }

---

## 6. Edge-case behavior

Scenario                       | Behavior
------------------------------ | --------
N == 0                         | Throw InvalidOperationException before any work
k <= 0                         | Throw ArgumentException
maxIter < 1                    | Throw ArgumentException
k >= N                         | Clamp k = N; one seeding step per point; one assignment iter
k == 1                         | Single centroid = global mean; inertia = total sq deviation
Duplicate points               | D^2 all 0 after first centroid -> uniform-random fallback
Empty cluster mid-run          | Reseed to farthest-from-its-centroid point (step 5.4.10)
maxIter == 1                   | One assignment + one centroid update; inertia from that assignment
All points identical           | Seeding uniform fallback; all centroids at same location; fine

---

## 7. Tests outline

File: Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/KMeansTests.fProxy.cs

Structure follows QueryTests.fProxy.cs: TestJob : IJob with TestType enum for Burst tests;
plain [Test] for managed-throw guards (no Burst context needed for throw paths).

T1 -- Separable blobs: known centroids and zero inertia
  N=20 points: rows 0..9 all at (0,0), rows 10..19 all at (10,10). D=2, k=2, seed=1, maxIter=10.
  Both centroids within 1e-4 of ground-truth means. Each point assigned to correct half. Inertia == 0.

T2 -- Three-cluster toy: correct assignment
  30 points in three groups at (0,0), (20,0), (0,20), 10 each. k=3, seed=7, maxIter=20.
  Each returned centroid within 1.0 of its true group mean. All points in correct group.

T3 -- Inertia upper bound
  Same data as T1. Run maxIter=10 to convergence. Assert inertia <= 0 + epsilon (known answer).
  Standard Lloyd is monotone non-increasing per iteration for fixed metric; documented in XML summary.

T4 -- Assignment matches brute-force nearest centroid
  After kmeans completes on any dataset, for each point n compute argmin_j ||x_n - c_j||^2 by
  direct loop. Compare to assignment[n]. Must be identical for every n.

T5 -- Determinism for fixed seed
  Call kmeans twice with identical X, k, seed, maxIter. Assert centroids and assignment bit-identical.

T6 -- k==1 equals global mean
  k=1, any X with N=6 and D=2, seed=1. Assert centroid[0,f] == colMean(X)[f] within 1e-4.
  Uses fProxyStats_OP.colMean (StatsOP.fProxy.cs line 329 for the ref-dest form).
  Inertia == sum_n ||x_n - mean||^2.

T7 -- k >= N: each point its own centroid
  5 distinct points at well-separated locations, k=10 (clamped to 5 internally).
  Assert iters==1, inertia <= 1e-6.

T8 -- Empty-cluster reseed: no exception, all centroids finite
  5 points with only 2 distinct values, k=3. Assert completes without exception.
  Assert all 3 centroids: each component is finite (math.isfinite check, no NaN, no Inf).

T9 -- Workspace overload and allocating wrapper agree
  Same X, k, seed, maxIter. Call both overloads. Assert centroids bit-exact, assignment identical,
  inertia bit-exact.

T10 -- Guard throws (managed [Test], not in Burst job)
  N==0 -> InvalidOperationException.
  k==0 -> ArgumentException.
  maxIter==0 -> ArgumentException.
  centroids shape mismatch -> ArgumentException.
  assignment.N mismatch -> ArgumentException.

---

## 8. Files to touch

Template paths -- coder edits ONLY these. Never touch Source/Generated.

  NEW:  Assets/LinearAlgebra/CodeGen/TemplateSource/ML/KMeans.fProxy.cs
  NEW:  Assets/LinearAlgebra/CodeGen/TemplateSource/ML/KMeans.Workspace.fProxy.cs
  NEW:  Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/KMeansTests.fProxy.cs

Referenced existing files (read-only for coder):
  Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QueryOP.fProxy.cs          rowArgMin line 115
  Assets/LinearAlgebra/CodeGen/TemplateSource/OP/OP.Dot.fProxy.cs           dot line 128, trans line 183
  Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.Workspace.fProxy.cs    workspace pattern
  Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Random_OP.fProxy.cs         weightedPick line 164
  Assets/LinearAlgebra/CodeGen/TemplateSource/Statistics/StatsOP.fProxy.cs  colMean line 329
  Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/Arena.cs                Indices(n) factory line 83

Do NOT edit Arena.cs. The factory goes in KMeans.Workspace.fProxy.cs as a partial struct Arena
extension, same pattern as SVD.Workspace.fProxy.cs line 21.

---

## 9. Acceptance criteria

1. Codegen produces KMeans.float.cs, KMeans.double.cs, KMeans.Workspace.float.cs, and
   KMeans.Workspace.double.cs with zero compile errors.
2. fProxyKMeans_WS is unmanaged (Burst-safe). Arena.fProxyKMeans_WS(N,D,k) compiles
   inside a Burst IJob with no managed-type errors.
3. The workspace overload contains no managed allocations in the compute path
   (no 
ew except 
ew Random(...)).
4. Tests T1 through T10 all pass in the Unity Test Runner headless (Tools/*.ps1).
5. Both float and double generated variants pass their own test runs independently.
6. T6: centroid[0] matches floatStats_OP.colMean output within 1e-4 (float).
7. T5: two identical-arg calls produce bit-identical centroids, assignment, and inertia.
8. No CS1750 compile error (no default-valued fProxy parameters in the template).

---

## 10. Open questions (flag for user decision)

OQ1 -- Squared-Euclidean only?
The GEMM trick is specific to L2^2. Supporting cosine or Manhattan requires per-point inner-loop
scoring and loses the GEMM speedup. Recommendation: start with L2^2 only; add a Metric enum
parameter in a v2 spec if needed.

OQ2 -- Uniform-random init option?
k-means++ costs O(k^2 * N * D). For large k or tight budgets, uniform-random init (O(k)) may be
preferred. Could be a KMeansInit enum {KMeansPlusPlus, UniformRandom} with a forwarding overload.
Not specced here; flag if desired.

OQ3 -- Mini-batch k-means?
Reduces per-iteration cost from O(N*D*k) to O(B*D*k) where B << N. A random-subset picker
(fProxyRandom_OP.weightedPickInpl already exists) would be needed. Separate spec recommended.

OQ4 -- Multiple restarts / best-inertia selection?
Trivially done by caller: call the workspace overload n_init times, compare inertia, keep best.
Document as a usage pattern in the XML summary of the allocating wrapper, not in the API itself.

OQ5 -- Masked k-means (cluster only a subset of rows)?
Consistent with QueryOP Indices-mask design. Deferred; flag if needed.
