using System;

using LinearAlgebra;
using LinearAlgebra.ML;        // opt-in: KMeans.fit, KMeansInit, fProxyKMeansCache

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// K-means (LinearAlgebra.ML.KMeans.fit) — squared-Euclidean Lloyd with GEMM assignment
// and k-means++ / Uniform seeding. Tests mirror the SolverBattery / RollingWindow idiom: a Burst
// [BurstCompile(FloatPrecision.High)] IJob carries a TestType enum, a Fail NativeArray diagnostic
// channel, and a [TestCaseSource] driver; the managed-throw guard paths run as plain [Test]s on the
// main thread (no Burst context needed to observe a throw).
//
// Cases (task T1–T10):
//   1 SeparableBlobs        — 3 well-separated coincident blobs → centroids == blob centers, inertia≈0
//   2 BruteForceNearest     — assignment[n] == argmin_j ‖xₙ−cⱼ‖² (independent brute force); also maxIter=1
//   3 InertiaRecompute      — inertia ≥ 0 and == recomputed SSE(assignment, centroids); ==0 on blobs
//   4 KEqualsOneMean        — k==1 centroid == colMean(X); inertia == Σ‖xₙ−mean‖²
//   5 KGreaterEqualN        — k≥N clamped; every point its own cluster; inertia≈0; iters≤2
//   6 Determinism{PlusPlus,Uniform} — same seed/init ⇒ bit-identical centroids/assignment/inertia/iters
//   7 WorkspaceVsAllocating — primitive(+factory ws) == allocating wrapper, bit-exact
//   8 EmptyClusterReseed    — forced empty clusters ⇒ no throw, all-finite centroids, ≥2 distinct
//   9 BothInitsValid        — KMeansPlusPlus and Uniform both reach inertia≈0 on separable blobs
//  10 Guard throws          — empty X / k≤0 / maxIter<1 / shape mismatches → Argument/InvalidOperation
//
// Tolerances scale with Consts.fProxySqrtEps (float ≈ 3.45e-4, double ≈ 1.49e-8) so the SAME
// expression is loose for float and tight for double. Inertia/centroid facts here are exact in
// exact arithmetic (coincident integer blobs, exact means), so tiny sqrtEps-scaled bands suffice.
public class fProxyKMeansTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            SeparableBlobs,
            BruteForceNearest,
            InertiaRecompute,
            KEqualsOneMean,
            KGreaterEqualN,
            DeterminismPlusPlus,
            DeterminismUniform,
            WorkspaceVsAllocating,
            EmptyClusterReseed,
            BothInitsValid,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<fProxy> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.SeparableBlobs:        SeparableBlobs();        break;
                case TestType.BruteForceNearest:     BruteForceNearest();     break;
                case TestType.InertiaRecompute:      InertiaRecompute();      break;
                case TestType.KEqualsOneMean:        KEqualsOneMean();        break;
                case TestType.KGreaterEqualN:        KGreaterEqualN();        break;
                case TestType.DeterminismPlusPlus:   Determinism(KMeansInit.KMeansPlusPlus); break;
                case TestType.DeterminismUniform:    Determinism(KMeansInit.Uniform);        break;
                case TestType.WorkspaceVsAllocating: WorkspaceVsAllocating(); break;
                case TestType.EmptyClusterReseed:    EmptyClusterReseed();    break;
                case TestType.BothInitsValid:        BothInitsValid();        break;
            }
        }

        // T1 — separable blobs: returned centroids equal the blob centers and
        // inertia is (exactly) zero. Centers (0,0),(10,0),(0,10), four
        // coincident points each → blob mean == center, SSE == 0.
        void SeparableBlobs()
        {
            var X = Blobs3();   // 12×2, k target 3
            int k = 3, D = 2;

            var centroids = new fProxyMxN(k, D, Allocator.Temp);
            var assign    = new Indices(12, Allocator.Temp);
            var ws        = new fProxyKMeansCache(12, D, k, Allocator.Temp);
            KMeans.fit(in X, k, 1u, 20, ref centroids, ref assign, out fProxy inertia, out int iters, ref ws);

            // each of the three centers is matched by exactly one centroid (tight band)
            fProxy ctol = (fProxy)50 * Consts.fProxySqrtEps;
            AssertTrue(MinCentroidDistSq((fProxy)0,  (fProxy)0,  in centroids, k) <= ctol);
            AssertTrue(MinCentroidDistSq((fProxy)10, (fProxy)0,  in centroids, k) <= ctol);
            AssertTrue(MinCentroidDistSq((fProxy)0,  (fProxy)10, in centroids, k) <= ctol);

            // inertia exactly zero for coincident blobs
            fProxy itol = (fProxy)100 * Consts.fProxySqrtEps;
            AssertTrue(inertia >= (fProxy)0);
            AssertClose(inertia, (fProxy)0, itol);

            // sanity: converged in a bounded number of iters
            AssertTrue(iters >= 1 && iters <= 20);
        }

        // T2 — assignment == brute-force nearest centroid (the final-sync contract).
        // Run on well-separated data (no fp tie ambiguity) at maxIter=1 (forces
        // the non-converged exit) and at maxIter=20.
        void BruteForceNearest()
        {
            var X = TwoSpreadClusters(); // 6×2, well separated, k target 2
            CheckBruteForce(in X, 2, 11u, 1);   // non-converged path
            CheckBruteForce(in X, 2, 11u, 20);  // converged path

            var B = Blobs3();            // 12×2, k target 3
            CheckBruteForce(in B, 3, 5u, 20);
        }

        void CheckBruteForce(in fProxyMxN X, int k, uint seed, int maxIter)
        {
            int N = X.M_Rows, D = X.N_Cols;
            var centroids = new fProxyMxN(k, D, Allocator.Temp);
            var assign    = new Indices(N, Allocator.Temp);
            var ws        = new fProxyKMeansCache(N, D, k, Allocator.Temp);
            KMeans.fit(in X, k, seed, maxIter, ref centroids, ref assign, out _, out _, ref ws);

            for (int n = 0; n < N; n++)
            {
                int bf = BruteArgMin(in X, n, in centroids, k, D);
                RecordEq(assign[n], bf);
            }
        }

        // T3 — inertia is non-negative and equals the independently recomputed
        // SSE Σ‖xₙ − c_{assignment[n]}‖² of the returned (assignment,centroids).
        // On separable blobs it is exactly 0.
        void InertiaRecompute()
        {
            // (a) spread clusters → positive inertia, must match recompute
            var X = TwoSpreadClusters();
            {
                int N = X.M_Rows, D = X.N_Cols, k = 2;
                var centroids = new fProxyMxN(k, D, Allocator.Temp);
                var assign    = new Indices(N, Allocator.Temp);
                var ws        = new fProxyKMeansCache(N, D, k, Allocator.Temp);
                KMeans.fit(in X, k, 3u, 20, ref centroids, ref assign, out fProxy inertia, out _, ref ws);

                AssertTrue(inertia >= (fProxy)0);
                fProxy sse = RecomputeSSE(in X, in centroids, in assign, N, D);
                fProxy tol = (inertia + (fProxy)1) * (fProxy)200 * Consts.fProxySqrtEps;
                AssertClose(inertia, sse, tol);
            }

            // (b) coincident blobs → inertia exactly 0
            var B = Blobs3();
            {
                int N = B.M_Rows, D = B.N_Cols, k = 3;
                var centroids = new fProxyMxN(k, D, Allocator.Temp);
                var assign    = new Indices(N, Allocator.Temp);
                var ws        = new fProxyKMeansCache(N, D, k, Allocator.Temp);
                KMeans.fit(in B, k, 9u, 20, ref centroids, ref assign, out fProxy inertia, out _, ref ws);

                AssertTrue(inertia >= (fProxy)0);
                AssertClose(inertia, (fProxy)0, (fProxy)100 * Consts.fProxySqrtEps);
            }
        }

        // T4 — k == 1: the single centroid is the global mean (colMean). Guards
        // against returning a seed point. inertia == Σ‖xₙ − mean‖².
        void KEqualsOneMean()
        {
            // 6 points with an easy-to-mean spread.
            int N = 6, D = 2;
            var X = new fProxyMxN(N, D, Allocator.Temp);
            X[0, 0] = (fProxy)1;  X[0, 1] = (fProxy)2;
            X[1, 0] = (fProxy)3;  X[1, 1] = (fProxy)(-4);
            X[2, 0] = (fProxy)5;  X[2, 1] = (fProxy)6;
            X[3, 0] = (fProxy)(-7); X[3, 1] = (fProxy)8;
            X[4, 0] = (fProxy)9;  X[4, 1] = (fProxy)0;
            X[5, 0] = (fProxy)2;  X[5, 1] = (fProxy)(-1);

            int k = 1;
            var centroids = new fProxyMxN(k, D, Allocator.Temp);
            var assign    = new Indices(N, Allocator.Temp);
            var ws        = new fProxyKMeansCache(N, D, k, Allocator.Temp);
            KMeans.fit(in X, k, 1u, 20, ref centroids, ref assign, out fProxy inertia, out _, ref ws);

            var mean = Stats.colMean(in X);   // length D
            fProxy ctol = (fProxy)50 * Consts.fProxySqrtEps;
            for (int f = 0; f < D; f++)
                AssertClose(centroids[0, f], mean[f], ctol);

            // all points belong to cluster 0
            for (int n = 0; n < N; n++)
                RecordEq(assign[n], 0);

            // inertia == Σ‖xₙ − mean‖²
            fProxy sse = (fProxy)0;
            for (int n = 0; n < N; n++)
            {
                fProxy d2 = (fProxy)0;
                for (int f = 0; f < D; f++) { fProxy diff = X[n, f] - mean[f]; d2 += diff * diff; }
                sse += d2;
            }
            fProxy itol = (sse + (fProxy)1) * (fProxy)200 * Consts.fProxySqrtEps;
            AssertTrue(inertia >= (fProxy)0);
            AssertClose(inertia, sse, itol);
        }

        // T5 — k ≥ N: k clamps to N, every point is its own cluster (sits on its
        // centroid), inertia ≈ 0, iters ≤ 2, assignment is a bijection.
        void KGreaterEqualN()
        {
            int N = 5, D = 2;
            var X = new fProxyMxN(N, D, Allocator.Temp);
            X[0, 0] = (fProxy)0;   X[0, 1] = (fProxy)0;
            X[1, 0] = (fProxy)100; X[1, 1] = (fProxy)0;
            X[2, 0] = (fProxy)0;   X[2, 1] = (fProxy)100;
            X[3, 0] = (fProxy)100; X[3, 1] = (fProxy)100;
            X[4, 0] = (fProxy)50;  X[4, 1] = (fProxy)200;

            // allocating wrapper clamps internally to kk = min(10, 5) = 5
            KMeans.fit(in X, 10, 7u, 20,
                out fProxyMxN centroids, out Indices assign, out fProxy inertia, out int iters, Allocator.Temp);

            RecordEq(centroids.M_Rows, N);   // clamped to N rows
            AssertTrue(iters <= 2);

            // each point sits exactly on its assigned centroid → inertia ≈ 0
            AssertTrue(inertia >= (fProxy)0);
            AssertClose(inertia, (fProxy)0, (fProxy)100 * Consts.fProxySqrtEps);
            for (int n = 0; n < N; n++)
                AssertClose(DistSq(in X, n, in centroids, assign[n], D), (fProxy)0, (fProxy)100 * Consts.fProxySqrtEps);

            // assignment is a bijection (all N labels distinct)
            RecordEq(DistinctCount(in assign, N), N);
        }

        // T6 — determinism: two runs with identical seed/init produce bit-identical
        // centroids, assignment, inertia, and iters. Exercised for both inits.
        void Determinism(KMeansInit init)
        {
            var X = TwoSpreadClusters();  // 6×2
            int N = X.M_Rows, D = X.N_Cols, k = 2;
            uint seed = 1234u;

            var c1 = new fProxyMxN(k, D, Allocator.Temp); var a1 = new Indices(N, Allocator.Temp); var w1 = new fProxyKMeansCache(N, D, k, Allocator.Temp);
            var c2 = new fProxyMxN(k, D, Allocator.Temp); var a2 = new Indices(N, Allocator.Temp); var w2 = new fProxyKMeansCache(N, D, k, Allocator.Temp);

            KMeans.fit(in X, k, seed, 20, init, ref c1, ref a1, out fProxy in1, out int it1, ref w1);
            KMeans.fit(in X, k, seed, 20, init, ref c2, ref a2, out fProxy in2, out int it2, ref w2);

            RecordEq(it1, it2);
            AssertExact(in1, in2);
            for (int n = 0; n < N; n++) RecordEq(a1[n], a2[n]);
            for (int j = 0; j < k; j++)
                for (int f = 0; f < D; f++)
                    AssertExact(c1[j, f], c2[j, f]);
        }

        // T7 — workspace (primitive + factory ws) and allocating wrapper agree
        // bit-exactly for identical inputs/seed/init.
        void WorkspaceVsAllocating()
        {
            var X = TwoSpreadClusters();  // 6×2
            int N = X.M_Rows, D = X.N_Cols, k = 2;
            uint seed = 99u;

            var cP = new fProxyMxN(k, D, Allocator.Temp); var aP = new Indices(N, Allocator.Temp); var ws = new fProxyKMeansCache(N, D, k, Allocator.Temp);
            KMeans.fit(in X, k, seed, 20, KMeansInit.KMeansPlusPlus, ref cP, ref aP, out fProxy inP, out int itP, ref ws);

            KMeans.fit(in X, k, seed, 20, KMeansInit.KMeansPlusPlus,
                out fProxyMxN cA, out Indices aA, out fProxy inA, out int itA, Allocator.Temp);

            RecordEq(itP, itA);
            AssertExact(inP, inA);
            for (int n = 0; n < N; n++) RecordEq(aP[n], aA[n]);
            for (int j = 0; j < k; j++)
                for (int f = 0; f < D; f++)
                    AssertExact(cP[j, f], cA[j, f]);
        }

        // T8 — empty-cluster reseed: a 2-location duplicate-point set with k=4
        // forces ≥2 empty clusters in the first update (k-means++ falls back to
        // uniform once all D² weights collapse → duplicate centroids → empties).
        // Assert: no throw, all centroid components finite, ≥2 distinct centroids.
        void EmptyClusterReseed()
        {
            // 4 points at only 2 distinct locations → with k=4, two clusters end up empty.
            int N = 4, D = 2, k = 4;
            var X = new fProxyMxN(N, D, Allocator.Temp);
            X[0, 0] = (fProxy)0;  X[0, 1] = (fProxy)0;
            X[1, 0] = (fProxy)0;  X[1, 1] = (fProxy)0;
            X[2, 0] = (fProxy)10; X[2, 1] = (fProxy)10;
            X[3, 0] = (fProxy)10; X[3, 1] = (fProxy)10;

            var centroids = new fProxyMxN(k, D, Allocator.Temp);
            var assign    = new Indices(N, Allocator.Temp);
            var ws        = new fProxyKMeansCache(N, D, k, Allocator.Temp);
            KMeans.fit(in X, k, 2u, 20, ref centroids, ref assign, out fProxy inertia, out _, ref ws);

            // every centroid component finite (reseed must not produce NaN/Inf via divide-by-zero)
            for (int j = 0; j < k; j++)
                for (int f = 0; f < D; f++)
                    AssertTrue(math.isfinite(centroids[j, f]));

            AssertTrue(inertia >= (fProxy)0);
            AssertTrue(math.isfinite(inertia));

            AssertTrue(DistinctCentroidCount(in centroids, k, D) >= 2);
        }

        // T9 — both seeding modes converge to inertia ≈ 0 on the separable blobs.
        void BothInitsValid()
        {
            var X = Blobs3();  // 12×2
            CheckZeroInertia(in X, 3, 4u, KMeansInit.KMeansPlusPlus);
            CheckZeroInertia(in X, 3, 4u, KMeansInit.Uniform);
        }

        void CheckZeroInertia(in fProxyMxN X, int k, uint seed, KMeansInit init)
        {
            int N = X.M_Rows, D = X.N_Cols;
            var centroids = new fProxyMxN(k, D, Allocator.Temp);
            var assign    = new Indices(N, Allocator.Temp);
            var ws        = new fProxyKMeansCache(N, D, k, Allocator.Temp);
            KMeans.fit(in X, k, seed, 30, init, ref centroids, ref assign, out fProxy inertia, out _, ref ws);

            AssertTrue(inertia >= (fProxy)0);
            AssertClose(inertia, (fProxy)0, (fProxy)100 * Consts.fProxySqrtEps);
        }

        // datasets

        // 12×2: three coincident blobs of 4 points each at (0,0),(10,0),(0,10).
        fProxyMxN Blobs3()
        {
            var X = new fProxyMxN(12, 2, Allocator.Temp);
            for (int i = 0; i < 4; i++)  { X[i, 0] = (fProxy)0;  X[i, 1] = (fProxy)0; }
            for (int i = 4; i < 8; i++)  { X[i, 0] = (fProxy)10; X[i, 1] = (fProxy)0; }
            for (int i = 8; i < 12; i++) { X[i, 0] = (fProxy)0;  X[i, 1] = (fProxy)10; }
            return X;
        }

        // 6×2: two well-separated spread clusters (gap ≫ intra-cluster spread).
        fProxyMxN TwoSpreadClusters()
        {
            var X = new fProxyMxN(6, 2, Allocator.Temp);
            X[0, 0] = (fProxy)0;  X[0, 1] = (fProxy)0;
            X[1, 0] = (fProxy)1;  X[1, 1] = (fProxy)0;
            X[2, 0] = (fProxy)0;  X[2, 1] = (fProxy)1;
            X[3, 0] = (fProxy)50; X[3, 1] = (fProxy)50;
            X[4, 0] = (fProxy)51; X[4, 1] = (fProxy)50;
            X[5, 0] = (fProxy)50; X[5, 1] = (fProxy)51;
            return X;
        }

        // numeric helpers

        // ‖X[n,:] − C[j,:]‖²
        fProxy DistSq(in fProxyMxN X, int n, in fProxyMxN C, int j, int D)
        {
            fProxy s = (fProxy)0;
            for (int f = 0; f < D; f++) { fProxy d = X[n, f] - C[j, f]; s += d * d; }
            return s;
        }

        // brute-force argmin_j ‖X[n,:] − C[j,:]‖² with first-index tie-break (matches rowArgMin's strict <).
        int BruteArgMin(in fProxyMxN X, int n, in fProxyMxN C, int k, int D)
        {
            fProxy best = DistSq(in X, n, in C, 0, D);
            int bestJ = 0;
            for (int j = 1; j < k; j++)
            {
                fProxy d = DistSq(in X, n, in C, j, D);
                if (d < best) { best = d; bestJ = j; }
            }
            return bestJ;
        }

        // Σₙ ‖X[n,:] − C[assignment[n],:]‖²
        fProxy RecomputeSSE(in fProxyMxN X, in fProxyMxN C, in Indices assign, int N, int D)
        {
            fProxy s = (fProxy)0;
            for (int n = 0; n < N; n++)
                s += DistSq(in X, n, in C, assign[n], D);
            return s;
        }

        // min over centroids of ‖(cx,cy) − C[j,:]‖² (used only for the 2-D blob test).
        fProxy MinCentroidDistSq(fProxy cx, fProxy cy, in fProxyMxN C, int k)
        {
            fProxy best = fProxy.MaxValue;
            for (int j = 0; j < k; j++)
            {
                fProxy dx = C[j, 0] - cx, dy = C[j, 1] - cy;
                fProxy d = dx * dx + dy * dy;
                if (d < best) best = d;
            }
            return best;
        }

        // number of distinct integer labels in the first N entries.
        int DistinctCount(in Indices a, int N)
        {
            int count = 0;
            for (int i = 0; i < N; i++)
            {
                bool seen = false;
                for (int j = 0; j < i; j++) if (a[j] == a[i]) { seen = true; break; }
                if (!seen) count++;
            }
            return count;
        }

        // number of distinct centroid rows (exact equality on every component).
        int DistinctCentroidCount(in fProxyMxN C, int k, int D)
        {
            int count = 0;
            for (int i = 0; i < k; i++)
            {
                bool seen = false;
                for (int j = 0; j < i; j++)
                {
                    bool same = true;
                    for (int f = 0; f < D; f++) if (C[i, f] != C[j, f]) { same = false; break; }
                    if (same) { seen = true; break; }
                }
                if (!seen) count++;
            }
            return count;
        }

        // ---- Fail-array diagnostics (layout: [0]=flag, [1]=got, [2]=expected, [3]=diff) ----
        void AssertClose(fProxy a, fProxy b, fProxy precision)
        {
            fProxy diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = a; Fail[2] = b; Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertExact(fProxy a, fProxy b)
        {
            if (!(a == b) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = a; Fail[2] = b; Fail[3] = a - b;
            }
            Assert.IsTrue(a == b);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = (fProxy)0; Fail[2] = (fProxy)1; Fail[3] = (fProxy)1;
            }
            Assert.IsTrue(ok);
        }

        void RecordEq(int got, int expected)
        {
            if (got != expected && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = got; Fail[2] = expected; Fail[3] = got - expected;
            }
            Assert.AreEqual(expected, got);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void KMeansTests(TestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff {fail[3]} ({e.Message})");
            throw;
        }
        finally { fail.Dispose(); }
    }

    // T10 — managed guard throws (main thread; throw paths need no Burst).

    [Test]
    public void EmptyXThrows()
    {
        var X = new fProxyMxN(0, 2, Allocator.Temp);   // N == 0
        Assert.Throws<InvalidOperationException>(() =>
            KMeans.fit(in X, 2, 1u, 10,
                out fProxyMxN c, out Indices a, out fProxy inertia, out int iters, Allocator.Temp));
    }

    [Test]
    public void NonPositiveKThrows()
    {
        var X = new fProxyMxN(4, 2, Allocator.Temp);
        Assert.Throws<ArgumentException>(() =>
            KMeans.fit(in X, 0, 1u, 10,
                out fProxyMxN c, out Indices a, out fProxy inertia, out int iters, Allocator.Temp));
    }

    [Test]
    public void NonPositiveMaxIterThrows()
    {
        var X = new fProxyMxN(4, 2, Allocator.Temp);
        Assert.Throws<ArgumentException>(() =>
            KMeans.fit(in X, 2, 1u, 0,
                out fProxyMxN c, out Indices a, out fProxy inertia, out int iters, Allocator.Temp));
    }

    [Test]
    public void CentroidShapeMismatchThrows()
    {
        int N = 6, D = 2, k = 2;
        var X  = new fProxyMxN(N, D, Allocator.Temp);
        var ws = new fProxyKMeansCache(N, D, k, Allocator.Temp);
        var assign = new Indices(N, Allocator.Temp);
        var badCentroids = new fProxyMxN(k + 1, D, Allocator.Temp);   // wrong row count
        Assert.Throws<ArgumentException>(() =>
            KMeans.fit(in X, k, 1u, 10, ref badCentroids, ref assign, out fProxy inertia, out int iters, ref ws));
    }

    [Test]
    public void AssignmentSizeMismatchThrows()
    {
        int N = 6, D = 2, k = 2;
        var X  = new fProxyMxN(N, D, Allocator.Temp);
        var ws = new fProxyKMeansCache(N, D, k, Allocator.Temp);
        var centroids = new fProxyMxN(k, D, Allocator.Temp);
        var badAssign = new Indices(N + 1, Allocator.Temp);           // wrong length
        Assert.Throws<ArgumentException>(() =>
            KMeans.fit(in X, k, 1u, 10, ref centroids, ref badAssign, out fProxy inertia, out int iters, ref ws));
    }

    [Test]
    public void WorkspaceShapeMismatchThrows()
    {
        int N = 6, D = 2, k = 2;
        var X  = new fProxyMxN(N, D, Allocator.Temp);
        var centroids = new fProxyMxN(k, D, Allocator.Temp);
        var assign    = new Indices(N, Allocator.Temp);
        var badWs = new fProxyKMeansCache(N, D, k + 1, Allocator.Temp);   // ws sized for wrong k
        Assert.Throws<ArgumentException>(() =>
            KMeans.fit(in X, k, 1u, 10, ref centroids, ref assign, out fProxy inertia, out int iters, ref badWs));
    }
}
