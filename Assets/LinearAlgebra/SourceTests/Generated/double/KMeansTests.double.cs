using System;

using LinearAlgebra;
using LinearAlgebra.ML;        // opt-in: doubleKMeans_OP.kmeans, KMeansInit, doubleKMeans_WS
using LinearAlgebra.Stats;     // doubleStats_OP.colMean (k==1 global-mean oracle)

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// K-means (LinearAlgebra.ML.doubleKMeans_OP.kmeans) — squared-Euclidean Lloyd with GEMM assignment
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
// Tolerances scale with Consts.doubleSqrtEps (float ≈ 3.45e-4, double ≈ 1.49e-8) so the SAME
// expression is loose for float and tight for double. Inertia/centroid facts here are exact in
// exact arithmetic (coincident integer blobs, exact means), so tiny sqrtEps-scaled bands suffice.
public class doubleKMeansTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
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
        public NativeArray<double> Fail;

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

        // =====================================================================
        // T1 — separable blobs: returned centroids equal the blob centers and
        //      inertia is (exactly) zero. Centers (0,0),(10,0),(0,10), four
        //      coincident points each → blob mean == center, SSE == 0.
        // =====================================================================
        void SeparableBlobs()
        {
            var arena = new Arena(Allocator.Persistent);

            var X = Blobs3(ref arena);   // 12×2, k target 3
            int k = 3, D = 2;

            var centroids = arena.doubleMat(k, D);
            var assign    = arena.Indices(12);
            var ws        = arena.doubleKMeans_WS(12, D, k);
            doubleKMeans_OP.kmeans(in X, k, 1u, 20, ref centroids, ref assign, out double inertia, out int iters, ref ws);

            // each of the three centers is matched by exactly one centroid (tight band)
            double ctol = (double)50 * Consts.doubleSqrtEps;
            AssertTrue(MinCentroidDistSq((double)0,  (double)0,  in centroids, k) <= ctol);
            AssertTrue(MinCentroidDistSq((double)10, (double)0,  in centroids, k) <= ctol);
            AssertTrue(MinCentroidDistSq((double)0,  (double)10, in centroids, k) <= ctol);

            // inertia exactly zero for coincident blobs
            double itol = (double)100 * Consts.doubleSqrtEps;
            AssertTrue(inertia >= (double)0);
            AssertClose(inertia, (double)0, itol);

            // sanity: converged in a bounded number of iters
            AssertTrue(iters >= 1 && iters <= 20);

            arena.Dispose();
        }

        // =====================================================================
        // T2 — assignment == brute-force nearest centroid (the final-sync contract).
        //      Run on well-separated data (no fp tie ambiguity) at maxIter=1 (forces
        //      the non-converged exit) and at maxIter=20.
        // =====================================================================
        void BruteForceNearest()
        {
            var arena = new Arena(Allocator.Persistent);

            var X = TwoSpreadClusters(ref arena); // 6×2, well separated, k target 2
            CheckBruteForce(ref arena, in X, 2, 11u, 1);   // non-converged path
            CheckBruteForce(ref arena, in X, 2, 11u, 20);  // converged path

            var B = Blobs3(ref arena);            // 12×2, k target 3
            CheckBruteForce(ref arena, in B, 3, 5u, 20);

            arena.Dispose();
        }

        void CheckBruteForce(ref Arena arena, in doubleMxN X, int k, uint seed, int maxIter)
        {
            int N = X.M_Rows, D = X.N_Cols;
            var centroids = arena.doubleMat(k, D);
            var assign    = arena.Indices(N);
            var ws        = arena.doubleKMeans_WS(N, D, k);
            doubleKMeans_OP.kmeans(in X, k, seed, maxIter, ref centroids, ref assign, out _, out _, ref ws);

            for (int n = 0; n < N; n++)
            {
                int bf = BruteArgMin(in X, n, in centroids, k, D);
                RecordEq(assign[n], bf);
            }
        }

        // =====================================================================
        // T3 — inertia is non-negative and equals the independently recomputed
        //      SSE Σ‖xₙ − c_{assignment[n]}‖² of the returned (assignment,centroids).
        //      On separable blobs it is exactly 0.
        // =====================================================================
        void InertiaRecompute()
        {
            var arena = new Arena(Allocator.Persistent);

            // (a) spread clusters → positive inertia, must match recompute
            var X = TwoSpreadClusters(ref arena);
            {
                int N = X.M_Rows, D = X.N_Cols, k = 2;
                var centroids = arena.doubleMat(k, D);
                var assign    = arena.Indices(N);
                var ws        = arena.doubleKMeans_WS(N, D, k);
                doubleKMeans_OP.kmeans(in X, k, 3u, 20, ref centroids, ref assign, out double inertia, out _, ref ws);

                AssertTrue(inertia >= (double)0);
                double sse = RecomputeSSE(in X, in centroids, in assign, N, D);
                double tol = (inertia + (double)1) * (double)200 * Consts.doubleSqrtEps;
                AssertClose(inertia, sse, tol);
            }

            // (b) coincident blobs → inertia exactly 0
            var B = Blobs3(ref arena);
            {
                int N = B.M_Rows, D = B.N_Cols, k = 3;
                var centroids = arena.doubleMat(k, D);
                var assign    = arena.Indices(N);
                var ws        = arena.doubleKMeans_WS(N, D, k);
                doubleKMeans_OP.kmeans(in B, k, 9u, 20, ref centroids, ref assign, out double inertia, out _, ref ws);

                AssertTrue(inertia >= (double)0);
                AssertClose(inertia, (double)0, (double)100 * Consts.doubleSqrtEps);
            }

            arena.Dispose();
        }

        // =====================================================================
        // T4 — k == 1: the single centroid is the global mean (colMean). Guards
        //      against returning a seed point. inertia == Σ‖xₙ − mean‖².
        // =====================================================================
        void KEqualsOneMean()
        {
            var arena = new Arena(Allocator.Persistent);

            // 6 points with an easy-to-mean spread.
            int N = 6, D = 2;
            var X = arena.doubleMat(N, D);
            X[0, 0] = (double)1;  X[0, 1] = (double)2;
            X[1, 0] = (double)3;  X[1, 1] = (double)(-4);
            X[2, 0] = (double)5;  X[2, 1] = (double)6;
            X[3, 0] = (double)(-7); X[3, 1] = (double)8;
            X[4, 0] = (double)9;  X[4, 1] = (double)0;
            X[5, 0] = (double)2;  X[5, 1] = (double)(-1);

            int k = 1;
            var centroids = arena.doubleMat(k, D);
            var assign    = arena.Indices(N);
            var ws        = arena.doubleKMeans_WS(N, D, k);
            doubleKMeans_OP.kmeans(in X, k, 1u, 20, ref centroids, ref assign, out double inertia, out _, ref ws);

            var mean = doubleStats_OP.colMean(in X);   // length D
            double ctol = (double)50 * Consts.doubleSqrtEps;
            for (int f = 0; f < D; f++)
                AssertClose(centroids[0, f], mean[f], ctol);

            // all points belong to cluster 0
            for (int n = 0; n < N; n++)
                RecordEq(assign[n], 0);

            // inertia == Σ‖xₙ − mean‖²
            double sse = (double)0;
            for (int n = 0; n < N; n++)
            {
                double d2 = (double)0;
                for (int f = 0; f < D; f++) { double diff = X[n, f] - mean[f]; d2 += diff * diff; }
                sse += d2;
            }
            double itol = (sse + (double)1) * (double)200 * Consts.doubleSqrtEps;
            AssertTrue(inertia >= (double)0);
            AssertClose(inertia, sse, itol);

            arena.Dispose();
        }

        // =====================================================================
        // T5 — k ≥ N: k clamps to N, every point is its own cluster (sits on its
        //      centroid), inertia ≈ 0, iters ≤ 2, assignment is a bijection.
        // =====================================================================
        void KGreaterEqualN()
        {
            var arena = new Arena(Allocator.Persistent);

            int N = 5, D = 2;
            var X = arena.doubleMat(N, D);
            X[0, 0] = (double)0;   X[0, 1] = (double)0;
            X[1, 0] = (double)100; X[1, 1] = (double)0;
            X[2, 0] = (double)0;   X[2, 1] = (double)100;
            X[3, 0] = (double)100; X[3, 1] = (double)100;
            X[4, 0] = (double)50;  X[4, 1] = (double)200;

            // allocating wrapper clamps internally to kk = min(10, 5) = 5
            doubleKMeans_OP.kmeans(ref arena, in X, 10, 7u, 20,
                out doubleMxN centroids, out Indices assign, out double inertia, out int iters);

            RecordEq(centroids.M_Rows, N);   // clamped to N rows
            AssertTrue(iters <= 2);

            // each point sits exactly on its assigned centroid → inertia ≈ 0
            AssertTrue(inertia >= (double)0);
            AssertClose(inertia, (double)0, (double)100 * Consts.doubleSqrtEps);
            for (int n = 0; n < N; n++)
                AssertClose(DistSq(in X, n, in centroids, assign[n], D), (double)0, (double)100 * Consts.doubleSqrtEps);

            // assignment is a bijection (all N labels distinct)
            RecordEq(DistinctCount(in assign, N), N);

            arena.Dispose();
        }

        // =====================================================================
        // T6 — determinism: two runs with identical seed/init produce bit-identical
        //      centroids, assignment, inertia, and iters. Exercised for both inits.
        // =====================================================================
        void Determinism(KMeansInit init)
        {
            var arena = new Arena(Allocator.Persistent);

            var X = TwoSpreadClusters(ref arena);  // 6×2
            int N = X.M_Rows, D = X.N_Cols, k = 2;
            uint seed = 1234u;

            var c1 = arena.doubleMat(k, D); var a1 = arena.Indices(N); var w1 = arena.doubleKMeans_WS(N, D, k);
            var c2 = arena.doubleMat(k, D); var a2 = arena.Indices(N); var w2 = arena.doubleKMeans_WS(N, D, k);

            doubleKMeans_OP.kmeans(in X, k, seed, 20, init, ref c1, ref a1, out double in1, out int it1, ref w1);
            doubleKMeans_OP.kmeans(in X, k, seed, 20, init, ref c2, ref a2, out double in2, out int it2, ref w2);

            RecordEq(it1, it2);
            AssertExact(in1, in2);
            for (int n = 0; n < N; n++) RecordEq(a1[n], a2[n]);
            for (int j = 0; j < k; j++)
                for (int f = 0; f < D; f++)
                    AssertExact(c1[j, f], c2[j, f]);

            arena.Dispose();
        }

        // =====================================================================
        // T7 — workspace (primitive + factory ws) and allocating wrapper agree
        //      bit-exactly for identical inputs/seed/init.
        // =====================================================================
        void WorkspaceVsAllocating()
        {
            var arena = new Arena(Allocator.Persistent);

            var X = TwoSpreadClusters(ref arena);  // 6×2
            int N = X.M_Rows, D = X.N_Cols, k = 2;
            uint seed = 99u;

            var cP = arena.doubleMat(k, D); var aP = arena.Indices(N); var ws = arena.doubleKMeans_WS(N, D, k);
            doubleKMeans_OP.kmeans(in X, k, seed, 20, KMeansInit.KMeansPlusPlus, ref cP, ref aP, out double inP, out int itP, ref ws);

            doubleKMeans_OP.kmeans(ref arena, in X, k, seed, 20, KMeansInit.KMeansPlusPlus,
                out doubleMxN cA, out Indices aA, out double inA, out int itA);

            RecordEq(itP, itA);
            AssertExact(inP, inA);
            for (int n = 0; n < N; n++) RecordEq(aP[n], aA[n]);
            for (int j = 0; j < k; j++)
                for (int f = 0; f < D; f++)
                    AssertExact(cP[j, f], cA[j, f]);

            arena.Dispose();
        }

        // =====================================================================
        // T8 — empty-cluster reseed: a 2-location duplicate-point set with k=4
        //      forces ≥2 empty clusters in the first update (k-means++ falls back to
        //      uniform once all D² weights collapse → duplicate centroids → empties).
        //      Assert: no throw, all centroid components finite, ≥2 distinct centroids.
        // =====================================================================
        void EmptyClusterReseed()
        {
            var arena = new Arena(Allocator.Persistent);

            // 4 points at only 2 distinct locations → with k=4, two clusters end up empty.
            int N = 4, D = 2, k = 4;
            var X = arena.doubleMat(N, D);
            X[0, 0] = (double)0;  X[0, 1] = (double)0;
            X[1, 0] = (double)0;  X[1, 1] = (double)0;
            X[2, 0] = (double)10; X[2, 1] = (double)10;
            X[3, 0] = (double)10; X[3, 1] = (double)10;

            var centroids = arena.doubleMat(k, D);
            var assign    = arena.Indices(N);
            var ws        = arena.doubleKMeans_WS(N, D, k);
            doubleKMeans_OP.kmeans(in X, k, 2u, 20, ref centroids, ref assign, out double inertia, out _, ref ws);

            // every centroid component finite (reseed must not produce NaN/Inf via divide-by-zero)
            for (int j = 0; j < k; j++)
                for (int f = 0; f < D; f++)
                    AssertTrue(math.isfinite(centroids[j, f]));

            AssertTrue(inertia >= (double)0);
            AssertTrue(math.isfinite(inertia));

            // both real locations are represented: ≥2 distinct centroid positions.
            AssertTrue(DistinctCentroidCount(in centroids, k, D) >= 2);

            arena.Dispose();
        }

        // =====================================================================
        // T9 — both seeding modes converge to inertia ≈ 0 on the separable blobs.
        // =====================================================================
        void BothInitsValid()
        {
            var arena = new Arena(Allocator.Persistent);

            var X = Blobs3(ref arena);  // 12×2
            CheckZeroInertia(ref arena, in X, 3, 4u, KMeansInit.KMeansPlusPlus);
            CheckZeroInertia(ref arena, in X, 3, 4u, KMeansInit.Uniform);

            arena.Dispose();
        }

        void CheckZeroInertia(ref Arena arena, in doubleMxN X, int k, uint seed, KMeansInit init)
        {
            int N = X.M_Rows, D = X.N_Cols;
            var centroids = arena.doubleMat(k, D);
            var assign    = arena.Indices(N);
            var ws        = arena.doubleKMeans_WS(N, D, k);
            doubleKMeans_OP.kmeans(in X, k, seed, 30, init, ref centroids, ref assign, out double inertia, out _, ref ws);

            AssertTrue(inertia >= (double)0);
            AssertClose(inertia, (double)0, (double)100 * Consts.doubleSqrtEps);
        }

        // =====================================================================
        // datasets
        // =====================================================================

        // 12×2: three coincident blobs of 4 points each at (0,0),(10,0),(0,10).
        doubleMxN Blobs3(ref Arena arena)
        {
            var X = arena.doubleMat(12, 2);
            for (int i = 0; i < 4; i++)  { X[i, 0] = (double)0;  X[i, 1] = (double)0; }
            for (int i = 4; i < 8; i++)  { X[i, 0] = (double)10; X[i, 1] = (double)0; }
            for (int i = 8; i < 12; i++) { X[i, 0] = (double)0;  X[i, 1] = (double)10; }
            return X;
        }

        // 6×2: two well-separated spread clusters (gap ≫ intra-cluster spread).
        doubleMxN TwoSpreadClusters(ref Arena arena)
        {
            var X = arena.doubleMat(6, 2);
            X[0, 0] = (double)0;  X[0, 1] = (double)0;
            X[1, 0] = (double)1;  X[1, 1] = (double)0;
            X[2, 0] = (double)0;  X[2, 1] = (double)1;
            X[3, 0] = (double)50; X[3, 1] = (double)50;
            X[4, 0] = (double)51; X[4, 1] = (double)50;
            X[5, 0] = (double)50; X[5, 1] = (double)51;
            return X;
        }

        // =====================================================================
        // numeric helpers
        // =====================================================================

        // ‖X[n,:] − C[j,:]‖²
        double DistSq(in doubleMxN X, int n, in doubleMxN C, int j, int D)
        {
            double s = (double)0;
            for (int f = 0; f < D; f++) { double d = X[n, f] - C[j, f]; s += d * d; }
            return s;
        }

        // brute-force argmin_j ‖X[n,:] − C[j,:]‖² with first-index tie-break (matches rowArgMin's strict <).
        int BruteArgMin(in doubleMxN X, int n, in doubleMxN C, int k, int D)
        {
            double best = DistSq(in X, n, in C, 0, D);
            int bestJ = 0;
            for (int j = 1; j < k; j++)
            {
                double d = DistSq(in X, n, in C, j, D);
                if (d < best) { best = d; bestJ = j; }
            }
            return bestJ;
        }

        // Σₙ ‖X[n,:] − C[assignment[n],:]‖²
        double RecomputeSSE(in doubleMxN X, in doubleMxN C, in Indices assign, int N, int D)
        {
            double s = (double)0;
            for (int n = 0; n < N; n++)
                s += DistSq(in X, n, in C, assign[n], D);
            return s;
        }

        // min over centroids of ‖(cx,cy) − C[j,:]‖² (used only for the 2-D blob test).
        double MinCentroidDistSq(double cx, double cy, in doubleMxN C, int k)
        {
            double best = double.MaxValue;
            for (int j = 0; j < k; j++)
            {
                double dx = C[j, 0] - cx, dy = C[j, 1] - cy;
                double d = dx * dx + dy * dy;
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
        int DistinctCentroidCount(in doubleMxN C, int k, int D)
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
        void AssertClose(double a, double b, double precision)
        {
            double diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1; Fail[1] = a; Fail[2] = b; Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertExact(double a, double b)
        {
            if (!(a == b) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1; Fail[1] = a; Fail[2] = b; Fail[3] = a - b;
            }
            Assert.IsTrue(a == b);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (double)0)
            {
                Fail[0] = (double)1; Fail[1] = (double)0; Fail[2] = (double)1; Fail[3] = (double)1;
            }
            Assert.IsTrue(ok);
        }

        void RecordEq(int got, int expected)
        {
            if (got != expected && Fail[0] == (double)0)
            {
                Fail[0] = (double)1; Fail[1] = got; Fail[2] = expected; Fail[3] = got - expected;
            }
            Assert.AreEqual(expected, got);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void KMeansTests(TestJob.TestType type)
    {
        var fail = new NativeArray<double>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff {fail[3]} ({e.Message})");
            throw;
        }
        finally { fail.Dispose(); }
    }

    // =====================================================================
    // T10 — managed guard throws (main thread; throw paths need no Burst).
    // =====================================================================

    [Test]
    public void EmptyXThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var X = arena.doubleMat(0, 2);   // N == 0
        Assert.Throws<InvalidOperationException>(() =>
            doubleKMeans_OP.kmeans(ref arena, in X, 2, 1u, 10,
                out doubleMxN c, out Indices a, out double inertia, out int iters));
        arena.Dispose();
    }

    [Test]
    public void NonPositiveKThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var X = arena.doubleMat(4, 2);
        Assert.Throws<ArgumentException>(() =>
            doubleKMeans_OP.kmeans(ref arena, in X, 0, 1u, 10,
                out doubleMxN c, out Indices a, out double inertia, out int iters));
        arena.Dispose();
    }

    [Test]
    public void NonPositiveMaxIterThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var X = arena.doubleMat(4, 2);
        Assert.Throws<ArgumentException>(() =>
            doubleKMeans_OP.kmeans(ref arena, in X, 2, 1u, 0,
                out doubleMxN c, out Indices a, out double inertia, out int iters));
        arena.Dispose();
    }

    [Test]
    public void CentroidShapeMismatchThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        int N = 6, D = 2, k = 2;
        var X  = arena.doubleMat(N, D);
        var ws = arena.doubleKMeans_WS(N, D, k);
        var assign = arena.Indices(N);
        var badCentroids = arena.doubleMat(k + 1, D);   // wrong row count
        Assert.Throws<ArgumentException>(() =>
            doubleKMeans_OP.kmeans(in X, k, 1u, 10, ref badCentroids, ref assign, out double inertia, out int iters, ref ws));
        arena.Dispose();
    }

    [Test]
    public void AssignmentSizeMismatchThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        int N = 6, D = 2, k = 2;
        var X  = arena.doubleMat(N, D);
        var ws = arena.doubleKMeans_WS(N, D, k);
        var centroids = arena.doubleMat(k, D);
        var badAssign = arena.Indices(N + 1);           // wrong length
        Assert.Throws<ArgumentException>(() =>
            doubleKMeans_OP.kmeans(in X, k, 1u, 10, ref centroids, ref badAssign, out double inertia, out int iters, ref ws));
        arena.Dispose();
    }

    [Test]
    public void WorkspaceShapeMismatchThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        int N = 6, D = 2, k = 2;
        var X  = arena.doubleMat(N, D);
        var centroids = arena.doubleMat(k, D);
        var assign    = arena.Indices(N);
        var badWs = arena.doubleKMeans_WS(N, D, k + 1);   // ws sized for wrong k
        Assert.Throws<ArgumentException>(() =>
            doubleKMeans_OP.kmeans(in X, k, 1u, 10, ref centroids, ref assign, out double inertia, out int iters, ref badWs));
        arena.Dispose();
    }
}
