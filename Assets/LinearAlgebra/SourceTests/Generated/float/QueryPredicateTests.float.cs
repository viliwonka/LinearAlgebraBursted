using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;

// Tests for the predicate-filtered / score-based QueryOP extension (Query partial class).
// Spec: docs/spec-predicate-queries.md (Section 6 = T1..T5).
//
// Groups under test (mirroring the spec):
//   A — Flat / scalar predicate ops: findFirst, count, any, all, findAll (vector + matrix flat).
//   B — Row / column filter: countRows/whichRows + countColumns/whichColumns.
//   C — Masked nearest / k-nearest: nearestRowWhere, kNearestRowsWhere + column twins;
//       the headline "reject the closest" correctness check, all-pass equivalence vs the
//       unmasked ops, and the AlwaysFalse -> index == -1 + WorstScoreForNearest(m) sentinel.
//   D — Score-based selection: argMaxRowBy / argMinRowBy / topKRowsBy + column twins, cross-checked
//       against argMaxRowNorm / argMaxColNorm (Norm.L2 argmax is monotone under sqrt).
//   Symmetry — a column op on A equals the row op on transpose(A) (spec P1).
//
// Burst-compatible computational tests live in TestJob; managed-throw guards are plain [Test]
// methods on the main thread. Functor structs are NESTED in the outer class so the generated
// float / double files do not collide on namespace-scope type names.
public class floatQueryPredicateTests
{
    // -------------------------------------------------------------------------
    // Test functor structs (nested -> per-type-distinct after codegen).
    // -------------------------------------------------------------------------

    struct GreaterThanScalar : IfloatPredicate
    {
        public float t;
        public bool Test(float x) => x > t;
    }

    struct RowSumAbove : IfloatRowPredicate
    {
        public float t;
        public bool Test(in floatMxN A, int r)
        {
            float s = (float)0;
            for (int c = 0; c < A.N_Cols; c++) s += A[r, c];
            return s > t;
        }
    }

    struct ColSumAbove : IfloatColPredicate
    {
        public float t;
        public bool Test(in floatMxN A, int c)
        {
            float s = (float)0;
            for (int r = 0; r < A.M_Rows; r++) s += A[r, c];
            return s > t;
        }
    }

    struct RowL2Score : IfloatRowScore
    {
        public float Score(in floatMxN A, int r)
        {
            float s = (float)0;
            for (int c = 0; c < A.N_Cols; c++) s += A[r, c] * A[r, c];
            return s;
        }
    }

    struct ColL2Score : IfloatColScore
    {
        public float Score(in floatMxN A, int c)
        {
            float s = (float)0;
            for (int r = 0; r < A.M_Rows; r++) s += A[r, c] * A[r, c];
            return s;
        }
    }

    struct EvenRow : IfloatRowPredicate { public bool Test(in floatMxN A, int r) => (r & 1) == 0; }
    struct EvenCol : IfloatColPredicate { public bool Test(in floatMxN A, int c) => (c & 1) == 0; }

    struct AlwaysTrueRow  : IfloatRowPredicate { public bool Test(in floatMxN A, int r) => true; }
    struct AlwaysFalseRow : IfloatRowPredicate { public bool Test(in floatMxN A, int r) => false; }
    struct AlwaysTrueCol  : IfloatColPredicate { public bool Test(in floatMxN A, int c) => true; }
    struct AlwaysFalseCol : IfloatColPredicate { public bool Test(in floatMxN A, int c) => false; }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            GroupAScalar,
            GroupBFilter,
            MaskedNearest,
            AllPassEquivalence,
            EmptyAndZeroK,
            GroupDScore,
            Symmetry,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected, [3] diff
        public NativeArray<float> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.GroupAScalar:       GroupAScalar();       break;
                case TestType.GroupBFilter:       GroupBFilter();       break;
                case TestType.MaskedNearest:      MaskedNearest();      break;
                case TestType.AllPassEquivalence: AllPassEquivalence(); break;
                case TestType.EmptyAndZeroK:      EmptyAndZeroK();      break;
                case TestType.GroupDScore:        GroupDScore();        break;
                case TestType.Symmetry:           Symmetry();           break;
            }
        }

        // ---------------------------------------------------------------------
        // GROUP A — FLAT / SCALAR PREDICATE OPS (T1)
        // ---------------------------------------------------------------------

        void GroupAScalar()
        {
            var arena = new Arena(Allocator.Persistent);

            // v = [-2, 0, 3, 1, 4, 2]; threshold 2.5 -> {3@2, 4@4} pass.
            var v = arena.floatVec(6);
            v[0] = (float)(-2); v[1] = (float)0; v[2] = (float)3;
            v[3] = (float)1;    v[4] = (float)4; v[5] = (float)2;

            var pass = new GreaterThanScalar { t = (float)2.5 };
            AssertEqI(Query.findFirst(in v, ref pass), 2);
            AssertEqI(Query.count(in v, ref pass), 2);
            AssertTrue(Query.any(in v, ref pass));
            // not all > 2.5 (the -2 fails) -> all == false.
            AssertTrue(!Query.all(in v, ref pass));

            var idx = arena.Indices(6);
            int fc = Query.findAll(in v, ref pass, ref idx);
            AssertEqI(fc, 2);
            AssertEqI(idx[0], 2); AssertEqI(idx[1], 4);
            AssertEqI(fc, Query.count(in v, ref pass));

            // No element matches -> findFirst -1, count 0, any false.
            var none = new GreaterThanScalar { t = (float)100 };
            AssertEqI(Query.findFirst(in v, ref none), -1);
            AssertEqI(Query.count(in v, ref none), 0);
            AssertTrue(!Query.any(in v, ref none));
            int nc = Query.findAll(in v, ref none, ref idx);
            AssertEqI(nc, 0);

            // Every element passes -> all true, any true.
            var allPass = new GreaterThanScalar { t = (float)(-10) };
            AssertTrue(Query.all(in v, ref allPass));
            AssertTrue(Query.any(in v, ref allPass));

            // Empty vector: findFirst -1, count 0, any false, all true (vacuous), findAll 0.
            var v0 = arena.floatVec(0);
            AssertEqI(Query.findFirst(in v0, ref pass), -1);
            AssertEqI(Query.count(in v0, ref pass), 0);
            AssertTrue(!Query.any(in v0, ref pass));
            AssertTrue(Query.all(in v0, ref pass));
            var idx0 = arena.Indices(1);
            AssertEqI(Query.findAll(in v0, ref pass, ref idx0), 0);

            // Matrix flat-index variant (generic T over floatMxN, row-major flat order).
            // A = [1 5; 2 5] -> flat [1,5,2,5]; threshold 4 -> {5@1, 5@3}.
            var A = arena.floatMat(2, 2);
            A[0, 0] = (float)1; A[0, 1] = (float)5;
            A[1, 0] = (float)2; A[1, 1] = (float)5;
            var matPass = new GreaterThanScalar { t = (float)4 };
            AssertEqI(Query.findFirst(in A, ref matPass), 1);
            AssertEqI(Query.count(in A, ref matPass), 2);
            var idxM = arena.Indices(4);
            int mc = Query.findAll(in A, ref matPass, ref idxM);
            AssertEqI(mc, 2);
            AssertEqI(idxM[0], 1); AssertEqI(idxM[1], 3);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // GROUP B — ROW / COLUMN FILTER (T2)
        // ---------------------------------------------------------------------

        void GroupBFilter()
        {
            var arena = new Arena(Allocator.Persistent);

            // 4x3 with known row + column sums:
            //  r0: 1 0 0  -> sum 1
            //  r1: 2 2 0  -> sum 4
            //  r2: 0 0 0  -> sum 0
            //  r3: 3 1 1  -> sum 5
            //  col sums: c0=6, c1=3, c2=1.
            var A = arena.floatMat(4, 3);
            A[0, 0] = (float)1; A[0, 1] = (float)0; A[0, 2] = (float)0;
            A[1, 0] = (float)2; A[1, 1] = (float)2; A[1, 2] = (float)0;
            A[2, 0] = (float)0; A[2, 1] = (float)0; A[2, 2] = (float)0;
            A[3, 0] = (float)3; A[3, 1] = (float)1; A[3, 2] = (float)1;

            // EvenRow -> rows {0,2}.
            var even = new EvenRow();
            var idxR = arena.Indices(4);
            int er = Query.whichRows(in A, ref even, ref idxR);
            AssertEqI(er, 2);
            AssertEqI(idxR[0], 0); AssertEqI(idxR[1], 2);
            AssertEqI(Query.countRows(in A, ref even), 2);

            // RowSumAbove(1.5) -> rows {1,3} (sums 4,5).
            var rsa = new RowSumAbove { t = (float)1.5 };
            int rr = Query.whichRows(in A, ref rsa, ref idxR);
            AssertEqI(rr, 2);
            AssertEqI(idxR[0], 1); AssertEqI(idxR[1], 3);
            AssertEqI(Query.countRows(in A, ref rsa), 2);

            // AlwaysTrueRow -> all rows in order.
            var atr = new AlwaysTrueRow();
            int allc = Query.whichRows(in A, ref atr, ref idxR);
            AssertEqI(allc, 4);
            AssertEqI(idxR[0], 0); AssertEqI(idxR[1], 1); AssertEqI(idxR[2], 2); AssertEqI(idxR[3], 3);
            AssertEqI(Query.countRows(in A, ref atr), 4);

            // AlwaysFalseRow -> 0.
            var afr = new AlwaysFalseRow();
            AssertEqI(Query.whichRows(in A, ref afr, ref idxR), 0);
            AssertEqI(Query.countRows(in A, ref afr), 0);

            // Column twin: EvenCol -> cols {0,2}; ColSumAbove(2) -> {c0=6, c1=3} = {0,1}.
            var evenC = new EvenCol();
            var idxC = arena.Indices(3);
            int ec = Query.whichColumns(in A, ref evenC, ref idxC);
            AssertEqI(ec, 2);
            AssertEqI(idxC[0], 0); AssertEqI(idxC[1], 2);
            AssertEqI(Query.countColumns(in A, ref evenC), 2);

            var csa = new ColSumAbove { t = (float)2 };
            int cc = Query.whichColumns(in A, ref csa, ref idxC);
            AssertEqI(cc, 2);
            AssertEqI(idxC[0], 0); AssertEqI(idxC[1], 1);
            AssertEqI(Query.countColumns(in A, ref csa), 2);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // GROUP C — MASKED NEAREST: the headline correctness check (T3).
        // ---------------------------------------------------------------------

        void MaskedNearest()
        {
            var arena = new Arena(Allocator.Persistent);

            // Rows as 1D points on x-axis; q=(0,0). SqEuclidean dist^2: r0=1, r1=9, r2=4.
            // Unmasked nearest = r0. RowSumAbove(1.5) REJECTS r0 (sum 1) -> nearest among {r1,r2} = r2.
            var A = arena.floatMat(3, 2);
            A[0, 0] = (float)1; A[0, 1] = (float)0;
            A[1, 0] = (float)3; A[1, 1] = (float)0;
            A[2, 0] = (float)2; A[2, 1] = (float)0;
            var q = arena.floatVec(2);
            q[0] = (float)0; q[1] = (float)0;

            // Oracle sanity: unmasked nearest really is r0.
            Query.nearestRow(in A, in q, Metric.SqEuclidean, out int ui, out float us);
            AssertEqI(ui, 0); AssertClose(us, (float)1, fEps());

            var pred = new RowSumAbove { t = (float)1.5 };
            Query.nearestRowWhere(in A, in q, Metric.SqEuclidean, ref pred, out int mi, out float ms);
            AssertEqI(mi, 2); AssertClose(ms, (float)4, fEps());

            // Column twin: columns are the points (M_Rows=2, q length 2).
            //  c0=(1,0) c1=(3,0) c2=(2,0); ColSumAbove(1.5) rejects c0 -> nearest = c2.
            var B = arena.floatMat(2, 3);
            B[0, 0] = (float)1; B[0, 1] = (float)3; B[0, 2] = (float)2;
            B[1, 0] = (float)0; B[1, 1] = (float)0; B[1, 2] = (float)0;
            var cpred = new ColSumAbove { t = (float)1.5 };
            Query.nearestColumnWhere(in B, in q, Metric.SqEuclidean, ref cpred, out int cmi, out float cms);
            AssertEqI(cmi, 2); AssertClose(cms, (float)4, fEps());

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // GROUP C — ALL-PASS EQUIVALENCE (AC#3/AC#4) + AlwaysFalse sentinel.
        // ---------------------------------------------------------------------

        void AllPassEquivalence()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 6, N = 3;
            var A = arena.floatRandomMat(M, N, -3f, 3f, 424242);
            var q = arena.floatVec(N);
            q[0] = (float)0.5; q[1] = (float)(-1); q[2] = (float)2;

            var atr = new AlwaysTrueRow();
            var afr = new AlwaysFalseRow();

            // --- nearestRowWhere(AlwaysTrue) == nearestRow exactly (same code path). ---
            Query.nearestRow(in A, in q, Metric.SqEuclidean, out int ni, out float ns);
            Query.nearestRowWhere(in A, in q, Metric.SqEuclidean, ref atr, out int wi, out float ws);
            AssertEqI(wi, ni); AssertClose(ws, ns, (float)0);

            Query.nearestRow(in A, in q, Metric.Dot, out int nid, out float nsd);
            Query.nearestRowWhere(in A, in q, Metric.Dot, ref atr, out int wid, out float wsd);
            AssertEqI(wid, nid); AssertClose(wsd, nsd, (float)0);

            // --- kNearestRowsWhere(AlwaysTrue) byte-identical to kNearestRows (SqEuclidean + Dot). ---
            int k = 3;
            for (int mm = 0; mm < 2; mm++)
            {
                Metric m = mm == 0 ? Metric.SqEuclidean : Metric.Dot;
                var idxU = arena.Indices(k); var scU = arena.floatVec(k);
                var idxW = arena.Indices(k); var scW = arena.floatVec(k);
                int cU = Query.kNearestRows(in A, in q, k, m, ref idxU, ref scU);
                int cW = Query.kNearestRowsWhere(in A, in q, k, m, ref atr, ref idxW, ref scW);
                AssertEqI(cW, cU);
                for (int i = 0; i < cU; i++)
                {
                    AssertEqI(idxW[i], idxU[i]);
                    AssertClose(scW[i], scU[i], (float)0);
                }
            }

            // --- AlwaysFalse: index == -1 and score == WorstScoreForNearest(m) for every metric. ---
            // Distance metrics -> float.MaxValue; similarity metrics (Cosine/Dot) -> float.MinValue.
            Query.nearestRowWhere(in A, in q, Metric.SqEuclidean, ref afr, out int fi1, out float fs1);
            AssertEqI(fi1, -1); AssertClose(fs1, float.MaxValue, (float)0);

            Query.nearestRowWhere(in A, in q, Metric.Manhattan, ref afr, out int fi2, out float fs2);
            AssertEqI(fi2, -1); AssertClose(fs2, float.MaxValue, (float)0);

            Query.nearestRowWhere(in A, in q, Metric.Cosine, ref afr, out int fi3, out float fs3);
            AssertEqI(fi3, -1); AssertClose(fs3, float.MinValue, (float)0);

            Query.nearestRowWhere(in A, in q, Metric.Dot, ref afr, out int fi4, out float fs4);
            AssertEqI(fi4, -1); AssertClose(fs4, float.MinValue, (float)0);

            // --- AlwaysFalse k-nearest -> 0. ---
            var idxF = arena.Indices(k);
            var scF = arena.floatVec(k);
            AssertEqI(Query.kNearestRowsWhere(in A, in q, k, Metric.SqEuclidean, ref afr, ref idxF, ref scF), 0);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // GROUP C — k <= 0 / empty matrix returns 0 without throwing (T5 partial).
        // ---------------------------------------------------------------------

        void EmptyAndZeroK()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 4, N = 3;
            var A = arena.floatRandomMat(M, N, -2f, 2f, 555);
            var q = arena.floatVec(N);
            q[0] = (float)1; q[1] = (float)0; q[2] = (float)(-1);

            var atr = new AlwaysTrueRow();
            var idx = arena.Indices(3);
            var sc = arena.floatVec(3);

            AssertEqI(Query.kNearestRowsWhere(in A, in q, 0, Metric.SqEuclidean, ref atr, ref idx, ref sc), 0);
            AssertEqI(Query.kNearestRowsWhere(in A, in q, -1, Metric.SqEuclidean, ref atr, ref idx, ref sc), 0);

            // 0-row matrix -> 0 (returns before any q / size check).
            var A0 = arena.floatMat(0, N);
            AssertEqI(Query.kNearestRowsWhere(in A0, in q, 3, Metric.SqEuclidean, ref atr, ref idx, ref sc), 0);

            // Column twin: k<=0 and 0-column matrix -> 0.
            var atc = new AlwaysTrueCol();
            var qc = arena.floatVec(M);
            for (int i = 0; i < M; i++) qc[i] = (float)(i - 1);
            AssertEqI(Query.kNearestColumnsWhere(in A, in qc, 0, Metric.SqEuclidean, ref atc, ref idx, ref sc), 0);
            var A0c = arena.floatMat(M, 0);
            AssertEqI(Query.kNearestColumnsWhere(in A0c, in qc, 3, Metric.SqEuclidean, ref atc, ref idx, ref sc), 0);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // GROUP D — SCORE-BASED SELECTION (T4)
        // ---------------------------------------------------------------------

        void GroupDScore()
        {
            var arena = new Arena(Allocator.Persistent);

            // Hand matrix: row L2^2 norms 1, 9, 4, 25 (distinct).
            //  c0 = (1,3,2,5) -> L2^2 = 39 ; c1 = (0,0,0,0) -> 0.
            var H = arena.floatMat(4, 2);
            H[0, 0] = (float)1; H[0, 1] = (float)0;
            H[1, 0] = (float)3; H[1, 1] = (float)0;
            H[2, 0] = (float)2; H[2, 1] = (float)0;
            H[3, 0] = (float)5; H[3, 1] = (float)0;

            var rs = new RowL2Score();
            // argMaxRowBy -> r3 (25); cross-check argMaxRowNorm(L2) (argmax monotone under sqrt).
            Query.argMaxRowBy(in H, ref rs, out int mxi, out float mxs);
            AssertEqI(mxi, 3); AssertClose(mxs, (float)25, fEps());
            AssertEqI(mxi, Query.argMaxRowNorm(in H, Norm.L2));

            // argMinRowBy -> r0 (1).
            Query.argMinRowBy(in H, ref rs, out int mni, out float mns);
            AssertEqI(mni, 0); AssertClose(mns, (float)1, fEps());

            // topKRowsBy k=2 -> best-first {r3=25, r1=9}, descending.
            var idxT = arena.Indices(2);
            var scT = arena.floatVec(2);
            int cT = Query.topKRowsBy(in H, ref rs, 2, ref idxT, ref scT);
            AssertEqI(cT, 2);
            AssertEqI(idxT[0], 3); AssertClose(scT[0], (float)25, fEps());
            AssertEqI(idxT[1], 1); AssertClose(scT[1], (float)9, fEps());
            AssertTrue(scT[0] >= scT[1]);

            // Column twin: argMaxColBy -> c0 (39); cross-check argMaxColNorm(L2).
            var cs = new ColL2Score();
            Query.argMaxColBy(in H, ref cs, out int cmi, out float cms);
            AssertEqI(cmi, 0); AssertClose(cms, (float)39, fEps());
            AssertEqI(cmi, Query.argMaxColNorm(in H, Norm.L2));

            // Random equivalence: argMaxRowBy == argMaxRowNorm(L2); argMaxColBy == argMaxColNorm(L2).
            var R = arena.floatRandomMat(7, 4, -3f, 3f, 909090);
            var rrs = new RowL2Score();
            Query.argMaxRowBy(in R, ref rrs, out int rmi, out float _);
            AssertEqI(rmi, Query.argMaxRowNorm(in R, Norm.L2));
            var rcs = new ColL2Score();
            Query.argMaxColBy(in R, ref rcs, out int rci, out float _);
            AssertEqI(rci, Query.argMaxColNorm(in R, Norm.L2));

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // SYMMETRY (spec P1): a column op on A == the row op on transpose(A).
        // ---------------------------------------------------------------------

        void Symmetry()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 5, N = 4;
            var A = arena.floatRandomMat(M, N, -3f, 3f, 20240628);
            var At = Blas.trans(A);   // N x M; column j of A == row j of At.

            // Column query length = A.M_Rows = M = At.N_Cols.
            var q = arena.floatVec(M);
            for (int i = 0; i < M; i++) q[i] = (float)(i - 2) * (float)0.7;

            // whichColumns(A) == whichRows(At) for equivalent sum predicates (same threshold).
            var cpred = new ColSumAbove { t = (float)0 };
            var rpred = new RowSumAbove { t = (float)0 };
            var cIdx = arena.Indices(N);
            var rIdx = arena.Indices(N);
            int cc = Query.whichColumns(in A, ref cpred, ref cIdx);
            int rc = Query.whichRows(in At, ref rpred, ref rIdx);
            AssertEqI(cc, rc);
            for (int i = 0; i < cc; i++) AssertEqI(cIdx[i], rIdx[i]);
            AssertEqI(Query.countColumns(in A, ref cpred),
                      Query.countRows(in At, ref rpred));

            // nearestColumnWhere(A) == nearestRowWhere(At).
            Query.nearestColumnWhere(in A, in q, Metric.SqEuclidean, ref cpred, out int cni, out float cns);
            Query.nearestRowWhere(in At, in q, Metric.SqEuclidean, ref rpred, out int rni, out float rns);
            AssertEqI(cni, rni); AssertClose(cns, rns, sqrtEps());

            // argMaxColBy(A) == argMaxRowBy(At) == argMaxColNorm(A, L2).
            var cscore = new ColL2Score();
            var rscore = new RowL2Score();
            Query.argMaxColBy(in A, ref cscore, out int aci, out float _);
            Query.argMaxRowBy(in At, ref rscore, out int ari, out float _);
            AssertEqI(aci, ari);
            AssertEqI(aci, Query.argMaxColNorm(in A, Norm.L2));

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // helpers (per-precision tolerances: float looser than double).
        // ---------------------------------------------------------------------

        static float sqrtEps() => (float)Consts.floatSqrtEps;
        static float fEps() => (float)(100 * Consts.floatEpsilon);

        // Fail layout: [0]=flag, [1]=got, [2]=expected, [3]=diff
        void AssertClose(float a, float b, float precision)
        {
            float diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertEqI(int got, int expected)
        {
            if (got != expected && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = got;
                Fail[2] = expected;
                Fail[3] = got - expected;
            }
            Assert.AreEqual(expected, got);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = (float)(-1);
                Fail[2] = (float)(-1);
                Fail[3] = (float)(-1);
            }
            Assert.IsTrue(ok);
        }
    }

    void RunJob(TestJob.TestType type)
    {
        var fail = new NativeArray<float>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (float)0)
                Assert.Fail($"got {fail[1]}, expected {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (float)0)
                Assert.Fail($"{type}: got {fail[1]}, expected {fail[2]}, diff {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    [Test] public void GroupAScalarTest()       => RunJob(TestJob.TestType.GroupAScalar);
    [Test] public void GroupBFilterTest()       => RunJob(TestJob.TestType.GroupBFilter);
    [Test] public void MaskedNearestTest()      => RunJob(TestJob.TestType.MaskedNearest);
    [Test] public void AllPassEquivalenceTest() => RunJob(TestJob.TestType.AllPassEquivalence);
    [Test] public void EmptyAndZeroKTest()      => RunJob(TestJob.TestType.EmptyAndZeroK);
    [Test] public void GroupDScoreTest()        => RunJob(TestJob.TestType.GroupDScore);
    [Test] public void SymmetryTest()           => RunJob(TestJob.TestType.Symmetry);

    // -------------------------------------------------------------------------
    // Managed-throw guards (main thread): undersized Indices, empty matrices,
    // and query-length mismatches (T5).
    // -------------------------------------------------------------------------

    [Test]
    public void UndersizedBufferThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.floatMat(4, 3);              // M_Rows=4, N_Cols=3
        var q = arena.floatVec(3);

        // findAll: idx.N < x.Data.Length.
        var v = arena.floatVec(5);
        var gt = new GreaterThanScalar { t = (float)0 };
        var smallFlat = arena.Indices(4);
        Assert.Throws<ArgumentException>(() => Query.findAll(in v, ref gt, ref smallFlat));

        // whichRows: idx.N < A.M_Rows.
        var even = new EvenRow();
        var smallRows = arena.Indices(3);
        Assert.Throws<ArgumentException>(() => Query.whichRows(in A, ref even, ref smallRows));

        // whichColumns: idx.N < A.N_Cols.
        var evenC = new EvenCol();
        var smallCols = arena.Indices(2);
        Assert.Throws<ArgumentException>(() => Query.whichColumns(in A, ref evenC, ref smallCols));

        // kNearestRowsWhere: idx.N < k (q valid, k>0 so it reaches the size guard).
        var atr = new AlwaysTrueRow();
        var smallK = arena.Indices(1);
        var scK = arena.floatVec(3);
        Assert.Throws<ArgumentException>(() =>
            Query.kNearestRowsWhere(in A, in q, 3, Metric.SqEuclidean, ref atr, ref smallK, ref scK));
        // kNearestRowsWhere: scores.N < k.
        var okK = arena.Indices(3);
        var smallScores = arena.floatVec(1);
        Assert.Throws<ArgumentException>(() =>
            Query.kNearestRowsWhere(in A, in q, 3, Metric.SqEuclidean, ref atr, ref okK, ref smallScores));

        // topKRowsBy: idx.N < k.
        var rs = new RowL2Score();
        var smallT = arena.Indices(1);
        var scT = arena.floatVec(3);
        Assert.Throws<ArgumentException>(() =>
            Query.topKRowsBy(in A, ref rs, 3, ref smallT, ref scT));

        arena.Dispose();
    }

    [Test]
    public void EmptyMatrixThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var q = arena.floatVec(3);
        var atr = new AlwaysTrueRow();
        var rs = new RowL2Score();

        // nearestRowWhere on a 0-row matrix -> InvalidOperationException.
        var A0 = arena.floatMat(0, 3);
        Assert.Throws<InvalidOperationException>(() =>
            Query.nearestRowWhere(in A0, in q, Metric.SqEuclidean, ref atr, out int _, out float _));
        // argMaxRowBy on a 0-row matrix -> InvalidOperationException.
        Assert.Throws<InvalidOperationException>(() =>
            Query.argMaxRowBy(in A0, ref rs, out int _, out float _));

        // Column twins: 0-column matrix.
        var atc = new AlwaysTrueCol();
        var cs = new ColL2Score();
        var qc = arena.floatVec(3);
        var A0c = arena.floatMat(3, 0);
        Assert.Throws<InvalidOperationException>(() =>
            Query.nearestColumnWhere(in A0c, in qc, Metric.SqEuclidean, ref atc, out int _, out float _));
        Assert.Throws<InvalidOperationException>(() =>
            Query.argMaxColBy(in A0c, ref cs, out int _, out float _));

        arena.Dispose();
    }

    [Test]
    public void QueryLengthMismatchThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.floatMat(3, 4);              // row ops need q.N==4; col ops need q.N==3
        var atr = new AlwaysTrueRow();
        var atc = new AlwaysTrueCol();

        var qBadRow = arena.floatVec(3);           // wrong for row ops (need 4)
        Assert.Throws<ArgumentException>(() =>
            Query.nearestRowWhere(in A, in qBadRow, Metric.SqEuclidean, ref atr, out int _, out float _));

        var idxK = arena.Indices(2);
        var scK = arena.floatVec(2);
        Assert.Throws<ArgumentException>(() =>
            Query.kNearestRowsWhere(in A, in qBadRow, 2, Metric.SqEuclidean, ref atr, ref idxK, ref scK));

        var qBadCol = arena.floatVec(4);           // wrong for col ops (need 3)
        Assert.Throws<ArgumentException>(() =>
            Query.nearestColumnWhere(in A, in qBadCol, Metric.SqEuclidean, ref atc, out int _, out float _));

        arena.Dispose();
    }
}
