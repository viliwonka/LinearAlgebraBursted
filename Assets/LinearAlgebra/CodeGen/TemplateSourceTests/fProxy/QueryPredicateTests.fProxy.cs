using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;

// Tests for the predicate-filtered / score-based QueryOP extension (fProxyQuery_OP partial class).
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
public class fProxyQueryPredicateTests
{
    // -------------------------------------------------------------------------
    // Test functor structs (nested -> per-type-distinct after codegen).
    // -------------------------------------------------------------------------

    struct GreaterThanScalar : IfProxyPredicate
    {
        public fProxy t;
        public bool Test(fProxy x) => x > t;
    }

    struct RowSumAbove : IfProxyRowPredicate
    {
        public fProxy t;
        public bool Test(in fProxyMxN A, int r)
        {
            fProxy s = (fProxy)0;
            for (int c = 0; c < A.N_Cols; c++) s += A[r, c];
            return s > t;
        }
    }

    struct ColSumAbove : IfProxyColPredicate
    {
        public fProxy t;
        public bool Test(in fProxyMxN A, int c)
        {
            fProxy s = (fProxy)0;
            for (int r = 0; r < A.M_Rows; r++) s += A[r, c];
            return s > t;
        }
    }

    struct RowL2Score : IfProxyRowScore
    {
        public fProxy Score(in fProxyMxN A, int r)
        {
            fProxy s = (fProxy)0;
            for (int c = 0; c < A.N_Cols; c++) s += A[r, c] * A[r, c];
            return s;
        }
    }

    struct ColL2Score : IfProxyColScore
    {
        public fProxy Score(in fProxyMxN A, int c)
        {
            fProxy s = (fProxy)0;
            for (int r = 0; r < A.M_Rows; r++) s += A[r, c] * A[r, c];
            return s;
        }
    }

    struct EvenRow : IfProxyRowPredicate { public bool Test(in fProxyMxN A, int r) => (r & 1) == 0; }
    struct EvenCol : IfProxyColPredicate { public bool Test(in fProxyMxN A, int c) => (c & 1) == 0; }

    struct AlwaysTrueRow  : IfProxyRowPredicate { public bool Test(in fProxyMxN A, int r) => true; }
    struct AlwaysFalseRow : IfProxyRowPredicate { public bool Test(in fProxyMxN A, int r) => false; }
    struct AlwaysTrueCol  : IfProxyColPredicate { public bool Test(in fProxyMxN A, int c) => true; }
    struct AlwaysFalseCol : IfProxyColPredicate { public bool Test(in fProxyMxN A, int c) => false; }

    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
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
        public NativeArray<fProxy> Fail;

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
            var v = arena.fProxyVec(6);
            v[0] = (fProxy)(-2); v[1] = (fProxy)0; v[2] = (fProxy)3;
            v[3] = (fProxy)1;    v[4] = (fProxy)4; v[5] = (fProxy)2;

            var pass = new GreaterThanScalar { t = (fProxy)2.5 };
            AssertEqI(fProxyQuery_OP.findFirst(in v, ref pass), 2);
            AssertEqI(fProxyQuery_OP.count(in v, ref pass), 2);
            AssertTrue(fProxyQuery_OP.any(in v, ref pass));
            // not all > 2.5 (the -2 fails) -> all == false.
            AssertTrue(!fProxyQuery_OP.all(in v, ref pass));

            var idx = arena.Indices(6);
            int fc = fProxyQuery_OP.findAll(in v, ref pass, ref idx);
            AssertEqI(fc, 2);
            AssertEqI(idx[0], 2); AssertEqI(idx[1], 4);
            // findAll count == count.
            AssertEqI(fc, fProxyQuery_OP.count(in v, ref pass));

            // No element matches -> findFirst -1, count 0, any false.
            var none = new GreaterThanScalar { t = (fProxy)100 };
            AssertEqI(fProxyQuery_OP.findFirst(in v, ref none), -1);
            AssertEqI(fProxyQuery_OP.count(in v, ref none), 0);
            AssertTrue(!fProxyQuery_OP.any(in v, ref none));
            int nc = fProxyQuery_OP.findAll(in v, ref none, ref idx);
            AssertEqI(nc, 0);

            // Every element passes -> all true, any true.
            var allPass = new GreaterThanScalar { t = (fProxy)(-10) };
            AssertTrue(fProxyQuery_OP.all(in v, ref allPass));
            AssertTrue(fProxyQuery_OP.any(in v, ref allPass));

            // Empty vector: findFirst -1, count 0, any false, all true (vacuous), findAll 0.
            var v0 = arena.fProxyVec(0);
            AssertEqI(fProxyQuery_OP.findFirst(in v0, ref pass), -1);
            AssertEqI(fProxyQuery_OP.count(in v0, ref pass), 0);
            AssertTrue(!fProxyQuery_OP.any(in v0, ref pass));
            AssertTrue(fProxyQuery_OP.all(in v0, ref pass));
            var idx0 = arena.Indices(1);
            AssertEqI(fProxyQuery_OP.findAll(in v0, ref pass, ref idx0), 0);

            // Matrix flat-index variant (generic T over fProxyMxN, row-major flat order).
            // A = [1 5; 2 5] -> flat [1,5,2,5]; threshold 4 -> {5@1, 5@3}.
            var A = arena.fProxyMat(2, 2);
            A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)5;
            A[1, 0] = (fProxy)2; A[1, 1] = (fProxy)5;
            var matPass = new GreaterThanScalar { t = (fProxy)4 };
            AssertEqI(fProxyQuery_OP.findFirst(in A, ref matPass), 1);
            AssertEqI(fProxyQuery_OP.count(in A, ref matPass), 2);
            var idxM = arena.Indices(4);
            int mc = fProxyQuery_OP.findAll(in A, ref matPass, ref idxM);
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
            var A = arena.fProxyMat(4, 3);
            A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)0; A[0, 2] = (fProxy)0;
            A[1, 0] = (fProxy)2; A[1, 1] = (fProxy)2; A[1, 2] = (fProxy)0;
            A[2, 0] = (fProxy)0; A[2, 1] = (fProxy)0; A[2, 2] = (fProxy)0;
            A[3, 0] = (fProxy)3; A[3, 1] = (fProxy)1; A[3, 2] = (fProxy)1;

            // EvenRow -> rows {0,2}.
            var even = new EvenRow();
            var idxR = arena.Indices(4);
            int er = fProxyQuery_OP.whichRows(in A, ref even, ref idxR);
            AssertEqI(er, 2);
            AssertEqI(idxR[0], 0); AssertEqI(idxR[1], 2);
            AssertEqI(fProxyQuery_OP.countRows(in A, ref even), 2);

            // RowSumAbove(1.5) -> rows {1,3} (sums 4,5).
            var rsa = new RowSumAbove { t = (fProxy)1.5 };
            int rr = fProxyQuery_OP.whichRows(in A, ref rsa, ref idxR);
            AssertEqI(rr, 2);
            AssertEqI(idxR[0], 1); AssertEqI(idxR[1], 3);
            AssertEqI(fProxyQuery_OP.countRows(in A, ref rsa), 2);

            // AlwaysTrueRow -> all rows in order.
            var atr = new AlwaysTrueRow();
            int allc = fProxyQuery_OP.whichRows(in A, ref atr, ref idxR);
            AssertEqI(allc, 4);
            AssertEqI(idxR[0], 0); AssertEqI(idxR[1], 1); AssertEqI(idxR[2], 2); AssertEqI(idxR[3], 3);
            AssertEqI(fProxyQuery_OP.countRows(in A, ref atr), 4);

            // AlwaysFalseRow -> 0.
            var afr = new AlwaysFalseRow();
            AssertEqI(fProxyQuery_OP.whichRows(in A, ref afr, ref idxR), 0);
            AssertEqI(fProxyQuery_OP.countRows(in A, ref afr), 0);

            // Column twin: EvenCol -> cols {0,2}; ColSumAbove(2) -> {c0=6, c1=3} = {0,1}.
            var evenC = new EvenCol();
            var idxC = arena.Indices(3);
            int ec = fProxyQuery_OP.whichColumns(in A, ref evenC, ref idxC);
            AssertEqI(ec, 2);
            AssertEqI(idxC[0], 0); AssertEqI(idxC[1], 2);
            AssertEqI(fProxyQuery_OP.countColumns(in A, ref evenC), 2);

            var csa = new ColSumAbove { t = (fProxy)2 };
            int cc = fProxyQuery_OP.whichColumns(in A, ref csa, ref idxC);
            AssertEqI(cc, 2);
            AssertEqI(idxC[0], 0); AssertEqI(idxC[1], 1);
            AssertEqI(fProxyQuery_OP.countColumns(in A, ref csa), 2);

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
            var A = arena.fProxyMat(3, 2);
            A[0, 0] = (fProxy)1; A[0, 1] = (fProxy)0;
            A[1, 0] = (fProxy)3; A[1, 1] = (fProxy)0;
            A[2, 0] = (fProxy)2; A[2, 1] = (fProxy)0;
            var q = arena.fProxyVec(2);
            q[0] = (fProxy)0; q[1] = (fProxy)0;

            // Oracle sanity: unmasked nearest really is r0.
            fProxyQuery_OP.nearestRow(in A, in q, Metric.SqEuclidean, out int ui, out fProxy us);
            AssertEqI(ui, 0); AssertClose(us, (fProxy)1, fEps());

            var pred = new RowSumAbove { t = (fProxy)1.5 };
            fProxyQuery_OP.nearestRowWhere(in A, in q, Metric.SqEuclidean, ref pred, out int mi, out fProxy ms);
            AssertEqI(mi, 2); AssertClose(ms, (fProxy)4, fEps());

            // Column twin: columns are the points (M_Rows=2, q length 2).
            //  c0=(1,0) c1=(3,0) c2=(2,0); ColSumAbove(1.5) rejects c0 -> nearest = c2.
            var B = arena.fProxyMat(2, 3);
            B[0, 0] = (fProxy)1; B[0, 1] = (fProxy)3; B[0, 2] = (fProxy)2;
            B[1, 0] = (fProxy)0; B[1, 1] = (fProxy)0; B[1, 2] = (fProxy)0;
            var cpred = new ColSumAbove { t = (fProxy)1.5 };
            fProxyQuery_OP.nearestColumnWhere(in B, in q, Metric.SqEuclidean, ref cpred, out int cmi, out fProxy cms);
            AssertEqI(cmi, 2); AssertClose(cms, (fProxy)4, fEps());

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // GROUP C — ALL-PASS EQUIVALENCE (AC#3/AC#4) + AlwaysFalse sentinel.
        // ---------------------------------------------------------------------

        void AllPassEquivalence()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 6, N = 3;
            var A = arena.fProxyRandomMatrix(M, N, -3f, 3f, 424242);
            var q = arena.fProxyVec(N);
            q[0] = (fProxy)0.5; q[1] = (fProxy)(-1); q[2] = (fProxy)2;

            var atr = new AlwaysTrueRow();
            var afr = new AlwaysFalseRow();

            // --- nearestRowWhere(AlwaysTrue) == nearestRow exactly (same code path). ---
            fProxyQuery_OP.nearestRow(in A, in q, Metric.SqEuclidean, out int ni, out fProxy ns);
            fProxyQuery_OP.nearestRowWhere(in A, in q, Metric.SqEuclidean, ref atr, out int wi, out fProxy ws);
            AssertEqI(wi, ni); AssertClose(ws, ns, (fProxy)0);

            fProxyQuery_OP.nearestRow(in A, in q, Metric.Dot, out int nid, out fProxy nsd);
            fProxyQuery_OP.nearestRowWhere(in A, in q, Metric.Dot, ref atr, out int wid, out fProxy wsd);
            AssertEqI(wid, nid); AssertClose(wsd, nsd, (fProxy)0);

            // --- kNearestRowsWhere(AlwaysTrue) byte-identical to kNearestRows (SqEuclidean + Dot). ---
            int k = 3;
            for (int mm = 0; mm < 2; mm++)
            {
                Metric m = mm == 0 ? Metric.SqEuclidean : Metric.Dot;
                var idxU = arena.Indices(k); var scU = arena.fProxyVec(k);
                var idxW = arena.Indices(k); var scW = arena.fProxyVec(k);
                int cU = fProxyQuery_OP.kNearestRows(in A, in q, k, m, ref idxU, ref scU);
                int cW = fProxyQuery_OP.kNearestRowsWhere(in A, in q, k, m, ref atr, ref idxW, ref scW);
                AssertEqI(cW, cU);
                for (int i = 0; i < cU; i++)
                {
                    AssertEqI(idxW[i], idxU[i]);
                    AssertClose(scW[i], scU[i], (fProxy)0);
                }
            }

            // --- AlwaysFalse: index == -1 and score == WorstScoreForNearest(m) for every metric. ---
            // Distance metrics -> fProxy.MaxValue; similarity metrics (Cosine/Dot) -> fProxy.MinValue.
            fProxyQuery_OP.nearestRowWhere(in A, in q, Metric.SqEuclidean, ref afr, out int fi1, out fProxy fs1);
            AssertEqI(fi1, -1); AssertClose(fs1, fProxy.MaxValue, (fProxy)0);

            fProxyQuery_OP.nearestRowWhere(in A, in q, Metric.Manhattan, ref afr, out int fi2, out fProxy fs2);
            AssertEqI(fi2, -1); AssertClose(fs2, fProxy.MaxValue, (fProxy)0);

            fProxyQuery_OP.nearestRowWhere(in A, in q, Metric.Cosine, ref afr, out int fi3, out fProxy fs3);
            AssertEqI(fi3, -1); AssertClose(fs3, fProxy.MinValue, (fProxy)0);

            fProxyQuery_OP.nearestRowWhere(in A, in q, Metric.Dot, ref afr, out int fi4, out fProxy fs4);
            AssertEqI(fi4, -1); AssertClose(fs4, fProxy.MinValue, (fProxy)0);

            // --- AlwaysFalse k-nearest -> 0. ---
            var idxF = arena.Indices(k);
            var scF = arena.fProxyVec(k);
            AssertEqI(fProxyQuery_OP.kNearestRowsWhere(in A, in q, k, Metric.SqEuclidean, ref afr, ref idxF, ref scF), 0);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // GROUP C — k <= 0 / empty matrix returns 0 without throwing (T5 partial).
        // ---------------------------------------------------------------------

        void EmptyAndZeroK()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 4, N = 3;
            var A = arena.fProxyRandomMatrix(M, N, -2f, 2f, 555);
            var q = arena.fProxyVec(N);
            q[0] = (fProxy)1; q[1] = (fProxy)0; q[2] = (fProxy)(-1);

            var atr = new AlwaysTrueRow();
            var idx = arena.Indices(3);
            var sc = arena.fProxyVec(3);

            AssertEqI(fProxyQuery_OP.kNearestRowsWhere(in A, in q, 0, Metric.SqEuclidean, ref atr, ref idx, ref sc), 0);
            AssertEqI(fProxyQuery_OP.kNearestRowsWhere(in A, in q, -1, Metric.SqEuclidean, ref atr, ref idx, ref sc), 0);

            // 0-row matrix -> 0 (returns before any q / size check).
            var A0 = arena.fProxyMat(0, N);
            AssertEqI(fProxyQuery_OP.kNearestRowsWhere(in A0, in q, 3, Metric.SqEuclidean, ref atr, ref idx, ref sc), 0);

            // Column twin: k<=0 and 0-column matrix -> 0.
            var atc = new AlwaysTrueCol();
            var qc = arena.fProxyVec(M);
            for (int i = 0; i < M; i++) qc[i] = (fProxy)(i - 1);
            AssertEqI(fProxyQuery_OP.kNearestColumnsWhere(in A, in qc, 0, Metric.SqEuclidean, ref atc, ref idx, ref sc), 0);
            var A0c = arena.fProxyMat(M, 0);
            AssertEqI(fProxyQuery_OP.kNearestColumnsWhere(in A0c, in qc, 3, Metric.SqEuclidean, ref atc, ref idx, ref sc), 0);

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
            var H = arena.fProxyMat(4, 2);
            H[0, 0] = (fProxy)1; H[0, 1] = (fProxy)0;
            H[1, 0] = (fProxy)3; H[1, 1] = (fProxy)0;
            H[2, 0] = (fProxy)2; H[2, 1] = (fProxy)0;
            H[3, 0] = (fProxy)5; H[3, 1] = (fProxy)0;

            var rs = new RowL2Score();
            // argMaxRowBy -> r3 (25); cross-check argMaxRowNorm(L2) (argmax monotone under sqrt).
            fProxyQuery_OP.argMaxRowBy(in H, ref rs, out int mxi, out fProxy mxs);
            AssertEqI(mxi, 3); AssertClose(mxs, (fProxy)25, fEps());
            AssertEqI(mxi, fProxyQuery_OP.argMaxRowNorm(in H, Norm.L2));

            // argMinRowBy -> r0 (1).
            fProxyQuery_OP.argMinRowBy(in H, ref rs, out int mni, out fProxy mns);
            AssertEqI(mni, 0); AssertClose(mns, (fProxy)1, fEps());

            // topKRowsBy k=2 -> best-first {r3=25, r1=9}, descending.
            var idxT = arena.Indices(2);
            var scT = arena.fProxyVec(2);
            int cT = fProxyQuery_OP.topKRowsBy(in H, ref rs, 2, ref idxT, ref scT);
            AssertEqI(cT, 2);
            AssertEqI(idxT[0], 3); AssertClose(scT[0], (fProxy)25, fEps());
            AssertEqI(idxT[1], 1); AssertClose(scT[1], (fProxy)9, fEps());
            AssertTrue(scT[0] >= scT[1]);

            // Column twin: argMaxColBy -> c0 (39); cross-check argMaxColNorm(L2).
            var cs = new ColL2Score();
            fProxyQuery_OP.argMaxColBy(in H, ref cs, out int cmi, out fProxy cms);
            AssertEqI(cmi, 0); AssertClose(cms, (fProxy)39, fEps());
            AssertEqI(cmi, fProxyQuery_OP.argMaxColNorm(in H, Norm.L2));

            // Random equivalence: argMaxRowBy == argMaxRowNorm(L2); argMaxColBy == argMaxColNorm(L2).
            var R = arena.fProxyRandomMatrix(7, 4, -3f, 3f, 909090);
            var rrs = new RowL2Score();
            fProxyQuery_OP.argMaxRowBy(in R, ref rrs, out int rmi, out fProxy _);
            AssertEqI(rmi, fProxyQuery_OP.argMaxRowNorm(in R, Norm.L2));
            var rcs = new ColL2Score();
            fProxyQuery_OP.argMaxColBy(in R, ref rcs, out int rci, out fProxy _);
            AssertEqI(rci, fProxyQuery_OP.argMaxColNorm(in R, Norm.L2));

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // SYMMETRY (spec P1): a column op on A == the row op on transpose(A).
        // ---------------------------------------------------------------------

        void Symmetry()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 5, N = 4;
            var A = arena.fProxyRandomMatrix(M, N, -3f, 3f, 20240628);
            var At = fProxy_OP.trans(A);   // N x M; column j of A == row j of At.

            // Column query length = A.M_Rows = M = At.N_Cols.
            var q = arena.fProxyVec(M);
            for (int i = 0; i < M; i++) q[i] = (fProxy)(i - 2) * (fProxy)0.7;

            // whichColumns(A) == whichRows(At) for equivalent sum predicates (same threshold).
            var cpred = new ColSumAbove { t = (fProxy)0 };
            var rpred = new RowSumAbove { t = (fProxy)0 };
            var cIdx = arena.Indices(N);
            var rIdx = arena.Indices(N);
            int cc = fProxyQuery_OP.whichColumns(in A, ref cpred, ref cIdx);
            int rc = fProxyQuery_OP.whichRows(in At, ref rpred, ref rIdx);
            AssertEqI(cc, rc);
            for (int i = 0; i < cc; i++) AssertEqI(cIdx[i], rIdx[i]);
            AssertEqI(fProxyQuery_OP.countColumns(in A, ref cpred),
                      fProxyQuery_OP.countRows(in At, ref rpred));

            // nearestColumnWhere(A) == nearestRowWhere(At).
            fProxyQuery_OP.nearestColumnWhere(in A, in q, Metric.SqEuclidean, ref cpred, out int cni, out fProxy cns);
            fProxyQuery_OP.nearestRowWhere(in At, in q, Metric.SqEuclidean, ref rpred, out int rni, out fProxy rns);
            AssertEqI(cni, rni); AssertClose(cns, rns, sqrtEps());

            // argMaxColBy(A) == argMaxRowBy(At) == argMaxColNorm(A, L2).
            var cscore = new ColL2Score();
            var rscore = new RowL2Score();
            fProxyQuery_OP.argMaxColBy(in A, ref cscore, out int aci, out fProxy _);
            fProxyQuery_OP.argMaxRowBy(in At, ref rscore, out int ari, out fProxy _);
            AssertEqI(aci, ari);
            AssertEqI(aci, fProxyQuery_OP.argMaxColNorm(in A, Norm.L2));

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // helpers (per-precision tolerances: float looser than double).
        // ---------------------------------------------------------------------

        static fProxy sqrtEps() => (fProxy)Consts.fProxySqrtEps;
        static fProxy fEps() => (fProxy)(100 * Consts.fProxyEpsilon);

        // Fail layout: [0]=flag, [1]=got, [2]=expected, [3]=diff
        void AssertClose(fProxy a, fProxy b, fProxy precision)
        {
            fProxy diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertEqI(int got, int expected)
        {
            if (got != expected && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = got;
                Fail[2] = expected;
                Fail[3] = got - expected;
            }
            Assert.AreEqual(expected, got);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = (fProxy)(-1);
                Fail[2] = (fProxy)(-1);
                Fail[3] = (fProxy)(-1);
            }
            Assert.IsTrue(ok);
        }
    }

    void RunJob(TestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"got {fail[1]}, expected {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (fProxy)0)
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
        var A = arena.fProxyMat(4, 3);              // M_Rows=4, N_Cols=3
        var q = arena.fProxyVec(3);

        // findAll: idx.N < x.Data.Length.
        var v = arena.fProxyVec(5);
        var gt = new GreaterThanScalar { t = (fProxy)0 };
        var smallFlat = arena.Indices(4);
        Assert.Throws<ArgumentException>(() => fProxyQuery_OP.findAll(in v, ref gt, ref smallFlat));

        // whichRows: idx.N < A.M_Rows.
        var even = new EvenRow();
        var smallRows = arena.Indices(3);
        Assert.Throws<ArgumentException>(() => fProxyQuery_OP.whichRows(in A, ref even, ref smallRows));

        // whichColumns: idx.N < A.N_Cols.
        var evenC = new EvenCol();
        var smallCols = arena.Indices(2);
        Assert.Throws<ArgumentException>(() => fProxyQuery_OP.whichColumns(in A, ref evenC, ref smallCols));

        // kNearestRowsWhere: idx.N < k (q valid, k>0 so it reaches the size guard).
        var atr = new AlwaysTrueRow();
        var smallK = arena.Indices(1);
        var scK = arena.fProxyVec(3);
        Assert.Throws<ArgumentException>(() =>
            fProxyQuery_OP.kNearestRowsWhere(in A, in q, 3, Metric.SqEuclidean, ref atr, ref smallK, ref scK));
        // kNearestRowsWhere: scores.N < k.
        var okK = arena.Indices(3);
        var smallScores = arena.fProxyVec(1);
        Assert.Throws<ArgumentException>(() =>
            fProxyQuery_OP.kNearestRowsWhere(in A, in q, 3, Metric.SqEuclidean, ref atr, ref okK, ref smallScores));

        // topKRowsBy: idx.N < k.
        var rs = new RowL2Score();
        var smallT = arena.Indices(1);
        var scT = arena.fProxyVec(3);
        Assert.Throws<ArgumentException>(() =>
            fProxyQuery_OP.topKRowsBy(in A, ref rs, 3, ref smallT, ref scT));

        arena.Dispose();
    }

    [Test]
    public void EmptyMatrixThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var q = arena.fProxyVec(3);
        var atr = new AlwaysTrueRow();
        var rs = new RowL2Score();

        // nearestRowWhere on a 0-row matrix -> InvalidOperationException.
        var A0 = arena.fProxyMat(0, 3);
        Assert.Throws<InvalidOperationException>(() =>
            fProxyQuery_OP.nearestRowWhere(in A0, in q, Metric.SqEuclidean, ref atr, out int _, out fProxy _));
        // argMaxRowBy on a 0-row matrix -> InvalidOperationException.
        Assert.Throws<InvalidOperationException>(() =>
            fProxyQuery_OP.argMaxRowBy(in A0, ref rs, out int _, out fProxy _));

        // Column twins: 0-column matrix.
        var atc = new AlwaysTrueCol();
        var cs = new ColL2Score();
        var qc = arena.fProxyVec(3);
        var A0c = arena.fProxyMat(3, 0);
        Assert.Throws<InvalidOperationException>(() =>
            fProxyQuery_OP.nearestColumnWhere(in A0c, in qc, Metric.SqEuclidean, ref atc, out int _, out fProxy _));
        Assert.Throws<InvalidOperationException>(() =>
            fProxyQuery_OP.argMaxColBy(in A0c, ref cs, out int _, out fProxy _));

        arena.Dispose();
    }

    [Test]
    public void QueryLengthMismatchThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.fProxyMat(3, 4);              // row ops need q.N==4; col ops need q.N==3
        var atr = new AlwaysTrueRow();
        var atc = new AlwaysTrueCol();

        var qBadRow = arena.fProxyVec(3);           // wrong for row ops (need 4)
        Assert.Throws<ArgumentException>(() =>
            fProxyQuery_OP.nearestRowWhere(in A, in qBadRow, Metric.SqEuclidean, ref atr, out int _, out fProxy _));

        var idxK = arena.Indices(2);
        var scK = arena.fProxyVec(2);
        Assert.Throws<ArgumentException>(() =>
            fProxyQuery_OP.kNearestRowsWhere(in A, in qBadRow, 2, Metric.SqEuclidean, ref atr, ref idxK, ref scK));

        var qBadCol = arena.fProxyVec(4);           // wrong for col ops (need 3)
        Assert.Throws<ArgumentException>(() =>
            fProxyQuery_OP.nearestColumnWhere(in A, in qBadCol, Metric.SqEuclidean, ref atc, out int _, out fProxy _));

        arena.Dispose();
    }
}
