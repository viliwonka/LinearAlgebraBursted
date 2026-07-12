using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;

// Tests for QueryOP Phase 2 (the iProxy integer subset: int/short/long): the integer-exact
// search & selection ops. One template expands to
// intQuery_OP / shortQuery_OP / longQuery_OP, so every literal must be exact AND safe for the
// TIGHTEST type (short): coordinates kept small so Manhattan/Chebyshev differences fit the
// type, and SqEuclidean/Dot accumulations fit short.MaxValue = 32767. Type extremes use the
// proxy constants iProxy.MinValue / iProxy.MaxValue (which expand per type).
//
// Coverage groups (mirroring the spec; decodeIndex is float-only so it is NOT exercised here):
//   1 — Extremes: argMaxAbs/argMinAbs (vec+matrix), rowArgMin/Max + colArgMin/Max (value+index
//       and index-only), strided columns.
//   2 — Norm-selection: argMaxRowNorm/argMaxColNorm for L1 and Linf; Norm.L2 throws (float-only).
//   3 — Search: distancesToRow/Column for every integer metric (Manhattan/Chebyshev/SqEuclidean/
//       Dot), nearest/farthest + Column twins, kNearest/kFarthest vs brute force, within-radius
//       boundary; Metric.Euclidean and Metric.Cosine throw (float-only).
//   4 — Value/mask: findValue, nonzero/countNonzero (tol=0 and tol>0).
//   MinValue edge — the iAbs() off-by-one fix: argMaxAbs maps MinValue -> MaxValue, countNonzero
//       counts it, findValue matches it.
//   Arena wrappers — each allocating wrapper matches the zero-alloc primitive (incl. NEW
//       iProxyKFarthestRows / iProxyKFarthestColumns).
//
// Burst-compatible computational tests live in TestJob (message-free asserts + Fail-buffer
// diagnostics); managed-throw guards are plain [Test] methods on the main thread.
public class iProxyQueryTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            ArgMaxMinAbsVector,
            ArgMaxMinAbsMatrix,
            RowColArgMinMax,
            ColArgStrided,
            ArgMaxRowNorm,
            ArgMaxColNorm,
            DistancesToRowAllMetrics,
            DistancesToColumnAllMetrics,
            NearestFarthestDistance,
            NearestFarthestSimilarity,
            KNearestBruteForce,
            KNearestClampAndZero,
            KFarthest,
            WithinRadiusBoundary,
            FindValue,
            NonzeroCountNonzero,
            MinValueEdge,
            ArenaWrappers,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected, [3] diff
        public NativeArray<iProxy> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ArgMaxMinAbsVector:          ArgMaxMinAbsVector();          break;
                case TestType.ArgMaxMinAbsMatrix:          ArgMaxMinAbsMatrix();          break;
                case TestType.RowColArgMinMax:             RowColArgMinMax();             break;
                case TestType.ColArgStrided:               ColArgStrided();               break;
                case TestType.ArgMaxRowNorm:               ArgMaxRowNorm();               break;
                case TestType.ArgMaxColNorm:               ArgMaxColNorm();               break;
                case TestType.DistancesToRowAllMetrics:    DistancesToRowAllMetrics();    break;
                case TestType.DistancesToColumnAllMetrics: DistancesToColumnAllMetrics(); break;
                case TestType.NearestFarthestDistance:     NearestFarthestDistance();     break;
                case TestType.NearestFarthestSimilarity:   NearestFarthestSimilarity();   break;
                case TestType.KNearestBruteForce:          KNearestBruteForce();          break;
                case TestType.KNearestClampAndZero:        KNearestClampAndZero();        break;
                case TestType.KFarthest:                   KFarthest();                   break;
                case TestType.WithinRadiusBoundary:        WithinRadiusBoundary();        break;
                case TestType.FindValue:                   FindValue();                   break;
                case TestType.NonzeroCountNonzero:         NonzeroCountNonzero();         break;
                case TestType.MinValueEdge:                MinValueEdge();                break;
                case TestType.ArenaWrappers:               ArenaWrappers();               break;
            }
        }

        // ---------------------------------------------------------------------
        // GROUP 1 — EXTREMES
        // ---------------------------------------------------------------------

        // argMaxAbs/argMinAbs over a vector: value + flat index; ties -> first occurrence.
        void ArgMaxMinAbsVector()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.iProxyVec(6);
            // |.| = 2, 5, 5, 1, 1, 4  => maxAbs first at index 1 (tie with 2), minAbs first at index 3.
            v[0] = (iProxy)(-2); v[1] = (iProxy)5; v[2] = (iProxy)(-5);
            v[3] = (iProxy)1;    v[4] = (iProxy)(-1); v[5] = (iProxy)4;

            Query.argMaxAbs(in v, out iProxy maxVal, out int maxIdx);
            AssertEqI(maxIdx, 1);                  // first of the two |5| entries
            AssertEqV(maxVal, (iProxy)5);

            Query.argMinAbs(in v, out iProxy minVal, out int minIdx);
            AssertEqI(minIdx, 3);                  // first of the two |1| entries
            AssertEqV(minVal, (iProxy)1);

            // 1x1 / single-element vector: index 0, value = |element|.
            var one = arena.iProxyVec(1);
            one[0] = (iProxy)(-7);
            Query.argMaxAbs(in one, out iProxy ov, out int oi);
            AssertEqI(oi, 0);
            AssertEqV(ov, (iProxy)7);

            arena.Dispose();
        }

        // argMaxAbs/argMinAbs over a matrix: index is row-major flat. (decodeIndex is float-only,
        // so the (r,c) split is checked by hand: flat 4 -> (1,1); flat 3 -> (1,0) for N_Cols=3.)
        void ArgMaxMinAbsMatrix()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.iProxyMat(2, 3);
            // row 0: 1 -3  2
            // row 1: 0  4 -4      flat: [1,-3,2,0,4,-4]; maxAbs first |4| at flat 4 (r1,c1).
            A[0, 0] = (iProxy)1;  A[0, 1] = (iProxy)(-3); A[0, 2] = (iProxy)2;
            A[1, 0] = (iProxy)0;  A[1, 1] = (iProxy)4;    A[1, 2] = (iProxy)(-4);

            Query.argMaxAbs(in A, out iProxy maxVal, out int maxIdx);
            AssertEqI(maxIdx, 4);                  // (r,c) = (4/3, 4%3) = (1,1)
            AssertEqV(maxVal, (iProxy)4);

            // minAbs is the 0 at flat 3 (r1,c0).
            Query.argMinAbs(in A, out iProxy minVal, out int minIdx);
            AssertEqI(minIdx, 3);                  // (r,c) = (3/3, 3%3) = (1,0)
            AssertEqV(minVal, (iProxy)0);

            arena.Dispose();
        }

        // rowArgMin/Max + colArgMin/Max value+index AND index-only forms vs a hand-built oracle.
        void RowColArgMinMax()
        {
            var arena = new Arena(Allocator.Persistent);

            // 3x3:
            //  3  1  2     rowMin@c1=1   rowMax@c0=3
            //  9  7  8     rowMin@c1=7   rowMax@c0=9
            //  0  5  4     rowMin@c0=0   rowMax@c1=5
            var A = arena.iProxyMat(3, 3);
            A[0, 0] = (iProxy)3; A[0, 1] = (iProxy)1; A[0, 2] = (iProxy)2;
            A[1, 0] = (iProxy)9; A[1, 1] = (iProxy)7; A[1, 2] = (iProxy)8;
            A[2, 0] = (iProxy)0; A[2, 1] = (iProxy)5; A[2, 2] = (iProxy)4;

            var idxR = arena.Indices(3);
            var valR = arena.iProxyVec(3);

            int nr = Query.rowArgMin(in A, ref idxR, ref valR);
            AssertEqI(nr, 3);
            AssertEqI(idxR[0], 1); AssertEqV(valR[0], (iProxy)1);
            AssertEqI(idxR[1], 1); AssertEqV(valR[1], (iProxy)7);
            AssertEqI(idxR[2], 0); AssertEqV(valR[2], (iProxy)0);

            // index-only form must match.
            var idxR2 = arena.Indices(3);
            Query.rowArgMin(in A, ref idxR2);
            AssertEqI(idxR2[0], 1); AssertEqI(idxR2[1], 1); AssertEqI(idxR2[2], 0);

            Query.rowArgMax(in A, ref idxR, ref valR);
            AssertEqI(idxR[0], 0); AssertEqV(valR[0], (iProxy)3);
            AssertEqI(idxR[1], 0); AssertEqV(valR[1], (iProxy)9);
            AssertEqI(idxR[2], 1); AssertEqV(valR[2], (iProxy)5);

            var idxR3 = arena.Indices(3);
            Query.rowArgMax(in A, ref idxR3);
            AssertEqI(idxR3[0], 0); AssertEqI(idxR3[1], 0); AssertEqI(idxR3[2], 1);

            // columns: colMin per column -> rows {2,0,0}; colMax per column -> rows {1,1,1}.
            var idxC = arena.Indices(3);
            var valC = arena.iProxyVec(3);
            int nc = Query.colArgMin(in A, ref idxC, ref valC);
            AssertEqI(nc, 3);
            AssertEqI(idxC[0], 2); AssertEqV(valC[0], (iProxy)0);
            AssertEqI(idxC[1], 0); AssertEqV(valC[1], (iProxy)1);
            AssertEqI(idxC[2], 0); AssertEqV(valC[2], (iProxy)2);

            var idxC2 = arena.Indices(3);
            Query.colArgMin(in A, ref idxC2);
            AssertEqI(idxC2[0], 2); AssertEqI(idxC2[1], 0); AssertEqI(idxC2[2], 0);

            Query.colArgMax(in A, ref idxC, ref valC);
            AssertEqI(idxC[0], 1); AssertEqV(valC[0], (iProxy)9);
            AssertEqI(idxC[1], 1); AssertEqV(valC[1], (iProxy)7);
            AssertEqI(idxC[2], 1); AssertEqV(valC[2], (iProxy)8);

            var idxC3 = arena.Indices(3);
            Query.colArgMax(in A, ref idxC3);
            AssertEqI(idxC3[0], 1); AssertEqI(idxC3[1], 1); AssertEqI(idxC3[2], 1);

            arena.Dispose();
        }

        // Strided-column correctness: a non-square matrix where a row-major misread would give a
        // different answer than the true strided column scan.
        void ColArgStrided()
        {
            var arena = new Arena(Allocator.Persistent);

            // 4x2. Column 0 (strided indices 0,2,4,6), column 1 (1,3,5,7).
            //  r0:  1   8
            //  r1:  2   6
            //  r2:  9   7
            //  r3:  3   4
            // col0 min@r0(1) max@r2(9); col1 min@r3(4) max@r0(8).
            var A = arena.iProxyMat(4, 2);
            A[0, 0] = (iProxy)1; A[0, 1] = (iProxy)8;
            A[1, 0] = (iProxy)2; A[1, 1] = (iProxy)6;
            A[2, 0] = (iProxy)9; A[2, 1] = (iProxy)7;
            A[3, 0] = (iProxy)3; A[3, 1] = (iProxy)4;

            var idxC = arena.Indices(2);
            var valC = arena.iProxyVec(2);

            Query.colArgMin(in A, ref idxC, ref valC);
            AssertEqI(idxC[0], 0); AssertEqV(valC[0], (iProxy)1);
            AssertEqI(idxC[1], 3); AssertEqV(valC[1], (iProxy)4);

            Query.colArgMax(in A, ref idxC, ref valC);
            AssertEqI(idxC[0], 2); AssertEqV(valC[0], (iProxy)9);
            AssertEqI(idxC[1], 0); AssertEqV(valC[1], (iProxy)8);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // GROUP 2 — NORM-SELECTION (L1 and Linf; L2 throws — tested on main thread)
        // ---------------------------------------------------------------------

        void ArgMaxRowNorm()
        {
            var arena = new Arena(Allocator.Persistent);

            // 3x3 with hand-computed row norms:
            //  r0:  3  0  0  -> L1=3,  Linf=3
            //  r1:  1  1  1  -> L1=3,  Linf=1
            //  r2: -2  2  0  -> L1=4,  Linf=2
            // L1 max -> r2 ; Linf max -> r0.
            var A = arena.iProxyMat(3, 3);
            A[0, 0] = (iProxy)3;    A[0, 1] = (iProxy)0; A[0, 2] = (iProxy)0;
            A[1, 0] = (iProxy)1;    A[1, 1] = (iProxy)1; A[1, 2] = (iProxy)1;
            A[2, 0] = (iProxy)(-2); A[2, 1] = (iProxy)2; A[2, 2] = (iProxy)0;

            AssertEqI(Query.argMaxRowNorm(in A, Norm.L1), 2);
            AssertEqI(Query.argMaxRowNorm(in A, Norm.Linf), 0);

            // Tie -> first occurrence. Two rows of identical L1 norm 5; first is row 0.
            var T = arena.iProxyMat(2, 2);
            T[0, 0] = (iProxy)5; T[0, 1] = (iProxy)0;
            T[1, 0] = (iProxy)0; T[1, 1] = (iProxy)5;
            AssertEqI(Query.argMaxRowNorm(in T, Norm.L1), 0);

            arena.Dispose();
        }

        void ArgMaxColNorm()
        {
            var arena = new Arena(Allocator.Persistent);

            //  cols (each length 3):
            //   c0:  3, 1, -2   -> L1=6, Linf=3
            //   c1:  0, 1,  2   -> L1=3, Linf=2
            //   c2:  4, 0,  0   -> L1=4, Linf=4
            // L1 max -> c0 ; Linf max -> c2.
            var A = arena.iProxyMat(3, 3);
            A[0, 0] = (iProxy)3;    A[0, 1] = (iProxy)0; A[0, 2] = (iProxy)4;
            A[1, 0] = (iProxy)1;    A[1, 1] = (iProxy)1; A[1, 2] = (iProxy)0;
            A[2, 0] = (iProxy)(-2); A[2, 1] = (iProxy)2; A[2, 2] = (iProxy)0;

            AssertEqI(Query.argMaxColNorm(in A, Norm.L1), 0);
            AssertEqI(Query.argMaxColNorm(in A, Norm.Linf), 2);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // GROUP 3 — SEARCH
        // ---------------------------------------------------------------------

        // distancesToRow for each integer metric against hand-computed distances.
        void DistancesToRowAllMetrics()
        {
            var arena = new Arena(Allocator.Persistent);

            // 2 rows, 2 cols; query q = (0,0).
            //  r0 = (3, 4)
            //  r1 = (1, 0)
            var A = arena.iProxyMat(2, 2);
            A[0, 0] = (iProxy)3; A[0, 1] = (iProxy)4;
            A[1, 0] = (iProxy)1; A[1, 1] = (iProxy)0;
            var q = arena.iProxyVec(2);
            q[0] = (iProxy)0; q[1] = (iProxy)0;

            var d = arena.iProxyVec(2);

            // Manhattan: |3|+|4|=7 ; |1|+|0|=1
            Query.distancesToRow(in A, in q, Metric.Manhattan, ref d);
            AssertEqV(d[0], (iProxy)7); AssertEqV(d[1], (iProxy)1);

            // SqEuclidean (unsquared, no sqrt): 25 ; 1
            Query.distancesToRow(in A, in q, Metric.SqEuclidean, ref d);
            AssertEqV(d[0], (iProxy)25); AssertEqV(d[1], (iProxy)1);

            // Chebyshev: max(3,4)=4 ; max(1,0)=1
            Query.distancesToRow(in A, in q, Metric.Chebyshev, ref d);
            AssertEqV(d[0], (iProxy)4); AssertEqV(d[1], (iProxy)1);

            // Dot with q=(0,0) is 0 for every row.
            Query.distancesToRow(in A, in q, Metric.Dot, ref d);
            AssertEqV(d[0], (iProxy)0); AssertEqV(d[1], (iProxy)0);

            // Dot with q2=(3,4): r0.q2 = 9+16 = 25 ; r1.q2 = 3+0 = 3.
            var q2 = arena.iProxyVec(2);
            q2[0] = (iProxy)3; q2[1] = (iProxy)4;
            Query.distancesToRow(in A, in q2, Metric.Dot, ref d);
            AssertEqV(d[0], (iProxy)25); AssertEqV(d[1], (iProxy)3);

            arena.Dispose();
        }

        // distancesToColumn for each integer metric. Columns are strided.
        void DistancesToColumnAllMetrics()
        {
            var arena = new Arena(Allocator.Persistent);

            // 2 rows, 2 cols; query q (length M_Rows = 2) = (0,0).
            //  c0 = (3,1)   c1 = (4,0)
            var A = arena.iProxyMat(2, 2);
            A[0, 0] = (iProxy)3; A[0, 1] = (iProxy)4;
            A[1, 0] = (iProxy)1; A[1, 1] = (iProxy)0;
            var q = arena.iProxyVec(2);
            q[0] = (iProxy)0; q[1] = (iProxy)0;

            var d = arena.iProxyVec(2);

            // Manhattan: c0=4, c1=4
            Query.distancesToColumn(in A, in q, Metric.Manhattan, ref d);
            AssertEqV(d[0], (iProxy)4); AssertEqV(d[1], (iProxy)4);

            // SqEuclidean: c0 = 9+1 = 10 ; c1 = 16+0 = 16
            Query.distancesToColumn(in A, in q, Metric.SqEuclidean, ref d);
            AssertEqV(d[0], (iProxy)10); AssertEqV(d[1], (iProxy)16);

            // Chebyshev: c0=max(3,1)=3 ; c1=max(4,0)=4
            Query.distancesToColumn(in A, in q, Metric.Chebyshev, ref d);
            AssertEqV(d[0], (iProxy)3); AssertEqV(d[1], (iProxy)4);

            // Dot with q=(0,0) is 0 for every column.
            Query.distancesToColumn(in A, in q, Metric.Dot, ref d);
            AssertEqV(d[0], (iProxy)0); AssertEqV(d[1], (iProxy)0);

            arena.Dispose();
        }

        // nearest=min / farthest=max for DISTANCE metrics, with score in metric units.
        void NearestFarthestDistance()
        {
            var arena = new Arena(Allocator.Persistent);

            //  r0=(0,0) r1=(3,4) r2=(1,1); q=(0,0).
            var A = arena.iProxyMat(3, 2);
            A[0, 0] = (iProxy)0; A[0, 1] = (iProxy)0;
            A[1, 0] = (iProxy)3; A[1, 1] = (iProxy)4;
            A[2, 0] = (iProxy)1; A[2, 1] = (iProxy)1;
            var q = arena.iProxyVec(2);
            q[0] = (iProxy)0; q[1] = (iProxy)0;

            // SqEuclidean: distances 0, 25, 2. nearest=r0 (score 0), farthest=r1 (score 25, squared).
            Query.nearestRow(in A, in q, Metric.SqEuclidean, out int ni, out iProxy ns);
            AssertEqI(ni, 0); AssertEqV(ns, (iProxy)0);

            Query.farthestRow(in A, in q, Metric.SqEuclidean, out int fi, out iProxy fs);
            AssertEqI(fi, 1); AssertEqV(fs, (iProxy)25);

            // Manhattan: distances 0, 7, 2. nearest=r0, farthest=r1 (score 7).
            Query.nearestRow(in A, in q, Metric.Manhattan, out int ni2, out iProxy ns2);
            AssertEqI(ni2, 0); AssertEqV(ns2, (iProxy)0);
            Query.farthestRow(in A, in q, Metric.Manhattan, out int fi2, out iProxy fs2);
            AssertEqI(fi2, 1); AssertEqV(fs2, (iProxy)7);

            arena.Dispose();
        }

        // The KEY direction flip: similarity metric (Dot) -> nearest = MAX, farthest = MIN.
        void NearestFarthestSimilarity()
        {
            var arena = new Arena(Allocator.Persistent);

            //  r0=(1,0) r1=(10,0) r2=(-5,0); q=(1,0).
            //  Dot: 1, 10, -5  -> nearest(max)=r1, farthest(min)=r2.
            var A = arena.iProxyMat(3, 2);
            A[0, 0] = (iProxy)1;    A[0, 1] = (iProxy)0;
            A[1, 0] = (iProxy)10;   A[1, 1] = (iProxy)0;
            A[2, 0] = (iProxy)(-5); A[2, 1] = (iProxy)0;
            var q = arena.iProxyVec(2);
            q[0] = (iProxy)1; q[1] = (iProxy)0;

            Query.nearestRow(in A, in q, Metric.Dot, out int ni, out iProxy ns);
            AssertEqI(ni, 1); AssertEqV(ns, (iProxy)10);

            Query.farthestRow(in A, in q, Metric.Dot, out int fi, out iProxy fs);
            AssertEqI(fi, 2); AssertEqV(fs, (iProxy)(-5));

            // Column twins: columns of A as vectors of length 3: c0=(1,10,-5), c1=(0,0,0). q3=(1,0,0).
            // Dot: c0.q3 = 1 ; c1.q3 = 0 -> nearest(max)=c0, farthest(min)=c1.
            var q3 = arena.iProxyVec(3);
            q3[0] = (iProxy)1; q3[1] = (iProxy)0; q3[2] = (iProxy)0;
            Query.nearestColumn(in A, in q3, Metric.Dot, out int cni, out iProxy cns);
            AssertEqI(cni, 0); AssertEqV(cns, (iProxy)1);
            Query.farthestColumn(in A, in q3, Metric.Dot, out int cfi, out iProxy cfs);
            AssertEqI(cfi, 1); AssertEqV(cfs, (iProxy)0);

            arena.Dispose();
        }

        // kNearestRows + kNearestColumns vs brute-force reference, sorted best-first.
        void KNearestBruteForce()
        {
            var arena = new Arena(Allocator.Persistent);

            // Small range keeps integer SqEuclidean/Dot inside short.MaxValue.
            int M = 6, N = 3;
            var A = arena.iProxyRandomMat(M, N, (iProxy)(-3), (iProxy)4, 424242);
            var q = arena.iProxyVec(N);
            q[0] = (iProxy)1; q[1] = (iProxy)(-2); q[2] = (iProxy)2;

            int k = 3;
            var idx = arena.Indices(k);
            var scores = arena.iProxyVec(k);

            // --- distance metric (SqEuclidean): best-first = ASCENDING ---
            int cnt = Query.kNearestRows(in A, in q, k, Metric.SqEuclidean, ref idx, ref scores);
            AssertEqI(cnt, k);

            var all = arena.iProxyVec(M);
            Query.distancesToRow(in A, in q, Metric.SqEuclidean, ref all);
            for (int i = 0; i + 1 < cnt; i++)
                AssertTrue(scores[i] <= scores[i + 1]);
            for (int i = 0; i < cnt; i++)
                AssertEqV(scores[i], all[idx[i]]);
            iProxy kth = scores[cnt - 1];
            for (int r = 0; r < M; r++)
            {
                bool selected = false;
                for (int i = 0; i < cnt; i++) if (idx[i] == r) selected = true;
                if (!selected) AssertTrue(all[r] >= kth);
            }

            // --- similarity metric (Dot): best-first = DESCENDING ---
            var idx2 = arena.Indices(k);
            var scores2 = arena.iProxyVec(k);
            int cnt2 = Query.kNearestRows(in A, in q, k, Metric.Dot, ref idx2, ref scores2);
            AssertEqI(cnt2, k);
            var allDot = arena.iProxyVec(M);
            Query.distancesToRow(in A, in q, Metric.Dot, ref allDot);
            for (int i = 0; i + 1 < cnt2; i++)
                AssertTrue(scores2[i] >= scores2[i + 1]);
            for (int i = 0; i < cnt2; i++)
                AssertEqV(scores2[i], allDot[idx2[i]]);
            iProxy kthDot = scores2[cnt2 - 1];
            for (int r = 0; r < M; r++)
            {
                bool selected = false;
                for (int i = 0; i < cnt2; i++) if (idx2[i] == r) selected = true;
                if (!selected) AssertTrue(allDot[r] <= kthDot);
            }

            // --- kNearestColumns (SqEuclidean): q length = M_Rows ---
            var qc = arena.iProxyVec(M);
            for (int i = 0; i < M; i++) qc[i] = (iProxy)(i - 3);
            int kc = 2;
            var idxC = arena.Indices(kc);
            var scoresC = arena.iProxyVec(kc);
            int cntC = Query.kNearestColumns(in A, in qc, kc, Metric.SqEuclidean, ref idxC, ref scoresC);
            AssertEqI(cntC, kc);
            var allC = arena.iProxyVec(N);
            Query.distancesToColumn(in A, in qc, Metric.SqEuclidean, ref allC);
            for (int i = 0; i + 1 < cntC; i++)
                AssertTrue(scoresC[i] <= scoresC[i + 1]);
            for (int i = 0; i < cntC; i++)
                AssertEqV(scoresC[i], allC[idxC[i]]);
            iProxy kthC = scoresC[cntC - 1];
            for (int c = 0; c < N; c++)
            {
                bool selected = false;
                for (int i = 0; i < cntC; i++) if (idxC[i] == c) selected = true;
                if (!selected) AssertTrue(allC[c] >= kthC);
            }

            arena.Dispose();
        }

        // k > M_Rows clamps to count; k=0 returns 0; pure-tie keeps first-seen order.
        void KNearestClampAndZero()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 3, N = 2;
            var A = arena.iProxyMat(M, N);
            // all rows at SqEuclidean distance 1 from q -> a full tie.
            A[0, 0] = (iProxy)1;    A[0, 1] = (iProxy)0;
            A[1, 0] = (iProxy)0;    A[1, 1] = (iProxy)1;
            A[2, 0] = (iProxy)(-1); A[2, 1] = (iProxy)0;
            var q = arena.iProxyVec(N);
            q[0] = (iProxy)0; q[1] = (iProxy)0;

            // k = 10 > M -> clamp to 3.
            int k = 10;
            var idx = arena.Indices(k);
            var scores = arena.iProxyVec(k);
            int cnt = Query.kNearestRows(in A, in q, k, Metric.SqEuclidean, ref idx, ref scores);
            AssertEqI(cnt, 3);
            for (int i = 0; i < cnt; i++) AssertEqV(scores[i], (iProxy)1);
            // pure tie -> insertion keeps first-seen order.
            AssertEqI(idx[0], 0); AssertEqI(idx[1], 1); AssertEqI(idx[2], 2);

            // k = 0 -> returns 0, writes nothing.
            int z = Query.kNearestRows(in A, in q, 0, Metric.SqEuclidean, ref idx, ref scores);
            AssertEqI(z, 0);

            arena.Dispose();
        }

        // kFarthestRows + kFarthestColumns: farthest-first ordering vs brute force.
        void KFarthest()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 5, N = 2;
            var A = arena.iProxyRandomMat(M, N, (iProxy)(-2), (iProxy)3, 99887);
            var q = arena.iProxyVec(N);
            q[0] = (iProxy)1; q[1] = (iProxy)(-1);

            int k = 2;
            var idx = arena.Indices(k);
            var scores = arena.iProxyVec(k);
            int cnt = Query.kFarthestRows(in A, in q, k, Metric.SqEuclidean, ref idx, ref scores);
            AssertEqI(cnt, k);

            var all = arena.iProxyVec(M);
            Query.distancesToRow(in A, in q, Metric.SqEuclidean, ref all);
            // farthest-first => descending distance
            for (int i = 0; i + 1 < cnt; i++)
                AssertTrue(scores[i] >= scores[i + 1]);
            for (int i = 0; i < cnt; i++)
                AssertEqV(scores[i], all[idx[i]]);
            iProxy kth = scores[cnt - 1];
            for (int r = 0; r < M; r++)
            {
                bool selected = false;
                for (int i = 0; i < cnt; i++) if (idx[i] == r) selected = true;
                if (!selected) AssertTrue(all[r] <= kth);
            }

            // kFarthestColumns twin (SqEuclidean), q length = M_Rows.
            var qc = arena.iProxyVec(M);
            for (int i = 0; i < M; i++) qc[i] = (iProxy)(i - 2);
            int kc = 1;
            var idxC = arena.Indices(kc);
            var scoresC = arena.iProxyVec(kc);
            int cntC = Query.kFarthestColumns(in A, in qc, kc, Metric.SqEuclidean, ref idxC, ref scoresC);
            AssertEqI(cntC, kc);
            var allC = arena.iProxyVec(N);
            Query.distancesToColumn(in A, in qc, Metric.SqEuclidean, ref allC);
            // the single farthest column must equal the max distance.
            iProxy maxC = allC[0];
            for (int c = 1; c < N; c++) if (allC[c] > maxC) maxC = allC[c];
            AssertEqV(scoresC[0], maxC);
            AssertEqV(allC[idxC[0]], maxC);

            arena.Dispose();
        }

        // within-radius: inclusive boundary (<= r distance, >= r similarity); buffer fill matches count.
        void WithinRadiusBoundary()
        {
            var arena = new Arena(Allocator.Persistent);

            //  r0=(0,0) r1=(3,4) r2=(1,1) ; q=(0,0).  SqEuclidean distances: 0, 25, 2.
            var A = arena.iProxyMat(3, 2);
            A[0, 0] = (iProxy)0; A[0, 1] = (iProxy)0;
            A[1, 0] = (iProxy)3; A[1, 1] = (iProxy)4;
            A[2, 0] = (iProxy)1; A[2, 1] = (iProxy)1;
            var q = arena.iProxyVec(2);
            q[0] = (iProxy)0; q[1] = (iProxy)0;

            // radius exactly 25 (boundary): inclusive -> all three rows qualify.
            var idx = arena.Indices(3);
            int cnt = Query.rowsWithinRadius(in A, in q, (iProxy)25, Metric.SqEuclidean, ref idx);
            int ccnt = Query.countWithinRadius(in A, in q, (iProxy)25, Metric.SqEuclidean);
            AssertEqI(cnt, 3); AssertEqI(ccnt, 3);
            AssertEqI(idx[0], 0); AssertEqI(idx[1], 1); AssertEqI(idx[2], 2);

            // radius 24 -> r1 (distance 25) excluded.
            int cnt2 = Query.rowsWithinRadius(in A, in q, (iProxy)24, Metric.SqEuclidean, ref idx);
            AssertEqI(cnt2, 2);
            AssertEqI(idx[0], 0); AssertEqI(idx[1], 2);
            AssertEqI(Query.countWithinRadius(in A, in q, (iProxy)24, Metric.SqEuclidean), 2);

            // similarity metric (Dot): inclusive >= r. q2=(1,0): dots 0, 3, 1.
            var q2 = arena.iProxyVec(2);
            q2[0] = (iProxy)1; q2[1] = (iProxy)0;
            // threshold exactly 1 -> rows with dot >= 1: r1(3) and r2(1). r0(0) excluded.
            int cnt3 = Query.rowsWithinRadius(in A, in q2, (iProxy)1, Metric.Dot, ref idx);
            AssertEqI(cnt3, 2);
            AssertEqI(idx[0], 1); AssertEqI(idx[1], 2);
            AssertEqI(Query.countWithinRadius(in A, in q2, (iProxy)1, Metric.Dot), 2);

            // Column twins. Columns of A length 3: c0=(0,3,1) c1=(0,4,1). qcol=(0,0,0).
            var qcol = arena.iProxyVec(3);
            qcol[0] = (iProxy)0; qcol[1] = (iProxy)0; qcol[2] = (iProxy)0;
            // SqEuclidean: c0=0+9+1=10, c1=0+16+1=17. radius 10 inclusive -> only c0.
            var idxc = arena.Indices(2);
            int ccol = Query.columnsWithinRadius(in A, in qcol, (iProxy)10, Metric.SqEuclidean, ref idxc);
            AssertEqI(ccol, 1); AssertEqI(idxc[0], 0);
            AssertEqI(Query.countWithinColumnRadius(in A, in qcol, (iProxy)10, Metric.SqEuclidean), 1);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // GROUP 4 — VALUE / MASK
        // ---------------------------------------------------------------------

        void FindValue()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.iProxyVec(5);
            v[0] = (iProxy)1; v[1] = (iProxy)2; v[2] = (iProxy)2; v[3] = (iProxy)3; v[4] = (iProxy)2;

            // first match (tol 0) at index 1
            AssertEqI(Query.findValue(in v, (iProxy)2, (iProxy)0), 1);
            // absent -> -1
            AssertEqI(Query.findValue(in v, (iProxy)9, (iProxy)0), -1);
            // integer tol: target 4, tol 1 -> first element with |x-4| <= 1 is the 3 at index 3.
            AssertEqI(Query.findValue(in v, (iProxy)4, (iProxy)1), 3);
            // tol 0 of an absent target -> no match.
            AssertEqI(Query.findValue(in v, (iProxy)4, (iProxy)0), -1);

            // matrix overload (flat index). 2x2 = [5, 6; 7, 6]; first 6 at flat 1.
            var A = arena.iProxyMat(2, 2);
            A[0, 0] = (iProxy)5; A[0, 1] = (iProxy)6;
            A[1, 0] = (iProxy)7; A[1, 1] = (iProxy)6;
            AssertEqI(Query.findValue(in A, (iProxy)6, (iProxy)0), 1);

            arena.Dispose();
        }

        void NonzeroCountNonzero()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.iProxyVec(6);
            v[0] = (iProxy)0; v[1] = (iProxy)2;    v[2] = (iProxy)0;
            v[3] = (iProxy)(-3); v[4] = (iProxy)1; v[5] = (iProxy)0;

            // tol=0: nonzero are indices 1,3,4 -> count 3
            AssertEqI(Query.countNonzero(in v, (iProxy)0), 3);
            var idx = arena.Indices(6);
            int c = Query.nonzero(in v, (iProxy)0, ref idx);
            AssertEqI(c, 3);
            AssertEqI(idx[0], 1); AssertEqI(idx[1], 3); AssertEqI(idx[2], 4);

            // tol=1 (strict |x|>tol): |1| filtered out -> indices 1,3 -> count 2
            AssertEqI(Query.countNonzero(in v, (iProxy)1), 2);
            int c2 = Query.nonzero(in v, (iProxy)1, ref idx);
            AssertEqI(c2, 2);
            AssertEqI(idx[0], 1); AssertEqI(idx[1], 3);

            // matrix overload, flat indices. 2x2 = [0,2;0,0] -> one nonzero at flat 1.
            var A = arena.iProxyMat(2, 2);
            A[0, 0] = (iProxy)0; A[0, 1] = (iProxy)2;
            A[1, 0] = (iProxy)0; A[1, 1] = (iProxy)0;
            AssertEqI(Query.countNonzero(in A, (iProxy)0), 1);
            var idxA = arena.Indices(4);
            int ca = Query.nonzero(in A, (iProxy)0, ref idxA);
            AssertEqI(ca, 1); AssertEqI(idxA[0], 1);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // MinValue EDGE — iAbs() saturates |iProxy.MinValue| to iProxy.MaxValue.
        // ---------------------------------------------------------------------

        void MinValueEdge()
        {
            var arena = new Arena(Allocator.Persistent);

            // MinValue at flat 0 so findValue returns before any other (potentially overflowing)
            // subtraction is evaluated. Other elements are small and non-extreme.
            var v = arena.iProxyVec(4);
            v[0] = (iProxy)iProxy.MinValue; v[1] = (iProxy)3; v[2] = (iProxy)0; v[3] = (iProxy)(-2);

            // argMaxAbs: |MinValue| saturates to MaxValue (the documented off-by-one) and wins.
            Query.argMaxAbs(in v, out iProxy mv, out int mi);
            AssertEqI(mi, 0);
            AssertEqV(mv, (iProxy)iProxy.MaxValue);

            // countNonzero(tol=0): MinValue classifies as nonzero -> {0,1,3} -> count 3.
            AssertEqI(Query.countNonzero(in v, (iProxy)0), 3);

            // findValue(target = MinValue, tol = 0): exact match at flat 0.
            AssertEqI(Query.findValue(in v, (iProxy)iProxy.MinValue, (iProxy)0), 0);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // ARENA WRAPPERS — each allocating wrapper matches the zero-alloc primitive.
        // ---------------------------------------------------------------------

        void ArenaWrappers()
        {
            var arena = new Arena(Allocator.Persistent);

            // Small range keeps integer SqEuclidean inside short.MaxValue.
            int M = 6, N = 4;
            var A = arena.iProxyRandomMat(M, N, (iProxy)(-2), (iProxy)3, 7777);
            var q = arena.iProxyVec(N);
            for (int i = 0; i < N; i++) q[i] = (iProxy)(i - 2);

            // --- iProxyDistancesToRow / Column wrappers vs primitive ---
            var dr = ArenaExtensions.iProxyDistancesToRow(in A, in q, Metric.SqEuclidean);
            var drRef = arena.iProxyVec(M);
            Query.distancesToRow(in A, in q, Metric.SqEuclidean, ref drRef);
            AssertEqI(dr.N, M);
            for (int i = 0; i < M; i++) AssertEqV(dr[i], drRef[i]);

            var qc = arena.iProxyVec(M);
            for (int i = 0; i < M; i++) qc[i] = (iProxy)(i - 3);
            var dcol = ArenaExtensions.iProxyDistancesToColumn(in A, in qc, Metric.SqEuclidean);
            var dcolRef = arena.iProxyVec(N);
            Query.distancesToColumn(in A, in qc, Metric.SqEuclidean, ref dcolRef);
            AssertEqI(dcol.N, N);
            for (int j = 0; j < N; j++) AssertEqV(dcol[j], dcolRef[j]);

            // --- iProxyNonzeroIndices: exact-sized, contents match the primitive ---
            var idxNz = arena.iProxyNonzeroIndices(in A, (iProxy)0);
            var refNz = arena.Indices(M * N);
            int refCnt = Query.nonzero(in A, (iProxy)0, ref refNz);
            AssertEqI(idxNz.N, refCnt);
            for (int i = 0; i < refCnt; i++) AssertEqI(idxNz[i], refNz[i]);

            // --- iProxyRowsWithinRadius: exact-sized, contents match the primitive ---
            iProxy radius = (iProxy)20;
            var idxRR = arena.iProxyRowsWithinRadius(in A, in q, radius, Metric.SqEuclidean);
            var refRR = arena.Indices(M);
            int refRRcnt = Query.rowsWithinRadius(in A, in q, radius, Metric.SqEuclidean, ref refRR);
            AssertEqI(idxRR.N, refRRcnt);
            for (int i = 0; i < refRRcnt; i++) AssertEqI(idxRR[i], refRR[i]);

            // --- iProxyColumnsWithinRadius ---
            var idxCR = arena.iProxyColumnsWithinRadius(in A, in qc, (iProxy)40, Metric.SqEuclidean);
            var refCR = arena.Indices(N);
            int refCRcnt = Query.columnsWithinRadius(in A, in qc, (iProxy)40, Metric.SqEuclidean, ref refCR);
            AssertEqI(idxCR.N, refCRcnt);
            for (int i = 0; i < refCRcnt; i++) AssertEqI(idxCR[i], refCR[i]);

            // --- iProxyKNearestRows: idx + scores match the primitive ---
            int k = 3;
            var idxK = arena.iProxyKNearestRows(in A, in q, k, Metric.SqEuclidean, out iProxyN scoresK, out int cntK);
            var refIdxK = arena.Indices(k);
            var refScoresK = arena.iProxyVec(k);
            int refCntK = Query.kNearestRows(in A, in q, k, Metric.SqEuclidean, ref refIdxK, ref refScoresK);
            AssertEqI(cntK, refCntK);
            AssertEqI(idxK.N, refCntK);
            for (int i = 0; i < refCntK; i++)
            {
                AssertEqI(idxK[i], refIdxK[i]);
                AssertEqV(scoresK[i], refScoresK[i]);
            }

            // --- iProxyKNearestColumns ---
            var idxKC = arena.iProxyKNearestColumns(in A, in qc, k, Metric.SqEuclidean, out iProxyN scoresKC, out int cntKC);
            var refIdxKC = arena.Indices(k);
            var refScoresKC = arena.iProxyVec(k);
            int refCntKC = Query.kNearestColumns(in A, in qc, k, Metric.SqEuclidean, ref refIdxKC, ref refScoresKC);
            AssertEqI(cntKC, refCntKC);
            for (int i = 0; i < refCntKC; i++)
            {
                AssertEqI(idxKC[i], refIdxKC[i]);
                AssertEqV(scoresKC[i], refScoresKC[i]);
            }

            // --- NEW iProxyKFarthestRows ---
            var idxKF = arena.iProxyKFarthestRows(in A, in q, k, Metric.SqEuclidean, out iProxyN scoresKF, out int cntKF);
            var refIdxKF = arena.Indices(k);
            var refScoresKF = arena.iProxyVec(k);
            int refCntKF = Query.kFarthestRows(in A, in q, k, Metric.SqEuclidean, ref refIdxKF, ref refScoresKF);
            AssertEqI(cntKF, refCntKF);
            AssertEqI(idxKF.N, refCntKF);
            for (int i = 0; i < refCntKF; i++)
            {
                AssertEqI(idxKF[i], refIdxKF[i]);
                AssertEqV(scoresKF[i], refScoresKF[i]);
            }

            // --- NEW iProxyKFarthestColumns ---
            var idxKFC = arena.iProxyKFarthestColumns(in A, in qc, k, Metric.SqEuclidean, out iProxyN scoresKFC, out int cntKFC);
            var refIdxKFC = arena.Indices(k);
            var refScoresKFC = arena.iProxyVec(k);
            int refCntKFC = Query.kFarthestColumns(in A, in qc, k, Metric.SqEuclidean, ref refIdxKFC, ref refScoresKFC);
            AssertEqI(cntKFC, refCntKFC);
            for (int i = 0; i < refCntKFC; i++)
            {
                AssertEqI(idxKFC[i], refIdxKFC[i]);
                AssertEqV(scoresKFC[i], refScoresKFC[i]);
            }

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // helpers (integer ops are exact — no tolerance)
        // ---------------------------------------------------------------------

        // Fail layout: [0]=flag, [1]=got, [2]=expected, [3]=diff
        void AssertEqV(iProxy got, iProxy expected)
        {
            if (got != expected && Fail[0] == (iProxy)0)
            {
                Fail[0] = (iProxy)1;
                Fail[1] = got;
                Fail[2] = expected;
                Fail[3] = (iProxy)(got - expected);
            }
            Assert.IsTrue(got == expected);
        }

        void AssertEqI(int got, int expected)
        {
            if (got != expected && Fail[0] == (iProxy)0)
            {
                Fail[0] = (iProxy)1;
                Fail[1] = (iProxy)got;
                Fail[2] = (iProxy)expected;
                Fail[3] = (iProxy)(got - expected);
            }
            Assert.AreEqual(expected, got);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (iProxy)0)
            {
                Fail[0] = (iProxy)1;
                Fail[1] = (iProxy)(-1);
                Fail[2] = (iProxy)(-1);
                Fail[3] = (iProxy)(-1);
            }
            Assert.IsTrue(ok);
        }
    }

    void RunJob(TestJob.TestType type)
    {
        var fail = new NativeArray<iProxy>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (iProxy)0)
                Assert.Fail($"got {fail[1]}, expected {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (iProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected {fail[2]}, diff {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    [Test] public void ArgMaxMinAbsVectorTest()          => RunJob(TestJob.TestType.ArgMaxMinAbsVector);
    [Test] public void ArgMaxMinAbsMatrixTest()          => RunJob(TestJob.TestType.ArgMaxMinAbsMatrix);
    [Test] public void RowColArgMinMaxTest()             => RunJob(TestJob.TestType.RowColArgMinMax);
    [Test] public void ColArgStridedTest()               => RunJob(TestJob.TestType.ColArgStrided);
    [Test] public void ArgMaxRowNormTest()               => RunJob(TestJob.TestType.ArgMaxRowNorm);
    [Test] public void ArgMaxColNormTest()               => RunJob(TestJob.TestType.ArgMaxColNorm);
    [Test] public void DistancesToRowAllMetricsTest()    => RunJob(TestJob.TestType.DistancesToRowAllMetrics);
    [Test] public void DistancesToColumnAllMetricsTest() => RunJob(TestJob.TestType.DistancesToColumnAllMetrics);
    [Test] public void NearestFarthestDistanceTest()     => RunJob(TestJob.TestType.NearestFarthestDistance);
    [Test] public void NearestFarthestSimilarityTest()   => RunJob(TestJob.TestType.NearestFarthestSimilarity);
    [Test] public void KNearestBruteForceTest()          => RunJob(TestJob.TestType.KNearestBruteForce);
    [Test] public void KNearestClampAndZeroTest()        => RunJob(TestJob.TestType.KNearestClampAndZero);
    [Test] public void KFarthestTest()                   => RunJob(TestJob.TestType.KFarthest);
    [Test] public void WithinRadiusBoundaryTest()        => RunJob(TestJob.TestType.WithinRadiusBoundary);
    [Test] public void FindValueTest()                   => RunJob(TestJob.TestType.FindValue);
    [Test] public void NonzeroCountNonzeroTest()         => RunJob(TestJob.TestType.NonzeroCountNonzero);
    [Test] public void MinValueEdgeTest()                => RunJob(TestJob.TestType.MinValueEdge);
    [Test] public void ArenaWrappersTest()               => RunJob(TestJob.TestType.ArenaWrappers);

    // -------------------------------------------------------------------------
    // Managed-throw guards (main thread): float-only norms/metrics rejected (spec P2/P6),
    // plus dimension-mismatch contracts.
    // -------------------------------------------------------------------------

    // Norm.L2 is float-only for integer norm-selection -> ArgumentException.
    [Test]
    public void NormL2ThrowsTest()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.iProxyMat(3, 3);
        A[0, 0] = (iProxy)1; A[0, 1] = (iProxy)2; A[0, 2] = (iProxy)3;
        A[1, 0] = (iProxy)4; A[1, 1] = (iProxy)5; A[1, 2] = (iProxy)6;
        A[2, 0] = (iProxy)7; A[2, 1] = (iProxy)8; A[2, 2] = (iProxy)9;

        Assert.Throws<ArgumentException>(() => Query.argMaxRowNorm(in A, Norm.L2));
        Assert.Throws<ArgumentException>(() => Query.argMaxColNorm(in A, Norm.L2));
        // L1 / Linf must NOT throw.
        Assert.DoesNotThrow(() => Query.argMaxRowNorm(in A, Norm.L1));
        Assert.DoesNotThrow(() => Query.argMaxColNorm(in A, Norm.Linf));

        arena.Dispose();
    }

    // Metric.Euclidean and Metric.Cosine are float-only -> ArgumentException on the integer ops.
    [Test]
    public void MetricEuclideanCosineThrowTest()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.iProxyMat(3, 2);
        A[0, 0] = (iProxy)1; A[0, 1] = (iProxy)2;
        A[1, 0] = (iProxy)3; A[1, 1] = (iProxy)4;
        A[2, 0] = (iProxy)5; A[2, 1] = (iProxy)6;
        var q = arena.iProxyVec(2);
        q[0] = (iProxy)0; q[1] = (iProxy)0;
        var dest = arena.iProxyVec(3);
        var idx = arena.Indices(3);
        var scores = arena.iProxyVec(2);

        Assert.Throws<ArgumentException>(() =>
            Query.distancesToRow(in A, in q, Metric.Euclidean, ref dest));
        Assert.Throws<ArgumentException>(() =>
            Query.distancesToRow(in A, in q, Metric.Cosine, ref dest));
        Assert.Throws<ArgumentException>(() =>
            Query.nearestRow(in A, in q, Metric.Euclidean, out int _, out iProxy _));
        Assert.Throws<ArgumentException>(() =>
            Query.farthestRow(in A, in q, Metric.Cosine, out int _, out iProxy _));
        Assert.Throws<ArgumentException>(() =>
            Query.kNearestRows(in A, in q, 2, Metric.Euclidean, ref idx, ref scores));
        Assert.Throws<ArgumentException>(() =>
            Query.kFarthestRows(in A, in q, 2, Metric.Cosine, ref idx, ref scores));
        Assert.Throws<ArgumentException>(() =>
            Query.rowsWithinRadius(in A, in q, (iProxy)1, Metric.Euclidean, ref idx));
        Assert.Throws<ArgumentException>(() =>
            Query.countWithinRadius(in A, in q, (iProxy)1, Metric.Cosine));

        arena.Dispose();
    }

    // Dimension-mismatch contracts: q length must match the relevant axis.
    [Test]
    public void DimensionMismatchThrowTest()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.iProxyMat(3, 4);            // row ops need q.N == N_Cols == 4; col ops need q.N == M_Rows == 3
        var qBadRow = arena.iProxyVec(3);         // wrong for row ops
        var destRow = arena.iProxyVec(3);
        Assert.Throws<ArgumentException>(() =>
            Query.distancesToRow(in A, in qBadRow, Metric.SqEuclidean, ref destRow));
        Assert.Throws<ArgumentException>(() =>
            Query.nearestRow(in A, in qBadRow, Metric.SqEuclidean, out int _, out iProxy _));

        var qBadCol = arena.iProxyVec(4);         // wrong for col ops
        var destCol = arena.iProxyVec(4);
        Assert.Throws<ArgumentException>(() =>
            Query.distancesToColumn(in A, in qBadCol, Metric.SqEuclidean, ref destCol));
        Assert.Throws<ArgumentException>(() =>
            Query.countWithinColumnRadius(in A, in qBadCol, (iProxy)1, Metric.SqEuclidean));

        arena.Dispose();
    }
}
