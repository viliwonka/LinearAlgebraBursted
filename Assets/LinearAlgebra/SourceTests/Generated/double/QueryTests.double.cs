using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;

// Tests for QueryOP Phase 1 (the double float/double core): search & selection over
// vectors / matrices treated as sets of vectors. Spec: docs/spec-query.md.
//
// Coverage groups (mirroring the spec):
//   1 — Extremes: argMaxAbs/argMinAbs (vec+matrix), decodeIndex, rowArgMin/Max + colArgMin/Max.
//   2 — Norm-selection: argMaxRowNorm/argMaxColNorm for each Norm (L1/L2/Linf).
//   3 — Search: distancesToRow/Column, nearest/farthest, kNearest/kFarthest, within-radius/count,
//       for each Metric; the similarity direction flip (Cosine/Dot -> nearest=MAX) is the key check.
//   4 — Value/mask: findValue, nonzero/countNonzero, BoolAnalysis.whichTrue/countTrue.
//   Symmetry — a column op on A equals the row op on transpose(A) (spec P1).
//   Arena wrappers — each allocating wrapper matches the zero-alloc primitive.
//
// Burst-compatible computational tests live in TestJob; managed-throw guards and the
// Indices out-of-range indexer test are plain [Test] methods on the main thread.
public class doubleQueryTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            ArgMaxMinAbsVector,
            ArgMaxMinAbsMatrix,
            DecodeIndex,
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
            Symmetry,
            ArenaWrappers,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected, [3] diff
        public NativeArray<double> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ArgMaxMinAbsVector:           ArgMaxMinAbsVector();           break;
                case TestType.ArgMaxMinAbsMatrix:           ArgMaxMinAbsMatrix();           break;
                case TestType.DecodeIndex:                  DecodeIndex();                  break;
                case TestType.RowColArgMinMax:              RowColArgMinMax();              break;
                case TestType.ColArgStrided:                ColArgStrided();                break;
                case TestType.ArgMaxRowNorm:                ArgMaxRowNorm();                break;
                case TestType.ArgMaxColNorm:                ArgMaxColNorm();                break;
                case TestType.DistancesToRowAllMetrics:     DistancesToRowAllMetrics();     break;
                case TestType.DistancesToColumnAllMetrics:  DistancesToColumnAllMetrics();  break;
                case TestType.NearestFarthestDistance:      NearestFarthestDistance();      break;
                case TestType.NearestFarthestSimilarity:    NearestFarthestSimilarity();    break;
                case TestType.KNearestBruteForce:           KNearestBruteForce();           break;
                case TestType.KNearestClampAndZero:         KNearestClampAndZero();         break;
                case TestType.KFarthest:                    KFarthest();                    break;
                case TestType.WithinRadiusBoundary:         WithinRadiusBoundary();         break;
                case TestType.FindValue:                    FindValue();                    break;
                case TestType.NonzeroCountNonzero:          NonzeroCountNonzero();          break;
                case TestType.Symmetry:                     Symmetry();                     break;
                case TestType.ArenaWrappers:                ArenaWrappers();                break;
            }
        }

        // ---------------------------------------------------------------------
        // GROUP 1 — EXTREMES
        // ---------------------------------------------------------------------

        // argMaxAbs/argMinAbs over a vector: value + flat index; ties -> first occurrence.
        void ArgMaxMinAbsVector()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.doubleVec(6);
            // |.| = 2, 5, 5, 1, 1, 4  => maxAbs first at index 1 (tie with 2), minAbs first at index 3.
            v[0] = (double)(-2); v[1] = (double)5; v[2] = (double)(-5);
            v[3] = (double)1;    v[4] = (double)(-1); v[5] = (double)4;

            doubleQueryOP.argMaxAbs(in v, out double maxVal, out int maxIdx);
            AssertEqI(maxIdx, 1);                  // first of the two |5| entries
            AssertClose(maxVal, (double)5, (double)0);

            doubleQueryOP.argMinAbs(in v, out double minVal, out int minIdx);
            AssertEqI(minIdx, 3);                  // first of the two |1| entries
            AssertClose(minVal, (double)1, (double)0);

            // 1x1 / single-element vector: index 0, value = |element|.
            var one = arena.doubleVec(1);
            one[0] = (double)(-7);
            doubleQueryOP.argMaxAbs(in one, out double ov, out int oi);
            AssertEqI(oi, 0);
            AssertClose(ov, (double)7, (double)0);

            arena.Dispose();
        }

        // argMaxAbs/argMinAbs over a matrix: index is row-major flat; decodeIndex round-trips it.
        void ArgMaxMinAbsMatrix()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleMat(2, 3);
            // row 0: 1 -3  2
            // row 1: 0  4 -4      flat: [1,-3,2,0,4,-4]; maxAbs first |4| at flat 4 (r1,c1).
            A[0, 0] = (double)1;  A[0, 1] = (double)(-3); A[0, 2] = (double)2;
            A[1, 0] = (double)0;  A[1, 1] = (double)4;    A[1, 2] = (double)(-4);

            doubleQueryOP.argMaxAbs(in A, out double maxVal, out int maxIdx);
            AssertEqI(maxIdx, 4);
            AssertClose(maxVal, (double)4, (double)0);
            doubleQueryOP.decodeIndex(maxIdx, A.N_Cols, out int mr, out int mc);
            AssertEqI(mr, 1); AssertEqI(mc, 1);

            // minAbs is the 0 at flat 3 (r1,c0).
            doubleQueryOP.argMinAbs(in A, out double minVal, out int minIdx);
            AssertEqI(minIdx, 3);
            AssertClose(minVal, (double)0, (double)0);
            doubleQueryOP.decodeIndex(minIdx, A.N_Cols, out int nr, out int nc);
            AssertEqI(nr, 1); AssertEqI(nc, 0);

            arena.Dispose();
        }

        // decodeIndex round-trips flat <-> (row,col) for a few hand values.
        void DecodeIndex()
        {
            doubleQueryOP.decodeIndex(0, 4, out int r, out int c);
            AssertEqI(r, 0); AssertEqI(c, 0);
            doubleQueryOP.decodeIndex(7, 4, out r, out c);   // 7 = 1*4 + 3
            AssertEqI(r, 1); AssertEqI(c, 3);
            doubleQueryOP.decodeIndex(11, 3, out r, out c);  // 11 = 3*3 + 2
            AssertEqI(r, 3); AssertEqI(c, 2);
            // round-trip: r*nCols+c -> (r,c)
            int nCols = 5;
            for (int rr = 0; rr < 4; rr++)
                for (int cc = 0; cc < nCols; cc++)
                {
                    doubleQueryOP.decodeIndex(rr * nCols + cc, nCols, out int gr, out int gc);
                    AssertEqI(gr, rr); AssertEqI(gc, cc);
                }
        }

        // rowArgMin/Max + colArgMin/Max value+index forms vs a hand-built oracle.
        void RowColArgMinMax()
        {
            var arena = new Arena(Allocator.Persistent);

            // 3x3:
            //  3  1  2     rowMin@c1=1   rowMax@c0=3
            //  9  7  8     rowMin@c1=7   rowMax@c0=9
            //  0  5  4     rowMin@c0=0   rowMax@c1=5
            var A = arena.doubleMat(3, 3);
            A[0, 0] = (double)3; A[0, 1] = (double)1; A[0, 2] = (double)2;
            A[1, 0] = (double)9; A[1, 1] = (double)7; A[1, 2] = (double)8;
            A[2, 0] = (double)0; A[2, 1] = (double)5; A[2, 2] = (double)4;

            var idxR = arena.Indices(3);
            var valR = arena.doubleVec(3);

            int nr = doubleQueryOP.rowArgMin(in A, ref idxR, ref valR);
            AssertEqI(nr, 3);
            AssertEqI(idxR[0], 1); AssertClose(valR[0], (double)1, (double)0);
            AssertEqI(idxR[1], 1); AssertClose(valR[1], (double)7, (double)0);
            AssertEqI(idxR[2], 0); AssertClose(valR[2], (double)0, (double)0);

            // index-only form must match.
            var idxR2 = arena.Indices(3);
            doubleQueryOP.rowArgMin(in A, ref idxR2);
            AssertEqI(idxR2[0], 1); AssertEqI(idxR2[1], 1); AssertEqI(idxR2[2], 0);

            doubleQueryOP.rowArgMax(in A, ref idxR, ref valR);
            AssertEqI(idxR[0], 0); AssertClose(valR[0], (double)3, (double)0);
            AssertEqI(idxR[1], 0); AssertClose(valR[1], (double)9, (double)0);
            AssertEqI(idxR[2], 1); AssertClose(valR[2], (double)5, (double)0);

            var idxR3 = arena.Indices(3);
            doubleQueryOP.rowArgMax(in A, ref idxR3);
            AssertEqI(idxR3[0], 0); AssertEqI(idxR3[1], 0); AssertEqI(idxR3[2], 1);

            // columns: colMin per column -> rows {2,0,0}; colMax per column -> rows {1,1,1}.
            var idxC = arena.Indices(3);
            var valC = arena.doubleVec(3);
            int nc = doubleQueryOP.colArgMin(in A, ref idxC, ref valC);
            AssertEqI(nc, 3);
            AssertEqI(idxC[0], 2); AssertClose(valC[0], (double)0, (double)0);
            AssertEqI(idxC[1], 0); AssertClose(valC[1], (double)1, (double)0);
            AssertEqI(idxC[2], 0); AssertClose(valC[2], (double)2, (double)0);

            var idxC2 = arena.Indices(3);
            doubleQueryOP.colArgMin(in A, ref idxC2);
            AssertEqI(idxC2[0], 2); AssertEqI(idxC2[1], 0); AssertEqI(idxC2[2], 0);

            doubleQueryOP.colArgMax(in A, ref idxC, ref valC);
            AssertEqI(idxC[0], 1); AssertClose(valC[0], (double)9, (double)0);
            AssertEqI(idxC[1], 1); AssertClose(valC[1], (double)7, (double)0);
            AssertEqI(idxC[2], 1); AssertClose(valC[2], (double)8, (double)0);

            var idxC3 = arena.Indices(3);
            doubleQueryOP.colArgMax(in A, ref idxC3);
            AssertEqI(idxC3[0], 1); AssertEqI(idxC3[1], 1); AssertEqI(idxC3[2], 1);

            arena.Dispose();
        }

        // Strided-column correctness: a non-square matrix where a row-major misread would give
        // a different answer than the true strided column scan.
        void ColArgStrided()
        {
            var arena = new Arena(Allocator.Persistent);

            // 4x2. Column 0 (strided indices 0,2,4,6), column 1 (1,3,5,7).
            //  r0:  1   8
            //  r1:  2   6
            //  r2:  9   7
            //  r3:  3   4
            // col0 min@r0(1) max@r2(9); col1 min@r3(4) max@r0(8).
            // If you (wrongly) read row-major contiguous, col0 would pick up {1,8} not {1,2,9,3}.
            var A = arena.doubleMat(4, 2);
            A[0, 0] = (double)1; A[0, 1] = (double)8;
            A[1, 0] = (double)2; A[1, 1] = (double)6;
            A[2, 0] = (double)9; A[2, 1] = (double)7;
            A[3, 0] = (double)3; A[3, 1] = (double)4;

            var idxC = arena.Indices(2);
            var valC = arena.doubleVec(2);

            doubleQueryOP.colArgMin(in A, ref idxC, ref valC);
            AssertEqI(idxC[0], 0); AssertClose(valC[0], (double)1, (double)0);
            AssertEqI(idxC[1], 3); AssertClose(valC[1], (double)4, (double)0);

            doubleQueryOP.colArgMax(in A, ref idxC, ref valC);
            AssertEqI(idxC[0], 2); AssertClose(valC[0], (double)9, (double)0);
            AssertEqI(idxC[1], 0); AssertClose(valC[1], (double)8, (double)0);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // GROUP 2 — NORM-SELECTION
        // ---------------------------------------------------------------------

        void ArgMaxRowNorm()
        {
            var arena = new Arena(Allocator.Persistent);

            // 3x3 with hand-computed row norms:
            //  r0:  3  0  0  -> L1=3,  L2²=9,   Linf=3
            //  r1:  1  1  1  -> L1=3,  L2²=3,   Linf=1
            //  r2: -2  2  0  -> L1=4,  L2²=8,   Linf=2
            // L1 max -> r2 ; L2 max -> r0 ; Linf max -> r0.
            var A = arena.doubleMat(3, 3);
            A[0, 0] = (double)3;    A[0, 1] = (double)0; A[0, 2] = (double)0;
            A[1, 0] = (double)1;    A[1, 1] = (double)1; A[1, 2] = (double)1;
            A[2, 0] = (double)(-2); A[2, 1] = (double)2; A[2, 2] = (double)0;

            AssertEqI(doubleQueryOP.argMaxRowNorm(in A, Norm.L1), 2);
            AssertEqI(doubleQueryOP.argMaxRowNorm(in A, Norm.L2), 0);
            AssertEqI(doubleQueryOP.argMaxRowNorm(in A, Norm.Linf), 0);

            // Tie -> first occurrence. Two rows of identical L1 norm 5; first is row 0.
            var T = arena.doubleMat(2, 2);
            T[0, 0] = (double)5; T[0, 1] = (double)0;
            T[1, 0] = (double)0; T[1, 1] = (double)5;
            AssertEqI(doubleQueryOP.argMaxRowNorm(in T, Norm.L1), 0);

            arena.Dispose();
        }

        void ArgMaxColNorm()
        {
            var arena = new Arena(Allocator.Persistent);

            // Build the transpose layout of the ArgMaxRowNorm matrix so columns carry the same
            // norms: c0:(3,0,0) c1:(0,1,2... ) — instead just hand-pick columns directly.
            //  cols (each length 3):
            //   c0:  3, 1, -2   -> L1=6, L2²=14, Linf=3
            //   c1:  0, 1,  2   -> L1=3, L2²=5,  Linf=2
            //   c2:  4, 0,  0   -> L1=4, L2²=16, Linf=4
            // L1 max -> c0 ; L2 max -> c2 ; Linf max -> c2.
            var A = arena.doubleMat(3, 3);
            A[0, 0] = (double)3;    A[0, 1] = (double)0; A[0, 2] = (double)4;
            A[1, 0] = (double)1;    A[1, 1] = (double)1; A[1, 2] = (double)0;
            A[2, 0] = (double)(-2); A[2, 1] = (double)2; A[2, 2] = (double)0;

            AssertEqI(doubleQueryOP.argMaxColNorm(in A, Norm.L1), 0);
            AssertEqI(doubleQueryOP.argMaxColNorm(in A, Norm.L2), 2);
            AssertEqI(doubleQueryOP.argMaxColNorm(in A, Norm.Linf), 2);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // GROUP 3 — SEARCH
        // ---------------------------------------------------------------------

        // distancesToRow for every metric against hand-computed distances.
        void DistancesToRowAllMetrics()
        {
            var arena = new Arena(Allocator.Persistent);

            // 2 rows, 2 cols; query q = (0,0).
            //  r0 = (3, 4)
            //  r1 = (1, 0)
            var A = arena.doubleMat(2, 2);
            A[0, 0] = (double)3; A[0, 1] = (double)4;
            A[1, 0] = (double)1; A[1, 1] = (double)0;
            var q = arena.doubleVec(2);
            q[0] = (double)0; q[1] = (double)0;

            var d = arena.doubleVec(2);

            // Manhattan: |3|+|4|=7 ; |1|+|0|=1
            doubleQueryOP.distancesToRow(in A, in q, Metric.Manhattan, ref d);
            AssertClose(d[0], (double)7, fEps()); AssertClose(d[1], (double)1, fEps());

            // Euclidean: 5 ; 1
            doubleQueryOP.distancesToRow(in A, in q, Metric.Euclidean, ref d);
            AssertClose(d[0], (double)5, sqrtEps()); AssertClose(d[1], (double)1, sqrtEps());

            // SqEuclidean (squared, no sqrt): 25 ; 1
            doubleQueryOP.distancesToRow(in A, in q, Metric.SqEuclidean, ref d);
            AssertClose(d[0], (double)25, fEps()); AssertClose(d[1], (double)1, fEps());

            // Chebyshev: max(3,4)=4 ; max(1,0)=1
            doubleQueryOP.distancesToRow(in A, in q, Metric.Chebyshev, ref d);
            AssertClose(d[0], (double)4, fEps()); AssertClose(d[1], (double)1, fEps());

            // Dot with q=(0,0) is 0 for every row.
            doubleQueryOP.distancesToRow(in A, in q, Metric.Dot, ref d);
            AssertClose(d[0], (double)0, fEps()); AssertClose(d[1], (double)0, fEps());

            // Cosine of a zero query vector = 0 (zero-vector guard).
            doubleQueryOP.distancesToRow(in A, in q, Metric.Cosine, ref d);
            AssertClose(d[0], (double)0, fEps()); AssertClose(d[1], (double)0, fEps());

            // Cosine with a non-zero query: q2 = (3,4) is exactly parallel to r0 -> cos=1;
            // r1=(1,0) -> cos = 3/5 = 0.6.
            var q2 = arena.doubleVec(2);
            q2[0] = (double)3; q2[1] = (double)4;
            doubleQueryOP.distancesToRow(in A, in q2, Metric.Cosine, ref d);
            AssertClose(d[0], (double)1,   sqrtEps());
            AssertClose(d[1], (double)0.6, sqrtEps());

            // Dot with q2: r0·q2 = 9+16 = 25 ; r1·q2 = 3+0 = 3.
            doubleQueryOP.distancesToRow(in A, in q2, Metric.Dot, ref d);
            AssertClose(d[0], (double)25, fEps()); AssertClose(d[1], (double)3, fEps());

            arena.Dispose();
        }

        // distancesToColumn for every metric. Columns are strided.
        void DistancesToColumnAllMetrics()
        {
            var arena = new Arena(Allocator.Persistent);

            // 2 rows, 2 cols; query q (length M_Rows = 2) = (0,0).
            //  c0 = (3,1)   c1 = (4,0)
            var A = arena.doubleMat(2, 2);
            A[0, 0] = (double)3; A[0, 1] = (double)4;
            A[1, 0] = (double)1; A[1, 1] = (double)0;
            var q = arena.doubleVec(2);
            q[0] = (double)0; q[1] = (double)0;

            var d = arena.doubleVec(2);

            // Manhattan: c0=4, c1=4
            doubleQueryOP.distancesToColumn(in A, in q, Metric.Manhattan, ref d);
            AssertClose(d[0], (double)4, fEps()); AssertClose(d[1], (double)4, fEps());

            // SqEuclidean: c0 = 9+1 = 10 ; c1 = 16+0 = 16
            doubleQueryOP.distancesToColumn(in A, in q, Metric.SqEuclidean, ref d);
            AssertClose(d[0], (double)10, fEps()); AssertClose(d[1], (double)16, fEps());

            // Euclidean: sqrt(10), 4
            doubleQueryOP.distancesToColumn(in A, in q, Metric.Euclidean, ref d);
            AssertClose(d[0], math.sqrt((double)10), sqrtEps()); AssertClose(d[1], (double)4, sqrtEps());

            // Chebyshev: c0=max(3,1)=3 ; c1=max(4,0)=4
            doubleQueryOP.distancesToColumn(in A, in q, Metric.Chebyshev, ref d);
            AssertClose(d[0], (double)3, fEps()); AssertClose(d[1], (double)4, fEps());

            arena.Dispose();
        }

        // nearest=min / farthest=max for DISTANCE metrics, with score in metric units.
        void NearestFarthestDistance()
        {
            var arena = new Arena(Allocator.Persistent);

            //  r0=(0,0) r1=(3,4) r2=(1,1); q=(0,0).
            var A = arena.doubleMat(3, 2);
            A[0, 0] = (double)0; A[0, 1] = (double)0;
            A[1, 0] = (double)3; A[1, 1] = (double)4;
            A[2, 0] = (double)1; A[2, 1] = (double)1;
            var q = arena.doubleVec(2);
            q[0] = (double)0; q[1] = (double)0;

            // SqEuclidean: distances 0, 25, 2. nearest=r0 (score 0), farthest=r1 (score 25).
            doubleQueryOP.nearestRow(in A, in q, Metric.SqEuclidean, out int ni, out double ns);
            AssertEqI(ni, 0); AssertClose(ns, (double)0, fEps());

            doubleQueryOP.farthestRow(in A, in q, Metric.SqEuclidean, out int fi, out double fs);
            AssertEqI(fi, 1); AssertClose(fs, (double)25, fEps());   // squared units, not sqrt

            // Euclidean score is in euclidean units: farthest = 5.
            doubleQueryOP.farthestRow(in A, in q, Metric.Euclidean, out int fi2, out double fs2);
            AssertEqI(fi2, 1); AssertClose(fs2, (double)5, sqrtEps());

            arena.Dispose();
        }

        // The KEY direction flip: similarity metrics (Cosine/Dot) -> nearest = MAX, farthest = MIN.
        void NearestFarthestSimilarity()
        {
            var arena = new Arena(Allocator.Persistent);

            //  r0=(1,0) r1=(10,0) r2=(-5,0); q=(1,0).
            //  Dot: 1, 10, -5  -> nearest(max)=r1, farthest(min)=r2.
            var A = arena.doubleMat(3, 2);
            A[0, 0] = (double)1;    A[0, 1] = (double)0;
            A[1, 0] = (double)10;   A[1, 1] = (double)0;
            A[2, 0] = (double)(-5); A[2, 1] = (double)0;
            var q = arena.doubleVec(2);
            q[0] = (double)1; q[1] = (double)0;

            doubleQueryOP.nearestRow(in A, in q, Metric.Dot, out int ni, out double ns);
            AssertEqI(ni, 1); AssertClose(ns, (double)10, fEps());

            doubleQueryOP.farthestRow(in A, in q, Metric.Dot, out int fi, out double fs);
            AssertEqI(fi, 2); AssertClose(fs, (double)(-5), fEps());

            // Cosine: r0 & r1 are parallel to q (cos=1), r2 anti-parallel (cos=-1).
            // nearest = max cosine -> first of the two cos=1 rows (r0); farthest = r2.
            doubleQueryOP.nearestRow(in A, in q, Metric.Cosine, out int cni, out double cns);
            AssertEqI(cni, 0); AssertClose(cns, (double)1, sqrtEps());

            doubleQueryOP.farthestRow(in A, in q, Metric.Cosine, out int cfi, out double cfs);
            AssertEqI(cfi, 2); AssertClose(cfs, (double)(-1), sqrtEps());

            // Column twins: nearestColumn/farthestColumn with a similarity metric.
            // Columns of A as vectors of length 3: c0=(1,10,-5), c1=(0,0,0). q3=(1,0,0).
            var q3 = arena.doubleVec(3);
            q3[0] = (double)1; q3[1] = (double)0; q3[2] = (double)0;
            // Dot: c0·q3 = 1 ; c1·q3 = 0 -> nearest(max)=c0, farthest(min)=c1.
            doubleQueryOP.nearestColumn(in A, in q3, Metric.Dot, out int cni2, out double cns2);
            AssertEqI(cni2, 0); AssertClose(cns2, (double)1, fEps());
            doubleQueryOP.farthestColumn(in A, in q3, Metric.Dot, out int cfi2, out double cfs2);
            AssertEqI(cfi2, 1); AssertClose(cfs2, (double)0, fEps());

            arena.Dispose();
        }

        // kNearestRows vs brute-force reference, sorted best-first, ties deterministic.
        void KNearestBruteForce()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 6, N = 3;
            var A = arena.doubleRandomMatrix(M, N, -3f, 3f, 424242);
            var q = arena.doubleVec(N);
            q[0] = (double)0.5; q[1] = (double)(-1); q[2] = (double)2;

            int k = 3;
            var idx = arena.Indices(k);
            var scores = arena.doubleVec(k);

            // --- distance metric (SqEuclidean) ---
            int cnt = doubleQueryOP.kNearestRows(in A, in q, k, Metric.SqEuclidean, ref idx, ref scores);
            AssertEqI(cnt, k);

            // Brute-force all scores then verify: returned set is the k smallest, sorted ascending.
            var all = arena.doubleVec(M);
            doubleQueryOP.distancesToRow(in A, in q, Metric.SqEuclidean, ref all);
            // sorted ascending => scores[0] <= scores[1] <= scores[2]
            for (int i = 0; i + 1 < cnt; i++)
                AssertTrue(scores[i] <= scores[i + 1] + sqrtEps());
            // each returned score equals the metric of its returned index
            for (int i = 0; i < cnt; i++)
                AssertClose(scores[i], all[idx[i]], sqrtEps());
            // the k-th best returned score is <= every non-selected row's score
            double kth = scores[cnt - 1];
            for (int r = 0; r < M; r++)
            {
                bool selected = false;
                for (int i = 0; i < cnt; i++) if (idx[i] == r) selected = true;
                if (!selected) AssertTrue(all[r] >= kth - sqrtEps());
            }

            // --- similarity metric (Dot): best-first = DESCENDING ---
            var idx2 = arena.Indices(k);
            var scores2 = arena.doubleVec(k);
            int cnt2 = doubleQueryOP.kNearestRows(in A, in q, k, Metric.Dot, ref idx2, ref scores2);
            AssertEqI(cnt2, k);
            var allDot = arena.doubleVec(M);
            doubleQueryOP.distancesToRow(in A, in q, Metric.Dot, ref allDot);
            for (int i = 0; i + 1 < cnt2; i++)
                AssertTrue(scores2[i] >= scores2[i + 1] - sqrtEps());
            for (int i = 0; i < cnt2; i++)
                AssertClose(scores2[i], allDot[idx2[i]], sqrtEps());
            double kthDot = scores2[cnt2 - 1];
            for (int r = 0; r < M; r++)
            {
                bool selected = false;
                for (int i = 0; i < cnt2; i++) if (idx2[i] == r) selected = true;
                if (!selected) AssertTrue(allDot[r] <= kthDot + sqrtEps());
            }

            arena.Dispose();
        }

        // k > M_Rows clamps to count; k=0 returns 0; ties handled deterministically.
        void KNearestClampAndZero()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 3, N = 2;
            var A = arena.doubleMat(M, N);
            // all rows identical distance from q -> a full tie; clamp must still return M and
            // produce a valid permutation of {0,1,2} (deterministic insertion order = ascending).
            A[0, 0] = (double)1; A[0, 1] = (double)0;
            A[1, 0] = (double)0; A[1, 1] = (double)1;
            A[2, 0] = (double)(-1); A[2, 1] = (double)0;
            var q = arena.doubleVec(N);
            q[0] = (double)0; q[1] = (double)0;   // every row at SqEuclidean distance 1.

            // k = 10 > M -> clamp to 3.
            int k = 10;
            var idx = arena.Indices(k);
            var scores = arena.doubleVec(k);
            int cnt = doubleQueryOP.kNearestRows(in A, in q, k, Metric.SqEuclidean, ref idx, ref scores);
            AssertEqI(cnt, 3);
            // all returned scores == 1
            for (int i = 0; i < cnt; i++) AssertClose(scores[i], (double)1, fEps());
            // indices form a permutation of {0,1,2}; on a pure tie insertion keeps first-seen order.
            AssertEqI(idx[0], 0); AssertEqI(idx[1], 1); AssertEqI(idx[2], 2);

            // k = 0 -> returns 0, writes nothing.
            int z = doubleQueryOP.kNearestRows(in A, in q, 0, Metric.SqEuclidean, ref idx, ref scores);
            AssertEqI(z, 0);

            arena.Dispose();
        }

        // kFarthestRows: farthest-first ordering vs brute force.
        void KFarthest()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 5, N = 2;
            var A = arena.doubleRandomMatrix(M, N, -2f, 2f, 99887);
            var q = arena.doubleVec(N);
            q[0] = (double)1; q[1] = (double)(-1);

            int k = 2;
            var idx = arena.Indices(k);
            var scores = arena.doubleVec(k);
            int cnt = doubleQueryOP.kFarthestRows(in A, in q, k, Metric.SqEuclidean, ref idx, ref scores);
            AssertEqI(cnt, k);

            var all = arena.doubleVec(M);
            doubleQueryOP.distancesToRow(in A, in q, Metric.SqEuclidean, ref all);
            // farthest-first => descending distance
            for (int i = 0; i + 1 < cnt; i++)
                AssertTrue(scores[i] >= scores[i + 1] - sqrtEps());
            for (int i = 0; i < cnt; i++)
                AssertClose(scores[i], all[idx[i]], sqrtEps());
            // the k-th farthest returned is >= every non-selected row's distance
            double kth = scores[cnt - 1];
            for (int r = 0; r < M; r++)
            {
                bool selected = false;
                for (int i = 0; i < cnt; i++) if (idx[i] == r) selected = true;
                if (!selected) AssertTrue(all[r] <= kth + sqrtEps());
            }

            arena.Dispose();
        }

        // within-radius: inclusive boundary (<= r distance, >= r similarity); buffer fill matches count.
        void WithinRadiusBoundary()
        {
            var arena = new Arena(Allocator.Persistent);

            //  r0=(0,0) r1=(3,4) r2=(1,1) ; q=(0,0).  Euclidean distances: 0, 5, sqrt2.
            var A = arena.doubleMat(3, 2);
            A[0, 0] = (double)0; A[0, 1] = (double)0;
            A[1, 0] = (double)3; A[1, 1] = (double)4;
            A[2, 0] = (double)1; A[2, 1] = (double)1;
            var q = arena.doubleVec(2);
            q[0] = (double)0; q[1] = (double)0;

            // radius exactly 5 (boundary): inclusive -> all three rows qualify.
            var idx = arena.Indices(3);
            int cnt = doubleQueryOP.rowsWithinRadius(in A, in q, (double)5, Metric.Euclidean, ref idx);
            int ccnt = doubleQueryOP.countWithinRadius(in A, in q, (double)5, Metric.Euclidean);
            AssertEqI(cnt, 3); AssertEqI(ccnt, 3);
            // filled indices are 0,1,2 in scan order
            AssertEqI(idx[0], 0); AssertEqI(idx[1], 1); AssertEqI(idx[2], 2);

            // radius just under 5 -> r1 excluded (only the two close rows).
            int cnt2 = doubleQueryOP.rowsWithinRadius(in A, in q, (double)4.9, Metric.Euclidean, ref idx);
            AssertEqI(cnt2, 2);
            AssertEqI(idx[0], 0); AssertEqI(idx[1], 2);
            AssertEqI(doubleQueryOP.countWithinRadius(in A, in q, (double)4.9, Metric.Euclidean), 2);

            // similarity metric (Dot): inclusive >= r. q2=(1,0): dots 0, 3, 1.
            var q2 = arena.doubleVec(2);
            q2[0] = (double)1; q2[1] = (double)0;
            // threshold exactly 1 -> rows with dot >= 1: r1(3) and r2(1). r0(0) excluded.
            int cnt3 = doubleQueryOP.rowsWithinRadius(in A, in q2, (double)1, Metric.Dot, ref idx);
            AssertEqI(cnt3, 2);
            AssertEqI(idx[0], 1); AssertEqI(idx[1], 2);
            AssertEqI(doubleQueryOP.countWithinRadius(in A, in q2, (double)1, Metric.Dot), 2);

            // Column twins. Columns of A length 3: c0=(0,3,1) c1=(0,4,1). qcol=(0,0,0).
            var qcol = arena.doubleVec(3);
            qcol[0] = (double)0; qcol[1] = (double)0; qcol[2] = (double)0;
            // SqEuclidean: c0=0+9+1=10, c1=0+16+1=17. radius 10 inclusive -> only c0.
            var idxc = arena.Indices(2);
            int ccol = doubleQueryOP.columnsWithinRadius(in A, in qcol, (double)10, Metric.SqEuclidean, ref idxc);
            AssertEqI(ccol, 1); AssertEqI(idxc[0], 0);
            AssertEqI(doubleQueryOP.countWithinColumnRadius(in A, in qcol, (double)10, Metric.SqEuclidean), 1);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // GROUP 4 — VALUE / MASK
        // ---------------------------------------------------------------------

        void FindValue()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.doubleVec(5);
            v[0] = (double)1; v[1] = (double)2; v[2] = (double)2; v[3] = (double)3; v[4] = (double)2;

            // first match (within tol 0) at index 1
            AssertEqI(doubleQueryOP.findValue(in v, (double)2, (double)0), 1);
            // absent -> -1
            AssertEqI(doubleQueryOP.findValue(in v, (double)9, (double)0), -1);
            // tol boundary, exactly representable in float+double: target 2.5, tol 0.5 matches the
            // 2's on the inclusive boundary (|2 - 2.5| = 0.5 <= 0.5) -> first 2 at index 1.
            AssertEqI(doubleQueryOP.findValue(in v, (double)2.5, (double)0.5), 1);
            // tol below the gap (|2 - 2.5| = 0.5 > 0.25, |3 - 2.5| = 0.5 > 0.25) -> no match.
            AssertEqI(doubleQueryOP.findValue(in v, (double)2.5, (double)0.25), -1);

            // matrix overload (flat index). 2x2 = [5, 6; 7, 6]; first 6 at flat 1.
            var A = arena.doubleMat(2, 2);
            A[0, 0] = (double)5; A[0, 1] = (double)6;
            A[1, 0] = (double)7; A[1, 1] = (double)6;
            AssertEqI(doubleQueryOP.findValue(in A, (double)6, (double)0), 1);

            arena.Dispose();
        }

        void NonzeroCountNonzero()
        {
            var arena = new Arena(Allocator.Persistent);

            var v = arena.doubleVec(6);
            v[0] = (double)0;   v[1] = (double)0.5; v[2] = (double)0;
            v[3] = (double)(-3); v[4] = (double)0.05; v[5] = (double)0;

            // tol=0: nonzero are indices 1,3,4 -> count 3
            AssertEqI(doubleQueryOP.countNonzero(in v, (double)0), 3);
            var idx = arena.Indices(6);
            int c = doubleQueryOP.nonzero(in v, (double)0, ref idx);
            AssertEqI(c, 3);
            AssertEqI(idx[0], 1); AssertEqI(idx[1], 3); AssertEqI(idx[2], 4);

            // tol=0.1: |0.05| filtered out -> indices 1,3 -> count 2
            AssertEqI(doubleQueryOP.countNonzero(in v, (double)0.1), 2);
            int c2 = doubleQueryOP.nonzero(in v, (double)0.1, ref idx);
            AssertEqI(c2, 2);
            AssertEqI(idx[0], 1); AssertEqI(idx[1], 3);

            // matrix overload, flat indices. 2x2 = [0,2;0,0] -> one nonzero at flat 1.
            var A = arena.doubleMat(2, 2);
            A[0, 0] = (double)0; A[0, 1] = (double)2;
            A[1, 0] = (double)0; A[1, 1] = (double)0;
            AssertEqI(doubleQueryOP.countNonzero(in A, (double)0), 1);
            var idxA = arena.Indices(4);
            int ca = doubleQueryOP.nonzero(in A, (double)0, ref idxA);
            AssertEqI(ca, 1); AssertEqI(idxA[0], 1);

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // SYMMETRY (spec P1): a column op on A == the row op on transpose(A).
        // ---------------------------------------------------------------------

        void Symmetry()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 5, N = 4;
            var A = arena.doubleRandomMatrix(M, N, -3f, 3f, 20240625);
            var At = doubleOP.trans(A);   // N x M

            // For a column op on A, the query length is M_Rows = M; this same q is the row query
            // for At (whose N_Cols = M). So nearestColumn(A,q) == nearestRow(At,q).
            var q = arena.doubleVec(M);
            for (int i = 0; i < M; i++) q[i] = (double)(i - 2) * (double)0.7;

            // nearestColumn(A) == nearestRow(transpose(A)) for a distance metric.
            doubleQueryOP.nearestColumn(in A, in q, Metric.SqEuclidean, out int ci, out double cs);
            doubleQueryOP.nearestRow(in At, in q, Metric.SqEuclidean, out int ri, out double rs);
            AssertEqI(ci, ri); AssertClose(cs, rs, sqrtEps());

            // distancesToColumn(A) == distancesToRow(transpose(A)).
            var dc = arena.doubleVec(N);
            var dr = arena.doubleVec(N);
            doubleQueryOP.distancesToColumn(in A, in q, Metric.Euclidean, ref dc);
            doubleQueryOP.distancesToRow(in At, in q, Metric.Euclidean, ref dr);
            for (int j = 0; j < N; j++)
                AssertClose(dc[j], dr[j], sqrtEps());

            // colArgMin(A) == rowArgMin(transpose(A)): same per-column extreme indices.
            var colIdx = arena.Indices(N);
            var colVal = arena.doubleVec(N);
            doubleQueryOP.colArgMin(in A, ref colIdx, ref colVal);
            var rowIdx = arena.Indices(N);
            var rowVal = arena.doubleVec(N);
            doubleQueryOP.rowArgMin(in At, ref rowIdx, ref rowVal);
            for (int j = 0; j < N; j++)
            {
                AssertEqI(colIdx[j], rowIdx[j]);
                AssertClose(colVal[j], rowVal[j], sqrtEps());
            }

            // argMaxColNorm(A) == argMaxRowNorm(transpose(A)).
            AssertEqI(doubleQueryOP.argMaxColNorm(in A, Norm.L2),
                      doubleQueryOP.argMaxRowNorm(in At, Norm.L2));

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // ARENA WRAPPERS — each allocating wrapper matches the zero-alloc primitive.
        // ---------------------------------------------------------------------

        void ArenaWrappers()
        {
            var arena = new Arena(Allocator.Persistent);

            int M = 6, N = 4;
            var A = arena.doubleRandomMatrix(M, N, -2f, 2f, 7777);
            var q = arena.doubleVec(N);
            for (int i = 0; i < N; i++) q[i] = (double)(i) * (double)0.3 - (double)0.5;

            // --- doubleDistancesToRow / Column wrappers vs primitive ---
            var dr = ArenaExtensions.doubleDistancesToRow(in A, in q, Metric.SqEuclidean);
            var drRef = arena.doubleVec(M);
            doubleQueryOP.distancesToRow(in A, in q, Metric.SqEuclidean, ref drRef);
            AssertEqI(dr.N, M);
            for (int i = 0; i < M; i++) AssertClose(dr[i], drRef[i], sqrtEps());

            var qc = arena.doubleVec(M);
            for (int i = 0; i < M; i++) qc[i] = (double)(i) * (double)0.2;
            var dcol = ArenaExtensions.doubleDistancesToColumn(in A, in qc, Metric.SqEuclidean);
            var dcolRef = arena.doubleVec(N);
            doubleQueryOP.distancesToColumn(in A, in qc, Metric.SqEuclidean, ref dcolRef);
            AssertEqI(dcol.N, N);
            for (int j = 0; j < N; j++) AssertClose(dcol[j], dcolRef[j], sqrtEps());

            // --- doubleNonzeroIndices: exact-sized, contents match the primitive ---
            var idxNz = arena.doubleNonzeroIndices(in A, (double)0.5);
            var refNz = arena.Indices(M * N);
            int refCnt = doubleQueryOP.nonzero(in A, (double)0.5, ref refNz);
            AssertEqI(idxNz.N, refCnt);
            for (int i = 0; i < refCnt; i++) AssertEqI(idxNz[i], refNz[i]);

            // --- doubleRowsWithinRadius: exact-sized, contents match the primitive ---
            double radius = (double)5;
            var idxRR = arena.doubleRowsWithinRadius(in A, in q, radius, Metric.SqEuclidean);
            var refRR = arena.Indices(M);
            int refRRcnt = doubleQueryOP.rowsWithinRadius(in A, in q, radius, Metric.SqEuclidean, ref refRR);
            AssertEqI(idxRR.N, refRRcnt);
            for (int i = 0; i < refRRcnt; i++) AssertEqI(idxRR[i], refRR[i]);

            // --- doubleColumnsWithinRadius ---
            var idxCR = arena.doubleColumnsWithinRadius(in A, in qc, (double)8, Metric.SqEuclidean);
            var refCR = arena.Indices(N);
            int refCRcnt = doubleQueryOP.columnsWithinRadius(in A, in qc, (double)8, Metric.SqEuclidean, ref refCR);
            AssertEqI(idxCR.N, refCRcnt);
            for (int i = 0; i < refCRcnt; i++) AssertEqI(idxCR[i], refCR[i]);

            // --- doubleKNearestRows: idx + scores match the primitive ---
            int k = 3;
            var idxK = arena.doubleKNearestRows(in A, in q, k, Metric.SqEuclidean, out doubleN scoresK, out int cntK);
            var refIdxK = arena.Indices(k);
            var refScoresK = arena.doubleVec(k);
            int refCntK = doubleQueryOP.kNearestRows(in A, in q, k, Metric.SqEuclidean, ref refIdxK, ref refScoresK);
            AssertEqI(cntK, refCntK);
            AssertEqI(idxK.N, refCntK);
            for (int i = 0; i < refCntK; i++)
            {
                AssertEqI(idxK[i], refIdxK[i]);
                AssertClose(scoresK[i], refScoresK[i], sqrtEps());
            }

            // --- doubleKNearestColumns ---
            var idxKC = arena.doubleKNearestColumns(in A, in qc, k, Metric.SqEuclidean, out doubleN scoresKC, out int cntKC);
            var refIdxKC = arena.Indices(k);
            var refScoresKC = arena.doubleVec(k);
            int refCntKC = doubleQueryOP.kNearestColumns(in A, in qc, k, Metric.SqEuclidean, ref refIdxKC, ref refScoresKC);
            AssertEqI(cntKC, refCntKC);
            for (int i = 0; i < refCntKC; i++)
            {
                AssertEqI(idxKC[i], refIdxKC[i]);
                AssertClose(scoresKC[i], refScoresKC[i], sqrtEps());
            }

            arena.Dispose();
        }

        // ---------------------------------------------------------------------
        // helpers
        // ---------------------------------------------------------------------

        // Per-precision tolerances: float needs looser epsilons than double.
        // doubleSqrtEps expands to float (3.45e-4) / double (1.49e-8).
        static double sqrtEps() => (double)Consts.doubleSqrtEps;
        static double fEps() => (double)(100 * Consts.doubleEpsilon);

        // Fail layout: [0]=flag, [1]=got, [2]=expected, [3]=diff
        void AssertClose(double a, double b, double precision)
        {
            double diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertEqI(int got, int expected)
        {
            if (got != expected && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = got;
                Fail[2] = expected;
                Fail[3] = got - expected;
            }
            Assert.AreEqual(expected, got);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = (double)(-1);
                Fail[2] = (double)(-1);
                Fail[3] = (double)(-1);
            }
            Assert.IsTrue(ok);
        }
    }

    void RunJob(TestJob.TestType type)
    {
        var fail = new NativeArray<double>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (double)0)
                Assert.Fail($"got {fail[1]}, expected {fail[2]}, diff {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (double)0)
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
    [Test] public void DecodeIndexTest()                 => RunJob(TestJob.TestType.DecodeIndex);
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
    [Test] public void SymmetryTest()                    => RunJob(TestJob.TestType.Symmetry);
    [Test] public void ArenaWrappersTest()               => RunJob(TestJob.TestType.ArenaWrappers);

    // -------------------------------------------------------------------------
    // GROUP 4 — BoolAnalysis.whichTrue / countTrue (bool mask bridge).
    // Bool ops live outside the per-type QueryOP; run on the main thread.
    // -------------------------------------------------------------------------

    [Test]
    public void WhichTrueCountTrueVector()
    {
        var arena = new Arena(Allocator.Persistent);

        var mask = arena.BoolVector(6);
        mask[0] = false; mask[1] = true; mask[2] = false;
        mask[3] = true;  mask[4] = true; mask[5] = false;

        Assert.AreEqual(3, BoolAnalysis.countTrue(in mask));

        var idx = arena.Indices(6);
        int c = BoolAnalysis.whichTrue(in mask, ref idx);
        Assert.AreEqual(3, c);
        Assert.AreEqual(1, idx[0]);
        Assert.AreEqual(3, idx[1]);
        Assert.AreEqual(4, idx[2]);

        // Arena wrapper returns exact-sized Indices matching the primitive.
        var idxW = arena.WhichTrue(in mask);
        Assert.AreEqual(3, idxW.N);
        Assert.AreEqual(1, idxW[0]); Assert.AreEqual(3, idxW[1]); Assert.AreEqual(4, idxW[2]);

        arena.Dispose();
    }

    [Test]
    public void WhichTrueCountTrueMatrix()
    {
        var arena = new Arena(Allocator.Persistent);

        // 2x3 mask, flat indices of true elements.
        //  T F T
        //  F F T   -> flat true at 0, 2, 5
        var mask = arena.BoolMatrix(2, 3);
        mask[0] = true;  mask[1] = false; mask[2] = true;
        mask[3] = false; mask[4] = false; mask[5] = true;

        Assert.AreEqual(3, BoolAnalysis.countTrue(in mask));

        var idx = arena.Indices(6);
        int c = BoolAnalysis.whichTrue(in mask, ref idx);
        Assert.AreEqual(3, c);
        Assert.AreEqual(0, idx[0]);
        Assert.AreEqual(2, idx[1]);
        Assert.AreEqual(5, idx[2]);

        var idxW = arena.WhichTrue(in mask);
        Assert.AreEqual(3, idxW.N);
        Assert.AreEqual(0, idxW[0]); Assert.AreEqual(2, idxW[1]); Assert.AreEqual(5, idxW[2]);

        arena.Dispose();
    }

    // -------------------------------------------------------------------------
    // Managed-throw guards (main thread): dimension-mismatch + empty-input contracts.
    // -------------------------------------------------------------------------

    [Test]
    public void EmptyArgAbsThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var v0 = arena.doubleVec(0);
        Assert.Throws<InvalidOperationException>(() =>
            doubleQueryOP.argMaxAbs(in v0, out double _, out int _));
        Assert.Throws<InvalidOperationException>(() =>
            doubleQueryOP.argMinAbs(in v0, out double _, out int _));
        arena.Dispose();
    }

    [Test]
    public void RowOpDimensionMismatchThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.doubleMat(3, 4);          // row ops need q.N == A.N_Cols == 4
        var qBad = arena.doubleVec(3);          // wrong length
        var dest = arena.doubleVec(3);

        Assert.Throws<ArgumentException>(() =>
            doubleQueryOP.distancesToRow(in A, in qBad, Metric.SqEuclidean, ref dest));
        Assert.Throws<ArgumentException>(() =>
            doubleQueryOP.nearestRow(in A, in qBad, Metric.SqEuclidean, out int _, out double _));
        Assert.Throws<ArgumentException>(() =>
            doubleQueryOP.countWithinRadius(in A, in qBad, (double)1, Metric.SqEuclidean));
        arena.Dispose();
    }

    [Test]
    public void ColOpDimensionMismatchThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.doubleMat(3, 4);          // col ops need q.N == A.M_Rows == 3
        var qBad = arena.doubleVec(4);          // wrong length
        var dest = arena.doubleVec(4);

        Assert.Throws<ArgumentException>(() =>
            doubleQueryOP.distancesToColumn(in A, in qBad, Metric.SqEuclidean, ref dest));
        Assert.Throws<ArgumentException>(() =>
            doubleQueryOP.nearestColumn(in A, in qBad, Metric.SqEuclidean, out int _, out double _));
        Assert.Throws<ArgumentException>(() =>
            doubleQueryOP.countWithinColumnRadius(in A, in qBad, (double)1, Metric.SqEuclidean));
        arena.Dispose();
    }

    // -------------------------------------------------------------------------
    // Indices type: out-of-range indexer access throws ArgumentOutOfRangeException.
    // -------------------------------------------------------------------------

    [Test]
    public void IndicesOutOfRangeThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        var idx = arena.Indices(3);
        idx[0] = 10; idx[1] = 20; idx[2] = 30;       // valid writes

        Assert.Throws<ArgumentOutOfRangeException>(() => { int _ = idx[3]; });
        Assert.Throws<ArgumentOutOfRangeException>(() => { int _ = idx[-1]; });
        Assert.Throws<ArgumentOutOfRangeException>(() => { idx[3] = 0; });
        arena.Dispose();
    }
}
