using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Tests for column-pivoted (rank-revealing) QR: OrthoOP.qrDecompositionColumnPivot.
// A*P = Q*R with the pivot chosen greedily (Businger–Golub) so |R[d,d]| is non-increasing.
//
// Test vectors / properties sourced from the literature:
//  - Reconstruction A[:,P[j]] == (Q*R)[:,j], Q orthogonal, R upper-triangular (definition).
//  - |R[0,0]| >= |R[1,1]| >= ... — the monotone-diagonal guarantee of column pivoting
//    (Golub & Van Loan, "Matrix Computations", QR with column pivoting; Higham, "What Is a
//    Rank-Revealing Factorization?", nhigham.com/2021/05/19).
//  - The first pivot is the column of largest 2-norm (greedy selection rule).
//  - Numerical rank is revealed by the count of non-negligible R diagonal entries.
//  - Kahan matrix: invariant under QR with column pivoting — no permutation is performed.
//    (Kahan's matrix; see nhigham.com and LAPACK lawn276; it is the canonical case where
//    column pivoting fails to reveal rank precisely because it never pivots.)
public class fProxyOrthoColumnPivotTests
{
    // Burst-compile smoke test.
    [BurstCompile]
    public struct AssemblyTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            var Q = arena.fProxyRandomMatrix(12, 6);
            var R = arena.fProxyMat(6);
            var P = new Pivot(6, Allocator.Persistent);

            OrthoOP.qrDecompositionColumnPivot(ref Q, ref R, ref P);

            P.Dispose();
            arena.Dispose();
        }
    }

    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            ReconstructRandomTall,
            ReconstructRandomSquare,
            FirstPivotLargestColumn,
            RankRevealingDeficient,
            KahanNoPivot,
            SingleElement,
            AllZero,
            ZeroColumnMiddle,
            DuplicateColumns,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<fProxy> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ReconstructRandomTall:    ReconstructRandomTall();    break;
                case TestType.ReconstructRandomSquare:  ReconstructRandomSquare();  break;
                case TestType.FirstPivotLargestColumn:  FirstPivotLargestColumn();  break;
                case TestType.RankRevealingDeficient:   RankRevealingDeficient();   break;
                case TestType.KahanNoPivot:             KahanNoPivot();             break;
                case TestType.SingleElement:            SingleElement();            break;
                case TestType.AllZero:                  AllZero();                  break;
                case TestType.ZeroColumnMiddle:         ZeroColumnMiddle();         break;
                case TestType.DuplicateColumns:         DuplicateColumns();         break;
            }
        }

        void ReconstructRandomTall()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 12, n = 6;
            var P = new Pivot(n, Allocator.Persistent); // size fixed across iterations; reset internally
            try
            {
                for (uint t = 0; t < 16; t++)
                {
                    var Q = arena.fProxyRandomMatrix(m, n, -3f, 3f, 7001 + t * 13);
                    var R = arena.fProxyMat(n);
                    var A = Q.Copy();

                    OrthoOP.qrDecompositionColumnPivot(ref Q, ref R, ref P);

                    AssertQRCP(in A, in Q, in R, in P, (fProxy)1E-4f);

                    arena.Clear();
                }
            }
            finally { P.Dispose(); arena.Dispose(); }
        }

        void ReconstructRandomSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;
            var P = new Pivot(dim, Allocator.Persistent);
            try
            {
                for (uint t = 0; t < 16; t++)
                {
                    var Q = arena.fProxyRandomMatrix(dim, dim, -3f, 3f, 4220 + t * 7);
                    var R = arena.fProxyMat(dim);
                    var A = Q.Copy();

                    OrthoOP.qrDecompositionColumnPivot(ref Q, ref R, ref P);

                    AssertQRCP(in A, in Q, in R, in P, (fProxy)1E-4f);

                    arena.Clear();
                }
            }
            finally { P.Dispose(); arena.Dispose(); }
        }

        // Greedy rule: P[0] selects the original column of largest 2-norm.
        void FirstPivotLargestColumn()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 7, n = 4;
            var Q = arena.fProxyRandomMatrix(m, n, -1f, 1f, 31337);

            // Scale columns to distinct magnitudes; column 2 is unambiguously the largest.
            for (int r = 0; r < m; r++)
            {
                Q[r, 0] *= (fProxy)1f;
                Q[r, 1] *= (fProxy)4f;
                Q[r, 2] *= (fProxy)12f;
                Q[r, 3] *= (fProxy)2f;
            }

            var R = arena.fProxyMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var A = Q.Copy();

            OrthoOP.qrDecompositionColumnPivot(ref Q, ref R, ref P);

            // Reconstruction must still hold...
            AssertQRCP(in A, in Q, in R, in P, (fProxy)1E-4f);
            // ...and the first pivot must be original column 2.
            RecordEq(P[0], 2);

            P.Dispose();
            arena.Dispose();
        }

        // A rank-3 matrix built as 5 columns with 2 exact linear dependencies. Column pivoting
        // must surface exactly 3 non-negligible R diagonal entries (numerical rank = 3) and drive
        // the trailing 2 to ~0.
        void RankRevealingDeficient()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 6, n = 5;
            var Q = arena.fProxyRandomMatrix(m, n, -1f, 1f, 90210);

            // col3 = 2*col0 - col1 ; col4 = col0 + col2  => exact rank 3.
            for (int r = 0; r < m; r++)
            {
                Q[r, 3] = (fProxy)2f * Q[r, 0] - Q[r, 1];
                Q[r, 4] = Q[r, 0] + Q[r, 2];
            }

            var R = arena.fProxyMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var A = Q.Copy();

            OrthoOP.qrDecompositionColumnPivot(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (fProxy)1E-4f);

            // Count non-negligible diagonal entries relative to the leading one. The gap here is
            // enormous (rank-3 entries are O(|R00|); the 2 dependent columns reduce to the float
            // round-off of forming them, ~1e-6 relative), so 1e-4 relative cleanly separates them
            // and is robust across float/double and seeds.
            fProxy lead = math.abs(R[0, 0]);
            fProxy rankTol = (fProxy)1E-4f * lead;
            int rank = 0;
            for (int d = 0; d < n; d++)
                if (math.abs(R[d, d]) > rankTol)
                    rank++;

            RecordEq(rank, 3);

            P.Dispose();
            arena.Dispose();
        }

        // Kahan matrix K = S*U, S=diag(s^i), U upper-tri with 1 on diagonal and -c above.
        // Every column has 2-norm exactly 1, and the matrix is invariant under column pivoting:
        // no permutation should occur (P is the identity).
        void KahanNoPivot()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 6;
            fProxy c = (fProxy)0.2f;
            fProxy s = math.sqrt((fProxy)1f - c * c);

            var Q = arena.fProxyMat(dim, dim); // zero-initialised
            fProxy si = 1f;                    // s^i
            for (int i = 0; i < dim; i++)
            {
                Q[i, i] = si;                  // diagonal: s^i
                for (int j = i + 1; j < dim; j++)
                    Q[i, j] = -c * si;         // above diagonal: -c*s^i
                si *= s;
            }

            var R = arena.fProxyMat(dim);
            var P = new Pivot(dim, Allocator.Persistent);
            var A = Q.Copy();

            OrthoOP.qrDecompositionColumnPivot(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (fProxy)1E-4f);

            // No pivoting: P must be the identity permutation.
            for (int d = 0; d < dim; d++)
                RecordEq(P[d], d);

            P.Dispose();
            arena.Dispose();
        }

        // Degenerate 1x1 input: Q = [+/-1], R = [+/-a], P = [0]; reconstruction must hold.
        void SingleElement()
        {
            var arena = new Arena(Allocator.Persistent);

            var Q = arena.fProxyMat(1, 1);
            Q[0, 0] = (fProxy)5f;
            var R = arena.fProxyMat(1);
            var P = new Pivot(1, Allocator.Persistent);
            var A = Q.Copy();

            OrthoOP.qrDecompositionColumnPivot(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (fProxy)1E-4f);
            RecordEq(P[0], 0);

            P.Dispose();
            arena.Dispose();
        }

        // Fully zero matrix: no column has any norm, so no pivot ever fires (P stays identity),
        // R is all-zero, Q stays orthogonal, and nothing produces a NaN (exercises the
        // degenerate-reflector branch at every step).
        void AllZero()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 4, n = 3;
            var Q = arena.fProxyMat(m, n); // zero-initialised
            var R = arena.fProxyMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var A = Q.Copy();

            OrthoOP.qrDecompositionColumnPivot(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (fProxy)1E-4f);
            for (int d = 0; d < n; d++)
                RecordEq(P[d], d); // no pivoting on an all-zero matrix

            P.Dispose();
            arena.Dispose();
        }

        // An exact zero column in the middle must be pivoted to the LAST position (it has the
        // smallest — zero — norm), and P must remain a valid permutation. Columns: 0 large,
        // 1 zero, 2 medium => expected pivot order P = [0, 2, 1].
        void ZeroColumnMiddle()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 5, n = 3;
            var Q = arena.fProxyMat(m, n); // zero-initialised
            // column 0: large norm
            Q[0, 0] = (fProxy)6f; Q[1, 0] = (fProxy)6f; Q[2, 0] = (fProxy)6f;
            // column 1: exact zero (left as 0)
            // column 2: medium norm (< column 0)
            Q[0, 2] = (fProxy)2f; Q[3, 2] = (fProxy)2f;

            var R = arena.fProxyMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var A = Q.Copy();

            OrthoOP.qrDecompositionColumnPivot(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (fProxy)1E-4f);
            RecordEq(P[0], 0); // largest stays first
            RecordEq(P[2], 1); // zero column pushed last

            P.Dispose();
            arena.Dispose();
        }

        // Two identical columns => rank deficiency by 1. Tie handling must stay deterministic and
        // the duplicate must reduce to a near-zero trailing diagonal (numerical rank = 2 of 3).
        void DuplicateColumns()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 4, n = 3;
            var Q = arena.fProxyRandomMatrix(m, n, -1f, 1f, 13579);
            for (int r = 0; r < m; r++)
                Q[r, 2] = Q[r, 0]; // column 2 == column 0

            var R = arena.fProxyMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var A = Q.Copy();

            OrthoOP.qrDecompositionColumnPivot(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (fProxy)1E-4f);

            fProxy lead = math.abs(R[0, 0]);
            fProxy rankTol = (fProxy)1E-4f * lead;
            int rank = 0;
            for (int d = 0; d < n; d++)
                if (math.abs(R[d, d]) > rankTol)
                    rank++;
            RecordEq(rank, 2);

            P.Dispose();
            arena.Dispose();
        }

        // Reconstruction (A permuted by P == Q*R), R upper-triangular, Q orthogonal, and the
        // monotone-magnitude diagonal guarantee of column pivoting.
        void AssertQRCP(in fProxyMxN A, in fProxyMxN Q, in fProxyMxN R, in Pivot P, fProxy precision)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            // Build A permuted by P: column j == original column P[j]. Copy() gives a same-shape
            // arena matrix; we overwrite every entry from the (untouched) original A.
            var Aperm = A.Copy();
            for (int r = 0; r < m; r++)
                for (int j = 0; j < n; j++)
                    Aperm[r, j] = A[r, P[j]];

            fProxyMxN shouldBeZero = Aperm - fProxyOP.dot(Q, R);

            if (Analysis.IsAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            fProxy zeroError = Analysis.MaxZeroError(shouldBeZero);
            RecordBound(zeroError, precision);

            Assert.IsTrue(Analysis.IsZero(in shouldBeZero, precision));
            Assert.IsTrue(Analysis.IsUpperTriangular(R, precision));
            Assert.IsTrue(Analysis.IsOrthogonal(Q, precision));

            // |R[d,d]| non-increasing (guaranteed by greedy column pivoting). Allow a small
            // absolute slack relative to the leading magnitude for float rounding.
            fProxy monoTol = precision * (math.abs(R[0, 0]) + (fProxy)1f);
            for (int d = 0; d + 1 < n; d++)
            {
                fProxy hi = math.abs(R[d, d]);
                fProxy lo = math.abs(R[d + 1, d + 1]);
                if (!(hi + monoTol >= lo) && Fail[0] == (fProxy)0)
                {
                    Fail[0] = (fProxy)1;
                    Fail[1] = lo;        // got (next diagonal)
                    Fail[2] = hi;        // expected upper bound (this diagonal)
                    Fail[3] = lo - hi;   // excess
                }
                Assert.IsTrue(hi + monoTol >= lo);
            }
        }

        void RecordBound(fProxy value, fProxy limit)
        {
            if (!(value <= limit) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = value;
                Fail[2] = limit;
                Fail[3] = value - limit;
            }
        }

        void RecordEq(int got, int expected)
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
    }

    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void ColumnPivotTests(TestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }
}
