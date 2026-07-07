using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Tests for row-pivoted (rank-revealing) LQ: LQRP.decomp / LQRP.solveInPlace — the transpose-dual
// of QRCP. P·A = L·Q with the pivot chosen greedily so |L[d,d]| is non-increasing.
//
// Properties (transpose-duals of the QRCP guarantees):
//  - Reconstruction A[P[j], :] == (L*Q)[j, :]; L lower-triangular; Q has orthonormal ROWS (QQᵀ = I_m).
//  - |L[0,0]| >= |L[1,1]| >= ... — the monotone-diagonal guarantee of row pivoting.
//  - The first pivot is the row of largest 2-norm (greedy selection rule).
//  - Numerical row rank is revealed by the count of non-negligible L diagonal entries.
//  - solveInPlace yields the BASIC solution: satisfies the r independent equations (so A x ≈ b for a
//    consistent RHS) but is NOT minimum-norm on a rank-deficient A. At full ROW rank it coincides with
//    LQ.minNormSolve (which IS min-norm) — this pins the basic-vs-min-norm distinction.
public class doubleLQRPTests
{
    // Burst-compile smoke test.
    [BurstCompile(CompileSynchronously = true)]
    public struct AssemblyTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleRandomMat(6, 12);
            var L = arena.doubleMat(6, 6);
            var Q = arena.doubleMat(6, 12);
            var P = new Pivot(6, Allocator.Persistent);

            LQRP.decomp(in A, ref L, ref Q, ref P);

            P.Dispose();
            arena.Dispose();
        }
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // DECOMPOSITION: LQRP.decomp — row-pivoted rank-revealing LQ.
    // ────────────────────────────────────────────────────────────────────────────────
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            ReconstructRandomWide,
            ReconstructRandomSquare,
            FirstPivotLargestRow,
            RankRevealingDeficient,
            SingleElement,
            AllZero,
            ZeroRowMiddle,
            DuplicateRows,
            DecompPreservesA,
            CacheEquivalence,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<double> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ReconstructRandomWide:   ReconstructRandomWide();   break;
                case TestType.ReconstructRandomSquare: ReconstructRandomSquare(); break;
                case TestType.FirstPivotLargestRow:    FirstPivotLargestRow();    break;
                case TestType.RankRevealingDeficient:  RankRevealingDeficient();  break;
                case TestType.SingleElement:           SingleElement();           break;
                case TestType.AllZero:                 AllZero();                 break;
                case TestType.ZeroRowMiddle:           ZeroRowMiddle();           break;
                case TestType.DuplicateRows:           DuplicateRows();           break;
                case TestType.DecompPreservesA:        DecompPreservesA();        break;
                case TestType.CacheEquivalence:        CacheEquivalence();        break;
            }
        }

        void ReconstructRandomWide()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 6, n = 12;
            var P = new Pivot(m, Allocator.Persistent);
            try
            {
                for (uint t = 0; t < 16; t++)
                {
                    var A = arena.doubleRandomMat(m, n, -3f, 3f, 7001 + t * 13);
                    var L = arena.doubleMat(m, m);
                    var Q = arena.doubleMat(m, n);

                    LQRP.decomp(in A, ref L, ref Q, ref P);
                    AssertLQRP(in A, in L, in Q, in P, (double)1E-4f);

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
                    var A = arena.doubleRandomMat(dim, dim, -3f, 3f, 4220 + t * 7);
                    var L = arena.doubleMat(dim, dim);
                    var Q = arena.doubleMat(dim, dim);

                    LQRP.decomp(in A, ref L, ref Q, ref P);
                    AssertLQRP(in A, in L, in Q, in P, (double)1E-4f);

                    arena.Clear();
                }
            }
            finally { P.Dispose(); arena.Dispose(); }
        }

        // Greedy rule: P[0] selects the original ROW of largest 2-norm.
        void FirstPivotLargestRow()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 4, n = 7;
            var A = arena.doubleRandomMat(m, n, -1f, 1f, 31337);

            // Scale rows to distinct magnitudes; row 2 is unambiguously the largest.
            for (int c = 0; c < n; c++)
            {
                A[0, c] *= (double)1f;
                A[1, c] *= (double)4f;
                A[2, c] *= (double)12f;
                A[3, c] *= (double)2f;
            }

            var L = arena.doubleMat(m, m);
            var Q = arena.doubleMat(m, n);
            var P = new Pivot(m, Allocator.Persistent);

            LQRP.decomp(in A, ref L, ref Q, ref P);

            AssertLQRP(in A, in L, in Q, in P, (double)1E-4f);
            RecordEq(P[0], 2); // first pivot is original row 2

            P.Dispose();
            arena.Dispose();
        }

        // A row-rank-3 matrix built as 5 rows with 2 exact linear dependencies. Row pivoting must
        // surface exactly 3 non-negligible L diagonal entries (numerical row rank = 3).
        void RankRevealingDeficient()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 5, n = 6;
            var A = arena.doubleRandomMat(m, n, -1f, 1f, 90210);

            // row3 = 2*row0 - row1 ; row4 = row0 + row2  => exact row rank 3.
            for (int c = 0; c < n; c++)
            {
                A[3, c] = (double)2f * A[0, c] - A[1, c];
                A[4, c] = A[0, c] + A[2, c];
            }

            var L = arena.doubleMat(m, m);
            var Q = arena.doubleMat(m, n);
            var P = new Pivot(m, Allocator.Persistent);

            LQRP.decomp(in A, ref L, ref Q, ref P);
            AssertLQRP(in A, in L, in Q, in P, (double)1E-4f);

            double lead = math.abs(L[0, 0]);
            double rankTol = (double)1E-4f * lead;
            int rank = 0;
            for (int d = 0; d < m; d++)
                if (math.abs(L[d, d]) > rankTol)
                    rank++;
            RecordEq(rank, 3);

            P.Dispose();
            arena.Dispose();
        }

        // Degenerate 1x1 input.
        void SingleElement()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleMat(1, 1);
            A[0, 0] = (double)5f;
            var L = arena.doubleMat(1, 1);
            var Q = arena.doubleMat(1, 1);
            var P = new Pivot(1, Allocator.Persistent);

            LQRP.decomp(in A, ref L, ref Q, ref P);
            AssertLQRP(in A, in L, in Q, in P, (double)1E-4f);
            RecordEq(P[0], 0);

            P.Dispose();
            arena.Dispose();
        }

        // Fully zero matrix: no row has any norm, so no pivot ever fires (P stays identity), L is
        // all-zero, Q stays orthonormal, nothing NaNs (exercises the degenerate-reflector branch).
        void AllZero()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 3, n = 4;
            var A = arena.doubleMat(m, n); // zero-initialised
            var L = arena.doubleMat(m, m);
            var Q = arena.doubleMat(m, n);
            var P = new Pivot(m, Allocator.Persistent);

            LQRP.decomp(in A, ref L, ref Q, ref P);
            AssertLQRP(in A, in L, in Q, in P, (double)1E-4f);
            for (int d = 0; d < m; d++)
                RecordEq(P[d], d);

            P.Dispose();
            arena.Dispose();
        }

        // An exact zero ROW in the middle must be pivoted to the LAST position (smallest — zero —
        // norm), and P must remain a valid permutation. Rows: 0 large, 1 zero, 2 medium => P = [0,2,1].
        void ZeroRowMiddle()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 3, n = 5;
            var A = arena.doubleMat(m, n); // zero-initialised
            // row 0: large norm
            A[0, 0] = (double)6f; A[0, 1] = (double)6f; A[0, 2] = (double)6f;
            // row 1: exact zero (left as 0)
            // row 2: medium norm (< row 0)
            A[2, 0] = (double)2f; A[2, 3] = (double)2f;

            var L = arena.doubleMat(m, m);
            var Q = arena.doubleMat(m, n);
            var P = new Pivot(m, Allocator.Persistent);

            LQRP.decomp(in A, ref L, ref Q, ref P);
            AssertLQRP(in A, in L, in Q, in P, (double)1E-4f);
            RecordEq(P[0], 0); // largest stays first
            RecordEq(P[2], 1); // zero row pushed last

            P.Dispose();
            arena.Dispose();
        }

        // Two identical rows => row-rank deficiency by 1. The duplicate must reduce to a near-zero
        // trailing diagonal (numerical row rank = 2 of 3).
        void DuplicateRows()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 3, n = 4;
            var A = arena.doubleRandomMat(m, n, -1f, 1f, 13579);
            for (int c = 0; c < n; c++)
                A[2, c] = A[0, c]; // row 2 == row 0

            var L = arena.doubleMat(m, m);
            var Q = arena.doubleMat(m, n);
            var P = new Pivot(m, Allocator.Persistent);

            LQRP.decomp(in A, ref L, ref Q, ref P);
            AssertLQRP(in A, in L, in Q, in P, (double)1E-4f);

            double lead = math.abs(L[0, 0]);
            double rankTol = (double)1E-4f * lead;
            int rank = 0;
            for (int d = 0; d < m; d++)
                if (math.abs(L[d, d]) > rankTol)
                    rank++;
            RecordEq(rank, 2);

            P.Dispose();
            arena.Dispose();
        }

        // LQRP.decomp must not modify A. Position-weighted checksum before/after.
        void DecompPreservesA()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 5, n = 10;
            var A = arena.doubleRandomMat(m, n, -3f, 3f, 606060);
            for (int d = 0; d < m; d++) A[d, d] += 4f;

            double checksumBefore = (double)0;
            for (int i = 0; i < A.Length; i++) checksumBefore += A[i] * (double)(i + 1);

            var L = arena.doubleMat(m, m);
            var Q = arena.doubleMat(m, n);
            var P = new Pivot(m, Allocator.Persistent);
            LQRP.decomp(in A, ref L, ref Q, ref P);

            double checksumAfter = (double)0;
            for (int i = 0; i < A.Length; i++) checksumAfter += A[i] * (double)(i + 1);

            if (checksumAfter != checksumBefore && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = checksumAfter;
                Fail[2] = checksumBefore;
                Fail[3] = checksumAfter - checksumBefore;
            }
            Assert.IsTrue(checksumAfter == checksumBefore);

            AssertLQRP(in A, in L, in Q, in P, (double)1E-4f);

            P.Dispose();
            arena.Dispose();
        }

        // Zero-alloc cache overload must match the allocating overload bit-for-bit (same unblocked
        // kernel, same scratch semantics).
        void CacheEquivalence()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 6, n = 11;
            var A = arena.doubleRandomMat(m, n, -3f, 3f, 246810);

            var L1 = arena.doubleMat(m, m); var Q1 = arena.doubleMat(m, n); var P1 = new Pivot(m, Allocator.Persistent);
            var L2 = arena.doubleMat(m, m); var Q2 = arena.doubleMat(m, n); var P2 = new Pivot(m, Allocator.Persistent);

            LQRP.decomp(in A, ref L1, ref Q1, ref P1);

            var ws = arena.doubleLQRPCache(m, n);
            LQRP.decomp(in A, ref L2, ref Q2, ref P2, ref ws);

            for (int i = 0; i < L1.Length; i++) RecordExact(L2[i], L1[i]);
            for (int i = 0; i < Q1.Length; i++) RecordExact(Q2[i], Q1[i]);
            for (int j = 0; j < m; j++) RecordEq(P2[j], P1[j]);

            P1.Dispose(); P2.Dispose();
            arena.Dispose();
        }

        // Reconstruction (A permuted by rows == L*Q), L lower-triangular, Q orthonormal rows, and
        // the monotone-magnitude diagonal guarantee of row pivoting.
        void AssertLQRP(in doubleMxN A, in doubleMxN L, in doubleMxN Q, in Pivot P, double precision)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            // Build A permuted by P: row j == original row P[j].
            var Aperm = A.Copy();
            for (int j = 0; j < m; j++)
                for (int c = 0; c < n; c++)
                    Aperm[j, c] = A[P[j], c];

            doubleMxN shouldBeZero = Aperm - Blas.dot(L, Q);

            if (Analysis.isAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            double zeroError = Analysis.MaxZeroError(shouldBeZero);
            RecordBound(zeroError, precision);

            Assert.IsTrue(Analysis.isZero(in shouldBeZero, precision));
            Assert.IsTrue(Analysis.isLowerTriangular(L, precision));

            // Q orthonormal rows: isOrthogonal(Qᵀ) checks (Qᵀ)ᵀ(Qᵀ) = QQᵀ = I_m.
            doubleMxN Qt = Blas.trans(Q);
            Assert.IsTrue(Analysis.isOrthogonal(in Qt, precision));

            // |L[d,d]| non-increasing (guaranteed by greedy row pivoting).
            double monoTol = precision * (math.abs(L[0, 0]) + (double)1f);
            for (int d = 0; d + 1 < m; d++)
            {
                double hi = math.abs(L[d, d]);
                double lo = math.abs(L[d + 1, d + 1]);
                if (!(hi + monoTol >= lo) && Fail[0] == (double)0)
                {
                    Fail[0] = (double)1;
                    Fail[1] = lo;
                    Fail[2] = hi;
                    Fail[3] = lo - hi;
                }
                Assert.IsTrue(hi + monoTol >= lo);
            }
        }

        void RecordBound(double value, double limit)
        {
            if (!(value <= limit) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = value;
                Fail[2] = limit;
                Fail[3] = value - limit;
            }
        }

        void RecordEq(int got, int expected)
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

        void RecordExact(double got, double expected)
        {
            if (got != expected && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = got;
                Fail[2] = expected;
                Fail[3] = got - expected;
            }
            Assert.IsTrue(got == expected);
        }
    }

    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void RowPivotTests(TestJob.TestType type)
    {
        var fail = new NativeArray<double>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (double)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally { fail.Dispose(); }
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // SOLVER: LQRP.solveInPlace — rank-safe BASIC solution of the underdetermined A x = b.
    // ────────────────────────────────────────────────────────────────────────────────
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SolveTestJob : IJob
    {
        public enum TestType
        {
            FullRowRankMatchesLQ,      // (1) full-row-rank wide: rank==m & x == LQ.minNormSolve (min-norm)
            FullRowRankConsistent,     // (2) full-row-rank, consistent b: A x == b exactly
            RankDeficientConsistent,   // (3) row dependency, consistent b: rank<m & A x ≈ b (basic ok)
            RankDeficientBasicNotMinNorm, // (4) basic solution norm >= LQ/SVD min-norm cross-check
            ZeroMatrix,                // (5) zero matrix: rank 0, x all zeros
            OneByOne,                  // (6) 1x(>1) system
            ExplicitScratchOverload,   // (7) primitive (L/P/v scratch) path
            UninitXContract,           // (8) x is output-only (prior NaN must not survive)
        }

        public TestType Type;
        public NativeArray<double> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.FullRowRankMatchesLQ:         FullRowRankMatchesLQ();         break;
                case TestType.FullRowRankConsistent:        FullRowRankConsistent();        break;
                case TestType.RankDeficientConsistent:      RankDeficientConsistent();      break;
                case TestType.RankDeficientBasicNotMinNorm: RankDeficientBasicNotMinNorm(); break;
                case TestType.ZeroMatrix:                   ZeroMatrix();                   break;
                case TestType.OneByOne:                     OneByOne();                     break;
                case TestType.ExplicitScratchOverload:      ExplicitScratchOverload();      break;
                case TestType.UninitXContract:              UninitXContract();              break;
            }
        }

        // (1) Full ROW rank, well-conditioned wide A. The basic solution must coincide with LQ's
        // minimum-norm solution (both min-norm here), and rank must be m. Allocating default overload.
        void FullRowRankMatchesLQ()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 4, n = 12;
            var A = arena.doubleRandomMat(m, n, -5f, 5f, 778231);
            for (int d = 0; d < m; d++) A[d, d] += (double)10f; // full row rank, well conditioned

            var b = arena.doubleRandomVec(m, -5f, 5f, 9091);

            var Alqrp = A.Copy(); // solveInPlace destroys A
            var x = arena.doubleVec(n);
            int rank = LQRP.solveInPlace(ref Alqrp, ref b, ref x).rank;

            RecordEq(rank, m);
            if (Analysis.isAnyNan(in x)) { Fail0(0, 0); return; }

            // reference: LQ.minNormSolve (min-norm; does not modify A)
            var Alq = A.Copy();
            var xRef = arena.doubleVec(n);
            LQ.minNormSolve(ref Alq, ref b, ref xRef);

            double tol = (double)Consts.doubleSqrtEps * (double)10;
            for (int k = 0; k < n; k++)
                AssertClose(x[k], xRef[k], tol * (math.abs(xRef[k]) + (double)1));

            arena.Dispose();
        }

        // (2) Full row rank, consistent b = A*xTrue: the basic solution reproduces b exactly.
        void FullRowRankConsistent()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 5, n = 9;
            var A = arena.doubleRandomMat(m, n, -3f, 3f, 314221);
            for (int d = 0; d < m; d++) A[d, d] += (double)8f;

            var xTrue = arena.doubleRandomVec(n, -2f, 2f, 1337);
            var b = arena.doubleVec(m);
            Blas.dot(in A, in xTrue, ref b);

            var A_copy = A.Copy();
            var b0 = b.Copy();
            var x = arena.doubleVec(n);
            int rank = LQRP.solveInPlace(ref A, ref b, ref x).rank;

            RecordEq(rank, m);
            if (!Analysis.isAnyNan(in x))
            {
                double tol = (double)Consts.doubleSqrtEps * (double)10;
                double res = ResidualNorm(in A_copy, in x, in b0);
                RecordBound(res, tol * ((double)1 + VecNorm(in b0)));
            }

            arena.Dispose();
        }

        // (3) Rank-deficient by an exact row dependency (row m-1 = row0 + row1), consistent b. Detected
        // rank must be m-1, and the basic solution must still satisfy A x ≈ b (dependent equation is a
        // linear combo of independent ones, so consistency carries over).
        void RankDeficientConsistent()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 4, n = 10;
            var A = arena.doubleRandomMat(m, n, -3f, 3f, 90211);
            for (int c = 0; c < n; c++)
                A[m - 1, c] = A[0, c] + A[1, c]; // exact row dependency -> row rank m-1

            var xTrue = arena.doubleRandomVec(n, -2f, 2f, 4242);
            var b = arena.doubleVec(m);
            Blas.dot(in A, in xTrue, ref b); // consistent RHS

            var A_copy = A.Copy();
            var b0 = b.Copy();
            var x = arena.doubleVec(n);
            int rank = LQRP.solveInPlace(ref A, ref b, ref x).rank;

            RecordEq(rank, m - 1);
            if (!Analysis.isAnyNan(in x))
            {
                double tol = (double)1E-4f;
                double res = ResidualNorm(in A_copy, in x, in b0);
                RecordBound(res, tol * ((double)1 + VecNorm(in b0)));
            }

            arena.Dispose();
        }

        // (4) Basic vs minimum-norm: on a rank-deficient A the LQRP basic solution has norm >= the
        // SVD.pinvSolve minimum-norm solution (both reproduce a consistent b). Pins the distinction.
        void RankDeficientBasicNotMinNorm()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 4, n = 8;
            var A = arena.doubleRandomMat(m, n, -3f, 3f, 55123);
            for (int c = 0; c < n; c++)
                A[m - 1, c] = (double)2f * A[0, c] - A[1, c]; // row dependency -> rank m-1

            var xTrue = arena.doubleRandomVec(n, -2f, 2f, 771);
            var b = arena.doubleVec(m);
            Blas.dot(in A, in xTrue, ref b);
            var b0 = b.Copy();

            var Albrp = A.Copy();
            var x = arena.doubleVec(n);
            LQRP.solveInPlace(ref Albrp, ref b, ref x);
            if (Analysis.isAnyNan(in x)) { Fail0(1, 0); return; }
            double normBasic = VecNorm(in x);

            // min-norm reference
            var Apinv = A.Copy();
            var xPinv = arena.doubleVec(n);
            SVD.pinvSolve(ref Apinv, in b0, ref xPinv);
            double normMin = VecNorm(in xPinv);

            // basic norm must be >= min norm (allow a hair of slack for rounding)
            double slack = (double)1E-4f * (normMin + (double)1);
            RecordBound(normMin, normBasic + slack);
        }

        // (5) Zero matrix: rank 0, x all zeros.
        void ZeroMatrix()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 3, n = 6;
            var A = arena.doubleMat(m, n);
            var b = arena.doubleRandomVec(m, -1f, 1f, 42);
            var x = arena.doubleVec(n);

            int rank = LQRP.solveInPlace(ref A, ref b, ref x).rank;
            RecordEq(rank, 0);
            for (int j = 0; j < n; j++)
                RecordExact(x[j], (double)0);

            arena.Dispose();
        }

        // (6) 1 x n system: x = min-norm solution of a·xᵀ = b0; A x reproduces b0.
        void OneByOne()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 1, n = 3;
            var A = arena.doubleRandomMat(m, n, -2f, 2f, 191);
            A[0, 0] += (double)5f;
            var b = arena.doubleRandomVec(m, -2f, 2f, 202);
            var A_copy = A.Copy();
            var b0 = b.Copy();

            var x = arena.doubleVec(n);
            int rank = LQRP.solveInPlace(ref A, ref b, ref x).rank;
            RecordEq(rank, 1);
            if (!Analysis.isAnyNan(in x))
            {
                double tol = (double)Consts.doubleSqrtEps * (double)10;
                double res = ResidualNorm(in A_copy, in x, in b0);
                RecordBound(res, tol * ((double)1 + VecNorm(in b0)));
            }

            arena.Dispose();
        }

        // (7) Primitive explicit-scratch overload must agree with the allocating overload.
        void ExplicitScratchOverload()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 4, n = 9;
            var A = arena.doubleRandomMat(m, n, -3f, 3f, 30303);
            for (int d = 0; d < m; d++) A[d, d] += (double)7f;
            var b = arena.doubleRandomVec(m, -3f, 3f, 40404);

            var A1 = A.Copy(); var x1 = arena.doubleVec(n);
            LQRP.solveInPlace(ref A1, ref b, ref x1);

            var A2 = A.Copy(); var x2 = arena.doubleVec(n);
            var L = arena.doubleMat(m, m);
            var P = new Pivot(m, Allocator.Persistent);
            var v = arena.doubleVec(n);
            LQRP.solveInPlace(ref A2, ref b, ref x2, ref L, ref P, ref v);

            for (int k = 0; k < n; k++) RecordExact(x2[k], x1[k]);

            P.Dispose();
            arena.Dispose();
        }

        // (8) Uninit-x contract: prior NaN garbage in x must not survive.
        void UninitXContract()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 3, n = 7;
            var A = arena.doubleRandomMat(m, n, -2f, 2f, 6363);
            for (int d = 0; d < m; d++) A[d, d] += (double)5f;
            var b = arena.doubleRandomVec(m, -1f, 1f, 7474);
            var x = arena.doubleVec(n);
            for (int i = 0; i < n; i++) x[i] = double.NaN;

            LQRP.solveInPlace(ref A, ref b, ref x);
            Assert.IsFalse(Analysis.isAnyNan(in x));

            arena.Dispose();
        }

        // ---- helpers ----

        double ResidualNorm(in doubleMxN A, in doubleN x, in doubleN b)
        {
            int m = A.M_Rows;
            double s = (double)0;
            for (int r = 0; r < m; r++)
            {
                double acc = (double)0;
                for (int c = 0; c < A.N_Cols; c++)
                    acc += A[r, c] * x[c];
                double d = acc - b[r];
                s += d * d;
            }
            return math.sqrt(s);
        }

        double VecNorm(in doubleN v)
        {
            double s = (double)0;
            for (int i = 0; i < v.N; i++) s += v[i] * v[i];
            return math.sqrt(s);
        }

        void AssertClose(double got, double expected, double tol)
        {
            double d = math.abs(got - expected);
            if (!(d <= tol) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = got;
                Fail[2] = expected;
                Fail[3] = d;
            }
            Assert.IsTrue(d <= tol);
        }

        void RecordEq(int got, int expected)
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

        void RecordExact(double got, double expected)
        {
            if (got != expected && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = got;
                Fail[2] = expected;
                Fail[3] = got - expected;
            }
            Assert.IsTrue(got == expected);
        }

        void RecordBound(double value, double limit)
        {
            if (!(value <= limit) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = value;
                Fail[2] = limit;
                Fail[3] = value - limit;
            }
            Assert.IsTrue(value <= limit);
        }

        // Records an early-out failure into the Fail[] array ONLY — no in-Burst Assert.Fail(string):
        // that overload is not Burst-compilable (BC1071) and, worse, silently drops the whole job to a
        // Mono fallback. The managed SolveTests wrapper reads Fail[0] after .Run() and fails there, so
        // the interpolated-string diagnostic is reported in managed context where it is legal.
        void Fail0(double code, double extra)
        {
            if (Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = code;
                Fail[2] = extra;
                Fail[3] = (double)0;
            }
        }
    }

    public static Array GetSolveEnums()
    {
        return Enum.GetValues(typeof(SolveTestJob.TestType));
    }

    [TestCaseSource("GetSolveEnums")]
    public void SolveTests(SolveTestJob.TestType type)
    {
        var fail = new NativeArray<double>(4, Allocator.TempJob);
        try
        {
            new SolveTestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (double)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally { fail.Dispose(); }
    }
}
