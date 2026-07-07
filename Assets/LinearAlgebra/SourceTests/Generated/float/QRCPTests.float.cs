using System;

using LinearAlgebra;
using LinearAlgebra.Gallery;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Tests for column-pivoted (rank-revealing) QR: QRCP.decompInPlace.
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
public class floatQRCPTests
{
    // Burst-compile smoke test.
    [BurstCompile(CompileSynchronously = true)]
    public struct AssemblyTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            var Q = arena.floatRandomMat(12, 6);
            var R = arena.floatMat(6);
            var P = new Pivot(6, Allocator.Persistent);

            QRCP.decompInPlace(ref Q, ref R, ref P);

            P.Dispose();
            arena.Dispose();
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            ReconstructRandomTall,
            ReconstructRandomSquare,
            FirstPivotLargestColumn,
            RankRevealingDeficient,
            KahanNoPivot,
            KahanGalleryReconstruct,
            SingleElement,
            AllZero,
            ZeroColumnMiddle,
            DuplicateColumns,
            // Solver API rework (commit 2) coverage.
            DecompPreservesA,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<float> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ReconstructRandomTall:    ReconstructRandomTall();    break;
                case TestType.ReconstructRandomSquare:  ReconstructRandomSquare();  break;
                case TestType.FirstPivotLargestColumn:  FirstPivotLargestColumn();  break;
                case TestType.RankRevealingDeficient:   RankRevealingDeficient();   break;
                case TestType.KahanNoPivot:             KahanNoPivot();             break;
                case TestType.KahanGalleryReconstruct:  KahanGalleryReconstruct();  break;
                case TestType.SingleElement:            SingleElement();            break;
                case TestType.AllZero:                  AllZero();                  break;
                case TestType.ZeroColumnMiddle:         ZeroColumnMiddle();         break;
                case TestType.DuplicateColumns:         DuplicateColumns();         break;
                case TestType.DecompPreservesA:         DecompPreservesA();         break;
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
                    var Q = arena.floatRandomMat(m, n, -3f, 3f, 7001 + t * 13);
                    var R = arena.floatMat(n);
                    var A = Q.Copy();

                    QRCP.decompInPlace(ref Q, ref R, ref P);

                    AssertQRCP(in A, in Q, in R, in P, (float)1E-4f);

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
                    var Q = arena.floatRandomMat(dim, dim, -3f, 3f, 4220 + t * 7);
                    var R = arena.floatMat(dim);
                    var A = Q.Copy();

                    QRCP.decompInPlace(ref Q, ref R, ref P);

                    AssertQRCP(in A, in Q, in R, in P, (float)1E-4f);

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
            var Q = arena.floatRandomMat(m, n, -1f, 1f, 31337);

            // Scale columns to distinct magnitudes; column 2 is unambiguously the largest.
            for (int r = 0; r < m; r++)
            {
                Q[r, 0] *= (float)1f;
                Q[r, 1] *= (float)4f;
                Q[r, 2] *= (float)12f;
                Q[r, 3] *= (float)2f;
            }

            var R = arena.floatMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var A = Q.Copy();

            QRCP.decompInPlace(ref Q, ref R, ref P);

            // Reconstruction must still hold...
            AssertQRCP(in A, in Q, in R, in P, (float)1E-4f);
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
            var Q = arena.floatRandomMat(m, n, -1f, 1f, 90210);

            // col3 = 2*col0 - col1 ; col4 = col0 + col2  => exact rank 3.
            for (int r = 0; r < m; r++)
            {
                Q[r, 3] = (float)2f * Q[r, 0] - Q[r, 1];
                Q[r, 4] = Q[r, 0] + Q[r, 2];
            }

            var R = arena.floatMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var A = Q.Copy();

            QRCP.decompInPlace(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (float)1E-4f);

            // Count non-negligible diagonal entries relative to the leading one. The gap here is
            // enormous (rank-3 entries are O(|R00|); the 2 dependent columns reduce to the float
            // round-off of forming them, ~1e-6 relative), so 1e-4 relative cleanly separates them
            // and is robust across float/double and seeds.
            float lead = math.abs(R[0, 0]);
            float rankTol = (float)1E-4f * lead;
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
            float c = (float)0.2f;
            float s = math.sqrt((float)1f - c * c);

            var Q = arena.floatMat(dim, dim); // zero-initialised
            float si = 1f;                    // s^i
            for (int i = 0; i < dim; i++)
            {
                Q[i, i] = si;                  // diagonal: s^i
                for (int j = i + 1; j < dim; j++)
                    Q[i, j] = -c * si;         // above diagonal: -c*s^i
                si *= s;
            }

            var R = arena.floatMat(dim);
            var P = new Pivot(dim, Allocator.Persistent);
            var A = Q.Copy();

            QRCP.decompInPlace(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (float)1E-4f);

            // No pivoting: P must be the identity permutation.
            for (int d = 0; d < dim; d++)
                RecordEq(P[d], d);

            P.Dispose();
            arena.Dispose();
        }

        // GALLERY KNOWN-ANSWER (Gallery.Special): n=8 Kahan matrix via the gallery generator (built
        // from theta, K[i,i]=s^i, K[i,j]=-c·s^i). It is ill-conditioned — the canonical QRCP stress
        // case. Independent of whether any pivot fires, the factorisation must satisfy the defining
        // properties: A·P ≈ Q·R, R upper-triangular, Q orthogonal, and |R[d,d]| non-increasing
        // (all checked by AssertQRCP). Uses a different n/theta than KahanNoPivot for extra coverage.
        void KahanGalleryReconstruct()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;
            float theta = (float)1.2f; // c=cos, s=sin both comfortably away from 0

            var Q = arena.floatKahan(dim, theta);
            var R = arena.floatMat(dim);
            var P = new Pivot(dim, Allocator.Persistent);
            var A = Q.Copy();

            QRCP.decompInPlace(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (float)1E-4f);

            P.Dispose();
            arena.Dispose();
        }

        // Degenerate 1x1 input: Q = [+/-1], R = [+/-a], P = [0]; reconstruction must hold.
        void SingleElement()
        {
            var arena = new Arena(Allocator.Persistent);

            var Q = arena.floatMat(1, 1);
            Q[0, 0] = (float)5f;
            var R = arena.floatMat(1);
            var P = new Pivot(1, Allocator.Persistent);
            var A = Q.Copy();

            QRCP.decompInPlace(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (float)1E-4f);
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
            var Q = arena.floatMat(m, n); // zero-initialised
            var R = arena.floatMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var A = Q.Copy();

            QRCP.decompInPlace(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (float)1E-4f);
            for (int d = 0; d < n; d++)
                RecordEq(P[d], d);

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
            var Q = arena.floatMat(m, n); // zero-initialised
            // column 0: large norm
            Q[0, 0] = (float)6f; Q[1, 0] = (float)6f; Q[2, 0] = (float)6f;
            // column 1: exact zero (left as 0)
            // column 2: medium norm (< column 0)
            Q[0, 2] = (float)2f; Q[3, 2] = (float)2f;

            var R = arena.floatMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var A = Q.Copy();

            QRCP.decompInPlace(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (float)1E-4f);
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
            var Q = arena.floatRandomMat(m, n, -1f, 1f, 13579);
            for (int r = 0; r < m; r++)
                Q[r, 2] = Q[r, 0]; // column 2 == column 0

            var R = arena.floatMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var A = Q.Copy();

            QRCP.decompInPlace(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (float)1E-4f);

            float lead = math.abs(R[0, 0]);
            float rankTol = (float)1E-4f * lead;
            int rank = 0;
            for (int d = 0; d < n; d++)
                if (math.abs(R[d, d]) > rankTol)
                    rank++;
            RecordEq(rank, 2);

            P.Dispose();
            arena.Dispose();
        }

        // Solver API rework (commit 2): QRCP.decomp must not modify A. Checksum (position-weighted
        // sum, so a permutation or a single altered entry both trip it) before/after the call.
        void DecompPreservesA()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 10, n = 5;
            var A = arena.floatRandomMat(m, n, -3f, 3f, 606060);
            for (int d = 0; d < n; d++) A[d, d] += 4f;

            float checksumBefore = (float)0;
            for (int i = 0; i < A.Length; i++) checksumBefore += A[i] * (float)(i + 1);

            var Q = arena.floatMat(m, n);
            var R = arena.floatMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            QRCP.decomp(in A, ref Q, ref R, ref P);

            float checksumAfter = (float)0;
            for (int i = 0; i < A.Length; i++) checksumAfter += A[i] * (float)(i + 1);

            if (checksumAfter != checksumBefore && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = checksumAfter;
                Fail[2] = checksumBefore;
                Fail[3] = checksumAfter - checksumBefore;
            }
            Assert.IsTrue(checksumAfter == checksumBefore);

            // and the decomposition itself must still be correct (A intact, matches Q*R via P).
            AssertQRCP(in A, in Q, in R, in P, (float)1E-4f);

            P.Dispose();
            arena.Dispose();
        }

        // Reconstruction (A permuted by P == Q*R), R upper-triangular, Q orthogonal, and the
        // monotone-magnitude diagonal guarantee of column pivoting.
        void AssertQRCP(in floatMxN A, in floatMxN Q, in floatMxN R, in Pivot P, float precision)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            // Build A permuted by P: column j == original column P[j]. Copy() gives a same-shape
            // arena matrix; we overwrite every entry from the (untouched) original A.
            var Aperm = A.Copy();
            for (int r = 0; r < m; r++)
                for (int j = 0; j < n; j++)
                    Aperm[r, j] = A[r, P[j]];

            floatMxN shouldBeZero = Aperm - Blas.dot(Q, R);

            if (Analysis.isAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            float zeroError = Analysis.MaxZeroError(shouldBeZero);
            RecordBound(zeroError, precision);

            Assert.IsTrue(Analysis.isZero(in shouldBeZero, precision));
            Assert.IsTrue(Analysis.isUpperTriangular(R, precision));
            Assert.IsTrue(Analysis.isOrthogonal(Q, precision));

            // |R[d,d]| non-increasing (guaranteed by greedy column pivoting). Allow a small
            // absolute slack relative to the leading magnitude for float rounding.
            float monoTol = precision * (math.abs(R[0, 0]) + (float)1f);
            for (int d = 0; d + 1 < n; d++)
            {
                float hi = math.abs(R[d, d]);
                float lo = math.abs(R[d + 1, d + 1]);
                if (!(hi + monoTol >= lo) && Fail[0] == (float)0)
                {
                    Fail[0] = (float)1;
                    Fail[1] = lo;        // got (next diagonal)
                    Fail[2] = hi;        // expected upper bound (this diagonal)
                    Fail[3] = lo - hi;   // excess
                }
                Assert.IsTrue(hi + monoTol >= lo);
            }
        }

        void RecordBound(float value, float limit)
        {
            if (!(value <= limit) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = value;
                Fail[2] = limit;
                Fail[3] = value - limit;
            }
        }

        void RecordEq(int got, int expected)
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
    }

    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void ColumnPivotTests(TestJob.TestType type)
    {
        var fail = new NativeArray<float>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (float)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (float)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // SOLVER: QRCP.solveInPlace — QRCP-based rank-safe least-squares (BASIC / truncated
    // solution). Solves min‖A x − b‖ for a possibly rank-deficient A (m >= n). Returns the
    // detected numerical rank and the basic solution (≤ rank nonzeros in permuted order):
    // minimal RESIDUAL but NOT minimum norm. At full column rank it reduces to ordinary QR-LS.
    //
    // Cross-checks (where used): SVD.pinvSolve gives the SAME residual (also residual-minimal)
    // but the MINIMUM-NORM solution, so ‖x_pinv‖ <= ‖x_qrcp‖ — this pins the basic-vs-min-norm
    // distinction. All four overloads are exercised across the cases below.
    // ────────────────────────────────────────────────────────────────────────────────
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SolveTestJob : IJob
    {
        public enum TestType
        {
            FullRankAgreesWithQR,   // (1) overdetermined full-rank: rank==n & x == QR.solveInPlace
            FullRankSquare,         // (2) square full-rank: A x == b
            RankDeficientResidual,  // (3) dependent column: rank, minimal residual, pinv cross-check
            Rank1Projection,        // (4) Pei(4,0) rank-1, n>1: residual minimal, A x == proj(b)
            OverdeterminedDeficient,// (5) m > n, r < n (two dependencies)
            ZeroMatrix,             // (6) zero matrix: rank 0, x all zeros
            OneByOne,               // (7) 1x1 system: x == b/a (projection formula)
            AutoSentinel,           // (8) relTol=-1 == default overload == explicit default tol
            KnownValueRegression,   // (9) hand-computable rank-deficient basic solution
            RankInfoStatus,         // (10) Stage-3: RankInfo.status/rank/Solved on rank-deficient A
            NoCopyEquivalenceFullRank,      // (11) commit-2: no-copy solveInPlace == copying-then-solveInPlace
            NoCopyEquivalenceRankDeficient, // (12) same, rank-deficient A
            BlockedFusedSolve,              // (13) large n (>= 2*QRCP_BLOCK): fused blocked solve == QR-LS
            MinNormRankDeficientTall,       // (14) COD: min-norm == SVD pinv, genuinely below the basic solution
            MinNormFullRankEqualsBasic,     // (15) full column rank: minNormSolveInPlace == basic solveInPlace (bit-identical)
            MinNormConsistent,              // (16) rank-deficient consistent b: reconstructs A x ≈ b, min-norm
            MinNormScratchEquivalence,      // (17) allocating minNormSolveInPlace == primitive scratch overload
            MinNormBlocked,                 // (18) large n (>= 2*QRCP_BLOCK) rank-deficient: blocked factor + COD == SVD pinv
            MinNormZeroMatrix,              // (19) zero matrix: rank 0, x all zeros
            MinNormKnownRank1,              // (20) literature: rank-1 A=[[1,0],[2,0]], closed-form A+ (Wikipedia)
            MinNormKnownRank2TallInconsistent, // (21) literature matrix (R ginv tutorial), hand-derived min-norm x
            MinNormMultiRHSMatchesSingle,   // (22) block minNormSolveInPlace == per-column single-RHS COD
            MinNormDecompSolveMatchesFused, // (23) factor-reuse minNormDecompSolve == fused block COD
            MinNormPseudoinverseIdentity,   // (24) B=I -> X=A+, verify A A+ A == A (Penrose)
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<float> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.FullRankAgreesWithQR:    FullRankAgreesWithQR();    break;
                case TestType.FullRankSquare:          FullRankSquare();          break;
                case TestType.RankDeficientResidual:   RankDeficientResidual();   break;
                case TestType.Rank1Projection:         Rank1Projection();         break;
                case TestType.OverdeterminedDeficient: OverdeterminedDeficient(); break;
                case TestType.ZeroMatrix:              ZeroMatrix();              break;
                case TestType.OneByOne:                OneByOne();                break;
                case TestType.AutoSentinel:            AutoSentinel();            break;
                case TestType.KnownValueRegression:    KnownValueRegression();    break;
                case TestType.RankInfoStatus:          RankInfoStatus();          break;
                case TestType.NoCopyEquivalenceFullRank:      NoCopyEquivalenceFullRank();      break;
                case TestType.NoCopyEquivalenceRankDeficient: NoCopyEquivalenceRankDeficient(); break;
                case TestType.BlockedFusedSolve:              BlockedFusedSolve();              break;
                case TestType.MinNormRankDeficientTall:       MinNormRankDeficientTall();       break;
                case TestType.MinNormFullRankEqualsBasic:     MinNormFullRankEqualsBasic();     break;
                case TestType.MinNormConsistent:              MinNormConsistent();              break;
                case TestType.MinNormScratchEquivalence:      MinNormScratchEquivalence();      break;
                case TestType.MinNormBlocked:                 MinNormBlocked();                 break;
                case TestType.MinNormZeroMatrix:              MinNormZeroMatrix();              break;
                case TestType.MinNormKnownRank1:              MinNormKnownRank1();              break;
                case TestType.MinNormKnownRank2TallInconsistent: MinNormKnownRank2TallInconsistent(); break;
                case TestType.MinNormMultiRHSMatchesSingle:   MinNormMultiRHSMatchesSingle();   break;
                case TestType.MinNormDecompSolveMatchesFused: MinNormDecompSolveMatchesFused(); break;
                case TestType.MinNormPseudoinverseIdentity:   MinNormPseudoinverseIdentity();   break;
            }
        }

        // (1) Overdetermined, full column rank, well-conditioned (diag-boosted random). The basic
        // solution must coincide with ordinary (un-pivoted) QR least-squares to tolerance, and the
        // detected rank must be the full n. Uses the ALLOCATING default overload.
        void FullRankAgreesWithQR()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 12, n = 4;
            var A = arena.floatRandomMat(m, n, -5f, 5f, 778231);
            for (int d = 0; d < n; d++)
                A[d, d] += (float)10f; // boost leading block -> full column rank, good conditioning

            // generic b (not in range(A)) so it is a genuine least-squares (not exact) problem
            var b = arena.floatRandomVec(m, -5f, 5f, 9091);
            var A_pristine = A.Copy(); // solveInPlace destroys A (reflectors, NOT Q now); preserve for the QR reference
            var bqr = b.Copy();        // solveInPlace destroys b too (overwritten with Qᵀb); copy for the QR reference

            var x = arena.floatVec(n);
            int rank = QRCP.solveInPlace(ref A, ref b, ref x).rank; // A and b BOTH destroyed (fused fast path)

            RecordEq(rank, n);
            if (Analysis.isAnyNan(in x)) { Fail0(0, 0); return; }

            // reference: ordinary QR-LS (destroys its inputs -> Aqr copy + bqr copy above)
            var Aqr = A_pristine.Copy();
            var xRef = arena.floatVec(n);
            QR.solveInPlace(ref Aqr, ref bqr, ref xRef);

            float tol = (float)Consts.floatSqrtEps * (float)10;
            for (int k = 0; k < n; k++)
                AssertClose(x[k], xRef[k], tol * (math.abs(xRef[k]) + (float)1));

            arena.Dispose();
        }

        // (2) Square full-rank: the basic solution solves A x = b exactly (residual ~ 0).
        // Uses the PRIMITIVE default-tolerance overload (R/P/u scratch; A_to_Q and b both destroyed).
        void FullRankSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;
            var A = arena.floatRandomMat(dim, dim, -5f, 5f, 314221);
            for (int d = 0; d < dim; d++)
                A[d, d] += (float)10f;

            var xOrig = arena.floatRandomVec(dim, -3f, 3f, 1337);
            var b = Blas.dot(A, xOrig); // b in range(A) -> exact solution exists
            var A_copy = A.Copy();          // for residual check after the solve
            var b0 = b.Copy();              // solveInPlace destroys b (fused); keep original for the residual

            var R = arena.floatMat(dim);
            var P = new Pivot(dim, Allocator.Persistent);
            var u = arena.floatVec(dim);
            var x = arena.floatVec(dim);

            int rank = QRCP.solveInPlace(ref A, ref b, ref x, ref R, ref P, ref u).rank;

            RecordEq(rank, dim);
            if (!Analysis.isAnyNan(in x))
            {
                float tol = (float)Consts.floatSqrtEps * (float)10;
                for (int k = 0; k < dim; k++)
                    AssertClose(x[k], xOrig[k], tol * (math.abs(xOrig[k]) + (float)1));

                // residual ~ 0
                float res = ResidualNorm(in A_copy, in x, in b0);
                RecordBound(res, tol * ((float)1 + VecNorm(in b0)));
            }

            P.Dispose();
            arena.Dispose();
        }

        // (3) Rank-deficient by an EXACT linear dependency (col3 = col0 + col1). Detected rank must
        // be the true rank, the residual must be the irreducible minimum (cross-checked against
        // SVD.pinvSolve, which minimizes the SAME residual), and the basic solution must have norm
        // >= the minimum-norm pinv solution. Uses the ALLOCATING default overload.
        void RankDeficientResidual()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 6, n = 4;
            var A = arena.floatRandomMat(m, n, -3f, 3f, 90211);
            for (int r = 0; r < m; r++)
                A[r, 3] = A[r, 0] + A[r, 1]; // exact dependency -> true rank 3
            var A_copy = A.Copy();

            var b = arena.floatRandomVec(m, -3f, 3f, 5511);
            var b0 = b.Copy();   // solveInPlace destroys b (fused); keep original for residual/pinv

            var x = arena.floatVec(n);
            int rank = QRCP.solveInPlace(ref A, ref b, ref x).rank;

            RecordEq(rank, 3);
            if (Analysis.isAnyNan(in x)) { Fail0(1, 0); return; }

            float resQrcp = ResidualNorm(in A_copy, in x, in b0);
            float normQrcp = VecNorm(in x);

            // pinv reference (no longer modifies A) — same residual, minimum norm
            var Apinv = A_copy.Copy();
            var xPinv = arena.floatVec(n);
            RankInfo pinvInfo = SVD.pinvSolve(ref Apinv, in b0, ref xPinv);
            bool converged = pinvInfo;
            int pinvRank = pinvInfo.rank;

            RecordEq(pinvRank, 3);
            float resPinv = ResidualNorm(in A_copy, in xPinv, in b0);
            float normPinv = VecNorm(in xPinv);

            // (a) SAME residual (both are residual-minimal). Residual is second-order flat at the
            // optimum, so even pinv's iterative x reproduces the minimum value tightly.
            float resTol = (float)Consts.floatSqrtEps * (float)4 * (resPinv + (float)1);
            AssertClose(resQrcp, resPinv, resTol);

            // (b) basic solution is NOT minimum-norm: ‖x_pinv‖ <= ‖x_qrcp‖ (with slack).
            float normSlack = (float)Consts.floatSqrtEps * (float)10 * (normQrcp + (float)1);
            RecordBound(normPinv - normQrcp, normSlack);

            arena.Dispose();
        }

        // (4) Rank-1 with n>1 (truncation + un-permute both exercised). Pei(4,0) is the all-ones
        // 4x4 (rank 1). For an all-ones A, (A x)[i] = Σ x_j is a single constant; least squares
        // fits that constant to the mean of b. So the reconstruction A x must equal mean(b)·ones,
        // the residual must be minimal (pinv cross-check), and ‖x_pinv‖ <= ‖x_qrcp‖.
        void Rank1Projection()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 4;
            var A = arena.floatPei(dim, (float)0); // all-ones, rank 1
            var A_copy = A.Copy();

            var b = arena.floatRandomVec(dim, -4f, 4f, 24680);
            var b0 = b.Copy();   // solveInPlace destroys b (fused); keep original for residual/pinv
            float mean = (float)0;
            for (int i = 0; i < dim; i++) mean += b[i];
            mean /= (float)dim;

            var x = arena.floatVec(dim);
            int rank = QRCP.solveInPlace(ref A, ref b, ref x).rank;

            RecordEq(rank, 1);
            if (Analysis.isAnyNan(in x)) { Fail0(1, 0); return; }

            // reconstruction A x must be the projection of b onto span(ones) = mean(b)*ones
            var Ax = Blas.dot(A_copy, x);
            float tol = (float)Consts.floatSqrtEps * (float)10;
            for (int i = 0; i < dim; i++)
                AssertClose(Ax[i], mean, tol * (math.abs(mean) + (float)1));

            // residual minimal vs pinv, and basic norm >= min norm
            float resQrcp = ResidualNorm(in A_copy, in x, in b0);
            float normQrcp = VecNorm(in x);

            var Apinv = A_copy.Copy();
            var xPinv = arena.floatVec(dim);
            RankInfo pinvInfo = SVD.pinvSolve(ref Apinv, in b0, ref xPinv);
            bool converged = pinvInfo;
            int pinvRank = pinvInfo.rank;
            RecordEq(pinvRank, 1);
            float resPinv = ResidualNorm(in A_copy, in xPinv, in b0);
            float normPinv = VecNorm(in xPinv);

            AssertClose(resQrcp, resPinv, (float)Consts.floatSqrtEps * (float)4 * (resPinv + (float)1));
            RecordBound(normPinv - normQrcp, (float)Consts.floatSqrtEps * (float)10 * (normQrcp + (float)1));

            arena.Dispose();
        }

        // (5) Overdetermined (m > n) AND rank-deficient (r < n) via two exact dependencies
        // (col3 = 2*col0 - col1 ; col4 = col0 + col2) => rank 3 of 5. Residual minimal (pinv check).
        // Uses the PRIMITIVE with an explicit positive tolerance (= the library default).
        void OverdeterminedDeficient()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 8, n = 5;
            var A = arena.floatRandomMat(m, n, -2f, 2f, 90210);
            for (int r = 0; r < m; r++)
            {
                A[r, 3] = (float)2f * A[r, 0] - A[r, 1];
                A[r, 4] = A[r, 0] + A[r, 2];
            }
            var A_copy = A.Copy();

            var b = arena.floatRandomVec(m, -2f, 2f, 1212);
            var b0 = b.Copy();   // solveInPlace destroys b (fused); keep original for residual/pinv

            var R = arena.floatMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var u = arena.floatVec(m);
            var x = arena.floatVec(n);

            float explicitTol = (float)(math.max(m, n)) * (float)Consts.floatZeroThreshold;
            int rank = QRCP.solveInPlace(ref A, ref b, ref x, ref R, ref P, ref u, explicitTol).rank;

            RecordEq(rank, 3);
            if (Analysis.isAnyNan(in x)) { Fail0(1, 0); return; }

            float resQrcp = ResidualNorm(in A_copy, in x, in b0);

            var Apinv = A_copy.Copy();
            var xPinv = arena.floatVec(n);
            RankInfo pinvInfo = SVD.pinvSolve(ref Apinv, in b0, ref xPinv);
            bool converged = pinvInfo;
            int pinvRank = pinvInfo.rank;
            RecordEq(pinvRank, 3);
            float resPinv = ResidualNorm(in A_copy, in xPinv, in b0);

            AssertClose(resQrcp, resPinv, (float)Consts.floatSqrtEps * (float)4 * (resPinv + (float)1));

            // basic norm >= min norm
            RecordBound(VecNorm(in xPinv) - VecNorm(in x),
                        (float)Consts.floatSqrtEps * (float)10 * (VecNorm(in x) + (float)1));

            P.Dispose();
            arena.Dispose();
        }

        // (6) Zero matrix (m=5, n=3): no column has any norm -> rank 0 and x is all zeros (no NaN).
        void ZeroMatrix()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 5, n = 3;
            var A = arena.floatMat(m, n);                          // zero-initialised
            var b = arena.floatRandomVec(m, -5f, 5f, 5151);

            var x = arena.floatVec(n);
            int rank = QRCP.solveInPlace(ref A, ref b, ref x).rank;

            RecordEq(rank, 0);
            if (Analysis.isAnyNan(in x)) { Fail0(1, 0); return; }
            for (int k = 0; k < n; k++)
                AssertClose(x[k], (float)0, (float)Consts.floatSqrtEps);

            arena.Dispose();
        }

        // (7) 1x1 system A=[a], b=[β]: the only column has full rank, so x[0] = (a·β)/(a·a) = β/a
        // (the projection formula). Pick a=4, β=10 -> x=2.5, residual 0, rank 1.
        void OneByOne()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.floatMat(1, 1);
            A[0, 0] = (float)4f;
            var A_copy = A.Copy();

            var b = arena.floatVec(1);
            b[0] = (float)10f;
            var b0 = b.Copy();   // solveInPlace destroys b (fused); keep original for the residual

            var x = arena.floatVec(1);
            int rank = QRCP.solveInPlace(ref A, ref b, ref x).rank;

            RecordEq(rank, 1);
            if (Analysis.isAnyNan(in x)) { Fail0(1, 0); return; }

            AssertClose(x[0], (float)2.5f, (float)Consts.floatSqrtEps * (float)10);
            RecordBound(ResidualNorm(in A_copy, in x, in b0), (float)Consts.floatSqrtEps * (float)10);

            arena.Dispose();
        }

        // (8) Auto sentinel: relTol = -1 must select the documented default
        // (max(m,n)*Consts.floatZeroThreshold). Verify it produces the SAME rank and the SAME x as
        // (a) the default overload and (b) the explicit positive default tolerance — bit-for-bit
        // (identical code path). Exercised on a rank-deficient system so rank/truncation matter.
        void AutoSentinel()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 6, n = 4;
            var A = arena.floatRandomMat(m, n, -3f, 3f, 4242);
            for (int r = 0; r < m; r++)
                A[r, 2] = A[r, 0] - A[r, 1]; // rank 3
            var b = arena.floatRandomVec(m, -3f, 3f, 2424);

            // solveInPlace destroys BOTH A and b (fused fast path) -- each call needs its own pristine
            // A copy AND its own b copy so all three exercise the IDENTICAL input.
            var xAuto = arena.floatVec(n);
            var Aauto = A.Copy(); var bAuto = b.Copy();
            int rankAuto = QRCP.solveInPlace(ref Aauto, ref bAuto, ref xAuto).rank; // default overload

            var xNeg = arena.floatVec(n);
            var Aneg = A.Copy(); var bNeg = b.Copy();
            int rankNeg = QRCP.solveInPlace(ref Aneg, ref bNeg, ref xNeg, (float)(-1)).rank; // sentinel

            float explicitTol = (float)(math.max(m, n)) * (float)Consts.floatZeroThreshold;
            var xExpl = arena.floatVec(n);
            var Aexpl = A.Copy(); var bExpl = b.Copy();
            int rankExpl = QRCP.solveInPlace(ref Aexpl, ref bExpl, ref xExpl, explicitTol).rank;

            RecordEq(rankNeg, rankAuto);
            RecordEq(rankExpl, rankAuto);

            for (int k = 0; k < n; k++)
            {
                // identical computation -> exact equality
                AssertClose(xNeg[k], xAuto[k], (float)0);
                AssertClose(xExpl[k], xAuto[k], (float)0);
            }

            arena.Dispose();
        }

        // (9) KNOWN-VALUE regression. A = [[1,2],[0,0],[0,0]] (3x2): col1 = 2*col0 -> rank 1, and
        // col1 has the larger norm so column pivoting promotes it to position 0. The basic solution
        // therefore zeros the FREE variable (original col0) and solves the leading 1x1 block on
        // col1: 2*x1 = b0 = 6 -> x1 = 3. After un-permuting, x = (0, 3) EXACTLY. Residual = ‖(0,-1,-1)‖
        // = sqrt(2). This pins truncation + un-permute against a hand-computed answer.
        void KnownValueRegression()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 3, n = 2;
            var A = arena.floatMat(m, n);  // zero-initialised
            A[0, 0] = (float)1f; A[0, 1] = (float)2f; // only row 0 is nonzero
            var A_copy = A.Copy();

            var b = arena.floatVec(m);
            b[0] = (float)6f; b[1] = (float)1f; b[2] = (float)1f;
            var b0 = b.Copy();   // solveInPlace destroys b (fused); keep original for the residual

            var R = arena.floatMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var u = arena.floatVec(m);
            var x = arena.floatVec(n);

            int rank = QRCP.solveInPlace(ref A, ref b, ref x, ref R, ref P, ref u, (float)(-1)).rank;

            RecordEq(rank, 1);
            if (Analysis.isAnyNan(in x)) { Fail0(1, 0); return; }

            float tol = (float)Consts.floatSqrtEps * (float)10;
            AssertClose(x[0], (float)0f, tol); // free variable (original col0) zeroed
            AssertClose(x[1], (float)3f, tol); // pivoted col1 carries the rank-1 solution

            float res = ResidualNorm(in A_copy, in x, in b0);
            AssertClose(res, math.sqrt((float)2f), tol);

            P.Dispose();
            arena.Dispose();
        }

        // (10) Stage-3 direct-solve-status coverage: on a rank-deficient A (exact linear
        // dependency, true rank 3 of 4), QRCP.solveInPlace must return a RankInfo with
        // status == RankDeficient, rank == the detected reduced rank, and Solved == true (a
        // rank-deficient basic solution is still usable) -- distinct from a hard failure.
        void RankInfoStatus()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 6, n = 4;
            var A = arena.floatRandomMat(m, n, -3f, 3f, 90211);
            for (int r = 0; r < m; r++)
                A[r, 3] = A[r, 0] + A[r, 1]; // exact dependency -> true rank 3

            var b = arena.floatRandomVec(m, -3f, 3f, 5511);
            var x = arena.floatVec(n);

            RankInfo info = QRCP.solveInPlace(ref A, ref b, ref x);

            RecordEq(info.rank, 3);
            RecordEq((int)info.status, (int)DirectSolveStatus.RankDeficient);
            RecordEq(info.Solved ? 1 : 0, 1);
            RecordEq(info ? 1 : 0, 1); // implicit bool must also be true

            arena.Dispose();
        }

        // (11)/(12) no-copy optimization: solveInPlace factors A_to_Q's own buffer directly (no
        // separate Q scratch param). Buffer identity must not matter: running the fused primitive on
        // one independent (A, b) copy pair must be bit-identical (x, status, rank, and the destroyed
        // A-buffer) to running it on a SEPARATE independent pair -- proving the no-copy/in-place design
        // is a pure perf choice, not an observable behavior change. (The fused solve no longer forms Q,
        // so there is no Q exit to cross-check against decompInPlace here — that lives in the decomp tests.)
        void NoCopyEquivalenceFullRank() => NoCopyEquivalence(10, 6, 707070, false);
        void NoCopyEquivalenceRankDeficient() => NoCopyEquivalence(8, 5, 808080, true);

        void NoCopyEquivalence(int m, int n, uint seed, bool rankDeficient)
        {
            var arena = new Arena(Allocator.Persistent);

            var A0 = arena.floatRandomMat(m, n, -3f, 3f, seed);
            for (int d = 0; d < n; d++) A0[d, d] += 6f;
            if (rankDeficient)
                for (int r = 0; r < m; r++)
                    A0[r, n - 1] = A0[r, 0] + A0[r, 1]; // exact dependency -> rank n-1

            var b0 = arena.floatRandomVec(m, -3f, 3f, seed + 1);

            // Path 1: solveInPlace on one independent (A, b) copy pair.
            var Adirect = A0.Copy();
            var bDirect = b0.Copy();
            var xDirect = arena.floatVec(n);
            RankInfo infoDirect = QRCP.solveInPlace(ref Adirect, ref bDirect, ref xDirect);

            // Path 2: solveInPlace on a SEPARATE independent copy pair -- proves buffer identity is
            // irrelevant to the result. (The fused fast path destroys BOTH A and b, so each run needs
            // its own pair; the old b-is-read-only reuse no longer holds.)
            var Acopy = A0.Copy();
            var bCopy = b0.Copy();
            var xCopy = arena.floatVec(n);
            RankInfo infoCopy = QRCP.solveInPlace(ref Acopy, ref bCopy, ref xCopy);

            RecordEq((int)infoDirect.status, (int)infoCopy.status);
            RecordEq(infoDirect.rank, infoCopy.rank);
            // (2f-iii) The two no-copy paths agreeing isn't enough: pin the ABSOLUTE detected rank so
            // a bug that consistently reports the WRONG rank (e.g. always full rank) can't slip
            // through and the rank-deficient/truncation path is actually exercised. The construction
            // forces column n-1 = col0 + col1, an exact dependency making true rank n-1.
            if (rankDeficient)
                RecordEq(infoDirect.rank, n - 1);
            for (int i = 0; i < n; i++)
                AssertBitIdentical(xDirect[i], xCopy[i]);

            // The destroyed A-buffers (stored reflectors + R, NOT Q — the fused solve never
            // reconstructs Q) must be bit-identical to each other too: the factorization is
            // deterministic and independent of b, so buffer identity can't change it. (Q itself is
            // produced by QRCP.decompInPlace and covered by the decomposition tests, not here.)
            for (int i = 0; i < Adirect.Length; i++)
                AssertBitIdentical(Adirect[i], Acopy[i]);

            arena.Dispose();
        }

        // (13) Large n (m=200, n=96 >= 2*QRCP_BLOCK): forces the fused BLOCKED solve path
        // (decompInPlaceBlockedCore's fusedSolve mode + Qᵀb + no Q reconstruction — the fast path all
        // the small cases above miss). Well-conditioned overdetermined full-rank system: the fused
        // solution must equal ordinary QR least-squares and be residual-minimal, and rank must be n.
        void BlockedFusedSolve()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 200, n = 96;
            var A = arena.floatRandomMat(m, n, -2f, 2f, 0xF0DE71u);
            for (int d = 0; d < n; d++)
                A[d, d] += (float)n; // diagonally dominant -> full column rank, well-conditioned
            var A_copy = A.Copy();

            var b = arena.floatRandomVec(m, -2f, 2f, 0x5013u);
            var b0 = b.Copy();  // solveInPlace destroys b (fused); keep original for residual + QR ref

            var x = arena.floatVec(n);
            int rank = QRCP.solveInPlace(ref A, ref b, ref x).rank;

            RecordEq(rank, n);
            if (Analysis.isAnyNan(in x)) { Fail0(1, 0); return; }

            // reference: ordinary (un-pivoted) QR least-squares on copies (it destroys its inputs).
            var Aqr = A_copy.Copy();
            var bqr = b0.Copy();
            var xRef = arena.floatVec(n);
            QR.solveInPlace(ref Aqr, ref bqr, ref xRef);

            float tol = (float)Consts.floatSqrtEps * (float)20;
            for (int k = 0; k < n; k++)
                AssertClose(x[k], xRef[k], tol * (math.abs(xRef[k]) + (float)1));

            // residual-minimal (QRCP basic solution == QR-LS residual at full rank).
            float resQrcp = ResidualNorm(in A_copy, in x, in b0);
            float resQr = ResidualNorm(in A_copy, in xRef, in b0);
            AssertClose(resQrcp, resQr, tol * (resQr + (float)1));

            arena.Dispose();
        }

        // ---- COD (minNormSolveInPlace): minimum-norm / pseudoinverse least-squares ----

        // (14) The core COD test. Tall, rank-deficient with SEVERAL free variables (m=12, n=8, rank 5:
        // cols 5,6,7 are combinations of cols 0..4), generic (inconsistent) b. The COD solution must:
        //   (a) equal the SVD pseudoinverse on BOTH norm and residual (it IS the min-norm LS solution),
        //   (b) be no larger in norm than the basic solution, and
        //   (c) be GENUINELY smaller than the basic solution here (proving COD is not a no-op) — this
        //       construction gives a large basic-vs-min-norm gap (see docs / the benchmark probe).
        void MinNormRankDeficientTall()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 12, n = 8, r = 5;
            var A = arena.floatRandomMat(m, n, -3f, 3f, 0xC0D1u);
            for (int row = 0; row < m; row++)
            {
                A[row, 5] = A[row, 0] + A[row, 1];               // 3 exact dependencies -> rank 5
                A[row, 6] = (float)2f * A[row, 2] - A[row, 3];
                A[row, 7] = A[row, 0] - A[row, 4];
            }
            var A0 = A.Copy();
            var b = arena.floatRandomVec(m, -3f, 3f, 0xB1Au);
            var b0 = b.Copy();

            // COD min-norm
            var Acod = A0.Copy(); var bcod = b0.Copy();
            var xCod = arena.floatVec(n);
            RankInfo codInfo = QRCP.minNormSolveInPlace(ref Acod, ref bcod, ref xCod);
            RecordEq(codInfo.rank, r);
            if (Analysis.isAnyNan(in xCod)) { Fail0(14, 0); return; }
            float normCod = VecNorm(in xCod);
            float resCod = ResidualNorm(in A0, in xCod, in b0);

            // basic (truncated) solution
            var Abas = A0.Copy(); var bbas = b0.Copy();
            var xBas = arena.floatVec(n);
            QRCP.solveInPlace(ref Abas, ref bbas, ref xBas);
            float normBas = VecNorm(in xBas);
            float resBas = ResidualNorm(in A0, in xBas, in b0);

            // SVD pseudoinverse oracle
            var Apinv = A0.Copy();
            var xPinv = arena.floatVec(n);
            RankInfo pinvInfo = SVD.pinvSolve(ref Apinv, in b0, ref xPinv);
            RecordEq(pinvInfo.rank, r);
            float normPinv = VecNorm(in xPinv);
            float resPinv = ResidualNorm(in A0, in xPinv, in b0);

            // (a) COD == pinv on norm AND residual (min-norm LS solution is unique).
            float tol = (float)Consts.floatSqrtEps * (float)20;
            AssertClose(normCod, normPinv, tol * (normPinv + (float)1));
            AssertClose(resCod, resPinv, tol * (resPinv + (float)1));
            // all three residuals coincide (basic is also LS-optimal on residual — only the NORM differs).
            AssertClose(resCod, resBas, tol * (resBas + (float)1));
            // (b) min-norm <= basic (+ slack).
            RecordBound(normCod - normBas, (float)Consts.floatSqrtEps * (float)10 * (normBas + (float)1));
            // (c) COD is not a no-op: the basic solution is DISTINGUISHABLY larger in norm than the
            // min-norm one (gap well above rounding noise). A COD that mistakenly returned the basic
            // solution would collapse the gap to ~0 and fail this. (The gap's magnitude varies with b /
            // precision — the real gap is far bigger on high-deficiency problems, see the benchmark
            // probe — so this floors only at "clearly nonzero", not a fixed fraction.)
            RecordBound((float)100 * Consts.floatSqrtEps * (normCod + (float)1), normBas - normCod);

            arena.Dispose();
        }

        // (15) Full COLUMN rank tall: there are no free variables, so the min-norm solution IS the basic
        // solution. minNormSolveInPlace routes r==n through the same fused factor + finish as
        // solveInPlace, so the two must be BIT-IDENTICAL (not merely close).
        void MinNormFullRankEqualsBasic()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 10, n = 5;
            var A = arena.floatRandomMat(m, n, -4f, 4f, 0x0F0Fu);
            for (int d = 0; d < n; d++) A[d, d] += (float)8f;   // full column rank, well-conditioned
            var A0 = A.Copy();
            var b = arena.floatRandomVec(m, -4f, 4f, 0x7A7Au);
            var b0 = b.Copy();

            var Amin = A0.Copy(); var bmin = b0.Copy();
            var xMin = arena.floatVec(n);
            int rankMin = QRCP.minNormSolveInPlace(ref Amin, ref bmin, ref xMin).rank;

            var Abas = A0.Copy(); var bbas = b0.Copy();
            var xBas = arena.floatVec(n);
            int rankBas = QRCP.solveInPlace(ref Abas, ref bbas, ref xBas).rank;

            RecordEq(rankMin, n);
            RecordEq(rankBas, n);
            for (int k = 0; k < n; k++)
                AssertBitIdentical(xMin[k], xBas[k]);

            arena.Dispose();
        }

        // (16) Rank-deficient with a CONSISTENT b (b = A·xTrue): the min-norm solution must reconstruct b
        // (residual ~ 0) and match the SVD pseudoinverse norm.
        void MinNormConsistent()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 7, n = 5, r = 3;
            var A = arena.floatRandomMat(m, n, -3f, 3f, 0x5A5Au);
            for (int row = 0; row < m; row++)
            {
                A[row, 3] = A[row, 0] + A[row, 1];               // rank 3 (2 dependencies)
                A[row, 4] = A[row, 0] - A[row, 2];
            }
            var A0 = A.Copy();
            var xTrue = arena.floatRandomVec(n, -2f, 2f, 0x1234u);
            var b = Blas.dot(A, xTrue);                          // consistent
            var b0 = b.Copy();

            var Acod = A0.Copy(); var bcod = b0.Copy();
            var xCod = arena.floatVec(n);
            int rank = QRCP.minNormSolveInPlace(ref Acod, ref bcod, ref xCod).rank;
            RecordEq(rank, r);
            if (Analysis.isAnyNan(in xCod)) { Fail0(16, 0); return; }

            // reconstruction: A x ≈ b (consistent -> residual ~ 0)
            float tol = (float)Consts.floatSqrtEps * (float)20;
            float res = ResidualNorm(in A0, in xCod, in b0);
            RecordBound(res, tol * ((float)1 + VecNorm(in b0)));

            // norm matches the SVD pseudoinverse
            var Apinv = A0.Copy();
            var xPinv = arena.floatVec(n);
            SVD.pinvSolve(ref Apinv, in b0, ref xPinv);
            AssertClose(VecNorm(in xCod), VecNorm(in xPinv), tol * (VecNorm(in xPinv) + (float)1));

            arena.Dispose();
        }

        // (17) The allocating overload and the explicit-scratch primitive must produce bit-identical
        // results on a rank-deficient system (same factor + COD path; only scratch ownership differs).
        void MinNormScratchEquivalence()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 9, n = 6;
            var A = arena.floatRandomMat(m, n, -3f, 3f, 0x2468u);
            for (int row = 0; row < m; row++)
                A[row, 5] = (float)2f * A[row, 0] - A[row, 1];  // rank 5
            var A0 = A.Copy();
            var b = arena.floatRandomVec(m, -3f, 3f, 0x1357u);
            var b0 = b.Copy();

            var A1 = A0.Copy(); var b1 = b0.Copy(); var x1 = arena.floatVec(n);
            RankInfo info1 = QRCP.minNormSolveInPlace(ref A1, ref b1, ref x1);   // allocating

            var A2 = A0.Copy(); var b2 = b0.Copy(); var x2 = arena.floatVec(n);
            var R = arena.floatMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var u = arena.floatVec(m);
            RankInfo info2 = QRCP.minNormSolveInPlace(ref A2, ref b2, ref x2, ref R, ref P, ref u);   // primitive

            RecordEq(info1.rank, info2.rank);
            for (int k = 0; k < n; k++)
                AssertBitIdentical(x1[k], x2[k]);

            P.Dispose();
            arena.Dispose();
        }

        // (18) Large n (m=180, n=80 >= 2*QRCP_BLOCK) rank-deficient: forces the BLOCKED fused factor
        // ahead of the COD completion (r=79 < 80). The min-norm solution must still match the SVD
        // pseudoinverse on norm and residual.
        void MinNormBlocked()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 180, n = 80;
            var A = arena.floatRandomMat(m, n, -2f, 2f, 0xB10Cu);
            for (int d = 0; d < n; d++) A[d, d] += (float)n;    // well-conditioned...
            for (int row = 0; row < m; row++)
                A[row, n - 1] = A[row, 0] + A[row, 1];           // ...except one exact dependency -> rank n-1
            var A0 = A.Copy();
            var b = arena.floatRandomVec(m, -2f, 2f, 0x50C0u);
            var b0 = b.Copy();

            var Acod = A0.Copy(); var bcod = b0.Copy();
            var xCod = arena.floatVec(n);
            int rank = QRCP.minNormSolveInPlace(ref Acod, ref bcod, ref xCod).rank;
            RecordEq(rank, n - 1);
            if (Analysis.isAnyNan(in xCod)) { Fail0(18, 0); return; }

            var Apinv = A0.Copy();
            var xPinv = arena.floatVec(n);
            SVD.pinvSolve(ref Apinv, in b0, ref xPinv);

            float tol = (float)Consts.floatSqrtEps * (float)40;
            AssertClose(VecNorm(in xCod), VecNorm(in xPinv), tol * (VecNorm(in xPinv) + (float)1));
            AssertClose(ResidualNorm(in A0, in xCod, in b0), ResidualNorm(in A0, in xPinv, in b0),
                        tol * (ResidualNorm(in A0, in xPinv, in b0) + (float)1));

            arena.Dispose();
        }

        // (19) Zero matrix: rank 0, x all zeros (degenerate COD path — never enters the LQ compress).
        void MinNormZeroMatrix()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 5, n = 3;
            var A = arena.floatMat(m, n);                       // zero-initialised
            var b = arena.floatRandomVec(m, -1f, 1f, 0x000Fu);
            var x = arena.floatVec(n);

            int rank = QRCP.minNormSolveInPlace(ref A, ref b, ref x).rank;
            RecordEq(rank, 0);
            for (int j = 0; j < n; j++)
                AssertBitIdentical(x[j], (float)0);

            arena.Dispose();
        }

        // (20) KNOWN-ANSWER (external ground truth, not our own SVD). Rank-1 A = [[1,0],[2,0]] has the
        // closed-form pseudoinverse A+ = [[1/5, 2/5],[0, 0]] (Wikipedia, "Moore–Penrose inverse":
        // denominators 5 = 1²+2²). So the min-norm least-squares solution of A x ≈ b is
        // x = A+ b = [ (b0 + 2 b1)/5 , 0 ]. For b = [1, 3] that is x = [7/5, 0] = [1.4, 0] exactly.
        void MinNormKnownRank1()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 2, n = 2;
            var A = arena.floatMat(m, n);
            A[0, 0] = (float)1; A[0, 1] = (float)0;
            A[1, 0] = (float)2; A[1, 1] = (float)0;
            var b = arena.floatVec(m);
            b[0] = (float)1; b[1] = (float)3;

            var x = arena.floatVec(n);
            int rank = QRCP.minNormSolveInPlace(ref A, ref b, ref x).rank;

            RecordEq(rank, 1);
            float tol = (float)Consts.floatSqrtEps * (float)20;
            AssertClose(x[0], (float)7 / (float)5, tol);   // 1.4
            AssertClose(x[1], (float)0, tol);

            arena.Dispose();
        }

        // (21) KNOWN-ANSWER, rank-2 TALL and INCONSISTENT. Matrix from the R MASS::ginv tutorial
        // (r-statistics.co) — A (4x3) with column 3 = column 1 + column 2, so rank 2. For b = [6,5,11,4]
        // the system is NOT consistent (that tutorial's quoted "residual 0" solution is for a different
        // setup); the true minimum-norm least-squares solution, derived independently from the rank-2
        // normal equations + the "x in row space" min-norm condition, is
        //   (p,q) = ([15 14;14 15]^{-1} [53;54]) = (39/29, 68/29),  x3 = (p+q)/3 = 107/87,
        //   x = [ 10/87, 97/87, 107/87 ] ≈ [0.114943, 1.114943, 1.229885].
        void MinNormKnownRank2TallInconsistent()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 4, n = 3;
            var A = arena.floatMat(m, n);
            A[0, 0] = (float)1; A[0, 1] = (float)2; A[0, 2] = (float)3;
            A[1, 0] = (float)2; A[1, 1] = (float)1; A[1, 2] = (float)3;
            A[2, 0] = (float)3; A[2, 1] = (float)3; A[2, 2] = (float)6;
            A[3, 0] = (float)1; A[3, 1] = (float)1; A[3, 2] = (float)2;
            var b = arena.floatVec(m);
            b[0] = (float)6; b[1] = (float)5; b[2] = (float)11; b[3] = (float)4;
            var A0 = A.Copy(); var b0 = b.Copy();

            var x = arena.floatVec(n);
            int rank = QRCP.minNormSolveInPlace(ref A, ref b, ref x).rank;

            RecordEq(rank, 2);
            float tol = (float)1E-4f;
            AssertClose(x[0], (float)10 / (float)87, tol);   // 0.114943
            AssertClose(x[1], (float)97 / (float)87, tol);   // 1.114943
            AssertClose(x[2], (float)107 / (float)87, tol);  // 1.229885

            // Cross-check it really is the LS optimum: residual equals the SVD pseudoinverse residual.
            var Apinv = A0.Copy();
            var xPinv = arena.floatVec(n);
            SVD.pinvSolve(ref Apinv, in b0, ref xPinv);
            AssertClose(ResidualNorm(in A0, in x, in b0), ResidualNorm(in A0, in xPinv, in b0),
                        (float)Consts.floatSqrtEps * (float)20);

            arena.Dispose();
        }

        // (22) Multi-RHS COD == per-column single-RHS COD. Since single-RHS COD is already validated
        // against SVD + literature vectors, this transitively validates the block path.
        void MinNormMultiRHSMatchesSingle()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 10, n = 6, k = 4;
            var A = arena.floatRandomMat(m, n, -3f, 3f, 0xB10Cu);
            for (int row = 0; row < m; row++)
            {
                A[row, 4] = A[row, 0] + A[row, 1];               // rank 4
                A[row, 5] = A[row, 2] - A[row, 3];
            }
            var A0 = A.Copy();
            var B = arena.floatRandomMat(m, k, -3f, 3f, 0x5013u);

            var Ablk = A0.Copy(); var Bblk = B.Copy();
            var Xblk = arena.floatMat(n, k);
            int rankBlk = QRCP.minNormSolveInPlace(ref Ablk, ref Bblk, ref Xblk).rank;
            RecordEq(rankBlk, 4);

            float tol = (float)Consts.floatSqrtEps * (float)50;
            for (int j = 0; j < k; j++)
            {
                var Aj = A0.Copy();
                var bj = arena.floatVec(m);
                for (int i = 0; i < m; i++) bj[i] = B[i, j];
                var xj = arena.floatVec(n);
                QRCP.minNormSolveInPlace(ref Aj, ref bj, ref xj);
                for (int i = 0; i < n; i++)
                    AssertClose(Xblk[i, j], xj[i], tol * (math.abs(xj[i]) + (float)1));
            }

            arena.Dispose();
        }

        // (23) Factor-reuse minNormDecompSolve (from a precomputed A·P=Q·R) == the fused block COD.
        void MinNormDecompSolveMatchesFused()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 9, n = 5, k = 3;
            var A = arena.floatRandomMat(m, n, -3f, 3f, 0x2468u);
            for (int row = 0; row < m; row++)
                A[row, 4] = (float)2f * A[row, 0] - A[row, 1];  // rank 4
            var A0 = A.Copy();
            var B = arena.floatRandomMat(m, k, -3f, 3f, 0x1357u);

            // fused block COD (destroys A + B)
            var Afu = A0.Copy(); var Bfu = B.Copy();
            var Xfu = arena.floatMat(n, k);
            QRCP.minNormSolveInPlace(ref Afu, ref Bfu, ref Xfu);

            // factor-reuse: decompose once (A preserved into Q), then minNormDecompSolve (B preserved)
            var Q = arena.floatMat(m, n);
            var R = arena.floatMat(n, n);
            var Pp = new Pivot(n, Allocator.Persistent);
            QRCP.decomp(in A0, ref Q, ref R, ref Pp);
            var Xre = arena.floatMat(n, k);
            var Bre = B.Copy();
            QRCP.minNormDecompSolve(ref Q, ref R, in Pp, ref Bre, ref Xre);

            float tol = (float)Consts.floatSqrtEps * (float)50;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < k; j++)
                    AssertClose(Xre[i, j], Xfu[i, j], tol * (math.abs(Xfu[i, j]) + (float)1));

            Pp.Dispose();
            arena.Dispose();
        }

        // (24) B = I -> X = A+ (the pseudoinverse itself). Verify the Penrose identity A A+ A == A.
        void MinNormPseudoinverseIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 8, n = 5;
            var A = arena.floatRandomMat(m, n, -3f, 3f, 0xF00Du);
            for (int row = 0; row < m; row++)
                A[row, 4] = A[row, 0] + A[row, 2];               // rank 4
            var A0 = A.Copy();

            // B = I_m ; X = A+ (n x m)
            var B = arena.floatMat(m, m);
            for (int i = 0; i < m; i++) B[i, i] = (float)1;
            var Apinvmat = arena.floatMat(n, m);
            int rank = QRCP.minNormSolveInPlace(ref A, ref B, ref Apinvmat).rank;
            RecordEq(rank, 4);

            // A A+ A == A (Penrose condition 1)
            var AAp = Blas.dot(A0, Apinvmat);        // m x m
            var AApA = Blas.dot(AAp, A0);            // m x n
            float tol = (float)Consts.floatSqrtEps * (float)50;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    AssertClose(AApA[i, j], A0[i, j], tol * (math.abs(A0[i, j]) + (float)1));

            arena.Dispose();
        }

        void AssertBitIdentical(float a, float b)
        {
            if (a != b && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = a - b;
            }
            Assert.IsTrue(a == b);
        }

        // ---- helpers ----

        // ‖A x − b‖2 using an UNMODIFIED copy of A (the live A may be consumed by a solver).
        float ResidualNorm(in floatMxN A, in floatN x, in floatN b)
        {
            var Ax = Blas.dot(A, x);
            float s = (float)0;
            for (int i = 0; i < b.N; i++)
            {
                float d = Ax[i] - b[i];
                s += d * d;
            }
            return math.sqrt(s);
        }

        float VecNorm(in floatN v)
        {
            float s = (float)0;
            for (int i = 0; i < v.N; i++)
                s += v[i] * v[i];
            return math.sqrt(s);
        }

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

        void RecordBound(float value, float limit)
        {
            if (!(value <= limit) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = value;
                Fail[2] = limit;
                Fail[3] = value - limit;
            }
            Assert.IsTrue(value <= limit);
        }

        void RecordEq(int got, int expected)
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

        void Fail0(float got, float expected)
        {
            if (Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = got;
                Fail[2] = expected;
                Fail[3] = got - expected;
            }
            // Mirror the existing job's Burst-safe NaN guard (literal-string throw, not Assert.Fail).
            throw new System.Exception("SolveTestJob: NaN detected in solution");
        }
    }

    public static Array GetSolveEnums()
    {
        return Enum.GetValues(typeof(SolveTestJob.TestType));
    }

    [TestCaseSource("GetSolveEnums")]
    public void ColumnPivotSolveTests(SolveTestJob.TestType type)
    {
        var fail = new NativeArray<float>(4, Allocator.TempJob);
        try
        {
            new SolveTestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (float)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (float)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    // Managed throw-tests: dimension validation runs on the main thread (not in a Burst job).

    [Test]
    public void QrcpSolveThrowsOnShortMatrix() // m < n is rejected
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.floatMat(2, 3);
        var b = arena.floatVec(2);
        var x = arena.floatVec(3);
        Assert.Catch<ArgumentException>(() => QRCP.solveInPlace(ref A, ref b, ref x));
        arena.Dispose();
    }

    [Test]
    public void QrcpSolveThrowsOnWrongBLength()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.floatMat(4, 3);
        var b = arena.floatVec(3); // should be 4
        var x = arena.floatVec(3);
        Assert.Catch<ArgumentException>(() => QRCP.solveInPlace(ref A, ref b, ref x));
        arena.Dispose();
    }

    [Test]
    public void QrcpSolveThrowsOnWrongXLength()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.floatMat(4, 3);
        var b = arena.floatVec(4);
        var x = arena.floatVec(2); // should be 3
        Assert.Catch<ArgumentException>(() => QRCP.solveInPlace(ref A, ref b, ref x));
        arena.Dispose();
    }
}
