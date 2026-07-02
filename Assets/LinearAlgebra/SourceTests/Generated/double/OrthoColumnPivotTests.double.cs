using System;

using LinearAlgebra;
using LinearAlgebra.Gallery;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Tests for column-pivoted (rank-revealing) QR: QR.qrDecompositionColumnPivot.
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
public class doubleOrthoColumnPivotTests
{
    // Burst-compile smoke test.
    [BurstCompile]
    public struct AssemblyTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            var Q = arena.doubleRandomMat(12, 6);
            var R = arena.doubleMat(6);
            var P = new Pivot(6, Allocator.Persistent);

            QR.qrDecompositionColumnPivot(ref Q, ref R, ref P);

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
            KahanGalleryReconstruct,
            SingleElement,
            AllZero,
            ZeroColumnMiddle,
            DuplicateColumns,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<double> Fail;

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
                    var Q = arena.doubleRandomMat(m, n, -3f, 3f, 7001 + t * 13);
                    var R = arena.doubleMat(n);
                    var A = Q.Copy();

                    QR.qrDecompositionColumnPivot(ref Q, ref R, ref P);

                    AssertQRCP(in A, in Q, in R, in P, (double)1E-4f);

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
                    var Q = arena.doubleRandomMat(dim, dim, -3f, 3f, 4220 + t * 7);
                    var R = arena.doubleMat(dim);
                    var A = Q.Copy();

                    QR.qrDecompositionColumnPivot(ref Q, ref R, ref P);

                    AssertQRCP(in A, in Q, in R, in P, (double)1E-4f);

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
            var Q = arena.doubleRandomMat(m, n, -1f, 1f, 31337);

            // Scale columns to distinct magnitudes; column 2 is unambiguously the largest.
            for (int r = 0; r < m; r++)
            {
                Q[r, 0] *= (double)1f;
                Q[r, 1] *= (double)4f;
                Q[r, 2] *= (double)12f;
                Q[r, 3] *= (double)2f;
            }

            var R = arena.doubleMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var A = Q.Copy();

            QR.qrDecompositionColumnPivot(ref Q, ref R, ref P);

            // Reconstruction must still hold...
            AssertQRCP(in A, in Q, in R, in P, (double)1E-4f);
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
            var Q = arena.doubleRandomMat(m, n, -1f, 1f, 90210);

            // col3 = 2*col0 - col1 ; col4 = col0 + col2  => exact rank 3.
            for (int r = 0; r < m; r++)
            {
                Q[r, 3] = (double)2f * Q[r, 0] - Q[r, 1];
                Q[r, 4] = Q[r, 0] + Q[r, 2];
            }

            var R = arena.doubleMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var A = Q.Copy();

            QR.qrDecompositionColumnPivot(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (double)1E-4f);

            // Count non-negligible diagonal entries relative to the leading one. The gap here is
            // enormous (rank-3 entries are O(|R00|); the 2 dependent columns reduce to the float
            // round-off of forming them, ~1e-6 relative), so 1e-4 relative cleanly separates them
            // and is robust across float/double and seeds.
            double lead = math.abs(R[0, 0]);
            double rankTol = (double)1E-4f * lead;
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
            double c = (double)0.2f;
            double s = math.sqrt((double)1f - c * c);

            var Q = arena.doubleMat(dim, dim); // zero-initialised
            double si = 1f;                    // s^i
            for (int i = 0; i < dim; i++)
            {
                Q[i, i] = si;                  // diagonal: s^i
                for (int j = i + 1; j < dim; j++)
                    Q[i, j] = -c * si;         // above diagonal: -c*s^i
                si *= s;
            }

            var R = arena.doubleMat(dim);
            var P = new Pivot(dim, Allocator.Persistent);
            var A = Q.Copy();

            QR.qrDecompositionColumnPivot(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (double)1E-4f);

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
            double theta = (double)1.2f; // c=cos, s=sin both comfortably away from 0

            var Q = arena.doubleKahan(dim, theta);
            var R = arena.doubleMat(dim);
            var P = new Pivot(dim, Allocator.Persistent);
            var A = Q.Copy();

            QR.qrDecompositionColumnPivot(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (double)1E-4f);

            P.Dispose();
            arena.Dispose();
        }

        // Degenerate 1x1 input: Q = [+/-1], R = [+/-a], P = [0]; reconstruction must hold.
        void SingleElement()
        {
            var arena = new Arena(Allocator.Persistent);

            var Q = arena.doubleMat(1, 1);
            Q[0, 0] = (double)5f;
            var R = arena.doubleMat(1);
            var P = new Pivot(1, Allocator.Persistent);
            var A = Q.Copy();

            QR.qrDecompositionColumnPivot(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (double)1E-4f);
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
            var Q = arena.doubleMat(m, n); // zero-initialised
            var R = arena.doubleMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var A = Q.Copy();

            QR.qrDecompositionColumnPivot(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (double)1E-4f);
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
            var Q = arena.doubleMat(m, n); // zero-initialised
            // column 0: large norm
            Q[0, 0] = (double)6f; Q[1, 0] = (double)6f; Q[2, 0] = (double)6f;
            // column 1: exact zero (left as 0)
            // column 2: medium norm (< column 0)
            Q[0, 2] = (double)2f; Q[3, 2] = (double)2f;

            var R = arena.doubleMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var A = Q.Copy();

            QR.qrDecompositionColumnPivot(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (double)1E-4f);
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
            var Q = arena.doubleRandomMat(m, n, -1f, 1f, 13579);
            for (int r = 0; r < m; r++)
                Q[r, 2] = Q[r, 0]; // column 2 == column 0

            var R = arena.doubleMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var A = Q.Copy();

            QR.qrDecompositionColumnPivot(ref Q, ref R, ref P);

            AssertQRCP(in A, in Q, in R, in P, (double)1E-4f);

            double lead = math.abs(R[0, 0]);
            double rankTol = (double)1E-4f * lead;
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
        void AssertQRCP(in doubleMxN A, in doubleMxN Q, in doubleMxN R, in Pivot P, double precision)
        {
            int m = A.M_Rows;
            int n = A.N_Cols;

            // Build A permuted by P: column j == original column P[j]. Copy() gives a same-shape
            // arena matrix; we overwrite every entry from the (untouched) original A.
            var Aperm = A.Copy();
            for (int r = 0; r < m; r++)
                for (int j = 0; j < n; j++)
                    Aperm[r, j] = A[r, P[j]];

            doubleMxN shouldBeZero = Aperm - Linear_OP.dot(Q, R);

            if (Analysis_OP.isAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            double zeroError = Analysis_OP.MaxZeroError(shouldBeZero);
            RecordBound(zeroError, precision);

            Assert.IsTrue(Analysis_OP.isZero(in shouldBeZero, precision));
            Assert.IsTrue(Analysis_OP.isUpperTriangular(R, precision));
            Assert.IsTrue(Analysis_OP.isOrthogonal(Q, precision));

            // |R[d,d]| non-increasing (guaranteed by greedy column pivoting). Allow a small
            // absolute slack relative to the leading magnitude for float rounding.
            double monoTol = precision * (math.abs(R[0, 0]) + (double)1f);
            for (int d = 0; d + 1 < n; d++)
            {
                double hi = math.abs(R[d, d]);
                double lo = math.abs(R[d + 1, d + 1]);
                if (!(hi + monoTol >= lo) && Fail[0] == (double)0)
                {
                    Fail[0] = (double)1;
                    Fail[1] = lo;        // got (next diagonal)
                    Fail[2] = hi;        // expected upper bound (this diagonal)
                    Fail[3] = lo - hi;   // excess
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
    }

    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void ColumnPivotTests(TestJob.TestType type)
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
        finally
        {
            fail.Dispose();
        }
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // SOLVER: QR.qrcpDirectSolve — QRCP-based rank-safe least-squares (BASIC / truncated
    // solution). Solves min‖A x − b‖ for a possibly rank-deficient A (m >= n). Returns the
    // detected numerical rank and the basic solution (≤ rank nonzeros in permuted order):
    // minimal RESIDUAL but NOT minimum norm. At full column rank it reduces to ordinary QR-LS.
    //
    // Cross-checks (where used): SVD.pinvSolve gives the SAME residual (also residual-minimal)
    // but the MINIMUM-NORM solution, so ‖x_pinv‖ <= ‖x_qrcp‖ — this pins the basic-vs-min-norm
    // distinction. All four overloads are exercised across the cases below.
    // ────────────────────────────────────────────────────────────────────────────────
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct SolveTestJob : IJob
    {
        public enum TestType
        {
            FullRankAgreesWithQR,   // (1) overdetermined full-rank: rank==n & x == qrDirectSolve
            FullRankSquare,         // (2) square full-rank: A x == b
            RankDeficientResidual,  // (3) dependent column: rank, minimal residual, pinv cross-check
            Rank1Projection,        // (4) Pei(4,0) rank-1, n>1: residual minimal, A x == proj(b)
            OverdeterminedDeficient,// (5) m > n, r < n (two dependencies)
            ZeroMatrix,             // (6) zero matrix: rank 0, x all zeros
            OneByOne,               // (7) 1x1 system: x == b/a (projection formula)
            AutoSentinel,           // (8) relTol=-1 == default overload == explicit default tol
            KnownValueRegression,   // (9) hand-computable rank-deficient basic solution
            RankRevealingInfoStatus,// (10) Stage-3: RankRevealingInfo.status/rank/Solved on rank-deficient A
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<double> Fail;

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
                case TestType.RankRevealingInfoStatus: RankRevealingInfoStatus(); break;
            }
        }

        // (1) Overdetermined, full column rank, well-conditioned (diag-boosted random). The basic
        // solution must coincide with ordinary (un-pivoted) QR least-squares to tolerance, and the
        // detected rank must be the full n. Uses the ALLOCATING default overload.
        void FullRankAgreesWithQR()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 12, n = 4;
            var A = arena.doubleRandomMat(m, n, -5f, 5f, 778231);
            for (int d = 0; d < n; d++)
                A[d, d] += (double)10f; // boost leading block -> full column rank, good conditioning

            // generic b (not in range(A)) so it is a genuine least-squares (not exact) problem
            var b = arena.doubleRandomVec(m, -5f, 5f, 9091);

            var x = arena.doubleVec(n);
            int rank = QR.qrcpDirectSolve(ref A, ref b, ref x).rank; // qrcp leaves A,b intact

            RecordEq(rank, n);
            if (Analysis_OP.isAnyNan(in x)) { Fail0(0, 0); return; }

            // reference: ordinary QR-LS (destroys its inputs -> feed copies)
            var Aqr = A.Copy();
            var bqr = b.Copy();
            var xRef = arena.doubleVec(n);
            QR.qrDirectSolve(ref Aqr, ref bqr, ref xRef);

            double tol = (double)Consts.doubleSqrtEps * (double)10;
            for (int k = 0; k < n; k++)
                AssertClose(x[k], xRef[k], tol * (math.abs(xRef[k]) + (double)1));

            arena.Dispose();
        }

        // (2) Square full-rank: the basic solution solves A x = b exactly (residual ~ 0).
        // Uses the PRIMITIVE default-tolerance overload (Q/R/P/u scratch).
        void FullRankSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;
            var A = arena.doubleRandomMat(dim, dim, -5f, 5f, 314221);
            for (int d = 0; d < dim; d++)
                A[d, d] += (double)10f;

            var xOrig = arena.doubleRandomVec(dim, -3f, 3f, 1337);
            var b = Linear_OP.dot(A, xOrig); // b in range(A) -> exact solution exists
            var A_copy = A.Copy();          // for residual check after the solve

            var Q = arena.doubleMat(dim, dim);
            var R = arena.doubleMat(dim);
            var P = new Pivot(dim, Allocator.Persistent);
            var u = arena.doubleVec(dim);
            var x = arena.doubleVec(dim);

            int rank = QR.qrcpDirectSolve(ref A, ref b, ref x, ref Q, ref R, ref P, ref u).rank;

            RecordEq(rank, dim);
            if (!Analysis_OP.isAnyNan(in x))
            {
                double tol = (double)Consts.doubleSqrtEps * (double)10;
                for (int k = 0; k < dim; k++)
                    AssertClose(x[k], xOrig[k], tol * (math.abs(xOrig[k]) + (double)1));

                // residual ~ 0
                double res = ResidualNorm(in A_copy, in x, in b);
                RecordBound(res, tol * ((double)1 + VecNorm(in b)));
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
            var A = arena.doubleRandomMat(m, n, -3f, 3f, 90211);
            for (int r = 0; r < m; r++)
                A[r, 3] = A[r, 0] + A[r, 1]; // exact dependency -> true rank 3
            var A_copy = A.Copy();

            var b = arena.doubleRandomVec(m, -3f, 3f, 5511);

            var x = arena.doubleVec(n);
            int rank = QR.qrcpDirectSolve(ref A, ref b, ref x).rank;

            RecordEq(rank, 3);
            if (Analysis_OP.isAnyNan(in x)) { Fail0(1, 0); return; }

            double resQrcp = ResidualNorm(in A_copy, in x, in b);
            double normQrcp = VecNorm(in x);

            // pinv reference (no longer modifies A) — same residual, minimum norm
            var Apinv = A_copy.Copy();
            var xPinv = arena.doubleVec(n);
            int pinvRank = SVD.pinvSolve(ref Apinv, in b, ref xPinv, out bool converged);

            RecordEq(pinvRank, 3);
            double resPinv = ResidualNorm(in A_copy, in xPinv, in b);
            double normPinv = VecNorm(in xPinv);

            // (a) SAME residual (both are residual-minimal). Residual is second-order flat at the
            // optimum, so even pinv's iterative x reproduces the minimum value tightly.
            double resTol = (double)Consts.doubleSqrtEps * (double)4 * (resPinv + (double)1);
            AssertClose(resQrcp, resPinv, resTol);

            // (b) basic solution is NOT minimum-norm: ‖x_pinv‖ <= ‖x_qrcp‖ (with slack).
            double normSlack = (double)Consts.doubleSqrtEps * (double)10 * (normQrcp + (double)1);
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
            var A = arena.doublePei(dim, (double)0); // all-ones, rank 1
            var A_copy = A.Copy();

            var b = arena.doubleRandomVec(dim, -4f, 4f, 24680);
            double mean = (double)0;
            for (int i = 0; i < dim; i++) mean += b[i];
            mean /= (double)dim;

            var x = arena.doubleVec(dim);
            int rank = QR.qrcpDirectSolve(ref A, ref b, ref x).rank;

            RecordEq(rank, 1);
            if (Analysis_OP.isAnyNan(in x)) { Fail0(1, 0); return; }

            // reconstruction A x must be the projection of b onto span(ones) = mean(b)*ones
            var Ax = Linear_OP.dot(A_copy, x);
            double tol = (double)Consts.doubleSqrtEps * (double)10;
            for (int i = 0; i < dim; i++)
                AssertClose(Ax[i], mean, tol * (math.abs(mean) + (double)1));

            // residual minimal vs pinv, and basic norm >= min norm
            double resQrcp = ResidualNorm(in A_copy, in x, in b);
            double normQrcp = VecNorm(in x);

            var Apinv = A_copy.Copy();
            var xPinv = arena.doubleVec(dim);
            int pinvRank = SVD.pinvSolve(ref Apinv, in b, ref xPinv, out bool converged);
            RecordEq(pinvRank, 1);
            double resPinv = ResidualNorm(in A_copy, in xPinv, in b);
            double normPinv = VecNorm(in xPinv);

            AssertClose(resQrcp, resPinv, (double)Consts.doubleSqrtEps * (double)4 * (resPinv + (double)1));
            RecordBound(normPinv - normQrcp, (double)Consts.doubleSqrtEps * (double)10 * (normQrcp + (double)1));

            arena.Dispose();
        }

        // (5) Overdetermined (m > n) AND rank-deficient (r < n) via two exact dependencies
        // (col3 = 2*col0 - col1 ; col4 = col0 + col2) => rank 3 of 5. Residual minimal (pinv check).
        // Uses the PRIMITIVE with an explicit positive tolerance (= the library default).
        void OverdeterminedDeficient()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 8, n = 5;
            var A = arena.doubleRandomMat(m, n, -2f, 2f, 90210);
            for (int r = 0; r < m; r++)
            {
                A[r, 3] = (double)2f * A[r, 0] - A[r, 1];
                A[r, 4] = A[r, 0] + A[r, 2];
            }
            var A_copy = A.Copy();

            var b = arena.doubleRandomVec(m, -2f, 2f, 1212);

            var Q = arena.doubleMat(m, n);
            var R = arena.doubleMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var u = arena.doubleVec(m);
            var x = arena.doubleVec(n);

            double explicitTol = (double)(math.max(m, n)) * (double)Consts.doubleZeroThreshold;
            int rank = QR.qrcpDirectSolve(ref A, ref b, ref x, ref Q, ref R, ref P, ref u, explicitTol).rank;

            RecordEq(rank, 3);
            if (Analysis_OP.isAnyNan(in x)) { Fail0(1, 0); return; }

            double resQrcp = ResidualNorm(in A_copy, in x, in b);

            var Apinv = A_copy.Copy();
            var xPinv = arena.doubleVec(n);
            int pinvRank = SVD.pinvSolve(ref Apinv, in b, ref xPinv, out bool converged);
            RecordEq(pinvRank, 3);
            double resPinv = ResidualNorm(in A_copy, in xPinv, in b);

            AssertClose(resQrcp, resPinv, (double)Consts.doubleSqrtEps * (double)4 * (resPinv + (double)1));

            // basic norm >= min norm
            RecordBound(VecNorm(in xPinv) - VecNorm(in x),
                        (double)Consts.doubleSqrtEps * (double)10 * (VecNorm(in x) + (double)1));

            P.Dispose();
            arena.Dispose();
        }

        // (6) Zero matrix (m=5, n=3): no column has any norm -> rank 0 and x is all zeros (no NaN).
        void ZeroMatrix()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 5, n = 3;
            var A = arena.doubleMat(m, n);                          // zero-initialised
            var b = arena.doubleRandomVec(m, -5f, 5f, 5151);

            var x = arena.doubleVec(n);
            int rank = QR.qrcpDirectSolve(ref A, ref b, ref x).rank;

            RecordEq(rank, 0);
            if (Analysis_OP.isAnyNan(in x)) { Fail0(1, 0); return; }
            for (int k = 0; k < n; k++)
                AssertClose(x[k], (double)0, (double)Consts.doubleSqrtEps);

            arena.Dispose();
        }

        // (7) 1x1 system A=[a], b=[β]: the only column has full rank, so x[0] = (a·β)/(a·a) = β/a
        // (the projection formula). Pick a=4, β=10 -> x=2.5, residual 0, rank 1.
        void OneByOne()
        {
            var arena = new Arena(Allocator.Persistent);

            var A = arena.doubleMat(1, 1);
            A[0, 0] = (double)4f;
            var A_copy = A.Copy();

            var b = arena.doubleVec(1);
            b[0] = (double)10f;

            var x = arena.doubleVec(1);
            int rank = QR.qrcpDirectSolve(ref A, ref b, ref x).rank;

            RecordEq(rank, 1);
            if (Analysis_OP.isAnyNan(in x)) { Fail0(1, 0); return; }

            AssertClose(x[0], (double)2.5f, (double)Consts.doubleSqrtEps * (double)10);
            RecordBound(ResidualNorm(in A_copy, in x, in b), (double)Consts.doubleSqrtEps * (double)10);

            arena.Dispose();
        }

        // (8) Auto sentinel: relTol = -1 must select the documented default
        // (max(m,n)*Consts.doubleZeroThreshold). Verify it produces the SAME rank and the SAME x as
        // (a) the default overload and (b) the explicit positive default tolerance — bit-for-bit
        // (identical code path). Exercised on a rank-deficient system so rank/truncation matter.
        void AutoSentinel()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 6, n = 4;
            var A = arena.doubleRandomMat(m, n, -3f, 3f, 4242);
            for (int r = 0; r < m; r++)
                A[r, 2] = A[r, 0] - A[r, 1]; // rank 3
            var b = arena.doubleRandomVec(m, -3f, 3f, 2424);

            var xAuto = arena.doubleVec(n);
            int rankAuto = QR.qrcpDirectSolve(ref A, ref b, ref xAuto).rank; // default overload

            var xNeg = arena.doubleVec(n);
            int rankNeg = QR.qrcpDirectSolve(ref A, ref b, ref xNeg, (double)(-1)).rank; // sentinel

            double explicitTol = (double)(math.max(m, n)) * (double)Consts.doubleZeroThreshold;
            var xExpl = arena.doubleVec(n);
            int rankExpl = QR.qrcpDirectSolve(ref A, ref b, ref xExpl, explicitTol).rank;

            RecordEq(rankNeg, rankAuto);
            RecordEq(rankExpl, rankAuto);

            for (int k = 0; k < n; k++)
            {
                // identical computation -> exact equality
                AssertClose(xNeg[k], xAuto[k], (double)0);
                AssertClose(xExpl[k], xAuto[k], (double)0);
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
            var A = arena.doubleMat(m, n);  // zero-initialised
            A[0, 0] = (double)1f; A[0, 1] = (double)2f; // only row 0 is nonzero
            var A_copy = A.Copy();

            var b = arena.doubleVec(m);
            b[0] = (double)6f; b[1] = (double)1f; b[2] = (double)1f;

            var Q = arena.doubleMat(m, n);
            var R = arena.doubleMat(n);
            var P = new Pivot(n, Allocator.Persistent);
            var u = arena.doubleVec(m);
            var x = arena.doubleVec(n);

            int rank = QR.qrcpDirectSolve(ref A, ref b, ref x, ref Q, ref R, ref P, ref u, (double)(-1)).rank;

            RecordEq(rank, 1);
            if (Analysis_OP.isAnyNan(in x)) { Fail0(1, 0); return; }

            double tol = (double)Consts.doubleSqrtEps * (double)10;
            AssertClose(x[0], (double)0f, tol); // free variable (original col0) zeroed
            AssertClose(x[1], (double)3f, tol); // pivoted col1 carries the rank-1 solution

            double res = ResidualNorm(in A_copy, in x, in b);
            AssertClose(res, math.sqrt((double)2f), tol);

            P.Dispose();
            arena.Dispose();
        }

        // (10) Stage-3 direct-solve-status coverage: on a rank-deficient A (exact linear
        // dependency, true rank 3 of 4), qrcpDirectSolve must return a RankRevealingInfo with
        // status == RankDeficient, rank == the detected reduced rank, and Solved == true (a
        // rank-deficient basic solution is still usable) -- distinct from a hard failure.
        void RankRevealingInfoStatus()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 6, n = 4;
            var A = arena.doubleRandomMat(m, n, -3f, 3f, 90211);
            for (int r = 0; r < m; r++)
                A[r, 3] = A[r, 0] + A[r, 1]; // exact dependency -> true rank 3

            var b = arena.doubleRandomVec(m, -3f, 3f, 5511);
            var x = arena.doubleVec(n);

            RankRevealingInfo info = QR.qrcpDirectSolve(ref A, ref b, ref x);

            RecordEq(info.rank, 3);
            RecordEq((int)info.status, (int)DirectSolveStatus.RankDeficient);
            RecordEq(info.Solved ? 1 : 0, 1);
            RecordEq(info ? 1 : 0, 1); // implicit bool must also be true

            arena.Dispose();
        }

        // ---- helpers ----

        // ‖A x − b‖2 using an UNMODIFIED copy of A (the live A may be consumed by a solver).
        double ResidualNorm(in doubleMxN A, in doubleN x, in doubleN b)
        {
            var Ax = Linear_OP.dot(A, x);
            double s = (double)0;
            for (int i = 0; i < b.N; i++)
            {
                double d = Ax[i] - b[i];
                s += d * d;
            }
            return math.sqrt(s);
        }

        double VecNorm(in doubleN v)
        {
            double s = (double)0;
            for (int i = 0; i < v.N; i++)
                s += v[i] * v[i];
            return math.sqrt(s);
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff
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

        void Fail0(double got, double expected)
        {
            if (Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
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
        var A = arena.doubleMat(2, 3);
        var b = arena.doubleVec(2);
        var x = arena.doubleVec(3);
        Assert.Catch<ArgumentException>(() => QR.qrcpDirectSolve(ref A, ref b, ref x));
        arena.Dispose();
    }

    [Test]
    public void QrcpSolveThrowsOnWrongBLength()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.doubleMat(4, 3);
        var b = arena.doubleVec(3); // should be 4
        var x = arena.doubleVec(3);
        Assert.Catch<ArgumentException>(() => QR.qrcpDirectSolve(ref A, ref b, ref x));
        arena.Dispose();
    }

    [Test]
    public void QrcpSolveThrowsOnWrongXLength()
    {
        var arena = new Arena(Allocator.Persistent);
        var A = arena.doubleMat(4, 3);
        var b = arena.doubleVec(4);
        var x = arena.doubleVec(2); // should be 3
        Assert.Catch<ArgumentException>(() => QR.qrcpDirectSolve(ref A, ref b, ref x));
        arena.Dispose();
    }
}
