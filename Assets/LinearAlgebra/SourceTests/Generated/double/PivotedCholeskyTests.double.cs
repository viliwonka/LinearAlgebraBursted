using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Tests for pivoted (rank-revealing) Cholesky: Cholesky.choleskyDecompositionPivot / choleskyPivotSolve.
// PᵀAP = LLᵀ with symmetric (diagonal) pivoting, largest remaining diagonal first (LAPACK xPSTRF).
//
// Properties / vectors exercised (see each test method for the exact numbers): full-rank SPD
// reconstruction + solve; diagonal PSD rank-reveal; rank-deficient PSD Gram-matrix reconstruction +
// min-norm solve; min-norm certificate ‖x‖ ≤ ‖xOrig‖; indefinite matrices rejected; zero matrix
// (rank 0) and rank-1 outer product (rank 1).
public class doublePivotedCholeskyTests
{
    [BurstCompile]
    public struct AssemblyTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            var B = arena.doubleRandomMat(6, 4);
            var A = Gram(in arena, in B);
            var L = arena.doubleMat(6);
            var P = new Pivot(6, Allocator.Persistent);

            Cholesky.choleskyDecompositionPivot(in A, ref L, ref P);

            P.Dispose();
            arena.Dispose();
        }

        static doubleMxN Gram(in Arena arena, in doubleMxN B)
        {
            int n = B.M_Rows, r = B.N_Cols;
            var A = arena.doubleMat(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    double s = 0;
                    for (int k = 0; k < r; k++)
                        s += B[i, k] * B[j, k];
                    A[i, j] = s;
                }
            return A;
        }
    }

    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            FullRankSPD,
            DiagonalRankReveal,
            RankDeficientReconstructAndSolve,
            MinNormCertificate,
            Indefinite2x2,
            Indefinite3x3,
            IndefiniteStatus,
            ZeroDiagonalIndefinite,
            ZeroMatrix,
            Rank1Outer,
            SingleElement,
            OutOfRangeLeastSquares,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<double> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.FullRankSPD:                      FullRankSPD();                      break;
                case TestType.DiagonalRankReveal:               DiagonalRankReveal();               break;
                case TestType.RankDeficientReconstructAndSolve: RankDeficientReconstructAndSolve(); break;
                case TestType.MinNormCertificate:               MinNormCertificate();               break;
                case TestType.Indefinite2x2:                    Indefinite2x2();                    break;
                case TestType.Indefinite3x3:                    Indefinite3x3();                    break;
                case TestType.IndefiniteStatus:                 IndefiniteStatus();                 break;
                case TestType.ZeroDiagonalIndefinite:           ZeroDiagonalIndefinite();           break;
                case TestType.ZeroMatrix:                       ZeroMatrix();                       break;
                case TestType.Rank1Outer:                       Rank1Outer();                       break;
                case TestType.SingleElement:                    SingleElement();                    break;
                case TestType.OutOfRangeLeastSquares:           OutOfRangeLeastSquares();           break;
            }
        }

        void FullRankSPD()
        {
            var arena = new Arena(Allocator.Persistent);

            for (uint t = 0; t < 12; t++)
            {
                int n = 8;
                var B = arena.doubleRandomMat(n, n, -1f, 1f, 6100 + t * 13);
                var A = Gram(in arena, in B);
                for (int d = 0; d < n; d++) A[d, d] += (double)n; // diagonal boost: well-conditioned SPD

                var L = arena.doubleMat(n);
                var P = new Pivot(n, Allocator.Persistent);

                var pivInfo = Cholesky.choleskyDecompositionPivot(in A, ref L, ref P);
                bool ok = pivInfo; int rank = pivInfo.rank;
                RecordEq(ok ? 1 : 0, 1);
                RecordEq(rank, n);
                AssertReconstruct(in A, in L, in P, rank, (double)1E-4f);

                // first pivot must be the largest diagonal of A.
                int argmaxDiag = 0; double best = A[0, 0];
                for (int j = 1; j < n; j++) if (A[j, j] > best) { best = A[j, j]; argmaxDiag = j; }
                RecordEq(P[0], argmaxDiag);

                // exact solve: b = A xOrig => x == xOrig.
                var xOrig = arena.doubleRandomVec(n, -3f, 3f, 71000 + t * 7);
                var b = Linear_OP.dot(A, xOrig);
                var Lc = arena.doubleMat(n);
                var Pc = new Pivot(n, Allocator.Persistent);
                Cholesky.choleskyPivotSolve(in A, ref Lc, ref Pc, ref b); // b <- x
                for (int i = 0; i < n; i++) b[i] -= xOrig[i];
                RecordBound(doubleNorms_OP.L2(in b), (double)1E-3f);

                Pc.Dispose();
                P.Dispose();
                arena.Clear();
            }

            arena.Dispose();
        }

        // diag(4,0,9,0,1): pivot picks 9,4,1 (indices 2,0,4); numerical rank 3.
        void DiagonalRankReveal()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            var A = arena.doubleMat(n, n);
            A[0, 0] = 4f; A[2, 2] = 9f; A[4, 4] = 1f; // others zero

            var L = arena.doubleMat(n);
            var P = new Pivot(n, Allocator.Persistent);

            var pivInfo = Cholesky.choleskyDecompositionPivot(in A, ref L, ref P);
            bool ok = pivInfo; int rank = pivInfo.rank;
            RecordEq(ok ? 1 : 0, 1);
            RecordEq(rank, 3);
            RecordEq(P[0], 2);
            RecordEq(P[1], 0);
            RecordEq(P[2], 4);
            AssertReconstruct(in A, in L, in P, rank, (double)1E-5f);

            P.Dispose();
            arena.Dispose();
        }

        // A = B Bᵀ with B (n x r), r < n => PSD of exact rank r. Reconstruction + exact min-norm
        // recovery when b ∈ range(A) (take xRange = A·w, then b = A·xRange => x == xRange).
        void RankDeficientReconstructAndSolve()
        {
            var arena = new Arena(Allocator.Persistent);

            for (uint t = 0; t < 12; t++)
            {
                int n = 7, r = 4;
                var B = arena.doubleRandomMat(n, r, -1f, 1f, 8200 + t * 11);
                var A = Gram(in arena, in B);

                var L = arena.doubleMat(n);
                var P = new Pivot(n, Allocator.Persistent);

                var pivInfo = Cholesky.choleskyDecompositionPivot(in A, ref L, ref P);
                bool ok = pivInfo; int rank = pivInfo.rank;
                RecordEq(ok ? 1 : 0, 1);
                RecordEq(rank, r);
                AssertReconstruct(in A, in L, in P, rank, (double)1E-4f);

                // xRange = A·w ∈ range(A); b = A·xRange => min-norm solution == xRange.
                var w = arena.doubleRandomVec(n, -2f, 2f, 51000 + t * 5);
                var xRange = Linear_OP.dot(A, w);
                var b = Linear_OP.dot(A, xRange);

                var Ls = arena.doubleMat(n);
                var Ps = new Pivot(n, Allocator.Persistent);
                Cholesky.choleskyPivotSolve(in A, ref Ls, ref Ps, ref b); // b <- x

                // A·x ≈ A·xRange (consistency) and x ≈ xRange (exact recovery, scaled by ‖xRange‖).
                double scale = doubleNorms_OP.L2(in xRange) + (double)1f;
                var diff = arena.doubleVec(n);
                for (int i = 0; i < n; i++) diff[i] = b[i] - xRange[i];
                RecordBound(doubleNorms_OP.L2(in diff) / scale, (double)1E-2f);

                Ps.Dispose();
                P.Dispose();
                arena.Clear();
            }

            arena.Dispose();
        }

        // For b = A·xOrig (xOrig arbitrary => b ∈ range(A)), the min-norm solution x satisfies
        // A·x ≈ b and ‖x‖ ≤ ‖xOrig‖ (xOrig is *a* solution, so the minimum-norm one is no larger).
        void MinNormCertificate()
        {
            var arena = new Arena(Allocator.Persistent);

            for (uint t = 0; t < 12; t++)
            {
                int n = 7, r = 3;
                var B = arena.doubleRandomMat(n, r, -1f, 1f, 3300 + t * 17);
                var A = Gram(in arena, in B);

                var xOrig = arena.doubleRandomVec(n, -2f, 2f, 42000 + t * 9);
                var b = Linear_OP.dot(A, xOrig);     // b ∈ range(A)
                var bForResidual = b.Copy();

                var L = arena.doubleMat(n);
                var P = new Pivot(n, Allocator.Persistent);
                Cholesky.choleskyPivotSolve(in A, ref L, ref P, ref b); // b <- x

                // consistency: A·x ≈ bForResidual.
                var Ax = Linear_OP.dot(A, b);
                var resid = arena.doubleVec(n);
                for (int i = 0; i < n; i++) resid[i] = Ax[i] - bForResidual[i];
                double bScale = doubleNorms_OP.L2(in bForResidual) + (double)1f;
                RecordBound(doubleNorms_OP.L2(in resid) / bScale, (double)1E-2f);

                // minimum-norm: ‖x‖ ≤ ‖xOrig‖ (+ tiny slack).
                double xNorm = doubleNorms_OP.L2(in b);
                double origNorm = doubleNorms_OP.L2(in xOrig);
                if (!(xNorm <= origNorm + (double)1E-3f * (origNorm + (double)1f)) && Fail[0] == (double)0)
                {
                    Fail[0] = (double)1;
                    Fail[1] = xNorm;
                    Fail[2] = origNorm;
                    Fail[3] = xNorm - origNorm;
                }

                P.Dispose();
                arena.Clear();
            }

            arena.Dispose();
        }

        void Indefinite2x2()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 2;
            var A = arena.doubleMat(n, n);
            A[0, 0] = 1f; A[0, 1] = 2f; A[1, 0] = 2f; A[1, 1] = 1f; // eigenvalues 3, -1

            var L = arena.doubleMat(n);
            var P = new Pivot(n, Allocator.Persistent);

            bool ok = Cholesky.choleskyDecompositionPivot(in A, ref L, ref P);
            RecordEq(ok ? 1 : 0, 0); // indefinite => false

            P.Dispose();
            arena.Dispose();
        }

        void Indefinite3x3()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 3;
            var A = arena.doubleMat(n, n);
            // PSD-looking 2x2 block plus a negative eigenvalue contribution.
            A[0, 0] = 2f; A[1, 1] = 2f; A[2, 2] = 2f;
            A[0, 1] = 0f; A[1, 0] = 0f;
            A[0, 2] = 3f; A[2, 0] = 3f; // big off-diagonal => negative eigenvalue
            A[1, 2] = 0f; A[2, 1] = 0f;

            var L = arena.doubleMat(n);
            var P = new Pivot(n, Allocator.Persistent);

            bool ok = Cholesky.choleskyDecompositionPivot(in A, ref L, ref P);
            RecordEq(ok ? 1 : 0, 0);

            P.Dispose();
            arena.Dispose();
        }

        // Stage-3 direct-solve-status coverage: an indefinite matrix must report
        // DirectSolveStatus.Indefinite (not just a falsy implicit-bool) from
        // choleskyDecompositionPivot, and RankRevealingInfo.Solved must be false.
        void IndefiniteStatus()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 2;
            var A = arena.doubleMat(n, n);
            A[0, 0] = 1f; A[0, 1] = 2f; A[1, 0] = 2f; A[1, 1] = 1f; // eigenvalues 3, -1

            var L = arena.doubleMat(n);
            var P = new Pivot(n, Allocator.Persistent);

            RankRevealingInfo info = Cholesky.choleskyDecompositionPivot(in A, ref L, ref P);
            RecordEq((int)info.status, (int)DirectSolveStatus.Indefinite);
            RecordEq(info.Solved ? 1 : 0, 0);
            RecordEq(info ? 1 : 0, 0);

            P.Dispose();
            arena.Dispose();
        }

        void ZeroMatrix()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;
            var A = arena.doubleMat(n, n); // zero
            var L = arena.doubleMat(n);
            var P = new Pivot(n, Allocator.Persistent);

            var pivInfo = Cholesky.choleskyDecompositionPivot(in A, ref L, ref P);
            bool ok = pivInfo; int rank = pivInfo.rank;
            RecordEq(ok ? 1 : 0, 1);
            RecordEq(rank, 0);

            // solve: x = 0 for any b.
            var b = arena.doubleRandomVec(n, -1f, 1f, 999);
            Cholesky.choleskyPivotSolve(ref L, in P, rank, ref b);
            RecordBound(doubleNorms_OP.L2(in b), (double)1E-6f);

            P.Dispose();
            arena.Dispose();
        }

        void Rank1Outer()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;
            var v = arena.doubleVec(n);
            v[0] = 1f; v[1] = 2f; v[2] = 3f; v[3] = 0.5f;

            var A = arena.doubleMat(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = v[i] * v[j]; // rank-1 PSD

            var L = arena.doubleMat(n);
            var P = new Pivot(n, Allocator.Persistent);

            var pivInfo = Cholesky.choleskyDecompositionPivot(in A, ref L, ref P);
            bool ok = pivInfo; int rank = pivInfo.rank;
            RecordEq(ok ? 1 : 0, 1);
            RecordEq(rank, 1);
            AssertReconstruct(in A, in L, in P, rank, (double)1E-4f);

            P.Dispose();
            arena.Dispose();
        }

        // Zero diagonal but nonzero off-diagonal => genuinely indefinite (eigenvalues +/-1), must NOT
        // be silently accepted as a rank-0 PSD matrix. Regression for the all-zero-diagonal absScale
        // hole: with the all-entries scale + trailing-block residual check, this returns false.
        void ZeroDiagonalIndefinite()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 2;
            var A = arena.doubleMat(n, n);
            A[0, 1] = 1f; A[1, 0] = 1f; // diagonal stays zero

            var L = arena.doubleMat(n);
            var P = new Pivot(n, Allocator.Persistent);

            bool ok = Cholesky.choleskyDecompositionPivot(in A, ref L, ref P);
            RecordEq(ok ? 1 : 0, 0); // indefinite => false

            P.Dispose();
            arena.Dispose();
        }

        // 1x1 SPD: A=[[4]], b=[8] => x=[2]; rank 1, reconstruction trivial.
        void SingleElement()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 1;
            var A = arena.doubleMat(n, n);
            A[0, 0] = 4f;

            var L = arena.doubleMat(n);
            var P = new Pivot(n, Allocator.Persistent);

            var pivInfo = Cholesky.choleskyDecompositionPivot(in A, ref L, ref P);
            bool ok = pivInfo; int rank = pivInfo.rank;
            RecordEq(ok ? 1 : 0, 1);
            RecordEq(rank, 1);
            AssertReconstruct(in A, in L, in P, rank, (double)1E-5f);

            var b = arena.doubleVec(n);
            b[0] = 8f;
            Cholesky.choleskyPivotSolve(ref L, in P, rank, ref b);
            RecordBound(math.abs(b[0] - (double)2f), (double)1E-5f);

            P.Dispose();
            arena.Dispose();
        }

        // Genuine least-squares: b has a component OUTSIDE range(A). The min-norm pseudoinverse must
        // return x with the residual (A·x − b) orthogonal to range(A) — i.e. A·x is the projection of
        // b onto range(A), so A·(A·x) == A·b (a normal-equations identity that holds for x = A⁺b but
        // NOT for an arbitrary solve). This is the only case where the (MᵀM)⁻² double-inverse matters.
        void OutOfRangeLeastSquares()
        {
            var arena = new Arena(Allocator.Persistent);

            for (uint t = 0; t < 12; t++)
            {
                int n = 7, r = 3;
                var B = arena.doubleRandomMat(n, r, -1f, 1f, 2600 + t * 19);
                var A = Gram(in arena, in B);

                // arbitrary b, generically NOT in range(A) since rank r < n.
                var b = arena.doubleRandomVec(n, -2f, 2f, 77000 + t * 13);

                var L = arena.doubleMat(n);
                var P = new Pivot(n, Allocator.Persistent);
                bool ok = Cholesky.choleskyPivotSolve(in A, ref L, ref P, ref b); // b <- x
                RecordEq(ok ? 1 : 0, 1);

                // normal equations: A(Ax) == A b  <=>  A(Ax - b) == 0  (residual ⟂ range(A)).
                var Ax = Linear_OP.dot(A, b);
                var AAx = Linear_OP.dot(A, Ax);
                // recompute A b — b now holds x, so rebuild b from the same seed.
                var bOrig = arena.doubleRandomVec(n, -2f, 2f, 77000 + t * 13);
                var Ab = Linear_OP.dot(A, bOrig);

                double scale = doubleNorms_OP.L2(in Ab) + (double)1f;
                var diff = arena.doubleVec(n);
                for (int i = 0; i < n; i++) diff[i] = AAx[i] - Ab[i];
                RecordBound(doubleNorms_OP.L2(in diff) / scale, (double)1E-2f);

                P.Dispose();
                arena.Clear();
            }

            arena.Dispose();
        }

        // PᵀAP == LLᵀ (computed directly as Σ_k L[i,k]L[j,k]), L lower-triangular, and columns
        // rank..n-1 of L are exactly zero.
        void AssertReconstruct(in doubleMxN A, in doubleMxN L, in Pivot P, int rank, double precision)
        {
            int n = A.M_Rows;

            double maxErr = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    double llt = 0;
                    for (int k = 0; k < n; k++)
                        llt += L[i, k] * L[j, k];

                    double aperm = A[P[i], P[j]];
                    double e = math.abs(aperm - llt);
                    if (e > maxErr) maxErr = e;
                }
            RecordBound(maxErr, precision);

            // strict upper triangle is zero, and columns >= rank are zero (clean n×rank factor).
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    if ((j > i || j >= rank) && math.abs(L[i, j]) > precision && Fail[0] == (double)0)
                    {
                        Fail[0] = (double)1;
                        Fail[1] = L[i, j];
                        Fail[2] = 0;
                        Fail[3] = L[i, j];
                    }
        }

        static doubleMxN Gram(in Arena arena, in doubleMxN B)
        {
            int n = B.M_Rows, r = B.N_Cols;
            var A = arena.doubleMat(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    double s = 0;
                    for (int k = 0; k < r; k++)
                        s += B[i, k] * B[j, k];
                    A[i, j] = s;
                }
            return A;
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
    }

    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void PivotedCholeskyTests(TestJob.TestType type)
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
}
