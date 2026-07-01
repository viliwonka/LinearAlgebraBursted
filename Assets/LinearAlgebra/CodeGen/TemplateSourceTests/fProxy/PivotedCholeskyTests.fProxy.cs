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
// Properties / vectors exercised:
//  - Full-rank SPD: reconstruction PᵀAP == LLᵀ, rank == n, exact solve, first pivot = largest diagonal.
//  - Diagonal PSD matrix diag(4,0,9,0,1): hand-verifiable pivot order (P[0]=2 [val 9], P[1]=0 [val 4],
//    P[2]=4 [val 1]) and numerical rank 3.
//  - Rank-deficient PSD Gram matrix A = B·Bᵀ (B is n×r): rank == r, reconstruction, and the minimum-norm
//    solve recovers x exactly when b ∈ range(A).
//  - Minimum-norm certificate: for b = A·xOrig, the solve returns x with A·x ≈ b and ‖x‖ ≤ ‖xOrig‖.
//  - Indefinite matrices => choleskyDecompositionPivot returns false (e.g. [[1,2],[2,1]], eigenvalues 3,-1).
//  - Zero matrix => rank 0, solve returns x = 0. Rank-1 outer product v·vᵀ => rank 1.
public class fProxyPivotedCholeskyTests
{
    [BurstCompile]
    public struct AssemblyTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            var B = arena.fProxyRandomMat(6, 4);
            var A = Gram(in arena, in B);
            var L = arena.fProxyMat(6);
            var P = new Pivot(6, Allocator.Persistent);

            Cholesky.choleskyDecompositionPivot(in A, ref L, ref P, out int rank);

            P.Dispose();
            arena.Dispose();
        }

        static fProxyMxN Gram(in Arena arena, in fProxyMxN B)
        {
            int n = B.M_Rows, r = B.N_Cols;
            var A = arena.fProxyMat(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    fProxy s = 0;
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
            ZeroDiagonalIndefinite,
            ZeroMatrix,
            Rank1Outer,
            SingleElement,
            OutOfRangeLeastSquares,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<fProxy> Fail;

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
                var B = arena.fProxyRandomMat(n, n, -1f, 1f, 6100 + t * 13);
                var A = Gram(in arena, in B);
                for (int d = 0; d < n; d++) A[d, d] += (fProxy)n; // diagonal boost: well-conditioned SPD

                var L = arena.fProxyMat(n);
                var P = new Pivot(n, Allocator.Persistent);

                bool ok = Cholesky.choleskyDecompositionPivot(in A, ref L, ref P, out int rank);
                RecordEq(ok ? 1 : 0, 1);
                RecordEq(rank, n);
                AssertReconstruct(in A, in L, in P, rank, (fProxy)1E-4f);

                // first pivot must be the largest diagonal of A.
                int argmaxDiag = 0; fProxy best = A[0, 0];
                for (int j = 1; j < n; j++) if (A[j, j] > best) { best = A[j, j]; argmaxDiag = j; }
                RecordEq(P[0], argmaxDiag);

                // exact solve: b = A xOrig => x == xOrig.
                var xOrig = arena.fProxyRandomVec(n, -3f, 3f, 71000 + t * 7);
                var b = fProxy_OP.dot(A, xOrig);
                var Lc = arena.fProxyMat(n);
                var Pc = new Pivot(n, Allocator.Persistent);
                Cholesky.choleskyPivotSolve(in A, ref Lc, ref Pc, ref b); // b <- x
                for (int i = 0; i < n; i++) b[i] -= xOrig[i];
                RecordBound(fProxyNorms_OP.L2(in b), (fProxy)1E-3f);

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
            var A = arena.fProxyMat(n, n);
            A[0, 0] = 4f; A[2, 2] = 9f; A[4, 4] = 1f; // others zero

            var L = arena.fProxyMat(n);
            var P = new Pivot(n, Allocator.Persistent);

            bool ok = Cholesky.choleskyDecompositionPivot(in A, ref L, ref P, out int rank);
            RecordEq(ok ? 1 : 0, 1);
            RecordEq(rank, 3);
            RecordEq(P[0], 2);
            RecordEq(P[1], 0);
            RecordEq(P[2], 4);
            AssertReconstruct(in A, in L, in P, rank, (fProxy)1E-5f);

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
                var B = arena.fProxyRandomMat(n, r, -1f, 1f, 8200 + t * 11);
                var A = Gram(in arena, in B);

                var L = arena.fProxyMat(n);
                var P = new Pivot(n, Allocator.Persistent);

                bool ok = Cholesky.choleskyDecompositionPivot(in A, ref L, ref P, out int rank);
                RecordEq(ok ? 1 : 0, 1);
                RecordEq(rank, r);
                AssertReconstruct(in A, in L, in P, rank, (fProxy)1E-4f);

                // xRange = A·w ∈ range(A); b = A·xRange => min-norm solution == xRange.
                var w = arena.fProxyRandomVec(n, -2f, 2f, 51000 + t * 5);
                var xRange = fProxy_OP.dot(A, w);
                var b = fProxy_OP.dot(A, xRange);

                var Ls = arena.fProxyMat(n);
                var Ps = new Pivot(n, Allocator.Persistent);
                Cholesky.choleskyPivotSolve(in A, ref Ls, ref Ps, ref b); // b <- x

                // A·x ≈ A·xRange (consistency) and x ≈ xRange (exact recovery, scaled by ‖xRange‖).
                fProxy scale = fProxyNorms_OP.L2(in xRange) + (fProxy)1f;
                var diff = arena.fProxyVec(n);
                for (int i = 0; i < n; i++) diff[i] = b[i] - xRange[i];
                RecordBound(fProxyNorms_OP.L2(in diff) / scale, (fProxy)1E-2f);

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
                var B = arena.fProxyRandomMat(n, r, -1f, 1f, 3300 + t * 17);
                var A = Gram(in arena, in B);

                var xOrig = arena.fProxyRandomVec(n, -2f, 2f, 42000 + t * 9);
                var b = fProxy_OP.dot(A, xOrig);     // b ∈ range(A)
                var bForResidual = b.Copy();

                var L = arena.fProxyMat(n);
                var P = new Pivot(n, Allocator.Persistent);
                Cholesky.choleskyPivotSolve(in A, ref L, ref P, ref b); // b <- x

                // consistency: A·x ≈ bForResidual.
                var Ax = fProxy_OP.dot(A, b);
                var resid = arena.fProxyVec(n);
                for (int i = 0; i < n; i++) resid[i] = Ax[i] - bForResidual[i];
                fProxy bScale = fProxyNorms_OP.L2(in bForResidual) + (fProxy)1f;
                RecordBound(fProxyNorms_OP.L2(in resid) / bScale, (fProxy)1E-2f);

                // minimum-norm: ‖x‖ ≤ ‖xOrig‖ (+ tiny slack).
                fProxy xNorm = fProxyNorms_OP.L2(in b);
                fProxy origNorm = fProxyNorms_OP.L2(in xOrig);
                if (!(xNorm <= origNorm + (fProxy)1E-3f * (origNorm + (fProxy)1f)) && Fail[0] == (fProxy)0)
                {
                    Fail[0] = (fProxy)1;
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
            var A = arena.fProxyMat(n, n);
            A[0, 0] = 1f; A[0, 1] = 2f; A[1, 0] = 2f; A[1, 1] = 1f; // eigenvalues 3, -1

            var L = arena.fProxyMat(n);
            var P = new Pivot(n, Allocator.Persistent);

            bool ok = Cholesky.choleskyDecompositionPivot(in A, ref L, ref P, out int rank);
            RecordEq(ok ? 1 : 0, 0); // indefinite => false

            P.Dispose();
            arena.Dispose();
        }

        void Indefinite3x3()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 3;
            var A = arena.fProxyMat(n, n);
            // PSD-looking 2x2 block plus a negative eigenvalue contribution.
            A[0, 0] = 2f; A[1, 1] = 2f; A[2, 2] = 2f;
            A[0, 1] = 0f; A[1, 0] = 0f;
            A[0, 2] = 3f; A[2, 0] = 3f; // big off-diagonal => negative eigenvalue
            A[1, 2] = 0f; A[2, 1] = 0f;

            var L = arena.fProxyMat(n);
            var P = new Pivot(n, Allocator.Persistent);

            bool ok = Cholesky.choleskyDecompositionPivot(in A, ref L, ref P, out int rank);
            RecordEq(ok ? 1 : 0, 0);

            P.Dispose();
            arena.Dispose();
        }

        void ZeroMatrix()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;
            var A = arena.fProxyMat(n, n); // zero
            var L = arena.fProxyMat(n);
            var P = new Pivot(n, Allocator.Persistent);

            bool ok = Cholesky.choleskyDecompositionPivot(in A, ref L, ref P, out int rank);
            RecordEq(ok ? 1 : 0, 1);
            RecordEq(rank, 0);

            // solve: x = 0 for any b.
            var b = arena.fProxyRandomVec(n, -1f, 1f, 999);
            Cholesky.choleskyPivotSolve(ref L, in P, rank, ref b);
            RecordBound(fProxyNorms_OP.L2(in b), (fProxy)1E-6f);

            P.Dispose();
            arena.Dispose();
        }

        void Rank1Outer()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 4;
            var v = arena.fProxyVec(n);
            v[0] = 1f; v[1] = 2f; v[2] = 3f; v[3] = 0.5f;

            var A = arena.fProxyMat(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    A[i, j] = v[i] * v[j]; // rank-1 PSD

            var L = arena.fProxyMat(n);
            var P = new Pivot(n, Allocator.Persistent);

            bool ok = Cholesky.choleskyDecompositionPivot(in A, ref L, ref P, out int rank);
            RecordEq(ok ? 1 : 0, 1);
            RecordEq(rank, 1);
            AssertReconstruct(in A, in L, in P, rank, (fProxy)1E-4f);

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
            var A = arena.fProxyMat(n, n);
            A[0, 1] = 1f; A[1, 0] = 1f; // diagonal stays zero

            var L = arena.fProxyMat(n);
            var P = new Pivot(n, Allocator.Persistent);

            bool ok = Cholesky.choleskyDecompositionPivot(in A, ref L, ref P, out int rank);
            RecordEq(ok ? 1 : 0, 0); // indefinite => false

            P.Dispose();
            arena.Dispose();
        }

        // 1x1 SPD: A=[[4]], b=[8] => x=[2]; rank 1, reconstruction trivial.
        void SingleElement()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 1;
            var A = arena.fProxyMat(n, n);
            A[0, 0] = 4f;

            var L = arena.fProxyMat(n);
            var P = new Pivot(n, Allocator.Persistent);

            bool ok = Cholesky.choleskyDecompositionPivot(in A, ref L, ref P, out int rank);
            RecordEq(ok ? 1 : 0, 1);
            RecordEq(rank, 1);
            AssertReconstruct(in A, in L, in P, rank, (fProxy)1E-5f);

            var b = arena.fProxyVec(n);
            b[0] = 8f;
            Cholesky.choleskyPivotSolve(ref L, in P, rank, ref b);
            RecordBound(math.abs(b[0] - (fProxy)2f), (fProxy)1E-5f);

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
                var B = arena.fProxyRandomMat(n, r, -1f, 1f, 2600 + t * 19);
                var A = Gram(in arena, in B);

                // arbitrary b, generically NOT in range(A) since rank r < n.
                var b = arena.fProxyRandomVec(n, -2f, 2f, 77000 + t * 13);

                var L = arena.fProxyMat(n);
                var P = new Pivot(n, Allocator.Persistent);
                bool ok = Cholesky.choleskyPivotSolve(in A, ref L, ref P, ref b); // b <- x
                RecordEq(ok ? 1 : 0, 1);

                // normal equations: A(Ax) == A b  <=>  A(Ax - b) == 0  (residual ⟂ range(A)).
                var Ax = fProxy_OP.dot(A, b);
                var AAx = fProxy_OP.dot(A, Ax);
                // recompute A b — b now holds x, so rebuild b from the same seed.
                var bOrig = arena.fProxyRandomVec(n, -2f, 2f, 77000 + t * 13);
                var Ab = fProxy_OP.dot(A, bOrig);

                fProxy scale = fProxyNorms_OP.L2(in Ab) + (fProxy)1f;
                var diff = arena.fProxyVec(n);
                for (int i = 0; i < n; i++) diff[i] = AAx[i] - Ab[i];
                RecordBound(fProxyNorms_OP.L2(in diff) / scale, (fProxy)1E-2f);

                P.Dispose();
                arena.Clear();
            }

            arena.Dispose();
        }

        // PᵀAP == LLᵀ (computed directly as Σ_k L[i,k]L[j,k]), L lower-triangular, and columns
        // rank..n-1 of L are exactly zero.
        void AssertReconstruct(in fProxyMxN A, in fProxyMxN L, in Pivot P, int rank, fProxy precision)
        {
            int n = A.M_Rows;

            fProxy maxErr = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    fProxy llt = 0;
                    for (int k = 0; k < n; k++)
                        llt += L[i, k] * L[j, k];

                    fProxy aperm = A[P[i], P[j]];
                    fProxy e = math.abs(aperm - llt);
                    if (e > maxErr) maxErr = e;
                }
            RecordBound(maxErr, precision);

            // strict upper triangle is zero, and columns >= rank are zero (clean n×rank factor).
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    if ((j > i || j >= rank) && math.abs(L[i, j]) > precision && Fail[0] == (fProxy)0)
                    {
                        Fail[0] = (fProxy)1;
                        Fail[1] = L[i, j];
                        Fail[2] = 0;
                        Fail[3] = L[i, j];
                    }
        }

        static fProxyMxN Gram(in Arena arena, in fProxyMxN B)
        {
            int n = B.M_Rows, r = B.N_Cols;
            var A = arena.fProxyMat(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    fProxy s = 0;
                    for (int k = 0; k < r; k++)
                        s += B[i, k] * B[j, k];
                    A[i, j] = s;
                }
            return A;
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
            Assert.IsTrue(value <= limit);
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
    public void PivotedCholeskyTests(TestJob.TestType type)
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
