using System;

using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Stats;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using UnityEngine;
using UnityEngine.TestTools;
using LinearAlgebra.Internal;

public class doubleLUTests
{

    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            LUDecompIdentity,
            LUDecompPredefined,
            LUDecompRandomDiagonal,
            LUDecompRandom,
            LUDecompSingular,
            LUDecompPivotRequired,
            LUDeterminant,
            LUDeterminantGallery,
            LUReusePivot,
            SwapOPTest,
            LUSolveSystem,
            LUSolveSystemInplace,
            // Blocked (level-3) LU path coverage (engages at M_Rows >= 256; LU_BLOCK=32).
            LUBlockedRefAccuracy256,
            LUBlockedRefAccuracy300,
            LUBlockedIllConditioned256,
            LUBlockedSingular256,
            LUBlockedSolve300
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff/index
        public NativeArray<double> Fail;

        public void Execute()
        {
            switch(Type)
            {
                case TestType.LUDecompIdentity:
                    LUDecompIdentity();
                break;
                case TestType.LUDecompPredefined:
                    LUDecompPredefined();
                break;
                case TestType.LUDecompRandomDiagonal:
                    LUDecompRandomDiagonal();
                break;
                case TestType.LUDecompRandom:
                    LUDecompRandom();
                break;
                case TestType.LUDecompSingular:
                    LUDecompSingular();
                break;
                case TestType.LUDecompPivotRequired:
                    LUDecompPivotRequired();
                break;
                case TestType.LUDeterminant:
                    LUDeterminant();
                break;
                case TestType.LUDeterminantGallery:
                    LUDeterminantGallery();
                break;
                case TestType.LUReusePivot:
                    LUReusePivot();
                break;
                case TestType.SwapOPTest:
                    SwapOPTest();
                break;
                case TestType.LUSolveSystem:
                    SolveSystem();
                break;
                case TestType.LUSolveSystemInplace:
                    SolveSystemInplace();
                    break;
                case TestType.LUBlockedRefAccuracy256:
                    LUBlockedRefAccuracy256();
                    break;
                case TestType.LUBlockedRefAccuracy300:
                    LUBlockedRefAccuracy300();
                    break;
                case TestType.LUBlockedIllConditioned256:
                    LUBlockedIllConditioned256();
                    break;
                case TestType.LUBlockedSingular256:
                    LUBlockedSingular256();
                    break;
                case TestType.LUBlockedSolve300:
                    LUBlockedSolve300();
                    break;

            }
        }

        private doubleMxN GetRandomMatrix(ref Arena arena, int dim, double min, double max, uint seed) {

            var mat = arena.doubleRandomMat(dim, dim, min, max, seed);

            return mat;
        }

        public void LUDecompIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var U = arena.doubleIdentityMat(dim);
            var L = arena.doubleIdentityMat(dim);

            var A = U.Copy();

            bool success = LU.luDecompositionNoPivot(ref U, ref L);

            Assert.IsTrue(success);

            AssertLU(in A, in L, in U, false);

            arena.Dispose();
        }
        public void LUDecompRandomDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var U = arena.doubleRandomDiagonalMat(dim, 1f, 3f);
            var L = arena.doubleIdentityMat(dim);

            var A = U.Copy();

            bool success = LU.luDecompositionNoPivot(ref U, ref L);

            Assert.IsTrue(success);

            AssertLU(in A, in L, in U, false);

            arena.Dispose();
        }

        public void LUDecompPredefined() {

            var arena = new Arena(Allocator.Persistent);

            var dim = 5;

            var U = arena.doubleMat(dim);
            var L = arena.doubleIdentityMat(dim);

            U[0] = -2f;
            U[1] = 1f;
            U[2] = -2f;
            U[3] = 3f;
            U[4] = 1f;

            U[5] = 1f;
            U[6] = -2f;
            U[7] = 3f;
            U[8] = -5f;
            U[9] = 4f;

            U[10] = 4f;
            U[11] = 3f;
            U[12] = -1f;
            U[13] = 2f;
            U[14] = -3f;

            U[15] = 1f;
            U[16] = 1f;
            U[17] = -1f;
            U[18] = -11f;
            U[19] = 11f;

            U[20] = -1f;
            U[21] = -9f;
            U[22] = -1f;
            U[23] = 7f;
            U[24] = 1f;

            var A = U.Copy();

            var pivot = new Pivot(dim, Allocator.Temp);

            bool success = LU.luDecomposition(ref U, ref L, ref pivot);

            Assert.IsTrue(success);

            // PA = LU  =>  A = P^-1 (LU). Apply inverse pivot to A to match L*U.
            pivot.ApplyInverseRow(ref A);

            AssertLU(in A, in L, in U, true, 1E-5f);

            pivot.Dispose();

            arena.Dispose();
        }

        public void LUDecompRandom()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 18;

            var U = arena.doubleRandomMat(dim, dim, 1f, 10f, 314221);
            var L = arena.doubleIdentityMat(dim);

            // add to diagonals of U
            for(int d = 0; d < dim; d++)
                U[d, d] += 5f;

            var A = U.Copy();

            var pivot = new Pivot(dim, Allocator.Temp);

            bool success = LU.luDecomposition(ref U, ref L, ref pivot);

            Assert.IsTrue(success);

            pivot.ApplyInverseRow(ref A);

            pivot.Dispose();

            AssertLU(in A, in L, in U, true, 1E-05f);

            arena.Dispose();
        }

        public void LUDecompSingular()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            // Case 1: zero matrix is singular -> all variants must return false.
            {
                var U = arena.doubleMat(dim, dim);
                var L = arena.doubleIdentityMat(dim);

                bool noPivot = LU.luDecompositionNoPivot(ref U, ref L);
                Assert.IsFalse(noPivot);
                Assert.IsFalse(Analysis_OP.isAnyNan(in U));
                Assert.IsFalse(Analysis_OP.isAnyNan(in L));

                var Up = arena.doubleMat(dim, dim);
                var Lp = arena.doubleIdentityMat(dim);
                var pivot = new Pivot(dim, Allocator.Temp);
                bool pivoted = LU.luDecomposition(ref Up, ref Lp, ref pivot);
                Assert.IsFalse(pivoted);
                Assert.IsFalse(Analysis_OP.isAnyNan(in Up));
                Assert.IsFalse(Analysis_OP.isAnyNan(in Lp));

                var LUmat = arena.doubleMat(dim, dim);
                bool inplace = LU.luDecompositionInpl(ref LUmat, ref pivot);
                Assert.IsFalse(inplace);
                Assert.IsFalse(Analysis_OP.isAnyNan(in LUmat));

                pivot.Dispose();
            }

            // Case 2: two identical rows -> rank deficient -> all variants return false.
            {
                var U = arena.doubleRandomMat(dim, dim, 1f, 10f, 8821);
                // force diagonal dominance so only the duplicated rows cause singularity
                for (int d = 0; d < dim; d++)
                    U[d, d] += 20f;
                // make row 5 an exact copy of row 2
                for (int c = 0; c < dim; c++)
                    U[5, c] = U[2, c];

                var A = U.Copy();
                var L = arena.doubleIdentityMat(dim);

                bool noPivot = LU.luDecompositionNoPivot(ref U, ref L);
                Assert.IsFalse(noPivot);
                Assert.IsFalse(Analysis_OP.isAnyNan(in U));
                Assert.IsFalse(Analysis_OP.isAnyNan(in L));

                var Up = A.Copy();
                var Lp = arena.doubleIdentityMat(dim);
                var pivot = new Pivot(dim, Allocator.Temp);
                bool pivoted = LU.luDecomposition(ref Up, ref Lp, ref pivot);
                Assert.IsFalse(pivoted);
                Assert.IsFalse(Analysis_OP.isAnyNan(in Up));
                Assert.IsFalse(Analysis_OP.isAnyNan(in Lp));

                var LUmat = A.Copy();
                bool inplace = LU.luDecompositionInpl(ref LUmat, ref pivot);
                Assert.IsFalse(inplace);
                Assert.IsFalse(Analysis_OP.isAnyNan(in LUmat));

                pivot.Dispose();
            }

            // Case 3: [[0,1],[1,0]] : no-pivot fails on zero leading pivot,
            // but inplace (with partial pivoting) succeeds.
            {
                var U = arena.doubleMat(2, 2);
                U[0, 0] = 0f; U[0, 1] = 1f;
                U[1, 0] = 1f; U[1, 1] = 0f;

                var L = arena.doubleIdentityMat(2);

                var Unp = U.Copy();
                bool noPivot = LU.luDecompositionNoPivot(ref Unp, ref L);
                Assert.IsFalse(noPivot);

                var LUmat = U.Copy();
                var pivot = new Pivot(2, Allocator.Temp);
                bool inplace = LU.luDecompositionInpl(ref LUmat, ref pivot);
                Assert.IsTrue(inplace);
                Assert.IsFalse(Analysis_OP.isAnyNan(in LUmat));

                pivot.Dispose();
            }

            arena.Dispose();
        }

        public void LUDecompPivotRequired()
        {
            var arena = new Arena(Allocator.Persistent);

            // Case A: 3x3 with A[0,0] == 0 but nonsingular, requires pivoting.
            {
                int dim = 3;
                var A = arena.doubleMat(dim, dim);
                // A[0,0] == 0 forces a row swap
                A[0, 0] = 0f; A[0, 1] = 2f; A[0, 2] = 1f;
                A[1, 0] = 1f; A[1, 1] = 1f; A[1, 2] = 1f;
                A[2, 0] = 2f; A[2, 1] = 1f; A[2, 2] = 0f;

                var x_Known = arena.doubleVec(dim);
                x_Known[0] = 3f; x_Known[1] = -2f; x_Known[2] = 5f;

                var b = Linear_OP.dot(A, x_Known);

                var LUmat = A.Copy();
                var pivot = new Pivot(dim, Allocator.Temp);

                bool success = LU.luDecompositionInpl(ref LUmat, ref pivot);
                Assert.IsTrue(success);

                var x_Solved = b.Copy();
                LU.luSolve(ref LUmat, in pivot, ref x_Solved);

                Assert.IsFalse(Analysis_OP.isAnyNan(in x_Solved));

                AssertVecClose(in x_Known, in x_Solved, dim, 1E-3f);

                pivot.Dispose();
            }

            // Case B: permutation contains a 3-cycle (not an involution).
            // Construct A so partial pivoting yields a cyclic permutation. We make the
            // sub-diagonal magnitudes strictly increasing down each column so the pivot
            // search always selects the last remaining row -> a long cycle, not pair swaps.
            {
                int dim = 4;
                var A = arena.doubleMat(dim, dim);
                A[0, 0] = 1f; A[0, 1] = 2f; A[0, 2] = 0f; A[0, 3] = 1f;
                A[1, 0] = 2f; A[1, 1] = 1f; A[1, 2] = 3f; A[1, 3] = 0f;
                A[2, 0] = 4f; A[2, 1] = 0f; A[2, 2] = 1f; A[2, 3] = 2f;
                A[3, 0] = 8f; A[3, 1] = 3f; A[3, 2] = 2f; A[3, 3] = 1f;

                var x_Known = arena.doubleVec(dim);
                x_Known[0] = 1f; x_Known[1] = -3f; x_Known[2] = 2f; x_Known[3] = 4f;

                var b = Linear_OP.dot(A, x_Known);

                var LUmat = A.Copy();
                var pivot = new Pivot(dim, Allocator.Temp);

                bool success = LU.luDecompositionInpl(ref LUmat, ref pivot);
                Assert.IsTrue(success);

                // Verify the permutation is not a simple involution (P applied twice != identity),
                // i.e. it really contains a cycle of length > 2.
                bool isInvolution = true;
                for (int i = 0; i < dim; i++)
                    if (pivot[pivot[i]] != i)
                        isInvolution = false;
                Assert.IsFalse(isInvolution);

                var x_Solved = b.Copy();
                LU.luSolve(ref LUmat, in pivot, ref x_Solved);

                Assert.IsFalse(Analysis_OP.isAnyNan(in x_Solved));

                AssertVecClose(in x_Known, in x_Solved, dim, 1E-3f);

                pivot.Dispose();
            }

            arena.Dispose();
        }

        public void LUDeterminant()
        {
            var arena = new Arena(Allocator.Persistent);

            // identity -> det = 1
            {
                int dim = 6;
                var I = arena.doubleIdentityMat(dim);
                var pivot = new Pivot(dim, Allocator.Temp);

                bool success = LU.luDecompositionInpl(ref I, ref pivot);
                Assert.IsTrue(success);

                double det = LU.determinant(in I, in pivot);
                AssertClose(det, (double)1f, 1E-4f);

                pivot.Dispose();
            }

            // diagonal -> det = product of diagonal
            {
                int dim = 4;
                var D = arena.doubleMat(dim, dim);
                D[0, 0] = 2f;
                D[1, 1] = -3f;
                D[2, 2] = 0.5f;
                D[3, 3] = 4f;
                double expected = 2f * -3f * 0.5f * 4f; // -12

                var pivot = new Pivot(dim, Allocator.Temp);
                bool success = LU.luDecompositionInpl(ref D, ref pivot);
                Assert.IsTrue(success);

                double det = LU.determinant(in D, in pivot);
                AssertCloseRel(det, expected, 1E-4f);

                pivot.Dispose();
            }

            // 3x3 with known determinant requiring a row swap (A[0,0]==0).
            // A = [[0,2,1],[1,1,1],[2,1,0]]; det = 3 (hand computed, nonsingular).
            {
                int dim = 3;
                var A = arena.doubleMat(dim, dim);
                A[0, 0] = 0f; A[0, 1] = 2f; A[0, 2] = 1f;
                A[1, 0] = 1f; A[1, 1] = 1f; A[1, 2] = 1f;
                A[2, 0] = 2f; A[2, 1] = 1f; A[2, 2] = 0f;

                var pivot = new Pivot(dim, Allocator.Temp);
                bool success = LU.luDecompositionInpl(ref A, ref pivot);
                Assert.IsTrue(success);

                double det = LU.determinant(in A, in pivot);
                AssertCloseRel(det, (double)3f, 1E-4f);

                pivot.Dispose();
            }

            // permutation matrix -> det = +-1 matching swap parity.
            // single transposition (rows 0 and 2 swapped) -> det = -1.
            {
                int dim = 3;
                var P = arena.doubleMat(dim, dim);
                P[0, 2] = 1f;
                P[1, 1] = 1f;
                P[2, 0] = 1f;

                var pivot = new Pivot(dim, Allocator.Temp);
                bool success = LU.luDecompositionInpl(ref P, ref pivot);
                Assert.IsTrue(success);

                double det = LU.determinant(in P, in pivot);
                AssertCloseRel(det, (double)(-1f), 1E-4f);

                pivot.Dispose();
            }

            arena.Dispose();
        }

        // GALLERY KNOWN-ANSWER: famous unit-determinant matrices. det is computed via
        // luDecompositionInpl + LU.determinant (the file's established sequence).
        //  - Pascal(5):  symmetric Pascal, det = 1.
        //  - MinIJ(5):   A[i,j]=min(i,j)+1, det = 1.
        //  - Frank(5):   upper-Hessenberg Frank, det = 1 (ill-conditioned but integer-valued, so
        //                float LU stays accurate — relerr ~1e-6 in single precision).
        public void LUDeterminantGallery()
        {
            var arena = new Arena(Allocator.Persistent);

            // Pascal(5): det = 1
            {
                int dim = 5;
                var A = arena.doublePascal(dim);
                var pivot = new Pivot(dim, Allocator.Temp);
                bool success = LU.luDecompositionInpl(ref A, ref pivot);
                Assert.IsTrue(success);

                double det = LU.determinant(in A, in pivot);
                AssertCloseRel(det, (double)1f, 1E-4f);

                pivot.Dispose();
            }

            // MinIJ(5): det = 1
            {
                int dim = 5;
                var A = arena.doubleMinIJ(dim);
                var pivot = new Pivot(dim, Allocator.Temp);
                bool success = LU.luDecompositionInpl(ref A, ref pivot);
                Assert.IsTrue(success);

                double det = LU.determinant(in A, in pivot);
                AssertCloseRel(det, (double)1f, 1E-4f);

                pivot.Dispose();
            }

            // Frank(5): det = 1
            {
                int dim = 5;
                var A = arena.doubleFrank(dim);
                var pivot = new Pivot(dim, Allocator.Temp);
                bool success = LU.luDecompositionInpl(ref A, ref pivot);
                Assert.IsTrue(success);

                double det = LU.determinant(in A, in pivot);
                AssertCloseRel(det, (double)1f, 1E-4f);

                pivot.Dispose();
            }

            arena.Dispose();
        }

        public void LUReusePivot()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 6;

            var pivot = new Pivot(dim, Allocator.Temp);

            // First decomposition with the pivot - permutes it.
            var A1 = arena.doubleRandomMat(dim, dim, -5f, 5f, 7777);
            for (int d = 0; d < dim; d++)
                A1[d, d] += 15f;
            var LU1 = A1.Copy();
            bool s1 = LU.luDecompositionInpl(ref LU1, ref pivot);
            Assert.IsTrue(s1);

            // Second decomposition reuses the SAME pivot object; Reset() must clean it.
            var A2 = arena.doubleRandomMat(dim, dim, -5f, 5f, 9999);
            for (int d = 0; d < dim; d++)
                A2[d, d] += 15f;

            var x_Known = arena.doubleVec(dim);
            for (int i = 0; i < dim; i++)
                x_Known[i] = (double)(i + 1);

            var b = Linear_OP.dot(A2, x_Known);

            var LU2 = A2.Copy();
            bool s2 = LU.luDecompositionInpl(ref LU2, ref pivot);
            Assert.IsTrue(s2);

            var x_Solved = b.Copy();
            LU.luSolve(ref LU2, in pivot, ref x_Solved);

            Assert.IsFalse(Analysis_OP.isAnyNan(in x_Solved));

            AssertVecClose(in x_Known, in x_Solved, dim, 1E-3f);

            pivot.Dispose();

            arena.Dispose();
        }

        public void SwapOPTest()
        {
            var arena = new Arena(Allocator.Persistent);

            // Swap_OP.Rows with default start/end swaps full rows.
            {
                int dim = 3;
                var mat = arena.doubleMat(dim, dim);
                for (int r = 0; r < dim; r++)
                    for (int c = 0; c < dim; c++)
                        mat[r, c] = (double)(r * 10 + c);

                Swap_OP.Rows(ref mat, 0, 1);

                // row 0 and row 1 fully swapped
                for (int c = 0; c < dim; c++) {
                    AssertClose(mat[0, c], (double)(10 + c), 1E-6f);
                    AssertClose(mat[1, c], (double)(0 + c), 1E-6f);
                    AssertClose(mat[2, c], (double)(20 + c), 1E-6f);
                }
            }

            // Swap_OP.Columns with explicit start/end swaps only that row-range.
            {
                int dim = 4;
                var mat = arena.doubleMat(dim, dim);
                for (int r = 0; r < dim; r++)
                    for (int c = 0; c < dim; c++)
                        mat[r, c] = (double)(r * 10 + c);

                // swap columns 0 and 1 only for rows [1,3)
                Swap_OP.Columns(ref mat, 0, 1, 1, 3);

                // rows 0 and 3 untouched
                AssertClose(mat[0, 0], (double)(0), 1E-6f);
                AssertClose(mat[0, 1], (double)(1), 1E-6f);
                AssertClose(mat[3, 0], (double)(30), 1E-6f);
                AssertClose(mat[3, 1], (double)(31), 1E-6f);

                // rows 1 and 2 have columns 0 and 1 swapped
                AssertClose(mat[1, 0], (double)(11), 1E-6f);
                AssertClose(mat[1, 1], (double)(10), 1E-6f);
                AssertClose(mat[2, 0], (double)(21), 1E-6f);
                AssertClose(mat[2, 1], (double)(20), 1E-6f);

                // other columns untouched
                AssertClose(mat[1, 2], (double)(12), 1E-6f);
                AssertClose(mat[2, 3], (double)(23), 1E-6f);
            }

            arena.Dispose();
        }

        public void SolveSystem() {

            var arena = new Arena(Allocator.Persistent);

            int dim = 512;

            var A = arena.doubleRandomMat(dim, dim, -10f, 10f, 314221);

            for (int d = 0; d < dim; d++) {
                A[d, d] *= 2f;
                if (Unity.Mathematics.math.abs(A[d, d]) < 0.01f)
                    A[d, d] *= 10f;
            }

            var x_Known = arena.doubleRandomVec(dim, 1f, 10f, 901);

            var b = Linear_OP.dot(A, x_Known);

            var U = A.Copy();
            var L = arena.doubleIdentityMat(dim);

            var pivot = new Pivot(dim, Allocator.Temp);

            bool success = LU.luDecomposition(ref U, ref L, ref pivot);

            Assert.IsTrue(success);

            var x_Solved = b.Copy();

            LU.luSolve(ref L, ref U, in pivot, ref x_Solved);

            if (Analysis_OP.isAnyNan(in x_Solved))
                throw new System.Exception("TestJob: NaN detected");

            var zeroError = Analysis_OP.MaxZeroError(x_Known - x_Solved);

            // Fail layout: [1]=zeroError, [2]=limit 1E-3, [3]=diff
            if (!(zeroError < (double)1E-03f) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = zeroError;
                Fail[2] = (double)1E-03f;
                Fail[3] = zeroError - (double)1E-03f;
            }
            Assert.IsTrue(zeroError < 1E-03f);

            pivot.Dispose();

            arena.Dispose();
        }

        public void SolveSystemInplace() {

            var arena = new Arena(Allocator.Persistent);

            int dim = 512;

            var A = arena.doubleRandomMat(dim, dim, -10f, 10f, 314221);

            for (int d = 0; d < dim; d++) {
                A[d, d] *= 2f;
                if (Unity.Mathematics.math.abs(A[d, d]) < 0.01f)
                    A[d, d] *= 10f;
            }

            var x_Known = arena.doubleRandomVec(dim, 1f, 10f, 901);

            var b = Linear_OP.dot(A, x_Known);

            var LUmat = A.Copy();

            var pivot = new Pivot(dim, Allocator.Temp);

            bool success = LU.luDecompositionInpl(ref LUmat, ref pivot);

            Assert.IsTrue(success);

            var x_Solved = b.Copy();

            LU.luSolve(ref LUmat, in pivot, ref x_Solved);

            if (Analysis_OP.isAnyNan(in x_Solved))
                throw new System.Exception("TestJob: NaN detected");

            var zeroError = Analysis_OP.MaxZeroError(x_Known - x_Solved);

            // Fail layout: [1]=zeroError, [2]=limit 1E-3, [3]=diff
            if (!(zeroError < (double)1E-03f) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = zeroError;
                Fail[2] = (double)1E-03f;
                Fail[3] = zeroError - (double)1E-03f;
            }
            Assert.IsTrue(zeroError < 1E-03f);

            pivot.Dispose();

            arena.Dispose();
        }

        // ================================================================================
        // BLOCKED (level-3) LU coverage.
        //
        // LU.luDecomposition(ref U, ref L, ref P) switches to the LAPACK-style right-looking
        // blocked (compact-WY GEMM trailing update) path at M_Rows >= LU_BLOCK_MIN_N = 8*32 = 256;
        // below that it runs the plain unblocked rank-1 sweep. The blocked path is DESIGNED to keep
        // the partial-pivoting sequence bit-identical to the unblocked form, so it must produce the
        // SAME pivot array and (within GEMM summation-order rounding) the same L/U as the independent,
        // untouched, level-2 compact factorization LU.luDecompositionInpl(ref LU, ref P) — which is
        // used here as the reference ORACLE for both correctness and accuracy.
        //
        // In the inpl compact form: factor row i lives at physical row P[i]; LU[P[i], j] with j < i
        // is the unit-lower L multiplier, and LU[P[i], j] with j >= i is U.
        // ================================================================================

        // (1) N=256: 8 aligned panels of LU_BLOCK=32 — exactly at the gate.
        public void LUBlockedRefAccuracy256()
        {
            var arena = new Arena(Allocator.Persistent);
            int dim = 256;
            var A = MakeWellConditionedPivoting(ref arena, dim, 260871);
            BlockedVsReference(ref arena, in A);
            arena.Dispose();
        }

        // (2) N=300 = 9*32 + 12 — non-aligned last panel.
        public void LUBlockedRefAccuracy300()
        {
            var arena = new Arena(Allocator.Persistent);
            int dim = 300;
            var A = MakeWellConditionedPivoting(ref arena, dim, 771013);
            BlockedVsReference(ref arena, in A);
            arena.Dispose();
        }

        // (3) Ill-conditioned N=256: Lehmer (SPD, totally nonnegative, cond < 4n^2 ~ 2.6e5 —
        // genuinely ill-conditioned for float, yet LU never hits a zero pivot). The KEY accuracy
        // test: the blocked backward error ||P A - L U|| stays small AND within a small factor of
        // the unblocked reference's residual on the SAME matrix, i.e. blocking did not amplify error.
        public void LUBlockedIllConditioned256()
        {
            var arena = new Arena(Allocator.Persistent);
            int dim = 256;
            var A = arena.doubleLehmer(dim);
            IllConditionedResidual(ref arena, in A);
            arena.Dispose();
        }

        // (4) Singular N=256: an exact duplicate row makes the matrix rank-deficient. Because the two
        // identical rows receive bit-identical updates until one is pivoted (then the other becomes an
        // exact zero row), a zero pivot is guaranteed and the blocked path must return false with no
        // NaN/Inf written.
        public void LUBlockedSingular256()
        {
            var arena = new Arena(Allocator.Persistent);
            int dim = 256;

            var A = arena.doubleRandomMat(dim, dim, 1f, 10f, 55221);
            // strong diagonal dominance so ONLY the duplicated rows cause singularity
            for (int d = 0; d < dim; d++)
                A[d, d] += (double)(2 * dim);
            // make row 137 an exact copy of row 42
            for (int c = 0; c < dim; c++)
                A[137, c] = A[42, c];

            var U = A.Copy();
            var L = arena.doubleIdentityMat(dim);
            var pivot = new Pivot(dim, Allocator.Temp);

            bool ok = LU.luDecomposition(ref U, ref L, ref pivot);

            Assert.IsFalse(ok);
            Assert.IsFalse(Analysis_OP.isAnyNan(in U));
            Assert.IsFalse(Analysis_OP.isAnyNan(in L));
            Assert.IsFalse(Analysis_OP.isAnyInf(in U));
            Assert.IsFalse(Analysis_OP.isAnyInf(in L));

            pivot.Dispose();
            arena.Dispose();
        }

        // (5) Solve round-trip N=300 (non-aligned last panel) using the separate-L/U blocked path.
        // Same recipe / tolerance as the existing dim=512 SolveSystem test.
        public void LUBlockedSolve300()
        {
            var arena = new Arena(Allocator.Persistent);
            int dim = 300;

            var A = arena.doubleRandomMat(dim, dim, -10f, 10f, 314221);

            for (int d = 0; d < dim; d++) {
                A[d, d] *= 2f;
                if (Unity.Mathematics.math.abs(A[d, d]) < 0.01f)
                    A[d, d] *= 10f;
            }

            var x_Known = arena.doubleRandomVec(dim, 1f, 10f, 901);

            var b = Linear_OP.dot(A, x_Known);

            var U = A.Copy();
            var L = arena.doubleIdentityMat(dim);

            var pivot = new Pivot(dim, Allocator.Temp);

            bool success = LU.luDecomposition(ref U, ref L, ref pivot);

            Assert.IsTrue(success);

            var x_Solved = b.Copy();

            LU.luSolve(ref L, ref U, in pivot, ref x_Solved);

            if (Analysis_OP.isAnyNan(in x_Solved))
                throw new System.Exception("TestJob: NaN detected");

            var zeroError = Analysis_OP.MaxZeroError(x_Known - x_Solved);

            // x-accuracy is condition-number-amplified (NOT the backward error, which stays ~eps·‖A‖).
            // This fixed random draw at n=300 (cond a few·10^3) lands at ~1.2e-3 max error in float —
            // just over the 1e-3 the dim=512 SolveSystem happens to hit — so use a per-precision band:
            // generous for float, tight for double. Deterministic (fixed seeds), so not flaky.
            double xtol = IsDouble() ? (double)1E-8 : (double)3E-3f;

            // Fail layout: [1]=zeroError, [2]=limit, [3]=diff
            if (!(zeroError < xtol) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = zeroError;
                Fail[2] = xtol;
                Fail[3] = zeroError - xtol;
            }
            Assert.IsTrue(zeroError < xtol);

            pivot.Dispose();
            arena.Dispose();
        }

        // true only when double expands to double (doubleEpsilon ≈ 2.2e-16 < 1e-10).
        private bool IsDouble() => (double)Consts.doubleEpsilon < 1e-10;

        // Well-conditioned input that forces a NONTRIVIAL but UNAMBIGUOUS pivot sequence: a strongly
        // (row+column) diagonally-dominant random matrix (each column has one big entry, all others
        // small) with its rows reversed. Partial pivoting must undo the reversal; the huge column-wise
        // gap (~2*dim vs ~1) means the argmax is robust to GEMM-vs-scalar summation-order rounding, so
        // the blocked and unblocked factorizations pick the SAME pivots for BOTH float and double.
        private doubleMxN MakeWellConditionedPivoting(ref Arena arena, int dim, uint seed)
        {
            var A = arena.doubleRandomMat(dim, dim, -1f, 1f, seed);
            for (int d = 0; d < dim; d++)
                A[d, d] += (double)(2 * dim);
            for (int i = 0; i < dim / 2; i++)
                Swap_OP.Rows(ref A, i, dim - 1 - i);
            return A;
        }

        // Points (1)/(2): blocked luDecomposition vs unblocked compact luDecompositionInpl oracle.
        // Asserts identical pivots, matching L/U factors, and no backward-error regression.
        private void BlockedVsReference(ref Arena arena, in doubleMxN A)
        {
            int dim = A.M_Rows;

            // --- blocked path (separate L, U, P) ---
            var U = A.Copy();
            var L = arena.doubleIdentityMat(dim);
            var pB = new Pivot(dim, Allocator.Temp);
            bool okB = LU.luDecomposition(ref U, ref L, ref pB);
            Assert.IsTrue(okB);
            Assert.IsFalse(Analysis_OP.isAnyNan(in U));
            Assert.IsFalse(Analysis_OP.isAnyNan(in L));

            // --- reference: independent unblocked compact inplace factorization ---
            var LUref = A.Copy();
            var pR = new Pivot(dim, Allocator.Temp);
            bool okR = LU.luDecompositionInpl(ref LUref, ref pR);
            Assert.IsTrue(okR);

            // (a) pivot arrays identical elementwise
            AssertPivotEqual(in pB, in pR, dim);

            // (b) factor accuracy: blocked L (strict lower) & U (upper) match the reference compact
            //     form at physical row pR[i]. Absolute tolerance scaled by matrix magnitude, matrix
            //     size (accumulation length) and machine eps — loose for float, tight for double.
            double aScale = MatMaxAbs(in A);
            double factorTol = (aScale + (double)1) * (double)dim * Consts.doubleEpsilon * (double)8;
            double maxFactorDiff = (double)0;
            for (int i = 0; i < dim; i++) {
                int prow = pR[i];
                for (int j = 0; j < dim; j++) {
                    double refVal = LUref[prow, j];
                    double blkVal = (j < i) ? L[i, j] : U[i, j];
                    maxFactorDiff = Unity.Mathematics.math.max(maxFactorDiff, Unity.Mathematics.math.abs(refVal - blkVal));
                }
            }
            if (!(maxFactorDiff <= factorTol) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1; Fail[1] = maxFactorDiff; Fail[2] = factorTol; Fail[3] = maxFactorDiff - factorTol;
            }
            Assert.IsTrue(maxFactorDiff <= factorTol);

            // (c) backward error: ||P A - L U|| for blocked vs reference (rebuilt from compact form).
            double resBlocked = ResidualPALU(ref arena, in A, in L, in U, in pB);

            var refL = arena.doubleIdentityMat(dim);
            var refU = arena.doubleMat(dim, dim);
            for (int i = 0; i < dim; i++) {
                int prow = pR[i];
                for (int j = 0; j < dim; j++) {
                    if (j < i) refL[i, j] = LUref[prow, j];
                    else       refU[i, j] = LUref[prow, j];
                }
            }
            double resRef = ResidualPALU(ref arena, in A, in refL, in refU, in pR);

            AssertResidualNotWorse(dim, aScale, resBlocked, resRef);

            pB.Dispose();
            pR.Dispose();
        }

        // Point (3): ill-conditioned residual comparison ONLY (no pivot/factor identity, since a
        // rounding-induced pivot flip is legitimate on a near-degenerate matrix).
        private void IllConditionedResidual(ref Arena arena, in doubleMxN A)
        {
            int dim = A.M_Rows;

            var U = A.Copy();
            var L = arena.doubleIdentityMat(dim);
            var pB = new Pivot(dim, Allocator.Temp);
            bool okB = LU.luDecomposition(ref U, ref L, ref pB);
            Assert.IsTrue(okB);
            Assert.IsFalse(Analysis_OP.isAnyNan(in U));
            Assert.IsFalse(Analysis_OP.isAnyNan(in L));

            var LUref = A.Copy();
            var pR = new Pivot(dim, Allocator.Temp);
            bool okR = LU.luDecompositionInpl(ref LUref, ref pR);
            Assert.IsTrue(okR);

            double resBlocked = ResidualPALU(ref arena, in A, in L, in U, in pB);

            var refL = arena.doubleIdentityMat(dim);
            var refU = arena.doubleMat(dim, dim);
            for (int i = 0; i < dim; i++) {
                int prow = pR[i];
                for (int j = 0; j < dim; j++) {
                    if (j < i) refL[i, j] = LUref[prow, j];
                    else       refU[i, j] = LUref[prow, j];
                }
            }
            double resRef = ResidualPALU(ref arena, in A, in refL, in refU, in pR);

            double aScale = MatMaxAbs(in A);
            AssertResidualNotWorse(dim, aScale, resBlocked, resRef);

            pB.Dispose();
            pR.Dispose();
        }

        // resBlocked must (i) stay within the O(n * sqrt(eps) * ||A||) backward-error ceiling (the true
        // backward error is O(n * eps * ||A||), well below this) and (ii) be within a small factor of
        // the unblocked reference residual — the accuracy-regression guard.
        private void AssertResidualNotWorse(int dim, double aScale, double resBlocked, double resRef)
        {
            double ceiling = (aScale + (double)1) * (double)dim * Consts.doubleSqrtEps;
            if (!(resBlocked <= ceiling) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1; Fail[1] = resBlocked; Fail[2] = ceiling; Fail[3] = resBlocked - ceiling;
            }
            Assert.IsTrue(resBlocked <= ceiling);

            double resFloor = (aScale + (double)1) * (double)dim * Consts.doubleEpsilon;
            double resLimit = (double)16 * resRef + resFloor;
            if (!(resBlocked <= resLimit) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1; Fail[1] = resBlocked; Fail[2] = resRef; Fail[3] = resBlocked - resLimit;
            }
            Assert.IsTrue(resBlocked <= resLimit);
        }

        // ||P A - L U||_max : apply the inverse row pivot to a copy of A (PA = LU convention, matching
        // AssertLU's usage) then compare against L*U.
        private double ResidualPALU(ref Arena arena, in doubleMxN A, in doubleMxN L, in doubleMxN U, in Pivot P)
        {
            var Aperm = A.Copy();
            P.ApplyInverseRow(ref Aperm);
            var shouldBeZero = Aperm - Linear_OP.dot(L, U);
            return Analysis_OP.MaxZeroError(shouldBeZero);
        }

        private void AssertPivotEqual(in Pivot a, in Pivot b, int dim)
        {
            for (int i = 0; i < dim; i++) {
                if (a[i] != b[i] && Fail[0] == (double)0)
                {
                    Fail[0] = (double)1; Fail[1] = (double)a[i]; Fail[2] = (double)b[i]; Fail[3] = (double)i;
                }
                Assert.IsTrue(a[i] == b[i]);
            }
        }

        // max |A[i,j]| — matrix magnitude, used to scale backward-stable tolerances.
        private double MatMaxAbs(in doubleMxN A)
        {
            double mx = (double)0;
            for (int i = 0; i < A.Length; i++)
                mx = Unity.Mathematics.math.max(mx, Unity.Mathematics.math.abs(A[i]));
            return mx;
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff
        private void AssertClose(double a, double b, double precision) {
            double diff = Unity.Mathematics.math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected, [3]=relative diff
        private void AssertCloseRel(double a, double b, double relPrecision) {
            double denom = Unity.Mathematics.math.max((double)1f, Unity.Mathematics.math.abs(b));
            double diff = Unity.Mathematics.math.abs(a - b) / denom;
            if (!(diff <= relPrecision) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= relPrecision);
        }

        // Fail layout: [0]=flag, [1]=got[i], [2]=expected[i], [3]=index cast to double
        private void AssertVecClose(in doubleN expected, in doubleN got, int dim, double precision) {
            for (int i = 0; i < dim; i++) {
                double diff = Unity.Mathematics.math.abs(expected[i] - got[i]);
                if (!(diff <= precision) && Fail[0] == (double)0)
                {
                    Fail[0] = (double)1;
                    Fail[1] = got[i];
                    Fail[2] = expected[i];
                    Fail[3] = (double)i;
                }
                Assert.IsTrue(diff <= precision);
            }
        }

        private void AssertLU(in doubleMxN A, in doubleMxN L, in doubleMxN U, bool pivoted) => AssertLU(in A, in L, in U, pivoted, 1E-6f);
        private void AssertLU(in doubleMxN A, in doubleMxN L, in doubleMxN U, bool pivoted, double precision)
        {
            doubleMxN shouldBeZero = A - Linear_OP.dot(L, U);

            if (Analysis_OP.isAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            // Fail layout: [1]=maxZeroError, [2]=precision, [3]=diff
            var zeroError = Analysis_OP.MaxZeroError(shouldBeZero);
            if (!(zeroError <= precision) && Fail[0] == (double)0)
            {
                Fail[0] = (double)1;
                Fail[1] = zeroError;
                Fail[2] = precision;
                Fail[3] = zeroError - precision;
            }
            Assert.IsTrue(Analysis_OP.isZero(in shouldBeZero, precision));
            Assert.IsTrue(Analysis_OP.isLowerTriangular(L, precision));
            Assert.IsTrue(Analysis_OP.isUpperTriangular(U, precision));

            if(pivoted)
            unsafe {
                var maxAbs = LinearAlgebra.Internal.Unsafe_OP.maxAbs(L.Data.Ptr, L.Length);

                if(maxAbs > 1f)
                    throw new System.Exception("TestJob: L has values greater than 1f");
            }
        }

    }

    public static Array GetEnums() {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void LUDecompTests(TestJob.TestType type)
    {
        var fail = new NativeArray<double>(4, Allocator.TempJob);
        try {
            new TestJob() { Type = type, Fail = fail }.Run();
            // Under Burst a failed in-job assert logs an exception and aborts the job without
            // throwing to the caller - surface the recorded diagnostics here as well.
            if (fail[0] != (double)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");

        }
        catch (Exception e) {
            if (fail[0] != (double)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally {
            fail.Dispose();
        }
    }

}
