using System;

using LinearAlgebra;
using LinearAlgebra.Gallery;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using UnityEngine;
using UnityEngine.TestTools;
using LinearAlgebra.Internal;

public class fProxyLUTests
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
            LUDecompSingularStatus,
            LUDecompPivotRequired,
            LUDeterminant,
            LUDeterminantGallery,
            LUReusePivot,
            SwapOPTest,
            LUSolveSystem,
            LUSolveSystemInPlace,
            // Blocked (level-3) LU path coverage (engages at M_Rows >= 256; LU_BLOCK=32).
            LUBlockedRefAccuracy256,
            LUBlockedRefAccuracy300,
            LUBlockedIllConditioned256,
            LUBlockedSingular256,
            LUBlockedSolve300,
            // Solver API rework (commit 2) coverage: safe decomp/decompNoPivot preserve A, and
            // solveInPlace's exit factor is a valid decompSolve input (bit-identical to fresh decomp).
            LUDecompVariantsPreserveA,
            LUSolveInPlaceExitIsUsableFactor,
            // Commit 2.5 hardening: solveInPlace driver short-circuit purity (singular input leaves
            // b_to_x bit-identical) + blocked-path (dim=256) A-preservation.
            LUSolveInPlaceShortCircuitPurity,
            LUDecompPreservesABlocked
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff/index
        public NativeArray<fProxy> Fail;

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
                case TestType.LUDecompSingularStatus:
                    LUDecompSingularStatus();
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
                case TestType.LUSolveSystemInPlace:
                    SolveSystemInPlace();
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
                case TestType.LUDecompVariantsPreserveA:
                    LUDecompVariantsPreserveA();
                    break;
                case TestType.LUSolveInPlaceExitIsUsableFactor:
                    LUSolveInPlaceExitIsUsableFactor();
                    break;
                case TestType.LUSolveInPlaceShortCircuitPurity:
                    LUSolveInPlaceShortCircuitPurity();
                    break;
                case TestType.LUDecompPreservesABlocked:
                    LUDecompPreservesABlocked();
                    break;

            }
        }

        private fProxyMxN GetRandomMatrix(ref Arena arena, int dim, fProxy min, fProxy max, uint seed) {

            var mat = arena.fProxyRandomMat(dim, dim, min, max, seed);

            return mat;
        }

        public void LUDecompIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var U = arena.fProxyIdentityMat(dim);
            var L = arena.fProxyIdentityMat(dim);

            var A = U.Copy();

            bool success = LU.decompNoPivot(in A, ref L, ref U);

            Assert.IsTrue(success);

            AssertLU(in A, in L, in U, false);

            arena.Dispose();
        }
        public void LUDecompRandomDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var U = arena.fProxyRandomDiagonalMat(dim, 1f, 3f);
            var L = arena.fProxyIdentityMat(dim);

            var A = U.Copy();

            bool success = LU.decompNoPivot(in A, ref L, ref U);

            Assert.IsTrue(success);

            AssertLU(in A, in L, in U, false);

            arena.Dispose();
        }

        public void LUDecompPredefined() {

            var arena = new Arena(Allocator.Persistent);

            var dim = 5;

            var U = arena.fProxyMat(dim);
            var L = arena.fProxyIdentityMat(dim);

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

            bool success = LU.decomp(in A, ref L, ref U, ref pivot);

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

            var U = arena.fProxyRandomMat(dim, dim, 1f, 10f, 314221);
            var L = arena.fProxyIdentityMat(dim);

            for(int d = 0; d < dim; d++)
                U[d, d] += 5f;

            var A = U.Copy();

            var pivot = new Pivot(dim, Allocator.Temp);

            bool success = LU.decomp(in A, ref L, ref U, ref pivot);

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
                var A = arena.fProxyMat(dim, dim);
                var L = arena.fProxyIdentityMat(dim);
                var U = arena.fProxyMat(dim, dim);

                bool noPivot = LU.decompNoPivot(in A, ref L, ref U);
                Assert.IsFalse(noPivot);
                Assert.IsFalse(Analysis.isAnyNan(in U));
                Assert.IsFalse(Analysis.isAnyNan(in L));

                var Ap = arena.fProxyMat(dim, dim);
                var Lp = arena.fProxyIdentityMat(dim);
                var Up = arena.fProxyMat(dim, dim);
                var pivot = new Pivot(dim, Allocator.Temp);
                bool pivoted = LU.decomp(in Ap, ref Lp, ref Up, ref pivot);
                Assert.IsFalse(pivoted);
                Assert.IsFalse(Analysis.isAnyNan(in Up));
                Assert.IsFalse(Analysis.isAnyNan(in Lp));

                var LUmat = arena.fProxyMat(dim, dim);
                bool inPlace = LU.decompInPlace(ref LUmat, ref pivot);
                Assert.IsFalse(inPlace);
                Assert.IsFalse(Analysis.isAnyNan(in LUmat));

                pivot.Dispose();
            }

            // Case 2: two identical rows -> rank deficient -> all variants return false.
            {
                var A = arena.fProxyRandomMat(dim, dim, 1f, 10f, 8821);
                // force diagonal dominance so only the duplicated rows cause singularity
                for (int d = 0; d < dim; d++)
                    A[d, d] += 20f;
                // make row 5 an exact copy of row 2
                for (int c = 0; c < dim; c++)
                    A[5, c] = A[2, c];

                var L = arena.fProxyIdentityMat(dim);
                var U = arena.fProxyMat(dim, dim);

                bool noPivot = LU.decompNoPivot(in A, ref L, ref U);
                Assert.IsFalse(noPivot);
                Assert.IsFalse(Analysis.isAnyNan(in U));
                Assert.IsFalse(Analysis.isAnyNan(in L));

                var Lp = arena.fProxyIdentityMat(dim);
                var Up = arena.fProxyMat(dim, dim);
                var pivot = new Pivot(dim, Allocator.Temp);
                bool pivoted = LU.decomp(in A, ref Lp, ref Up, ref pivot);
                Assert.IsFalse(pivoted);
                Assert.IsFalse(Analysis.isAnyNan(in Up));
                Assert.IsFalse(Analysis.isAnyNan(in Lp));

                var LUmat = A.Copy();
                bool inPlace = LU.decompInPlace(ref LUmat, ref pivot);
                Assert.IsFalse(inPlace);
                Assert.IsFalse(Analysis.isAnyNan(in LUmat));

                pivot.Dispose();
            }

            // Case 3: [[0,1],[1,0]] : no-pivot fails on zero leading pivot,
            // but in-place (with partial pivoting) succeeds.
            {
                var A = arena.fProxyMat(2, 2);
                A[0, 0] = 0f; A[0, 1] = 1f;
                A[1, 0] = 1f; A[1, 1] = 0f;

                var L = arena.fProxyIdentityMat(2);
                var Unp = arena.fProxyMat(2, 2);

                bool noPivot = LU.decompNoPivot(in A, ref L, ref Unp);
                Assert.IsFalse(noPivot);

                var LUmat = A.Copy();
                var pivot = new Pivot(2, Allocator.Temp);
                bool inPlace = LU.decompInPlace(ref LUmat, ref pivot);
                Assert.IsTrue(inPlace);
                Assert.IsFalse(Analysis.isAnyNan(in LUmat));

                pivot.Dispose();
            }

            arena.Dispose();
        }

        // Stage-3 direct-solve-status coverage: a singular matrix must report
        // DirectSolveStatus.Singular (not just a falsy implicit-bool) from all three LU
        // decomposition entry points, and DirectSolveInfo.Solved must be false.
        public void LUDecompSingularStatus()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;
            var A = arena.fProxyMat(dim, dim); // zero matrix -> singular
            var L = arena.fProxyIdentityMat(dim);
            var U = arena.fProxyMat(dim, dim);

            DirectSolveInfo noPivotInfo = LU.decompNoPivot(in A, ref L, ref U);
            Assert.IsTrue(noPivotInfo.status == DirectSolveStatus.Singular);
            Assert.IsFalse(noPivotInfo.Solved);
            Assert.IsFalse(noPivotInfo);

            var Lp = arena.fProxyIdentityMat(dim);
            var Up = arena.fProxyMat(dim, dim);
            var pivot = new Pivot(dim, Allocator.Temp);
            DirectSolveInfo pivotedInfo = LU.decomp(in A, ref Lp, ref Up, ref pivot);
            Assert.IsTrue(pivotedInfo.status == DirectSolveStatus.Singular);
            Assert.IsFalse(pivotedInfo.Solved);

            var LUmat = arena.fProxyMat(dim, dim);
            DirectSolveInfo inPlaceInfo = LU.decompInPlace(ref LUmat, ref pivot);
            Assert.IsTrue(inPlaceInfo.status == DirectSolveStatus.Singular);
            Assert.IsFalse(inPlaceInfo.Solved);

            pivot.Dispose();
            arena.Dispose();
        }

        public void LUDecompPivotRequired()
        {
            var arena = new Arena(Allocator.Persistent);

            // Case A: 3x3 with A[0,0] == 0 but nonsingular, requires pivoting.
            {
                int dim = 3;
                var A = arena.fProxyMat(dim, dim);
                // A[0,0] == 0 forces a row swap
                A[0, 0] = 0f; A[0, 1] = 2f; A[0, 2] = 1f;
                A[1, 0] = 1f; A[1, 1] = 1f; A[1, 2] = 1f;
                A[2, 0] = 2f; A[2, 1] = 1f; A[2, 2] = 0f;

                var x_Known = arena.fProxyVec(dim);
                x_Known[0] = 3f; x_Known[1] = -2f; x_Known[2] = 5f;

                var b = Blas.dot(A, x_Known);

                var LUmat = A.Copy();
                var pivot = new Pivot(dim, Allocator.Temp);

                bool success = LU.decompInPlace(ref LUmat, ref pivot);
                Assert.IsTrue(success);

                var x_Solved = b.Copy();
                LU.decompSolve(ref LUmat, in pivot, ref x_Solved);

                Assert.IsFalse(Analysis.isAnyNan(in x_Solved));

                AssertVecClose(in x_Known, in x_Solved, dim, 1E-3f);

                pivot.Dispose();
            }

            // Case B: permutation contains a 3-cycle (not an involution).
            // Construct A so partial pivoting yields a cyclic permutation. We make the
            // sub-diagonal magnitudes strictly increasing down each column so the pivot
            // search always selects the last remaining row -> a long cycle, not pair swaps.
            {
                int dim = 4;
                var A = arena.fProxyMat(dim, dim);
                A[0, 0] = 1f; A[0, 1] = 2f; A[0, 2] = 0f; A[0, 3] = 1f;
                A[1, 0] = 2f; A[1, 1] = 1f; A[1, 2] = 3f; A[1, 3] = 0f;
                A[2, 0] = 4f; A[2, 1] = 0f; A[2, 2] = 1f; A[2, 3] = 2f;
                A[3, 0] = 8f; A[3, 1] = 3f; A[3, 2] = 2f; A[3, 3] = 1f;

                var x_Known = arena.fProxyVec(dim);
                x_Known[0] = 1f; x_Known[1] = -3f; x_Known[2] = 2f; x_Known[3] = 4f;

                var b = Blas.dot(A, x_Known);

                var LUmat = A.Copy();
                var pivot = new Pivot(dim, Allocator.Temp);

                bool success = LU.decompInPlace(ref LUmat, ref pivot);
                Assert.IsTrue(success);

                // Verify the permutation is not a simple involution (P applied twice != identity),
                // i.e. it really contains a cycle of length > 2.
                bool isInvolution = true;
                for (int i = 0; i < dim; i++)
                    if (pivot[pivot[i]] != i)
                        isInvolution = false;
                Assert.IsFalse(isInvolution);

                var x_Solved = b.Copy();
                LU.decompSolve(ref LUmat, in pivot, ref x_Solved);

                Assert.IsFalse(Analysis.isAnyNan(in x_Solved));

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
                var I = arena.fProxyIdentityMat(dim);
                var pivot = new Pivot(dim, Allocator.Temp);

                bool success = LU.decompInPlace(ref I, ref pivot);
                Assert.IsTrue(success);

                fProxy det = LU.determinant(in I, in pivot);
                AssertClose(det, (fProxy)1f, 1E-4f);

                pivot.Dispose();
            }

            // diagonal -> det = product of diagonal
            {
                int dim = 4;
                var D = arena.fProxyMat(dim, dim);
                D[0, 0] = 2f;
                D[1, 1] = -3f;
                D[2, 2] = 0.5f;
                D[3, 3] = 4f;
                fProxy expected = 2f * -3f * 0.5f * 4f; // -12

                var pivot = new Pivot(dim, Allocator.Temp);
                bool success = LU.decompInPlace(ref D, ref pivot);
                Assert.IsTrue(success);

                fProxy det = LU.determinant(in D, in pivot);
                AssertCloseRel(det, expected, 1E-4f);

                pivot.Dispose();
            }

            // 3x3 with known determinant requiring a row swap (A[0,0]==0).
            // A = [[0,2,1],[1,1,1],[2,1,0]]; det = 3 (hand computed, nonsingular).
            {
                int dim = 3;
                var A = arena.fProxyMat(dim, dim);
                A[0, 0] = 0f; A[0, 1] = 2f; A[0, 2] = 1f;
                A[1, 0] = 1f; A[1, 1] = 1f; A[1, 2] = 1f;
                A[2, 0] = 2f; A[2, 1] = 1f; A[2, 2] = 0f;

                var pivot = new Pivot(dim, Allocator.Temp);
                bool success = LU.decompInPlace(ref A, ref pivot);
                Assert.IsTrue(success);

                fProxy det = LU.determinant(in A, in pivot);
                AssertCloseRel(det, (fProxy)3f, 1E-4f);

                pivot.Dispose();
            }

            // permutation matrix -> det = +-1 matching swap parity.
            // single transposition (rows 0 and 2 swapped) -> det = -1.
            {
                int dim = 3;
                var P = arena.fProxyMat(dim, dim);
                P[0, 2] = 1f;
                P[1, 1] = 1f;
                P[2, 0] = 1f;

                var pivot = new Pivot(dim, Allocator.Temp);
                bool success = LU.decompInPlace(ref P, ref pivot);
                Assert.IsTrue(success);

                fProxy det = LU.determinant(in P, in pivot);
                AssertCloseRel(det, (fProxy)(-1f), 1E-4f);

                pivot.Dispose();
            }

            arena.Dispose();
        }

        // GALLERY KNOWN-ANSWER: famous unit-determinant matrices. det is computed via
        // LU.decompInPlace + LU.determinant (the file's established sequence).
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
                var A = arena.fProxyPascal(dim);
                var pivot = new Pivot(dim, Allocator.Temp);
                bool success = LU.decompInPlace(ref A, ref pivot);
                Assert.IsTrue(success);

                fProxy det = LU.determinant(in A, in pivot);
                AssertCloseRel(det, (fProxy)1f, 1E-4f);

                pivot.Dispose();
            }

            // MinIJ(5): det = 1
            {
                int dim = 5;
                var A = arena.fProxyMinIJ(dim);
                var pivot = new Pivot(dim, Allocator.Temp);
                bool success = LU.decompInPlace(ref A, ref pivot);
                Assert.IsTrue(success);

                fProxy det = LU.determinant(in A, in pivot);
                AssertCloseRel(det, (fProxy)1f, 1E-4f);

                pivot.Dispose();
            }

            // Frank(5): det = 1
            {
                int dim = 5;
                var A = arena.fProxyFrank(dim);
                var pivot = new Pivot(dim, Allocator.Temp);
                bool success = LU.decompInPlace(ref A, ref pivot);
                Assert.IsTrue(success);

                fProxy det = LU.determinant(in A, in pivot);
                AssertCloseRel(det, (fProxy)1f, 1E-4f);

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
            var A1 = arena.fProxyRandomMat(dim, dim, -5f, 5f, 7777);
            for (int d = 0; d < dim; d++)
                A1[d, d] += 15f;
            var LU1 = A1.Copy();
            bool s1 = LU.decompInPlace(ref LU1, ref pivot);
            Assert.IsTrue(s1);

            // Second decomposition reuses the SAME pivot object; Reset() must clean it.
            var A2 = arena.fProxyRandomMat(dim, dim, -5f, 5f, 9999);
            for (int d = 0; d < dim; d++)
                A2[d, d] += 15f;

            var x_Known = arena.fProxyVec(dim);
            for (int i = 0; i < dim; i++)
                x_Known[i] = (fProxy)(i + 1);

            var b = Blas.dot(A2, x_Known);

            var LU2 = A2.Copy();
            bool s2 = LU.decompInPlace(ref LU2, ref pivot);
            Assert.IsTrue(s2);

            var x_Solved = b.Copy();
            LU.decompSolve(ref LU2, in pivot, ref x_Solved);

            Assert.IsFalse(Analysis.isAnyNan(in x_Solved));

            AssertVecClose(in x_Known, in x_Solved, dim, 1E-3f);

            pivot.Dispose();

            arena.Dispose();
        }

        public void SwapOPTest()
        {
            var arena = new Arena(Allocator.Persistent);

            // Swap.Rows with default start/end swaps full rows.
            {
                int dim = 3;
                var mat = arena.fProxyMat(dim, dim);
                for (int r = 0; r < dim; r++)
                    for (int c = 0; c < dim; c++)
                        mat[r, c] = (fProxy)(r * 10 + c);

                Swap.Rows(ref mat, 0, 1);

                for (int c = 0; c < dim; c++) {
                    AssertClose(mat[0, c], (fProxy)(10 + c), 1E-6f);
                    AssertClose(mat[1, c], (fProxy)(0 + c), 1E-6f);
                    AssertClose(mat[2, c], (fProxy)(20 + c), 1E-6f);
                }
            }

            // Swap.Columns with explicit start/end swaps only that row-range.
            {
                int dim = 4;
                var mat = arena.fProxyMat(dim, dim);
                for (int r = 0; r < dim; r++)
                    for (int c = 0; c < dim; c++)
                        mat[r, c] = (fProxy)(r * 10 + c);

                // swap columns 0 and 1 only for rows [1,3)
                Swap.Columns(ref mat, 0, 1, 1, 3);

                AssertClose(mat[0, 0], (fProxy)(0), 1E-6f);
                AssertClose(mat[0, 1], (fProxy)(1), 1E-6f);
                AssertClose(mat[3, 0], (fProxy)(30), 1E-6f);
                AssertClose(mat[3, 1], (fProxy)(31), 1E-6f);

                AssertClose(mat[1, 0], (fProxy)(11), 1E-6f);
                AssertClose(mat[1, 1], (fProxy)(10), 1E-6f);
                AssertClose(mat[2, 0], (fProxy)(21), 1E-6f);
                AssertClose(mat[2, 1], (fProxy)(20), 1E-6f);

                AssertClose(mat[1, 2], (fProxy)(12), 1E-6f);
                AssertClose(mat[2, 3], (fProxy)(23), 1E-6f);
            }

            arena.Dispose();
        }

        public void SolveSystem() {

            var arena = new Arena(Allocator.Persistent);

            int dim = 512;

            var A = arena.fProxyRandomMat(dim, dim, -10f, 10f, 314221);

            for (int d = 0; d < dim; d++) {
                A[d, d] *= 2f;
                if (Unity.Mathematics.math.abs(A[d, d]) < 0.01f)
                    A[d, d] *= 10f;
            }

            var x_Known = arena.fProxyRandomVec(dim, 1f, 10f, 901);

            var b = Blas.dot(A, x_Known);

            var U = arena.fProxyMat(dim, dim);
            var L = arena.fProxyIdentityMat(dim);

            var pivot = new Pivot(dim, Allocator.Temp);

            bool success = LU.decomp(in A, ref L, ref U, ref pivot);

            Assert.IsTrue(success);

            var x_Solved = b.Copy();

            LU.decompSolve(ref L, ref U, in pivot, ref x_Solved);

            if (Analysis.isAnyNan(in x_Solved))
                throw new System.Exception("TestJob: NaN detected");

            var zeroError = Analysis.MaxZeroError(x_Known - x_Solved);

            // Fail layout: [1]=zeroError, [2]=limit 1E-3, [3]=diff
            if (!(zeroError < (fProxy)1E-03f) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = zeroError;
                Fail[2] = (fProxy)1E-03f;
                Fail[3] = zeroError - (fProxy)1E-03f;
            }
            Assert.IsTrue(zeroError < 1E-03f);

            pivot.Dispose();

            arena.Dispose();
        }

        // Same Fail-layout convention as SolveSystem above (see there).
        public void SolveSystemInPlace() {

            var arena = new Arena(Allocator.Persistent);

            int dim = 512;

            var A = arena.fProxyRandomMat(dim, dim, -10f, 10f, 314221);

            for (int d = 0; d < dim; d++) {
                A[d, d] *= 2f;
                if (Unity.Mathematics.math.abs(A[d, d]) < 0.01f)
                    A[d, d] *= 10f;
            }

            var x_Known = arena.fProxyRandomVec(dim, 1f, 10f, 901);

            var b = Blas.dot(A, x_Known);

            var LUmat = A.Copy();

            var pivot = new Pivot(dim, Allocator.Temp);

            bool success = LU.decompInPlace(ref LUmat, ref pivot);

            Assert.IsTrue(success);

            var x_Solved = b.Copy();

            LU.decompSolve(ref LUmat, in pivot, ref x_Solved);

            if (Analysis.isAnyNan(in x_Solved))
                throw new System.Exception("TestJob: NaN detected");

            var zeroError = Analysis.MaxZeroError(x_Known - x_Solved);

            if (!(zeroError < (fProxy)1E-03f) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = zeroError;
                Fail[2] = (fProxy)1E-03f;
                Fail[3] = zeroError - (fProxy)1E-03f;
            }
            Assert.IsTrue(zeroError < 1E-03f);

            pivot.Dispose();

            arena.Dispose();
        }

        // ================================================================================
        // BLOCKED (level-3) LU coverage.
        //
        // LU.decomp(in A, ref L, ref U, ref P) switches to the LAPACK-style right-looking
        // blocked (compact-WY GEMM trailing update) path at M_Rows >= LU_BLOCK_MIN_N = 8*32 = 256;
        // below that it runs the plain unblocked rank-1 sweep. The blocked path is DESIGNED to keep
        // the partial-pivoting sequence bit-identical to the unblocked form, so it must produce the
        // SAME pivot array and (within GEMM summation-order rounding) the same L/U as the independent,
        // untouched, level-2 compact factorization LU.decompInPlace(ref LU, ref P) — which is
        // used here as the reference ORACLE for both correctness and accuracy.
        //
        // In the in-place compact form: factor row i lives at physical row P[i]; LU[P[i], j] with j < i
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
            var A = arena.fProxyLehmer(dim);
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

            var A = arena.fProxyRandomMat(dim, dim, 1f, 10f, 55221);
            // strong diagonal dominance so ONLY the duplicated rows cause singularity
            for (int d = 0; d < dim; d++)
                A[d, d] += (fProxy)(2 * dim);
            // make row 137 an exact copy of row 42
            for (int c = 0; c < dim; c++)
                A[137, c] = A[42, c];

            var U = arena.fProxyMat(dim, dim);
            var L = arena.fProxyIdentityMat(dim);
            var pivot = new Pivot(dim, Allocator.Temp);

            bool ok = LU.decomp(in A, ref L, ref U, ref pivot);

            Assert.IsFalse(ok);
            Assert.IsFalse(Analysis.isAnyNan(in U));
            Assert.IsFalse(Analysis.isAnyNan(in L));
            Assert.IsFalse(Analysis.isAnyInf(in U));
            Assert.IsFalse(Analysis.isAnyInf(in L));

            pivot.Dispose();
            arena.Dispose();
        }

        // (5) Solve round-trip N=300 (non-aligned last panel) using the separate-L/U blocked path.
        // Same recipe / tolerance as the existing dim=512 SolveSystem test.
        public void LUBlockedSolve300()
        {
            var arena = new Arena(Allocator.Persistent);
            int dim = 300;

            var A = arena.fProxyRandomMat(dim, dim, -10f, 10f, 314221);

            for (int d = 0; d < dim; d++) {
                A[d, d] *= 2f;
                if (Unity.Mathematics.math.abs(A[d, d]) < 0.01f)
                    A[d, d] *= 10f;
            }

            var x_Known = arena.fProxyRandomVec(dim, 1f, 10f, 901);

            var b = Blas.dot(A, x_Known);

            var U = arena.fProxyMat(dim, dim);
            var L = arena.fProxyIdentityMat(dim);

            var pivot = new Pivot(dim, Allocator.Temp);

            bool success = LU.decomp(in A, ref L, ref U, ref pivot);

            Assert.IsTrue(success);

            var x_Solved = b.Copy();

            LU.decompSolve(ref L, ref U, in pivot, ref x_Solved);

            if (Analysis.isAnyNan(in x_Solved))
                throw new System.Exception("TestJob: NaN detected");

            var zeroError = Analysis.MaxZeroError(x_Known - x_Solved);

            // x-accuracy is condition-number-amplified (NOT the backward error, which stays ~eps·‖A‖).
            // This fixed random draw at n=300 (cond a few·10^3) lands at ~1.2e-3 max error in float —
            // just over the 1e-3 the dim=512 SolveSystem happens to hit — so use a per-precision band:
            // generous for float, tight for double. Deterministic (fixed seeds), so not flaky.
            fProxy xtol = IsDouble() ? (fProxy)1E-8 : (fProxy)3E-3f;

            if (!(zeroError < xtol) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = zeroError;
                Fail[2] = xtol;
                Fail[3] = zeroError - xtol;
            }
            Assert.IsTrue(zeroError < xtol);

            pivot.Dispose();
            arena.Dispose();
        }

        // ================================================================================
        // Solver API rework (commit 2): safe decomp/decompNoPivot preserve A; solveInPlace's exit
        // factor is a usable decompSolve input.
        // ================================================================================

        // LU.decomp and LU.decompNoPivot must not modify A. Checksum (position-weighted sum, so a
        // permutation or a single altered entry both trip it) before/after each call.
        void LUDecompVariantsPreserveA()
        {
            var arena = new Arena(Allocator.Persistent);
            int dim = 9;

            // LU.decomp (pivoted, safe)
            {
                var A = arena.fProxyRandomMat(dim, dim, -5f, 5f, 424242);
                for (int d = 0; d < dim; d++) A[d, d] += 15f;
                fProxy checksumBefore = Checksum(in A);

                var L = arena.fProxyIdentityMat(dim);
                var U = arena.fProxyMat(dim, dim);
                var pivot = new Pivot(dim, Allocator.Temp);
                bool ok = LU.decomp(in A, ref L, ref U, ref pivot);
                Assert.IsTrue(ok);

                AssertExactEqual(checksumBefore, Checksum(in A));
                pivot.Dispose();
            }

            // LU.decompNoPivot (safe)
            {
                var A = arena.fProxyRandomMat(dim, dim, -5f, 5f, 535353);
                for (int d = 0; d < dim; d++) A[d, d] += 15f;
                fProxy checksumBefore = Checksum(in A);

                var L = arena.fProxyIdentityMat(dim);
                var U = arena.fProxyMat(dim, dim);
                bool ok = LU.decompNoPivot(in A, ref L, ref U);
                Assert.IsTrue(ok);

                AssertExactEqual(checksumBefore, Checksum(in A));
            }

            arena.Dispose();
        }

        // LU.solveInPlace's exit (A_to_LU, P) must be a valid decompSolve input: solving a SECOND
        // right-hand side through it must be bit-identical to a completely independent
        // decompInPlace + decompSolve on the same original matrix.
        void LUSolveInPlaceExitIsUsableFactor()
        {
            var arena = new Arena(Allocator.Persistent);
            int dim = 10;

            // Non-trivial pivoting: MakeWellConditionedPivoting row-reverses a diagonally-dominant
            // matrix, so partial pivoting must undo a genuine (non-identity) permutation. A plain
            // diagonally-dominant fill would swap no rows, hiding any pivot-indexing bug in the fused
            // solveInPlace vs the independent decompInPlace+decompSolve oracle.
            var A = MakeWellConditionedPivoting(ref arena, dim, 314159);

            var xKnown1 = arena.fProxyRandomVec(dim, 1f, 5f, 111);
            var b1 = Blas.dot(A, xKnown1);
            var xKnown2 = arena.fProxyRandomVec(dim, 1f, 5f, 222);
            var b2 = Blas.dot(A, xKnown2);

            // path under test: solveInPlace (first RHS), then decompSolve (second RHS) off its exit.
            var Afused = A.Copy();
            var pivotFused = new Pivot(dim, Allocator.Temp);
            var x1 = b1.Copy();
            var info = LU.solveInPlace(ref Afused, ref pivotFused, ref x1);
            Assert.IsTrue(info.Solved);

            var x2 = b2.Copy();
            LU.decompSolve(ref Afused, in pivotFused, ref x2);

            // oracle: fresh decompInPlace + decompSolve on an independent copy, same second RHS.
            var Aref = A.Copy();
            var pivotRef = new Pivot(dim, Allocator.Temp);
            var infoRef = LU.decompInPlace(ref Aref, ref pivotRef);
            Assert.IsTrue(infoRef.Solved);

            var x2ref = b2.Copy();
            LU.decompSolve(ref Aref, in pivotRef, ref x2ref);

            for (int i = 0; i < dim; i++)
                AssertExactEqual(x2ref[i], x2[i]);

            pivotFused.Dispose();
            pivotRef.Dispose();
            arena.Dispose();
        }

        // (2a) Driver short-circuit purity: LU.solveInPlace on a SINGULAR matrix must (a) report the
        // Singular failure status and (b) leave b_to_x BIT-IDENTICAL to its pre-call snapshot. This
        // guards the `if (!info.Solved) return info;` early return in the fused GESV driver: if that
        // short-circuit were removed, decompSolve would run on the garbage/partial factor and corrupt
        // b_to_x.
        void LUSolveInPlaceShortCircuitPurity()
        {
            var arena = new Arena(Allocator.Persistent);
            int dim = 8;

            // Singular: two identical rows (row 5 == row 2), diagonally boosted so ONLY that
            // duplication causes singularity (reuses LUDecompSingular's construction).
            var A = arena.fProxyRandomMat(dim, dim, 1f, 10f, 8821);
            for (int d = 0; d < dim; d++) A[d, d] += 20f;
            for (int c = 0; c < dim; c++) A[5, c] = A[2, c];

            var b = arena.fProxyRandomVec(dim, -3f, 3f, 246810);
            var bSnapshot = b.Copy(); // capture BEFORE the call

            var pivot = new Pivot(dim, Allocator.Temp);
            DirectSolveInfo info = LU.solveInPlace(ref A, ref pivot, ref b);

            // (a) failure status forwarded; not Solved.
            if (info.status != DirectSolveStatus.Singular && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = (fProxy)(int)info.status;
                Fail[2] = (fProxy)(int)DirectSolveStatus.Singular; Fail[3] = (fProxy)0;
            }
            Assert.IsTrue(info.status == DirectSolveStatus.Singular);
            Assert.IsFalse(info.Solved);

            // (b) b_to_x untouched: bit-identical (==, not within-tolerance) to its snapshot.
            for (int i = 0; i < dim; i++)
                AssertExactEqual(bSnapshot[i], b[i]);

            pivot.Dispose();
            arena.Dispose();
        }

        // (2f-i) Blocked-path A-preservation: LU.decomp at dim=256 engages the level-3 blocked
        // (compact-WY GEMM trailing-update) path (LU_BLOCK_MIN_N = 8*32 = 256); it still must not
        // modify A. Checksum (position-weighted) before/after. The existing LUDecompVariantsPreserveA
        // only reaches the unblocked path (dim=9).
        void LUDecompPreservesABlocked()
        {
            var arena = new Arena(Allocator.Persistent);
            int dim = 256;

            var A = MakeWellConditionedPivoting(ref arena, dim, 424243);
            fProxy checksumBefore = Checksum(in A);

            var L = arena.fProxyIdentityMat(dim);
            var U = arena.fProxyMat(dim, dim);
            var pivot = new Pivot(dim, Allocator.Temp);
            bool ok = LU.decomp(in A, ref L, ref U, ref pivot);
            Assert.IsTrue(ok);

            AssertExactEqual(checksumBefore, Checksum(in A));

            pivot.Dispose();
            arena.Dispose();
        }

        // Position-weighted sum: differs if ANY entry changes or two entries are transposed.
        private fProxy Checksum(in fProxyMxN M)
        {
            fProxy s = (fProxy)0;
            for (int i = 0; i < M.Length; i++)
                s += M[i] * (fProxy)(i + 1);
            return s;
        }

        // Fail layout: [1]=got, [2]=expected, [3]=diff
        private void AssertExactEqual(fProxy expected, fProxy got)
        {
            if (got != expected && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = got;
                Fail[2] = expected;
                Fail[3] = got - expected;
            }
            Assert.IsTrue(got == expected);
        }

        // true only when fProxy expands to double (doubleEpsilon ≈ 2.2e-16 < 1e-10).
        private bool IsDouble() => (double)Consts.fProxyEpsilon < 1e-10;

        // Well-conditioned input that forces a NONTRIVIAL but UNAMBIGUOUS pivot sequence: a strongly
        // (row+column) diagonally-dominant random matrix (each column has one big entry, all others
        // small) with its rows reversed. Partial pivoting must undo the reversal; the huge column-wise
        // gap (~2*dim vs ~1) means the argmax is robust to GEMM-vs-scalar summation-order rounding, so
        // the blocked and unblocked factorizations pick the SAME pivots for BOTH float and double.
        private fProxyMxN MakeWellConditionedPivoting(ref Arena arena, int dim, uint seed)
        {
            var A = arena.fProxyRandomMat(dim, dim, -1f, 1f, seed);
            for (int d = 0; d < dim; d++)
                A[d, d] += (fProxy)(2 * dim);
            for (int i = 0; i < dim / 2; i++)
                Swap.Rows(ref A, i, dim - 1 - i);
            return A;
        }

        // Points (1)/(2): blocked LU.decomp (in A, ref L, ref U, ref P) vs unblocked compact LU.decompInPlace (2-arg) oracle.
        // Asserts identical pivots, matching L/U factors, and no backward-error regression.
        private void BlockedVsReference(ref Arena arena, in fProxyMxN A)
        {
            int dim = A.M_Rows;

            // --- blocked path (separate L, U, P) ---
            var U = arena.fProxyMat(dim, dim);
            var L = arena.fProxyIdentityMat(dim);
            var pB = new Pivot(dim, Allocator.Temp);
            bool okB = LU.decomp(in A, ref L, ref U, ref pB);
            Assert.IsTrue(okB);
            Assert.IsFalse(Analysis.isAnyNan(in U));
            Assert.IsFalse(Analysis.isAnyNan(in L));

            // --- reference: independent unblocked compact in-place factorization ---
            var LUref = A.Copy();
            var pR = new Pivot(dim, Allocator.Temp);
            bool okR = LU.decompInPlace(ref LUref, ref pR);
            Assert.IsTrue(okR);

            // (a) pivot arrays identical elementwise
            AssertPivotEqual(in pB, in pR, dim);

            // (b) factor accuracy: blocked L (strict lower) & U (upper) match the reference compact
            //     form at physical row pR[i]. Absolute tolerance scaled by matrix magnitude, matrix
            //     size (accumulation length) and machine eps — loose for float, tight for double.
            fProxy aScale = MatMaxAbs(in A);
            fProxy factorTol = (aScale + (fProxy)1) * (fProxy)dim * Consts.fProxyEpsilon * (fProxy)8;
            fProxy maxFactorDiff = (fProxy)0;
            for (int i = 0; i < dim; i++) {
                int prow = pR[i];
                for (int j = 0; j < dim; j++) {
                    fProxy refVal = LUref[prow, j];
                    fProxy blkVal = (j < i) ? L[i, j] : U[i, j];
                    maxFactorDiff = Unity.Mathematics.math.max(maxFactorDiff, Unity.Mathematics.math.abs(refVal - blkVal));
                }
            }
            if (!(maxFactorDiff <= factorTol) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = maxFactorDiff; Fail[2] = factorTol; Fail[3] = maxFactorDiff - factorTol;
            }
            Assert.IsTrue(maxFactorDiff <= factorTol);

            // (c) backward error: ||P A - L U|| for blocked vs reference (rebuilt from compact form).
            fProxy resBlocked = ResidualPALU(ref arena, in A, in L, in U, in pB);

            var refL = arena.fProxyIdentityMat(dim);
            var refU = arena.fProxyMat(dim, dim);
            for (int i = 0; i < dim; i++) {
                int prow = pR[i];
                for (int j = 0; j < dim; j++) {
                    if (j < i) refL[i, j] = LUref[prow, j];
                    else       refU[i, j] = LUref[prow, j];
                }
            }
            fProxy resRef = ResidualPALU(ref arena, in A, in refL, in refU, in pR);

            AssertResidualNotWorse(dim, aScale, resBlocked, resRef);

            pB.Dispose();
            pR.Dispose();
        }

        // Point (3): ill-conditioned residual comparison ONLY (no pivot/factor identity, since a
        // rounding-induced pivot flip is legitimate on a near-degenerate matrix).
        private void IllConditionedResidual(ref Arena arena, in fProxyMxN A)
        {
            int dim = A.M_Rows;

            var U = arena.fProxyMat(dim, dim);
            var L = arena.fProxyIdentityMat(dim);
            var pB = new Pivot(dim, Allocator.Temp);
            bool okB = LU.decomp(in A, ref L, ref U, ref pB);
            Assert.IsTrue(okB);
            Assert.IsFalse(Analysis.isAnyNan(in U));
            Assert.IsFalse(Analysis.isAnyNan(in L));

            var LUref = A.Copy();
            var pR = new Pivot(dim, Allocator.Temp);
            bool okR = LU.decompInPlace(ref LUref, ref pR);
            Assert.IsTrue(okR);

            fProxy resBlocked = ResidualPALU(ref arena, in A, in L, in U, in pB);

            var refL = arena.fProxyIdentityMat(dim);
            var refU = arena.fProxyMat(dim, dim);
            for (int i = 0; i < dim; i++) {
                int prow = pR[i];
                for (int j = 0; j < dim; j++) {
                    if (j < i) refL[i, j] = LUref[prow, j];
                    else       refU[i, j] = LUref[prow, j];
                }
            }
            fProxy resRef = ResidualPALU(ref arena, in A, in refL, in refU, in pR);

            fProxy aScale = MatMaxAbs(in A);
            AssertResidualNotWorse(dim, aScale, resBlocked, resRef);

            pB.Dispose();
            pR.Dispose();
        }

        // resBlocked must (i) stay within the O(n * sqrt(eps) * ||A||) backward-error ceiling (the true
        // backward error is O(n * eps * ||A||), well below this) and (ii) be within a small factor of
        // the unblocked reference residual — the accuracy-regression guard.
        private void AssertResidualNotWorse(int dim, fProxy aScale, fProxy resBlocked, fProxy resRef)
        {
            fProxy ceiling = (aScale + (fProxy)1) * (fProxy)dim * Consts.fProxySqrtEps;
            if (!(resBlocked <= ceiling) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = resBlocked; Fail[2] = ceiling; Fail[3] = resBlocked - ceiling;
            }
            Assert.IsTrue(resBlocked <= ceiling);

            fProxy resFloor = (aScale + (fProxy)1) * (fProxy)dim * Consts.fProxyEpsilon;
            fProxy resLimit = (fProxy)16 * resRef + resFloor;
            if (!(resBlocked <= resLimit) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = resBlocked; Fail[2] = resRef; Fail[3] = resBlocked - resLimit;
            }
            Assert.IsTrue(resBlocked <= resLimit);
        }

        // ||P A - L U||_max : apply the inverse row pivot to a copy of A (PA = LU convention, matching
        // AssertLU's usage) then compare against L*U.
        private fProxy ResidualPALU(ref Arena arena, in fProxyMxN A, in fProxyMxN L, in fProxyMxN U, in Pivot P)
        {
            var Aperm = A.Copy();
            P.ApplyInverseRow(ref Aperm);
            var shouldBeZero = Aperm - Blas.dot(L, U);
            return Analysis.MaxZeroError(shouldBeZero);
        }

        private void AssertPivotEqual(in Pivot a, in Pivot b, int dim)
        {
            for (int i = 0; i < dim; i++) {
                if (a[i] != b[i] && Fail[0] == (fProxy)0)
                {
                    Fail[0] = (fProxy)1; Fail[1] = (fProxy)a[i]; Fail[2] = (fProxy)b[i]; Fail[3] = (fProxy)i;
                }
                Assert.IsTrue(a[i] == b[i]);
            }
        }

        // max |A[i,j]| — matrix magnitude, used to scale backward-stable tolerances.
        private fProxy MatMaxAbs(in fProxyMxN A)
        {
            fProxy mx = (fProxy)0;
            for (int i = 0; i < A.Length; i++)
                mx = Unity.Mathematics.math.max(mx, Unity.Mathematics.math.abs(A[i]));
            return mx;
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff
        private void AssertClose(fProxy a, fProxy b, fProxy precision) {
            fProxy diff = Unity.Mathematics.math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected, [3]=relative diff
        private void AssertCloseRel(fProxy a, fProxy b, fProxy relPrecision) {
            fProxy denom = Unity.Mathematics.math.max((fProxy)1f, Unity.Mathematics.math.abs(b));
            fProxy diff = Unity.Mathematics.math.abs(a - b) / denom;
            if (!(diff <= relPrecision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= relPrecision);
        }

        // Fail layout: [0]=flag, [1]=got[i], [2]=expected[i], [3]=index cast to fProxy
        private void AssertVecClose(in fProxyN expected, in fProxyN got, int dim, fProxy precision) {
            for (int i = 0; i < dim; i++) {
                fProxy diff = Unity.Mathematics.math.abs(expected[i] - got[i]);
                if (!(diff <= precision) && Fail[0] == (fProxy)0)
                {
                    Fail[0] = (fProxy)1;
                    Fail[1] = got[i];
                    Fail[2] = expected[i];
                    Fail[3] = (fProxy)i;
                }
                Assert.IsTrue(diff <= precision);
            }
        }

        private void AssertLU(in fProxyMxN A, in fProxyMxN L, in fProxyMxN U, bool pivoted) => AssertLU(in A, in L, in U, pivoted, 1E-6f);
        private void AssertLU(in fProxyMxN A, in fProxyMxN L, in fProxyMxN U, bool pivoted, fProxy precision)
        {
            fProxyMxN shouldBeZero = A - Blas.dot(L, U);

            if (Analysis.isAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            // Fail layout: [1]=maxZeroError, [2]=precision, [3]=diff
            var zeroError = Analysis.MaxZeroError(shouldBeZero);
            if (!(zeroError <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = zeroError;
                Fail[2] = precision;
                Fail[3] = zeroError - precision;
            }
            Assert.IsTrue(Analysis.isZero(in shouldBeZero, precision));
            Assert.IsTrue(Analysis.isLowerTriangular(L, precision));
            Assert.IsTrue(Analysis.isUpperTriangular(U, precision));

            if(pivoted)
            unsafe {
                var maxAbs = LinearAlgebra.Internal.UnsafeOP.maxAbs(L.Data.Ptr, L.Length);

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
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try {
            new TestJob() { Type = type, Fail = fail }.Run();
            // Under Burst a failed in-job assert logs an exception and aborts the job without
            // throwing to the caller - surface the recorded diagnostics here as well.
            if (fail[0] != (fProxy)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");

        }
        catch (Exception e) {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally {
            fail.Dispose();
        }
    }

}
