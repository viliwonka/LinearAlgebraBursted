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

public class floatLUTests
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
            LUSolveSystemInplace
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff/index
        public NativeArray<float> Fail;

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

            }
        }

        private floatMxN GetRandomMatrix(ref Arena arena, int dim, float min, float max, uint seed) {

            var mat = arena.floatRandomMat(dim, dim, min, max, seed);

            return mat;
        }

        public void LUDecompIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var U = arena.floatIdentityMat(dim);
            var L = arena.floatIdentityMat(dim);

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

            var U = arena.floatRandomDiagonalMat(dim, 1f, 3f);
            var L = arena.floatIdentityMat(dim);

            var A = U.Copy();

            bool success = LU.luDecompositionNoPivot(ref U, ref L);

            Assert.IsTrue(success);

            AssertLU(in A, in L, in U, false);

            arena.Dispose();
        }

        public void LUDecompPredefined() {

            var arena = new Arena(Allocator.Persistent);

            var dim = 5;

            var U = arena.floatMat(dim);
            var L = arena.floatIdentityMat(dim);

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

            var U = arena.floatRandomMat(dim, dim, 1f, 10f, 314221);
            var L = arena.floatIdentityMat(dim);

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
                var U = arena.floatMat(dim, dim);
                var L = arena.floatIdentityMat(dim);

                bool noPivot = LU.luDecompositionNoPivot(ref U, ref L);
                Assert.IsFalse(noPivot);
                Assert.IsFalse(Analysis_OP.isAnyNan(in U));
                Assert.IsFalse(Analysis_OP.isAnyNan(in L));

                var Up = arena.floatMat(dim, dim);
                var Lp = arena.floatIdentityMat(dim);
                var pivot = new Pivot(dim, Allocator.Temp);
                bool pivoted = LU.luDecomposition(ref Up, ref Lp, ref pivot);
                Assert.IsFalse(pivoted);
                Assert.IsFalse(Analysis_OP.isAnyNan(in Up));
                Assert.IsFalse(Analysis_OP.isAnyNan(in Lp));

                var LUmat = arena.floatMat(dim, dim);
                bool inplace = LU.luDecompositionInpl(ref LUmat, ref pivot);
                Assert.IsFalse(inplace);
                Assert.IsFalse(Analysis_OP.isAnyNan(in LUmat));

                pivot.Dispose();
            }

            // Case 2: two identical rows -> rank deficient -> all variants return false.
            {
                var U = arena.floatRandomMat(dim, dim, 1f, 10f, 8821);
                // force diagonal dominance so only the duplicated rows cause singularity
                for (int d = 0; d < dim; d++)
                    U[d, d] += 20f;
                // make row 5 an exact copy of row 2
                for (int c = 0; c < dim; c++)
                    U[5, c] = U[2, c];

                var A = U.Copy();
                var L = arena.floatIdentityMat(dim);

                bool noPivot = LU.luDecompositionNoPivot(ref U, ref L);
                Assert.IsFalse(noPivot);
                Assert.IsFalse(Analysis_OP.isAnyNan(in U));
                Assert.IsFalse(Analysis_OP.isAnyNan(in L));

                var Up = A.Copy();
                var Lp = arena.floatIdentityMat(dim);
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
                var U = arena.floatMat(2, 2);
                U[0, 0] = 0f; U[0, 1] = 1f;
                U[1, 0] = 1f; U[1, 1] = 0f;

                var L = arena.floatIdentityMat(2);

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
                var A = arena.floatMat(dim, dim);
                // A[0,0] == 0 forces a row swap
                A[0, 0] = 0f; A[0, 1] = 2f; A[0, 2] = 1f;
                A[1, 0] = 1f; A[1, 1] = 1f; A[1, 2] = 1f;
                A[2, 0] = 2f; A[2, 1] = 1f; A[2, 2] = 0f;

                var x_Known = arena.floatVec(dim);
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
                var A = arena.floatMat(dim, dim);
                A[0, 0] = 1f; A[0, 1] = 2f; A[0, 2] = 0f; A[0, 3] = 1f;
                A[1, 0] = 2f; A[1, 1] = 1f; A[1, 2] = 3f; A[1, 3] = 0f;
                A[2, 0] = 4f; A[2, 1] = 0f; A[2, 2] = 1f; A[2, 3] = 2f;
                A[3, 0] = 8f; A[3, 1] = 3f; A[3, 2] = 2f; A[3, 3] = 1f;

                var x_Known = arena.floatVec(dim);
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
                var I = arena.floatIdentityMat(dim);
                var pivot = new Pivot(dim, Allocator.Temp);

                bool success = LU.luDecompositionInpl(ref I, ref pivot);
                Assert.IsTrue(success);

                float det = LU.determinant(in I, in pivot);
                AssertClose(det, (float)1f, 1E-4f);

                pivot.Dispose();
            }

            // diagonal -> det = product of diagonal
            {
                int dim = 4;
                var D = arena.floatMat(dim, dim);
                D[0, 0] = 2f;
                D[1, 1] = -3f;
                D[2, 2] = 0.5f;
                D[3, 3] = 4f;
                float expected = 2f * -3f * 0.5f * 4f; // -12

                var pivot = new Pivot(dim, Allocator.Temp);
                bool success = LU.luDecompositionInpl(ref D, ref pivot);
                Assert.IsTrue(success);

                float det = LU.determinant(in D, in pivot);
                AssertCloseRel(det, expected, 1E-4f);

                pivot.Dispose();
            }

            // 3x3 with known determinant requiring a row swap (A[0,0]==0).
            // A = [[0,2,1],[1,1,1],[2,1,0]]; det = 3 (hand computed, nonsingular).
            {
                int dim = 3;
                var A = arena.floatMat(dim, dim);
                A[0, 0] = 0f; A[0, 1] = 2f; A[0, 2] = 1f;
                A[1, 0] = 1f; A[1, 1] = 1f; A[1, 2] = 1f;
                A[2, 0] = 2f; A[2, 1] = 1f; A[2, 2] = 0f;

                var pivot = new Pivot(dim, Allocator.Temp);
                bool success = LU.luDecompositionInpl(ref A, ref pivot);
                Assert.IsTrue(success);

                float det = LU.determinant(in A, in pivot);
                AssertCloseRel(det, (float)3f, 1E-4f);

                pivot.Dispose();
            }

            // permutation matrix -> det = +-1 matching swap parity.
            // single transposition (rows 0 and 2 swapped) -> det = -1.
            {
                int dim = 3;
                var P = arena.floatMat(dim, dim);
                P[0, 2] = 1f;
                P[1, 1] = 1f;
                P[2, 0] = 1f;

                var pivot = new Pivot(dim, Allocator.Temp);
                bool success = LU.luDecompositionInpl(ref P, ref pivot);
                Assert.IsTrue(success);

                float det = LU.determinant(in P, in pivot);
                AssertCloseRel(det, (float)(-1f), 1E-4f);

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
                var A = arena.floatPascal(dim);
                var pivot = new Pivot(dim, Allocator.Temp);
                bool success = LU.luDecompositionInpl(ref A, ref pivot);
                Assert.IsTrue(success);

                float det = LU.determinant(in A, in pivot);
                AssertCloseRel(det, (float)1f, 1E-4f);

                pivot.Dispose();
            }

            // MinIJ(5): det = 1
            {
                int dim = 5;
                var A = arena.floatMinIJ(dim);
                var pivot = new Pivot(dim, Allocator.Temp);
                bool success = LU.luDecompositionInpl(ref A, ref pivot);
                Assert.IsTrue(success);

                float det = LU.determinant(in A, in pivot);
                AssertCloseRel(det, (float)1f, 1E-4f);

                pivot.Dispose();
            }

            // Frank(5): det = 1
            {
                int dim = 5;
                var A = arena.floatFrank(dim);
                var pivot = new Pivot(dim, Allocator.Temp);
                bool success = LU.luDecompositionInpl(ref A, ref pivot);
                Assert.IsTrue(success);

                float det = LU.determinant(in A, in pivot);
                AssertCloseRel(det, (float)1f, 1E-4f);

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
            var A1 = arena.floatRandomMat(dim, dim, -5f, 5f, 7777);
            for (int d = 0; d < dim; d++)
                A1[d, d] += 15f;
            var LU1 = A1.Copy();
            bool s1 = LU.luDecompositionInpl(ref LU1, ref pivot);
            Assert.IsTrue(s1);

            // Second decomposition reuses the SAME pivot object; Reset() must clean it.
            var A2 = arena.floatRandomMat(dim, dim, -5f, 5f, 9999);
            for (int d = 0; d < dim; d++)
                A2[d, d] += 15f;

            var x_Known = arena.floatVec(dim);
            for (int i = 0; i < dim; i++)
                x_Known[i] = (float)(i + 1);

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
                var mat = arena.floatMat(dim, dim);
                for (int r = 0; r < dim; r++)
                    for (int c = 0; c < dim; c++)
                        mat[r, c] = (float)(r * 10 + c);

                Swap_OP.Rows(ref mat, 0, 1);

                // row 0 and row 1 fully swapped
                for (int c = 0; c < dim; c++) {
                    AssertClose(mat[0, c], (float)(10 + c), 1E-6f);
                    AssertClose(mat[1, c], (float)(0 + c), 1E-6f);
                    AssertClose(mat[2, c], (float)(20 + c), 1E-6f);
                }
            }

            // Swap_OP.Columns with explicit start/end swaps only that row-range.
            {
                int dim = 4;
                var mat = arena.floatMat(dim, dim);
                for (int r = 0; r < dim; r++)
                    for (int c = 0; c < dim; c++)
                        mat[r, c] = (float)(r * 10 + c);

                // swap columns 0 and 1 only for rows [1,3)
                Swap_OP.Columns(ref mat, 0, 1, 1, 3);

                // rows 0 and 3 untouched
                AssertClose(mat[0, 0], (float)(0), 1E-6f);
                AssertClose(mat[0, 1], (float)(1), 1E-6f);
                AssertClose(mat[3, 0], (float)(30), 1E-6f);
                AssertClose(mat[3, 1], (float)(31), 1E-6f);

                // rows 1 and 2 have columns 0 and 1 swapped
                AssertClose(mat[1, 0], (float)(11), 1E-6f);
                AssertClose(mat[1, 1], (float)(10), 1E-6f);
                AssertClose(mat[2, 0], (float)(21), 1E-6f);
                AssertClose(mat[2, 1], (float)(20), 1E-6f);

                // other columns untouched
                AssertClose(mat[1, 2], (float)(12), 1E-6f);
                AssertClose(mat[2, 3], (float)(23), 1E-6f);
            }

            arena.Dispose();
        }

        public void SolveSystem() {

            var arena = new Arena(Allocator.Persistent);

            int dim = 512;

            var A = arena.floatRandomMat(dim, dim, -10f, 10f, 314221);

            for (int d = 0; d < dim; d++) {
                A[d, d] *= 2f;
                if (Unity.Mathematics.math.abs(A[d, d]) < 0.01f)
                    A[d, d] *= 10f;
            }

            var x_Known = arena.floatRandomVec(dim, 1f, 10f, 901);

            var b = Linear_OP.dot(A, x_Known);

            var U = A.Copy();
            var L = arena.floatIdentityMat(dim);

            var pivot = new Pivot(dim, Allocator.Temp);

            bool success = LU.luDecomposition(ref U, ref L, ref pivot);

            Assert.IsTrue(success);

            var x_Solved = b.Copy();

            LU.luSolve(ref L, ref U, in pivot, ref x_Solved);

            if (Analysis_OP.isAnyNan(in x_Solved))
                throw new System.Exception("TestJob: NaN detected");

            var zeroError = Analysis_OP.MaxZeroError(x_Known - x_Solved);

            // Fail layout: [1]=zeroError, [2]=limit 1E-3, [3]=diff
            if (!(zeroError < (float)1E-03f) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = zeroError;
                Fail[2] = (float)1E-03f;
                Fail[3] = zeroError - (float)1E-03f;
            }
            Assert.IsTrue(zeroError < 1E-03f);

            pivot.Dispose();

            arena.Dispose();
        }

        public void SolveSystemInplace() {

            var arena = new Arena(Allocator.Persistent);

            int dim = 512;

            var A = arena.floatRandomMat(dim, dim, -10f, 10f, 314221);

            for (int d = 0; d < dim; d++) {
                A[d, d] *= 2f;
                if (Unity.Mathematics.math.abs(A[d, d]) < 0.01f)
                    A[d, d] *= 10f;
            }

            var x_Known = arena.floatRandomVec(dim, 1f, 10f, 901);

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
            if (!(zeroError < (float)1E-03f) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = zeroError;
                Fail[2] = (float)1E-03f;
                Fail[3] = zeroError - (float)1E-03f;
            }
            Assert.IsTrue(zeroError < 1E-03f);

            pivot.Dispose();

            arena.Dispose();
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff
        private void AssertClose(float a, float b, float precision) {
            float diff = Unity.Mathematics.math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        // Fail layout: [0]=flag, [1]=got, [2]=expected, [3]=relative diff
        private void AssertCloseRel(float a, float b, float relPrecision) {
            float denom = Unity.Mathematics.math.max((float)1f, Unity.Mathematics.math.abs(b));
            float diff = Unity.Mathematics.math.abs(a - b) / denom;
            if (!(diff <= relPrecision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= relPrecision);
        }

        // Fail layout: [0]=flag, [1]=got[i], [2]=expected[i], [3]=index cast to float
        private void AssertVecClose(in floatN expected, in floatN got, int dim, float precision) {
            for (int i = 0; i < dim; i++) {
                float diff = Unity.Mathematics.math.abs(expected[i] - got[i]);
                if (!(diff <= precision) && Fail[0] == (float)0)
                {
                    Fail[0] = (float)1;
                    Fail[1] = got[i];
                    Fail[2] = expected[i];
                    Fail[3] = (float)i;
                }
                Assert.IsTrue(diff <= precision);
            }
        }

        private void AssertLU(in floatMxN A, in floatMxN L, in floatMxN U, bool pivoted) => AssertLU(in A, in L, in U, pivoted, 1E-6f);
        private void AssertLU(in floatMxN A, in floatMxN L, in floatMxN U, bool pivoted, float precision)
        {
            floatMxN shouldBeZero = A - Linear_OP.dot(L, U);

            if (Analysis_OP.isAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            // Fail layout: [1]=maxZeroError, [2]=precision, [3]=diff
            var zeroError = Analysis_OP.MaxZeroError(shouldBeZero);
            if (!(zeroError <= precision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
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
        var fail = new NativeArray<float>(4, Allocator.TempJob);
        try {
            new TestJob() { Type = type, Fail = fail }.Run();
            // Under Burst a failed in-job assert logs an exception and aborts the job without
            // throwing to the caller - surface the recorded diagnostics here as well.
            if (fail[0] != (float)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");

        }
        catch (Exception e) {
            if (fail[0] != (float)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally {
            fail.Dispose();
        }
    }

}
