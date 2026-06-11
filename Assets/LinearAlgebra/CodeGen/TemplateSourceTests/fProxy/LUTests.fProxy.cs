using System;

using LinearAlgebra;
using LinearAlgebra.Stats;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using UnityEngine;
using UnityEngine.TestTools;

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
            LUDecompPivotRequired,
            LUDeterminant,
            LUReusePivot,
            SwapOPTest,
            LUSolveSystem,
            LUSolveSystemInplace
        }

        public TestType Type;


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

        private fProxyMxN GetRandomMatrix(ref Arena arena, int dim, fProxy min, fProxy max, uint seed) {

            var mat = arena.fProxyRandomMatrix(dim, dim, min, max, seed);

            return mat;
        }

        public void LUDecompIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var U = arena.fProxyIdentityMatrix(dim);
            var L = arena.fProxyIdentityMatrix(dim);

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

            var U = arena.fProxyRandomDiagonalMatrix(dim, 1f, 3f);
            var L = arena.fProxyIdentityMatrix(dim);

            var A = U.Copy();

            bool success = LU.luDecompositionNoPivot(ref U, ref L);

            Assert.IsTrue(success);

            AssertLU(in A, in L, in U, false);

            arena.Dispose();
        }

        public void LUDecompPredefined() {

            var arena = new Arena(Allocator.Persistent);

            var dim = 5;

            var U = arena.fProxyMat(dim);
            var L = arena.fProxyIdentityMatrix(dim);

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

            var U = arena.fProxyRandomMatrix(dim, dim, 1f, 10f, 314221);
            var L = arena.fProxyIdentityMatrix(dim);

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
                var U = arena.fProxyMat(dim, dim);
                var L = arena.fProxyIdentityMatrix(dim);

                bool noPivot = LU.luDecompositionNoPivot(ref U, ref L);
                Assert.IsFalse(noPivot);
                Assert.IsFalse(Analysis.IsAnyNan(in U));
                Assert.IsFalse(Analysis.IsAnyNan(in L));

                var Up = arena.fProxyMat(dim, dim);
                var Lp = arena.fProxyIdentityMatrix(dim);
                var pivot = new Pivot(dim, Allocator.Temp);
                bool pivoted = LU.luDecomposition(ref Up, ref Lp, ref pivot);
                Assert.IsFalse(pivoted);
                Assert.IsFalse(Analysis.IsAnyNan(in Up));
                Assert.IsFalse(Analysis.IsAnyNan(in Lp));

                var LUmat = arena.fProxyMat(dim, dim);
                bool inplace = LU.luDecompositionInplace(ref LUmat, ref pivot);
                Assert.IsFalse(inplace);
                Assert.IsFalse(Analysis.IsAnyNan(in LUmat));

                pivot.Dispose();
            }

            // Case 2: two identical rows -> rank deficient -> all variants return false.
            {
                var U = arena.fProxyRandomMatrix(dim, dim, 1f, 10f, 8821);
                // force diagonal dominance so only the duplicated rows cause singularity
                for (int d = 0; d < dim; d++)
                    U[d, d] += 20f;
                // make row 5 an exact copy of row 2
                for (int c = 0; c < dim; c++)
                    U[5, c] = U[2, c];

                var A = U.Copy();
                var L = arena.fProxyIdentityMatrix(dim);

                bool noPivot = LU.luDecompositionNoPivot(ref U, ref L);
                Assert.IsFalse(noPivot);
                Assert.IsFalse(Analysis.IsAnyNan(in U));
                Assert.IsFalse(Analysis.IsAnyNan(in L));

                var Up = A.Copy();
                var Lp = arena.fProxyIdentityMatrix(dim);
                var pivot = new Pivot(dim, Allocator.Temp);
                bool pivoted = LU.luDecomposition(ref Up, ref Lp, ref pivot);
                Assert.IsFalse(pivoted);
                Assert.IsFalse(Analysis.IsAnyNan(in Up));
                Assert.IsFalse(Analysis.IsAnyNan(in Lp));

                var LUmat = A.Copy();
                bool inplace = LU.luDecompositionInplace(ref LUmat, ref pivot);
                Assert.IsFalse(inplace);
                Assert.IsFalse(Analysis.IsAnyNan(in LUmat));

                pivot.Dispose();
            }

            // Case 3: [[0,1],[1,0]] : no-pivot fails on zero leading pivot,
            // but inplace (with partial pivoting) succeeds.
            {
                var U = arena.fProxyMat(2, 2);
                U[0, 0] = 0f; U[0, 1] = 1f;
                U[1, 0] = 1f; U[1, 1] = 0f;

                var L = arena.fProxyIdentityMatrix(2);

                var Unp = U.Copy();
                bool noPivot = LU.luDecompositionNoPivot(ref Unp, ref L);
                Assert.IsFalse(noPivot);

                var LUmat = U.Copy();
                var pivot = new Pivot(2, Allocator.Temp);
                bool inplace = LU.luDecompositionInplace(ref LUmat, ref pivot);
                Assert.IsTrue(inplace);
                Assert.IsFalse(Analysis.IsAnyNan(in LUmat));

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
                var A = arena.fProxyMat(dim, dim);
                // A[0,0] == 0 forces a row swap
                A[0, 0] = 0f; A[0, 1] = 2f; A[0, 2] = 1f;
                A[1, 0] = 1f; A[1, 1] = 1f; A[1, 2] = 1f;
                A[2, 0] = 2f; A[2, 1] = 1f; A[2, 2] = 0f;

                var x_Known = arena.fProxyVec(dim);
                x_Known[0] = 3f; x_Known[1] = -2f; x_Known[2] = 5f;

                var b = fProxyOP.dot(A, x_Known);

                var LUmat = A.Copy();
                var pivot = new Pivot(dim, Allocator.Temp);

                bool success = LU.luDecompositionInplace(ref LUmat, ref pivot);
                Assert.IsTrue(success);

                var x_Solved = b.Copy();
                LU.LUSolve(ref LUmat, in pivot, ref x_Solved);

                Assert.IsFalse(Analysis.IsAnyNan(in x_Solved));

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

                var b = fProxyOP.dot(A, x_Known);

                var LUmat = A.Copy();
                var pivot = new Pivot(dim, Allocator.Temp);

                bool success = LU.luDecompositionInplace(ref LUmat, ref pivot);
                Assert.IsTrue(success);

                // Verify the permutation is not a simple involution (P applied twice != identity),
                // i.e. it really contains a cycle of length > 2.
                bool isInvolution = true;
                for (int i = 0; i < dim; i++)
                    if (pivot[pivot[i]] != i)
                        isInvolution = false;
                Assert.IsFalse(isInvolution);

                var x_Solved = b.Copy();
                LU.LUSolve(ref LUmat, in pivot, ref x_Solved);

                Assert.IsFalse(Analysis.IsAnyNan(in x_Solved));

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
                var I = arena.fProxyIdentityMatrix(dim);
                var pivot = new Pivot(dim, Allocator.Temp);

                bool success = LU.luDecompositionInplace(ref I, ref pivot);
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
                bool success = LU.luDecompositionInplace(ref D, ref pivot);
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
                bool success = LU.luDecompositionInplace(ref A, ref pivot);
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
                bool success = LU.luDecompositionInplace(ref P, ref pivot);
                Assert.IsTrue(success);

                fProxy det = LU.determinant(in P, in pivot);
                AssertCloseRel(det, (fProxy)(-1f), 1E-4f);

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
            var A1 = arena.fProxyRandomMatrix(dim, dim, -5f, 5f, 7777);
            for (int d = 0; d < dim; d++)
                A1[d, d] += 15f;
            var LU1 = A1.Copy();
            bool s1 = LU.luDecompositionInplace(ref LU1, ref pivot);
            Assert.IsTrue(s1);

            // Second decomposition reuses the SAME pivot object; Reset() must clean it.
            var A2 = arena.fProxyRandomMatrix(dim, dim, -5f, 5f, 9999);
            for (int d = 0; d < dim; d++)
                A2[d, d] += 15f;

            var x_Known = arena.fProxyVec(dim);
            for (int i = 0; i < dim; i++)
                x_Known[i] = (fProxy)(i + 1);

            var b = fProxyOP.dot(A2, x_Known);

            var LU2 = A2.Copy();
            bool s2 = LU.luDecompositionInplace(ref LU2, ref pivot);
            Assert.IsTrue(s2);

            var x_Solved = b.Copy();
            LU.LUSolve(ref LU2, in pivot, ref x_Solved);

            Assert.IsFalse(Analysis.IsAnyNan(in x_Solved));

            AssertVecClose(in x_Known, in x_Solved, dim, 1E-3f);

            pivot.Dispose();

            arena.Dispose();
        }

        public void SwapOPTest()
        {
            var arena = new Arena(Allocator.Persistent);

            // SwapOP.Rows with default start/end swaps full rows.
            {
                int dim = 3;
                var mat = arena.fProxyMat(dim, dim);
                for (int r = 0; r < dim; r++)
                    for (int c = 0; c < dim; c++)
                        mat[r, c] = (fProxy)(r * 10 + c);

                SwapOP.Rows(ref mat, 0, 1);

                // row 0 and row 1 fully swapped
                for (int c = 0; c < dim; c++) {
                    AssertClose(mat[0, c], (fProxy)(10 + c), 1E-6f);
                    AssertClose(mat[1, c], (fProxy)(0 + c), 1E-6f);
                    AssertClose(mat[2, c], (fProxy)(20 + c), 1E-6f);
                }
            }

            // SwapOP.Columns with explicit start/end swaps only that row-range.
            {
                int dim = 4;
                var mat = arena.fProxyMat(dim, dim);
                for (int r = 0; r < dim; r++)
                    for (int c = 0; c < dim; c++)
                        mat[r, c] = (fProxy)(r * 10 + c);

                // swap columns 0 and 1 only for rows [1,3)
                SwapOP.Columns(ref mat, 0, 1, 1, 3);

                // rows 0 and 3 untouched
                AssertClose(mat[0, 0], (fProxy)(0), 1E-6f);
                AssertClose(mat[0, 1], (fProxy)(1), 1E-6f);
                AssertClose(mat[3, 0], (fProxy)(30), 1E-6f);
                AssertClose(mat[3, 1], (fProxy)(31), 1E-6f);

                // rows 1 and 2 have columns 0 and 1 swapped
                AssertClose(mat[1, 0], (fProxy)(11), 1E-6f);
                AssertClose(mat[1, 1], (fProxy)(10), 1E-6f);
                AssertClose(mat[2, 0], (fProxy)(21), 1E-6f);
                AssertClose(mat[2, 1], (fProxy)(20), 1E-6f);

                // other columns untouched
                AssertClose(mat[1, 2], (fProxy)(12), 1E-6f);
                AssertClose(mat[2, 3], (fProxy)(23), 1E-6f);
            }

            arena.Dispose();
        }

        public void SolveSystem() {

            var arena = new Arena(Allocator.Persistent);

            int dim = 512;

            var A = arena.fProxyRandomMatrix(dim, dim, -10f, 10f, 314221);

            for (int d = 0; d < dim; d++) {
                A[d, d] *= 2f;
                if (Unity.Mathematics.math.abs(A[d, d]) < 0.01f)
                    A[d, d] *= 10f;
            }

            var x_Known = arena.fProxyRandomVector(dim, 1f, 10f, 901);

            var b = fProxyOP.dot(A, x_Known);

            var U = A.Copy();
            var L = arena.fProxyIdentityMatrix(dim);

            var pivot = new Pivot(dim, Allocator.Temp);

            bool success = LU.luDecomposition(ref U, ref L, ref pivot);

            Assert.IsTrue(success);

            var x_Solved = b.Copy();

            LU.LUSolve(ref L, ref U, in pivot, ref x_Solved);

            if (Analysis.IsAnyNan(in x_Solved))
                throw new System.Exception("TestJob: NaN detected");

            var zeroError = Analysis.MaxZeroError(x_Known - x_Solved);

            Debug.Log($"Error of max(abs(x_Known - x_Solved)): {zeroError}");


            Assert.IsTrue(zeroError < 1E-03f);

            pivot.Dispose();

            arena.Dispose();
        }

        public void SolveSystemInplace() {

            var arena = new Arena(Allocator.Persistent);

            int dim = 512;

            var A = arena.fProxyRandomMatrix(dim, dim, -10f, 10f, 314221);

            for (int d = 0; d < dim; d++) {
                A[d, d] *= 2f;
                if (Unity.Mathematics.math.abs(A[d, d]) < 0.01f)
                    A[d, d] *= 10f;
            }

            var x_Known = arena.fProxyRandomVector(dim, 1f, 10f, 901);

            var b = fProxyOP.dot(A, x_Known);

            var LUmat = A.Copy();

            var pivot = new Pivot(dim, Allocator.Temp);

            bool success = LU.luDecompositionInplace(ref LUmat, ref pivot);

            Assert.IsTrue(success);

            var x_Solved = b.Copy();

            LU.LUSolve(ref LUmat, in pivot, ref x_Solved);

            if (Analysis.IsAnyNan(in x_Solved))
                throw new System.Exception("TestJob: NaN detected");

            var zeroError = Analysis.MaxZeroError(x_Known - x_Solved);

            Debug.Log($"Error of max(abs(x_Known - x_Solved)): {zeroError}");

            Assert.IsTrue(zeroError < 1E-03f);

            pivot.Dispose();

            arena.Dispose();
        }

        private void AssertClose(fProxy a, fProxy b, fProxy precision) {
            fProxy diff = Unity.Mathematics.math.abs(a - b);
            Assert.IsTrue(diff <= precision, $"Expected {b} got {a} (diff {diff})");
        }

        private void AssertCloseRel(fProxy a, fProxy b, fProxy relPrecision) {
            fProxy denom = Unity.Mathematics.math.max((fProxy)1f, Unity.Mathematics.math.abs(b));
            fProxy diff = Unity.Mathematics.math.abs(a - b) / denom;
            Assert.IsTrue(diff <= relPrecision, $"Expected {b} got {a} (rel diff {diff})");
        }

        private void AssertVecClose(in fProxyN expected, in fProxyN got, int dim, fProxy precision) {
            for (int i = 0; i < dim; i++) {
                fProxy diff = Unity.Mathematics.math.abs(expected[i] - got[i]);
                Assert.IsTrue(diff <= precision, $"x[{i}] expected {expected[i]} got {got[i]} (diff {diff})");
            }
        }

        private void AssertLU(in fProxyMxN A, in fProxyMxN L, in fProxyMxN U, bool pivoted) => AssertLU(in A, in L, in U, pivoted, 1E-6f);
        private void AssertLU(in fProxyMxN A, in fProxyMxN L, in fProxyMxN U, bool pivoted, fProxy precision)
        {
            fProxyMxN shouldBeZero = A - fProxyOP.dot(L, U);

            var zeroError = Analysis.MaxZeroError(shouldBeZero);

            if (Analysis.IsAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            Debug.Log($"Error of max(abs(A - LU)): {zeroError}");

            Assert.IsTrue(Analysis.IsZero(in shouldBeZero, precision));
            Assert.IsTrue(Analysis.IsLowerTriangular(L, precision));
            Assert.IsTrue(Analysis.IsUpperTriangular(U, precision));

            if(pivoted)
            unsafe {
                var maxAbs = LinearAlgebra.UnsafeOP.maxAbs(L.Data.Ptr, L.Length);

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
        new TestJob() { Type = type }.Run();
    }

}
