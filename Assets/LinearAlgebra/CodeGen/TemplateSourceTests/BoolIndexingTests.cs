using LinearAlgebra;
using NUnit.Framework;
using System;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

public class BoolIndexingTests
{
    [BurstCompile]
    public struct TestsJob : IJob
    {
        public enum TestType
        {
            VectorIndexing,
            MatrixIndexing1D,
            MatrixIndexing2D,
            VectorCopyGuard,
            MatrixCopyNullArenaGuard,
        }

        public TestType Type;

        public void Execute()
        {
            Arena arena = new Arena(Allocator.Temp);
            try
            {
                switch (Type)
                {
                    case TestType.VectorIndexing:
                        VectorIndexing(ref arena);
                        break;
                    case TestType.MatrixIndexing1D:
                        MatrixIndexing1D(ref arena);
                        break;
                    case TestType.MatrixIndexing2D:
                        MatrixIndexing2D(ref arena);
                        break;
                    case TestType.VectorCopyGuard:
                        VectorCopyGuard(ref arena);
                        break;
                    case TestType.MatrixCopyNullArenaGuard:
                        MatrixCopyNullArenaGuard(ref arena);
                        break;
                    default:
                        throw new NotImplementedException();
                }
            }
            finally
            {
                arena.Dispose();
            }
        }

        void VectorIndexing(ref Arena arena)
        {
            int dim = 17;

            boolN vec = arena.BoolVector(dim);

            // Forward-fill a distinct-enough pattern via the plain int indexer (oracle).
            for (int i = 0; i < dim; i++)
                vec[i] = (i % 3 == 0);

            // From-end accessor must equal the forward element at (Length - k), for ANY pattern.
            for (int k = 1; k <= dim; k++)
                Assert.IsTrue(vec[^k] == vec[dim - k]);

            // ^1 is the LAST element.
            Assert.IsTrue(vec[^1] == vec[dim - 1]);

            // Write through the from-end accessor, read back through the plain forward int accessor.
            for (int k = 1; k <= dim; k++)
                vec[^k] = (k % 2 == 0);

            for (int k = 1; k <= dim; k++)
                Assert.IsTrue(vec[dim - k] == (k % 2 == 0));
        }

        void MatrixIndexing1D(ref Arena arena)
        {
            int rows = 5;
            int cols = 7;

            boolMxN mat = arena.BoolMatrix(rows, cols);

            int len = rows * cols;

            // Forward-fill a distinct-enough pattern via the plain int indexer (oracle).
            for (int i = 0; i < len; i++)
                mat[i] = (i % 3 == 0);

            // From-end accessor must equal the forward element at (Length - k).
            for (int k = 1; k <= len; k++)
                Assert.IsTrue(mat[^k] == mat[len - k]);

            // ^1 is the LAST element.
            Assert.IsTrue(mat[^1] == mat[len - 1]);

            // Write through the from-end accessor, read back through the plain forward int accessor.
            for (int k = 1; k <= len; k++)
                mat[^k] = (k % 2 == 0);

            for (int k = 1; k <= len; k++)
                Assert.IsTrue(mat[len - k] == (k % 2 == 0));
        }

        void MatrixIndexing2D(ref Arena arena)
        {
            int rows = 5;
            int cols = 7;

            boolMxN mat = arena.BoolMatrix(rows, cols);

            // Forward-fill a distinct-enough pattern via the plain [r, c] oracle.
            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                mat[r, c] = ((r * cols + c) % 3 == 0);

            // [^r, c] : from-end row, forward col -> forward [rows - r, c].
            for (int r = 1; r <= rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.IsTrue(mat[^r, c] == mat[rows - r, c]);

            // [r, ^c] : forward row, from-end col -> forward [r, cols - c].
            for (int r = 0; r < rows; r++)
            for (int c = 1; c <= cols; c++)
                Assert.IsTrue(mat[r, ^c] == mat[r, cols - c]);

            // [^r, ^c] : both from-end -> forward [rows - r, cols - c].
            for (int r = 1; r <= rows; r++)
            for (int c = 1; c <= cols; c++)
                Assert.IsTrue(mat[^r, ^c] == mat[rows - r, cols - c]);

            // ^1, ^1 is the LAST element.
            Assert.IsTrue(mat[^1, ^1] == mat[rows - 1, cols - 1]);

            // Write through the from-end accessor (both axes), read back through forward [r, c].
            for (int r = 1; r <= rows; r++)
            for (int c = 1; c <= cols; c++)
                mat[^r, ^c] = ((r + c) % 2 == 0);

            for (int r = 1; r <= rows; r++)
            for (int c = 1; c <= cols; c++)
                Assert.IsTrue(mat[rows - r, cols - c] == ((r + c) % 2 == 0));
        }

        // boolN has no standalone (null-arena) ctor, so exercise the copy-constructor guard's
        // non-null branch: copying an arena-backed vector with the DEFAULT allocator must resolve
        // the allocator from the (non-null) arena pointer and produce an equal, independent copy.
        void VectorCopyGuard(ref Arena arena)
        {
            int dim = 16;

            boolN orig = arena.BoolVector(dim);
            for (int i = 0; i < dim; i++)
                orig[i] = (i % 2 == 0);

            // Default allocator -> guard takes the non-null branch (arena allocator). Must not crash.
            boolN copy = new boolN(in orig);

            Assert.IsTrue(copy.N == orig.N);
            for (int i = 0; i < dim; i++)
                Assert.IsTrue(copy[i] == orig[i]);

            // Copy must be independent of the original.
            copy[0] = !copy[0];
            Assert.IsTrue(copy[0] != orig[0]);

            copy.Dispose();
        }

        // boolMxN HAS a standalone (null-arena) ctor, so fully exercise the copy-constructor guard's
        // null branch: copying a standalone matrix with the DEFAULT allocator previously dereferenced
        // a null _arenaPtr; it must now fall back to Allocator.Temp without crashing and copy equally.
        void MatrixCopyNullArenaGuard(ref Arena arena)
        {
            int rows = 4;
            int cols = 6;

            // Standalone matrix: the non-arena ctor leaves _arenaPtr null.
            boolMxN standalone = new boolMxN(rows, cols, Allocator.Temp);
            for (int i = 0; i < standalone.Length; i++)
                standalone[i] = (i % 2 == 0);

            // Default allocator -> guard MUST hit the null-arena branch (fallback Allocator.Temp).
            boolMxN copy = new boolMxN(standalone);

            Assert.IsTrue(copy.M_Rows == standalone.M_Rows);
            Assert.IsTrue(copy.N_Cols == standalone.N_Cols);
            Assert.IsTrue(copy.Length == standalone.Length);
            for (int i = 0; i < standalone.Length; i++)
                Assert.IsTrue(copy[i] == standalone[i]);

            // Copy must be independent of the original.
            copy[0] = !copy[0];
            Assert.IsTrue(copy[0] != standalone[0]);

            copy.Dispose();
            standalone.Dispose();
        }
    }

    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(TestsJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void Tests(TestsJob.TestType testType)
    {
        new TestsJob() { Type = testType }.Run();
    }
}
