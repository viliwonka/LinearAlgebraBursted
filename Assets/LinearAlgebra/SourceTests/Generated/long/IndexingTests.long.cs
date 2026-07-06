using System.Collections;
using System.Collections.Generic;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

using UnityEngine;
using UnityEngine.TestTools;

public class longIndexingTests {
    public enum TestType
    {
        TestVector,
        TestMatrix1D,
        TestMatrix2D,
        RandomCalc,
    }

    [BurstCompile(CompileSynchronously = true)]
    public struct IndexingTestJob : IJob
    {
        public TestType TestType;

        public void Execute()
        {
            switch(TestType)
            {
                case TestType.TestVector:
                    VectorIndexing();
                    break;
                case TestType.TestMatrix1D:
                    MatrixIndexing1D();
                    break;
                case TestType.TestMatrix2D:
                    MatrixIndexing2D();
                    break;
                case TestType.RandomCalc:
                    RandomCalc();
                break; 
            }
        }

        public void VectorIndexing()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;

            longN vec = arena.longVec(dim);

            // Forward-fill DISTINCT ground-truth values via the plain int indexer (oracle).
            for (int i = 0; i < dim; i++)
                vec[i] = (long)(i + 1);

            for (int i = 0; i < dim; i++)
                Assert.IsTrue(vec[i] == (long)(i + 1));

            // From-end accessor must equal the forward element at (Length - k).
            for (int k = 1; k <= dim; k++)
                Assert.IsTrue(vec[^k] == vec[dim - k]);

            // ^1 is the LAST element.
            Assert.IsTrue(vec[^1] == vec[dim - 1]);

            // Write through the from-end accessor with fresh distinct values,
            // read them back through the plain forward int accessor.
            for (int k = 1; k <= dim; k++)
                vec[^k] = (long)(1000 + k);

            for (int k = 1; k <= dim; k++)
                Assert.IsTrue(vec[dim - k] == (long)(1000 + k));

            arena.Dispose();
        }

        // Same invariants as VectorIndexing (forward-fill oracle, from-end == Length-k, ^1 is
        // last, write-through-then-readback), on the flat 1D matrix indexer.
        public void MatrixIndexing1D()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;

            longMxN mat = arena.longMat(dim, dim);

            int len = dim * dim;

            for (int i = 0; i < len; i++)
                mat[i] = (long)(i + 1);

            for (int i = 0; i < len; i++)
                Assert.IsTrue(mat[i] == (long)(i + 1));

            for (int k = 1; k <= len; k++)
                Assert.IsTrue(mat[^k] == mat[len - k]);

            Assert.IsTrue(mat[^1] == mat[len - 1]);

            for (int k = 1; k <= len; k++)
                mat[^k] = (long)(1000 + k);

            for (int k = 1; k <= len; k++)
                Assert.IsTrue(mat[len - k] == (long)(1000 + k));

            arena.Dispose();
        }

        public void MatrixIndexing2D()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 16;

            longMxN mat = arena.longMat(rows, cols);

            // Forward-fill DISTINCT ground-truth values via the plain [r, c] oracle.
            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                    mat[r, c] = (long)(r * cols + c + 1);

            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.IsTrue(mat[r, c] == (long)(r * cols + c + 1));

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
                mat[^r, ^c] = (long)(1000 + r * cols + c);

            for (int r = 1; r <= rows; r++)
            for (int c = 1; c <= cols; c++)
                Assert.IsTrue(mat[rows - r, cols - c] == (long)(1000 + r * cols + c));

            arena.Dispose();
        }

        public void RandomCalc()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 16;

            longMxN mat = arena.longMat(rows, cols);

            for(int r = 0; r < rows; r++)
            for(int c = 0; c < cols; c++)
                mat[r, c] = (long)(r * c);

            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                mat[r, c] = (long)(mat[r, c] + mat[r, c]);

            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.IsTrue(mat[r, c] == (long)(r * c * 2));
        }
    }

    [Test]
    public void VectorIndexingTest()
    {
        new IndexingTestJob() { TestType = TestType.TestVector}.Run();
    }


    [Test]
    public void MatrixIndexing1DTest()
    {
        new IndexingTestJob() { TestType = TestType.TestMatrix1D }.Run();
    }
    
    [Test]
    public void MatrixIndexing2DTest()
    {
        new IndexingTestJob() { TestType = TestType.TestMatrix2D }.Run();
    }

    [Test]
    public void RandomCalc()
    {
        new IndexingTestJob() { TestType = TestType.RandomCalc }.Run();
    }
}
