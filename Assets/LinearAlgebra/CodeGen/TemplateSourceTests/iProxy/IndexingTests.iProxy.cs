using System.Collections;
using System.Collections.Generic;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

using UnityEngine;
using UnityEngine.TestTools;

public class iProxyIndexingTests {
    public enum TestType
    {
        TestVector,
        TestMatrix1D,
        TestMatrix2D,
        RandomCalc,
    }

    [BurstCompile]
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

            iProxyN vec = arena.iProxyVec(dim);

            // Forward-fill DISTINCT ground-truth values via the plain int indexer (oracle).
            for (int i = 0; i < dim; i++)
                vec[i] = (iProxy)(i + 1);

            for (int i = 0; i < dim; i++)
                Assert.IsTrue(vec[i] == (iProxy)(i + 1));

            // From-end accessor must equal the forward element at (Length - k).
            for (int k = 1; k <= dim; k++)
                Assert.IsTrue(vec[^k] == vec[dim - k]);

            // ^1 is the LAST element.
            Assert.IsTrue(vec[^1] == vec[dim - 1]);

            // Write through the from-end accessor with fresh distinct values,
            // read them back through the plain forward int accessor.
            for (int k = 1; k <= dim; k++)
                vec[^k] = (iProxy)(1000 + k);

            for (int k = 1; k <= dim; k++)
                Assert.IsTrue(vec[dim - k] == (iProxy)(1000 + k));

            arena.Dispose();
        }

        public void MatrixIndexing1D()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;

            iProxyMxN mat = arena.iProxyMat(dim, dim);

            int len = dim * dim;

            // Forward-fill DISTINCT ground-truth values via the plain int indexer (oracle).
            for (int i = 0; i < len; i++)
                mat[i] = (iProxy)(i + 1);

            for (int i = 0; i < len; i++)
                Assert.IsTrue(mat[i] == (iProxy)(i + 1));

            // From-end accessor must equal the forward element at (Length - k).
            for (int k = 1; k <= len; k++)
                Assert.IsTrue(mat[^k] == mat[len - k]);

            // ^1 is the LAST element.
            Assert.IsTrue(mat[^1] == mat[len - 1]);

            // Write through the from-end accessor, read back through forward int accessor.
            for (int k = 1; k <= len; k++)
                mat[^k] = (iProxy)(1000 + k);

            for (int k = 1; k <= len; k++)
                Assert.IsTrue(mat[len - k] == (iProxy)(1000 + k));

            arena.Dispose();
        }

        public void MatrixIndexing2D()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 16;

            iProxyMxN mat = arena.iProxyMat(rows, cols);

            // Forward-fill DISTINCT ground-truth values via the plain [r, c] oracle.
            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                    mat[r, c] = (iProxy)(r * cols + c + 1);

            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.IsTrue(mat[r, c] == (iProxy)(r * cols + c + 1));

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
                mat[^r, ^c] = (iProxy)(1000 + r * cols + c);

            for (int r = 1; r <= rows; r++)
            for (int c = 1; c <= cols; c++)
                Assert.IsTrue(mat[rows - r, cols - c] == (iProxy)(1000 + r * cols + c));

            arena.Dispose();
        }

        public void RandomCalc()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 16;

            iProxyMxN mat = arena.iProxyMat(rows, cols);

            for(int r = 0; r < rows; r++)
            for(int c = 0; c < cols; c++)
                mat[r, c] = (iProxy)(r * c);

            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                mat[r, c] = (iProxy)(mat[r, c] + mat[r, c]);

            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.IsTrue(mat[r, c] == (iProxy)(r * c * 2));
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
