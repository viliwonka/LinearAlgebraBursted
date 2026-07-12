using System.Collections;
using System.Collections.Generic;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

using UnityEngine;
using UnityEngine.TestTools;

public class fProxyIndexingTests {
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

            fProxyN vec = arena.fProxyVec(dim);

            // Forward-fill DISTINCT ground-truth values via the plain int indexer (oracle).
            for (int i = 0; i < dim; i++)
                vec[i] = (fProxy)(i + 1);

            for (int i = 0; i < dim; i++)
                Assert.IsTrue(vec[i] == (fProxy)(i + 1));

            // From-end accessor must equal the forward element at (Length - k).
            for (int k = 1; k <= dim; k++)
                Assert.IsTrue(vec[^k] == vec[dim - k]);

            // ^1 is the LAST element.
            Assert.IsTrue(vec[^1] == vec[dim - 1]);

            // Write through the from-end accessor with fresh distinct values,
            // read them back through the plain forward int accessor.
            for (int k = 1; k <= dim; k++)
                vec[^k] = (fProxy)(1000 + k);

            for (int k = 1; k <= dim; k++)
                Assert.IsTrue(vec[dim - k] == (fProxy)(1000 + k));

            arena.Dispose();
        }

        public void MatrixIndexing1D()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;

            fProxyMxN mat = arena.fProxyMat(dim, dim);

            int len = dim * dim;

            // Same oracle/from-end pattern as VectorIndexing, over the flat 1D indexer.
            for (int i = 0; i < len; i++)
                mat[i] = (fProxy)(i + 1);

            for (int i = 0; i < len; i++)
                Assert.IsTrue(mat[i] == (fProxy)(i + 1));

            for (int k = 1; k <= len; k++)
                Assert.IsTrue(mat[^k] == mat[len - k]);

            Assert.IsTrue(mat[^1] == mat[len - 1]);

            for (int k = 1; k <= len; k++)
                mat[^k] = (fProxy)(1000 + k);

            for (int k = 1; k <= len; k++)
                Assert.IsTrue(mat[len - k] == (fProxy)(1000 + k));

            arena.Dispose();
        }

        public void MatrixIndexing2D()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 16;

            fProxyMxN mat = arena.fProxyMat(rows, cols);

            // Same oracle pattern, via the plain [r, c] indexer; from-end checked per axis below.
            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                    mat[r, c] = (fProxy)(r * cols + c + 1);

            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.IsTrue(mat[r, c] == (fProxy)(r * cols + c + 1));

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

            Assert.IsTrue(mat[^1, ^1] == mat[rows - 1, cols - 1]);

            for (int r = 1; r <= rows; r++)
            for (int c = 1; c <= cols; c++)
                mat[^r, ^c] = (fProxy)(1000 + r * cols + c);

            for (int r = 1; r <= rows; r++)
            for (int c = 1; c <= cols; c++)
                Assert.IsTrue(mat[rows - r, cols - c] == (fProxy)(1000 + r * cols + c));

            arena.Dispose();
        }

        public void RandomCalc()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 16;

            fProxyMxN mat = arena.fProxyMat(rows, cols);

            for(int r = 0; r < rows; r++)
            for(int c = 0; c < cols; c++)
                mat[r, c] = r * c;

            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                mat[r, c] = mat[r, c] + mat[r, c];

            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.IsTrue(mat[r, c] == (fProxy)(r * c * 2));

            arena.Dispose();
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
