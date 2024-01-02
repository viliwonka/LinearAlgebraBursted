using System.Collections;
using System.Collections.Generic;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;

using UnityEngine;
using UnityEngine.TestTools;

public class doubleIndexingTests {
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

            doubleN vec = arena.doubleVec(dim);

            for (int i = 0; i < dim; i++)
                vec[i] = i;

            for (int i = 0; i < dim; i++)
                Assert.AreEqual((double)i, vec[i]);

            for (int i = 0; i < dim; i++)
                vec[^i] = i;

            for (int i = 0; i < dim; i++)
                Assert.AreEqual((double)i, vec[^i]);

            arena.Dispose();
        }

        public void MatrixIndexing1D()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;

            doubleMxN mat = arena.doubleMat(dim, dim);

            int len = dim * dim;

            for (int i = 0; i < len; i++)
                mat[i] = i;

            for (int i = 0; i < len; i++)
                Assert.AreEqual((double)i, mat[i]);

            for (int i = 0; i < len; i++)
                mat[^i] = i;

            for (int i = 0; i < len; i++)
                Assert.AreEqual((double)i, mat[^i]);

            arena.Dispose();
        }

        public void MatrixIndexing2D()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 16;

            doubleMxN mat = arena.doubleMat(rows, cols);
            
            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                    mat[r, c] = r*c;

            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.AreEqual((double)(r * c), mat[r, c]);

            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                    mat[^r, c] = r * c;

            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.AreEqual((double)(r * c), mat[^r, c]);

            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                mat[r, ^c] = r * c;

            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.AreEqual((double)(r * c), mat[r, ^c]);

            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                mat[^r, ^c] = r * c;

            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.AreEqual((double)(r * c), mat[^r, ^c]);

            arena.Dispose();
        }

        public void RandomCalc()
        {
            var arena = new Arena(Allocator.Persistent);

            int rows = 8;
            int cols = 16;

            doubleMxN mat = arena.doubleMat(rows, cols);

            for(int r = 0; r < rows; r++)
            for(int c = 0; c < cols; c++)
                mat[r, c] = r * c;

            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                mat[r, c] = mat[r, c] + mat[r, c];

            for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.AreEqual((double)(r * c * 2), mat[r, c]);
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
