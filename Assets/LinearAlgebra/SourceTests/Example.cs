using System.Collections;
using System.Collections.Generic;

using LinearAlgebra.Stats;

using Unity.Collections;

using UnityEngine;

namespace LinearAlgebra {

    public class Example {
        void Main() {

            // memory management struct
            var arena = new Arena(Allocator.Persistent);

            int dim = 128;
            // creates a zero vector of 128 dimensions 
            floatN vecA = arena.floatVec(dim);
            // creates a vector of 128 dimensions with all elements set to 1
            floatN vecB = arena.floatVec(dim, 1f);

            // add per component (will allocate a new temporary vec)
            floatN vecAdd = vecA + vecB;
            // mul per component (will allocate a new temporary vec)
            floatN vecMul = vecA * vecB;

            // create identity matrix
            floatMxN matI = arena.floatIdentityMatrix(dim);
            floatMxN matRand = arena.floatRandomMatrix(16, 16);

            // per component sum, allocates new matrix
            floatMxN compSumMat = matI + matRand;

            // adds 1f to compSumMat inplace, allocating nothing
            floatOP.addInpl(compSumMat, 1f);

            // mulls matI into compSumMat inplace, allocating nothing 
            floatOP.compMulInpl(compSumMat, matI);

            // creates random matrix with range from -3f to 3f
            floatMxN A = arena.floatRandomDiagonalMatrix(dim, -3f, 3f);
            floatMxN B = arena.floatRandomDiagonalMatrix(dim, -3f, 3f);

            // dot multiply A and B, will allocate new matrix
            floatMxN C = floatOP.dot(A, B);

            // adds 5f to element on [0, 0] coords
            C[0, 0] += 5f;

            floatN b = arena.floatVec(dim, 1f);
            floatN x_result = arena.floatVec(dim, 1f);


            // solves linear system Ax = b inplace using QR, will allocate nothing permament
            // but will modify A and b
            OrthoOP.qrDirectSolve(ref A, ref b, ref x_result);

            // calculate L1 norm
            float norm = floatNormsOP.L1(x_result);

            // prints C matrix, although it will be cutoff because of big dimensions
            Print.Log(C);

            // returns true for all elements c_ij > a_ij, else false
            // will allocate
            boolMxN matCompare = C > A;

            // flips booleans, will allocate
            matCompare = !matCompare;

            // creates 3 new allocations
            boolMxN matCompare2 = C > A | C < B;

            // clears all temporary allocations
            arena.ClearTemp();

            // creates new int vector with dimensions of 10 and valued at 32
            intN intVec = arena.intVec(10, 32);

            // applies bitwise OR to elements, allocates new vector
            intVec |= 64;

            // also allocates, inplace methods do exist though
            intVec = 2 + (intVec << 2) + intVec;

            // creates new integer matrix
            intMxN intMat = arena.intRandomMatrix(10, 10, 0, 10);

            // creates new double matrix
            doubleMxN doubleMat = arena.doubleRandomMatrix(10, 10, 0, 10);

            // creates new short matrix
            shortMxN shortMat = arena.shortRandomMatrix(10, 10, 0, 10);

            // creates new long matrix
            longMxN longMat = arena.longRandomMatrix(10, 10, 0, 10);

            // mean of a vec
            double mean = doubleStatsOP.mean(in doubleMat);

            // mean of a vec
            double max = doubleStatsOP.max(in doubleMat);

            // vector of means of each row
            doubleN rowMean = doubleStatsOP.rowMean(in doubleMat);

            // clears and dispose all allocated vectors/matrices, disposes also arena
            arena.Dispose();
        }

    }

}