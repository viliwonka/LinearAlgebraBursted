#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace LinearAlgebra
{
    /// <summary>           
    /// Inpl = inplace
    /// </summary>
    public static partial class OrthoOP {

        public static void householderInpl(ref floatMxN matrix, in floatN u)
        {
            if(matrix.IsSquare == false)
                throw new System.Exception("OrthoOP.householder: Matrix must be square");

            if(matrix.M_Rows < matrix.N_Cols)
                throw new System.Exception("OrthoOP.householder: Matrix must be square or tall (more or equal rows than cols)");

            var maxDim = math.max(matrix.M_Rows, matrix.N_Cols);

            if(u.N < maxDim)
                throw new System.Exception("OrthoOP.householder: Vector must be at least as long as the largest dimension of the matrix");

            float vTv = floatOP.dot(u, u); // Inline dot product calculation
            
            float scaleFactor = 2 / vTv;

            for (int i = 0; i < matrix.M_Rows; i++)
            {
                for (int j = 0; j < matrix.N_Cols; j++)
                {
                    float vvT_element = scaleFactor * u[i] * u[j];
                    matrix[i, j] -= vvT_element; // Apply directly to matrix
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void genHouseholder(ref floatMxN Q, ref floatN u, int k) {
            // copy column d of A into u
            // here we are forming x vector
            for (int r = k; r < Q.M_Rows; r++)
                u[r] = Q[r, k];

            // norm of vector, with range
            float xNorm = floatNormsOP.L2Range(u, k, u.N);

            // Check if xNorm is zero or very small
            if (Math.Abs(xNorm) < Consts.floatZeroTreshold) {
                // Set v to be 'identity' vector
                for (int r = k; r < u.N; r++)
                    u[r] = (r == k) ? 1 : 0;

                return; // Early return as no further processing is needed
            }

            // v = x + sign(x[0]) * ||x|| * e_0
            u[k] += sign(u[k]) * xNorm;

            floatNormsOP.NormalizeL2(u, k, u.N);
        }

        static float sign(float x) {
            return x < 0 ? -1 : 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void genHouseholderPete(ref floatMxN Q, ref floatN u, int k) {

            // copy column d of A into u
            // here we are forming x vector
            for (int r = k; r < u.N; r++)
                u[r] = Q[r, k];
            
            float xNorm = floatNormsOP.L2Range(u, k, u.N);

            if (math.abs(xNorm) > Consts.floatZeroTreshold) {

                for (int r = k; r < u.N; r++)
                    u[r] = u[r] / xNorm;

                u[k] = u[k] + sign(u[k]);

                var div = math.sqrt(math.abs(u[k]));
                for (int r = k; r < u.N; r++) {
                    u[r] = u[r] / div;
                }
            }
            else {

                u[k] = math.SQRT2;
                //for (int r = k; r < v.N; r++)
                //    v[k] = (r == k) ? math.SQRT2 : 0;
            }
        }

        // Q is original matrix A, R is identity matrix
        // Q becomes orthogonal matrix, R becomes upper triangular matrix
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void qrDecomposition(ref floatMxN Q, ref floatMxN R)
        {
            if (Q.M_Rows < Q.N_Cols)
                throw new System.Exception("OrthoOP.qrDecomposition: Matrix R must be square or tall (more or equal rows than cols)");

            // vector to be copied from A below diagonal
            // auxilary vector, will be reused in each step
            var u = new floatN(Q.M_Rows, Allocator.Temp, false);

            int qrSteps = Q.N_Cols;

            // forming R inside Q (will be copied into R later)
            // d = step and diagonal index
            for (int d = 0; d < qrSteps; d++)
            {
                genHouseholderPete(ref Q, ref u, d);;
                                 
                for (int c = d; c < Q.N_Cols; c++) 
                {
                    float dotProduct = 0;
                    for (int k = d; k < Q.M_Rows; k++)
                    {
                        dotProduct += u[k] * Q[k, c];
                    }

                    //dotProduct *= 2;
                    for (int r = d; r < Q.M_Rows; r++)
                    {
                        Q[r, c] -= u[r] * dotProduct;
                    }
                }

                // copy current Q diagonal element into R
                // it will be over-written in the next step
                R[d, d] = Q[d, d];

                // copy v into Q below diagonal, will be used to reconstruct Q
                for (int i = d; i < Q.M_Rows; i++)
                {
                    Q[i, d] = u[i];
                }
            }
            // End or R orthogonalization construction

            // Copy the upper triangular part of Q into R
            for (int r = 0; r < R.M_Rows; r++)
            for (int c = 0; c < R.N_Cols; c++)
            {
                if (c < r)
                {
                    // Below diagonal, set to 0
                    R[r, c] = 0;
                }
                else if (c > r)
                {
                    // above diagonal, copy from Q
                    R[r, c] = Q[r, c];
                }
            }

            /// Reconstruct Q from vectors stored inside Q columns

            // Initialize upper part of Q to identity matrix, including diagonals
            for (int r = 0; r < Q.M_Rows; r++)
            {
                for (int c = r; c < Q.N_Cols; c++)
                {
                    // On and above diagonal
                    if (c > r)
                    {   
                        Q[r, c] = 0;
                    }
                }
            }
            
            // Apply Householder transformations in reverse order
            // Reconstruct the Householder vector v from the original Q
            for (int d = Q.N_Cols - 1; d >= 0; d--)
            {               
                // includes diagonal elements
                for (int i = d; i < Q.M_Rows; i++)
                {
                    u[i] = Q[i, d];
                    Q[i, d] = i == d? 1 : 0;
                }

                // Apply the Householder transformation to Q
                for (int c = d; c < Q.N_Cols; c++) {

                    float dotProduct = 0;
                    for (int k = d; k < Q.M_Rows; k++) {
                        dotProduct += u[k] * Q[k, c];
                    }
                    //dotProduct *= 2;
                    for (int r = d; r < Q.M_Rows; r++) {
                        Q[r, c] -= u[r] * dotProduct;
                    }
                }
            }

            u.Dispose();
        }

        // Q is original matrix A that will be turned into R (upper triangular) non square matrix
        // Q becomes R
        // b will be transformed into y, where y = Q^T b, and then solved for x
        // x is the solution
        // Q and b get modified (destroyed)
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void qrDirectSolve(ref floatMxN A, ref floatN b, ref floatN x) {
            if (A.M_Rows < A.N_Cols)
                throw new System.Exception("OrthoOP.qrDecomposition: Matrix R must be square or tall (more or equal rows than cols)");

            // vector to be copied from A below diagonal
            // auxilary vector, will be reused in each step
            var u = new floatN(A.M_Rows, Allocator.Temp, false);

            int qrSteps = A.N_Cols;

            float dotProduct = 0; 
            // forming R inside Q (will be copied into R later)
            // d = step and diagonal index
            for (int d = 0; d < qrSteps; d++) {

                genHouseholderPete(ref A, ref u, d);

                for (int c = d; c < A.N_Cols; c++) {

                    dotProduct = 0;
                    for (int r = d; r < A.M_Rows; r++)
                        dotProduct += u[r] * A[r, c];

                    //dotProduct *= 2;
                    for (int r = d; r < A.M_Rows; r++)
                        A[r, c] -= u[r] * dotProduct;
                }

                // apply same transformation to b vector
                dotProduct = 0;
                for (int r = d; r < A.M_Rows; r++)
                    dotProduct += u[r] * b[r];

                //dotProduct *= 2;
                for (int r = d; r < A.M_Rows; r++)
                    b[r] -= u[r] * dotProduct;
            }

            // copy b into x (x may be smaller dimension than b)
            for (int r = 0; r < A.N_Cols; r++)
                x[r] = b[r];

            // b was transformed to y, where y = Q^T b
            // Solve Rx = y

            Solvers.SolveUpperTriangular(ref A, ref x);

            u.Dispose();
        }
    }

}
