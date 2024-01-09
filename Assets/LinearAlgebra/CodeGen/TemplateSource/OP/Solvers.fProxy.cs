#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS 

using Unity.Collections;

namespace LinearAlgebra
{
    /// <summary>           
    /// Inpl = inplace
    /// </summary>
    public static partial class Solvers {

        // Solve Ux = b for x
        public static void SolveUpperTriangular(ref fProxyMxN U, ref fProxyN x)
        {
            //if(U.IsSquare == false)
                //throw new System.Exception("Solvers.SolveUpperTriangular: Matrix must be square");

            if(U.N_Cols != x.N)
                throw new System.Exception("Solvers.SolveUpperTriangular: Matrix and vector must have same number of columns");

            for (int r = U.N_Cols - 1; r >= 0; r--)
            {
                fProxy sum = 0;

                for (int c = r + 1; c < U.N_Cols; c++)
                    sum += U[r, c] * x[c];
                
                x[r] = (x[r] - sum) / U[r, r];
            }
        }

        // Solve Lx = b for x
        public static void SolveLowerTriangular(ref fProxyMxN L, ref fProxyN x)
        {
            if (L.IsSquare == false)
                throw new System.Exception("Solvers.SolveLowerTriangular: Matrix must be square");

            if (L.M_Rows != x.N)
                throw new System.Exception("Solvers.SolveLowerTriangular: Matrix and vector must have same number of rows");

            for (int r = 0; r < L.M_Rows; r++)
            {
                fProxy sum = 0;

                for (int c = 0; c < r; c++)
                    sum += L[r, c] * x[c];

                x[r] = (x[r] - sum) / L[r, r];
            }
        }

        // Solve Ly = b for, where y = Ux
        public static void SolveLowerTriangularLU(ref fProxyMxN L, ref fProxyN x) {
            if (L.IsSquare == false)
                throw new System.Exception("Solvers.SolveLowerTriangular: Matrix must be square");

            if (L.M_Rows != x.N)
                throw new System.Exception("Solvers.SolveLowerTriangular: Matrix and vector must have same number of rows");

            for (int r = 0; r < L.M_Rows; r++) {
                fProxy sum = 0;

                for (int c = 0; c < r; c++)
                    sum += L[r, c] * x[c];

                x[r] = (x[r] - sum);
            }
        }

        // Solve Ly = b for, where y = Ux
        // RP = Row Pivot
        public static void SolveLowerTriangularLU(ref fProxyMxN L, ref Pivot RP, ref fProxyN x) {
            if (L.IsSquare == false)
                throw new System.Exception("Solvers.SolveLowerTriangular: Matrix must be square");

            if (L.M_Rows != x.N)
                throw new System.Exception("Solvers.SolveLowerTriangular: Matrix and vector must have same number of rows");

            for (int r = 0; r < L.M_Rows; r++) {
                fProxy sum = 0;

                for (int c = 0; c < r; c++)
                    sum += L[RP[r], c] * x[c];

                x[r] = (x[r] - sum);
            }
        }

        public static void SolveUpperTriangularLU(ref fProxyMxN U, ref Pivot RP, ref fProxyN x) {
            //if(U.IsSquare == false)
            //throw new System.Exception("Solvers.SolveUpperTriangular: Matrix must be square");

            if (U.N_Cols != x.N)
                throw new System.Exception("Solvers.SolveUpperTriangular: Matrix and vector must have same number of columns");

            for (int r = U.N_Cols - 1; r >= 0; r--) {
                fProxy sum = 0;

                for (int c = r + 1; c < U.N_Cols; c++)
                    sum += U[RP[r], c] * x[c];

                x[r] = (x[r] - sum) / U[RP[r], r];
            }
        }

        /// <summary>
        /// Solve QRx = b for x
        /// Use if you intend to solve for multiple b vectors, you have to compute QR decomposition only once
        /// dim(b) >= dim(x)
        /// </summary>
        /// <param name="Q">Ortho matrix Q from QR decomposition</param>
        /// <param name="R">Upper triangular matrix R from QR decomposition</param>
        /// <param name="b">Known vector</param>
        /// <param name="x">Unknown vector</param>
        public static void SolveQR(ref fProxyMxN Q, ref fProxyMxN R, ref fProxyN b, out fProxyN x) {
            // Solve Ax = b for x
            // A = QR
            // QRx = b
            // Rx = Q^T b
            // x = R^-1 Q^T b

            // y = Q^T b (or b^T Q)
            fProxyN y = fProxyOP.dot(b, Q);
            // Solve Rx = Q^T b for x
            SolveUpperTriangular(ref R, ref y);

            x = y;
        }

        // Solve Ax = b for x                                                                     
        public static void SolveQR(ref fProxyMxN A, ref fProxyN b, ref fProxyN x)
        {            
            OrthoOP.qrDirectSolve(ref A, ref b, ref x);

        }
    }

}
