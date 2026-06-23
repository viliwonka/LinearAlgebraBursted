#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using Unity.Collections;

namespace LinearAlgebra
{
    /// <summary>
    /// Inpl = inplace
    /// </summary>
    public static partial class Solvers {

        // Solve Ux = b for x
        // U may be tall (M_Rows >= N_Cols): only the top N_Cols x N_Cols block is read,
        // which is the R block produced by QR on overdetermined systems.
        public static void SolveUpperTriangular(ref fProxyMxN U, ref fProxyN x)
        {
            if(U.M_Rows < U.N_Cols)
                throw new System.Exception("Solvers.SolveUpperTriangular: Matrix must be square or tall (M_Rows >= N_Cols)");

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
        // RP = Row Pivot
        public static void SolveLowerTriangularLU(ref fProxyMxN L, in Pivot RP, ref fProxyN x) {
            if (L.IsSquare == false)
                throw new System.Exception("Solvers.SolveLowerTriangularLU: Matrix must be square");

            if (L.M_Rows != x.N)
                throw new System.Exception("Solvers.SolveLowerTriangularLU: Matrix and vector must have same number of rows");

            for (int r = 0; r < L.M_Rows; r++) {
                fProxy sum = 0;

                for (int c = 0; c < r; c++)
                    sum += L[RP[r], c] * x[c];

                x[r] = (x[r] - sum);
            }
        }

        public static void SolveUpperTriangularLU(ref fProxyMxN U, in Pivot RP, ref fProxyN x) {
            if(U.IsSquare == false)
                throw new System.Exception("Solvers.SolveUpperTriangularLU: Matrix must be square");

            if (U.N_Cols != x.N)
                throw new System.Exception("Solvers.SolveUpperTriangularLU: Matrix and vector must have same number of columns");

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

        /// <summary>
        /// Zero-alloc Conjugate Gradient solver for symmetric positive-definite (SPD) systems A x = b.
        /// Caller provides x (initial guess, overwritten with solution) and three scratch vectors
        /// r, p, Ap (all length A.M_Rows). Returns true if converged within maxIterations to the
        /// relative residual tolerance; false if not converged or non-positive curvature p·Ap <= 0
        /// is encountered (A not SPD or numerical breakdown). On a false return x is undefined
        /// (it may have been partially updated) — only read x when the call returns true.
        /// </summary>
        public static bool conjugateGradient(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                             ref fProxyN r, ref fProxyN p, ref fProxyN Ap,
                                             int maxIterations, fProxy tolerance)
        {
            if (!A.IsSquare)
                throw new ArgumentException("conjugateGradient: A must be square");

            if (b.N != A.M_Rows)
                throw new ArgumentException("conjugateGradient: b.N must equal A.M_Rows");

            if (x.N != A.M_Rows)
                throw new ArgumentException("conjugateGradient: x.N must equal A.M_Rows");

            if (r.N != A.M_Rows)
                throw new ArgumentException("conjugateGradient: r.N must equal A.M_Rows");

            if (p.N != A.M_Rows)
                throw new ArgumentException("conjugateGradient: p.N must equal A.M_Rows");

            if (Ap.N != A.M_Rows)
                throw new ArgumentException("conjugateGradient: Ap.N must equal A.M_Rows");

            if (maxIterations < 1)
                throw new ArgumentException("conjugateGradient: maxIterations must be >= 1");

            fProxy bb = fProxyOP.dot(b, b);

            // b is the zero vector — x = 0 is the exact solution. Copy b (all zeros)
            // rather than multiplying by 0, so a NaN/Inf initial guess is sanitized
            // (NaN * 0 = NaN would otherwise leak through).
            if (bb == (fProxy)0)
            {
                x.Data.CopyFrom(b.Data);
                return true;
            }

            // r = b - A x
            fProxyOP.dot(in A, in x, ref Ap);           // Ap = A x (temp use of Ap)
            r.Data.CopyFrom(b.Data);                     // r  = b
            r.addScaledInpl((fProxy)(-1), Ap);           // r -= Ap  =>  r = b - A x

            // p = r
            p.Data.CopyFrom(r.Data);

            fProxy rsold = fProxyOP.dot(r, r);
            fProxy threshold = tolerance * tolerance * bb;

            if (rsold <= threshold)
                return true;

            for (int k = 0; k < maxIterations; k++)
            {
                fProxyOP.dot(in A, in p, ref Ap);        // Ap = A p

                fProxy pAp = fProxyOP.dot(p, Ap);

                if (!(pAp > (fProxy)0))                  // NaN-safe: also catches breakdown
                    return false;

                fProxy alpha = rsold / pAp;

                x.addScaledInpl(alpha, p);               // x += alpha p
                r.addScaledInpl(-alpha, Ap);             // r -= alpha Ap

                fProxy rsnew = fProxyOP.dot(r, r);

                if (rsnew <= threshold)
                    return true;

                fProxy beta = rsnew / rsold;

                p.scaleAddInpl(beta, r);                 // p = beta p + r

                rsold = rsnew;
            }

            return false;
        }

        /// <summary>
        /// Conjugate Gradient solver — allocates three scratch vectors from the arena and calls
        /// the zero-alloc primitive. x is overwritten with the solution on convergence.
        /// </summary>
        public static bool conjugateGradient(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                             int maxIterations, fProxy tolerance)
        {
            fProxyN r  = b.tempfProxyVec(A.M_Rows);
            fProxyN p  = b.tempfProxyVec(A.M_Rows);
            fProxyN Ap = b.tempfProxyVec(A.M_Rows);
            return conjugateGradient(in A, in b, ref x, ref r, ref p, ref Ap, maxIterations, tolerance);
        }

        /// <summary>
        /// Conjugate Gradient solver with default maxIterations (A.M_Rows) and tolerance
        /// (Consts.fProxySqrtEps). x is overwritten with the solution on convergence.
        /// </summary>
        public static bool conjugateGradient(in fProxyMxN A, in fProxyN b, ref fProxyN x)
        {
            return conjugateGradient(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }
    }

}
