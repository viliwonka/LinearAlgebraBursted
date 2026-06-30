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
        // PRECONDITION: U is non-singular — every diagonal U[r,r] must be nonzero. A zero diagonal
        // (a singular/rank-deficient triangular factor) divides by zero and yields Inf/NaN; this
        // primitive does not guard it. For rank-deficient systems use the rank-revealing paths
        // (Ortho_OP.qrDecompositionColumnPivot, SVD.pinvSolve, or Cholesky.choleskyPivotSolve).
        public static void SolveUpperTriangular(ref doubleMxN U, ref doubleN x)
        {
            if(U.M_Rows < U.N_Cols)
                throw new System.Exception("Solvers.SolveUpperTriangular: Matrix must be square or tall (M_Rows >= N_Cols)");

            if(U.N_Cols != x.N)
                throw new System.Exception("Solvers.SolveUpperTriangular: Matrix and vector must have same number of columns");

            for (int r = U.N_Cols - 1; r >= 0; r--)
            {
                double sum = 0;

                for (int c = r + 1; c < U.N_Cols; c++)
                    sum += U[r, c] * x[c];

                x[r] = (x[r] - sum) / U[r, r];
            }
        }

        // Solve Lx = b for x
        // PRECONDITION: L is non-singular — every diagonal L[r,r] must be nonzero (see
        // SolveUpperTriangular; a zero diagonal divides by zero -> Inf/NaN, unguarded).
        public static void SolveLowerTriangular(ref doubleMxN L, ref doubleN x)
        {
            if (L.IsSquare == false)
                throw new System.Exception("Solvers.SolveLowerTriangular: Matrix must be square");

            if (L.M_Rows != x.N)
                throw new System.Exception("Solvers.SolveLowerTriangular: Matrix and vector must have same number of rows");

            for (int r = 0; r < L.M_Rows; r++)
            {
                double sum = 0;

                for (int c = 0; c < r; c++)
                    sum += L[r, c] * x[c];

                x[r] = (x[r] - sum) / L[r, r];
            }
        }

        // Solve Ly = b for, where y = Ux
        // RP = Row Pivot
        public static void SolveLowerTriangularLU(ref doubleMxN L, in Pivot RP, ref doubleN x) {
            if (L.IsSquare == false)
                throw new System.Exception("Solvers.SolveLowerTriangularLU: Matrix must be square");

            if (L.M_Rows != x.N)
                throw new System.Exception("Solvers.SolveLowerTriangularLU: Matrix and vector must have same number of rows");

            for (int r = 0; r < L.M_Rows; r++) {
                double sum = 0;

                for (int c = 0; c < r; c++)
                    sum += L[RP[r], c] * x[c];

                x[r] = (x[r] - sum);
            }
        }

        public static void SolveUpperTriangularLU(ref doubleMxN U, in Pivot RP, ref doubleN x) {
            if(U.IsSquare == false)
                throw new System.Exception("Solvers.SolveUpperTriangularLU: Matrix must be square");

            if (U.N_Cols != x.N)
                throw new System.Exception("Solvers.SolveUpperTriangularLU: Matrix and vector must have same number of columns");

            for (int r = U.N_Cols - 1; r >= 0; r--) {
                double sum = 0;

                for (int c = r + 1; c < U.N_Cols; c++)
                    sum += U[RP[r], c] * x[c];

                x[r] = (x[r] - sum) / U[RP[r], r];
            }
        }

        /// <summary>
        /// Solve QRx = b for x, with Q,R from a precomputed QR decomposition (solve for multiple
        /// b vectors reusing one decomposition). Caller provides the destination x (length
        /// Q.N_Cols); x must be distinct from b. Zero-alloc: Qᵀb is formed directly into x with
        /// the ref-dest dot — no internal temporary. dim(b) = Q.M_Rows >= dim(x) = Q.N_Cols.
        /// </summary>
        /// <param name="Q">Ortho matrix Q from QR decomposition</param>
        /// <param name="R">Upper triangular matrix R from QR decomposition</param>
        /// <param name="b">Known vector (length Q.M_Rows)</param>
        /// <param name="x">Solution destination (length Q.N_Cols), must not alias b</param>
        public static void SolveQR(ref doubleMxN Q, ref doubleMxN R, ref doubleN b, ref doubleN x) {
            // Solve Ax = b for x
            // A = QR
            // QRx = b
            // Rx = Q^T b
            // x = R^-1 Q^T b

            if (x.N != Q.N_Cols)
                throw new ArgumentException("SolveQR: x.N must equal Q.N_Cols");

            // x = Q^T b (or b^T Q). The ref-dest dot guards x-aliases-b and zeroes x first.
            double_OP.dot(in b, in Q, ref x);
            // Solve Rx = Q^T b for x, in place
            SolveUpperTriangular(ref R, ref x);
        }

        /// <summary>
        /// SolveQR convenience: allocates the solution vector x (length Q.N_Cols) from the arena
        /// and returns it. Use the ref-destination overload in hot loops to avoid the allocation.
        /// </summary>
        public static doubleN SolveQR(ref doubleMxN Q, ref doubleMxN R, ref doubleN b) {
            doubleN x = b.tempdoubleVec(Q.N_Cols);
            SolveQR(ref Q, ref R, ref b, ref x);
            return x;
        }

        // Solve Ax = b for x
        public static void SolveQR(ref doubleMxN A, ref doubleN b, ref doubleN x)
        {
            Ortho_OP.qrDirectSolve(ref A, ref b, ref x);

        }

        /// <summary>
        /// Zero-alloc Conjugate Gradient solver for symmetric positive-definite (SPD) systems A x = b.
        /// Caller provides x (initial guess, overwritten with solution) and three scratch vectors
        /// r, p, Ap (all length A.M_Rows). Returns true if converged within maxIterations to the
        /// relative residual tolerance; false if not converged or non-positive curvature p·Ap <= 0
        /// is encountered (A not SPD or numerical breakdown). On a false return x is undefined
        /// (it may have been partially updated) — only read x when the call returns true.
        /// </summary>
        public static bool conjugateGradient(in doubleMxN A, in doubleN b, ref doubleN x,
                                             ref doubleN r, ref doubleN p, ref doubleN Ap,
                                             int maxIterations, double tolerance)
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

            double bb = double_OP.dot(b, b);

            // b is the zero vector — x = 0 is the exact solution. Copy b (all zeros)
            // rather than multiplying by 0, so a NaN/Inf initial guess is sanitized
            // (NaN * 0 = NaN would otherwise leak through).
            if (bb == (double)0)
            {
                x.Data.CopyFrom(b.Data);
                return true;
            }

            // r = b - A x
            double_OP.dot(in A, in x, ref Ap);           // Ap = A x (temp use of Ap)
            r.Data.CopyFrom(b.Data);                     // r  = b
            r.addScaledInpl((double)(-1), Ap);           // r -= Ap  =>  r = b - A x

            // p = r
            p.Data.CopyFrom(r.Data);

            double rsold = double_OP.dot(r, r);
            double threshold = tolerance * tolerance * bb;

            if (rsold <= threshold)
                return true;

            for (int k = 0; k < maxIterations; k++)
            {
                double_OP.dot(in A, in p, ref Ap);        // Ap = A p

                double pAp = double_OP.dot(p, Ap);

                if (!(pAp > (double)0))                  // NaN-safe: also catches breakdown
                    return false;

                double alpha = rsold / pAp;

                x.addScaledInpl(alpha, p);               // x += alpha p
                r.addScaledInpl(-alpha, Ap);             // r -= alpha Ap

                double rsnew = double_OP.dot(r, r);

                if (rsnew <= threshold)
                    return true;

                double beta = rsnew / rsold;

                p.scaleAddInpl(beta, r);                 // p = beta p + r

                rsold = rsnew;
            }

            return false;
        }

        /// <summary>
        /// Conjugate Gradient solver — allocates three scratch vectors from the arena and calls
        /// the zero-alloc primitive. x is overwritten with the solution on convergence.
        /// </summary>
        public static bool conjugateGradient(in doubleMxN A, in doubleN b, ref doubleN x,
                                             int maxIterations, double tolerance)
        {
            doubleN r  = b.tempdoubleVec(A.M_Rows);
            doubleN p  = b.tempdoubleVec(A.M_Rows);
            doubleN Ap = b.tempdoubleVec(A.M_Rows);
            return conjugateGradient(in A, in b, ref x, ref r, ref p, ref Ap, maxIterations, tolerance);
        }

        /// <summary>
        /// Conjugate Gradient solver with default maxIterations (A.M_Rows) and tolerance
        /// (Consts.doubleSqrtEps). x is overwritten with the solution on convergence.
        /// </summary>
        public static bool conjugateGradient(in doubleMxN A, in doubleN b, ref doubleN x)
        {
            return conjugateGradient(in A, in b, ref x, A.M_Rows, Consts.doubleSqrtEps);
        }
    }

}
