#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using Unity.Collections;
using LinearAlgebra.Sparse;

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
        // (QR.qrDecompositionColumnPivot, SVD.pinvSolve, or Cholesky.choleskyPivotSolve).
        public static void solveUpperTriangular(ref floatMxN U, ref floatN x)
        {
            if(U.M_Rows < U.N_Cols)
                throw new ArgumentException("Solvers.solveUpperTriangular: Matrix must be square or tall (M_Rows >= N_Cols)");

            if(U.N_Cols != x.N)
                throw new ArgumentException("Solvers.solveUpperTriangular: Matrix and vector must have same number of columns");

            for (int r = U.N_Cols - 1; r >= 0; r--)
            {
                float sum = 0;

                for (int c = r + 1; c < U.N_Cols; c++)
                    sum += U[r, c] * x[c];

                x[r] = (x[r] - sum) / U[r, r];
            }
        }

        // Solve Lx = b for x
        // PRECONDITION: L is non-singular — every diagonal L[r,r] must be nonzero (see
        // solveUpperTriangular; a zero diagonal divides by zero -> Inf/NaN, unguarded).
        public static void solveLowerTriangular(ref floatMxN L, ref floatN x)
        {
            if (L.IsSquare == false)
                throw new ArgumentException("Solvers.solveLowerTriangular: Matrix must be square");

            if (L.M_Rows != x.N)
                throw new ArgumentException("Solvers.solveLowerTriangular: Matrix and vector must have same number of rows");

            for (int r = 0; r < L.M_Rows; r++)
            {
                float sum = 0;

                for (int c = 0; c < r; c++)
                    sum += L[r, c] * x[c];

                x[r] = (x[r] - sum) / L[r, r];
            }
        }

        // Solve Ly = b for, where y = Ux
        // RP = Row Pivot
        public static void solveLowerTriangularLU(ref floatMxN L, in Pivot RP, ref floatN x) {
            if (L.IsSquare == false)
                throw new ArgumentException("Solvers.solveLowerTriangularLU: Matrix must be square");

            if (L.M_Rows != x.N)
                throw new ArgumentException("Solvers.solveLowerTriangularLU: Matrix and vector must have same number of rows");

            for (int r = 0; r < L.M_Rows; r++) {
                float sum = 0;

                for (int c = 0; c < r; c++)
                    sum += L[RP[r], c] * x[c];

                x[r] = (x[r] - sum);
            }
        }

        public static void solveUpperTriangularLU(ref floatMxN U, in Pivot RP, ref floatN x) {
            if(U.IsSquare == false)
                throw new ArgumentException("Solvers.solveUpperTriangularLU: Matrix must be square");

            if (U.N_Cols != x.N)
                throw new ArgumentException("Solvers.solveUpperTriangularLU: Matrix and vector must have same number of columns");

            for (int r = U.N_Cols - 1; r >= 0; r--) {
                float sum = 0;

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
        public static void solveQR(ref floatMxN Q, ref floatMxN R, ref floatN b, ref floatN x) {
            // Solve Ax = b for x
            // A = QR
            // QRx = b
            // Rx = Q^T b
            // x = R^-1 Q^T b

            if (x.N != Q.N_Cols)
                throw new ArgumentException("solveQR: x.N must equal Q.N_Cols");

            // x = Q^T b (or b^T Q). The ref-dest dot guards x-aliases-b and zeroes x first.
            Linear_OP.dot(in b, in Q, ref x);
            // Solve Rx = Q^T b for x, in place
            solveUpperTriangular(ref R, ref x);
        }

        /// <summary>
        /// solveQR convenience: allocates the solution vector x (length Q.N_Cols) from the arena
        /// and returns it. Use the ref-destination overload in hot loops to avoid the allocation.
        /// </summary>
        public static floatN solveQR(ref floatMxN Q, ref floatMxN R, ref floatN b) {
            floatN x = b.tempfloatVec(Q.N_Cols);
            solveQR(ref Q, ref R, ref b, ref x);
            return x;
        }

        // Solve Ax = b for x
        public static void solveQR(ref floatMxN A, ref floatN b, ref floatN x)
        {
            QR.qrDirectSolve(ref A, ref b, ref x);

        }

        /// <summary>
        /// Zero-alloc Conjugate Gradient solver for symmetric positive-definite (SPD) systems A x = b,
        /// generic over any <see cref="IfloatLinearOperator"/> (Burst-monomorphized static
        /// dispatch, no vtable/managed delegate). This is the SINGLE SOURCE OF TRUTH for the CG
        /// loop — the concrete dense (<c>conjugateGradient(in floatMxN, ...)</c>) and BSM
        /// (<c>conjugateGradient(in floatBSM, ...)</c>) overloads below are thin forwarders that
        /// wrap their matrix in <see cref="floatDenseOperator"/> / <c>floatBSMOperator</c> and
        /// call this method.
        ///
        /// Caller provides x (initial guess, overwritten with solution — WARM-STARTABLE: seed x
        /// with a previous solution to resume/refine) and three scratch vectors r, p, Ap (all
        /// length A.Rows). Returns true if converged within maxIterations to the relative residual
        /// tolerance; false if not converged or non-positive curvature p·Ap <= 0 is encountered (A
        /// not SPD or numerical breakdown). On a false return x is undefined (it may have been
        /// partially updated) — only read x when the call returns true.
        /// </summary>
        public static bool cg<TOp>(in TOp A, in floatN b, ref floatN x,
                                   ref floatN r, ref floatN p, ref floatN Ap,
                                   int maxIterations, float tolerance)
            where TOp : struct, IfloatLinearOperator
        {
            if (A.Rows != A.Cols)
                throw new ArgumentException("cg: A must be square");

            if (b.N != A.Rows)
                throw new ArgumentException("cg: b.N must equal A.Rows");

            if (x.N != A.Rows)
                throw new ArgumentException("cg: x.N must equal A.Rows");

            if (r.N != A.Rows)
                throw new ArgumentException("cg: r.N must equal A.Rows");

            if (p.N != A.Rows)
                throw new ArgumentException("cg: p.N must equal A.Rows");

            if (Ap.N != A.Rows)
                throw new ArgumentException("cg: Ap.N must equal A.Rows");

            if (maxIterations < 1)
                throw new ArgumentException("cg: maxIterations must be >= 1");

            // Aliasing guard: the loop below mixes plain elementwise scratch updates
            // (addScaledInpl/scaleAddInpl) with reads of "old" values, and those primitives do
            // NOT self-check aliasing the way A.Apply's own dot/spMV call does. E.g. r aliasing
            // Ap turns `r.addScaledInpl(-1, Ap)` (r -= Ap) into r -= r == 0 elementwise -- a
            // silent false convergence instead of a thrown exception. Check every pair up front.
            unsafe
            {
                float* rPtr = r.Data.Ptr, pPtr = p.Data.Ptr, ApPtr = Ap.Data.Ptr, xPtr = x.Data.Ptr, bPtr = b.Data.Ptr;

                if (rPtr == pPtr || rPtr == ApPtr || rPtr == xPtr || rPtr == bPtr ||
                    pPtr == ApPtr || pPtr == xPtr || pPtr == bPtr ||
                    ApPtr == xPtr || ApPtr == bPtr ||
                    xPtr == bPtr)
                    throw new ArgumentException("cg: r/p/Ap/x/b must be distinct");
            }

            float bb = Linear_OP.dot(b, b);

            // b is the zero vector — x = 0 is the exact solution. Copy b (all zeros)
            // rather than multiplying by 0, so a NaN/Inf initial guess is sanitized
            // (NaN * 0 = NaN would otherwise leak through).
            if (bb == (float)0)
            {
                x.Data.CopyFrom(b.Data);
                return true;
            }

            // r = b - A x
            A.Apply(in x, ref Ap);                       // Ap = A x (temp use of Ap)
            r.Data.CopyFrom(b.Data);                     // r  = b
            r.addScaledInpl((float)(-1), Ap);           // r -= Ap  =>  r = b - A x

            // p = r
            p.Data.CopyFrom(r.Data);

            float rsold = Linear_OP.dot(r, r);
            float threshold = tolerance * tolerance * bb;

            if (rsold <= threshold)
                return true;

            for (int k = 0; k < maxIterations; k++)
            {
                A.Apply(in p, ref Ap);                    // Ap = A p

                float pAp = Linear_OP.dot(p, Ap);

                if (!(pAp > (float)0))                  // NaN-safe: also catches breakdown
                    return false;

                float alpha = rsold / pAp;

                x.addScaledInpl(alpha, p);               // x += alpha p
                r.addScaledInpl(-alpha, Ap);             // r -= alpha Ap

                float rsnew = Linear_OP.dot(r, r);

                if (rsnew <= threshold)
                    return true;

                float beta = rsnew / rsold;

                p.scaleAddInpl(beta, r);                 // p = beta p + r

                rsold = rsnew;
            }

            return false;
        }

        /// <summary>
        /// Zero-alloc Conjugate Gradient solver for symmetric positive-definite (SPD) systems A x = b.
        /// Caller provides x (initial guess, overwritten with solution) and three scratch vectors
        /// r, p, Ap (all length A.M_Rows). Returns true if converged within maxIterations to the
        /// relative residual tolerance; false if not converged or non-positive curvature p·Ap <= 0
        /// is encountered (A not SPD or numerical breakdown). On a false return x is undefined
        /// (it may have been partially updated) — only read x when the call returns true.
        /// Forwards into <see cref="cg{TOp}"/> via <see cref="floatDenseOperator"/> — see that
        /// method for the actual loop.
        /// </summary>
        public static bool conjugateGradient(in floatMxN A, in floatN b, ref floatN x,
                                             ref floatN r, ref floatN p, ref floatN Ap,
                                             int maxIterations, float tolerance)
        {
            return cg(new floatDenseOperator(in A), in b, ref x, ref r, ref p, ref Ap, maxIterations, tolerance);
        }

        /// <summary>
        /// Conjugate Gradient solver — allocates three scratch vectors from the arena and calls
        /// the zero-alloc primitive. x is overwritten with the solution on convergence.
        /// </summary>
        public static bool conjugateGradient(in floatMxN A, in floatN b, ref floatN x,
                                             int maxIterations, float tolerance)
        {
            floatN r  = b.tempfloatVec(A.M_Rows);
            floatN p  = b.tempfloatVec(A.M_Rows);
            floatN Ap = b.tempfloatVec(A.M_Rows);
            return conjugateGradient(in A, in b, ref x, ref r, ref p, ref Ap, maxIterations, tolerance);
        }

        /// <summary>
        /// Conjugate Gradient solver with default maxIterations (A.M_Rows) and tolerance
        /// (Consts.floatSqrtEps). x is overwritten with the solution on convergence.
        /// </summary>
        public static bool conjugateGradient(in floatMxN A, in floatN b, ref floatN x)
        {
            return conjugateGradient(in A, in b, ref x, A.M_Rows, Consts.floatSqrtEps);
        }

        /// <summary>
        /// Conjugate Gradient solver over a block-sparse (BSR) SPD matrix. Same semantics as
        /// the dense overload — see <see cref="conjugateGradient(in floatMxN, in floatN, ref floatN, ref floatN, ref floatN, ref floatN, int, float)"/>.
        /// Forwards into <see cref="cg{TOp}"/> via <c>floatBSMOperator</c>.
        /// </summary>
        public static bool conjugateGradient(in floatBSM A, in floatN b, ref floatN x,
                                             ref floatN r, ref floatN p, ref floatN Ap,
                                             int maxIterations, float tolerance)
        {
            return cg(new floatBSMOperator(in A), in b, ref x, ref r, ref p, ref Ap, maxIterations, tolerance);
        }

        /// <summary>
        /// Conjugate Gradient solver over a block-sparse (BSR) SPD matrix — allocates three
        /// scratch vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static bool conjugateGradient(in floatBSM A, in floatN b, ref floatN x,
                                             int maxIterations, float tolerance)
        {
            floatN r  = b.tempfloatVec(A.M_Rows);
            floatN p  = b.tempfloatVec(A.M_Rows);
            floatN Ap = b.tempfloatVec(A.M_Rows);
            return conjugateGradient(in A, in b, ref x, ref r, ref p, ref Ap, maxIterations, tolerance);
        }

        /// <summary>
        /// Conjugate Gradient solver over a block-sparse (BSR) SPD matrix, with default
        /// maxIterations (A.M_Rows) and tolerance (Consts.floatSqrtEps).
        /// </summary>
        public static bool conjugateGradient(in floatBSM A, in floatN b, ref floatN x)
        {
            return conjugateGradient(in A, in b, ref x, A.M_Rows, Consts.floatSqrtEps);
        }

        /// <summary>
        /// Zero-alloc Preconditioned Conjugate Gradient solver for SPD systems A x = b, generic
        /// over both the operator (<see cref="IfloatLinearOperator"/>) and the preconditioner
        /// (<see cref="IfloatPreconditioner"/>) — same Burst static-dispatch shape as
        /// <see cref="cg{TOp}"/>. Standard PCG: p is combined with z = M⁻¹r (not r), and β uses
        /// ⟨r,z⟩ instead of ⟨r,r⟩.
        ///
        /// Caller provides x (initial guess, overwritten with solution — warm-startable) and four
        /// scratch vectors r, p, Ap, z (all length A.Rows). The convergence test compares the
        /// TRUE (unpreconditioned) residual ||r||² against tolerance²·||b||² — the same criterion
        /// as <see cref="cg{TOp}"/> — so iteration counts between cg and pcg on the same system are
        /// directly comparable. Returns true if converged within maxIterations; false if not
        /// converged or non-positive curvature p·Ap <= 0 is encountered. On a false return x is
        /// undefined — only read x when the call returns true.
        /// </summary>
        public static bool pcg<TOp, TPre>(in TOp A, in TPre M, in floatN b, ref floatN x,
                                          ref floatN r, ref floatN p, ref floatN Ap, ref floatN z,
                                          int maxIterations, float tolerance)
            where TOp : struct, IfloatLinearOperator
            where TPre : struct, IfloatPreconditioner
        {
            if (A.Rows != A.Cols)
                throw new ArgumentException("pcg: A must be square");

            if (b.N != A.Rows)
                throw new ArgumentException("pcg: b.N must equal A.Rows");

            if (x.N != A.Rows)
                throw new ArgumentException("pcg: x.N must equal A.Rows");

            if (r.N != A.Rows)
                throw new ArgumentException("pcg: r.N must equal A.Rows");

            if (p.N != A.Rows)
                throw new ArgumentException("pcg: p.N must equal A.Rows");

            if (Ap.N != A.Rows)
                throw new ArgumentException("pcg: Ap.N must equal A.Rows");

            if (z.N != A.Rows)
                throw new ArgumentException("pcg: z.N must equal A.Rows");

            if (maxIterations < 1)
                throw new ArgumentException("pcg: maxIterations must be >= 1");

            // Aliasing guard -- see the matching comment in cg<TOp>. z joins the set here since
            // PCG additionally mixes p/r into the preconditioned residual via M.Apply / axpy.
            unsafe
            {
                float* rPtr = r.Data.Ptr, pPtr = p.Data.Ptr, ApPtr = Ap.Data.Ptr, zPtr = z.Data.Ptr, xPtr = x.Data.Ptr, bPtr = b.Data.Ptr;

                if (rPtr == pPtr || rPtr == ApPtr || rPtr == zPtr || rPtr == xPtr || rPtr == bPtr ||
                    pPtr == ApPtr || pPtr == zPtr || pPtr == xPtr || pPtr == bPtr ||
                    ApPtr == zPtr || ApPtr == xPtr || ApPtr == bPtr ||
                    zPtr == xPtr || zPtr == bPtr ||
                    xPtr == bPtr)
                    throw new ArgumentException("pcg: r/p/Ap/z/x/b must be distinct");
            }

            float bb = Linear_OP.dot(b, b);

            if (bb == (float)0)
            {
                x.Data.CopyFrom(b.Data);
                return true;
            }

            // r = b - A x
            A.Apply(in x, ref Ap);
            r.Data.CopyFrom(b.Data);
            r.addScaledInpl((float)(-1), Ap);

            float threshold = tolerance * tolerance * bb;

            if (Linear_OP.dot(r, r) <= threshold)
                return true;

            // z = M^-1 r ; p = z
            M.Apply(in r, ref z);
            p.Data.CopyFrom(z.Data);

            float rzold = Linear_OP.dot(r, z);

            // Block-Jacobi is SPD so this never trips on the shipped path, but a user-supplied
            // preconditioner is not guaranteed SPD; a non-positive <r,z> yields a wrong-signed
            // alpha/beta and silent divergence instead of a clean bailout. Mirrors cg's
            // NaN-safe !(pAp > 0) breakdown guard.
            if (!(rzold > (float)0))
                return false;

            for (int k = 0; k < maxIterations; k++)
            {
                A.Apply(in p, ref Ap);                    // Ap = A p

                float pAp = Linear_OP.dot(p, Ap);

                if (!(pAp > (float)0))                  // NaN-safe: also catches breakdown
                    return false;

                float alpha = rzold / pAp;

                x.addScaledInpl(alpha, p);               // x += alpha p
                r.addScaledInpl(-alpha, Ap);             // r -= alpha Ap

                if (Linear_OP.dot(r, r) <= threshold)
                    return true;

                M.Apply(in r, ref z);                     // z = M^-1 r

                float rznew = Linear_OP.dot(r, z);

                if (!(rznew > (float)0))                 // NaN-safe: same breakdown guard, fresh <r,z>
                    return false;

                float beta = rznew / rzold;

                p.scaleAddInpl(beta, z);                 // p = beta p + z

                rzold = rznew;
            }

            return false;
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient solver — allocates four scratch vectors from the
        /// arena and calls the zero-alloc primitive.
        /// </summary>
        public static bool pcg<TOp, TPre>(in TOp A, in TPre M, in floatN b, ref floatN x,
                                          int maxIterations, float tolerance)
            where TOp : struct, IfloatLinearOperator
            where TPre : struct, IfloatPreconditioner
        {
            floatN r  = b.tempfloatVec(A.Rows);
            floatN p  = b.tempfloatVec(A.Rows);
            floatN Ap = b.tempfloatVec(A.Rows);
            floatN z  = b.tempfloatVec(A.Rows);
            return pcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIterations, tolerance);
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient solver with default maxIterations (A.Rows) and
        /// tolerance (Consts.floatSqrtEps).
        /// </summary>
        public static bool pcg<TOp, TPre>(in TOp A, in TPre M, in floatN b, ref floatN x)
            where TOp : struct, IfloatLinearOperator
            where TPre : struct, IfloatPreconditioner
        {
            return pcg(in A, in M, in b, ref x, A.Rows, Consts.floatSqrtEps);
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient over a block-sparse (BSR) SPD matrix with its
        /// matching block-Jacobi preconditioner. Forwards into <see cref="pcg{TOp,TPre}"/> via
        /// <c>floatBSMOperator</c>.
        /// </summary>
        public static bool pcg(in floatBSM A, in floatBlockJacobi M, in floatN b, ref floatN x,
                               ref floatN r, ref floatN p, ref floatN Ap, ref floatN z,
                               int maxIterations, float tolerance)
        {
            return pcg(new floatBSMOperator(in A), in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIterations, tolerance);
        }

        /// <summary>
        /// Block-Jacobi Preconditioned Conjugate Gradient over a BSR SPD matrix — allocates four
        /// scratch vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static bool pcg(in floatBSM A, in floatBlockJacobi M, in floatN b, ref floatN x,
                               int maxIterations, float tolerance)
        {
            floatN r  = b.tempfloatVec(A.M_Rows);
            floatN p  = b.tempfloatVec(A.M_Rows);
            floatN Ap = b.tempfloatVec(A.M_Rows);
            floatN z  = b.tempfloatVec(A.M_Rows);
            return pcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIterations, tolerance);
        }

        /// <summary>
        /// Block-Jacobi Preconditioned Conjugate Gradient over a BSR SPD matrix, with default
        /// maxIterations (A.M_Rows) and tolerance (Consts.floatSqrtEps).
        /// </summary>
        public static bool pcg(in floatBSM A, in floatBlockJacobi M, in floatN b, ref floatN x)
        {
            return pcg(in A, in M, in b, ref x, A.M_Rows, Consts.floatSqrtEps);
        }
    }

}
