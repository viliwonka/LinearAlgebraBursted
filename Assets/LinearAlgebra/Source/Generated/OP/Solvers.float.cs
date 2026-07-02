#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using Unity.Collections;
using Unity.Mathematics;
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

        // ===================================================================================
        // Phase 3: MINRES (symmetric indefinite), BiCGSTAB (non-symmetric), CGLS/LSQR
        // (rectangular least-squares). Same generic-operator pattern as cg&lt;TOp&gt;/
        // pcg&lt;TOp,TPre&gt; above -- see cg&lt;TOp&gt;'s doc comment for the shared "why an
        // up-front aliasing guard" rationale. These four solvers carry more scratch vectors than
        // cg/pcg (6-9 vs 3-4), so their guards use RequireDistinctBuffers (a small loop-based
        // helper) instead of a hand-expanded OR chain -- see that helper's doc comment.
        // ===================================================================================

        /// <summary>
        /// Zero-alloc MINRES (Paige-Saunders) solver for symmetric systems A x = b, generic over
        /// <see cref="IfloatLinearOperator"/> (Burst-monomorphized static dispatch, no vtable).
        /// Unlike <see cref="cg{TOp}"/>, A need NOT be positive definite -- MINRES minimizes the
        /// 2-norm residual ‖b-Ax‖ over the same Krylov subspace via a short Lanczos recurrence
        /// plus an incrementally-updated QR factorization (Givens rotations) of the resulting
        /// tridiagonal system, so it converges cleanly on symmetric INDEFINITE (and singular/
        /// semidefinite) systems where CG's p·Ap&gt;0 curvature requirement breaks down. A MUST
        /// be symmetric -- this is a caller precondition, not verified at runtime (same contract
        /// as CG's "A must be SPD").
        ///
        /// Caller provides x (initial guess, overwritten with solution -- WARM-STARTABLE) and
        /// seven scratch vectors (y, r1, r2, v, w, w1, w2, all length A.Rows) matching the
        /// classic MINRES variable names (Paige &amp; Saunders 1975; Choi/Saunders' minres.m).
        /// y/r1/r2 carry the 3-term Lanczos recurrence; v is the current normalized Lanczos
        /// vector; w/w1/w2 carry the 3-term search-direction recurrence driven by the
        /// Givens-rotated tridiagonal system. A well-known MINRES identity means the true
        /// residual norm ‖b-Ax‖ falls out of the recurrence for free (the running <c>phibar</c>
        /// variable) -- no extra dot product or matvec is needed to test convergence.
        ///
        /// Returns true if the residual falls within the relative tolerance (‖r‖ &lt;=
        /// tolerance*‖b‖) inside maxIterations; false if not converged, or if the Lanczos
        /// recurrence exactly exhausts the Krylov subspace short of tolerance (beta==0, an
        /// exact-arithmetic invariant-subspace breakdown). On a false return x is undefined (it
        /// may have been partially updated) -- only read x when the call returns true.
        /// </summary>
        public static bool minres<TOp>(in TOp A, in floatN b, ref floatN x,
                                       ref floatN y, ref floatN r1, ref floatN r2, ref floatN v,
                                       ref floatN w, ref floatN w1, ref floatN w2,
                                       int maxIterations, float tolerance)
            where TOp : struct, IfloatLinearOperator
        {
            if (A.Rows != A.Cols)
                throw new ArgumentException("minres: A must be square");

            if (b.N != A.Rows) throw new ArgumentException("minres: b.N must equal A.Rows");
            if (x.N != A.Rows) throw new ArgumentException("minres: x.N must equal A.Rows");
            if (y.N != A.Rows) throw new ArgumentException("minres: y.N must equal A.Rows");
            if (r1.N != A.Rows) throw new ArgumentException("minres: r1.N must equal A.Rows");
            if (r2.N != A.Rows) throw new ArgumentException("minres: r2.N must equal A.Rows");
            if (v.N != A.Rows) throw new ArgumentException("minres: v.N must equal A.Rows");
            if (w.N != A.Rows) throw new ArgumentException("minres: w.N must equal A.Rows");
            if (w1.N != A.Rows) throw new ArgumentException("minres: w1.N must equal A.Rows");
            if (w2.N != A.Rows) throw new ArgumentException("minres: w2.N must equal A.Rows");

            if (maxIterations < 1)
                throw new ArgumentException("minres: maxIterations must be >= 1");

            unsafe
            {
                long* ptrs = stackalloc long[9];
                ptrs[0] = (long)y.Data.Ptr;  ptrs[1] = (long)r1.Data.Ptr; ptrs[2] = (long)r2.Data.Ptr;
                ptrs[3] = (long)v.Data.Ptr;  ptrs[4] = (long)w.Data.Ptr;  ptrs[5] = (long)w1.Data.Ptr;
                ptrs[6] = (long)w2.Data.Ptr; ptrs[7] = (long)x.Data.Ptr;  ptrs[8] = (long)b.Data.Ptr;
                RequireDistinctBuffers("minres: y/r1/r2/v/w/w1/w2/x/b must be distinct", ptrs, 9);
            }

            float bb = Linear_OP.dot(b, b);

            if (bb == (float)0)
            {
                x.Data.CopyFrom(b.Data);
                return true;
            }

            // r1 = b - A x
            A.Apply(in x, ref y);                       // y = A x (temp use of y)
            r1.Data.CopyFrom(b.Data);
            r1.addScaledInpl((float)(-1), y);           // r1 = b - A x

            float beta1 = math.sqrt(Linear_OP.dot(r1, r1));
            float threshold = tolerance * tolerance * bb;

            if (beta1 * beta1 <= threshold)
                return true;

            r2.Data.CopyFrom(r1.Data);

            // Zero the 3-term search-direction history (w/w1/w2 start at 0 in exact MINRES).
            for (int i = 0; i < A.Rows; i++) { w[i] = (float)0; w1[i] = (float)0; w2[i] = (float)0; }

            float oldb = (float)0;
            float beta = beta1;
            float dbar = (float)0;
            float epsln = (float)0;
            float phibar = beta1;
            float cs = (float)(-1);
            float sn = (float)0;
            float gammaFloor = Consts.floatEpsilon;

            for (int k = 0; k < maxIterations; k++)
            {
                // ---- Lanczos step: extend the tridiagonalization by one vector ----
                v.Data.CopyFrom(r2.Data);
                v.divInpl(beta);                          // v = r2 / beta

                A.Apply(in v, ref y);                      // y = A v

                if (k >= 1)
                    y.addScaledInpl(-(beta / oldb), r1);   // y -= (beta/oldb) r1

                float alfa = Linear_OP.dot(v, y);
                y.addScaledInpl(-(alfa / beta), r2);       // y -= (alfa/beta) r2

                r1.Data.CopyFrom(r2.Data);
                r2.Data.CopyFrom(y.Data);

                oldb = beta;
                beta = math.sqrt(Linear_OP.dot(r2, r2));

                // ---- apply the PREVIOUS Givens rotation (cs,sn) to the new tridiagonal column ----
                float oldeps = epsln;
                float delta = cs * dbar + sn * alfa;
                float gbar = sn * dbar - cs * alfa;
                epsln = sn * beta;
                dbar = -cs * beta;

                // ---- compute the NEW Givens rotation that zeros the subdiagonal entry ----
                float gamma = math.sqrt(gbar * gbar + beta * beta);
                gamma = math.max(gamma, gammaFloor);
                cs = gbar / gamma;
                sn = beta / gamma;
                float phi = cs * phibar;
                phibar = sn * phibar;

                // ---- update the 3-term search direction, then the solution ----
                w1.Data.CopyFrom(w2.Data);
                w2.Data.CopyFrom(w.Data);

                w.Data.CopyFrom(v.Data);
                w.addScaledInpl(-oldeps, w1);
                w.addScaledInpl(-delta, w2);
                w.divInpl(gamma);                          // w = (v - oldeps*w1 - delta*w2) / gamma

                x.addScaledInpl(phi, w);

                // phibar IS the true residual norm ‖b-Ax‖ at this step (MINRES identity) --
                // no extra dot product needed.
                if (phibar * phibar <= threshold)
                    return true;

                if (!(beta > (float)0))
                    break; // Lanczos breakdown: invariant subspace exhausted, no further progress possible
            }

            return false;
        }

        /// <summary>
        /// MINRES over a dense <see cref="floatMxN"/> -- zero-alloc primitive. Forwards into
        /// <see cref="minres{TOp}"/> via <see cref="floatDenseOperator"/>. See that method for
        /// the actual loop and buffer semantics.
        /// </summary>
        public static bool minres(in floatMxN A, in floatN b, ref floatN x,
                                  ref floatN y, ref floatN r1, ref floatN r2, ref floatN v,
                                  ref floatN w, ref floatN w1, ref floatN w2,
                                  int maxIterations, float tolerance)
        {
            return minres(new floatDenseOperator(in A), in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, maxIterations, tolerance);
        }

        /// <summary>MINRES over a dense matrix -- allocates seven scratch vectors from the arena.</summary>
        public static bool minres(in floatMxN A, in floatN b, ref floatN x, int maxIterations, float tolerance)
        {
            floatN y  = b.tempfloatVec(A.M_Rows);
            floatN r1 = b.tempfloatVec(A.M_Rows);
            floatN r2 = b.tempfloatVec(A.M_Rows);
            floatN v  = b.tempfloatVec(A.M_Rows);
            floatN w  = b.tempfloatVec(A.M_Rows);
            floatN w1 = b.tempfloatVec(A.M_Rows);
            floatN w2 = b.tempfloatVec(A.M_Rows);
            return minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, maxIterations, tolerance);
        }

        /// <summary>MINRES over a dense matrix with default maxIterations (A.M_Rows) and tolerance (Consts.floatSqrtEps).</summary>
        public static bool minres(in floatMxN A, in floatN b, ref floatN x)
        {
            return minres(in A, in b, ref x, A.M_Rows, Consts.floatSqrtEps);
        }

        /// <summary>
        /// MINRES over a symmetric block-sparse (BSR) matrix -- zero-alloc primitive. Forwards
        /// into <see cref="minres{TOp}"/> via <c>floatBSMOperator</c>.
        /// </summary>
        public static bool minres(in floatBSM A, in floatN b, ref floatN x,
                                  ref floatN y, ref floatN r1, ref floatN r2, ref floatN v,
                                  ref floatN w, ref floatN w1, ref floatN w2,
                                  int maxIterations, float tolerance)
        {
            return minres(new floatBSMOperator(in A), in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, maxIterations, tolerance);
        }

        /// <summary>MINRES over a BSR matrix -- allocates seven scratch vectors from the arena.</summary>
        public static bool minres(in floatBSM A, in floatN b, ref floatN x, int maxIterations, float tolerance)
        {
            floatN y  = b.tempfloatVec(A.M_Rows);
            floatN r1 = b.tempfloatVec(A.M_Rows);
            floatN r2 = b.tempfloatVec(A.M_Rows);
            floatN v  = b.tempfloatVec(A.M_Rows);
            floatN w  = b.tempfloatVec(A.M_Rows);
            floatN w1 = b.tempfloatVec(A.M_Rows);
            floatN w2 = b.tempfloatVec(A.M_Rows);
            return minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, maxIterations, tolerance);
        }

        /// <summary>MINRES over a BSR matrix with default maxIterations (A.M_Rows) and tolerance (Consts.floatSqrtEps).</summary>
        public static bool minres(in floatBSM A, in floatN b, ref floatN x)
        {
            return minres(in A, in b, ref x, A.M_Rows, Consts.floatSqrtEps);
        }

        /// <summary>
        /// Zero-alloc BiCGSTAB (van der Vorst 1992, stabilized Bi-Conjugate Gradient) solver for
        /// NON-symmetric (general) square systems A x = b, generic over
        /// <see cref="IfloatLinearOperator"/>. Short two-sided recurrence, flat O(n) memory (no
        /// growing Krylov basis like GMRES) -- the non-symmetric counterpart to CG/MINRES, for
        /// e.g. frictional-LCP or MNA-circuit operators.
        ///
        /// Caller provides x (initial guess, overwritten with solution -- WARM-STARTABLE) and
        /// five scratch vectors r, rHat0, p, v, t (all length A.Rows). r doubles as the
        /// intermediate "s" half-step residual from the classic two-half-step presentation (s = r
        /// - alpha*v, updated into r in place -- the standard buffer-count reduction); rHat0 is
        /// the fixed "shadow" residual (rHat0 = r0, chosen once at the start and never mutated
        /// after).
        ///
        /// Returns true if the residual falls within the relative tolerance (‖r‖ &lt;=
        /// tolerance*‖b‖) inside maxIterations; false on non-convergence or one of the standard
        /// BiCGSTAB breakdowns (rho == 0, rHat0·v == 0, or omega == 0 -- A not amenable to
        /// BiCGSTAB from this shadow residual, or numerical breakdown). On a false return x is
        /// undefined (it may have been partially updated) -- only read x when the call returns
        /// true.
        /// </summary>
        public static bool biCGStab<TOp>(in TOp A, in floatN b, ref floatN x,
                                         ref floatN r, ref floatN rHat0, ref floatN p, ref floatN v, ref floatN t,
                                         int maxIterations, float tolerance)
            where TOp : struct, IfloatLinearOperator
        {
            if (A.Rows != A.Cols)
                throw new ArgumentException("biCGStab: A must be square");

            if (b.N != A.Rows) throw new ArgumentException("biCGStab: b.N must equal A.Rows");
            if (x.N != A.Rows) throw new ArgumentException("biCGStab: x.N must equal A.Rows");
            if (r.N != A.Rows) throw new ArgumentException("biCGStab: r.N must equal A.Rows");
            if (rHat0.N != A.Rows) throw new ArgumentException("biCGStab: rHat0.N must equal A.Rows");
            if (p.N != A.Rows) throw new ArgumentException("biCGStab: p.N must equal A.Rows");
            if (v.N != A.Rows) throw new ArgumentException("biCGStab: v.N must equal A.Rows");
            if (t.N != A.Rows) throw new ArgumentException("biCGStab: t.N must equal A.Rows");

            if (maxIterations < 1)
                throw new ArgumentException("biCGStab: maxIterations must be >= 1");

            unsafe
            {
                long* ptrs = stackalloc long[7];
                ptrs[0] = (long)r.Data.Ptr; ptrs[1] = (long)rHat0.Data.Ptr; ptrs[2] = (long)p.Data.Ptr;
                ptrs[3] = (long)v.Data.Ptr; ptrs[4] = (long)t.Data.Ptr;
                ptrs[5] = (long)x.Data.Ptr; ptrs[6] = (long)b.Data.Ptr;
                RequireDistinctBuffers("biCGStab: r/rHat0/p/v/t/x/b must be distinct", ptrs, 7);
            }

            float bb = Linear_OP.dot(b, b);

            if (bb == (float)0)
            {
                x.Data.CopyFrom(b.Data);
                return true;
            }

            // r = b - A x
            A.Apply(in x, ref v);                          // v = A x (temp use, overwritten below)
            r.Data.CopyFrom(b.Data);
            r.addScaledInpl((float)(-1), v);

            float threshold = tolerance * tolerance * bb;

            if (Linear_OP.dot(r, r) <= threshold)
                return true;

            rHat0.Data.CopyFrom(r.Data);

            // p_0 = v_0 = 0 (standard BiCGSTAB init).
            for (int i = 0; i < A.Rows; i++) { p[i] = (float)0; v[i] = (float)0; }

            float rho = (float)1, alpha = (float)1, omega = (float)1;

            for (int k = 0; k < maxIterations; k++)
            {
                float rhoNew = Linear_OP.dot(rHat0, r);

                if (rhoNew == (float)0 || math.isnan(rhoNew))
                    return false; // serious breakdown: r has gone orthogonal to the shadow residual

                float beta = (rhoNew / rho) * (alpha / omega);

                p.addScaledInpl(-omega, v);                // p -= omega v      (still old p, old v)
                p.scaleAddInpl(beta, r);                    // p = beta p + r

                A.Apply(in p, ref v);                       // v = A p

                float rv = Linear_OP.dot(rHat0, v);

                if (rv == (float)0 || math.isnan(rv))
                    return false; // breakdown: alpha undefined

                alpha = rhoNew / rv;

                r.addScaledInpl(-alpha, v);                 // r := s = r - alpha v

                float ss = Linear_OP.dot(r, r);

                if (ss <= threshold)
                {
                    // Early exit: the half-step residual s is already small enough -- finish
                    // with x += alpha p (skipping the t = A s stabilization matvec entirely).
                    x.addScaledInpl(alpha, p);
                    return true;
                }

                A.Apply(in r, ref t);                       // t = A s   (r currently holds s)

                float tt = Linear_OP.dot(t, t);

                if (!(tt > (float)0))                       // NaN-safe: tt is a norm^2, nonnegative
                    return false; // breakdown: omega undefined

                omega = Linear_OP.dot(t, r) / tt;

                if (omega == (float)0 || math.isnan(omega))
                    return false; // breakdown: next iteration's beta would divide by zero

                x.addScaledInpl(alpha, p);
                x.addScaledInpl(omega, r);                  // r still holds s here

                r.addScaledInpl(-omega, t);                 // r := s - omega t   (new residual)

                float rr = Linear_OP.dot(r, r);

                if (rr <= threshold)
                    return true;

                rho = rhoNew;
            }

            return false;
        }

        /// <summary>
        /// BiCGSTAB over a dense <see cref="floatMxN"/> -- zero-alloc primitive. Forwards into
        /// <see cref="biCGStab{TOp}"/> via <see cref="floatDenseOperator"/>.
        /// </summary>
        public static bool biCGStab(in floatMxN A, in floatN b, ref floatN x,
                                    ref floatN r, ref floatN rHat0, ref floatN p, ref floatN v, ref floatN t,
                                    int maxIterations, float tolerance)
        {
            return biCGStab(new floatDenseOperator(in A), in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, maxIterations, tolerance);
        }

        /// <summary>BiCGSTAB over a dense matrix -- allocates five scratch vectors from the arena.</summary>
        public static bool biCGStab(in floatMxN A, in floatN b, ref floatN x, int maxIterations, float tolerance)
        {
            floatN r     = b.tempfloatVec(A.M_Rows);
            floatN rHat0 = b.tempfloatVec(A.M_Rows);
            floatN p     = b.tempfloatVec(A.M_Rows);
            floatN v     = b.tempfloatVec(A.M_Rows);
            floatN t     = b.tempfloatVec(A.M_Rows);
            return biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, maxIterations, tolerance);
        }

        /// <summary>BiCGSTAB over a dense matrix with default maxIterations (A.M_Rows) and tolerance (Consts.floatSqrtEps).</summary>
        public static bool biCGStab(in floatMxN A, in floatN b, ref floatN x)
        {
            return biCGStab(in A, in b, ref x, A.M_Rows, Consts.floatSqrtEps);
        }

        /// <summary>
        /// BiCGSTAB over a block-sparse (BSR) matrix -- zero-alloc primitive. Forwards into
        /// <see cref="biCGStab{TOp}"/> via <c>floatBSMOperator</c>.
        /// </summary>
        public static bool biCGStab(in floatBSM A, in floatN b, ref floatN x,
                                    ref floatN r, ref floatN rHat0, ref floatN p, ref floatN v, ref floatN t,
                                    int maxIterations, float tolerance)
        {
            return biCGStab(new floatBSMOperator(in A), in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, maxIterations, tolerance);
        }

        /// <summary>BiCGSTAB over a BSR matrix -- allocates five scratch vectors from the arena.</summary>
        public static bool biCGStab(in floatBSM A, in floatN b, ref floatN x, int maxIterations, float tolerance)
        {
            floatN r     = b.tempfloatVec(A.M_Rows);
            floatN rHat0 = b.tempfloatVec(A.M_Rows);
            floatN p     = b.tempfloatVec(A.M_Rows);
            floatN v     = b.tempfloatVec(A.M_Rows);
            floatN t     = b.tempfloatVec(A.M_Rows);
            return biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, maxIterations, tolerance);
        }

        /// <summary>BiCGSTAB over a BSR matrix with default maxIterations (A.M_Rows) and tolerance (Consts.floatSqrtEps).</summary>
        public static bool biCGStab(in floatBSM A, in floatN b, ref floatN x)
        {
            return biCGStab(in A, in b, ref x, A.M_Rows, Consts.floatSqrtEps);
        }

        /// <summary>
        /// Zero-alloc CGLS solver for RECTANGULAR least-squares systems: minimizes ‖Ax-b‖₂ for
        /// possibly non-square A (over- or under-determined), generic over
        /// <see cref="IfloatLinearOperator"/>. This is CG applied to the normal equations
        /// AᵀA x = Aᵀb, but NEVER explicitly forms AᵀA -- every AᵀA-vector product is one
        /// <see cref="IfloatLinearOperator.Apply"/> plus one
        /// <see cref="IfloatLinearOperator.ApplyT"/>. The normal-equation residual s = Aᵀr is
        /// recomputed FRESH from r = b-Ax every iteration (rather than updated incrementally via
        /// s -= alpha·Aᵀq), the numerically-stable "CGLS" variant (Björck, "Numerical Methods for
        /// Least Squares Problems", Algorithm 7.17) rather than the classic-but-drift-prone CGNR
        /// update -- same op count (1 Apply + 1 ApplyT/iteration), just recomputed instead of
        /// incrementally accumulated.
        ///
        /// Caller provides x (initial guess, length A.Cols -- overwritten with solution, WARM-
        /// STARTABLE) and four scratch vectors: r, q (length A.Rows) and s, p (length A.Cols).
        /// Converges when the normal-equation residual ‖Aᵀr‖ falls within the relative tolerance
        /// (‖Aᵀr‖ &lt;= tolerance*‖Aᵀb‖, a fixed scale reference independent of x0, mirroring cg's
        /// ‖b‖ reference). For a CONSISTENT system (b in range(A)) this drives r itself to zero
        /// (exact recovery); for an INCONSISTENT system it converges to the least-squares
        /// solution with r left orthogonal to range(A) (‖Aᵀr‖≈0, ‖r‖ generally nonzero -- the
        /// normal-equations optimality condition).
        ///
        /// <paramref name="damp"/> (&gt;= 0) applies Tikhonov regularization: minimizes
        /// ‖Ax-b‖² + damp²‖x‖², i.e. runs CG on the SHIFTED normal equations (AᵀA + damp²I)x = Aᵀb
        /// -- the residual becomes s = Aᵀr - damp²x and the curvature ‖Ap‖² + damp²‖p‖², never
        /// forming AᵀA. damp == 0 is BIT-IDENTICAL to the plain solve. Because s uses the FULL x
        /// (not the residual), cgls regularizes ‖x‖ for ANY initial x -- warm start included --
        /// unlike lsqr/lsmr, which regularize ‖x - x₀‖ under a nonzero warm start.
        ///
        /// Returns false on non-convergence or non-positive curvature ‖Ap‖²&lt;=0 (breakdown,
        /// mirrors cg's p·Ap&lt;=0 guard: p is in null(A), or p==0). On a false return x is
        /// undefined -- only read x when the call returns true.
        /// </summary>
        public static bool cgls<TOp>(in TOp A, in floatN b, ref floatN x,
                                     ref floatN r, ref floatN s, ref floatN p, ref floatN q,
                                     int maxIterations, float tolerance, float damp)
            where TOp : struct, IfloatLinearOperator
        {
            if (b.N != A.Rows) throw new ArgumentException("cgls: b.N must equal A.Rows");
            if (x.N != A.Cols) throw new ArgumentException("cgls: x.N must equal A.Cols");
            if (r.N != A.Rows) throw new ArgumentException("cgls: r.N must equal A.Rows");
            if (q.N != A.Rows) throw new ArgumentException("cgls: q.N must equal A.Rows");
            if (s.N != A.Cols) throw new ArgumentException("cgls: s.N must equal A.Cols");
            if (p.N != A.Cols) throw new ArgumentException("cgls: p.N must equal A.Cols");

            if (maxIterations < 1)
                throw new ArgumentException("cgls: maxIterations must be >= 1");

            unsafe
            {
                long* ptrs = stackalloc long[6];
                ptrs[0] = (long)r.Data.Ptr; ptrs[1] = (long)s.Data.Ptr; ptrs[2] = (long)p.Data.Ptr;
                ptrs[3] = (long)q.Data.Ptr; ptrs[4] = (long)x.Data.Ptr; ptrs[5] = (long)b.Data.Ptr;
                RequireDistinctBuffers("cgls: r/s/p/q/x/b must be distinct", ptrs, 6);
            }

            // Fixed scale reference for the relative tolerance, independent of x0 (mirrors cg's
            // bb = dot(b,b)): AtB = A^T b, atbSq = ||AtB||^2. s doubles as scratch for this
            // one-off computation -- the main loop overwrites it every iteration from here on.
            A.ApplyT(in b, ref s);
            float atbSq = Linear_OP.dot(s, s);

            if (atbSq == (float)0)
            {
                // A^T b == 0 -> x=0 is a valid least-squares minimizer regardless of warm start
                // (mirrors cg's bb==0 shortcut: a deterministic, NaN-sanitizing exact answer).
                for (int i = 0; i < x.N; i++) x[i] = (float)0;
                return true;
            }

            float threshold = tolerance * tolerance * atbSq;

            // r = b - A x
            A.Apply(in x, ref q);                          // q = A x (temp use of q)
            r.Data.CopyFrom(b.Data);
            r.addScaledInpl((float)(-1), q);

            // s = A^T r - damp^2 x  (damped: the residual of the normal equations
            // (A^T A + damp^2 I) x = A^T b; damp==0 -> s = A^T r exactly, bit-identical).
            A.ApplyT(in r, ref s);
            if (damp != (float)0) s.addScaledInpl(-(damp * damp), x);

            float gamma = Linear_OP.dot(s, s);

            if (gamma <= threshold)
                return true;

            p.Data.CopyFrom(s.Data);

            for (int k = 0; k < maxIterations; k++)
            {
                A.Apply(in p, ref q);                       // q = A p

                float delta = Linear_OP.dot(q, q);
                if (damp != (float)0) delta += (damp * damp) * Linear_OP.dot(p, p);   // p^T(A^T A + damp^2 I)p

                if (!(delta > (float)0))                   // NaN-safe: also catches breakdown
                    return false;

                float alpha = gamma / delta;

                x.addScaledInpl(alpha, p);
                r.addScaledInpl(-alpha, q);

                A.ApplyT(in r, ref s);                       // s = A^T r, recomputed fresh (stability)
                if (damp != (float)0) s.addScaledInpl(-(damp * damp), x);   // - damp^2 x (damped gradient)

                float gammaNew = Linear_OP.dot(s, s);

                if (gammaNew <= threshold)
                    return true;

                float beta = gammaNew / gamma;

                p.scaleAddInpl(beta, s);                     // p = beta p + s

                gamma = gammaNew;
            }

            return false;
        }

        /// <summary>Undamped CGLS (damp = 0): plain least-squares. Forwards to the damped core.</summary>
        public static bool cgls<TOp>(in TOp A, in floatN b, ref floatN x,
                                     ref floatN r, ref floatN s, ref floatN p, ref floatN q,
                                     int maxIterations, float tolerance)
            where TOp : struct, IfloatLinearOperator
            => cgls(in A, in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance, (float)0);

        /// <summary>
        /// CGLS over a dense <see cref="floatMxN"/> (possibly rectangular) -- zero-alloc
        /// primitive. Forwards into <see cref="cgls{TOp}"/> via <see cref="floatDenseOperator"/>.
        /// </summary>
        public static bool cgls(in floatMxN A, in floatN b, ref floatN x,
                                ref floatN r, ref floatN s, ref floatN p, ref floatN q,
                                int maxIterations, float tolerance)
        {
            return cgls(new floatDenseOperator(in A), in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance);
        }

        /// <summary>CGLS over a dense matrix -- allocates four scratch vectors from the arena.</summary>
        public static bool cgls(in floatMxN A, in floatN b, ref floatN x, int maxIterations, float tolerance)
        {
            floatN r = b.tempfloatVec(A.M_Rows);
            floatN s = b.tempfloatVec(A.N_Cols);
            floatN p = b.tempfloatVec(A.N_Cols);
            floatN q = b.tempfloatVec(A.M_Rows);
            return cgls(in A, in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance);
        }

        /// <summary>
        /// Damped (Tikhonov) CGLS over a dense matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates
        /// four scratch vectors from the arena. damp == 0 reproduces the plain least-squares solve.
        /// </summary>
        public static bool cgls(in floatMxN A, in floatN b, ref floatN x, int maxIterations, float tolerance, float damp)
        {
            floatN r = b.tempfloatVec(A.M_Rows);
            floatN s = b.tempfloatVec(A.N_Cols);
            floatN p = b.tempfloatVec(A.N_Cols);
            floatN q = b.tempfloatVec(A.M_Rows);
            return cgls(new floatDenseOperator(in A), in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance, damp);
        }

        /// <summary>CGLS over a dense matrix with default maxIterations (A.N_Cols) and tolerance (Consts.floatSqrtEps).</summary>
        public static bool cgls(in floatMxN A, in floatN b, ref floatN x)
        {
            return cgls(in A, in b, ref x, A.N_Cols, Consts.floatSqrtEps);
        }

        /// <summary>
        /// CGLS over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc primitive.
        /// Forwards into <see cref="cgls{TOp}"/> via <c>floatBSMOperator</c>. This is the payoff
        /// of rectangular BR x BC blocks: matrix-free least squares over a sparse Jacobian-like
        /// operator, never forming AᵀA.
        /// </summary>
        public static bool cgls(in floatBSM A, in floatN b, ref floatN x,
                                ref floatN r, ref floatN s, ref floatN p, ref floatN q,
                                int maxIterations, float tolerance)
        {
            return cgls(new floatBSMOperator(in A), in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance);
        }

        /// <summary>
        /// CGLS over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc primitive
        /// variant that takes a CALLER-PROVIDED precomputed transpose AT (e.g. built once via
        /// <c>arena.floatBSMTranspose(in A)</c> outside a hot loop / before a benchmark's timed
        /// region) and routes every ApplyT call through the resulting cache-friendly forward
        /// spMV(AT, x) instead of the scatter-heavy on-the-fly spMVT(A, x) -- see
        /// <see cref="floatBSMOperator"/>'s two-arg ctor. Caller is responsible for AT actually
        /// being A's transpose; this overload does not verify it. Prefer this over the allocating
        /// <see cref="cgls(in floatBSM, in floatN, ref floatN, int, float)"/> overload when
        /// solving repeatedly against the same A (build AT once, reuse it across many solves).
        /// </summary>
        public static bool cgls(in floatBSM A, in floatBSM AT, in floatN b, ref floatN x,
                                ref floatN r, ref floatN s, ref floatN p, ref floatN q,
                                int maxIterations, float tolerance)
        {
            return cgls(new floatBSMOperator(in A, in AT), in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance);
        }

        /// <summary>
        /// CGLS over a BSR matrix -- allocates four scratch vectors AND materializes A^T ONCE
        /// via <c>arena.floatBSMTranspose</c> (same arena as the scratch vectors, taken from
        /// b), then drives CGLS with the two-arg <see cref="floatBSMOperator"/> so every
        /// ApplyT call routes through a cache-friendly forward spMV(A^T, x) instead of the
        /// scatter-heavy on-the-fly spMVT(A, x) every iteration -- this is the fix for the
        /// rectangular CGLS/LSQR transpose-matvec cache-unfriendliness (the one-time O(nnz)
        /// transpose build is amortized over every iteration). For a build-free zero-alloc path
        /// (e.g. many solves reusing the same A), build A^T yourself once (<c>arena.
        /// floatBSMTranspose(in A)</c>) and call the zero-alloc <see cref="cgls(in floatBSM,
        /// in floatBSM, in floatN, ref floatN, ref floatN, ref floatN, ref floatN, ref
        /// floatN, int, float)"/> overload above with your own scratch vectors, or the generic
        /// <see cref="cgls{TOp}"/> overload directly with <c>new floatBSMOperator(in A, in
        /// AT)</c>.
        /// </summary>
        public static bool cgls(in floatBSM A, in floatN b, ref floatN x, int maxIterations, float tolerance)
        {
            floatN r = b.tempfloatVec(A.M_Rows);
            floatN s = b.tempfloatVec(A.N_Cols);
            floatN p = b.tempfloatVec(A.N_Cols);
            floatN q = b.tempfloatVec(A.M_Rows);
            floatBSM AT = b.floatBSMTranspose(in A);
            return cgls(new floatBSMOperator(in A, in AT), in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance);
        }

        /// <summary>
        /// Damped (Tikhonov) CGLS over a BSR matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates four
        /// scratch vectors AND materializes A^T once (see the undamped allocating overload). damp == 0
        /// reproduces the plain least-squares solve.
        /// </summary>
        public static bool cgls(in floatBSM A, in floatN b, ref floatN x, int maxIterations, float tolerance, float damp)
        {
            floatN r = b.tempfloatVec(A.M_Rows);
            floatN s = b.tempfloatVec(A.N_Cols);
            floatN p = b.tempfloatVec(A.N_Cols);
            floatN q = b.tempfloatVec(A.M_Rows);
            floatBSM AT = b.floatBSMTranspose(in A);
            return cgls(new floatBSMOperator(in A, in AT), in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance, damp);
        }

        /// <summary>CGLS over a BSR matrix with default maxIterations (A.N_Cols) and tolerance (Consts.floatSqrtEps).</summary>
        public static bool cgls(in floatBSM A, in floatN b, ref floatN x)
        {
            return cgls(in A, in b, ref x, A.N_Cols, Consts.floatSqrtEps);
        }

        /// <summary>
        /// Zero-alloc LSQR (Paige-Saunders 1982) solver for RECTANGULAR least-squares systems:
        /// minimizes ‖Ax-b‖₂ for possibly non-square A, generic over
        /// <see cref="IfloatLinearOperator"/>. Builds an implicit bidiagonalization of A via the
        /// Golub-Kahan process (alternating <see cref="IfloatLinearOperator.Apply"/> /
        /// <see cref="IfloatLinearOperator.ApplyT"/> calls) and folds it through an incremental
        /// Givens-rotated QR factorization -- the same "short recurrence + running rotation"
        /// shape as <see cref="minres{TOp}"/>, generalized to rectangular A. More robust than
        /// <see cref="cgls{TOp}"/> on ill-conditioned A (the bidiagonalization never squares A's
        /// condition number the way the normal equations implicitly do), at the same O(n+m)
        /// memory and per-iteration cost (1 Apply + 1 ApplyT).
        ///
        /// Caller provides x (initial guess, length A.Cols -- overwritten with solution, WARM-
        /// STARTABLE) and five scratch vectors: u, tmpM (length A.Rows) and v, w, tmpN (length
        /// A.Cols). The normal-equation residual norm ‖Aᵀr‖ (arnorm) falls out of the recurrence
        /// for free (no extra ApplyT call, a well-known LSQR identity) -- same convergence
        /// contract as <see cref="cgls{TOp}"/>: converges when ‖Aᵀr‖ &lt;= tolerance*‖Aᵀb‖.
        ///
        /// <paramref name="damp"/> (&gt;= 0) applies Tikhonov regularization: minimizes
        /// ‖Ax-b‖² + damp²‖x‖² (equivalently solves (AᵀA + damp²I)x = Aᵀb) via one extra rotation
        /// per step folding damp into the bidiagonal diagonal -- no augmented matrix formed.
        /// damp == 0 is BIT-IDENTICAL to the plain least-squares solve.
        ///
        /// WARM START + DAMPING: like <see cref="lsmr{TOp}"/>, lsqr bidiagonalizes the residual
        /// b - A·x₀, so a NONZERO initial x₀ makes it minimize ‖Ax-b‖² + damp²‖x - x₀‖² (regularizing
        /// the CORRECTION), not ‖x‖. Start from x = 0 for the ‖x‖-regularized minimizer. (cgls
        /// regularizes ‖x‖ for any x₀.)
        ///
        /// Returns false on non-convergence or a total bidiagonalization breakdown (the current
        /// alpha and beta both collapse to zero in the same step -- the Golub-Kahan recurrence
        /// exhausted). On a false return x is undefined -- only read x when the call returns
        /// true.
        /// </summary>
        public static bool lsqr<TOp>(in TOp A, in floatN b, ref floatN x,
                                     ref floatN u, ref floatN v, ref floatN w,
                                     ref floatN tmpM, ref floatN tmpN,
                                     int maxIterations, float tolerance, float damp)
            where TOp : struct, IfloatLinearOperator
        {
            if (b.N != A.Rows) throw new ArgumentException("lsqr: b.N must equal A.Rows");
            if (x.N != A.Cols) throw new ArgumentException("lsqr: x.N must equal A.Cols");
            if (u.N != A.Rows) throw new ArgumentException("lsqr: u.N must equal A.Rows");
            if (tmpM.N != A.Rows) throw new ArgumentException("lsqr: tmpM.N must equal A.Rows");
            if (v.N != A.Cols) throw new ArgumentException("lsqr: v.N must equal A.Cols");
            if (w.N != A.Cols) throw new ArgumentException("lsqr: w.N must equal A.Cols");
            if (tmpN.N != A.Cols) throw new ArgumentException("lsqr: tmpN.N must equal A.Cols");

            if (maxIterations < 1)
                throw new ArgumentException("lsqr: maxIterations must be >= 1");

            unsafe
            {
                long* ptrs = stackalloc long[7];
                ptrs[0] = (long)u.Data.Ptr; ptrs[1] = (long)v.Data.Ptr; ptrs[2] = (long)w.Data.Ptr;
                ptrs[3] = (long)tmpM.Data.Ptr; ptrs[4] = (long)tmpN.Data.Ptr;
                ptrs[5] = (long)x.Data.Ptr; ptrs[6] = (long)b.Data.Ptr;
                RequireDistinctBuffers("lsqr: u/v/w/tmpM/tmpN/x/b must be distinct", ptrs, 7);
            }

            // Fixed scale reference for the relative tolerance (mirrors cgls's atbSq).
            A.ApplyT(in b, ref tmpN);
            float atbSq = Linear_OP.dot(tmpN, tmpN);

            if (atbSq == (float)0)
            {
                for (int i = 0; i < x.N; i++) x[i] = (float)0;
                return true;
            }

            float threshold = tolerance * tolerance * atbSq;

            // u = b - A x ; beta = ||u||
            A.Apply(in x, ref tmpM);
            u.Data.CopyFrom(b.Data);
            u.addScaledInpl((float)(-1), tmpM);

            float beta = math.sqrt(Linear_OP.dot(u, u));

            if (beta == (float)0)
                return true; // x already exact (r = 0)

            u.divInpl(beta);

            // v = A^T u ; alpha = ||v||
            A.ApplyT(in u, ref tmpN);
            v.Data.CopyFrom(tmpN.Data);

            float alpha = math.sqrt(Linear_OP.dot(v, v));

            if (alpha == (float)0)
                return true; // x already least-squares-stationary (A^T r = 0)

            v.divInpl(alpha);

            if ((alpha * beta) * (alpha * beta) <= threshold)
                return true; // already within tolerance before the first bidiagonalization step

            w.Data.CopyFrom(v.Data);

            float phibar = beta;
            float rhobar = alpha;

            for (int k = 0; k < maxIterations; k++)
            {
                // ---- bidiagonalization step (Golub-Kahan) ----
                A.Apply(in v, ref tmpM);
                u.scaleAddInpl(-alpha, tmpM);              // u = -alpha*u + tmpM = A v - alpha u
                beta = math.sqrt(Linear_OP.dot(u, u));
                if (beta > (float)0) u.divInpl(beta);

                A.ApplyT(in u, ref tmpN);
                v.scaleAddInpl(-beta, tmpN);                // v = -beta*v + tmpN = A^T u - beta v
                alpha = math.sqrt(Linear_OP.dot(v, v));
                if (alpha > (float)0) v.divInpl(alpha);

                // ---- fold Tikhonov damping into rhobar: rotate (rhobar, damp) -> (rhobar1, 0),
                // scaling phibar by the rotation cosine. damp==0 -> rhobar1==rhobar and phibar is
                // untouched, so the undamped path is bit-identical. ----
                float rhobar1 = rhobar;
                if (damp != (float)0)
                {
                    rhobar1 = math.sqrt(rhobar * rhobar + damp * damp);
                    phibar = (rhobar / rhobar1) * phibar;   // cs1 * phibar
                }

                // ---- Givens rotation folding (rhobar1, beta) -> (rho, 0) ----
                float rho = math.sqrt(rhobar1 * rhobar1 + beta * beta);

                if (!(rho > (float)0))
                    break; // total breakdown: rhobar1 and beta both zero

                float c = rhobar1 / rho;
                float sn = beta / rho;
                float theta = sn * alpha;
                rhobar = -c * alpha;
                float phi = c * phibar;
                phibar = sn * phibar;

                // ---- update x using the OLD w, then update w ----
                x.addScaledInpl(phi / rho, w);
                w.scaleAddInpl(-theta / rho, v);             // w = -(theta/rho)*w + v

                float arnorm = phibar * alpha * math.abs(c);

                if (arnorm * arnorm <= threshold)
                    return true;

                if (!(beta > (float)0) || !(alpha > (float)0)) // NaN-safe: both are norms, nonnegative
                    break; // bidiagonalization breakdown: Krylov space exhausted, no further progress
            }

            return false;
        }

        /// <summary>Undamped LSQR (damp = 0): plain least-squares. Forwards to the damped core.</summary>
        public static bool lsqr<TOp>(in TOp A, in floatN b, ref floatN x,
                                     ref floatN u, ref floatN v, ref floatN w,
                                     ref floatN tmpM, ref floatN tmpN,
                                     int maxIterations, float tolerance)
            where TOp : struct, IfloatLinearOperator
            => lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance, (float)0);

        /// <summary>
        /// LSQR over a dense <see cref="floatMxN"/> (possibly rectangular) -- zero-alloc
        /// primitive. Forwards into <see cref="lsqr{TOp}"/> via <see cref="floatDenseOperator"/>.
        /// </summary>
        public static bool lsqr(in floatMxN A, in floatN b, ref floatN x,
                                ref floatN u, ref floatN v, ref floatN w,
                                ref floatN tmpM, ref floatN tmpN,
                                int maxIterations, float tolerance)
        {
            return lsqr(new floatDenseOperator(in A), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>LSQR over a dense matrix -- allocates five scratch vectors from the arena.</summary>
        public static bool lsqr(in floatMxN A, in floatN b, ref floatN x, int maxIterations, float tolerance)
        {
            floatN u    = b.tempfloatVec(A.M_Rows);
            floatN v    = b.tempfloatVec(A.N_Cols);
            floatN w    = b.tempfloatVec(A.N_Cols);
            floatN tmpM = b.tempfloatVec(A.M_Rows);
            floatN tmpN = b.tempfloatVec(A.N_Cols);
            return lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// Damped (Tikhonov) LSQR over a dense matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates
        /// five scratch vectors from the arena. damp == 0 reproduces the plain least-squares solve.
        /// </summary>
        public static bool lsqr(in floatMxN A, in floatN b, ref floatN x, int maxIterations, float tolerance, float damp)
        {
            floatN u    = b.tempfloatVec(A.M_Rows);
            floatN v    = b.tempfloatVec(A.N_Cols);
            floatN w    = b.tempfloatVec(A.N_Cols);
            floatN tmpM = b.tempfloatVec(A.M_Rows);
            floatN tmpN = b.tempfloatVec(A.N_Cols);
            return lsqr(new floatDenseOperator(in A), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance, damp);
        }

        /// <summary>LSQR over a dense matrix with default maxIterations (A.N_Cols) and tolerance (Consts.floatSqrtEps).</summary>
        public static bool lsqr(in floatMxN A, in floatN b, ref floatN x)
        {
            return lsqr(in A, in b, ref x, A.N_Cols, Consts.floatSqrtEps);
        }

        /// <summary>
        /// LSQR over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc primitive.
        /// Forwards into <see cref="lsqr{TOp}"/> via <c>floatBSMOperator</c>. This is the payoff
        /// of rectangular BR x BC blocks: matrix-free least squares over a sparse Jacobian-like
        /// operator, never forming AᵀA, with better ill-conditioned behavior than <see
        /// cref="cgls{TOp}"/>.
        /// </summary>
        public static bool lsqr(in floatBSM A, in floatN b, ref floatN x,
                                ref floatN u, ref floatN v, ref floatN w,
                                ref floatN tmpM, ref floatN tmpN,
                                int maxIterations, float tolerance)
        {
            return lsqr(new floatBSMOperator(in A), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// LSQR over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc primitive
        /// variant that takes a CALLER-PROVIDED precomputed transpose AT (e.g. built once via
        /// <c>arena.floatBSMTranspose(in A)</c> outside a hot loop / before a benchmark's timed
        /// region) and routes every ApplyT call through the resulting cache-friendly forward
        /// spMV(AT, x) instead of the scatter-heavy on-the-fly spMVT(A, x) -- see
        /// <see cref="floatBSMOperator"/>'s two-arg ctor. Caller is responsible for AT actually
        /// being A's transpose; this overload does not verify it. Prefer this over the allocating
        /// <see cref="lsqr(in floatBSM, in floatN, ref floatN, int, float)"/> overload when
        /// solving repeatedly against the same A (build AT once, reuse it across many solves).
        /// </summary>
        public static bool lsqr(in floatBSM A, in floatBSM AT, in floatN b, ref floatN x,
                                ref floatN u, ref floatN v, ref floatN w,
                                ref floatN tmpM, ref floatN tmpN,
                                int maxIterations, float tolerance)
        {
            return lsqr(new floatBSMOperator(in A, in AT), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// LSQR over a BSR matrix -- allocates five scratch vectors AND materializes A^T ONCE
        /// via <c>arena.floatBSMTranspose</c> (same arena as the scratch vectors, taken from
        /// b), then drives LSQR with the two-arg <see cref="floatBSMOperator"/> so every
        /// ApplyT call routes through a cache-friendly forward spMV(A^T, x) instead of the
        /// scatter-heavy on-the-fly spMVT(A, x) every iteration -- same fix and same tradeoff as
        /// <see cref="cgls(in floatBSM, in floatN, ref floatN, int, float)"/>: for a
        /// build-free zero-alloc path, build A^T yourself once (<c>arena.floatBSMTranspose(in
        /// A)</c>) and call the zero-alloc <see cref="lsqr(in floatBSM, in floatBSM, in
        /// floatN, ref floatN, ref floatN, ref floatN, ref floatN, ref floatN, int,
        /// float)"/> overload above with your own scratch vectors, or the generic
        /// <see cref="lsqr{TOp}"/> overload directly with <c>new floatBSMOperator(in A, in
        /// AT)</c>.
        /// </summary>
        public static bool lsqr(in floatBSM A, in floatN b, ref floatN x, int maxIterations, float tolerance)
        {
            floatN u    = b.tempfloatVec(A.M_Rows);
            floatN v    = b.tempfloatVec(A.N_Cols);
            floatN w    = b.tempfloatVec(A.N_Cols);
            floatN tmpM = b.tempfloatVec(A.M_Rows);
            floatN tmpN = b.tempfloatVec(A.N_Cols);
            floatBSM AT = b.floatBSMTranspose(in A);
            return lsqr(new floatBSMOperator(in A, in AT), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// Damped (Tikhonov) LSQR over a BSR matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates five
        /// scratch vectors AND materializes A^T once (see the undamped allocating overload). damp == 0
        /// reproduces the plain least-squares solve.
        /// </summary>
        public static bool lsqr(in floatBSM A, in floatN b, ref floatN x, int maxIterations, float tolerance, float damp)
        {
            floatN u    = b.tempfloatVec(A.M_Rows);
            floatN v    = b.tempfloatVec(A.N_Cols);
            floatN w    = b.tempfloatVec(A.N_Cols);
            floatN tmpM = b.tempfloatVec(A.M_Rows);
            floatN tmpN = b.tempfloatVec(A.N_Cols);
            floatBSM AT = b.floatBSMTranspose(in A);
            return lsqr(new floatBSMOperator(in A, in AT), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance, damp);
        }

        /// <summary>LSQR over a BSR matrix with default maxIterations (A.N_Cols) and tolerance (Consts.floatSqrtEps).</summary>
        public static bool lsqr(in floatBSM A, in floatN b, ref floatN x)
        {
            return lsqr(in A, in b, ref x, A.N_Cols, Consts.floatSqrtEps);
        }

        /// <summary>
        /// Zero-alloc LSMR (Fong-Saunders 2011) solver for RECTANGULAR least-squares systems:
        /// minimizes ‖Ax-b‖₂ for possibly non-square A, generic over
        /// <see cref="IfloatLinearOperator"/>. Built on the SAME Golub-Kahan bidiagonalization as
        /// <see cref="lsqr{TOp}"/> (alternating <see cref="IfloatLinearOperator.Apply"/> /
        /// <see cref="IfloatLinearOperator.ApplyT"/>), but folds it through a rotation sequence
        /// equivalent to applying MINRES to the normal equations AᵀA x = Aᵀb -- so the
        /// normal-equation residual ‖Aᵀrₖ‖ decreases MONOTONICALLY (it equals |ζ̄ₖ₊₁|, produced
        /// for free by the recurrence). That makes LSMR's stopping test cleaner and its early
        /// termination safer than LSQR's on ill-conditioned least-squares problems, at the same
        /// O(n+m) memory and per-iteration cost (1 Apply + 1 ApplyT).
        ///
        /// Caller provides x (initial guess, length A.Cols -- overwritten with solution, WARM-
        /// STARTABLE) and six scratch vectors: u, tmpM (length A.Rows) and v, h, hbar, tmpN
        /// (length A.Cols). One more A.Cols-length vector than LSQR: LSMR carries both the
        /// Golub-Kahan search direction h and the MINRES-folded direction hbar. Same convergence
        /// contract as <see cref="cgls{TOp}"/> / <see cref="lsqr{TOp}"/>: converges when
        /// ‖Aᵀr‖ &lt;= tolerance*‖Aᵀb‖.
        ///
        /// <paramref name="damp"/> (&gt;= 0) applies Tikhonov regularization: minimizes
        /// ‖Ax-b‖² + damp²‖x‖² (equivalently solves (AᵀA + damp²I)x = Aᵀb) via one extra rotation
        /// per step that folds damp into the bidiagonal diagonal -- no augmented matrix formed.
        /// damp == 0 is the plain least-squares solve and is BIT-IDENTICAL to the undamped path.
        /// Damping regularizes rank-deficient / ill-posed / noisy systems and stabilizes the
        /// underdetermined minimum-norm solution.
        ///
        /// WARM START + DAMPING: the damp²‖x‖² penalty is measured against the COLD start (x = 0).
        /// Because lsmr bidiagonalizes the residual b - A·x₀, a NONZERO initial x₀ makes it minimize
        /// ‖Ax-b‖² + damp²‖x - x₀‖² (regularizing the CORRECTION), not ‖x‖. Start from x = 0 for the
        /// ‖x‖-regularized minimizer. (cgls regularizes ‖x‖ for any x₀; lsqr matches lsmr here.)
        ///
        /// Returns false on non-convergence or a bidiagonalization breakdown (a rotation radius
        /// collapses to zero -- the Golub-Kahan recurrence exhausted). On a false return x is
        /// undefined -- only read x when the call returns true.
        /// </summary>
        public static bool lsmr<TOp>(in TOp A, in floatN b, ref floatN x,
                                     ref floatN u, ref floatN v, ref floatN h,
                                     ref floatN hbar, ref floatN tmpM, ref floatN tmpN,
                                     int maxIterations, float tolerance, float damp)
            where TOp : struct, IfloatLinearOperator
        {
            if (b.N != A.Rows) throw new ArgumentException("lsmr: b.N must equal A.Rows");
            if (x.N != A.Cols) throw new ArgumentException("lsmr: x.N must equal A.Cols");
            if (u.N != A.Rows) throw new ArgumentException("lsmr: u.N must equal A.Rows");
            if (tmpM.N != A.Rows) throw new ArgumentException("lsmr: tmpM.N must equal A.Rows");
            if (v.N != A.Cols) throw new ArgumentException("lsmr: v.N must equal A.Cols");
            if (h.N != A.Cols) throw new ArgumentException("lsmr: h.N must equal A.Cols");
            if (hbar.N != A.Cols) throw new ArgumentException("lsmr: hbar.N must equal A.Cols");
            if (tmpN.N != A.Cols) throw new ArgumentException("lsmr: tmpN.N must equal A.Cols");

            if (maxIterations < 1)
                throw new ArgumentException("lsmr: maxIterations must be >= 1");

            unsafe
            {
                long* ptrs = stackalloc long[8];
                ptrs[0] = (long)u.Data.Ptr; ptrs[1] = (long)v.Data.Ptr; ptrs[2] = (long)h.Data.Ptr;
                ptrs[3] = (long)hbar.Data.Ptr; ptrs[4] = (long)tmpM.Data.Ptr; ptrs[5] = (long)tmpN.Data.Ptr;
                ptrs[6] = (long)x.Data.Ptr; ptrs[7] = (long)b.Data.Ptr;
                RequireDistinctBuffers("lsmr: u/v/h/hbar/tmpM/tmpN/x/b must be distinct", ptrs, 8);
            }

            // Fixed scale reference for the relative tolerance (identical contract to lsqr/cgls).
            A.ApplyT(in b, ref tmpN);
            float atbSq = Linear_OP.dot(tmpN, tmpN);

            if (atbSq == (float)0)
            {
                for (int i = 0; i < x.N; i++) x[i] = (float)0;
                return true;
            }

            float threshold = tolerance * tolerance * atbSq;

            // u = b - A x ; beta = ||u||   (warm-startable: bidiagonalize the residual)
            A.Apply(in x, ref tmpM);
            u.Data.CopyFrom(b.Data);
            u.addScaledInpl((float)(-1), tmpM);

            float beta = math.sqrt(Linear_OP.dot(u, u));

            if (beta == (float)0)
                return true; // x already exact (r = 0)

            u.divInpl(beta);

            // v = A^T u ; alpha = ||v||
            A.ApplyT(in u, ref tmpN);
            v.Data.CopyFrom(tmpN.Data);

            float alpha = math.sqrt(Linear_OP.dot(v, v));

            if (alpha == (float)0)
                return true; // x already least-squares-stationary (A^T r = 0)

            v.divInpl(alpha);

            // ||A^T r_0|| = alpha*beta = |zetabar_1|; matches lsqr's pre-loop early-out.
            if ((alpha * beta) * (alpha * beta) <= threshold)
                return true;

            // h = v ; hbar = 0
            h.Data.CopyFrom(v.Data);
            for (int i = 0; i < hbar.N; i++) hbar[i] = (float)0;

            // MINRES-on-normal-equations rotation state.
            float alphabar = alpha;
            float zetabar  = alpha * beta;
            float rho = (float)1, rhobar = (float)1, cbar = (float)1, sbar = (float)0;

            for (int k = 0; k < maxIterations; k++)
            {
                // ---- bidiagonalization step (Golub-Kahan) ----
                A.Apply(in v, ref tmpM);
                u.scaleAddInpl(-alpha, tmpM);              // u = A v - alpha u
                beta = math.sqrt(Linear_OP.dot(u, u));
                if (beta > (float)0)
                {
                    u.divInpl(beta);
                    A.ApplyT(in u, ref tmpN);
                    v.scaleAddInpl(-beta, tmpN);            // v = A^T u - beta v
                    alpha = math.sqrt(Linear_OP.dot(v, v));
                    if (alpha > (float)0) v.divInpl(alpha);
                }

                // ---- rotation P_k : (alphahat, beta) -> (rho, 0) ----
                // alphahat folds in the Tikhonov damping: alphahat = sqrt(alphabar^2 + damp^2).
                // damp==0 -> alphahat==alphabar exactly, so the undamped path is bit-identical.
                float rhoold = rho;
                float alphahat = damp != (float)0 ? math.sqrt(alphabar * alphabar + damp * damp) : alphabar;
                rho = math.sqrt(alphahat * alphahat + beta * beta);
                if (!(rho > (float)0))
                    break; // breakdown: alphahat and beta both zero
                float c = alphahat / rho;
                float s = beta / rho;
                float thetanew = s * alpha;
                alphabar = c * alpha;

                // ---- rotation Pbar_k : fold R^T into Rbar (the MINRES layer) ----
                float rhobarold = rhobar;
                float thetabar = sbar * rho;
                float cbarrho = cbar * rho;
                rhobar = math.sqrt(cbarrho * cbarrho + thetanew * thetanew);
                if (!(rhobar > (float)0))
                    break; // breakdown
                cbar = cbarrho / rhobar;
                sbar = thetanew / rhobar;
                float zeta = cbar * zetabar;
                zetabar = -sbar * zetabar;

                // ---- updates: hbar, x, h ----
                // hbar = h - (thetabar*rho / (rhoold*rhobarold)) * hbar
                float coefHbar = thetabar * rho / (rhoold * rhobarold);
                hbar.scaleAddInpl(-coefHbar, h);           // hbar = -coefHbar*hbar + h
                // x = x + (zeta / (rho*rhobar)) * hbar
                x.addScaledInpl(zeta / (rho * rhobar), hbar);
                // h = v - (thetanew/rho) * h
                h.scaleAddInpl(-thetanew / rho, v);         // h = -(thetanew/rho)*h + v

                // ‖A^T r‖ for the just-updated x = |zetabar| (falls out for free, decreases
                // monotonically). With damping this is the DAMPED normal-equation residual
                // ‖AᵀA x + damp² x − Aᵀb‖ = ‖Aᵀr − damp² x‖.
                if (zetabar * zetabar <= threshold)
                    return true;

                if (!(beta > (float)0) || !(alpha > (float)0)) // NaN-safe: both are norms, nonnegative
                    break; // bidiagonalization breakdown: Krylov space exhausted, no further progress
            }

            return false;
        }

        /// <summary>Undamped LSMR (damp = 0): plain least-squares. Forwards to the damped core.</summary>
        public static bool lsmr<TOp>(in TOp A, in floatN b, ref floatN x,
                                     ref floatN u, ref floatN v, ref floatN h,
                                     ref floatN hbar, ref floatN tmpM, ref floatN tmpN,
                                     int maxIterations, float tolerance)
            where TOp : struct, IfloatLinearOperator
            => lsmr(in A, in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance, (float)0);

        /// <summary>
        /// LSMR over a dense <see cref="floatMxN"/> (possibly rectangular) -- zero-alloc
        /// primitive. Forwards into <see cref="lsmr{TOp}"/> via <see cref="floatDenseOperator"/>.
        /// </summary>
        public static bool lsmr(in floatMxN A, in floatN b, ref floatN x,
                                ref floatN u, ref floatN v, ref floatN h,
                                ref floatN hbar, ref floatN tmpM, ref floatN tmpN,
                                int maxIterations, float tolerance)
        {
            return lsmr(new floatDenseOperator(in A), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>LSMR over a dense matrix -- allocates six scratch vectors from the arena.</summary>
        public static bool lsmr(in floatMxN A, in floatN b, ref floatN x, int maxIterations, float tolerance)
        {
            floatN u    = b.tempfloatVec(A.M_Rows);
            floatN v    = b.tempfloatVec(A.N_Cols);
            floatN h    = b.tempfloatVec(A.N_Cols);
            floatN hbar = b.tempfloatVec(A.N_Cols);
            floatN tmpM = b.tempfloatVec(A.M_Rows);
            floatN tmpN = b.tempfloatVec(A.N_Cols);
            return lsmr(in A, in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// Damped (Tikhonov) LSMR over a dense matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates
        /// six scratch vectors from the arena. damp == 0 reproduces the plain least-squares solve.
        /// </summary>
        public static bool lsmr(in floatMxN A, in floatN b, ref floatN x, int maxIterations, float tolerance, float damp)
        {
            floatN u    = b.tempfloatVec(A.M_Rows);
            floatN v    = b.tempfloatVec(A.N_Cols);
            floatN h    = b.tempfloatVec(A.N_Cols);
            floatN hbar = b.tempfloatVec(A.N_Cols);
            floatN tmpM = b.tempfloatVec(A.M_Rows);
            floatN tmpN = b.tempfloatVec(A.N_Cols);
            return lsmr(new floatDenseOperator(in A), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance, damp);
        }

        /// <summary>LSMR over a dense matrix with default maxIterations (A.N_Cols) and tolerance (Consts.floatSqrtEps).</summary>
        public static bool lsmr(in floatMxN A, in floatN b, ref floatN x)
        {
            return lsmr(in A, in b, ref x, A.N_Cols, Consts.floatSqrtEps);
        }

        /// <summary>
        /// LSMR over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc primitive.
        /// Forwards into <see cref="lsmr{TOp}"/> via <c>floatBSMOperator</c>. Matrix-free least
        /// squares over a sparse Jacobian-like operator, never forming AᵀA, with LSMR's monotone
        /// ‖Aᵀr‖ decrease (see the generic overload).
        /// </summary>
        public static bool lsmr(in floatBSM A, in floatN b, ref floatN x,
                                ref floatN u, ref floatN v, ref floatN h,
                                ref floatN hbar, ref floatN tmpM, ref floatN tmpN,
                                int maxIterations, float tolerance)
        {
            return lsmr(new floatBSMOperator(in A), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// LSMR over a (possibly rectangular) BSR matrix -- zero-alloc primitive that takes a
        /// CALLER-PROVIDED precomputed transpose AT (e.g. <c>arena.floatBSMTranspose(in A)</c>
        /// built once outside a hot loop) and routes every ApplyT through the cache-friendly
        /// forward spMV(AT, x) instead of on-the-fly spMVT(A, x) -- see
        /// <see cref="floatBSMOperator"/>'s two-arg ctor. Caller is responsible for AT being A's
        /// transpose; this overload does not verify it.
        /// </summary>
        public static bool lsmr(in floatBSM A, in floatBSM AT, in floatN b, ref floatN x,
                                ref floatN u, ref floatN v, ref floatN h,
                                ref floatN hbar, ref floatN tmpM, ref floatN tmpN,
                                int maxIterations, float tolerance)
        {
            return lsmr(new floatBSMOperator(in A, in AT), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// LSMR over a BSR matrix -- allocates six scratch vectors AND materializes A^T ONCE via
        /// <c>arena.floatBSMTranspose</c>, then drives LSMR with the two-arg
        /// <see cref="floatBSMOperator"/> so every ApplyT routes through a cache-friendly forward
        /// spMV(A^T, x). For a build-free zero-alloc path, build A^T yourself once and call the
        /// zero-alloc AT overload above with your own scratch vectors.
        /// </summary>
        public static bool lsmr(in floatBSM A, in floatN b, ref floatN x, int maxIterations, float tolerance)
        {
            floatN u    = b.tempfloatVec(A.M_Rows);
            floatN v    = b.tempfloatVec(A.N_Cols);
            floatN h    = b.tempfloatVec(A.N_Cols);
            floatN hbar = b.tempfloatVec(A.N_Cols);
            floatN tmpM = b.tempfloatVec(A.M_Rows);
            floatN tmpN = b.tempfloatVec(A.N_Cols);
            floatBSM AT = b.floatBSMTranspose(in A);
            return lsmr(new floatBSMOperator(in A, in AT), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// Damped (Tikhonov) LSMR over a BSR matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates six
        /// scratch vectors AND materializes A^T once (see the undamped allocating overload). damp == 0
        /// reproduces the plain least-squares solve.
        /// </summary>
        public static bool lsmr(in floatBSM A, in floatN b, ref floatN x, int maxIterations, float tolerance, float damp)
        {
            floatN u    = b.tempfloatVec(A.M_Rows);
            floatN v    = b.tempfloatVec(A.N_Cols);
            floatN h    = b.tempfloatVec(A.N_Cols);
            floatN hbar = b.tempfloatVec(A.N_Cols);
            floatN tmpM = b.tempfloatVec(A.M_Rows);
            floatN tmpN = b.tempfloatVec(A.N_Cols);
            floatBSM AT = b.floatBSMTranspose(in A);
            return lsmr(new floatBSMOperator(in A, in AT), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance, damp);
        }

        /// <summary>LSMR over a BSR matrix with default maxIterations (A.N_Cols) and tolerance (Consts.floatSqrtEps).</summary>
        public static bool lsmr(in floatBSM A, in floatN b, ref floatN x)
        {
            return lsmr(in A, in b, ref x, A.N_Cols, Consts.floatSqrtEps);
        }

        /// <summary>
        /// Zero-alloc CGNE / Craig's method (Saad Alg. 8.5) for CONSISTENT systems: finds the
        /// MINIMUM-NORM solution of A x = b (requires b in range(A)) for possibly rectangular
        /// (typically UNDER-determined, m &lt; n) A, generic over <see cref="IfloatLinearOperator"/>.
        /// Mathematically CG applied to A Aᵀ y = b with x = Aᵀ y, but recurred directly on x (no y
        /// stored): r = b - A x (length A.Rows) is the residual, p (length A.Cols) the search
        /// direction. Complementary to <see cref="cgls{TOp}"/>: CGLS minimizes ‖Ax-b‖ (OVER-
        /// determined / inconsistent), CGNE minimizes ‖x‖ subject to A x = b (consistent UNDER-
        /// determined). One Apply + one ApplyT per iteration, O(n+m) memory.
        ///
        /// Caller provides x (initial guess, length A.Cols -- overwritten, warm-startable) and four
        /// scratch vectors: r, q (length A.Rows) and p, tmpN (length A.Cols). Converges when
        /// ‖b - A x‖ &lt;= tolerance*‖b‖ (a fixed scale, mirroring cg's ‖b‖ reference). For an
        /// INCONSISTENT system (b not in range(A)) the residual cannot reach zero -- CGNE then runs
        /// to maxIterations and returns false; use cgls/lsqr/lsmr for least-squares instead.
        ///
        /// Returns false on non-convergence or breakdown (‖p‖² &lt;= 0, i.e. Aᵀr = 0 while r != 0 --
        /// r is orthogonal to range(A), which for a consistent system means r is already 0). On a
        /// false return x is undefined -- only read x when the call returns true.
        /// </summary>
        public static bool cgne<TOp>(in TOp A, in floatN b, ref floatN x,
                                     ref floatN r, ref floatN p, ref floatN q, ref floatN tmpN,
                                     int maxIterations, float tolerance)
            where TOp : struct, IfloatLinearOperator
        {
            if (b.N != A.Rows) throw new ArgumentException("cgne: b.N must equal A.Rows");
            if (x.N != A.Cols) throw new ArgumentException("cgne: x.N must equal A.Cols");
            if (r.N != A.Rows) throw new ArgumentException("cgne: r.N must equal A.Rows");
            if (q.N != A.Rows) throw new ArgumentException("cgne: q.N must equal A.Rows");
            if (p.N != A.Cols) throw new ArgumentException("cgne: p.N must equal A.Cols");
            if (tmpN.N != A.Cols) throw new ArgumentException("cgne: tmpN.N must equal A.Cols");

            if (maxIterations < 1)
                throw new ArgumentException("cgne: maxIterations must be >= 1");

            unsafe
            {
                long* ptrs = stackalloc long[6];
                ptrs[0] = (long)r.Data.Ptr; ptrs[1] = (long)p.Data.Ptr; ptrs[2] = (long)q.Data.Ptr;
                ptrs[3] = (long)tmpN.Data.Ptr; ptrs[4] = (long)x.Data.Ptr; ptrs[5] = (long)b.Data.Ptr;
                RequireDistinctBuffers("cgne: r/p/q/tmpN/x/b must be distinct", ptrs, 6);
            }

            // Fixed scale reference for the relative tolerance, independent of x0 (mirrors cg's bb).
            float bb = Linear_OP.dot(b, b);

            if (bb == (float)0)
            {
                // b == 0 -> the minimum-norm solution of A x = 0 is x = 0.
                for (int i = 0; i < x.N; i++) x[i] = (float)0;
                return true;
            }

            float threshold = tolerance * tolerance * bb;

            // r = b - A x
            A.Apply(in x, ref q);                          // q = A x (temp use of q)
            r.Data.CopyFrom(b.Data);
            r.addScaledInpl((float)(-1), q);

            float rr = Linear_OP.dot(r, r);

            if (rr <= threshold)
                return true;

            // p = A^T r
            A.ApplyT(in r, ref p);

            for (int k = 0; k < maxIterations; k++)
            {
                float pp = Linear_OP.dot(p, p);

                if (!(pp > (float)0))                      // NaN-safe: A^T r == 0 (r ⟂ range(A)) or p == 0
                    return false;

                float alpha = rr / pp;

                x.addScaledInpl(alpha, p);                  // x += alpha p
                A.Apply(in p, ref q);                       // q = A p
                r.addScaledInpl(-alpha, q);                 // r -= alpha A p

                float rrNew = Linear_OP.dot(r, r);

                if (rrNew <= threshold)
                    return true;

                float beta = rrNew / rr;

                A.ApplyT(in r, ref tmpN);                   // tmpN = A^T r
                p.scaleAddInpl(beta, tmpN);                 // p = beta p + A^T r

                rr = rrNew;
            }

            return false;
        }

        /// <summary>
        /// CGNE / Craig over a dense <see cref="floatMxN"/> (possibly rectangular) -- zero-alloc
        /// primitive. Forwards into <see cref="cgne{TOp}"/> via <see cref="floatDenseOperator"/>.
        /// </summary>
        public static bool cgne(in floatMxN A, in floatN b, ref floatN x,
                                ref floatN r, ref floatN p, ref floatN q, ref floatN tmpN,
                                int maxIterations, float tolerance)
        {
            return cgne(new floatDenseOperator(in A), in b, ref x, ref r, ref p, ref q, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>CGNE over a dense matrix -- allocates four scratch vectors from the arena.</summary>
        public static bool cgne(in floatMxN A, in floatN b, ref floatN x, int maxIterations, float tolerance)
        {
            floatN r    = b.tempfloatVec(A.M_Rows);
            floatN p    = b.tempfloatVec(A.N_Cols);
            floatN q    = b.tempfloatVec(A.M_Rows);
            floatN tmpN = b.tempfloatVec(A.N_Cols);
            return cgne(in A, in b, ref x, ref r, ref p, ref q, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>CGNE over a dense matrix with default maxIterations (A.N_Cols) and tolerance (Consts.floatSqrtEps).</summary>
        public static bool cgne(in floatMxN A, in floatN b, ref floatN x)
        {
            return cgne(in A, in b, ref x, A.N_Cols, Consts.floatSqrtEps);
        }

        /// <summary>
        /// CGNE / Craig over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc
        /// primitive. Forwards into <see cref="cgne{TOp}"/> via <c>floatBSMOperator</c>. Matrix-
        /// free minimum-norm solve over a sparse Jacobian-like operator, never forming A Aᵀ.
        /// </summary>
        public static bool cgne(in floatBSM A, in floatN b, ref floatN x,
                                ref floatN r, ref floatN p, ref floatN q, ref floatN tmpN,
                                int maxIterations, float tolerance)
        {
            return cgne(new floatBSMOperator(in A), in b, ref x, ref r, ref p, ref q, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// CGNE / Craig over a BSR matrix with a CALLER-PROVIDED precomputed transpose AT (built
        /// once via <c>arena.floatBSMTranspose(in A)</c>), routing every ApplyT through the
        /// cache-friendly forward spMV(AT, x) instead of on-the-fly spMVT(A, x) -- see
        /// <see cref="floatBSMOperator"/>'s two-arg ctor. Zero-alloc; caller owns AT.
        /// </summary>
        public static bool cgne(in floatBSM A, in floatBSM AT, in floatN b, ref floatN x,
                                ref floatN r, ref floatN p, ref floatN q, ref floatN tmpN,
                                int maxIterations, float tolerance)
        {
            return cgne(new floatBSMOperator(in A, in AT), in b, ref x, ref r, ref p, ref q, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// CGNE over a BSR matrix -- allocates four scratch vectors AND materializes A^T once via
        /// <c>arena.floatBSMTranspose</c>, driving CGNE with the two-arg
        /// <see cref="floatBSMOperator"/> so every ApplyT routes through a cache-friendly forward
        /// spMV(A^T, x). For a build-free zero-alloc path, build A^T yourself once and call the
        /// caller-AT overload above.
        /// </summary>
        public static bool cgne(in floatBSM A, in floatN b, ref floatN x, int maxIterations, float tolerance)
        {
            floatN r    = b.tempfloatVec(A.M_Rows);
            floatN p    = b.tempfloatVec(A.N_Cols);
            floatN q    = b.tempfloatVec(A.M_Rows);
            floatN tmpN = b.tempfloatVec(A.N_Cols);
            floatBSM AT = b.floatBSMTranspose(in A);
            return cgne(new floatBSMOperator(in A, in AT), in b, ref x, ref r, ref p, ref q, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>CGNE over a BSR matrix with default maxIterations (A.N_Cols) and tolerance (Consts.floatSqrtEps).</summary>
        public static bool cgne(in floatBSM A, in floatN b, ref floatN x)
        {
            return cgne(in A, in b, ref x, A.N_Cols, Consts.floatSqrtEps);
        }
    }

}
