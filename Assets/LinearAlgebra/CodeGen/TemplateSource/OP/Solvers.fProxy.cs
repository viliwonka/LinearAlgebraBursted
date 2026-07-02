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
        public static void solveUpperTriangular(ref fProxyMxN U, ref fProxyN x)
        {
            if(U.M_Rows < U.N_Cols)
                throw new ArgumentException("Solvers.solveUpperTriangular: Matrix must be square or tall (M_Rows >= N_Cols)");

            if(U.N_Cols != x.N)
                throw new ArgumentException("Solvers.solveUpperTriangular: Matrix and vector must have same number of columns");

            for (int r = U.N_Cols - 1; r >= 0; r--)
            {
                fProxy sum = 0;

                for (int c = r + 1; c < U.N_Cols; c++)
                    sum += U[r, c] * x[c];

                x[r] = (x[r] - sum) / U[r, r];
            }
        }

        // Solve Lx = b for x
        // PRECONDITION: L is non-singular — every diagonal L[r,r] must be nonzero (see
        // solveUpperTriangular; a zero diagonal divides by zero -> Inf/NaN, unguarded).
        public static void solveLowerTriangular(ref fProxyMxN L, ref fProxyN x)
        {
            if (L.IsSquare == false)
                throw new ArgumentException("Solvers.solveLowerTriangular: Matrix must be square");

            if (L.M_Rows != x.N)
                throw new ArgumentException("Solvers.solveLowerTriangular: Matrix and vector must have same number of rows");

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
        public static void solveLowerTriangularLU(ref fProxyMxN L, in Pivot RP, ref fProxyN x) {
            if (L.IsSquare == false)
                throw new ArgumentException("Solvers.solveLowerTriangularLU: Matrix must be square");

            if (L.M_Rows != x.N)
                throw new ArgumentException("Solvers.solveLowerTriangularLU: Matrix and vector must have same number of rows");

            for (int r = 0; r < L.M_Rows; r++) {
                fProxy sum = 0;

                for (int c = 0; c < r; c++)
                    sum += L[RP[r], c] * x[c];

                x[r] = (x[r] - sum);
            }
        }

        public static void solveUpperTriangularLU(ref fProxyMxN U, in Pivot RP, ref fProxyN x) {
            if(U.IsSquare == false)
                throw new ArgumentException("Solvers.solveUpperTriangularLU: Matrix must be square");

            if (U.N_Cols != x.N)
                throw new ArgumentException("Solvers.solveUpperTriangularLU: Matrix and vector must have same number of columns");

            for (int r = U.N_Cols - 1; r >= 0; r--) {
                fProxy sum = 0;

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
        public static void solveQR(ref fProxyMxN Q, ref fProxyMxN R, ref fProxyN b, ref fProxyN x) {
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
        public static fProxyN solveQR(ref fProxyMxN Q, ref fProxyMxN R, ref fProxyN b) {
            fProxyN x = b.tempfProxyVec(Q.N_Cols);
            solveQR(ref Q, ref R, ref b, ref x);
            return x;
        }

        // Solve Ax = b for x
        public static void solveQR(ref fProxyMxN A, ref fProxyN b, ref fProxyN x)
        {
            QR.qrDirectSolve(ref A, ref b, ref x);

        }

        // Shared factory for the square-solver diagnostics struct (cg/pcg/minres/biCGStab/cgne).
        // rnorm is ALWAYS a value the solver already holds -- a tracked residual norm, or a single
        // dot on its live residual r -- never a fresh A*x, honoring the "free diagnostics" contract.
        static SolveInfo MakeSolveInfo(IterativeSolveStatus status, int iterations, fProxy rnorm)
            => new SolveInfo { rnorm = rnorm, iterations = iterations, status = status };

        /// <summary>
        /// Zero-alloc Conjugate Gradient solver for symmetric positive-definite (SPD) systems A x = b,
        /// generic over any <see cref="IfProxyLinearOperator"/> (Burst-monomorphized static
        /// dispatch, no vtable/managed delegate). This is the SINGLE SOURCE OF TRUTH for the CG
        /// loop — the concrete dense (<c>conjugateGradient(in fProxyMxN, ...)</c>) and BSM
        /// (<c>conjugateGradient(in fProxyBSM, ...)</c>) overloads below are thin forwarders that
        /// wrap their matrix in <see cref="fProxyDenseOperator"/> / <c>fProxyBSMOperator</c> and
        /// call this method.
        ///
        /// Caller provides x (initial guess, overwritten with solution — WARM-STARTABLE: seed x
        /// with a previous solution to resume/refine) and three scratch vectors r, p, Ap (all
        /// length A.Rows). Returns an <see cref="SolveInfo"/> (rnorm = ‖b-Ax‖, iterations,
        /// status) that converts implicitly to bool (== Converged), so <c>if (cg(...))</c> keeps
        /// working. Status is Converged iff it reached the relative residual tolerance within
        /// maxIterations; Breakdown on non-positive curvature p·Ap &lt;= 0 (A not SPD or numerical
        /// breakdown); MaxIterations otherwise. On a non-Converged return x is undefined (it may
        /// have been partially updated) — only read x when Solved.
        /// </summary>
        public static SolveInfo cg<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                   ref fProxyN r, ref fProxyN p, ref fProxyN Ap,
                                   int maxIterations, fProxy tolerance)
            where TOp : struct, IfProxyLinearOperator
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
                fProxy* rPtr = r.Data.Ptr, pPtr = p.Data.Ptr, ApPtr = Ap.Data.Ptr, xPtr = x.Data.Ptr, bPtr = b.Data.Ptr;

                if (rPtr == pPtr || rPtr == ApPtr || rPtr == xPtr || rPtr == bPtr ||
                    pPtr == ApPtr || pPtr == xPtr || pPtr == bPtr ||
                    ApPtr == xPtr || ApPtr == bPtr ||
                    xPtr == bPtr)
                    throw new ArgumentException("cg: r/p/Ap/x/b must be distinct");
            }

            fProxy bb = Linear_OP.dot(b, b);

            // b is the zero vector — x = 0 is the exact solution. Copy b (all zeros)
            // rather than multiplying by 0, so a NaN/Inf initial guess is sanitized
            // (NaN * 0 = NaN would otherwise leak through).
            if (bb == (fProxy)0)
            {
                x.Data.CopyFrom(b.Data);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }

            // r = b - A x
            A.Apply(in x, ref Ap);                       // Ap = A x (temp use of Ap)
            r.Data.CopyFrom(b.Data);                     // r  = b
            r.addScaledInpl((fProxy)(-1), Ap);           // r -= Ap  =>  r = b - A x

            // p = r
            p.Data.CopyFrom(r.Data);

            fProxy rsold = Linear_OP.dot(r, r);
            fProxy threshold = tolerance * tolerance * bb;

            if (rsold <= threshold)
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, math.sqrt(rsold));

            for (int k = 0; k < maxIterations; k++)
            {
                A.Apply(in p, ref Ap);                    // Ap = A p

                fProxy pAp = Linear_OP.dot(p, Ap);

                if (!(pAp > (fProxy)0))                  // NaN-safe: also catches breakdown
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rsold));

                fProxy alpha = rsold / pAp;

                x.addScaledInpl(alpha, p);               // x += alpha p
                r.addScaledInpl(-alpha, Ap);             // r -= alpha Ap

                fProxy rsnew = Linear_OP.dot(r, r);

                if (rsnew <= threshold)
                    return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(rsnew));

                fProxy beta = rsnew / rsold;

                p.scaleAddInpl(beta, r);                 // p = beta p + r

                rsold = rsnew;
            }

            return MakeSolveInfo(IterativeSolveStatus.MaxIterations, maxIterations, math.sqrt(rsold));
        }

        /// <summary>
        /// Zero-alloc Conjugate Gradient solver for symmetric positive-definite (SPD) systems A x = b.
        /// Caller provides x (initial guess, overwritten with solution) and three scratch vectors
        /// r, p, Ap (all length A.M_Rows). Returns an <see cref="SolveInfo"/> (implicit-bool
        /// == Converged); Converged within maxIterations to the relative residual tolerance,
        /// Breakdown on non-positive curvature p·Ap &lt;= 0 (A not SPD or numerical breakdown),
        /// MaxIterations otherwise. On a non-Converged return x is undefined — only read x when Solved.
        /// Forwards into <see cref="cg{TOp}"/> via <see cref="fProxyDenseOperator"/> — see that
        /// method for the actual loop.
        /// </summary>
        public static SolveInfo conjugateGradient(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                             ref fProxyN r, ref fProxyN p, ref fProxyN Ap,
                                             int maxIterations, fProxy tolerance)
        {
            return cg(new fProxyDenseOperator(in A), in b, ref x, ref r, ref p, ref Ap, maxIterations, tolerance);
        }

        /// <summary>
        /// Conjugate Gradient solver — allocates three scratch vectors from the arena and calls
        /// the zero-alloc primitive. x is overwritten with the solution on convergence.
        /// </summary>
        public static SolveInfo conjugateGradient(in fProxyMxN A, in fProxyN b, ref fProxyN x,
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
        public static SolveInfo conjugateGradient(in fProxyMxN A, in fProxyN b, ref fProxyN x)
        {
            return conjugateGradient(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Conjugate Gradient solver over a block-sparse (BSR) SPD matrix. Same semantics as
        /// the dense overload — see <see cref="conjugateGradient(in fProxyMxN, in fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, int, fProxy)"/>.
        /// Forwards into <see cref="cg{TOp}"/> via <c>fProxyBSMOperator</c>.
        /// </summary>
        public static SolveInfo conjugateGradient(in fProxyBSM A, in fProxyN b, ref fProxyN x,
                                             ref fProxyN r, ref fProxyN p, ref fProxyN Ap,
                                             int maxIterations, fProxy tolerance)
        {
            return cg(new fProxyBSMOperator(in A), in b, ref x, ref r, ref p, ref Ap, maxIterations, tolerance);
        }

        /// <summary>
        /// Conjugate Gradient solver over a block-sparse (BSR) SPD matrix — allocates three
        /// scratch vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo conjugateGradient(in fProxyBSM A, in fProxyN b, ref fProxyN x,
                                             int maxIterations, fProxy tolerance)
        {
            fProxyN r  = b.tempfProxyVec(A.M_Rows);
            fProxyN p  = b.tempfProxyVec(A.M_Rows);
            fProxyN Ap = b.tempfProxyVec(A.M_Rows);
            return conjugateGradient(in A, in b, ref x, ref r, ref p, ref Ap, maxIterations, tolerance);
        }

        /// <summary>
        /// Conjugate Gradient solver over a block-sparse (BSR) SPD matrix, with default
        /// maxIterations (A.M_Rows) and tolerance (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo conjugateGradient(in fProxyBSM A, in fProxyN b, ref fProxyN x)
        {
            return conjugateGradient(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Zero-alloc Preconditioned Conjugate Gradient solver for SPD systems A x = b, generic
        /// over both the operator (<see cref="IfProxyLinearOperator"/>) and the preconditioner
        /// (<see cref="IfProxyPreconditioner"/>) — same Burst static-dispatch shape as
        /// <see cref="cg{TOp}"/>. Standard PCG: p is combined with z = M⁻¹r (not r), and β uses
        /// ⟨r,z⟩ instead of ⟨r,r⟩.
        ///
        /// Caller provides x (initial guess, overwritten with solution — warm-startable) and four
        /// scratch vectors r, p, Ap, z (all length A.Rows). The convergence test compares the
        /// TRUE (unpreconditioned) residual ||r||² against tolerance²·||b||² — the same criterion
        /// as <see cref="cg{TOp}"/> — so iteration counts between cg and pcg on the same system are
        /// directly comparable. Returns an <see cref="SolveInfo"/> (implicit-bool ==
        /// Converged): Converged within maxIterations, Breakdown on non-positive curvature
        /// p·Ap &lt;= 0 (or a non-SPD preconditioner's non-positive ⟨r,z⟩), MaxIterations
        /// otherwise. On a non-Converged return x is undefined — only read x when Solved.
        /// </summary>
        public static SolveInfo pcg<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x,
                                          ref fProxyN r, ref fProxyN p, ref fProxyN Ap, ref fProxyN z,
                                          int maxIterations, fProxy tolerance)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
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
                fProxy* rPtr = r.Data.Ptr, pPtr = p.Data.Ptr, ApPtr = Ap.Data.Ptr, zPtr = z.Data.Ptr, xPtr = x.Data.Ptr, bPtr = b.Data.Ptr;

                if (rPtr == pPtr || rPtr == ApPtr || rPtr == zPtr || rPtr == xPtr || rPtr == bPtr ||
                    pPtr == ApPtr || pPtr == zPtr || pPtr == xPtr || pPtr == bPtr ||
                    ApPtr == zPtr || ApPtr == xPtr || ApPtr == bPtr ||
                    zPtr == xPtr || zPtr == bPtr ||
                    xPtr == bPtr)
                    throw new ArgumentException("pcg: r/p/Ap/z/x/b must be distinct");
            }

            fProxy bb = Linear_OP.dot(b, b);

            if (bb == (fProxy)0)
            {
                x.Data.CopyFrom(b.Data);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }

            // r = b - A x
            A.Apply(in x, ref Ap);
            r.Data.CopyFrom(b.Data);
            r.addScaledInpl((fProxy)(-1), Ap);

            fProxy threshold = tolerance * tolerance * bb;

            // rr tracks ‖r‖² of the CURRENT residual across the whole solve -- it is exactly the
            // quantity the convergence test already needs, so reporting rnorm = √rr is free.
            fProxy rr = Linear_OP.dot(r, r);
            if (rr <= threshold)
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, math.sqrt(rr));

            // z = M^-1 r ; p = z
            M.Apply(in r, ref z);
            p.Data.CopyFrom(z.Data);

            fProxy rzold = Linear_OP.dot(r, z);

            // Block-Jacobi is SPD so this never trips on the shipped path, but a user-supplied
            // preconditioner is not guaranteed SPD; a non-positive <r,z> yields a wrong-signed
            // alpha/beta and silent divergence instead of a clean bailout. Mirrors cg's
            // NaN-safe !(pAp > 0) breakdown guard.
            if (!(rzold > (fProxy)0))
                return MakeSolveInfo(IterativeSolveStatus.Breakdown, 0, math.sqrt(rr));

            for (int k = 0; k < maxIterations; k++)
            {
                A.Apply(in p, ref Ap);                    // Ap = A p

                fProxy pAp = Linear_OP.dot(p, Ap);

                if (!(pAp > (fProxy)0))                  // NaN-safe: also catches breakdown
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr));

                fProxy alpha = rzold / pAp;

                x.addScaledInpl(alpha, p);               // x += alpha p
                r.addScaledInpl(-alpha, Ap);             // r -= alpha Ap

                rr = Linear_OP.dot(r, r);
                if (rr <= threshold)
                    return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(rr));

                M.Apply(in r, ref z);                     // z = M^-1 r

                fProxy rznew = Linear_OP.dot(r, z);

                if (!(rznew > (fProxy)0))                 // NaN-safe: same breakdown guard, fresh <r,z>
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k + 1, math.sqrt(rr));

                fProxy beta = rznew / rzold;

                p.scaleAddInpl(beta, z);                 // p = beta p + z

                rzold = rznew;
            }

            return MakeSolveInfo(IterativeSolveStatus.MaxIterations, maxIterations, math.sqrt(rr));
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient solver — allocates four scratch vectors from the
        /// arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo pcg<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x,
                                          int maxIterations, fProxy tolerance)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            fProxyN r  = b.tempfProxyVec(A.Rows);
            fProxyN p  = b.tempfProxyVec(A.Rows);
            fProxyN Ap = b.tempfProxyVec(A.Rows);
            fProxyN z  = b.tempfProxyVec(A.Rows);
            return pcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIterations, tolerance);
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient solver with default maxIterations (A.Rows) and
        /// tolerance (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo pcg<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            return pcg(in A, in M, in b, ref x, A.Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient over a block-sparse (BSR) SPD matrix with its
        /// matching block-Jacobi preconditioner. Forwards into <see cref="pcg{TOp,TPre}"/> via
        /// <c>fProxyBSMOperator</c>.
        /// </summary>
        public static SolveInfo pcg(in fProxyBSM A, in fProxyBlockJacobi M, in fProxyN b, ref fProxyN x,
                               ref fProxyN r, ref fProxyN p, ref fProxyN Ap, ref fProxyN z,
                               int maxIterations, fProxy tolerance)
        {
            return pcg(new fProxyBSMOperator(in A), in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIterations, tolerance);
        }

        /// <summary>
        /// Block-Jacobi Preconditioned Conjugate Gradient over a BSR SPD matrix — allocates four
        /// scratch vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo pcg(in fProxyBSM A, in fProxyBlockJacobi M, in fProxyN b, ref fProxyN x,
                               int maxIterations, fProxy tolerance)
        {
            fProxyN r  = b.tempfProxyVec(A.M_Rows);
            fProxyN p  = b.tempfProxyVec(A.M_Rows);
            fProxyN Ap = b.tempfProxyVec(A.M_Rows);
            fProxyN z  = b.tempfProxyVec(A.M_Rows);
            return pcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIterations, tolerance);
        }

        /// <summary>
        /// Block-Jacobi Preconditioned Conjugate Gradient over a BSR SPD matrix, with default
        /// maxIterations (A.M_Rows) and tolerance (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo pcg(in fProxyBSM A, in fProxyBlockJacobi M, in fProxyN b, ref fProxyN x)
        {
            return pcg(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
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
        /// <see cref="IfProxyLinearOperator"/> (Burst-monomorphized static dispatch, no vtable).
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
        /// Returns an <see cref="SolveInfo"/> (rnorm = phibar = ‖b-Ax‖, iterations, status;
        /// implicit-bool == Converged): Converged when the residual falls within the relative
        /// tolerance (‖r‖ &lt;= tolerance*‖b‖) inside maxIterations; Breakdown if the Lanczos
        /// recurrence exactly exhausts the Krylov subspace short of tolerance (beta==0, an
        /// exact-arithmetic invariant-subspace breakdown); MaxIterations otherwise. On a
        /// non-Converged return x is undefined — only read x when Solved.
        /// </summary>
        public static SolveInfo minres<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                       ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                                       ref fProxyN w, ref fProxyN w1, ref fProxyN w2,
                                       int maxIterations, fProxy tolerance)
            where TOp : struct, IfProxyLinearOperator
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

            fProxy bb = Linear_OP.dot(b, b);

            if (bb == (fProxy)0)
            {
                x.Data.CopyFrom(b.Data);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }

            // r1 = b - A x
            A.Apply(in x, ref y);                       // y = A x (temp use of y)
            r1.Data.CopyFrom(b.Data);
            r1.addScaledInpl((fProxy)(-1), y);           // r1 = b - A x

            fProxy beta1 = math.sqrt(Linear_OP.dot(r1, r1));
            fProxy threshold = tolerance * tolerance * bb;

            if (beta1 * beta1 <= threshold)
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, beta1);

            r2.Data.CopyFrom(r1.Data);

            // Zero the 3-term search-direction history (w/w1/w2 start at 0 in exact MINRES).
            for (int i = 0; i < A.Rows; i++) { w[i] = (fProxy)0; w1[i] = (fProxy)0; w2[i] = (fProxy)0; }

            fProxy oldb = (fProxy)0;
            fProxy beta = beta1;
            fProxy dbar = (fProxy)0;
            fProxy epsln = (fProxy)0;
            fProxy phibar = beta1;
            fProxy cs = (fProxy)(-1);
            fProxy sn = (fProxy)0;
            fProxy gammaFloor = Consts.fProxyEpsilon;

            for (int k = 0; k < maxIterations; k++)
            {
                // ---- Lanczos step: extend the tridiagonalization by one vector ----
                v.Data.CopyFrom(r2.Data);
                v.divInpl(beta);                          // v = r2 / beta

                A.Apply(in v, ref y);                      // y = A v

                if (k >= 1)
                    y.addScaledInpl(-(beta / oldb), r1);   // y -= (beta/oldb) r1

                fProxy alfa = Linear_OP.dot(v, y);
                y.addScaledInpl(-(alfa / beta), r2);       // y -= (alfa/beta) r2

                r1.Data.CopyFrom(r2.Data);
                r2.Data.CopyFrom(y.Data);

                oldb = beta;
                beta = math.sqrt(Linear_OP.dot(r2, r2));

                // ---- apply the PREVIOUS Givens rotation (cs,sn) to the new tridiagonal column ----
                fProxy oldeps = epsln;
                fProxy delta = cs * dbar + sn * alfa;
                fProxy gbar = sn * dbar - cs * alfa;
                epsln = sn * beta;
                dbar = -cs * beta;

                // ---- compute the NEW Givens rotation that zeros the subdiagonal entry ----
                fProxy gamma = math.sqrt(gbar * gbar + beta * beta);
                gamma = math.max(gamma, gammaFloor);
                cs = gbar / gamma;
                sn = beta / gamma;
                fProxy phi = cs * phibar;
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
                // no extra dot product needed, so rnorm = phibar is free.
                if (phibar * phibar <= threshold)
                    return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, phibar);

                if (!(beta > (fProxy)0))
                    // Lanczos breakdown: invariant subspace exhausted, no further progress possible.
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k + 1, phibar);
            }

            return MakeSolveInfo(IterativeSolveStatus.MaxIterations, maxIterations, phibar);
        }

        /// <summary>
        /// MINRES over a dense <see cref="fProxyMxN"/> -- zero-alloc primitive. Forwards into
        /// <see cref="minres{TOp}"/> via <see cref="fProxyDenseOperator"/>. See that method for
        /// the actual loop and buffer semantics.
        /// </summary>
        public static SolveInfo minres(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                  ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                                  ref fProxyN w, ref fProxyN w1, ref fProxyN w2,
                                  int maxIterations, fProxy tolerance)
        {
            return minres(new fProxyDenseOperator(in A), in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, maxIterations, tolerance);
        }

        /// <summary>MINRES over a dense matrix -- allocates seven scratch vectors from the arena.</summary>
        public static SolveInfo minres(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance)
        {
            fProxyN y  = b.tempfProxyVec(A.M_Rows);
            fProxyN r1 = b.tempfProxyVec(A.M_Rows);
            fProxyN r2 = b.tempfProxyVec(A.M_Rows);
            fProxyN v  = b.tempfProxyVec(A.M_Rows);
            fProxyN w  = b.tempfProxyVec(A.M_Rows);
            fProxyN w1 = b.tempfProxyVec(A.M_Rows);
            fProxyN w2 = b.tempfProxyVec(A.M_Rows);
            return minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, maxIterations, tolerance);
        }

        /// <summary>MINRES over a dense matrix with default maxIterations (A.M_Rows) and tolerance (Consts.fProxySqrtEps).</summary>
        public static SolveInfo minres(in fProxyMxN A, in fProxyN b, ref fProxyN x)
        {
            return minres(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// MINRES over a symmetric block-sparse (BSR) matrix -- zero-alloc primitive. Forwards
        /// into <see cref="minres{TOp}"/> via <c>fProxyBSMOperator</c>.
        /// </summary>
        public static SolveInfo minres(in fProxyBSM A, in fProxyN b, ref fProxyN x,
                                  ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                                  ref fProxyN w, ref fProxyN w1, ref fProxyN w2,
                                  int maxIterations, fProxy tolerance)
        {
            return minres(new fProxyBSMOperator(in A), in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, maxIterations, tolerance);
        }

        /// <summary>MINRES over a BSR matrix -- allocates seven scratch vectors from the arena.</summary>
        public static SolveInfo minres(in fProxyBSM A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance)
        {
            fProxyN y  = b.tempfProxyVec(A.M_Rows);
            fProxyN r1 = b.tempfProxyVec(A.M_Rows);
            fProxyN r2 = b.tempfProxyVec(A.M_Rows);
            fProxyN v  = b.tempfProxyVec(A.M_Rows);
            fProxyN w  = b.tempfProxyVec(A.M_Rows);
            fProxyN w1 = b.tempfProxyVec(A.M_Rows);
            fProxyN w2 = b.tempfProxyVec(A.M_Rows);
            return minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, maxIterations, tolerance);
        }

        /// <summary>MINRES over a BSR matrix with default maxIterations (A.M_Rows) and tolerance (Consts.fProxySqrtEps).</summary>
        public static SolveInfo minres(in fProxyBSM A, in fProxyN b, ref fProxyN x)
        {
            return minres(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Zero-alloc BiCGSTAB (van der Vorst 1992, stabilized Bi-Conjugate Gradient) solver for
        /// NON-symmetric (general) square systems A x = b, generic over
        /// <see cref="IfProxyLinearOperator"/>. Short two-sided recurrence, flat O(n) memory (no
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
        /// Returns an <see cref="SolveInfo"/> (rnorm = ‖b-Ax‖, iterations, status;
        /// implicit-bool == Converged): Converged when the residual falls within the relative
        /// tolerance (‖r‖ &lt;= tolerance*‖b‖) inside maxIterations; Breakdown on one of the
        /// standard BiCGSTAB breakdowns (rho == 0, rHat0·v == 0, or omega == 0 -- A not amenable
        /// to BiCGSTAB from this shadow residual, or numerical breakdown); MaxIterations otherwise.
        /// On a non-Converged return x is undefined — only read x when Solved.
        /// </summary>
        public static SolveInfo biCGStab<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                         ref fProxyN r, ref fProxyN rHat0, ref fProxyN p, ref fProxyN v, ref fProxyN t,
                                         int maxIterations, fProxy tolerance)
            where TOp : struct, IfProxyLinearOperator
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

            fProxy bb = Linear_OP.dot(b, b);

            if (bb == (fProxy)0)
            {
                x.Data.CopyFrom(b.Data);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }

            // r = b - A x
            A.Apply(in x, ref v);                          // v = A x (temp use, overwritten below)
            r.Data.CopyFrom(b.Data);
            r.addScaledInpl((fProxy)(-1), v);

            fProxy threshold = tolerance * tolerance * bb;

            // rr tracks ‖current residual‖²; ss the ‖half-step residual s‖². Both are already
            // computed for the convergence tests, so every exit reports rnorm from a held value.
            fProxy rr = Linear_OP.dot(r, r);
            if (rr <= threshold)
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, math.sqrt(rr));

            rHat0.Data.CopyFrom(r.Data);

            // p_0 = v_0 = 0 (standard BiCGSTAB init).
            for (int i = 0; i < A.Rows; i++) { p[i] = (fProxy)0; v[i] = (fProxy)0; }

            fProxy rho = (fProxy)1, alpha = (fProxy)1, omega = (fProxy)1;

            for (int k = 0; k < maxIterations; k++)
            {
                fProxy rhoNew = Linear_OP.dot(rHat0, r);

                if (rhoNew == (fProxy)0 || math.isnan(rhoNew))
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr)); // r orthogonal to shadow residual

                fProxy beta = (rhoNew / rho) * (alpha / omega);

                p.addScaledInpl(-omega, v);                // p -= omega v      (still old p, old v)
                p.scaleAddInpl(beta, r);                    // p = beta p + r

                A.Apply(in p, ref v);                       // v = A p

                fProxy rv = Linear_OP.dot(rHat0, v);

                if (rv == (fProxy)0 || math.isnan(rv))
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr)); // breakdown: alpha undefined

                alpha = rhoNew / rv;

                r.addScaledInpl(-alpha, v);                 // r := s = r - alpha v

                fProxy ss = Linear_OP.dot(r, r);

                if (ss <= threshold)
                {
                    // Early exit: the half-step residual s is already small enough -- finish
                    // with x += alpha p (skipping the t = A s stabilization matvec entirely).
                    x.addScaledInpl(alpha, p);
                    return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(ss));
                }

                A.Apply(in r, ref t);                       // t = A s   (r currently holds s)

                fProxy tt = Linear_OP.dot(t, t);

                if (!(tt > (fProxy)0))                       // NaN-safe: tt is a norm^2, nonnegative
                    // breakdown: omega undefined. x is still x_old here (the alpha·p / omega·r
                    // updates are below), so its residual is rr -- NOT ss (ss = ‖b - A(x_old+alpha·p)‖,
                    // an iterate this path never commits to x).
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr));

                omega = Linear_OP.dot(t, r) / tt;

                if (omega == (fProxy)0 || math.isnan(omega))
                    // breakdown: beta would divide by zero. x is still x_old (see above) -> report rr.
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr));

                x.addScaledInpl(alpha, p);
                x.addScaledInpl(omega, r);                  // r still holds s here

                r.addScaledInpl(-omega, t);                 // r := s - omega t   (new residual)

                rr = Linear_OP.dot(r, r);

                if (rr <= threshold)
                    return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(rr));

                rho = rhoNew;
            }

            return MakeSolveInfo(IterativeSolveStatus.MaxIterations, maxIterations, math.sqrt(rr));
        }

        /// <summary>
        /// BiCGSTAB over a dense <see cref="fProxyMxN"/> -- zero-alloc primitive. Forwards into
        /// <see cref="biCGStab{TOp}"/> via <see cref="fProxyDenseOperator"/>.
        /// </summary>
        public static SolveInfo biCGStab(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                    ref fProxyN r, ref fProxyN rHat0, ref fProxyN p, ref fProxyN v, ref fProxyN t,
                                    int maxIterations, fProxy tolerance)
        {
            return biCGStab(new fProxyDenseOperator(in A), in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, maxIterations, tolerance);
        }

        /// <summary>BiCGSTAB over a dense matrix -- allocates five scratch vectors from the arena.</summary>
        public static SolveInfo biCGStab(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance)
        {
            fProxyN r     = b.tempfProxyVec(A.M_Rows);
            fProxyN rHat0 = b.tempfProxyVec(A.M_Rows);
            fProxyN p     = b.tempfProxyVec(A.M_Rows);
            fProxyN v     = b.tempfProxyVec(A.M_Rows);
            fProxyN t     = b.tempfProxyVec(A.M_Rows);
            return biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, maxIterations, tolerance);
        }

        /// <summary>BiCGSTAB over a dense matrix with default maxIterations (A.M_Rows) and tolerance (Consts.fProxySqrtEps).</summary>
        public static SolveInfo biCGStab(in fProxyMxN A, in fProxyN b, ref fProxyN x)
        {
            return biCGStab(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// BiCGSTAB over a block-sparse (BSR) matrix -- zero-alloc primitive. Forwards into
        /// <see cref="biCGStab{TOp}"/> via <c>fProxyBSMOperator</c>.
        /// </summary>
        public static SolveInfo biCGStab(in fProxyBSM A, in fProxyN b, ref fProxyN x,
                                    ref fProxyN r, ref fProxyN rHat0, ref fProxyN p, ref fProxyN v, ref fProxyN t,
                                    int maxIterations, fProxy tolerance)
        {
            return biCGStab(new fProxyBSMOperator(in A), in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, maxIterations, tolerance);
        }

        /// <summary>BiCGSTAB over a BSR matrix -- allocates five scratch vectors from the arena.</summary>
        public static SolveInfo biCGStab(in fProxyBSM A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance)
        {
            fProxyN r     = b.tempfProxyVec(A.M_Rows);
            fProxyN rHat0 = b.tempfProxyVec(A.M_Rows);
            fProxyN p     = b.tempfProxyVec(A.M_Rows);
            fProxyN v     = b.tempfProxyVec(A.M_Rows);
            fProxyN t     = b.tempfProxyVec(A.M_Rows);
            return biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, maxIterations, tolerance);
        }

        /// <summary>BiCGSTAB over a BSR matrix with default maxIterations (A.M_Rows) and tolerance (Consts.fProxySqrtEps).</summary>
        public static SolveInfo biCGStab(in fProxyBSM A, in fProxyN b, ref fProxyN x)
        {
            return biCGStab(in A, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Certified-exact least-squares residual diagnostics: given any solution x, recomputes
        /// rnorm = ‖b - A x‖, Arnorm = ‖Aᵀr - damp²x‖ (the damped normal-equation residual; damp==0
        /// -> ‖Aᵀr‖), and xnorm = ‖x‖ FRESH from x with one extra Apply + one ApplyT. This is the
        /// independent, matvec-paying counterpart to the FREE tracked norms the solvers put in their
        /// returned <see cref="LstsqInfo"/>: use it to audit a solution (or a warm-start seed)
        /// to full accuracy when the couple-extra-matvec cost is acceptable. Reuses two caller
        /// scratch buffers -- <paramref name="rScratch"/> (length A.Rows) and
        /// <paramref name="sScratch"/> (length A.Cols) -- so it allocates nothing. The returned
        /// struct carries only the three norms (iterations = 0, status = Converged as placeholders):
        /// it describes x, not a solve.
        ///
        /// DAMPED CONVENTION: Arnorm uses the ‖x‖-Tikhonov gradient ‖Aᵀr - damp²x‖ -- the optimality
        /// residual for cgls (any start) and for cold-start (x₀=0) lsqr/lsmr. Auditing a WARM-STARTED
        /// lsqr/lsmr result with damp!=0 will report a nonzero Arnorm even at that solver's optimum,
        /// because it minimizes the CORRECTION-penalized ‖Aᵀr - damp²(x-x₀)‖ instead; rnorm = ‖b-Ax‖
        /// is unaffected by the convention and is always exact.
        /// </summary>
        public static LstsqInfo lstsqResidual<TOp>(in TOp A, in fProxyN b, in fProxyN x, fProxy damp,
                                                         ref fProxyN rScratch, ref fProxyN sScratch)
            where TOp : struct, IfProxyLinearOperator
        {
            // r = b - A x
            A.Apply(in x, ref rScratch);
            rScratch.scaleAddInpl((fProxy)(-1), b);          // rScratch = -A x + b = b - A x
            fProxy rnorm = math.sqrt(Linear_OP.dot(rScratch, rScratch));

            // s = Aᵀr - damp²x  (the same optimality residual cgls's loop tracks)
            A.ApplyT(in rScratch, ref sScratch);
            if (damp != (fProxy)0) sScratch.addScaledInpl(-(damp * damp), x);
            fProxy arnorm = math.sqrt(Linear_OP.dot(sScratch, sScratch));

            fProxy xnorm = math.sqrt(Linear_OP.dot(x, x));

            return new LstsqInfo
            {
                rnorm = rnorm,
                Arnorm = arnorm,
                xnorm = xnorm,
                iterations = 0,
                status = IterativeSolveStatus.Converged,
            };
        }

        /// <summary>
        /// Zero-alloc CGLS solver for RECTANGULAR least-squares systems: minimizes ‖Ax-b‖₂ for
        /// possibly non-square A (over- or under-determined), generic over
        /// <see cref="IfProxyLinearOperator"/>. This is CG applied to the normal equations
        /// AᵀA x = Aᵀb, but NEVER explicitly forms AᵀA -- every AᵀA-vector product is one
        /// <see cref="IfProxyLinearOperator.Apply"/> plus one
        /// <see cref="IfProxyLinearOperator.ApplyT"/>. The normal-equation residual s = Aᵀr is
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
        /// Returns an <see cref="LstsqInfo"/> (implicit-bool == Solved): Breakdown on
        /// non-positive curvature ‖Ap‖²&lt;=0 (mirrors cg's p·Ap&lt;=0 guard: p is in null(A), or
        /// p==0), MaxIterations if it runs out. On a Breakdown return x is undefined -- only read
        /// x when Solved.
        /// </summary>
        public static LstsqInfo cgls<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                     ref fProxyN r, ref fProxyN s, ref fProxyN p, ref fProxyN q,
                                     int maxIterations, fProxy tolerance, fProxy damp)
            where TOp : struct, IfProxyLinearOperator
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
            fProxy atbSq = Linear_OP.dot(s, s);

            if (atbSq == (fProxy)0)
            {
                // A^T b == 0 -> x=0 is a valid least-squares minimizer regardless of warm start
                // (mirrors cg's bb==0 shortcut: a deterministic, NaN-sanitizing exact answer).
                for (int i = 0; i < x.N; i++) x[i] = (fProxy)0;
                // r = b, Aᵀr = Aᵀb = 0, x = 0.
                return new LstsqInfo { rnorm = math.sqrt(Linear_OP.dot(b, b)), Arnorm = (fProxy)0, xnorm = (fProxy)0, iterations = 0, status = IterativeSolveStatus.Converged };
            }

            fProxy threshold = tolerance * tolerance * atbSq;

            // r = b - A x
            A.Apply(in x, ref q);                          // q = A x (temp use of q)
            r.Data.CopyFrom(b.Data);
            r.addScaledInpl((fProxy)(-1), q);

            // s = A^T r - damp^2 x  (damped: the residual of the normal equations
            // (A^T A + damp^2 I) x = A^T b; damp==0 -> s = A^T r exactly, bit-identical).
            A.ApplyT(in r, ref s);
            if (damp != (fProxy)0) s.addScaledInpl(-(damp * damp), x);

            fProxy gamma = Linear_OP.dot(s, s);

            // rnorm/Arnorm/xnorm are all FREE here: r is live (one dot), Arnorm = √gamma is the
            // tracked normal-equation residual, xnorm one dot on x. No extra matvec.
            if (gamma <= threshold)
                return CglsInfo(IterativeSolveStatus.Converged, 0, gamma, in r, in x);

            p.Data.CopyFrom(s.Data);

            for (int k = 0; k < maxIterations; k++)
            {
                A.Apply(in p, ref q);                       // q = A p

                fProxy delta = Linear_OP.dot(q, q);
                if (damp != (fProxy)0) delta += (damp * damp) * Linear_OP.dot(p, p);   // p^T(A^T A + damp^2 I)p

                if (!(delta > (fProxy)0))                   // NaN-safe: also catches breakdown
                    return CglsInfo(IterativeSolveStatus.Breakdown, k + 1, gamma, in r, in x);

                fProxy alpha = gamma / delta;

                x.addScaledInpl(alpha, p);
                r.addScaledInpl(-alpha, q);

                A.ApplyT(in r, ref s);                       // s = A^T r, recomputed fresh (stability)
                if (damp != (fProxy)0) s.addScaledInpl(-(damp * damp), x);   // - damp^2 x (damped gradient)

                fProxy gammaNew = Linear_OP.dot(s, s);

                if (gammaNew <= threshold)
                    return CglsInfo(IterativeSolveStatus.Converged, k + 1, gammaNew, in r, in x);

                fProxy beta = gammaNew / gamma;

                p.scaleAddInpl(beta, s);                     // p = beta p + s

                gamma = gammaNew;
            }

            return CglsInfo(IterativeSolveStatus.MaxIterations, maxIterations, gamma, in r, in x);
        }

        /// <summary>Assemble a CGLS <see cref="LstsqInfo"/> from live state: rnorm = ‖r‖
        /// (r is CGLS's live residual b - A x), Arnorm = √gamma (its tracked ‖Aᵀr - damp²x‖²),
        /// xnorm = ‖x‖. Two dots on vectors already in cache -- no matvec.</summary>
        static LstsqInfo CglsInfo(IterativeSolveStatus status, int iterations, fProxy gamma, in fProxyN r, in fProxyN x)
            => new LstsqInfo
            {
                rnorm = math.sqrt(Linear_OP.dot(r, r)),
                Arnorm = math.sqrt(gamma),
                xnorm = math.sqrt(Linear_OP.dot(x, x)),
                iterations = iterations,
                status = status,
            };

        /// <summary>Undamped CGLS (damp = 0): plain least-squares. Forwards to the damped core.</summary>
        public static LstsqInfo cgls<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                     ref fProxyN r, ref fProxyN s, ref fProxyN p, ref fProxyN q,
                                     int maxIterations, fProxy tolerance)
            where TOp : struct, IfProxyLinearOperator
            => cgls(in A, in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance, (fProxy)0);

        /// <summary>
        /// CGLS over a dense <see cref="fProxyMxN"/> (possibly rectangular) -- zero-alloc
        /// primitive. Forwards into <see cref="cgls{TOp}"/> via <see cref="fProxyDenseOperator"/>.
        /// </summary>
        public static LstsqInfo cgls(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                ref fProxyN r, ref fProxyN s, ref fProxyN p, ref fProxyN q,
                                int maxIterations, fProxy tolerance)
        {
            return cgls(new fProxyDenseOperator(in A), in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance);
        }

        /// <summary>CGLS over a dense matrix -- allocates four scratch vectors from the arena.</summary>
        public static LstsqInfo cgls(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance)
        {
            fProxyN r = b.tempfProxyVec(A.M_Rows);
            fProxyN s = b.tempfProxyVec(A.N_Cols);
            fProxyN p = b.tempfProxyVec(A.N_Cols);
            fProxyN q = b.tempfProxyVec(A.M_Rows);
            return cgls(in A, in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance);
        }

        /// <summary>
        /// Damped (Tikhonov) CGLS over a dense matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates
        /// four scratch vectors from the arena. damp == 0 reproduces the plain least-squares solve.
        /// </summary>
        public static LstsqInfo cgls(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance, fProxy damp)
        {
            fProxyN r = b.tempfProxyVec(A.M_Rows);
            fProxyN s = b.tempfProxyVec(A.N_Cols);
            fProxyN p = b.tempfProxyVec(A.N_Cols);
            fProxyN q = b.tempfProxyVec(A.M_Rows);
            return cgls(new fProxyDenseOperator(in A), in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance, damp);
        }

        /// <summary>CGLS over a dense matrix with default maxIterations (A.N_Cols) and tolerance (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo cgls(in fProxyMxN A, in fProxyN b, ref fProxyN x)
        {
            return cgls(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// CGLS over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc primitive.
        /// Forwards into <see cref="cgls{TOp}"/> via <c>fProxyBSMOperator</c>. This is the payoff
        /// of rectangular BR x BC blocks: matrix-free least squares over a sparse Jacobian-like
        /// operator, never forming AᵀA.
        /// </summary>
        public static LstsqInfo cgls(in fProxyBSM A, in fProxyN b, ref fProxyN x,
                                ref fProxyN r, ref fProxyN s, ref fProxyN p, ref fProxyN q,
                                int maxIterations, fProxy tolerance)
        {
            return cgls(new fProxyBSMOperator(in A), in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance);
        }

        /// <summary>
        /// CGLS over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc primitive
        /// variant that takes a CALLER-PROVIDED precomputed transpose AT (e.g. built once via
        /// <c>arena.fProxyBSMTranspose(in A)</c> outside a hot loop / before a benchmark's timed
        /// region) and routes every ApplyT call through the resulting cache-friendly forward
        /// spMV(AT, x) instead of the scatter-heavy on-the-fly spMVT(A, x) -- see
        /// <see cref="fProxyBSMOperator"/>'s two-arg ctor. Caller is responsible for AT actually
        /// being A's transpose; this overload does not verify it. Prefer this over the allocating
        /// <see cref="cgls(in fProxyBSM, in fProxyN, ref fProxyN, int, fProxy)"/> overload when
        /// solving repeatedly against the same A (build AT once, reuse it across many solves).
        /// </summary>
        public static LstsqInfo cgls(in fProxyBSM A, in fProxyBSM AT, in fProxyN b, ref fProxyN x,
                                ref fProxyN r, ref fProxyN s, ref fProxyN p, ref fProxyN q,
                                int maxIterations, fProxy tolerance)
        {
            return cgls(new fProxyBSMOperator(in A, in AT), in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance);
        }

        /// <summary>
        /// CGLS over a BSR matrix -- allocates four scratch vectors AND materializes A^T ONCE
        /// via <c>arena.fProxyBSMTranspose</c> (same arena as the scratch vectors, taken from
        /// b), then drives CGLS with the two-arg <see cref="fProxyBSMOperator"/> so every
        /// ApplyT call routes through a cache-friendly forward spMV(A^T, x) instead of the
        /// scatter-heavy on-the-fly spMVT(A, x) every iteration -- this is the fix for the
        /// rectangular CGLS/LSQR transpose-matvec cache-unfriendliness (the one-time O(nnz)
        /// transpose build is amortized over every iteration). For a build-free zero-alloc path
        /// (e.g. many solves reusing the same A), build A^T yourself once (<c>arena.
        /// fProxyBSMTranspose(in A)</c>) and call the zero-alloc <see cref="cgls(in fProxyBSM,
        /// in fProxyBSM, in fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref
        /// fProxyN, int, fProxy)"/> overload above with your own scratch vectors, or the generic
        /// <see cref="cgls{TOp}"/> overload directly with <c>new fProxyBSMOperator(in A, in
        /// AT)</c>.
        /// </summary>
        public static LstsqInfo cgls(in fProxyBSM A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance)
        {
            fProxyN r = b.tempfProxyVec(A.M_Rows);
            fProxyN s = b.tempfProxyVec(A.N_Cols);
            fProxyN p = b.tempfProxyVec(A.N_Cols);
            fProxyN q = b.tempfProxyVec(A.M_Rows);
            fProxyBSM AT = b.fProxyBSMTranspose(in A);
            return cgls(new fProxyBSMOperator(in A, in AT), in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance);
        }

        /// <summary>
        /// Damped (Tikhonov) CGLS over a BSR matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates four
        /// scratch vectors AND materializes A^T once (see the undamped allocating overload). damp == 0
        /// reproduces the plain least-squares solve.
        /// </summary>
        public static LstsqInfo cgls(in fProxyBSM A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance, fProxy damp)
        {
            fProxyN r = b.tempfProxyVec(A.M_Rows);
            fProxyN s = b.tempfProxyVec(A.N_Cols);
            fProxyN p = b.tempfProxyVec(A.N_Cols);
            fProxyN q = b.tempfProxyVec(A.M_Rows);
            fProxyBSM AT = b.fProxyBSMTranspose(in A);
            return cgls(new fProxyBSMOperator(in A, in AT), in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance, damp);
        }

        /// <summary>CGLS over a BSR matrix with default maxIterations (A.N_Cols) and tolerance (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo cgls(in fProxyBSM A, in fProxyN b, ref fProxyN x)
        {
            return cgls(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Zero-alloc LSQR (Paige-Saunders 1982) solver for RECTANGULAR least-squares systems:
        /// minimizes ‖Ax-b‖₂ for possibly non-square A, generic over
        /// <see cref="IfProxyLinearOperator"/>. Builds an implicit bidiagonalization of A via the
        /// Golub-Kahan process (alternating <see cref="IfProxyLinearOperator.Apply"/> /
        /// <see cref="IfProxyLinearOperator.ApplyT"/> calls) and folds it through an incremental
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
        /// Returns an <see cref="LstsqInfo"/> (implicit-bool == Solved): Breakdown on a total
        /// bidiagonalization breakdown (the current alpha and beta both collapse to zero in the same
        /// step -- the Golub-Kahan recurrence exhausted), MaxIterations if it runs out. On a
        /// Breakdown return x is undefined -- only read x when Solved.
        /// </summary>
        public static LstsqInfo lsqr<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                     ref fProxyN u, ref fProxyN v, ref fProxyN w,
                                     ref fProxyN tmpM, ref fProxyN tmpN,
                                     int maxIterations, fProxy tolerance, fProxy damp)
            where TOp : struct, IfProxyLinearOperator
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
            fProxy atbSq = Linear_OP.dot(tmpN, tmpN);

            if (atbSq == (fProxy)0)
            {
                for (int i = 0; i < x.N; i++) x[i] = (fProxy)0;
                // r = b, Aᵀr = Aᵀb = 0, x = 0.
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, math.sqrt(Linear_OP.dot(b, b)), (fProxy)0, (fProxy)0, in x);
            }

            fProxy threshold = tolerance * tolerance * atbSq;

            // u = b - A x ; beta = ||u||
            A.Apply(in x, ref tmpM);
            u.Data.CopyFrom(b.Data);
            u.addScaledInpl((fProxy)(-1), tmpM);

            fProxy beta = math.sqrt(Linear_OP.dot(u, u));

            if (beta == (fProxy)0)
                // x already exact (r = 0): rnorm = 0, Aᵀr = 0.
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, (fProxy)0, (fProxy)0, (fProxy)0, in x);

            u.divInpl(beta);

            // v = A^T u ; alpha = ||v||
            A.ApplyT(in u, ref tmpN);
            v.Data.CopyFrom(tmpN.Data);

            fProxy alpha = math.sqrt(Linear_OP.dot(v, v));

            if (alpha == (fProxy)0)
                // x already least-squares-stationary (A^T r = 0). ‖r‖ = beta.
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, beta, (fProxy)0, (fProxy)0, in x);

            v.divInpl(alpha);

            // phibar tracks ‖r‖ (LSQR identity); arnorm tracks ‖Aᵀr‖ = alpha*beta pre-loop.
            fProxy phibar = beta;
            fProxy rhobar = alpha;
            fProxy arnorm = alpha * beta;

            // Σψ²: energy the damping rotations peel off phibar into the residual. With damp>0 the
            // residual LSQR actually reduces is the AUGMENTED one ‖[b-Ax; -damp·x]‖, whose square is
            // sumPsiSq + phibar² -- phibar ALONE is neither the plain nor the augmented residual once
            // damping folds in. LstsqInfoTracked recovers the plain ‖b-Ax‖ from the augmented norm.
            // damp==0 -> sumPsiSq stays 0, so the undamped path reports rnorm = phibar unchanged.
            fProxy sumPsiSq = (fProxy)0;

            if (arnorm * arnorm <= threshold)
                // already within tolerance before the first bidiagonalization step
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, phibar, arnorm, (fProxy)0, in x);

            w.Data.CopyFrom(v.Data);

            for (int k = 0; k < maxIterations; k++)
            {
                // ---- bidiagonalization step (Golub-Kahan) ----
                A.Apply(in v, ref tmpM);
                u.scaleAddInpl(-alpha, tmpM);              // u = -alpha*u + tmpM = A v - alpha u
                beta = math.sqrt(Linear_OP.dot(u, u));
                if (beta > (fProxy)0) u.divInpl(beta);

                A.ApplyT(in u, ref tmpN);
                v.scaleAddInpl(-beta, tmpN);                // v = -beta*v + tmpN = A^T u - beta v
                alpha = math.sqrt(Linear_OP.dot(v, v));
                if (alpha > (fProxy)0) v.divInpl(alpha);

                // ---- fold Tikhonov damping into rhobar: rotate (rhobar, damp) -> (rhobar1, 0),
                // scaling phibar by the rotation cosine. damp==0 -> rhobar1==rhobar and phibar is
                // untouched, so the undamped path is bit-identical. ----
                fProxy rhobar1 = rhobar;
                if (damp != (fProxy)0)
                {
                    rhobar1 = math.sqrt(rhobar * rhobar + damp * damp);
                    fProxy psi = (damp / rhobar1) * phibar;  // sn1 * phibar: residual rotated out by damping
                    sumPsiSq += psi * psi;
                    phibar = (rhobar / rhobar1) * phibar;   // cs1 * phibar
                }

                // ---- Givens rotation folding (rhobar1, beta) -> (rho, 0) ----
                fProxy rho = math.sqrt(rhobar1 * rhobar1 + beta * beta);

                if (!(rho > (fProxy)0))
                    // total breakdown: rhobar1 and beta both zero. phibar/arnorm carry the last
                    // pre-rotation values (arnorm from the previous completed step).
                    return LstsqInfoTracked(IterativeSolveStatus.Breakdown, k + 1, math.sqrt(sumPsiSq + phibar * phibar), arnorm, damp, in x);

                fProxy c = rhobar1 / rho;
                fProxy sn = beta / rho;
                fProxy theta = sn * alpha;
                rhobar = -c * alpha;
                fProxy phi = c * phibar;
                phibar = sn * phibar;

                // ---- update x using the OLD w, then update w ----
                x.addScaledInpl(phi / rho, w);
                w.scaleAddInpl(-theta / rho, v);             // w = -(theta/rho)*w + v

                arnorm = phibar * alpha * math.abs(c);        // ‖Aᵀr‖ for the just-updated x (free)

                if (arnorm * arnorm <= threshold)
                    return LstsqInfoTracked(IterativeSolveStatus.Converged, k + 1, math.sqrt(sumPsiSq + phibar * phibar), arnorm, damp, in x);

                if (!(beta > (fProxy)0) || !(alpha > (fProxy)0)) // NaN-safe: both are norms, nonnegative
                    // bidiagonalization breakdown: Krylov space exhausted, no further progress
                    return LstsqInfoTracked(IterativeSolveStatus.Breakdown, k + 1, math.sqrt(sumPsiSq + phibar * phibar), arnorm, damp, in x);
            }

            return LstsqInfoTracked(IterativeSolveStatus.MaxIterations, maxIterations, math.sqrt(sumPsiSq + phibar * phibar), arnorm, damp, in x);
        }

        /// <summary>Assemble an <see cref="LstsqInfo"/> from a solver's tracked residual and
        /// ‖Aᵀr‖ scalars, filling xnorm = ‖x‖ with one dot on x. Shared by lsqr/lsmr (cgls uses
        /// <see cref="CglsInfo"/>, which reads ‖r‖ from its live residual instead).
        ///
        /// <paramref name="resNorm"/> is the residual norm of the system the solver actually
        /// bidiagonalizes. When <paramref name="dampAug"/> != 0 that is the AUGMENTED residual
        /// √(‖b-Ax‖² + damp²‖x - x₀‖²) (lsqr/lsmr regularize by bidiagonalizing [A; damp·I] on the
        /// residual b - A·x₀), so we recover the plain ‖b-Ax‖ = √(resNorm² − damp²‖x‖²) here -- FREE,
        /// reusing the xnorm we already compute. This gives rnorm = ‖b-Ax‖ consistently across every
        /// solver for all UNDAMPED solves and for the documented COLD-START (x₀=0) damped usage, where
        /// ‖x - x₀‖ = ‖x‖. CAVEAT: under the niche combination of a NONZERO warm start AND damping,
        /// the augmented residual penalizes ‖x - x₀‖ (not ‖x‖), so this recovery (which does not
        /// retain x₀) does NOT return ‖b-Ax‖ -- start damped solves from x=0, or read ‖b-Ax‖ from
        /// Solvers.lstsqResidual on the returned x. Call sites whose resNorm is ALREADY the plain
        /// residual (the pre-loop early exits, where no bidiagonalization/damping rotation has folded
        /// in yet, so resNorm = beta = ‖b - A·x₀‖) pass dampAug = 0 to skip the recovery. dampAug = 0
        /// makes this the identity, so the undamped path is unchanged.</summary>
        static LstsqInfo LstsqInfoTracked(IterativeSolveStatus status, int iterations, fProxy resNorm, fProxy Arnorm, fProxy dampAug, in fProxyN x)
        {
            fProxy xnorm = math.sqrt(Linear_OP.dot(x, x));
            fProxy rr = resNorm * resNorm - dampAug * dampAug * xnorm * xnorm;
            fProxy rnorm = rr > (fProxy)0 ? math.sqrt(rr) : (fProxy)0;   // guard estimate noise when ‖b-Ax‖≈0
            return new LstsqInfo
            {
                rnorm = rnorm,
                Arnorm = Arnorm,
                xnorm = xnorm,
                iterations = iterations,
                status = status,
            };
        }

        /// <summary>Undamped LSQR (damp = 0): plain least-squares. Forwards to the damped core.</summary>
        public static LstsqInfo lsqr<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                     ref fProxyN u, ref fProxyN v, ref fProxyN w,
                                     ref fProxyN tmpM, ref fProxyN tmpN,
                                     int maxIterations, fProxy tolerance)
            where TOp : struct, IfProxyLinearOperator
            => lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance, (fProxy)0);

        /// <summary>
        /// LSQR over a dense <see cref="fProxyMxN"/> (possibly rectangular) -- zero-alloc
        /// primitive. Forwards into <see cref="lsqr{TOp}"/> via <see cref="fProxyDenseOperator"/>.
        /// </summary>
        public static LstsqInfo lsqr(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v, ref fProxyN w,
                                ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIterations, fProxy tolerance)
        {
            return lsqr(new fProxyDenseOperator(in A), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>LSQR over a dense matrix -- allocates five scratch vectors from the arena.</summary>
        public static LstsqInfo lsqr(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance)
        {
            fProxyN u    = b.tempfProxyVec(A.M_Rows);
            fProxyN v    = b.tempfProxyVec(A.N_Cols);
            fProxyN w    = b.tempfProxyVec(A.N_Cols);
            fProxyN tmpM = b.tempfProxyVec(A.M_Rows);
            fProxyN tmpN = b.tempfProxyVec(A.N_Cols);
            return lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// Damped (Tikhonov) LSQR over a dense matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates
        /// five scratch vectors from the arena. damp == 0 reproduces the plain least-squares solve.
        /// </summary>
        public static LstsqInfo lsqr(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance, fProxy damp)
        {
            fProxyN u    = b.tempfProxyVec(A.M_Rows);
            fProxyN v    = b.tempfProxyVec(A.N_Cols);
            fProxyN w    = b.tempfProxyVec(A.N_Cols);
            fProxyN tmpM = b.tempfProxyVec(A.M_Rows);
            fProxyN tmpN = b.tempfProxyVec(A.N_Cols);
            return lsqr(new fProxyDenseOperator(in A), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance, damp);
        }

        /// <summary>LSQR over a dense matrix with default maxIterations (A.N_Cols) and tolerance (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo lsqr(in fProxyMxN A, in fProxyN b, ref fProxyN x)
        {
            return lsqr(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// LSQR over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc primitive.
        /// Forwards into <see cref="lsqr{TOp}"/> via <c>fProxyBSMOperator</c>. This is the payoff
        /// of rectangular BR x BC blocks: matrix-free least squares over a sparse Jacobian-like
        /// operator, never forming AᵀA, with better ill-conditioned behavior than <see
        /// cref="cgls{TOp}"/>.
        /// </summary>
        public static LstsqInfo lsqr(in fProxyBSM A, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v, ref fProxyN w,
                                ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIterations, fProxy tolerance)
        {
            return lsqr(new fProxyBSMOperator(in A), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// LSQR over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc primitive
        /// variant that takes a CALLER-PROVIDED precomputed transpose AT (e.g. built once via
        /// <c>arena.fProxyBSMTranspose(in A)</c> outside a hot loop / before a benchmark's timed
        /// region) and routes every ApplyT call through the resulting cache-friendly forward
        /// spMV(AT, x) instead of the scatter-heavy on-the-fly spMVT(A, x) -- see
        /// <see cref="fProxyBSMOperator"/>'s two-arg ctor. Caller is responsible for AT actually
        /// being A's transpose; this overload does not verify it. Prefer this over the allocating
        /// <see cref="lsqr(in fProxyBSM, in fProxyN, ref fProxyN, int, fProxy)"/> overload when
        /// solving repeatedly against the same A (build AT once, reuse it across many solves).
        /// </summary>
        public static LstsqInfo lsqr(in fProxyBSM A, in fProxyBSM AT, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v, ref fProxyN w,
                                ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIterations, fProxy tolerance)
        {
            return lsqr(new fProxyBSMOperator(in A, in AT), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// LSQR over a BSR matrix -- allocates five scratch vectors AND materializes A^T ONCE
        /// via <c>arena.fProxyBSMTranspose</c> (same arena as the scratch vectors, taken from
        /// b), then drives LSQR with the two-arg <see cref="fProxyBSMOperator"/> so every
        /// ApplyT call routes through a cache-friendly forward spMV(A^T, x) instead of the
        /// scatter-heavy on-the-fly spMVT(A, x) every iteration -- same fix and same tradeoff as
        /// <see cref="cgls(in fProxyBSM, in fProxyN, ref fProxyN, int, fProxy)"/>: for a
        /// build-free zero-alloc path, build A^T yourself once (<c>arena.fProxyBSMTranspose(in
        /// A)</c>) and call the zero-alloc <see cref="lsqr(in fProxyBSM, in fProxyBSM, in
        /// fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, int,
        /// fProxy)"/> overload above with your own scratch vectors, or the generic
        /// <see cref="lsqr{TOp}"/> overload directly with <c>new fProxyBSMOperator(in A, in
        /// AT)</c>.
        /// </summary>
        public static LstsqInfo lsqr(in fProxyBSM A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance)
        {
            fProxyN u    = b.tempfProxyVec(A.M_Rows);
            fProxyN v    = b.tempfProxyVec(A.N_Cols);
            fProxyN w    = b.tempfProxyVec(A.N_Cols);
            fProxyN tmpM = b.tempfProxyVec(A.M_Rows);
            fProxyN tmpN = b.tempfProxyVec(A.N_Cols);
            fProxyBSM AT = b.fProxyBSMTranspose(in A);
            return lsqr(new fProxyBSMOperator(in A, in AT), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// Damped (Tikhonov) LSQR over a BSR matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates five
        /// scratch vectors AND materializes A^T once (see the undamped allocating overload). damp == 0
        /// reproduces the plain least-squares solve.
        /// </summary>
        public static LstsqInfo lsqr(in fProxyBSM A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance, fProxy damp)
        {
            fProxyN u    = b.tempfProxyVec(A.M_Rows);
            fProxyN v    = b.tempfProxyVec(A.N_Cols);
            fProxyN w    = b.tempfProxyVec(A.N_Cols);
            fProxyN tmpM = b.tempfProxyVec(A.M_Rows);
            fProxyN tmpN = b.tempfProxyVec(A.N_Cols);
            fProxyBSM AT = b.fProxyBSMTranspose(in A);
            return lsqr(new fProxyBSMOperator(in A, in AT), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance, damp);
        }

        /// <summary>LSQR over a BSR matrix with default maxIterations (A.N_Cols) and tolerance (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo lsqr(in fProxyBSM A, in fProxyN b, ref fProxyN x)
        {
            return lsqr(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Zero-alloc LSMR (Fong-Saunders 2011) solver for RECTANGULAR least-squares systems:
        /// minimizes ‖Ax-b‖₂ for possibly non-square A, generic over
        /// <see cref="IfProxyLinearOperator"/>. Built on the SAME Golub-Kahan bidiagonalization as
        /// <see cref="lsqr{TOp}"/> (alternating <see cref="IfProxyLinearOperator.Apply"/> /
        /// <see cref="IfProxyLinearOperator.ApplyT"/>), but folds it through a rotation sequence
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
        /// Returns an <see cref="LstsqInfo"/> (implicit-bool == Solved): Breakdown on a
        /// bidiagonalization breakdown (a rotation radius collapses to zero -- the Golub-Kahan
        /// recurrence exhausted), MaxIterations if it runs out. On a Breakdown return x is undefined
        /// -- only read x when Solved.
        /// </summary>
        public static LstsqInfo lsmr<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                     ref fProxyN u, ref fProxyN v, ref fProxyN h,
                                     ref fProxyN hbar, ref fProxyN tmpM, ref fProxyN tmpN,
                                     int maxIterations, fProxy tolerance, fProxy damp)
            where TOp : struct, IfProxyLinearOperator
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
            fProxy atbSq = Linear_OP.dot(tmpN, tmpN);

            if (atbSq == (fProxy)0)
            {
                for (int i = 0; i < x.N; i++) x[i] = (fProxy)0;
                // r = b, Aᵀr = Aᵀb = 0, x = 0.
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, math.sqrt(Linear_OP.dot(b, b)), (fProxy)0, (fProxy)0, in x);
            }

            fProxy threshold = tolerance * tolerance * atbSq;

            // u = b - A x ; beta = ||u||   (warm-startable: bidiagonalize the residual)
            A.Apply(in x, ref tmpM);
            u.Data.CopyFrom(b.Data);
            u.addScaledInpl((fProxy)(-1), tmpM);

            fProxy beta = math.sqrt(Linear_OP.dot(u, u));

            if (beta == (fProxy)0)
                // x already exact (r = 0).
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, (fProxy)0, (fProxy)0, (fProxy)0, in x);

            u.divInpl(beta);

            // v = A^T u ; alpha = ||v||
            A.ApplyT(in u, ref tmpN);
            v.Data.CopyFrom(tmpN.Data);

            fProxy alpha = math.sqrt(Linear_OP.dot(v, v));

            if (alpha == (fProxy)0)
                // x already least-squares-stationary (A^T r = 0). ‖r‖ = beta.
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, beta, (fProxy)0, (fProxy)0, in x);

            v.divInpl(alpha);

            // ||A^T r_0|| = alpha*beta = |zetabar_1|; matches lsqr's pre-loop early-out.
            if ((alpha * beta) * (alpha * beta) <= threshold)
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, beta, alpha * beta, (fProxy)0, in x);

            // h = v ; hbar = 0
            h.Data.CopyFrom(v.Data);
            for (int i = 0; i < hbar.N; i++) hbar[i] = (fProxy)0;

            // MINRES-on-normal-equations rotation state.
            fProxy alphabar = alpha;
            fProxy zetabar  = alpha * beta;
            fProxy rho = (fProxy)1, rhobar = (fProxy)1, cbar = (fProxy)1, sbar = (fProxy)0;

            // ---- ‖r‖ estimate state (Fong & Saunders 2011, "LSMR" §5.4 / SciPy lsmr). LSMR does
            // not hold the residual r = b - A x, but ‖r‖ falls out of a short scalar recurrence
            // over the SAME rotations at O(1)/iteration -- no extra matvec/dot. beta here is
            // beta1 = ‖b - A x0‖; undamped (damp==0) -> chat==1, shat==0 -> betacheck==0. ----
            fProxy betadd = beta;
            fProxy betad = (fProxy)0;
            fProxy rhodold = (fProxy)1;
            fProxy tautildeold = (fProxy)0;
            fProxy thetatilde = (fProxy)0;
            fProxy zeta = (fProxy)0;
            fProxy dnorm = (fProxy)0;   // accumulates betacheck^2
            fProxy normr = beta;

            for (int k = 0; k < maxIterations; k++)
            {
                // ---- bidiagonalization step (Golub-Kahan) ----
                A.Apply(in v, ref tmpM);
                u.scaleAddInpl(-alpha, tmpM);              // u = A v - alpha u
                beta = math.sqrt(Linear_OP.dot(u, u));
                if (beta > (fProxy)0)
                {
                    u.divInpl(beta);
                    A.ApplyT(in u, ref tmpN);
                    v.scaleAddInpl(-beta, tmpN);            // v = A^T u - beta v
                    alpha = math.sqrt(Linear_OP.dot(v, v));
                    if (alpha > (fProxy)0) v.divInpl(alpha);
                }

                // ---- rotation P_k : (alphahat, beta) -> (rho, 0) ----
                // alphahat folds in the Tikhonov damping: alphahat = sqrt(alphabar^2 + damp^2).
                // damp==0 -> alphahat==alphabar exactly, so the undamped path is bit-identical.
                // (chat, shat) is the rotation folding damp -- needed by the ‖r‖ recurrence.
                fProxy rhoold = rho;
                fProxy alphahat = damp != (fProxy)0 ? math.sqrt(alphabar * alphabar + damp * damp) : alphabar;
                fProxy chat, shat;
                if (alphahat > (fProxy)0) { chat = alphabar / alphahat; shat = damp / alphahat; }
                else { chat = (fProxy)1; shat = (fProxy)0; }
                rho = math.sqrt(alphahat * alphahat + beta * beta);
                if (!(rho > (fProxy)0))
                    // breakdown: alphahat and beta both zero. normr/zetabar carry the prior step's values.
                    return LstsqInfoTracked(IterativeSolveStatus.Breakdown, k + 1, normr, math.abs(zetabar), damp, in x);
                fProxy c = alphahat / rho;
                fProxy s = beta / rho;
                fProxy thetanew = s * alpha;
                alphabar = c * alpha;

                // ---- rotation Pbar_k : fold R^T into Rbar (the MINRES layer) ----
                fProxy rhobarold = rhobar;
                fProxy thetabar = sbar * rho;
                fProxy cbarrho = cbar * rho;
                rhobar = math.sqrt(cbarrho * cbarrho + thetanew * thetanew);
                if (!(rhobar > (fProxy)0))
                    return LstsqInfoTracked(IterativeSolveStatus.Breakdown, k + 1, normr, math.abs(zetabar), damp, in x);
                cbar = cbarrho / rhobar;
                sbar = thetanew / rhobar;
                fProxy zetaold = zeta;
                zeta = cbar * zetabar;
                zetabar = -sbar * zetabar;

                // ---- updates: hbar, x, h ----
                // hbar = h - (thetabar*rho / (rhoold*rhobarold)) * hbar
                fProxy coefHbar = thetabar * rho / (rhoold * rhobarold);
                hbar.scaleAddInpl(-coefHbar, h);           // hbar = -coefHbar*hbar + h
                // x = x + (zeta / (rho*rhobar)) * hbar
                x.addScaledInpl(zeta / (rho * rhobar), hbar);
                // h = v - (thetanew/rho) * h
                h.scaleAddInpl(-thetanew / rho, v);         // h = -(thetanew/rho)*h + v

                // ---- ‖r‖ recurrence for the just-updated x (this step's rotations; no matvec) ----
                fProxy betaacute = chat * betadd;
                fProxy betacheck = -shat * betadd;
                fProxy betahat = c * betaacute;
                betadd = -s * betaacute;

                fProxy thetatildeold = thetatilde;
                fProxy rhotildeold = math.sqrt(rhodold * rhodold + thetabar * thetabar);
                fProxy ctildeold = rhotildeold > (fProxy)0 ? rhodold / rhotildeold : (fProxy)1;
                fProxy stildeold = rhotildeold > (fProxy)0 ? thetabar / rhotildeold : (fProxy)0;
                thetatilde = stildeold * rhobar;
                rhodold = ctildeold * rhobar;
                betad = -stildeold * betad + ctildeold * betahat;

                tautildeold = rhotildeold > (fProxy)0 ? (zetaold - thetatildeold * tautildeold) / rhotildeold : (fProxy)0;
                fProxy taud = rhodold > (fProxy)0 ? (zeta - thetatilde * tautildeold) / rhodold : (fProxy)0;
                dnorm = dnorm + betacheck * betacheck;
                normr = math.sqrt(dnorm + (betad - taud) * (betad - taud) + betadd * betadd);

                // ‖A^T r‖ for the just-updated x = |zetabar| (falls out for free, decreases
                // monotonically). With damping this is the DAMPED normal-equation residual
                // ‖AᵀA x + damp² x − Aᵀb‖ = ‖Aᵀr − damp² x‖.
                if (zetabar * zetabar <= threshold)
                    return LstsqInfoTracked(IterativeSolveStatus.Converged, k + 1, normr, math.abs(zetabar), damp, in x);

                if (!(beta > (fProxy)0) || !(alpha > (fProxy)0)) // NaN-safe: both are norms, nonnegative
                    // bidiagonalization breakdown: Krylov space exhausted, no further progress
                    return LstsqInfoTracked(IterativeSolveStatus.Breakdown, k + 1, normr, math.abs(zetabar), damp, in x);
            }

            return LstsqInfoTracked(IterativeSolveStatus.MaxIterations, maxIterations, normr, math.abs(zetabar), damp, in x);
        }

        /// <summary>Undamped LSMR (damp = 0): plain least-squares. Forwards to the damped core.</summary>
        public static LstsqInfo lsmr<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                     ref fProxyN u, ref fProxyN v, ref fProxyN h,
                                     ref fProxyN hbar, ref fProxyN tmpM, ref fProxyN tmpN,
                                     int maxIterations, fProxy tolerance)
            where TOp : struct, IfProxyLinearOperator
            => lsmr(in A, in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance, (fProxy)0);

        /// <summary>
        /// LSMR over a dense <see cref="fProxyMxN"/> (possibly rectangular) -- zero-alloc
        /// primitive. Forwards into <see cref="lsmr{TOp}"/> via <see cref="fProxyDenseOperator"/>.
        /// </summary>
        public static LstsqInfo lsmr(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v, ref fProxyN h,
                                ref fProxyN hbar, ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIterations, fProxy tolerance)
        {
            return lsmr(new fProxyDenseOperator(in A), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>LSMR over a dense matrix -- allocates six scratch vectors from the arena.</summary>
        public static LstsqInfo lsmr(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance)
        {
            fProxyN u    = b.tempfProxyVec(A.M_Rows);
            fProxyN v    = b.tempfProxyVec(A.N_Cols);
            fProxyN h    = b.tempfProxyVec(A.N_Cols);
            fProxyN hbar = b.tempfProxyVec(A.N_Cols);
            fProxyN tmpM = b.tempfProxyVec(A.M_Rows);
            fProxyN tmpN = b.tempfProxyVec(A.N_Cols);
            return lsmr(in A, in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// Damped (Tikhonov) LSMR over a dense matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates
        /// six scratch vectors from the arena. damp == 0 reproduces the plain least-squares solve.
        /// </summary>
        public static LstsqInfo lsmr(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance, fProxy damp)
        {
            fProxyN u    = b.tempfProxyVec(A.M_Rows);
            fProxyN v    = b.tempfProxyVec(A.N_Cols);
            fProxyN h    = b.tempfProxyVec(A.N_Cols);
            fProxyN hbar = b.tempfProxyVec(A.N_Cols);
            fProxyN tmpM = b.tempfProxyVec(A.M_Rows);
            fProxyN tmpN = b.tempfProxyVec(A.N_Cols);
            return lsmr(new fProxyDenseOperator(in A), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance, damp);
        }

        /// <summary>LSMR over a dense matrix with default maxIterations (A.N_Cols) and tolerance (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo lsmr(in fProxyMxN A, in fProxyN b, ref fProxyN x)
        {
            return lsmr(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// LSMR over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc primitive.
        /// Forwards into <see cref="lsmr{TOp}"/> via <c>fProxyBSMOperator</c>. Matrix-free least
        /// squares over a sparse Jacobian-like operator, never forming AᵀA, with LSMR's monotone
        /// ‖Aᵀr‖ decrease (see the generic overload).
        /// </summary>
        public static LstsqInfo lsmr(in fProxyBSM A, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v, ref fProxyN h,
                                ref fProxyN hbar, ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIterations, fProxy tolerance)
        {
            return lsmr(new fProxyBSMOperator(in A), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// LSMR over a (possibly rectangular) BSR matrix -- zero-alloc primitive that takes a
        /// CALLER-PROVIDED precomputed transpose AT (e.g. <c>arena.fProxyBSMTranspose(in A)</c>
        /// built once outside a hot loop) and routes every ApplyT through the cache-friendly
        /// forward spMV(AT, x) instead of on-the-fly spMVT(A, x) -- see
        /// <see cref="fProxyBSMOperator"/>'s two-arg ctor. Caller is responsible for AT being A's
        /// transpose; this overload does not verify it.
        /// </summary>
        public static LstsqInfo lsmr(in fProxyBSM A, in fProxyBSM AT, in fProxyN b, ref fProxyN x,
                                ref fProxyN u, ref fProxyN v, ref fProxyN h,
                                ref fProxyN hbar, ref fProxyN tmpM, ref fProxyN tmpN,
                                int maxIterations, fProxy tolerance)
        {
            return lsmr(new fProxyBSMOperator(in A, in AT), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// LSMR over a BSR matrix -- allocates six scratch vectors AND materializes A^T ONCE via
        /// <c>arena.fProxyBSMTranspose</c>, then drives LSMR with the two-arg
        /// <see cref="fProxyBSMOperator"/> so every ApplyT routes through a cache-friendly forward
        /// spMV(A^T, x). For a build-free zero-alloc path, build A^T yourself once and call the
        /// zero-alloc AT overload above with your own scratch vectors.
        /// </summary>
        public static LstsqInfo lsmr(in fProxyBSM A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance)
        {
            fProxyN u    = b.tempfProxyVec(A.M_Rows);
            fProxyN v    = b.tempfProxyVec(A.N_Cols);
            fProxyN h    = b.tempfProxyVec(A.N_Cols);
            fProxyN hbar = b.tempfProxyVec(A.N_Cols);
            fProxyN tmpM = b.tempfProxyVec(A.M_Rows);
            fProxyN tmpN = b.tempfProxyVec(A.N_Cols);
            fProxyBSM AT = b.fProxyBSMTranspose(in A);
            return lsmr(new fProxyBSMOperator(in A, in AT), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// Damped (Tikhonov) LSMR over a BSR matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates six
        /// scratch vectors AND materializes A^T once (see the undamped allocating overload). damp == 0
        /// reproduces the plain least-squares solve.
        /// </summary>
        public static LstsqInfo lsmr(in fProxyBSM A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance, fProxy damp)
        {
            fProxyN u    = b.tempfProxyVec(A.M_Rows);
            fProxyN v    = b.tempfProxyVec(A.N_Cols);
            fProxyN h    = b.tempfProxyVec(A.N_Cols);
            fProxyN hbar = b.tempfProxyVec(A.N_Cols);
            fProxyN tmpM = b.tempfProxyVec(A.M_Rows);
            fProxyN tmpN = b.tempfProxyVec(A.N_Cols);
            fProxyBSM AT = b.fProxyBSMTranspose(in A);
            return lsmr(new fProxyBSMOperator(in A, in AT), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance, damp);
        }

        /// <summary>LSMR over a BSR matrix with default maxIterations (A.N_Cols) and tolerance (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo lsmr(in fProxyBSM A, in fProxyN b, ref fProxyN x)
        {
            return lsmr(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);
        }

        // ==================== AᵀA-Jacobi (column-equilibration) convenience overloads ====================
        // cglsJacobi / lsqrJacobi / lsmrJacobi build the column scale d[j] = 1/||A_:,j|| from
        // columnNormsSquared, wrap A in a fProxyColScaledOperator, solve the equilibrated system
        // (A*D) y = b with the underlying solver (COLD start -- x is zeroed internally; column
        // scaling is a change of variable, so a warm start would need pre-mapping y0 = D^-1 x0), and
        // unscale x = D*y in place. On an ill-conditioned least-squares problem this converges in
        // fewer iterations than the un-preconditioned solve to the SAME solution. Everything is
        // temp-pool allocated from b. BSM forms materialize A^T once (ApplyT-heavy). For explicit
        // control (custom d, warm start, damping semantics, zero-alloc) use the composable path
        // directly: Linear_OP.columnNormsSquared + buildJacobiScale + fProxyColScaledOperator + the
        // generic solver overload.
        //
        // DIAGNOSTICS: the returned LstsqInfo is reported in ORIGINAL coordinates. The
        // equilibrated solve tracks rnorm/Arnorm/xnorm in scaled y-space (Arnorm = ‖D·Aᵀr‖ can be
        // wildly off), so JacobiFinish recomputes all three exactly on the unscaled A via
        // lstsqResidual (one Apply + ApplyT) and keeps the solve's iteration count + status. So
        // info.Arnorm here is the true ‖Aᵀr‖ and info.Solved still reflects the equilibrated solve.

        /// <summary>Shared tail for the *Jacobi convenience wrappers: unscale the solution
        /// (x = D·y) back to the ORIGINAL variables, then report diagnostics in original coordinates.
        /// The equilibrated solve of (A·D)y=b tracks rnorm/Arnorm/xnorm in the SCALED y-space --
        /// rnorm happens to coincide (‖b-(AD)y‖ = ‖b-Ax‖) but Arnorm = ‖(AD)ᵀr‖ = ‖D·Aᵀr‖ is off by
        /// the column scaling (badly so on exactly the ill-scaled systems the preconditioner targets).
        /// So we recompute all three exactly on the UNSCALED operator via <see cref="lstsqResidual"/>
        /// (one Apply + one ApplyT -- negligible next to the solve and the Aᵀ build these wrappers
        /// already pay), keeping the solve's iteration count and status. <paramref name="Aop"/> is the
        /// UNSCALED operator; <paramref name="mScratch"/>/<paramref name="nScratch"/> are Rows-/Cols-
        /// length scratch (the solver's own buffers, free to reuse post-solve).</summary>
        static LstsqInfo JacobiFinish<TOp>(in TOp Aop, in fProxyN b, ref fProxyN x, in fProxyN d,
                                                 int iterations, IterativeSolveStatus status,
                                                 ref fProxyN mScratch, ref fProxyN nScratch)
            where TOp : struct, IfProxyLinearOperator
        {
            for (int j = 0; j < d.N; j++) x[j] *= d[j];      // unscale x = D y
            var info = lstsqResidual(in Aop, in b, in x, (fProxy)0, ref mScratch, ref nScratch);
            info.iterations = iterations;
            info.status = status;
            return info;
        }

        // ---- CGLS + Jacobi ----
        /// <summary>CGLS with an AᵀA-Jacobi column-equilibration preconditioner over a dense matrix.</summary>
        public static LstsqInfo cglsJacobi(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance)
        {
            int m = A.M_Rows, n = A.N_Cols;
            fProxyN d = b.tempfProxyVec(n), d2 = b.tempfProxyVec(n), scratch = b.tempfProxyVec(n);
            Linear_OP.columnNormsSquared(in A, ref d2);
            Linear_OP.buildJacobiScale(in d2, ref d);
            var op = new fProxyColScaledOperator<fProxyDenseOperator>(new fProxyDenseOperator(in A), d, scratch);

            for (int j = 0; j < n; j++) x[j] = (fProxy)0;                 // cold start (change of variable)
            fProxyN r = b.tempfProxyVec(m), s = b.tempfProxyVec(n), p = b.tempfProxyVec(n), q = b.tempfProxyVec(m);
            var solveInfo = cgls(op, in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance);
            return JacobiFinish(new fProxyDenseOperator(in A), in b, ref x, in d, solveInfo.iterations, solveInfo.status, ref r, ref s);
        }

        /// <summary>CGLS + Jacobi (dense), default maxIterations (A.N_Cols) / tolerance (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo cglsJacobi(in fProxyMxN A, in fProxyN b, ref fProxyN x)
            => cglsJacobi(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);

        /// <summary>CGLS with an AᵀA-Jacobi preconditioner over a BSR matrix (materializes Aᵀ once).</summary>
        public static LstsqInfo cglsJacobi(in fProxyBSM A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance)
        {
            int m = A.M_Rows, n = A.N_Cols;
            fProxyN d = b.tempfProxyVec(n), d2 = b.tempfProxyVec(n), scratch = b.tempfProxyVec(n);
            Sparse_OP.columnNormsSquared(in A, ref d2);
            Linear_OP.buildJacobiScale(in d2, ref d);
            fProxyBSM AT = b.fProxyBSMTranspose(in A);
            var op = new fProxyColScaledOperator<fProxyBSMOperator>(new fProxyBSMOperator(in A, in AT), d, scratch);

            for (int j = 0; j < n; j++) x[j] = (fProxy)0;
            fProxyN r = b.tempfProxyVec(m), s = b.tempfProxyVec(n), p = b.tempfProxyVec(n), q = b.tempfProxyVec(m);
            var solveInfo = cgls(op, in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance);
            return JacobiFinish(new fProxyBSMOperator(in A, in AT), in b, ref x, in d, solveInfo.iterations, solveInfo.status, ref r, ref s);
        }

        /// <summary>CGLS + Jacobi (BSR), default maxIterations (A.N_Cols) / tolerance (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo cglsJacobi(in fProxyBSM A, in fProxyN b, ref fProxyN x)
            => cglsJacobi(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);

        // ---- LSQR + Jacobi ----
        /// <summary>LSQR with an AᵀA-Jacobi column-equilibration preconditioner over a dense matrix.</summary>
        public static LstsqInfo lsqrJacobi(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance)
        {
            int m = A.M_Rows, n = A.N_Cols;
            fProxyN d = b.tempfProxyVec(n), d2 = b.tempfProxyVec(n), scratch = b.tempfProxyVec(n);
            Linear_OP.columnNormsSquared(in A, ref d2);
            Linear_OP.buildJacobiScale(in d2, ref d);
            var op = new fProxyColScaledOperator<fProxyDenseOperator>(new fProxyDenseOperator(in A), d, scratch);

            for (int j = 0; j < n; j++) x[j] = (fProxy)0;
            fProxyN u = b.tempfProxyVec(m), v = b.tempfProxyVec(n), w = b.tempfProxyVec(n), tmpM = b.tempfProxyVec(m), tmpN = b.tempfProxyVec(n);
            var solveInfo = lsqr(op, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance);
            return JacobiFinish(new fProxyDenseOperator(in A), in b, ref x, in d, solveInfo.iterations, solveInfo.status, ref u, ref v);
        }

        /// <summary>LSQR + Jacobi (dense), default maxIterations (A.N_Cols) / tolerance (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo lsqrJacobi(in fProxyMxN A, in fProxyN b, ref fProxyN x)
            => lsqrJacobi(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);

        /// <summary>LSQR with an AᵀA-Jacobi preconditioner over a BSR matrix (materializes Aᵀ once).</summary>
        public static LstsqInfo lsqrJacobi(in fProxyBSM A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance)
        {
            int m = A.M_Rows, n = A.N_Cols;
            fProxyN d = b.tempfProxyVec(n), d2 = b.tempfProxyVec(n), scratch = b.tempfProxyVec(n);
            Sparse_OP.columnNormsSquared(in A, ref d2);
            Linear_OP.buildJacobiScale(in d2, ref d);
            fProxyBSM AT = b.fProxyBSMTranspose(in A);
            var op = new fProxyColScaledOperator<fProxyBSMOperator>(new fProxyBSMOperator(in A, in AT), d, scratch);

            for (int j = 0; j < n; j++) x[j] = (fProxy)0;
            fProxyN u = b.tempfProxyVec(m), v = b.tempfProxyVec(n), w = b.tempfProxyVec(n), tmpM = b.tempfProxyVec(m), tmpN = b.tempfProxyVec(n);
            var solveInfo = lsqr(op, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance);
            return JacobiFinish(new fProxyBSMOperator(in A, in AT), in b, ref x, in d, solveInfo.iterations, solveInfo.status, ref u, ref v);
        }

        /// <summary>LSQR + Jacobi (BSR), default maxIterations (A.N_Cols) / tolerance (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo lsqrJacobi(in fProxyBSM A, in fProxyN b, ref fProxyN x)
            => lsqrJacobi(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);

        // ---- LSMR + Jacobi ----
        /// <summary>LSMR with an AᵀA-Jacobi column-equilibration preconditioner over a dense matrix.</summary>
        public static LstsqInfo lsmrJacobi(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance)
        {
            int m = A.M_Rows, n = A.N_Cols;
            fProxyN d = b.tempfProxyVec(n), d2 = b.tempfProxyVec(n), scratch = b.tempfProxyVec(n);
            Linear_OP.columnNormsSquared(in A, ref d2);
            Linear_OP.buildJacobiScale(in d2, ref d);
            var op = new fProxyColScaledOperator<fProxyDenseOperator>(new fProxyDenseOperator(in A), d, scratch);

            for (int j = 0; j < n; j++) x[j] = (fProxy)0;
            fProxyN u = b.tempfProxyVec(m), v = b.tempfProxyVec(n), h = b.tempfProxyVec(n), hbar = b.tempfProxyVec(n), tmpM = b.tempfProxyVec(m), tmpN = b.tempfProxyVec(n);
            var solveInfo = lsmr(op, in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance);
            return JacobiFinish(new fProxyDenseOperator(in A), in b, ref x, in d, solveInfo.iterations, solveInfo.status, ref u, ref v);
        }

        /// <summary>LSMR + Jacobi (dense), default maxIterations (A.N_Cols) / tolerance (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo lsmrJacobi(in fProxyMxN A, in fProxyN b, ref fProxyN x)
            => lsmrJacobi(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);

        /// <summary>LSMR with an AᵀA-Jacobi preconditioner over a BSR matrix (materializes Aᵀ once).</summary>
        public static LstsqInfo lsmrJacobi(in fProxyBSM A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance)
        {
            int m = A.M_Rows, n = A.N_Cols;
            fProxyN d = b.tempfProxyVec(n), d2 = b.tempfProxyVec(n), scratch = b.tempfProxyVec(n);
            Sparse_OP.columnNormsSquared(in A, ref d2);
            Linear_OP.buildJacobiScale(in d2, ref d);
            fProxyBSM AT = b.fProxyBSMTranspose(in A);
            var op = new fProxyColScaledOperator<fProxyBSMOperator>(new fProxyBSMOperator(in A, in AT), d, scratch);

            for (int j = 0; j < n; j++) x[j] = (fProxy)0;
            fProxyN u = b.tempfProxyVec(m), v = b.tempfProxyVec(n), h = b.tempfProxyVec(n), hbar = b.tempfProxyVec(n), tmpM = b.tempfProxyVec(m), tmpN = b.tempfProxyVec(n);
            var solveInfo = lsmr(op, in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance);
            return JacobiFinish(new fProxyBSMOperator(in A, in AT), in b, ref x, in d, solveInfo.iterations, solveInfo.status, ref u, ref v);
        }

        /// <summary>LSMR + Jacobi (BSR), default maxIterations (A.N_Cols) / tolerance (Consts.fProxySqrtEps).</summary>
        public static LstsqInfo lsmrJacobi(in fProxyBSM A, in fProxyN b, ref fProxyN x)
            => lsmrJacobi(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);

        /// <summary>
        /// Zero-alloc CGNE / Craig's method (Saad Alg. 8.5) for CONSISTENT systems: finds the
        /// MINIMUM-NORM solution of A x = b (requires b in range(A)) for possibly rectangular
        /// (typically UNDER-determined, m &lt; n) A, generic over <see cref="IfProxyLinearOperator"/>.
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
        /// to maxIterations and reports MaxIterations; use cgls/lsqr/lsmr for least-squares instead.
        ///
        /// Returns an <see cref="SolveInfo"/> (rnorm = ‖b-Ax‖, iterations, status; implicit-
        /// bool == Converged). Breakdown when ‖p‖² &lt;= 0 (Aᵀr = 0 while r is still above
        /// tolerance): for a CONSISTENT system r lies in range(A), so Aᵀr = 0 forces r = 0 in exact
        /// arithmetic -- a breakdown here therefore means the iteration reached the exact solution
        /// (to floating-point precision) or the system is inconsistent (r has stalled orthogonal to
        /// range(A) at the least-squares residual). On a non-Converged return x is undefined --
        /// only read x when Solved.
        /// </summary>
        public static SolveInfo cgne<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                                     ref fProxyN r, ref fProxyN p, ref fProxyN q, ref fProxyN tmpN,
                                     int maxIterations, fProxy tolerance)
            where TOp : struct, IfProxyLinearOperator
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
            fProxy bb = Linear_OP.dot(b, b);

            if (bb == (fProxy)0)
            {
                // b == 0 -> the unique minimum-norm solution of A x = 0 is x = 0 (any warm start
                // in x is discarded: x = 0 is the exact answer, matching cg's bb==0 shortcut).
                for (int i = 0; i < x.N; i++) x[i] = (fProxy)0;
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }

            fProxy threshold = tolerance * tolerance * bb;

            // r = b - A x
            A.Apply(in x, ref q);                          // q = A x (temp use of q)
            r.Data.CopyFrom(b.Data);
            r.addScaledInpl((fProxy)(-1), q);

            fProxy rr = Linear_OP.dot(r, r);

            if (rr <= threshold)
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, math.sqrt(rr));

            // p = A^T r
            A.ApplyT(in r, ref p);

            for (int k = 0; k < maxIterations; k++)
            {
                fProxy pp = Linear_OP.dot(p, p);

                if (!(pp > (fProxy)0))                      // NaN-safe: A^T r == 0 (r ⟂ range(A)) or p == 0
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr));

                fProxy alpha = rr / pp;

                x.addScaledInpl(alpha, p);                  // x += alpha p
                A.Apply(in p, ref q);                       // q = A p
                r.addScaledInpl(-alpha, q);                 // r -= alpha A p

                fProxy rrNew = Linear_OP.dot(r, r);

                if (rrNew <= threshold)
                    return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(rrNew));

                fProxy beta = rrNew / rr;

                A.ApplyT(in r, ref tmpN);                   // tmpN = A^T r
                p.scaleAddInpl(beta, tmpN);                 // p = beta p + A^T r

                rr = rrNew;
            }

            return MakeSolveInfo(IterativeSolveStatus.MaxIterations, maxIterations, math.sqrt(rr));
        }

        /// <summary>
        /// CGNE / Craig over a dense <see cref="fProxyMxN"/> (possibly rectangular) -- zero-alloc
        /// primitive. Forwards into <see cref="cgne{TOp}"/> via <see cref="fProxyDenseOperator"/>.
        /// </summary>
        public static SolveInfo cgne(in fProxyMxN A, in fProxyN b, ref fProxyN x,
                                ref fProxyN r, ref fProxyN p, ref fProxyN q, ref fProxyN tmpN,
                                int maxIterations, fProxy tolerance)
        {
            return cgne(new fProxyDenseOperator(in A), in b, ref x, ref r, ref p, ref q, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>CGNE over a dense matrix -- allocates four scratch vectors from the arena.</summary>
        public static SolveInfo cgne(in fProxyMxN A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance)
        {
            fProxyN r    = b.tempfProxyVec(A.M_Rows);
            fProxyN p    = b.tempfProxyVec(A.N_Cols);
            fProxyN q    = b.tempfProxyVec(A.M_Rows);
            fProxyN tmpN = b.tempfProxyVec(A.N_Cols);
            return cgne(in A, in b, ref x, ref r, ref p, ref q, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>CGNE over a dense matrix with default maxIterations (A.N_Cols) and tolerance (Consts.fProxySqrtEps).</summary>
        public static SolveInfo cgne(in fProxyMxN A, in fProxyN b, ref fProxyN x)
        {
            return cgne(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// CGNE / Craig over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc
        /// primitive. Forwards into <see cref="cgne{TOp}"/> via <c>fProxyBSMOperator</c>. Matrix-
        /// free minimum-norm solve over a sparse Jacobian-like operator, never forming A Aᵀ.
        /// </summary>
        public static SolveInfo cgne(in fProxyBSM A, in fProxyN b, ref fProxyN x,
                                ref fProxyN r, ref fProxyN p, ref fProxyN q, ref fProxyN tmpN,
                                int maxIterations, fProxy tolerance)
        {
            return cgne(new fProxyBSMOperator(in A), in b, ref x, ref r, ref p, ref q, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// CGNE / Craig over a BSR matrix with a CALLER-PROVIDED precomputed transpose AT (built
        /// once via <c>arena.fProxyBSMTranspose(in A)</c>), routing every ApplyT through the
        /// cache-friendly forward spMV(AT, x) instead of on-the-fly spMVT(A, x) -- see
        /// <see cref="fProxyBSMOperator"/>'s two-arg ctor. Zero-alloc; caller owns AT.
        /// </summary>
        public static SolveInfo cgne(in fProxyBSM A, in fProxyBSM AT, in fProxyN b, ref fProxyN x,
                                ref fProxyN r, ref fProxyN p, ref fProxyN q, ref fProxyN tmpN,
                                int maxIterations, fProxy tolerance)
        {
            return cgne(new fProxyBSMOperator(in A, in AT), in b, ref x, ref r, ref p, ref q, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// CGNE over a BSR matrix -- allocates four scratch vectors AND materializes A^T once via
        /// <c>arena.fProxyBSMTranspose</c>, driving CGNE with the two-arg
        /// <see cref="fProxyBSMOperator"/> so every ApplyT routes through a cache-friendly forward
        /// spMV(A^T, x). For a build-free zero-alloc path, build A^T yourself once and call the
        /// caller-AT overload above.
        /// </summary>
        public static SolveInfo cgne(in fProxyBSM A, in fProxyN b, ref fProxyN x, int maxIterations, fProxy tolerance)
        {
            fProxyN r    = b.tempfProxyVec(A.M_Rows);
            fProxyN p    = b.tempfProxyVec(A.N_Cols);
            fProxyN q    = b.tempfProxyVec(A.M_Rows);
            fProxyN tmpN = b.tempfProxyVec(A.N_Cols);
            fProxyBSM AT = b.fProxyBSMTranspose(in A);
            return cgne(new fProxyBSMOperator(in A, in AT), in b, ref x, ref r, ref p, ref q, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>CGNE over a BSR matrix with default maxIterations (A.N_Cols) and tolerance (Consts.fProxySqrtEps).</summary>
        public static SolveInfo cgne(in fProxyBSM A, in fProxyN b, ref fProxyN x)
        {
            return cgne(in A, in b, ref x, A.N_Cols, Consts.fProxySqrtEps);
        }
    }

}
