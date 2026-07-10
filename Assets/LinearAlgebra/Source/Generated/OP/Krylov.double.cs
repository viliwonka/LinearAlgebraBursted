#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using Unity.Collections;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov {

        // Shared factory for the square-solver diagnostics struct (cg/pcg/minres/biCGStab/cgne).
        // rnorm is ALWAYS a value the solver already holds -- a tracked residual norm, or a single
        // dot on its live residual r -- never a fresh A*x, honoring the "free diagnostics" contract.
        static SolveInfo MakeSolveInfo(IterativeSolveStatus status, int iterations, double rnorm)
            => new SolveInfo { rnorm = rnorm, iterations = iterations, status = status };

        /// <summary>
        /// Zero-alloc Conjugate Gradient solver for symmetric positive-definite (SPD) systems A x = b,
        /// generic over any <see cref="IdoubleLinearOperator"/>. This is the single implementation of
        /// the CG loop — the concrete dense (<c>cg(in doubleMxN, ...)</c>) and BSR
        /// (<c>cg(in doubleBSR, ...)</c>) overloads below are thin forwarders that
        /// wrap their matrix in <see cref="doubleDenseOperator"/> / <c>doubleBSROperator</c> and
        /// call this method.
        ///
        /// Caller provides x (initial guess, overwritten with solution — WARM-STARTABLE: seed x
        /// with a previous solution to resume/refine) and three scratch vectors r, p, Ap (all
        /// length A.Rows). Returns a <see cref="SolveInfo"/> (rnorm = ‖b-Ax‖) — see that struct
        /// for the implicit-bool/status/undefined-x contract. Breakdown on non-positive curvature
        /// p·Ap &lt;= 0 (A not SPD or numerical breakdown).
        /// </summary>
        public static SolveInfo cg<TOp>(in TOp A, in doubleN b, ref doubleN x,
                                   ref doubleN r, ref doubleN p, ref doubleN Ap,
                                   int maxIterations, double tolerance)
            where TOp : struct, IdoubleLinearOperator
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
            // (addScaledInPlace/scaleAddInPlace) with reads of "old" values, and those primitives do
            // NOT self-check aliasing the way A.Apply's own dot/spMV call does. E.g. r aliasing
            // Ap turns `r.addScaledInPlace(-1, Ap)` (r -= Ap) into r -= r == 0 elementwise -- a
            // silent false convergence instead of a thrown exception. Check every pair up front.
            unsafe
            {
                double* rPtr = r.Data.Ptr, pPtr = p.Data.Ptr, ApPtr = Ap.Data.Ptr, xPtr = x.Data.Ptr, bPtr = b.Data.Ptr;

                if (rPtr == pPtr || rPtr == ApPtr || rPtr == xPtr || rPtr == bPtr ||
                    pPtr == ApPtr || pPtr == xPtr || pPtr == bPtr ||
                    ApPtr == xPtr || ApPtr == bPtr ||
                    xPtr == bPtr)
                    throw new ArgumentException("cg: r/p/Ap/x/b must be distinct");
            }

            double bb = Blas.dot(b, b);

            // b is the zero vector — x = 0 is the exact solution. Copy b (all zeros)
            // rather than multiplying by 0, so a NaN/Inf initial guess is sanitized
            // (NaN * 0 = NaN would otherwise leak through).
            if (bb == (double)0)
            {
                x.Data.CopyFrom(b.Data);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (double)0);
            }

            // r = b - A x
            A.Apply(in x, ref Ap);                       // Ap = A x (temp use of Ap)
            r.Data.CopyFrom(b.Data);                     // r  = b
            r.addScaledInPlace((double)(-1), Ap);           // r -= Ap  =>  r = b - A x

            // p = r
            p.Data.CopyFrom(r.Data);

            double rsold = Blas.dot(r, r);
            double threshold = tolerance * tolerance * bb;

            if (rsold <= threshold)
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, math.sqrt(rsold));

            for (int k = 0; k < maxIterations; k++)
            {
                // Ap = A p ; pAp = dot(p, Ap) -- Krylov R2's ApplyDot: one call site instead of
                // two (see IdoubleLinearOperator.ApplyDot's doc comment for why every operator
                // composes rather than fuses -- a fused version was tried and measured slower).
                double pAp = A.ApplyDot(in p, ref Ap);

                if (!(pAp > (double)0))                  // NaN-safe: also catches breakdown
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rsold));

                double alpha = rsold / pAp;

                // x += alpha p ; r -= alpha Ap ; rsnew = ||r||^2 folded into the r-update pass
                // (Blas.updateXR), eliminating the separate Blas.dot(r,r) traversal.
                double rsnew = Blas.updateXR(alpha, p, ref x, Ap, ref r);

                if (rsnew <= threshold)
                    return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(rsnew));

                double beta = rsnew / rsold;

                p.scaleAddInPlace(beta, r);                 // p = beta p + r

                rsold = rsnew;
            }

            return MakeSolveInfo(IterativeSolveStatus.MaxIterations, maxIterations, math.sqrt(rsold));
        }

        /// <summary>
        /// CG over a dense <see cref="doubleMxN"/> -- zero-alloc primitive. Forwards into
        /// <see cref="cg{TOp}"/> via <see cref="doubleDenseOperator"/>. See that method for the
        /// actual loop and buffer semantics.
        /// </summary>
        public static SolveInfo cg(in doubleMxN A, in doubleN b, ref doubleN x,
                                             ref doubleN r, ref doubleN p, ref doubleN Ap,
                                             int maxIterations, double tolerance)
        {
            return cg(new doubleDenseOperator(in A), in b, ref x, ref r, ref p, ref Ap, maxIterations, tolerance);
        }

        /// <summary>
        /// Conjugate Gradient solver — allocates three scratch vectors from the arena and calls
        /// the zero-alloc primitive. x is overwritten with the solution on convergence.
        /// </summary>
        public static SolveInfo cg(in doubleMxN A, in doubleN b, ref doubleN x,
                                             int maxIterations, double tolerance)
        {
            doubleN r  = b.doubleTempVec(A.M_Rows);
            doubleN p  = b.doubleTempVec(A.M_Rows);
            doubleN Ap = b.doubleTempVec(A.M_Rows);
            return cg(in A, in b, ref x, ref r, ref p, ref Ap, maxIterations, tolerance);
        }

        /// <summary>
        /// Conjugate Gradient solver with default maxIterations (A.M_Rows) and tolerance
        /// (Consts.doubleSqrtEps). x is overwritten with the solution on convergence.
        /// </summary>
        public static SolveInfo cg(in doubleMxN A, in doubleN b, ref doubleN x)
        {
            return cg(in A, in b, ref x, A.M_Rows, Consts.doubleSqrtEps);
        }

        /// <summary>
        /// Conjugate Gradient solver over a block-sparse (BSR) SPD matrix. Same semantics as
        /// the dense overload — see <see cref="cg(in doubleMxN, in doubleN, ref doubleN, ref doubleN, ref doubleN, ref doubleN, int, double)"/>.
        /// Forwards into <see cref="cg{TOp}"/> via <c>doubleBSROperator</c>.
        /// </summary>
        public static SolveInfo cg(in doubleBSR A, in doubleN b, ref doubleN x,
                                             ref doubleN r, ref doubleN p, ref doubleN Ap,
                                             int maxIterations, double tolerance)
        {
            return cg(new doubleBSROperator(in A), in b, ref x, ref r, ref p, ref Ap, maxIterations, tolerance);
        }

        /// <summary>
        /// Conjugate Gradient solver over a block-sparse (BSR) SPD matrix — allocates three
        /// scratch vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo cg(in doubleBSR A, in doubleN b, ref doubleN x,
                                             int maxIterations, double tolerance)
        {
            doubleN r  = b.doubleTempVec(A.M_Rows);
            doubleN p  = b.doubleTempVec(A.M_Rows);
            doubleN Ap = b.doubleTempVec(A.M_Rows);
            return cg(in A, in b, ref x, ref r, ref p, ref Ap, maxIterations, tolerance);
        }

        /// <summary>
        /// Conjugate Gradient solver over a block-sparse (BSR) SPD matrix, with default
        /// maxIterations (A.M_Rows) and tolerance (Consts.doubleSqrtEps).
        /// </summary>
        public static SolveInfo cg(in doubleBSR A, in doubleN b, ref doubleN x)
        {
            return cg(in A, in b, ref x, A.M_Rows, Consts.doubleSqrtEps);
        }

        /// <summary>
        /// Zero-alloc Preconditioned Conjugate Gradient solver for SPD systems A x = b, generic
        /// over both the operator (<see cref="IdoubleLinearOperator"/>) and the preconditioner
        /// (<see cref="IdoublePreconditioner"/>). Standard PCG: p is combined with z = M⁻¹r (not r), and β uses
        /// ⟨r,z⟩ instead of ⟨r,r⟩.
        ///
        /// Caller provides x (initial guess, overwritten with solution — warm-startable) and four
        /// scratch vectors r, p, Ap, z (all length A.Rows). The convergence test compares the
        /// TRUE (unpreconditioned) residual ||r||² against tolerance²·||b||² — the same criterion
        /// as <see cref="cg{TOp}"/> — so iteration counts between cg and pcg on the same system are
        /// directly comparable. Returns a <see cref="SolveInfo"/> — see that struct for the
        /// implicit-bool/status/undefined-x contract. Breakdown on non-positive curvature
        /// p·Ap &lt;= 0 (or a non-SPD preconditioner's non-positive ⟨r,z⟩).
        /// </summary>
        public static SolveInfo pcg<TOp, TPre>(in TOp A, in TPre M, in doubleN b, ref doubleN x,
                                          ref doubleN r, ref doubleN p, ref doubleN Ap, ref doubleN z,
                                          int maxIterations, double tolerance)
            where TOp : struct, IdoubleLinearOperator
            where TPre : struct, IdoublePreconditioner
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
                double* rPtr = r.Data.Ptr, pPtr = p.Data.Ptr, ApPtr = Ap.Data.Ptr, zPtr = z.Data.Ptr, xPtr = x.Data.Ptr, bPtr = b.Data.Ptr;

                if (rPtr == pPtr || rPtr == ApPtr || rPtr == zPtr || rPtr == xPtr || rPtr == bPtr ||
                    pPtr == ApPtr || pPtr == zPtr || pPtr == xPtr || pPtr == bPtr ||
                    ApPtr == zPtr || ApPtr == xPtr || ApPtr == bPtr ||
                    zPtr == xPtr || zPtr == bPtr ||
                    xPtr == bPtr)
                    throw new ArgumentException("pcg: r/p/Ap/z/x/b must be distinct");
            }

            double bb = Blas.dot(b, b);

            if (bb == (double)0)
            {
                x.Data.CopyFrom(b.Data);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (double)0);
            }

            // r = b - A x
            A.Apply(in x, ref Ap);
            r.Data.CopyFrom(b.Data);
            r.addScaledInPlace((double)(-1), Ap);

            double threshold = tolerance * tolerance * bb;

            // rr tracks ‖r‖² of the CURRENT residual across the whole solve -- it is exactly the
            // quantity the convergence test already needs, so reporting rnorm = √rr is free.
            double rr = Blas.dot(r, r);
            if (rr <= threshold)
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, math.sqrt(rr));

            // z = M^-1 r ; p = z
            M.Apply(in r, ref z);
            p.Data.CopyFrom(z.Data);

            double rzold = Blas.dot(r, z);

            // Block-Jacobi is SPD so this never trips on the shipped path, but a user-supplied
            // preconditioner is not guaranteed SPD; a non-positive <r,z> yields a wrong-signed
            // alpha/beta and silent divergence instead of a clean bailout. Mirrors cg's
            // NaN-safe !(pAp > 0) breakdown guard.
            if (!(rzold > (double)0))
                return MakeSolveInfo(IterativeSolveStatus.Breakdown, 0, math.sqrt(rr));

            for (int k = 0; k < maxIterations; k++)
            {
                // Ap = A p ; pAp = dot(p, Ap) -- Krylov R2's ApplyDot: one call site instead of
                // two (see IdoubleLinearOperator.ApplyDot's doc comment for why every operator
                // composes rather than fuses -- a fused version was tried and measured slower).
                double pAp = A.ApplyDot(in p, ref Ap);

                if (!(pAp > (double)0))                  // NaN-safe: also catches breakdown
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr));

                double alpha = rzold / pAp;

                // x += alpha p ; r -= alpha Ap ; rr = ||r||^2 folded into the r-update pass
                // (Blas.updateXR), eliminating the separate Blas.dot(r,r) traversal.
                rr = Blas.updateXR(alpha, p, ref x, Ap, ref r);
                if (rr <= threshold)
                    return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(rr));

                M.Apply(in r, ref z);                     // z = M^-1 r

                double rznew = Blas.dot(r, z);

                if (!(rznew > (double)0))                 // NaN-safe: same breakdown guard, fresh <r,z>
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k + 1, math.sqrt(rr));

                double beta = rznew / rzold;

                p.scaleAddInPlace(beta, z);                 // p = beta p + z

                rzold = rznew;
            }

            return MakeSolveInfo(IterativeSolveStatus.MaxIterations, maxIterations, math.sqrt(rr));
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient solver — allocates four scratch vectors from the
        /// arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo pcg<TOp, TPre>(in TOp A, in TPre M, in doubleN b, ref doubleN x,
                                          int maxIterations, double tolerance)
            where TOp : struct, IdoubleLinearOperator
            where TPre : struct, IdoublePreconditioner
        {
            doubleN r  = b.doubleTempVec(A.Rows);
            doubleN p  = b.doubleTempVec(A.Rows);
            doubleN Ap = b.doubleTempVec(A.Rows);
            doubleN z  = b.doubleTempVec(A.Rows);
            return pcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIterations, tolerance);
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient solver with default maxIterations (A.Rows) and
        /// tolerance (Consts.doubleSqrtEps).
        /// </summary>
        public static SolveInfo pcg<TOp, TPre>(in TOp A, in TPre M, in doubleN b, ref doubleN x)
            where TOp : struct, IdoubleLinearOperator
            where TPre : struct, IdoublePreconditioner
        {
            return pcg(in A, in M, in b, ref x, A.Rows, Consts.doubleSqrtEps);
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient over a block-sparse (BSR) SPD matrix with its
        /// matching block-Jacobi preconditioner. Forwards into <see cref="pcg{TOp,TPre}"/> via
        /// <c>doubleBSROperator</c>.
        /// </summary>
        public static SolveInfo pcg(in doubleBSR A, in doubleBlockJacobi M, in doubleN b, ref doubleN x,
                               ref doubleN r, ref doubleN p, ref doubleN Ap, ref doubleN z,
                               int maxIterations, double tolerance)
        {
            return pcg(new doubleBSROperator(in A), in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIterations, tolerance);
        }

        /// <summary>
        /// Block-Jacobi Preconditioned Conjugate Gradient over a BSR SPD matrix — allocates four
        /// scratch vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo pcg(in doubleBSR A, in doubleBlockJacobi M, in doubleN b, ref doubleN x,
                               int maxIterations, double tolerance)
        {
            doubleN r  = b.doubleTempVec(A.M_Rows);
            doubleN p  = b.doubleTempVec(A.M_Rows);
            doubleN Ap = b.doubleTempVec(A.M_Rows);
            doubleN z  = b.doubleTempVec(A.M_Rows);
            return pcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIterations, tolerance);
        }

        /// <summary>
        /// Block-Jacobi Preconditioned Conjugate Gradient over a BSR SPD matrix, with default
        /// maxIterations (A.M_Rows) and tolerance (Consts.doubleSqrtEps).
        /// </summary>
        public static SolveInfo pcg(in doubleBSR A, in doubleBlockJacobi M, in doubleN b, ref doubleN x)
        {
            return pcg(in A, in M, in b, ref x, A.M_Rows, Consts.doubleSqrtEps);
        }

        /// <summary>
        /// Preconditioned Conjugate Gradient over a block-sparse (BSR) SPD matrix with its
        /// matching SSOR preconditioner (Krylov R3, docs/draft-spec-krylov-optimization.md).
        /// Forwards into <see cref="pcg{TOp,TPre}"/> via <c>doubleBSROperator</c> -- same
        /// three-rung BSR convenience pattern as the block-Jacobi overloads above.
        /// </summary>
        public static SolveInfo pcg(in doubleBSR A, in doubleSSOR M, in doubleN b, ref doubleN x,
                               ref doubleN r, ref doubleN p, ref doubleN Ap, ref doubleN z,
                               int maxIterations, double tolerance)
        {
            return pcg(new doubleBSROperator(in A), in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIterations, tolerance);
        }

        /// <summary>
        /// SSOR Preconditioned Conjugate Gradient over a BSR SPD matrix -- allocates four scratch
        /// vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo pcg(in doubleBSR A, in doubleSSOR M, in doubleN b, ref doubleN x,
                               int maxIterations, double tolerance)
        {
            doubleN r  = b.doubleTempVec(A.M_Rows);
            doubleN p  = b.doubleTempVec(A.M_Rows);
            doubleN Ap = b.doubleTempVec(A.M_Rows);
            doubleN z  = b.doubleTempVec(A.M_Rows);
            return pcg(in A, in M, in b, ref x, ref r, ref p, ref Ap, ref z, maxIterations, tolerance);
        }

        /// <summary>
        /// SSOR Preconditioned Conjugate Gradient over a BSR SPD matrix, with default
        /// maxIterations (A.M_Rows) and tolerance (Consts.doubleSqrtEps).
        /// </summary>
        public static SolveInfo pcg(in doubleBSR A, in doubleSSOR M, in doubleN b, ref doubleN x)
        {
            return pcg(in A, in M, in b, ref x, A.M_Rows, Consts.doubleSqrtEps);
        }

        // Phase 3: MINRES (symmetric indefinite), BiCGSTAB (non-symmetric), CGLS/LSQR
        // (rectangular least-squares). Same generic-operator pattern as cg&lt;TOp&gt;/
        // pcg&lt;TOp,TPre&gt; above -- see cg&lt;TOp&gt;'s doc comment for the shared "why an
        // up-front aliasing guard" rationale. These four solvers carry more scratch vectors than
        // cg/pcg (6-9 vs 3-4), so their guards use RequireDistinctBuffers (a small loop-based
        // helper) instead of a hand-expanded OR chain -- see that helper's doc comment.

        /// <summary>
        /// Zero-alloc MINRES (Paige-Saunders) solver for symmetric systems A x = b, generic over
        /// <see cref="IdoubleLinearOperator"/>.
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
        /// Returns a <see cref="SolveInfo"/> (rnorm = phibar = ‖b-Ax‖) — see that struct for the
        /// implicit-bool/status/undefined-x contract. Breakdown if the Lanczos recurrence exactly
        /// exhausts the Krylov subspace short of tolerance (beta==0, an exact-arithmetic
        /// invariant-subspace breakdown).
        /// </summary>
        public static SolveInfo minres<TOp>(in TOp A, in doubleN b, ref doubleN x,
                                       ref doubleN y, ref doubleN r1, ref doubleN r2, ref doubleN v,
                                       ref doubleN w, ref doubleN w1, ref doubleN w2,
                                       int maxIterations, double tolerance)
            where TOp : struct, IdoubleLinearOperator
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

            double bb = Blas.dot(b, b);

            if (bb == (double)0)
            {
                x.Data.CopyFrom(b.Data);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (double)0);
            }

            // r1 = b - A x
            A.Apply(in x, ref y);                       // y = A x (temp use of y)
            r1.Data.CopyFrom(b.Data);
            r1.addScaledInPlace((double)(-1), y);           // r1 = b - A x

            double beta1 = math.sqrt(Blas.dot(r1, r1));
            double threshold = tolerance * tolerance * bb;

            if (beta1 * beta1 <= threshold)
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, beta1);

            r2.Data.CopyFrom(r1.Data);

            // Zero the 3-term search-direction history (w/w1/w2 start at 0 in exact MINRES).
            for (int i = 0; i < A.Rows; i++) { w[i] = (double)0; w1[i] = (double)0; w2[i] = (double)0; }

            double oldb = (double)0;
            double beta = beta1;
            double dbar = (double)0;
            double epsln = (double)0;
            double phibar = beta1;
            double cs = (double)(-1);
            double sn = (double)0;
            double gammaFloor = Consts.doubleEpsilon;

            for (int k = 0; k < maxIterations; k++)
            {
                // ---- Lanczos step: extend the tridiagonalization by one vector ----
                // v = r2 / beta, one pass (Blas.scaledCopy with a = 1/beta) -- rounding-only vs the
                // original CopyFrom+divInPlace (reciprocal-multiply instead of a per-element divide).
                Blas.scaledCopy(1 / beta, r2, ref v);

                A.Apply(in v, ref y);                      // y = A v

                if (k >= 1)
                    y.addScaledInPlace(-(beta / oldb), r1);   // y -= (beta/oldb) r1

                double alfa = Blas.dot(v, y);
                y.addScaledInPlace(-(alfa / beta), r2);       // y -= (alfa/beta) r2

                // Buffer rotation (r1,r2,y) -> (r2,y,r1): swap the local doubleN handles instead of
                // Data.CopyFrom. r1's old buffer is fully consumed above (last read this iteration)
                // and is recycled as next iteration's y, which A.Apply fully overwrites regardless of
                // its incoming contents -- contract-clean (see spec's buffer-rotation rationale).
                { doubleN tmp = r1; r1 = r2; r2 = y; y = tmp; }

                oldb = beta;
                beta = math.sqrt(Blas.dot(r2, r2));

                // ---- apply the PREVIOUS Givens rotation (cs,sn) to the new tridiagonal column ----
                double oldeps = epsln;
                double delta = cs * dbar + sn * alfa;
                double gbar = sn * dbar - cs * alfa;
                epsln = sn * beta;
                dbar = -cs * beta;

                // ---- compute the NEW Givens rotation that zeros the subdiagonal entry ----
                double gamma = math.sqrt(gbar * gbar + beta * beta);
                gamma = math.max(gamma, gammaFloor);
                cs = gbar / gamma;
                sn = beta / gamma;
                double phi = cs * phibar;
                phibar = sn * phibar;

                // ---- update the 3-term search direction, then the solution ----
                // Buffer rotation (w1,w2,w) -> (w2,w,w1), mirroring the r-rotation above.
                { doubleN tmp = w1; w1 = w2; w2 = w; w = tmp; }

                // w = (v - oldeps*w1 - delta*w2) / gamma, one pass (Blas.combine3 with s = 1/gamma) --
                // rounding-only vs the original copy+axpy+axpy+divInPlace chain (reciprocal-multiply
                // instead of a per-element divide at the end; the (v + a*w1 + b*w2) grouping matches).
                Blas.combine3(ref w, v, -oldeps, w1, -delta, w2, 1 / gamma);

                x.addScaledInPlace(phi, w);

                // phibar IS the true residual norm ‖b-Ax‖ at this step (MINRES identity) --
                // no extra dot product needed, so rnorm = phibar is free.
                if (phibar * phibar <= threshold)
                    return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, phibar);

                if (!(beta > (double)0))
                    // Lanczos breakdown: invariant subspace exhausted, no further progress possible.
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k + 1, phibar);
            }

            return MakeSolveInfo(IterativeSolveStatus.MaxIterations, maxIterations, phibar);
        }

        /// <summary>
        /// MINRES over a dense <see cref="doubleMxN"/> -- zero-alloc primitive. Forwards into
        /// <see cref="minres{TOp}"/> via <see cref="doubleDenseOperator"/>. See that method for
        /// the actual loop and buffer semantics.
        /// </summary>
        public static SolveInfo minres(in doubleMxN A, in doubleN b, ref doubleN x,
                                  ref doubleN y, ref doubleN r1, ref doubleN r2, ref doubleN v,
                                  ref doubleN w, ref doubleN w1, ref doubleN w2,
                                  int maxIterations, double tolerance)
        {
            return minres(new doubleDenseOperator(in A), in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, maxIterations, tolerance);
        }

        /// <summary>MINRES over a dense matrix -- allocates seven scratch vectors from the arena.</summary>
        public static SolveInfo minres(in doubleMxN A, in doubleN b, ref doubleN x, int maxIterations, double tolerance)
        {
            doubleN y  = b.doubleTempVec(A.M_Rows);
            doubleN r1 = b.doubleTempVec(A.M_Rows);
            doubleN r2 = b.doubleTempVec(A.M_Rows);
            doubleN v  = b.doubleTempVec(A.M_Rows);
            doubleN w  = b.doubleTempVec(A.M_Rows);
            doubleN w1 = b.doubleTempVec(A.M_Rows);
            doubleN w2 = b.doubleTempVec(A.M_Rows);
            return minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, maxIterations, tolerance);
        }

        /// <summary>MINRES over a dense matrix with default maxIterations (A.M_Rows) and tolerance (Consts.doubleSqrtEps).</summary>
        public static SolveInfo minres(in doubleMxN A, in doubleN b, ref doubleN x)
        {
            return minres(in A, in b, ref x, A.M_Rows, Consts.doubleSqrtEps);
        }

        /// <summary>
        /// MINRES over a symmetric block-sparse (BSR) matrix -- zero-alloc primitive. Forwards
        /// into <see cref="minres{TOp}"/> via <c>doubleBSROperator</c>.
        /// </summary>
        public static SolveInfo minres(in doubleBSR A, in doubleN b, ref doubleN x,
                                  ref doubleN y, ref doubleN r1, ref doubleN r2, ref doubleN v,
                                  ref doubleN w, ref doubleN w1, ref doubleN w2,
                                  int maxIterations, double tolerance)
        {
            return minres(new doubleBSROperator(in A), in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, maxIterations, tolerance);
        }

        /// <summary>MINRES over a BSR matrix -- allocates seven scratch vectors from the arena.</summary>
        public static SolveInfo minres(in doubleBSR A, in doubleN b, ref doubleN x, int maxIterations, double tolerance)
        {
            doubleN y  = b.doubleTempVec(A.M_Rows);
            doubleN r1 = b.doubleTempVec(A.M_Rows);
            doubleN r2 = b.doubleTempVec(A.M_Rows);
            doubleN v  = b.doubleTempVec(A.M_Rows);
            doubleN w  = b.doubleTempVec(A.M_Rows);
            doubleN w1 = b.doubleTempVec(A.M_Rows);
            doubleN w2 = b.doubleTempVec(A.M_Rows);
            return minres(in A, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, maxIterations, tolerance);
        }

        /// <summary>MINRES over a BSR matrix with default maxIterations (A.M_Rows) and tolerance (Consts.doubleSqrtEps).</summary>
        public static SolveInfo minres(in doubleBSR A, in doubleN b, ref doubleN x)
        {
            return minres(in A, in b, ref x, A.M_Rows, Consts.doubleSqrtEps);
        }

        /// <summary>
        /// Zero-alloc BiCGSTAB (van der Vorst 1992, stabilized Bi-Conjugate Gradient) solver for
        /// NON-symmetric (general) square systems A x = b, generic over
        /// <see cref="IdoubleLinearOperator"/>. Short two-sided recurrence, flat O(n) memory (no
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
        /// Returns a <see cref="SolveInfo"/> (rnorm = ‖b-Ax‖) — see that struct for the
        /// implicit-bool/status/undefined-x contract. Breakdown on one of the standard BiCGSTAB
        /// breakdowns (rho == 0, rHat0·v == 0, or omega == 0 -- A not amenable to BiCGSTAB from
        /// this shadow residual, or numerical breakdown).
        /// </summary>
        public static SolveInfo biCGStab<TOp>(in TOp A, in doubleN b, ref doubleN x,
                                         ref doubleN r, ref doubleN rHat0, ref doubleN p, ref doubleN v, ref doubleN t,
                                         int maxIterations, double tolerance)
            where TOp : struct, IdoubleLinearOperator
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

            double bb = Blas.dot(b, b);

            if (bb == (double)0)
            {
                x.Data.CopyFrom(b.Data);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (double)0);
            }

            // r = b - A x
            A.Apply(in x, ref v);                          // v = A x (temp use, overwritten below)
            r.Data.CopyFrom(b.Data);
            r.addScaledInPlace((double)(-1), v);

            double threshold = tolerance * tolerance * bb;

            // rr tracks ‖current residual‖²; ss the ‖half-step residual s‖². Both are already
            // computed for the convergence tests, so every exit reports rnorm from a held value.
            double rr = Blas.dot(r, r);
            if (rr <= threshold)
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, math.sqrt(rr));

            rHat0.Data.CopyFrom(r.Data);

            // p_0 = v_0 = 0 (standard BiCGSTAB init).
            for (int i = 0; i < A.Rows; i++) { p[i] = (double)0; v[i] = (double)0; }

            double rho = (double)1, alpha = (double)1, omega = (double)1;

            for (int k = 0; k < maxIterations; k++)
            {
                double rhoNew = Blas.dot(rHat0, r);

                if (rhoNew == (double)0 || math.isnan(rhoNew))
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr)); // r orthogonal to shadow residual

                double beta = (rhoNew / rho) * (alpha / omega);

                p.addScaledInPlace(-omega, v);                // p -= omega v      (still old p, old v)
                p.scaleAddInPlace(beta, r);                    // p = beta p + r

                A.Apply(in p, ref v);                       // v = A p

                double rv = Blas.dot(rHat0, v);

                if (rv == (double)0 || math.isnan(rv))
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr)); // breakdown: alpha undefined

                alpha = rhoNew / rv;

                // r := s = r - alpha v ; ss = ||s||^2, fused into one pass (Blas.axpyNormSq).
                double ss = Blas.axpyNormSq(-alpha, v, ref r);

                if (ss <= threshold)
                {
                    // Early exit: the half-step residual s is already small enough -- finish
                    // with x += alpha p (skipping the t = A s stabilization matvec entirely).
                    x.addScaledInPlace(alpha, p);
                    return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(ss));
                }

                A.Apply(in r, ref t);                       // t = A s   (r currently holds s)

                double tt = Blas.dot(t, t);

                if (!(tt > (double)0))                       // NaN-safe: tt is a norm^2, nonnegative
                    // breakdown: omega undefined. x is still x_old here (the alpha·p / omega·r
                    // updates are below), so its residual is rr -- NOT ss (ss = ‖b - A(x_old+alpha·p)‖,
                    // an iterate this path never commits to x).
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr));

                omega = Blas.dot(t, r) / tt;

                if (omega == (double)0 || math.isnan(omega))
                    // breakdown: beta would divide by zero. x is still x_old (see above) -> report rr.
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr));

                x.addScaledInPlace(alpha, p);
                x.addScaledInPlace(omega, r);                  // r still holds s here

                // r := s - omega t (new residual) ; rr = ||r||^2, fused into one pass (Blas.axpyNormSq).
                rr = Blas.axpyNormSq(-omega, t, ref r);

                if (rr <= threshold)
                    return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(rr));

                rho = rhoNew;
            }

            return MakeSolveInfo(IterativeSolveStatus.MaxIterations, maxIterations, math.sqrt(rr));
        }

        /// <summary>
        /// BiCGSTAB over a dense <see cref="doubleMxN"/> -- zero-alloc primitive. Forwards into
        /// <see cref="biCGStab{TOp}"/> via <see cref="doubleDenseOperator"/>.
        /// </summary>
        public static SolveInfo biCGStab(in doubleMxN A, in doubleN b, ref doubleN x,
                                    ref doubleN r, ref doubleN rHat0, ref doubleN p, ref doubleN v, ref doubleN t,
                                    int maxIterations, double tolerance)
        {
            return biCGStab(new doubleDenseOperator(in A), in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, maxIterations, tolerance);
        }

        /// <summary>BiCGSTAB over a dense matrix -- allocates five scratch vectors from the arena.</summary>
        public static SolveInfo biCGStab(in doubleMxN A, in doubleN b, ref doubleN x, int maxIterations, double tolerance)
        {
            doubleN r     = b.doubleTempVec(A.M_Rows);
            doubleN rHat0 = b.doubleTempVec(A.M_Rows);
            doubleN p     = b.doubleTempVec(A.M_Rows);
            doubleN v     = b.doubleTempVec(A.M_Rows);
            doubleN t     = b.doubleTempVec(A.M_Rows);
            return biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, maxIterations, tolerance);
        }

        /// <summary>BiCGSTAB over a dense matrix with default maxIterations (A.M_Rows) and tolerance (Consts.doubleSqrtEps).</summary>
        public static SolveInfo biCGStab(in doubleMxN A, in doubleN b, ref doubleN x)
        {
            return biCGStab(in A, in b, ref x, A.M_Rows, Consts.doubleSqrtEps);
        }

        /// <summary>
        /// BiCGSTAB over a block-sparse (BSR) matrix -- zero-alloc primitive. Forwards into
        /// <see cref="biCGStab{TOp}"/> via <c>doubleBSROperator</c>.
        /// </summary>
        public static SolveInfo biCGStab(in doubleBSR A, in doubleN b, ref doubleN x,
                                    ref doubleN r, ref doubleN rHat0, ref doubleN p, ref doubleN v, ref doubleN t,
                                    int maxIterations, double tolerance)
        {
            return biCGStab(new doubleBSROperator(in A), in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, maxIterations, tolerance);
        }

        /// <summary>BiCGSTAB over a BSR matrix -- allocates five scratch vectors from the arena.</summary>
        public static SolveInfo biCGStab(in doubleBSR A, in doubleN b, ref doubleN x, int maxIterations, double tolerance)
        {
            doubleN r     = b.doubleTempVec(A.M_Rows);
            doubleN rHat0 = b.doubleTempVec(A.M_Rows);
            doubleN p     = b.doubleTempVec(A.M_Rows);
            doubleN v     = b.doubleTempVec(A.M_Rows);
            doubleN t     = b.doubleTempVec(A.M_Rows);
            return biCGStab(in A, in b, ref x, ref r, ref rHat0, ref p, ref v, ref t, maxIterations, tolerance);
        }

        /// <summary>BiCGSTAB over a BSR matrix with default maxIterations (A.M_Rows) and tolerance (Consts.doubleSqrtEps).</summary>
        public static SolveInfo biCGStab(in doubleBSR A, in doubleN b, ref doubleN x)
        {
            return biCGStab(in A, in b, ref x, A.M_Rows, Consts.doubleSqrtEps);
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
        public static LstsqInfo lstsqResidual<TOp>(in TOp A, in doubleN b, in doubleN x, double damp,
                                                         ref doubleN rScratch, ref doubleN sScratch)
            where TOp : struct, IdoubleLinearOperator
        {
            // r = b - A x
            A.Apply(in x, ref rScratch);
            rScratch.scaleAddInPlace((double)(-1), b);          // rScratch = -A x + b = b - A x
            double rnorm = math.sqrt(Blas.dot(rScratch, rScratch));

            // s = Aᵀr - damp²x  (the same optimality residual cgls's loop tracks)
            A.ApplyT(in rScratch, ref sScratch);
            if (damp != (double)0) sScratch.addScaledInPlace(-(damp * damp), x);
            double arnorm = math.sqrt(Blas.dot(sScratch, sScratch));

            double xnorm = math.sqrt(Blas.dot(x, x));

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
        /// <see cref="IdoubleLinearOperator"/>. This is CG applied to the normal equations
        /// AᵀA x = Aᵀb, but NEVER explicitly forms AᵀA -- every AᵀA-vector product is one
        /// <see cref="IdoubleLinearOperator.Apply"/> plus one
        /// <see cref="IdoubleLinearOperator.ApplyT"/>. The normal-equation residual s = Aᵀr is
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
        /// Returns an <see cref="LstsqInfo"/> — see that struct for the implicit-bool/status/
        /// undefined-x contract. Breakdown on non-positive curvature ‖Ap‖²&lt;=0 (mirrors cg's
        /// p·Ap&lt;=0 guard: p is in null(A), or p==0).
        /// </summary>
        public static LstsqInfo cgls<TOp>(in TOp A, in doubleN b, ref doubleN x,
                                     ref doubleN r, ref doubleN s, ref doubleN p, ref doubleN q,
                                     int maxIterations, double tolerance, double damp)
            where TOp : struct, IdoubleLinearOperator
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
            double atbSq = Blas.dot(s, s);

            if (atbSq == (double)0)
            {
                // A^T b == 0 -> x=0 is a valid least-squares minimizer regardless of warm start
                // (mirrors cg's bb==0 shortcut: a deterministic, NaN-sanitizing exact answer).
                for (int i = 0; i < x.N; i++) x[i] = (double)0;
                // r = b, Aᵀr = Aᵀb = 0, x = 0.
                return new LstsqInfo { rnorm = math.sqrt(Blas.dot(b, b)), Arnorm = (double)0, xnorm = (double)0, iterations = 0, status = IterativeSolveStatus.Converged };
            }

            double threshold = tolerance * tolerance * atbSq;

            // r = b - A x
            A.Apply(in x, ref q);                          // q = A x (temp use of q)
            r.Data.CopyFrom(b.Data);
            r.addScaledInPlace((double)(-1), q);

            // s = A^T r - damp^2 x  (damped: the residual of the normal equations
            // (A^T A + damp^2 I) x = A^T b; damp==0 -> s = A^T r exactly, bit-identical).
            A.ApplyT(in r, ref s);
            if (damp != (double)0) s.addScaledInPlace(-(damp * damp), x);

            double gamma = Blas.dot(s, s);

            // rnorm/Arnorm/xnorm are all FREE here: r is live (one dot), Arnorm = √gamma is the
            // tracked normal-equation residual, xnorm one dot on x. No extra matvec.
            if (gamma <= threshold)
                return CglsInfo(IterativeSolveStatus.Converged, 0, gamma, in r, in x);

            p.Data.CopyFrom(s.Data);

            for (int k = 0; k < maxIterations; k++)
            {
                A.Apply(in p, ref q);                       // q = A p

                double delta = Blas.dot(q, q);
                if (damp != (double)0) delta += (damp * damp) * Blas.dot(p, p);   // p^T(A^T A + damp^2 I)p

                if (!(delta > (double)0))                   // NaN-safe: also catches breakdown
                    return CglsInfo(IterativeSolveStatus.Breakdown, k + 1, gamma, in r, in x);

                double alpha = gamma / delta;

                // NOTE: x (length A.Cols) and r (length A.Rows) generally differ in size here (cgls is
                // rectangular), so Blas.updateXR cannot interleave them into a shared-index pass and
                // cgls does not need updateXR's ||r||^2 byproduct (convergence is tracked via gamma =
                // ||A^T r||^2). Left as two plain axpy calls -- updateXR would only add a wasted
                // reduction here, not remove a sweep.
                x.addScaledInPlace(alpha, p);
                r.addScaledInPlace(-alpha, q);

                A.ApplyT(in r, ref s);                       // s = A^T r, recomputed fresh (stability)
                if (damp != (double)0) s.addScaledInPlace(-(damp * damp), x);   // - damp^2 x (damped gradient)

                double gammaNew = Blas.dot(s, s);

                if (gammaNew <= threshold)
                    return CglsInfo(IterativeSolveStatus.Converged, k + 1, gammaNew, in r, in x);

                double beta = gammaNew / gamma;

                p.scaleAddInPlace(beta, s);                     // p = beta p + s

                gamma = gammaNew;
            }

            return CglsInfo(IterativeSolveStatus.MaxIterations, maxIterations, gamma, in r, in x);
        }

        /// <summary>Assemble a CGLS <see cref="LstsqInfo"/> from live state: rnorm = ‖r‖
        /// (r is CGLS's live residual b - A x), Arnorm = √gamma (its tracked ‖Aᵀr - damp²x‖²),
        /// xnorm = ‖x‖. Two dots on vectors already in cache -- no matvec.</summary>
        static LstsqInfo CglsInfo(IterativeSolveStatus status, int iterations, double gamma, in doubleN r, in doubleN x)
            => new LstsqInfo
            {
                rnorm = math.sqrt(Blas.dot(r, r)),
                Arnorm = math.sqrt(gamma),
                xnorm = math.sqrt(Blas.dot(x, x)),
                iterations = iterations,
                status = status,
            };

        /// <summary>Undamped CGLS (damp = 0): plain least-squares. Forwards to the damped core.</summary>
        public static LstsqInfo cgls<TOp>(in TOp A, in doubleN b, ref doubleN x,
                                     ref doubleN r, ref doubleN s, ref doubleN p, ref doubleN q,
                                     int maxIterations, double tolerance)
            where TOp : struct, IdoubleLinearOperator
            => cgls(in A, in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance, (double)0);

        /// <summary>
        /// CGLS over a dense <see cref="doubleMxN"/> (possibly rectangular) -- zero-alloc
        /// primitive. Forwards into <see cref="cgls{TOp}"/> via <see cref="doubleDenseOperator"/>.
        /// </summary>
        public static LstsqInfo cgls(in doubleMxN A, in doubleN b, ref doubleN x,
                                ref doubleN r, ref doubleN s, ref doubleN p, ref doubleN q,
                                int maxIterations, double tolerance)
        {
            return cgls(new doubleDenseOperator(in A), in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance);
        }

        /// <summary>CGLS over a dense matrix -- allocates four scratch vectors from the arena.</summary>
        public static LstsqInfo cgls(in doubleMxN A, in doubleN b, ref doubleN x, int maxIterations, double tolerance)
        {
            doubleN r = b.doubleTempVec(A.M_Rows);
            doubleN s = b.doubleTempVec(A.N_Cols);
            doubleN p = b.doubleTempVec(A.N_Cols);
            doubleN q = b.doubleTempVec(A.M_Rows);
            return cgls(in A, in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance);
        }

        /// <summary>
        /// Damped (Tikhonov) CGLS over a dense matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates
        /// four scratch vectors from the arena. damp == 0 reproduces the plain least-squares solve.
        /// </summary>
        public static LstsqInfo cgls(in doubleMxN A, in doubleN b, ref doubleN x, int maxIterations, double tolerance, double damp)
        {
            doubleN r = b.doubleTempVec(A.M_Rows);
            doubleN s = b.doubleTempVec(A.N_Cols);
            doubleN p = b.doubleTempVec(A.N_Cols);
            doubleN q = b.doubleTempVec(A.M_Rows);
            return cgls(new doubleDenseOperator(in A), in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance, damp);
        }

        /// <summary>CGLS over a dense matrix with default maxIterations (A.N_Cols) and tolerance (Consts.doubleSqrtEps).</summary>
        public static LstsqInfo cgls(in doubleMxN A, in doubleN b, ref doubleN x)
        {
            return cgls(in A, in b, ref x, A.N_Cols, Consts.doubleSqrtEps);
        }

        /// <summary>
        /// CGLS over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc primitive.
        /// Forwards into <see cref="cgls{TOp}"/> via <c>doubleBSROperator</c>. This is the payoff
        /// of rectangular BR x BC blocks: matrix-free least squares over a sparse Jacobian-like
        /// operator, never forming AᵀA.
        /// </summary>
        public static LstsqInfo cgls(in doubleBSR A, in doubleN b, ref doubleN x,
                                ref doubleN r, ref doubleN s, ref doubleN p, ref doubleN q,
                                int maxIterations, double tolerance)
        {
            return cgls(new doubleBSROperator(in A), in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance);
        }

        /// <summary>
        /// CGLS over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc primitive
        /// variant that takes a CALLER-PROVIDED precomputed transpose AT (e.g. built once via
        /// <c>arena.doubleBSRTranspose(in A)</c> outside a hot loop / before a benchmark's timed
        /// region) and routes every ApplyT call through the resulting cache-friendly forward
        /// spMV(AT, x) instead of the scatter-heavy on-the-fly spMVT(A, x) -- see
        /// <see cref="doubleBSROperator"/>'s two-arg ctor. Caller is responsible for AT actually
        /// being A's transpose; this overload does not verify it. Prefer this over the allocating
        /// <see cref="cgls(in doubleBSR, in doubleN, ref doubleN, int, double)"/> overload when
        /// solving repeatedly against the same A (build AT once, reuse it across many solves).
        /// </summary>
        public static LstsqInfo cgls(in doubleBSR A, in doubleBSR AT, in doubleN b, ref doubleN x,
                                ref doubleN r, ref doubleN s, ref doubleN p, ref doubleN q,
                                int maxIterations, double tolerance)
        {
            return cgls(new doubleBSROperator(in A, in AT), in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance);
        }

        /// <summary>
        /// CGLS over a BSR matrix -- allocates four scratch vectors AND materializes A^T ONCE
        /// via <c>arena.doubleBSRTranspose</c> (same arena as the scratch vectors, taken from
        /// b), then drives CGLS with the two-arg <see cref="doubleBSROperator"/> so every
        /// ApplyT call routes through a cache-friendly forward spMV(A^T, x) instead of the
        /// scatter-heavy on-the-fly spMVT(A, x) every iteration -- this is the fix for the
        /// rectangular CGLS/LSQR transpose-matvec cache-unfriendliness (the one-time O(nnz)
        /// transpose build is amortized over every iteration). For a build-free zero-alloc path
        /// (e.g. many solves reusing the same A), build A^T yourself once (<c>arena.
        /// doubleBSRTranspose(in A)</c>) and call the zero-alloc <see cref="cgls(in doubleBSR,
        /// in doubleBSR, in doubleN, ref doubleN, ref doubleN, ref doubleN, ref doubleN, ref
        /// doubleN, int, double)"/> overload above with your own scratch vectors, or the generic
        /// <see cref="cgls{TOp}"/> overload directly with <c>new doubleBSROperator(in A, in
        /// AT)</c>.
        /// </summary>
        public static LstsqInfo cgls(in doubleBSR A, in doubleN b, ref doubleN x, int maxIterations, double tolerance)
        {
            doubleN r = b.doubleTempVec(A.M_Rows);
            doubleN s = b.doubleTempVec(A.N_Cols);
            doubleN p = b.doubleTempVec(A.N_Cols);
            doubleN q = b.doubleTempVec(A.M_Rows);
            doubleBSR AT = b.doubleBSRTranspose(in A);
            return cgls(new doubleBSROperator(in A, in AT), in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance);
        }

        /// <summary>
        /// Damped (Tikhonov) CGLS over a BSR matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates four
        /// scratch vectors AND materializes A^T once (see the undamped allocating overload). damp == 0
        /// reproduces the plain least-squares solve.
        /// </summary>
        public static LstsqInfo cgls(in doubleBSR A, in doubleN b, ref doubleN x, int maxIterations, double tolerance, double damp)
        {
            doubleN r = b.doubleTempVec(A.M_Rows);
            doubleN s = b.doubleTempVec(A.N_Cols);
            doubleN p = b.doubleTempVec(A.N_Cols);
            doubleN q = b.doubleTempVec(A.M_Rows);
            doubleBSR AT = b.doubleBSRTranspose(in A);
            return cgls(new doubleBSROperator(in A, in AT), in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance, damp);
        }

        /// <summary>CGLS over a BSR matrix with default maxIterations (A.N_Cols) and tolerance (Consts.doubleSqrtEps).</summary>
        public static LstsqInfo cgls(in doubleBSR A, in doubleN b, ref doubleN x)
        {
            return cgls(in A, in b, ref x, A.N_Cols, Consts.doubleSqrtEps);
        }

        /// <summary>
        /// Zero-alloc LSQR (Paige-Saunders 1982) solver for RECTANGULAR least-squares systems:
        /// minimizes ‖Ax-b‖₂ for possibly non-square A, generic over
        /// <see cref="IdoubleLinearOperator"/>. Builds an implicit bidiagonalization of A via the
        /// Golub-Kahan process (alternating <see cref="IdoubleLinearOperator.Apply"/> /
        /// <see cref="IdoubleLinearOperator.ApplyT"/> calls) and folds it through an incremental
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
        /// Returns an <see cref="LstsqInfo"/> — see that struct for the implicit-bool/status/
        /// undefined-x contract. Breakdown on a total bidiagonalization breakdown (the current
        /// alpha and beta both collapse to zero in the same step -- the Golub-Kahan recurrence
        /// exhausted).
        /// </summary>
        public static LstsqInfo lsqr<TOp>(in TOp A, in doubleN b, ref doubleN x,
                                     ref doubleN u, ref doubleN v, ref doubleN w,
                                     ref doubleN tmpM, ref doubleN tmpN,
                                     int maxIterations, double tolerance, double damp)
            where TOp : struct, IdoubleLinearOperator
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
            double atbSq = Blas.dot(tmpN, tmpN);

            if (atbSq == (double)0)
            {
                for (int i = 0; i < x.N; i++) x[i] = (double)0;
                // r = b, Aᵀr = Aᵀb = 0, x = 0.
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, math.sqrt(Blas.dot(b, b)), (double)0, (double)0, in x);
            }

            double threshold = tolerance * tolerance * atbSq;

            // u = b - A x ; beta = ||u||
            A.Apply(in x, ref tmpM);
            u.Data.CopyFrom(b.Data);
            u.addScaledInPlace((double)(-1), tmpM);

            double beta = math.sqrt(Blas.dot(u, u));

            if (beta == (double)0)
                // x already exact (r = 0): rnorm = 0, Aᵀr = 0.
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, (double)0, (double)0, (double)0, in x);

            u.divInPlace(beta);

            // v = A^T u ; alpha = ||v||
            A.ApplyT(in u, ref tmpN);
            v.Data.CopyFrom(tmpN.Data);

            double alpha = math.sqrt(Blas.dot(v, v));

            if (alpha == (double)0)
                // x already least-squares-stationary (A^T r = 0). ‖r‖ = beta.
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, beta, (double)0, (double)0, in x);

            v.divInPlace(alpha);

            // phibar tracks ‖r‖ (LSQR identity); arnorm tracks ‖Aᵀr‖ = alpha*beta pre-loop.
            double phibar = beta;
            double rhobar = alpha;
            double arnorm = alpha * beta;

            // Σψ²: energy the damping rotations peel off phibar into the residual. With damp>0 the
            // residual LSQR actually reduces is the AUGMENTED one ‖[b-Ax; -damp·x]‖, whose square is
            // sumPsiSq + phibar² -- phibar ALONE is neither the plain nor the augmented residual once
            // damping folds in. LstsqInfoTracked recovers the plain ‖b-Ax‖ from the augmented norm.
            // damp==0 -> sumPsiSq stays 0, so the undamped path reports rnorm = phibar unchanged.
            double sumPsiSq = (double)0;

            if (arnorm * arnorm <= threshold)
                // already within tolerance before the first bidiagonalization step
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, phibar, arnorm, (double)0, in x);

            w.Data.CopyFrom(v.Data);

            for (int k = 0; k < maxIterations; k++)
            {
                // ---- bidiagonalization step (Golub-Kahan) ----
                // u = A v - alpha u ; beta = ||u||, fused (Blas.xpayNormSq) into one pass over u.
                A.Apply(in v, ref tmpM);
                beta = math.sqrt(Blas.xpayNormSq(-alpha, tmpM, ref u));
                if (beta > (double)0) u.divInPlace(beta);

                // v = A^T u - beta v ; alpha = ||v||, same fusion.
                A.ApplyT(in u, ref tmpN);
                alpha = math.sqrt(Blas.xpayNormSq(-beta, tmpN, ref v));
                if (alpha > (double)0) v.divInPlace(alpha);

                // ---- fold Tikhonov damping into rhobar: rotate (rhobar, damp) -> (rhobar1, 0),
                // scaling phibar by the rotation cosine. damp==0 -> rhobar1==rhobar and phibar is
                // untouched, so the undamped path is bit-identical. ----
                double rhobar1 = rhobar;
                if (damp != (double)0)
                {
                    rhobar1 = math.sqrt(rhobar * rhobar + damp * damp);
                    double psi = (damp / rhobar1) * phibar;  // sn1 * phibar: residual rotated out by damping
                    sumPsiSq += psi * psi;
                    phibar = (rhobar / rhobar1) * phibar;   // cs1 * phibar
                }

                // ---- Givens rotation folding (rhobar1, beta) -> (rho, 0) ----
                double rho = math.sqrt(rhobar1 * rhobar1 + beta * beta);

                if (!(rho > (double)0))
                    // total breakdown: rhobar1 and beta both zero. phibar/arnorm carry the last
                    // pre-rotation values (arnorm from the previous completed step).
                    return LstsqInfoTracked(IterativeSolveStatus.Breakdown, k + 1, math.sqrt(sumPsiSq + phibar * phibar), arnorm, damp, in x);

                double c = rhobar1 / rho;
                double sn = beta / rho;
                double theta = sn * alpha;
                rhobar = -c * alpha;
                double phi = c * phibar;
                phibar = sn * phibar;

                // ---- update x using the OLD w, then update w ----
                x.addScaledInPlace(phi / rho, w);
                w.scaleAddInPlace(-theta / rho, v);             // w = -(theta/rho)*w + v

                arnorm = phibar * alpha * math.abs(c);        // ‖Aᵀr‖ for the just-updated x (free)

                if (arnorm * arnorm <= threshold)
                    return LstsqInfoTracked(IterativeSolveStatus.Converged, k + 1, math.sqrt(sumPsiSq + phibar * phibar), arnorm, damp, in x);

                if (!(beta > (double)0) || !(alpha > (double)0)) // NaN-safe: both are norms, nonnegative
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
        /// Krylov.lstsqResidual on the returned x. Call sites whose resNorm is ALREADY the plain
        /// residual (the pre-loop early exits, where no bidiagonalization/damping rotation has folded
        /// in yet, so resNorm = beta = ‖b - A·x₀‖) pass dampAug = 0 to skip the recovery. dampAug = 0
        /// makes this the identity, so the undamped path is unchanged.</summary>
        static LstsqInfo LstsqInfoTracked(IterativeSolveStatus status, int iterations, double resNorm, double Arnorm, double dampAug, in doubleN x)
        {
            double xnorm = math.sqrt(Blas.dot(x, x));
            double rr = resNorm * resNorm - dampAug * dampAug * xnorm * xnorm;
            double rnorm = rr > (double)0 ? math.sqrt(rr) : (double)0;   // guard estimate noise when ‖b-Ax‖≈0
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
        public static LstsqInfo lsqr<TOp>(in TOp A, in doubleN b, ref doubleN x,
                                     ref doubleN u, ref doubleN v, ref doubleN w,
                                     ref doubleN tmpM, ref doubleN tmpN,
                                     int maxIterations, double tolerance)
            where TOp : struct, IdoubleLinearOperator
            => lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance, (double)0);

        /// <summary>
        /// LSQR over a dense <see cref="doubleMxN"/> (possibly rectangular) -- zero-alloc
        /// primitive. Forwards into <see cref="lsqr{TOp}"/> via <see cref="doubleDenseOperator"/>.
        /// </summary>
        public static LstsqInfo lsqr(in doubleMxN A, in doubleN b, ref doubleN x,
                                ref doubleN u, ref doubleN v, ref doubleN w,
                                ref doubleN tmpM, ref doubleN tmpN,
                                int maxIterations, double tolerance)
        {
            return lsqr(new doubleDenseOperator(in A), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>LSQR over a dense matrix -- allocates five scratch vectors from the arena.</summary>
        public static LstsqInfo lsqr(in doubleMxN A, in doubleN b, ref doubleN x, int maxIterations, double tolerance)
        {
            doubleN u    = b.doubleTempVec(A.M_Rows);
            doubleN v    = b.doubleTempVec(A.N_Cols);
            doubleN w    = b.doubleTempVec(A.N_Cols);
            doubleN tmpM = b.doubleTempVec(A.M_Rows);
            doubleN tmpN = b.doubleTempVec(A.N_Cols);
            return lsqr(in A, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// Damped (Tikhonov) LSQR over a dense matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates
        /// five scratch vectors from the arena. damp == 0 reproduces the plain least-squares solve.
        /// </summary>
        public static LstsqInfo lsqr(in doubleMxN A, in doubleN b, ref doubleN x, int maxIterations, double tolerance, double damp)
        {
            doubleN u    = b.doubleTempVec(A.M_Rows);
            doubleN v    = b.doubleTempVec(A.N_Cols);
            doubleN w    = b.doubleTempVec(A.N_Cols);
            doubleN tmpM = b.doubleTempVec(A.M_Rows);
            doubleN tmpN = b.doubleTempVec(A.N_Cols);
            return lsqr(new doubleDenseOperator(in A), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance, damp);
        }

        /// <summary>LSQR over a dense matrix with default maxIterations (A.N_Cols) and tolerance (Consts.doubleSqrtEps).</summary>
        public static LstsqInfo lsqr(in doubleMxN A, in doubleN b, ref doubleN x)
        {
            return lsqr(in A, in b, ref x, A.N_Cols, Consts.doubleSqrtEps);
        }

        /// <summary>
        /// LSQR over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc primitive.
        /// Forwards into <see cref="lsqr{TOp}"/> via <c>doubleBSROperator</c>. This is the payoff
        /// of rectangular BR x BC blocks: matrix-free least squares over a sparse Jacobian-like
        /// operator, never forming AᵀA, with better ill-conditioned behavior than <see
        /// cref="cgls{TOp}"/>.
        /// </summary>
        public static LstsqInfo lsqr(in doubleBSR A, in doubleN b, ref doubleN x,
                                ref doubleN u, ref doubleN v, ref doubleN w,
                                ref doubleN tmpM, ref doubleN tmpN,
                                int maxIterations, double tolerance)
        {
            return lsqr(new doubleBSROperator(in A), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// LSQR over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc primitive
        /// variant that takes a CALLER-PROVIDED precomputed transpose AT (e.g. built once via
        /// <c>arena.doubleBSRTranspose(in A)</c> outside a hot loop / before a benchmark's timed
        /// region) and routes every ApplyT call through the resulting cache-friendly forward
        /// spMV(AT, x) instead of the scatter-heavy on-the-fly spMVT(A, x) -- see
        /// <see cref="doubleBSROperator"/>'s two-arg ctor. Caller is responsible for AT actually
        /// being A's transpose; this overload does not verify it. Prefer this over the allocating
        /// <see cref="lsqr(in doubleBSR, in doubleN, ref doubleN, int, double)"/> overload when
        /// solving repeatedly against the same A (build AT once, reuse it across many solves).
        /// </summary>
        public static LstsqInfo lsqr(in doubleBSR A, in doubleBSR AT, in doubleN b, ref doubleN x,
                                ref doubleN u, ref doubleN v, ref doubleN w,
                                ref doubleN tmpM, ref doubleN tmpN,
                                int maxIterations, double tolerance)
        {
            return lsqr(new doubleBSROperator(in A, in AT), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// LSQR over a BSR matrix -- allocates five scratch vectors AND materializes A^T ONCE
        /// via <c>arena.doubleBSRTranspose</c> (same arena as the scratch vectors, taken from
        /// b), then drives LSQR with the two-arg <see cref="doubleBSROperator"/> so every
        /// ApplyT call routes through a cache-friendly forward spMV(A^T, x) instead of the
        /// scatter-heavy on-the-fly spMVT(A, x) every iteration -- same fix and same tradeoff as
        /// <see cref="cgls(in doubleBSR, in doubleN, ref doubleN, int, double)"/>: for a
        /// build-free zero-alloc path, build A^T yourself once (<c>arena.doubleBSRTranspose(in
        /// A)</c>) and call the zero-alloc <see cref="lsqr(in doubleBSR, in doubleBSR, in
        /// doubleN, ref doubleN, ref doubleN, ref doubleN, ref doubleN, ref doubleN, int,
        /// double)"/> overload above with your own scratch vectors, or the generic
        /// <see cref="lsqr{TOp}"/> overload directly with <c>new doubleBSROperator(in A, in
        /// AT)</c>.
        /// </summary>
        public static LstsqInfo lsqr(in doubleBSR A, in doubleN b, ref doubleN x, int maxIterations, double tolerance)
        {
            doubleN u    = b.doubleTempVec(A.M_Rows);
            doubleN v    = b.doubleTempVec(A.N_Cols);
            doubleN w    = b.doubleTempVec(A.N_Cols);
            doubleN tmpM = b.doubleTempVec(A.M_Rows);
            doubleN tmpN = b.doubleTempVec(A.N_Cols);
            doubleBSR AT = b.doubleBSRTranspose(in A);
            return lsqr(new doubleBSROperator(in A, in AT), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// Damped (Tikhonov) LSQR over a BSR matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates five
        /// scratch vectors AND materializes A^T once (see the undamped allocating overload). damp == 0
        /// reproduces the plain least-squares solve.
        /// </summary>
        public static LstsqInfo lsqr(in doubleBSR A, in doubleN b, ref doubleN x, int maxIterations, double tolerance, double damp)
        {
            doubleN u    = b.doubleTempVec(A.M_Rows);
            doubleN v    = b.doubleTempVec(A.N_Cols);
            doubleN w    = b.doubleTempVec(A.N_Cols);
            doubleN tmpM = b.doubleTempVec(A.M_Rows);
            doubleN tmpN = b.doubleTempVec(A.N_Cols);
            doubleBSR AT = b.doubleBSRTranspose(in A);
            return lsqr(new doubleBSROperator(in A, in AT), in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance, damp);
        }

        /// <summary>LSQR over a BSR matrix with default maxIterations (A.N_Cols) and tolerance (Consts.doubleSqrtEps).</summary>
        public static LstsqInfo lsqr(in doubleBSR A, in doubleN b, ref doubleN x)
        {
            return lsqr(in A, in b, ref x, A.N_Cols, Consts.doubleSqrtEps);
        }

        /// <summary>
        /// Zero-alloc LSMR (Fong-Saunders 2011) solver for RECTANGULAR least-squares systems:
        /// minimizes ‖Ax-b‖₂ for possibly non-square A, generic over
        /// <see cref="IdoubleLinearOperator"/>. Built on the SAME Golub-Kahan bidiagonalization as
        /// <see cref="lsqr{TOp}"/> (alternating <see cref="IdoubleLinearOperator.Apply"/> /
        /// <see cref="IdoubleLinearOperator.ApplyT"/>), but folds it through a rotation sequence
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
        /// Returns an <see cref="LstsqInfo"/> — see that struct for the implicit-bool/status/
        /// undefined-x contract. Breakdown on a bidiagonalization breakdown (a rotation radius
        /// collapses to zero -- the Golub-Kahan recurrence exhausted).
        /// </summary>
        public static LstsqInfo lsmr<TOp>(in TOp A, in doubleN b, ref doubleN x,
                                     ref doubleN u, ref doubleN v, ref doubleN h,
                                     ref doubleN hbar, ref doubleN tmpM, ref doubleN tmpN,
                                     int maxIterations, double tolerance, double damp)
            where TOp : struct, IdoubleLinearOperator
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
            double atbSq = Blas.dot(tmpN, tmpN);

            if (atbSq == (double)0)
            {
                for (int i = 0; i < x.N; i++) x[i] = (double)0;
                // r = b, Aᵀr = Aᵀb = 0, x = 0.
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, math.sqrt(Blas.dot(b, b)), (double)0, (double)0, in x);
            }

            double threshold = tolerance * tolerance * atbSq;

            // u = b - A x ; beta = ||u||   (warm-startable: bidiagonalization of the residual)
            A.Apply(in x, ref tmpM);
            u.Data.CopyFrom(b.Data);
            u.addScaledInPlace((double)(-1), tmpM);

            double beta = math.sqrt(Blas.dot(u, u));

            if (beta == (double)0)
                // x already exact (r = 0).
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, (double)0, (double)0, (double)0, in x);

            u.divInPlace(beta);

            // v = A^T u ; alpha = ||v||
            A.ApplyT(in u, ref tmpN);
            v.Data.CopyFrom(tmpN.Data);

            double alpha = math.sqrt(Blas.dot(v, v));

            if (alpha == (double)0)
                // x already least-squares-stationary (A^T r = 0). ‖r‖ = beta.
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, beta, (double)0, (double)0, in x);

            v.divInPlace(alpha);

            // ||A^T r_0|| = alpha*beta = |zetabar_1|; matches lsqr's pre-loop early-out.
            if ((alpha * beta) * (alpha * beta) <= threshold)
                return LstsqInfoTracked(IterativeSolveStatus.Converged, 0, beta, alpha * beta, (double)0, in x);

            // h = v ; hbar = 0
            h.Data.CopyFrom(v.Data);
            for (int i = 0; i < hbar.N; i++) hbar[i] = (double)0;

            // MINRES-on-normal-equations rotation state.
            double alphabar = alpha;
            double zetabar  = alpha * beta;
            double rho = (double)1, rhobar = (double)1, cbar = (double)1, sbar = (double)0;

            // ---- ‖r‖ estimate state (Fong & Saunders 2011, "LSMR" §5.4 / SciPy lsmr). LSMR does
            // not hold the residual r = b - A x, but ‖r‖ falls out of a short scalar recurrence
            // over the SAME rotations at O(1)/iteration -- no extra matvec/dot. beta here is
            // beta1 = ‖b - A x0‖; undamped (damp==0) -> chat==1, shat==0 -> betacheck==0. ----
            double betadd = beta;
            double betad = (double)0;
            double rhodold = (double)1;
            double tautildeold = (double)0;
            double thetatilde = (double)0;
            double zeta = (double)0;
            double dnorm = (double)0;   // accumulates betacheck^2
            double normr = beta;

            for (int k = 0; k < maxIterations; k++)
            {
                // ---- bidiagonalization step (Golub-Kahan) ----
                // u = A v - alpha u ; beta = ||u||, fused (Blas.xpayNormSq) into one pass over u.
                A.Apply(in v, ref tmpM);
                beta = math.sqrt(Blas.xpayNormSq(-alpha, tmpM, ref u));
                if (beta > (double)0)
                {
                    u.divInPlace(beta);
                    // v = A^T u - beta v ; alpha = ||v||, same fusion.
                    A.ApplyT(in u, ref tmpN);
                    alpha = math.sqrt(Blas.xpayNormSq(-beta, tmpN, ref v));
                    if (alpha > (double)0) v.divInPlace(alpha);
                }

                // ---- rotation P_k : (alphahat, beta) -> (rho, 0) ----
                // alphahat folds in the Tikhonov damping: alphahat = sqrt(alphabar^2 + damp^2).
                // damp==0 -> alphahat==alphabar exactly, so the undamped path is bit-identical.
                // (chat, shat) is the rotation folding damp -- needed by the ‖r‖ recurrence.
                double rhoold = rho;
                double alphahat = damp != (double)0 ? math.sqrt(alphabar * alphabar + damp * damp) : alphabar;
                double chat, shat;
                if (alphahat > (double)0) { chat = alphabar / alphahat; shat = damp / alphahat; }
                else { chat = (double)1; shat = (double)0; }
                rho = math.sqrt(alphahat * alphahat + beta * beta);
                if (!(rho > (double)0))
                    // breakdown: alphahat and beta both zero. normr/zetabar carry the prior step's values.
                    return LstsqInfoTracked(IterativeSolveStatus.Breakdown, k + 1, normr, math.abs(zetabar), damp, in x);
                double c = alphahat / rho;
                double s = beta / rho;
                double thetanew = s * alpha;
                alphabar = c * alpha;

                // ---- rotation Pbar_k : fold R^T into Rbar (the MINRES layer) ----
                double rhobarold = rhobar;
                double thetabar = sbar * rho;
                double cbarrho = cbar * rho;
                rhobar = math.sqrt(cbarrho * cbarrho + thetanew * thetanew);
                if (!(rhobar > (double)0))
                    return LstsqInfoTracked(IterativeSolveStatus.Breakdown, k + 1, normr, math.abs(zetabar), damp, in x);
                cbar = cbarrho / rhobar;
                sbar = thetanew / rhobar;
                double zetaold = zeta;
                zeta = cbar * zetabar;
                zetabar = -sbar * zetabar;

                // ---- updates: hbar, x, h ----
                // hbar = h - (thetabar*rho / (rhoold*rhobarold)) * hbar
                double coefHbar = thetabar * rho / (rhoold * rhobarold);
                hbar.scaleAddInPlace(-coefHbar, h);           // hbar = -coefHbar*hbar + h
                // x = x + (zeta / (rho*rhobar)) * hbar
                x.addScaledInPlace(zeta / (rho * rhobar), hbar);
                // h = v - (thetanew/rho) * h
                h.scaleAddInPlace(-thetanew / rho, v);         // h = -(thetanew/rho)*h + v

                // ---- ‖r‖ recurrence for the just-updated x (this step's rotations; no matvec) ----
                double betaacute = chat * betadd;
                double betacheck = -shat * betadd;
                double betahat = c * betaacute;
                betadd = -s * betaacute;

                double thetatildeold = thetatilde;
                double rhotildeold = math.sqrt(rhodold * rhodold + thetabar * thetabar);
                double ctildeold = rhotildeold > (double)0 ? rhodold / rhotildeold : (double)1;
                double stildeold = rhotildeold > (double)0 ? thetabar / rhotildeold : (double)0;
                thetatilde = stildeold * rhobar;
                rhodold = ctildeold * rhobar;
                betad = -stildeold * betad + ctildeold * betahat;

                tautildeold = rhotildeold > (double)0 ? (zetaold - thetatildeold * tautildeold) / rhotildeold : (double)0;
                double taud = rhodold > (double)0 ? (zeta - thetatilde * tautildeold) / rhodold : (double)0;
                dnorm = dnorm + betacheck * betacheck;
                normr = math.sqrt(dnorm + (betad - taud) * (betad - taud) + betadd * betadd);

                // ‖A^T r‖ for the just-updated x = |zetabar| (falls out for free, decreases
                // monotonically). With damping this is the DAMPED normal-equation residual
                // ‖AᵀA x + damp² x − Aᵀb‖ = ‖Aᵀr − damp² x‖.
                if (zetabar * zetabar <= threshold)
                    return LstsqInfoTracked(IterativeSolveStatus.Converged, k + 1, normr, math.abs(zetabar), damp, in x);

                if (!(beta > (double)0) || !(alpha > (double)0)) // NaN-safe: both are norms, nonnegative
                    // bidiagonalization breakdown: Krylov space exhausted, no further progress
                    return LstsqInfoTracked(IterativeSolveStatus.Breakdown, k + 1, normr, math.abs(zetabar), damp, in x);
            }

            return LstsqInfoTracked(IterativeSolveStatus.MaxIterations, maxIterations, normr, math.abs(zetabar), damp, in x);
        }

        /// <summary>Undamped LSMR (damp = 0): plain least-squares. Forwards to the damped core.</summary>
        public static LstsqInfo lsmr<TOp>(in TOp A, in doubleN b, ref doubleN x,
                                     ref doubleN u, ref doubleN v, ref doubleN h,
                                     ref doubleN hbar, ref doubleN tmpM, ref doubleN tmpN,
                                     int maxIterations, double tolerance)
            where TOp : struct, IdoubleLinearOperator
            => lsmr(in A, in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance, (double)0);

        /// <summary>
        /// LSMR over a dense <see cref="doubleMxN"/> (possibly rectangular) -- zero-alloc
        /// primitive. Forwards into <see cref="lsmr{TOp}"/> via <see cref="doubleDenseOperator"/>.
        /// </summary>
        public static LstsqInfo lsmr(in doubleMxN A, in doubleN b, ref doubleN x,
                                ref doubleN u, ref doubleN v, ref doubleN h,
                                ref doubleN hbar, ref doubleN tmpM, ref doubleN tmpN,
                                int maxIterations, double tolerance)
        {
            return lsmr(new doubleDenseOperator(in A), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>LSMR over a dense matrix -- allocates six scratch vectors from the arena.</summary>
        public static LstsqInfo lsmr(in doubleMxN A, in doubleN b, ref doubleN x, int maxIterations, double tolerance)
        {
            doubleN u    = b.doubleTempVec(A.M_Rows);
            doubleN v    = b.doubleTempVec(A.N_Cols);
            doubleN h    = b.doubleTempVec(A.N_Cols);
            doubleN hbar = b.doubleTempVec(A.N_Cols);
            doubleN tmpM = b.doubleTempVec(A.M_Rows);
            doubleN tmpN = b.doubleTempVec(A.N_Cols);
            return lsmr(in A, in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// Damped (Tikhonov) LSMR over a dense matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates
        /// six scratch vectors from the arena. damp == 0 reproduces the plain least-squares solve.
        /// </summary>
        public static LstsqInfo lsmr(in doubleMxN A, in doubleN b, ref doubleN x, int maxIterations, double tolerance, double damp)
        {
            doubleN u    = b.doubleTempVec(A.M_Rows);
            doubleN v    = b.doubleTempVec(A.N_Cols);
            doubleN h    = b.doubleTempVec(A.N_Cols);
            doubleN hbar = b.doubleTempVec(A.N_Cols);
            doubleN tmpM = b.doubleTempVec(A.M_Rows);
            doubleN tmpN = b.doubleTempVec(A.N_Cols);
            return lsmr(new doubleDenseOperator(in A), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance, damp);
        }

        /// <summary>LSMR over a dense matrix with default maxIterations (A.N_Cols) and tolerance (Consts.doubleSqrtEps).</summary>
        public static LstsqInfo lsmr(in doubleMxN A, in doubleN b, ref doubleN x)
        {
            return lsmr(in A, in b, ref x, A.N_Cols, Consts.doubleSqrtEps);
        }

        /// <summary>
        /// LSMR over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc primitive.
        /// Forwards into <see cref="lsmr{TOp}"/> via <c>doubleBSROperator</c>. Matrix-free least
        /// squares over a sparse Jacobian-like operator, never forming AᵀA, with LSMR's monotone
        /// ‖Aᵀr‖ decrease (see the generic overload).
        /// </summary>
        public static LstsqInfo lsmr(in doubleBSR A, in doubleN b, ref doubleN x,
                                ref doubleN u, ref doubleN v, ref doubleN h,
                                ref doubleN hbar, ref doubleN tmpM, ref doubleN tmpN,
                                int maxIterations, double tolerance)
        {
            return lsmr(new doubleBSROperator(in A), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// LSMR over a (possibly rectangular) BSR matrix -- zero-alloc primitive that takes a
        /// CALLER-PROVIDED precomputed transpose AT (e.g. <c>arena.doubleBSRTranspose(in A)</c>
        /// built once outside a hot loop) and routes every ApplyT through the cache-friendly
        /// forward spMV(AT, x) instead of on-the-fly spMVT(A, x) -- see
        /// <see cref="doubleBSROperator"/>'s two-arg ctor. Caller is responsible for AT being A's
        /// transpose; this overload does not verify it.
        /// </summary>
        public static LstsqInfo lsmr(in doubleBSR A, in doubleBSR AT, in doubleN b, ref doubleN x,
                                ref doubleN u, ref doubleN v, ref doubleN h,
                                ref doubleN hbar, ref doubleN tmpM, ref doubleN tmpN,
                                int maxIterations, double tolerance)
        {
            return lsmr(new doubleBSROperator(in A, in AT), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// LSMR over a BSR matrix -- allocates six scratch vectors AND materializes A^T ONCE via
        /// <c>arena.doubleBSRTranspose</c>, then drives LSMR with the two-arg
        /// <see cref="doubleBSROperator"/> so every ApplyT routes through a cache-friendly forward
        /// spMV(A^T, x). For a build-free zero-alloc path, build A^T yourself once and call the
        /// zero-alloc AT overload above with your own scratch vectors.
        /// </summary>
        public static LstsqInfo lsmr(in doubleBSR A, in doubleN b, ref doubleN x, int maxIterations, double tolerance)
        {
            doubleN u    = b.doubleTempVec(A.M_Rows);
            doubleN v    = b.doubleTempVec(A.N_Cols);
            doubleN h    = b.doubleTempVec(A.N_Cols);
            doubleN hbar = b.doubleTempVec(A.N_Cols);
            doubleN tmpM = b.doubleTempVec(A.M_Rows);
            doubleN tmpN = b.doubleTempVec(A.N_Cols);
            doubleBSR AT = b.doubleBSRTranspose(in A);
            return lsmr(new doubleBSROperator(in A, in AT), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// Damped (Tikhonov) LSMR over a BSR matrix -- minimizes ‖Ax-b‖² + damp²‖x‖². Allocates six
        /// scratch vectors AND materializes A^T once (see the undamped allocating overload). damp == 0
        /// reproduces the plain least-squares solve.
        /// </summary>
        public static LstsqInfo lsmr(in doubleBSR A, in doubleN b, ref doubleN x, int maxIterations, double tolerance, double damp)
        {
            doubleN u    = b.doubleTempVec(A.M_Rows);
            doubleN v    = b.doubleTempVec(A.N_Cols);
            doubleN h    = b.doubleTempVec(A.N_Cols);
            doubleN hbar = b.doubleTempVec(A.N_Cols);
            doubleN tmpM = b.doubleTempVec(A.M_Rows);
            doubleN tmpN = b.doubleTempVec(A.N_Cols);
            doubleBSR AT = b.doubleBSRTranspose(in A);
            return lsmr(new doubleBSROperator(in A, in AT), in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance, damp);
        }

        /// <summary>LSMR over a BSR matrix with default maxIterations (A.N_Cols) and tolerance (Consts.doubleSqrtEps).</summary>
        public static LstsqInfo lsmr(in doubleBSR A, in doubleN b, ref doubleN x)
        {
            return lsmr(in A, in b, ref x, A.N_Cols, Consts.doubleSqrtEps);
        }

        // AᵀA-Jacobi (column-equilibration) convenience overloads.
        // cglsJacobi / lsqrJacobi / lsmrJacobi build the column scale d[j] = 1/||A_:,j|| from
        // columnNormsSquared, wrap A in a doubleColScaledOperator, solve the equilibrated system
        // (A*D) y = b with the underlying solver (COLD start -- x is zeroed internally; column
        // scaling is a change of variable, so a warm start would need pre-mapping y0 = D^-1 x0), and
        // unscale x = D*y in place. On an ill-conditioned least-squares problem this converges in
        // fewer iterations than the un-preconditioned solve to the SAME solution. Everything is
        // temp-pool allocated from b. BSR forms materialize A^T once (ApplyT-heavy). For explicit
        // control (custom d, warm start, damping semantics, zero-alloc) use the composable path
        // directly: Blas.columnNormsSquared + buildJacobiScale + doubleColScaledOperator + the
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
        static LstsqInfo JacobiFinish<TOp>(in TOp Aop, in doubleN b, ref doubleN x, in doubleN d,
                                                 int iterations, IterativeSolveStatus status,
                                                 ref doubleN mScratch, ref doubleN nScratch)
            where TOp : struct, IdoubleLinearOperator
        {
            for (int j = 0; j < d.N; j++) x[j] *= d[j];      // unscale x = D y
            var info = lstsqResidual(in Aop, in b, in x, (double)0, ref mScratch, ref nScratch);
            info.iterations = iterations;
            info.status = status;
            return info;
        }

        // ---- CGLS + Jacobi ----
        /// <summary>CGLS with an AᵀA-Jacobi column-equilibration preconditioner over a dense matrix.</summary>
        public static LstsqInfo cglsJacobi(in doubleMxN A, in doubleN b, ref doubleN x, int maxIterations, double tolerance)
        {
            int m = A.M_Rows, n = A.N_Cols;
            doubleN d = b.doubleTempVec(n), d2 = b.doubleTempVec(n), scratch = b.doubleTempVec(n);
            Blas.columnNormsSquared(in A, ref d2);
            Blas.buildJacobiScale(in d2, ref d);
            var op = new doubleColScaledOperator<doubleDenseOperator>(new doubleDenseOperator(in A), d, scratch);

            for (int j = 0; j < n; j++) x[j] = (double)0;                 // cold start (change of variable)
            doubleN r = b.doubleTempVec(m), s = b.doubleTempVec(n), p = b.doubleTempVec(n), q = b.doubleTempVec(m);
            var solveInfo = cgls(op, in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance);
            return JacobiFinish(new doubleDenseOperator(in A), in b, ref x, in d, solveInfo.iterations, solveInfo.status, ref r, ref s);
        }

        /// <summary>CGLS + Jacobi (dense), default maxIterations (A.N_Cols) / tolerance (Consts.doubleSqrtEps).</summary>
        public static LstsqInfo cglsJacobi(in doubleMxN A, in doubleN b, ref doubleN x)
            => cglsJacobi(in A, in b, ref x, A.N_Cols, Consts.doubleSqrtEps);

        /// <summary>CGLS with an AᵀA-Jacobi preconditioner over a BSR matrix (materializes Aᵀ once).</summary>
        public static LstsqInfo cglsJacobi(in doubleBSR A, in doubleN b, ref doubleN x, int maxIterations, double tolerance)
        {
            int m = A.M_Rows, n = A.N_Cols;
            doubleN d = b.doubleTempVec(n), d2 = b.doubleTempVec(n), scratch = b.doubleTempVec(n);
            BSR.columnNormsSquared(in A, ref d2);
            Blas.buildJacobiScale(in d2, ref d);
            doubleBSR AT = b.doubleBSRTranspose(in A);
            var op = new doubleColScaledOperator<doubleBSROperator>(new doubleBSROperator(in A, in AT), d, scratch);

            for (int j = 0; j < n; j++) x[j] = (double)0;
            doubleN r = b.doubleTempVec(m), s = b.doubleTempVec(n), p = b.doubleTempVec(n), q = b.doubleTempVec(m);
            var solveInfo = cgls(op, in b, ref x, ref r, ref s, ref p, ref q, maxIterations, tolerance);
            return JacobiFinish(new doubleBSROperator(in A, in AT), in b, ref x, in d, solveInfo.iterations, solveInfo.status, ref r, ref s);
        }

        /// <summary>CGLS + Jacobi (BSR), default maxIterations (A.N_Cols) / tolerance (Consts.doubleSqrtEps).</summary>
        public static LstsqInfo cglsJacobi(in doubleBSR A, in doubleN b, ref doubleN x)
            => cglsJacobi(in A, in b, ref x, A.N_Cols, Consts.doubleSqrtEps);

        // ---- LSQR + Jacobi ----
        /// <summary>LSQR with an AᵀA-Jacobi column-equilibration preconditioner over a dense matrix.</summary>
        public static LstsqInfo lsqrJacobi(in doubleMxN A, in doubleN b, ref doubleN x, int maxIterations, double tolerance)
        {
            int m = A.M_Rows, n = A.N_Cols;
            doubleN d = b.doubleTempVec(n), d2 = b.doubleTempVec(n), scratch = b.doubleTempVec(n);
            Blas.columnNormsSquared(in A, ref d2);
            Blas.buildJacobiScale(in d2, ref d);
            var op = new doubleColScaledOperator<doubleDenseOperator>(new doubleDenseOperator(in A), d, scratch);

            for (int j = 0; j < n; j++) x[j] = (double)0;
            doubleN u = b.doubleTempVec(m), v = b.doubleTempVec(n), w = b.doubleTempVec(n), tmpM = b.doubleTempVec(m), tmpN = b.doubleTempVec(n);
            var solveInfo = lsqr(op, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance);
            return JacobiFinish(new doubleDenseOperator(in A), in b, ref x, in d, solveInfo.iterations, solveInfo.status, ref u, ref v);
        }

        /// <summary>LSQR + Jacobi (dense), default maxIterations (A.N_Cols) / tolerance (Consts.doubleSqrtEps).</summary>
        public static LstsqInfo lsqrJacobi(in doubleMxN A, in doubleN b, ref doubleN x)
            => lsqrJacobi(in A, in b, ref x, A.N_Cols, Consts.doubleSqrtEps);

        /// <summary>LSQR with an AᵀA-Jacobi preconditioner over a BSR matrix (materializes Aᵀ once).</summary>
        public static LstsqInfo lsqrJacobi(in doubleBSR A, in doubleN b, ref doubleN x, int maxIterations, double tolerance)
        {
            int m = A.M_Rows, n = A.N_Cols;
            doubleN d = b.doubleTempVec(n), d2 = b.doubleTempVec(n), scratch = b.doubleTempVec(n);
            BSR.columnNormsSquared(in A, ref d2);
            Blas.buildJacobiScale(in d2, ref d);
            doubleBSR AT = b.doubleBSRTranspose(in A);
            var op = new doubleColScaledOperator<doubleBSROperator>(new doubleBSROperator(in A, in AT), d, scratch);

            for (int j = 0; j < n; j++) x[j] = (double)0;
            doubleN u = b.doubleTempVec(m), v = b.doubleTempVec(n), w = b.doubleTempVec(n), tmpM = b.doubleTempVec(m), tmpN = b.doubleTempVec(n);
            var solveInfo = lsqr(op, in b, ref x, ref u, ref v, ref w, ref tmpM, ref tmpN, maxIterations, tolerance);
            return JacobiFinish(new doubleBSROperator(in A, in AT), in b, ref x, in d, solveInfo.iterations, solveInfo.status, ref u, ref v);
        }

        /// <summary>LSQR + Jacobi (BSR), default maxIterations (A.N_Cols) / tolerance (Consts.doubleSqrtEps).</summary>
        public static LstsqInfo lsqrJacobi(in doubleBSR A, in doubleN b, ref doubleN x)
            => lsqrJacobi(in A, in b, ref x, A.N_Cols, Consts.doubleSqrtEps);

        // ---- LSMR + Jacobi ----
        /// <summary>LSMR with an AᵀA-Jacobi column-equilibration preconditioner over a dense matrix.</summary>
        public static LstsqInfo lsmrJacobi(in doubleMxN A, in doubleN b, ref doubleN x, int maxIterations, double tolerance)
        {
            int m = A.M_Rows, n = A.N_Cols;
            doubleN d = b.doubleTempVec(n), d2 = b.doubleTempVec(n), scratch = b.doubleTempVec(n);
            Blas.columnNormsSquared(in A, ref d2);
            Blas.buildJacobiScale(in d2, ref d);
            var op = new doubleColScaledOperator<doubleDenseOperator>(new doubleDenseOperator(in A), d, scratch);

            for (int j = 0; j < n; j++) x[j] = (double)0;
            doubleN u = b.doubleTempVec(m), v = b.doubleTempVec(n), h = b.doubleTempVec(n), hbar = b.doubleTempVec(n), tmpM = b.doubleTempVec(m), tmpN = b.doubleTempVec(n);
            var solveInfo = lsmr(op, in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance);
            return JacobiFinish(new doubleDenseOperator(in A), in b, ref x, in d, solveInfo.iterations, solveInfo.status, ref u, ref v);
        }

        /// <summary>LSMR + Jacobi (dense), default maxIterations (A.N_Cols) / tolerance (Consts.doubleSqrtEps).</summary>
        public static LstsqInfo lsmrJacobi(in doubleMxN A, in doubleN b, ref doubleN x)
            => lsmrJacobi(in A, in b, ref x, A.N_Cols, Consts.doubleSqrtEps);

        /// <summary>LSMR with an AᵀA-Jacobi preconditioner over a BSR matrix (materializes Aᵀ once).</summary>
        public static LstsqInfo lsmrJacobi(in doubleBSR A, in doubleN b, ref doubleN x, int maxIterations, double tolerance)
        {
            int m = A.M_Rows, n = A.N_Cols;
            doubleN d = b.doubleTempVec(n), d2 = b.doubleTempVec(n), scratch = b.doubleTempVec(n);
            BSR.columnNormsSquared(in A, ref d2);
            Blas.buildJacobiScale(in d2, ref d);
            doubleBSR AT = b.doubleBSRTranspose(in A);
            var op = new doubleColScaledOperator<doubleBSROperator>(new doubleBSROperator(in A, in AT), d, scratch);

            for (int j = 0; j < n; j++) x[j] = (double)0;
            doubleN u = b.doubleTempVec(m), v = b.doubleTempVec(n), h = b.doubleTempVec(n), hbar = b.doubleTempVec(n), tmpM = b.doubleTempVec(m), tmpN = b.doubleTempVec(n);
            var solveInfo = lsmr(op, in b, ref x, ref u, ref v, ref h, ref hbar, ref tmpM, ref tmpN, maxIterations, tolerance);
            return JacobiFinish(new doubleBSROperator(in A, in AT), in b, ref x, in d, solveInfo.iterations, solveInfo.status, ref u, ref v);
        }

        /// <summary>LSMR + Jacobi (BSR), default maxIterations (A.N_Cols) / tolerance (Consts.doubleSqrtEps).</summary>
        public static LstsqInfo lsmrJacobi(in doubleBSR A, in doubleN b, ref doubleN x)
            => lsmrJacobi(in A, in b, ref x, A.N_Cols, Consts.doubleSqrtEps);

        /// <summary>
        /// Zero-alloc CGNE / Craig's method (Saad Alg. 8.5) for CONSISTENT systems: finds the
        /// MINIMUM-NORM solution of A x = b (requires b in range(A)) for possibly rectangular
        /// (typically UNDER-determined, m &lt; n) A, generic over <see cref="IdoubleLinearOperator"/>.
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
        /// Returns a <see cref="SolveInfo"/> (rnorm = ‖b-Ax‖) — see that struct for the
        /// implicit-bool/status/undefined-x contract. Breakdown when ‖p‖² &lt;= 0 (Aᵀr = 0 while r
        /// is still above tolerance): for a CONSISTENT system r lies in range(A), so Aᵀr = 0 forces
        /// r = 0 in exact arithmetic -- a breakdown here therefore means the iteration reached the
        /// exact solution (to floating-point precision) or the system is inconsistent (r has
        /// stalled orthogonal to range(A) at the least-squares residual).
        /// </summary>
        public static SolveInfo cgne<TOp>(in TOp A, in doubleN b, ref doubleN x,
                                     ref doubleN r, ref doubleN p, ref doubleN q, ref doubleN tmpN,
                                     int maxIterations, double tolerance)
            where TOp : struct, IdoubleLinearOperator
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
            double bb = Blas.dot(b, b);

            if (bb == (double)0)
            {
                // b == 0 -> the unique minimum-norm solution of A x = 0 is x = 0 (any warm start
                // in x is discarded: x = 0 is the exact answer, matching cg's bb==0 shortcut).
                for (int i = 0; i < x.N; i++) x[i] = (double)0;
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (double)0);
            }

            double threshold = tolerance * tolerance * bb;

            // r = b - A x
            A.Apply(in x, ref q);                          // q = A x (temp use of q)
            r.Data.CopyFrom(b.Data);
            r.addScaledInPlace((double)(-1), q);

            double rr = Blas.dot(r, r);

            if (rr <= threshold)
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, math.sqrt(rr));

            // p = A^T r
            A.ApplyT(in r, ref p);

            for (int k = 0; k < maxIterations; k++)
            {
                double pp = Blas.dot(p, p);

                if (!(pp > (double)0))                      // NaN-safe: A^T r == 0 (r ⟂ range(A)) or p == 0
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k, math.sqrt(rr));

                double alpha = rr / pp;

                // A.Apply(p,q) does not read x, so it can run BEFORE the x/r update (reordered from
                // the original x-then-Apply-then-r sequence) without changing the computed q -- this
                // lets the r-update fold its reduction in via Blas.updateXR: x += alpha p ; r -= alpha
                // q ; rrNew = ||r||^2, replacing the separate Blas.dot(r,r) call (one fewer pass over r).
                A.Apply(in p, ref q);                       // q = A p
                double rrNew = Blas.updateXR(alpha, p, ref x, q, ref r);

                if (rrNew <= threshold)
                    return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(rrNew));

                double beta = rrNew / rr;

                A.ApplyT(in r, ref tmpN);                   // tmpN = A^T r
                p.scaleAddInPlace(beta, tmpN);                 // p = beta p + A^T r

                rr = rrNew;
            }

            return MakeSolveInfo(IterativeSolveStatus.MaxIterations, maxIterations, math.sqrt(rr));
        }

        /// <summary>
        /// CGNE / Craig over a dense <see cref="doubleMxN"/> (possibly rectangular) -- zero-alloc
        /// primitive. Forwards into <see cref="cgne{TOp}"/> via <see cref="doubleDenseOperator"/>.
        /// </summary>
        public static SolveInfo cgne(in doubleMxN A, in doubleN b, ref doubleN x,
                                ref doubleN r, ref doubleN p, ref doubleN q, ref doubleN tmpN,
                                int maxIterations, double tolerance)
        {
            return cgne(new doubleDenseOperator(in A), in b, ref x, ref r, ref p, ref q, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>CGNE over a dense matrix -- allocates four scratch vectors from the arena.</summary>
        public static SolveInfo cgne(in doubleMxN A, in doubleN b, ref doubleN x, int maxIterations, double tolerance)
        {
            doubleN r    = b.doubleTempVec(A.M_Rows);
            doubleN p    = b.doubleTempVec(A.N_Cols);
            doubleN q    = b.doubleTempVec(A.M_Rows);
            doubleN tmpN = b.doubleTempVec(A.N_Cols);
            return cgne(in A, in b, ref x, ref r, ref p, ref q, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>CGNE over a dense matrix with default maxIterations (A.N_Cols) and tolerance (Consts.doubleSqrtEps).</summary>
        public static SolveInfo cgne(in doubleMxN A, in doubleN b, ref doubleN x)
        {
            return cgne(in A, in b, ref x, A.N_Cols, Consts.doubleSqrtEps);
        }

        /// <summary>
        /// CGNE / Craig over a (possibly rectangular) block-sparse (BSR) matrix -- zero-alloc
        /// primitive. Forwards into <see cref="cgne{TOp}"/> via <c>doubleBSROperator</c>. Matrix-
        /// free minimum-norm solve over a sparse Jacobian-like operator, never forming A Aᵀ.
        /// </summary>
        public static SolveInfo cgne(in doubleBSR A, in doubleN b, ref doubleN x,
                                ref doubleN r, ref doubleN p, ref doubleN q, ref doubleN tmpN,
                                int maxIterations, double tolerance)
        {
            return cgne(new doubleBSROperator(in A), in b, ref x, ref r, ref p, ref q, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// CGNE / Craig over a BSR matrix with a CALLER-PROVIDED precomputed transpose AT (built
        /// once via <c>arena.doubleBSRTranspose(in A)</c>), routing every ApplyT through the
        /// cache-friendly forward spMV(AT, x) instead of on-the-fly spMVT(A, x) -- see
        /// <see cref="doubleBSROperator"/>'s two-arg ctor. Zero-alloc; caller owns AT.
        /// </summary>
        public static SolveInfo cgne(in doubleBSR A, in doubleBSR AT, in doubleN b, ref doubleN x,
                                ref doubleN r, ref doubleN p, ref doubleN q, ref doubleN tmpN,
                                int maxIterations, double tolerance)
        {
            return cgne(new doubleBSROperator(in A, in AT), in b, ref x, ref r, ref p, ref q, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>
        /// CGNE over a BSR matrix -- allocates four scratch vectors AND materializes A^T once via
        /// <c>arena.doubleBSRTranspose</c>, driving CGNE with the two-arg
        /// <see cref="doubleBSROperator"/> so every ApplyT routes through a cache-friendly forward
        /// spMV(A^T, x). For a build-free zero-alloc path, build A^T yourself once and call the
        /// caller-AT overload above.
        /// </summary>
        public static SolveInfo cgne(in doubleBSR A, in doubleN b, ref doubleN x, int maxIterations, double tolerance)
        {
            doubleN r    = b.doubleTempVec(A.M_Rows);
            doubleN p    = b.doubleTempVec(A.N_Cols);
            doubleN q    = b.doubleTempVec(A.M_Rows);
            doubleN tmpN = b.doubleTempVec(A.N_Cols);
            doubleBSR AT = b.doubleBSRTranspose(in A);
            return cgne(new doubleBSROperator(in A, in AT), in b, ref x, ref r, ref p, ref q, ref tmpN, maxIterations, tolerance);
        }

        /// <summary>CGNE over a BSR matrix with default maxIterations (A.N_Cols) and tolerance (Consts.doubleSqrtEps).</summary>
        public static SolveInfo cgne(in doubleBSR A, in doubleN b, ref doubleN x)
        {
            return cgne(in A, in b, ref x, A.N_Cols, Consts.doubleSqrtEps);
        }
    }

}
