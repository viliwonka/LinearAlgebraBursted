using System;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov
    {
        /// <summary>
        /// Zero-alloc preconditioned MINRES (Paige-Saunders) solver for symmetric (possibly
        /// indefinite) systems A x = b, generic over the operator (<see
        /// cref="IfProxyLinearOperator"/>) and an SPD preconditioner (<see
        /// cref="IfProxyPreconditioner"/>). Same regime as <see cref="minres{TOp}"/> -- A need
        /// NOT be positive definite -- but the Lanczos recurrence is driven by the preconditioned
        /// residual z = M⁻¹r rather than r itself. A MUST be symmetric and M MUST be SPD --
        /// caller preconditions, not verified at runtime beyond the NaN-safe breakdown guards
        /// below.
        ///
        /// Caller provides x (initial guess, overwritten with solution -- WARM-STARTABLE) and
        /// eight scratch vectors (y, r1, r2, v, w, w1, w2, z, all length A.Rows) -- <see
        /// cref="minres{TOp}"/>'s seven Paige-Saunders variables plus z for the preconditioned
        /// residual.
        ///
        /// Unlike <see cref="minres{TOp}"/>, the recursively tracked phibar is the M⁻¹-weighted
        /// residual norm once M ≠ I, not ‖b-Ax‖ -- so a claimed Converged exit is verified with
        /// one fresh r = b-Ax before it is trusted (falls through and keeps iterating on a failed
        /// verify, mirroring <see cref="pcg{TOp,TPre}"/>), and the MaxIterations exit also
        /// reports one freshly computed true residual instead of phibar. Only Breakdown reports
        /// the unverified phibar estimate -- the same carve-out <see cref="SolveInfo"/> documents
        /// for every solver's Breakdown exit.
        /// </summary>
        public static SolveInfo pminres<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x,
                                       ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                                       ref fProxyN w, ref fProxyN w1, ref fProxyN w2, ref fProxyN z,
                                       int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            if (A.Rows != A.Cols)
                throw new ArgumentException("pminres: A must be square");

            if (b.N != A.Rows) throw new ArgumentException("pminres: b.N must equal A.Rows");
            if (x.N != A.Rows) throw new ArgumentException("pminres: x.N must equal A.Rows");
            if (y.N != A.Rows) throw new ArgumentException("pminres: y.N must equal A.Rows");
            if (r1.N != A.Rows) throw new ArgumentException("pminres: r1.N must equal A.Rows");
            if (r2.N != A.Rows) throw new ArgumentException("pminres: r2.N must equal A.Rows");
            if (v.N != A.Rows) throw new ArgumentException("pminres: v.N must equal A.Rows");
            if (w.N != A.Rows) throw new ArgumentException("pminres: w.N must equal A.Rows");
            if (w1.N != A.Rows) throw new ArgumentException("pminres: w1.N must equal A.Rows");
            if (w2.N != A.Rows) throw new ArgumentException("pminres: w2.N must equal A.Rows");
            if (z.N != A.Rows) throw new ArgumentException("pminres: z.N must equal A.Rows");

            if (maxIter < 1)
                throw new ArgumentException("pminres: maxIter must be >= 1");

            unsafe
            {
                long* ptrs = stackalloc long[10];
                ptrs[0] = (long)y.Data.Ptr;  ptrs[1] = (long)r1.Data.Ptr; ptrs[2] = (long)r2.Data.Ptr;
                ptrs[3] = (long)v.Data.Ptr;  ptrs[4] = (long)w.Data.Ptr;  ptrs[5] = (long)w1.Data.Ptr;
                ptrs[6] = (long)w2.Data.Ptr; ptrs[7] = (long)z.Data.Ptr;  ptrs[8] = (long)x.Data.Ptr;
                ptrs[9] = (long)b.Data.Ptr;
                RequireDistinctBuffers("pminres: y/r1/r2/v/w/w1/w2/z/x/b must be distinct", ptrs, 10);
            }

            fProxy bb = Blas.dot(b, b);

            if (bb == (fProxy)0)
            {
                x.Data.CopyFrom(b.Data);
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, (fProxy)0);
            }

            // r1 = b - A x
            A.Apply(in x, ref y);                       // y = A x (temp use of y)
            r1.Data.CopyFrom(b.Data);
            r1.addScaledInPlace((fProxy)(-1), y);           // r1 = b - A x

            fProxy threshold = tol * tol * bb;
            fProxy trueRR0 = Blas.dot(r1, r1);

            if (trueRR0 <= threshold)
                return MakeSolveInfo(IterativeSolveStatus.Converged, 0, math.sqrt(trueRR0));

            // z = M^-1 r1 -- preconditioned residual, drives the Lanczos normalization below.
            M.Apply(in r1, ref z);
            fProxy betaSq = Blas.dot(r1, z);

            // Non-SPD preconditioner (or a non-positive <r1, M^-1 r1>): mirrors pcg's
            // !(rzold > 0) breakdown guard.
            if (!(betaSq > (fProxy)0))
                return MakeSolveInfo(IterativeSolveStatus.Breakdown, 0, math.sqrt(trueRR0));

            r2.Data.CopyFrom(r1.Data);

            // Zero the 3-term search-direction history (w/w1/w2 start at 0 in exact MINRES).
            for (int i = 0; i < A.Rows; i++) { w[i] = (fProxy)0; w1[i] = (fProxy)0; w2[i] = (fProxy)0; }

            fProxy oldb = (fProxy)0;
            fProxy beta = math.sqrt(betaSq);
            fProxy dbar = (fProxy)0;
            fProxy epsln = (fProxy)0;
            fProxy phibar = beta;
            fProxy cs = (fProxy)(-1);
            fProxy sn = (fProxy)0;
            fProxy gammaFloor = Consts.fProxyEpsilon;

            for (int k = 0; k < maxIter; k++)
            {
                // ---- preconditioned Lanczos step: extend the tridiagonalization by one vector ----
                // v = z / beta (z holds M^-1 of the CURRENT unpreconditioned Lanczos vector r2),
                // one pass (Blas.scaledCopy with a = 1/beta, i.e. reciprocal-multiply instead of a
                // per-element divide).
                Blas.scaledCopy(1 / beta, z, ref v);

                A.Apply(in v, ref y);                      // y = A v

                if (k >= 1)
                    y.addScaledInPlace(-(beta / oldb), r1);   // y -= (beta/oldb) r1

                fProxy alfa = Blas.dot(v, y);
                y.addScaledInPlace(-(alfa / beta), r2);       // y -= (alfa/beta) r2

                // Buffer rotation (r1,r2,y) -> (r2,y,r1): swap the local fProxyN handles instead of
                // Data.CopyFrom. r1's old buffer is fully consumed above (last read this iteration)
                // and is recycled as next iteration's y, which A.Apply fully overwrites regardless of
                // its incoming contents.
                { fProxyN tmp = r1; r1 = r2; r2 = y; y = tmp; }

                oldb = beta;

                // z = M^-1 r2 -- precondition the NEW unpreconditioned Lanczos vector, feeding the
                // NEXT iteration's v (and this iteration's beta below).
                M.Apply(in r2, ref z);
                fProxy betaNewSq = Blas.dot(r2, z);

                // Non-SPD preconditioner: <r2, M^-1 r2> < 0 -> beta = sqrt(negative) = NaN. Bail
                // BEFORE the Givens/x update below poisons a warm-started x; x is left untouched.
                // (betaNewSq == 0, a true invariant subspace, is left to the beta>0 guard below.)
                if (!(betaNewSq >= (fProxy)0))
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k + 1, phibar);

                beta = math.sqrt(betaNewSq);

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
                // Buffer rotation (w1,w2,w) -> (w2,w,w1), mirroring the r-rotation above.
                { fProxyN tmp = w1; w1 = w2; w2 = w; w = tmp; }

                // w = (v - oldeps*w1 - delta*w2) / gamma, one pass (Blas.combine3 with s = 1/gamma,
                // i.e. reciprocal-multiply instead of a per-element divide at the end).
                Blas.combine3(ref w, v, -oldeps, w1, -delta, w2, 1 / gamma);

                x.addScaledInPlace(phi, w);

                // phibar is the M^-1-weighted residual norm once preconditioned, not ‖b-Ax‖ --
                // verify with one fresh residual before trusting a claimed convergence (mirrors
                // pcg's verify-at-exit); fall through and keep iterating if the verify fails. y and
                // v are both idle at this point in the iteration (y: recycled garbage awaiting next
                // iteration's A.Apply; v: fully consumed by the combine3 call above), so they are
                // reused as scratch instead of allocating.
                if (phibar * phibar <= threshold)
                {
                    A.Apply(in x, ref y);                     // y = A x
                    v.Data.CopyFrom(b.Data);
                    v.addScaledInPlace((fProxy)(-1), y);         // v = b - A x
                    fProxy trueRR = Blas.dot(v, v);

                    if (trueRR <= threshold)
                        return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(trueRR));
                }

                if (!(beta > (fProxy)0))
                    // Lanczos breakdown: invariant subspace exhausted, no further progress possible.
                    return MakeSolveInfo(IterativeSolveStatus.Breakdown, k + 1, phibar);
            }

            // MaxIterations: report the TRUE residual (one fresh Apply) rather than phibar -- see
            // the preconditioned-phibar note above.
            A.Apply(in x, ref y);                             // y = A x
            v.Data.CopyFrom(b.Data);
            v.addScaledInPlace((fProxy)(-1), y);                 // v = b - A x
            fProxy finalRR = Blas.dot(v, v);
            return MakeSolveInfo(IterativeSolveStatus.MaxIterations, maxIter, math.sqrt(finalRR));
        }

        /// <summary>
        /// Preconditioned MINRES solver -- allocates eight scratch vectors from the arena and
        /// calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo pminres<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x,
                                          int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            fProxyN y  = b.fProxyTempVec(A.Rows);
            fProxyN r1 = b.fProxyTempVec(A.Rows);
            fProxyN r2 = b.fProxyTempVec(A.Rows);
            fProxyN v  = b.fProxyTempVec(A.Rows);
            fProxyN w  = b.fProxyTempVec(A.Rows);
            fProxyN w1 = b.fProxyTempVec(A.Rows);
            fProxyN w2 = b.fProxyTempVec(A.Rows);
            fProxyN z  = b.fProxyTempVec(A.Rows);
            return pminres(in A, in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// Preconditioned MINRES solver with default maxIter (A.Rows) and tol
        /// (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo pminres<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            return pminres(in A, in M, in b, ref x, A.Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned MINRES over a block-sparse (BSR) matrix with its matching block-Jacobi
        /// preconditioner. Forwards into <see cref="pminres{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c>.
        /// </summary>
        public static SolveInfo pminres(in fProxyBSR A, in fProxyBlockJacobi M, in fProxyN b, ref fProxyN x,
                               ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                               ref fProxyN w, ref fProxyN w1, ref fProxyN w2, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            return pminres(new fProxyBSROperator(in A), in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// Block-Jacobi Preconditioned MINRES over a BSR matrix -- allocates eight scratch
        /// vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo pminres(in fProxyBSR A, in fProxyBlockJacobi M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN y  = b.fProxyTempVec(A.M_Rows);
            fProxyN r1 = b.fProxyTempVec(A.M_Rows);
            fProxyN r2 = b.fProxyTempVec(A.M_Rows);
            fProxyN v  = b.fProxyTempVec(A.M_Rows);
            fProxyN w  = b.fProxyTempVec(A.M_Rows);
            fProxyN w1 = b.fProxyTempVec(A.M_Rows);
            fProxyN w2 = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return pminres(in A, in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// Block-Jacobi Preconditioned MINRES over a BSR matrix, with default maxIter (A.M_Rows)
        /// and tol (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo pminres(in fProxyBSR A, in fProxyBlockJacobi M, in fProxyN b, ref fProxyN x)
        {
            return pminres(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned MINRES over a block-sparse (BSR) matrix with its matching SSOR
        /// preconditioner. Forwards into <see cref="pminres{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c> -- same three-rung BSR convenience pattern as the block-Jacobi
        /// overloads above.
        /// </summary>
        public static SolveInfo pminres(in fProxyBSR A, in fProxySSOR M, in fProxyN b, ref fProxyN x,
                               ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                               ref fProxyN w, ref fProxyN w1, ref fProxyN w2, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            return pminres(new fProxyBSROperator(in A), in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// SSOR Preconditioned MINRES over a BSR matrix -- allocates eight scratch vectors from
        /// the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo pminres(in fProxyBSR A, in fProxySSOR M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN y  = b.fProxyTempVec(A.M_Rows);
            fProxyN r1 = b.fProxyTempVec(A.M_Rows);
            fProxyN r2 = b.fProxyTempVec(A.M_Rows);
            fProxyN v  = b.fProxyTempVec(A.M_Rows);
            fProxyN w  = b.fProxyTempVec(A.M_Rows);
            fProxyN w1 = b.fProxyTempVec(A.M_Rows);
            fProxyN w2 = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return pminres(in A, in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// SSOR Preconditioned MINRES over a BSR matrix, with default maxIter (A.M_Rows) and tol
        /// (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo pminres(in fProxyBSR A, in fProxySSOR M, in fProxyN b, ref fProxyN x)
        {
            return pminres(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned MINRES over a block-sparse (BSR) matrix with its matching block IC(0)
        /// preconditioner. Forwards into <see cref="pminres{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c> -- same three-rung BSR convenience pattern as the block-Jacobi
        /// and SSOR overloads above.
        /// </summary>
        public static SolveInfo pminres(in fProxyBSR A, in fProxyIC0 M, in fProxyN b, ref fProxyN x,
                               ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                               ref fProxyN w, ref fProxyN w1, ref fProxyN w2, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            return pminres(new fProxyBSROperator(in A), in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// IC(0) Preconditioned MINRES over a BSR matrix -- allocates eight scratch vectors from
        /// the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo pminres(in fProxyBSR A, in fProxyIC0 M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN y  = b.fProxyTempVec(A.M_Rows);
            fProxyN r1 = b.fProxyTempVec(A.M_Rows);
            fProxyN r2 = b.fProxyTempVec(A.M_Rows);
            fProxyN v  = b.fProxyTempVec(A.M_Rows);
            fProxyN w  = b.fProxyTempVec(A.M_Rows);
            fProxyN w1 = b.fProxyTempVec(A.M_Rows);
            fProxyN w2 = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return pminres(in A, in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// IC(0) Preconditioned MINRES over a BSR matrix, with default maxIter (A.M_Rows) and tol
        /// (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo pminres(in fProxyBSR A, in fProxyIC0 M, in fProxyN b, ref fProxyN x)
        {
            return pminres(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned MINRES over a block-sparse (BSR) matrix with its matching FSAI
        /// preconditioner. Forwards into <see cref="pminres{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c> -- same three-rung BSR convenience pattern as the block-Jacobi
        /// and IC0 overloads above. FSAI's local SPD solves need A[J,J] SPD; on an indefinite A
        /// build may fall back to shifted rows (same practical caveat IC0 already carries on
        /// pminres).
        /// </summary>
        public static SolveInfo pminres(in fProxyBSR A, in fProxyFSAI M, in fProxyN b, ref fProxyN x,
                               ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                               ref fProxyN w, ref fProxyN w1, ref fProxyN w2, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            return pminres(new fProxyBSROperator(in A), in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// FSAI Preconditioned MINRES over a BSR matrix -- allocates eight scratch vectors from
        /// the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo pminres(in fProxyBSR A, in fProxyFSAI M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN y  = b.fProxyTempVec(A.M_Rows);
            fProxyN r1 = b.fProxyTempVec(A.M_Rows);
            fProxyN r2 = b.fProxyTempVec(A.M_Rows);
            fProxyN v  = b.fProxyTempVec(A.M_Rows);
            fProxyN w  = b.fProxyTempVec(A.M_Rows);
            fProxyN w1 = b.fProxyTempVec(A.M_Rows);
            fProxyN w2 = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return pminres(in A, in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// FSAI Preconditioned MINRES over a BSR matrix, with default maxIter (A.M_Rows) and tol
        /// (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo pminres(in fProxyBSR A, in fProxyFSAI M, in fProxyN b, ref fProxyN x)
        {
            return pminres(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned MINRES over a block-sparse (BSR) matrix with its matching Chebyshev
        /// preconditioner. Forwards into <see cref="pminres{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c> -- same three-rung BSR convenience pattern as the block-Jacobi,
        /// SSOR, and IC0 overloads above.
        /// </summary>
        public static SolveInfo pminres(in fProxyBSR A, in fProxyChebyshev M, in fProxyN b, ref fProxyN x,
                               ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                               ref fProxyN w, ref fProxyN w1, ref fProxyN w2, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            return pminres(new fProxyBSROperator(in A), in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// Chebyshev Preconditioned MINRES over a BSR matrix -- allocates eight scratch vectors
        /// from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo pminres(in fProxyBSR A, in fProxyChebyshev M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN y  = b.fProxyTempVec(A.M_Rows);
            fProxyN r1 = b.fProxyTempVec(A.M_Rows);
            fProxyN r2 = b.fProxyTempVec(A.M_Rows);
            fProxyN v  = b.fProxyTempVec(A.M_Rows);
            fProxyN w  = b.fProxyTempVec(A.M_Rows);
            fProxyN w1 = b.fProxyTempVec(A.M_Rows);
            fProxyN w2 = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return pminres(in A, in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// Chebyshev Preconditioned MINRES over a BSR matrix, with default maxIter (A.M_Rows) and
        /// tol (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo pminres(in fProxyBSR A, in fProxyChebyshev M, in fProxyN b, ref fProxyN x)
        {
            return pminres(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }

        /// <summary>
        /// Preconditioned MINRES over a block-sparse (BSR) matrix with its matching symmetric
        /// additive-Schwarz preconditioner. Forwards into <see cref="pminres{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c> -- same three-rung BSR convenience pattern as the block-Jacobi
        /// and IC0 overloads above. AS is SPD whenever its build reports Success, so it is a valid
        /// MINRES preconditioner; restricted Schwarz (RAS) is NOT symmetric and has no pminres rung.
        /// </summary>
        public static SolveInfo pminres(in fProxyBSR A, in fProxyAdditiveSchwarz M, in fProxyN b, ref fProxyN x,
                               ref fProxyN y, ref fProxyN r1, ref fProxyN r2, ref fProxyN v,
                               ref fProxyN w, ref fProxyN w1, ref fProxyN w2, ref fProxyN z,
                               int maxIter, fProxy tol)
        {
            return pminres(new fProxyBSROperator(in A), in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// Additive-Schwarz Preconditioned MINRES over a BSR matrix -- allocates eight scratch
        /// vectors from the arena and calls the zero-alloc primitive.
        /// </summary>
        public static SolveInfo pminres(in fProxyBSR A, in fProxyAdditiveSchwarz M, in fProxyN b, ref fProxyN x,
                               int maxIter, fProxy tol)
        {
            fProxyN y  = b.fProxyTempVec(A.M_Rows);
            fProxyN r1 = b.fProxyTempVec(A.M_Rows);
            fProxyN r2 = b.fProxyTempVec(A.M_Rows);
            fProxyN v  = b.fProxyTempVec(A.M_Rows);
            fProxyN w  = b.fProxyTempVec(A.M_Rows);
            fProxyN w1 = b.fProxyTempVec(A.M_Rows);
            fProxyN w2 = b.fProxyTempVec(A.M_Rows);
            fProxyN z  = b.fProxyTempVec(A.M_Rows);
            return pminres(in A, in M, in b, ref x, ref y, ref r1, ref r2, ref v, ref w, ref w1, ref w2, ref z, maxIter, tol);
        }

        /// <summary>
        /// Additive-Schwarz Preconditioned MINRES over a BSR matrix, with default maxIter (A.M_Rows)
        /// and tol (Consts.fProxySqrtEps).
        /// </summary>
        public static SolveInfo pminres(in fProxyBSR A, in fProxyAdditiveSchwarz M, in fProxyN b, ref fProxyN x)
        {
            return pminres(in A, in M, in b, ref x, A.M_Rows, Consts.fProxySqrtEps);
        }
    }
}
