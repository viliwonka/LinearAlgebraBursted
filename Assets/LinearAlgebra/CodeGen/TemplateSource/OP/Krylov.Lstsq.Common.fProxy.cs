using Unity.Mathematics;

namespace LinearAlgebra
{
    public static partial class Krylov {

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
        /// residual for a cold-start (x₀=0) lsqr/lsmr. Auditing a WARM-STARTED
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
            rScratch.scaleAddInPlace((fProxy)(-1), b);          // rScratch = -A x + b = b - A x
            fProxy rnorm = math.sqrt(Blas.dot(rScratch, rScratch));

            // s = Aᵀr - damp²x  (the ‖x‖-Tikhonov optimality residual)
            A.ApplyT(in rScratch, ref sScratch);
            if (damp != (fProxy)0) sScratch.addScaledInPlace(-(damp * damp), x);
            fProxy arnorm = math.sqrt(Blas.dot(sScratch, sScratch));

            fProxy xnorm = math.sqrt(Blas.dot(x, x));

            return new LstsqInfo
            {
                rnorm = rnorm,
                Arnorm = arnorm,
                xnorm = xnorm,
                iterations = 0,
                status = IterativeSolveStatus.Converged,
            };
        }

        /// <summary>Assemble an <see cref="LstsqInfo"/> from a solver's tracked residual and
        /// ‖Aᵀr‖ scalars, filling xnorm = ‖x‖ with one dot on x. Shared by lsqr/lsmr.
        ///
        /// Recovers the plain ‖b-Ax‖ from the (possibly damping-augmented) tracked residual; exact
        /// only for damp == 0 or a cold start (x₀ = 0). Under a NONZERO warm start with damping, this
        /// does NOT return ‖b-Ax‖ -- start damped solves from x=0, or read ‖b-Ax‖ from
        /// <see cref="lstsqResidual{TOp}"/> on the returned x.</summary>
        static LstsqInfo LstsqInfoTracked(IterativeSolveStatus status, int iterations, fProxy resNorm, fProxy Arnorm, fProxy dampAug, in fProxyN x)
        {
            fProxy xnorm = math.sqrt(Blas.dot(x, x));
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

        // AᵀA-Jacobi (column-equilibration) convenience overloads.
        // lsqrJacobi / lsmrJacobi build the column scale d[j] = 1/||A_:,j|| from
        // columnNormsSquared, wrap A in a fProxyColScaledOperator, solve the equilibrated system
        // (A*D) y = b with the underlying solver (COLD start -- x is zeroed internally; column
        // scaling is a change of variable, so a warm start would need pre-mapping y0 = D^-1 x0), and
        // unscale x = D*y in place. On an ill-conditioned least-squares problem this converges in
        // fewer iterations than the un-preconditioned solve to the SAME solution. Everything is
        // temp-pool allocated from b. BSR forms materialize A^T once (ApplyT-heavy). For explicit
        // control (custom d, warm start, damping semantics, zero-alloc) use the composable path
        // directly: Blas.columnNormsSquared + buildJacobiScale + fProxyColScaledOperator + the
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
    }
}
