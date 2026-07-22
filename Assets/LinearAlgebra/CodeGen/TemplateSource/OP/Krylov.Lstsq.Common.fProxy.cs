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

        /// <summary>Assembles the returned <see cref="LstsqInfo"/> from a certified-exact residual
        /// audit (<see cref="lstsqResidual{TOp}"/>: one Apply + one ApplyT, damp = 0) plus the
        /// caller's iteration count and status. Shared by the undamped least-norm solvers
        /// (cgne/craig). <paramref name="rScratch"/> (length A.Rows) / <paramref name="sScratch"/>
        /// (length A.Cols) are the solver's own buffers, free to reuse post-solve.</summary>
        static LstsqInfo LstsqInfoAudited<TOp>(IterativeSolveStatus status, int iterations, in TOp A, in fProxyN b,
                                                 ref fProxyN x, ref fProxyN rScratch, ref fProxyN sScratch)
            where TOp : struct, IfProxyLinearOperator
        {
            var info = lstsqResidual(in A, in b, in x, (fProxy)0, ref rScratch, ref sScratch);
            info.iterations = iterations;
            info.status = status;
            return info;
        }

        // Right (column) preconditioned convenience overloads.
        // lsqrRightPre / lsmrRightPre solve min ‖Ax-b‖ through a change of variables x = N·y with a
        // caller-supplied SYMMETRIC preconditioner N (n×n, IfProxyPreconditioner): wrap A in a
        // fProxyRightPreconditionedOperator, solve (A·N) y = b with the underlying solver (COLD
        // start -- x is zeroed internally; the change of variable means a warm start would need
        // pre-mapping y0 = N⁻¹ x0), recover x = N·y, and report diagnostics in ORIGINAL coordinates
        // (RightPreFinish). lsqrJacobi / lsmrJacobi are the DIAGONAL case: build the column scale
        // d[j] = 1/||A_:,j|| from columnNormsSquared + buildJacobiScale, wrap it in a
        // fProxyDiagonalPreconditioner, and forward. On an ill-conditioned least-squares problem a
        // good N converges in fewer iterations than the un-preconditioned solve to the SAME
        // solution. Everything is temp-pool allocated from b. BSR forms materialize A^T once
        // (ApplyT-heavy). For explicit control (warm start, zero-alloc, custom scratch) use the
        // composable path directly: fProxyRightPreconditionedOperator + the generic solver overload.
        //
        // DIAGNOSTICS: the returned LstsqInfo is reported in ORIGINAL coordinates. The
        // preconditioned solve tracks rnorm/Arnorm/xnorm in transformed y-space (Arnorm = ‖N·Aᵀr‖
        // can be wildly off), so RightPreFinish recomputes all three exactly on the unwrapped A via
        // lstsqResidual (one Apply + ApplyT) and keeps the solve's iteration count + status. So
        // info.Arnorm here is the true ‖Aᵀr‖ and info.Solved still reflects the preconditioned solve.

        /// <summary>Shared tail for the right-preconditioned convenience wrappers: map the solution
        /// (x = N·y, y arriving in x) back to the ORIGINAL variables, then report diagnostics in
        /// original coordinates. The preconditioned solve of (A·N)y=b tracks rnorm/Arnorm/xnorm in
        /// the transformed y-space -- rnorm happens to coincide (‖b-(AN)y‖ = ‖b-Ax‖) but
        /// Arnorm = ‖(AN)ᵀr‖ = ‖N·Aᵀr‖ is off by the preconditioner (badly so on exactly the
        /// ill-scaled systems it targets). So we recompute all three exactly on the UNWRAPPED
        /// operator via <see cref="lstsqResidual"/> (one Apply + one ApplyT -- negligible next to
        /// the solve these wrappers already pay), keeping the solve's iteration count and status.
        /// <paramref name="Aop"/> is the UNWRAPPED operator; <paramref name="mScratch"/>/
        /// <paramref name="nScratch"/> are Rows-/Cols-length scratch (the solver's own buffers, free
        /// to reuse post-solve; nScratch also stages the N·y map). With nonzero
        /// <paramref name="damp"/> the reported Arnorm is ‖Aᵀr - damp²x‖, the plain ‖x‖-ridge
        /// gradient; the damped PRECONDITIONED solve minimizes the ‖N⁻¹x‖-weighted ridge instead,
        /// so even a converged damped solve generally leaves it nonzero.</summary>
        static LstsqInfo RightPreFinish<TOp, TPre>(in TOp Aop, in TPre N, in fProxyN b, ref fProxyN x,
                                                 fProxy damp, int iterations, IterativeSolveStatus status,
                                                 ref fProxyN mScratch, ref fProxyN nScratch)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            N.Apply(in x, ref nScratch);                     // x holds y: nScratch = N·y
            x.CopyFrom(in nScratch);
            var info = lstsqResidual(in Aop, in b, in x, damp, ref mScratch, ref nScratch);
            info.iterations = iterations;
            info.status = status;
            return info;
        }

        /// <summary>Right-preconditioned tail for a GENERAL (operator-valued) preconditioner N: same
        /// as <see cref="RightPreFinish{TOp,TPre}"/> but N is an <see cref="IfProxyLinearOperator"/>
        /// (need not be symmetric). Maps the solution x = N·y (y arriving in x) back to the ORIGINAL
        /// variables via <c>N.Apply</c>, then re-audits rnorm/Arnorm/xnorm on the UNWRAPPED operator
        /// via <see cref="lstsqResidual"/>, keeping the solve's iteration count and status.</summary>
        static LstsqInfo RightPreFinishOp<TOp, TPreN>(in TOp Aop, in TPreN N, in fProxyN b, ref fProxyN x,
                                                 fProxy damp, int iterations, IterativeSolveStatus status,
                                                 ref fProxyN mScratch, ref fProxyN nScratch)
            where TOp : struct, IfProxyLinearOperator
            where TPreN : struct, IfProxyLinearOperator
        {
            N.Apply(in x, ref nScratch);                     // x holds y: nScratch = N·y
            x.CopyFrom(in nScratch);
            var info = lstsqResidual(in Aop, in b, in x, damp, ref mScratch, ref nScratch);
            info.iterations = iterations;
            info.status = status;
            return info;
        }
    }
}
