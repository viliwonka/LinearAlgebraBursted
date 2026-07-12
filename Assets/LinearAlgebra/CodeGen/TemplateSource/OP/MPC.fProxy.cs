using System;

using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // ================================================================================================
    // Linear model predictive control: MPC.solve re-solves the condensed QP fProxyMPCState precomputed
    // (see that file's own header comment for the condensing math) every frame, from the CURRENT
    // measured state and reference, and returns the receding-horizon FIRST input. Warm-starts BOTH the
    // active-set working set (QP.qpActiveSetCoreWarm, persisted in state.wstatus) and the QP's own
    // feasible starting point (a shift-forward of the last plan, tail-filled via the cached LQR gain,
    // and clipped to the hard input bounds -- feasible by construction, so the phase-1-skipping QP seam
    // applies directly, no feasibility search needed every frame).
    //
    // Tracking: the reference enters only the condensed gradient (never the Hessian) -- see
    // BuildGradient. DeltaU penalty (if configured) adds one more gradient term from the last applied
    // input. Soft-row slack values are guessed (minimal violation under the warm-started trajectory)
    // alongside the input plan, so the whole starting point -- inputs AND slacks -- is feasible before
    // the QP ever runs.
    //
    // Failure contract: on anything other than QPStatus.Optimal/MaxIterations (should not happen for a
    // well-posed problem -- defensive only), MPC.solve returns the shifted PREVIOUS plan's first input
    // rather than propagating a QP failure into the caller's control loop. That value is captured
    // (ExtractU0) BEFORE the QP call, not re-derived from state.z afterward: qpActiveSetCoreWarm leaves
    // state.z/state.wstatus untouched on QPStatus.Infeasible (it short-circuits before either is
    // touched), but NOT on QPStatus.Unbounded (state.z can hold a partial iterate and state.wstatus is
    // always persisted on that path) -- capturing the guess's own u0 up front makes the fallback
    // contract hold for BOTH statuses without depending on that distinction.
    // ================================================================================================
    public static partial class MPC
    {
        /// <summary>
        /// Re-solves the condensed MPC QP from the current state and reference, returning the receding-
        /// horizon optimal first input. Warm-started from <paramref name="s"/>'s carried plan and working
        /// set (see this file's header comment); zero managed allocation, small <c>Allocator.Temp</c>
        /// scratch only.
        /// </summary>
        /// <param name="s">Must be constructed via <c>new fProxyMPCState(...)</c>.</param>
        /// <param name="x0">Current measured/estimated state, length n.</param>
        /// <param name="reference">Either a constant setpoint (length n, held across the whole horizon)
        /// or a per-stage reference trajectory (length N*n, one state-sized block per predicted stage
        /// x_1..x_N) -- detected from its own length.</param>
        /// <param name="u0out">Output, length m: the first input of the receding-horizon plan.</param>
        /// <param name="maxIter">Condensed QP active-set pivot budget; &lt;=0 picks a size-based default
        /// (matching <see cref="QP.qpActiveSetCoreWarm"/>'s own convention).</param>
        public static MPCInfo solve(ref fProxyMPCState s, in fProxyN x0, in fProxyN reference,
                                    ref fProxyN u0out, int maxIter = 0)
        {
            if (!s.IsCreated)
                throw new ArgumentException("MPC.solve: state must be constructed via new fProxyMPCState(...)");
            if (x0.N != s.n)
                throw new ArgumentException("MPC.solve: x0.N must equal the state dimension");
            if (u0out.N != s.m)
                throw new ArgumentException("MPC.solve: u0out.N must equal the input dimension");
            if (reference.N != s.n && reference.N != s.N * s.n)
                throw new ArgumentException("MPC.solve: reference.N must equal n (constant setpoint) or N*n (per-stage trajectory)");

            BuildWarmStartGuess(ref s, in x0);
            // Captured BEFORE the QP call (see this file's header comment): the ONLY value the Fallback
            // path is allowed to trust, since qpActiveSetCoreWarm's "leave state untouched" guarantee
            // does not extend to QPStatus.Unbounded.
            ExtractU0(in s, in x0, ref u0out);

            BuildGradient(ref s, in x0, in reference);
            BuildGeneralRHS(ref s, in x0);

            var qpInfo = QP.qpActiveSetCoreWarm(in s.H, in s.cScratch, in s.Arows, in s.bScratch, in s.senses,
                                                in s.xl, in s.xu, ref s.z, out double objective, maxIter,
                                                s.wstatus, out int changes);

            MPCInfo info;
            if (qpInfo.status == QPStatus.Optimal || qpInfo.status == QPStatus.MaxIterations)
            {
                ExtractU0(in s, in x0, ref u0out);   // overwrite the pre-solve guess with the real solved value
                RecoverPhysicalUPlan(ref s, in x0);
                s.populated = true;

                double maxSlack = 0;
                if (s.hasSoftRows)
                    for (int j = 0; j < s.nSlack; j++)
                        maxSlack = math.max(maxSlack, (double)s.z[s.nu + j]);

                info = new MPCInfo
                {
                    status = qpInfo.status == QPStatus.Optimal ? MPCStatus.Optimal : MPCStatus.MaxIterations,
                    iterations = qpInfo.iterations,
                    activeSetChanges = changes,
                    maxSlackViolation = maxSlack,
                    objective = objective,
                };
            }
            else
            {
                // Fallback (defensive-only, see this file's header comment): u0out already holds the
                // shifted/clipped guess's own first input, captured above before this call could mutate
                // anything. s.uPlan/s.populated are left untouched either way (never written on this
                // path) so the next frame retries from the last known-good plan; s.wstatus may have been
                // perturbed by qpActiveSetCoreWarm on Unbounded (not on Infeasible), which is harmless --
                // RepairWorkingSet re-validates every entry against the next frame's own x0 regardless.
                info = new MPCInfo
                {
                    status = MPCStatus.Fallback,
                    iterations = 0,
                    activeSetChanges = 0,
                    maxSlackViolation = 0,
                    objective = double.PositiveInfinity,
                };
            }

            return info;
        }

        // u0 = z[0:m] directly (z holds physical inputs), or -Kstab x0 + z[0:m] when prestabilized (z
        // holds v; x0 is the CURRENT, exactly-known state, so recovering u0 needs no simulation).
        static void ExtractU0(in fProxyMPCState s, in fProxyN x0, ref fProxyN u0out)
        {
            if (s.hasPrestab)
            {
                for (int i = 0; i < s.m; i++)
                {
                    fProxy sum = (fProxy)0;
                    for (int p = 0; p < s.n; p++) sum += s.Kstab[i, p] * x0[p];
                    u0out[i] = -sum + s.z[i];
                }
            }
            else
            {
                for (int i = 0; i < s.m; i++) u0out[i] = s.z[i];
            }
        }

        // Builds a FEASIBLE-by-construction condensed QP starting point into s.z, and its implied
        // predicted trajectory into s.xTrajScratch (reused right after by BuildGradient -- see that
        // method's own comment):
        //   1. shift: block k (0..N-2) <- s.uPlan's block k+1 (the previous plan's own look-ahead).
        //   2. tail-fill block N-1 via the cached LQR gain (s.Kinf) against the trajectory this SAME
        //      loop is simulating.
        //   3. clip every block to the hard input bounds (s.uLo/s.uHi) -- always in PHYSICAL u-space.
        //   4. forward-simulate with the REAL (not closed-loop) dynamics to get the predicted trajectory,
        //      converting to v = u + Kstab x THIS STEP's state when prestabilized (algebraically exact:
        //      see fProxyMPCState's file header for why this reproduces the closed-loop condensing's own
        //      implied trajectory).
        //   5. soft-row slack guess: max(0, C x_k - d) under the just-simulated trajectory.
        static void BuildWarmStartGuess(ref fProxyMPCState s, in fProxyN x0)
        {
            int n = s.n, m = s.m, N = s.N, nu = s.nu;

            var uGuess = new fProxyN(nu, Allocator.Temp, true);
            for (int k = 0; k < N - 1; k++)
                for (int i = 0; i < m; i++)
                    uGuess[k * m + i] = s.uPlan[(k + 1) * m + i];
            // block N-1 is left uninitialized here -- the loop below always overwrites it (tail-fill)
            // before it is ever read.

            var xCurr = new fProxyN(n, Allocator.Temp, true);
            xCurr.CopyFrom(x0);
            var xNext = new fProxyN(n, Allocator.Temp, true);

            for (int k = 0; k < N; k++)
            {
                if (k == N - 1)
                {
                    for (int i = 0; i < m; i++)
                    {
                        fProxy sum = (fProxy)0;
                        for (int p = 0; p < n; p++) sum += s.Kinf[i, p] * xCurr[p];
                        uGuess[k * m + i] = -sum;
                    }
                }
                for (int i = 0; i < m; i++)
                {
                    fProxy v = uGuess[k * m + i];
                    if (v < s.uLo[i]) v = s.uLo[i];
                    else if (v > s.uHi[i]) v = s.uHi[i];
                    uGuess[k * m + i] = v;
                }

                if (s.hasPrestab)
                {
                    for (int i = 0; i < m; i++)
                    {
                        fProxy sum = (fProxy)0;
                        for (int p = 0; p < n; p++) sum += s.Kstab[i, p] * xCurr[p];
                        s.z[k * m + i] = uGuess[k * m + i] + sum;
                    }
                }
                else
                {
                    for (int i = 0; i < m; i++) s.z[k * m + i] = uGuess[k * m + i];
                }

                for (int p = 0; p < n; p++)
                {
                    fProxy sum = (fProxy)0;
                    for (int q = 0; q < n; q++) sum += s.A[p, q] * xCurr[q];
                    for (int q = 0; q < m; q++) sum += s.B[p, q] * uGuess[k * m + q];
                    xNext[p] = sum;
                }
                for (int p = 0; p < n; p++) s.xTrajScratch[k * n + p] = xNext[p];
                xCurr.CopyFrom(xNext);
            }

            if (s.hasSoftRows)
            {
                int rowIdx = 0;
                for (int k = 0; k < N; k++)
                {
                    for (int si = 0; si < s.nSoftPerStage; si++)
                    {
                        fProxy act = (fProxy)0;
                        for (int p = 0; p < n; p++) act += s.C[si, p] * s.xTrajScratch[k * n + p];
                        fProxy viol = act - s.d[si];
                        s.z[nu + rowIdx] = viol > (fProxy)0 ? viol : (fProxy)0;
                        rowIdx++;
                    }
                }
            }

            xNext.Dispose(); xCurr.Dispose(); uGuess.Dispose();
        }

        // Condensed gradient c (length nz): tracking term 2*GtQbar*(Phi x0 - reference) into c[0:nu]
        // (reusing s.xTrajScratch as scratch -- its warm-start-guess content from BuildWarmStartGuess is
        // no longer needed once that method returns), a deltaU adjustment on c[0:m] from the LAST applied
        // input (s.uPlan's own block 0, still the OLD value here -- RecoverPhysicalUPlan has not run yet
        // this call), a prestabilization correction (Rcross @ x0, see fProxyMPCState's file header --
        // the u_k^T R u_k cost term's x0-dependent part under the u_k = -Kstab x_k + v_k substitution),
        // and the soft-row L1 weight into c[nu:nz].
        static void BuildGradient(ref fProxyMPCState s, in fProxyN x0, in fProxyN reference)
        {
            int n = s.n, m = s.m, N = s.N, nu = s.nu, Nn = N * n;
            bool perStage = reference.N == Nn;

            Blas.dot(in s.Phi, in x0, ref s.xTrajScratch);   // xTrajScratch <- Phi x0
            for (int k = 0; k < N; k++)
                for (int i = 0; i < n; i++)
                    s.xTrajScratch[k * n + i] -= perStage ? reference[k * n + i] : reference[i];

            for (int row = 0; row < nu; row++)
            {
                fProxy sum = (fProxy)0;
                for (int col = 0; col < Nn; col++) sum += s.GtQbar[row, col] * s.xTrajScratch[col];
                s.cScratch[row] = (fProxy)2 * sum;
            }

            if (s.hasPrestab)
            {
                for (int row = 0; row < nu; row++)
                {
                    fProxy sum = (fProxy)0;
                    for (int p = 0; p < n; p++) sum += s.Rcross[row, p] * x0[p];
                    s.cScratch[row] += sum;
                }
            }

            if (s.hasDeltaU)
            {
                for (int i = 0; i < m; i++)
                {
                    fProxy sum = (fProxy)0;
                    for (int j = 0; j < m; j++) sum += s.S[i, j] * s.uPlan[j];
                    s.cScratch[i] -= (fProxy)2 * sum;
                }
            }

            if (s.hasSoftRows)
                for (int j = 0; j < s.nSlack; j++) s.cScratch[nu + j] = s.rho1;
        }

        // Condensed general-row RHS (length nGeneral): rowConstRHS - qCoupling@x0, rebuilt every call
        // (the coefficients themselves are fixed at construction -- see fProxyMPCState's own fields).
        static void BuildGeneralRHS(ref fProxyMPCState s, in fProxyN x0)
        {
            int n = s.n;
            for (int row = 0; row < s.nGeneral; row++)
            {
                fProxy sum = (fProxy)0;
                for (int p = 0; p < n; p++) sum += s.qCoupling[row, p] * x0[p];
                s.bScratch[row] = s.rowConstRHS[row] - sum;
            }
        }

        // Recovers the PHYSICAL input plan from the just-solved s.z into s.uPlan (the warm-start basis
        // for the NEXT call's shift). Direct copy when not prestabilized (z already holds u); otherwise
        // forward-simulates with the closed-loop relation u_k = -Kstab x_k + v_k, x_{k+1} = A x_k + B u_k
        // (REAL A/B; see fProxyMPCState's file header for why this recovers the SAME trajectory the
        // condensed QP itself assumed).
        static void RecoverPhysicalUPlan(ref fProxyMPCState s, in fProxyN x0)
        {
            if (!s.hasPrestab)
            {
                for (int i = 0; i < s.nu; i++) s.uPlan[i] = s.z[i];
                return;
            }

            int n = s.n, m = s.m, N = s.N;
            var xCurr = new fProxyN(n, Allocator.Temp, true);
            xCurr.CopyFrom(x0);
            var xNext = new fProxyN(n, Allocator.Temp, true);

            for (int k = 0; k < N; k++)
            {
                for (int i = 0; i < m; i++)
                {
                    fProxy sum = (fProxy)0;
                    for (int p = 0; p < n; p++) sum += s.Kstab[i, p] * xCurr[p];
                    s.uPlan[k * m + i] = -sum + s.z[k * m + i];
                }
                for (int p = 0; p < n; p++)
                {
                    fProxy sum = (fProxy)0;
                    for (int q = 0; q < n; q++) sum += s.A[p, q] * xCurr[q];
                    for (int q = 0; q < m; q++) sum += s.B[p, q] * s.uPlan[k * m + q];
                    xNext[p] = sum;
                }
                xCurr.CopyFrom(xNext);
            }

            xNext.Dispose(); xCurr.Dispose();
        }
    }
}
