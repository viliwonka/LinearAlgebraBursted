using System;

using Unity.Collections;
using Unity.Mathematics;
using BULA.Internal;

namespace BULA.Control
{
    // ================================================================================================
    // Buffer-carrying state for MPC.solve: a linear time-invariant model x_{k+1} = A x_k + B u_k is
    // condensed ONCE, at construction, into the standard batch/dense QP form used throughout linear MPC
    // (Borrelli, Bemporad & Morari, "Predictive Control for Linear and Hybrid Systems", ch. 2's
    // "condensed" formulation; the same condense-then-QP shape acados/HPIPM's condensing routines and
    // TinyMPC use, read for product-shape reference only -- no source from either was transcribed).
    //
    // Decision vector z = [u_0; ...; u_{N-1}] (length N*m; or [v_0; ...; v_{N-1}] when prestabilized --
    // see fProxyMPCState.Kstab's own doc comment) followed by [s_0; ...; s_{N-1}] soft-row slacks
    // (length N*nSoftRows, 0 if disabled). Predicted states X = Phi x0 + Gamma z stack x_1..x_N; the
    // condensed cost sum_k (x_k-r_k)^T Q (x_k-r_k) + (x_N-r_N)^T P (x_N-r_N) + u_k^T R u_k (+ deltaU +
    // slack terms) becomes z^T H z + g^T z, with H FIXED at construction (data does not change frame to
    // frame) and g rebuilt from the current x0/reference every MPC.solve call.
    //
    // Soft state rows (Kerrigan & Maciejowski, "Soft Constraints and Exact Penalty Functions in Model
    // Predictive Control", 2000): a row C x_k <= d becomes C x_k - s_k <= d, s_k >= 0, with cost penalty
    // rho1*s_k + rho2*s_k^2. The L1 term rho1*s_k is what makes the relaxation EXACT (zero violation
    // whenever the original hard row is satisfiable) once rho1 exceeds the row's own hard-constraint
    // multiplier; rho2*s_k^2 only adds strict convexity/conditioning, it does not affect exactness.
    //
    // Terminal cost P defaults to the infinite-horizon DARE solution (via LQR.lqr's own public warm
    // overload, which exposes the converged Riccati S directly -- no need to reach into LQR's
    // internals for this) -- the SAME call also yields Kinf, the LQR tail gain used to fill the newly-
    // exposed last stage of the warm-started next-frame guess after shifting.
    //
    // Prestabilization (owner-approved float32 conditioning insurance): substituting u_k = -Kstab x_k +
    // v_k and condensing the CLOSED LOOP (A - B Kstab instead of A) keeps Phi/Gamma bounded even when the
    // raw A is unstable and rho(A)^N would otherwise blow up Phi/Gamma (and hence H's condition number)
    // at a long horizon in float32. Use it when rho(A)^N is large relative to the horizon (an unstable
    // plant with N large enough that A^N would already be enormous). Prestabilization is a PURE change of
    // coordinates -- the physical solution (u_0..u_{N-1}, the predicted trajectory) is IDENTICAL to
    // solving the same (A,B,Q,R,uLo,uHi) problem without it; only the QP's own decision vector and
    // conditioning differ. Two consequences, both routed through the same affine map
    // u_k = M_row_k @ V + c_k built once here (its Phi/Gamma block index is off by one -- see DEVLOG):
    //   1. Hard input bounds become GENERAL rows, not a box on z, since u_k depends on the predicted
    //      state, which depends on V.
    //   2. The u_k^T R u_k term is NOT Rbar-on-v -- that silently drops the -Kstab*x_k cross-coupling.
    // Combining prestabilization with the deltaU penalty is NOT supported in v1 (throws at
    // construction).
    // ================================================================================================
    public partial struct fProxyMPCState
    {
        /// <summary>State dimension.</summary>
        public int n;
        /// <summary>Input dimension.</summary>
        public int m;
        /// <summary>Prediction horizon (number of stages).</summary>
        public int N;
        /// <summary>N*m -- length of the condensed decision vector's input/v block.</summary>
        public int nu;
        /// <summary>Soft rows PER STAGE (0 if soft rows are disabled).</summary>
        public int nSoftPerStage;
        /// <summary>N*nSoftPerStage -- total slack count (0 if disabled).</summary>
        public int nSlack;
        /// <summary>nu + nSlack -- length of the condensed QP decision vector.</summary>
        public int nz;
        /// <summary>General (non-box) condensed row count: N*nSoftPerStage soft rows plus, if
        /// prestabilized, 2*N*m input-bound rows.</summary>
        public int nGeneral;

        /// <summary>True if a deltaU penalty (<see cref="S"/>) is configured.</summary>
        public bool hasDeltaU;
        /// <summary>True if soft state rows (<see cref="C"/>/<see cref="d"/>) are configured.</summary>
        public bool hasSoftRows;
        /// <summary>True if prestabilization (<see cref="Kstab"/>) is configured.</summary>
        public bool hasPrestab;
        NativeReference<int> _populated;

        /// <summary>False until the first solve that reaches the condensed QP (<see cref="MPCStatus.Optimal"/>
        /// or <see cref="MPCStatus.MaxIterations"/>) completes. Native-backed exactly like
        /// <see cref="fProxyLQRState.populated"/> -- a plain field would not survive an <c>IJob</c>
        /// by-value copy (silently forcing every warm call back onto the cold path), so this lives
        /// behind a shared handle that writes persist through.</summary>
        public bool populated
        {
            get => _populated.IsCreated && _populated.Value != 0;
            set => _populated.Value = value ? 1 : 0;
        }

        // ---- condensed prediction matrices (fixed at construction) ----

        /// <summary>Prediction matrix, N*n x n: predicted state block k (rows [k*n,(k+1)*n)) is
        /// x_{k+1}'s x0-coefficient. Uses the CLOSED-LOOP dynamics (A - B Kstab) when prestabilized.</summary>
        public fProxyMxN Phi;

        /// <summary>Prediction matrix, N*n x nu: predicted state block k's z-coefficient (z = inputs, or
        /// prestabilization's v). Block lower triangular. Same closed-loop convention as <see cref="Phi"/>.</summary>
        public fProxyMxN Gamma;

        /// <summary>Condensed QP Hessian, nz x nz -- already the QP convention's 2*(physical Hessian)
        /// (<see cref="QP.qpActiveSetCoreWarm"/>'s 1/2 zᵀHz + gᵀz form), so it plugs in directly.</summary>
        public fProxyMxN H;

        /// <summary>Gamma^T Qbar (Qbar block-diagonal(Q,...,Q,P)), nu x (N*n) -- precomputed so the
        /// per-call tracking-gradient assembly is a single matrix-vector apply against (Phi x0 - r).</summary>
        public fProxyMxN GtQbar;

        // ---- condensed general rows (soft state rows + prestabilized input bounds), fixed coefficients ----

        /// <summary>General-row coefficient matrix, nGeneral x nz (0 rows if <see cref="nGeneral"/> is 0).
        /// Fixed at construction; the row's RHS (x0-dependent) is rebuilt every solve call from
        /// <see cref="qCoupling"/>/<see cref="rowConstRHS"/>.</summary>
        public fProxyMxN Arows;

        /// <summary>Per-row x0-coupling vector, nGeneral x n: row's RHS this call =
        /// <see cref="rowConstRHS"/>[row] - qCoupling[row,:]@x0.</summary>
        public fProxyMxN qCoupling;

        /// <summary>Per-row constant RHS term, length nGeneral (before subtracting the x0-coupling).</summary>
        public fProxyN rowConstRHS;

        /// <summary>Per-row sense, length nGeneral. Always <see cref="ConstraintSense.LessEqual"/> in v1
        /// (both soft rows and prestabilized input-bound rows are assembled as LessEqual).</summary>
        public NativeArray<ConstraintSense> senses;

        /// <summary>Box lower bound, length nz. Input block is the literal per-input lower bound tiled N
        /// times when NOT prestabilized (-1e30 sentinel tiled instead when prestabilized -- those bounds
        /// become general rows), slack block (if any) is 0.</summary>
        public fProxyN xl;

        /// <summary>Box upper bound, length nz. Slack block (if any) is the +1e30 sentinel (slacks are
        /// only bounded below).</summary>
        public fProxyN xu;

        // ---- model copies (needed every solve call for warm-start forward simulation) ----

        /// <summary>Dynamics, n x n (the ORIGINAL, not closed-loop, plant -- needed every solve call to
        /// physically forward-simulate the shifted warm-start guess).</summary>
        public fProxyMxN A;

        /// <summary>Control input, n x m.</summary>
        public fProxyMxN B;

        /// <summary>Hard input lower bound, length m -- kept ALWAYS in physical u-space (even when
        /// prestabilized, where the box bound on z itself is +-infinity -- see <see cref="xl"/>) since
        /// the warm-start guess is always clipped in physical space.</summary>
        public fProxyN uLo;

        /// <summary>Hard input upper bound, length m. See <see cref="uLo"/>.</summary>
        public fProxyN uHi;

        /// <summary>Infinite-horizon LQR gain (from the SAME DARE solve as the default terminal cost --
        /// see this file's header comment), m x n. Used to fill the newly-exposed last stage of the
        /// warm-started next-frame guess after shifting -- ALWAYS computed, independent of whether
        /// <see cref="MPC.State.fProxy.cs"/>'s caller supplied an explicit terminal P.</summary>
        public fProxyMxN Kinf;

        /// <summary>Prestabilization gain, m x n. <see cref="fProxyMxN.IsCreated"/> false if disabled.</summary>
        public fProxyMxN Kstab;

        /// <summary>Prestabilization-only gradient correction, nu x n: the physical input is an affine
        /// map of (x0, V) under prestabilization (u_k = -Kstab x_k + v_k), so the u_k^T R u_k cost term
        /// contributes a per-call linear term on top of the usual tracking gradient -- added as
        /// <c>Rcross @ x0</c> every <see cref="MPC.solve"/> call (see <see cref="MPC.fProxy.cs"/>'s
        /// BuildGradient). <see cref="fProxyMxN.IsCreated"/> false if not prestabilized.</summary>
        public fProxyMxN Rcross;

        /// <summary>DeltaU penalty weight, m x m. <see cref="fProxyMxN.IsCreated"/> false if disabled.</summary>
        public fProxyMxN S;

        /// <summary>Soft-row coefficients, nSoftPerStage x n. <see cref="fProxyMxN.IsCreated"/> false if
        /// disabled.</summary>
        public fProxyMxN C;

        /// <summary>Soft-row right-hand sides, length nSoftPerStage.</summary>
        public fProxyN d;

        /// <summary>Soft-row L1 (exact-penalty) weight.</summary>
        public fProxy rho1;

        /// <summary>Soft-row quadratic weight.</summary>
        public fProxy rho2;

        // ---- persistent QP scratch / warm-start carry ----

        /// <summary>Condensed QP decision vector -- rebuilt into a feasible warm-started guess at the
        /// start of every <see cref="MPC.solve"/> call, then overwritten in place by the QP solve.</summary>
        public fProxyN z;

        /// <summary>Persistent working-set status array for <see cref="QP.qpActiveSetCoreWarm"/>, length
        /// nGeneral + nz. Zero-initialized (every row Inactive) at construction.</summary>
        public NativeArray<byte> wstatus;

        /// <summary>Condensed QP gradient scratch, length nz. Rebuilt every solve call.</summary>
        public fProxyN cScratch;

        /// <summary>Condensed QP general-row RHS scratch, length nGeneral. Rebuilt every solve call.</summary>
        public fProxyN bScratch;

        /// <summary>Physical (never prestabilization-substituted) input plan from the last solve that
        /// reached the QP, u_0..u_{N-1}, length nu. The warm-start basis for the NEXT call's shift; left
        /// UNCHANGED on <see cref="MPCStatus.Fallback"/> (see that status's own doc comment).</summary>
        public fProxyN uPlan;

        /// <summary>Forward-simulated state trajectory scratch, length N*n. Dual purpose within one
        /// solve call: first the warm-start guess's predicted trajectory (for tail-filling, slack
        /// guessing, and prestabilization's u-to-v conversion), then reused as the tracking-gradient's
        /// (Phi x0 - reference) buffer once the guess is built.</summary>
        public fProxyN xTrajScratch;

        /// <summary>Persistent condensed-QP working-set QR factorization, carried across solves so a
        /// steady-state tick (unchanged working set) reuses it instead of refactoring -- see
        /// <see cref="QP.qpActiveSetCoreWarmPersistent"/>. Sized for n = <see cref="nz"/>. INTERNAL (the
        /// field type is an internal solver detail); the public surface is <see cref="MPC.solve"/>.</summary>
        internal fProxyQPFactorState qpFactor;

        /// <summary>Persistent condensed-QP reduced space (Z / QZ / H_Z), carried across solves so a
        /// steady-state tick skips the O(nz²·(nz_free)) reduced-space rebuild. Sized for n =
        /// <see cref="nz"/>. INTERNAL, see <see cref="qpFactor"/>.</summary>
        internal fProxyQPReducedState qpReduced;

        /// <summary>Native-backed persistence of the scalar factorization metadata that the
        /// <see cref="qpFactor"/>/<see cref="qpReduced"/> STRUCTS carry as plain fields (k, reflCount,
        /// stale, changeCount, ...): those plain fields do NOT survive an <c>IJob.Run()</c>/<c>Schedule</c>
        /// by-value copy of this state, but the factorization's native BUFFERS do -- so
        /// <see cref="QP.qpActiveSetCoreWarmPersistent"/> rehydrates the counters from here at the start of
        /// every solve and writes them back at the end, making cross-tick reuse work through the job path.
        /// Length 8: [0]=factorValid, [1..5]=k/reflCount/rotCount/opCount/deadCount, [6]=stale,
        /// [7]=changeCount. Initialized to "no factor yet" (factorValid=0, stale=1).</summary>
        internal NativeArray<int> qpMeta;

        /// <summary>True once every buffer is allocated (regardless of content validity).</summary>
        public bool IsCreated => z.Data.IsCreated;

        /// <summary>True iff created AND sized for exactly (n, m, N).</summary>
        public bool IsValid(int n, int m, int N) => IsCreated && this.n == n && this.m == m && this.N == N;

        /// <summary>Releases every buffer. Safe to call on an empty/already-disposed instance.</summary>
        public void Dispose()
        {
            if (Phi.Data.IsCreated) Phi.Dispose();
            if (Gamma.Data.IsCreated) Gamma.Dispose();
            if (H.Data.IsCreated) H.Dispose();
            if (GtQbar.Data.IsCreated) GtQbar.Dispose();
            if (Arows.Data.IsCreated) Arows.Dispose();
            if (qCoupling.Data.IsCreated) qCoupling.Dispose();
            if (rowConstRHS.Data.IsCreated) rowConstRHS.Dispose();
            if (senses.IsCreated) senses.Dispose();
            if (xl.Data.IsCreated) xl.Dispose();
            if (xu.Data.IsCreated) xu.Dispose();
            if (A.Data.IsCreated) A.Dispose();
            if (B.Data.IsCreated) B.Dispose();
            if (uLo.Data.IsCreated) uLo.Dispose();
            if (uHi.Data.IsCreated) uHi.Dispose();
            if (Kinf.Data.IsCreated) Kinf.Dispose();
            if (Kstab.Data.IsCreated) Kstab.Dispose();
            if (Rcross.Data.IsCreated) Rcross.Dispose();
            if (S.Data.IsCreated) S.Dispose();
            if (C.Data.IsCreated) C.Dispose();
            if (d.Data.IsCreated) d.Dispose();
            if (z.Data.IsCreated) z.Dispose();
            if (wstatus.IsCreated) wstatus.Dispose();
            if (cScratch.Data.IsCreated) cScratch.Dispose();
            if (bScratch.Data.IsCreated) bScratch.Dispose();
            if (uPlan.Data.IsCreated) uPlan.Dispose();
            if (xTrajScratch.Data.IsCreated) xTrajScratch.Dispose();
            if (qpFactor.V.Data.IsCreated) qpFactor.Dispose();
            if (qpReduced.Z.Data.IsCreated) qpReduced.Dispose();
            if (qpMeta.IsCreated) qpMeta.Dispose();
            if (_populated.IsCreated) _populated.Dispose();
        }

        // ================================================================================================
        // Construction
        // ================================================================================================

        /// <summary>
        /// Full constructor -- every optional knob explicit. Pass <c>default(fProxyMxN)</c> for
        /// <paramref name="P"/> (auto-DARE terminal cost), <paramref name="S"/> (no deltaU penalty),
        /// <paramref name="C"/> (no soft rows -- <paramref name="d"/>/<paramref name="rho1"/>/
        /// <paramref name="rho2"/> are then ignored), and/or <paramref name="Kstab"/> (no
        /// prestabilization) to disable that feature; convenience overloads below cover the common
        /// single-feature cases.
        /// </summary>
        /// <param name="n">State dimension.</param>
        /// <param name="m">Input dimension.</param>
        /// <param name="N">Prediction horizon, &gt;= 1.</param>
        /// <param name="allocator">Backs every persistent buffer this state owns.</param>
        /// <param name="A">Dynamics, n x n.</param>
        /// <param name="B">Control input, n x m.</param>
        /// <param name="Q">Stage state cost, n x n (assumed symmetric PSD; not numerically validated,
        /// matching <see cref="LQR.lqr"/>'s own contract).</param>
        /// <param name="R">Stage input cost, m x m (assumed symmetric PD).</param>
        /// <param name="uLo">Hard input lower bound, length m.</param>
        /// <param name="uHi">Hard input upper bound, length m.</param>
        /// <param name="P">Terminal state cost, n x n, or <c>default(fProxyMxN)</c> for the infinite-
        /// horizon DARE solution (via <see cref="LQR.lqr"/>).</param>
        /// <param name="S">DeltaU penalty weight, m x m, or <c>default(fProxyMxN)</c> to disable.</param>
        /// <param name="C">Soft-row coefficients, k x n (k &gt;= 1), or <c>default(fProxyMxN)</c> to
        /// disable soft rows entirely.</param>
        /// <param name="d">Soft-row right-hand sides, length k (matching <paramref name="C"/>'s row
        /// count). Ignored if <paramref name="C"/> is not created.</param>
        /// <param name="rho1">Soft-row L1 (exact-penalty) weight. Must exceed the corresponding hard-
        /// constraint multiplier for the exact-penalty property to hold (Kerrigan &amp; Maciejowski 2000)
        /// -- see this file's header comment.</param>
        /// <param name="rho2">Soft-row quadratic weight (conditioning only, does not affect exactness).</param>
        /// <param name="Kstab">Prestabilization gain, m x n, or <c>default(fProxyMxN)</c> to disable.
        /// NOT supported together with a created <paramref name="S"/> (throws).</param>
        public fProxyMPCState(int n, int m, int N, Allocator allocator,
                              in fProxyMxN A, in fProxyMxN B, in fProxyMxN Q, in fProxyMxN R,
                              in fProxyN uLo, in fProxyN uHi,
                              in fProxyMxN P, in fProxyMxN S,
                              in fProxyMxN C, in fProxyN d, fProxy rho1, fProxy rho2,
                              in fProxyMxN Kstab)
        {
            if (n <= 0 || m <= 0 || N <= 0)
                throw new ArgumentException("fProxyMPCState: n, m, N must all be >= 1");
            if (!A.IsSquare || A.M_Rows != n) throw new ArgumentException("fProxyMPCState: A must be n x n");
            if (B.M_Rows != n || B.N_Cols != m) throw new ArgumentException("fProxyMPCState: B must be n x m");
            if (!Q.IsSquare || Q.M_Rows != n) throw new ArgumentException("fProxyMPCState: Q must be n x n");
            if (!R.IsSquare || R.M_Rows != m) throw new ArgumentException("fProxyMPCState: R must be m x m");
            if (uLo.N != m || uHi.N != m) throw new ArgumentException("fProxyMPCState: uLo/uHi must have length m");
            for (int i = 0; i < m; i++)
                if (uLo[i] > uHi[i]) throw new ArgumentException("fProxyMPCState: uLo must be <= uHi componentwise");

            bool hasP = P.IsCreated;
            if (hasP && (!P.IsSquare || P.M_Rows != n)) throw new ArgumentException("fProxyMPCState: P must be n x n");
            bool hasDeltaU = S.IsCreated;
            if (hasDeltaU && (!S.IsSquare || S.M_Rows != m)) throw new ArgumentException("fProxyMPCState: S must be m x m");
            bool hasSoftRows = C.IsCreated;
            int nSoftPerStageLocal = 0;
            if (hasSoftRows)
            {
                if (C.N_Cols != n || C.M_Rows < 1) throw new ArgumentException("fProxyMPCState: C must be k x n, k >= 1");
                if (d.N != C.M_Rows) throw new ArgumentException("fProxyMPCState: d.N must equal C.M_Rows");
                if (!math.isfinite(rho1) || rho1 < (fProxy)0 || !math.isfinite(rho2) || rho2 < (fProxy)0)
                    throw new ArgumentException("fProxyMPCState: rho1/rho2 must be finite and non-negative");
                nSoftPerStageLocal = C.M_Rows;
            }
            bool hasPrestab = Kstab.IsCreated;
            if (hasPrestab && (Kstab.M_Rows != m || Kstab.N_Cols != n)) throw new ArgumentException("fProxyMPCState: Kstab must be m x n");
            if (hasDeltaU && hasPrestab)
                throw new ArgumentException("fProxyMPCState: deltaU weighting (S) and prestabilization (Kstab) cannot both be enabled");

            this.n = n; this.m = m; this.N = N;
            nu = N * m;
            nSoftPerStage = nSoftPerStageLocal;
            nSlack = N * nSoftPerStage;
            nz = nu + nSlack;
            int nPrestabRows = hasPrestab ? 2 * N * m : 0;
            nGeneral = N * nSoftPerStage + nPrestabRows;
            this.hasDeltaU = hasDeltaU; this.hasSoftRows = hasSoftRows; this.hasPrestab = hasPrestab;
            _populated = new NativeReference<int>(allocator);   // zero-initialised => not populated

            this.A = new fProxyMxN(in A, allocator);
            this.B = new fProxyMxN(in B, allocator);
            this.uLo = new fProxyN(in uLo, allocator);
            this.uHi = new fProxyN(in uHi, allocator);
            this.S = hasDeltaU ? new fProxyMxN(in S, allocator) : default;
            this.C = hasSoftRows ? new fProxyMxN(in C, allocator) : default;
            this.d = hasSoftRows ? new fProxyN(in d, allocator) : default;
            this.rho1 = rho1; this.rho2 = rho2;
            this.Kstab = hasPrestab ? new fProxyMxN(in Kstab, allocator) : default;

            // ---- terminal cost + tail gain: LQR.lqr's own PUBLIC warm overload exposes the
            // converged Riccati solution directly (state.S), so no internal Control access is needed
            // here (unlike Kalman.steadyStateGain, which reuses Riccati.dare directly because no
            // public entry point already produces what it needs). ----
            Kinf = new fProxyMxN(m, n, allocator);
            var lqrState = new fProxyLQRState(n, Allocator.Temp);
            var lqrInfo = LQR.lqr(in A, in B, in Q, in R, ref Kinf, ref lqrState);
            if (lqrInfo.status != RiccatiStatus.Converged)
            {
                lqrState.Dispose(); Kinf.Dispose();
                this.A.Dispose(); this.B.Dispose(); this.uLo.Dispose(); this.uHi.Dispose();
                if (hasDeltaU) this.S.Dispose();
                if (hasSoftRows) { this.C.Dispose(); this.d.Dispose(); }
                if (hasPrestab) this.Kstab.Dispose();
                _populated.Dispose();
                throw new ArgumentException("fProxyMPCState: terminal DARE did not converge -- (A,B) must be stabilizable");
            }
            fProxyMxN Pterm = hasP ? P : lqrState.S;

            // ---- closed-loop (or plain) condensing dynamics ----
            var Acond = new fProxyMxN(n, n, Allocator.Temp);
            if (hasPrestab)
            {
                var BK = new fProxyMxN(n, n, Allocator.Temp);
                Blas.dot(in B, in Kstab, ref BK);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        Acond[i, j] = A[i, j] - BK[i, j];
                BK.Dispose();
            }
            else
            {
                Acond.CopyFrom(in A);
            }

            // ---- Phi (N*n x n): Phi block k = Acond^(k+1) ----
            Phi = new fProxyMxN(N * n, n, allocator, true);
            var Ak = new fProxyMxN(n, n, Allocator.Temp);
            var AkNext = new fProxyMxN(n, n, Allocator.Temp);
            Ak.CopyFrom(in Acond);
            for (int k = 0; k < N; k++)
            {
                if (k > 0)
                {
                    Blas.dot(in Acond, in Ak, ref AkNext);
                    Ak.CopyFrom(in AkNext);
                }
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        Phi[k * n + i, j] = Ak[i, j];
            }

            // ---- Gamma (N*n x nu): block (k,j) = Acond^(k-j) @ B for j <= k, else 0 ----
            Gamma = new fProxyMxN(N * n, nu, allocator);   // zero-initialized
            var prevBlock = new fProxyMxN(n, m, Allocator.Temp);
            var newBlock = new fProxyMxN(n, m, Allocator.Temp);
            for (int k = 0; k < N; k++)
            {
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < m; j++)
                        Gamma[k * n + i, k * m + j] = B[i, j];

                for (int jb = 0; jb < k; jb++)
                {
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < m; j++)
                            prevBlock[i, j] = Gamma[(k - 1) * n + i, jb * m + j];
                    Blas.dot(in Acond, in prevBlock, ref newBlock);
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < m; j++)
                            Gamma[k * n + i, jb * m + j] = newBlock[i, j];
                }
            }
            prevBlock.Dispose(); newBlock.Dispose(); AkNext.Dispose(); Ak.Dispose(); Acond.Dispose();

            // ---- QGamma (N*n x nu) = Qbar @ Gamma, blockwise (Qbar = blockdiag(Q,...,Q,P)) ----
            var QGamma = new fProxyMxN(N * n, nu, Allocator.Temp);
            for (int k = 0; k < N; k++)
            {
                bool terminal = k == N - 1;
                for (int i = 0; i < n; i++)
                {
                    for (int col = 0; col < nu; col++)
                    {
                        fProxy sum = (fProxy)0;
                        for (int p = 0; p < n; p++)
                            sum += (terminal ? Pterm[i, p] : Q[i, p]) * Gamma[k * n + p, col];
                        QGamma[k * n + i, col] = sum;
                    }
                }
            }

            var H_UU = new fProxyMxN(nu, nu, Allocator.Temp);
            Blas.dotSym(in Gamma, in QGamma, ref H_UU);   // Γᵀ(Q̄Γ), symmetric since Q̄ is
            GtQbar = new fProxyMxN(nu, N * n, allocator);
            Blas.trans(in QGamma, ref GtQbar);
            QGamma.Dispose();
            lqrState.Dispose();   // Pterm (if auto) is no longer needed past this point

            // Rbar (block-diagonal R repeated N times): for the PLAIN (non-prestabilized) case u_k ==
            // v_k, so this is simply added block-diagonally. For the PRESTABILIZED case, the physical
            // input is an AFFINE function of V (u_k = M_row_k @ V + c_k, c_k = -KPhiPre_row_k @ x0 --
            // built below), so the u_k^T R u_k cost term expands to M^T Rbar M (a genuinely different,
            // non-block-diagonal matrix) plus a per-call gradient correction (Rcross) -- naive Rbar-on-v
            // silently drops the -Kstab*x_k cross-coupling entirely (see OP/DEVLOG.md).
            fProxyMxN M = default, KPhiPre = default;
            Rcross = default;   // unconditional -- the hasPrestab branch below overwrites it; struct
                                 // definite-assignment requires every field set on every path
            if (!hasPrestab)
            {
                for (int k = 0; k < N; k++)
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < m; j++)
                            H_UU[k * m + i, k * m + j] += R[i, j];
            }
            else
            {
                // u_k = M_row_k @ V + c_k, c_k = -KPhiPre_row_k @ x0. Row block 0: x_0 = x0 EXACTLY (no
                // V-coupling at all -- M stays identity there, KPhiPre = Kstab directly). Row block
                // k >= 1: x_k = Phi/Gamma BLOCK (k-1) (that block's own convention is x_{(k-1)+1} = x_k
                // -- see this file's header) -- using block k instead (x_{k+1}) is the off-by-one this
                // file's DEVLOG documents fixing.
                M = new fProxyMxN(nu, nu, Allocator.Temp);   // zero-initialized
                for (int i = 0; i < nu; i++) M[i, i] = (fProxy)1;
                KPhiPre = new fProxyMxN(nu, n, Allocator.Temp);   // zero-initialized

                for (int i = 0; i < m; i++)
                    for (int q = 0; q < n; q++)
                        KPhiPre[i, q] = Kstab[i, q];

                for (int k = 1; k < N; k++)
                {
                    for (int i = 0; i < m; i++)
                    {
                        for (int col = 0; col < nu; col++)
                        {
                            fProxy sum = (fProxy)0;
                            for (int p = 0; p < n; p++) sum += Kstab[i, p] * Gamma[(k - 1) * n + p, col];
                            M[k * m + i, col] -= sum;
                        }
                        for (int q = 0; q < n; q++)
                        {
                            fProxy sum = (fProxy)0;
                            for (int p = 0; p < n; p++) sum += Kstab[i, p] * Phi[(k - 1) * n + p, q];
                            KPhiPre[k * m + i, q] = sum;
                        }
                    }
                }

                // Rbar is block-diagonal (N copies of R), so RbarM = Rbar @ M is formed blockwise —
                // row block k is R (m x m) times M's row block k, one unit-stride axpy per (i,p) —
                // instead of materializing the dense nu x nu Rbar and paying a full GEMM against it.
                // M^T Rbar M is symmetric by construction, so it comes from the symmetric kernel
                // (half the GEMM); M^T Rbar = (Rbar M)^T (Rbar symmetric) feeds Rcross directly.
                var RM = new fProxyMxN(nu, nu, Allocator.Temp);   // zero-initialized
                unsafe
                {
                    fProxy* Mp = M.Data.Ptr;
                    fProxy* RMp = RM.Data.Ptr;
                    for (int k = 0; k < N; k++)
                        for (int i = 0; i < m; i++)
                        {
                            fProxy* dst = RMp + (long)(k * m + i) * nu;
                            for (int p = 0; p < m; p++)
                                UnsafeOP.axpy(dst, Mp + (long)(k * m + p) * nu, R[i, p], nu);
                        }
                }

                var MtRbarM = new fProxyMxN(nu, nu, Allocator.Temp);
                Blas.dotSym(in M, in RM, ref MtRbarM);                    // M^T @ (Rbar M), symmetric
                for (int i = 0; i < nu; i++)
                    for (int j = 0; j < nu; j++)
                        H_UU[i, j] += MtRbarM[i, j];

                Rcross = new fProxyMxN(nu, n, allocator);
                Blas.dot(in RM, in KPhiPre, ref Rcross, transposeA: true); // (Rbar M)^T @ KPhiPre = M^T Rbar KPhiPre
                for (int i = 0; i < nu; i++)
                    for (int j = 0; j < n; j++)
                        Rcross[i, j] = (fProxy)(-2) * Rcross[i, j];

                MtRbarM.Dispose(); RM.Dispose();
            }

            // deltaU blocks (tridiagonal in the input blocks) -- see this file's header comment
            if (hasDeltaU)
            {
                for (int k = 0; k < N; k++)
                {
                    fProxy scale = k < N - 1 ? (fProxy)2 : (fProxy)1;
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < m; j++)
                            H_UU[k * m + i, k * m + j] += scale * S[i, j];
                }
                for (int k = 1; k < N; k++)
                {
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < m; j++)
                        {
                            H_UU[k * m + i, (k - 1) * m + j] -= S[i, j];
                            H_UU[(k - 1) * m + i, k * m + j] -= S[i, j];
                        }
                }
            }

            H = new fProxyMxN(nz, nz, allocator);   // zero-initialized
            for (int i = 0; i < nu; i++)
                for (int j = 0; j < nu; j++)
                    H[i, j] = (fProxy)2 * H_UU[i, j];
            if (hasSoftRows)
                for (int j = 0; j < nSlack; j++)
                    H[nu + j, nu + j] = (fProxy)2 * rho2;
            H_UU.Dispose();
            Riccati.SymmetrizeInPlace(ref H);   // roundoff hygiene, reusing Riccati's own helper directly

            // ---- general rows: soft state rows, then (if prestabilized) input-bound rows ----
            Arows = new fProxyMxN(nGeneral, nz, allocator);
            qCoupling = new fProxyMxN(nGeneral, n, allocator);
            rowConstRHS = new fProxyN(nGeneral, allocator);
            // Exactly nGeneral (NOT padded to >= 1): this array is passed straight through as
            // QP.qpActiveSetCoreWarm's own `senses` parameter, which validates senses.Length ==
            // A.M_Rows == nGeneral exactly -- see bScratch's own comment below for the same reasoning.
            senses = new NativeArray<ConstraintSense>(nGeneral, allocator);

            int rowIdx = 0;
            if (hasSoftRows)
            {
                for (int k = 0; k < N; k++)
                {
                    for (int si = 0; si < nSoftPerStage; si++)
                    {
                        for (int col = 0; col < nu; col++)
                        {
                            fProxy sum = (fProxy)0;
                            for (int p = 0; p < n; p++) sum += C[si, p] * Gamma[k * n + p, col];
                            Arows[rowIdx, col] = sum;
                        }
                        Arows[rowIdx, nu + rowIdx] = (fProxy)(-1);
                        for (int q = 0; q < n; q++)
                        {
                            fProxy sum = (fProxy)0;
                            for (int p = 0; p < n; p++) sum += C[si, p] * Phi[k * n + p, q];
                            qCoupling[rowIdx, q] = sum;
                        }
                        rowConstRHS[rowIdx] = d[si];
                        senses[rowIdx] = ConstraintSense.LessEqual;
                        rowIdx++;
                    }
                }
            }
            if (hasPrestab)
            {
                // Reuses M/KPhiPre built above (Rbar section): u_k = M_row_k @ V + c_k,
                // c_k = -KPhiPre_row_k @ x0 -- the SAME affine map the cost correction uses, so the row
                // and the cost agree on stage k's physical input by construction (no re-derivation, no
                // chance of the two drifting apart under a future edit).
                for (int k = 0; k < N; k++)
                {
                    for (int i = 0; i < m; i++)
                    {
                        int row = k * m + i;

                        // upper: u_k[i] <= uHi[i]  =>  M_row @ V <= uHi[i] + KPhiPre_row @ x0
                        for (int col = 0; col < nu; col++) Arows[rowIdx, col] = M[row, col];
                        for (int q = 0; q < n; q++) qCoupling[rowIdx, q] = -KPhiPre[row, q];
                        rowConstRHS[rowIdx] = uHi[i];
                        senses[rowIdx] = ConstraintSense.LessEqual;
                        rowIdx++;

                        // lower: -u_k[i] <= -uLo[i]  =>  -M_row @ V <= -uLo[i] + KPhiPre_row @ x0
                        for (int col = 0; col < nu; col++) Arows[rowIdx, col] = -M[row, col];
                        for (int q = 0; q < n; q++) qCoupling[rowIdx, q] = KPhiPre[row, q];
                        rowConstRHS[rowIdx] = -uLo[i];
                        senses[rowIdx] = ConstraintSense.LessEqual;
                        rowIdx++;
                    }
                }
                KPhiPre.Dispose(); M.Dispose();
            }

            // ---- box bounds ----
            xl = new fProxyN(nz, allocator);
            xu = new fProxyN(nz, allocator);
            fProxy INF = (fProxy)1e30;
            if (!hasPrestab)
            {
                for (int k = 0; k < N; k++)
                    for (int i = 0; i < m; i++)
                    {
                        xl[k * m + i] = uLo[i];
                        xu[k * m + i] = uHi[i];
                    }
            }
            else
            {
                for (int j = 0; j < nu; j++) { xl[j] = -INF; xu[j] = INF; }
            }
            if (hasSoftRows)
                for (int j = 0; j < nSlack; j++) { xl[nu + j] = (fProxy)0; xu[nu + j] = INF; }

            // ---- persistent scratch / warm-start carry ----
            z = new fProxyN(nz, allocator);
            wstatus = new NativeArray<byte>(nGeneral + nz, allocator);
            cScratch = new fProxyN(nz, allocator);
            // Exactly nGeneral (not padded): passed straight through as qpActiveSetCoreWarm's own `b`
            // parameter every solve call, which validates b.N == A.M_Rows == nGeneral exactly.
            bScratch = new fProxyN(nGeneral, allocator);
            uPlan = new fProxyN(nu, allocator);
            xTrajScratch = new fProxyN(N * n, allocator);
            // Persistent condensed-QP factorization + reduced space (n = nz), carried across ticks by
            // QP.qpActiveSetCoreWarmPersistent. qpMeta[0]=factorValid stays 0, qpMeta[6]=stale=1 until the
            // first solve fills them (native-backed so the scalar metadata survives job by-value copies).
            qpFactor = fProxyQPFactorState.Create(nz, allocator);
            qpReduced = fProxyQPReducedState.Create(nz, allocator);
            qpMeta = new NativeArray<int>(8, allocator);
            qpMeta[6] = 1;   // stale
        }

        /// <summary>Convenience overload: no terminal-P override, no deltaU, no soft rows, no
        /// prestabilization -- the common case.</summary>
        public fProxyMPCState(int n, int m, int N, Allocator allocator,
                              in fProxyMxN A, in fProxyMxN B, in fProxyMxN Q, in fProxyMxN R,
                              in fProxyN uLo, in fProxyN uHi)
            : this(n, m, N, allocator, in A, in B, in Q, in R, in uLo, in uHi,
                   default, default, default, default(fProxyN), (fProxy)0, (fProxy)0, default)
        {
        }

        /// <summary>Convenience overload: explicit terminal cost <paramref name="P"/>, nothing else.</summary>
        public fProxyMPCState(int n, int m, int N, Allocator allocator,
                              in fProxyMxN A, in fProxyMxN B, in fProxyMxN Q, in fProxyMxN R,
                              in fProxyN uLo, in fProxyN uHi, in fProxyMxN P)
            : this(n, m, N, allocator, in A, in B, in Q, in R, in uLo, in uHi,
                   in P, default, default, default(fProxyN), (fProxy)0, (fProxy)0, default)
        {
        }

        /// <summary>Convenience overload: soft state rows <paramref name="C"/>/<paramref name="d"/> using
        /// the library's default exact-penalty weights (<see cref="MPC.DEFAULT_RHO1"/>/
        /// <see cref="MPC.DEFAULT_RHO2"/>) -- nothing else.</summary>
        public fProxyMPCState(int n, int m, int N, Allocator allocator,
                              in fProxyMxN A, in fProxyMxN B, in fProxyMxN Q, in fProxyMxN R,
                              in fProxyN uLo, in fProxyN uHi, in fProxyMxN C, in fProxyN d)
            : this(n, m, N, allocator, in A, in B, in Q, in R, in uLo, in uHi,
                   default, default, in C, in d, (fProxy)MPC.DEFAULT_RHO1, (fProxy)MPC.DEFAULT_RHO2, default)
        {
        }

        /// <summary>Convenience overload: soft state rows with explicit <paramref name="rho1"/>/
        /// <paramref name="rho2"/> -- nothing else.</summary>
        public fProxyMPCState(int n, int m, int N, Allocator allocator,
                              in fProxyMxN A, in fProxyMxN B, in fProxyMxN Q, in fProxyMxN R,
                              in fProxyN uLo, in fProxyN uHi, in fProxyMxN C, in fProxyN d, fProxy rho1, fProxy rho2)
            : this(n, m, N, allocator, in A, in B, in Q, in R, in uLo, in uHi,
                   default, default, in C, in d, rho1, rho2, default)
        {
        }

        // NOTE: deltaU-only and prestabilization-only convenience overloads are deliberately NOT
        // provided -- both would need an extra fProxyMxN-typed parameter (S / Kstab) in the SAME
        // position as the "with terminal P" overload's P, which is a genuine C# overload-signature
        // collision (parameter NAMES don't participate in overload resolution, only types), not a style
        // choice. Use the full constructor directly for either feature (pass `default` for the other
        // optional matrix params).
    }
}
