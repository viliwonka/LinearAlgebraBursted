#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    // ================================================================================================
    // Convex quadratic programming (docs/draft-spec-qp.md) -- a HiGHS-style dense primal null-space
    // active-set method, ported per Nocedal & Wright, Numerical Optimization (2nd ed.), ch. 16
    // ("Quadratic Programming"), specifically 16.2 "The Null-Space Method" and 16.5 "Active-Set
    // Methods for Convex QP".
    //
    // STAGE 1 (this file, current): the EQUALITY-constrained QP core (EQP) -- a FIXED working set,
    // algorithm steps 2-3 only (the null-space Newton step + multiplier recovery). No add/drop loop,
    // no ratio test, no phase 1 -- those are stage 2 (inequality loop) and stage 3 (phase 1 +
    // hardening), layered on top of the SAME kernel functions below without touching them (see each
    // function's own doc comment for which future stage calls it).
    //
    // Problem solved by eqpSolve / eqpNullSpaceStep:
    //
    //     minimize    ½xᵀQx + cᵀx
    //     subject to  A_W x = b_W        (A_W: k x n, k <= n, rows independent -- "the working set")
    //
    // Q is symmetric PSD (v1 contract, draft-spec-qp.md open question 3): a singular reduced Hessian
    // is regularized (δ·‖Q‖∞·I retry) rather than handled via negative-curvature machinery, matching
    // HiGHS's own v1 scope. Indefinite Q is out of scope (NP-hard in general).
    //
    // ---- Why one null-space Newton step is EXACT for an equality-constrained quadratic ----
    //
    // Parameterize every feasible point as x = x0 + Zy, x0 any point with A_W x0 = b_W, Z an
    // orthonormal basis for null(A_W) (A_W Z = 0, so A_W x = A_W x0 = b_W for ANY y). Substituting,
    // the equality-constrained problem becomes the UNCONSTRAINED reduced problem
    //     minimize_y   ½yᵀ(ZᵀQZ)y + (Zᵀg(x0))ᵀy + const,   g(x0) = Qx0 + c,
    // an ordinary (unconstrained) quadratic in y with Hessian H_Z = ZᵀQZ and gradient
    // Zᵀg(x0) + H_Z y. For ANY quadratic, Newton's method reaches the exact minimizer in ONE step
    // regardless of the starting point (the model IS the function), so solving H_Z y = -Zᵀg(x0) and
    // setting x1 = x0 + Zy lands exactly on the equality-constrained optimum -- no line search, no
    // iteration (Nocedal & Wright ch. 16.2, eq. 16.16-16.19; the "null-space method").
    //
    // ---- Keeping Q (QR's orthogonal factor, not this file's Q the Hessian -- unfortunate letter
    // clash inherited from the spec/textbook, disambiguated by context below) implicit ----
    //
    // QR.decompInPlace's public API FORMS the dense (n x k) "thin" Q1 -- it factors AND reconstructs
    // in one call, with no split entry point. To avoid ever materializing that n x k matrix (per the
    // spec's explicit ask), this file bypasses QR.decompInPlace entirely and instead drives QR's own
    // per-step primitives directly (QR.genHouseholder / QR.applyReflectorRight, both `internal` --
    // the same functions decompInPlace itself is built from): FactorWorkingSetTranspose below is
    // exactly decompInPlace's factorization half (store R + stash each Householder vector into A_Wᵀ's
    // own columns), replicated rather than called through the public API specifically so the
    // reconstruction half never runs. Two more primitives close the loop without ever forming Q1:
    //   * ApplyWorkingSetQtForward -- Q_full = H_0 H_1 ... H_{k-1} is an n x n orthogonal matrix (each
    //     reflector acts on the full ambient n-dimensional space; only k of them exist because A_Wᵀ
    //     has k columns), so Q_fullᵀ = H_{k-1} ... H_0 and a FORWARD sweep (d = 0..k-1) of the k
    //     stashed reflectors over any n-vector v computes Q_fullᵀv = (Q1ᵀv ; Zᵀv) in one pass -- top k
    //     entries and bottom n-k entries. This is the exact trick QR.solveInPlace already uses for its
    //     `b` argument (computing Qᵀb without ever forming Q), generalized and replayed from STORED
    //     reflectors instead of freshly-generated ones. It replaces BOTH QR.decompSolve's "Qᵀg, then
    //     R-solve" (used for the multiplier recovery) AND the reduced gradient Zᵀg -- one sweep gives
    //     both halves.
    //   * FormNullSpaceBasis -- Z itself (n x (n-k), needed explicitly because the reduced Hessian
    //     ZᵀQZ and the step p = Zy are GEMM/GEMV operands) is formed by REVERSE-sweeping (d = k-1..0)
    //     the same stashed reflectors over the seed [0; I_{n-k}] -- exactly QR.decompInPlace's own
    //     Q-reconstruction phase, just seeded with the TRAILING identity block instead of the leading
    //     one and targeting a separate n x (n-k) buffer instead of overwriting A_Wᵀ. Z is smaller than
    //     the full n x n Q (by construction, only the n-k null-space columns), so this is the
    //     documented exception to "don't form dense Q" -- forming Z, not Q.
    //
    // ---- Structuring for stage 2-3 reuse ----
    //
    // Every function below is `internal static` (not a buried local), matching the structuring rule
    // LP.RevisedSimplex.fProxy.cs set for LP.DualSimplex.fProxy.cs: the stage-2 active-set loop (ratio
    // test, add/drop, Dantzig pricing) will call eqpNullSpaceStep (or its constituent pieces) once per
    // iteration, re-factoring A_Wᵀ from scratch after every working-set change -- see
    // FactorWorkingSetTranspose's own doc comment for that cost and why it is deliberately NOT
    // incremental (v1 scope decision, draft-spec-qp.md "Judgment").
    // ================================================================================================
    public static partial class QP
    {
        /// <summary>
        /// Solve the EQUALITY-constrained QP  min ½xᵀQx + cᵀx  s.t. A_W x = b_W  EXACTLY, from
        /// scratch: reaches a feasible point via <see cref="LQ.minNormSolve(ref fProxyMxN, ref fProxyN, ref fProxyN)"/>
        /// (A_W is k x n, k &lt;= n, independent rows -- exactly the "wide, full row rank"
        /// underdetermined system that targets), then takes ONE exact null-space Newton step -- see
        /// <see cref="eqpNullSpaceStep"/> and this file's header comment for why one step suffices.
        ///
        /// STAGE 1 of docs/draft-spec-qp.md: a FIXED working set (algorithm steps 2-3 only). INTERNAL:
        /// the public surface for QP is the future inequality-constrained <c>QP.solve</c> facade
        /// (stage 2-3); this is the reusable EQP kernel entry it will call once the active set is
        /// pinned down for a given iteration. Hand-written tests reach this via InternalsVisibleTo
        /// (BurstLinearAlgebra.Tests, see AssemblyInfo.cs), the same route as
        /// LP.ladFrischNewtonCore / LadFrischNewtonQuantileTests.cs.
        /// </summary>
        /// <param name="Q">Symmetric PSD Hessian, n x n. Only referenced via matrix products (Qx,
        /// ZᵀQZ) -- not verified to be exactly symmetric in storage (both triangles are read).</param>
        /// <param name="c">Linear cost, length n.</param>
        /// <param name="A_W">Working-set constraint matrix, k x n (1 &lt;= k &lt;= n), rows
        /// independent. Not modified.</param>
        /// <param name="b_W">Working-set right-hand side, length k. Not modified.</param>
        /// <param name="x">Output only; prior contents ignored; safe to allocate with uninit: true.
        /// Solution, length n.</param>
        /// <param name="lambda">Output only; prior contents ignored; safe to allocate with uninit:
        /// true. Multipliers for A_W's rows (A_Wᵀlambda = Qx+c at the optimum), length k. NOT written
        /// on <see cref="QPStatus.Unbounded"/>.</param>
        internal static QPInfo eqpSolve(in fProxyMxN Q, in fProxyN c, in fProxyMxN A_W, in fProxyN b_W,
                                        ref fProxyN x, ref fProxyN lambda)
        {
            // LQ.minNormSolve does not modify its A/b arguments (they are copied into its own working
            // buffers internally) -- a local struct copy (same handle, zero-cost) is enough to satisfy
            // its `ref` parameters from our `in` ones.
            var A_W_rw = A_W;
            var b_W_rw = b_W;
            LQ.minNormSolve(ref A_W_rw, ref b_W_rw, ref x);

            return eqpNullSpaceStep(in Q, in c, in A_W, in b_W, ref x, ref lambda);
        }

        /// <summary>
        /// Given an ALREADY FEASIBLE <paramref name="x"/> (A_W x = b_W) and a fixed working set
        /// <paramref name="A_W"/>, take the exact null-space Newton step to the equality-constrained
        /// optimum for THIS working set: assemble A_Wᵀ, factor it (<see cref="FactorWorkingSetTranspose"/>),
        /// form the reduced gradient and Z implicitly from the same factor
        /// (<see cref="ApplyWorkingSetQtForward"/>, <see cref="FormNullSpaceBasis"/>), solve the reduced
        /// Hessian system with regularization on breakdown (<see cref="SolveReducedNewtonStep"/>), take
        /// the step, and recover the multipliers from the SAME QR factor (R-solve of the reduced
        /// gradient's leading k entries). Updates <paramref name="x"/> and <paramref name="lambda"/> in
        /// place; returns the diagnostics (see <see cref="QPInfo"/>).
        ///
        /// STAGE 1 (fixed working set): called once by <see cref="eqpSolve"/> and that is the whole
        /// algorithm (draft-spec-qp.md steps 2-3, no ratio test / add-drop loop -- that is stage 2).
        /// STAGE 2-3 (future): the active-set loop will call this once PER ITERATION with the CURRENT
        /// working set (which changes as constraints are added/dropped) -- re-factoring A_Wᵀ from
        /// scratch every call, matching the spec's v1 judgment that an incremental QR update is not
        /// worth porting at this library's dense target sizes (see this file's header comment); stage
        /// 2 will need to intercept BETWEEN computing the step and applying it in full (the ratio test
        /// may truncate the step to alpha*p, alpha &lt; 1), which this stage-1 version does not do --
        /// expect that seam to require splitting the "compute p" and "apply p, recover multipliers"
        /// halves of this function when stage 2 is built.
        /// </summary>
        /// <param name="Q">Symmetric PSD Hessian, n x n.</param>
        /// <param name="c">Linear cost, length n.</param>
        /// <param name="A_W">Working-set constraint matrix, k x n (1 &lt;= k &lt;= n), rows
        /// independent. Not modified.</param>
        /// <param name="b_W">Working-set right-hand side, length k. Not modified. Used only for the
        /// cheap feasibility-residual diagnostic (the step itself never needs it -- it stays within
        /// null(A_W) by construction).</param>
        /// <param name="x">On entry a feasible point (A_W x = b_W); on exit the equality-constrained
        /// optimum (unchanged from entry on <see cref="QPStatus.Unbounded"/>).</param>
        /// <param name="lambda">Output only; prior contents ignored. Multipliers, length k. NOT
        /// written on <see cref="QPStatus.Unbounded"/>.</param>
        internal static QPInfo eqpNullSpaceStep(in fProxyMxN Q, in fProxyN c, in fProxyMxN A_W, in fProxyN b_W,
                                                ref fProxyN x, ref fProxyN lambda)
        {
            int n = Q.M_Rows;
            int k = A_W.M_Rows;
            int nz = n - k;

            if (!Q.IsSquare)
                throw new ArgumentException("QP.eqpNullSpaceStep: Q must be square");
            if (A_W.N_Cols != n)
                throw new ArgumentException("QP.eqpNullSpaceStep: A_W.N_Cols must equal Q.M_Rows");
            if (c.N != n)
                throw new ArgumentException("QP.eqpNullSpaceStep: c.N must equal Q.M_Rows");
            if (b_W.N != k)
                throw new ArgumentException("QP.eqpNullSpaceStep: b_W.N must equal A_W.M_Rows");
            if (x.N != n)
                throw new ArgumentException("QP.eqpNullSpaceStep: x.N must equal Q.M_Rows");
            if (lambda.N != k)
                throw new ArgumentException("QP.eqpNullSpaceStep: lambda.N must equal A_W.M_Rows");
            // k == 0 (an empty working set) is deliberately unsupported in stage 1 -- it would need a
            // separate Z-is-implicitly-identity fast path (no A_Wᵀ to factor at all) that stage 1's
            // test matrix (draft-spec-qp.md: k in {1, n/4, n/2, n-1}) never exercises; revisit if
            // stage 2's active-set loop can transiently reach an empty working set.
            if (k < 1 || k > n)
                throw new ArgumentException("QP.eqpNullSpaceStep: A_W.M_Rows (k) must be between 1 and Q.M_Rows (n) inclusive");

            fProxy normInfQ = Norms.LInf(in Q);
            fProxy zeroThreshold = Consts.fProxyZeroThreshold * math.max(normInfQ, (fProxy)1);

            var AWT = new fProxyMxN(n, k, Allocator.Temp, true);
            Blas.trans(in A_W, ref AWT);

            var R = new fProxyMxN(k, k, Allocator.Temp, true);
            var u = new fProxyN(n, Allocator.Temp, false);
            var w = new fProxyN(math.max(k, math.max(nz, 1)), Allocator.Temp, false);

            FactorWorkingSetTranspose(ref AWT, ref R, ref u, ref w, zeroThreshold);

            var Qx = new fProxyN(n, Allocator.Temp, false);
            var g = new fProxyN(n, Allocator.Temp, false);
            Blas.dot(in Q, in x, ref Qx);
            for (int i = 0; i < n; i++) g[i] = Qx[i] + c[i];

            // g[0..k) <- Q1ᵀg, g[k..n) <- Zᵀg  (at the feasible start x) -- see file header.
            ApplyWorkingSetQtForward(ref AWT, ref g, ref u, k);

            QPStatus status = QPStatus.Optimal;

            if (nz == 0)
            {
                // Fully determined working set (k == n): A_W is n x n and full rank, so x is ALREADY
                // pinned down exactly by the constraints alone -- there is no null space to step
                // through. lambda solves A_Wᵀlambda = g directly (R-solve of the now-full-length
                // transformed g).
                for (int i = 0; i < k; i++) lambda[i] = g[i];
                Blas.triUpper(ref R, ref lambda);
            }
            else
            {
                var gz = new fProxyN(nz, Allocator.Temp, false);
                for (int j = 0; j < nz; j++) gz[j] = g[k + j];

                var Z = new fProxyMxN(n, nz, Allocator.Temp, true);
                FormNullSpaceBasis(ref AWT, ref Z, ref u, ref w, k);

                var QZ = new fProxyMxN(n, nz, Allocator.Temp, true);
                var Hz = new fProxyMxN(nz, nz, Allocator.Temp, true);
                var y = new fProxyN(nz, Allocator.Temp, false);

                var choInfo = SolveReducedNewtonStep(in Q, ref Z, ref QZ, ref Hz, ref gz, ref y, normInfQ, out bool regularized);

                bool unbounded = !choInfo.Solved;
                if (!unbounded && regularized)
                {
                    // Descent-direction check on the REGULARIZED step (draft-spec-qp.md step 2): a
                    // successful regularized solve mathematically guarantees pᵀg = yᵀgz <= 0 (H_Z+δI
                    // is PD, so yᵀ(H_Z+δI)y >= 0 => -yᵀgz >= 0), so in exact arithmetic this branch is
                    // a formality that should never fire for a genuinely PSD Q -- it exists as the
                    // spec's documented catch-all for "Q singular along an unbounded ray", detectable
                    // only if roundoff or a not-quite-PSD Q defeats the guarantee above.
                    double pdotg = 0;
                    for (int j = 0; j < nz; j++) pdotg += (double)y[j] * (double)gz[j];
                    if (pdotg > (double)zeroThreshold)
                        unbounded = true;
                }

                if (unbounded)
                {
                    status = QPStatus.Unbounded;
                }
                else
                {
                    var p = new fProxyN(n, Allocator.Temp, false);
                    Blas.dot(in Z, in y, ref p);
                    for (int i = 0; i < n; i++) x[i] += p[i];
                    p.Dispose();

                    // Recompute g at the NEW x and replay the SAME stashed reflectors -- the
                    // multipliers must satisfy A_Wᵀlambda = g(x_new), not g(x0) (see file header: the
                    // null-space step changes g by Q*(step), which the multiplier solve at x0 alone
                    // cannot account for).
                    Blas.dot(in Q, in x, ref Qx);
                    for (int i = 0; i < n; i++) g[i] = Qx[i] + c[i];
                    ApplyWorkingSetQtForward(ref AWT, ref g, ref u, k);

                    for (int i = 0; i < k; i++) lambda[i] = g[i];
                    Blas.triUpper(ref R, ref lambda);
                }

                y.Dispose(); Hz.Dispose(); QZ.Dispose(); Z.Dispose(); gz.Dispose();
            }

            // Stationarity residual: ‖Zᵀg‖∞ at the FINAL x -- already sitting in g's tail after the
            // last ApplyWorkingSetQtForward call above (both branches leave it there); 0 on Unbounded
            // (no valid step was taken, the residual would be stale) or when nz == 0 (no null space).
            double stationarity = 0;
            if (status == QPStatus.Optimal)
                for (int j = k; j < n; j++)
                    stationarity = math.max(stationarity, math.abs((double)g[j]));

            // Objective + feasibility at the FINAL x. Qx already equals Q*x_final in every path above
            // (nz==0: x never changed after the first Qx; Unbounded: x never changed, same Qx still
            // valid; the normal step path: Qx was refreshed right after x += p) -- no redundant GEMV.
            double objective = 0;
            for (int i = 0; i < n; i++)
                objective += 0.5 * (double)x[i] * (double)Qx[i] + (double)c[i] * (double)x[i];

            var Ax = new fProxyN(k, Allocator.Temp, false);
            Blas.dot(in A_W, in x, ref Ax);
            double feasibility = 0;
            for (int i = 0; i < k; i++)
                feasibility = math.max(feasibility, math.abs((double)Ax[i] - (double)b_W[i]));
            Ax.Dispose();

            g.Dispose(); Qx.Dispose(); w.Dispose(); u.Dispose(); R.Dispose(); AWT.Dispose();

            return new QPInfo
            {
                status = status,
                iterations = 1,
                objective = objective,
                stationarityResidual = stationarity,
                feasibilityResidual = feasibility,
            };
        }

        // Factor A_Wᵀ (AWT, n x k) into R (k x k, upper triangular, caller-allocated and expected
        // zero-initialized -- only entries with c >= r are written) plus k stashed Householder
        // reflectors left in AWT's own columns (row i, column d holds u_d[i] for i >= d) -- exactly
        // decompInPlace's OWN storage convention, WITHOUT its subsequent Q-reconstruction phase (see
        // file header for why: this keeps Q1 implicit). u is scratch of length AWT.M_Rows (= n); w is
        // scratch of length >= AWT.N_Cols (= k). Reuses QR's per-step primitives (QR.genHouseholder /
        // QR.applyReflectorRight, both `internal`) -- not a re-derivation of the Householder math.
        //
        // v1 scope (draft-spec-qp.md "Judgment"): this ALWAYS re-factors A_Wᵀ from scratch. HiGHS
        // instead maintains an incrementally-updated factorization of the working-set basis across
        // add/drop changes -- deliberately not ported here (dense v1 target sizes make an O(n²k)
        // re-factor per change cheap enough, and simple/correct beats incremental/subtle). If stage 2
        // ever needs it, the QRCP downdating machinery (docs/spec-qrcp-downdate.md's rank-1 column
        // downdate) is the natural donor -- it already solves the adjacent problem (removing a column
        // from a QR factor without refactoring) for QRCP; adapting it to A_Wᵀ's row-add/row-drop shape
        // (an add/drop of a WORKING-SET CONSTRAINT is a column add/drop on A_Wᵀ) would be the v2 path.
        internal static void FactorWorkingSetTranspose(ref fProxyMxN AWT, ref fProxyMxN R, ref fProxyN u, ref fProxyN w, fProxy zeroThreshold)
        {
            int n = AWT.M_Rows;
            int k = AWT.N_Cols;

            for (int d = 0; d < k; d++)
            {
                QR.genHouseholder(ref AWT, ref u, d, zeroThreshold);
                QR.applyReflectorRight(ref AWT, ref u, ref w, d);

                R[d, d] = AWT[d, d];
                for (int i = d; i < n; i++)
                    AWT[i, d] = u[i];
            }

            // R's strict upper triangle (c > r): read AFTER the full loop, since row r's entries in
            // columns c > r are finalized by step d = r's own reflector apply and never touched again
            // by any later step (which only ever modifies rows >= its own d > r) -- see file header.
            for (int r = 0; r < k; r++)
            for (int c = r + 1; c < k; c++)
                R[r, c] = AWT[r, c];
        }

        // v[0..k) <- Q1ᵀv, v[k..n) <- Zᵀv, applying the k reflectors STASHED by
        // FactorWorkingSetTranspose (read back from AWT's columns, NOT regenerated) in FORWARD order
        // (d = 0..k-1). See file header for why this single sweep produces both halves at once and why
        // it never needs to form Q1. u is scratch of length AWT.M_Rows (= n), overwritten (used to
        // stage each reflector as it is read back). Mirrors QR.solveInPlace's own `b`-transform loop
        // (plain scalar accumulation, not the matrix-apply's vectorised UnsafeOP.axpy path -- v is a
        // single vector here, not a trailing submatrix).
        internal static void ApplyWorkingSetQtForward(ref fProxyMxN AWT, ref fProxyN v, ref fProxyN u, int k)
        {
            int n = AWT.M_Rows;

            for (int d = 0; d < k; d++)
            {
                for (int r = d; r < n; r++) u[r] = AWT[r, d];

                fProxy dot = 0;
                for (int r = d; r < n; r++) dot += u[r] * v[r];
                for (int r = d; r < n; r++) v[r] -= u[r] * dot;
            }
        }

        // Z (n x (n-k)): an orthonormal basis for null(A_W) -- the trailing n-k columns of the FULL
        // orthogonal factor Q_full = H_0 H_1 ... H_{k-1} -- formed by REVERSE-sweeping (d = k-1..0)
        // the reflectors FactorWorkingSetTranspose stashed in AWT's columns over the seed [0; I_{n-k}]
        // (n x (n-k)). Z must be caller-allocated n x (n-k) and is fully overwritten (any prior
        // contents are irrelevant -- it is zeroed here). u is scratch of length AWT.M_Rows (= n); w is
        // scratch of length >= Z.N_Cols (= n-k).
        //
        // Deliberately does NOT call QR.applyReflectorRight (unlike FactorWorkingSetTranspose above) --
        // that primitive restricts each reflector d's column range to [d, N_Cols), which is exact ONLY
        // for decompInPlace's own reconstruction, seeded with the LEADING identity: there, column j's
        // seed is e_j, and H_i (support on rows >= i) satisfies H_i·e_j = e_j EXACTLY whenever i > j
        // (row j < i is outside u_i's support), so skipping columns < d when applying H_d is a true
        // per-column no-op, not just an optimization. Z's seed is the TRAILING identity block instead
        // (column j = e_{k+j}), and every reflector index d < k <= k+j, so u_d's support (rows >= d)
        // ALWAYS includes row k+j -- H_d is generally NOT a no-op on any column of Z, for any d. The
        // column restriction would therefore silently skip real work (caught by the Stage-1 KKT-oracle
        // check at k=2, n=8: k=1 has only one reflector at d=0, whose "columns >= 0" restriction never
        // actually excludes anything, so the bug is invisible there). ApplyReflectorFullWidth below is
        // the same rank-1 update with that restriction removed -- full column width every step.
        internal static void FormNullSpaceBasis(ref fProxyMxN AWT, ref fProxyMxN Z, ref fProxyN u, ref fProxyN w, int k)
        {
            int n = AWT.M_Rows;
            int nz = Z.N_Cols;

            unsafe { UnsafeUtility.MemClear(Z.Data.Ptr, (long)Z.Data.Length * UnsafeUtility.SizeOf<fProxy>()); }
            for (int j = 0; j < nz; j++)
                Z[k + j, j] = (fProxy)1;

            for (int d = k - 1; d >= 0; d--)
            {
                for (int r = d; r < n; r++) u[r] = AWT[r, d];
                ApplyReflectorFullWidth(ref Z, ref u, ref w, d);
            }
        }

        // Apply Householder reflector H_d = I - u·uᵀ (u supported on rows >= d) to M's FULL column
        // width, rows [d, M_Rows): M[d:, :] -= u·(uᵀ·M[d:, :]). Same two-pass vectorised shape as
        // QR.applyReflectorRightCols (UnsafeOP.axpy, the GEMM pointer path), but WITHOUT that
        // primitive's "columns >= d" restriction -- see FormNullSpaceBasis's doc comment for why that
        // restriction does not generalize to a seed other than the leading identity. w is scratch of
        // length >= M.N_Cols.
        internal static unsafe void ApplyReflectorFullWidth(ref fProxyMxN M, ref fProxyN u, ref fProxyN w, int d)
        {
            int rows = M.M_Rows;
            int cols = M.N_Cols;
            if (cols <= 0) return;

            fProxy* mp = M.Data.Ptr;
            fProxy* up = u.Data.Ptr;
            fProxy* wp = w.Data.Ptr;

            UnsafeUtility.MemClear(wp, (long)cols * UnsafeUtility.SizeOf<fProxy>());
            for (int r = d; r < rows; r++)
                UnsafeOP.axpy(wp, mp + (long)r * cols, up[r], cols);
            for (int r = d; r < rows; r++)
                UnsafeOP.axpy(mp + (long)r * cols, wp, -up[r], cols);
        }

        // Solve the reduced-Hessian Newton system H_Z y = -gz for y, H_Z = ZᵀQZ (nz x nz, PSD since Q
        // is PSD -- two matMatDot-shaped calls via Blas.dot, per draft-spec-qp.md step 2). On a
        // Cholesky breakdown (H_Z numerically singular -- possible even though Q is only PSD, not PD:
        // Q's null space can overlap Z's span), retries ONCE with H_Z + delta*normInfQ*I,
        // delta = sqrt(Consts.fProxyEpsilon) -- a PSD matrix plus a strictly positive multiple of I is
        // always PD, so this retry cannot itself break down for a genuinely PSD Q. QZ/Hz are
        // caller-allocated scratch (n x nz / nz x nz respectively, sized by the caller since it also
        // owns Z); y is the caller-allocated destination (length nz). Returns the CHO status from the
        // (possibly regularized) solve and, via <paramref name="regularized"/>, whether the retry
        // path was taken -- the caller uses that to run the descent-direction / Unbounded check
        // (draft-spec-qp.md step 2's "declare Unbounded" clause), which needs gz and y regardless of
        // which path produced them.
        internal static DirectSolveInfo SolveReducedNewtonStep(in fProxyMxN Q, ref fProxyMxN Z, ref fProxyMxN QZ, ref fProxyMxN Hz,
                                                                ref fProxyN gz, ref fProxyN y, fProxy normInfQ, out bool regularized)
        {
            int nz = y.N;

            Blas.dot(in Q, in Z, ref QZ);
            Blas.dot(in Z, in QZ, ref Hz, transposeA: true);

            for (int j = 0; j < nz; j++) y[j] = -gz[j];
            var info = CHO.solveInPlace(ref Hz, ref y);

            regularized = false;
            if (!info.Solved)
            {
                regularized = true;
                fProxy delta = math.sqrt(Consts.fProxyEpsilon) * math.max(normInfQ, (fProxy)1);

                // CHO.decompInPlace leaves a failed factor PARTIALLY overwritten (documented
                // "destroyed on failure"); rebuild H_Z cleanly from the still-intact Z/QZ before
                // adding the regularizer, rather than trying to patch the partial factor in place.
                Blas.dot(in Z, in QZ, ref Hz, transposeA: true);
                for (int j = 0; j < nz; j++) Hz[j, j] += delta;

                for (int j = 0; j < nz; j++) y[j] = -gz[j];
                info = CHO.solveInPlace(ref Hz, ref y);
            }

            return info;
        }
    }
}
