using System;
using Unity.Collections;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Krylov {

        // ---- btfqmr private row helpers -------------------------------------------------------------
        // btfqmr runs s independent scalar-TFQMR recurrences (one per RHS row) in lockstep, sharing a
        // single ApplyBlock per half-step instead of s separate Apply calls -- see btfqmr's own doc for
        // why this is a PSEUDO-block (batched, non-mixing) design, not a subspace-mixing true block
        // method. Every per-row coefficient (alpha/eta/theta/tau/rho/beta) is an independent scalar
        // stored at index i of a length-s fProxyN; these helpers apply one such scalar to a single row
        // of an s x n block.

        // Y[row,:] += coeff * T[row,:].
        static void RowAddScaled(ref fProxyMxN Y, int row, fProxy coeff, in fProxyMxN T, int n)
        {
            for (int c = 0; c < n; c++) Y[row, c] += coeff * T[row, c];
        }

        // Y[row,:] = coeff*Y[row,:] + T[row,:].
        static void RowScaleAddSelf(ref fProxyMxN Y, int row, fProxy coeff, in fProxyMxN T, int n)
        {
            for (int c = 0; c < n; c++) Y[row, c] = coeff * Y[row, c] + T[row, c];
        }

        // Y[row,:] *= scale.
        static void RowScaleInPlace(ref fProxyMxN Y, int row, fProxy scale, int n)
        {
            for (int c = 0; c < n; c++) Y[row, c] *= scale;
        }

        // dot(P[row,:], Q[row,:]).
        static fProxy RowDot(in fProxyMxN P, in fProxyMxN Q, int row, int n)
        {
            fProxy acc = (fProxy)0;
            for (int c = 0; c < n; c++) acc += P[row, c] * Q[row, c];
            return acc;
        }

        // True iff any NON-FROZEN entry in [0,s) is exactly zero or NaN (shared shape of every
        // rho/vtrstar/alpha/tau breakdown denominator check). Rows frozen at init (already converged --
        // zero RHS row, or a warm-started row already at its exact solution) are skipped: their per-row
        // scalars stay pinned at their init value (0 for a truly-zero row) for the rest of the solve,
        // which would otherwise look like a spurious breakdown; it isn't one.
        static bool AnyZeroOrNaNActive(in fProxyN v, in NativeArray<bool> frozen, int s)
        {
            for (int i = 0; i < s; i++)
            {
                if (frozen[i]) continue;
                if (v[i] == (fProxy)0 || math.isnan(v[i])) return true;
            }
            return false;
        }

        // True iff any NON-FROZEN entry in [0,s) is NaN.
        static bool AnyNaNActive(in fProxyN v, in NativeArray<bool> frozen, int s)
        {
            for (int i = 0; i < s; i++)
            {
                if (frozen[i]) continue;
                if (math.isnan(v[i])) return true;
            }
            return false;
        }

        // Counts rows whose quasi-residual bound already meets its threshold; also returns the worst
        // bound (mirrors CountConverged's shape for the exact-residual block solvers).
        static int CountConvergedByBound(in fProxyN bound, in fProxyN thresh, int s, out double maxRnorm)
        {
            int conv = 0; double worst = 0;
            for (int j = 0; j < s; j++)
            {
                if (bound[j] <= thresh[j]) conv++;
                double bd = (double)bound[j];
                if (bd > worst) worst = bd;
            }
            maxRnorm = worst;
            return conv;
        }

        // ---- pseudo-block TFQMR core (btfqmr) --------------------------------------------------------

        /// <summary>
        /// Zero-alloc PSEUDO-block TFQMR for a NON-symmetric (general) square A and s simultaneous
        /// right-hand sides, generic over BOTH the operator (<see cref="IfProxyLinearOperator"/>) and
        /// the preconditioner (<see cref="IfProxyPreconditioner"/>). No published true (subspace-mixing)
        /// block TFQMR exists -- see OP/DEVLOG.md "Krylov.Block.TFQMR" for why the natural m x m block
        /// generalization of <see cref="tfqmr{TOp, TPre}"/>'s quasi-minimal-residual smoothing is
        /// ill-defined. Instead this runs s INDEPENDENT copies of the scalar <see cref="tfqmr{TOp,
        /// TPre}"/> recurrence in lockstep (every coefficient -- alpha/rho/beta/theta/eta/tau -- is a
        /// per-row scalar, never mixed across rows), sharing one <c>ApplyBlock</c> call per half-step
        /// instead of s separate <c>Apply</c> calls (ported from Belos's <c>PseudoBlockTFQMRIter</c>
        /// batching pattern, reference-only, BSD-3). Same per-row rigor and breakdown guards as scalar
        /// tfqmr; no block-Krylov-subspace advantage over looping scalar tfqmr s times.
        ///
        /// B and X are s ROWS x n COLS (row j = the j-th RHS/solution, length n = A.Rows); X is
        /// warm-startable. Rhat0/U/W/V/AU/D are s x n block scratch (Rhat0 is the fixed per-row shadow
        /// residual); UHat is s x n block scratch, UNUSED under the identity preconditioner (pass
        /// <c>default</c>). Convergence is per row against its own quasi-residual bound (tau[j] *
        /// sqrt(half-steps+1), Freund's rigorous upper bound on ‖B[j]-AX[j]‖) crossing tol*‖B[j]‖ -- a
        /// Converged row's true residual is guaranteed within tolerance, never independently recomputed.
        /// A row already converged at entry (all-zero RHS row, or a warm-started X row already exact)
        /// is frozen at construction: it is skipped in every per-row update for the rest of the solve
        /// (state stays at its init value, X row stays as-is) and excluded from every breakdown scan,
        /// so it cannot trigger a spurious whole-solve Breakdown. No other column locking/deflation
        /// happens mid-solve: every non-frozen row is advanced every half-step. A breakdown on any
        /// non-frozen row (zero/NaN bi-orthogonality dot, degenerate alpha, zero/NaN rhoLast, or tau
        /// collapsing before its bound catches it) reports <see cref="IterativeSolveStatus.Breakdown"/>
        /// for the WHOLE solve, X holding the last committed iterate -- never NaN, never a false
        /// Converged. Returns a <see cref="BlockSolveInfo"/> whose <see cref="BlockSolveInfo.minActive"/>
        /// is always s (no deflation).
        /// </summary>
        public static BlockSolveInfo btfqmr<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X,
                                         ref fProxyMxN Rhat0, ref fProxyMxN U, ref fProxyMxN W, ref fProxyMxN V,
                                         ref fProxyMxN AU, ref fProxyMxN D, ref fProxyMxN UHat,
                                         int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            if (A.Rows != A.Cols) throw new ArgumentException("btfqmr: A must be square");
            int n = A.Rows;
            int s = B.M_Rows;
            if (B.N_Cols != n) throw new ArgumentException("btfqmr: B.N_Cols must equal A.Rows");
            if (X.M_Rows != s || X.N_Cols != n) throw new ArgumentException("btfqmr: X must match B");
            if (Rhat0.M_Rows != s || Rhat0.N_Cols != n) throw new ArgumentException("btfqmr: Rhat0 must match B");
            if (U.M_Rows != s || U.N_Cols != n) throw new ArgumentException("btfqmr: U must match B");
            if (W.M_Rows != s || W.N_Cols != n) throw new ArgumentException("btfqmr: W must match B");
            if (V.M_Rows != s || V.N_Cols != n) throw new ArgumentException("btfqmr: V must match B");
            if (AU.M_Rows != s || AU.N_Cols != n) throw new ArgumentException("btfqmr: AU must match B");
            if (D.M_Rows != s || D.N_Cols != n) throw new ArgumentException("btfqmr: D must match B");
            if (!M.IsIdentity && (UHat.M_Rows != s || UHat.N_Cols != n))
                throw new ArgumentException("btfqmr: UHat must match B");
            if (maxIter < 1) throw new ArgumentException("btfqmr: maxIter must be >= 1");

            unsafe
            {
                int cnt = M.IsIdentity ? 8 : 9;
                long* ptrs = stackalloc long[9];
                ptrs[0] = (long)Rhat0.Data.Ptr; ptrs[1] = (long)U.Data.Ptr; ptrs[2] = (long)W.Data.Ptr;
                ptrs[3] = (long)V.Data.Ptr; ptrs[4] = (long)AU.Data.Ptr; ptrs[5] = (long)D.Data.Ptr;
                ptrs[6] = (long)X.Data.Ptr; ptrs[7] = (long)B.Data.Ptr;
                if (!M.IsIdentity) ptrs[8] = (long)UHat.Data.Ptr;
                RequireDistinctBuffers("btfqmr: Rhat0/U/W/V/AU/D/UHat/X/B must be distinct", ptrs, cnt);
            }

            fProxy bbAll = BlockFrobDot(in B, in B);
            if (bbAll == (fProxy)0)
            {
                CopyBlock(in B, ref X, s, n);
                return new BlockSolveInfo { rhs = s, converged = s, iterations = 0, maxRnorm = 0.0, minActive = s, status = IterativeSolveStatus.Converged };
            }

            var thresh = new fProxyN(s);
            for (int j = 0; j < s; j++)
            {
                fProxy bb = (fProxy)0;
                for (int c = 0; c < n; c++) bb += B[j, c] * B[j, c];
                thresh[j] = tol * math.sqrt(bb);
            }

            // Rhat0 = B - A X -- the fixed initial residual AND per-row shadow vector for the rest of the
            // solve (never mutated again). V doubles as scratch for A X here (overwritten below), mirroring
            // scalar tfqmr's own reuse of v.
            BlockResidual(in A, in X, in B, ref V, ref Rhat0, s, n);

            var tau = new fProxyN(s);
            for (int i = 0; i < s; i++) tau[i] = math.sqrt(RowDot(in Rhat0, in Rhat0, i, n));

            var bound = new fProxyN(s);
            for (int i = 0; i < s; i++) bound[i] = tau[i];
            int convergedInit = CountConvergedByBound(in bound, in thresh, s, out double maxrInit);
            if (convergedInit == s)
            {
                thresh.Dispose(); tau.Dispose(); bound.Dispose();
                return new BlockSolveInfo { rhs = s, converged = s, iterations = 0, maxRnorm = maxrInit, minActive = s, status = IterativeSolveStatus.Converged };
            }

            // Rows already meeting thresh at init (zero RHS row, or a warm-started X row already exact)
            // are frozen: their Rhat0/U/W/V/D rows are exactly 0 (or, for a non-zero-but-converged warm
            // start, their per-row scalars never need updating), so every per-row loop below skips them
            // and every breakdown scan excludes them -- a frozen row's 0 is expected, not a breakdown.
            var frozen = new NativeArray<bool>(s, Allocator.Temp);
            for (int j = 0; j < s; j++) frozen[j] = bound[j] <= thresh[j];

            CopyBlock(in Rhat0, ref U, s, n);
            CopyBlock(in Rhat0, ref W, s, n);

            fProxyN rowIn = default, rowOut = default;
            if (!M.IsIdentity) { rowIn = new fProxyN(n); rowOut = new fProxyN(n); }

            // AU = A (M^-1 U). Identity: UHat = U, so AU = A U directly (UHat untouched).
            if (M.IsIdentity) A.ApplyBlock(in U, ref AU, s);
            else { BlockApplyPre(in M, in U, ref UHat, s, n, ref rowIn, ref rowOut); A.ApplyBlock(in UHat, ref AU, s); }

            CopyBlock(in AU, ref V, s, n);
            for (int i = 0; i < s; i++)
                for (int c = 0; c < n; c++) D[i, c] = (fProxy)0;

            var theta = new fProxyN(s);
            var eta = new fProxyN(s);
            var alpha = new fProxyN(s);
            var rho = new fProxyN(s);
            var rhoLast = new fProxyN(s);
            for (int i = 0; i < s; i++)
            {
                theta[i] = (fProxy)0; eta[i] = (fProxy)0; alpha[i] = (fProxy)0;
                rho[i] = tau[i] * tau[i];       // <Rhat0[i], Rhat0[i]> == tau[i]^2 (shadow == r0)
                rhoLast[i] = rho[i];
            }

            var vtrstar = new fProxyN(s);
            var rhoNew = new fProxyN(s);
            var beta = new fProxyN(s);

            IterativeSolveStatus status = IterativeSolveStatus.MaxIterations;
            int itersDone = maxIter;
            int converged = convergedInit;
            double maxr = maxrInit;

            for (int k = 0; k < maxIter; k++)
            {
                bool even = (k & 1) == 0;

                if (even)
                {
                    for (int i = 0; i < s; i++) { if (frozen[i]) continue; vtrstar[i] = RowDot(in Rhat0, in V, i, n); }
                    if (AnyZeroOrNaNActive(in vtrstar, in frozen, s))
                    { status = IterativeSolveStatus.Breakdown; itersDone = k; goto cleanup; } // row orthogonal to its shadow

                    for (int i = 0; i < s; i++) { if (frozen[i]) continue; alpha[i] = rho[i] / vtrstar[i]; }
                    if (AnyZeroOrNaNActive(in alpha, in frozen, s))
                    { status = IterativeSolveStatus.Breakdown; itersDone = k; goto cleanup; } // degenerate alpha
                }

                for (int i = 0; i < s; i++)
                {
                    if (frozen[i]) continue; // state stays pinned at its init value; X row stays as-is

                    // W -= alpha AU -- AU already holds A(M^-1 U) for the CURRENT U (computed at the
                    // tail of the previous half-step, or just above for k == 0).
                    RowAddScaled(ref W, i, -alpha[i], in AU, n);

                    // D = UHat + (theta^2/alpha) eta D -- UHat is M^-1 of the CURRENT U (identity: U
                    // itself), already right-preconditioned so eta*D below folds straight into X.
                    fProxy dCoeff = (theta[i] * theta[i] / alpha[i]) * eta[i];
                    if (M.IsIdentity) RowScaleAddSelf(ref D, i, dCoeff, in U, n);
                    else              RowScaleAddSelf(ref D, i, dCoeff, in UHat, n);

                    fProxy thetaI = math.sqrt(RowDot(in W, in W, i, n)) / tau[i];
                    fProxy c = (fProxy)1 / math.sqrt((fProxy)1 + thetaI * thetaI);
                    tau[i] = tau[i] * thetaI * c;
                    theta[i] = thetaI;
                    eta[i] = c * c * alpha[i];

                    RowAddScaled(ref X, i, eta[i], in D, n);

                    bound[i] = tau[i] * math.sqrt((fProxy)(k + 1));
                }

                converged = CountConvergedByBound(in bound, in thresh, s, out maxr);
                if (converged == s) { status = IterativeSolveStatus.Converged; itersDone = k + 1; goto cleanup; }
                if (AnyZeroOrNaNActive(in tau, in frozen, s))
                { status = IterativeSolveStatus.Breakdown; itersDone = k + 1; goto cleanup; } // tau collapsed on some row

                if (even)
                {
                    // Advance U = U - alpha V, then refresh AU (and UHat) from the new U. ApplyBlock runs
                    // over the whole block (frozen rows carry alpha=0/U row=0 through unguarded -- A of a
                    // zero row is exactly zero, so this is harmless and keeps the frozen invariant).
                    for (int i = 0; i < s; i++) { if (frozen[i]) continue; RowAddScaled(ref U, i, -alpha[i], in V, n); }
                    if (M.IsIdentity) A.ApplyBlock(in U, ref AU, s);
                    else { BlockApplyPre(in M, in U, ref UHat, s, n, ref rowIn, ref rowOut); A.ApplyBlock(in UHat, ref AU, s); }
                    for (int i = 0; i < s; i++) { if (frozen[i]) continue; rhoLast[i] = rho[i]; }
                }
                else
                {
                    for (int i = 0; i < s; i++) { if (frozen[i]) continue; rhoNew[i] = RowDot(in Rhat0, in W, i, n); }
                    if (AnyZeroOrNaNActive(in rhoLast, in frozen, s))
                    { status = IterativeSolveStatus.Breakdown; itersDone = k + 1; goto cleanup; } // beta undefined

                    for (int i = 0; i < s; i++) { if (frozen[i]) continue; beta[i] = rhoNew[i] / rhoLast[i]; }
                    if (AnyNaNActive(in beta, in frozen, s))
                    { status = IterativeSolveStatus.Breakdown; itersDone = k + 1; goto cleanup; }
                    for (int i = 0; i < s; i++) { if (frozen[i]) continue; rho[i] = rhoNew[i]; }

                    for (int i = 0; i < s; i++)
                    {
                        if (frozen[i]) continue;
                        RowScaleAddSelf(ref U, i, beta[i], in W, n);              // U = beta U + W
                        RowScaleInPlace(ref V, i, beta[i] * beta[i], n);          // V = beta^2 V ...
                        RowAddScaled(ref V, i, beta[i], in AU, n);                // ... + beta AU (AU still OLD here)
                    }
                    if (M.IsIdentity) A.ApplyBlock(in U, ref AU, s);
                    else { BlockApplyPre(in M, in U, ref UHat, s, n, ref rowIn, ref rowOut); A.ApplyBlock(in UHat, ref AU, s); }
                    for (int i = 0; i < s; i++) { if (frozen[i]) continue; RowAddScaled(ref V, i, (fProxy)1, in AU, n); }  // V += AU (now refreshed)
                }
            }

        cleanup:
            thresh.Dispose(); tau.Dispose(); bound.Dispose();
            theta.Dispose(); eta.Dispose(); alpha.Dispose(); rho.Dispose(); rhoLast.Dispose();
            vtrstar.Dispose(); rhoNew.Dispose(); beta.Dispose();
            frozen.Dispose();
            if (!M.IsIdentity) { rowIn.Dispose(); rowOut.Dispose(); }

            // No width reduction -- frozen rows are skipped, not deflated, so minActive = s always.
            return new BlockSolveInfo { rhs = s, converged = converged, iterations = itersDone, maxRnorm = maxr, minActive = s, status = status };
        }

        // ---- unpreconditioned + concrete forwarders ------------------------------------------------

        /// <summary>Unpreconditioned pseudo-block TFQMR -- forwards into the merged
        /// <see cref="btfqmr{TOp, TPre}"/> with the identity preconditioner (whose IsIdentity fold strips
        /// the UHat traffic), so this needs no UHat buffer.</summary>
        public static BlockSolveInfo btfqmr<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X,
                                         ref fProxyMxN Rhat0, ref fProxyMxN U, ref fProxyMxN W, ref fProxyMxN V,
                                         ref fProxyMxN AU, ref fProxyMxN D,
                                         int maxIter, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
        {
            fProxyMxN UHat = default;
            return btfqmr(in A, default(fProxyIdentityPreconditioner), in B, ref X, ref Rhat0, ref U, ref W, ref V, ref AU, ref D, ref UHat, maxIter, tol);
        }

        /// <summary>Pseudo-block TFQMR over a dense NON-symmetric <see cref="fProxyMxN"/> A, via
        /// <see cref="fProxyDenseOperatorGeneral"/> (general block apply -- <see cref="fProxyDenseOperator"/>'s
        /// ApplyBlock is symmetric-only and would silently solve Aᵀx=b here).</summary>
        public static BlockSolveInfo btfqmr(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X,
                                    ref fProxyMxN Rhat0, ref fProxyMxN U, ref fProxyMxN W, ref fProxyMxN V,
                                    ref fProxyMxN AU, ref fProxyMxN D,
                                    int maxIter, fProxy tol)
        {
            return btfqmr(new fProxyDenseOperatorGeneral(in A), in B, ref X, ref Rhat0, ref U, ref W, ref V, ref AU, ref D, maxIter, tol);
        }

        /// <summary>Pseudo-block TFQMR over a dense non-symmetric A -- allocates block scratch from the
        /// arena.</summary>
        public static BlockSolveInfo btfqmr(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)
        {
            int s = B.M_Rows, n = A.M_Rows;
            fProxyMxN Rhat0 = B.fProxyTempMat(s, n, true), U = B.fProxyTempMat(s, n, true), W = B.fProxyTempMat(s, n, true),
                      V = B.fProxyTempMat(s, n, true), AU = B.fProxyTempMat(s, n, true), D = B.fProxyTempMat(s, n, true);
            return btfqmr(in A, in B, ref X, ref Rhat0, ref U, ref W, ref V, ref AU, ref D, maxIter, tol);
        }

        /// <summary>Pseudo-block TFQMR over a dense non-symmetric A with default maxIter (A.M_Rows) and
        /// tol (Consts.fProxySqrtEps).</summary>
        public static BlockSolveInfo btfqmr(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X)
            => btfqmr(in A, in B, ref X, A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>Pseudo-block TFQMR over a block-sparse (BSR) non-symmetric A -- zero-alloc primitive.
        /// Forwards into <see cref="btfqmr{TOp}"/> via <c>fProxyBSROperator</c>.</summary>
        public static BlockSolveInfo btfqmr(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X,
                                    ref fProxyMxN Rhat0, ref fProxyMxN U, ref fProxyMxN W, ref fProxyMxN V,
                                    ref fProxyMxN AU, ref fProxyMxN D,
                                    int maxIter, fProxy tol)
        {
            return btfqmr(new fProxyBSROperator(in A), in B, ref X, ref Rhat0, ref U, ref W, ref V, ref AU, ref D, maxIter, tol);
        }

        /// <summary>Pseudo-block TFQMR over a BSR non-symmetric A -- allocates block scratch from the
        /// arena.</summary>
        public static BlockSolveInfo btfqmr(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)
        {
            int s = B.M_Rows, n = A.M_Rows;
            fProxyMxN Rhat0 = B.fProxyTempMat(s, n, true), U = B.fProxyTempMat(s, n, true), W = B.fProxyTempMat(s, n, true),
                      V = B.fProxyTempMat(s, n, true), AU = B.fProxyTempMat(s, n, true), D = B.fProxyTempMat(s, n, true);
            return btfqmr(in A, in B, ref X, ref Rhat0, ref U, ref W, ref V, ref AU, ref D, maxIter, tol);
        }

        /// <summary>Pseudo-block TFQMR over a BSR non-symmetric A with default maxIter (A.M_Rows) and
        /// tol (Consts.fProxySqrtEps).</summary>
        public static BlockSolveInfo btfqmr(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X)
            => btfqmr(in A, in B, ref X, A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>Right-preconditioned pseudo-block TFQMR over a block-sparse (BSR) non-symmetric
        /// A with ANY <see cref="IfProxyPreconditioner"/> (ILU0) -- forwards into
        /// <see cref="btfqmr{TOp,TPre}"/> via <c>fProxyBSROperator</c>.</summary>
        public static BlockSolveInfo btfqmr<TPre>(in fProxyBSR A, in TPre M, in fProxyMxN B, ref fProxyMxN X,
                               int maxIter, fProxy tol)
            where TPre : struct, IfProxyPreconditioner
        {
            int s = B.M_Rows, n = A.M_Rows;
            fProxyMxN Rhat0 = B.fProxyTempMat(s, n, true), U = B.fProxyTempMat(s, n, true), W = B.fProxyTempMat(s, n, true),
                      V = B.fProxyTempMat(s, n, true), AU = B.fProxyTempMat(s, n, true), D = B.fProxyTempMat(s, n, true),
                      UHat = B.fProxyTempMat(s, n, true);
            return btfqmr(new fProxyBSROperator(in A), in M, in B, ref X,
                             ref Rhat0, ref U, ref W, ref V, ref AU, ref D, ref UHat,
                             maxIter, tol);
        }

        /// <summary>Preconditioned pseudo-block TFQMR over BSR with ANY <see cref="IfProxyPreconditioner"/>
        /// (ILU0), with default maxIter (A.M_Rows) and tolerance (Consts.fProxySqrtEps).</summary>
        public static BlockSolveInfo btfqmr<TPre>(in fProxyBSR A, in TPre M, in fProxyMxN B, ref fProxyMxN X)
            where TPre : struct, IfProxyPreconditioner
            => btfqmr(in A, in M, in B, ref X, A.M_Rows, Consts.fProxySqrtEps);
    }
}
