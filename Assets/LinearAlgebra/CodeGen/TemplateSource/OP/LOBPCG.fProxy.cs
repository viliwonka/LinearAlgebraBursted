using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
using Unity.Mathematics;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    /// <summary>
    /// Blocked Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG): the k SMALLEST
    /// eigenpairs of a symmetric operator A (A.Rows == A.Cols), generic over any
    /// <see cref="IfProxyLinearOperator"/> and an optional <see cref="IfProxyPreconditioner"/>.
    /// Reuses the dense <see cref="Eigen.symmetric(ref fProxyMxN, ref fProxyN, ref fProxyMxN)"/>
    /// solver for the small (&lt;= 3k) Rayleigh-Ritz sub-problem and
    /// <see cref="CHO.decomp(in fProxyMxN, ref fProxyMxN)"/> for orthogonalization and the
    /// generalized-to-standard reduction.
    ///
    /// <b>Locking:</b> once a pair's relative residual meets <c>tol</c> it is locked (frozen at its
    /// converged value, no further matvec/preconditioner/Rayleigh-Ritz) and deflated out of the
    /// active subspace, so already-found eigenvectors are never re-discovered. Locked pairs stay in
    /// the output X.
    ///
    /// <b>Robustness:</b> the active W and P blocks are deflated against the locked+active X and
    /// Cholesky-QR-orthonormalized before the Rayleigh-Ritz step, keeping its Gram matrix
    /// well-conditioned. The Gram Cholesky has a Tikhonov-ridge retry; if it still fails the
    /// iteration drops P and retries with just [X, W], and failing that stalls (X/lambda unchanged)
    /// rather than producing NaN. Selected Ritz values are sanity-checked against the individual
    /// basis-row Rayleigh quotients and rejected if implausible. A non-finite residual aborts with
    /// <see cref="IterativeSolveStatus.Breakdown"/>.
    ///
    /// <b>Guard vectors:</b> the working block size is taken from the cache (<c>ws.X.M_Rows</c>), not
    /// from <c>k</c>. Allocating <c>Arena.fProxyLOBPCGCache(n, k + guard)</c> and calling with wanted
    /// count <c>k</c> makes the extra <c>guard</c> rows GUARD ("ghost") vectors: full participants in
    /// every matvec / orthogonalization / Rayleigh-Ritz step, but only the <c>k</c> smallest pairs gate
    /// convergence (see the wanted-subset exit) and are returned. Guards give the wanted pairs spectral
    /// separation room, so a clustered / near-degenerate bottom (e.g. a grid Laplacian, where a square
    /// grid has exact multiplicities) converges in far fewer iterations -- the standard LOBPCG lever for
    /// clustered spectra. A cache sized exactly <c>k</c> (<c>guard == 0</c>) has no guard rows and every
    /// step is bit-identical to the pre-guard implementation. The allocating dense/BSR
    /// <c>lobpcg(ref arena, A, k, guard, ...)</c> overloads wrap this; the zero-alloc primitive picks it
    /// up automatically from an oversized cache.
    ///
    /// <b>Zero-alloc scope:</b> all O(n)-scale buffers live in <see cref="fProxyLOBPCGCache"/>,
    /// allocated once via <c>Arena.fProxyLOBPCGCache(n, k)</c> and reused across calls. The
    /// O(k)-scale Rayleigh-Ritz dense matrices are also cache fields. The only bounded
    /// <c>Allocator.Temp</c> allocations are inside
    /// <see cref="Eigen.symmetric(ref fProxyMxN, ref fProxyN, ref fProxyMxN)"/> and the
    /// triangular-solve helpers.
    ///
    /// <b>Convergence:</b> per-pair 2-norm relative residual ||A x_i - lambda_i B x_i|| /
    /// max(|lambda_i|, 1) &lt;= tol. Returns a <see cref="LOBPCGInfo"/>
    /// (Converged/MaxIterations/Breakdown).
    ///
    /// <b>Output order:</b> eigenvalues/eigenvectors are returned ASCENDING (index 0 = smallest) --
    /// unlike the other Eigen methods (which sort descending) -- because LOBPCG targets the k
    /// smallest.
    ///
    /// <b>Generalized eigenproblem (A x = lambda B x, B SPD):</b> overloads taking a second operator
    /// B solve the pencil by replacing Euclidean inner products with B-inner products (Gram = S^T B S)
    /// and residuals r_i = A x_i - theta_i B x_i; convergence uses the Euclidean 2-norm of that
    /// residual. Only B must be SPD -- A may be INDEFINITE (the buckling case below) and convergence
    /// to the algebraically smallest pencil eigenvalues still holds. The standard (single-operator)
    /// overloads forward into this same core with B = <see cref="fProxyIdentityOperator"/>, giving a
    /// bit-identical Euclidean path.
    ///
    /// <b>Buckling mapping (K_E phi = -lambda K_G phi convention):</b> the linear-buckling problem
    /// K_E*phi + lambda*K_G*phi = 0 (K_E SPD elastic stiffness, K_G indefinite geometric stiffness at
    /// a reference load; lambda the load multiplier -- the Nastran SOL 105 / Abaqus *BUCKLE
    /// convention) rearranges to the pencil K_G*phi = mu*K_E*phi with mu = -1/lambda_cr, i.e. K_G in
    /// the A slot and K_E (SPD) in the B slot. The smallest (most negative) mu gives the first
    /// critical load, exactly what LOBPCG targets:
    /// <code>
    ///   // K_E: SPD elastic stiffness. K_G: geometric stiffness at the reference load (indefinite).
    ///   var mu = Eigen.lobpcg(in K_G, in K_E, ref ws, k, tol, maxIter); // A=K_G, B=K_E
    ///   // mu is ASCENDING; mu[0] is the most negative (first/critical) mode. A mu[i] &gt;= 0 is not
    ///   // a buckling mode under this reference load direction -- discard/flag it, do not divide.
    ///   for (int i = 0; i &lt; k; i++)
    ///       if (mu[i] &lt; 0) lambdaCritical[i] = -1 / mu[i]; // buckling load multiplier
    /// </code>
    /// For the opposite sign convention (K_E*phi = +lambda*K_G*phi) use lambda_cr = +1/mu; the pencil
    /// construction and smallest-mu targeting are unchanged.
    /// </summary>
    public static partial class Eigen
    {
        /// <summary>
        /// Zero-alloc (at O(n) scale) LOBPCG primitive for the GENERALIZED symmetric eigenproblem
        /// A x = lambda B x (B SPD -- see the class doc comment's "Generalized eigenproblem" /
        /// "B=I strategy" / "Fresh-matvec principle extends to B" sections for the full math and
        /// design rationale). <paramref name="ws"/> is warm-startable: pre-fill <c>ws.X</c> with a
        /// guess (it is orthonormalized unconditionally at the start, whether seeded or supplied);
        /// the all-zero block is seeded deterministically first.
        /// </summary>
        public static LOBPCGInfo lobpcg<TOp, TBOp, TPre>(in TOp A, in TBOp B, in TPre M, ref fProxyLOBPCGCache ws,
                                                          int k, fProxy tol, int maxIter)
            where TOp : struct, IfProxyLinearOperator
            where TBOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
        {
            if (A.Rows != A.Cols)
                throw new ArgumentException("LOBPCG: A must be square");

            int n = A.Rows;

            if (B.Rows != n || B.Cols != n)
                throw new ArgumentException("LOBPCG: B must be n x n, matching A (B must be SPD -- not verified at runtime, the same unchecked contract as A's symmetry)");

            // The working BLOCK size is taken from the cache (ws.X has kWork rows); the requested number
            // of wanted pairs is `k`. When kWork > k the extra (kWork - k) rows are GUARD ("ghost")
            // vectors: full participants in every matvec / orthogonalization / Rayleigh-Ritz step, but
            // only the k SMALLEST pairs gate convergence and are returned. Guard vectors give the wanted
            // pairs spectral separation room, which is what accelerates -- and, for a near-degenerate
            // bottom, rescues -- convergence (see the class doc's "Guard vectors" note). kWork == k (a
            // cache allocated for exactly k -- every pre-guard call site) is the no-guard degenerate
            // case, and every step below is then bit-identical to the pre-guard implementation.
            int kWork = ws.X.M_Rows;

            if (kWork < 1 || kWork > n)
                throw new ArgumentException("LOBPCG: block size (cache k) must be in [1, A.Rows]");

            if (k < 1 || k > kWork)
                throw new ArgumentException("LOBPCG: k must be in [1, cache block size] (allocate fProxyLOBPCGCache(n, k + guard) to use guard vectors)");

            if (maxIter < 1)
                throw new ArgumentException("LOBPCG: maxIter must be >= 1");

            if (tol <= (fProxy)0)
                throw new ArgumentException("LOBPCG: tol must be > 0");

            RequireLOBPCGWorkspace(in ws, n, kWork);
            RequireDistinctBuffers(in ws);

            // ---- seed X if all-zero, then orthonormalize unconditionally ----
            bool allZero = true;
            for (int i = 0; i < kWork && allZero; i++)
                for (int c = 0; c < n; c++)
                    if (ws.X[i, c] != (fProxy)0) { allZero = false; break; }

            // Deterministic pseudo-random fill (fixed seed -- same inputs always produce the same
            // result), NOT a periodic formula: an earlier version used `(i + c*3 + 1) & 3`, which
            // repeats with period 4 in BOTH i and c, so the seeded X has AT MOST 4 distinct rows --
            // EXACTLY rank-deficient for any k > 4. That degeneracy is silently absorbed by
            // FactorGram's ridge retry (see safeguard 2), so the solver would iterate correctly
            // within only a 4-dimensional subspace instead of the full R^n, never converging to
            // eigenpairs 5+ (or converging to duplicates/garbage for them). A fixed-seed
            // Unity.Mathematics.Random fill (mirrors the same deterministic-seed pattern used by
            // SVD.LowRank/SVD.Randomized's own `rng.NextFloat() * 2f - 1f` seed vectors) has
            // effectively zero chance of landing in any low-dimensional subspace for realistic n/k.
            if (allZero)
            {
                var seedRng = new Unity.Mathematics.Random(0x9E3779B1u);
                for (int i = 0; i < kWork; i++)
                    for (int c = 0; c < n; c++)
                        ws.X[i, c] = (fProxy)(seedRng.NextFloat() * 2f - 1f);
            }

            // ws.AX is not yet meaningful here (freshly recomputed via A.Apply right below) --
            // passed through purely so the SAME combination applied to X is mirrored onto it too
            // (harmless busywork on not-yet-meaningful data), letting the initial seed reuse the
            // exact same helper every OTHER block orthonormalization in this file uses. This
            // initial orthonormalization is DELIBERATELY EUCLIDEAN ONLY (V^T V, not the
            // B-inner-product V^T B V): it only needs to leave X non-degenerate/well-conditioned
            // before the first real Rayleigh-Ritz iteration, which correctly forms and reduces the
            // ACTUAL Gram = X^T B X of whatever X turns out to be (Cholesky/ridge-retry are already
            // robust to an arbitrary SPD Gram, not just an identity one). Keeping this call
            // UNCHANGED (same helper, same arguments) is what makes the standard (B=I) path's
            // bootstrap bit-identical to the pre-generalization implementation -- see
            // <see cref="OrthonormalizeBlockB"/> for the B-aware sibling used by W/P below, which
            // genuinely needs a pre-computed B-image and so cannot reuse this same bootstrap shape.
            if (!OrthonormalizeBlock(ref ws.X, ref ws.AX, kWork, n, ref ws.Gram, ref ws.L, ref ws.rowIn, ref ws.rowOut))
                return new LOBPCGInfo { iterations = 0, converged = 0, maxResidual = double.NaN, status = IterativeSolveStatus.Breakdown };

            A.ApplyBlock(in ws.X, ref ws.AX, kWork);

            // BX = B.Apply(X), fresh, right after AX -- mirrors AX's own freshness exactly (see the
            // class doc's "fresh-matvec principle extends to B"). For the B=I forwarding path
            // (TBOp == fProxyIdentityOperator) this is an exact bit-copy of X, so every later
            // computation reading BX in place of a Euclidean X reproduces the pre-generalization
            // formula bit-for-bit.
            B.ApplyBlock(in ws.X, ref ws.BX, kWork);

            // Bootstrap Rayleigh-quotient seed for lambda -- deliberately the plain EUCLIDEAN
            // quotient dot(X,AX), NOT divided by dot(X,BX) (the "more correct" generalized
            // quotient): see the class doc's "B=I strategy" note for why (this seed is immediately
            // superseded by the first real Rayleigh-Ritz iteration regardless of its accuracy, so
            // the division would only cost bit-identical-ness for B=I with no practical benefit).
            for (int i = 0; i < kWork; i++)
            {
                fProxy d = (fProxy)0;
                for (int c = 0; c < n; c++) d += ws.X[i, c] * ws.AX[i, c];
                ws.lambda[i] = d;
            }

            int numActive = kWork;
            bool haveP = false;

            // Freeze (lock) a pair only at a fraction of tol; convergence is still DECLARED at tol via
            // allWithinTol below. Locking is destructive -- once a pair is frozen, the remaining active
            // pairs are confined B-orthogonal to it (W/P are deflated against all X), and the best
            // residual achievable under that confinement is ~0.87x the frozen pair's lock residual. So
            // lock with margin (0.087*tol induced floor) instead of at tol, which would leave later
            // pairs stuck just above tol. See the (d1) re-deflation block below.
            fProxy lockTol = tol * (fProxy)0.1;

            for (int iter = 0; iter < maxIter; iter++)
            {
                // ---- residual + lock newly converged pairs (scan back-to-front so a swap-in
                //      from the back is still checked) ----
                bool allWithinTol = true;
                for (int i = numActive - 1; i >= 0; i--)
                {
                    fProxy rn2 = (fProxy)0;
                    for (int c = 0; c < n; c++)
                    {
                        // Generalized residual r_i = A x_i - lambda_i B x_i (standard practice --
                        // see the class doc). For B=I, ws.BX[i,c] is bit-identical to ws.X[i,c], so
                        // this reproduces the original "A x - lambda x" formula bit-for-bit.
                        fProxy rv = ws.AX[i, c] - ws.lambda[i] * ws.BX[i, c];
                        ws.R[i, c] = rv;
                        rn2 += rv * rv;
                    }
                    fProxy rnorm = math.sqrt(rn2);

                    if (!math.isfinite(rnorm))
                    {
                        SortAscending(ref ws, kWork);
                        return new LOBPCGInfo { iterations = iter, converged = math.min(k, kWork - numActive), maxResidual = double.NaN, status = IterativeSolveStatus.Breakdown };
                    }

                    ws.residual[i] = rnorm;

                    fProxy scale = math.abs(ws.lambda[i]);
                    if (scale < (fProxy)1) scale = (fProxy)1;

                    // Convergence is measured at tol (this drives the honest exit below); a pair need
                    // not be locked to count as converged.
                    if (!(rnorm <= tol * scale)) allWithinTol = false;

                    if (rnorm <= lockTol * scale)
                    {
                        int last = numActive - 1;
                        if (i != last)
                        {
                            Swap.Rows(ref ws.X, i, last);
                            Swap.Rows(ref ws.AX, i, last);
                            Swap.Rows(ref ws.BX, i, last);
                            Swap.Vec(ref ws.lambda, i, last);
                            Swap.Vec(ref ws.residual, i, last);

                            // P/AP/BP (the search direction) and R (this row's own just-computed raw
                            // residual, which forms W a few lines below) must move WITH the pair
                            // that is moving into row i (previously row `last`), or the next steps
                            // read a search direction / residual belonging to a DIFFERENT
                            // (just-locked) pair -- a desync bug that only manifests once a
                            // lock-swap actually happens (k==1, or a run that converges before any
                            // lock, never exercises it).
                            Swap.Rows(ref ws.P, i, last);
                            Swap.Rows(ref ws.AP, i, last);
                            Swap.Rows(ref ws.BP, i, last);
                            Swap.Rows(ref ws.R, i, last);
                        }
                        numActive--;
                    }
                }

                // Honest exit: the k WANTED (smallest) pairs are all within tol. With NO guards
                // (kWork == k) this is exactly allWithinTol -- every pair within tol, computed inline
                // above -- so the pre-guard path is bit-identical. With guards it ignores the residuals
                // of the (kWork - k) guard rows, which are not required to converge (see WantedWithinTol);
                // waiting on them would defeat the purpose (guards typically sit higher in the spectrum
                // and converge slowest). `iter` (not iter+1): iterations 0..iter-1 each did a full
                // W/Rayleigh-Ritz work pass; THIS iteration only ran the residual check before finding
                // everyone converged, so it contributes no additional work -- matches SolveInfo's
                // "0 when converged before the first step" convention.
                bool converged = (kWork == k) ? allWithinTol : WantedWithinTol(in ws, kWork, k, tol);
                if (converged)
                {
                    SortAscending(ref ws, kWork);
                    return new LOBPCGInfo { iterations = iter, converged = k, maxResidual = MaxRelResidual(in ws, k), status = IterativeSolveStatus.Converged };
                }

                // (d1) Re-B-orthogonalize the ACTIVE X block against the LOCKED rows [numActive, k).
                // Deflation of the search directions (W/P below) against ALL X removes locked-row
                // components from the DIRECTIONS, but the active X rows themselves can retain a fixed
                // B-component along a just-frozen row that no later direction can then cancel -- a
                // hard-locking FIXED POINT that freezes the residual at ~|component|*|dLambda|*||B x||
                // (the buckling smoke test hit exactly this in float). Projecting the active block
                // B-orthogonal to the locked rows here removes that trapped component; AX/BX ride along
                // by linearity, and we renormalize in the B-inner-product so the next Rayleigh-Ritz
                // Gram stays near the identity. No-op until at least one pair has locked (numActive < k).
                if (numActive < kWork)
                {
                    Deflate(ref ws.X, ref ws.AX, ref ws.BX, numActive, in ws.X, in ws.AX, in ws.BX, numActive, kWork - numActive, n);
                    for (int i = 0; i < numActive; i++)
                    {
                        fProxy bn2 = (fProxy)0;
                        for (int c = 0; c < n; c++) bn2 += ws.X[i, c] * ws.BX[i, c];
                        if (bn2 > (fProxy)0)
                        {
                            fProxy inv = (fProxy)1 / math.sqrt(bn2);
                            for (int c = 0; c < n; c++)
                            {
                                ws.X[i, c]  *= inv;
                                ws.AX[i, c] *= inv;
                                ws.BX[i, c] *= inv;
                            }
                        }
                    }
                }

                // ---- W = M^-1 R (active), AW = A W, BW = B W (active) -- ONE matvec batch each/iteration ----
                for (int i = 0; i < numActive; i++)
                {
                    for (int c = 0; c < n; c++) ws.rowIn[c] = ws.R[i, c];
                    M.Apply(in ws.rowIn, ref ws.rowOut);
                    for (int c = 0; c < n; c++) ws.W[i, c] = ws.rowOut[c];
                }
                A.ApplyBlock(in ws.W, ref ws.AW, numActive);
                // BW = B W, fresh -- mirrors AW's own single fresh compute this iteration (see the
                // class doc's "fresh-matvec principle extends to B"); maintained via linearity
                // through the Deflate/OrthonormalizeBlockB calls below within THIS iteration only
                // (bounded, not chained across iterations -- exactly how AW already works).
                B.ApplyBlock(in ws.W, ref ws.BW, numActive);

                // ---- B-deflate against X, then INTERNALLY B-orthonormalize (safeguard 1) ----
                // Deflation alone leaves W's OWN Gram an arbitrary SPD matrix (its rows can differ
                // in scale by orders of magnitude, e.g. once some pairs' residuals have shrunk much
                // more than others) -- feeding that directly into the combined Rayleigh-Ritz Gram
                // relies on ONE Cholesky to absorb both the deflation AND that scale spread, which
                // is exactly the ill-conditioned-basis failure mode that produces spurious Ritz
                // values (below lambda_min, even negative) instead of tripping the rank-deficiency
                // safeguard. Cholesky-QR-normalizing W (and P) INTERNALLY right here, w.r.t. the
                // B-inner-product -- via the same FactorGram (with its ridge retry) used everywhere
                // else -- keeps the combined B-Gram close to the identity, so the final
                // Rayleigh-Ritz Cholesky is well-conditioned by construction and a genuine rank
                // deficiency reliably trips the correct safeguard.
                Deflate(ref ws.W, ref ws.AW, ref ws.BW, numActive, in ws.X, in ws.AX, in ws.BX, 0, kWork, n);

                if (!OrthonormalizeBlockB(ref ws.W, ref ws.AW, ref ws.BW, numActive, n, ref ws.Gram, ref ws.L, ref ws.rowIn, ref ws.rowOut, ref ws.rowAux))
                {
                    // W collapsed entirely onto the already-known (locked+active X) subspace --
                    // a genuinely degenerate iteration. Stall (leave X/lambda untouched) rather
                    // than feed a degenerate basis into Rayleigh-Ritz; try again next iteration.
                    haveP = false;
                    continue;
                }

                bool haveP0 = haveP;
                if (haveP0)
                {
                    Deflate(ref ws.P, ref ws.AP, ref ws.BP, numActive, in ws.X, in ws.AX, in ws.BX, 0, kWork, n);
                    Deflate(ref ws.P, ref ws.AP, ref ws.BP, numActive, in ws.W, in ws.AW, in ws.BW, 0, numActive, n);

                    // If P has become (nearly) linearly dependent on X/W -- e.g. after many
                    // iterations P can drift toward the span already covered -- drop it for just
                    // this iteration (safeguard 2's "standard fix") rather than treating it as a
                    // hard stall; W alone is still a perfectly good (steepest-descent) basis.
                    if (!OrthonormalizeBlockB(ref ws.P, ref ws.AP, ref ws.BP, numActive, n, ref ws.Gram, ref ws.L, ref ws.rowIn, ref ws.rowOut, ref ws.rowAux))
                        haveP0 = false;
                }

                // ---- Rayleigh-Ritz, 3-block with a 2-block ("drop P") fallback (safeguard 2) ----
                bool usedP = haveP0;
                bool ok = TryRayleighRitz(ref ws, numActive, usedP, n);

                if (!ok && usedP)
                {
                    usedP = false;
                    ok = TryRayleighRitz(ref ws, numActive, usedP, n);
                }

                if (!ok)
                {
                    // Pathological stall: skip this iteration's Ritz update (lambda and the Ritz
                    // combination of X are untouched) and retry next time; discard P's history since it
                    // (or even just [X,W]) was implicated. Note X may already carry the (d1) re-deflation
                    // applied above -- a legitimate improvement, not part of the discarded RR step.
                    haveP = false;
                    continue;
                }

                UpdateActiveBlock(ref ws, numActive, usedP, n, kWork);

                // Recompute AX/BX FRESH via a matvec each -- UpdateActiveBlock deliberately does
                // NOT also mirror-combine AX/BX (see its own doc comment): propagating AX through
                // many iterations of Cholesky-QR/Rayleigh-Ritz combinations (never re-touching A)
                // accumulates rounding error that compounds -- exactly the observed failure mode
                // (residual shrinks nicely for ~15-20 iterations, then stalls and creeps back up
                // instead of continuing to converge). This is the canonical "R = A X - X diag(theta)"
                // fresh-residual formulation (generalized: "- B X diag(theta)"); the extra
                // matvecs/iteration (over numActive rows only) are a small, worthwhile price for a
                // residual that stays exact to working precision indefinitely.
                A.ApplyBlock(in ws.X, ref ws.AX, numActive);
                B.ApplyBlock(in ws.X, ref ws.BX, numActive);

                // Same fix for AP/BP: P is reformed EVERY iteration from a combination of the
                // CURRENT W and the OLD P (chained iteration to iteration, just like AX used to
                // be), and -- unlike AX, which only feeds the residual/convergence check -- an
                // inaccurate AP corrupts next iteration's [X,W,P] Gram/H directly (H's P-columns
                // are dot(*, AP)), which is a much more direct route to visibly wrong Ritz values
                // (this is what actually produced Ritz values below lambda_min, even wildly
                // negative, as soon as P entered the mix -- not merely a conditioning threshold
                // issue, since the SAME marginal conditioning is completely harmless in the
                // P-less 2-block path above). BP is refreshed for the SAME reason -- it feeds the
                // next iteration's B-Gram directly.
                A.ApplyBlock(in ws.P, ref ws.AP, numActive);
                B.ApplyBlock(in ws.P, ref ws.BP, numActive);

                haveP = true;
            }

            SortAscending(ref ws, kWork);
            return new LOBPCGInfo { iterations = maxIter, converged = ConvergedWithinTol(in ws, k, tol), maxResidual = MaxRelResidual(in ws, k), status = IterativeSolveStatus.MaxIterations };
        }

        /// <summary>lobpcg (generalized, preconditioned) with default maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg<TOp, TBOp, TPre>(in TOp A, in TBOp B, in TPre M, ref fProxyLOBPCGCache ws, int k, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TBOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => lobpcg(in A, in B, in M, ref ws, k, tol, 1000);

        /// <summary>lobpcg (generalized, preconditioned) with default tol (Consts.fProxySqrtEps) and maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg<TOp, TBOp, TPre>(in TOp A, in TBOp B, in TPre M, ref fProxyLOBPCGCache ws, int k)
            where TOp : struct, IfProxyLinearOperator
            where TBOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => lobpcg(in A, in B, in M, ref ws, k, Consts.fProxySqrtEps, 1000);

        /// <summary>
        /// lobpcg (STANDARD, B=I) primitive, preconditioned. Forwards into the generalized
        /// <see cref="lobpcg{TOp,TBOp,TPre}"/> with <see cref="fProxyIdentityOperator"/> in the B
        /// slot -- see the class doc comment's "B=I strategy" note for why this reproduces the
        /// pre-generalization algorithm bit-for-bit (identity's Apply is an exact bit-copy) rather
        /// than requiring a hand-duplicated standard-only implementation.
        /// </summary>
        public static LOBPCGInfo lobpcg<TOp, TPre>(in TOp A, in TPre M, ref fProxyLOBPCGCache ws, int k, fProxy tol, int maxIter)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => lobpcg(in A, new fProxyIdentityOperator(A.Rows), in M, ref ws, k, tol, maxIter);

        /// <summary>lobpcg (preconditioned) with default maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg<TOp, TPre>(in TOp A, in TPre M, ref fProxyLOBPCGCache ws, int k, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => lobpcg(in A, in M, ref ws, k, tol, 1000);

        /// <summary>lobpcg (preconditioned) with default tol (Consts.fProxySqrtEps) and maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg<TOp, TPre>(in TOp A, in TPre M, ref fProxyLOBPCGCache ws, int k)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner
            => lobpcg(in A, in M, ref ws, k, Consts.fProxySqrtEps, 1000);

        /// <summary>
        /// Zero-alloc (at O(n) scale) LOBPCG primitive, UNPRECONDITIONED. Forwards into
        /// <see cref="lobpcg{TOp,TPre}"/> via <see cref="fProxyIdentityPreconditioner"/> -- a
        /// one-line forwarder rather than a hand-duplicated loop (unlike <see cref="Krylov.cg{TOp}"/>
        /// / <see cref="Krylov.pcg{TOp,TPre}"/>'s literal duplication -- LOBPCG's loop is
        /// considerably larger, so this method mirrors the SAME "single source of truth, thin
        /// forwarder" pattern already used everywhere else in this file for dense/BSR wrapping,
        /// just applied one level further).
        /// </summary>
        public static LOBPCGInfo lobpcg<TOp>(in TOp A, ref fProxyLOBPCGCache ws, int k, fProxy tol, int maxIter)
            where TOp : struct, IfProxyLinearOperator
            => lobpcg(in A, new fProxyIdentityPreconditioner(), ref ws, k, tol, maxIter);

        /// <summary>lobpcg (unpreconditioned) with default maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg<TOp>(in TOp A, ref fProxyLOBPCGCache ws, int k, fProxy tol)
            where TOp : struct, IfProxyLinearOperator
            => lobpcg(in A, ref ws, k, tol, 1000);

        /// <summary>lobpcg (unpreconditioned) with default tol (Consts.fProxySqrtEps) and maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg<TOp>(in TOp A, ref fProxyLOBPCGCache ws, int k)
            where TOp : struct, IfProxyLinearOperator
            => lobpcg(in A, ref ws, k, Consts.fProxySqrtEps, 1000);

        // NOTE: there is deliberately NO standalone `lobpcg<TOp,TBOp>(in TOp A, in TBOp B, ref ws,
        // int k, fProxy tol, int maxIter)` unpreconditioned-generic-generalized convenience overload
        // mirroring `lobpcg<TOp>` above: with 2 open type parameters and the same positional value-
        // argument shape (T, T, ref cache, int, fProxy, int), it would be IDENTICAL, at the C#
        // signature level, to the existing `lobpcg<TOp,TPre>(in TOp A, in TPre M, ref ws, int k,
        // fProxy tol, int maxIter)` above -- generic constraints (IfProxyLinearOperator vs
        // IfProxyPreconditioner) do NOT participate in overload/signature matching, so declaring
        // both would be a compile error (CS0111, "already defines a member with the same parameter
        // types"), not merely an ambiguous call. Callers who have their own custom TOp/TBOp operator
        // structs and want an unpreconditioned generalized solve call the 3-type-param core directly
        // with an explicit identity preconditioner: <c>lobpcg(in A, in B, new
        // fProxyIdentityPreconditioner(), ref ws, k, tol, maxIter)</c>. The CONCRETE (dense/BSR)
        // unpreconditioned-generalized overloads below are unaffected (they forward straight into
        // the 3-type-param core with an inline identity preconditioner) since concrete parameter
        // types never collide the way two same-shaped open generic signatures do.

        /// <summary>
        /// LOBPCG over a dense <see cref="fProxyMxN"/> -- zero-alloc primitive, unpreconditioned.
        /// Forwards into <see cref="lobpcg{TOp}"/> via <see cref="fProxyDenseOperator"/>.
        /// </summary>
        public static LOBPCGInfo lobpcg(in fProxyMxN A, ref fProxyLOBPCGCache ws, int k, fProxy tol, int maxIter)
            => lobpcg(new fProxyDenseOperator(in A), ref ws, k, tol, maxIter);

        /// <summary>lobpcg over a dense matrix with default maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in fProxyMxN A, ref fProxyLOBPCGCache ws, int k, fProxy tol)
            => lobpcg(in A, ref ws, k, tol, 1000);

        /// <summary>lobpcg over a dense matrix with default tol (Consts.fProxySqrtEps) and maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in fProxyMxN A, ref fProxyLOBPCGCache ws, int k)
            => lobpcg(in A, ref ws, k, Consts.fProxySqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a dense pencil (A, B) -- GENERALIZED eigenproblem A x = lambda B x, zero-alloc
        /// primitive, unpreconditioned. Forwards DIRECTLY into the 3-type-param
        /// <see cref="lobpcg{TOp,TBOp,TPre}"/> core with an inline <see cref="fProxyIdentityPreconditioner"/>
        /// (see the NOTE above this dense group's generic sibling would have occupied -- a standalone
        /// generic unpreconditioned-generalized rung is not declarable, but this CONCRETE overload is
        /// unaffected). See the class doc comment's "Buckling mapping" note for the canonical
        /// A=K_G/B=K_E truss-buckling usage.
        /// </summary>
        public static LOBPCGInfo lobpcg(in fProxyMxN A, in fProxyMxN B, ref fProxyLOBPCGCache ws, int k, fProxy tol, int maxIter)
            => lobpcg(new fProxyDenseOperator(in A), new fProxyDenseOperator(in B), new fProxyIdentityPreconditioner(), ref ws, k, tol, maxIter);

        /// <summary>lobpcg (generalized) over a dense pencil with default maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in fProxyMxN A, in fProxyMxN B, ref fProxyLOBPCGCache ws, int k, fProxy tol)
            => lobpcg(in A, in B, ref ws, k, tol, 1000);

        /// <summary>lobpcg (generalized) over a dense pencil with default tol (Consts.fProxySqrtEps) and maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in fProxyMxN A, in fProxyMxN B, ref fProxyLOBPCGCache ws, int k)
            => lobpcg(in A, in B, ref ws, k, Consts.fProxySqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a dense SYMMETRIC matrix -- allocates the workspace from <paramref
        /// name="arena"/> and calls the zero-alloc primitive. Returns the eigenvalues (length k,
        /// ASCENDING); <paramref name="eigenvectors"/> is k x n (row i = eigenvector i). Both
        /// buffers are the workspace's OWN X/lambda (no extra copy) -- the rest of the workspace
        /// stays allocated in the arena, unused after this call, exactly like every other
        /// arena-convenience wrapper in this library (e.g. <see cref="Eigen.lanczosVectors{TOp}"/>).
        /// </summary>
        public static fProxyN lobpcg(ref Arena arena, in fProxyMxN A, int k, out fProxyMxN eigenvectors,
                                      out LOBPCGInfo info, fProxy tol, int maxIter)
        {
            var ws = arena.fProxyLOBPCGCache(A.M_Rows, k);
            info = lobpcg(in A, ref ws, k, tol, maxIter);
            eigenvectors = ws.X;
            return ws.lambda;
        }

        /// <summary>lobpcg (allocating) over a dense matrix with default tol/maxIter.</summary>
        public static fProxyN lobpcg(ref Arena arena, in fProxyMxN A, int k, out fProxyMxN eigenvectors, out LOBPCGInfo info)
            => lobpcg(ref arena, in A, k, out eigenvectors, out info, Consts.fProxySqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a dense symmetric matrix with <paramref name="guard"/> GUARD ("ghost") vectors --
        /// allocating. Iterates on a block of k + guard vectors but converges and RETURNS only the k
        /// smallest pairs; the guards are full participants in the iteration and give the wanted pairs
        /// spectral separation room, which accelerates -- and, for a near-degenerate/clustered bottom,
        /// rescues -- convergence. A typical guard is a small handful (e.g. 3-8); <paramref name="guard"/>
        /// = 0 reproduces the guard-free overload exactly. <paramref name="eigenvectors"/> is the leading
        /// k x n block (ascending-sorted); the returned values are a fresh length-k vector.
        /// </summary>
        public static fProxyN lobpcg(ref Arena arena, in fProxyMxN A, int k, int guard, out fProxyMxN eigenvectors,
                                      out LOBPCGInfo info, fProxy tol, int maxIter)
        {
            if (guard < 0) throw new ArgumentException("LOBPCG: guard must be >= 0");
            var ws = arena.fProxyLOBPCGCache(A.M_Rows, k + guard);
            info = lobpcg(in A, ref ws, k, tol, maxIter);
            eigenvectors = ws.X;
            eigenvectors.M_Rows = k;                       // leading k rows are the k smallest (SortAscending ran)
            var vals = arena.fProxyVec(k);
            for (int i = 0; i < k; i++) vals[i] = ws.lambda[i];
            return vals;
        }

        /// <summary>lobpcg (allocating, guard vectors) over a dense matrix with default tol/maxIter.</summary>
        public static fProxyN lobpcg(ref Arena arena, in fProxyMxN A, int k, int guard, out fProxyMxN eigenvectors, out LOBPCGInfo info)
            => lobpcg(ref arena, in A, k, guard, out eigenvectors, out info, Consts.fProxySqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a dense pencil (A, B) -- GENERALIZED eigenproblem, allocating. See the
        /// standard dense overload's doc comment for the buffer-ownership contract.
        /// </summary>
        public static fProxyN lobpcg(ref Arena arena, in fProxyMxN A, in fProxyMxN B, int k, out fProxyMxN eigenvectors,
                                      out LOBPCGInfo info, fProxy tol, int maxIter)
        {
            var ws = arena.fProxyLOBPCGCache(A.M_Rows, k);
            info = lobpcg(in A, in B, ref ws, k, tol, maxIter);
            eigenvectors = ws.X;
            return ws.lambda;
        }

        /// <summary>lobpcg (allocating, generalized) over a dense pencil with default tol/maxIter.</summary>
        public static fProxyN lobpcg(ref Arena arena, in fProxyMxN A, in fProxyMxN B, int k, out fProxyMxN eigenvectors, out LOBPCGInfo info)
            => lobpcg(ref arena, in A, in B, k, out eigenvectors, out info, Consts.fProxySqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a block-sparse (BSR) matrix -- zero-alloc primitive, unpreconditioned.
        /// Forwards into <see cref="lobpcg{TOp}"/> via <c>fProxyBSROperator</c>.
        /// </summary>
        public static LOBPCGInfo lobpcg(in fProxyBSR A, ref fProxyLOBPCGCache ws, int k, fProxy tol, int maxIter)
            => lobpcg(new fProxyBSROperator(in A), ref ws, k, tol, maxIter);

        /// <summary>lobpcg over a BSR matrix with default maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in fProxyBSR A, ref fProxyLOBPCGCache ws, int k, fProxy tol)
            => lobpcg(in A, ref ws, k, tol, 1000);

        /// <summary>lobpcg over a BSR matrix with default tol (Consts.fProxySqrtEps) and maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in fProxyBSR A, ref fProxyLOBPCGCache ws, int k)
            => lobpcg(in A, ref ws, k, Consts.fProxySqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a block-sparse pencil (A, B) -- GENERALIZED eigenproblem, zero-alloc
        /// primitive, unpreconditioned. Forwards DIRECTLY into the 3-type-param
        /// <see cref="lobpcg{TOp,TBOp,TPre}"/> core (via <c>fProxyBSROperator</c> for both A and B)
        /// with an inline <see cref="fProxyIdentityPreconditioner"/> -- see the dense pencil
        /// overload's NOTE for why there is no generic unpreconditioned-generalized rung to route
        /// through here.
        /// </summary>
        public static LOBPCGInfo lobpcg(in fProxyBSR A, in fProxyBSR B, ref fProxyLOBPCGCache ws, int k, fProxy tol, int maxIter)
            => lobpcg(new fProxyBSROperator(in A), new fProxyBSROperator(in B), new fProxyIdentityPreconditioner(), ref ws, k, tol, maxIter);

        /// <summary>lobpcg (generalized) over a BSR pencil with default maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in fProxyBSR A, in fProxyBSR B, ref fProxyLOBPCGCache ws, int k, fProxy tol)
            => lobpcg(in A, in B, ref ws, k, tol, 1000);

        /// <summary>lobpcg (generalized) over a BSR pencil with default tol (Consts.fProxySqrtEps) and maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in fProxyBSR A, in fProxyBSR B, ref fProxyLOBPCGCache ws, int k)
            => lobpcg(in A, in B, ref ws, k, Consts.fProxySqrtEps, 1000);

        /// <summary>lobpcg (allocating) over a BSR matrix. See the dense overload's doc comment.</summary>
        public static fProxyN lobpcg(ref Arena arena, in fProxyBSR A, int k, out fProxyMxN eigenvectors,
                                      out LOBPCGInfo info, fProxy tol, int maxIter)
        {
            var ws = arena.fProxyLOBPCGCache(A.M_Rows, k);
            info = lobpcg(in A, ref ws, k, tol, maxIter);
            eigenvectors = ws.X;
            return ws.lambda;
        }

        /// <summary>lobpcg (allocating) over a BSR matrix with default tol/maxIter.</summary>
        public static fProxyN lobpcg(ref Arena arena, in fProxyBSR A, int k, out fProxyMxN eigenvectors, out LOBPCGInfo info)
            => lobpcg(ref arena, in A, k, out eigenvectors, out info, Consts.fProxySqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a block-sparse (BSR) matrix with <paramref name="guard"/> GUARD ("ghost") vectors
        /// -- allocating. See the dense guard overload's doc comment: iterates on k + guard vectors,
        /// returns the k smallest. This is the recommended entry point for the smallest eigenpairs of a
        /// large sparse operator whose bottom spectrum is clustered/near-degenerate (e.g. a grid
        /// Laplacian -- see <c>fProxyGallery.fProxyLaplacian2D</c>).
        /// </summary>
        public static fProxyN lobpcg(ref Arena arena, in fProxyBSR A, int k, int guard, out fProxyMxN eigenvectors,
                                      out LOBPCGInfo info, fProxy tol, int maxIter)
        {
            if (guard < 0) throw new ArgumentException("LOBPCG: guard must be >= 0");
            var ws = arena.fProxyLOBPCGCache(A.M_Rows, k + guard);
            info = lobpcg(in A, ref ws, k, tol, maxIter);
            eigenvectors = ws.X;
            eigenvectors.M_Rows = k;
            var vals = arena.fProxyVec(k);
            for (int i = 0; i < k; i++) vals[i] = ws.lambda[i];
            return vals;
        }

        /// <summary>lobpcg (allocating, guard vectors) over a BSR matrix with default tol/maxIter.</summary>
        public static fProxyN lobpcg(ref Arena arena, in fProxyBSR A, int k, int guard, out fProxyMxN eigenvectors, out LOBPCGInfo info)
            => lobpcg(ref arena, in A, k, guard, out eigenvectors, out info, Consts.fProxySqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a block-sparse pencil (A, B) -- GENERALIZED eigenproblem, allocating. See
        /// the standard BSR overload's doc comment for the buffer-ownership contract.
        /// </summary>
        public static fProxyN lobpcg(ref Arena arena, in fProxyBSR A, in fProxyBSR B, int k, out fProxyMxN eigenvectors,
                                      out LOBPCGInfo info, fProxy tol, int maxIter)
        {
            var ws = arena.fProxyLOBPCGCache(A.M_Rows, k);
            info = lobpcg(in A, in B, ref ws, k, tol, maxIter);
            eigenvectors = ws.X;
            return ws.lambda;
        }

        /// <summary>lobpcg (allocating, generalized) over a BSR pencil with default tol/maxIter.</summary>
        public static fProxyN lobpcg(ref Arena arena, in fProxyBSR A, in fProxyBSR B, int k, out fProxyMxN eigenvectors, out LOBPCGInfo info)
            => lobpcg(ref arena, in A, in B, k, out eigenvectors, out info, Consts.fProxySqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a block-sparse (BSR) matrix with its matching block-Jacobi preconditioner --
        /// zero-alloc primitive. Forwards into <see cref="lobpcg{TOp,TPre}"/> via
        /// <c>fProxyBSROperator</c>/<c>fProxyBlockJacobi</c>. This is the preconditioned entry point
        /// the sparse-BSM eigensolver roadmap calls out (matvec + block-Jacobi, matching how
        /// <see cref="Krylov.pcg(in fProxyBSR, in fProxyBlockJacobi, in fProxyN, ref fProxyN)"/>
        /// consumes it).
        /// </summary>
        public static LOBPCGInfo lobpcg(in fProxyBSR A, in fProxyBlockJacobi M, ref fProxyLOBPCGCache ws,
                                         int k, fProxy tol, int maxIter)
            => lobpcg(new fProxyBSROperator(in A), in M, ref ws, k, tol, maxIter);

        /// <summary>lobpcg (BSR + block-Jacobi) with default maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in fProxyBSR A, in fProxyBlockJacobi M, ref fProxyLOBPCGCache ws, int k, fProxy tol)
            => lobpcg(in A, in M, ref ws, k, tol, 1000);

        /// <summary>lobpcg (BSR + block-Jacobi) with default tol (Consts.fProxySqrtEps) and maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in fProxyBSR A, in fProxyBlockJacobi M, ref fProxyLOBPCGCache ws, int k)
            => lobpcg(in A, in M, ref ws, k, Consts.fProxySqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a block-sparse pencil (A, B) -- GENERALIZED eigenproblem -- with A's matching
        /// block-Jacobi preconditioner. Forwards into <see cref="lobpcg{TOp,TBOp,TPre}"/> via
        /// <c>fProxyBSROperator</c>/<c>fProxyBlockJacobi</c>. Note the preconditioner M is built from
        /// (and approximates the inverse of) A only -- it operates on the RAW residual r = A x -
        /// lambda B x exactly like the standard-path block-Jacobi preconditioner does, B does not
        /// enter M's construction or Apply.
        /// </summary>
        public static LOBPCGInfo lobpcg(in fProxyBSR A, in fProxyBSR B, in fProxyBlockJacobi M, ref fProxyLOBPCGCache ws,
                                         int k, fProxy tol, int maxIter)
            => lobpcg(new fProxyBSROperator(in A), new fProxyBSROperator(in B), in M, ref ws, k, tol, maxIter);

        /// <summary>lobpcg (generalized, BSR + block-Jacobi) with default maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in fProxyBSR A, in fProxyBSR B, in fProxyBlockJacobi M, ref fProxyLOBPCGCache ws, int k, fProxy tol)
            => lobpcg(in A, in B, in M, ref ws, k, tol, 1000);

        /// <summary>lobpcg (generalized, BSR + block-Jacobi) with default tol (Consts.fProxySqrtEps) and maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in fProxyBSR A, in fProxyBSR B, in fProxyBlockJacobi M, ref fProxyLOBPCGCache ws, int k)
            => lobpcg(in A, in B, in M, ref ws, k, Consts.fProxySqrtEps, 1000);

        /// <summary>lobpcg (allocating) over a BSR matrix with block-Jacobi. See the dense overload's doc comment.</summary>
        public static fProxyN lobpcg(ref Arena arena, in fProxyBSR A, in fProxyBlockJacobi M, int k,
                                      out fProxyMxN eigenvectors, out LOBPCGInfo info, fProxy tol, int maxIter)
        {
            var ws = arena.fProxyLOBPCGCache(A.M_Rows, k);
            info = lobpcg(in A, in M, ref ws, k, tol, maxIter);
            eigenvectors = ws.X;
            return ws.lambda;
        }

        /// <summary>lobpcg (allocating) over a BSR matrix with block-Jacobi and default tol/maxIter.</summary>
        public static fProxyN lobpcg(ref Arena arena, in fProxyBSR A, in fProxyBlockJacobi M, int k,
                                      out fProxyMxN eigenvectors, out LOBPCGInfo info)
            => lobpcg(ref arena, in A, in M, k, out eigenvectors, out info, Consts.fProxySqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a block-sparse pencil (A, B) with block-Jacobi -- GENERALIZED eigenproblem,
        /// allocating. See the standard BSR+block-Jacobi overload's doc comment.
        /// </summary>
        public static fProxyN lobpcg(ref Arena arena, in fProxyBSR A, in fProxyBSR B, in fProxyBlockJacobi M, int k,
                                      out fProxyMxN eigenvectors, out LOBPCGInfo info, fProxy tol, int maxIter)
        {
            var ws = arena.fProxyLOBPCGCache(A.M_Rows, k);
            info = lobpcg(in A, in B, in M, ref ws, k, tol, maxIter);
            eigenvectors = ws.X;
            return ws.lambda;
        }

        /// <summary>lobpcg (allocating, generalized) over a BSR pencil with block-Jacobi and default tol/maxIter.</summary>
        public static fProxyN lobpcg(ref Arena arena, in fProxyBSR A, in fProxyBSR B, in fProxyBlockJacobi M, int k,
                                      out fProxyMxN eigenvectors, out LOBPCGInfo info)
            => lobpcg(ref arena, in A, in B, in M, k, out eigenvectors, out info, Consts.fProxySqrtEps, 1000);

        // ==================================================================================
        // Private helpers
        // ==================================================================================

        // Aliasing guard: every scratch buffer in the workspace must be distinct -- same rationale
        // as cg<TOp>'s guard (elementwise updates below don't self-check aliasing). A local
        // loop-based check (mirrors Krylov.RequireDistinctBuffers) rather than a hand-expanded OR
        // chain: 25 buffers -> 300 pairs, impractical to hand-write/review. Includes the O(k)-scale
        // Rayleigh-Ritz scratch (Gram/H/L/Atrans/Y/C) alongside the O(n)-scale buffers -- all six
        // live simultaneously within a single TryRayleighRitz call (Gram/H built together, L
        // factored from Gram, Atrans formed from H/L, Y from Atrans, C from Y/L), so an aliased
        // pair among THEM is just as much a correctness hazard as an aliased O(n) pair. BX/BW/BP
        // (the generalized-eigenproblem B-images of X/W/P) and rowAux (OrthonormalizeBlockB's third
        // row-combination scratch) are included for the SAME reason -- every one of them is live
        // simultaneously with the buffers it is combined against.
        static unsafe void RequireDistinctBuffers(in fProxyLOBPCGCache ws)
        {
            const int count = 25;
            long* ptrs = stackalloc long[count];
            ptrs[0] = (long)ws.X.Data.Ptr;
            ptrs[1] = (long)ws.AX.Data.Ptr;
            ptrs[2] = (long)ws.W.Data.Ptr;
            ptrs[3] = (long)ws.AW.Data.Ptr;
            ptrs[4] = (long)ws.P.Data.Ptr;
            ptrs[5] = (long)ws.AP.Data.Ptr;
            ptrs[6] = (long)ws.R.Data.Ptr;
            ptrs[7] = (long)ws.Xnext.Data.Ptr;
            ptrs[8] = (long)ws.AXnext.Data.Ptr;
            ptrs[9] = (long)ws.Pnext.Data.Ptr;
            ptrs[10] = (long)ws.APnext.Data.Ptr;
            ptrs[11] = (long)ws.lambda.Data.Ptr;
            ptrs[12] = (long)ws.residual.Data.Ptr;
            ptrs[13] = (long)ws.rowIn.Data.Ptr;
            ptrs[14] = (long)ws.rowOut.Data.Ptr;
            ptrs[15] = (long)ws.Gram.Data.Ptr;
            ptrs[16] = (long)ws.H.Data.Ptr;
            ptrs[17] = (long)ws.L.Data.Ptr;
            ptrs[18] = (long)ws.Atrans.Data.Ptr;
            ptrs[19] = (long)ws.Y.Data.Ptr;
            ptrs[20] = (long)ws.C.Data.Ptr;
            ptrs[21] = (long)ws.BX.Data.Ptr;
            ptrs[22] = (long)ws.BW.Data.Ptr;
            ptrs[23] = (long)ws.BP.Data.Ptr;
            ptrs[24] = (long)ws.rowAux.Data.Ptr;

            for (int i = 0; i < count; i++)
                for (int j = i + 1; j < count; j++)
                    if (ptrs[i] == ptrs[j])
                        throw new ArgumentException("LOBPCG: workspace buffers must be distinct (use Arena.fProxyLOBPCGCache(n, k))");
        }

        // Same-buffer, smaller-shaped logical view: fProxyMxN.M_Rows/N_Cols are plain mutable
        // fields independent of the backing Data store, so a value-copy with adjusted dims is a
        // free reinterpretation of the SAME (larger, cache-owned) buffer's leading m x m block --
        // not a new allocation. See the class doc comment's "Zero-alloc scope" note.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static fProxyMxN View(in fProxyMxN buf, int m)
        {
            var v = buf;
            v.M_Rows = m;
            v.N_Cols = m;
            return v;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void SwapMat(ref fProxyMxN a, ref fProxyMxN b)
        {
            var t = a;
            a = b;
            b = t;
        }

        // Cholesky-QR-orthonormalizes the first `rows` rows of V IN PLACE, carrying AV along via
        // the SAME combination (linearity: A(sum_r c_r v_r) = sum_r c_r (A v_r), so AV never needs
        // a fresh matvec): G = V V^T, Cholesky-factor it (with FactorGram's own ridge retry), then
        // Vortho[i,:] = (V[i,:] - sum_{r<i} L[i,r]*Vortho[r,:]) / L[i,i] row by row (each row's
        // update only reads ALREADY-finalized earlier rows, so overwriting V/AV in place, top row
        // first, is safe), with the identical row combination applied to AV in lockstep. Used both
        // to orthonormalize the initial X seed and, every iteration, to internally orthonormalize
        // the (already X-deflated) W and P blocks -- see the class doc comment's "Orthogonalization"
        // note on why this is not redundant with the Rayleigh-Ritz Gram's own Cholesky step: without
        // it, W/P's arbitrary internal scale (e.g. once residuals have shrunk by very different
        // amounts across active pairs) can make the COMBINED Gram ill-conditioned enough that its
        // one Cholesky produces spurious Ritz values instead of tripping the rank-deficiency
        // safeguard. `rowTmp`/`rowTmp2` (length n each) are caller-provided scratch. Returns false
        // if V's rows cannot be orthonormalized at all (rank-deficient even after FactorGram's ridge
        // retry -- e.g. k > n for the initial seed, or W/P has collapsed onto the already-known
        // subspace); callers stall or drop P accordingly.
        static bool OrthonormalizeBlock(ref fProxyMxN V, ref fProxyMxN AV, int rows, int n,
                                         ref fProxyMxN Gram, ref fProxyMxN L, ref fProxyN rowTmp, ref fProxyN rowTmp2)
        {
            var G = View(in Gram, rows);
            var Lv = View(in L, rows);

            FillGramSub(ref G, 0, in V, rows, 0, in V, rows, n, true);

            if (!FactorGram(ref G, ref Lv, rows))
                return false;

            for (int i = 0; i < rows; i++)
            {
                for (int c = 0; c < n; c++) { rowTmp[c] = V[i, c]; rowTmp2[c] = AV[i, c]; }

                for (int r = 0; r < i; r++)
                {
                    fProxy coef = Lv[i, r];
                    for (int c = 0; c < n; c++)
                    {
                        rowTmp[c] -= coef * V[r, c];
                        rowTmp2[c] -= coef * AV[r, c];
                    }
                }

                fProxy inv = (fProxy)1 / Lv[i, i];
                for (int c = 0; c < n; c++)
                {
                    V[i, c] = rowTmp[c] * inv;
                    AV[i, c] = rowTmp2[c] * inv;
                }
            }

            return true;
        }

        // GENERALIZED (B-inner-product) sibling of OrthonormalizeBlock: Cholesky-QR-orthonormalizes
        // the first `rows` rows of V IN PLACE w.r.t. the B-inner-product (Gram = V^T B V instead of
        // V^T V), carrying AV *and* BV along via the SAME combination (linearity applies to BOTH A
        // and B: A(sum c_r v_r) = sum c_r (A v_r), likewise for B). Requires BV to already hold a
        // VALID B-image of V on entry (this routine only ever WRITES BV via the same row-combination
        // used for V/AV, it never independently recomputes it -- see the class doc's "fresh-matvec
        // principle extends to B": the caller is responsible for BV's freshness, this helper just
        // mirrors whatever transform it applies to V onto BV too). Used for the per-iteration W/P
        // blocks (never for the initial X seed, which stays Euclidean-only -- see
        // <see cref="OrthonormalizeBlock"/>'s own call site comment for why). `rowTmp`/`rowTmp2`/
        // `rowTmp3` (length n each) are caller-provided scratch for V/AV/BV respectively. Returns
        // false if V's rows cannot be B-orthonormalized at all (rank-deficient even after
        // FactorGram's ridge retry); callers stall or drop P accordingly, exactly like
        // OrthonormalizeBlock's own failure contract.
        static bool OrthonormalizeBlockB(ref fProxyMxN V, ref fProxyMxN AV, ref fProxyMxN BV, int rows, int n,
                                          ref fProxyMxN Gram, ref fProxyMxN L, ref fProxyN rowTmp, ref fProxyN rowTmp2, ref fProxyN rowTmp3)
        {
            var G = View(in Gram, rows);
            var Lv = View(in L, rows);

            FillGramSub(ref G, 0, in V, rows, 0, in BV, rows, n, true);

            if (!FactorGram(ref G, ref Lv, rows))
                return false;

            for (int i = 0; i < rows; i++)
            {
                for (int c = 0; c < n; c++) { rowTmp[c] = V[i, c]; rowTmp2[c] = AV[i, c]; rowTmp3[c] = BV[i, c]; }

                for (int r = 0; r < i; r++)
                {
                    fProxy coef = Lv[i, r];
                    for (int c = 0; c < n; c++)
                    {
                        rowTmp[c] -= coef * V[r, c];
                        rowTmp2[c] -= coef * AV[r, c];
                        rowTmp3[c] -= coef * BV[r, c];
                    }
                }

                fProxy inv = (fProxy)1 / Lv[i, i];
                for (int c = 0; c < n; c++)
                {
                    V[i, c] = rowTmp[c] * inv;
                    AV[i, c] = rowTmp2[c] * inv;
                    BV[i, c] = rowTmp3[c] * inv;
                }
            }

            return true;
        }

        // Fills Gram[rowOff+i, colOff+j] = dot(Vb[i,:], Wb[j,:]) and mirrors the SAME value into
        // [colOff+j, rowOff+i] (Gram is symmetric by construction). `sameBlock` (Vb/Wb identical,
        // rowOff==colOff) restricts the fill to the upper triangle i<=j to avoid redundant work.
        static void FillGramSub(ref fProxyMxN Gram, int rowOff, in fProxyMxN Vb, int rows,
                                 int colOff, in fProxyMxN Wb, int cols, int n, bool sameBlock)
        {
            for (int i = 0; i < rows; i++)
            {
                int jStart = sameBlock ? i : 0;
                for (int j = jStart; j < cols; j++)
                {
                    fProxy s = (fProxy)0;
                    for (int c = 0; c < n; c++) s += Vb[i, c] * Wb[j, c];
                    Gram[rowOff + i, colOff + j] = s;
                    Gram[colOff + j, rowOff + i] = s;
                }
            }
        }

        // Fills H[rowOff+i, colOff+j] = dot(Vb[i,:], AWb[j,:]) and mirrors the SAME value into
        // [colOff+j, rowOff+i]. Valid because A is symmetric: Vb_i . A Wb_j == Wb_j . A Vb_i
        // exactly, so one dot product correctly serves both slots -- guaranteeing exact symmetry
        // rather than "equal only to roundoff, computed twice".
        static void FillHSub(ref fProxyMxN H, int rowOff, in fProxyMxN Vb, int rows,
                              int colOff, in fProxyMxN AWb, int cols, int n, bool sameBlock)
        {
            for (int i = 0; i < rows; i++)
            {
                int jStart = sameBlock ? i : 0;
                for (int j = jStart; j < cols; j++)
                {
                    fProxy s = (fProxy)0;
                    for (int c = 0; c < n; c++) s += Vb[i, c] * AWb[j, c];
                    H[rowOff + i, colOff + j] = s;
                    H[colOff + j, rowOff + i] = s;
                }
            }
        }

        // Deflates each of the first `activeCount` rows of V/AV/BV against the first `againstCount`
        // rows of Against/AgainstA/AgainstB, TWICE (classical "reorthogonalize twice" folklore --
        // mirrors Eigen.lanczos's own choice), w.r.t. the B-INNER-PRODUCT: coeff = <Against_i,
        // V_a>_B = dot(AgainstB_i, V_a) (AgainstB = B * Against, already the B-image of the
        // reference block -- valid on entry, e.g. ws.BX or an already-B-deflated ws.BW). AV/BV are
        // updated with the SAME coefficient (linearity: A(v - c x) = Av - c(Ax), likewise for B: B(v
        // - c x) = Bv - c(Bx)), so this never costs an extra matvec for either. For B=I,
        // AgainstB bits are identical to Against bits (see the class doc's "B=I strategy"), so
        // coeff and every update below reproduce the pre-generalization Euclidean Deflate formula
        // bit-for-bit.
        static void Deflate(ref fProxyMxN V, ref fProxyMxN AV, ref fProxyMxN BV, int activeCount,
                             in fProxyMxN Against, in fProxyMxN AgainstA, in fProxyMxN AgainstB, int againstStart, int againstCount, int n)
        {
            for (int pass = 0; pass < 2; pass++)
                for (int a = 0; a < activeCount; a++)
                    for (int i = againstStart; i < againstStart + againstCount; i++)
                    {
                        fProxy coeff = (fProxy)0;
                        for (int c = 0; c < n; c++) coeff += AgainstB[i, c] * V[a, c];

                        for (int c = 0; c < n; c++)
                        {
                            V[a, c] -= coeff * Against[i, c];
                            AV[a, c] -= coeff * AgainstA[i, c];
                            BV[a, c] -= coeff * AgainstB[i, c];
                        }
                    }
        }

        // Builds the combined Gram/H (2-block [X,W] or 3-block [X,W,P]) for the active window.
        // Gram is now the B-Gram (S^T B S): every FillGramSub call reads the RAW block as its first
        // operand and that block's B-IMAGE (BX/BW/BP) as its second -- for B=I those B-images are
        // bit-identical copies (see the class doc's "B=I strategy"), so this reproduces the
        // pre-generalization Euclidean Gram (S^T S) bit-for-bit. H stays S^T A S, unaffected by B.
        static void BuildProjected(ref fProxyMxN Gram, ref fProxyMxN H, int numActive, bool useP,
                                    in fProxyMxN X, in fProxyMxN AX, in fProxyMxN BX,
                                    in fProxyMxN W, in fProxyMxN AW, in fProxyMxN BW,
                                    in fProxyMxN P, in fProxyMxN AP, in fProxyMxN BP, int n)
        {
            int offX = 0, offW = numActive, offP = 2 * numActive;

            FillGramSub(ref Gram, offX, in X, numActive, offX, in BX, numActive, n, true);
            FillGramSub(ref Gram, offX, in X, numActive, offW, in BW, numActive, n, false);
            FillGramSub(ref Gram, offW, in W, numActive, offW, in BW, numActive, n, true);

            FillHSub(ref H, offX, in X, numActive, offX, in AX, numActive, n, true);
            FillHSub(ref H, offX, in X, numActive, offW, in AW, numActive, n, false);
            FillHSub(ref H, offW, in W, numActive, offW, in AW, numActive, n, true);

            if (useP)
            {
                FillGramSub(ref Gram, offX, in X, numActive, offP, in BP, numActive, n, false);
                FillGramSub(ref Gram, offW, in W, numActive, offP, in BP, numActive, n, false);
                FillGramSub(ref Gram, offP, in P, numActive, offP, in BP, numActive, n, true);

                FillHSub(ref H, offX, in X, numActive, offP, in AP, numActive, n, false);
                FillHSub(ref H, offW, in W, numActive, offP, in AP, numActive, n, false);
                FillHSub(ref H, offP, in P, numActive, offP, in AP, numActive, n, true);
            }
        }

        // Attempts Cholesky of an m x m Gram view; on failure OR a suspiciously tiny relative
        // pivot, adds a small Tikhonov ridge (scaled to Gram's own diagonal) and retries once --
        // mirrors CHOP.decompSolve's own ridge-retry recovery for a borderline
        // semidefinite Gram. The ridge-retry attempt is checked against the SAME pivotRelTol (not
        // merely "Cholesky did not report a negative pivot") -- a ridge just barely large enough to
        // make Gram numerically SPD can still leave L badly conditioned, and accepting that
        // unconditionally would reintroduce exactly the failure this check exists to catch. Returns
        // false only if BOTH attempts fail the threshold.
        static bool FactorGram(ref fProxyMxN Gram, ref fProxyMxN L, int m)
        {
            // Relative-pivot tolerance for the "tiny diagonal" rank-deficiency check. A method-
            // local value (not a class-level const): Cholesky/QR/etc. in this codebase already
            // hoist their tuning constants into method scope for the same reason -- a class-level
            // const of the same name would collide across the float/double generated partials
            // (CS0102; see CHO.fProxy.cs's CHOL_BLOCK comment).
            fProxy pivotRelTol = Consts.fProxySqrtEps;

            var info = CHO.decomp(in Gram, ref L);
            if (info.Solved && MinMaxDiagRatio(in L, m) >= pivotRelTol)
                return true;

            fProxy scale = (fProxy)0;
            for (int i = 0; i < m; i++) { fProxy d = math.abs(Gram[i, i]); if (d > scale) scale = d; }
            fProxy ridge = (fProxy)m * Consts.fProxyEpsilon * scale;
            if (!(ridge > (fProxy)0)) ridge = Consts.fProxyEpsilon;

            for (int i = 0; i < m; i++) Gram[i, i] += ridge;

            info = CHO.decomp(in Gram, ref L);
            return info.Solved && MinMaxDiagRatio(in L, m) >= pivotRelTol;
        }

        static fProxy MinMaxDiagRatio(in fProxyMxN L, int m)
        {
            fProxy mn = math.abs(L[0, 0]);
            fProxy mx = mn;
            for (int i = 1; i < m; i++)
            {
                fProxy d = math.abs(L[i, i]);
                if (d < mn) mn = d;
                if (d > mx) mx = d;
            }
            return mx > (fProxy)0 ? mn / mx : (fProxy)0;
        }

        // Attempts the full Rayleigh-Ritz reduction for the active window (2-block or 3-block per
        // `useP`): build Gram/H, factor Gram (with retry), reduce to the standard eigenproblem
        // Ahat = L^-1 H L^-T, solve it, recover the combination coefficients C = L^-T Y, and write
        // the selected (smallest numActive) Ritz values directly into ws.lambda[0..numActive) --
        // done here (rather than returned) because the eigenvalues live in a small Allocator.Temp
        // buffer that is disposed before this method returns. Returns false (caller falls back or
        // stalls) if Cholesky or the small eigensolve fails.
        static bool TryRayleighRitz(ref fProxyLOBPCGCache ws, int numActive, bool useP, int n)
        {
            int m = useP ? 3 * numActive : 2 * numActive;

            var G = View(in ws.Gram, m);
            var Hv = View(in ws.H, m);
            var Lv = View(in ws.L, m);

            BuildProjected(ref G, ref Hv, numActive, useP, in ws.X, in ws.AX, in ws.BX, in ws.W, in ws.AW, in ws.BW, in ws.P, in ws.AP, in ws.BP, n);

            // Cheap, numerically TRUSTWORTHY plausibility envelope for the eventual Ritz values
            // (safeguard 3), computed BEFORE the Cholesky-based reduction below: H[i,i]/G[i,i] is
            // the exact GENERALIZED Rayleigh quotient (s^T A s)/(s^T B s) of one individual,
            // already-B-unit-normalized basis row (a row of X, W, or P) -- a plain ratio of two
            // already-computed dot products, no matrix inversion involved, so it is immune to the
            // ill-conditioning that can corrupt the L^-1 H L^-T transform. Every individual
            // generalized Rayleigh quotient is, by definition, within [lambda_min, lambda_max] of
            // the pencil (A, B) -- the SAME immunity argument as the standard (B=I) case, which is
            // just this formula's Gram[i,i]==1 special case. If the Ritz values symmetric
            // returns fall wildly outside the range spanned by these trustworthy individual
            // quotients, that is conclusive evidence the transform corrupted the problem -- observed: Ritz values far below
            // lambda_min, even wildly negative (down to -1E13 and beyond), while the Cholesky
            // diag-ratio check (FactorGram's pivotRelTol) reported a perfectly comfortable pivot;
            // diagRatio alone is a poor proxy for THIS failure mode. Reject the whole attempt (so
            // the caller falls back to dropping P, or stalls) instead of locking in garbage.
            fProxy qMin = fProxy.MaxValue, qMax = -fProxy.MaxValue;
            for (int i = 0; i < m; i++)
            {
                fProxy gi = G[i, i];
                if (!(gi > (fProxy)0)) continue; // shouldn't happen for a normalized block; skip defensively
                fProxy q = Hv[i, i] / gi;
                if (q < qMin) qMin = q;
                if (q > qMax) qMax = q;
            }
            // Generous margin: a genuine Ritz value from superposing these rows can exceed their
            // individual quotient range somewhat, but never by orders of magnitude -- 1000x the
            // observed span (plus a small additive floor for the degenerate span==0 case)
            // comfortably separates that from actual numerical garbage (observed magnitudes
            // exceeded this envelope by 1E5-1E30x).
            fProxy margin = (fProxy)1000 * (qMax - qMin + (fProxy)1);
            fProxy envMin = qMin - margin;
            fProxy envMax = qMax + margin;

            if (!FactorGram(ref G, ref Lv, m))
                return false;

            var Atrans = View(in ws.Atrans, m);
            FormAtrans(ref Hv, ref Lv, ref Atrans, m);

            // Symmetrize (roundoff insurance -- symmetric requires exact-within-eps symmetry;
            // the two triangular-solve passes above are mathematically symmetric but not bit-exact).
            for (int i = 0; i < m; i++)
                for (int j = i + 1; j < m; j++)
                {
                    fProxy avg = (fProxy)0.5 * (Atrans[i, j] + Atrans[j, i]);
                    Atrans[i, j] = avg;
                    Atrans[j, i] = avg;
                }

            var Yv = View(in ws.Y, m);
            var eigSmall = new fProxyN(m, Allocator.Temp);
            bool eigOk = Eigen.symmetric(ref Atrans, ref eigSmall, ref Yv);

            if (eigOk)
            {
                // Validate the selected candidates against the trustworthy envelope BEFORE
                // committing them to ws.lambda (see safeguard 3 above).
                for (int j = 0; j < numActive; j++)
                {
                    fProxy candidate = eigSmall[m - numActive + j];
                    if (!math.isfinite(candidate) || candidate < envMin || candidate > envMax)
                    {
                        eigOk = false;
                        break;
                    }
                }
            }

            if (eigOk)
            {
                var Cv = View(in ws.C, m);
                RecoverC(ref Yv, ref Lv, ref Cv, m);

                // symmetric sorts DESCENDING -- the numActive SMALLEST real eigenvalues are
                // the LAST numActive entries.
                for (int j = 0; j < numActive; j++)
                    ws.lambda[j] = eigSmall[m - numActive + j];
            }

            eigSmall.Dispose();
            return eigOk;
        }

        // Ahat = L^-1 H L^-T via two triangular-solve passes: first each ROW of H is forward-solved
        // against L (giving H L^-T, since (H_row L^-T)^T = L^-1 H_row^T), stored into Atrans; then
        // each COLUMN of that result is forward-solved against L again (L^-1 applied on the left).
        // `m` is small (<= 3k) so the row/column scratch is a bounded Allocator.Temp vector.
        static void FormAtrans(ref fProxyMxN H, ref fProxyMxN L, ref fProxyMxN Atrans, int m)
        {
            var tmp = new fProxyN(m, Allocator.Temp);

            for (int r = 0; r < m; r++)
            {
                for (int c = 0; c < m; c++) tmp[c] = H[r, c];
                Blas.triLower(ref L, ref tmp);
                for (int c = 0; c < m; c++) Atrans[r, c] = tmp[c];
            }

            for (int c = 0; c < m; c++)
            {
                for (int r = 0; r < m; r++) tmp[r] = Atrans[r, c];
                Blas.triLower(ref L, ref tmp);
                for (int r = 0; r < m; r++) Atrans[r, c] = tmp[r];
            }

            tmp.Dispose();
        }

        // c_j = L^-T y_j for every column j of Y (back-substitution against L read transposed,
        // (L^T)[r,c] = L[c,r] -- L^T is upper triangular). Mirrors CHO's own private
        // SolveUpperTriangularTransposed in miniature (duplicated here rather than exposing that
        // private helper across files for one small, stable, easily-reviewed piece of math).
        static void RecoverC(ref fProxyMxN Y, ref fProxyMxN L, ref fProxyMxN C, int m)
        {
            var tmp = new fProxyN(m, Allocator.Temp);

            for (int col = 0; col < m; col++)
            {
                for (int r = 0; r < m; r++) tmp[r] = Y[r, col];

                for (int r = m - 1; r >= 0; r--)
                {
                    fProxy sum = (fProxy)0;
                    for (int c = r + 1; c < m; c++) sum += L[c, r] * tmp[c];
                    tmp[r] = (tmp[r] - sum) / L[r, r];
                }

                for (int r = 0; r < m; r++) C[r, col] = tmp[r];
            }

            tmp.Dispose();
        }

        // Forms the new active X/P block from the recovered combination coefficients C
        // (ws.lambda[0..numActive) already holds the selected Ritz values -- see TryRayleighRitz).
        // X_new combines the FULL basis (X/W/[P] parts); P_new (the new "conjugate direction")
        // combines only the W/[P] parts (zero X-part). Written into the ping-pong Xnext/Pnext
        // buffers (the combination reads the CURRENT X/W/P, so it cannot safely write in place),
        // then swapped in; the frozen locked rows [numActive, k) are carried forward first since
        // Xnext is about to become the new X wholesale.
        //
        // Deliberately does NOT also mirror-combine AX/AP (or BX/BP) the same way (an earlier
        // version did, for AX/AP): the caller ALWAYS immediately recomputes AX/BX/AP/BP via a fresh
        // A.Apply/B.Apply right after this call returns (see the "AX/AP freshness" / "fresh-matvec
        // principle extends to B" class doc notes), which unconditionally overwrites whatever this
        // method would have written -- so computing axv/bxv/apv/bpv here was pure wasted work (an
        // extra O(3k) multiply-adds per element, i.e. O(3k^2 n) per iteration, doubled again for the
        // B-images) with the result immediately discarded.
        static void UpdateActiveBlock(ref fProxyLOBPCGCache ws, int numActive, bool useP, int n, int k)
        {
            int m = useP ? 3 * numActive : 2 * numActive;
            var Cv = View(in ws.C, m);

            for (int j = 0; j < numActive; j++)
            {
                int col = m - numActive + j;

                for (int c = 0; c < n; c++)
                {
                    fProxy xv = (fProxy)0;

                    for (int r = 0; r < numActive; r++)
                        xv += Cv[r, col] * ws.X[r, c];
                    for (int r = 0; r < numActive; r++)
                        xv += Cv[numActive + r, col] * ws.W[r, c];
                    if (useP)
                        for (int r = 0; r < numActive; r++)
                            xv += Cv[2 * numActive + r, col] * ws.P[r, c];

                    ws.Xnext[j, c] = xv;
                }

                for (int c = 0; c < n; c++)
                {
                    fProxy pv = (fProxy)0;

                    for (int r = 0; r < numActive; r++)
                        pv += Cv[numActive + r, col] * ws.W[r, c];
                    if (useP)
                        for (int r = 0; r < numActive; r++)
                            pv += Cv[2 * numActive + r, col] * ws.P[r, c];

                    ws.Pnext[j, c] = pv;
                }
            }

            for (int i = numActive; i < k; i++)
                for (int c = 0; c < n; c++)
                    ws.Xnext[i, c] = ws.X[i, c];

            SwapMat(ref ws.X, ref ws.Xnext);
            SwapMat(ref ws.P, ref ws.Pnext);
        }

        static void SortAscending(ref fProxyLOBPCGCache ws, int k)
        {
            for (int i = 0; i < k - 1; i++)
            {
                int best = i;
                for (int j = i + 1; j < k; j++)
                    if (ws.lambda[j] < ws.lambda[best]) best = j;

                if (best != i)
                {
                    Swap.Vec(ref ws.lambda, i, best);
                    Swap.Vec(ref ws.residual, i, best);
                    Swap.Rows(ref ws.X, i, best);
                    Swap.Rows(ref ws.AX, i, best);
                    // BX kept row-consistent with X/AX for the SAME "internally introspectable
                    // cache state" reason AX itself is swapped here -- neither actually matters
                    // functionally for a follow-up call on this SAME cache (both AX and BX are
                    // unconditionally refreshed via a fresh Apply at the very start of any call,
                    // warm-started or not).
                    Swap.Rows(ref ws.BX, i, best);
                }
            }
        }

        static double MaxRelResidual(in fProxyLOBPCGCache ws, int k)
        {
            double worst = 0;
            for (int i = 0; i < k; i++)
            {
                fProxy scale = math.abs(ws.lambda[i]);
                if (scale < (fProxy)1) scale = (fProxy)1;
                double rel = (double)(ws.residual[i] / scale);
                if (rel > worst) worst = rel;
            }
            return worst;
        }

        // GUARD-VECTOR convergence gate: true iff the k WANTED pairs -- the k SMALLEST by current Ritz
        // value among all kWork block rows -- are each within the requested tol. The (kWork - k) guard
        // rows are ignored. Rank is computed by value with a stable index tie-break so EXACTLY k rows
        // count as wanted even when Ritz values coincide (the square-grid multiplicity case). residual[]
        // is current for every row here (active rows from the just-finished scan, locked rows frozen at
        // their lock-time value <= lockTol < tol), and lambda[] holds every row's current Ritz value, so
        // no row ordering is assumed. kWork is tiny (<= a few dozen) so the O(kWork^2) rank scan is
        // negligible. Only ever called when kWork > k; the kWork == k path uses allWithinTol inline and
        // stays bit-identical to the pre-guard implementation.
        static bool WantedWithinTol(in fProxyLOBPCGCache ws, int kWork, int k, fProxy tol)
        {
            for (int i = 0; i < kWork; i++)
            {
                int rank = 0;
                for (int j = 0; j < kWork; j++)
                    if (ws.lambda[j] < ws.lambda[i] || (ws.lambda[j] == ws.lambda[i] && j < i)) rank++;
                if (rank >= k) continue;                 // row i is a guard (not among the k smallest)

                fProxy scale = math.abs(ws.lambda[i]);
                if (scale < (fProxy)1) scale = (fProxy)1;
                if (!(ws.residual[i] <= tol * scale)) return false;   // NaN-safe (!(NaN<=x) == true)
            }
            return true;
        }

        // How many of the k pairs are within the requested tol (NOT the stricter lock margin lockTol).
        // On a non-converged exit, k - numActive would count only pairs frozen to lockTol (= 0.1*tol) and
        // undercount pairs that reached tol but were never locked -- so report this instead. residual[] is
        // populated for all k here (active pairs from the last scan, locked pairs frozen at lock time).
        static int ConvergedWithinTol(in fProxyLOBPCGCache ws, int k, fProxy tol)
        {
            int c = 0;
            for (int i = 0; i < k; i++)
            {
                fProxy scale = math.abs(ws.lambda[i]);
                if (scale < (fProxy)1) scale = (fProxy)1;
                if (ws.residual[i] <= tol * scale) c++;
            }
            return c;
        }
    }
}
