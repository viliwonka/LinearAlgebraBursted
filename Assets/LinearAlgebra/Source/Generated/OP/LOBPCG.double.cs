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
    /// <see cref="IdoubleLinearOperator"/> / optional <see cref="IdoublePreconditioner"/> --
    /// same Burst-monomorphized static-dispatch shape as <see cref="Krylov.cg{TOp}"/> /
    /// <see cref="Krylov.pcg{TOp,TPre}"/>. Reuses the dense
    /// <see cref="Eigen.symmetric(ref doubleMxN, ref doubleN, ref doubleMxN)"/> solver for the
    /// small (&lt;= 3k) Rayleigh-Ritz sub-problem and <see cref="CHO.decomp(in doubleMxN, ref doubleMxN)"/>
    /// for both orthogonalization and the generalized-to-standard eigenproblem reduction.
    ///
    /// DESIGN NOTES (see the coder's final report for the full write-up; summarized here for
    /// future maintainers):
    ///
    /// <b>Block size / guard vectors:</b> the active block is exactly k (the requested count) --
    /// no extra guard vectors. Locking (below) already shrinks the active Rayleigh-Ritz problem as
    /// pairs converge, which is the main lever real implementations use guard vectors for; adding
    /// k+g guard columns would speed up the slowest-converging pair at the cost of a meaningfully
    /// larger per-iteration Gram/eigensolve, so this implementation leaves it out (a documented,
    /// reportable scope decision, not an oversight).
    ///
    /// <b>Locking:</b> deflation-based. Once a pair's relative residual meets <c>tol</c> it is
    /// LOCKED -- swapped to the back of the k-wide window (frozen, no longer touched: no matvec,
    /// no preconditioner Apply, no Rayleigh-Ritz) -- and every remaining active W/P direction is
    /// explicitly projected (deflated) off the FULL locked+active X block, so the active subspace
    /// never re-discovers an already-found eigenvector. This differs slightly from the most literal
    /// reading of "soft locking" (which keeps locked columns IN the Rayleigh-Ritz mix so they can
    /// still drift); deflating them out entirely is simpler, strictly cheaper (zero cost per locked
    /// pair per iteration, matching the spec's "excluded from W computation" requirement), and
    /// standard in the literature (Knyazev). Locked pairs stay exposed in the output X (frozen at
    /// their converged values) exactly as the spec requires.
    ///
    /// <b>Orthogonalization (safeguard 1):</b> two stages, both required. First, W is DEFLATED
    /// (projected, modified-Gram-Schmidt style, TWICE for stability -- mirrors
    /// <see cref="Eigen.lanczos{TOp}"/>'s own "reorthogonalize twice" folklore) against the full
    /// locked+active X; P (once formed) is deflated against X and then against the
    /// already-deflated W. Second, W (then P) is Cholesky-QR-orthonormalized INTERNALLY (see
    /// <see cref="OrthonormalizeBlock"/>) -- an EARLIER version of this method skipped this second
    /// stage, reasoning that the Rayleigh-Ritz step's own Cholesky-based generalized-to-standard
    /// reduction (below) would absorb it "for free"; that was wrong in practice: deflation alone
    /// leaves W's/P's OWN internal scale arbitrary (active pairs' residuals routinely shrink by very
    /// different amounts), and feeding that directly into the combined Gram relies on ONE Cholesky
    /// factorization to absorb both the deflation and that scale spread -- which is exactly the
    /// ill-conditioned-basis failure mode that manufactures spurious Ritz values (observed: values
    /// BELOW lambda_min, even negative, for an SPD test operator) instead of tripping the
    /// rank-deficiency safeguard. Internally orthonormalizing first keeps the combined Gram close to
    /// the identity, so the final Rayleigh-Ritz Cholesky is well-conditioned by construction and a
    /// genuine rank deficiency reliably trips safeguard 2 instead. Every orthogonalization step
    /// mirrors its combination onto A*V using the SAME coefficients (linearity: A(sum_r c_r v_r) =
    /// sum_r c_r (A v_r)), so AW never needs a fresh matvec -- but see the "AX/AP freshness" note
    /// below for why X's and P's own A-images do NOT rely purely on this mirroring.
    ///
    /// <b>AX/AP freshness (rescue-task fix):</b> an earlier version of this method maintained BOTH
    /// AX and AP purely via the linearity-mirroring above, across EVERY iteration, with no
    /// independent recomputation -- i.e. A was never re-applied to X or P once the initial AX seed
    /// was formed. This is a correctness bug, not just an accuracy nicety: AX/AP are each reformed
    /// from a NEW linear combination every iteration (chained onto the PREVIOUS iteration's AX/AP),
    /// so any rounding error introduced by one iteration's Cholesky-QR/Rayleigh-Ritz combination
    /// compounds into the next. For AX this manifested as slow drift -- the residual would shrink
    /// geometrically for the first ~15-20 iterations, then stall and creep back up instead of
    /// continuing to converge (X's OWN combination coefficients are usually well-behaved, so the
    /// drift is slow). For AP the effect was far more direct and severe: AP feeds straight into
    /// next iteration's H (the P-columns are dot(*, AP)), so a single iteration's inaccurate AP
    /// corrupts the NEXT Rayleigh-Ritz problem's energy matrix directly -- this, combined with
    /// safeguard 2's diag-ratio check being a poor proxy for THIS specific failure (see safeguard 3
    /// below), is what actually produced Ritz values far below lambda_min, even wildly negative, as
    /// soon as P entered the mix. The fix: after <see cref="UpdateActiveBlock"/> forms the new
    /// X/P for this iteration, AX and AP are each recomputed via a FRESH <c>A.Apply</c> (the
    /// canonical "R = A X - X diag(theta)" formulation), rather than trusted from the combination.
    /// This costs two extra matvec batches per iteration (over numActive rows) on top of AW's one --
    /// a small, worthwhile price for AX/AP that stay exact to working precision indefinitely.
    ///
    /// <b>Rayleigh-Ritz / rank deficiency (safeguard 2):</b> forms Gram = S^T S and H = S^T A S for
    /// S = [X_active; W_active(deflated); P_active(deflated)] (m = 3*numActive, or 2*numActive
    /// before P exists), Cholesky-factors Gram (with one Tikhonov-ridge retry on failure/a tiny
    /// relative pivot -- mirrors <see cref="CHOP.decompSolve(ref doubleMxN, in Pivot, int, ref doubleN, ref doubleCHOPCache)"/>'s
    /// own ridge-retry recovery, and the retry attempt is re-checked against the SAME pivot tolerance
    /// rather than accepted on bare Cholesky success), reduces to the standard eigenproblem Ahat = L^-1 H L^-T,
    /// solves it with <see cref="Eigen.symmetric(ref doubleMxN, ref doubleN, ref doubleMxN)"/>,
    /// and recovers the combination coefficients C = L^-T Y. If the 3-block Cholesky fails/is too
    /// ill-conditioned this iteration DROPS P and retries with just [X_active; W_active] -- "the
    /// standard fix" the spec calls for. If even THAT fails (after its own ridge retry), this
    /// iteration is a no-op STALL: X/lambda are left unchanged, P's history is discarded, and the
    /// loop tries again next iteration -- never NaN, never diverges, just makes no progress that
    /// one iteration.
    ///
    /// <b>Ritz-value plausibility (safeguard 3, rescue-task addition):</b> safeguard 2's Cholesky
    /// diag-ratio check turned out to be a poor proxy for the ACTUAL failure this class was built
    /// to prevent -- diagnostic runs found diag ratios of 5.7E-08, 8.1E-04 and 1.25E-03 (all safely
    /// clear of the sqrt(eps) pivot threshold, i.e. "comfortably well-conditioned" by that check)
    /// that nonetheless produced Ritz values of -328042.9 and worse. Tightening the pivot threshold
    /// was tried and rejected: it also rejects benign, already-convergent 2-block [X,W] cases with
    /// SIMILAR diag ratios that were converging perfectly correctly, so the threshold is not
    /// discriminating the failure at all -- something else was wrong (the AX/AP freshness bug
    /// above), and no Cholesky-conditioning threshold can distinguish "well-conditioned reduction of
    /// a subtly-corrupted problem" from "well-conditioned reduction of a correct one". Instead, this
    /// safeguard checks the RESULT directly against a cheap, numerically TRUSTWORTHY bound computed
    /// before the Cholesky reduction: H[i,i]/Gram[i,i] is the exact Rayleigh quotient of one
    /// individual, already-unit-normalized basis row (a row of X, W, or P) -- a plain dot-product
    /// ratio, no matrix inversion involved, so it is immune to whatever ill-conditioning may corrupt
    /// the L^-1 H L^-T transform. Every individual Rayleigh quotient is, by definition, within
    /// [lambda_min(A), lambda_max(A)] of the FULL operator; a selected Ritz value falling wildly
    /// outside a generous (1000x) envelope around the range spanned by these quotients is rejected
    /// (triggering the same drop-P/stall fallback as safeguard 2), rather than locked in as
    /// numerical garbage.
    ///
    /// <b>Guard against tiny/non-finite residuals (safeguard 4):</b> a non-finite (NaN/Inf) residual
    /// norm aborts the whole solve immediately with <see cref="IterativeSolveStatus.Breakdown"/>
    /// rather than feeding garbage into the preconditioner/orthogonalization; an exactly-zero (or
    /// tiny) residual is simply locked immediately (it already satisfies the relative-tolerance
    /// check for any tol &gt; 0), which is this implementation's reading of "guard before
    /// normalizing" -- there is no separate per-vector unit-normalization step to guard (Cholesky-QR
    /// style orthogonalization here operates on the whole active block via its Gram matrix, which
    /// is scale-covariant), so the guard is folded into the convergence/lock check instead of a
    /// standalone division.
    ///
    /// <b>Zero-alloc scope:</b> every O(n)-scale buffer (X/W/P/AX/AW/AP/R and their "next"
    /// ping-pong twins, plus the row scratch) lives in <see cref="doubleLOBPCGCache"/>, allocated
    /// once via <c>Arena.doubleLOBPCGCache(n, k)</c> and reused across calls -- zero allocation at
    /// the O(n) scale. The O(k)-scale Rayleigh-Ritz sub-problem's DENSE matrices (Gram/H/L/Atrans/
    /// Y/C, each up to 3k x 3k) are ALSO cache fields, reused every iteration via a same-buffer,
    /// smaller-shaped logical view (<see cref="View"/> -- <see cref="doubleMxN.M_Rows"/>/
    /// <see cref="doubleMxN.N_Cols"/> are plain mutable fields independent of the backing store, so
    /// a value-copy with different dims is a free reinterpretation of the SAME buffer, not a new
    /// allocation). The one exception: <see cref="Eigen.symmetric(ref doubleMxN, ref doubleN, ref doubleMxN)"/>
    /// itself is not zero-alloc (it allocates three length-m Temp vectors internally, already true
    /// of every existing caller e.g. <see cref="Eigen.lanczosVectors{TOp}"/>), and this method's own
    /// small O(m) row/column scratch inside the triangular-solve helpers is likewise a bounded
    /// <c>Allocator.Temp</c> vector -- consistent with, not a regression from, that established
    /// precedent.
    ///
    /// <b>Convergence:</b> per-pair 2-norm relative residual ||A x_i - lambda_i B x_i|| /
    /// max(|lambda_i|, 1) &lt;= tol (B=I reduces this to the Euclidean ||A x_i - lambda_i x_i||
    /// used throughout the discussion above -- see "Generalized eigenproblem" below for B != I).
    /// Returns a <see cref="LOBPCGInfo"/> (reuses <see cref="IterativeSolveStatus"/> --
    /// Converged/MaxIterations/Breakdown, no new enum).
    ///
    /// <b>Output order:</b> eigenvalues/eigenvectors are returned ASCENDING (index 0 = smallest) --
    /// unlike every OTHER Eigen method in this library (which sorts descending), because LOBPCG's
    /// entire purpose is "the k smallest", so smallest-first is the natural presentation here.
    ///
    /// <b>Generalized eigenproblem (A x = lambda B x, B SPD):</b> every overload above the
    /// standard (single-operator) form threads a second operator B through the SAME algorithm --
    /// B-inner products (u^T B v) replace Euclidean ones (u^T v) everywhere a Gram matrix is
    /// formed: the basis blocks X/W/P are B-ORTHONORMALIZED (Gram = S^T B S instead of S^T S) and
    /// Rayleigh-Ritz solves the pencil (S^T A S, S^T B S) -- the SAME Cholesky-based
    /// generalized-to-standard reduction already used for the standard case's Rayleigh-Ritz step
    /// (<see cref="TryRayleighRitz"/>) IS the textbook generalized-eigenproblem reduction once its
    /// Gram argument is S^T B S rather than S^T S; nothing about that reduction itself changed.
    /// Residuals are r_i = A x_i - theta_i B x_i (theta_i the current Ritz value estimate);
    /// convergence remains the EUCLIDEAN 2-norm of this residual relative to max(|theta_i|, 1) --
    /// standard practice (the residual measures how far x_i is from satisfying the pencil, it is
    /// not itself a B-weighted distance). Requires BX/BW/BP (the B-images of X/W/P, mirroring
    /// AX/AW/AP) -- see <see cref="doubleLOBPCGCache"/>'s BX/BW/BP fields. The safeguard-3
    /// plausibility envelope generalizes verbatim: each individual quotient becomes
    /// (s^T A s)/(s^T B s) = H[i,i]/Gram[i,i], where Gram is now the B-Gram -- the SAME immunity
    /// argument applies (a plain ratio of two already-computed dot products, no matrix inversion,
    /// so it stays trustworthy regardless of the Cholesky-based reduction's own conditioning).
    /// Convergence theory: LOBPCG requires only B SPD -- A itself may be INDEFINITE (this is
    /// exactly the buckling case below) and the method is still guaranteed to converge to the
    /// algebraically smallest eigenvalues of the pencil; it does NOT require A positive (semi)definite.
    ///
    /// <b>B=I strategy (bit-identical standard path):</b> rather than hand-duplicating the whole
    /// algorithm for the plain (Euclidean, B=I) case, every standard-path overload
    /// (<see cref="lobpcg{TOp,TPre}"/> and everything built on it) forwards into the SAME
    /// generalized core with B played by <see cref="doubleIdentityOperator"/> (Apply is an exact
    /// bit-copy, z = r). Every place the generalized algorithm reads a "B-image" in place of a raw
    /// Euclidean block (BX for X, BW for W, BP for P) is a direct SUBSTITUTION -- no new arithmetic
    /// operation (no extra division/normalization) was introduced alongside it -- so for B=I, every
    /// substituted quantity holds BITS IDENTICAL to the block it substitutes, and every downstream
    /// formula that reads it reproduces the pre-generalization Euclidean formula bit-for-bit (each
    /// substitution site is documented inline with this reasoning). The ONE intentional exception:
    /// the bootstrap Rayleigh-quotient seed for lambda (dot(X,AX), computed once before iteration 0)
    /// deliberately stays the plain Euclidean quotient rather than dividing by dot(X,BX) -- the
    /// "more correct" generalized quotient -- specifically because that division would cost
    /// bit-identical-ness for B=I (dot(X,BX) is only extremely close to, not exactly, 1.0 in
    /// floating point after Cholesky-QR) for zero practical benefit: this seed is immediately
    /// superseded by the first real Rayleigh-Ritz iteration regardless of its accuracy (unlike e.g.
    /// Newton's method, LOBPCG's subspace correction does not depend sensitively on a bootstrap
    /// value's precision). This is the sole documented deviation from "every substitution is a
    /// bit-identical no-op for B=I" in this file. Verification note: since this replaces the
    /// prior standard-only implementation rather than living alongside it, "bit-identical" is
    /// supported by the by-construction argument above plus the unmodified 27-test standard-path
    /// regression suite continuing to pass -- not by a mechanical byte-diff against a captured
    /// pre-change baseline (the old code path no longer exists to diff against).
    ///
    /// Cost trade-off: the unified implementation pays a real (if small) constant-factor overhead
    /// on the standard path even when B is the identity -- three extra "B.Apply" batches per
    /// iteration (BX/BW/BP), each just a buffer copy for the identity case, plus the extra Deflate/
    /// OrthonormalizeBlockB bookkeeping over BW/BP. This was accepted deliberately in exchange for
    /// zero code duplication and a strong bit-identical guarantee, matching the spec's explicit
    /// preference; a hand-specialized Euclidean-only fast path was considered and rejected as the
    /// higher-risk, higher-maintenance option (two divergent copies of a very safeguard-heavy loop).
    ///
    /// <b>Fresh-matvec principle extends to B:</b> exactly like AX/AP (see the "AX/AP freshness"
    /// note above), BX and BP are recomputed via a FRESH <c>B.Apply</c> at the same points AX/AP
    /// are -- NEVER maintained by mirroring a linear combination across an iteration boundary. BW,
    /// like AW, gets exactly ONE fresh <c>B.Apply</c> per iteration (right when W is formed from
    /// the preconditioned residual) and is then carried through THAT SAME iteration's Deflate/
    /// OrthonormalizeBlockB calls via linearity (B is linear, so this mirroring is exact up to
    /// ordinary floating-point rounding of a SINGLE transform -- it does not compound across
    /// iterations the way the rejected AX/AP-mirroring design did, because it is never carried past
    /// one iteration's boundary before the next fresh <c>B.Apply</c> supersedes it). The one-time
    /// initial X seed is a partial exception: its own Euclidean orthonormalization is UNCHANGED
    /// (deliberately not B-aware -- see <see cref="OrthonormalizeBlock"/>'s call site comment for
    /// why), so BX's very first value comes from a single fresh <c>B.Apply</c> issued right after
    /// that seed step, mirroring AX's own initial treatment exactly.
    ///
    /// <b>Buckling mapping (K_E phi = -lambda K_G phi convention):</b> the standard linear-buckling
    /// eigenproblem is K_E*phi + lambda*K_G*phi = 0, i.e. K_E*phi = -lambda*K_G*phi, where K_E is
    /// the (SPD) elastic stiffness matrix and K_G is the geometric/stress stiffness matrix evaluated
    /// at some REFERENCE load level (typically INDEFINITE: members in compression contribute
    /// negative-definite-like terms, tension members positive) -- the same convention used by e.g.
    /// Nastran SOL 105 / Abaqus *BUCKLE. lambda is the LOAD MULTIPLIER: the critical buckling load
    /// is lambda_cr times the reference load. Rearranging: K_G*phi = mu*K_E*phi where
    /// mu = -1/lambda_cr -- a pencil with K_G (indefinite) in the A slot and K_E (SPD) in the B
    /// slot, exactly this method's required shape (only B needs to be SPD; A indefinite is fine --
    /// see "Convergence theory" above). The SMALLEST (most negative) mu returned corresponds to the
    /// SMALLEST positive lambda_cr -- i.e. the FIRST critical load, exactly the quantity a buckling
    /// analysis wants, and exactly what LOBPCG natively targets with NO extra mode-selection needed:
    /// <code>
    ///   // K_E: SPD elastic stiffness. K_G: geometric stiffness at the reference load (indefinite).
    ///   var mu = Eigen.lobpcg(in K_G, in K_E, ref ws, k, tol, maxIter); // A=K_G, B=K_E
    ///   // mu is ASCENDING; mu[0] is the most negative (first/critical) mode, PROVIDED it is
    ///   // actually negative -- a mu[i] &gt;= 0 is not a buckling mode under this reference load
    ///   // direction (no positive critical multiplier exists for that mode) and should be
    ///   // discarded/flagged, not divided.
    ///   for (int i = 0; i &lt; k; i++)
    ///       if (mu[i] &lt; 0) lambdaCritical[i] = -1 / mu[i]; // buckling load multiplier
    /// </code>
    /// Verified on a small analytic example (see the LOBPCG generalized smoke tests): a
    /// diagonal-congruent K_G/K_E pair with a KNOWN mixed-sign spectrum reproduces the expected
    /// mu's, and the recovered lambda_cr matches a direct K_E*phi = -lambda*K_G*phi solve for the
    /// SAME analytic system. If a different sign convention is used upstream (some texts define
    /// K_G with the OPPOSITE sign, i.e. K_E*phi = +lambda*K_G*phi), flip the sign in the final
    /// division (lambda_cr = +1/mu) accordingly -- the pencil construction and "smallest mu"
    /// targeting are unaffected; only the final scalar's sign flips with the convention.
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
        public static LOBPCGInfo lobpcg<TOp, TBOp, TPre>(in TOp A, in TBOp B, in TPre M, ref doubleLOBPCGCache ws,
                                                          int k, double tol, int maxIter)
            where TOp : struct, IdoubleLinearOperator
            where TBOp : struct, IdoubleLinearOperator
            where TPre : struct, IdoublePreconditioner
        {
            if (A.Rows != A.Cols)
                throw new ArgumentException("LOBPCG: A must be square");

            int n = A.Rows;

            if (B.Rows != n || B.Cols != n)
                throw new ArgumentException("LOBPCG: B must be n x n, matching A (B must be SPD -- not verified at runtime, the same unchecked contract as A's symmetry)");

            if (k < 1 || k > n)
                throw new ArgumentException("LOBPCG: k must be in [1, A.Rows]");

            if (maxIter < 1)
                throw new ArgumentException("LOBPCG: maxIter must be >= 1");

            if (tol <= (double)0)
                throw new ArgumentException("LOBPCG: tol must be > 0");

            RequireLOBPCGWorkspace(in ws, n, k);
            RequireDistinctBuffers(in ws);

            // ---- seed X if all-zero, then orthonormalize unconditionally ----
            bool allZero = true;
            for (int i = 0; i < k && allZero; i++)
                for (int c = 0; c < n; c++)
                    if (ws.X[i, c] != (double)0) { allZero = false; break; }

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
                for (int i = 0; i < k; i++)
                    for (int c = 0; c < n; c++)
                        ws.X[i, c] = (double)(seedRng.NextFloat() * 2f - 1f);
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
            if (!OrthonormalizeBlock(ref ws.X, ref ws.AX, k, n, ref ws.Gram, ref ws.L, ref ws.rowIn, ref ws.rowOut))
                return new LOBPCGInfo { iterations = 0, converged = 0, maxResidual = double.NaN, status = IterativeSolveStatus.Breakdown };

            for (int i = 0; i < k; i++)
            {
                for (int c = 0; c < n; c++) ws.rowIn[c] = ws.X[i, c];
                A.Apply(in ws.rowIn, ref ws.rowOut);
                for (int c = 0; c < n; c++) ws.AX[i, c] = ws.rowOut[c];
            }

            // BX = B.Apply(X), fresh, right after AX -- mirrors AX's own freshness exactly (see the
            // class doc's "fresh-matvec principle extends to B"). For the B=I forwarding path
            // (TBOp == doubleIdentityOperator) this is an exact bit-copy of X, so every later
            // computation reading BX in place of a Euclidean X reproduces the pre-generalization
            // formula bit-for-bit.
            for (int i = 0; i < k; i++)
            {
                for (int c = 0; c < n; c++) ws.rowIn[c] = ws.X[i, c];
                B.Apply(in ws.rowIn, ref ws.rowOut);
                for (int c = 0; c < n; c++) ws.BX[i, c] = ws.rowOut[c];
            }

            // Bootstrap Rayleigh-quotient seed for lambda -- deliberately the plain EUCLIDEAN
            // quotient dot(X,AX), NOT divided by dot(X,BX) (the "more correct" generalized
            // quotient): see the class doc's "B=I strategy" note for why (this seed is immediately
            // superseded by the first real Rayleigh-Ritz iteration regardless of its accuracy, so
            // the division would only cost bit-identical-ness for B=I with no practical benefit).
            for (int i = 0; i < k; i++)
            {
                double d = (double)0;
                for (int c = 0; c < n; c++) d += ws.X[i, c] * ws.AX[i, c];
                ws.lambda[i] = d;
            }

            int numActive = k;
            bool haveP = false;

            for (int iter = 0; iter < maxIter; iter++)
            {
                // ---- residual + lock newly converged pairs (scan back-to-front so a swap-in
                //      from the back is still checked) ----
                for (int i = numActive - 1; i >= 0; i--)
                {
                    double rn2 = (double)0;
                    for (int c = 0; c < n; c++)
                    {
                        // Generalized residual r_i = A x_i - lambda_i B x_i (standard practice --
                        // see the class doc). For B=I, ws.BX[i,c] is bit-identical to ws.X[i,c], so
                        // this reproduces the original "A x - lambda x" formula bit-for-bit.
                        double rv = ws.AX[i, c] - ws.lambda[i] * ws.BX[i, c];
                        ws.R[i, c] = rv;
                        rn2 += rv * rv;
                    }
                    double rnorm = math.sqrt(rn2);

                    if (!math.isfinite(rnorm))
                    {
                        SortAscending(ref ws, k);
                        return new LOBPCGInfo { iterations = iter, converged = k - numActive, maxResidual = double.NaN, status = IterativeSolveStatus.Breakdown };
                    }

                    ws.residual[i] = rnorm;

                    double scale = math.abs(ws.lambda[i]);
                    if (scale < (double)1) scale = (double)1;

                    if (rnorm <= tol * scale)
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

                if (numActive == 0)
                {
                    // `iter` (not iter+1): iterations 0..iter-1 each did a full W/Rayleigh-Ritz
                    // work pass; THIS iteration only ran the residual check before finding
                    // everyone already converged, so it contributes no additional work -- matches
                    // SolveInfo's "0 when converged before the first step" convention.
                    SortAscending(ref ws, k);
                    return new LOBPCGInfo { iterations = iter, converged = k, maxResidual = MaxRelResidual(in ws, k), status = IterativeSolveStatus.Converged };
                }

                // ---- W = M^-1 R (active), AW = A W, BW = B W (active) -- ONE matvec batch each/iteration ----
                for (int i = 0; i < numActive; i++)
                {
                    for (int c = 0; c < n; c++) ws.rowIn[c] = ws.R[i, c];
                    M.Apply(in ws.rowIn, ref ws.rowOut);
                    for (int c = 0; c < n; c++) ws.W[i, c] = ws.rowOut[c];
                }
                for (int i = 0; i < numActive; i++)
                {
                    for (int c = 0; c < n; c++) ws.rowIn[c] = ws.W[i, c];
                    A.Apply(in ws.rowIn, ref ws.rowOut);
                    for (int c = 0; c < n; c++) ws.AW[i, c] = ws.rowOut[c];
                }
                // BW = B W, fresh -- mirrors AW's own single fresh compute this iteration (see the
                // class doc's "fresh-matvec principle extends to B"); maintained via linearity
                // through the Deflate/OrthonormalizeBlockB calls below within THIS iteration only
                // (bounded, not chained across iterations -- exactly how AW already works).
                for (int i = 0; i < numActive; i++)
                {
                    for (int c = 0; c < n; c++) ws.rowIn[c] = ws.W[i, c];
                    B.Apply(in ws.rowIn, ref ws.rowOut);
                    for (int c = 0; c < n; c++) ws.BW[i, c] = ws.rowOut[c];
                }

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
                Deflate(ref ws.W, ref ws.AW, ref ws.BW, numActive, in ws.X, in ws.AX, in ws.BX, k, n);

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
                    Deflate(ref ws.P, ref ws.AP, ref ws.BP, numActive, in ws.X, in ws.AX, in ws.BX, k, n);
                    Deflate(ref ws.P, ref ws.AP, ref ws.BP, numActive, in ws.W, in ws.AW, in ws.BW, numActive, n);

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
                    // Pathological stall: leave X/lambda untouched this iteration and retry next
                    // time; discard P's history since it (or even just [X,W]) was implicated.
                    haveP = false;
                    continue;
                }

                UpdateActiveBlock(ref ws, numActive, usedP, n, k);

                // Recompute AX/BX FRESH via a matvec each -- UpdateActiveBlock deliberately does
                // NOT also mirror-combine AX/BX (see its own doc comment): propagating AX through
                // many iterations of Cholesky-QR/Rayleigh-Ritz combinations (never re-touching A)
                // accumulates rounding error that compounds -- exactly the observed failure mode
                // (residual shrinks nicely for ~15-20 iterations, then stalls and creeps back up
                // instead of continuing to converge). This is the canonical "R = A X - X diag(theta)"
                // fresh-residual formulation (generalized: "- B X diag(theta)"); the extra
                // matvecs/iteration (over numActive rows only) are a small, worthwhile price for a
                // residual that stays exact to working precision indefinitely.
                for (int i = 0; i < numActive; i++)
                {
                    for (int c = 0; c < n; c++) ws.rowIn[c] = ws.X[i, c];
                    A.Apply(in ws.rowIn, ref ws.rowOut);
                    for (int c = 0; c < n; c++) ws.AX[i, c] = ws.rowOut[c];
                }
                for (int i = 0; i < numActive; i++)
                {
                    for (int c = 0; c < n; c++) ws.rowIn[c] = ws.X[i, c];
                    B.Apply(in ws.rowIn, ref ws.rowOut);
                    for (int c = 0; c < n; c++) ws.BX[i, c] = ws.rowOut[c];
                }

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
                for (int i = 0; i < numActive; i++)
                {
                    for (int c = 0; c < n; c++) ws.rowIn[c] = ws.P[i, c];
                    A.Apply(in ws.rowIn, ref ws.rowOut);
                    for (int c = 0; c < n; c++) ws.AP[i, c] = ws.rowOut[c];
                }
                for (int i = 0; i < numActive; i++)
                {
                    for (int c = 0; c < n; c++) ws.rowIn[c] = ws.P[i, c];
                    B.Apply(in ws.rowIn, ref ws.rowOut);
                    for (int c = 0; c < n; c++) ws.BP[i, c] = ws.rowOut[c];
                }

                haveP = true;
            }

            SortAscending(ref ws, k);
            return new LOBPCGInfo { iterations = maxIter, converged = k - numActive, maxResidual = MaxRelResidual(in ws, k), status = IterativeSolveStatus.MaxIterations };
        }

        /// <summary>lobpcg (generalized, preconditioned) with default maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg<TOp, TBOp, TPre>(in TOp A, in TBOp B, in TPre M, ref doubleLOBPCGCache ws, int k, double tol)
            where TOp : struct, IdoubleLinearOperator
            where TBOp : struct, IdoubleLinearOperator
            where TPre : struct, IdoublePreconditioner
            => lobpcg(in A, in B, in M, ref ws, k, tol, 1000);

        /// <summary>lobpcg (generalized, preconditioned) with default tol (Consts.doubleSqrtEps) and maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg<TOp, TBOp, TPre>(in TOp A, in TBOp B, in TPre M, ref doubleLOBPCGCache ws, int k)
            where TOp : struct, IdoubleLinearOperator
            where TBOp : struct, IdoubleLinearOperator
            where TPre : struct, IdoublePreconditioner
            => lobpcg(in A, in B, in M, ref ws, k, Consts.doubleSqrtEps, 1000);

        /// <summary>
        /// lobpcg (STANDARD, B=I) primitive, preconditioned. Forwards into the generalized
        /// <see cref="lobpcg{TOp,TBOp,TPre}"/> with <see cref="doubleIdentityOperator"/> in the B
        /// slot -- see the class doc comment's "B=I strategy" note for why this reproduces the
        /// pre-generalization algorithm bit-for-bit (identity's Apply is an exact bit-copy) rather
        /// than requiring a hand-duplicated standard-only implementation.
        /// </summary>
        public static LOBPCGInfo lobpcg<TOp, TPre>(in TOp A, in TPre M, ref doubleLOBPCGCache ws, int k, double tol, int maxIter)
            where TOp : struct, IdoubleLinearOperator
            where TPre : struct, IdoublePreconditioner
            => lobpcg(in A, new doubleIdentityOperator(A.Rows), in M, ref ws, k, tol, maxIter);

        /// <summary>lobpcg (preconditioned) with default maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg<TOp, TPre>(in TOp A, in TPre M, ref doubleLOBPCGCache ws, int k, double tol)
            where TOp : struct, IdoubleLinearOperator
            where TPre : struct, IdoublePreconditioner
            => lobpcg(in A, in M, ref ws, k, tol, 1000);

        /// <summary>lobpcg (preconditioned) with default tol (Consts.doubleSqrtEps) and maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg<TOp, TPre>(in TOp A, in TPre M, ref doubleLOBPCGCache ws, int k)
            where TOp : struct, IdoubleLinearOperator
            where TPre : struct, IdoublePreconditioner
            => lobpcg(in A, in M, ref ws, k, Consts.doubleSqrtEps, 1000);

        /// <summary>
        /// Zero-alloc (at O(n) scale) LOBPCG primitive, UNPRECONDITIONED. Forwards into
        /// <see cref="lobpcg{TOp,TPre}"/> via <see cref="doubleIdentityPreconditioner"/> -- a
        /// one-line forwarder rather than a hand-duplicated loop (unlike <see cref="Krylov.cg{TOp}"/>
        /// / <see cref="Krylov.pcg{TOp,TPre}"/>'s literal duplication -- LOBPCG's loop is
        /// considerably larger, so this method mirrors the SAME "single source of truth, thin
        /// forwarder" pattern already used everywhere else in this file for dense/BSR wrapping,
        /// just applied one level further).
        /// </summary>
        public static LOBPCGInfo lobpcg<TOp>(in TOp A, ref doubleLOBPCGCache ws, int k, double tol, int maxIter)
            where TOp : struct, IdoubleLinearOperator
            => lobpcg(in A, new doubleIdentityPreconditioner(), ref ws, k, tol, maxIter);

        /// <summary>lobpcg (unpreconditioned) with default maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg<TOp>(in TOp A, ref doubleLOBPCGCache ws, int k, double tol)
            where TOp : struct, IdoubleLinearOperator
            => lobpcg(in A, ref ws, k, tol, 1000);

        /// <summary>lobpcg (unpreconditioned) with default tol (Consts.doubleSqrtEps) and maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg<TOp>(in TOp A, ref doubleLOBPCGCache ws, int k)
            where TOp : struct, IdoubleLinearOperator
            => lobpcg(in A, ref ws, k, Consts.doubleSqrtEps, 1000);

        // NOTE: there is deliberately NO standalone `lobpcg<TOp,TBOp>(in TOp A, in TBOp B, ref ws,
        // int k, double tol, int maxIter)` unpreconditioned-generic-generalized convenience overload
        // mirroring `lobpcg<TOp>` above: with 2 open type parameters and the same positional value-
        // argument shape (T, T, ref cache, int, double, int), it would be IDENTICAL, at the C#
        // signature level, to the existing `lobpcg<TOp,TPre>(in TOp A, in TPre M, ref ws, int k,
        // double tol, int maxIter)` above -- generic constraints (IdoubleLinearOperator vs
        // IdoublePreconditioner) do NOT participate in overload/signature matching, so declaring
        // both would be a compile error (CS0111, "already defines a member with the same parameter
        // types"), not merely an ambiguous call. Callers who have their own custom TOp/TBOp operator
        // structs and want an unpreconditioned generalized solve call the 3-type-param core directly
        // with an explicit identity preconditioner: <c>lobpcg(in A, in B, new
        // doubleIdentityPreconditioner(), ref ws, k, tol, maxIter)</c>. The CONCRETE (dense/BSR)
        // unpreconditioned-generalized overloads below are unaffected (they forward straight into
        // the 3-type-param core with an inline identity preconditioner) since concrete parameter
        // types never collide the way two same-shaped open generic signatures do.

        /// <summary>
        /// LOBPCG over a dense <see cref="doubleMxN"/> -- zero-alloc primitive, unpreconditioned.
        /// Forwards into <see cref="lobpcg{TOp}"/> via <see cref="doubleDenseOperator"/>.
        /// </summary>
        public static LOBPCGInfo lobpcg(in doubleMxN A, ref doubleLOBPCGCache ws, int k, double tol, int maxIter)
            => lobpcg(new doubleDenseOperator(in A), ref ws, k, tol, maxIter);

        /// <summary>lobpcg over a dense matrix with default maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in doubleMxN A, ref doubleLOBPCGCache ws, int k, double tol)
            => lobpcg(in A, ref ws, k, tol, 1000);

        /// <summary>lobpcg over a dense matrix with default tol (Consts.doubleSqrtEps) and maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in doubleMxN A, ref doubleLOBPCGCache ws, int k)
            => lobpcg(in A, ref ws, k, Consts.doubleSqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a dense pencil (A, B) -- GENERALIZED eigenproblem A x = lambda B x, zero-alloc
        /// primitive, unpreconditioned. Forwards DIRECTLY into the 3-type-param
        /// <see cref="lobpcg{TOp,TBOp,TPre}"/> core with an inline <see cref="doubleIdentityPreconditioner"/>
        /// (see the NOTE above this dense group's generic sibling would have occupied -- a standalone
        /// generic unpreconditioned-generalized rung is not declarable, but this CONCRETE overload is
        /// unaffected). See the class doc comment's "Buckling mapping" note for the canonical
        /// A=K_G/B=K_E truss-buckling usage.
        /// </summary>
        public static LOBPCGInfo lobpcg(in doubleMxN A, in doubleMxN B, ref doubleLOBPCGCache ws, int k, double tol, int maxIter)
            => lobpcg(new doubleDenseOperator(in A), new doubleDenseOperator(in B), new doubleIdentityPreconditioner(), ref ws, k, tol, maxIter);

        /// <summary>lobpcg (generalized) over a dense pencil with default maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in doubleMxN A, in doubleMxN B, ref doubleLOBPCGCache ws, int k, double tol)
            => lobpcg(in A, in B, ref ws, k, tol, 1000);

        /// <summary>lobpcg (generalized) over a dense pencil with default tol (Consts.doubleSqrtEps) and maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in doubleMxN A, in doubleMxN B, ref doubleLOBPCGCache ws, int k)
            => lobpcg(in A, in B, ref ws, k, Consts.doubleSqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a dense SYMMETRIC matrix -- allocates the workspace from <paramref
        /// name="arena"/> and calls the zero-alloc primitive. Returns the eigenvalues (length k,
        /// ASCENDING); <paramref name="eigenvectors"/> is k x n (row i = eigenvector i). Both
        /// buffers are the workspace's OWN X/lambda (no extra copy) -- the rest of the workspace
        /// stays allocated in the arena, unused after this call, exactly like every other
        /// arena-convenience wrapper in this library (e.g. <see cref="Eigen.lanczosVectors{TOp}"/>).
        /// </summary>
        public static doubleN lobpcg(ref Arena arena, in doubleMxN A, int k, out doubleMxN eigenvectors,
                                      out LOBPCGInfo info, double tol, int maxIter)
        {
            var ws = arena.doubleLOBPCGCache(A.M_Rows, k);
            info = lobpcg(in A, ref ws, k, tol, maxIter);
            eigenvectors = ws.X;
            return ws.lambda;
        }

        /// <summary>lobpcg (allocating) over a dense matrix with default tol/maxIter.</summary>
        public static doubleN lobpcg(ref Arena arena, in doubleMxN A, int k, out doubleMxN eigenvectors, out LOBPCGInfo info)
            => lobpcg(ref arena, in A, k, out eigenvectors, out info, Consts.doubleSqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a dense pencil (A, B) -- GENERALIZED eigenproblem, allocating. See the
        /// standard dense overload's doc comment for the buffer-ownership contract.
        /// </summary>
        public static doubleN lobpcg(ref Arena arena, in doubleMxN A, in doubleMxN B, int k, out doubleMxN eigenvectors,
                                      out LOBPCGInfo info, double tol, int maxIter)
        {
            var ws = arena.doubleLOBPCGCache(A.M_Rows, k);
            info = lobpcg(in A, in B, ref ws, k, tol, maxIter);
            eigenvectors = ws.X;
            return ws.lambda;
        }

        /// <summary>lobpcg (allocating, generalized) over a dense pencil with default tol/maxIter.</summary>
        public static doubleN lobpcg(ref Arena arena, in doubleMxN A, in doubleMxN B, int k, out doubleMxN eigenvectors, out LOBPCGInfo info)
            => lobpcg(ref arena, in A, in B, k, out eigenvectors, out info, Consts.doubleSqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a block-sparse (BSR) matrix -- zero-alloc primitive, unpreconditioned.
        /// Forwards into <see cref="lobpcg{TOp}"/> via <c>doubleBSROperator</c>.
        /// </summary>
        public static LOBPCGInfo lobpcg(in doubleBSR A, ref doubleLOBPCGCache ws, int k, double tol, int maxIter)
            => lobpcg(new doubleBSROperator(in A), ref ws, k, tol, maxIter);

        /// <summary>lobpcg over a BSR matrix with default maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in doubleBSR A, ref doubleLOBPCGCache ws, int k, double tol)
            => lobpcg(in A, ref ws, k, tol, 1000);

        /// <summary>lobpcg over a BSR matrix with default tol (Consts.doubleSqrtEps) and maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in doubleBSR A, ref doubleLOBPCGCache ws, int k)
            => lobpcg(in A, ref ws, k, Consts.doubleSqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a block-sparse pencil (A, B) -- GENERALIZED eigenproblem, zero-alloc
        /// primitive, unpreconditioned. Forwards DIRECTLY into the 3-type-param
        /// <see cref="lobpcg{TOp,TBOp,TPre}"/> core (via <c>doubleBSROperator</c> for both A and B)
        /// with an inline <see cref="doubleIdentityPreconditioner"/> -- see the dense pencil
        /// overload's NOTE for why there is no generic unpreconditioned-generalized rung to route
        /// through here.
        /// </summary>
        public static LOBPCGInfo lobpcg(in doubleBSR A, in doubleBSR B, ref doubleLOBPCGCache ws, int k, double tol, int maxIter)
            => lobpcg(new doubleBSROperator(in A), new doubleBSROperator(in B), new doubleIdentityPreconditioner(), ref ws, k, tol, maxIter);

        /// <summary>lobpcg (generalized) over a BSR pencil with default maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in doubleBSR A, in doubleBSR B, ref doubleLOBPCGCache ws, int k, double tol)
            => lobpcg(in A, in B, ref ws, k, tol, 1000);

        /// <summary>lobpcg (generalized) over a BSR pencil with default tol (Consts.doubleSqrtEps) and maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in doubleBSR A, in doubleBSR B, ref doubleLOBPCGCache ws, int k)
            => lobpcg(in A, in B, ref ws, k, Consts.doubleSqrtEps, 1000);

        /// <summary>lobpcg (allocating) over a BSR matrix. See the dense overload's doc comment.</summary>
        public static doubleN lobpcg(ref Arena arena, in doubleBSR A, int k, out doubleMxN eigenvectors,
                                      out LOBPCGInfo info, double tol, int maxIter)
        {
            var ws = arena.doubleLOBPCGCache(A.M_Rows, k);
            info = lobpcg(in A, ref ws, k, tol, maxIter);
            eigenvectors = ws.X;
            return ws.lambda;
        }

        /// <summary>lobpcg (allocating) over a BSR matrix with default tol/maxIter.</summary>
        public static doubleN lobpcg(ref Arena arena, in doubleBSR A, int k, out doubleMxN eigenvectors, out LOBPCGInfo info)
            => lobpcg(ref arena, in A, k, out eigenvectors, out info, Consts.doubleSqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a block-sparse pencil (A, B) -- GENERALIZED eigenproblem, allocating. See
        /// the standard BSR overload's doc comment for the buffer-ownership contract.
        /// </summary>
        public static doubleN lobpcg(ref Arena arena, in doubleBSR A, in doubleBSR B, int k, out doubleMxN eigenvectors,
                                      out LOBPCGInfo info, double tol, int maxIter)
        {
            var ws = arena.doubleLOBPCGCache(A.M_Rows, k);
            info = lobpcg(in A, in B, ref ws, k, tol, maxIter);
            eigenvectors = ws.X;
            return ws.lambda;
        }

        /// <summary>lobpcg (allocating, generalized) over a BSR pencil with default tol/maxIter.</summary>
        public static doubleN lobpcg(ref Arena arena, in doubleBSR A, in doubleBSR B, int k, out doubleMxN eigenvectors, out LOBPCGInfo info)
            => lobpcg(ref arena, in A, in B, k, out eigenvectors, out info, Consts.doubleSqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a block-sparse (BSR) matrix with its matching block-Jacobi preconditioner --
        /// zero-alloc primitive. Forwards into <see cref="lobpcg{TOp,TPre}"/> via
        /// <c>doubleBSROperator</c>/<c>doubleBlockJacobi</c>. This is the preconditioned entry point
        /// the sparse-BSM eigensolver roadmap calls out (matvec + block-Jacobi, matching how
        /// <see cref="Krylov.pcg(in doubleBSR, in doubleBlockJacobi, in doubleN, ref doubleN)"/>
        /// consumes it).
        /// </summary>
        public static LOBPCGInfo lobpcg(in doubleBSR A, in doubleBlockJacobi M, ref doubleLOBPCGCache ws,
                                         int k, double tol, int maxIter)
            => lobpcg(new doubleBSROperator(in A), in M, ref ws, k, tol, maxIter);

        /// <summary>lobpcg (BSR + block-Jacobi) with default maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in doubleBSR A, in doubleBlockJacobi M, ref doubleLOBPCGCache ws, int k, double tol)
            => lobpcg(in A, in M, ref ws, k, tol, 1000);

        /// <summary>lobpcg (BSR + block-Jacobi) with default tol (Consts.doubleSqrtEps) and maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in doubleBSR A, in doubleBlockJacobi M, ref doubleLOBPCGCache ws, int k)
            => lobpcg(in A, in M, ref ws, k, Consts.doubleSqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a block-sparse pencil (A, B) -- GENERALIZED eigenproblem -- with A's matching
        /// block-Jacobi preconditioner. Forwards into <see cref="lobpcg{TOp,TBOp,TPre}"/> via
        /// <c>doubleBSROperator</c>/<c>doubleBlockJacobi</c>. Note the preconditioner M is built from
        /// (and approximates the inverse of) A only -- it operates on the RAW residual r = A x -
        /// lambda B x exactly like the standard-path block-Jacobi preconditioner does, B does not
        /// enter M's construction or Apply.
        /// </summary>
        public static LOBPCGInfo lobpcg(in doubleBSR A, in doubleBSR B, in doubleBlockJacobi M, ref doubleLOBPCGCache ws,
                                         int k, double tol, int maxIter)
            => lobpcg(new doubleBSROperator(in A), new doubleBSROperator(in B), in M, ref ws, k, tol, maxIter);

        /// <summary>lobpcg (generalized, BSR + block-Jacobi) with default maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in doubleBSR A, in doubleBSR B, in doubleBlockJacobi M, ref doubleLOBPCGCache ws, int k, double tol)
            => lobpcg(in A, in B, in M, ref ws, k, tol, 1000);

        /// <summary>lobpcg (generalized, BSR + block-Jacobi) with default tol (Consts.doubleSqrtEps) and maxIter (1000).</summary>
        public static LOBPCGInfo lobpcg(in doubleBSR A, in doubleBSR B, in doubleBlockJacobi M, ref doubleLOBPCGCache ws, int k)
            => lobpcg(in A, in B, in M, ref ws, k, Consts.doubleSqrtEps, 1000);

        /// <summary>lobpcg (allocating) over a BSR matrix with block-Jacobi. See the dense overload's doc comment.</summary>
        public static doubleN lobpcg(ref Arena arena, in doubleBSR A, in doubleBlockJacobi M, int k,
                                      out doubleMxN eigenvectors, out LOBPCGInfo info, double tol, int maxIter)
        {
            var ws = arena.doubleLOBPCGCache(A.M_Rows, k);
            info = lobpcg(in A, in M, ref ws, k, tol, maxIter);
            eigenvectors = ws.X;
            return ws.lambda;
        }

        /// <summary>lobpcg (allocating) over a BSR matrix with block-Jacobi and default tol/maxIter.</summary>
        public static doubleN lobpcg(ref Arena arena, in doubleBSR A, in doubleBlockJacobi M, int k,
                                      out doubleMxN eigenvectors, out LOBPCGInfo info)
            => lobpcg(ref arena, in A, in M, k, out eigenvectors, out info, Consts.doubleSqrtEps, 1000);

        /// <summary>
        /// LOBPCG over a block-sparse pencil (A, B) with block-Jacobi -- GENERALIZED eigenproblem,
        /// allocating. See the standard BSR+block-Jacobi overload's doc comment.
        /// </summary>
        public static doubleN lobpcg(ref Arena arena, in doubleBSR A, in doubleBSR B, in doubleBlockJacobi M, int k,
                                      out doubleMxN eigenvectors, out LOBPCGInfo info, double tol, int maxIter)
        {
            var ws = arena.doubleLOBPCGCache(A.M_Rows, k);
            info = lobpcg(in A, in B, in M, ref ws, k, tol, maxIter);
            eigenvectors = ws.X;
            return ws.lambda;
        }

        /// <summary>lobpcg (allocating, generalized) over a BSR pencil with block-Jacobi and default tol/maxIter.</summary>
        public static doubleN lobpcg(ref Arena arena, in doubleBSR A, in doubleBSR B, in doubleBlockJacobi M, int k,
                                      out doubleMxN eigenvectors, out LOBPCGInfo info)
            => lobpcg(ref arena, in A, in B, in M, k, out eigenvectors, out info, Consts.doubleSqrtEps, 1000);

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
        static unsafe void RequireDistinctBuffers(in doubleLOBPCGCache ws)
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
                        throw new ArgumentException("LOBPCG: workspace buffers must be distinct (use Arena.doubleLOBPCGCache(n, k))");
        }

        // Same-buffer, smaller-shaped logical view: doubleMxN.M_Rows/N_Cols are plain mutable
        // fields independent of the backing Data store, so a value-copy with adjusted dims is a
        // free reinterpretation of the SAME (larger, cache-owned) buffer's leading m x m block --
        // not a new allocation. See the class doc comment's "Zero-alloc scope" note.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static doubleMxN View(in doubleMxN buf, int m)
        {
            var v = buf;
            v.M_Rows = m;
            v.N_Cols = m;
            return v;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void SwapMat(ref doubleMxN a, ref doubleMxN b)
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
        static bool OrthonormalizeBlock(ref doubleMxN V, ref doubleMxN AV, int rows, int n,
                                         ref doubleMxN Gram, ref doubleMxN L, ref doubleN rowTmp, ref doubleN rowTmp2)
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
                    double coef = Lv[i, r];
                    for (int c = 0; c < n; c++)
                    {
                        rowTmp[c] -= coef * V[r, c];
                        rowTmp2[c] -= coef * AV[r, c];
                    }
                }

                double inv = (double)1 / Lv[i, i];
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
        static bool OrthonormalizeBlockB(ref doubleMxN V, ref doubleMxN AV, ref doubleMxN BV, int rows, int n,
                                          ref doubleMxN Gram, ref doubleMxN L, ref doubleN rowTmp, ref doubleN rowTmp2, ref doubleN rowTmp3)
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
                    double coef = Lv[i, r];
                    for (int c = 0; c < n; c++)
                    {
                        rowTmp[c] -= coef * V[r, c];
                        rowTmp2[c] -= coef * AV[r, c];
                        rowTmp3[c] -= coef * BV[r, c];
                    }
                }

                double inv = (double)1 / Lv[i, i];
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
        static void FillGramSub(ref doubleMxN Gram, int rowOff, in doubleMxN Vb, int rows,
                                 int colOff, in doubleMxN Wb, int cols, int n, bool sameBlock)
        {
            for (int i = 0; i < rows; i++)
            {
                int jStart = sameBlock ? i : 0;
                for (int j = jStart; j < cols; j++)
                {
                    double s = (double)0;
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
        static void FillHSub(ref doubleMxN H, int rowOff, in doubleMxN Vb, int rows,
                              int colOff, in doubleMxN AWb, int cols, int n, bool sameBlock)
        {
            for (int i = 0; i < rows; i++)
            {
                int jStart = sameBlock ? i : 0;
                for (int j = jStart; j < cols; j++)
                {
                    double s = (double)0;
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
        static void Deflate(ref doubleMxN V, ref doubleMxN AV, ref doubleMxN BV, int activeCount,
                             in doubleMxN Against, in doubleMxN AgainstA, in doubleMxN AgainstB, int againstCount, int n)
        {
            for (int pass = 0; pass < 2; pass++)
                for (int a = 0; a < activeCount; a++)
                    for (int i = 0; i < againstCount; i++)
                    {
                        double coeff = (double)0;
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
        static void BuildProjected(ref doubleMxN Gram, ref doubleMxN H, int numActive, bool useP,
                                    in doubleMxN X, in doubleMxN AX, in doubleMxN BX,
                                    in doubleMxN W, in doubleMxN AW, in doubleMxN BW,
                                    in doubleMxN P, in doubleMxN AP, in doubleMxN BP, int n)
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
        static bool FactorGram(ref doubleMxN Gram, ref doubleMxN L, int m)
        {
            // Relative-pivot tolerance for the "tiny diagonal" rank-deficiency check. A method-
            // local value (not a class-level const): Cholesky/QR/etc. in this codebase already
            // hoist their tuning constants into method scope for the same reason -- a class-level
            // const of the same name would collide across the float/double generated partials
            // (CS0102; see CHO.double.cs's CHOL_BLOCK comment).
            double pivotRelTol = Consts.doubleSqrtEps;

            var info = CHO.decomp(in Gram, ref L);
            if (info.Solved && MinMaxDiagRatio(in L, m) >= pivotRelTol)
                return true;

            double scale = (double)0;
            for (int i = 0; i < m; i++) { double d = math.abs(Gram[i, i]); if (d > scale) scale = d; }
            double ridge = (double)m * Consts.doubleEpsilon * scale;
            if (!(ridge > (double)0)) ridge = Consts.doubleEpsilon;

            for (int i = 0; i < m; i++) Gram[i, i] += ridge;

            info = CHO.decomp(in Gram, ref L);
            return info.Solved && MinMaxDiagRatio(in L, m) >= pivotRelTol;
        }

        static double MinMaxDiagRatio(in doubleMxN L, int m)
        {
            double mn = math.abs(L[0, 0]);
            double mx = mn;
            for (int i = 1; i < m; i++)
            {
                double d = math.abs(L[i, i]);
                if (d < mn) mn = d;
                if (d > mx) mx = d;
            }
            return mx > (double)0 ? mn / mx : (double)0;
        }

        // Attempts the full Rayleigh-Ritz reduction for the active window (2-block or 3-block per
        // `useP`): build Gram/H, factor Gram (with retry), reduce to the standard eigenproblem
        // Ahat = L^-1 H L^-T, solve it, recover the combination coefficients C = L^-T Y, and write
        // the selected (smallest numActive) Ritz values directly into ws.lambda[0..numActive) --
        // done here (rather than returned) because the eigenvalues live in a small Allocator.Temp
        // buffer that is disposed before this method returns. Returns false (caller falls back or
        // stalls) if Cholesky or the small eigensolve fails.
        static bool TryRayleighRitz(ref doubleLOBPCGCache ws, int numActive, bool useP, int n)
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
            double qMin = double.MaxValue, qMax = -double.MaxValue;
            for (int i = 0; i < m; i++)
            {
                double gi = G[i, i];
                if (!(gi > (double)0)) continue; // shouldn't happen for a normalized block; skip defensively
                double q = Hv[i, i] / gi;
                if (q < qMin) qMin = q;
                if (q > qMax) qMax = q;
            }
            // Generous margin: a genuine Ritz value from superposing these rows can exceed their
            // individual quotient range somewhat, but never by orders of magnitude -- 1000x the
            // observed span (plus a small additive floor for the degenerate span==0 case)
            // comfortably separates that from actual numerical garbage (observed magnitudes
            // exceeded this envelope by 1E5-1E30x).
            double margin = (double)1000 * (qMax - qMin + (double)1);
            double envMin = qMin - margin;
            double envMax = qMax + margin;

            if (!FactorGram(ref G, ref Lv, m))
                return false;

            var Atrans = View(in ws.Atrans, m);
            FormAtrans(ref Hv, ref Lv, ref Atrans, m);

            // Symmetrize (roundoff insurance -- symmetric requires exact-within-eps symmetry;
            // the two triangular-solve passes above are mathematically symmetric but not bit-exact).
            for (int i = 0; i < m; i++)
                for (int j = i + 1; j < m; j++)
                {
                    double avg = (double)0.5 * (Atrans[i, j] + Atrans[j, i]);
                    Atrans[i, j] = avg;
                    Atrans[j, i] = avg;
                }

            var Yv = View(in ws.Y, m);
            var eigSmall = new doubleN(m, Allocator.Temp);
            bool eigOk = Eigen.symmetric(ref Atrans, ref eigSmall, ref Yv);

            if (eigOk)
            {
                // Validate the selected candidates against the trustworthy envelope BEFORE
                // committing them to ws.lambda (see safeguard 3 above).
                for (int j = 0; j < numActive; j++)
                {
                    double candidate = eigSmall[m - numActive + j];
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
        static void FormAtrans(ref doubleMxN H, ref doubleMxN L, ref doubleMxN Atrans, int m)
        {
            var tmp = new doubleN(m, Allocator.Temp);

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
        static void RecoverC(ref doubleMxN Y, ref doubleMxN L, ref doubleMxN C, int m)
        {
            var tmp = new doubleN(m, Allocator.Temp);

            for (int col = 0; col < m; col++)
            {
                for (int r = 0; r < m; r++) tmp[r] = Y[r, col];

                for (int r = m - 1; r >= 0; r--)
                {
                    double sum = (double)0;
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
        static void UpdateActiveBlock(ref doubleLOBPCGCache ws, int numActive, bool useP, int n, int k)
        {
            int m = useP ? 3 * numActive : 2 * numActive;
            var Cv = View(in ws.C, m);

            for (int j = 0; j < numActive; j++)
            {
                int col = m - numActive + j;

                for (int c = 0; c < n; c++)
                {
                    double xv = (double)0;

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
                    double pv = (double)0;

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

        static void SortAscending(ref doubleLOBPCGCache ws, int k)
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

        static double MaxRelResidual(in doubleLOBPCGCache ws, int k)
        {
            double worst = 0;
            for (int i = 0; i < k; i++)
            {
                double scale = math.abs(ws.lambda[i]);
                if (scale < (double)1) scale = (double)1;
                double rel = (double)(ws.residual[i] / scale);
                if (rel > worst) worst = rel;
            }
            return worst;
        }
    }
}
