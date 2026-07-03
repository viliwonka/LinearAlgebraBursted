#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using LinearAlgebra.Internal;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Eigen {

        /// <summary>
        /// Power iteration with Rayleigh-quotient eigenvalue estimate, generic over any
        /// <see cref="IfProxyLinearOperator"/> (Burst-monomorphized static dispatch, no
        /// vtable/managed delegate). This is the SINGLE SOURCE OF TRUTH for the power-iteration
        /// loop — the concrete dense (<c>powerIteration(in fProxyMxN, ...)</c>) and BSR
        /// (<c>powerIteration(in fProxyBSR, ...)</c>) overloads below are thin forwarders that
        /// wrap their matrix in <see cref="fProxyDenseOperator"/> / <c>fProxyBSROperator</c> and
        /// call this method (mirrors <see cref="Solvers.cg{TOp}"/>).
        ///
        /// Finds the dominant eigenpair (lambda, v) of a square operator A (A.Rows == A.Cols).
        ///
        /// On input: v (length A.Rows) is the initial guess for the eigenvector; w (length
        /// A.Rows) is caller-provided scratch storage — it is overwritten and must NOT be the
        /// same array as v. On output: v is the unit eigenvector estimate; lambda is the
        /// Rayleigh quotient estimate (v^T A v).
        ///
        /// If the supplied v has zero 2-norm it is seeded deterministically as
        /// v[i] = 1 + (i &amp; 3), then normalized before iterating.
        ///
        /// Convergence criterion: the infinity norm of the residual r = A*v - lambda*v
        /// satisfies r &lt;= tol * max(1, |lambda|). Returns an <see cref="EigenSolveInfo"/>
        /// (implicit-bool == Converged) carrying that residual (widened to double) and the
        /// iteration count; power iteration has no Breakdown status -- only Converged or
        /// MaxIterations.
        ///
        /// Notes:
        ///   - Converges to the dominant eigenpair when |lambda_1| &gt; |lambda_2|;
        ///     the rate is |lambda_2 / lambda_1| per iteration.
        ///   - For a negative dominant eigenvalue the eigenvector sign may alternate
        ///     between iterations, but the residual still converges.
        ///   - When the dominant eigenvalue is a complex conjugate pair (e.g. rotation
        ///     matrices) the iteration cannot converge and the method reports MaxIterations
        ///     after maxIter iterations.
        ///   - Inputs of extreme magnitude (entries whose squares overflow the type) are
        ///     not rescaled in this version; keep element magnitudes moderate.
        ///   - Does not allocate.
        /// </summary>
        public static EigenSolveInfo powerIteration<TOp>(in TOp A, ref fProxyN v, ref fProxyN w,
                                               out fProxy lambda, fProxy tol, int maxIter)
            where TOp : struct, IfProxyLinearOperator
        {
            if (A.Rows != A.Cols)
                throw new ArgumentException("powerIteration: A must be square");

            if (v.N != A.Rows)
                throw new ArgumentException("powerIteration: v.N must equal A.Rows");

            if (w.N != A.Rows)
                throw new ArgumentException("powerIteration: w.N must equal A.Rows");

            unsafe {
                if (v.Data.Ptr == w.Data.Ptr)
                    throw new ArgumentException("powerIteration: w must not alias v");
            }

            if (maxIter < 1)
                throw new ArgumentException("powerIteration: maxIter must be >= 1");

            if (tol <= (fProxy)0)
                throw new ArgumentException("powerIteration: tol must be > 0");

            int n = A.Rows;

            // Seed v deterministically if the caller supplied the zero vector
            fProxy vNormSq = (fProxy)0;
            for (int i = 0; i < n; i++)
                vNormSq += v[i] * v[i];

            if (vNormSq == (fProxy)0) {
                for (int i = 0; i < n; i++)
                    v[i] = (fProxy)(1 + (i & 3));
                vNormSq = (fProxy)0;
                for (int i = 0; i < n; i++)
                    vNormSq += v[i] * v[i];
            }

            // Normalize v to unit length
            fProxy vNorm = math.sqrt(vNormSq);
            fProxy invVNorm = (fProxy)1 / vNorm;
            for (int i = 0; i < n; i++)
                v[i] = v[i] * invVNorm;

            lambda = (fProxy)0;

            for (int iter = 0; iter < maxIter; iter++) {

                // Step 1: w = A * v (no allocation — the operator's own Apply, e.g. a manual
                // matvec for dense or spMV for a BSR)
                A.Apply(in v, ref w);

                // Step 2: lambda = v . w (Rayleigh quotient; ||v||_2 = 1)
                lambda = (fProxy)0;
                for (int i = 0; i < n; i++)
                    lambda += v[i] * w[i];

                // Step 3: residual r = max_i |w[i] - lambda * v[i]|  (infinity norm)
                fProxy residual = (fProxy)0;
                for (int i = 0; i < n; i++) {
                    fProxy ri = math.abs(w[i] - lambda * v[i]);
                    if (ri > residual)
                        residual = ri;
                }

                // Step 4: convergence check
                fProxy scale = math.abs(lambda);
                if (scale < (fProxy)1)
                    scale = (fProxy)1;
                if (residual <= tol * scale)
                    return new EigenSolveInfo { iterations = iter + 1, residual = (double)residual, status = IterativeSolveStatus.Converged };

                // Step 5: compute ||w||_2; handle exact null-space case
                fProxy nw = (fProxy)0;
                for (int i = 0; i < n; i++)
                    nw += w[i] * w[i];
                nw = math.sqrt(nw);

                if (nw == (fProxy)0) {
                    lambda = (fProxy)0;
                    return new EigenSolveInfo { iterations = iter + 1, residual = 0.0, status = IterativeSolveStatus.Converged };
                }

                // Step 6: v = w / ||w||
                fProxy invNw = (fProxy)1 / nw;
                for (int i = 0; i < n; i++)
                    v[i] = w[i] * invNw;
            }

            // Post-loop: recompute w = A*v, lambda, residual with final v
            A.Apply(in v, ref w);

            lambda = (fProxy)0;
            for (int i = 0; i < n; i++)
                lambda += v[i] * w[i];

            fProxy finalResidual = (fProxy)0;
            for (int i = 0; i < n; i++) {
                fProxy ri = math.abs(w[i] - lambda * v[i]);
                if (ri > finalResidual)
                    finalResidual = ri;
            }

            fProxy finalScale = math.abs(lambda);
            if (finalScale < (fProxy)1)
                finalScale = (fProxy)1;

            bool finalOk = finalResidual <= tol * finalScale;
            return new EigenSolveInfo {
                iterations = maxIter,
                residual = (double)finalResidual,
                status = finalOk ? IterativeSolveStatus.Converged : IterativeSolveStatus.MaxIterations
            };
        }

        /// <summary>
        /// Power iteration with Rayleigh-quotient eigenvalue estimate over a dense
        /// <see cref="fProxyMxN"/>. Forwards into <see cref="powerIteration{TOp}"/> via
        /// <see cref="fProxyDenseOperator"/> — see that method for the actual loop and the full
        /// algorithm documentation (deterministic seeding, convergence criterion, notes).
        /// </summary>
        public static EigenSolveInfo powerIteration(in fProxyMxN A, ref fProxyN v, ref fProxyN w,
                                          out fProxy lambda, fProxy tol, int maxIter)
        {
            return powerIteration(new fProxyDenseOperator(in A), ref v, ref w, out lambda, tol, maxIter);
        }

        /// <summary>powerIteration with default maxIter (1000).</summary>
        public static EigenSolveInfo powerIteration(in fProxyMxN A, ref fProxyN v, ref fProxyN w,
                                          out fProxy lambda, fProxy tol)
            => powerIteration(in A, ref v, ref w, out lambda, tol, 1000);

        /// <summary>powerIteration with default tol (Consts.fProxyZeroThreshold) and maxIter (1000).</summary>
        public static EigenSolveInfo powerIteration(in fProxyMxN A, ref fProxyN v, ref fProxyN w,
                                          out fProxy lambda)
            => powerIteration(in A, ref v, ref w, out lambda, Consts.fProxyZeroThreshold, 1000);

        /// <summary>
        /// Power iteration with Rayleigh-quotient eigenvalue estimate over a block-sparse (BSR)
        /// matrix. Same semantics as the dense overload — see
        /// <see cref="powerIteration(in fProxyMxN, ref fProxyN, ref fProxyN, out fProxy, fProxy, int)"/>.
        /// Forwards into <see cref="powerIteration{TOp}"/> via <c>fProxyBSROperator</c>.
        /// </summary>
        public static EigenSolveInfo powerIteration(in fProxyBSR A, ref fProxyN v, ref fProxyN w,
                                          out fProxy lambda, fProxy tol, int maxIter)
        {
            return powerIteration(new fProxyBSROperator(in A), ref v, ref w, out lambda, tol, maxIter);
        }

        /// <summary>powerIteration over a block-sparse (BSR) matrix with default maxIter (1000).</summary>
        public static EigenSolveInfo powerIteration(in fProxyBSR A, ref fProxyN v, ref fProxyN w,
                                          out fProxy lambda, fProxy tol)
            => powerIteration(in A, ref v, ref w, out lambda, tol, 1000);

        /// <summary>
        /// powerIteration over a block-sparse (BSR) matrix with default tol
        /// (Consts.fProxyZeroThreshold) and maxIter (1000).
        /// </summary>
        public static EigenSolveInfo powerIteration(in fProxyBSR A, ref fProxyN v, ref fProxyN w,
                                          out fProxy lambda)
            => powerIteration(in A, ref v, ref w, out lambda, Consts.fProxyZeroThreshold, 1000);

        /// <summary>
        /// Inverse power iteration for the SMALLEST eigenpair (lambda_min, v) of a symmetric
        /// positive-definite (SPD) operator A (A.Rows == A.Cols), generic over any
        /// <see cref="IfProxyLinearOperator"/> (Burst-monomorphized static dispatch, no
        /// vtable/managed delegate) -- same shape as <see cref="powerIteration{TOp}"/>. This is
        /// the SINGLE SOURCE OF TRUTH for the inverse-iteration loop -- the concrete dense
        /// (<c>inversePowerIteration(in fProxyMxN, ...)</c>) and BSR
        /// (<c>inversePowerIteration(in fProxyBSR, ...)</c>) overloads below are thin forwarders
        /// that wrap their matrix in <see cref="fProxyDenseOperator"/> / <c>fProxyBSROperator</c>
        /// and call this method (mirrors <see cref="powerIteration{TOp}"/>).
        ///
        /// A^-1 amplifies the SMALLEST-magnitude eigencomponent of A, so ordinary power iteration
        /// on A^-1 converges to the eigenvector of A's smallest eigenvalue -- this is the roadmap's
        /// lambda_min capability (e.g. the Fiedler vector of a graph Laplacian, or the lowest
        /// vibration mode of a stiffness matrix). Rather than forming/factoring A^-1, each outer
        /// iteration solves A y = v with the zero-alloc generic <see cref="Solvers.cg{TOp}"/> (A
        /// must be SPD and nonsingular for CG to converge -- e.g.
        /// <c>LinearAlgebra.Gallery.fProxyGallery.fProxyLaplacian1D</c> qualifies), then normalizes
        /// y into v.
        ///
        /// PRECONDITION (caller responsibility, not verified at runtime -- same contract as CG's
        /// "A must be SPD"): A is symmetric positive-definite and nonsingular (lambda_min &gt; 0).
        ///
        /// Scratch layout: v (length A.Rows) is the eigenvector estimate, in/out -- WARM-STARTABLE
        /// and, like <see cref="powerIteration{TOp}"/>, deterministically seeded as
        /// v[i] = 1 + (i &amp; 3) then normalized if the caller supplies the zero vector. y is the
        /// inner solve's solution scratch (A y = v). r, p, Ap are <see cref="Solvers.cg{TOp}"/>'s
        /// own scratch, reused across every outer iteration -- zero-alloc overall. No extra scratch
        /// vector is needed for the Rayleigh-quotient recompute: once CG returns, r/p/Ap are free,
        /// so Ap doubles as the A*v scratch for that step.
        ///
        /// On output: v is the unit eigenvector estimate for A's smallest eigenvalue; lambda is the
        /// Rayleigh quotient v^T A v / v^T v (recomputed via A.Apply -- not carried over from CG).
        ///
        /// Convergence (checked once per outer iteration, OR'd): (1) the eigenvector settles -- the
        /// infinity norm of v_new - v_old is &lt;= tol, where v_new is sign-aligned against v_old
        /// first (inverse iteration, like power iteration, can flip the eigenvector's sign between
        /// iterations); or (2) the Rayleigh quotient stabilizes -- |lambda_new - lambda_old| &lt;=
        /// tol * max(1, |lambda_new|). Returns an <see cref="EigenSolveInfo"/> (implicit-bool ==
        /// Converged); Converged within maxIter outer iterations, MaxIterations if it runs out,
        /// Breakdown if the inner CG solve fails (see below). On Converged/MaxIterations, residual
        /// is ‖Av-λv‖∞ computed once from the last outer iteration's already-held A*v (no extra
        /// matvec); on Breakdown, residual is <see cref="double.NaN"/>.
        ///
        /// IMPORTANT: pick tol no tighter than (and ideally a small multiple of) cgTol. Every outer
        /// iteration's v/y comes from a FRESH CG solve accurate only to ~cgTol -- consecutive
        /// eigenpair estimates stop shrinking once that noise floor is reached (further outer
        /// iterations do not refine it further, unlike <see cref="powerIteration{TOp}"/>'s pure
        /// matvecs, which can drive the residual to machine precision). A tol tighter than cgTol's
        /// noise floor may never be satisfied, spinning to maxIter and reporting MaxIterations even
        /// though the eigenpair estimate is already as good as this cgTol allows.
        ///
        /// If the inner CG solve fails to converge within cgMaxIter iterations to cgTol (A not SPD,
        /// or numerical breakdown -- see <see cref="Solvers.cg{TOp}"/>), this method bails out
        /// immediately and reports Breakdown; lambda is then set to 0 (undefined) and v holds
        /// whatever CG last produced (partially updated) -- only read v/lambda when Solved.
        ///
        /// SEEDING CAVEAT: when the caller passes the zero vector, v is seeded with the fixed
        /// deterministic pattern (1,2,3,4,1,2,3,4,...). If the target smallest eigenvector is
        /// (near-)orthogonal to that pattern -- possible for structured/symmetric matrices -- the
        /// iteration converges to the next eigenpair and reports Converged with a WRONG lambda (the
        /// Rayleigh quotient stabilizes on the wrong-but-stable pair). Callers with structured A
        /// should pass their own nonzero seed rather than relying on the default pattern. Same
        /// caveat as <see cref="powerIteration{TOp}"/>, but higher risk here: the CG-noise-floor
        /// convergence branch can accept a wrong pair that pure matvecs would eventually escape.
        ///
        /// Does not allocate.
        /// </summary>
        public static EigenSolveInfo inversePowerIteration<TOp>(in TOp A, ref fProxyN v, ref fProxyN y,
                                                      ref fProxyN r, ref fProxyN p, ref fProxyN Ap,
                                                      out fProxy lambda,
                                                      fProxy tol, int maxIter, int cgMaxIter, fProxy cgTol)
            where TOp : struct, IfProxyLinearOperator
        {
            if (A.Rows != A.Cols)
                throw new ArgumentException("inversePowerIteration: A must be square");

            int n = A.Rows;

            if (v.N != n)
                throw new ArgumentException("inversePowerIteration: v.N must equal A.Rows");

            if (y.N != n)
                throw new ArgumentException("inversePowerIteration: y.N must equal A.Rows");

            if (r.N != n)
                throw new ArgumentException("inversePowerIteration: r.N must equal A.Rows");

            if (p.N != n)
                throw new ArgumentException("inversePowerIteration: p.N must equal A.Rows");

            if (Ap.N != n)
                throw new ArgumentException("inversePowerIteration: Ap.N must equal A.Rows");

            if (maxIter < 1)
                throw new ArgumentException("inversePowerIteration: maxIter must be >= 1");

            if (tol <= (fProxy)0)
                throw new ArgumentException("inversePowerIteration: tol must be > 0");

            // Aliasing guard: v/y/r/p/Ap must all be distinct buffers -- same rationale as
            // cg<TOp>'s guard (the loop below, and cg's own loop, mix elementwise scratch updates
            // that don't self-check aliasing, so silent corruption would replace a thrown
            // exception). Five buffers -> a hand-expanded OR chain, same style as cg<TOp>.
            unsafe
            {
                fProxy* vPtr = v.Data.Ptr, yPtr = y.Data.Ptr, rPtr = r.Data.Ptr, pPtr = p.Data.Ptr, ApPtr = Ap.Data.Ptr;

                if (vPtr == yPtr || vPtr == rPtr || vPtr == pPtr || vPtr == ApPtr ||
                    yPtr == rPtr || yPtr == pPtr || yPtr == ApPtr ||
                    rPtr == pPtr || rPtr == ApPtr ||
                    pPtr == ApPtr)
                    throw new ArgumentException("inversePowerIteration: v/y/r/p/Ap must be distinct");
            }

            // Seed v deterministically if the caller supplied the zero vector (mirrors powerIteration).
            fProxy vNormSq = (fProxy)0;
            for (int i = 0; i < n; i++)
                vNormSq += v[i] * v[i];

            if (vNormSq == (fProxy)0) {
                for (int i = 0; i < n; i++)
                    v[i] = (fProxy)(1 + (i & 3));
                vNormSq = (fProxy)0;
                for (int i = 0; i < n; i++)
                    vNormSq += v[i] * v[i];
            }

            fProxy vNorm = math.sqrt(vNormSq);
            fProxy invVNorm = (fProxy)1 / vNorm;
            for (int i = 0; i < n; i++)
                v[i] = v[i] * invVNorm;

            lambda = (fProxy)0;
            fProxy lambdaPrev = fProxy.NaN;   // sentinel: no previous estimate yet (NaN-safe compare below)

            for (int iter = 0; iter < maxIter; iter++) {

                // Step 1: solve A y = v via CG (reuses r/p/Ap as CG's own scratch every outer
                // iteration -- zero additional allocation). A false return means CG broke down
                // (A not SPD from this v, or numerical breakdown); bail out immediately.
                bool cgOk = Solvers.cg(in A, in v, ref y, ref r, ref p, ref Ap, cgMaxIter, cgTol);
                if (!cgOk) {
                    lambda = (fProxy)0;
                    return new EigenSolveInfo { iterations = iter, residual = double.NaN, status = IterativeSolveStatus.Breakdown };
                }

                // Step 2: ||y|| for normalization. y == 0 can only happen if v == 0, which cannot
                // occur here (v is unit-norm going into CG) -- guarded anyway rather than dividing
                // by zero into Inf/NaN.
                fProxy yNormSq = (fProxy)0;
                for (int i = 0; i < n; i++)
                    yNormSq += y[i] * y[i];

                if (yNormSq == (fProxy)0) {
                    lambda = (fProxy)0;
                    return new EigenSolveInfo { iterations = iter, residual = double.NaN, status = IterativeSolveStatus.Breakdown };
                }

                fProxy invYNorm = (fProxy)1 / math.sqrt(yNormSq);

                // Step 3: sign-aware convergence check against the PREVIOUS v (before it is
                // overwritten). Align the new candidate's sign to the old v via their dot product
                // (mirrors powerIteration's note: the eigenvector may flip sign between
                // iterations), then commit the sign-aligned, normalized candidate as the new v.
                fProxy alignDot = (fProxy)0;
                for (int i = 0; i < n; i++)
                    alignDot += v[i] * (y[i] * invYNorm);
                fProxy sign = alignDot >= (fProxy)0 ? (fProxy)1 : (fProxy)(-1);

                fProxy vecDiff = (fProxy)0;
                for (int i = 0; i < n; i++) {
                    fProxy vNew = sign * y[i] * invYNorm;
                    fProxy di = math.abs(vNew - v[i]);
                    if (di > vecDiff) vecDiff = di;
                    v[i] = vNew;
                }

                // Step 4: Rayleigh quotient lambda = v^T A v / v^T v. v is unit-norm by
                // construction, but v^T v is recomputed (not assumed exactly 1) to absorb
                // roundoff. Ap is free here -- CG's own use of it ended when CG returned above --
                // so it doubles as the A*v scratch; no extra scratch vector is needed.
                A.Apply(in v, ref Ap);

                fProxy vtv = (fProxy)0, vtAv = (fProxy)0;
                for (int i = 0; i < n; i++) {
                    vtv += v[i] * v[i];
                    vtAv += v[i] * Ap[i];
                }
                lambda = vtAv / vtv;

                // Step 5: convergence -- eigenvector settled OR Rayleigh quotient stabilized.
                fProxy lambdaScale = math.abs(lambda);
                if (lambdaScale < (fProxy)1) lambdaScale = (fProxy)1;
                fProxy lambdaChange = math.abs(lambda - lambdaPrev);   // NaN on iter 0 -> false below

                if (vecDiff <= tol || lambdaChange <= tol * lambdaScale)
                    return new EigenSolveInfo {
                        iterations = iter + 1,
                        residual = InversePowerResidual(in Ap, in v, lambda, n),
                        status = IterativeSolveStatus.Converged
                    };

                lambdaPrev = lambda;
            }

            return new EigenSolveInfo {
                iterations = maxIter,
                residual = InversePowerResidual(in Ap, in v, lambda, n),
                status = IterativeSolveStatus.MaxIterations
            };
        }

        // ‖Ap - lambda*v‖_inf -- inversePowerIteration's Converged/MaxIterations returns compute
        // their residual from here. Ap already holds A*v from the loop's last A.Apply (Step 4), so
        // this is a single extra O(n) pass at the return site, not an extra matvec.
        static double InversePowerResidual(in fProxyN Ap, in fProxyN v, fProxy lambda, int n)
        {
            fProxy residual = (fProxy)0;
            for (int i = 0; i < n; i++) {
                fProxy ri = math.abs(Ap[i] - lambda * v[i]);
                if (ri > residual) residual = ri;
            }
            return (double)residual;
        }

        /// <summary>
        /// Inverse power iteration for the smallest eigenpair of a SPD dense
        /// <see cref="fProxyMxN"/>. Forwards into <see cref="inversePowerIteration{TOp}"/> via
        /// <see cref="fProxyDenseOperator"/> -- see that method for the actual loop and the full
        /// algorithm documentation (deterministic seeding, convergence criteria, scratch layout,
        /// SPD/nonsingular precondition).
        /// </summary>
        public static EigenSolveInfo inversePowerIteration(in fProxyMxN A, ref fProxyN v, ref fProxyN y,
                                                 ref fProxyN r, ref fProxyN p, ref fProxyN Ap,
                                                 out fProxy lambda,
                                                 fProxy tol, int maxIter, int cgMaxIter, fProxy cgTol)
        {
            return inversePowerIteration(new fProxyDenseOperator(in A), ref v, ref y, ref r, ref p, ref Ap, out lambda, tol, maxIter, cgMaxIter, cgTol);
        }

        /// <summary>inversePowerIteration with default cgMaxIter (A.M_Rows) and cgTol (Consts.fProxySqrtEps).</summary>
        public static EigenSolveInfo inversePowerIteration(in fProxyMxN A, ref fProxyN v, ref fProxyN y,
                                                 ref fProxyN r, ref fProxyN p, ref fProxyN Ap,
                                                 out fProxy lambda, fProxy tol, int maxIter)
            => inversePowerIteration(in A, ref v, ref y, ref r, ref p, ref Ap, out lambda, tol, maxIter, A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>
        /// Inverse power iteration over a dense SPD matrix -- allocates the inner-solve scratch
        /// (y, r, p, Ap; all length A.M_Rows) from the arena that <paramref name="v"/> carries and
        /// calls the zero-alloc primitive. Use the ref-scratch overload in hot loops to avoid the
        /// allocation.
        /// </summary>
        public static EigenSolveInfo inversePowerIteration(in fProxyMxN A, ref fProxyN v, out fProxy lambda,
                                                 fProxy tol, int maxIter, int cgMaxIter, fProxy cgTol)
        {
            fProxyN y  = v.fProxyTempVec(A.M_Rows);
            fProxyN r  = v.fProxyTempVec(A.M_Rows);
            fProxyN p  = v.fProxyTempVec(A.M_Rows);
            fProxyN Ap = v.fProxyTempVec(A.M_Rows);
            return inversePowerIteration(in A, ref v, ref y, ref r, ref p, ref Ap, out lambda, tol, maxIter, cgMaxIter, cgTol);
        }

        /// <summary>
        /// inversePowerIteration (allocating) with default tol (10 * Consts.fProxySqrtEps),
        /// maxIter (1000), cgMaxIter (A.M_Rows) and cgTol (Consts.fProxySqrtEps). tol defaults to
        /// a multiple of cgTol (NOT the much tighter Consts.fProxyZeroThreshold) on purpose: the
        /// outer convergence checks compare CONSECUTIVE eigenpair estimates, each derived from its
        /// own fresh CG solve accurate only to ~cgTol -- an outer tolerance tighter than that noise
        /// floor could spin to maxIter without ever detecting convergence (the residual genuinely
        /// bottoms out around cgTol, it does not keep shrinking with more outer iterations).
        /// </summary>
        public static EigenSolveInfo inversePowerIteration(in fProxyMxN A, ref fProxyN v, out fProxy lambda)
            => inversePowerIteration(in A, ref v, out lambda, (fProxy)10 * Consts.fProxySqrtEps, 1000, A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>
        /// Inverse power iteration for the smallest eigenpair of a SPD block-sparse (BSR) matrix.
        /// Same semantics as the dense overload -- see
        /// <see cref="inversePowerIteration(in fProxyMxN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, ref fProxyN, out fProxy, fProxy, int, int, fProxy)"/>.
        /// Forwards into <see cref="inversePowerIteration{TOp}"/> via <c>fProxyBSROperator</c>.
        /// </summary>
        public static EigenSolveInfo inversePowerIteration(in fProxyBSR A, ref fProxyN v, ref fProxyN y,
                                                 ref fProxyN r, ref fProxyN p, ref fProxyN Ap,
                                                 out fProxy lambda,
                                                 fProxy tol, int maxIter, int cgMaxIter, fProxy cgTol)
        {
            return inversePowerIteration(new fProxyBSROperator(in A), ref v, ref y, ref r, ref p, ref Ap, out lambda, tol, maxIter, cgMaxIter, cgTol);
        }

        /// <summary>inversePowerIteration over a BSR matrix with default cgMaxIter (A.M_Rows) and cgTol (Consts.fProxySqrtEps).</summary>
        public static EigenSolveInfo inversePowerIteration(in fProxyBSR A, ref fProxyN v, ref fProxyN y,
                                                 ref fProxyN r, ref fProxyN p, ref fProxyN Ap,
                                                 out fProxy lambda, fProxy tol, int maxIter)
            => inversePowerIteration(in A, ref v, ref y, ref r, ref p, ref Ap, out lambda, tol, maxIter, A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>
        /// Inverse power iteration over a BSR SPD matrix -- allocates the inner-solve scratch from
        /// the arena that <paramref name="v"/> carries and calls the zero-alloc primitive.
        /// </summary>
        public static EigenSolveInfo inversePowerIteration(in fProxyBSR A, ref fProxyN v, out fProxy lambda,
                                                 fProxy tol, int maxIter, int cgMaxIter, fProxy cgTol)
        {
            fProxyN y  = v.fProxyTempVec(A.M_Rows);
            fProxyN r  = v.fProxyTempVec(A.M_Rows);
            fProxyN p  = v.fProxyTempVec(A.M_Rows);
            fProxyN Ap = v.fProxyTempVec(A.M_Rows);
            return inversePowerIteration(in A, ref v, ref y, ref r, ref p, ref Ap, out lambda, tol, maxIter, cgMaxIter, cgTol);
        }

        /// <summary>
        /// inversePowerIteration (allocating) over a BSR matrix with default tol
        /// (10 * Consts.fProxySqrtEps), maxIter (1000), cgMaxIter (A.M_Rows) and cgTol
        /// (Consts.fProxySqrtEps). See the dense overload's doc comment for why tol defaults to a
        /// multiple of cgTol rather than the much tighter Consts.fProxyZeroThreshold.
        /// </summary>
        public static EigenSolveInfo inversePowerIteration(in fProxyBSR A, ref fProxyN v, out fProxy lambda)
            => inversePowerIteration(in A, ref v, out lambda, (fProxy)10 * Consts.fProxySqrtEps, 1000, A.M_Rows, Consts.fProxySqrtEps);

        /// <summary>
        /// Lanczos tridiagonalization of a SYMMETRIC operator A (A.Rows == A.Cols), generic over
        /// any <see cref="IfProxyLinearOperator"/> (Burst-monomorphized static dispatch, no
        /// vtable/managed delegate) -- same shape as <see cref="powerIteration{TOp}"/> /
        /// <see cref="inversePowerIteration{TOp}"/>. This is the SINGLE SOURCE OF TRUTH for the
        /// Lanczos loop -- the concrete dense (<c>lanczos(in fProxyMxN, ...)</c>) and BSR
        /// (<c>lanczos(in fProxyBSR, ...)</c>) overloads below are thin forwarders that wrap their
        /// matrix in <see cref="fProxyDenseOperator"/> / <c>fProxyBSROperator</c> and call this
        /// method.
        ///
        /// Builds an orthonormal Krylov basis v_1..v_m (m = <paramref name="steps"/>) via the
        /// classical 3-term Lanczos recurrence and the corresponding symmetric tridiagonal T (diag
        /// alpha_1..alpha_m, off-diag beta_2..beta_m), then reuses
        /// <see cref="eigenvaluesSymmetric(ref fProxyMxN, ref fProxyN, ref fProxyEigenSymCache)"/> on
        /// T to obtain the Ritz values -- approximate eigenvalues of A. The EXTREMAL Ritz values
        /// (largest and smallest) converge fastest and are already accurate for m &lt;&lt; A.Rows;
        /// with m == A.Rows and full reorthogonalization, T is orthogonally similar to A and the
        /// Ritz values reproduce A's ENTIRE spectrum.
        ///
        /// PRECONDITION (caller responsibility, not verified at runtime -- same contract as
        /// <see cref="Solvers.minres{TOp}"/>'s "A must be symmetric"): A is symmetric. Lanczos is
        /// undefined for a non-symmetric operator (the 3-term recurrence assumes it).
        ///
        /// FULL REORTHOGONALIZATION (against every previously computed v_1..v_j), performed TWICE
        /// per iteration: bare/naive Lanczos loses orthogonality among the v_j to rounding error
        /// after relatively few steps, which manufactures spurious "ghost" duplicate eigenvalues in
        /// T. A single reorthogonalization pass already restores orthogonality to working
        /// precision; the second pass ("reorthogonalize twice" -- classical folklore, e.g.
        /// Parlett's "Symmetric Eigenvalue Problem") is cheap insurance (an extra O(m*n) sweep per
        /// iteration, negligible next to the O(m^2*n) total the first pass already costs) traded
        /// for extra robustness -- this implementation prioritizes correctness over the last bit of
        /// speed, per its purpose as a general-purpose extremal eigensolver.
        ///
        /// Workspace (see <see cref="fProxyLanczosCache"/>, allocate via
        /// <c>Arena.fProxyLanczosCache(A.Rows, steps)</c>): <c>ws.V</c> (steps x n) accumulates the
        /// Krylov basis (row j = v_(j+1)); <c>ws.vCur</c>/<c>ws.w</c> (length n) are the current
        /// Krylov vector (copied out of V for A.Apply, which requires distinct in/out buffers) and
        /// the work vector; <c>ws.alpha</c>/<c>ws.beta</c> (length steps) are T's diagonal/off-
        /// diagonal; <c>ws.T</c> (steps x steps) and <c>ws.symWs</c> back the
        /// eigenvaluesSymmetric call. On input, row 0 of <c>ws.V</c> is the seed for v_1: if it has
        /// zero 2-norm it is seeded deterministically as V[0,i] = 1 + (i &amp; 3) (mirrors
        /// <see cref="powerIteration{TOp}"/>'s seeding), then normalized either way.
        ///
        /// EARLY BREAKDOWN: if at some iteration j &lt; steps the residual norm beta_(j+1) falls to
        /// <paramref name="breakdownTol"/> or below, an invariant subspace of A has been found --
        /// the j Ritz values already computed are EXACT eigenvalues of A (restricted to that
        /// subspace), and there is no further information to extract, so the process truncates
        /// there. The returned <see cref="LanczosInfo.produced"/> reports how many of the
        /// m = <paramref name="steps"/> requested vectors were actually produced (produced &lt;=
        /// steps; produced == steps unless breakdown occurred). To keep the workspace's T/symWs a
        /// FIXED steps x steps shape across calls (so the same workspace serves every call
        /// regardless of whether breakdown occurs), the unused steps-produced rows/columns of T are
        /// padded with a block that is exactly DECOUPLED from the real produced x produced block
        /// (the connecting off-diagonal entry is left at its zero-initialized value) and whose
        /// diagonal entries lie strictly below every real Ritz value (offset past a Gershgorin bound
        /// on the real block), so the padding can never mix into the real spectrum under sorting.
        ///
        /// On output: <paramref name="eigenvalues"/> (length steps, caller-allocated) holds the
        /// Ritz values sorted DESCENDING (same convention as eigenvaluesSymmetric) in its first
        /// <c>produced</c> entries -- eigenvalues[0] is the largest Ritz value,
        /// eigenvalues[produced-1] the smallest. Entries [produced, steps) are the padding
        /// described above and are MEANINGLESS -- ignore them. The returned
        /// <see cref="LanczosInfo.status"/> is Converged iff eigenvaluesSymmetric's QL iteration
        /// converged on the (possibly padded) tridiagonal; only trust the eigenvalues when Solved.
        ///
        /// Does not allocate.
        /// </summary>
        public static LanczosInfo lanczos<TOp>(in TOp A, ref fProxyLanczosCache ws, ref fProxyN eigenvalues,
                                        int steps, fProxy breakdownTol)
            where TOp : struct, IfProxyLinearOperator
        {
            if (eigenvalues.N != steps)
                throw new ArgumentException("lanczos: eigenvalues.N must equal steps");

            lanczosTridiag(in A, ref ws, out int produced, steps, breakdownTol);
            bool ok = eigenvaluesSymmetric(ref ws.T, ref eigenvalues, ref ws.symWs);
            return new LanczosInfo { produced = produced, status = ok ? IterativeSolveStatus.Converged : IterativeSolveStatus.MaxIterations };
        }

        // Shared Lanczos front-half used by both lanczos (eigenvalues only) and lanczosVectors
        // (eigenvalues + Ritz vectors): seed v_1, run the twice-reorthogonalized bidiagonalization
        // building the Krylov basis ws.V, and assemble the (early-breakdown-padded) symmetric
        // tridiagonal ws.T. Sets `produced`. Callers then apply their own symmetric eigensolver to
        // ws.T (eigenvaluesSymmetric for values, eigenSymmetric for values + eigenvectors).
        static void lanczosTridiag<TOp>(in TOp A, ref fProxyLanczosCache ws, out int produced,
                                        int steps, fProxy breakdownTol)
            where TOp : struct, IfProxyLinearOperator
        {
            if (A.Rows != A.Cols)
                throw new ArgumentException("lanczos: A must be square");

            int n = A.Rows;

            if (steps < 1 || steps > n)
                throw new ArgumentException("lanczos: steps must be in [1, A.Rows]");

            if (breakdownTol < (fProxy)0)
                throw new ArgumentException("lanczos: breakdownTol must be >= 0");

            RequireLanczosWorkspace(in ws, n, steps);

            unsafe {
                if (ws.vCur.Data.Ptr == ws.w.Data.Ptr)
                    throw new ArgumentException("lanczos: workspace vCur and w must be distinct buffers");
            }

            // Seed v_1 (row 0 of V) deterministically if the caller left it at zero (mirrors
            // powerIteration's seeding), then normalize either way.
            fProxy seedNormSq = (fProxy)0;
            for (int i = 0; i < n; i++)
                seedNormSq += ws.V[0, i] * ws.V[0, i];

            if (seedNormSq == (fProxy)0) {
                for (int i = 0; i < n; i++)
                    ws.V[0, i] = (fProxy)(1 + (i & 3));
                seedNormSq = (fProxy)0;
                for (int i = 0; i < n; i++)
                    seedNormSq += ws.V[0, i] * ws.V[0, i];
            }

            fProxy invSeedNorm = (fProxy)1 / math.sqrt(seedNormSq);
            for (int i = 0; i < n; i++)
                ws.V[0, i] = ws.V[0, i] * invSeedNorm;

            produced = steps;

            for (int jj = 0; jj < steps; jj++) {

                // vCur = V row jj (A.Apply requires a standalone fProxyN distinct from its output).
                for (int i = 0; i < n; i++)
                    ws.vCur[i] = ws.V[jj, i];

                // w = A * v_j
                A.Apply(in ws.vCur, ref ws.w);

                // Local 3-term recurrence: w -= beta_j * v_(j-1) (skipped for j == 1, i.e. jj == 0).
                if (jj > 0) {
                    fProxy bj = ws.beta[jj - 1];
                    for (int i = 0; i < n; i++)
                        ws.w[i] = ws.w[i] - bj * ws.V[jj - 1, i];
                }

                // alpha_j = v_j . w; w -= alpha_j * v_j
                fProxy alphaJ = (fProxy)0;
                for (int i = 0; i < n; i++)
                    alphaJ += ws.vCur[i] * ws.w[i];
                ws.alpha[jj] = alphaJ;

                for (int i = 0; i < n; i++)
                    ws.w[i] = ws.w[i] - alphaJ * ws.vCur[i];

                // Full reorthogonalization against v_1..v_j, done TWICE (see doc comment).
                for (int pass = 0; pass < 2; pass++) {
                    for (int k = 0; k <= jj; k++) {
                        fProxy proj = (fProxy)0;
                        for (int i = 0; i < n; i++)
                            proj += ws.V[k, i] * ws.w[i];
                        for (int i = 0; i < n; i++)
                            ws.w[i] = ws.w[i] - proj * ws.V[k, i];
                    }
                }

                fProxy wNormSq = (fProxy)0;
                for (int i = 0; i < n; i++)
                    wNormSq += ws.w[i] * ws.w[i];
                fProxy wNorm = math.sqrt(wNormSq);

                if (wNorm <= breakdownTol) {
                    // Invariant subspace found: the jj+1 Ritz values already gathered are exact.
                    produced = jj + 1;
                    break;
                }

                ws.beta[jj] = wNorm;

                if (jj + 1 < steps) {
                    fProxy invW = (fProxy)1 / wNorm;
                    for (int i = 0; i < n; i++)
                        ws.V[jj + 1, i] = ws.w[i] * invW;
                }
            }

            // Assemble the steps x steps symmetric tridiagonal T from alpha/beta (zeroed first so
            // a workspace reused from a previous, larger-`produced` call has no stale entries).
            for (int i = 0; i < steps; i++)
                for (int j = 0; j < steps; j++)
                    ws.T[i, j] = (fProxy)0;

            for (int i = 0; i < produced; i++)
                ws.T[i, i] = ws.alpha[i];

            for (int i = 0; i < produced - 1; i++) {
                ws.T[i, i + 1] = ws.beta[i];
                ws.T[i + 1, i] = ws.beta[i];
            }

            if (produced < steps) {
                // Pad with a block that is exactly decoupled from the real produced x produced
                // block (the off-diagonal entry T[produced-1, produced] stays at its zero-init
                // value) and whose diagonal lies strictly below a Gershgorin bound on the real
                // block, so sorting can never mix the padding into the real Ritz values.
                fProxy bound = (fProxy)0;
                for (int i = 0; i < produced; i++) {
                    fProxy radius = math.abs(ws.alpha[i]);
                    if (i > 0) radius += math.abs(ws.beta[i - 1]);
                    if (i < produced - 1) radius += math.abs(ws.beta[i]);
                    if (radius > bound) bound = radius;
                }

                // Separation between the real block (all Ritz values >= -bound) and the padding is
                // scaled by max(1, bound), not a fixed 1: eigenvaluesSymmetric's QL computes each
                // value with ~eps*||T|| error, so at large ||T|| a fixed unit gap could be swamped
                // and let a real Ritz value sort below the padding. A relative gap keeps the padding
                // strictly, robustly below every real value regardless of spectrum magnitude.
                fProxy pad = bound > (fProxy)1 ? bound : (fProxy)1;
                for (int i = produced; i < steps; i++)
                    ws.T[i, i] = -bound - (fProxy)(i - produced + 1) * pad;
            }
        }

        /// <summary>lanczos with default breakdownTol (Consts.fProxyZeroThreshold).</summary>
        public static LanczosInfo lanczos<TOp>(in TOp A, ref fProxyLanczosCache ws, ref fProxyN eigenvalues,
                                        int steps)
            where TOp : struct, IfProxyLinearOperator
            => lanczos(in A, ref ws, ref eigenvalues, steps, Consts.fProxyZeroThreshold);

        /// <summary>
        /// Lanczos tridiagonalization of a SYMMETRIC dense <see cref="fProxyMxN"/>. Forwards into
        /// <see cref="lanczos{TOp}"/> via <see cref="fProxyDenseOperator"/> -- see that method for
        /// the actual loop and the full algorithm documentation (seeding, full
        /// reorthogonalization, workspace layout, early-breakdown padding, output convention).
        /// </summary>
        public static LanczosInfo lanczos(in fProxyMxN A, ref fProxyLanczosCache ws, ref fProxyN eigenvalues,
                                   int steps, fProxy breakdownTol)
        {
            return lanczos(new fProxyDenseOperator(in A), ref ws, ref eigenvalues, steps, breakdownTol);
        }

        /// <summary>lanczos over a dense matrix with default breakdownTol (Consts.fProxyZeroThreshold).</summary>
        public static LanczosInfo lanczos(in fProxyMxN A, ref fProxyLanczosCache ws, ref fProxyN eigenvalues,
                                   int steps)
            => lanczos(in A, ref ws, ref eigenvalues, steps, Consts.fProxyZeroThreshold);

        /// <summary>
        /// Lanczos over a dense SYMMETRIC matrix -- allocates the workspace (see
        /// <see cref="fProxyLanczosCache"/>) and the output eigenvalues buffer (length steps) from
        /// <paramref name="arena"/> and calls the zero-alloc primitive. Returns the Ritz values;
        /// only the first <c>info.produced</c> entries are meaningful (see
        /// <see cref="lanczos{TOp}"/>'s doc comment). Use the ref-workspace overload in hot loops
        /// to avoid the allocation.
        /// </summary>
        public static fProxyN lanczos(ref Arena arena, in fProxyMxN A, int steps, out LanczosInfo info,
                                      fProxy breakdownTol)
        {
            var ws = arena.fProxyLanczosCache(A.M_Rows, steps);
            var eigenvalues = arena.fProxyVec(steps);
            info = lanczos(in A, ref ws, ref eigenvalues, steps, breakdownTol);
            return eigenvalues;
        }

        /// <summary>lanczos (allocating) over a dense matrix with default breakdownTol (Consts.fProxyZeroThreshold).</summary>
        public static fProxyN lanczos(ref Arena arena, in fProxyMxN A, int steps, out LanczosInfo info)
            => lanczos(ref arena, in A, steps, out info, Consts.fProxyZeroThreshold);

        /// <summary>
        /// Lanczos tridiagonalization of a SYMMETRIC block-sparse (BSR) matrix. Same semantics as
        /// the dense overload -- see
        /// <see cref="lanczos(in fProxyMxN, ref fProxyLanczosCache, ref fProxyN, int, fProxy)"/>.
        /// Forwards into <see cref="lanczos{TOp}"/> via <c>fProxyBSROperator</c>.
        /// </summary>
        public static LanczosInfo lanczos(in fProxyBSR A, ref fProxyLanczosCache ws, ref fProxyN eigenvalues,
                                   int steps, fProxy breakdownTol)
        {
            return lanczos(new fProxyBSROperator(in A), ref ws, ref eigenvalues, steps, breakdownTol);
        }

        /// <summary>lanczos over a BSR matrix with default breakdownTol (Consts.fProxyZeroThreshold).</summary>
        public static LanczosInfo lanczos(in fProxyBSR A, ref fProxyLanczosCache ws, ref fProxyN eigenvalues,
                                   int steps)
            => lanczos(in A, ref ws, ref eigenvalues, steps, Consts.fProxyZeroThreshold);

        /// <summary>
        /// Lanczos over a BSR SYMMETRIC matrix -- allocates the workspace and the output
        /// eigenvalues buffer from <paramref name="arena"/> and calls the zero-alloc primitive.
        /// See the dense overload's doc comment for the return-value convention.
        /// </summary>
        public static fProxyN lanczos(ref Arena arena, in fProxyBSR A, int steps, out LanczosInfo info,
                                      fProxy breakdownTol)
        {
            var ws = arena.fProxyLanczosCache(A.M_Rows, steps);
            var eigenvalues = arena.fProxyVec(steps);
            info = lanczos(in A, ref ws, ref eigenvalues, steps, breakdownTol);
            return eigenvalues;
        }

        /// <summary>lanczos (allocating) over a BSR matrix with default breakdownTol (Consts.fProxyZeroThreshold).</summary>
        public static fProxyN lanczos(ref Arena arena, in fProxyBSR A, int steps, out LanczosInfo info)
            => lanczos(ref arena, in A, steps, out info, Consts.fProxyZeroThreshold);

        /// <summary>
        /// Lanczos with RITZ VECTORS: same twice-reorthogonalized tridiagonalization as
        /// <see cref="lanczos{TOp}"/> (via the shared <c>lanczosTridiag</c>), but also returns
        /// approximate EIGENVECTORS. After building the Krylov basis ws.V and tridiagonal ws.T, it
        /// eigendecomposes T with <see cref="eigenSymmetric(ref fProxyMxN, ref fProxyN, ref fProxyMxN)"/>
        /// (eigenvalues DESCENDING into <paramref name="eigenvalues"/>, T's eigenvectors into the
        /// COLUMNS of <paramref name="Yt"/>), then forms each Ritz vector
        /// ritz[i] = sum_j Yt[j,i]·v_(j+1) -- the i-th approximate eigenvector of A, stored as ROW i
        /// of <paramref name="ritz"/>.
        ///
        /// Only the first <c>info.produced</c> rows of ritz / entries of eigenvalues are
        /// meaningful (lanczos's early-breakdown/padding convention): a real Ritz vector has zero
        /// weight on the decoupled padded coordinates, so summing j &lt; produced is exact. The Ritz
        /// vectors are orthonormal to working precision (Yt orthonormal, ws.V rows orthonormal). At
        /// FULL steps == n they reproduce A's eigenvectors; for partial steps &lt; n only the EXTREMAL
        /// ones have converged (Kaniel-Paige-Saad), matching the Ritz values.
        ///
        /// <paramref name="Yt"/> is steps x steps scratch (T's eigenvectors); <paramref name="ritz"/>
        /// is steps x A.Rows output. NOTE: unlike <see cref="lanczos{TOp}"/> this is NOT zero-alloc --
        /// <see cref="eigenSymmetric(ref fProxyMxN, ref fProxyN, ref fProxyMxN)"/> allocates three
        /// length-steps Temp vectors internally. Returns a <see cref="LanczosInfo"/> (produced,
        /// status = eigenSymmetric's convergence flag).
        /// </summary>
        public static LanczosInfo lanczosVectors<TOp>(in TOp A, ref fProxyLanczosCache ws, ref fProxyMxN Yt,
                                               ref fProxyN eigenvalues, ref fProxyMxN ritz,
                                               int steps, fProxy breakdownTol)
            where TOp : struct, IfProxyLinearOperator
        {
            if (eigenvalues.N != steps)
                throw new ArgumentException("lanczosVectors: eigenvalues.N must equal steps");
            if (!Yt.IsSquare || Yt.M_Rows != steps)
                throw new ArgumentException("lanczosVectors: Yt must be steps x steps");
            if (ritz.M_Rows != steps || ritz.N_Cols != A.Rows)
                throw new ArgumentException("lanczosVectors: ritz must be steps x A.Rows");

            int n = A.Rows;

            lanczosTridiag(in A, ref ws, out int produced, steps, breakdownTol);

            bool ok = eigenSymmetric(ref ws.T, ref eigenvalues, ref Yt);

            // Ritz vector i (row i of ritz) = sum_{j < produced} Yt[j,i] * v_(j+1) (row j of ws.V).
            for (int i = 0; i < produced; i++)
                for (int col = 0; col < n; col++)
                {
                    fProxy acc = (fProxy)0;
                    for (int j = 0; j < produced; j++)
                        acc += Yt[j, i] * ws.V[j, col];
                    ritz[i, col] = acc;
                }

            // Rows [produced, steps) belong to the decoupled/padded eigenpairs and carry no
            // meaning. Zero them so an accidental read of ritz[i >= produced] fails LOUD (a zero
            // vector trips any unit-norm / residual check) instead of returning plausible arena
            // garbage that happens to look like an eigenvector.
            for (int i = produced; i < steps; i++)
                for (int col = 0; col < n; col++)
                    ritz[i, col] = (fProxy)0;

            return new LanczosInfo { produced = produced, status = ok ? IterativeSolveStatus.Converged : IterativeSolveStatus.MaxIterations };
        }

        /// <summary>lanczosVectors with default breakdownTol (Consts.fProxyZeroThreshold).</summary>
        public static LanczosInfo lanczosVectors<TOp>(in TOp A, ref fProxyLanczosCache ws, ref fProxyMxN Yt,
                                               ref fProxyN eigenvalues, ref fProxyMxN ritz,
                                               int steps)
            where TOp : struct, IfProxyLinearOperator
            => lanczosVectors(in A, ref ws, ref Yt, ref eigenvalues, ref ritz, steps, Consts.fProxyZeroThreshold);

        /// <summary>
        /// Lanczos with Ritz vectors over a dense SYMMETRIC matrix -- allocates the workspace, T's
        /// eigenvector scratch, the eigenvalues buffer (length steps) and the Ritz-vector matrix
        /// (steps x A.Rows) from <paramref name="arena"/>. Returns the eigenvalues; Ritz vectors are
        /// written to <paramref name="ritz"/> (row i = approximate eigenvector i). Only the first
        /// <c>info.produced</c> entries/rows are meaningful. See <see cref="lanczosVectors{TOp}"/>.
        /// </summary>
        public static fProxyN lanczosVectors(ref Arena arena, in fProxyMxN A, int steps,
                                             out fProxyMxN ritz, out LanczosInfo info,
                                             fProxy breakdownTol)
        {
            var ws = arena.fProxyLanczosCache(A.M_Rows, steps);
            var Yt = arena.fProxyMat(steps, steps);
            var eigenvalues = arena.fProxyVec(steps);
            ritz = arena.fProxyMat(steps, A.M_Rows);
            info = lanczosVectors(new fProxyDenseOperator(in A), ref ws, ref Yt, ref eigenvalues, ref ritz, steps, breakdownTol);
            return eigenvalues;
        }

        /// <summary>lanczosVectors (allocating) over a dense matrix with default breakdownTol.</summary>
        public static fProxyN lanczosVectors(ref Arena arena, in fProxyMxN A, int steps, out fProxyMxN ritz, out LanczosInfo info)
            => lanczosVectors(ref arena, in A, steps, out ritz, out info, Consts.fProxyZeroThreshold);

        /// <summary>
        /// Lanczos with Ritz vectors over a BSR SYMMETRIC matrix -- allocates workspace/scratch and
        /// the eigenvalues + Ritz-vector outputs from <paramref name="arena"/>. See the dense overload.
        /// </summary>
        public static fProxyN lanczosVectors(ref Arena arena, in fProxyBSR A, int steps,
                                             out fProxyMxN ritz, out LanczosInfo info,
                                             fProxy breakdownTol)
        {
            var ws = arena.fProxyLanczosCache(A.M_Rows, steps);
            var Yt = arena.fProxyMat(steps, steps);
            var eigenvalues = arena.fProxyVec(steps);
            ritz = arena.fProxyMat(steps, A.M_Rows);
            info = lanczosVectors(new fProxyBSROperator(in A), ref ws, ref Yt, ref eigenvalues, ref ritz, steps, breakdownTol);
            return eigenvalues;
        }

        /// <summary>lanczosVectors (allocating) over a BSR matrix with default breakdownTol.</summary>
        public static fProxyN lanczosVectors(ref Arena arena, in fProxyBSR A, int steps, out fProxyMxN ritz, out LanczosInfo info)
            => lanczosVectors(ref arena, in A, steps, out ritz, out info, Consts.fProxyZeroThreshold);

        /// <summary>
        /// Full symmetric eigendecomposition via classical two-sided (cyclic) Jacobi iteration.
        /// Computes A = V * diag(eigenvalues) * V^T where V is orthonormal.
        ///
        /// On input: A must be square and symmetric. On output: A is DESTROYED (driven to
        /// approximately diagonal); eigenvalues (length n) holds the eigenvalues;
        /// V (n x n) holds the eigenvectors as columns (V is overwritten and initialized
        /// to the identity internally).
        ///
        /// Eigenvalues are sorted in DESCENDING ORDER BY VALUE (not magnitude), so
        /// lambda[0] &gt;= lambda[1] &gt;= ... &gt;= lambda[n-1]. This means negative eigenvalues
        /// appear last. The corresponding eigenvector columns of V are reordered to match.
        ///
        /// Returns true if convergence was reached within maxSweeps (a sweep with zero
        /// Jacobi rotations), false if the sweep limit was exhausted.
        ///
        /// Notes:
        ///   - Works for any real symmetric matrix including indefinite ones; eigenvalues
        ///     are always real.
        ///   - For positive semi-definite matrices the result matches SVD up to column
        ///     sign differences.
        ///   - Does not allocate.
        /// </summary>
        [System.Obsolete("Prefer Eigen.eigenSymmetric (Householder tridiagonal + QL, ~30x faster) for symmetric eigenpairs, or Eigen.eigenvaluesSymmetric for eigenvalues only. This cyclic-Jacobi solver is retained for reference.", false)]
        public static bool eigenDecomposition(ref fProxyMxN A, ref fProxyN eigenvalues,
                                              ref fProxyMxN V, int maxSweeps, fProxy eps)
        {
            if (!A.IsSquare)
                throw new ArgumentException("Eigen.eigenDecomposition: A must be square");

            int n = A.N_Cols;

            if (eigenvalues.N != n)
                throw new ArgumentException("Eigen.eigenDecomposition: eigenvalues.N must equal A dimension");

            if (!V.IsSquare || V.M_Rows != n)
                throw new ArgumentException("Eigen.eigenDecomposition: V must be square with side equal to A dimension");

            if (maxSweeps < 1)
                throw new ArgumentException("Eigen.eigenDecomposition: maxSweeps must be >= 1");

            if (eps <= (fProxy)0)
                throw new ArgumentException("Eigen.eigenDecomposition: eps must be > 0");

            // Symmetry guard: check that A is symmetric within eps-relative tolerance
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    fProxy aij = A[i, j];
                    fProxy aji = A[j, i];
                    fProxy diff = math.abs(aij - aji);
                    fProxy relScale = (fProxy)1 + math.abs(aij) + math.abs(aji);
                    if (diff > eps * relScale)
                        throw new ArgumentException("Eigen.eigenDecomposition: Matrix must be symmetric");
                }
            }

            if (n == 0)
                return true;

            // Initialize V to identity
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    V[i, j] = (i == j) ? (fProxy)1 : (fProxy)0;

            bool converged = false;

            for (int sweep = 0; sweep < maxSweeps; sweep++) {

                int rotations = 0;

                for (int p = 0; p < n - 1; p++) {
                    for (int q = p + 1; q < n; q++) {

                        fProxy apq = A[p, q];

                        // Skip exact zeros
                        if (apq == (fProxy)0)
                            continue;

                        // Skip when off-diagonal is negligible relative to the diagonal
                        if (math.abs(apq) <= eps * (fProxy)0.5 * (math.abs(A[p, p]) + math.abs(A[q, q])))
                            continue;

                        // Compute rotation angle: theta = (A[q,q] - A[p,p]) / (2 * A[p,q])
                        fProxy theta = (A[q, q] - A[p, p]) / ((fProxy)2 * apq);

                        // sign(theta) with 0 -> +1
                        fProxy signTheta = theta >= (fProxy)0 ? (fProxy)1 : (fProxy)(-1);
                        fProxy absTheta = math.abs(theta);

                        fProxy t;
                        if (absTheta > (fProxy)1) {
                            // Factor out |theta| to avoid theta*theta overflow
                            fProxy inv = (fProxy)1 / theta;
                            t = signTheta / (absTheta * ((fProxy)1 + math.sqrt((fProxy)1 + inv * inv)));
                        } else {
                            // |theta| <= 1 -> theta*theta <= 1, safe
                            t = signTheta / (absTheta + math.sqrt((fProxy)1 + theta * theta));
                        }

                        fProxy c = (fProxy)1 / math.sqrt((fProxy)1 + t * t);
                        fProxy s = t * c;

                        // Apply symmetric rotation to A
                        fProxy app = A[p, p];
                        fProxy aqq = A[q, q];
                        A[p, p] = app - t * apq;
                        A[q, q] = aqq + t * apq;
                        A[p, q] = (fProxy)0;
                        A[q, p] = (fProxy)0;

                        for (int i = 0; i < n; i++) {
                            if (i == p || i == q)
                                continue;
                            fProxy aip = A[i, p];
                            fProxy aiq = A[i, q];
                            fProxy newAip = c * aip - s * aiq;
                            fProxy newAiq = s * aip + c * aiq;
                            A[i, p] = newAip;
                            A[p, i] = newAip;
                            A[i, q] = newAiq;
                            A[q, i] = newAiq;
                        }

                        // Rotate columns p and q of V
                        for (int i = 0; i < n; i++) {
                            fProxy vip = V[i, p];
                            fProxy viq = V[i, q];
                            V[i, p] = c * vip - s * viq;
                            V[i, q] = s * vip + c * viq;
                        }

                        rotations++;
                    }
                }

                if (rotations == 0) {
                    converged = true;
                    break;
                }
            }

            // Extract diagonal of (now approximately diagonal) A into eigenvalues
            for (int i = 0; i < n; i++)
                eigenvalues[i] = A[i, i];

            // Selection sort: descending by value (not magnitude)
            for (int j = 0; j < n; j++) {
                int maxIdx = j;
                fProxy maxVal = eigenvalues[j];

                for (int k = j + 1; k < n; k++) {
                    if (eigenvalues[k] > maxVal) {
                        maxIdx = k;
                        maxVal = eigenvalues[k];
                    }
                }

                if (maxIdx != j) {
                    // Swap eigenvalues
                    fProxy tmp = eigenvalues[j];
                    eigenvalues[j] = eigenvalues[maxIdx];
                    eigenvalues[maxIdx] = tmp;

                    // Swap corresponding columns of V only (A's diagonal traveled into eigenvalues)
                    Swap_OP.Columns(ref V, j, maxIdx);
                }
            }

            return converged;
        }

        // The default-argument overloads forward to the deprecated primitive; suppress the
        // self-referential obsolete warning (618) on the forwarding calls.
#pragma warning disable 618
        /// <summary>eigenDecomposition with default eps (Consts.fProxyZeroThreshold).</summary>
        [System.Obsolete("Prefer Eigen.eigenSymmetric (Householder tridiagonal + QL, ~30x faster) for symmetric eigenpairs, or Eigen.eigenvaluesSymmetric for eigenvalues only. This cyclic-Jacobi solver is retained for reference.", false)]
        public static bool eigenDecomposition(ref fProxyMxN A, ref fProxyN eigenvalues,
                                              ref fProxyMxN V, int maxSweeps)
            => eigenDecomposition(ref A, ref eigenvalues, ref V, maxSweeps, Consts.fProxyZeroThreshold);

        /// <summary>eigenDecomposition with default maxSweeps (30) and eps (Consts.fProxyZeroThreshold).</summary>
        [System.Obsolete("Prefer Eigen.eigenSymmetric (Householder tridiagonal + QL, ~30x faster) for symmetric eigenpairs, or Eigen.eigenvaluesSymmetric for eigenvalues only. This cyclic-Jacobi solver is retained for reference.", false)]
        public static bool eigenDecomposition(ref fProxyMxN A, ref fProxyN eigenvalues,
                                              ref fProxyMxN V)
            => eigenDecomposition(ref A, ref eigenvalues, ref V, 30, Consts.fProxyZeroThreshold);
#pragma warning restore 618

        // copysign: magnitude of a with the sign of b (b >= 0 -> +|a|). EISPACK SIGN(a,b).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static fProxy copysign(fProxy a, fProxy b) => b >= (fProxy)0 ? math.abs(a) : -math.abs(a);

        // sqrt(a^2 + b^2) computed so neither square overflows/underflows prematurely.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static fProxy pythag(fProxy a, fProxy b)
        {
            fProxy aa = math.abs(a), ab = math.abs(b);
            if (aa > ab) { fProxy r = ab / aa; return aa * math.sqrt((fProxy)1 + r * r); }
            if (ab == (fProxy)0) return (fProxy)0;
            { fProxy r = aa / ab; return ab * math.sqrt((fProxy)1 + r * r); }
        }

        /// <summary>
        /// All eigenVALUES of a SYMMETRIC real matrix, via Householder tridiagonalization followed by
        /// the implicit-shift QL iteration (EISPACK tred1 + tql1, GVL Alg. 8.3.1). Much faster than the
        /// cyclic-Jacobi eigenDecomposition: the O(n^3) reduction is a sequence of gemv + symmetric
        /// rank-2 updates (the rank-2 update is axpy → vectorises), and the QL sweep that follows is
        /// only O(n^2). No eigenvectors (use eigenDecomposition if you need them).
        ///
        /// A must be symmetric (checked within eps-relative tolerance) and is DESTROYED. On output
        /// eigenvalues[i] holds the i-th eigenvalue, sorted DESCENDING. Returns true on convergence;
        /// false if QL hit maxIterPerEig for some eigenvalue (outputs then undefined). Does not allocate
        /// beyond three length-n Temp scratch vectors.
        /// </summary>
        public static bool eigenvaluesSymmetric(ref fProxyMxN A, ref fProxyN eigenvalues, int maxIterPerEig, fProxy eps,
                                                 ref fProxyEigenSymCache ws)
        {
            if (!A.IsSquare)
                throw new ArgumentException("Eigen.eigenvaluesSymmetric: A must be square");

            int n = A.M_Rows;

            if (eigenvalues.N != n)
                throw new ArgumentException("Eigen.eigenvaluesSymmetric: eigenvalues.N must equal A dimension");

            if (maxIterPerEig < 1)
                throw new ArgumentException("Eigen.eigenvaluesSymmetric: maxIterPerEig must be >= 1");

            if (eps <= (fProxy)0)
                throw new ArgumentException("Eigen.eigenvaluesSymmetric: eps must be > 0");

            // Symmetry guard (same as eigenDecomposition). The reduction reads the full symmetric
            // matrix (the gemv uses whole rows), so both triangles must agree.
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    fProxy aij = A[i, j], aji = A[j, i];
                    fProxy diff = math.abs(aij - aji);
                    fProxy relScale = (fProxy)1 + math.abs(aij) + math.abs(aji);
                    if (diff > eps * relScale)
                        throw new ArgumentException("Eigen.eigenvaluesSymmetric: Matrix must be symmetric");
                }

            RequireEigenSymWorkspace(in ws, n);

            if (n == 0) return true;
            if (n == 1) { eigenvalues[0] = A[0, 0]; return true; }

            var eVec = ws.eVec;   // off-diagonal e[i] couples d[i], d[i+1]
            var vVec = ws.vVec;   // Householder vector (entries m0..n-1)
            var pVec = ws.pVec;   // p = beta*A*v, then q = p - K v

            unsafe
            {
                fProxy* ap = A.Data.Ptr;
                fProxy* v  = vVec.Data.Ptr;
                fProxy* p  = pVec.Data.Ptr;

                // Matrix scale (max |entry|) for the column-deflation test in the reduction below.
                fProxy matScale = (fProxy)0;
                for (long ii = 0; ii < (long)n * n; ii++)
                {
                    fProxy a = math.abs(ap[ii]);
                    if (a > matScale) matScale = a;
                }
                fProxy belowNormTol = (fProxy)n * Consts.fProxyEpsilon * matScale;

                // ---- Householder tridiagonalization (full symmetric storage, values only) ----
                // The trailing submatrix stays symmetric; column k below the subdiagonal is never read
                // again, so (values-only) we record the subdiagonal in e[k] and skip zeroing it.
                for (int k = 0; k < n - 2; k++)
                {
                    int m0 = k + 1;

                    // x = A[m0.., k]; sigma = ||x[1..]||^2 (entries strictly below the leading one).
                    fProxy sigma = 0;
                    for (int i = m0 + 1; i < n; i++)
                    {
                        fProxy aik = ap[(long)i * n + k];
                        sigma += aik * aik;
                    }
                    fProxy x0 = ap[(long)m0 * n + k];

                    // Deflate a column whose below-subdiagonal norm is negligible vs the matrix scale.
                    // Exact (sigma == 0) is not enough: for rank-deficient/structured matrices sigma
                    // shrinks to denormal (nonzero), vtv underflows and beta = 2/vtv OVERFLOWS to Inf,
                    // and the rank-2 update then forms Inf - Inf = NaN. Deflate cleanly before that.
                    if (math.sqrt(sigma) <= belowNormTol)
                    {
                        // column already (effectively) in tridiagonal form
                        eVec[k] = x0;
                        continue;
                    }

                    fProxy xnorm = math.sqrt(x0 * x0 + sigma);
                    fProxy alpha = (x0 >= (fProxy)0) ? -xnorm : xnorm;   // -sign(x0)*||x||

                    // Householder vector v (entries m0..n-1): v[m0] = x0 - alpha, v[i>m0] = x[i].
                    v[m0] = x0 - alpha;
                    for (int i = m0 + 1; i < n; i++) v[i] = ap[(long)i * n + k];

                    fProxy vtv  = v[m0] * v[m0] + sigma;
                    fProxy beta = (fProxy)2 / vtv;

                    // p = beta * A_sub * v   (A_sub = A[m0:n, m0:n], symmetric). Row dots (contiguous).
                    for (int r = m0; r < n; r++)
                    {
                        fProxy* arow = ap + (long)r * n;
                        fProxy s = 0;
                        for (int c = m0; c < n; c++) s += arow[c] * v[c];
                        p[r] = beta * s;
                    }

                    // K = beta * (vᵀp) / 2;  q = p - K v   (overwrite p with q)
                    fProxy vp = 0;
                    for (int i = m0; i < n; i++) vp += v[i] * p[i];
                    fProxy K = beta * vp / (fProxy)2;
                    for (int i = m0; i < n; i++) p[i] -= K * v[i];

                    // Symmetric rank-2 update: A_sub -= v qᵀ + q vᵀ  (two contiguous axpys per row).
                    int len = n - m0;
                    for (int r = m0; r < n; r++)
                    {
                        fProxy* arow = ap + (long)r * n;
                        Unsafe_OP.axpy(arow + m0, p + m0, -v[r], len);   // -= v[r] * q
                        Unsafe_OP.axpy(arow + m0, v + m0, -p[r], len);   // -= q[r] * v
                    }

                    eVec[k] = alpha;
                }

                // trailing subdiagonal + diagonal
                eVec[n - 2] = ap[(long)(n - 1) * n + (n - 2)];
                eVec[n - 1] = (fProxy)0;
                for (int i = 0; i < n; i++) eigenvalues[i] = ap[(long)i * n + i];
            }

            // Global tridiagonal scale. The deflation test below is floored by this so a cluster of
            // ZERO eigenvalues can still deflate: there the local |d[m]|+|d[m+1]| collapses to ~0, but
            // the sub-diagonal noise floor is set by the GLOBAL scale, so a purely local threshold
            // never triggers in float and QL spins to maxIter (the rank-deficient svdValues case).
            fProxy anorm = math.abs(eigenvalues[0]) + math.abs(eVec[0]);
            for (int i = 1; i < n; i++)
            {
                fProxy rowSum = math.abs(eVec[i - 1]) + math.abs(eigenvalues[i]) + math.abs(eVec[i]);
                if (rowSum > anorm) anorm = rowSum;
            }

            // ---- implicit-shift QL on the tridiagonal (d = eigenvalues, e), values only ----
            // e[i] couples d[i] and d[i+1]; e[n-1] = 0.
            for (int l = 0; l < n; l++)
            {
                int iter = 0;
                int m;
                do
                {
                    for (m = l; m < n - 1; m++)
                    {
                        fProxy dd = math.abs(eigenvalues[m]) + math.abs(eigenvalues[m + 1]);
                        // machine-eps relative, floored by the global scale `anorm` (see above)
                        if (math.abs(eVec[m]) <= (fProxy)8 * Consts.fProxyEpsilon * (dd + anorm)) break;
                    }
                    if (m != l)
                    {
                        if (iter++ >= maxIterPerEig) { return false; }

                        fProxy g = (eigenvalues[l + 1] - eigenvalues[l]) / ((fProxy)2 * eVec[l]);
                        fProxy r = pythag(g, (fProxy)1);
                        g = eigenvalues[m] - eigenvalues[l] + eVec[l] / (g + copysign(r, g));
                        fProxy s = 1, c = 1, pp = 0;
                        int i;
                        for (i = m - 1; i >= l; i--)
                        {
                            fProxy f = s * eVec[i];
                            fProxy b = c * eVec[i];
                            r = pythag(f, g);
                            eVec[i + 1] = r;
                            if (r == (fProxy)0) { eigenvalues[i + 1] -= pp; eVec[m] = 0; break; }
                            s = f / r; c = g / r;
                            g = eigenvalues[i + 1] - pp;
                            r = (eigenvalues[i] - g) * s + (fProxy)2 * c * b;
                            pp = s * r;
                            eigenvalues[i + 1] = g + pp;
                            g = c * r - b;
                        }
                        if (r == (fProxy)0 && i >= l) continue;
                        eigenvalues[l] -= pp; eVec[l] = g; eVec[m] = 0;
                    }
                } while (m != l);
            }

            // sort descending (selection sort, matching eigenDecomposition)
            for (int j = 0; j < n; j++)
            {
                int maxIdx = j;
                fProxy maxVal = eigenvalues[j];
                for (int k = j + 1; k < n; k++)
                    if (eigenvalues[k] > maxVal) { maxIdx = k; maxVal = eigenvalues[k]; }
                if (maxIdx != j)
                {
                    fProxy tmp = eigenvalues[j];
                    eigenvalues[j] = eigenvalues[maxIdx];
                    eigenvalues[maxIdx] = tmp;
                }
            }

            return true;
        }

        /// <summary>eigenvaluesSymmetric (ref workspace) with default maxIterPerEig (30) and eps (Consts.fProxyZeroThreshold).</summary>
        public static bool eigenvaluesSymmetric(ref fProxyMxN A, ref fProxyN eigenvalues, ref fProxyEigenSymCache ws)
            => eigenvaluesSymmetric(ref A, ref eigenvalues, 30, Consts.fProxyZeroThreshold, ref ws);

        /// <summary>
        /// eigenvaluesSymmetric allocating its tridiagonalization scratch (three length-n vectors) from
        /// Allocator.Temp. See the ref-workspace overload for semantics. A is overwritten (destroyed).
        /// </summary>
        public static bool eigenvaluesSymmetric(ref fProxyMxN A, ref fProxyN eigenvalues, int maxIterPerEig, fProxy eps)
        {
            int n = A.M_Rows;
            var ws = new fProxyEigenSymCache
            {
                eVec = new fProxyN(n, Allocator.Temp, false),
                vVec = new fProxyN(n, Allocator.Temp, false),
                pVec = new fProxyN(n, Allocator.Temp, false)
            };
            bool ok = eigenvaluesSymmetric(ref A, ref eigenvalues, maxIterPerEig, eps, ref ws);
            ws.eVec.Dispose();
            ws.vVec.Dispose();
            ws.pVec.Dispose();
            return ok;
        }

        /// <summary>eigenvaluesSymmetric with default maxIterPerEig (30) and eps (Consts.fProxyZeroThreshold).</summary>
        public static bool eigenvaluesSymmetric(ref fProxyMxN A, ref fProxyN eigenvalues)
            => eigenvaluesSymmetric(ref A, ref eigenvalues, 30, Consts.fProxyZeroThreshold);

        /// <summary>
        /// Full eigenDECOMPOSITION of a SYMMETRIC real matrix via Householder tridiagonalization with
        /// orthogonal accumulation (tred2) + implicit-shift QL with eigenvector accumulation (tql2).
        /// Same result as the cyclic-Jacobi eigenDecomposition but far faster: the O(n^3)
        /// tridiagonalization is gemv + rank-2 axpy updates (vectorises) and runs ONCE, where Jacobi
        /// does several full sweeps of strided column rotations.
        ///
        /// A must be symmetric (checked within eps) and is DESTROYED. On output eigenvalues[i] is the
        /// i-th eigenvalue (sorted DESCENDING) and column i of V is its unit eigenvector, so
        /// A = V * diag(eigenvalues) * Vᵀ and VᵀV = I. Returns true on convergence; false if QL hit
        /// maxIterPerEig (outputs then undefined). Allocates three length-n Temp scratch vectors.
        /// </summary>
        public static bool eigenSymmetric(ref fProxyMxN A, ref fProxyN eigenvalues, ref fProxyMxN V,
                                          int maxIterPerEig, fProxy eps)
        {
            if (!A.IsSquare)
                throw new ArgumentException("Eigen.eigenSymmetric: A must be square");

            int n = A.M_Rows;

            if (eigenvalues.N != n)
                throw new ArgumentException("Eigen.eigenSymmetric: eigenvalues.N must equal A dimension");

            if (!V.IsSquare || V.M_Rows != n)
                throw new ArgumentException("Eigen.eigenSymmetric: V must be square with side equal to A dimension");

            if (maxIterPerEig < 1)
                throw new ArgumentException("Eigen.eigenSymmetric: maxIterPerEig must be >= 1");

            if (eps <= (fProxy)0)
                throw new ArgumentException("Eigen.eigenSymmetric: eps must be > 0");

            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    fProxy aij = A[i, j], aji = A[j, i];
                    fProxy diff = math.abs(aij - aji);
                    fProxy relScale = (fProxy)1 + math.abs(aij) + math.abs(aji);
                    if (diff > eps * relScale)
                        throw new ArgumentException("Eigen.eigenSymmetric: Matrix must be symmetric");
                }

            if (n == 0) return true;

            // V starts as identity (it accumulates Q = H_0 H_1 ... then the QL rotations).
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    V[i, j] = (i == j) ? (fProxy)1 : (fProxy)0;

            if (n == 1) { eigenvalues[0] = A[0, 0]; return true; }

            var eVec = new fProxyN(n, Allocator.Temp, false);
            var vVec = new fProxyN(n, Allocator.Temp, false);
            var pVec = new fProxyN(n, Allocator.Temp, false);

            unsafe
            {
                fProxy* ap = A.Data.Ptr;
                fProxy* qp = V.Data.Ptr;
                fProxy* v  = vVec.Data.Ptr;
                fProxy* p  = pVec.Data.Ptr;

                // Matrix scale for the deflation test — see eigenvaluesSymmetric.
                fProxy matScale = (fProxy)0;
                for (long ii = 0; ii < (long)n * n; ii++)
                {
                    fProxy a = math.abs(ap[ii]);
                    if (a > matScale) matScale = a;
                }
                fProxy belowNormTol = (fProxy)n * Consts.fProxyEpsilon * matScale;

                // ---- Householder tridiagonalization with Q accumulation into V ----
                for (int k = 0; k < n - 2; k++)
                {
                    int m0 = k + 1;

                    fProxy sigma = 0;
                    for (int i = m0 + 1; i < n; i++)
                    {
                        fProxy aik = ap[(long)i * n + k];
                        sigma += aik * aik;
                    }
                    fProxy x0 = ap[(long)m0 * n + k];

                    // See eigenvaluesSymmetric: deflate near-negligible columns before vtv underflows
                    // and beta = 2/vtv overflows to Inf (which would make the rank-2 update form NaN).
                    if (math.sqrt(sigma) <= belowNormTol)
                    {
                        eVec[k] = x0;
                        continue;
                    }

                    fProxy xnorm = math.sqrt(x0 * x0 + sigma);
                    fProxy alpha = (x0 >= (fProxy)0) ? -xnorm : xnorm;

                    v[m0] = x0 - alpha;
                    for (int i = m0 + 1; i < n; i++) v[i] = ap[(long)i * n + k];

                    fProxy vtv  = v[m0] * v[m0] + sigma;
                    fProxy beta = (fProxy)2 / vtv;

                    for (int r = m0; r < n; r++)
                    {
                        fProxy* arow = ap + (long)r * n;
                        fProxy s = 0;
                        for (int c = m0; c < n; c++) s += arow[c] * v[c];
                        p[r] = beta * s;
                    }

                    fProxy vp = 0;
                    for (int i = m0; i < n; i++) vp += v[i] * p[i];
                    fProxy K = beta * vp / (fProxy)2;
                    for (int i = m0; i < n; i++) p[i] -= K * v[i];

                    int len = n - m0;
                    for (int r = m0; r < n; r++)
                    {
                        fProxy* arow = ap + (long)r * n;
                        Unsafe_OP.axpy(arow + m0, p + m0, -v[r], len);
                        Unsafe_OP.axpy(arow + m0, v + m0, -p[r], len);
                    }

                    // Accumulate Q: V := V * H_k  (H_k = I - beta v vᵀ on columns [m0,n)).
                    // For each row r: V[r, m0:] -= beta*(V[r,m0:]·v) * v.
                    for (int r = 0; r < n; r++)
                    {
                        fProxy* qrow = qp + (long)r * n;
                        fProxy s = 0;
                        for (int c = m0; c < n; c++) s += qrow[c] * v[c];
                        Unsafe_OP.axpy(qrow + m0, v + m0, -(beta * s), len);
                    }

                    eVec[k] = alpha;
                }

                eVec[n - 2] = ap[(long)(n - 1) * n + (n - 2)];
                eVec[n - 1] = (fProxy)0;
                for (int i = 0; i < n; i++) eigenvalues[i] = ap[(long)i * n + i];

                // Global tridiagonal scale (see eigenvaluesSymmetric): floors the deflation threshold
                // so clustered zero eigenvalues still deflate instead of spinning QL to maxIter.
                fProxy anorm = math.abs(eigenvalues[0]) + math.abs(eVec[0]);
                for (int i = 1; i < n; i++)
                {
                    fProxy rowSum = math.abs(eVec[i - 1]) + math.abs(eigenvalues[i]) + math.abs(eVec[i]);
                    if (rowSum > anorm) anorm = rowSum;
                }

                // Transpose Q in place so the QL plane rotations below hit CONTIGUOUS rows (unit
                // stride → vectorizes) instead of strided columns. Transposed back after the sweep.
                for (int ti = 0; ti < n; ti++)
                    for (int tj = ti + 1; tj < n; tj++)
                    {
                        fProxy* pa = qp + (long)ti * n + tj;
                        fProxy* pb = qp + (long)tj * n + ti;
                        fProxy t = *pa; *pa = *pb; *pb = t;
                    }

                // ---- implicit-shift QL with eigenvector accumulation (tql2) ----
                for (int l = 0; l < n; l++)
                {
                    int iter = 0;
                    int m;
                    do
                    {
                        for (m = l; m < n - 1; m++)
                        {
                            fProxy dd = math.abs(eigenvalues[m]) + math.abs(eigenvalues[m + 1]);
                            // machine-eps relative, floored by anorm — see eigenvaluesSymmetric.
                            if (math.abs(eVec[m]) <= (fProxy)8 * Consts.fProxyEpsilon * (dd + anorm)) break;
                        }
                        if (m != l)
                        {
                            if (iter++ >= maxIterPerEig) { eVec.Dispose(); vVec.Dispose(); pVec.Dispose(); return false; }

                            fProxy g = (eigenvalues[l + 1] - eigenvalues[l]) / ((fProxy)2 * eVec[l]);
                            fProxy r = pythag(g, (fProxy)1);
                            g = eigenvalues[m] - eigenvalues[l] + eVec[l] / (g + copysign(r, g));
                            fProxy s = 1, c = 1, pp = 0;
                            int i;
                            for (i = m - 1; i >= l; i--)
                            {
                                fProxy f = s * eVec[i];
                                fProxy b = c * eVec[i];
                                r = pythag(f, g);
                                eVec[i + 1] = r;
                                if (r == (fProxy)0) { eigenvalues[i + 1] -= pp; eVec[m] = 0; break; }
                                s = f / r; c = g / r;
                                g = eigenvalues[i + 1] - pp;
                                r = (eigenvalues[i] - g) * s + (fProxy)2 * c * b;
                                pp = s * r;
                                eigenvalues[i + 1] = g + pp;
                                g = c * r - b;

                                // Apply the plane rotation to ROWS i, i+1 of the transposed eigenvector
                                // matrix — contiguous + [NoAlias] (distinct rows) so Burst vectorizes it.
                                Unsafe_OP.jacobiRotate(qp + (long)i * n, qp + (long)(i + 1) * n, c, s, n);
                            }
                            if (r == (fProxy)0 && i >= l) continue;
                            eigenvalues[l] -= pp; eVec[l] = g; eVec[m] = 0;
                        }
                    } while (m != l);
                }

                // Transpose Q back: rows → columns, so column i is eigenvector i again.
                for (int ti = 0; ti < n; ti++)
                    for (int tj = ti + 1; tj < n; tj++)
                    {
                        fProxy* pa = qp + (long)ti * n + tj;
                        fProxy* pb = qp + (long)tj * n + ti;
                        fProxy t = *pa; *pa = *pb; *pb = t;
                    }
            }

            eVec.Dispose();
            vVec.Dispose();
            pVec.Dispose();

            // sort descending by eigenvalue, carrying eigenvector columns along
            for (int j = 0; j < n; j++)
            {
                int maxIdx = j;
                fProxy maxVal = eigenvalues[j];
                for (int k = j + 1; k < n; k++)
                    if (eigenvalues[k] > maxVal) { maxIdx = k; maxVal = eigenvalues[k]; }
                if (maxIdx != j)
                {
                    fProxy tmp = eigenvalues[j];
                    eigenvalues[j] = eigenvalues[maxIdx];
                    eigenvalues[maxIdx] = tmp;
                    Swap_OP.Columns(ref V, j, maxIdx);
                }
            }

            return true;
        }

        /// <summary>eigenSymmetric with default maxIterPerEig (30) and eps (Consts.fProxyZeroThreshold).</summary>
        public static bool eigenSymmetric(ref fProxyMxN A, ref fProxyN eigenvalues, ref fProxyMxN V)
            => eigenSymmetric(ref A, ref eigenvalues, ref V, 30, Consts.fProxyZeroThreshold);

        /// <summary>
        /// All eigenvalues of a GENERAL (non-symmetric) real square matrix, via the QR algorithm:
        /// reduction to upper Hessenberg form (elimination with partial pivoting) followed by the
        /// Francis double-shift QR iteration to the real Schur form (EISPACK elmhes + hqr). Real
        /// arithmetic only — complex-conjugate eigenvalue pairs are produced from the 2x2 Schur
        /// blocks, so NO complex number type is needed.
        ///
        /// Unlike eigenDecomposition (symmetric-only Jacobi) and powerIteration (dominant pair only),
        /// this handles arbitrary real matrices including those with complex eigenvalues (e.g.
        /// rotations). It returns eigenVALUES only (no eigenvectors).
        ///
        /// On input A must be square; A is DESTROYED (overwritten during reduction/iteration).
        /// On output eigenvaluesReal[i] / eigenvaluesImag[i] are the real and imaginary parts of the
        /// i-th eigenvalue. Results are sorted by (real, then imaginary) DESCENDING, so a conjugate
        /// pair a±bi appears as (a,+b) immediately before (a,-b). Read the outputs only when the
        /// method returns true.
        ///
        /// Returns true if every eigenvalue converged within maxIterPerRoot iterations; false if the
        /// iteration limit was hit (outputs then undefined). Does not allocate.
        /// </summary>
        public static unsafe bool eigenvaluesQR(ref fProxyMxN A, ref fProxyN eigenvaluesReal,
                                                ref fProxyN eigenvaluesImag, int maxIterPerRoot)
        {
            if (!A.IsSquare)
                throw new ArgumentException("Eigen.eigenvaluesQR: A must be square");

            int n = A.N_Cols;
            fProxy* ap = A.Data.Ptr;   // row r starts at ap + (long)r * n (square: stride = n)

            if (eigenvaluesReal.N != n)
                throw new ArgumentException("Eigen.eigenvaluesQR: eigenvaluesReal.N must equal A dimension");

            if (eigenvaluesImag.N != n)
                throw new ArgumentException("Eigen.eigenvaluesQR: eigenvaluesImag.N must equal A dimension");

            if (maxIterPerRoot < 1)
                throw new ArgumentException("Eigen.eigenvaluesQR: maxIterPerRoot must be >= 1");

            if (n == 0)
                return true;

            // ---- Step 1: reduce A to upper Hessenberg form (elmhes: Gaussian elimination with
            //      partial pivoting via similarity transforms; preserves eigenvalues). ----
            for (int m = 1; m < n - 1; m++)
            {
                // pivot: largest |A[j, m-1]| over rows j >= m.
                fProxy x = (fProxy)0;
                int piv = m;
                for (int j = m; j < n; j++)
                {
                    if (math.abs(A[j, m - 1]) > math.abs(x))
                    {
                        x = A[j, m - 1];
                        piv = j;
                    }
                }

                // interchange rows and columns piv <-> m (a similarity transform).
                if (piv != m)
                {
                    for (int j = m - 1; j < n; j++)
                    {
                        fProxy tmp = A[piv, j]; A[piv, j] = A[m, j]; A[m, j] = tmp;
                    }
                    for (int j = 0; j < n; j++)
                    {
                        fProxy tmp = A[j, piv]; A[j, piv] = A[j, m]; A[j, m] = tmp;
                    }
                }

                // eliminate below the subdiagonal in column m-1.
                if (x != (fProxy)0)
                {
                    for (int i = m + 1; i < n; i++)
                    {
                        fProxy y = A[i, m - 1];
                        if (y != (fProxy)0)
                        {
                            y /= x;
                            A[i, m - 1] = y;                          // store multiplier (cleared below)
                            // row update A[i, m:] -= y * A[m, m:] — unit-stride, vectorized.
                            Unsafe_OP.axpy(ap + (long)i * n + m, ap + (long)m * n + m, -y, n - m);
                            // column update A[:, m] += y * A[:, i] — column-strided, left scalar.
                            for (int j = 0; j < n; j++)
                                A[j, m] += y * A[j, i];
                        }
                    }
                }
            }

            // clear the stored multipliers below the subdiagonal -> clean upper Hessenberg H in A.
            for (int i = 2; i < n; i++)
                for (int j = 0; j < i - 1; j++)
                    A[i, j] = (fProxy)0;

            // ---- Step 2: Francis double-shift QR on the Hessenberg matrix (hqr). ----
            fProxy anorm = (fProxy)0;
            for (int i = 0; i < n; i++)
                for (int j = math.max(i - 1, 0); j < n; j++)
                    anorm += math.abs(A[i, j]);

            int nn = n - 1;     // index of the current bottom-right active row/col
            fProxy t = (fProxy)0;

            while (nn >= 0)
            {
                int its = 0;
                int l;
                do
                {
                    // look for a single negligible subdiagonal element to split off.
                    for (l = nn; l >= 1; l--)
                    {
                        fProxy s0 = math.abs(A[l - 1, l - 1]) + math.abs(A[l, l]);
                        if (s0 == (fProxy)0) s0 = anorm;
                        if (math.abs(A[l, l - 1]) + s0 == s0)
                        {
                            A[l, l - 1] = (fProxy)0;
                            break;
                        }
                    }
                    if (l < 0) l = 0;

                    fProxy x = A[nn, nn];

                    if (l == nn)
                    {
                        // one real root.
                        eigenvaluesReal[nn] = x + t;
                        eigenvaluesImag[nn] = (fProxy)0;
                        nn--;
                    }
                    else
                    {
                        fProxy y = A[nn - 1, nn - 1];
                        fProxy w = A[nn, nn - 1] * A[nn - 1, nn];

                        if (l == nn - 1)
                        {
                            // two roots from the trailing 2x2 block.
                            fProxy p = (fProxy)0.5 * (y - x);
                            fProxy q = p * p + w;
                            fProxy z = math.sqrt(math.abs(q));
                            x += t;
                            if (q >= (fProxy)0)
                            {
                                // real pair.
                                z = p + copysign(z, p);
                                eigenvaluesReal[nn - 1] = x + z;
                                eigenvaluesReal[nn] = (z != (fProxy)0) ? (x - w / z) : (x + z);
                                eigenvaluesImag[nn - 1] = (fProxy)0;
                                eigenvaluesImag[nn] = (fProxy)0;
                            }
                            else
                            {
                                // complex-conjugate pair a +/- bi.
                                eigenvaluesReal[nn - 1] = x + p;
                                eigenvaluesReal[nn] = x + p;
                                eigenvaluesImag[nn - 1] = z;
                                eigenvaluesImag[nn] = -z;
                            }
                            nn -= 2;
                        }
                        else
                        {
                            // no root yet: perform a double-shift QR sweep.
                            if (its >= maxIterPerRoot)
                                return false;   // not converged

                            if (its == 10 || its == 20)
                            {
                                // exceptional shift to break a cycle.
                                t += x;
                                for (int i = 0; i <= nn; i++)
                                    A[i, i] -= x;
                                fProxy s1 = math.abs(A[nn, nn - 1]) + math.abs(A[nn - 1, nn - 2]);
                                y = x = (fProxy)0.75 * s1;
                                w = (fProxy)(-0.4375) * s1 * s1;
                            }
                            its++;

                            // find two consecutive negligible subdiagonals to start the sweep.
                            fProxy p = (fProxy)0, q = (fProxy)0, r = (fProxy)0;
                            int m;
                            for (m = nn - 2; m >= l; m--)
                            {
                                fProxy z = A[m, m];
                                fProxy rr = x - z;
                                fProxy ss = y - z;
                                p = (rr * ss - w) / A[m + 1, m] + A[m, m + 1];
                                q = A[m + 1, m + 1] - z - rr - ss;
                                r = A[m + 2, m + 1];
                                fProxy s2 = math.abs(p) + math.abs(q) + math.abs(r);
                                // guard the normalization (matches the guarded analog in the QR sweep
                                // below): if p,q,r are all exactly zero, leave them zero rather than
                                // dividing 0/0 -> NaN, which would poison the convergence test.
                                if (s2 != (fProxy)0) { p /= s2; q /= s2; r /= s2; }
                                if (m == l) break;
                                fProxy u = math.abs(A[m, m - 1]) * (math.abs(q) + math.abs(r));
                                fProxy v = math.abs(p) * (math.abs(A[m - 1, m - 1]) + math.abs(z) + math.abs(A[m + 1, m + 1]));
                                if (u + v == v) break;
                            }

                            for (int i = m + 2; i <= nn; i++)
                            {
                                A[i, i - 2] = (fProxy)0;
                                if (i != m + 2) A[i, i - 3] = (fProxy)0;
                            }

                            // the double QR step over rows/cols m..nn.
                            for (int k = m; k <= nn - 1; k++)
                            {
                                if (k != m)
                                {
                                    p = A[k, k - 1];
                                    q = A[k + 1, k - 1];
                                    r = (fProxy)0;
                                    if (k != nn - 1) r = A[k + 2, k - 1];
                                    x = math.abs(p) + math.abs(q) + math.abs(r);
                                    if (x != (fProxy)0)
                                    {
                                        p /= x; q /= x; r /= x;
                                    }
                                }

                                fProxy s = copysign(math.sqrt(p * p + q * q + r * r), p);
                                if (s != (fProxy)0)
                                {
                                    if (k == m)
                                    {
                                        if (l != m)
                                            A[k, k - 1] = -A[k, k - 1];
                                    }
                                    else
                                    {
                                        A[k, k - 1] = -s * x;
                                    }
                                    p += s;
                                    fProxy xx = p / s;
                                    fProxy yy = q / s;
                                    fProxy zz = r / s;
                                    q /= p;
                                    r /= p;

                                    // row modification over columns j = k..nn (unit-stride). Rows
                                    // k, k+1, k+2 are distinct -> [NoAlias] Francis butterfly SIMDs it.
                                    int rowLen = nn - k + 1;
                                    if (k != nn - 1)
                                        Unsafe_OP.francisRow3(ap + (long)k * n + k, ap + (long)(k + 1) * n + k,
                                                             ap + (long)(k + 2) * n + k, q, r, xx, yy, zz, rowLen);
                                    else
                                        Unsafe_OP.francisRow2(ap + (long)k * n + k, ap + (long)(k + 1) * n + k,
                                                             q, xx, yy, rowLen);

                                    int mmin = nn < k + 3 ? nn : k + 3;
                                    // column modification.
                                    for (int i = l; i <= mmin; i++)
                                    {
                                        p = xx * A[i, k] + yy * A[i, k + 1];
                                        if (k != nn - 1)
                                        {
                                            p += zz * A[i, k + 2];
                                            A[i, k + 2] -= p * r;
                                        }
                                        A[i, k + 1] -= p * q;
                                        A[i, k] -= p;
                                    }
                                }
                            }
                        }
                    }
                } while (l < nn - 1);
            }

            // ---- sort by (real, then imaginary) descending; keep re/im paired. ----
            for (int a = 0; a < n - 1; a++)
            {
                int best = a;
                for (int b = a + 1; b < n; b++)
                {
                    if (eigenvaluesReal[b] > eigenvaluesReal[best] ||
                        (eigenvaluesReal[b] == eigenvaluesReal[best] && eigenvaluesImag[b] > eigenvaluesImag[best]))
                        best = b;
                }
                if (best != a)
                {
                    fProxy tr = eigenvaluesReal[a]; eigenvaluesReal[a] = eigenvaluesReal[best]; eigenvaluesReal[best] = tr;
                    fProxy ti = eigenvaluesImag[a]; eigenvaluesImag[a] = eigenvaluesImag[best]; eigenvaluesImag[best] = ti;
                }
            }

            return true;
        }

        /// <summary>eigenvaluesQR with default maxIterPerRoot (30, the EISPACK hqr limit).</summary>
        public static bool eigenvaluesQR(ref fProxyMxN A, ref fProxyN eigenvaluesReal,
                                         ref fProxyN eigenvaluesImag)
            => eigenvaluesQR(ref A, ref eigenvaluesReal, ref eigenvaluesImag, 30);
    }
}
