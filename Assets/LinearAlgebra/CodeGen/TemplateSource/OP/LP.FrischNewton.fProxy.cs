#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;
using System.Runtime.CompilerServices;

using Unity.Burst;
using Unity.Collections;
using Unity.Mathematics;

using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    public static partial class LP
    {
        // ============================================================================================
        // Frisch-Newton exact LAD / quantile-regression solver (Portnoy & Koenker 1997, "The Gaussian
        // Hare and the Laplacian Tortoise"). Port of Daniel Morillo & Roger Koenker's `rq_fnm` /
        // `lp_fnm`. A structure-exploiting primal-dual interior point on the LAD DUAL, working
        // directly on the original m x n design A -- no LP reformulation, no m x m matrix: every
        // Newton step is an n x n weighted normal solve built by ONE pass over A.
        //
        // SIGN CONVENTION (get this wrong and every other formula is moot): lp_fnm's OUTPUT variable
        // is literally named "y" -- the equality multipliers of Ãv=b̃ -- and the caller negates it:
        // `b_coef = -lp_fnm(...)'`. So the regression coefficients returned by this core are `x = -y`,
        // y being this file's own "y" (n-vector). Get this sign backwards and the fit comes out
        // reflected through zero.
        //
        // Factorization: CHOP (pivoted/rank-revealing Cholesky), NOT plain CHO -- the IPM weights q_i
        // polarize toward 0/infinity as (a,s) approach the [0,1] boundary near convergence, so AᵀQA
        // becomes numerically near-semidefinite exactly in the endgame, in float. Plain CHO hard-fails
        // the instant a pivot goes non-positive; CHOP instead reports a (possibly reduced) numerical
        // rank and CHOP.decompSolve's rank-deficient branch still returns a well-defined minimum-norm
        // direction. One CHOP.decomp per iteration, reused for both the affine-predictor and the
        // centering-corrector solve via CHOP.decompSolve. The one-time least-squares INIT stays on
        // plain CHO: it runs once, outside the loop, on the UNWEIGHTED normal matrix AᵀA, which has
        // none of the endgame's polarization problem.
        //
        // Best-iterate safeguard: mirrors LP.InteriorPoint -- yBest tracks the multiplier vector at
        // the smallest duality gap seen (a NaN/Inf blow-up or an unrecoverable Indefinite
        // factorization stops the loop but never corrupts the returned x).
        //
        // Job-safe: all scratch is Allocator.Temp, disposed on every return path.
        // ============================================================================================

        /// <summary>
        /// Exact least-absolute-deviation (L1) regression via the Frisch-Newton primal-dual interior
        /// point method (Portnoy &amp; Koenker 1997) -- tau = 0.5 of the tau-parameterized quantile-
        /// regression core (<see cref="ladFrischNewtonCore"/>). Solves the LAD DUAL directly over the
        /// original m x n design: every Newton step is an n x n weighted normal solve (pivoted
        /// Cholesky, library CHOP) built by one pass over A -- never a (2n+2m)-variable LP
        /// reformulation like <see cref="lad"/>'s simplex/interior/revised/dual backends, and never an
        /// m x m matrix.
        ///
        /// <see cref="objective"/> is the L1 residual ‖A x − b‖₁, recomputed from the returned x
        /// (honest, not the internal duality gap -- an internal gap can under-report the true residual
        /// on a not-quite-converged iterate, exactly like <see cref="lad"/>'s own objective convention).
        /// FN on this bounded dual is never Infeasible/Unbounded for finite data: only
        /// <see cref="LPStatus.Optimal"/> (duality gap reached tolerance) or
        /// <see cref="LPStatus.MaxIterations"/> (best iterate on the iteration cap) are returned.
        ///
        /// <see cref="lad"/>'s own default routing is UNCHANGED by this method -- call ladFN directly
        /// for the exact Frisch-Newton route.
        /// </summary>
        /// <param name="A">Design matrix, m×n (m observations, n coefficients). m ≥ n typical.</param>
        /// <param name="b">Observations, length m.</param>
        /// <param name="x">Output coefficients, length n (overwritten). May be negative.</param>
        /// <param name="objective">Output L1 residual ‖A x − b‖₁.</param>
        /// <param name="maxIter">Iteration budget; ≤0 picks the default (50, matching the reference
        /// implementation's own max_it).</param>
        public static LPInfo ladFN(in fProxyMxN A, in fProxyN b, ref fProxyN x, out double objective, int maxIter = 0)
        {
            int m = A.M_Rows, n = A.N_Cols;

            if (b.N != m) throw new ArgumentException("LP.ladFN: b.N must equal A.M_Rows");
            if (x.N != n) throw new ArgumentException("LP.ladFN: x.N must equal A.N_Cols");

            return ladFrischNewtonCore(in A, in b, 0.5, ref x, out objective, maxIter);
        }

        /// <summary>
        /// Quantile regression via the same Frisch–Newton interior point: fits the conditional
        /// τ-quantile of b given A by minimizing the check loss Σᵢ ρ_τ(bᵢ − Aᵢ·x) over a FREE x,
        /// where ρ_τ(u) = u·(τ − 1[u&lt;0]). τ = 0.5 is median regression (identical fit to the
        /// τ-less <see cref="ladFN(in fProxyMxN, in fProxyN, ref fProxyN, out double, int)"/>);
        /// τ = 0.9 fits the 90th conditional percentile, and so on. Exact and reformulation-free —
        /// each iteration is one n×n weighted normal solve streamed from the original A.
        /// </summary>
        /// <param name="A">Design matrix, m×n (m observations, n coefficients). m ≥ n typical.</param>
        /// <param name="b">Observations, length m.</param>
        /// <param name="tau">Quantile level, strictly inside (0, 1). 0.5 = LAD/median.</param>
        /// <param name="x">Output coefficients, length n (overwritten). May be negative.</param>
        /// <param name="objective">Output ‖A x − b‖₁ at the returned fit (the plain L1 residual,
        /// reported for cross-method comparability; the τ-quantile fit does not minimize this
        /// unweighted sum unless τ = 0.5).</param>
        /// <param name="maxIter">Iteration budget; ≤0 picks the default (50).</param>
        public static LPInfo ladFN(in fProxyMxN A, in fProxyN b, double tau, ref fProxyN x,
                                   out double objective, int maxIter = 0)
        {
            int m = A.M_Rows, n = A.N_Cols;

            if (b.N != m) throw new ArgumentException("LP.ladFN: b.N must equal A.M_Rows");
            if (x.N != n) throw new ArgumentException("LP.ladFN: x.N must equal A.N_Cols");
            if (!(tau > 0.0 && tau < 1.0)) throw new ArgumentException("LP.ladFN: tau must be strictly inside (0, 1)");

            return ladFrischNewtonCore(in A, in b, tau, ref x, out objective, maxIter);
        }

        // tau-parameterized core. internal (not private): the InternalsVisibleTo("BurstLinearAlgebra.Tests")
        // grant (see AssemblyInfo.cs) also lets the hand-written test suite call it directly; the public
        // tau surface above is the supported entry (tau=0.5 is LAD up to a factor 2 in the raw LP
        // objective -- irrelevant here since `objective` is always the honest recomputed L1 residual).
        internal static LPInfo ladFrischNewtonCore(in fProxyMxN A, in fProxyN b, double tau,
                                                   ref fProxyN x, out double objective, int maxIter)
        {
            int m = A.M_Rows, n = A.N_Cols;

            // Strict-interior guard: a=(1-tau) and s=tau are the fixed starting values below, and
            // q = 1/(z/a + w/s) divides by both -- keep tau (hence a, s) safely away from the [0,1]
            // boundary regardless of what the caller passes.
            fProxy tauC = (fProxy)math.clamp(tau, 1e-6, 1.0 - 1e-6);
            fProxy oneMinusTau = (fProxy)1 - tauC;

            var a = new fProxyN(m, Allocator.Temp);
            var s = new fProxyN(m, Allocator.Temp);
            var y = new fProxyN(n, Allocator.Temp);          // dual multipliers; x_coef = -y (see file header)
            var z = new fProxyN(m, Allocator.Temp);
            var w = new fProxyN(m, Allocator.Temp);
            var q = new fProxyN(m, Allocator.Temp);
            var zw = new fProxyN(m, Allocator.Temp);         // z - w, recomputed each iteration
            var Av = new fProxyN(m, Allocator.Temp);         // scratch: A * (n-vector), reused per use
            var tmpN = new fProxyN(n, Allocator.Temp);        // scratch: CHOP.decompSolve in/out (n-length)
            var rhs = new fProxyN(n, Allocator.Temp);         // affine-predictor RHS, kept for the corrector
            var rhs2 = new fProxyN(n, Allocator.Temp);
            var dyAff = new fProxyN(n, Allocator.Temp);
            var daAff = new fProxyN(m, Allocator.Temp);
            var dsAff = new fProxyN(m, Allocator.Temp);
            var dzAff = new fProxyN(m, Allocator.Temp);
            var dwAff = new fProxyN(m, Allocator.Temp);
            var dy = new fProxyN(n, Allocator.Temp);
            var da = new fProxyN(m, Allocator.Temp);
            var ds = new fProxyN(m, Allocator.Temp);
            var dz = new fProxyN(m, Allocator.Temp);
            var dw = new fProxyN(m, Allocator.Temp);
            var dadz = new fProxyN(m, Allocator.Temp);
            var dsdw = new fProxyN(m, Allocator.Temp);
            var xi = new fProxyN(m, Allocator.Temp);
            var qCorr = new fProxyN(m, Allocator.Temp);       // q .* (dadz - dsdw - xi), corrector RHS term
            var M = new fProxyMxN(n, n, Allocator.Temp);
            var L = new fProxyMxN(n, n, Allocator.Temp);
            var yBest = new fProxyN(n, Allocator.Temp);
            var bLP = new fProxyN(n, Allocator.Temp);         // Aᵀ((1-tau)·1), the dual LP's constraint RHS
            var Linit = new fProxyMxN(n, n, Allocator.Temp);  // plain-CHO factor, LS init only

            var P = new Pivot(n, Allocator.Temp);
            var ws = new fProxyCHOPCache { W = new fProxyMxN(n, n, Allocator.Temp), bt = new fProxyN(n, Allocator.Temp) };

            fProxy reg = Consts.fProxyZeroThreshold;
            fProxy BIG = (fProxy)1e30;
            fProxy beta = (fProxy)0.9995;
            fProxy zwFloor = Consts.fProxyZeroThreshold;

            for (int i = 0; i < m; i++) { a[i] = oneMinusTau; s[i] = tauC; }
            ATmul(A, a, bLP, m, n);

            double bNorm = 0;
            for (int i = 0; i < m; i++) bNorm += (double)b[i] * (double)b[i];
            bNorm = math.sqrt(bNorm);

            // ---- init: y from an ORDINARY LEAST-SQUARES fit (plain CHO on the unweighted normal
            // equations AᵀA w = Aᵀb; q=1 uniformly here -- no endgame polarization at this one-time,
            // outside-the-loop solve, so plain CHO is the right (cleanest) tool, per spec), then
            // y = -w (rq_fnm's sign convention -- see file header). ----
            for (int i = 0; i < m; i++) q[i] = (fProxy)1;
            BuildATQA(A, q, M, m, n, reg);
            bool okInit = CHO.decomp(in M, ref Linit);
            if (okInit)
            {
                ATmul(A, b, tmpN, m, n);
                CHO.decompSolve(ref Linit, ref tmpN);
                for (int j = 0; j < n; j++) y[j] = -tmpN[j];
            }
            else
                for (int j = 0; j < n; j++) y[j] = (fProxy)0;

            // r = c - yᵀÃ = -b - Ay  (c = -b; see file header's Ã=Aᵀ, c=-b dual construction)
            Amul(A, y, Av, m, n);
            for (int i = 0; i < m; i++)
            {
                fProxy r = -b[i] - Av[i];
                if (math.abs(r) < zwFloor)
                {
                    z[i] = math.max(r, (fProxy)0) + zwFloor;
                    w[i] = math.max(-r, (fProxy)0) + zwFloor;
                }
                else
                {
                    z[i] = math.max(r, (fProxy)0);
                    w[i] = math.max(-r, (fProxy)0);
                }
            }

            double gap = DualityGap(b, a, y, bLP, w, m, n);
            double gapTol = (double)Consts.fProxySqrtEps * (1.0 + bNorm);

            int budget = maxIter > 0 ? maxIter : 50;
            LPStatus status = gap <= gapTol ? LPStatus.Optimal : LPStatus.MaxIterations;
            int iters = 0;

            for (int j = 0; j < n; j++) yBest[j] = y[j];
            double bestGap = gap;

            while (status != LPStatus.Optimal && iters < budget)
            {
                // 1. diagonal weights q_i = 1 / (z_i/a_i + w_i/s_i), and zw_i = z_i - w_i (used
                // together to build Av = q.*zw a few lines below), computed in one pass over m -- see
                // BuildFNWeights.
                unsafe { BuildFNWeights(z.Data.Ptr, a.Data.Ptr, w.Data.Ptr, s.Data.Ptr, q.Data.Ptr, zw.Data.Ptr, m); }

                // 2. the kernel: M = AᵀQA (one row-streaming pass over A), pivoted-Cholesky factor.
                BuildATQA(A, q, M, m, n, reg);
                RankInfo rinfo = CHOP.decomp(in M, ref L, ref P, ref ws);
                fProxy bump = reg;
                for (int t = 0; rinfo.status == DirectSolveStatus.Indefinite && t < 4; t++)
                {
                    bump *= (fProxy)1e3;
                    for (int r = 0; r < n; r++) M[r, r] += bump;
                    rinfo = CHOP.decomp(in M, ref L, ref P, ref ws);
                }
                if (!rinfo.Solved) break;   // unrecoverable -> stop, keep yBest (status stays MaxIterations)
                int rank = rinfo.rank;

                // 3a. affine-predictor solve: rhs = Aᵀ(q .* zw); dyAff = M⁻¹ rhs
                for (int i = 0; i < m; i++) Av[i] = q[i] * zw[i];
                ATmul(A, Av, rhs, m, n);
                for (int j = 0; j < n; j++) tmpN[j] = rhs[j];
                CHOP.decompSolve(ref L, in P, rank, ref tmpN, ref ws);
                for (int j = 0; j < n; j++) dyAff[j] = tmpN[j];

                Amul(A, dyAff, Av, m, n);
                for (int i = 0; i < m; i++)
                {
                    daAff[i] = q[i] * (Av[i] - zw[i]);
                    dsAff[i] = -daAff[i];
                    dzAff[i] = -z[i] * ((fProxy)1 + daAff[i] / a[i]);
                    dwAff[i] = -w[i] * ((fProxy)1 + dsAff[i] / s[i]);
                }

                fProxy fa = math.min(MaxStep(a, daAff, m, BIG), MaxStep(s, dsAff, m, BIG));
                fProxy fd = math.min(MaxStep(w, dwAff, m, BIG), MaxStep(z, dzAff, m, BIG));
                fa = math.min(beta * fa, (fProxy)1);
                fd = math.min(beta * fd, (fProxy)1);

                if (math.min(fa, fd) < (fProxy)1)
                {
                    // 3b. Mehrotra centering + corrector, REUSING the SAME factor (two solves, one factor).
                    double muCur = 0;
                    for (int i = 0; i < m; i++) muCur += (double)z[i] * (double)a[i] + (double)w[i] * (double)s[i];
                    double g = 0;
                    for (int i = 0; i < m; i++)
                    {
                        g += (double)(z[i] + fd * dzAff[i]) * (double)(a[i] + fa * daAff[i])
                           + (double)(w[i] + fd * dwAff[i]) * (double)(s[i] + fa * dsAff[i]);
                    }
                    double ratio = g / math.max(muCur, 1e-300);
                    double muTarget = muCur * ratio * ratio * ratio / (2.0 * m);

                    for (int i = 0; i < m; i++)
                    {
                        dadz[i] = daAff[i] * dzAff[i];
                        dsdw[i] = dsAff[i] * dwAff[i];
                        xi[i] = (fProxy)(muTarget * (1.0 / (double)a[i] - 1.0 / (double)s[i]));
                        qCorr[i] = q[i] * (dadz[i] - dsdw[i] - xi[i]);
                    }
                    ATmul(A, qCorr, tmpN, m, n);
                    for (int j = 0; j < n; j++) rhs2[j] = rhs[j] + tmpN[j];
                    for (int j = 0; j < n; j++) tmpN[j] = rhs2[j];
                    CHOP.decompSolve(ref L, in P, rank, ref tmpN, ref ws);
                    for (int j = 0; j < n; j++) dy[j] = tmpN[j];

                    Amul(A, dy, Av, m, n);
                    for (int i = 0; i < m; i++)
                    {
                        da[i] = q[i] * (Av[i] + xi[i] - zw[i] - dadz[i] + dsdw[i]);
                        ds[i] = -da[i];
                        dz[i] = (fProxy)(muTarget / (double)a[i]) - z[i] - (z[i] / a[i]) * da[i] - dadz[i];
                        dw[i] = (fProxy)(muTarget / (double)s[i]) - w[i] - (w[i] / s[i]) * ds[i] - dsdw[i];
                    }

                    fa = math.min(MaxStep(a, da, m, BIG), MaxStep(s, ds, m, BIG));
                    fd = math.min(MaxStep(w, dw, m, BIG), MaxStep(z, dz, m, BIG));
                    fa = math.min(beta * fa, (fProxy)1);
                    fd = math.min(beta * fd, (fProxy)1);
                }
                else
                {
                    for (int j = 0; j < n; j++) dy[j] = dyAff[j];
                    for (int i = 0; i < m; i++) { da[i] = daAff[i]; ds[i] = dsAff[i]; dz[i] = dzAff[i]; dw[i] = dwAff[i]; }
                }

                // 4. take the step
                for (int i = 0; i < m; i++) { a[i] += fa * da[i]; s[i] += fa * ds[i]; }
                for (int j = 0; j < n; j++) y[j] += fd * dy[j];
                for (int i = 0; i < m; i++) { w[i] += fd * dw[i]; z[i] += fd * dz[i]; }

                gap = DualityGap(b, a, y, bLP, w, m, n);
                iters++;

                if (!(gap < 1e300)) break;                              // NaN/Inf blow-up -> keep yBest
                if (gap <= bestGap) { bestGap = gap; for (int j = 0; j < n; j++) yBest[j] = y[j]; }
                if (gap <= gapTol) status = LPStatus.Optimal;
            }

            for (int j = 0; j < n; j++) x[j] = -yBest[j];
            double obj = 0;
            for (int i = 0; i < m; i++)
            {
                double rowDot = 0;
                for (int j = 0; j < n; j++) rowDot += (double)A[i, j] * (double)x[j];
                obj += math.abs(rowDot - (double)b[i]);
            }
            objective = obj;

            a.Dispose(); s.Dispose(); y.Dispose(); z.Dispose(); w.Dispose(); q.Dispose(); zw.Dispose();
            Av.Dispose(); tmpN.Dispose(); rhs.Dispose(); rhs2.Dispose();
            dyAff.Dispose(); daAff.Dispose(); dsAff.Dispose(); dzAff.Dispose(); dwAff.Dispose();
            dy.Dispose(); da.Dispose(); ds.Dispose(); dz.Dispose(); dw.Dispose();
            dadz.Dispose(); dsdw.Dispose(); xi.Dispose(); qCorr.Dispose();
            M.Dispose(); L.Dispose(); Linit.Dispose(); yBest.Dispose(); bLP.Dispose();
            P.Dispose(); ws.W.Dispose(); ws.bt.Dispose();

            return new LPInfo { status = status, iterations = iters, objective = obj };
        }

        // M (n x n) = Aᵀ diag(q) A + reg·I, built as ONE cache-friendly pass over A's ROWS: each row i
        // contributes q_i · A[i,:] ⊗ A[i,:] (an outer product) to M's upper triangle, then mirrored.
        // Row-major storage (the library's row-major matrix convention) makes reading A[i,:] unit-
        // stride, which a column-contracted loop order (fixed column, varying row) would not be -- the
        // same row-streaming rationale as LP.InteriorPoint's BuildNormal, just contracting over the
        // opposite axis (m/rows here vs nv/columns there) to land on this n x n shape.
        //
        // The inner "M[r,c] += v*A[i,c]" sweep over c in [r,n) is an AXPY (M's row r += v * A's row i,
        // both read/written unit-stride in this row-major layout), routed through UnsafeOP.axpy.
        static unsafe void BuildATQA(fProxyMxN A, fProxyN q, fProxyMxN M, int m, int n, fProxy reg)
        {
            for (int r = 0; r < n; r++)
                for (int c = r; c < n; c++)
                    M[r, c] = (fProxy)0;

            fProxy* Ap = A.Data.Ptr;
            fProxy* Mp = M.Data.Ptr;
            for (int i = 0; i < m; i++)
            {
                fProxy qi = q[i];
                fProxy* Arow = Ap + (long)i * n;
                for (int r = 0; r < n; r++)
                {
                    fProxy v = qi * Arow[r];
                    if (v == (fProxy)0) continue;
                    UnsafeOP.axpy(Mp + (long)r * n + r, Arow + r, v, n - r);
                }
            }

            for (int r = 0; r < n; r++)
            {
                M[r, r] += reg;
                for (int c = r + 1; c < n; c++)
                    M[c, r] = M[r, c];
            }
        }

        // q[i] = 1/(z[i]/a[i] + w[i]/s[i]);  zw[i] = z[i]-w[i] -- FN's per-iteration IPM barrier weight
        // and the affine-predictor's z-w term, computed in one pass over m. [NoAlias] is truthful (six
        // genuinely distinct Allocator.Temp buffers), which is what lets Burst vectorize this
        // elementwise (no-reduction) formula.
        [MethodImpl(MethodImplOptions.NoInlining)]
        static unsafe void BuildFNWeights([NoAlias] fProxy* z, [NoAlias] fProxy* a, [NoAlias] fProxy* w, [NoAlias] fProxy* s,
                                          [NoAlias] fProxy* q, [NoAlias] fProxy* zw, int m)
        {
            for (int i = 0; i < m; i++)
            {
                fProxy zi = z[i], ai = a[i], wi = w[i], si = s[i];
                q[i] = (fProxy)1 / (zi / ai + wi / si);
                zw[i] = zi - wi;
            }
        }

        // Duality gap of the bounded dual LP (rq_fnm's own `gap = c*x - y*b + w*u`, u=1): here
        // c=-b, x=a (the dual weight, NOT the returned coefficients), and b (rq_fnm's constraint RHS)
        // is bLP -- see the file header's sign map. A pure-local double accumulator (diagnostic sum),
        // matching every other IPM core's convention in this library.
        static double DualityGap(fProxyN b, fProxyN a, fProxyN y, fProxyN bLP, fProxyN w, int m, int n)
        {
            double cx = 0;
            for (int i = 0; i < m; i++) cx -= (double)b[i] * (double)a[i];
            double yb = 0;
            for (int j = 0; j < n; j++) yb += (double)y[j] * (double)bLP[j];
            double wu = 0;
            for (int i = 0; i < m; i++) wu += (double)w[i];
            return cx - yb + wu;
        }
    }
}
