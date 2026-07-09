#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra
{
    public static partial class LP
    {
        // ============================================================================================
        // Frisch-Newton exact LAD / quantile-regression solver (Portnoy & Koenker 1997, "The Gaussian
        // Hare and the Laplacian Tortoise"). A structure-exploiting primal-dual interior point on the
        // LAD DUAL, working directly on the original m x n design A -- no LP reformulation (no 2n+2m
        // blow-up like LP.lad's simplex/interior/revised/dual backends), no m x m matrix: every Newton
        // step is an n x n weighted normal solve built by ONE pass over A.
        //
        // Ported and verified line-by-line against the canonical reference: Daniel Morillo & Roger
        // Koenker's `rq_fnm` / `lp_fnm` (originally Ox, translated to MATLAB by Paul Eilers 1999,
        // modified by Koenker April 2001), fetched from
        // https://github.com/karenamckinnon/summer-temperature-distributions/blob/master/rq.m
        // (mirrors the file distributed with R's quantreg package; the same algorithm is also in
        // quantreg's Fortran rqfnb.f). Every update formula below (predictor, centering parameter,
        // corrector, step-length ratio test, the 0.9995 factor) is that source's, not reconstructed
        // from memory -- see the derivation trail in docs/spec-lad-frisch-newton.md's authoring history.
        //
        // ---- Problem, dual, and the SIGN CONVENTION (get this wrong and every other formula is moot)
        // ----
        // Quantile regression at level tau in (0,1) (tau=0.5 == LAD up to a factor 2):
        //     min_x  sum_i rho_tau(b_i - A_i.x),   rho_tau(u) = u*(tau - 1[u<0])
        // Its dual (rq_fnm's construction): max_a b.a  s.t. Aᵀa = (1-tau)Aᵀ1,  a in [0,1]^m -- solved by
        // lp_fnm as  min c.v  s.t. Ãv = b̃, 0<=v<=1  with Ã=Aᵀ, c=-b, b̃=Aᵀ((1-tau)1), and the LP's own
        // primal variable v IS this dual weight a (rq_fnm's outer "x"/"a" argument doubles as both the
        // RHS-defining vector AND v's initial interior point). lp_fnm's OUTPUT variable is literally
        // named "y" -- the equality multipliers of Ãv=b̃ -- and the caller negates it:
        // `b_coef = -lp_fnm(...)'`. So the regression coefficients returned by this core are `x = -y`,
        // y being this file's own "y" (n-vector). Get this sign backwards and the fit comes out
        // reflected through zero; verified against LadStackloss's published coefficients in testing.
        //
        // ---- Kernel: reuse, per the standing rule ----
        // The n x n normal matrix AᵀQA (Q = diag(q), q_i = 1/(z_i/a_i + w_i/s_i)) is built by
        // BuildATQA below, ONE row-streaming pass over A (unit-stride reads, matching A's row-major
        // storage -- see the matrix row-major convention note) -- the mirror of LP.InteriorPoint's
        // BuildNormal, just contracting over the opposite axis (rows/m here vs columns/nv there) to
        // land on an n x n shape instead of m x m.
        //
        // Factorization: CHOP (pivoted/rank-revealing Cholesky), NOT plain CHO, per this feature's
        // review -- the IPM weights q_i polarize toward 0/infinity as (a,s) approach the [0,1]
        // boundary near convergence, so AᵀQA becomes numerically near-semidefinite exactly in the
        // endgame the exact-fit degenerate test (docs/spec-lad-frisch-newton.md test 4) exercises
        // hardest, in float. Plain CHO hard-fails the instant a pivot goes non-positive; CHOP instead
        // reports a (possibly reduced) numerical rank and CHOP.decompSolve's rank-deficient branch
        // still returns a well-defined minimum-norm direction (RankInfo.Solved is true for BOTH
        // Success and RankDeficient) -- exactly the graceful degradation this endgame needs instead of
        // a NaN blow-up. Pivoting cost is O(n) selection per column, negligible at LAD's typical
        // n (a handful to a few dozen coefficients). One CHOP.decomp (with its own bump-retry ladder
        // on a genuine DirectSolveStatus.Indefinite, mirroring LP.InteriorPoint's CHO bump-retry) per
        // iteration, reused for BOTH the affine-predictor and the centering-corrector solve via
        // CHOP.decompSolve -- "two solves, one factor", same contract as every other IPM core in this
        // library. The one-time least-squares INIT (below) stays on plain CHO: it runs once, outside
        // the loop, on the UNWEIGHTED (q=1) normal matrix AᵀA, which has none of the endgame's
        // polarization problem.
        //
        // Best-iterate safeguard: mirrors LP.InteriorPoint -- yBest tracks the multiplier vector at the
        // smallest duality gap seen (a NaN/Inf blow-up or an unrecoverable Indefinite factorization
        // stops the loop but never corrupts the returned x). Iteration cap default 50 (rq_fnm's own
        // max_it), duality-gap tolerance derived from Consts.doubleSqrtEps (double: ~1.49e-8, matching
        // the spec's "1e-8 double-equivalent"; float: ~3.45e-4), scaled by (1+||b||) for problem-scale
        // robustness (the same `* (1.0 + scale)` idiom simplexCore/RevisedSimplex use for their own
        // tolerances).
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
        /// m x m matrix. See docs/spec-lad-frisch-newton.md.
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
        public static LPInfo ladFN(in doubleMxN A, in doubleN b, ref doubleN x, out double objective, int maxIter = 0)
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
        /// τ-less <see cref="ladFN(in doubleMxN, in doubleN, ref doubleN, out double, int)"/>);
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
        public static LPInfo ladFN(in doubleMxN A, in doubleN b, double tau, ref doubleN x,
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
        internal static LPInfo ladFrischNewtonCore(in doubleMxN A, in doubleN b, double tau,
                                                   ref doubleN x, out double objective, int maxIter)
        {
            int m = A.M_Rows, n = A.N_Cols;

            // Strict-interior guard: a=(1-tau) and s=tau are the fixed starting values below, and
            // q = 1/(z/a + w/s) divides by both -- keep tau (hence a, s) safely away from the [0,1]
            // boundary regardless of what the caller passes.
            double tauC = (double)math.clamp(tau, 1e-6, 1.0 - 1e-6);
            double oneMinusTau = (double)1 - tauC;

            var a = new doubleN(m, Allocator.Temp);
            var s = new doubleN(m, Allocator.Temp);
            var y = new doubleN(n, Allocator.Temp);          // dual multipliers; x_coef = -y (see file header)
            var z = new doubleN(m, Allocator.Temp);
            var w = new doubleN(m, Allocator.Temp);
            var q = new doubleN(m, Allocator.Temp);
            var zw = new doubleN(m, Allocator.Temp);         // z - w, recomputed each iteration
            var Av = new doubleN(m, Allocator.Temp);         // scratch: A * (n-vector), reused per use
            var tmpN = new doubleN(n, Allocator.Temp);        // scratch: CHOP.decompSolve in/out (n-length)
            var rhs = new doubleN(n, Allocator.Temp);         // affine-predictor RHS, kept for the corrector
            var rhs2 = new doubleN(n, Allocator.Temp);
            var dyAff = new doubleN(n, Allocator.Temp);
            var daAff = new doubleN(m, Allocator.Temp);
            var dsAff = new doubleN(m, Allocator.Temp);
            var dzAff = new doubleN(m, Allocator.Temp);
            var dwAff = new doubleN(m, Allocator.Temp);
            var dy = new doubleN(n, Allocator.Temp);
            var da = new doubleN(m, Allocator.Temp);
            var ds = new doubleN(m, Allocator.Temp);
            var dz = new doubleN(m, Allocator.Temp);
            var dw = new doubleN(m, Allocator.Temp);
            var dadz = new doubleN(m, Allocator.Temp);
            var dsdw = new doubleN(m, Allocator.Temp);
            var xi = new doubleN(m, Allocator.Temp);
            var qCorr = new doubleN(m, Allocator.Temp);       // q .* (dadz - dsdw - xi), corrector RHS term
            var M = new doubleMxN(n, n, Allocator.Temp);
            var L = new doubleMxN(n, n, Allocator.Temp);
            var yBest = new doubleN(n, Allocator.Temp);
            var bLP = new doubleN(n, Allocator.Temp);         // Aᵀ((1-tau)·1), the dual LP's constraint RHS
            var Linit = new doubleMxN(n, n, Allocator.Temp);  // plain-CHO factor, LS init only

            var P = new Pivot(n, Allocator.Temp);
            var ws = new doubleCHOPCache { W = new doubleMxN(n, n, Allocator.Temp), bt = new doubleN(n, Allocator.Temp) };

            double reg = Consts.doubleZeroThreshold;
            double BIG = (double)1e30;
            double beta = (double)0.9995;
            double zwFloor = Consts.doubleZeroThreshold;

            for (int i = 0; i < m; i++) { a[i] = oneMinusTau; s[i] = tauC; }
            ATmul(A, a, bLP, m, n);

            double bNorm = 0;
            for (int i = 0; i < m; i++) bNorm += (double)b[i] * (double)b[i];
            bNorm = math.sqrt(bNorm);

            // ---- init: y from an ORDINARY LEAST-SQUARES fit (plain CHO on the unweighted normal
            // equations AᵀA w = Aᵀb; q=1 uniformly here -- no endgame polarization at this one-time,
            // outside-the-loop solve, so plain CHO is the right (cleanest) tool, per spec), then
            // y = -w (rq_fnm's sign convention -- see file header). ----
            for (int i = 0; i < m; i++) q[i] = (double)1;
            BuildATQA(A, q, M, m, n, reg);
            bool okInit = CHO.decomp(in M, ref Linit);
            if (okInit)
            {
                ATmul(A, b, tmpN, m, n);
                CHO.decompSolve(ref Linit, ref tmpN);
                for (int j = 0; j < n; j++) y[j] = -tmpN[j];
            }
            else
                for (int j = 0; j < n; j++) y[j] = (double)0;

            // r = c - yᵀÃ = -b - Ay  (c = -b; see file header's Ã=Aᵀ, c=-b dual construction)
            Amul(A, y, Av, m, n);
            for (int i = 0; i < m; i++)
            {
                double r = -b[i] - Av[i];
                if (math.abs(r) < zwFloor)
                {
                    z[i] = math.max(r, (double)0) + zwFloor;
                    w[i] = math.max(-r, (double)0) + zwFloor;
                }
                else
                {
                    z[i] = math.max(r, (double)0);
                    w[i] = math.max(-r, (double)0);
                }
            }

            double gap = DualityGap(b, a, y, bLP, w, m, n);
            double gapTol = (double)Consts.doubleSqrtEps * (1.0 + bNorm);

            int budget = maxIter > 0 ? maxIter : 50;
            LPStatus status = gap <= gapTol ? LPStatus.Optimal : LPStatus.MaxIterations;
            int iters = 0;

            for (int j = 0; j < n; j++) yBest[j] = y[j];
            double bestGap = gap;

            while (status != LPStatus.Optimal && iters < budget)
            {
                // 1. diagonal weights q_i = 1 / (z_i/a_i + w_i/s_i)
                for (int i = 0; i < m; i++) q[i] = (double)1 / (z[i] / a[i] + w[i] / s[i]);
                for (int i = 0; i < m; i++) zw[i] = z[i] - w[i];

                // 2. the kernel: M = AᵀQA (one row-streaming pass over A), pivoted-Cholesky factor.
                BuildATQA(A, q, M, m, n, reg);
                RankInfo rinfo = CHOP.decomp(in M, ref L, ref P, ref ws);
                double bump = reg;
                for (int t = 0; rinfo.status == DirectSolveStatus.Indefinite && t < 4; t++)
                {
                    bump *= (double)1e3;
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
                    dzAff[i] = -z[i] * ((double)1 + daAff[i] / a[i]);
                    dwAff[i] = -w[i] * ((double)1 + dsAff[i] / s[i]);
                }

                double fa = math.min(MaxStep(a, daAff, m, BIG), MaxStep(s, dsAff, m, BIG));
                double fd = math.min(MaxStep(w, dwAff, m, BIG), MaxStep(z, dzAff, m, BIG));
                fa = math.min(beta * fa, (double)1);
                fd = math.min(beta * fd, (double)1);

                if (math.min(fa, fd) < (double)1)
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
                        xi[i] = (double)(muTarget * (1.0 / (double)a[i] - 1.0 / (double)s[i]));
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
                        dz[i] = (double)(muTarget / (double)a[i]) - z[i] - (z[i] / a[i]) * da[i] - dadz[i];
                        dw[i] = (double)(muTarget / (double)s[i]) - w[i] - (w[i] / s[i]) * ds[i] - dsdw[i];
                    }

                    fa = math.min(MaxStep(a, da, m, BIG), MaxStep(s, ds, m, BIG));
                    fd = math.min(MaxStep(w, dw, m, BIG), MaxStep(z, dz, m, BIG));
                    fa = math.min(beta * fa, (double)1);
                    fd = math.min(beta * fd, (double)1);
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
        static void BuildATQA(doubleMxN A, doubleN q, doubleMxN M, int m, int n, double reg)
        {
            for (int r = 0; r < n; r++)
                for (int c = r; c < n; c++)
                    M[r, c] = (double)0;

            for (int i = 0; i < m; i++)
            {
                double qi = q[i];
                for (int r = 0; r < n; r++)
                {
                    double v = qi * A[i, r];
                    if (v == (double)0) continue;
                    for (int c = r; c < n; c++)
                        M[r, c] += v * A[i, c];
                }
            }

            for (int r = 0; r < n; r++)
            {
                M[r, r] += reg;
                for (int c = r + 1; c < n; c++)
                    M[c, r] = M[r, c];
            }
        }

        // Duality gap of the bounded dual LP (rq_fnm's own `gap = c*x - y*b + w*u`, u=1): here
        // c=-b, x=a (the dual weight, NOT the returned coefficients), and b (rq_fnm's constraint RHS)
        // is bLP -- see the file header's sign map. A pure-local double accumulator (diagnostic sum),
        // matching every other IPM core's convention in this library.
        static double DualityGap(doubleN b, doubleN a, doubleN y, doubleN bLP, doubleN w, int m, int n)
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
