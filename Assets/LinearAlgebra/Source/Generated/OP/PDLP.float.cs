#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra
{
    public static partial class LP
    {
        // ============================================================================================
        // PDLP -- matrix-free first-order LP via (restarted) PDHG, after Applegate et al. (arXiv:2501.07018
        // / 2106.04756; ref impl FirstOrderLp.jl, Apache-2.0). See docs/spec-pdlp.md.
        //
        // Solves    min cᵀx   s.t.   ℓ_c ≤ A x ≤ u_c ,   ℓ_v ≤ x ≤ u_v
        // using ONLY A·v / Aᵀ·v (matrix-free over any IfloatLinearOperator) plus elementwise clamps --
        // no normal equations, no Cholesky, no preconditioner, robust in float, deterministic.
        //
        // *** STAGE 1d ***: restarted PDHG (running-average restart) + primal weight ω + adaptive step
        // size η (Malitsky-Pock line search) + Ruiz / Pock-Chambolle diagonal preconditioning (Â=Dr·A·Dc,
        // applied matrix-free via floatRowColScaledOperator). A BSR overload + large-sparse benchmark is
        // stage 1e. Preconditioning is a positive diagonal similarity: it cannot move the optimum, only
        // reshape the geometry PDHG converges through -- so every correctness test stays a valid gate.
        //
        // Job-safe: all scratch is Allocator.Temp, disposed before return.
        // ============================================================================================

        // Normalized KKT residual μ(x,y) used for restart decisions + termination. Sign-consistent with
        // the PDHG update (reduced cost r = c + Aᵀy). Combines THREE normalized residuals -- all three must
        // be small at a true optimum (spec §4):
        //   * primal ROW feasibility  ‖A x − clamp(A x, ℓ_c,u_c)‖ / (1+‖b‖)
        //   * primal STATIONARITY     ‖x − clamp(x − r, ℓ_v,u_v)‖ / (1+‖c‖)  (projected gradient; zero iff
        //     x minimizes the box-constrained Lagrangian for the current y)
        //   * DUALITY GAP  |cᵀx − D(y)| / (1+|cᵀx|+|D(y)|),  D(y) = Σ_j min_{[ℓ_v,u_v]} r_j x_j
        //     + Σ_i min_{[ℓ_c,u_c]}(−y_i z_i) the Lagrangian dual value. Without the gap, x=0 passes the
        //     first two whenever the dual is merely FEASIBLE (r ≥ 0) even if far from optimal -- the gap is
        //     what distinguishes "feasible + Lagrangian-stationary" from "optimal". When the dual is
        //     infeasible (a box term is −∞) the gap is set to 1 (not optimal); the stationarity term is
        //     already nonzero there, so this never blocks a true stop. Two matvecs; Ax/Aty are scratch.
        static double pdlpKkt<TOp>(in TOp A, in floatN x, in floatN y,
                                   in floatN lc, in floatN uc, in floatN lv, in floatN uv, in floatN c,
                                   ref floatN Ax, ref floatN Aty, double bScale, double cScale)
            where TOp : struct, IfloatLinearOperator
        {
            int m = A.Rows, n = A.Cols;
            A.Apply(in x, ref Ax);
            double pres = 0;
            for (int i = 0; i < m; i++)
            {
                double axi = (double)Ax[i];
                double v = axi - math.clamp(axi, (double)lc[i], (double)uc[i]);
                pres += v * v;
            }
            A.ApplyT(in y, ref Aty);
            double sres = 0, P = 0, D = 0;
            bool dualInf = false;
            const double tol = 1e-6;                  // ignore sub-tol reduced-cost sign slop at ±∞ bounds
            for (int j = 0; j < n; j++)
            {
                double r = (double)c[j] + (double)Aty[j];
                double xj = (double)x[j];
                double lvj = (double)lv[j], uvj = (double)uv[j];
                double d = xj - math.clamp(xj - r, lvj, uvj);
                sres += d * d;
                P += (double)c[j] * xj;
                // dual box term: min_{x∈[ℓ_v,u_v]} r·x  = r>=0 ? r·ℓ_v : r·u_v  (−∞ if the bound is ±∞)
                if (r >= 0) { if (lvj <= -1e29) { if (r >  tol) dualInf = true; } else D += r * lvj; }
                else        { if (uvj >=  1e29) { if (r < -tol) dualInf = true; } else D += r * uvj; }
            }
            for (int i = 0; i < m; i++)
            {
                double s = -(double)y[i], lci = (double)lc[i], uci = (double)uc[i];
                if (s >= 0) { if (lci <= -1e29) { if (s >  tol) dualInf = true; } else D += s * lci; }
                else        { if (uci >=  1e29) { if (s < -tol) dualInf = true; } else D += s * uci; }
            }
            pres = math.sqrt(pres) / (1.0 + bScale);
            sres = math.sqrt(sres) / (1.0 + cScale);
            double gap = dualInf ? 1.0 : math.abs(P - D) / (1.0 + math.abs(P) + math.abs(D));
            return math.sqrt(pres * pres + sres * sres + gap * gap);
        }

        // Stage-1c core: primal-first PDHG with primal weight ω, adaptive step size η, and running-average
        // adaptive restart (spec §3.1-§3.3). Primal-first ordering keeps the step-size line-search cheap:
        // the interaction term (Δy)ᵀA(Δx) reuses A·x⁺ (needed for the dual step anyway) minus a cached A·x,
        // and step-size retries reuse Aᵀy (y is fixed while η shrinks). Generic over the constraint operator.
        static LPInfo pdlpCore<TOp>(in TOp A, in floatN lc, in floatN uc, in floatN lv, in floatN uv,
                                    in floatN c, ref floatN x, out double objective, int maxIter, double epsOpt)
            where TOp : struct, IfloatLinearOperator
        {
            int m = A.Rows, n = A.Cols;

            var y    = new floatN(m, Allocator.Temp);   // dual (row multipliers); Temp => starts at 0
            var xnew = new floatN(n, Allocator.Temp);   // trial x⁺
            var ynew = new floatN(m, Allocator.Temp);   // trial y⁺
            var Ax   = new floatN(m, Allocator.Temp);   // A·x  (cached across iterations)
            var Axp  = new floatN(m, Allocator.Temp);   // A·x⁺ (trial; also kkt scratch)
            var Aty  = new floatN(n, Allocator.Temp);   // Aᵀ·y (reused across step-size retries)
            var sumX = new floatN(n, Allocator.Temp);   // running-average accumulators
            var sumY = new floatN(m, Allocator.Temp);
            var avgX = new floatN(n, Allocator.Temp);
            var avgY = new floatN(m, Allocator.Temp);
            var xPrev = new floatN(n, Allocator.Temp);  // previous restart point (drives the primal weight)
            var yPrev = new floatN(m, Allocator.Temp);
            var xBest = new floatN(n, Allocator.Temp);  // best (lowest-μ) average seen -- reported at the end

            for (int j = 0; j < n; j++) x[j] = math.clamp(x[j], lv[j], uv[j]);   // start inside the box
            for (int j = 0; j < n; j++) xPrev[j] = x[j];
            // yPrev, y already 0

            // ‖A‖₂ = sqrt(λ_max(AᵀA)) via 20 power-iteration sweeps (matrix-free; reuse xnew/Axp/Aty as
            // scratch -- unused until the main loop). Only needed for the INITIAL η; the line search adapts.
            float normA;
            {
                float inv = (float)(1.0 / math.sqrt((double)n));
                for (int j = 0; j < n; j++) xnew[j] = inv;                 // v
                double lam = 1;
                for (int k = 0; k < 20; k++)
                {
                    A.Apply(in xnew, ref Axp);                            // Av
                    A.ApplyT(in Axp, ref Aty);                            // AᵀAv
                    double nrm = 0;
                    for (int j = 0; j < n; j++) nrm += (double)Aty[j] * (double)Aty[j];
                    nrm = math.sqrt(nrm);
                    if (!(nrm > 0)) break;
                    lam = nrm;
                    float s = (float)(1.0 / nrm);
                    for (int j = 0; j < n; j++) xnew[j] = Aty[j] * s;
                }
                normA = (float)math.sqrt(lam);
            }
            double eta = 0.9 / math.max((double)normA, 1e-30);            // step size η (adapts below)
            double omega = 1.0;                                            // primal weight ω (τ=η/ω, σ=η·ω)

            // scales for relative residuals (ignore the ±inf sentinels)
            double bScale = 0, cScale = 0;
            for (int i = 0; i < m; i++)
            {
                double a = math.abs((double)uc[i]); if (a < 1e29) bScale = math.max(bScale, a);
                double b = math.abs((double)lc[i]); if (b < 1e29) bScale = math.max(bScale, b);
            }
            for (int j = 0; j < n; j++) cScale = math.max(cScale, math.abs((double)c[j]));

            A.Apply(in x, ref Ax);                                         // seed the A·x cache

            int iters = 0, innerT = 0;
            long cnt = 0;
            LPStatus status = LPStatus.MaxIterations;
            int budget = maxIter > 0 ? maxIter : 200000;

            // reference KKT residual for the restart ratio tests, and the previous check's value
            double muRef = pdlpKkt(in A, in x, in y, in lc, in uc, in lv, in uv, in c, ref Axp, ref Aty, bScale, cScale);
            double muLast = muRef;
            double muBest = muRef;
            for (int j = 0; j < n; j++) xBest[j] = x[j];
            if (muRef < epsOpt) status = LPStatus.Optimal;

            while (status != LPStatus.Optimal && iters < budget)
            {
                A.ApplyT(in y, ref Aty);                                   // Aᵀy (fixed while η shrinks)

                // ---- adaptive step size: trial x⁺,y⁺ with η; accept iff η ≤ η̄ (Malitsky-Pock) ----
                double etaTry = eta, etaNext = eta;
                for (int trial = 0; ; trial++)
                {
                    float tau = (float)(etaTry / omega), sigma = (float)(etaTry * omega);

                    // primal:  x⁺ = clamp(x − τ(c + Aᵀy))
                    for (int j = 0; j < n; j++)
                        xnew[j] = math.clamp(x[j] - tau * (Aty[j] + c[j]), lv[j], uv[j]);
                    A.Apply(in xnew, ref Axp);                            // A·x⁺

                    // dual:  y⁺ = w − σ·clamp(w/σ),  w = y + σ·A(2x⁺ − x) = y + σ(2Axp − Ax)
                    for (int i = 0; i < m; i++)
                    {
                        float aext = (float)2 * Axp[i] - Ax[i];
                        float w = y[i] + sigma * aext;
                        ynew[i] = w - sigma * math.clamp(w / sigma, lc[i], uc[i]);
                    }

                    // interaction (Δy)ᵀA(Δx) = (Δy)ᵀ(Axp − Ax); ω-weighted move ‖Δ‖²_ω = ω‖Δx‖²+‖Δy‖²/ω
                    double inter = 0, dxn = 0, dyn = 0;
                    for (int i = 0; i < m; i++)
                    {
                        double dy = (double)ynew[i] - (double)y[i];
                        inter += dy * ((double)Axp[i] - (double)Ax[i]);
                        dyn += dy * dy;
                    }
                    for (int j = 0; j < n; j++)
                    {
                        double dx = (double)xnew[j] - (double)x[j];
                        dxn += dx * dx;
                    }
                    double normSqW = omega * dxn + dyn / omega;
                    double denom = 2.0 * math.abs(inter);
                    double etaBar = denom > 1e-30 ? normSqW / denom : 1e30;
                    double kp = (double)(iters + 1);
                    etaNext = math.min((1.0 - math.pow(kp, -0.3)) * etaBar,
                                       (1.0 + math.pow(kp, -0.6)) * etaTry);
                    etaNext = math.clamp(etaNext, 1e-12, 1e30);
                    if (etaTry <= etaBar || trial >= 60) break;           // accept
                    etaTry = etaNext;                                     // reject: shrink, retry (reuse Aᵀy)
                }
                eta = etaNext;

                // ---- commit, refresh the A·x cache, accumulate the ergodic average ----
                for (int j = 0; j < n; j++) x[j] = xnew[j];
                for (int i = 0; i < m; i++) y[i] = ynew[i];
                for (int i = 0; i < m; i++) Ax[i] = Axp[i];
                for (int j = 0; j < n; j++) sumX[j] += x[j];
                for (int i = 0; i < m; i++) sumY[i] += y[i];
                cnt++; iters++; innerT++;

                // ---- restart / termination on the running AVERAGE (spec §3.2), ω update at restart ----
                if ((iters & 63) == 0)
                {
                    float invc = (float)(1.0 / (double)cnt);
                    for (int j = 0; j < n; j++) avgX[j] = sumX[j] * invc;
                    for (int i = 0; i < m; i++) avgY[i] = sumY[i] * invc;

                    // kkt uses Axp/Aty as scratch (both recomputed next iteration); Ax cache is untouched
                    double mu = pdlpKkt(in A, in avgX, in avgY, in lc, in uc, in lv, in uv, in c, ref Axp, ref Aty, bScale, cScale);
                    if (mu < muBest) { muBest = mu; for (int j = 0; j < n; j++) xBest[j] = avgX[j]; }
                    if (mu < epsOpt)
                    {
                        for (int j = 0; j < n; j++) x[j] = avgX[j];
                        status = LPStatus.Optimal;
                        break;
                    }
                    // (i) sufficient decay, (ii) necessary decay + stall, (iii) long-loop safeguard
                    bool restart = (mu <= 0.1 * muRef)
                                || (mu <= 0.9 * muRef && mu > muLast)
                                || (innerT >= 0.5 * iters);
                    if (restart)
                    {
                        // primal weight ω from restart-point movement (spec §3.3), geometric blend
                        double dxp = 0, dyp = 0;
                        for (int j = 0; j < n; j++) { double d = (double)avgX[j] - (double)xPrev[j]; dxp += d * d; }
                        for (int i = 0; i < m; i++) { double d = (double)avgY[i] - (double)yPrev[i]; dyp += d * d; }
                        dxp = math.sqrt(dxp); dyp = math.sqrt(dyp);
                        if (dxp > 1e-20 && dyp > 1e-20)
                            omega = math.exp(0.5 * math.log(dyp / dxp) + 0.5 * math.log(omega));
                        omega = math.clamp(omega, 1e-8, 1e8);

                        for (int j = 0; j < n; j++) { x[j] = avgX[j]; xPrev[j] = avgX[j]; sumX[j] = (float)0; }
                        for (int i = 0; i < m; i++) { y[i] = avgY[i]; yPrev[i] = avgY[i]; sumY[i] = (float)0; }
                        A.Apply(in x, ref Ax);                            // refresh the cache after the jump
                        cnt = 0; innerT = 0; muRef = mu; muLast = mu;
                    }
                    else muLast = mu;
                }
            }

            // report the BEST (lowest-μ) average seen -- robust to a stalled/oscillating tail where the plain
            // running average drifts (e.g. once μ plateaus at float's residual floor and restarts stop
            // firing, the ever-growing window averages to a worse point). On an Optimal break x already holds
            // the converged average, which by construction is the best.
            if (status != LPStatus.Optimal)
                for (int j = 0; j < n; j++) x[j] = xBest[j];

            objective = 0;
            for (int j = 0; j < n; j++) objective += (double)c[j] * (double)x[j];

            y.Dispose(); xnew.Dispose(); ynew.Dispose(); Ax.Dispose(); Axp.Dispose(); Aty.Dispose();
            sumX.Dispose(); sumY.Dispose(); avgX.Dispose(); avgY.Dispose(); xPrev.Dispose(); yPrev.Dispose();
            xBest.Dispose();
            return new LPInfo { status = status, iterations = iters, objective = objective };
        }

        // Ruiz (ℓ∞) equilibration from the DENSE matrix entries: fills Dr (length m) / Dc (length n) so that
        // Â = Dr·A·Dc has ~unit row/column inf-norms (‖Â‖≈1 -- the geometry PDHG's step size lives in). Done
        // once, up front. A positive diagonal can't move the optimum, so this only reshapes convergence.
        // O(mn) per sweep, ~10 sweeps. (NB: a Pock-Chambolle ℓ1 pass on top of Ruiz over-shrinks -- it
        // divides by sqrt(row/col 1-norm ~ n) -- driving ‖Â‖→0 and η→∞; Ruiz alone hits the right target.)
        static void pdlpEquilibrateDense(in floatMxN A, ref floatN Dr, ref floatN Dc)
        {
            int m = A.M_Rows, n = A.N_Cols;
            for (int i = 0; i < m; i++) Dr[i] = (float)1;
            for (int j = 0; j < n; j++) Dc[j] = (float)1;

            for (int it = 0; it < 10; it++)
            {
                for (int i = 0; i < m; i++)                       // row inf-norms of the current Â
                {
                    double mx = 0;
                    for (int j = 0; j < n; j++) { double a = math.abs((double)A[i, j]) * (double)Dc[j]; if (a > mx) mx = a; }
                    double R = (double)Dr[i] * mx;
                    if (R > 1e-30) Dr[i] = (float)((double)Dr[i] / math.sqrt(R));
                }
                for (int j = 0; j < n; j++)                       // column inf-norms (using the updated Dr)
                {
                    double mx = 0;
                    for (int i = 0; i < m; i++) { double a = math.abs((double)A[i, j]) * (double)Dr[i]; if (a > mx) mx = a; }
                    double C = (double)Dc[j] * mx;
                    if (C > 1e-30) Dc[j] = (float)((double)Dc[j] / math.sqrt(C));
                }
            }

            for (int i = 0; i < m; i++) Dr[i] = (float)math.clamp((double)Dr[i], 1e-12, 1e12);
            for (int j = 0; j < n; j++) Dc[j] = (float)math.clamp((double)Dc[j], 1e-12, 1e12);
        }

        // Solve the equilibrated problem on Â = Dr·A·Dc (matrix-free, via floatRowColScaledOperator) and
        // map the answer back to the original variable. Generic over the inner operator so the dense entry
        // point and the (stage-1e) BSR overload share this glue; they differ only in how Dr/Dc are built.
        //   rows i:   ℓ̂_c = Dr·ℓ_c , û_c = Dr·u_c            (equality/ineq senses preserved: Dr > 0)
        //   box  j:   ℓ̂_v = ℓ_v/Dc , û_v = u_v/Dc , ĉ = Dc·c , x̂ = x/Dc   (then x = Dc·x̂ on the way out)
        // ±1e30 unbounded sentinels pass through unscaled. Objective is recomputed as the original cᵀx.
        static LPInfo pdlpScaledSolve<TOp>(in TOp inner, in floatN Dr, in floatN Dc,
                                           in floatN lc, in floatN uc, in floatN lv, in floatN uv, in floatN c,
                                           ref floatN x, out double objective, int maxIter, double epsOpt)
            where TOp : struct, IfloatLinearOperator
        {
            int m = inner.Rows, n = inner.Cols;
            var lcS = new floatN(m, Allocator.Temp);
            var ucS = new floatN(m, Allocator.Temp);
            var lvS = new floatN(n, Allocator.Temp);
            var uvS = new floatN(n, Allocator.Temp);
            var cS  = new floatN(n, Allocator.Temp);
            var scratchN = new floatN(n, Allocator.Temp);
            var scratchM = new floatN(m, Allocator.Temp);

            for (int i = 0; i < m; i++)
            {
                lcS[i] = math.abs((double)lc[i]) >= 1e29 ? lc[i] : (float)((double)lc[i] * (double)Dr[i]);
                ucS[i] = math.abs((double)uc[i]) >= 1e29 ? uc[i] : (float)((double)uc[i] * (double)Dr[i]);
            }
            for (int j = 0; j < n; j++)
            {
                lvS[j] = math.abs((double)lv[j]) >= 1e29 ? lv[j] : (float)((double)lv[j] / (double)Dc[j]);
                uvS[j] = math.abs((double)uv[j]) >= 1e29 ? uv[j] : (float)((double)uv[j] / (double)Dc[j]);
                cS[j]  = (float)((double)c[j] * (double)Dc[j]);
                x[j]   = (float)((double)x[j] / (double)Dc[j]);          // x̂ = x / Dc
            }

            var op = new floatRowColScaledOperator<TOp>(in inner, in Dr, in Dc, in scratchN, in scratchM);
            var info = pdlpCore(in op, in lcS, in ucS, in lvS, in uvS, in cS, ref x, out double _, maxIter, epsOpt);

            objective = 0;
            for (int j = 0; j < n; j++)
            {
                x[j] = (float)((double)Dc[j] * (double)x[j]);           // x = Dc · x̂
                objective += (double)c[j] * (double)x[j];
            }
            info.objective = objective;

            lcS.Dispose(); ucS.Dispose(); lvS.Dispose(); uvS.Dispose(); cS.Dispose();
            scratchN.Dispose(); scratchM.Dispose();
            return info;
        }

        /// <summary>
        /// Dense PDLP (matrix-free first-order LP): minimize cᵀx s.t. ℓ_c ≤ A x ≤ u_c, ℓ_v ≤ x ≤ u_v.
        /// Use ±<c>1e30</c> in a bound to mean unbounded (so ℓ_c=u_c is an equality row, ℓ_v=0/u_v=1e30 is
        /// x ≥ 0). Restarted PDHG with adaptive step size + primal weight + Ruiz/Pock-Chambolle
        /// preconditioning (see docs/spec-pdlp.md). <paramref name="x"/> (length A.N_Cols) is overwritten
        /// with the solution.
        /// </summary>
        public static LPInfo pdlp(in floatMxN A, in floatN lc, in floatN uc, in floatN lv, in floatN uv,
                                  in floatN c, ref floatN x, out double objective, int maxIter = 0, double epsOpt = 1e-6)
        {
            int m = A.M_Rows, n = A.N_Cols;
            if (lc.N != m || uc.N != m) throw new System.ArgumentException("LP.pdlp: lc/uc length must equal A.M_Rows");
            if (lv.N != n || uv.N != n) throw new System.ArgumentException("LP.pdlp: lv/uv length must equal A.N_Cols");
            if (c.N != n) throw new System.ArgumentException("LP.pdlp: c length must equal A.N_Cols");
            if (x.N != n) throw new System.ArgumentException("LP.pdlp: x length must equal A.N_Cols");

            var Dr = new floatN(m, Allocator.Temp);
            var Dc = new floatN(n, Allocator.Temp);
            pdlpEquilibrateDense(in A, ref Dr, ref Dc);

            var op = new floatDenseOperator(in A);
            var info = pdlpScaledSolve(in op, in Dr, in Dc, in lc, in uc, in lv, in uv, in c, ref x, out objective, maxIter, epsOpt);

            Dr.Dispose(); Dc.Dispose();
            return info;
        }
    }
}
