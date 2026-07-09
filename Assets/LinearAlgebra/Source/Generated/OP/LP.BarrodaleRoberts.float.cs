#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra
{
    public static partial class LP
    {
        // ============================================================================================
        // Barrodale-Roberts specialized-simplex EXACT least-absolute-deviation (L1) / quantile-
        // regression solver (Barrodale, I. & Roberts, F.D.K. 1973, "An improved algorithm for discrete
        // l1 linear approximation", SIAM J. Numer. Anal. 10(5), 839-848; also ACM TOMS Algorithm 478).
        // The second reformulation-free exact LAD engine (LP.ladBR), alongside Frisch-Newton
        // (LP.ladFN, LP.FrischNewton.float.cs) -- see docs/spec-lad-barrodale-roberts.md.
        //
        // ---- Source + verification ----
        // Transcribed line-by-line from the Koenker-d'Orey Fortran port `rqbr` (R `quantreg` package,
        // src/rqbr.f -- a Ratfor-generated translation/extension of the 1973 algorithm that also does
        // quantile regression + confidence intervals), fetched from
        // https://cdn.jsdelivr.net/gh/cran/quantreg@master/src/rqbr.f (the same mirror pattern that
        // worked for LP.FrischNewton.float.cs's source). Cross-checked against the R wrapper
        // `rq.fit.br` (R/quantreg.R, same repo) for the `ift`/`flag` status-code semantics (0 =
        // success, 1 = "Solution may be nonunique", 2 = "Premature end - possible conditioning
        // problem in x") and the toler/big conventions (reference: toler = machine-eps^(2/3), big =
        // .Machine$double.xmax -- this port instead reuses this library's own established
        // ratio/pivot-tolerance convention, see deviation 1 below).
        //
        // This file implements the core simplex ONLY -- rqbr's confidence-interval machinery
        // (lci1/lci2, entered only when the caller requests CIs) and its full tau-PATH continuation
        // (the outer loop, entered only when the caller passes an out-of-[0,1] tau as a "compute the
        // whole path" sentinel) are both DELIBERATELY NOT PORTED. Neither is reachable with a fixed
        // tau in (0,1) and CIs off (lci1=false), the only mode LP.ladBR exposes, so dropping them is a
        // straight, provable dead-branch elimination (the guards are literal booleans this port fixes
        // at compile time), not an algorithmic approximation. Every remaining line maps to a specific
        // rqbr statement label, called out in the comments below.
        //
        // ---- Deviations from the literal reference (documented, not incidental) ----
        // 1. Tolerances: rqbr's own toler = eps^(2/3) (~3.7e-11 double) is tighter than this library's
        //    established simplex ratio/pivot tolerance. This port uses the SAME pivTol as the tableau
        //    simplex's own ratio test (LP.float.cs's RatioTest: max(Consts.floatZeroThreshold,
        //    1e-9)), for consistency with the rest of the LP surface rather than importing a one-off
        //    literal.
        // 2. ift=2 ("premature end", label 23047 -- the stage-2 ratio test finds no candidate row):
        //    the raw Fortran (and R's rq.fit.br) leave x UNTOUCHED at whatever it was before this
        //    call (all-zero, on a first pass). This library's own status convention is stronger and
        //    already covers this exact case: LPStatus.Unbounded's own doc comment promises "x is the
        //    last vertex before the unbounded edge was detected" -- stage 1 has ALWAYS completed by
        //    the time stage 2's ratio test can fail this way, so a fully-formed structural extraction
        //    is available, and this port performs it instead of discarding it. Mechanically the
        //    condition (an entering column with no limiting ratio among the remaining candidates) is
        //    identical to simplexCore's own Unbounded detection, hence the LPStatus mapping; for LAD
        //    specifically (objective bounded below by 0) this is a numerical-conditioning signal, not
        //    a genuine unbounded ray -- matching R's own "possible conditioning problem" wording.
        // 3. ift=1 ("solution may be nonunique", label 23132's kr==1 degenerate-optimum scan): a
        //    WARNING the reference emits without altering x. Not surfaced -- LPStatus has no "Optimal
        //    but nonunique" state, and the returned x/objective are unaffected either way.
        // 4. Diagnostic-only reference outputs (dsol/sol/h/e -- signed residuals, the tau-path R1
        //    stat, per-solution dual weights) are dropped entirely; this file's own honest-recomputed
        //    objective (below) replaces the reference's internal running sum, for the same reason
        //    ladFN's does (an internal sum can under/over-report against a not-fully-resolved
        //    iterate).
        //
        // ---- Algorithm shape (matches docs/spec-lad-barrodale-roberts.md's summary) ----
        // Two stages worked directly on an m x n condensed tableau of the ORIGINAL A (rqbr's `wa`,
        // here split into a plain m x n `T` plus a length-m `rhs` column -- rqbr's own n1/n2 RHS
        // columns are PROVABLY identical for every row throughout this port, since neither ever
        // receives the tau-path-only `idxcf` perturbation that would separate them, so they collapse
        // to one array; rqbr's n3 column is PROVABLY always zero for the same reason and is dropped
        // outright):
        //   Stage 1 (kount+kr < n): selects a BASIS OF n OBSERVATIONS -- one structural variable
        //     pivoted into each of rows [0,n) in turn (row i's tag records WHICH coefficient), so at
        //     the end of stage 1 the fit interpolates those n points exactly (the vertex property --
        //     n residuals exactly zero -- that only this engine, not ladFN's interior point, can
        //     certify). A column with no candidate pivot row degenerates directly into position kr
        //     (rare; leaves that coefficient at its default 0).
        //   Stage 2: a full reduced-cost simplex over the DISPLACED observations now occupying columns
        //     [kr,n), restricted to leaving-row candidates among rows [n,m) (the free residuals). Its
        //     signature trick (verified against rqbr's own do-loop shape): the ratio test doesn't stop
        //     at the FIRST candidate row -- it walks every candidate in increasing-ratio order,
        //     "folding" (negating in place, label 23094) each one whose reduced-cost budget
        //     (cost[in] - 2*pivot) hasn't yet dropped to the pivot threshold, so ONE entering-column
        //     choice can cross many residuals' sign changes before landing on the true
        //     (weighted-median) breakpoint -- the mechanism behind the spec's ~O(n), not O(m),
        //     iteration count.
        // x is extracted from whichever rows [0,kl) are resolved at exit (label 80's formula,
        // generalized to run at ANY exit point -- Optimal, Unbounded, or a maxIter cutoff mid-stage --
        // so MaxIterations always returns a well-defined, finite partial iterate, never NaN/stale
        // data).
        //
        // Job-safe: all scratch is Allocator.Temp, disposed on every return path.
        // ============================================================================================

        /// <summary>
        /// Exact least-absolute-deviation (L1) regression via the Barrodale-Roberts specialized
        /// simplex (Barrodale &amp; Roberts 1973; the Koenker-d'Orey <c>rqbr</c> core). The second
        /// reformulation-free exact LAD engine in this library, alongside
        /// <see cref="ladFN(in floatMxN, in floatN, ref floatN, out double, int)"/>: a primal
        /// simplex worked directly on an m x n condensed tableau of the original design (no
        /// 2n+2m-variable LP reformulation), converging to an EXACT VERTEX -- at the optimum, n of
        /// the m residuals are exactly zero (see docs/spec-lad-barrodale-roberts.md test 4), a
        /// certificate ladFN's interior-point path only approaches. Iteration count is ~O(n)
        /// regardless of m (the weighted-median long step folds many residual sign-changes into a
        /// single entering-column choice) -- BR is competitive with or faster than ladFN at small-to-
        /// moderate m, ladFN wins at large m (Portnoy &amp; Koenker 1997's crossover).
        ///
        /// <see cref="objective"/> is the L1 residual ‖A x - b‖₁, recomputed from the returned x
        /// (honest, matching every other LAD entry point's convention -- never the reference's own
        /// internal running sum). <see cref="LP.lad"/>'s own default routing is UNCHANGED by this
        /// method -- call ladBR directly for the exact Barrodale-Roberts route.
        /// </summary>
        /// <param name="A">Design matrix, m×n (m observations, n coefficients). m ≥ n typical.</param>
        /// <param name="b">Observations, length m.</param>
        /// <param name="x">Output coefficients, length n (overwritten). May be negative.</param>
        /// <param name="objective">Output L1 residual ‖A x − b‖₁.</param>
        /// <param name="maxIter">Iteration budget; ≤0 picks a size-based default (10n+100 -- stage 1
        /// alone always needs n rounds, so unlike ladFN's fixed cap this must scale with n).</param>
        public static LPInfo ladBR(in floatMxN A, in floatN b, ref floatN x, out double objective, int maxIter = 0)
        {
            int m = A.M_Rows, n = A.N_Cols;

            if (b.N != m) throw new ArgumentException("LP.ladBR: b.N must equal A.M_Rows");
            if (x.N != n) throw new ArgumentException("LP.ladBR: x.N must equal A.N_Cols");

            return ladBarrodaleRobertsCore(in A, in b, 0.5, ref x, out objective, maxIter);
        }

        /// <summary>
        /// Quantile regression via the same Barrodale-Roberts specialized simplex: fits the
        /// conditional τ-quantile of b given A by minimizing the check loss
        /// Σᵢ ρ_τ(bᵢ − Aᵢ·x) over a FREE x, where ρ_τ(u) = u·(τ − 1[u&lt;0]). τ = 0.5 is median
        /// regression (identical fit to the τ-less
        /// <see cref="ladBR(in floatMxN, in floatN, ref floatN, out double, int)"/>). Exact and
        /// reformulation-free, same vertex-exact / O(n)-iteration behavior as the τ=0.5 overload.
        /// </summary>
        /// <param name="A">Design matrix, m×n (m observations, n coefficients). m ≥ n typical.</param>
        /// <param name="b">Observations, length m.</param>
        /// <param name="tau">Quantile level, strictly inside (0, 1). 0.5 = LAD/median.</param>
        /// <param name="x">Output coefficients, length n (overwritten). May be negative.</param>
        /// <param name="objective">Output ‖A x − b‖₁ at the returned fit (the plain L1 residual,
        /// reported for cross-method comparability; the τ-quantile fit does not minimize this
        /// unweighted sum unless τ = 0.5 -- same convention as ladFN's tau overload).</param>
        /// <param name="maxIter">Iteration budget; ≤0 picks the size-based default (10n+100).</param>
        public static LPInfo ladBR(in floatMxN A, in floatN b, double tau, ref floatN x,
                                   out double objective, int maxIter = 0)
        {
            int m = A.M_Rows, n = A.N_Cols;

            if (b.N != m) throw new ArgumentException("LP.ladBR: b.N must equal A.M_Rows");
            if (x.N != n) throw new ArgumentException("LP.ladBR: x.N must equal A.N_Cols");
            if (!(tau > 0.0 && tau < 1.0)) throw new ArgumentException("LP.ladBR: tau must be strictly inside (0, 1)");

            return ladBarrodaleRobertsCore(in A, in b, tau, ref x, out objective, maxIter);
        }

        // tau-parameterized core. internal (not private), mirroring ladFrischNewtonCore's own
        // visibility rationale (see that file's header) -- the public tau surface above is the
        // supported entry.
        internal static LPInfo ladBarrodaleRobertsCore(in floatMxN A, in floatN b, double tau,
                                                        ref floatN x, out double objective, int maxIter)
        {
            int m = A.M_Rows, n = A.N_Cols;

            var T = new floatMxN(m, n, Allocator.Temp);
            var rhs = new floatN(m, Allocator.Temp);
            var cost = new floatN(n, Allocator.Temp);
            var rowTag = new NativeArray<int>(m, Allocator.Temp);
            var colTag = new NativeArray<int>(n, Allocator.Temp);
            var candRow = new NativeArray<int>(m, Allocator.Temp);
            var candRatio = new floatN(m, Allocator.Temp);

            // ---- setup (rqbr's "23019" loop): copy A/b into the working tableau, tag row i with its
            // observation index n+i+1 (1-based semantics kept for the tag VALUES throughout, matching
            // the reference exactly, regardless of this port's 0-based array indexing), then normalize
            // every row to a nonnegative rhs (negating the whole row -- data AND tag -- when it
            // isn't). idxcf is always 0 in this port (no tau-path/CI), so rqbr's n3 perturbation column
            // (wa(i,n3) = tnew*a(i,idxcf)) is always zero and n1==n2 always -- both collapse into the
            // single `rhs` column here (see file header). ----
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++) T[i, j] = A[i, j];
                rhs[i] = b[i];
                rowTag[i] = n + i + 1;
                if (rhs[i] < (float)0)
                {
                    for (int j = 0; j < n; j++) T[i, j] = -T[i, j];
                    rhs[i] = -rhs[i];
                    rowTag[i] = -rowTag[i];
                }
            }

            // ---- setup (rqbr's "23035" loop): cost[j] = cost0[j] + costT[j]*tau, the tau-dependent
            // reduced-cost row (rqbr's wa(m1,*) after its own do23048 recombination). cost0/costT
            // (rqbr's wa(m2,*)/wa(m3,*)) are transient locals here -- this port's single fixed-tau pass
            // never re-derives cost[] from them again after this point (that only happens on rqbr's
            // tau-PATH continuation, not ported -- see file header), so there is nothing to gain from
            // persisting them as maintained rows. aux below reads the ALREADY row-negated T (matching
            // the reference's own setup order: row negation happens in the loop above, THIS loop reads
            // its result), and aux*sign(rowTag) recovers the ORIGINAL (pre-negation) A[i,j] -- the
            // negation and the sign-undo cancel, which is why costT[j] below equals 2x the ORIGINAL
            // column sum. ----
            for (int j = 0; j < n; j++)
            {
                colTag[j] = j + 1;
                double costConst = 0, costT = 0;
                for (int i = 0; i < m; i++)
                {
                    double aux = (double)T[i, j];
                    if (rowTag[i] < 0) { costConst += 2.0 * aux; costT -= aux; }
                    else costT += aux;
                }
                costT *= 2.0;
                cost[j] = (float)(costConst + costT * tau);
            }

            // Pivot/ratio tolerance -- see deviation 1 in the file header (this library's own
            // established convention, not the reference's literal eps^(2/3)).
            float pivTol = math.max(Consts.floatZeroThreshold, (float)1e-9);

            int kr = 0, kl = 0, kount = 0, iters = 0;
            bool stage = true;
            LPStatus status = LPStatus.Optimal;
            int budget = maxIter > 0 ? maxIter : 10 * n + 100;

            while (true)
            {
                if (iters >= budget) { status = LPStatus.MaxIterations; break; }

                int enter = -1;
                if (stage)
                {
                    // ---- stage 1 entering-column selection (label 30): among still-eligible columns
                    // (|colTag[j]| <= n -- i.e. not yet resolved) in [kr,n), the one with the largest
                    // |cost|. ----
                    float best = (float)(-1);
                    for (int j = kr; j < n; j++)
                    {
                        if (math.abs(colTag[j]) <= n)
                        {
                            float d = math.abs(cost[j]);
                            if (d > best) { best = d; enter = j; }
                        }
                    }
                    if (enter < 0) { status = LPStatus.MaxIterations; break; }   // defensive; loop invariant precludes this
                    if (cost[enter] < (float)0)
                    {
                        for (int i = 0; i < m; i++) T[i, enter] = -T[i, enter];
                        cost[enter] = -cost[enter];
                        colTag[enter] = -colTag[enter];
                    }
                }
                else
                {
                    // ---- stage 2 entering-column selection (label 23055): full reduced-cost search
                    // over [kr,n) with the "-2" free-variable folding trick (x is UNRESTRICTED in
                    // sign, so a column whose reduced cost sits in (-2,0) is not yet competitive but
                    // isn't excluded outright -- its FOLDED value -d-2 re-enters the max search). ----
                    float BIG = (float)1e30;
                    float best = -BIG;
                    for (int j = kr; j < n; j++)
                    {
                        float d = cost[j];
                        if (d < (float)0)
                        {
                            if (d > (float)(-2)) continue;
                            d = -d - (float)2;
                        }
                        if (d > best) { best = d; enter = j; }
                    }
                    if (best <= pivTol) break;    // stage-2 optimal (label 23054) -- status stays Optimal
                    if (cost[enter] <= (float)0)
                    {
                        for (int i = 0; i < m; i++) T[i, enter] = -T[i, enter];
                        cost[enter] = -cost[enter] - (float)2;
                        colTag[enter] = -colTag[enter];
                    }
                }

                iters++;

                // ---- ratio test + weighted-median long step (labels 23072/23079/10): collect every
                // candidate row in [kl,m) with a positive entry in the entering column, then repeatedly
                // take the SMALLEST remaining ratio -- either it's the true pivot point
                // (cost[enter] - 2*pivot <= pivTol) or it's a breakpoint the objective still improves
                // past, in which case that row is FOLDED (negated in place) and the search continues
                // among the rest, all within this ONE entering-column choice. ----
                int nCand = 0;
                for (int i = kl; i < m; i++)
                {
                    float d = T[i, enter];
                    if (d > pivTol) { candRow[nCand] = i; candRatio[nCand] = rhs[i] / d; nCand++; }
                }

                int leave = -1;
                while (nCand > 0)
                {
                    int pick = 0;
                    float minRatio = candRatio[0];
                    for (int k = 1; k < nCand; k++) if (candRatio[k] < minRatio) { minRatio = candRatio[k]; pick = k; }
                    int cand = candRow[pick];
                    candRow[pick] = candRow[nCand - 1]; candRatio[pick] = candRatio[nCand - 1]; nCand--;

                    float pivot = T[cand, enter];
                    if (cost[enter] - pivot - pivot <= pivTol) { leave = cand; break; }   // label 10's own test

                    // fold (label 23094): pass THROUGH this breakpoint without pivoting -- negate row
                    // `cand` in place (columns [kr,n) AND rhs AND its tag), and debit its contribution
                    // from the entering column's remaining reduced-cost budget.
                    for (int j = kr; j < n; j++)
                    {
                        float d = T[cand, j];
                        cost[j] -= d + d;
                        T[cand, j] = -d;
                    }
                    float dr = rhs[cand];
                    rhs[cand] = -dr;
                    rowTag[cand] = -rowTag[cand];
                }

                if (leave < 0)
                {
                    if (stage)
                    {
                        // ---- degenerate column exchange (label 23081): no candidate row at all for
                        // this entering column -- swap it directly into position kr without a real
                        // pivot (that coefficient never gets a row assigned; x[.] stays 0 for it). ----
                        for (int i = 0; i < m; i++) { float tmp = T[i, kr]; T[i, kr] = T[i, enter]; T[i, enter] = tmp; }
                        float tc = cost[kr]; cost[kr] = cost[enter]; cost[enter] = tc;
                        int ttag = colTag[kr]; colTag[kr] = colTag[enter]; colTag[enter] = ttag;
                        kr++;
                    }
                    else
                    {
                        // ---- label 23047 (rqbr's ift=2, "premature end -- possible conditioning
                        // problem"): see file header deviation 2 for the LPStatus.Unbounded mapping and
                        // why x is still extracted below rather than discarded. ----
                        status = LPStatus.Unbounded;
                        break;
                    }
                }
                else
                {
                    // ---- the actual pivot (label 10) ----
                    BRPivot(T, rhs, cost, rowTag, colTag, m, n, kr, leave, enter);
                    kount++;
                    if (stage)
                    {
                        if (leave != kount - 1)
                        {
                            for (int j = kr; j < n; j++) { float tmp = T[leave, j]; T[leave, j] = T[kount - 1, j]; T[kount - 1, j] = tmp; }
                            float tr = rhs[leave]; rhs[leave] = rhs[kount - 1]; rhs[kount - 1] = tr;
                            int trg = rowTag[leave]; rowTag[leave] = rowTag[kount - 1]; rowTag[kount - 1] = trg;
                        }
                        kl++;
                    }
                }

                if (stage && kount + kr == n) stage = false;   // stage 1 complete -> stage 2 (labels 23057/23052)
            }

            // ---- extract solution (label 80's x(k) formula, generalized to run at ANY exit point:
            // rows [0,kl) are exactly the ones a structural variable has been resolved into so far --
            // a full n of them on Optimal/Unbounded (stage 1 always finishes before stage 2 can run),
            // fewer on a maxIter cutoff mid-stage-1, in which case every unresolved x[j] simply keeps
            // its zero-initialized value. ----
            for (int j = 0; j < n; j++) x[j] = (float)0;
            for (int i = 0; i < kl; i++)
            {
                int k = math.abs(rowTag[i]) - 1;
                float sgn = rowTag[i] < 0 ? -(float)1 : (float)1;
                x[k] = rhs[i] * sgn;
            }

            double obj = 0;
            for (int i = 0; i < m; i++)
            {
                double rowDot = 0;
                for (int j = 0; j < n; j++) rowDot += (double)A[i, j] * (double)x[j];
                obj += math.abs(rowDot - (double)b[i]);
            }
            objective = obj;

            T.Dispose(); rhs.Dispose(); cost.Dispose();
            rowTag.Dispose(); colTag.Dispose(); candRow.Dispose(); candRatio.Dispose();

            return new LPInfo { status = status, iterations = iters, objective = obj };
        }

        // Gauss-Jordan pivot at (leave, enter) -- rqbr's label 10, shared by both stages. Normalizes
        // row `leave` (excluding column `enter`), eliminates column `enter` from every OTHER row
        // (observation rows AND the cost row, matching the reference's own i=1..m3 elimination sweep --
        // see file header), then swaps the leaving row's tag with the entering column's tag.
        static void BRPivot(floatMxN T, floatN rhs, floatN cost, NativeArray<int> rowTag, NativeArray<int> colTag,
                            int m, int n, int kr, int leave, int enter)
        {
            float pivot = T[leave, enter];

            for (int j = kr; j < n; j++) if (j != enter) T[leave, j] /= pivot;
            rhs[leave] /= pivot;

            for (int i = 0; i < m; i++)
            {
                if (i == leave) continue;
                float d = T[i, enter];
                for (int j = kr; j < n; j++) if (j != enter) T[i, j] -= d * T[leave, j];
                rhs[i] -= d * rhs[leave];
                T[i, enter] = -d / pivot;
            }

            float dCost = cost[enter];
            for (int j = kr; j < n; j++) if (j != enter) cost[j] -= dCost * T[leave, j];
            cost[enter] = -dCost / pivot;

            T[leave, enter] = (float)1 / pivot;

            int tmp = rowTag[leave]; rowTag[leave] = colTag[enter]; colTag[enter] = tmp;
        }
    }
}
