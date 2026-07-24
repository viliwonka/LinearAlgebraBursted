using System;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    public static partial class LP
    {
        // ============================================================================================
        // LICENSE: ported from `rqbr.f` (R quantreg, GPL >= 2); distributed here under this
        // package's MIT license with the original author's permission. See "Third Party Notices.md"
        // in the package root.
        //
        // Barrodale-Roberts specialized-simplex EXACT least-absolute-deviation (L1) / quantile-
        // regression solver. Port of the Koenker-d'Orey Fortran `rqbr` (R `quantreg` package,
        // src/rqbr.f), itself derived from Barrodale & Roberts 1973 ("An improved algorithm for
        // discrete l1 linear approximation", SIAM J. Numer. Anal. 10(5), 839-848). The second
        // reformulation-free exact LAD engine (LP.ladBR), alongside Frisch-Newton (LP.ladFN,
        // LP.FrischNewton.fProxy.cs).
        //
        // This file implements the core simplex only -- rqbr's confidence-interval machinery and its
        // tau-path continuation are not ported (unreachable in LP.ladBR's fixed-tau, CIs-off mode).
        //
        // Deviations from the literal reference: (1) uses this library's own simplex ratio/pivot
        // tolerance (pivTol below) instead of rqbr's tighter toler=eps^(2/3); (2) on
        // "premature end" (ift=2), extracts the last-vertex structural solution instead of leaving x
        // untouched, matching LPStatus.Unbounded's own contract; (3) the "solution may be nonunique"
        // warning (ift=1) is not surfaced, since LPStatus has no such state; (4) diagnostic-only
        // reference outputs are dropped -- objective is honestly recomputed from the returned x.
        //
        // Algorithm shape: stage 1 pivots one structural variable into each of n rows so the fit
        // interpolates those n points exactly (a vertex property ladFN's interior point can only
        // approach); stage 2 runs a reduced-cost simplex that folds many residuals' sign changes into
        // a single entering-column choice, giving ~O(n) iterations regardless of m. x is extracted
        // from whatever rows are resolved at exit (Optimal, Unbounded, or a maxIter cutoff), so
        // MaxIterations always returns a well-defined, finite partial iterate.
        //
        // Job-safe: all scratch is Allocator.Temp, disposed on every return path.
        // ============================================================================================

        /// <summary>
        /// Exact least-absolute-deviation (L1) regression via the Barrodale-Roberts specialized
        /// simplex (Barrodale &amp; Roberts 1973; the Koenker-d'Orey <c>rqbr</c> core). The second
        /// reformulation-free exact LAD engine in this library, alongside
        /// <see cref="ladFN(in fProxyMxN, in fProxyN, ref fProxyN, out double, int)"/>: a primal
        /// simplex worked directly on an m x n condensed tableau of the original design (no
        /// 2n+2m-variable LP reformulation), converging to an EXACT VERTEX -- at the optimum, n of
        /// the m residuals are exactly zero, a certificate ladFN's interior-point path only
        /// approaches. Iteration count is ~O(n)
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
        public static LPInfo ladBR(in fProxyMxN A, in fProxyN b, ref fProxyN x, out double objective, int maxIter = 0)
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
        /// <see cref="ladBR(in fProxyMxN, in fProxyN, ref fProxyN, out double, int)"/>). Exact and
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
        public static LPInfo ladBR(in fProxyMxN A, in fProxyN b, double tau, ref fProxyN x,
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
        internal static LPInfo ladBarrodaleRobertsCore(in fProxyMxN A, in fProxyN b, double tau,
                                                        ref fProxyN x, out double objective, int maxIter)
        {
            int m = A.M_Rows, n = A.N_Cols;

            var T = new fProxyMxN(m, n, Allocator.Temp);
            var rhs = new fProxyN(m, Allocator.Temp);
            var cost = new fProxyN(n, Allocator.Temp);
            var rowTag = new NativeArray<int>(m, Allocator.Temp);
            var colTag = new NativeArray<int>(n, Allocator.Temp);
            var candRow = new NativeArray<int>(m, Allocator.Temp);
            var candRatio = new fProxyN(m, Allocator.Temp);

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
                if (rhs[i] < (fProxy)0)
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
                cost[j] = (fProxy)(costConst + costT * tau);
            }

            // Pivot/ratio tolerance -- see deviation 1 in the file header (this library's own
            // established convention, not the reference's literal eps^(2/3)).
            fProxy pivTol = math.max(Consts.fProxyZeroThreshold, (fProxy)1e-9);

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
                    fProxy best = (fProxy)(-1);
                    for (int j = kr; j < n; j++)
                    {
                        if (math.abs(colTag[j]) <= n)
                        {
                            fProxy d = math.abs(cost[j]);
                            if (d > best) { best = d; enter = j; }
                        }
                    }
                    if (enter < 0) { status = LPStatus.MaxIterations; break; }   // defensive; loop invariant precludes this
                    if (cost[enter] < (fProxy)0)
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
                    fProxy BIG = (fProxy)1e30;
                    fProxy best = -BIG;
                    for (int j = kr; j < n; j++)
                    {
                        fProxy d = cost[j];
                        if (d < (fProxy)0)
                        {
                            if (d > (fProxy)(-2)) continue;
                            d = -d - (fProxy)2;
                        }
                        if (d > best) { best = d; enter = j; }
                    }
                    if (best <= pivTol) break;    // stage-2 optimal (label 23054) -- status stays Optimal
                    if (cost[enter] <= (fProxy)0)
                    {
                        for (int i = 0; i < m; i++) T[i, enter] = -T[i, enter];
                        cost[enter] = -cost[enter] - (fProxy)2;
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
                // Column-strided read: T[i,enter] is a fixed COLUMN scanned over varying rows i, stride
                // n in this row-major tableau (the reference Fortran's own wa(i,in) would have been
                // unit-stride under Fortran's column-major convention -- this port's row-major storage
                // inverts that). This collection pass runs once per entering-column choice (<= iters
                // times total, NOT once per fold -- see below).
                int nCand = 0;
                for (int i = kl; i < m; i++)
                {
                    fProxy d = T[i, enter];
                    if (d > pivTol) { candRow[nCand] = i; candRatio[nCand] = rhs[i] / d; nCand++; }
                }

                // Candidate consumption: process candidates in ASCENDING ratio order until the terminal
                // pivot test fires; every non-terminal candidate FOLDS (see TryPivotOrFold) and the scan
                // continues. `iters` only counts real PIVOTS (incremented above, once per entering-
                // column choice), not folds -- so a single entering-column choice can fold arbitrarily
                // many of the nCand candidates before landing on its pivot, and the flat, size-
                // independent iters count BR is known for (~O(n) pivots regardless of m) can hide an
                // unbounded amount of fold work behind it.
                //
                // The ORIGINAL algorithm (kept below, UNCHANGED, for nCand <= CandSortThreshold) finds
                // this order via an O(nCand^2) selection sort, which becomes the dominant cost at large
                // m. Above BR_CAND_SORT_THRESHOLD, sort the candidates ONCE (UnsafeOP.sortByKeyAscending,
                // O(nCand log nCand) heapsort) and walk them in a single linear pass instead -- same
                // visitation order as the original whenever ratios are distinct.
                int leave = -1;
                if (nCand > BR_CAND_SORT_THRESHOLD)
                {
                    unsafe { UnsafeOP.sortByKeyAscending(candRatio.Data.Ptr, (int*)NativeArrayUnsafeUtility.GetUnsafePtr(candRow), nCand); }
                    for (int p = 0; p < nCand; p++)
                    {
                        int cand = candRow[p];
                        if (TryPivotOrFold(T, rhs, cost, rowTag, n, kr, enter, cand, pivTol)) { leave = cand; break; }
                    }
                }
                else
                {
                    while (nCand > 0)
                    {
                        int pick = 0;
                        fProxy minRatio = candRatio[0];
                        for (int k = 1; k < nCand; k++) if (candRatio[k] < minRatio) { minRatio = candRatio[k]; pick = k; }
                        int cand = candRow[pick];
                        candRow[pick] = candRow[nCand - 1]; candRatio[pick] = candRatio[nCand - 1]; nCand--;

                        if (TryPivotOrFold(T, rhs, cost, rowTag, n, kr, enter, cand, pivTol)) { leave = cand; break; }
                    }
                }

                if (leave < 0)
                {
                    if (stage)
                    {
                        // ---- degenerate column exchange (label 23081): no candidate row at all for
                        // this entering column -- swap it directly into position kr without a real
                        // pivot (that coefficient never gets a row assigned; x[.] stays 0 for it). ----
                        for (int i = 0; i < m; i++) { fProxy tmp = T[i, kr]; T[i, kr] = T[i, enter]; T[i, enter] = tmp; }
                        fProxy tc = cost[kr]; cost[kr] = cost[enter]; cost[enter] = tc;
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
                            for (int j = kr; j < n; j++) { fProxy tmp = T[leave, j]; T[leave, j] = T[kount - 1, j]; T[kount - 1, j] = tmp; }
                            fProxy tr = rhs[leave]; rhs[leave] = rhs[kount - 1]; rhs[kount - 1] = tr;
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
            for (int j = 0; j < n; j++) x[j] = (fProxy)0;
            for (int i = 0; i < kl; i++)
            {
                int k = math.abs(rowTag[i]) - 1;
                fProxy sgn = rowTag[i] < 0 ? -(fProxy)1 : (fProxy)1;
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

        // Processes ONE ratio-test candidate row `cand` for the current entering column: either it is
        // the terminal pivot (the reduced-cost budget cost[enter] has dropped to the pivot threshold,
        // label 10's own test) and the caller should stop and pivot there (returns true), or it is a
        // breakpoint the objective still improves past and gets FOLDED in place -- negated across
        // columns [kr,n), its rhs, and its tag, and its contribution debited from cost[enter] -- so the
        // scan can continue (returns false). Factored out so the small-nCand (linear-scan) and
        // large-nCand (pre-sorted) candidate-consumption paths above share this ONE definition of what
        // "process a candidate" means, rather than the fold logic being duplicated between them.
        static bool TryPivotOrFold(fProxyMxN T, fProxyN rhs, fProxyN cost, NativeArray<int> rowTag,
                                   int n, int kr, int enter, int cand, fProxy pivTol)
        {
            fProxy pivot = T[cand, enter];
            if (cost[enter] - pivot - pivot <= pivTol) return true;   // label 10's own test -- this row leaves

            // fold (label 23094): pass THROUGH this breakpoint without pivoting -- negate row `cand`
            // in place (columns [kr,n) AND rhs AND its tag), and debit its contribution from the
            // entering column's remaining reduced-cost budget.
            for (int j = kr; j < n; j++)
            {
                fProxy d = T[cand, j];
                cost[j] -= d + d;
                T[cand, j] = -d;
            }
            fProxy dr = rhs[cand];
            rhs[cand] = -dr;
            rowTag[cand] = -rowTag[cand];
            return false;
        }

        // Gauss-Jordan pivot at (leave, enter) -- rqbr's label 10, shared by both stages. Normalizes
        // row `leave` (excluding column `enter`), eliminates column `enter` from every OTHER row
        // (observation rows and the cost row), then swaps the leaving row's tag with the entering
        // column's tag. Column `enter` is a bookkeeping column with its own -d/pivot formula (not the
        // neighbor update), so the elimination is split into the exact ranges [kr,enter) and
        // (enter,n), routed through UnsafeOP.scalDiv/axpy for vectorization -- bit-identical to a
        // branchy per-column loop.
        static unsafe void BRPivot(fProxyMxN T, fProxyN rhs, fProxyN cost, NativeArray<int> rowTag, NativeArray<int> colTag,
                            int m, int n, int kr, int leave, int enter)
        {
            fProxy pivot = T[leave, enter];
            fProxy* Tp = T.Data.Ptr;
            fProxy* leaveRow = Tp + (long)leave * n;
            int lenLo = enter - kr;        // [kr, enter)
            int lenHi = n - enter - 1;     // (enter, n)

            if (lenLo > 0) UnsafeOP.scalDiv(leaveRow + kr, lenLo, pivot);
            if (lenHi > 0) UnsafeOP.scalDiv(leaveRow + enter + 1, lenHi, pivot);
            rhs[leave] /= pivot;

            for (int i = 0; i < m; i++)
            {
                if (i == leave) continue;
                fProxy* rowI = Tp + (long)i * n;
                fProxy d = rowI[enter];
                if (lenLo > 0) UnsafeOP.axpy(rowI + kr, leaveRow + kr, -d, lenLo);
                if (lenHi > 0) UnsafeOP.axpy(rowI + enter + 1, leaveRow + enter + 1, -d, lenHi);
                rhs[i] -= d * rhs[leave];
                rowI[enter] = -d / pivot;
            }

            fProxy dCost = cost[enter];
            fProxy* costP = cost.Data.Ptr;
            if (lenLo > 0) UnsafeOP.axpy(costP + kr, leaveRow + kr, -dCost, lenLo);
            if (lenHi > 0) UnsafeOP.axpy(costP + enter + 1, leaveRow + enter + 1, -dCost, lenHi);
            cost[enter] = -dCost / pivot;

            T[leave, enter] = (fProxy)1 / pivot;

            int tmp = rowTag[leave]; rowTag[leave] = colTag[enter]; colTag[enter] = tmp;
        }
    }
}
