#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // ================================================================================================
    // Bounded-variable DUAL revised simplex -- the LPMethod.DualSimplex backend, stage 2 of the
    // HiGHS-style dense revised-simplex port (docs/spec-revised-simplex.md). Builds on stage 1's kernel
    // layer (LP.RevisedSimplex.double.cs: Refactorize/Ftran/Btran/the eta-apply pair/RebuildXB) instead
    // of duplicating it -- ordinary partial-class member resolution, same generated type calls same
    // generated type (float calls float, double calls double), exactly like interiorCore already reuses
    // Amul/ATmul across this same partial class. Where the primal keeps a primal-feasible basis and
    // hunts for dual feasibility, the dual simplex keeps a DUAL-feasible basis and hunts for primal
    // feasibility: choose an infeasible BASIC row (dual steepest edge picks which), price the pivot ROW
    // (BTRAN + O(mn)), then a ratio test over nonbasic COLUMNS picks the entering variable so dual
    // feasibility survives the pivot. HiGHS is (again) the reference: dual steepest edge (Forrest-
    // Goldfarb) pricing, a long-step (bound-flipping) Harris ratio test, artificial-bounds dual phase 1,
    // cost perturbation, and -- the signature HiGHS composition -- handing the terminal dual basis to
    // the PRIMAL core (stage 1) as a cleanup pass once real bounds are restored, rather than writing a
    // second cleanup algorithm.
    //
    // Numerics: ordinary double throughout (matrix/vector storage, the ratio test, DSE weights), exactly
    // like stage 1 -- see that file's header for the full rationale (reuses the library's own LU/Blas;
    // tolerances derived from Consts.doubleZeroThreshold/doubleEpsilon, the SAME expressions stage 1
    // uses, computed INLINE in DualSimplexCore rather than via a shared helper method -- an double-
    // returning helper with no double-typed parameter differs only in return type between the float- and
    // double-generated fragments of this partial class, which C# does not overload on, so it would
    // collide as a duplicate member; see stage 1's file header for the same pitfall). The DSE weight
    // floor (1e-4) is HiGHS's own literal constant (kMinDualSteepestEdgeWeight), not tolerance-derived,
    // so it stays the same literal for both dtypes -- see below.
    //
    // DSE update formula verified line-by-line against HiGHS source (not just the spec's paraphrase):
    // https://github.com/ERGO-Code/HiGHS  highs/simplex/HEkk.cpp::updateDualSteepestEdgeWeights (the
    // `dual_edge_weight_[iRow] += aa_iRow*(new_pivotal_edge_weight*aa_iRow + Kai*dse_array_value)` line)
    // called from highs/simplex/HEkkDual.cpp::updatePrimal with `Kai = -2/alpha_col`,
    // `new_pivotal_edge_weight = edge_weight[row_out]/alpha_col^2`, and the DSE array itself built as
    // `col_DSE = Ftran(row_ep)` i.e. tau = B^-1 rho_r -- exactly the spec's
    // `w_i' = w_i - 2(alpha_qi/alpha_qr)*tau_i + (alpha_qi/alpha_qr)^2*w_r, then w_r' = w_r/alpha_qr^2`.
    // The 1e-4 floor is HiGHS's `kMinDualSteepestEdgeWeight` (highs/simplex/SimplexConst.h).
    // ================================================================================================
    public static partial class LP
    {
        // ---- double-typed entry point: builds the computational form (reusing stage 1's
        // BuildComputationalForm verbatim -- same partial class, same per-dtype fragment), hands off to
        // the dual core, copies the result back. ----
        static LPInfo dualSimplexCore(in doubleMxN A, in doubleN b, in doubleN c,
                                     in NativeArray<ConstraintSense> senses,
                                     ref doubleN x, out double objective, int maxIter)
        {
            int m = A.M_Rows, n = A.N_Cols, N = n + m;

            var M = new doubleMxN(m, N, Allocator.Temp);
            var lower = new doubleN(N, Allocator.Temp);
            var upper = new doubleN(N, Allocator.Temp);
            var cost = new doubleN(N, Allocator.Temp);
            var rhs = new doubleN(m, Allocator.Temp);

            BuildComputationalForm(in A, in b, in c, in senses, M, lower, upper, cost, rhs, m, n, N);

            var xFull = new doubleN(N, Allocator.Temp);
            var info = DualSimplexCore(M, lower, upper, cost, rhs, m, n, N, maxIter, xFull);

            for (int j = 0; j < n; j++) x[j] = xFull[j];

            double obj = 0;
            for (int j = 0; j < n; j++) obj += (double)c[j] * (double)xFull[j];
            objective = obj;
            info.objective = obj;

            M.Dispose(); lower.Dispose(); upper.Dispose(); cost.Dispose(); rhs.Dispose(); xFull.Dispose();
            return info;
        }

        // Prices alpha_r[j] = dot(M[:,j], rho) over every NONBASIC column j (basic columns left 0,
        // unused) -- the O(mn) PRICE step of the dual iteration's pivot row (rho = B^-T e_r via BTRAN).
        internal static void PriceRow(doubleMxN M, doubleN rho, NativeArray<byte> status, int N, int m, doubleN alphaRow)
        {
            for (int j = 0; j < N; j++)
            {
                if (status[j] == STATUS_BASIC) { alphaRow[j] = (double)0; continue; }
                double s = (double)0;
                for (int i = 0; i < m; i++) s += M[i, j] * rho[i];
                alphaRow[j] = s;
            }
        }

        // Prices reduced costs d_j = cost[j] - dot(M[:,j], y) over every nonbasic column j (y = B^-T c_B
        // via BTRAN of the basic costs) -- the dual ratio test's numerator. Mirrors the pricing half of
        // stage 1's SelectEntering but fills the WHOLE array (the ratio test needs every sign-correct
        // candidate's d_j, not just the single best one).
        internal static void PriceReducedCosts(doubleMxN M, doubleN y, doubleN cost, NativeArray<byte> status, int N, int m, doubleN dj)
        {
            for (int j = 0; j < N; j++)
            {
                if (status[j] == STATUS_BASIC) { dj[j] = (double)0; continue; }
                double d = cost[j];
                for (int i = 0; i < m; i++) d -= M[i, j] * y[i];
                dj[j] = d;
            }
        }

        // Dual ratio test: Harris two-pass + bound-flipping ratio test (BFRT), the dual analogue of
        // stage 1's HarrisRatioTest. Leaving row r is already chosen; `needIncrease` says which bound
        // its basic variable is heading for (true: below lower, moving up; false: above upper, moving
        // down). alphaRow[j] = rho_r . a_j (the PRICE step); dj[j] the reduced cost.
        //
        // alphaHat_j = s*alphaRow[j] (s = -1 if needIncrease else +1) makes the sign condition uniform:
        // AtLower wants alphaHat_j > pivTol, AtUpper wants alphaHat_j < -pivTol (see derivation in the
        // spec / this file's accompanying notes). Ratio d_j/|alphaHat_j| >= 0 for every candidate since
        // dual feasibility is an invariant (up to dualTol noise).
        //
        // A REAL pivot happens every time this returns with anyCandidate=true -- BFRT flips are only ever
        // a PREFIX of the walk, never a substitute for the final pivot (verified against HiGHS's
        // HEkkDualRow::chooseFinal: it always selects a workPivot and FTRANs its column via updateFtran,
        // unconditionally; the flip list from updateFlip is an ADDITIONAL, separate side effect applied
        // alongside that pivot, never instead of one). This matters beyond style: a "flip-only" iteration
        // would leave the basis (hence y = B^-T c_B, hence every dj) COMPLETELY UNCHANGED, so a column
        // just flipped from AtLower to AtUpper would still carry its OLD (AtLower-side) reduced cost --
        // generally the WRONG sign for its new status -- with no future iteration ever positioned to
        // notice, since dj is only ever re-examined when a column happens to be priced as a candidate for
        // SOME row again. Only an actual pivot changes cB and therefore y, which is what makes a flipped
        // column's stale dj become the mathematically CORRECT (fresh-BTRAN-recomputed) value next
        // iteration. (An earlier version of this method allowed flips to fully resolve a row with no
        // pivot; it passed every test at n<=24 but produced a false Infeasible on a 48-variable random
        // instance -- exactly the failure mode this derivation predicts, since the corruption needs
        // enough iterations/columns to surface.)
        //
        // Walks candidates (alphaHat_j = s*alphaRow[j], s=-1 if needIncrease else +1) in ascending
        // d_j/|alphaHat_j| ratio order (ties broken toward the largest |alphaHat|, mirroring the primal
        // Harris test's stability rule). A BOXED candidate (finite range) whose full flip would STILL
        // leave row r's infeasibility positive is flipped (absorbing |alphaHat_j|*(u_j-l_j) of it) and
        // the walk continues; the first candidate that is either NOT boxed, or boxed but sufficient to
        // finish absorbing the remaining infeasibility on its own, becomes the actual entering pivot
        // (never flipped, even if boxed). No sign-correct candidate at all, or the combined capacity of
        // every candidate still falling short of row r's infeasibility, -> primal Infeasible (dual
        // unbounded), reported via anyCandidate = false.
        internal static void DualRatioTest(doubleN alphaRow, doubleN dj,
                                           NativeArray<byte> status, doubleN lower, doubleN upper,
                                           int N, bool needIncrease, double pivTol, double infeasR, double feasTol,
                                           NativeArray<int> flipCols, out int flipCount,
                                           out int enterCol, out double thetaD, out bool anyCandidate)
        {
            double s = needIncrease ? (double)(-1) : (double)1;
            flipCount = 0; enterCol = -1; thetaD = (double)0; anyCandidate = false;

            var considered = new NativeArray<bool>(N, Allocator.Temp);   // true = excluded from candidacy
            for (int j = 0; j < N; j++)
            {
                if (status[j] == STATUS_BASIC || upper[j] - lower[j] <= (double)1e-13) { considered[j] = true; continue; }
                double ah = s * alphaRow[j];
                bool cand = (status[j] == STATUS_AT_LOWER && ah > pivTol) || (status[j] == STATUS_AT_UPPER && ah < -pivTol);
                considered[j] = !cand;
                if (cand) anyCandidate = true;
            }

            if (!anyCandidate) { considered.Dispose(); return; }

            double remaining = infeasR;
            while (true)
            {
                int best = -1; double bestRatio = (double)0; double bestAbsA = (double)0;
                for (int j = 0; j < N; j++)
                {
                    if (considered[j]) continue;
                    double ah = s * alphaRow[j];
                    double djr = math.max(status[j] == STATUS_AT_LOWER ? dj[j] : -dj[j], (double)0);
                    double ratio = djr / math.abs(ah);
                    if (best < 0 || ratio < bestRatio - (double)1e-9 ||
                        (ratio <= bestRatio + (double)1e-9 && math.abs(ah) > bestAbsA))
                    { best = j; bestRatio = ratio; bestAbsA = math.abs(ah); }
                }
                if (best < 0) { anyCandidate = false; break; }   // exhausted: combined capacity insufficient

                considered[best] = true;
                bool boxed = upper[best] - lower[best] < (double)1e29;
                double fullAbsorb = bestAbsA * (upper[best] - lower[best]);

                if (boxed && remaining - fullAbsorb > feasTol)
                {
                    flipCols[flipCount++] = best;
                    remaining -= fullAbsorb;
                    continue;
                }

                enterCol = best; thetaD = bestRatio;   // terminal candidate: the ACTUAL pivot, never a flip
                break;
            }

            considered.Dispose();
        }

        // Top-level driver: dual-feasibility precondition (+ artificial-bounds dual phase 1 folded in,
        // since at the all-logical start y=0 makes both coincide -- see below), dual steepest-edge
        // pricing + long-step Harris ratio test to drive primal feasibility while staying dual feasible
        // throughout, then restore real bounds and hand the terminal basis to stage 1's primal core as
        // a cleanup pass (removes the cost perturbation's effect and fixes any primal infeasibility left
        // by the bound restoration -- the HiGHS composition, see file header).
        internal static LPInfo DualSimplexCore(doubleMxN M, doubleN lower, doubleN upper,
                                               doubleN cost, doubleN rhs, int m, int n, int N,
                                               int maxIter, doubleN xFull)
        {
            // Same per-dtype tolerance expressions as stage 1 (see that file's header for why these are
            // inlined rather than a shared helper method).
            double feasTol = (double)math.max(math.sqrt((double)Consts.doubleEpsilon), 1e-7);
            double dualTol = feasTol;
            double pivTolFloor = math.max(Consts.doubleZeroThreshold, (double)1e-9);
            double weightFloor = (double)1e-4;
            double realINF = (double)1e30;

            // Artificial-bounds dual phase 1 (spec) uses a FIXED [0, 1e7] box in HiGHS, which is tuned
            // for HiGHS's own internally-SCALED (equilibrated) problem data. This solver does not scale,
            // so a literal 1e7 against O(1) problem data is a scale mismatch that is harmless for double
            // (headroom to spare) but catastrophic for float: RebuildXB's adjusted rhs sums the artificial
            // bound's contribution over every simultaneously-artificial column (up to ~n/2 of them for a
            // mixed-sign-cost problem), landing xB around -(artificialBound * n/2 * |A|) -- at 1e7 with
            // n~48 that is order 1e8, and float's ~1.19e-7 relative precision at that magnitude is an
            // ABSOLUTE error of order 10, which swamps feasTol (~3.45e-4) outright and was observed to
            // produce a false Infeasible within the first few dual iterations. Scaling the artificial
            // bound to the PROBLEM's own data magnitude (matching simplexCore's bScale convention) keeps
            // it proportionally huge -- 100x the largest |cost|/|rhs| entry, still far beyond where any
            // genuine optimum would sit for a well-posed LP -- while keeping the induced xB magnitude,
            // and hence the float rounding it carries, well under feasTol.
            double dataScale = 1.0;
            for (int i = 0; i < m; i++) dataScale = math.max(dataScale, math.abs((double)rhs[i]));
            for (int j = 0; j < n; j++) dataScale = math.max(dataScale, math.abs((double)cost[j]));
            double artificialBound = (double)(100.0 * dataScale);

            var basis = new NativeArray<int>(m, Allocator.Temp);
            var status = new NativeArray<byte>(N, Allocator.Temp);
            var xB = new doubleN(m, Allocator.Temp);
            var perturbedCost = new doubleN(N, Allocator.Temp);
            var isArtificial = new NativeArray<bool>(n, Allocator.Temp);

            for (int i = 0; i < m; i++) { basis[i] = n + i; status[n + i] = STATUS_BASIC; }
            for (int j = 0; j < n; j++) status[j] = STATUS_AT_LOWER;

            // HiGHS-style cost perturbation (degeneracy defence, deterministic seed): <= 1e-5*(1+|c_j|).
            // Deterministic per-column pseudo-random unit value in (-1, 1) via a cheap integer hash
            // (MurmurHash3 finalizer mix) of the column index -- inlined rather than a standalone helper
            // for the same return-type-collision reason as the tolerances above (an double-returning
            // helper with only an `int` parameter would differ solely in return type between the float-
            // and double-generated fragments).
            for (int j = 0; j < N; j++)
            {
                uint h = (uint)j * 2654435761u + 0x9E3779B9u;
                h ^= h >> 15; h *= 0x85EBCA6Bu;
                h ^= h >> 13; h *= 0xC2B2AE35u;
                h ^= h >> 16;
                double unit = (double)(h * (1.0 / 4294967295.0) * 2.0 - 1.0);
                perturbedCost[j] = cost[j] + unit * (double)1e-5 * ((double)1 + math.abs(cost[j]));
            }

            // Dual-feasibility precondition from the all-logical basis: c_B = 0 here, so y = B^-T c_B =
            // 0 and d_j = cost[j] for every nonbasic (only structurals are nonbasic at this basis). A
            // negative d_j needs status AtUpper to be dual-feasible, but structurals' real upper is +INF
            // -- give it the artificial-bounds dual-phase-1 box instead, which makes the flip possible.
            // At THIS specific starting basis, "flip to the matching bound" and "artificial-bounds phase
            // 1" are the same pass (spec: both derive purely from cost sign since y=0), so they are
            // folded into one loop rather than run as two separate stages.
            //
            // Uses the ORIGINAL cost, not perturbedCost: this is a one-time TRUE-dual-feasibility
            // decision, not part of the iterative pricing the perturbation is meant to stabilize. Using
            // perturbedCost here was an actual bug -- a column with cost[j] EXACTLY 0 (e.g. every x+/x-
            // column in LP.lad's reformulation, which has none of its own cost) is dual-feasible as-is
            // (d_j=0 trivially satisfies d_j>=0), but the perturbation's random sign could nudge it
            // slightly negative and give it a pointless artificial bound; multiplied across the MANY
            // exactly-zero-cost columns LP.lad's [x+|x-] block always has, this corrupted the warm-started
            // basis handed to the primal cleanup badly enough to report a false Unbounded.
            for (int j = 0; j < n; j++)
            {
                if (cost[j] < -dualTol)
                {
                    upper[j] = artificialBound;
                    isArtificial[j] = true;
                    status[j] = STATUS_AT_UPPER;
                }
            }

            var B = new doubleMxN(m, m, Allocator.Temp);
            var P = new Pivot(m, Allocator.Temp);
            var etaAlpha = new doubleMxN(REFACTOR_INTERVAL, m, Allocator.Temp);
            var etaRow = new NativeArray<int>(REFACTOR_INTERVAL, Allocator.Temp);
            int etaCount = 0;

            var y = new doubleN(m, Allocator.Temp);
            var cB = new doubleN(m, Allocator.Temp);
            var dj = new doubleN(N, Allocator.Temp);
            var rho = new doubleN(m, Allocator.Temp);
            var tau = new doubleN(m, Allocator.Temp);
            var alphaRow = new doubleN(N, Allocator.Temp);
            var alphaCol = new doubleN(m, Allocator.Temp);
            var flipCols = new NativeArray<int>(N, Allocator.Temp);
            var flipRHS = new doubleN(m, Allocator.Temp);

            // DSE weights: w_i = 1 exactly at the all-logical basis (spec), maintained (never reset)
            // across refactorizations for the rest of this run.
            var weight = new doubleN(m, Allocator.Temp);
            for (int i = 0; i < m; i++) weight[i] = (double)1;

            LPStatus resultStatus = LPStatus.Optimal;
            int iters = 0;

            bool ok = Refactorize(M, basis, B, ref P, m, N);
            if (!ok) resultStatus = LPStatus.MaxIterations;
            else RebuildXB(M, rhs, status, lower, upper, B, in P, m, N, xB);

            int budget = maxIter > 0 ? maxIter : 50 * (m + N) + 200;

            while (resultStatus == LPStatus.Optimal)
            {
                if (iters >= budget) { resultStatus = LPStatus.MaxIterations; break; }

                // ---- leaving row: dual steepest edge, r = argmax infeas_r^2 / w_r ----
                int r = -1; double bestScore = (double)(-1); bool rNeedsIncrease = false; double infeasR = (double)0;
                for (int i = 0; i < m; i++)
                {
                    int v = basis[i];
                    double xi = xB[i];
                    double infeas; bool needInc;
                    if (xi < lower[v] - feasTol) { infeas = lower[v] - xi; needInc = true; }
                    else if (xi > upper[v] + feasTol) { infeas = xi - upper[v]; needInc = false; }
                    else continue;

                    double score = infeas * infeas / math.max(weight[i], (double)1e-12);
                    if (score > bestScore) { bestScore = score; r = i; rNeedsIncrease = needInc; infeasR = infeas; }
                }

                if (r < 0) break;   // no basic bound violation -> primal feasible too -> optimal

                // ---- pivot row: rho = B^-T e_r (BTRAN), PRICE alpha_r = N^T rho over nonbasics ----
                for (int i = 0; i < m; i++) rho[i] = (i == r) ? (double)1 : (double)0;
                Btran(B, in P, etaAlpha, etaRow, etaCount, rho, m);
                PriceRow(M, rho, status, N, m, alphaRow);

                // ---- reduced costs (ratio test numerator): y = B^-T c_B (BTRAN), price over nonbasics ----
                for (int i = 0; i < m; i++) cB[i] = perturbedCost[basis[i]];
                for (int i = 0; i < m; i++) y[i] = cB[i];
                Btran(B, in P, etaAlpha, etaRow, etaCount, y, m);
                PriceReducedCosts(M, y, perturbedCost, status, N, m, dj);

                double rowMax = (double)0;
                for (int j = 0; j < N; j++) if (status[j] != STATUS_BASIC) rowMax = math.max(rowMax, math.abs(alphaRow[j]));
                double pivTol = math.max(pivTolFloor, (double)1e-6 * rowMax);

                DualRatioTest(alphaRow, dj, status, lower, upper, N, rNeedsIncrease, pivTol, infeasR, feasTol,
                              flipCols, out int flipCount, out int enterCol, out _, out bool anyCandidate);

                if (!anyCandidate) { resultStatus = LPStatus.Infeasible; break; }

                // ---- apply accumulated bound flips (BFRT) with ONE extra FTRAN of the summed columns ----
                if (flipCount > 0)
                {
                    for (int i = 0; i < m; i++) flipRHS[i] = (double)0;
                    for (int f = 0; f < flipCount; f++)
                    {
                        int j = flipCols[f];
                        double delta = status[j] == STATUS_AT_LOWER ? (upper[j] - lower[j]) : -(upper[j] - lower[j]);
                        for (int i = 0; i < m; i++) flipRHS[i] += delta * M[i, j];
                        status[j] = status[j] == STATUS_AT_LOWER ? STATUS_AT_UPPER : STATUS_AT_LOWER;
                    }
                    Ftran(B, in P, etaAlpha, etaRow, etaCount, flipRHS, m);
                    for (int i = 0; i < m; i++) xB[i] -= flipRHS[i];
                }

                // DualRatioTest guarantees enterCol >= 0 whenever anyCandidate is true (a REAL pivot
                // always terminates the walk -- see that method's doc comment for why a flip-only
                // iteration would be unsound). This check is a defensive backstop against that invariant,
                // not a normal code path.
                if (enterCol < 0) { resultStatus = LPStatus.MaxIterations; break; }

                // ---- real pivot: FTRAN the entering column + tau = Ftran(rho) for the DSE update ----
                for (int i = 0; i < m; i++) alphaCol[i] = M[i, enterCol];
                Ftran(B, in P, etaAlpha, etaRow, etaCount, alphaCol, m);
                for (int i = 0; i < m; i++) tau[i] = rho[i];
                Ftran(B, in P, etaAlpha, etaRow, etaCount, tau, m);

                double pivotElt = alphaCol[r];
                int sigma = status[enterCol] == STATUS_AT_LOWER ? 1 : -1;
                double sig = (double)sigma;
                int leavingVar = basis[r];

                // Invariant: a nonbasic variable must rest on a FINITE bound (see the matching guard in
                // LP.RevisedSimplex.double.cs's primal core for the full rationale). Row selection already
                // requires xB[r] to violate the ACTUAL lower/upper by more than feasTol to set
                // rNeedsIncrease, so an infinite target would essentially never trigger in a well-posed
                // problem, but guard defensively anyway: fall back to the other (finite) bound if the
                // intended one isn't finite, and drive thetaPrimal/the final status off that SAME decision
                // so xB[r] and status stay consistent.
                bool leavingNeedsIncrease = rNeedsIncrease;
                if (leavingNeedsIncrease && math.abs(lower[leavingVar]) >= (double)1e29) leavingNeedsIncrease = false;
                else if (!leavingNeedsIncrease && math.abs(upper[leavingVar]) >= (double)1e29) leavingNeedsIncrease = true;

                double targetBound = leavingNeedsIncrease ? lower[leavingVar] : upper[leavingVar];
                double safePivot = math.abs(pivotElt) < pivTolFloor ? (pivotElt >= (double)0 ? pivTolFloor : -pivTolFloor) : pivotElt;
                double thetaPrimal = (xB[r] - targetBound) / (sig * safePivot);
                if (thetaPrimal < (double)0) thetaPrimal = (double)0;

                double enteringValue = (sigma > 0 ? lower[enterCol] : upper[enterCol]) + sig * thetaPrimal;

                for (int i = 0; i < m; i++) xB[i] -= sig * thetaPrimal * alphaCol[i];
                xB[r] = enteringValue;

                status[leavingVar] = leavingNeedsIncrease ? STATUS_AT_LOWER : STATUS_AT_UPPER;
                basis[r] = enterCol;
                status[enterCol] = STATUS_BASIC;

                // ---- DSE weight update (Forrest-Goldfarb, verified against HiGHS -- see file header) ----
                double wr = weight[r];
                double newWr = wr / (safePivot * safePivot);
                for (int i = 0; i < m; i++)
                {
                    if (i == r) continue;
                    double ratio = alphaCol[i] / safePivot;
                    double wi = weight[i] - (double)2 * ratio * tau[i] + ratio * ratio * wr;
                    weight[i] = math.max(wi, weightFloor);
                }
                weight[r] = newWr;

                bool needRefactor = etaCount >= REFACTOR_INTERVAL || math.abs(pivotElt) < pivTolFloor;
                if (needRefactor)
                {
                    ok = Refactorize(M, basis, B, ref P, m, N);
                    etaCount = 0;
                    if (!ok) { resultStatus = LPStatus.MaxIterations; break; }
                    RebuildXB(M, rhs, status, lower, upper, B, in P, m, N, xB);
                }
                else
                {
                    for (int i = 0; i < m; i++) etaAlpha[etaCount, i] = alphaCol[i];
                    etaRow[etaCount] = r;
                    etaCount++;
                }

                iters++;
            }

            // ---- restore real bounds; a nonbasic still sitting at the artificial upper becomes
            //      primal-infeasible (or simply invalid, since the real bound is now +INF) and is reset
            //      to AtLower for the primal cleanup below to sort out (spec: "variables stuck at an
            //      artificial bound become primal-infeasible ... handled by the primal cleanup") ----
            for (int j = 0; j < n; j++)
            {
                if (!isArtificial[j]) continue;
                upper[j] = realINF;
                if (status[j] == STATUS_AT_UPPER) status[j] = STATUS_AT_LOWER;
            }

            LPInfo info;
            if (resultStatus == LPStatus.Infeasible)
            {
                for (int j = 0; j < N; j++)
                    xFull[j] = status[j] == STATUS_BASIC ? (double)0 : (status[j] == STATUS_AT_LOWER ? lower[j] : upper[j]);
                for (int i = 0; i < m; i++) xFull[basis[i]] = xB[i];
                info = new LPInfo { status = LPStatus.Infeasible, iterations = iters, objective = 0 };
            }
            else
            {
                // Primal cleanup (stage 1's core, warm-started): removes the perturbation's effect and
                // fixes any primal infeasibility left by the bound restoration, using the REAL cost.
                var cleanup = RevisedPrimalCore(M, lower, upper, cost, rhs, m, n, N, maxIter, xFull, basis, status);
                info = new LPInfo { status = cleanup.status, iterations = iters + cleanup.iterations, objective = 0 };
            }

            basis.Dispose(); status.Dispose(); xB.Dispose(); perturbedCost.Dispose(); isArtificial.Dispose();
            B.Dispose(); P.Dispose(); etaAlpha.Dispose(); etaRow.Dispose();
            y.Dispose(); cB.Dispose(); dj.Dispose(); rho.Dispose(); tau.Dispose();
            alphaRow.Dispose(); alphaCol.Dispose(); flipCols.Dispose(); flipRHS.Dispose(); weight.Dispose();

            return info;
        }
    }
}
