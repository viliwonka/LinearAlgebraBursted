#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // ================================================================================================
    // Bounded-variable DUAL revised simplex -- the LPMethod.DualSimplex backend, stage 2 of the
    // HiGHS-style dense revised-simplex port (docs/spec-revised-simplex.md). Builds on stage 1's kernel
    // layer (LP.RevisedSimplex.float.cs: Refactorize/Ftran/Btran/the eta-apply pair/RebuildXB) instead
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
    // Numerics: ordinary float throughout (matrix/vector storage, the ratio test, DSE weights), exactly
    // like stage 1 -- see that file's header for the full rationale (reuses the library's own LU/Blas;
    // tolerances derived from Consts.floatZeroThreshold/floatEpsilon, the SAME expressions stage 1
    // uses, computed INLINE in DualSimplexCore rather than via a shared helper method -- an float-
    // returning helper with no float-typed parameter differs only in return type between the float- and
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
    //
    // WARM-START (docs/draft-spec-mip.md stage 1, LPBasis in LP.Info.cs): DualSimplexCore below is
    // split fresh-overload-forwards-to-warm-overload, exactly like LP.RevisedSimplex.float.cs's
    // RevisedPrimalCore. The warm overload's dual-feasibility repair (bound flips / temporary
    // artificial bounds keyed off a REAL BTRAN-computed reduced cost) is a strict generalization of
    // the former cold-only precondition -- provably bit-identical at the all-logical basis, since
    // y = B^-T c_B is then exactly the zero vector (c_B = 0, and BTRAN of an all-zero vector stays
    // all-zero through every forward/back-substitution step) -- see that overload's own comments for
    // the full argument. LP.solve's `ref LPBasis` overload (LP.float.cs) is the only entry point;
    // the ordinary `LP.solve(..., LPMethod.DualSimplex)` call keeps hitting the UNCHANGED fresh
    // overload.
    // ================================================================================================
    public static partial class LP
    {
        // ---- float-typed entry point: builds the computational form (reusing stage 1's
        // BuildComputationalForm verbatim -- same partial class, same per-dtype fragment), hands off to
        // the dual core, copies the result back. ----
        static LPInfo dualSimplexCore(in floatMxN A, in floatN b, in floatN c,
                                     in NativeArray<ConstraintSense> senses,
                                     ref floatN x, out double objective, int maxIter)
        {
            int m = A.M_Rows, n = A.N_Cols, N = n + m;

            var M = new floatMxN(m, N, Allocator.Temp);
            var lower = new floatN(N, Allocator.Temp);
            var upper = new floatN(N, Allocator.Temp);
            var cost = new floatN(N, Allocator.Temp);
            var rhs = new floatN(m, Allocator.Temp);

            BuildComputationalForm(in A, in b, in c, in senses, M, lower, upper, cost, rhs, m, n, N);

            var xFull = new floatN(N, Allocator.Temp);
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
        //
        // MTmul (LP.RevisedSimplex.float.cs) fills alphaRow for EVERY column via its row-major sweep
        // (was a per-column loop reading M[i,j] with stride N -- the worst pattern for a row-major
        // matrix); the basic columns are then zeroed in a cheap second pass, matching the original
        // contract (basic entries left 0, unused). REORDERING vs the original per-column running sum:
        // see MTmul's own comment.
        internal static void PriceRow(floatMxN M, floatN rho, NativeArray<byte> status, int N, int m, floatN alphaRow)
        {
            MTmul(M, rho, alphaRow, m, N);
            for (int j = 0; j < N; j++)
                if (status[j] == STATUS_BASIC) alphaRow[j] = (float)0;
        }

        // Prices reduced costs d_j = cost[j] - dot(M[:,j], y) over every nonbasic column j (y = B^-T c_B
        // via BTRAN of the basic costs) -- the dual ratio test's numerator. Mirrors the pricing half of
        // stage 1's SelectEntering but fills the WHOLE array (the ratio test needs every sign-correct
        // candidate's d_j, not just the single best one).
        //
        // Same MTmul-then-fixup shape as PriceRow above: dj[j] is first filled with dot(M[:,j], y) for
        // every column, then combined with cost[j] (0 for basic).
        internal static void PriceReducedCosts(floatMxN M, floatN y, floatN cost, NativeArray<byte> status, int N, int m, floatN dj)
        {
            MTmul(M, y, dj, m, N);
            for (int j = 0; j < N; j++)
                dj[j] = status[j] == STATUS_BASIC ? (float)0 : cost[j] - dj[j];
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
        internal static void DualRatioTest(floatN alphaRow, floatN dj,
                                           NativeArray<byte> status, floatN lower, floatN upper,
                                           int N, bool needIncrease, float pivTol, float infeasR, float feasTol,
                                           NativeArray<int> flipCols, out int flipCount,
                                           out int enterCol, out float thetaD, out bool anyCandidate)
        {
            float s = needIncrease ? (float)(-1) : (float)1;
            flipCount = 0; enterCol = -1; thetaD = (float)0; anyCandidate = false;

            var considered = new NativeArray<bool>(N, Allocator.Temp);   // true = excluded from candidacy
            for (int j = 0; j < N; j++)
            {
                if (status[j] == STATUS_BASIC || upper[j] - lower[j] <= (float)1e-13) { considered[j] = true; continue; }
                float ah = s * alphaRow[j];
                bool cand = (status[j] == STATUS_AT_LOWER && ah > pivTol) || (status[j] == STATUS_AT_UPPER && ah < -pivTol);
                considered[j] = !cand;
                if (cand) anyCandidate = true;
            }

            if (!anyCandidate) { considered.Dispose(); return; }

            // Near-tie window for the largest-|alphaHat| stability preference below; per-dtype (a fixed
            // 1e-9 sits below float's own rounding noise at O(1) ratios, disabling the preference).
            float ratioTieTol = (float)1e-5f;

            float remaining = infeasR;
            while (true)
            {
                int best = -1; float bestRatio = (float)0; float bestAbsA = (float)0;
                for (int j = 0; j < N; j++)
                {
                    if (considered[j]) continue;
                    float ah = s * alphaRow[j];
                    float djr = math.max(status[j] == STATUS_AT_LOWER ? dj[j] : -dj[j], (float)0);
                    float ratio = djr / math.abs(ah);
                    if (best < 0 || ratio < bestRatio - ratioTieTol ||
                        (ratio <= bestRatio + ratioTieTol && math.abs(ah) > bestAbsA))
                    { best = j; bestRatio = ratio; bestAbsA = math.abs(ah); }
                }
                if (best < 0) { anyCandidate = false; break; }   // exhausted: combined capacity insufficient

                considered[best] = true;
                bool boxed = upper[best] - lower[best] < (float)1e29;
                float fullAbsorb = bestAbsA * (upper[best] - lower[best]);

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
        //
        // Fresh all-logical start: builds that basis/status, forwards to the warm-start overload below
        // (mirrors LP.RevisedSimplex.float.cs's RevisedPrimalCore split -- the fresh-start case is
        // exactly the warm-start case's math evaluated at a specific, all-logical starting point, so it
        // forwards instead of duplicating the pivot loop -- see that overload's comments for why this is
        // provably behavior-preserving), and owns (disposes) the basis/status it allocated.
        internal static LPInfo DualSimplexCore(floatMxN M, floatN lower, floatN upper,
                                               floatN cost, floatN rhs, int m, int n, int N,
                                               int maxIter, floatN xFull)
        {
            var basis = new NativeArray<int>(m, Allocator.Temp);
            var status = new NativeArray<byte>(N, Allocator.Temp);
            for (int i = 0; i < m; i++) { basis[i] = n + i; status[n + i] = STATUS_BASIC; }
            for (int j = 0; j < n; j++) status[j] = STATUS_AT_LOWER;

            var info = DualSimplexCore(M, lower, upper, cost, rhs, m, n, N, maxIter, xFull, basis, status);

            basis.Dispose(); status.Dispose();
            return info;
        }

        // Warm-start overload -- added for LP.solve's `ref LPBasis` re-solve entry point (LP.float.cs;
        // LPBasis in LP.Info.cs captures exactly this (status[], basis[]) pair in this computational-
        // form indexing). `basis`/`status` (sized m / N) must already describe a VALID assignment --
        // every nonbasic sitting exactly on one of its (current) bounds -- but need NOT be feasible or
        // dual-feasible; the caller retains ownership (this method reads/mutates them in place, never
        // allocates or disposes them), exactly like LP.RevisedSimplex.float.cs's RevisedPrimalCore warm
        // overload. The fresh-start overload above forwards here with the all-logical basis/status, so
        // this single body serves both -- the dual-feasibility repair below replaces the FORMER cold-
        // only precondition with a strict generalization of it, provably bit-identical at that specific
        // starting point (see the repair's own comment).
        //
        // A basis matrix that fails to factor at the very first Refactorize (garbage/stale `basis[]`,
        // e.g. carried over from an unrelated problem -- see LPBasis's doc comment) falls back to the
        // all-logical start rather than failing outright -- defensive, and that fallback's own
        // Refactorize cannot itself fail (the logical columns are exactly the identity, see
        // BuildComputationalForm), so `resultStatus` only reports MaxIterations here in a genuinely
        // pathological case (e.g. M containing NaN/Inf).
        internal static LPInfo DualSimplexCore(floatMxN M, floatN lower, floatN upper,
                                               floatN cost, floatN rhs, int m, int n, int N,
                                               int maxIter, floatN xFull,
                                               NativeArray<int> basis, NativeArray<byte> status)
        {
            // Same per-dtype tolerance expressions as stage 1 (see that file's header for why these are
            // inlined rather than a shared helper method).
            float feasTol = (float)math.max(math.sqrt((double)Consts.floatEpsilon), 1e-7);
            float dualTol = feasTol;
            float pivTolFloor = math.max(Consts.floatZeroThreshold, (float)1e-9);
            float weightFloor = (float)1e-4;
            float realINF = (float)1e30;

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
            float artificialBound = (float)(100.0 * dataScale);

            var xB = new floatN(m, Allocator.Temp);
            var perturbedCost = new floatN(N, Allocator.Temp);
            // 0 = untouched, +1 = given a temporary artificial UPPER bound (real upper was +INF),
            // -1 = given a temporary artificial LOWER bound (real lower was -INF) -- generalizes the
            // former cold-only `isArtificial` (bool[n], upper-only) to any nonbasic column (structural
            // OR logical) in either direction; see the repair loop below.
            var artificialDir = new NativeArray<sbyte>(N, Allocator.Temp);
            // True the moment ANY column gets a temporary artificial bound in the repair below --
            // gates the zero-pivot fast path near the bottom of this method (see that comment).
            bool anyArtificial = false;

            // Cost perturbation (degeneracy defence), ported from HEkk::initialiseCost. Structural
            // columns: base = 5e-7 * max|c| (dampened by sqrt(sqrt()) above 100, clamped to 1 when
            // <1% of variables are boxed), xpert = (1+r)*(|c_j|+1)*base, POSITIVE and signed by bound
            // structure so a dual-feasible d_j stays dual-feasible; free/fixed columns untouched.
            // Logical (row) columns: symmetric +-0.5 * 1e-12 -- exact-tie breaking only, ~7 orders
            // smaller than the column base. Both bases expressed in dualTol units (5*dualTol and
            // 1e-5*dualTol = HiGHS's literal 5e-7 / 1e-12 in double exactly; float scales with its own
            // tolerance). Deterministic per-column hash r in [0,1) replaces HiGHS's random vector
            // (MurmurHash3 finalizer mix, inlined -- see the tolerance comment above).
            double maxAbsCost = 0.0;
            for (int j = 0; j < n; j++) maxAbsCost = math.max(maxAbsCost, math.abs((double)cost[j]));
            if (maxAbsCost > 100.0) maxAbsCost = math.sqrt(math.sqrt(maxAbsCost));
            int boxedCount = 0;
            for (int j = 0; j < N; j++)
                if ((double)(upper[j] - lower[j]) < 1e30) boxedCount++;
            if (boxedCount < 0.01 * N) maxAbsCost = math.min(maxAbsCost, 1.0);
            double colPerturbBase = 5.0 * (double)dualTol * maxAbsCost;
            double rowPerturbBase = 1e-5 * (double)dualTol;
            for (int j = 0; j < N; j++)
            {
                uint h = (uint)j * 2654435761u + 0x9E3779B9u;
                h ^= h >> 15; h *= 0x85EBCA6Bu;
                h ^= h >> 13; h *= 0xC2B2AE35u;
                h ^= h >> 16;
                double r = h * (1.0 / 4294967296.0);
                if (j >= n)
                {
                    perturbedCost[j] = cost[j] + (float)((0.5 - r) * rowPerturbBase);
                    continue;
                }
                bool loInf = lower[j] <= (float)(-1e29);
                bool upInf = upper[j] >= (float)1e29;
                double xpert = (1.0 + r) * (math.abs((double)cost[j]) + 1.0) * colPerturbBase;
                if (loInf && upInf) perturbedCost[j] = cost[j];                                  // free
                else if (upInf) perturbedCost[j] = cost[j] + (float)xpert;                      // lower-bounded
                else if (loInf) perturbedCost[j] = cost[j] - (float)xpert;                      // upper-bounded
                else if (lower[j] != upper[j])                                                   // boxed
                    perturbedCost[j] = cost[j] + (cost[j] >= (float)0 ? (float)xpert : (float)(-xpert));
                else perturbedCost[j] = cost[j];                                                 // fixed
            }

            var B = new floatMxN(m, m, Allocator.Temp);
            var P = new Pivot(m, Allocator.Temp);
            var etaAlpha = new floatMxN(REFACTOR_INTERVAL, m, Allocator.Temp);
            var etaRow = new NativeArray<int>(REFACTOR_INTERVAL, Allocator.Temp);
            int etaCount = 0;

            var y = new floatN(m, Allocator.Temp);
            var cB = new floatN(m, Allocator.Temp);
            var dj = new floatN(N, Allocator.Temp);
            var rho = new floatN(m, Allocator.Temp);
            var tau = new floatN(m, Allocator.Temp);
            var alphaRow = new floatN(N, Allocator.Temp);
            var alphaCol = new floatN(m, Allocator.Temp);
            var flipCols = new NativeArray<int>(N, Allocator.Temp);
            var flipRHS = new floatN(m, Allocator.Temp);

            // DSE weights: seeded w_i = 1, exact only at the all-logical basis. HiGHS reuses weights
            // carried on the HEkk instance and computes EXACT weights (one BTRAN per row) for a
            // non-logical warm basis; unit seeding at a warm basis is a KNOWN simplification here --
            // affects pricing quality only, never correctness. The faithful fix (persist weights
            // alongside LPBasis across calls) belongs to the LPBasis factor-persistence redesign.
            // Maintained (never reset) across refactorizations for the rest of this run.
            var weight = new floatN(m, Allocator.Temp);
            for (int i = 0; i < m; i++) weight[i] = (float)1;

            LPStatus resultStatus = LPStatus.Optimal;
            int iters = 0;

            bool ok = Refactorize(M, basis, B, ref P, m, N);
            if (!ok)
            {
                // Supplied basis was singular -- fall back to the standard all-logical start (see this
                // overload's header comment) rather than reporting failure outright.
                for (int i = 0; i < m; i++) { basis[i] = n + i; status[n + i] = STATUS_BASIC; }
                for (int j = 0; j < n; j++) status[j] = STATUS_AT_LOWER;
                ok = Refactorize(M, basis, B, ref P, m, N);
            }

            if (!ok) resultStatus = LPStatus.MaxIterations;
            else
            {
                // ---- dual-feasibility repair: bound flips (or, when the natural bound is infinite, a
                // temporary artificial bound) on every nonbasic column whose ACTUAL reduced cost sign is
                // wrong for its current status. y = B^-T c_B via BTRAN against the JUST-refactorized
                // basis (etaCount == 0 here, a clean base solve, no eta corrections yet).
                //
                // At the all-logical basis c_B = 0 (cost[n+i] = 0 for every logical, BuildComputational
                // Form), so y starts as the exact zero vector and BTRAN of an all-zero vector STAYS all-
                // zero through every forward/back-substitution step (each step is either an assignment of
                // 0, a multiply-by-0, a subtract-of-0, or a divide-of-0-by-a-nonzero-pivot -- all exact
                // in IEEE754, no rounding). So d_j collapses to EXACTLY cost[j] for every nonbasic j, and
                // since logicals are BASIC at that basis (skipped by the STATUS_BASIC check below, same
                // as the old code's `j < n` loop range did implicitly), this is BIT-IDENTICAL to the
                // former cold-only precondition it replaces. At an arbitrary warm-started basis y is the
                // real B^-T c_B, so every nonbasic column -- structural OR logical -- is checked against
                // its TRUE reduced cost, and EITHER bound is eligible for the artificial-bound trick (a
                // GreaterEqual row's logical has real lower = -INF, the mirror image of a structural's
                // real upper = +INF).
                //
                // Uses the ORIGINAL cost, not perturbedCost: this is a one-time TRUE-dual-feasibility
                // decision, not part of the iterative pricing the perturbation is meant to stabilize. Using
                // perturbedCost here was an actual bug -- a column with cost[j] EXACTLY 0 (e.g. every x+/x-
                // column in LP.lad's reformulation, which has none of its own cost) is dual-feasible as-is
                // (d_j=0 trivially satisfies d_j>=0), but the perturbation's random sign could nudge it
                // slightly negative and give it a pointless artificial bound; multiplied across the MANY
                // exactly-zero-cost columns LP.lad's [x+|x-] block always has, this corrupted the warm-started
                // basis handed to the primal cleanup badly enough to report a false Unbounded.
                for (int i = 0; i < m; i++) y[i] = cost[basis[i]];
                Btran(B, in P, etaAlpha, etaRow, etaCount, y, m);

                // dj (declared above, not yet used at this point in the run -- its per-iteration use is
                // inside the while loop below, via PriceReducedCosts) is reused as scratch here: MTmul
                // (LP.RevisedSimplex.float.cs) fills it with dot(M[:,j], y) for every column in one
                // row-major sweep, replacing the original per-column loop (M[i,j] read at stride N). See
                // MTmul's own comment for the reordering note.
                MTmul(M, y, dj, m, N);

                for (int j = 0; j < N; j++)
                {
                    if (status[j] == STATUS_BASIC || upper[j] - lower[j] <= (float)1e-13) continue;

                    float d = cost[j] - dj[j];

                    if (status[j] == STATUS_AT_LOWER && d < -dualTol)
                    {
                        if (upper[j] < (float)1e29) status[j] = STATUS_AT_UPPER;
                        else { upper[j] = artificialBound; artificialDir[j] = 1; status[j] = STATUS_AT_UPPER; anyArtificial = true; }
                    }
                    else if (status[j] == STATUS_AT_UPPER && d > dualTol)
                    {
                        if (lower[j] > (float)(-1e29)) status[j] = STATUS_AT_LOWER;
                        else { lower[j] = -artificialBound; artificialDir[j] = -1; status[j] = STATUS_AT_LOWER; anyArtificial = true; }
                    }
                }

                RebuildXB(M, rhs, status, lower, upper, B, in P, m, N, xB);
            }

            int budget = maxIter > 0 ? maxIter : 50 * (m + N) + 200;

            while (resultStatus == LPStatus.Optimal)
            {
                if (iters >= budget) { resultStatus = LPStatus.MaxIterations; break; }

                // ---- leaving row: dual steepest edge, r = argmax infeas_r^2 / w_r ----
                int r = -1; float bestScore = (float)(-1); bool rNeedsIncrease = false; float infeasR = (float)0;
                for (int i = 0; i < m; i++)
                {
                    int v = basis[i];
                    float xi = xB[i];
                    float infeas; bool needInc;
                    if (xi < lower[v] - feasTol) { infeas = lower[v] - xi; needInc = true; }
                    else if (xi > upper[v] + feasTol) { infeas = xi - upper[v]; needInc = false; }
                    else continue;

                    float score = infeas * infeas / math.max(weight[i], (float)1e-12);
                    if (score > bestScore) { bestScore = score; r = i; rNeedsIncrease = needInc; infeasR = infeas; }
                }

                if (r < 0) break;   // no basic bound violation -> primal feasible too -> optimal

                // ---- pivot row: rho = B^-T e_r (BTRAN), PRICE alpha_r = N^T rho over nonbasics ----
                for (int i = 0; i < m; i++) rho[i] = (i == r) ? (float)1 : (float)0;
                Btran(B, in P, etaAlpha, etaRow, etaCount, rho, m);
                PriceRow(M, rho, status, N, m, alphaRow);

                // ---- reduced costs (ratio test numerator): y = B^-T c_B (BTRAN), price over nonbasics ----
                for (int i = 0; i < m; i++) cB[i] = perturbedCost[basis[i]];
                for (int i = 0; i < m; i++) y[i] = cB[i];
                Btran(B, in P, etaAlpha, etaRow, etaCount, y, m);
                PriceReducedCosts(M, y, perturbedCost, status, N, m, dj);

                float rowMax = (float)0;
                for (int j = 0; j < N; j++) if (status[j] != STATUS_BASIC) rowMax = math.max(rowMax, math.abs(alphaRow[j]));
                float pivTol = math.max(pivTolFloor, (float)1e-6 * rowMax);

                DualRatioTest(alphaRow, dj, status, lower, upper, N, rNeedsIncrease, pivTol, infeasR, feasTol,
                              flipCols, out int flipCount, out int enterCol, out _, out bool anyCandidate);

                if (!anyCandidate) { resultStatus = LPStatus.Infeasible; break; }

                // ---- apply accumulated bound flips (BFRT) with ONE extra FTRAN of the summed columns ----
                //
                // The inner `flipRHS[i] += delta * M[i, j]` read is column-strided too (M[i,j] at stride
                // N over i, fixed j), but LEFT AS-IS deliberately: flipCount is normally small (a handful
                // of boxed nonbasics absorbed per iteration, not O(N)), so its cost is O(flipCount * m),
                // already far below the O(mN) PRICE passes this file's other column-strided loops were.
                // Routing it through a dense Mmul(M, deltaVec, ..., m, N) GEMV (deltaVec sparse, nonzero
                // only at flipCols) would touch all N columns unconditionally -- a regression whenever
                // flipCount << N, the common case -- for a loop that was never the O(mN) bottleneck.
                if (flipCount > 0)
                {
                    for (int i = 0; i < m; i++) flipRHS[i] = (float)0;
                    for (int f = 0; f < flipCount; f++)
                    {
                        int j = flipCols[f];
                        float delta = status[j] == STATUS_AT_LOWER ? (upper[j] - lower[j]) : -(upper[j] - lower[j]);
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

                float pivotElt = alphaCol[r];
                int sigma = status[enterCol] == STATUS_AT_LOWER ? 1 : -1;
                float sig = (float)sigma;
                int leavingVar = basis[r];

                // Invariant: a nonbasic variable must rest on a FINITE bound (see the matching guard in
                // LP.RevisedSimplex.float.cs's primal core for the full rationale). Row selection already
                // requires xB[r] to violate the ACTUAL lower/upper by more than feasTol to set
                // rNeedsIncrease, so an infinite target would essentially never trigger in a well-posed
                // problem, but guard defensively anyway: fall back to the other (finite) bound if the
                // intended one isn't finite, and drive thetaPrimal/the final status off that SAME decision
                // so xB[r] and status stay consistent.
                bool leavingNeedsIncrease = rNeedsIncrease;
                if (leavingNeedsIncrease && math.abs(lower[leavingVar]) >= (float)1e29) leavingNeedsIncrease = false;
                else if (!leavingNeedsIncrease && math.abs(upper[leavingVar]) >= (float)1e29) leavingNeedsIncrease = true;

                float targetBound = leavingNeedsIncrease ? lower[leavingVar] : upper[leavingVar];
                float safePivot = math.abs(pivotElt) < pivTolFloor ? (pivotElt >= (float)0 ? pivTolFloor : -pivTolFloor) : pivotElt;
                float thetaPrimal = (xB[r] - targetBound) / (sig * safePivot);
                if (thetaPrimal < (float)0) thetaPrimal = (float)0;

                float enteringValue = (sigma > 0 ? lower[enterCol] : upper[enterCol]) + sig * thetaPrimal;

                for (int i = 0; i < m; i++) xB[i] -= sig * thetaPrimal * alphaCol[i];
                xB[r] = enteringValue;

                status[leavingVar] = leavingNeedsIncrease ? STATUS_AT_LOWER : STATUS_AT_UPPER;
                basis[r] = enterCol;
                status[enterCol] = STATUS_BASIC;

                // ---- DSE weight update (Forrest-Goldfarb, verified against HiGHS -- see file header) ----
                float wr = weight[r];
                float newWr = wr / (safePivot * safePivot);
                for (int i = 0; i < m; i++)
                {
                    if (i == r) continue;
                    float ratio = alphaCol[i] / safePivot;
                    float wi = weight[i] - (float)2 * ratio * tau[i] + ratio * ratio * wr;
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

            // ---- restore real bounds; a nonbasic still sitting at an artificial bound becomes primal-
            //      infeasible (or simply invalid, since the real bound is now infinite again) and is
            //      reset to the OTHER (now real) status for the primal cleanup below to sort out (spec:
            //      "variables stuck at an artificial bound become primal-infeasible ... handled by the
            //      primal cleanup") -- generalized to both directions, see artificialDir above ----
            for (int j = 0; j < N; j++)
            {
                if (artificialDir[j] == 0) continue;
                if (artificialDir[j] > 0)
                {
                    upper[j] = realINF;
                    if (status[j] == STATUS_AT_UPPER) status[j] = STATUS_AT_LOWER;
                }
                else
                {
                    lower[j] = -realINF;
                    if (status[j] == STATUS_AT_LOWER) status[j] = STATUS_AT_UPPER;
                }
            }

            LPInfo info;
            if (resultStatus == LPStatus.Infeasible)
            {
                for (int j = 0; j < N; j++)
                    xFull[j] = status[j] == STATUS_BASIC ? (float)0 : (status[j] == STATUS_AT_LOWER ? lower[j] : upper[j]);
                for (int i = 0; i < m; i++) xFull[basis[i]] = xB[i];
                info = new LPInfo { status = LPStatus.Infeasible, iterations = iters, objective = 0 };
            }
            else if (resultStatus == LPStatus.Optimal && iters == 0 && !anyArtificial)
            {
                // Zero-pivot fast path (warm-start payoff): the dual loop broke immediately (r < 0, no
                // basic bound violation) with NO pivot ever applied and NO temporary artificial bound
                // ever installed. The repair above already established dual feasibility against the
                // REAL cost (not perturbedCost -- see its own comment), and the loop's own exit
                // condition IS primal feasibility against the REAL bounds (no artificial ones are live).
                // Both hold simultaneously with the basis UNCHANGED since that repair, so this state is
                // already the true LP optimum -- calling RevisedPrimalCore here would provably perform
                // its own zero pivots too (SelectEntering finds no wrong-signed nonbasic, since dual
                // feasibility against the real cost already holds) after paying a SECOND full O(m^3)
                // Refactorize + O(mN) RebuildXB that can only reproduce state already sitting in
                // status/basis/xB. Skipping it roughly halves a warm re-solve's fixed per-call cost
                // whenever it applies (measured ~0.12ms/call -> ~0.06ms/call at mAug~80 on an isolated
                // warm LP.solve(ref LPBasis) benchmark, MIP perf investigation 2026-07-10) -- a genuine
                // but MINORITY case for MIP/strong-branch-trial re-solves in practice (most single-bound
                // tightenings still cost >=1 real pivot to restore primal feasibility, which this path
                // does not shortcut). Reuses the SAME extraction as the Infeasible branch above, just
                // with status = Optimal.
                for (int j = 0; j < N; j++)
                    xFull[j] = status[j] == STATUS_BASIC ? (float)0 : (status[j] == STATUS_AT_LOWER ? lower[j] : upper[j]);
                for (int i = 0; i < m; i++) xFull[basis[i]] = xB[i];
                info = new LPInfo { status = LPStatus.Optimal, iterations = iters, objective = 0 };
            }
            else
            {
                // Primal cleanup (stage 1's core, warm-started): removes the perturbation's effect and
                // fixes any primal infeasibility left by the bound restoration, using the REAL cost.
                var cleanup = RevisedPrimalCore(M, lower, upper, cost, rhs, m, n, N, maxIter, xFull, basis, status);
                info = new LPInfo { status = cleanup.status, iterations = iters + cleanup.iterations, objective = 0 };
            }

            xB.Dispose(); perturbedCost.Dispose(); artificialDir.Dispose();
            B.Dispose(); P.Dispose(); etaAlpha.Dispose(); etaRow.Dispose();
            y.Dispose(); cB.Dispose(); dj.Dispose(); rho.Dispose(); tau.Dispose();
            alphaRow.Dispose(); alphaCol.Dispose(); flipCols.Dispose(); flipRHS.Dispose(); weight.Dispose();

            return info;
        }
    }
}
