#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    // ================================================================================================
    // Bounded-variable PRIMAL revised simplex -- the LPMethod.RevisedSimplex backend, stage 1 of the
    // HiGHS-style dense revised-simplex port (docs/spec-revised-simplex.md). Instead of updating a full
    // m x (n+m) tableau every pivot (simplexCore, LP.double.cs -- left untouched), this keeps the
    // ORIGINAL constraint matrix and an LU-factored basis, reconstructing whatever a pivot needs via
    // triangular solves (FTRAN/BTRAN) against a Product-Form-of-the-Inverse (PFI) eta file. Bounded
    // variables are native (no slack row-doubling for ranges) and periodic refactorization bounds error
    // growth instead of letting it accumulate in a tableau.
    //
    // Computational form:  min cᵀx   s.t.   A x + s = b,   l <= (x, s) <= u.
    //   N = n + m variables: j < n are structural (column A[:,j], bounds [0, +INF)); j = n+i is the
    //   logical of row i (column e_i; bounds by sense -- LessEqual: [0,+INF), Equal: [0,0],
    //   GreaterEqual: (-INF,0]). INF = 1e30. No free variables arise here.
    //
    // Numerics: everything here -- matrix/vector STORAGE, the basis factorization, FTRAN/BTRAN, the eta
    // file, reduced costs, ratios -- is ordinary double, exactly like every other solver in this library
    // (simplexCore, interiorCore). Tolerances are derived from the SAME per-dtype Consts the tableau
    // simplex already uses (Consts.doubleZeroThreshold / doubleEpsilon), which resolve to the float or
    // double constant via the usual token substitution -- no separate double-only code path, and no
    // inline per-type literal marker (see GenUtils.cs) needed for these since Consts already IS
    // per-dtype. They are computed INLINE at each call site rather than behind a shared helper method:
    // a helper returning double with no double-typed parameter (e.g. `double Tol()`) differs only in its
    // RETURN type between the float- and double-generated fragments of this partial class, and C# does
    // not overload on return type alone -- both copies would collide as duplicate members (CS0111). Any
    // new shared double-returning helper needs at least one double-typed parameter to stay a genuine
    // per-type overload; otherwise, inline it. Objective/diagnostic SUMS that are pure locals (never fed
    // back into array storage) still accumulate in `double` for precision, matching simplexCore's own
    // convention (its phase-1 infeasibility sum) and this file's double entry point (the final cᵀx
    // recompute).
    //
    // Basis factorization: reuses the library's OWN `LU` class (LU.decompInPlace, the compact in-place
    // form) for both triangular directions -- FTRAN via LU.decompSolve (Blas's compact-LU triangular
    // solves triLowerLU/triUpperLU) and BTRAN (solve Bᵀy = v) via LU.decompSolveTransA, the getrs
    // trans='T' counterpart promoted into LU itself (this file used to carry its own hand-written
    // SolveTranspose; it is now just a call to the library primitive) -- no reimplementation of GETRF.
    // Both directions run against the SAME compact doubleMxN + Pivot factor LU.decompInPlace produces;
    // only the read pattern (forward vs transposed) differs.
    //
    // Every kernel function below is `internal static` (not buried locals) precisely so
    // LP.DualSimplex.double.cs (stage 2) can call them directly: ordinary partial-class member
    // resolution within the same generated type (float calls float, double calls double), exactly like
    // interiorCore's Amul/ATmul/BuildNormalStructured already work across this same file. REFACTOR_
    // INTERVAL and the STATUS_* tags are type-agnostic and therefore declared once in LP.Info.cs
    // (singularFile) instead of here, where a per-dtype duplicate would collide (CS0102) -- see that
    // file's comment.
    // ================================================================================================
    public static partial class LP
    {
        // ---- double-typed entry point: builds the computational form, hands off to the (now equally
        // double-typed) core, copies the result back. ----
        static LPInfo revisedSimplexCore(in doubleMxN A, in doubleN b, in doubleN c,
                                         in NativeArray<ConstraintSense> senses,
                                         ref doubleN x, out double objective, int maxIter)
        {
            int m = A.M_Rows, n = A.N_Cols, N = n + m;

            var M = new doubleMxN(m, N, Allocator.Temp);      // zero-initialized: [A | I], row-major
            var lower = new doubleN(N, Allocator.Temp);
            var upper = new doubleN(N, Allocator.Temp);
            var cost = new doubleN(N, Allocator.Temp);
            var rhs = new doubleN(m, Allocator.Temp);

            BuildComputationalForm(in A, in b, in c, in senses, M, lower, upper, cost, rhs, m, n, N);

            var xFull = new doubleN(N, Allocator.Temp);
            var info = RevisedPrimalCore(M, lower, upper, cost, rhs, m, n, N, maxIter, xFull);

            for (int j = 0; j < n; j++) x[j] = xFull[j];

            // Fresh recompute from the caller's ORIGINAL c -- matches simplexCore's "report objective
            // from fresh recompute cᵀx" convention. Accumulated in double for reporting precision
            // regardless of solve dtype (a pure local, never fed back into any array -- see file header).
            double obj = 0;
            for (int j = 0; j < n; j++) obj += (double)c[j] * (double)xFull[j];
            objective = obj;
            info.objective = obj;

            M.Dispose(); lower.Dispose(); upper.Dispose(); cost.Dispose(); rhs.Dispose(); xFull.Dispose();
            return info;
        }

        // Builds M=[A|I] (m x N), bounds, cost and rhs from the double inputs. Shared verbatim by stage
        // 2 (LP.DualSimplex.double.cs calls this too -- same partial class, same per-dtype fragment).
        internal static void BuildComputationalForm(in doubleMxN A, in doubleN b, in doubleN c,
                                           in NativeArray<ConstraintSense> senses,
                                           doubleMxN M, doubleN lower, doubleN upper,
                                           doubleN cost, doubleN rhs, int m, int n, int N)
        {
            double INF = (double)1e30;

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++) M[i, j] = A[i, j];
                M[i, n + i] = (double)1;            // logical column e_i (M was zero-initialized)
                rhs[i] = b[i];
            }

            for (int j = 0; j < n; j++) { lower[j] = (double)0; upper[j] = INF; cost[j] = c[j]; }

            for (int i = 0; i < m; i++)
            {
                int j = n + i;
                cost[j] = (double)0;
                switch (senses[i])
                {
                    case ConstraintSense.LessEqual: lower[j] = (double)0; upper[j] = INF; break;
                    case ConstraintSense.Equal: lower[j] = (double)0; upper[j] = (double)0; break;
                    default: lower[j] = -INF; upper[j] = (double)0; break;      // GreaterEqual
                }
            }
        }

        // ---- Shared row-major GEMV kernels (SIMD-routing helpers) ----
        //
        // Every "price a reduced cost over nonbasic columns" pass in this file and
        // LP.DualSimplex.double.cs used to be a per-column loop (j outer, i inner) reading M[i,j] with
        // stride N between consecutive i -- the worst access pattern for a row-major matrix, and the
        // dominant O(mN)-per-iteration cost. MTmul reshapes that into UnsafeOP.vecMatDot's row-major
        // sweep (i outer, j inner unit-stride: outv[j] += v[i]*M[i,j], zeroed first) -- mirrors
        // LP.InteriorPoint.double.cs's ATmul (kept as an independent copy here rather than reused across
        // files, so the two solver stages stay independently readable/deletable). REORDERING vs the
        // original per-column running-subtraction chain (cost[j] - t1 - t2 - ... vs a separate summed
        // dot product subtracted once): rounding-only, not bitwise-identical -- tolerance-safe, matching
        // every other kernel this campaign has touched (see docs/perf-vectorization-lessons.md).
        internal static unsafe void MTmul(doubleMxN M, doubleN v, doubleN outv, int m, int N)
        {
            UnsafeOP.vecMatDot(v.Data.Ptr, M.Data.Ptr, outv.Data.Ptr, m, N);
        }

        // out[i] = Σ_j M[i,j] v[j]  (M m×N, v length N, out length m) -- Mv (Mmul), used by RebuildXB
        // (adj = rhs − M·valVec, a column-strided scatter in its original per-nonbasic-column form). Routes
        // through UnsafeOP.matVecDot (two double4 SIMD accumulators per row, [NoAlias] pointers); outv
        // is zeroed first since matVecDot ACCUMULATES.
        internal static unsafe void Mmul(doubleMxN M, doubleN v, doubleN outv, int m, int N)
        {
            UnsafeUtility.MemClear(outv.Data.Ptr, (long)m * UnsafeUtility.SizeOf<double>());
            UnsafeOP.matVecDot(M.Data.Ptr, v.Data.Ptr, outv.Data.Ptr, m, N);
        }

        // Rebuilds the m x m basis matrix B (column k = column basis[k] of M) and LU-factors it in place
        // via the library's own LU.decompInPlace (compact form, partial pivoting -- no reimplementation
        // of GETRF). Returns false on a singular basis.
        //
        // Loop order is i (outer) / k (inner), not k / i: the natural "one basis column at a time" order
        // reads M[i,col] with stride N between consecutive i (a column gather from a row-major matrix)
        // AND writes B[i,k] with stride m between consecutive i (a column-strided write too). Swapping
        // to i outer / k inner turns the read into m arbitrary-offset reads WITHIN one row of M (which,
        // unlike striding across N-separated rows, is likely already resident once that row's cache
        // lines are touched) and the write into a fully contiguous row of B. Pure data movement, no
        // arithmetic -- bit-identical to the original order, zero drift risk.
        internal static bool Refactorize(doubleMxN M, NativeArray<int> basis, doubleMxN B, ref Pivot P, int m, int N)
        {
            for (int i = 0; i < m; i++)
                for (int k = 0; k < m; k++)
                    B[i, k] = M[i, basis[k]];
            return LU.decompInPlace(ref B, ref P).Solved;
        }

        // FTRAN: solve (current basis) * alpha = v for alpha, in place. Base LU solve (library's
        // LU.decompSolve, compact form) against the last refactorization, then the PFI eta file applied
        // in chronological (creation) order.
        internal static void Ftran(doubleMxN B, in Pivot P,
                                   doubleMxN etaAlpha, NativeArray<int> etaRow, int etaCount,
                                   doubleN v, int m)
        {
            LU.decompSolve(ref B, in P, ref v);
            for (int k = 0; k < etaCount; k++)
                ApplyEtaForward(etaAlpha, etaRow[k], k, v, m);
        }

        // BTRAN: solve (current basis)ᵀ * y = v for y. Eta file applied in REVERSE + transposed FIRST,
        // then the transposed base solve against the last refactorization (LU.decompSolveTransA).
        internal static void Btran(doubleMxN B, in Pivot P,
                                   doubleMxN etaAlpha, NativeArray<int> etaRow, int etaCount,
                                   doubleN v, int m)
        {
            for (int k = etaCount - 1; k >= 0; k--)
                ApplyEtaTransposed(etaAlpha, etaRow[k], k, v, m);

            LU.decompSolveTransA(ref B, in P, ref v);
        }

        // Applies eta slot `slot` (leaving row `row`, column alpha_q stored in etaAlpha[slot,:]) via
        // E^-1: v[row] /= alpha[row]; v[i] -= alpha[i]*v[row] for i != row (spec's PFI formula; v[row]'s
        // NEW value is what every other entry subtracts, so it must be updated first).
        //
        // The i != row exclusion is resolved by running UnsafeOP.axpy (a plain independent-per-lane
        // update, so this is bit-identical to the branchy scalar loop for every i != row -- no
        // reduction, no reassociation) over the WHOLE row [0,m) and then restoring v[row] to vr
        // afterward, rather than branching inside the hot loop: v[row] += (-vr)*etaAlpha[slot,row]
        // computes a value the code immediately discards, so the branch-free overwrite is exact, not
        // an approximation.
        internal static unsafe void ApplyEtaForward(doubleMxN etaAlpha, int row, int slot, doubleN v, int m)
        {
            double* etaRow = etaAlpha.Data.Ptr + (long)slot * etaAlpha.N_Cols;
            v[row] = v[row] / etaRow[row];
            double vr = v[row];
            UnsafeOP.axpy(v.Data.Ptr, etaRow, -vr, m);
            v[row] = vr;
        }

        // Applies (E^-1)^T: y[row] = (v[row] - sum_{i!=row} alpha[i]*v[i]) / alpha[row]; every other
        // entry of v is left untouched (derived in the file header comment).
        //
        // The i != row exclusion: compute the FULL dot (including the row term) via UnsafeOP.vecDot's
        // 2x width-4 SIMD-accumulator reduction, then subtract the one excluded term back out. This is
        // a genuine reduction reorder (SIMD lane tree vs strict left-to-right scalar sum) -- rounding-
        // only, tolerance-safe, same idiom as every other dot product this campaign has touched.
        internal static unsafe void ApplyEtaTransposed(doubleMxN etaAlpha, int row, int slot, doubleN v, int m)
        {
            double* etaRow = etaAlpha.Data.Ptr + (long)slot * etaAlpha.N_Cols;
            double full = UnsafeOP.vecDot(etaRow, v.Data.Ptr, m);
            double t = full - etaRow[row] * v[row];
            v[row] = (v[row] - t) / etaRow[row];
        }

        // Dantzig (or Bland's-rule, when useBland) pricing over nonbasic columns. d_j = costN[j] -
        // dot(M[:,j], y); AtLower wants d_j < -tol (sigma=+1), AtUpper wants d_j > +tol (sigma=-1).
        // Fixed nonbasics (upper==lower) are never candidates (their self bound-flip step is 0).
        //
        // dWork (length N, caller-owned scratch -- see RevisedPrimalCore) holds dot(M[:,j], y) for
        // EVERY column, filled once via MTmul's row-major sweep instead of the original per-column
        // loop (see that method's own comment for the reordering note). Computing it for basic/fixed
        // columns too (which the candidate scan below still skips) is wasted work proportional to m,
        // negligible next to the O(mN) sweep it replaces.
        internal static int SelectEntering(doubleMxN M, doubleN y, doubleN cost, bool phase1,
                                           NativeArray<byte> status, doubleN lower, doubleN upper,
                                           int N, int m, bool useBland, double tol, doubleN dWork, out int sigma, out double dj)
        {
            sigma = 0; dj = (double)0;
            int best = -1; double bestMag = tol;

            MTmul(M, y, dWork, m, N);

            for (int j = 0; j < N; j++)
            {
                if (status[j] == STATUS_BASIC) continue;
                if (upper[j] - lower[j] <= (double)1e-13) continue;   // fixed: zero-length self-flip, never useful

                double costN = phase1 ? (double)0 : cost[j];
                double d = costN - dWork[j];

                int s = 0;
                if (status[j] == STATUS_AT_LOWER && d < -tol) s = 1;
                else if (status[j] == STATUS_AT_UPPER && d > tol) s = -1;
                if (s == 0) continue;

                if (useBland) { sigma = s; dj = d; return j; }   // ascending j -> first hit is smallest index

                double mag = math.abs(d);
                if (mag > bestMag) { bestMag = mag; best = j; sigma = s; dj = d; }
            }

            return best;
        }

        // Sum of basic-variable bound violations (the phase-1 composite objective's value). 0 exactly
        // at a primal-feasible basis. A pure-local double accumulator (see file header), matching
        // simplexCore's own phase-1 infeasibility sum.
        internal static double InfeasibilitySum(doubleN xB, NativeArray<int> basis, doubleN lower, doubleN upper, int m, double feasTol)
        {
            double s = 0;
            for (int i = 0; i < m; i++)
            {
                int v = basis[i];
                double xi = xB[i];
                if (xi < lower[v] - feasTol) s += (double)(lower[v] - xi);
                else if (xi > upper[v] + feasTol) s += (double)(xi - upper[v]);
            }
            return s;
        }

        // Recomputes xB fresh (solve of the adjusted rhs b - N x_N) against the CURRENT (just
        // refactorized, eta-file-empty) factorization -- the accuracy check the spec calls for at
        // refactorization time. Always trusted as authoritative.
        //
        // The nonbasic contribution was originally a per-column scatter (outer loop over nonbasic j,
        // inner loop over i reading M[i,j] with stride N -- a column gather from a row-major matrix).
        // Reshaped into one dense GEMV: build valVec (the nonbasic value per column, 0 for basic/
        // AT_LOWER-zero -- computing it for every column rather than skipping zeros is harmless, 0
        // contributes 0 to the sum either way) then adj = rhs - M*valVec via Mmul's row-major sweep.
        // Called only at refactorization events (not every iteration), so the two extra Temp
        // allocations below are inconsequential next to the per-iteration kernels above.
        internal static void RebuildXB(doubleMxN M, doubleN rhs, NativeArray<byte> status,
                                       doubleN lower, doubleN upper,
                                       doubleMxN B, in Pivot P, int m, int N, doubleN xB)
        {
            var valVec = new doubleN(N, Allocator.Temp);
            for (int j = 0; j < N; j++)
                valVec[j] = status[j] == STATUS_BASIC ? (double)0 : (status[j] == STATUS_AT_LOWER ? lower[j] : upper[j]);

            var adj = new doubleN(m, Allocator.Temp);
            Mmul(M, valVec, adj, m, N);
            for (int i = 0; i < m; i++) adj[i] = rhs[i] - adj[i];

            LU.decompSolve(ref B, in P, ref adj);
            for (int i = 0; i < m; i++) xB[i] = adj[i];
            valVec.Dispose(); adj.Dispose();
        }

        // Bounded-variable Harris two-pass ratio test (composite/far-bound rule folded in so it also
        // handles phase 1's "travel through a violated bound to the far one" requirement -- see the
        // file header/spec: for a row currently WITHIN bounds this reduces to the plain bounded ratio
        // test; for a row currently infeasible it either targets the far bound (healing direction) or
        // imposes no limit at all (worsening direction, excluded from the min).
        //   d_i = -sigma*alpha_i is the rate xB_i moves per unit step t.
        // Pass 1 (relaxed by feasTol) finds Theta = min(thetaSelf, best relaxed row limit). Pass 2 keeps
        // rows whose EXACT ratio <= Theta and picks the largest |alpha_i| among them for stability,
        // using ITS exact ratio as the actual step; the self bound-flip wins ties (it needs no pivot).
        //
        // FAR-BOUND FALLBACK (bug fix): "travel through to the far bound" assumes the far bound is
        // FINITE. In this computational form exactly one bound is ever infinite per variable kind
        // (structurals/LessEqual-logicals: lower=0 finite, upper=+INF; GreaterEqual-logicals: lower=-INF,
        // upper=0 finite) -- so a row that is CURRENTLY INFEASIBLE and healing toward an INFINITE far
        // bound contributes NO limit at all (t evaluates to a huge, INF-scale number). That is harmless
        // when at least one OTHER row still contributes a finite limit (the common case, e.g.
        // RevisedMixedSense: one infeasible >= row plus two ordinary <= rows). It is NOT harmless when
        // EVERY simultaneously-infeasible row shares this property, which happens whenever every basic
        // row is infeasible in the SAME direction with the SAME infinite far bound -- e.g. a dense
        // covering LP (min cx s.t. Ax>=b, x>=0, A,b,c>0): every >=-row logical starts basic and above its
        // upper bound (0) with an unreachable lower bound (-INF), for every row simultaneously. Then NO
        // row ever contributes a finite limit, thetaRelaxed never drops below thetaSelf, and the pass-1
        // unbounded check (thetaRelaxed >= 1e29) fires -- a false "Unbounded" despite phase 1's composite
        // objective being bounded below by 0 by construction (it can never truly be an unbounded ray).
        // Caught by the LP benchmark: RevisedSimplex returned Optimal with 0 iterations / objective 0 on
        // every dense-covering-LP instance while tableau/interior/dual all agreed on the true optimum --
        // a silent phase-1 bail (declared unbounded, which the outer driver reports as LPStatus.Unbounded,
        // extracting x=0 from a basis nothing ever pivoted into) rather than a precision issue. Reproduced
        // in LPTests.double.cs as RevisedDenseCovering (failed before this fix, passes after).
        //
        // Fix: run the SAME two passes twice. The first attempt (useFallback=false) is BYTE-IDENTICAL to
        // the original algorithm -- if it finds a finite limit from ANY row (the common case), it returns
        // immediately with UNCHANGED behavior, so every already-passing scenario (RevisedMixedSense
        // included) pivots exactly as before. Only when the first attempt would report Unbounded does a
        // second attempt run with the fallback engaged: for a row that is CURRENTLY infeasible (violating
        // its NEAR bound) and whose FAR bound is not finite (|bound| >= 1e29), use the NEAR (violated)
        // bound as the target instead -- the step at which THIS row's own violation first reaches zero,
        // always finite. This is the smallest step that cannot make phase 1's objective worse, matching
        // the standard bounded-variable composite ratio test's fallback for an unreachable far bound.
        // hitsUpper is flipped alongside the bound swap so the leaving variable still lands on the bound
        // it actually reached (the primal core's defensive finite-bound guard at the pivot site is an
        // independent second safety net for this, not a substitute for getting it right here).
        internal static void HarrisRatioTest(doubleN alpha, NativeArray<int> basis, doubleN xB,
                                             doubleN lower, doubleN upper, int m, int sigma,
                                             double thetaSelf, double feasTol, double pivTolCol,
                                             out double theta, out int leaveRow, out bool leaveHitsUpper, out bool unbounded)
        {
            double sig = (double)sigma;

            for (int attempt = 0; attempt < 2; attempt++)
            {
                bool useFallback = attempt == 1;
                double thetaRelaxed = thetaSelf;

                for (int i = 0; i < m; i++)
                {
                    double ai = alpha[i];
                    if (math.abs(ai) <= pivTolCol) continue;
                    double d = -sig * ai;
                    int v = basis[i];
                    double lo = lower[v], hi = upper[v], xi = xB[i];

                    double bound; bool limited;
                    if (d > (double)0)
                    {
                        if (xi > hi + feasTol) { limited = false; bound = (double)0; }
                        else
                        {
                            limited = true; bound = hi + feasTol;
                            if (useFallback && math.abs(hi) >= (double)1e29 && xi < lo - feasTol) bound = lo;
                        }
                    }
                    else
                    {
                        if (xi < lo - feasTol) { limited = false; bound = (double)0; }
                        else
                        {
                            limited = true; bound = lo - feasTol;
                            if (useFallback && math.abs(lo) >= (double)1e29 && xi > hi + feasTol) bound = hi;
                        }
                    }
                    if (!limited) continue;

                    double t = (bound - xi) / d;
                    if (t < (double)0) t = (double)0;
                    if (t < thetaRelaxed) thetaRelaxed = t;
                }

                if (thetaRelaxed >= (double)1e29)
                {
                    if (useFallback) { theta = thetaRelaxed; leaveRow = -1; leaveHitsUpper = false; unbounded = true; return; }
                    continue;   // retry with the far-bound fallback before declaring unbounded
                }
                unbounded = false;

                int winner = -1; double winnerAlphaMag = (double)(-1); double winnerExactT = (double)0; bool winnerHitsUpper = false;
                for (int i = 0; i < m; i++)
                {
                    double ai = alpha[i];
                    double absA = math.abs(ai);
                    if (absA <= pivTolCol) continue;
                    double d = -sig * ai;
                    int v = basis[i];
                    double lo = lower[v], hi = upper[v], xi = xB[i];

                    double bound; bool limited; bool hitsUpper;
                    if (d > (double)0)
                    {
                        hitsUpper = true;
                        if (xi > hi + feasTol) { limited = false; bound = (double)0; }
                        else
                        {
                            limited = true; bound = hi;
                            if (useFallback && math.abs(hi) >= (double)1e29 && xi < lo - feasTol) { bound = lo; hitsUpper = false; }
                        }
                    }
                    else
                    {
                        hitsUpper = false;
                        if (xi < lo - feasTol) { limited = false; bound = (double)0; }
                        else
                        {
                            limited = true; bound = lo;
                            if (useFallback && math.abs(lo) >= (double)1e29 && xi > hi + feasTol) { bound = hi; hitsUpper = true; }
                        }
                    }
                    if (!limited) continue;

                    double tExact = (bound - xi) / d;
                    if (tExact < (double)0) tExact = (double)0;
                    if (tExact <= thetaRelaxed + feasTol && absA > winnerAlphaMag)
                    {
                        winnerAlphaMag = absA; winner = i; winnerExactT = tExact; winnerHitsUpper = hitsUpper;
                    }
                }

                if (winner < 0 || thetaSelf <= winnerExactT + (double)1e-12)
                {
                    theta = thetaSelf; leaveRow = -1; leaveHitsUpper = false;
                }
                else
                {
                    theta = winnerExactT; leaveRow = winner; leaveHitsUpper = winnerHitsUpper;
                }
                return;
            }

            // Unreachable (both loop iterations return), kept only so every path assigns the out params.
            theta = thetaSelf; leaveRow = -1; leaveHitsUpper = false; unbounded = true;
        }

        // Top-level driver: phase 1 (composite objective, driving basic infeasibilities to 0) then
        // phase 2 (minimize cost), sharing the kernel above. Fills xFull (length N, caller-allocated)
        // and returns the terminal LPInfo (objective left 0 here; the double entry point recomputes it
        // from the caller's original c). Fresh all-logical start: builds that basis/status, forwards to
        // the warm-start overload below, and owns (disposes) the basis/status it allocated.
        internal static LPInfo RevisedPrimalCore(doubleMxN M, doubleN lower, doubleN upper,
                                                 doubleN cost, doubleN rhs, int m, int n, int N,
                                                 int maxIter, doubleN xFull)
        {
            var basis = new NativeArray<int>(m, Allocator.Temp);
            var status = new NativeArray<byte>(N, Allocator.Temp);
            for (int i = 0; i < m; i++) { basis[i] = n + i; status[n + i] = STATUS_BASIC; }
            for (int j = 0; j < n; j++) status[j] = STATUS_AT_LOWER;

            var info = RevisedPrimalCore(M, lower, upper, cost, rhs, m, n, N, maxIter, xFull, basis, status);

            basis.Dispose(); status.Dispose();
            return info;
        }

        // Warm-start overload -- added for LPMethod.DualSimplex's HiGHS-style composition
        // (LP.DualSimplex.double.cs hands its terminal basis to this primal core as a cleanup pass once
        // real bounds are restored). Non-breaking: the fresh-start overload above now simply builds the
        // all-logical basis/status and forwards here, so its behavior and public surface are unchanged.
        //
        // `basis`/`status` (sized m / N) must already describe a VALID assignment -- every nonbasic
        // sitting exactly on one of its (current) bounds -- but need not be feasible or all-logical; the
        // caller retains ownership (this method reads/mutates them in place, never allocates or disposes
        // them). Phase 1 vs phase 2 is decided the same way as always (InfeasibilitySum against the
        // FRESH xB rebuilt from this basis), so an infeasible warm start is cleaned up automatically.
        internal static LPInfo RevisedPrimalCore(doubleMxN M, doubleN lower, doubleN upper,
                                                 doubleN cost, doubleN rhs, int m, int n, int N,
                                                 int maxIter, doubleN xFull,
                                                 NativeArray<int> basis, NativeArray<byte> status)
        {
            // Per-dtype tolerances, derived from the SAME Consts the tableau simplex (simplexCore)
            // already uses -- see file header. pivTol: absolute pivot-rejection floor. feasTol/dualTol:
            // feasibility/dual tolerance shared by the ratio test and entering-column pricing (mirrors
            // stage 1's original feasTol/dualTol, which were equal constants). Computed inline (not a
            // shared named helper) because a helper returning double with no double-typed parameter
            // would differ ONLY by return type between the float and double generated fragments -- C#
            // does not overload on return type alone, so float's and double's copies would collide as
            // duplicate members of the same partial class. Stage 2 (LP.DualSimplex.double.cs) computes
            // the identical expressions inline for the same reason.
            double pivTol = math.max(Consts.doubleZeroThreshold, (double)1e-9);
            double feasTol = (double)math.max(math.sqrt((double)Consts.doubleEpsilon), 1e-7);
            double dualTol = feasTol;

            var xB = new doubleN(m, Allocator.Temp);

            var B = new doubleMxN(m, m, Allocator.Temp);
            var P = new Pivot(m, Allocator.Temp);
            var etaAlpha = new doubleMxN(REFACTOR_INTERVAL, m, Allocator.Temp);
            var etaRow = new NativeArray<int>(REFACTOR_INTERVAL, Allocator.Temp);
            int etaCount = 0;

            var y = new doubleN(m, Allocator.Temp);
            var cB = new doubleN(m, Allocator.Temp);
            var alpha = new doubleN(m, Allocator.Temp);
            var dWork = new doubleN(N, Allocator.Temp);   // SelectEntering's per-iteration MTmul scratch

            LPStatus resultStatus = LPStatus.Optimal;
            int iters = 0;

            bool ok = Refactorize(M, basis, B, ref P, m, N);
            if (!ok) resultStatus = LPStatus.MaxIterations;
            else RebuildXB(M, rhs, status, lower, upper, B, in P, m, N, xB);

            int budget = maxIter > 0 ? maxIter : 50 * (m + N) + 200;
            int degenCount = 0;
            bool useBland = false;
            int phase = (resultStatus == LPStatus.Optimal && InfeasibilitySum(xB, basis, lower, upper, m, feasTol) > (double)feasTol * math.max(m, 1)) ? 1 : 2;

            while (resultStatus == LPStatus.Optimal)
            {
                if (iters >= budget) { resultStatus = LPStatus.MaxIterations; break; }

                for (int i = 0; i < m; i++)
                {
                    int v = basis[i];
                    if (phase == 1)
                    {
                        double xi = xB[i];
                        cB[i] = xi > upper[v] + feasTol ? (double)1 : (xi < lower[v] - feasTol ? (double)(-1) : (double)0);
                    }
                    else cB[i] = cost[v];
                }
                for (int i = 0; i < m; i++) y[i] = cB[i];
                Btran(B, in P, etaAlpha, etaRow, etaCount, y, m);

                int enter = SelectEntering(M, y, cost, phase == 1, status, lower, upper, N, m, useBland, dualTol, dWork, out int sigma, out _);

                if (enter < 0)
                {
                    if (phase == 1)
                    {
                        double infeas = InfeasibilitySum(xB, basis, lower, upper, m, feasTol);
                        if (infeas <= (double)feasTol * math.max(m, 1)) { phase = 2; degenCount = 0; useBland = false; continue; }
                        resultStatus = LPStatus.Infeasible;
                        break;
                    }
                    resultStatus = LPStatus.Optimal;
                    break;
                }

                for (int i = 0; i < m; i++) alpha[i] = M[i, enter];
                Ftran(B, in P, etaAlpha, etaRow, etaCount, alpha, m);

                double colMax = (double)0;
                for (int i = 0; i < m; i++) colMax = math.max(colMax, math.abs(alpha[i]));
                double pivTolCol = math.max(pivTol, (double)1e-6 * colMax);

                double thetaSelf = upper[enter] - lower[enter];

                HarrisRatioTest(alpha, basis, xB, lower, upper, m, sigma, thetaSelf, feasTol, pivTolCol,
                                out double theta, out int leaveRow, out bool leaveHitsUpper, out bool unbounded);

                if (unbounded) { resultStatus = LPStatus.Unbounded; break; }

                bool degenerate = theta < pivTol;
                if (degenerate) { degenCount++; if (degenCount >= 3 * math.max(m, 1)) useBland = true; }
                else { degenCount = 0; useBland = false; }

                double sig = (double)sigma;
                if (leaveRow < 0)
                {
                    // bound flip: entering variable reaches its own opposite bound, no basis change
                    for (int i = 0; i < m; i++) xB[i] -= sig * theta * alpha[i];
                    status[enter] = sigma > 0 ? STATUS_AT_UPPER : STATUS_AT_LOWER;
                }
                else
                {
                    double enteringValue = (sigma > 0 ? lower[enter] : upper[enter]) + sig * theta;
                    int leavingVar = basis[leaveRow];

                    for (int i = 0; i < m; i++) xB[i] -= sig * theta * alpha[i];
                    xB[leaveRow] = enteringValue;

                    bool fixedLeaving = upper[leavingVar] - lower[leavingVar] <= (double)1e-13;
                    bool hitsUpperFinal = leaveHitsUpper && !fixedLeaving;
                    // Invariant: a nonbasic variable must rest on a FINITE bound -- RebuildXB and the
                    // xFull extraction both read lower[]/upper[] straight into xB/x, so resting on an
                    // INF (1e30) sentinel would poison every downstream solve with 1e30-scale arithmetic.
                    // HarrisRatioTest's own `unbounded` check already excludes this in a well-posed
                    // problem (a row can only "win" with an infinite bound if nothing else limits the
                    // step, which is exactly the unbounded case, caught before reaching here), but guard
                    // the assignment defensively anyway: fall back to the OTHER bound if the intended one
                    // isn't finite. Every variable in this computational form has at least one finite
                    // bound (structurals/LessEqual/Equal's lower and GreaterEqual's upper are always 0).
                    if (hitsUpperFinal && math.abs(upper[leavingVar]) >= (double)1e29) hitsUpperFinal = false;
                    else if (!hitsUpperFinal && math.abs(lower[leavingVar]) >= (double)1e29) hitsUpperFinal = true;
                    status[leavingVar] = hitsUpperFinal ? STATUS_AT_UPPER : STATUS_AT_LOWER;
                    basis[leaveRow] = enter;
                    status[enter] = STATUS_BASIC;

                    bool needRefactor = etaCount >= REFACTOR_INTERVAL || math.abs(alpha[leaveRow]) < pivTol;
                    if (needRefactor)
                    {
                        ok = Refactorize(M, basis, B, ref P, m, N);
                        etaCount = 0;
                        if (!ok) { resultStatus = LPStatus.MaxIterations; break; }
                        RebuildXB(M, rhs, status, lower, upper, B, in P, m, N, xB);
                    }
                    else
                    {
                        for (int i = 0; i < m; i++) etaAlpha[etaCount, i] = alpha[i];
                        etaRow[etaCount] = leaveRow;
                        etaCount++;
                    }
                }

                iters++;
            }

            for (int j = 0; j < N; j++)
                xFull[j] = status[j] == STATUS_BASIC ? (double)0 : (status[j] == STATUS_AT_LOWER ? lower[j] : upper[j]);
            for (int i = 0; i < m; i++) xFull[basis[i]] = xB[i];

            xB.Dispose();
            B.Dispose(); P.Dispose();
            etaAlpha.Dispose(); etaRow.Dispose();
            y.Dispose(); cB.Dispose(); alpha.Dispose(); dWork.Dispose();

            return new LPInfo { status = resultStatus, iterations = iters, objective = 0 };
        }
    }
}
