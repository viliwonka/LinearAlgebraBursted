#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // ================================================================================================
    // Bounded-variable PRIMAL revised simplex -- the LPMethod.RevisedSimplex backend, stage 1 of the
    // HiGHS-style dense revised-simplex port (docs/spec-revised-simplex.md). Instead of updating a full
    // m x (n+m) tableau every pivot (simplexCore, LP.fProxy.cs -- left untouched), this keeps the
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
    // file, reduced costs, ratios -- is ordinary fProxy, exactly like every other solver in this library
    // (simplexCore, interiorCore). Tolerances are derived from the SAME per-dtype Consts the tableau
    // simplex already uses (Consts.fProxyZeroThreshold / fProxyEpsilon), which resolve to the float or
    // double constant via the usual token substitution -- no separate double-only code path, and no
    // inline per-type literal marker (see GenUtils.cs) needed for these since Consts already IS
    // per-dtype. They are computed INLINE at each call site rather than behind a shared helper method:
    // a helper returning fProxy with no fProxy-typed parameter (e.g. `fProxy Tol()`) differs only in its
    // RETURN type between the float- and double-generated fragments of this partial class, and C# does
    // not overload on return type alone -- both copies would collide as duplicate members (CS0111). Any
    // new shared fProxy-returning helper needs at least one fProxy-typed parameter to stay a genuine
    // per-type overload; otherwise, inline it. Objective/diagnostic SUMS that are pure locals (never fed
    // back into array storage) still accumulate in `double` for precision, matching simplexCore's own
    // convention (its phase-1 infeasibility sum) and this file's fProxy entry point (the final cᵀx
    // recompute).
    //
    // Basis factorization: reuses the library's OWN `LU` class (LU.decompInPlace, the compact in-place
    // form) and `Blas`'s compact-LU triangular solves (triLowerLU/triUpperLU via LU.decompSolve) for
    // FTRAN -- no reimplementation of GETRF. BTRAN (solve Bᵀy = v) has no library primitive to reuse
    // (Blas only ships forward compact-LU solves), so it is a small hand-written transposed forward/back
    // substitution (SolveTranspose below) over the SAME compact fProxyMxN + Pivot factor LU.decompInPlace
    // produces -- not a different algorithm, just the transposed read pattern of the identical factor
    // (see SolveTranspose's doc comment for the derivation: A = Pᵀ L U ⟹ Aᵀ = Uᵀ Lᵀ P).
    //
    // Every kernel function below is `internal static` (not buried locals) precisely so
    // LP.DualSimplex.fProxy.cs (stage 2) can call them directly: ordinary partial-class member
    // resolution within the same generated type (float calls float, double calls double), exactly like
    // interiorCore's Amul/ATmul/BuildNormalStructured already work across this same file. REFACTOR_
    // INTERVAL and the STATUS_* tags are type-agnostic and therefore declared once in LP.Info.cs
    // (singularFile) instead of here, where a per-dtype duplicate would collide (CS0102) -- see that
    // file's comment.
    // ================================================================================================
    public static partial class LP
    {
        // ---- fProxy-typed entry point: builds the computational form, hands off to the (now equally
        // fProxy-typed) core, copies the result back. ----
        static LPInfo revisedSimplexCore(in fProxyMxN A, in fProxyN b, in fProxyN c,
                                         in NativeArray<ConstraintSense> senses,
                                         ref fProxyN x, out double objective, int maxIter)
        {
            int m = A.M_Rows, n = A.N_Cols, N = n + m;

            var M = new fProxyMxN(m, N, Allocator.Temp);      // zero-initialized: [A | I], row-major
            var lower = new fProxyN(N, Allocator.Temp);
            var upper = new fProxyN(N, Allocator.Temp);
            var cost = new fProxyN(N, Allocator.Temp);
            var rhs = new fProxyN(m, Allocator.Temp);

            BuildComputationalForm(in A, in b, in c, in senses, M, lower, upper, cost, rhs, m, n, N);

            var xFull = new fProxyN(N, Allocator.Temp);
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

        // Builds M=[A|I] (m x N), bounds, cost and rhs from the fProxy inputs. Shared verbatim by stage
        // 2 (LP.DualSimplex.fProxy.cs calls this too -- same partial class, same per-dtype fragment).
        internal static void BuildComputationalForm(in fProxyMxN A, in fProxyN b, in fProxyN c,
                                           in NativeArray<ConstraintSense> senses,
                                           fProxyMxN M, fProxyN lower, fProxyN upper,
                                           fProxyN cost, fProxyN rhs, int m, int n, int N)
        {
            fProxy INF = (fProxy)1e30;

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++) M[i, j] = A[i, j];
                M[i, n + i] = (fProxy)1;            // logical column e_i (M was zero-initialized)
                rhs[i] = b[i];
            }

            for (int j = 0; j < n; j++) { lower[j] = (fProxy)0; upper[j] = INF; cost[j] = c[j]; }

            for (int i = 0; i < m; i++)
            {
                int j = n + i;
                cost[j] = (fProxy)0;
                switch (senses[i])
                {
                    case ConstraintSense.LessEqual: lower[j] = (fProxy)0; upper[j] = INF; break;
                    case ConstraintSense.Equal: lower[j] = (fProxy)0; upper[j] = (fProxy)0; break;
                    default: lower[j] = -INF; upper[j] = (fProxy)0; break;      // GreaterEqual
                }
            }
        }

        // Solve (Bᵀ) y = v given B's compact LU factor (A = Pᵀ L U, see LU.decompInPlace's own doc: "row
        // i lives at physical row P[i]", i.e. the permuted matrix A'[i,:] = A[P[i],:] equals L*U). Then
        // Aᵀ = Uᵀ Lᵀ P, so y = Pᵀ (L⁻ᵀ (U⁻ᵀ v)): forward through Uᵀ (lower-triangular, reading LU[P[j],k]
        // for j<=k -- the same physical entries Blas.triUpperLU reads as U[k,c] for c>=k), backward
        // through Lᵀ (unit-upper, reading LU[P[j],k] for j>k -- the same entries Blas.triLowerLU reads as
        // L[r,c] for c<r), then scatter by P. No library primitive does this (Blas ships only the
        // forward compact-LU solves triLowerLU/triUpperLU), so it is hand-written here over the SAME
        // factor LU.decompInPlace produces -- the transposed read pattern of an existing factor, not a
        // different algorithm. In place in v.
        internal static void SolveTranspose(fProxyMxN LU, in Pivot P, fProxyN v, int m)
        {
            var w = new fProxyN(m, Allocator.Temp);
            for (int i = 0; i < m; i++) w[i] = v[i];

            for (int k = 0; k < m; k++)
            {
                fProxy sum = (fProxy)0;
                for (int j = 0; j < k; j++) sum += LU[P[j], k] * w[j];
                w[k] = (w[k] - sum) / LU[P[k], k];
            }
            for (int k = m - 1; k >= 0; k--)
            {
                fProxy sum = (fProxy)0;
                for (int j = k + 1; j < m; j++) sum += LU[P[j], k] * w[j];
                w[k] -= sum;
            }

            for (int k = 0; k < m; k++) v[P[k]] = w[k];
            w.Dispose();
        }

        // Rebuilds the m x m basis matrix B (column k = column basis[k] of M) and LU-factors it in place
        // via the library's own LU.decompInPlace (compact form, partial pivoting -- no reimplementation
        // of GETRF). Returns false on a singular basis.
        internal static bool Refactorize(fProxyMxN M, NativeArray<int> basis, fProxyMxN B, ref Pivot P, int m, int N)
        {
            for (int k = 0; k < m; k++)
            {
                int col = basis[k];
                for (int i = 0; i < m; i++) B[i, k] = M[i, col];
            }
            return LU.decompInPlace(ref B, ref P).Solved;
        }

        // FTRAN: solve (current basis) * alpha = v for alpha, in place. Base LU solve (library's
        // LU.decompSolve, compact form) against the last refactorization, then the PFI eta file applied
        // in chronological (creation) order.
        internal static void Ftran(fProxyMxN B, in Pivot P,
                                   fProxyMxN etaAlpha, NativeArray<int> etaRow, int etaCount,
                                   fProxyN v, int m)
        {
            LU.decompSolve(ref B, in P, ref v);
            for (int k = 0; k < etaCount; k++)
                ApplyEtaForward(etaAlpha, etaRow[k], k, v, m);
        }

        // BTRAN: solve (current basis)ᵀ * y = v for y. Eta file applied in REVERSE + transposed FIRST,
        // then the transposed base solve (SolveTranspose above).
        internal static void Btran(fProxyMxN B, in Pivot P,
                                   fProxyMxN etaAlpha, NativeArray<int> etaRow, int etaCount,
                                   fProxyN v, int m)
        {
            for (int k = etaCount - 1; k >= 0; k--)
                ApplyEtaTransposed(etaAlpha, etaRow[k], k, v, m);

            SolveTranspose(B, in P, v, m);
        }

        // Applies eta slot `slot` (leaving row `row`, column alpha_q stored in etaAlpha[slot,:]) via
        // E^-1: v[row] /= alpha[row]; v[i] -= alpha[i]*v[row] for i != row (spec's PFI formula; v[row]'s
        // NEW value is what every other entry subtracts, so it must be updated first).
        internal static void ApplyEtaForward(fProxyMxN etaAlpha, int row, int slot, fProxyN v, int m)
        {
            v[row] = v[row] / etaAlpha[slot, row];
            fProxy vr = v[row];
            for (int i = 0; i < m; i++)
                if (i != row) v[i] -= etaAlpha[slot, i] * vr;
        }

        // Applies (E^-1)^T: y[row] = (v[row] - sum_{i!=row} alpha[i]*v[i]) / alpha[row]; every other
        // entry of v is left untouched (derived in the file header comment).
        internal static void ApplyEtaTransposed(fProxyMxN etaAlpha, int row, int slot, fProxyN v, int m)
        {
            fProxy t = (fProxy)0;
            for (int i = 0; i < m; i++)
                if (i != row) t += etaAlpha[slot, i] * v[i];
            v[row] = (v[row] - t) / etaAlpha[slot, row];
        }

        // Dantzig (or Bland's-rule, when useBland) pricing over nonbasic columns. d_j = costN[j] -
        // dot(M[:,j], y); AtLower wants d_j < -tol (sigma=+1), AtUpper wants d_j > +tol (sigma=-1).
        // Fixed nonbasics (upper==lower) are never candidates (their self bound-flip step is 0).
        internal static int SelectEntering(fProxyMxN M, fProxyN y, fProxyN cost, bool phase1,
                                           NativeArray<byte> status, fProxyN lower, fProxyN upper,
                                           int N, int m, bool useBland, fProxy tol, out int sigma, out fProxy dj)
        {
            sigma = 0; dj = (fProxy)0;
            int best = -1; fProxy bestMag = tol;

            for (int j = 0; j < N; j++)
            {
                if (status[j] == STATUS_BASIC) continue;
                if (upper[j] - lower[j] <= (fProxy)1e-13) continue;   // fixed: zero-length self-flip, never useful

                fProxy costN = phase1 ? (fProxy)0 : cost[j];
                fProxy d = costN;
                for (int i = 0; i < m; i++) d -= M[i, j] * y[i];

                int s = 0;
                if (status[j] == STATUS_AT_LOWER && d < -tol) s = 1;
                else if (status[j] == STATUS_AT_UPPER && d > tol) s = -1;
                if (s == 0) continue;

                if (useBland) { sigma = s; dj = d; return j; }   // ascending j -> first hit is smallest index

                fProxy mag = math.abs(d);
                if (mag > bestMag) { bestMag = mag; best = j; sigma = s; dj = d; }
            }

            return best;
        }

        // Sum of basic-variable bound violations (the phase-1 composite objective's value). 0 exactly
        // at a primal-feasible basis. A pure-local double accumulator (see file header), matching
        // simplexCore's own phase-1 infeasibility sum.
        internal static double InfeasibilitySum(fProxyN xB, NativeArray<int> basis, fProxyN lower, fProxyN upper, int m, fProxy feasTol)
        {
            double s = 0;
            for (int i = 0; i < m; i++)
            {
                int v = basis[i];
                fProxy xi = xB[i];
                if (xi < lower[v] - feasTol) s += (double)(lower[v] - xi);
                else if (xi > upper[v] + feasTol) s += (double)(xi - upper[v]);
            }
            return s;
        }

        // Recomputes xB fresh (solve of the adjusted rhs b - N x_N) against the CURRENT (just
        // refactorized, eta-file-empty) factorization -- the accuracy check the spec calls for at
        // refactorization time. Always trusted as authoritative.
        internal static void RebuildXB(fProxyMxN M, fProxyN rhs, NativeArray<byte> status,
                                       fProxyN lower, fProxyN upper,
                                       fProxyMxN B, in Pivot P, int m, int N, fProxyN xB)
        {
            var adj = new fProxyN(m, Allocator.Temp);
            for (int i = 0; i < m; i++) adj[i] = rhs[i];

            for (int j = 0; j < N; j++)
            {
                if (status[j] == STATUS_BASIC) continue;
                fProxy val = status[j] == STATUS_AT_LOWER ? lower[j] : upper[j];
                if (val == (fProxy)0) continue;
                for (int i = 0; i < m; i++) adj[i] -= M[i, j] * val;
            }

            LU.decompSolve(ref B, in P, ref adj);
            for (int i = 0; i < m; i++) xB[i] = adj[i];
            adj.Dispose();
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
        // in LPTests.fProxy.cs as RevisedDenseCovering (failed before this fix, passes after).
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
        internal static void HarrisRatioTest(fProxyN alpha, NativeArray<int> basis, fProxyN xB,
                                             fProxyN lower, fProxyN upper, int m, int sigma,
                                             fProxy thetaSelf, fProxy feasTol, fProxy pivTolCol,
                                             out fProxy theta, out int leaveRow, out bool leaveHitsUpper, out bool unbounded)
        {
            fProxy sig = (fProxy)sigma;

            for (int attempt = 0; attempt < 2; attempt++)
            {
                bool useFallback = attempt == 1;
                fProxy thetaRelaxed = thetaSelf;

                for (int i = 0; i < m; i++)
                {
                    fProxy ai = alpha[i];
                    if (math.abs(ai) <= pivTolCol) continue;
                    fProxy d = -sig * ai;
                    int v = basis[i];
                    fProxy lo = lower[v], hi = upper[v], xi = xB[i];

                    fProxy bound; bool limited;
                    if (d > (fProxy)0)
                    {
                        if (xi > hi + feasTol) { limited = false; bound = (fProxy)0; }
                        else
                        {
                            limited = true; bound = hi + feasTol;
                            if (useFallback && math.abs(hi) >= (fProxy)1e29 && xi < lo - feasTol) bound = lo;
                        }
                    }
                    else
                    {
                        if (xi < lo - feasTol) { limited = false; bound = (fProxy)0; }
                        else
                        {
                            limited = true; bound = lo - feasTol;
                            if (useFallback && math.abs(lo) >= (fProxy)1e29 && xi > hi + feasTol) bound = hi;
                        }
                    }
                    if (!limited) continue;

                    fProxy t = (bound - xi) / d;
                    if (t < (fProxy)0) t = (fProxy)0;
                    if (t < thetaRelaxed) thetaRelaxed = t;
                }

                if (thetaRelaxed >= (fProxy)1e29)
                {
                    if (useFallback) { theta = thetaRelaxed; leaveRow = -1; leaveHitsUpper = false; unbounded = true; return; }
                    continue;   // retry with the far-bound fallback before declaring unbounded
                }
                unbounded = false;

                int winner = -1; fProxy winnerAlphaMag = (fProxy)(-1); fProxy winnerExactT = (fProxy)0; bool winnerHitsUpper = false;
                for (int i = 0; i < m; i++)
                {
                    fProxy ai = alpha[i];
                    fProxy absA = math.abs(ai);
                    if (absA <= pivTolCol) continue;
                    fProxy d = -sig * ai;
                    int v = basis[i];
                    fProxy lo = lower[v], hi = upper[v], xi = xB[i];

                    fProxy bound; bool limited; bool hitsUpper;
                    if (d > (fProxy)0)
                    {
                        hitsUpper = true;
                        if (xi > hi + feasTol) { limited = false; bound = (fProxy)0; }
                        else
                        {
                            limited = true; bound = hi;
                            if (useFallback && math.abs(hi) >= (fProxy)1e29 && xi < lo - feasTol) { bound = lo; hitsUpper = false; }
                        }
                    }
                    else
                    {
                        hitsUpper = false;
                        if (xi < lo - feasTol) { limited = false; bound = (fProxy)0; }
                        else
                        {
                            limited = true; bound = lo;
                            if (useFallback && math.abs(lo) >= (fProxy)1e29 && xi > hi + feasTol) { bound = hi; hitsUpper = true; }
                        }
                    }
                    if (!limited) continue;

                    fProxy tExact = (bound - xi) / d;
                    if (tExact < (fProxy)0) tExact = (fProxy)0;
                    if (tExact <= thetaRelaxed + feasTol && absA > winnerAlphaMag)
                    {
                        winnerAlphaMag = absA; winner = i; winnerExactT = tExact; winnerHitsUpper = hitsUpper;
                    }
                }

                if (winner < 0 || thetaSelf <= winnerExactT + (fProxy)1e-12)
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
        // and returns the terminal LPInfo (objective left 0 here; the fProxy entry point recomputes it
        // from the caller's original c). Fresh all-logical start: builds that basis/status, forwards to
        // the warm-start overload below, and owns (disposes) the basis/status it allocated.
        internal static LPInfo RevisedPrimalCore(fProxyMxN M, fProxyN lower, fProxyN upper,
                                                 fProxyN cost, fProxyN rhs, int m, int n, int N,
                                                 int maxIter, fProxyN xFull)
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
        // (LP.DualSimplex.fProxy.cs hands its terminal basis to this primal core as a cleanup pass once
        // real bounds are restored). Non-breaking: the fresh-start overload above now simply builds the
        // all-logical basis/status and forwards here, so its behavior and public surface are unchanged.
        //
        // `basis`/`status` (sized m / N) must already describe a VALID assignment -- every nonbasic
        // sitting exactly on one of its (current) bounds -- but need not be feasible or all-logical; the
        // caller retains ownership (this method reads/mutates them in place, never allocates or disposes
        // them). Phase 1 vs phase 2 is decided the same way as always (InfeasibilitySum against the
        // FRESH xB rebuilt from this basis), so an infeasible warm start is cleaned up automatically.
        internal static LPInfo RevisedPrimalCore(fProxyMxN M, fProxyN lower, fProxyN upper,
                                                 fProxyN cost, fProxyN rhs, int m, int n, int N,
                                                 int maxIter, fProxyN xFull,
                                                 NativeArray<int> basis, NativeArray<byte> status)
        {
            // Per-dtype tolerances, derived from the SAME Consts the tableau simplex (simplexCore)
            // already uses -- see file header. pivTol: absolute pivot-rejection floor. feasTol/dualTol:
            // feasibility/dual tolerance shared by the ratio test and entering-column pricing (mirrors
            // stage 1's original feasTol/dualTol, which were equal constants). Computed inline (not a
            // shared named helper) because a helper returning fProxy with no fProxy-typed parameter
            // would differ ONLY by return type between the float and double generated fragments -- C#
            // does not overload on return type alone, so float's and double's copies would collide as
            // duplicate members of the same partial class. Stage 2 (LP.DualSimplex.fProxy.cs) computes
            // the identical expressions inline for the same reason.
            fProxy pivTol = math.max(Consts.fProxyZeroThreshold, (fProxy)1e-9);
            fProxy feasTol = (fProxy)math.max(math.sqrt((double)Consts.fProxyEpsilon), 1e-7);
            fProxy dualTol = feasTol;

            var xB = new fProxyN(m, Allocator.Temp);

            var B = new fProxyMxN(m, m, Allocator.Temp);
            var P = new Pivot(m, Allocator.Temp);
            var etaAlpha = new fProxyMxN(REFACTOR_INTERVAL, m, Allocator.Temp);
            var etaRow = new NativeArray<int>(REFACTOR_INTERVAL, Allocator.Temp);
            int etaCount = 0;

            var y = new fProxyN(m, Allocator.Temp);
            var cB = new fProxyN(m, Allocator.Temp);
            var alpha = new fProxyN(m, Allocator.Temp);

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
                        fProxy xi = xB[i];
                        cB[i] = xi > upper[v] + feasTol ? (fProxy)1 : (xi < lower[v] - feasTol ? (fProxy)(-1) : (fProxy)0);
                    }
                    else cB[i] = cost[v];
                }
                for (int i = 0; i < m; i++) y[i] = cB[i];
                Btran(B, in P, etaAlpha, etaRow, etaCount, y, m);

                int enter = SelectEntering(M, y, cost, phase == 1, status, lower, upper, N, m, useBland, dualTol, out int sigma, out _);

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

                fProxy colMax = (fProxy)0;
                for (int i = 0; i < m; i++) colMax = math.max(colMax, math.abs(alpha[i]));
                fProxy pivTolCol = math.max(pivTol, (fProxy)1e-6 * colMax);

                fProxy thetaSelf = upper[enter] - lower[enter];

                HarrisRatioTest(alpha, basis, xB, lower, upper, m, sigma, thetaSelf, feasTol, pivTolCol,
                                out fProxy theta, out int leaveRow, out bool leaveHitsUpper, out bool unbounded);

                if (unbounded) { resultStatus = LPStatus.Unbounded; break; }

                bool degenerate = theta < pivTol;
                if (degenerate) { degenCount++; if (degenCount >= 3 * math.max(m, 1)) useBland = true; }
                else { degenCount = 0; useBland = false; }

                fProxy sig = (fProxy)sigma;
                if (leaveRow < 0)
                {
                    // bound flip: entering variable reaches its own opposite bound, no basis change
                    for (int i = 0; i < m; i++) xB[i] -= sig * theta * alpha[i];
                    status[enter] = sigma > 0 ? STATUS_AT_UPPER : STATUS_AT_LOWER;
                }
                else
                {
                    fProxy enteringValue = (sigma > 0 ? lower[enter] : upper[enter]) + sig * theta;
                    int leavingVar = basis[leaveRow];

                    for (int i = 0; i < m; i++) xB[i] -= sig * theta * alpha[i];
                    xB[leaveRow] = enteringValue;

                    bool fixedLeaving = upper[leavingVar] - lower[leavingVar] <= (fProxy)1e-13;
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
                    if (hitsUpperFinal && math.abs(upper[leavingVar]) >= (fProxy)1e29) hitsUpperFinal = false;
                    else if (!hitsUpperFinal && math.abs(lower[leavingVar]) >= (fProxy)1e29) hitsUpperFinal = true;
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
                xFull[j] = status[j] == STATUS_BASIC ? (fProxy)0 : (status[j] == STATUS_AT_LOWER ? lower[j] : upper[j]);
            for (int i = 0; i < m; i++) xFull[basis[i]] = xB[i];

            xB.Dispose();
            B.Dispose(); P.Dispose();
            etaAlpha.Dispose(); etaRow.Dispose();
            y.Dispose(); cB.Dispose(); alpha.Dispose();

            return new LPInfo { status = resultStatus, iterations = iters, objective = 0 };
        }
    }
}
