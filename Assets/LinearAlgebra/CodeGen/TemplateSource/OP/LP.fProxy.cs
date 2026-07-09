#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // ================================================================================================
    // Linear programming + L1 (least-absolute-deviation) optimization.
    //
    // Canonical primal form solved by the public entry points:
    //
    //     minimize    cᵀx
    //     subject to  Aᵢ·x  {≤, =, ≥}  bᵢ    (per-row sense in `senses`)
    //                 x ≥ 0
    //
    // Two backends reach the same optimal vertex on a bounded, feasible problem (see LPMethod):
    //   * Simplex       -- two-phase dense tableau, Bland's anti-cycling rule (this file).
    //   * InteriorPoint -- Mehrotra predictor-corrector (LP.InteriorPoint.fProxy.cs).
    //
    // L1 regression (least absolute deviation) is the flagship application: minimize ‖Ax − b‖₁ over a
    // FREE x is exactly an LP once each residual is split into a +/− pair (see `lad`). A fast,
    // approximate iteratively-reweighted-least-squares alternative lives in Optimize.ladIRLS.
    //
    // Job-safe: the cores allocate their (problem-size-dependent) scratch from Allocator.Temp and
    // dispose it before returning, so the whole thing runs inside a [BurstCompile] IJob with no arena.
    // ================================================================================================
    public static partial class LP
    {
        /// <summary>
        /// Solve the linear program  min cᵀx  s.t.  A x {≤,=,≥} b (per-row <paramref name="senses"/>),
        /// x ≥ 0.  Writes the optimal x (length A.N_Cols) and returns an <see cref="LPInfo"/>
        /// (objective = cᵀx, implicit-bool == reached an optimum). Variables are non-negative; model a
        /// free variable by splitting it into a +/− pair, or use <see cref="lad"/> which does that for
        /// you.
        /// </summary>
        /// <param name="A">Constraint coefficients, m×n (m constraints, n variables).</param>
        /// <param name="b">Right-hand sides, length m. Any sign (negative rows are normalized internally).</param>
        /// <param name="c">Objective coefficients, length n (minimized).</param>
        /// <param name="senses">Per-row constraint sense, length m.</param>
        /// <param name="x">Output solution, length n (overwritten).</param>
        /// <param name="objective">Output cᵀx at the returned x.</param>
        /// <param name="method">Backend (default RevisedSimplex — fastest exact backend at every
        /// benchmarked size on cold solves and the fastest infeasibility certifier (1-2 pivots);
        /// pick <see cref="LPMethod.DualSimplex"/> explicitly for re-solves from a near-dual-feasible
        /// state, <see cref="LPMethod.InteriorPoint"/> for very ill-conditioned vertices, and
        /// <see cref="LPMethod.Simplex"/> (dense tableau) as the reference implementation.</param>
        /// <param name="maxIter">Pivot/iteration budget; ≤0 picks a size-based default.</param>
        public static LPInfo solve(in fProxyMxN A, in fProxyN b, in fProxyN c,
                                   in NativeArray<ConstraintSense> senses,
                                   ref fProxyN x, out double objective,
                                   LPMethod method = LPMethod.RevisedSimplex, int maxIter = 0)
        {
            int m = A.M_Rows, n = A.N_Cols;

            if (b.N != m) throw new ArgumentException("LP.solve: b.N must equal A.M_Rows");
            if (c.N != n) throw new ArgumentException("LP.solve: c.N must equal A.N_Cols");
            if (senses.Length != m) throw new ArgumentException("LP.solve: senses.Length must equal A.M_Rows");
            if (x.N != n) throw new ArgumentException("LP.solve: x.N must equal A.N_Cols");

            if (method == LPMethod.Simplex)
                return simplexCore(in A, in b, in c, in senses, ref x, out objective, maxIter);
            if (method == LPMethod.RevisedSimplex)
                return revisedSimplexCore(in A, in b, in c, in senses, ref x, out objective, maxIter);
            if (method == LPMethod.DualSimplex)
                return dualSimplexCore(in A, in b, in c, in senses, ref x, out objective, maxIter);
            return interiorCore(in A, in b, in c, in senses, ref x, out objective, maxIter);
        }

        /// <summary>
        /// Least absolute deviation (L1 regression): minimize ‖A x − b‖₁ over a FREE x ∈ ℝⁿ. Robust to
        /// outliers where ordinary least squares (which minimizes the L2 norm) is not. This overload
        /// is a HYBRID: it picks between this library's two reformulation-free exact engines by problem
        /// size (m = A.M_Rows) --
        /// <see cref="ladBR(in fProxyMxN, in fProxyN, ref fProxyN, out double, int)"/> (Barrodale-
        /// Roberts specialized simplex) at or below the per-dtype threshold in the dispatch below,
        /// <see cref="ladFN(in fProxyMxN, in fProxyN, ref fProxyN, out double, int)"/> (Frisch-Newton
        /// interior point) above it. BR's weighted-median long step wins at small-to-moderate m (near-
        /// constant, few-microsecond latency -- the common low-observation-count case); FN's fixed
        /// ~10-iteration n×n normal solve wins once m grows large enough that BR's per-pivot sweep over
        /// m rows dominates. The crossover is a measured, re-tunable, PER-DTYPE value (see the comment
        /// on the dispatch expression below for the benchmark it was set from), not
        /// a fixed property of either algorithm. Call <see cref="ladBR"/> / <see cref="ladFN"/> directly
        /// to bypass this routing and force one engine regardless of size. Use the <see cref="LPMethod"/>
        /// overload to route through the general LP backends instead (the classic split-variable
        /// reformulation; exact but far slower than either hybrid route -- retained mainly as
        /// independent cross-checks). For the approximate iteratively-reweighted alternative, see
        /// <see cref="Optimize.ladIRLS"/>.
        /// </summary>
        /// <param name="A">Design matrix, m×n (m observations, n coefficients). m ≥ n typical.</param>
        /// <param name="b">Observations, length m.</param>
        /// <param name="x">Output coefficients, length n (overwritten). May be negative.</param>
        /// <param name="objective">Output L1 residual ‖A x − b‖₁.</param>
        /// <param name="maxIter">Iteration budget; ≤0 picks the routed engine's own default.</param>
        public static LPInfo lad(in fProxyMxN A, in fProxyN b, ref fProxyN x, out double objective,
                                 int maxIter = 0)
            // MEASURED, RE-TUNABLE, PER-DTYPE crossover (LPBenchmark Section 2b, 2026-07-09, AFTER the
            // BR sort-path + FN SIMD optimization round): double -- BR wins through m=4096 (2.49ms vs
            // FN 2.71ms) and loses only ~11% at m=16384, so the threshold sits at the last measured
            // BR-win size, 4096; float -- FN's SIMD gains moved its win boundary down to m=1024
            // (FN 0.47ms vs BR 0.62ms) while BR still wins at m=384, so 512 splits the measured
            // bracket. Re-measure Section 2b (and re-tune here) whenever either engine's per-iteration
            // cost changes; this is benchmark data, not theory.
            => A.M_Rows <= /*+choose[512|4096]*/512/*-choose*/
                ? ladBR(in A, in b, ref x, out objective, maxIter)
                : ladFN(in A, in b, ref x, out objective, maxIter);

        /// <summary>
        /// Least absolute deviation via an explicitly chosen general-LP backend: split x = x⁺ − x⁻ and
        /// each residual rᵢ = uᵢ − vᵢ (u, v ≥ 0), then minimize Σ(uᵢ + vᵢ) subject to
        /// A(x⁺−x⁻) − u + v = b. At the optimum uᵢ + vᵢ = |rᵢ|, so the objective returned in
        /// <see cref="LPInfo.objective"/> IS the L1 residual ‖A x − b‖₁. Exact but far slower than the
        /// default <see cref="lad(in fProxyMxN, in fProxyN, ref fProxyN, out double, int)"/> overload
        /// (Frisch–Newton) — the reformulation has m equality rows over 2n+2m variables, so every
        /// backend pays for m where Frisch–Newton pays for n. Retained as independent exact
        /// cross-checks and for callers who specifically want a vertex solution.
        /// </summary>
        /// <param name="A">Design matrix, m×n (m observations, n coefficients). m ≥ n typical.</param>
        /// <param name="b">Observations, length m.</param>
        /// <param name="x">Output coefficients, length n (overwritten). May be negative.</param>
        /// <param name="objective">Output L1 residual ‖A x − b‖₁.</param>
        /// <param name="method">LP backend.</param>
        /// <param name="maxIter">Pivot/iteration budget; ≤0 picks a size-based default.</param>
        public static LPInfo lad(in fProxyMxN A, in fProxyN b, ref fProxyN x, out double objective,
                                 LPMethod method, int maxIter = 0)
        {
            int m = A.M_Rows, n = A.N_Cols;

            if (b.N != m) throw new ArgumentException("LP.lad: b.N must equal A.M_Rows");
            if (x.N != n) throw new ArgumentException("LP.lad: x.N must equal A.N_Cols");

            // Standard-form variables: [ x⁺(n) | x⁻(n) | u(m) | v(m) ], all ≥ 0.
            // m equality constraints:  A x⁺ − A x⁻ − u + v = b.
            int nv = 2 * n + 2 * m;
            var Alad = new fProxyMxN(m, nv, Allocator.Temp);            // zero-initialized
            var blad = new fProxyN(m, Allocator.Temp);
            var clad = new fProxyN(nv, Allocator.Temp);
            var senses = new NativeArray<ConstraintSense>(m, Allocator.Temp); // default = Equal (0)
            var xstd = new fProxyN(nv, Allocator.Temp);

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    fProxy a = A[i, j];
                    Alad[i, j] = a;            // x⁺
                    Alad[i, n + j] = -a;       // x⁻
                }
                Alad[i, 2 * n + i] = (fProxy)(-1);         // −u_i
                Alad[i, 2 * n + m + i] = (fProxy)1;        // +v_i
                blad[i] = b[i];
            }
            for (int i = 0; i < 2 * m; i++) clad[2 * n + i] = (fProxy)1;   // cost 1 on every u, v

            LPInfo info;
            if (method == LPMethod.Simplex)
                info = simplexCore(in Alad, in blad, in clad, in senses, ref xstd, out objective, maxIter);
            else if (method == LPMethod.RevisedSimplex)
                info = revisedSimplexCore(in Alad, in blad, in clad, in senses, ref xstd, out objective, maxIter);
            else if (method == LPMethod.DualSimplex)
                info = dualSimplexCore(in Alad, in blad, in clad, in senses, ref xstd, out objective, maxIter);
            else
                info = interiorCore(in Alad, in blad, in clad, in senses, ref xstd, out objective, maxIter);

            for (int j = 0; j < n; j++) x[j] = xstd[j] - xstd[n + j];   // x = x⁺ − x⁻

            Alad.Dispose(); blad.Dispose(); clad.Dispose(); senses.Dispose(); xstd.Dispose();
            return info;
        }

        // ============================================================================================
        // Two-phase dense tableau simplex.
        //
        // Builds standard form  min cᵀx  s.t.  M x = r, x ≥ 0  by (per row) normalizing rhs ≥ 0,
        // then adding one slack/surplus per inequality and one artificial per row that lacks a natural
        // unit-column basis (every = row, and every ≥ row / negated ≤ row). Column layout:
        //     [ structural (n) | slack (nSlack) | artificial (nArt) ]
        // Phase 1 minimizes Σ artificials to a feasible vertex (empty feasible region ⇒ Infeasible).
        // Phase 2 minimizes cᵀx from that vertex (no limiting ratio on an improving column ⇒ Unbounded).
        // Both reduced-cost rows are carried through every pivot, so phase 2 starts already priced-out
        // against phase 1's terminal basis. Bland's rule (entering = smallest index with negative
        // reduced cost; leaving = min-ratio, ties → smallest basic-variable index) precludes cycling.
        // ============================================================================================
        static LPInfo simplexCore(in fProxyMxN A, in fProxyN b, in fProxyN c,
                                  in NativeArray<ConstraintSense> senses,
                                  ref fProxyN xStruct, out double objective, int maxIter)
        {
            int m = A.M_Rows, n = A.N_Cols;

            // --- pass 1: normalize each row to rhs ≥ 0, decide slack sign & who needs an artificial ---
            var rowNeg = new NativeArray<bool>(m, Allocator.Temp);
            var slackSign = new NativeArray<int>(m, Allocator.Temp);   // +1, −1, or 0 (equality: no slack)
            var needsArt = new NativeArray<bool>(m, Allocator.Temp);
            int nSlack = 0, nArt = 0;
            double bScale = 0;

            for (int i = 0; i < m; i++)
            {
                bool neg = b[i] < (fProxy)0;
                rowNeg[i] = neg;
                bScale = math.max(bScale, math.abs((double)b[i]));

                // Sense as seen AFTER a potential row negation (negation flips ≤ ↔ ≥, leaves = alone).
                ConstraintSense s = senses[i];
                int dir = (int)s;                       // −1 ≤, 0 =, +1 ≥
                if (neg) dir = -dir;

                if (dir == 0) { slackSign[i] = 0; needsArt[i] = true; }        // equality
                else { nSlack++; if (dir < 0) { slackSign[i] = 1; needsArt[i] = false; }   // ≤ : +slack, natural basis
                                 else { slackSign[i] = -1; needsArt[i] = true; } }          // ≥ : −surplus, needs artificial
                if (needsArt[i]) nArt++;
            }

            int nCols = n + nSlack + nArt;

            // --- build tableau T (m × nCols), rhs, both reduced-cost rows, basis, artificial mask -----
            var T = new fProxyMxN(m, nCols, Allocator.Temp);      // zero-initialized
            var rhs = new fProxyN(m, Allocator.Temp);
            var cost1 = new fProxyN(nCols, Allocator.Temp);       // phase-1 reduced costs (Σ artificials)
            var cost2 = new fProxyN(nCols, Allocator.Temp);       // phase-2 reduced costs (cᵀx)
            var basis = new NativeArray<int>(m, Allocator.Temp);
            var isArt = new NativeArray<bool>(nCols, Allocator.Temp);

            int slackCol = n, artCol = n + nSlack;
            for (int i = 0; i < m; i++)
            {
                fProxy sgn = rowNeg[i] ? (fProxy)(-1) : (fProxy)1;
                for (int j = 0; j < n; j++) T[i, j] = sgn * A[i, j];
                rhs[i] = sgn * b[i];

                int myBasis;
                if (slackSign[i] != 0)
                {
                    T[i, slackCol] = (fProxy)slackSign[i];
                    myBasis = slackSign[i] > 0 ? slackCol : -1;   // −surplus is not a basis column
                    slackCol++;
                }
                else myBasis = -1;

                if (needsArt[i])
                {
                    T[i, artCol] = (fProxy)1;
                    isArt[artCol] = true;
                    myBasis = artCol;      // artificial is the basic variable for this row
                    artCol++;
                }
                basis[i] = myBasis;
            }

            // Original costs: phase 1 = 1 on artificials; phase 2 = c on structural, 0 elsewhere.
            // Price both rows out against the initial basis (reduced cost = origCost − c_B·T).
            for (int j = 0; j < n; j++) cost2[j] = c[j];
            for (int j = 0; j < nCols; j++) if (isArt[j]) cost1[j] = (fProxy)1;
            for (int i = 0; i < m; i++)
            {
                int bcol = basis[i];
                if (bcol < 0) continue;
                fProxy c1b = isArt[bcol] ? (fProxy)1 : (fProxy)0;
                fProxy c2b = bcol < n ? c[bcol] : (fProxy)0;
                if (c1b != (fProxy)0) for (int j = 0; j < nCols; j++) cost1[j] -= c1b * T[i, j];
                if (c2b != (fProxy)0) for (int j = 0; j < nCols; j++) cost2[j] -= c2b * T[i, j];
            }

            // Simplex needs a tolerance LOOSER than machine precision: near-zero reduced costs and
            // near-zero pivot elements are noise, not signal. Consts.fProxyZeroThreshold is 1e-6
            // (float) but 1e-14 (double) -- the latter is machine-tight and makes the double solve
            // over-sensitive to roundoff on larger, degenerate problems (near-zero pivots amplify,
            // phase-1 leaves a spurious residual). Floor both tolerances so double stays sane while
            // float is unchanged (its 1e-6 / sqrt-eps values already dominate the floors).
            fProxy pivTol = math.max(Consts.fProxyZeroThreshold, (fProxy)1e-9);
            double feasTol = math.max(math.sqrt((double)Consts.fProxyEpsilon), 1e-7) * (1.0 + bScale);
            int budget = maxIter > 0 ? maxIter : 50 * (m + nCols) + 200;

            LPStatus status = LPStatus.Optimal;
            int iters = 0;

            // ---- phase 1: drive artificials to zero ----
            if (nArt > 0)
            {
                while (true)
                {
                    if (iters >= budget) { status = LPStatus.MaxIterations; break; }
                    int enter = -1;
                    for (int j = 0; j < nCols; j++) if (cost1[j] < -pivTol) { enter = j; break; }
                    if (enter < 0) break;                                   // phase-1 optimal
                    int leave = RatioTest(T, rhs, basis, m, enter, pivTol);
                    if (leave < 0) break;                                   // Σ artificials is bounded below by 0
                    Pivot(T, rhs, cost1, cost2, basis, m, nCols, leave, enter);
                    iters++;
                }

                if (status == LPStatus.Optimal)
                {
                    double infeas = 0;
                    for (int i = 0; i < m; i++) if (basis[i] >= 0 && isArt[basis[i]]) infeas += (double)rhs[i];
                    if (infeas > feasTol) status = LPStatus.Infeasible;
                    else
                    {
                        // Pivot any still-basic artificial out of the basis onto a real column so
                        // phase 2 never has to reason about it (redundant rows keep a zero-valued
                        // artificial basic -- harmless, it is excluded from entering below).
                        for (int i = 0; i < m; i++)
                        {
                            if (basis[i] < 0 || !isArt[basis[i]]) continue;
                            int piv = -1;
                            for (int j = 0; j < nCols; j++)
                                if (!isArt[j] && math.abs(T[i, j]) > pivTol) { piv = j; break; }
                            if (piv >= 0) Pivot(T, rhs, cost1, cost2, basis, m, nCols, i, piv);
                        }
                    }
                }
            }

            // ---- phase 2: minimize cᵀx (artificials forbidden to re-enter) ----
            if (status == LPStatus.Optimal)
            {
                while (true)
                {
                    if (iters >= budget) { status = LPStatus.MaxIterations; break; }
                    int enter = -1;
                    for (int j = 0; j < nCols; j++) if (!isArt[j] && cost2[j] < -pivTol) { enter = j; break; }
                    if (enter < 0) break;                                   // optimal
                    int leave = RatioTest(T, rhs, basis, m, enter, pivTol);
                    if (leave < 0) { status = LPStatus.Unbounded; break; }
                    Pivot(T, rhs, cost1, cost2, basis, m, nCols, leave, enter);
                    iters++;
                }
            }

            // ---- extract structural solution & objective ----
            for (int j = 0; j < n; j++) xStruct[j] = (fProxy)0;
            for (int i = 0; i < m; i++)
            {
                int col = basis[i];
                if (col >= 0 && col < n) xStruct[col] = rhs[i] > (fProxy)0 ? rhs[i] : (fProxy)0;
            }
            double obj = 0;
            for (int j = 0; j < n; j++) obj += (double)c[j] * (double)xStruct[j];
            objective = obj;

            rowNeg.Dispose(); slackSign.Dispose(); needsArt.Dispose();
            T.Dispose(); rhs.Dispose(); cost1.Dispose(); cost2.Dispose(); basis.Dispose(); isArt.Dispose();

            return new LPInfo { status = status, iterations = iters, objective = obj };
        }

        // Bland ratio test: leaving row = min rhs/T[·,enter] over positive entries, ties broken by the
        // smallest basic-variable index. Returns −1 when the column has no positive entry (unbounded).
        static int RatioTest(fProxyMxN T, fProxyN rhs, NativeArray<int> basis, int m, int enter, fProxy pivTol)
        {
            int leave = -1;
            fProxy best = (fProxy)0;
            for (int i = 0; i < m; i++)
            {
                fProxy a = T[i, enter];
                if (a > pivTol)
                {
                    fProxy ratio = rhs[i] / a;
                    if (leave < 0 || ratio < best - pivTol) { best = ratio; leave = i; }
                    else if (ratio <= best + pivTol && basis[i] < basis[leave]) { leave = i; }
                }
            }
            return leave;
        }

        // Gauss-Jordan pivot on (prow, pcol): normalize the pivot row, eliminate the pivot column from
        // every other constraint row AND both reduced-cost rows, then record the new basic variable.
        static void Pivot(fProxyMxN T, fProxyN rhs, fProxyN cost1, fProxyN cost2,
                          NativeArray<int> basis, int m, int nCols, int prow, int pcol)
        {
            fProxy inv = (fProxy)1 / T[prow, pcol];
            for (int j = 0; j < nCols; j++) T[prow, j] *= inv;
            rhs[prow] *= inv;

            for (int i = 0; i < m; i++)
            {
                if (i == prow) continue;
                fProxy f = T[i, pcol];
                if (f == (fProxy)0) continue;
                for (int j = 0; j < nCols; j++) T[i, j] -= f * T[prow, j];
                rhs[i] -= f * rhs[prow];
            }

            fProxy f1 = cost1[pcol];
            if (f1 != (fProxy)0) for (int j = 0; j < nCols; j++) cost1[j] -= f1 * T[prow, j];
            fProxy f2 = cost2[pcol];
            if (f2 != (fProxy)0) for (int j = 0; j < nCols; j++) cost2[j] -= f2 * T[prow, j];

            basis[prow] = pcol;
        }
    }
}
