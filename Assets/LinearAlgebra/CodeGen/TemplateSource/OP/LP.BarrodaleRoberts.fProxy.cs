using System;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

using LinearAlgebra.Internal;

namespace LinearAlgebra
{
    // ================================================================================================
    // Barrodale-Roberts primal simplex for L1 (least-absolute-deviation) / quantile regression.
    //
    // Implemented from the algorithm DESCRIPTIONS in Barrodale & Roberts 1973 ("An improved algorithm
    // for discrete l1 linear approximation", SIAM J. Numer. Anal. 10(5) 839-848) and Koenker & d'Orey
    // 1987 ("Computing Regression Quantiles", Appl. Statist. 36) -- clean-room: no reference source
    // code was read. See this folder's DEVLOG.md for the derivation and validation history.
    //
    // Problem:  minimize sum_i rho_tau(b_i - A_i.x)  over a FREE x, rho_tau(r) = tau*r (r>=0) or
    // (tau-1)*r (r<0). Convex piecewise-linear in x. A vertex interpolates n of the m rows exactly
    // (index set S, |S| = n, B = A[S,:] invertible); this file maintains only B's inverse (via a
    // fresh small LU factorization each iteration -- n is a coefficient count, always small) rather
    // than a materialized m x (something) tableau, and reconstructs whatever a pivot needs from it.
    //
    // Stage 1 (colRow[j] < 0 for some j): places one structural column per row, growing S from empty
    // to n. Stage 2: S is full; each pivot replaces one interpolated row with another. Both stages run
    // through the SAME loop -- an unplaced column's optimality "box" is unbounded (any nonzero
    // gradient is a violator), a placed column's box is [-tau, 1-tau]; whichever column violates its
    // box the most enters next, so stage 1 finishes naturally before stage 2 without a separate path.
    //
    // Entering column's ROW is chosen by an exact line search along the pivot direction: candidate
    // breakpoints (rows whose residual would cross zero) are sorted once and walked once, accumulating
    // the direction's slope (which only ever increases, by convexity) until it reaches zero -- the
    // walk can pass through many candidates in a single pivot (the "weighted-median long step" that is
    // this algorithm's namesake speedup). Rank-deficient columns (zero leverage against every
    // remaining row) are detected and pinned at 0 rather than forced into a singular basis.
    //
    // Job-safe: all scratch is Allocator.Temp, disposed on every return path.
    // ================================================================================================
    public static partial class LP
    {
        /// <summary>
        /// Least absolute deviation (L1 regression): minimize ||A x - b||_1 over a FREE x, via the
        /// Barrodale-Roberts specialized primal simplex (tau = 0.5). Reaches an EXACT VERTEX (n of the
        /// m residuals are exactly zero -- see <see cref="LPInfo"/>). See
        /// <see cref="lad(in fProxyMxN, in fProxyN, ref fProxyN, out double, int)"/> for the
        /// size-routed hybrid default, and
        /// <see cref="ladFN(in fProxyMxN, in fProxyN, ref fProxyN, out double, int)"/> for the
        /// interior-point alternative.
        /// </summary>
        /// <param name="A">Design matrix, m x n (m observations, n coefficients).</param>
        /// <param name="b">Observations, length m.</param>
        /// <param name="x">Output coefficients, length n (overwritten). May be negative.</param>
        /// <param name="objective">Output L1 residual ||A x - b||_1, recomputed from the returned x.</param>
        /// <param name="maxIter">Pivot budget; &lt;=0 picks 10n+100 (stage 1 alone needs n pivots).</param>
        public static LPInfo ladBR(in fProxyMxN A, in fProxyN b, ref fProxyN x, out double objective,
                                   int maxIter = 0)
        {
            if (b.N != A.M_Rows) throw new ArgumentException("LP.ladBR: b.N must equal A.M_Rows");
            if (x.N != A.N_Cols) throw new ArgumentException("LP.ladBR: x.N must equal A.N_Cols");

            return ladBarrodaleRobertsCore(in A, in b, 0.5, ref x, out objective, maxIter);
        }

        /// <summary>
        /// Quantile regression at level <paramref name="tau"/>: minimize the tau check-loss
        /// sum rho_tau(b - A x) over a FREE x, via the Barrodale-Roberts primal simplex generalized
        /// per Koenker &amp; d'Orey 1987 (asymmetric [tau-1, tau] reduced-cost bounds in place of the
        /// classical LAD's symmetric [-0.5, 0.5]). tau = 0.5 reproduces <see cref="ladBR(in fProxyMxN, in fProxyN, ref fProxyN, out double, int)"/>.
        /// </summary>
        /// <param name="A">Design matrix, m x n (m observations, n coefficients).</param>
        /// <param name="b">Observations, length m.</param>
        /// <param name="tau">Quantile level, strictly between 0 and 1.</param>
        /// <param name="x">Output coefficients, length n (overwritten). May be negative.</param>
        /// <param name="objective">Output L1 residual ||A x - b||_1, recomputed from the returned x
        /// (NOT the tau check-loss -- an honest, tau-independent diagnostic, matching every other LAD
        /// entry point in this file).</param>
        /// <param name="maxIter">Pivot budget; &lt;=0 picks 10n+100.</param>
        public static LPInfo ladBR(in fProxyMxN A, in fProxyN b, double tau, ref fProxyN x,
                                   out double objective, int maxIter = 0)
        {
            if (b.N != A.M_Rows) throw new ArgumentException("LP.ladBR: b.N must equal A.M_Rows");
            if (x.N != A.N_Cols) throw new ArgumentException("LP.ladBR: x.N must equal A.N_Cols");
            if (tau <= 0.0 || tau >= 1.0) throw new ArgumentException("LP.ladBR: tau must be strictly between 0 and 1");

            return ladBarrodaleRobertsCore(in A, in b, tau, ref x, out objective, maxIter);
        }

        /// <summary>
        /// Core Barrodale-Roberts solve. Assumes dimensions/tau already validated by the public
        /// overloads above; internal (not private) so the InternalsVisibleTo test grant can call it
        /// directly. <paramref name="maxIter"/> &lt;=0 picks 10n+100.
        ///
        /// Status contract: <see cref="LPStatus.Optimal"/> on reaching the box-optimality certificate
        /// OR (rare, near-degenerate data only) on accepting the best iterate found after persistent
        /// cycling was detected and an anti-cycling escalation (Bland's rule, then a widened residual-
        /// zero threshold) failed to resolve it within one more history window -- the combinatorial
        /// vertex identity can be ambiguous at a given dtype's precision even though the solution
        /// itself is stable; <see cref="LPStatus.MaxIterations"/> if the pivot budget is exhausted
        /// first; <see cref="LPStatus.Unbounded"/> only on a genuinely degenerate pivot direction with
        /// no stopping row (should not occur for a well-posed problem). <paramref name="x"/> is a
        /// finite, well-defined iterate on EVERY exit path, including
        /// <see cref="LPStatus.MaxIterations"/>; unresolved coefficients (stage 1 incomplete, or
        /// detected rank-deficient) keep 0. <paramref name="objective"/> is always the honest
        /// ||A x - b||_1 at the returned x, independent of <paramref name="tau"/>.
        /// </summary>
        internal static LPInfo ladBarrodaleRobertsCore(in fProxyMxN A, in fProxyN b, double tau,
                                                       ref fProxyN x, out double objective, int maxIter)
        {
            int m = A.M_Rows, n = A.N_Cols;
            if (maxIter <= 0) maxIter = 10 * n + 100;

            fProxy tauF = (fProxy)tau;
            fProxy oneMinusTau = (fProxy)1 - tauF;
            // tol: the LOOSE optimality margin (box-violation / stopping-slope slack). pivTolFloor:
            // TIGHT "is this raw value zero" floor (nonzero-rate / nonzero-pivot detection).
            // zeroResidTol starts at pivTolFloor and is kept SEPARATE from tol for classifying a
            // residual's sign -- using tol there was measured to zero out legitimate small-but-real
            // residuals (in float, tol's sqrt-eps scale is ~300x looser than a residual step actually
            // needs), corrupting the price-out sum; see this file's DEVLOG entry.
            fProxy tol = (fProxy)math.max(math.sqrt((double)Consts.fProxyEpsilon), 1e-7);
            fProxy pivTolFloor = math.max(Consts.fProxyZeroThreshold, (fProxy)1e-9);
            fProxy zeroResidTol = pivTolFloor;

            var colRow = new NativeArray<int>(n, Allocator.Temp);      // coefficient -> row, or -1
            for (int j = 0; j < n; j++) colRow[j] = -1;
            var rowUsed = new NativeArray<bool>(m, Allocator.Temp);    // row currently interpolated?
            var redundant = new NativeArray<bool>(n, Allocator.Temp);  // rank-deficient, pinned at 0
            var placedCols = new NativeArray<int>(n, Allocator.Temp);  // gathered each iteration
            var srows = new NativeArray<int>(n, Allocator.Temp);       // srows[a] = colRow[placedCols[a]]

            var g = new fProxyN(n, Allocator.Temp);
            var val = new fProxyN(n, Allocator.Temp);
            var adjustedG = new fProxyN(n, Allocator.Temp);
            var R = new fProxyN(m, Allocator.Temp);
            var tcol = new fProxyN(m, Allocator.Temp);
            var candBreak = new fProxyN(m, Allocator.Temp);
            var candRow = new NativeArray<int>(m, Allocator.Temp);

            // Anti-cycling: a fixed-size ring of the last HIST (col, selfRow, enterRow) pivots.
            // Committing a pivot identical to one already in the ring means a short cycle is being
            // revisited (confirmed to occur at float precision on genuinely near-degenerate data --
            // several rows simultaneously within roundoff of the fitted line -- see DEVLOG); switch to
            // Bland's rule (deterministic smallest-index selection, both entering column and tied
            // leaving row) from that point on. bestX/bestChk track the lowest check-loss iterate seen;
            // if Bland's rule alone does not resolve the cycle within one more history window, the
            // combinatorial vertex identity is ambiguous at this precision but the SOLUTION is not (x
            // stays essentially fixed across such cycles) -- accept the best iterate rather than grind
            // to MaxIterations on a certificate this precision cannot stably certify.
            const int HIST = 16;
            var histCol = new NativeArray<int>(HIST, Allocator.Temp);
            var histSelf = new NativeArray<int>(HIST, Allocator.Temp);
            var histEnter = new NativeArray<int>(HIST, Allocator.Temp);
            for (int h = 0; h < HIST; h++) { histCol[h] = -1; histSelf[h] = -1; histEnter[h] = -1; }
            int histPos = 0;
            bool useBland = false;
            int cyclingSince = -1;
            var bestX = new fProxyN(n, Allocator.Temp);
            double bestChk = 0;
            bool haveBest = false;
            bool acceptedBest = false;

            for (int j = 0; j < n; j++) x[j] = (fProxy)0;

            LPStatus status = LPStatus.MaxIterations;
            int iters = 0;

            while (iters < maxIter)
            {
                int k = 0;
                for (int j = 0; j < n; j++)
                    if (colRow[j] >= 0) { placedCols[k] = j; srows[k] = colRow[j]; k++; }

                fProxyMxN Bmat = default;
                Pivot piv = default;
                bool haveFactor = false;

                if (k > 0)
                {
                    Bmat = new fProxyMxN(k, k, Allocator.Temp);
                    for (int a = 0; a < k; a++)
                        for (int c = 0; c < k; c++)
                            Bmat[a, c] = A[srows[a], placedCols[c]];

                    piv = new Pivot(k, Allocator.Temp);
                    var luInfo = LU.decompInPlace(ref Bmat, ref piv);
                    if (luInfo.status == DirectSolveStatus.Singular)
                    {
                        // Should not occur: redundant-column detection below keeps every placed
                        // column linearly independent of the others. Defensive fallback only.
                        status = LPStatus.Unbounded;
                        Bmat.Dispose(); piv.Dispose();
                        break;
                    }
                    haveFactor = true;

                    var rhs = new fProxyN(k, Allocator.Temp);
                    for (int a = 0; a < k; a++) rhs[a] = b[srows[a]];
                    LU.decompSolve(ref Bmat, in piv, ref rhs);
                    for (int a = 0; a < k; a++) x[placedCols[a]] = rhs[a];
                    rhs.Dispose();
                }
                for (int j = 0; j < n; j++) if (colRow[j] < 0) x[j] = (fProxy)0;

                for (int j = 0; j < n; j++) g[j] = (fProxy)0;
                double chk = 0;
                for (int i = 0; i < m; i++)
                {
                    if (rowUsed[i]) continue;
                    fProxy dot = (fProxy)0;
                    for (int j = 0; j < n; j++) dot += A[i, j] * x[j];
                    fProxy r = b[i] - dot;
                    R[i] = r;
                    chk += r >= (fProxy)0 ? tau * (double)r : (tau - 1.0) * (double)r;
                    fProxy sg = r > zeroResidTol ? tauF : (r < -zeroResidTol ? tauF - (fProxy)1 : (fProxy)0);
                    if (sg != (fProxy)0)
                        for (int j = 0; j < n; j++) g[j] += A[i, j] * sg;
                }

                if (!haveBest || chk < bestChk)
                {
                    haveBest = true;
                    bestChk = chk;
                    for (int j = 0; j < n; j++) bestX[j] = x[j];
                }

                for (int j = 0; j < n; j++) val[j] = (fProxy)0;
                if (k > 0)
                {
                    var gp = new fProxyN(k, Allocator.Temp);
                    for (int a = 0; a < k; a++) gp[a] = g[placedCols[a]];
                    LU.decompSolveTransA(ref Bmat, in piv, ref gp);
                    for (int a = 0; a < k; a++) val[placedCols[a]] = gp[a];
                    gp.Dispose();
                }

                for (int j = 0; j < n; j++)
                {
                    if (colRow[j] >= 0) { adjustedG[j] = (fProxy)0; continue; }
                    fProxy adj = g[j];
                    for (int a = 0; a < k; a++) adj -= A[srows[a], j] * val[placedCols[a]];
                    adjustedG[j] = adj;
                }

                int bestCol = -1;
                fProxy bestViol = tol;
                int bestDir = 0;
                for (int j = 0; j < n; j++)
                {
                    if (redundant[j]) continue;
                    if (colRow[j] < 0)
                    {
                        fProxy v = math.abs(adjustedG[j]);
                        if (v > bestViol)
                        {
                            bestViol = v; bestCol = j; bestDir = adjustedG[j] > (fProxy)0 ? -1 : 1;
                            if (useBland) break;   // ascending j -> first hit is smallest index
                        }
                    }
                    else
                    {
                        fProxy lo = -tauF, hi = oneMinusTau;
                        fProxy vv = val[j];
                        if (vv < lo - tol)
                        {
                            fProxy v = lo - vv;
                            if (v > bestViol) { bestViol = v; bestCol = j; bestDir = 1; if (useBland) break; }
                        }
                        else if (vv > hi + tol)
                        {
                            fProxy v = vv - hi;
                            if (v > bestViol) { bestViol = v; bestCol = j; bestDir = -1; if (useBland) break; }
                        }
                    }
                }

                if (bestCol < 0)
                {
                    int redundantCount = 0;
                    for (int j = 0; j < n; j++) if (redundant[j]) redundantCount++;
                    if (k == n - redundantCount)
                    {
                        status = LPStatus.Optimal;
                        if (haveFactor) { Bmat.Dispose(); piv.Dispose(); }
                        break;
                    }
                    // S incomplete but nothing strictly violates -- force-place the least-marginal
                    // unplaced column so stage 1 can still complete.
                    fProxy best = (fProxy)(-1);
                    int bc = -1;
                    for (int j = 0; j < n; j++)
                    {
                        if (colRow[j] >= 0 || redundant[j]) continue;
                        fProxy v = math.abs(adjustedG[j]);
                        if (v > best) { best = v; bc = j; }
                    }
                    bestCol = bc;
                    bestDir = adjustedG[bc] <= (fProxy)0 ? 1 : -1;
                }

                int col = bestCol;
                int dir = bestDir;
                bool placingNew = colRow[col] < 0;
                int selfRow = placingNew ? -1 : colRow[col];

                fProxy initSlope;
                if (placingNew)
                {
                    if (k > 0)
                    {
                        var scolVals = new fProxyN(k, Allocator.Temp);
                        for (int a = 0; a < k; a++) scolVals[a] = A[srows[a], col];
                        LU.decompSolve(ref Bmat, in piv, ref scolVals);
                        for (int i = 0; i < m; i++)
                        {
                            if (rowUsed[i]) continue;
                            fProxy proj = (fProxy)0;
                            for (int a = 0; a < k; a++) proj += A[i, placedCols[a]] * scolVals[a];
                            tcol[i] = A[i, col] - proj;
                        }
                        scolVals.Dispose();
                    }
                    else
                    {
                        for (int i = 0; i < m; i++) if (!rowUsed[i]) tcol[i] = A[i, col];
                    }
                    initSlope = dir * adjustedG[col];
                }
                else
                {
                    int slot = -1;
                    for (int a = 0; a < k; a++) if (placedCols[a] == col) { slot = a; break; }

                    var e = new fProxyN(k, Allocator.Temp);
                    for (int a = 0; a < k; a++) e[a] = a == slot ? (fProxy)1 : (fProxy)0;
                    LU.decompSolve(ref Bmat, in piv, ref e);
                    for (int i = 0; i < m; i++)
                    {
                        if (rowUsed[i]) continue;
                        fProxy tv = (fProxy)0;
                        for (int a = 0; a < k; a++) tv += A[i, placedCols[a]] * e[a];
                        tcol[i] = tv;
                    }
                    e.Dispose();
                    initSlope = dir == 1 ? (val[col] + tauF) : (-(val[col]) + oneMinusTau);
                }

                int nCand = 0;
                for (int i = 0; i < m; i++)
                {
                    if (rowUsed[i]) continue;
                    fProxy rate = dir * tcol[i];
                    if (math.abs(rate) <= pivTolFloor) continue;
                    fProxy ti = -R[i] / rate;
                    // Tight floor (not the loose optimality-margin tol) admitting a "just barely
                    // negative" breakpoint as an immediate (t=0) candidate: a row already slightly
                    // past the crossing (by more than roundoff) is NOT a legitimate candidate, and
                    // admitting it under too generous a slack was observed to seed exactly the
                    // near-degenerate reversals this file's anti-cycling machinery has to correct for.
                    if (ti < -pivTolFloor) continue;
                    candBreak[nCand] = math.max(ti, (fProxy)0);
                    candRow[nCand] = i;
                    nCand++;
                }

                int enterRow = -1;
                if (nCand == 0)
                {
                    if (placingNew)
                    {
                        int forceRow = -1;
                        fProxy forceMag = (fProxy)0;
                        for (int i = 0; i < m; i++)
                        {
                            if (rowUsed[i]) continue;
                            fProxy mag = math.abs(tcol[i]);
                            if (mag > forceMag) { forceMag = mag; forceRow = i; }
                        }
                        if (forceRow >= 0 && forceMag > pivTolFloor)
                        {
                            enterRow = forceRow;
                        }
                        else
                        {
                            redundant[col] = true;
                            if (haveFactor) { Bmat.Dispose(); piv.Dispose(); }
                            iters++;
                            continue;
                        }
                    }
                    else
                    {
                        status = LPStatus.Unbounded;
                        if (haveFactor) { Bmat.Dispose(); piv.Dispose(); }
                        break;
                    }
                }
                else
                {
                    unsafe
                    {
                        UnsafeOP.sortByKeyAscending((fProxy*)candBreak.Data.Ptr,
                                                    (int*)candRow.GetUnsafePtr(), nCand);
                    }

                    fProxy cum = initSlope;
                    fProxy stopWinTheta = (fProxy)0;
                    for (int idx = 0; idx < nCand; idx++)
                    {
                        int row = candRow[idx];
                        fProxy rate = dir * tcol[row];
                        cum += math.abs(rate);
                        if (cum >= -tol) { enterRow = row; stopWinTheta = candBreak[idx]; break; }
                    }
                    if (enterRow < 0)
                    {
                        status = LPStatus.Unbounded;
                        if (haveFactor) { Bmat.Dispose(); piv.Dispose(); }
                        break;
                    }

                    // Harris-style tie-break (mirrors LP.RevisedSimplex's ratio-test pattern): among
                    // all candidates numerically tied with the winning breakpoint (TWO-SIDED window --
                    // an upper-bound-only check trivially includes every candidate already swept by
                    // the walk above and silently picks the largest-magnitude one seen so far,
                    // regardless of whether it is actually tied), pick the one with LARGEST |rate| for
                    // a better-conditioned pivot. Under Bland's-rule fallback, smallest ROW INDEX
                    // instead (deterministic, anti-cycling).
                    fProxy tieTol = tol * ((fProxy)1 + math.abs(stopWinTheta));
                    if (useBland)
                    {
                        int bestRow = enterRow;
                        for (int idx = 0; idx < nCand; idx++)
                        {
                            int row = candRow[idx];
                            if (math.abs(candBreak[idx] - stopWinTheta) <= tieTol && row < bestRow) bestRow = row;
                        }
                        enterRow = bestRow;
                    }
                    else
                    {
                        fProxy bestMag = (fProxy)0;
                        int bestRow = enterRow;
                        for (int idx = 0; idx < nCand; idx++)
                        {
                            int row = candRow[idx];
                            if (math.abs(candBreak[idx] - stopWinTheta) <= tieTol)
                            {
                                fProxy mag = math.abs(dir * tcol[row]);
                                if (mag > bestMag) { bestMag = mag; bestRow = row; }
                            }
                        }
                        enterRow = bestRow;
                    }
                }

                bool repeated = false;
                for (int h = 0; h < HIST; h++)
                    if (histCol[h] == col && histSelf[h] == selfRow && histEnter[h] == enterRow) { repeated = true; break; }
                if (repeated)
                {
                    if (!useBland) { useBland = true; cyclingSince = iters; }
                    else if (zeroResidTol < tol) zeroResidTol = math.min(zeroResidTol * (fProxy)10, tol);
                }
                histCol[histPos] = col; histSelf[histPos] = selfRow; histEnter[histPos] = enterRow;
                histPos = (histPos + 1) % HIST;

                if (useBland && cyclingSince >= 0 && iters - cyclingSince > 6 * HIST)
                {
                    for (int j = 0; j < n; j++) x[j] = bestX[j];
                    status = LPStatus.Optimal;
                    acceptedBest = true;
                    if (haveFactor) { Bmat.Dispose(); piv.Dispose(); }
                    break;
                }

                if (!placingNew) rowUsed[selfRow] = false;
                rowUsed[enterRow] = true;
                colRow[col] = enterRow;

                if (haveFactor) { Bmat.Dispose(); piv.Dispose(); }
                iters++;
            }

            // Final extraction -- guarantees x matches colRow's LATEST state even when the loop
            // above exited via the maxIter budget right after applying its last pivot. Skipped when
            // the anti-cycling fallback already wrote the accepted best-seen iterate into x directly
            // (colRow's final state there is just wherever the still-cycling walk last landed, not
            // the accepted answer).
            if (!acceptedBest)
            {
                int k = 0;
                for (int j = 0; j < n; j++)
                    if (colRow[j] >= 0) { placedCols[k] = j; srows[k] = colRow[j]; k++; }

                if (k > 0)
                {
                    var Bmat2 = new fProxyMxN(k, k, Allocator.Temp);
                    for (int a = 0; a < k; a++)
                        for (int c = 0; c < k; c++)
                            Bmat2[a, c] = A[srows[a], placedCols[c]];

                    var piv2 = new Pivot(k, Allocator.Temp);
                    var luInfo2 = LU.decompInPlace(ref Bmat2, ref piv2);
                    if (luInfo2.status == DirectSolveStatus.Success)
                    {
                        var rhs2 = new fProxyN(k, Allocator.Temp);
                        for (int a = 0; a < k; a++) rhs2[a] = b[srows[a]];
                        LU.decompSolve(ref Bmat2, in piv2, ref rhs2);
                        for (int a = 0; a < k; a++) x[placedCols[a]] = rhs2[a];
                        rhs2.Dispose();
                    }
                    Bmat2.Dispose(); piv2.Dispose();
                }
                for (int j = 0; j < n; j++) if (colRow[j] < 0) x[j] = (fProxy)0;
            }

            double obj = 0;
            for (int i = 0; i < m; i++)
            {
                fProxy dot = (fProxy)0;
                for (int j = 0; j < n; j++) dot += A[i, j] * x[j];
                obj += math.abs((double)(b[i] - dot));
            }
            objective = obj;

            colRow.Dispose(); rowUsed.Dispose(); redundant.Dispose(); placedCols.Dispose(); srows.Dispose();
            g.Dispose(); val.Dispose(); adjustedG.Dispose(); R.Dispose(); tcol.Dispose();
            candBreak.Dispose(); candRow.Dispose();
            histCol.Dispose(); histSelf.Dispose(); histEnter.Dispose(); bestX.Dispose();

            return new LPInfo { status = status, iterations = iters, objective = obj };
        }
    }
}
