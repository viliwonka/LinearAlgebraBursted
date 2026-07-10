using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // Bound-change undo stack for MIP's search (MIP.fProxy.cs's SearchCore): push a tightening, undo
    // back to a marker. Used for the current plunge's dive step, throwaway strong-branch trials
    // (MIP.Pseudocost.fProxy.cs), and stage 4's PropagateFixpoint below, which pushes every tightening
    // it finds through the same Push/UndoToMarker pair. Jumping to a different open node (a best-bound
    // queue pop) uses ApplyNodeBounds below instead, since the target is not necessarily an ancestor
    // reachable by undoing this stack.
    //
    // Bounds are recorded in ORIGINAL x-space; Push/UndoToMarker translate to the shifted y-space row
    // rhs internally (see MIP.fProxy.cs's header for the shift).
    //
    // UB-row activation: an integer variable's UB row starts INERT (coefficient 0, rhs 0) when its root
    // xu is infinite, instead of using a 1e30 sentinel rhs directly -- a row's rhs feeds into
    // DualSimplexCore's dataScale/artificialBound scan (unlike a variable bound, which is excluded), so
    // a 1e30 rhs inflates artificialBound to ~1e32 and can return a false Infeasible (reproduced on the
    // Gomory/Wolsey instance: sentinel=1e30 -> false Infeasible after 63 pivots; sentinel<=1e10 ->
    // correct). It is also a correctness risk, not just numerics: a finite sentinel can silently bound a
    // genuinely unbounded direction. PushBoundChange activates the row (coefficient -> 1) the first time
    // a branch tightens it (oldBound still >= 1e29); UndoToMarker deactivates it back (coefficient and
    // rhs -> 0) when undoing past that point.
    internal struct fProxyBoundChange
    {
        /// <summary>Index into the ORIGINAL (length-n) variable space.</summary>
        public int varIndex;

        /// <summary>True: tightened the upper bound. False: tightened the lower bound.</summary>
        public bool isUpper;

        /// <summary>Bound value before this change.</summary>
        public fProxy oldBound;

        /// <summary>Bound value after this change.</summary>
        public fProxy newBound;
    }

    public static partial class MIP
    {
        /// <summary>
        /// Tightens variable <paramref name="varIndex"/>'s lower or upper bound to
        /// <paramref name="newBound"/>, records the change on <paramref name="stack"/>, and writes
        /// through to <paramref name="curLB"/>/<paramref name="curUB"/> and the matching bound row's rhs
        /// in <paramref name="bAug"/>. Activates the UB row's coefficient if it was inert -- bumps
        /// <paramref name="cache"/>.matrixVersion when (and only when) that coefficient write happens
        /// (docs/spec-lpbasis-persistence.md: rhs-only bound updates, the common case, leave it alone).
        /// Caller must pass an INTEGER <paramref name="varIndex"/> (the only kind ever branched).
        /// </summary>
        internal static void PushBoundChange(ref UnsafeList<fProxyBoundChange> stack, int varIndex, bool isUpper,
                                             fProxy oldBound, fProxy newBound,
                                             fProxyN curLB, fProxyN curUB, fProxyN xlRoot,
                                             fProxyMxN Aaug, NativeArray<int> col,
                                             fProxyN bAug, NativeArray<int> rowLB, NativeArray<int> rowUB,
                                             ref fProxyLPCache cache)
        {
            stack.Add(new fProxyBoundChange { varIndex = varIndex, isUpper = isUpper, oldBound = oldBound, newBound = newBound });

            if (isUpper)
            {
                curUB[varIndex] = newBound;
                if ((double)oldBound >= 1e29) { Aaug[rowUB[varIndex], col[varIndex]] = (fProxy)1; cache.matrixVersion++; }   // activate
                bAug[rowUB[varIndex]] = newBound - xlRoot[varIndex];
            }
            else
            {
                curLB[varIndex] = newBound;
                bAug[rowLB[varIndex]] = newBound - xlRoot[varIndex];
            }
        }

        /// <summary>
        /// Undoes every change on <paramref name="stack"/> at or after <paramref name="marker"/>, LIFO,
        /// restoring <paramref name="curLB"/>/<paramref name="curUB"/> and each bound row's rhs in
        /// <paramref name="bAug"/> -- the inverse of <see cref="PushBoundChange"/>. Deactivates a UB
        /// row's coefficient back to 0 when undoing past its first tightening -- bumps
        /// <paramref name="cache"/>.matrixVersion exactly then, mirroring <see cref="PushBoundChange"/>.
        /// Truncates <paramref name="stack"/> to <paramref name="marker"/>.
        /// </summary>
        internal static void UndoToMarker(ref UnsafeList<fProxyBoundChange> stack, int marker,
                                          fProxyN curLB, fProxyN curUB, fProxyN xlRoot,
                                          fProxyMxN Aaug, NativeArray<int> col,
                                          fProxyN bAug, NativeArray<int> rowLB, NativeArray<int> rowUB,
                                          ref fProxyLPCache cache)
        {
            for (int k = stack.Length - 1; k >= marker; k--)
            {
                fProxyBoundChange bc = stack[k];
                if (bc.isUpper)
                {
                    curUB[bc.varIndex] = bc.oldBound;
                    if ((double)bc.oldBound >= 1e29)
                    {
                        Aaug[rowUB[bc.varIndex], col[bc.varIndex]] = (fProxy)0;   // deactivate
                        bAug[rowUB[bc.varIndex]] = (fProxy)0;
                        cache.matrixVersion++;
                    }
                    else
                    {
                        bAug[rowUB[bc.varIndex]] = bc.oldBound - xlRoot[bc.varIndex];
                    }
                }
                else
                {
                    curLB[bc.varIndex] = bc.oldBound;
                    bAug[rowLB[bc.varIndex]] = bc.oldBound - xlRoot[bc.varIndex];
                }
            }
            stack.Length = marker;
        }

        /// <summary>
        /// Overwrites every INTEGER variable's live bound state (<paramref name="curLB"/>/
        /// <paramref name="curUB"/> and each bound row's rhs/coefficient in <paramref name="bAug"/>/
        /// <paramref name="Aaug"/>) from a queued node's full snapshot <paramref name="L"/>/
        /// <paramref name="U"/> -- the wholesale counterpart to <see cref="PushBoundChange"/>/
        /// <see cref="UndoToMarker"/>'s incremental delta, used by the stage-3 best-bound search when
        /// jumping to a queued node that is not an ancestor of the current plunge (so there is no
        /// shared marker to replay from). Continuous variables are untouched -- their bound rows, if
        /// any, never change after the root build. Same UB-row inert/active convention as
        /// <see cref="PushBoundChange"/>: inert (coefficient 0, rhs 0) exactly when
        /// <paramref name="U"/>[j] is still the +infinity sentinel.
        ///
        /// Bumps <paramref name="cache"/>.matrixVersion ONCE per call (not once per rewritten
        /// coefficient): every integer bound row is rewritten wholesale here regardless of whether its
        /// UB-row activation state actually changed, so a queue jump always forces the next
        /// <c>LP.solve</c> to rebuild cold -- the expected regime (MIP.fProxy.cs's SearchCore header).
        /// </summary>
        internal static void ApplyNodeBounds(fProxyN L, fProxyN U,
                                             fProxyN curLB, fProxyN curUB, fProxyN xlRoot,
                                             fProxyMxN Aaug, NativeArray<int> col,
                                             fProxyN bAug, NativeArray<int> rowLB, NativeArray<int> rowUB,
                                             NativeArray<byte> integrality, int n,
                                             ref fProxyLPCache cache)
        {
            cache.matrixVersion++;

            for (int j = 0; j < n; j++)
            {
                if (integrality[j] == 0) continue;

                curLB[j] = L[j];
                bAug[rowLB[j]] = L[j] - xlRoot[j];

                curUB[j] = U[j];
                bool hiFinite = (double)U[j] < 1e29;
                Aaug[rowUB[j], col[j]] = hiFinite ? (fProxy)1 : (fProxy)0;
                bAug[rowUB[j]] = hiFinite ? (U[j] - xlRoot[j]) : (fProxy)0;
            }
        }

        /// <summary>
        /// Activity-based bound tightening (docs/draft-spec-mip.md stage 4), ported from
        /// mip/HighsDomain.cpp's <c>propagate</c>/<c>propagateRowUpper</c>/<c>propagateRowLower</c>:
        /// WORKLIST-driven (a row is only (re-)examined when a variable it touches just changed --
        /// HighsDomain's <c>markPropagate</c>/column-incidence loop -- not a blind repeated sweep of
        /// every row). For the row at the head of the queue, compute the min/max row activity from the
        /// current live integer bounds (<paramref name="curLB"/>/<paramref name="curUB"/>; continuous
        /// variables keep their root bounds, which never change) and derive a tightened bound for every
        /// INTEGER variable with a nonzero row coefficient. A row with more than one variable whose
        /// relevant bound is still infinite yields no tightening from that row (HiGHS's <c>ninfmin</c>/
        /// <c>ninfmax</c> infinite-contributor counts); the closed form used when exactly one (or zero)
        /// such contributor exists is <c>(rhs - (act - ownContribution)) / a_ij</c> -- HiGHS's
        /// <c>minresact</c>/<c>maxresact</c>. Tightened integer bounds round inward (floor for an upper
        /// bound, ceil for a lower bound). Every tightening is applied via <see cref="PushBoundChange"/>
        /// (same undo-stack entries as a branch decision, same UB-row inert/active handling for a
        /// variable whose bound was infinite) and re-queues every OTHER row with a nonzero coefficient
        /// on that variable (dense column scan standing in for HiGHS's sparse column-index list).
        /// Terminates when the queue drains (the true fixpoint, mirroring HiGHS's
        /// <c>havePropagationRows</c>) or a total-row-visit cap (<see cref="PROPAGATION_MAX_PASSES"/>
        /// times the row count -- HiGHS has no such cap because its incremental activity bookkeeping
        /// makes each visit O(row length) instead of a full recompute; a cap here is a deliberate,
        /// bounded-cost adaptation for a fixpoint that is not persisted/maintained incrementally across
        /// the whole B&amp;B tree the way HighsDomain's activity arrays are -- see the file header).
        /// </summary>
        /// <returns>False as soon as a row or a variable's own range is proven empty (L &gt; U) --
        /// caller fathoms the node WITHOUT an LP solve. True otherwise (a fixpoint, possibly a no-op,
        /// was reached).</returns>
        internal static bool PropagateFixpoint(fProxyMxN A, fProxyN b, NativeArray<ConstraintSense> senses,
                                               int m0, int n, NativeArray<byte> integrality, int nInt,
                                               fProxyN curLB, fProxyN curUB, fProxyN xlRoot,
                                               fProxyMxN Aaug, NativeArray<int> col, fProxyN bAug,
                                               NativeArray<int> rowLB, NativeArray<int> rowUB,
                                               ref UnsafeList<fProxyBoundChange> stack, ref fProxyLPCache cache)
        {
            if (nInt == 0 || m0 == 0) return true;   // nothing propagation can tighten

            // Worklist of row indices still due for (re-)examination, plus membership flags so a row is
            // never queued twice at once (HighsDomain's propagateinds_ + an implicit membership test).
            var inQueue = new NativeArray<bool>(m0, Allocator.Temp);
            var queue = new UnsafeList<int>(m0, Allocator.Temp);
            for (int i = 0; i < m0; i++) { queue.Add(i); inQueue[i] = true; }

            bool feasible = true;
            int qHead = 0;
            int visits = 0;
            int maxVisits = PROPAGATION_MAX_PASSES * m0;

            while (feasible && qHead < queue.Length && visits < maxVisits)
            {
                int i = queue[qHead]; qHead++;
                inQueue[i] = false;
                visits++;

                ConstraintSense se = senses[i];
                double bi = (double)b[i];
                bool hasHi = se != ConstraintSense.GreaterEqual;   // row upper limit: <= or =
                bool hasLo = se != ConstraintSense.LessEqual;      // row lower limit: >= or =

                double minAct = 0, maxAct = 0;
                int ninfMin = 0, ninfMax = 0;
                for (int j = 0; j < n; j++)
                {
                    double a = (double)A[i, j];
                    if (a == 0) continue;
                    double Lj = (double)curLB[j], Uj = (double)curUB[j];
                    bool loInf = Lj <= -1e29, hiInf = Uj >= 1e29;
                    if (a > 0)
                    {
                        if (loInf) ninfMin++; else minAct += a * Lj;
                        if (hiInf) ninfMax++; else maxAct += a * Uj;
                    }
                    else
                    {
                        if (hiInf) ninfMin++; else minAct += a * Uj;
                        if (loInf) ninfMax++; else maxAct += a * Lj;
                    }
                }

                // Row itself already infeasible against every current bound -> empty domain.
                if (hasHi && ninfMin == 0 && minAct > bi + PROPAGATION_TOL) feasible = false;
                if (feasible && hasLo && ninfMax == 0 && maxAct < bi - PROPAGATION_TOL) feasible = false;

                for (int j = 0; feasible && j < n; j++)
                {
                    if (integrality[j] == 0) continue;
                    double a = (double)A[i, j];
                    if (a == 0) continue;

                    double Lj = (double)curLB[j], Uj = (double)curUB[j];
                    bool loInf = Lj <= -1e29, hiInf = Uj >= 1e29;
                    bool tightened = false;

                    // From the row's upper limit (uses minAct): tightens U_j when a>0, L_j when a<0.
                    if (hasHi)
                    {
                        bool jInf = (a > 0) ? loInf : hiInf;
                        if (ninfMin - (jInf ? 1 : 0) == 0)
                        {
                            double own = jInf ? 0.0 : (a > 0 ? a * Lj : a * Uj);
                            double cand = (bi - (minAct - own)) / a;
                            if (a > 0)
                            {
                                double newU = math.floor(cand + PROPAGATION_TOL);
                                if (newU < Uj - PROPAGATION_TOL)
                                {
                                    PushBoundChange(ref stack, j, true, (fProxy)Uj, (fProxy)newU,
                                                    curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB, ref cache);
                                    tightened = true;
                                }
                            }
                            else
                            {
                                double newL = math.ceil(cand - PROPAGATION_TOL);
                                if (newL > Lj + PROPAGATION_TOL)
                                {
                                    PushBoundChange(ref stack, j, false, (fProxy)Lj, (fProxy)newL,
                                                    curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB, ref cache);
                                    tightened = true;
                                }
                            }
                        }
                    }

                    // From the row's lower limit (uses maxAct): tightens L_j when a>0, U_j when a<0.
                    if (hasLo)
                    {
                        bool jInf = (a > 0) ? hiInf : loInf;
                        if (ninfMax - (jInf ? 1 : 0) == 0)
                        {
                            double own = jInf ? 0.0 : (a > 0 ? a * Uj : a * Lj);
                            double cand = (bi - (maxAct - own)) / a;
                            if (a > 0)
                            {
                                double newL = math.ceil(cand - PROPAGATION_TOL);
                                if (newL > Lj + PROPAGATION_TOL)
                                {
                                    PushBoundChange(ref stack, j, false, (fProxy)Lj, (fProxy)newL,
                                                    curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB, ref cache);
                                    tightened = true;
                                }
                            }
                            else
                            {
                                double newU = math.floor(cand + PROPAGATION_TOL);
                                if (newU < Uj - PROPAGATION_TOL)
                                {
                                    PushBoundChange(ref stack, j, true, (fProxy)Uj, (fProxy)newU,
                                                    curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB, ref cache);
                                    tightened = true;
                                }
                            }
                        }
                    }

                    if ((double)curLB[j] > (double)curUB[j] + PROPAGATION_TOL) { feasible = false; break; }

                    // Column-to-row incidence (HighsDomain::markPropagate over the column's nonzeros):
                    // re-queue every row touching j, including this one (a later variable in this same
                    // row can unlock a stronger bound for j once j itself moved).
                    if (tightened)
                    {
                        for (int i2 = 0; i2 < m0; i2++)
                        {
                            if (!inQueue[i2] && (double)A[i2, j] != 0.0)
                            {
                                queue.Add(i2);
                                inQueue[i2] = true;
                            }
                        }
                    }
                }
            }

            queue.Dispose();
            inQueue.Dispose();

            // Safety net for a variable with a zero coefficient in every row (so it is never visited
            // above): HighsDomain::changeBound checks L<=U immediately on every single bound change,
            // independent of row activity, catching exactly this case; PushBoundChange itself is shared
            // with the branching/strong-branch call sites (out of this stage's scope to change), so the
            // same check is done here as one final O(n) sweep instead. A branching-created empty child
            // with no row coefficients at all is still caught correctly regardless, by the augmented LP's
            // own bound-row infeasibility on the next solve -- this sweep only saves that one LP solve.
            if (feasible)
                for (int j = 0; j < n; j++)
                    if (integrality[j] != 0 && (double)curLB[j] > (double)curUB[j] + PROPAGATION_TOL)
                        { feasible = false; break; }

            return feasible;
        }
    }
}
