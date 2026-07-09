using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace LinearAlgebra
{
    // Bound-change undo stack for MIP's search (MIP.fProxy.cs's SearchCore): push a tightening, undo
    // back to a marker. Stage 3 uses this only for the current plunge's dive step and for throwaway
    // strong-branch trials (MIP.Pseudocost.fProxy.cs); jumping to a different open node (a best-bound
    // queue pop) uses ApplyNodeBounds below instead, since the target is not necessarily an ancestor
    // reachable by undoing this stack. Stage 4's propagation will push several changes per node
    // through the same Push/UndoToMarker pair.
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
        /// in <paramref name="bAug"/>. Activates the UB row's coefficient if it was inert. Caller must
        /// pass an INTEGER <paramref name="varIndex"/> (the only kind ever branched).
        /// </summary>
        internal static void PushBoundChange(ref UnsafeList<fProxyBoundChange> stack, int varIndex, bool isUpper,
                                             fProxy oldBound, fProxy newBound,
                                             fProxyN curLB, fProxyN curUB, fProxyN xlRoot,
                                             fProxyMxN Aaug, NativeArray<int> col,
                                             fProxyN bAug, NativeArray<int> rowLB, NativeArray<int> rowUB)
        {
            stack.Add(new fProxyBoundChange { varIndex = varIndex, isUpper = isUpper, oldBound = oldBound, newBound = newBound });

            if (isUpper)
            {
                curUB[varIndex] = newBound;
                if ((double)oldBound >= 1e29) Aaug[rowUB[varIndex], col[varIndex]] = (fProxy)1;   // activate
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
        /// row's coefficient back to 0 when undoing past its first tightening. Truncates
        /// <paramref name="stack"/> to <paramref name="marker"/>.
        /// </summary>
        internal static void UndoToMarker(ref UnsafeList<fProxyBoundChange> stack, int marker,
                                          fProxyN curLB, fProxyN curUB, fProxyN xlRoot,
                                          fProxyMxN Aaug, NativeArray<int> col,
                                          fProxyN bAug, NativeArray<int> rowLB, NativeArray<int> rowUB)
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
        /// </summary>
        internal static void ApplyNodeBounds(fProxyN L, fProxyN U,
                                             fProxyN curLB, fProxyN curUB, fProxyN xlRoot,
                                             fProxyMxN Aaug, NativeArray<int> col,
                                             fProxyN bAug, NativeArray<int> rowLB, NativeArray<int> rowUB,
                                             NativeArray<byte> integrality, int n)
        {
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
    }
}
