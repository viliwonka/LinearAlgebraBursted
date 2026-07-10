#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // Mixed-integer programming: branch & bound over the dual simplex (docs/draft-spec-mip.md).
    // Pseudocost + reliability branching (MIP.Pseudocost.double.cs) picks the branching variable;
    // search order is best-bound-with-plunging -- dive one child immediately, push the sibling to a
    // best-bound priority queue, and on reaching a leaf jump to the queue's best node instead of
    // backtracking to the DFS parent. Stage 4: every node is domain-propagated to a fixpoint before
    // its LP solve (MIP.Domain.double.cs's PropagateFixpoint, fathoms empty domains without an LP
    // solve), a rounding heuristic tries to install an incumbent from each fractional node's LP
    // solution (TryRoundingHeuristic below), and absGap/relGap make MIPStatus.GapLimit reachable.
    //
    // min cᵀx s.t. Ax {≤,=,≥} b, xl≤x≤xu, x_j integer for integrality[j]!=0.
    //
    // Bounds-as-rows: LP.solve only supports x>=0, so every variable is shifted to a non-negative y
    // (anchor-low/anchor-high/free-split, same substitution as QP.PhaseOneFeasibleStart). Integer
    // variables require a finite xl and get two pre-allocated rows (y<=U, y>=L); branching only rewrites
    // their rhs, keeping the augmented LP's shape fixed so the same LPBasis stays warm-startable.
    //
    // An integer variable's UB row starts INERT (coefficient 0) when xu is infinite, and is activated
    // (coefficient set to 1) on its first branch -- a literal 1e30 sentinel rhs corrupts the dual
    // simplex's dataScale/artificialBound scaling and can silently bound a truly unbounded direction.
    // See MIP.Domain.double.cs.
    //
    // Warm start: one LPBasis persists across the whole search, including strong-branch trials. The
    // dual simplex's dual-feasibility repair makes a stale basis (right after a plunge dive, an undone
    // strong-branch trial, or a queue jump) a correct, not just fast, starting point.
    //
    // Node state: the current plunge's dive steps use the incremental bound-change stack
    // (MIP.Domain.double.cs's PushBoundChange/UndoToMarker), same as stage 2. A queue jump is not
    // generally to an ancestor, so it cannot replay/undo that stack -- instead each queued node carries
    // its own full length-n bound snapshot (doubleMIPQueueNode) and a jump overwrites the live bound
    // state wholesale (ApplyNodeBounds) and resets the stack.
    //
    // dualBound = min over every still-open node's own parent-LP bound -- the current plunge frontier
    // plus everything still in the queue (tighter than stage 2's DFS-ancestor approximation).
    internal struct doubleMIPQueueNode
    {
        public doubleN L;          // full lower-bound snapshot, length n (owned -- Dispose on pop/drain)
        public doubleN U;          // full upper-bound snapshot, length n (owned -- Dispose on pop/drain)
        public double parentBound; // the branching LP's own bound -- a sound lower bound for this node
        public int branchVar;      // parent's branching variable
        public double fracValue;   // branchVar's fractional LP value at the parent
        public double newBound;    // the bound this node applies to branchVar (floor or ceil of fracValue)
        public bool isUp;          // true: branchVar's LOWER bound was tightened to newBound
    }

    public static partial class MIP
    {
        // Binary min-heap over `heap`, keyed by parentBound (best-bound = smallest, minimization).
        internal static void HeapPush(ref UnsafeList<doubleMIPQueueNode> heap, doubleMIPQueueNode node)
        {
            heap.Add(node);
            int i = heap.Length - 1;
            while (i > 0)
            {
                int parent = (i - 1) / 2;
                if (heap[parent].parentBound <= heap[i].parentBound) break;
                doubleMIPQueueNode tmp = heap[parent]; heap[parent] = heap[i]; heap[i] = tmp;
                i = parent;
            }
        }

        internal static doubleMIPQueueNode HeapPopMin(ref UnsafeList<doubleMIPQueueNode> heap)
        {
            doubleMIPQueueNode top = heap[0];
            int last = heap.Length - 1;
            heap[0] = heap[last];
            heap.Length = last;
            int i = 0;
            while (true)
            {
                int l = 2 * i + 1, r = 2 * i + 2, smallest = i;
                if (l < heap.Length && heap[l].parentBound < heap[smallest].parentBound) smallest = l;
                if (r < heap.Length && heap[r].parentBound < heap[smallest].parentBound) smallest = r;
                if (smallest == i) break;
                doubleMIPQueueNode tmp = heap[smallest]; heap[smallest] = heap[i]; heap[i] = tmp;
                i = smallest;
            }
            return top;
        }

        /// <summary>
        /// Solve the mixed-integer program min cᵀx s.t. A x {≤,=,≥} b (per-row <paramref name="senses"/>),
        /// xl ≤ x ≤ xu, x_j ∈ ℤ for every flagged <paramref name="integrality"/>[j]. Branch &amp; bound
        /// over the dense dual simplex (docs/draft-spec-mip.md: pseudocost + reliability branching,
        /// best-bound node queue with plunging, activity-based domain propagation at every node, and a
        /// rounding heuristic).
        ///
        /// Every INTEGER variable needs a finite <paramref name="xl"/>[j] (throws
        /// <see cref="ArgumentException"/> otherwise). Continuous variables support the full general
        /// bound range (finite, +-infinite via the 1e30 sentinel, or free), matching <see cref="QP.solve"/>.
        /// </summary>
        /// <param name="A">Constraint coefficients, m×n (m constraints, n variables).</param>
        /// <param name="b">Right-hand sides, length m.</param>
        /// <param name="c">Objective coefficients, length n (minimized).</param>
        /// <param name="senses">Per-row constraint sense, length m.</param>
        /// <param name="xl">Variable lower bounds, length n. &lt;= -1e29 means unbounded below -- but
        /// every integer variable needs a finite one.</param>
        /// <param name="xu">Variable upper bounds, length n. &gt;= 1e29 means unbounded above.</param>
        /// <param name="integrality">Per-variable flag, length n: 0 = continuous, 1 = integer. A binary
        /// variable is simply integer with xl=0, xu=1.</param>
        /// <param name="x">Output solution, length n (overwritten): the best incumbent found, or all
        /// zeros if none (see <see cref="MIPInfo"/> for the exact per-status contract).</param>
        /// <param name="objective">Output cᵀx at the returned x -- same value as
        /// <see cref="MIPInfo.objective"/>.</param>
        /// <param name="maxNodes">B&amp;B node budget; &lt;= 0 means unlimited.</param>
        /// <param name="maxIter">Cumulative LP-iteration budget across every node solved so far;
        /// &lt;= 0 means unlimited.</param>
        /// <param name="absGap">Stop (<see cref="MIPStatus.GapLimit"/>) once <c>objective - dualBound
        /// &lt;= absGap</c>, given an incumbent. &lt;= 0 means no absolute-gap limit.</param>
        /// <param name="relGap">Stop (<see cref="MIPStatus.GapLimit"/>) once
        /// <c>(objective - dualBound) / max(1, |objective|) &lt;= relGap</c>, given an incumbent.
        /// &lt;= 0 means no relative-gap limit.</param>
        public static MIPInfo solve(in doubleMxN A, in doubleN b, in doubleN c,
                                    in NativeArray<ConstraintSense> senses,
                                    in doubleN xl, in doubleN xu,
                                    in NativeArray<byte> integrality,
                                    ref doubleN x, out double objective,
                                    int maxNodes = 0, int maxIter = 0,
                                    double absGap = 0.0, double relGap = 0.0)
        {
            int m0 = A.M_Rows, n = A.N_Cols;

            if (b.N != m0) throw new ArgumentException("MIP.solve: b.N must equal A.M_Rows");
            if (c.N != n) throw new ArgumentException("MIP.solve: c.N must equal A.N_Cols");
            if (senses.Length != m0) throw new ArgumentException("MIP.solve: senses.Length must equal A.M_Rows");
            if (xl.N != n) throw new ArgumentException("MIP.solve: xl.N must equal A.N_Cols");
            if (xu.N != n) throw new ArgumentException("MIP.solve: xu.N must equal A.N_Cols");
            if (integrality.Length != n) throw new ArgumentException("MIP.solve: integrality.Length must equal A.N_Cols");
            if (x.N != n) throw new ArgumentException("MIP.solve: x.N must equal A.N_Cols");

            for (int j = 0; j < n; j++)
                if (xl[j] > xu[j]) throw new ArgumentException("MIP.solve: xl must be <= xu componentwise");

            // kind/col: shift every variable to y>=0. 0=anchor-low, 1=anchor-high, 2=free-split.
            var kind = new NativeArray<byte>(math.max(n, 1), Allocator.Temp);
            var col = new NativeArray<int>(math.max(n, 1), Allocator.Temp);
            int nExtraCols = 0;
            for (int j = 0; j < n; j++)
            {
                bool loFinite = (double)xl[j] > -1e29;
                bool hiFinite = (double)xu[j] < 1e29;
                if (integrality[j] != 0 && !loFinite)
                    throw new ArgumentException($"MIP.solve: integer variable {j} needs a finite xl");

                if (loFinite) kind[j] = 0;
                else if (hiFinite) kind[j] = 1;
                else { kind[j] = 2; nExtraCols++; }
            }
            int nY = n + nExtraCols;
            { int next = 0; for (int j = 0; j < n; j++) { col[j] = next; next += (kind[j] == 2 ? 2 : 1); } }

            // Pre-allocated bound rows for kind==0 (anchor-low) variables.
            var rowLB = new NativeArray<int>(math.max(n, 1), Allocator.Temp);
            var rowUB = new NativeArray<int>(math.max(n, 1), Allocator.Temp);
            int extraRows = 0;
            for (int j = 0; j < n; j++)
            {
                rowLB[j] = -1; rowUB[j] = -1;
                if (kind[j] != 0) continue;
                bool isInt = integrality[j] != 0;
                bool hiFinite = (double)xu[j] < 1e29;
                if (isInt || hiFinite) rowUB[j] = extraRows++;
                if (isInt) rowLB[j] = extraRows++;
            }
            for (int j = 0; j < n; j++)
            {
                if (rowUB[j] >= 0) rowUB[j] += m0;
                if (rowLB[j] >= 0) rowLB[j] += m0;
            }
            int mAug = m0 + extraRows;

            var Aaug = new doubleMxN(mAug, nY, Allocator.Temp);      // zero-initialized
            var bAug = new doubleN(mAug, Allocator.Temp);
            var sensesAug = new NativeArray<ConstraintSense>(mAug, Allocator.Temp);
            var costY = new doubleN(nY, Allocator.Temp);             // zero-initialized

            for (int i = 0; i < m0; i++)
            {
                double shiftSum = (double)0;
                for (int j = 0; j < n; j++)
                {
                    double a = A[i, j];
                    if (a == (double)0) continue;
                    if (kind[j] == 0) { Aaug[i, col[j]] = a; shiftSum += a * xl[j]; }
                    else if (kind[j] == 1) { Aaug[i, col[j]] = -a; shiftSum += a * xu[j]; }
                    else { Aaug[i, col[j]] = a; Aaug[i, col[j] + 1] = -a; }
                }
                bAug[i] = b[i] - shiftSum;
                sensesAug[i] = senses[i];
            }
            for (int j = 0; j < n; j++)
            {
                double cj = c[j];
                if (kind[j] == 0) costY[col[j]] = cj;
                else if (kind[j] == 1) costY[col[j]] = -cj;
                else { costY[col[j]] = cj; costY[col[j] + 1] = -cj; }
            }
            for (int j = 0; j < n; j++)
            {
                if (kind[j] != 0) continue;
                if (rowUB[j] >= 0)
                {
                    int r = rowUB[j];
                    bool hiFiniteHere = (double)xu[j] < 1e29;
                    // Inert (0/0) when xu[j] is infinite; PushBoundChange activates it on first branch.
                    Aaug[r, col[j]] = hiFiniteHere ? (double)1 : (double)0;
                    bAug[r] = hiFiniteHere ? (xu[j] - xl[j]) : (double)0;
                    sensesAug[r] = ConstraintSense.LessEqual;
                }
                if (rowLB[j] >= 0)
                {
                    int r = rowLB[j];
                    Aaug[r, col[j]] = (double)1;
                    bAug[r] = (double)0;   // root bound == xl[j] -> shifted rhs 0
                    sensesAug[r] = ConstraintSense.GreaterEqual;
                }
            }

            var curLB = new doubleN(math.max(n, 1), Allocator.Temp);
            var curUB = new doubleN(math.max(n, 1), Allocator.Temp);
            for (int j = 0; j < n; j++) { curLB[j] = xl[j]; curUB[j] = xu[j]; }

            var info = SearchCore(Aaug, bAug, costY, sensesAug, c, integrality, kind, col, xl, xu,
                                  curLB, curUB, rowLB, rowUB, A, b, senses, m0, n, nY, mAug,
                                  maxNodes, maxIter, absGap, relGap, x);

            objective = info.objective;

            curLB.Dispose(); curUB.Dispose();
            Aaug.Dispose(); bAug.Dispose(); sensesAug.Dispose(); costY.Dispose();
            rowLB.Dispose(); rowUB.Dispose(); kind.Dispose(); col.Dispose();

            return info;
        }

        // y (length nY, shift space) -> x (length n, caller space).
        internal static void UnshiftToX(doubleN y, NativeArray<byte> kind, NativeArray<int> col,
                                        doubleN xlRoot, doubleN xuRoot, int n, doubleN xOut)
        {
            for (int j = 0; j < n; j++)
            {
                if (kind[j] == 0) xOut[j] = xlRoot[j] + y[col[j]];
                else if (kind[j] == 1) xOut[j] = xuRoot[j] - y[col[j]];
                else xOut[j] = y[col[j]] - y[col[j] + 1];
            }
        }

        // Best-bound search with plunging: dive one child immediately (reusing the persistent basis,
        // still a DFS descent for as long as branching continues), push the other child to the
        // best-bound heap with its own full bound snapshot. On reaching a leaf (integral incumbent,
        // pruned, or an infeasible/unbounded/maxiter child), jump to the heap's best node instead of
        // backtracking to the DFS parent -- discarding any queued node the incumbent has since caught
        // up to without solving it. `xOut` (length n) gets the best incumbent, or zeros.
        internal static MIPInfo SearchCore(doubleMxN Aaug, doubleN bAug, doubleN costY, NativeArray<ConstraintSense> sensesAug,
                                           doubleN c, NativeArray<byte> integrality,
                                           NativeArray<byte> kind, NativeArray<int> col,
                                           doubleN xlRoot, doubleN xuRoot,
                                           doubleN curLB, doubleN curUB,
                                           NativeArray<int> rowLB, NativeArray<int> rowUB,
                                           doubleMxN A, doubleN b, NativeArray<ConstraintSense> senses, int m0,
                                           int n, int nY, int mAug,
                                           int maxNodes, int maxIter, double absGap, double relGap,
                                           doubleN xOut)
        {
            var basis = new LPBasis(nY, mAug, Allocator.Temp);   // job-safe: unpopulated, seeded by first solve
            var y = new doubleN(nY, Allocator.Temp);
            var trialY = new doubleN(nY, Allocator.Temp);
            var xNode = new doubleN(math.max(n, 1), Allocator.Temp);
            var trialX = new doubleN(math.max(n, 1), Allocator.Temp);
            var incumbentX = new doubleN(math.max(n, 1), Allocator.Temp);
            var xRound = new doubleN(math.max(n, 1), Allocator.Temp);   // TryRoundingHeuristic scratch

            var boundStack = new UnsafeList<doubleBoundChange>(64, Allocator.Temp);
            var heap = new UnsafeList<doubleMIPQueueNode>(64, Allocator.Temp);

            var pcUpSum = new doubleN(math.max(n, 1), Allocator.Temp);      // zero-initialized
            var pcDownSum = new doubleN(math.max(n, 1), Allocator.Temp);    // zero-initialized
            var pcUpCount = new NativeArray<int>(math.max(n, 1), Allocator.Temp);
            var pcDownCount = new NativeArray<int>(math.max(n, 1), Allocator.Temp);
            double globalPCSum = 0; int globalPCCount = 0;

            int nInt = 0; for (int j = 0; j < n; j++) if (integrality[j] != 0) nInt++;
            int sbBudget = STRONG_BRANCH_CALLS_PER_INT_VAR * math.max(nInt, 1);
            int sbCallsUsed = 0;

            // Up/down locks (TryRoundingHeuristic below), ported from
            // HighsMipSolverData::runSetup's uplocks/downlocks: static over the ORIGINAL rows, computed
            // once per search, not per node. A row with a finite lower limit locks j UP when a_ij<0, DOWN
            // when a_ij>=0; a row with a finite upper limit locks j DOWN when a_ij<0, UP when a_ij>=0.
            var uplocks = new NativeArray<int>(math.max(n, 1), Allocator.Temp);
            var downlocks = new NativeArray<int>(math.max(n, 1), Allocator.Temp);
            for (int j = 0; j < n; j++)
            {
                int up = 0, down = 0;
                for (int i = 0; i < m0; i++)
                {
                    double a = (double)A[i, j];
                    if (a == 0) continue;
                    ConstraintSense se = senses[i];
                    bool hasHiL = se != ConstraintSense.GreaterEqual;
                    bool hasLoL = se != ConstraintSense.LessEqual;
                    if (hasLoL) { if (a < 0) up++; else down++; }
                    if (hasHiL) { if (a < 0) down++; else up++; }
                }
                uplocks[j] = up; downlocks[j] = down;
            }
            // Fixed internal seed (same convention as LOBPCG.double.cs's seedRng): MIP.solve has no
            // public seed parameter and must stay bit-deterministic across repeated identical calls
            // (docs/draft-spec-mip.md open question 6), unlike HiGHS's own randomizedRounding which
            // advances a solver-wide RNG. Draws happen only inside the deterministic search sequence, so
            // repeated calls on identical inputs draw the identical sequence.
            var roundRng = new Unity.Mathematics.Random(0x9E3779B1u);

            bool haveIncumbent = false;
            double incumbentObj = double.PositiveInfinity;
            double frontierBound = double.NegativeInfinity;   // current plunge frontier's own LP bound
            int nodes = 0;
            int totalLpIter = 0;
            MIPStatus status = MIPStatus.Optimal;

            // Pending pseudocost attribution for whichever node is about to be solved (its parent's
            // branch decision), consumed the moment that solve completes.
            bool havePending = false;
            int pendingVar = -1; double pendingFrac = (double)0, pendingNewBound = (double)0;
            bool pendingIsUp = false; double pendingParentBound = 0;

            while (true)
            {
                nodes++;

                // Domain propagation (docs/draft-spec-mip.md stage 4): tighten integer bounds to a
                // fixpoint from the ORIGINAL rows before this node's LP solve. An emptied domain
                // fathoms the node without ever calling LP.solve (usable stays false below).
                bool domainOk = PropagateFixpoint(A, b, senses, m0, n, integrality, nInt,
                                                  curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB, ref boundStack);

                LPInfo info = default;
                bool usable = false;
                double nodeObj = 0;

                if (domainOk)
                {
                    info = LP.solve(in Aaug, in bAug, in costY, in sensesAug, ref y, out double _, ref basis, 0);
                    totalLpIter += info.iterations;
                    usable = info.status == LPStatus.Optimal;
                }

                if (nodes == 1)
                {
                    if (!domainOk || info.status == LPStatus.Infeasible) { status = MIPStatus.Infeasible; break; }
                    if (domainOk && info.status == LPStatus.Unbounded) { status = MIPStatus.Unbounded; break; }
                    if (domainOk && info.status == LPStatus.MaxIterations) { status = MIPStatus.MaxIterations; break; }
                }

                if (usable)
                {
                    UnshiftToX(y, kind, col, xlRoot, xuRoot, n, xNode);
                    for (int j = 0; j < n; j++) nodeObj += (double)c[j] * (double)xNode[j];
                    frontierBound = nodeObj;

                    if (havePending)
                    {
                        double delta = (double)pendingNewBound - (double)pendingFrac;
                        double objDelta = math.max(0.0, nodeObj - pendingParentBound);
                        AddPseudocostObservation(pendingVar, pendingIsUp, delta, objDelta,
                                                 pcUpSum, pcUpCount, pcDownSum, pcDownCount, ref globalPCSum, ref globalPCCount);
                    }
                }
                havePending = false;

                if ((maxNodes > 0 && nodes >= maxNodes) || (maxIter > 0 && totalLpIter >= maxIter))
                {
                    status = (maxNodes > 0 && nodes >= maxNodes) ? MIPStatus.NodeLimit : MIPStatus.MaxIterations;
                    break;
                }

                // Gap limit (docs/draft-spec-mip.md stage 4): dualBound peeked cheaply as min(current
                // plunge frontier, best-bound heap root) -- same quantity the final drain below folds
                // into MIPInfo.dualBound, just without popping.
                if (haveIncumbent && (absGap > 0.0 || relGap > 0.0))
                {
                    double dBoundNow = heap.Length > 0 ? math.min(frontierBound, heap[0].parentBound) : frontierBound;
                    double gapAbsNow = incumbentObj - dBoundNow;
                    double gapRelNow = gapAbsNow / math.max(1.0, math.abs(incumbentObj));
                    if ((absGap > 0.0 && gapAbsNow <= absGap) || (relGap > 0.0 && gapRelNow <= relGap))
                    {
                        status = MIPStatus.GapLimit;
                        break;
                    }
                }

                // Non-root Infeasible/Unbounded/MaxIterations, or a propagation-emptied domain: pruned,
                // unusable bound.
                bool prune = !usable || (haveIncumbent && nodeObj >= incumbentObj - ABS_GAP);

                if (!prune)
                {
                    int branchVar = SelectBranchVariable(xNode, integrality, n, nodeObj,
                                                         pcUpSum, pcUpCount, pcDownSum, pcDownCount, ref globalPCSum, ref globalPCCount,
                                                         ref sbCallsUsed, sbBudget, ref boundStack, ref basis,
                                                         Aaug, bAug, costY, sensesAug, c, kind, col, xlRoot, xuRoot, curLB, curUB,
                                                         rowLB, rowUB, trialY, trialX, ref totalLpIter);

                    if (branchVar < 0)
                    {
                        haveIncumbent = true;
                        incumbentObj = nodeObj;
                        for (int j = 0; j < n; j++) incumbentX[j] = xNode[j];
                    }
                    else
                    {
                        // Rounding heuristic (docs/draft-spec-mip.md stage 4): the LP solution is
                        // fractional here, so try it -- a cheap, non-branching shot at a better incumbent.
                        TryRoundingHeuristic(xNode, integrality, n, uplocks, downlocks, ref roundRng,
                                             A, b, senses, m0, c, curLB, curUB,
                                             xRound, ref haveIncumbent, ref incumbentObj, incumbentX);

                        double v = xNode[branchVar];
                        double floorV = math.floor(v);
                        double ceilV = math.ceil(v);
                        bool downFirst = (v - floorV) <= (double)0.5;

                        // Snapshot the sibling (the child NOT dived into) from the live bound state
                        // before the dive mutates it, and push it to the best-bound queue.
                        var qL = new doubleN(math.max(n, 1), Allocator.Temp);
                        var qU = new doubleN(math.max(n, 1), Allocator.Temp);
                        for (int j = 0; j < n; j++) { qL[j] = curLB[j]; qU[j] = curUB[j]; }
                        if (downFirst) qL[branchVar] = ceilV; else qU[branchVar] = floorV;

                        HeapPush(ref heap, new doubleMIPQueueNode
                        {
                            L = qL,
                            U = qU,
                            parentBound = nodeObj,
                            branchVar = branchVar,
                            fracValue = v,
                            newBound = downFirst ? ceilV : floorV,
                            isUp = downFirst,   // sibling is the OPPOSITE direction from the dive child
                        });

                        // Dive into the preferred child directly on the live state.
                        if (downFirst)
                            PushBoundChange(ref boundStack, branchVar, true, curUB[branchVar], floorV, curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB);
                        else
                            PushBoundChange(ref boundStack, branchVar, false, curLB[branchVar], ceilV, curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB);

                        havePending = true;
                        pendingVar = branchVar; pendingFrac = v;
                        pendingNewBound = downFirst ? floorV : ceilV;
                        pendingIsUp = !downFirst; pendingParentBound = nodeObj;

                        continue;   // solve the dive child next
                    }
                }

                // Leaf: fetch the next work item from the best-bound queue, discarding entries the
                // incumbent has already caught up to (bound-based pruning without an LP solve).
                bool advanced = false;
                while (heap.Length > 0)
                {
                    doubleMIPQueueNode entry = HeapPopMin(ref heap);
                    if (haveIncumbent && entry.parentBound >= incumbentObj - ABS_GAP)
                    {
                        entry.L.Dispose(); entry.U.Dispose();
                        continue;
                    }

                    boundStack.Length = 0;   // the jump target may not be an ancestor -- wholesale rewrite
                    ApplyNodeBounds(entry.L, entry.U, curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB, integrality, n);
                    entry.L.Dispose(); entry.U.Dispose();

                    havePending = true;
                    pendingVar = entry.branchVar; pendingFrac = entry.fracValue; pendingNewBound = entry.newBound;
                    pendingIsUp = entry.isUp; pendingParentBound = entry.parentBound;
                    frontierBound = entry.parentBound;

                    advanced = true;
                    break;
                }

                if (!advanced) { status = haveIncumbent ? MIPStatus.Optimal : MIPStatus.Infeasible; break; }
            }

            // Drain any remaining queue entries (early stop via a limit) -- disposal, and fold their
            // bounds into the reported dualBound.
            double dualBound = frontierBound;
            while (heap.Length > 0)
            {
                doubleMIPQueueNode entry = HeapPopMin(ref heap);
                dualBound = math.min(dualBound, entry.parentBound);
                entry.L.Dispose(); entry.U.Dispose();
            }

            double objective, gap;
            if (status == MIPStatus.Infeasible || status == MIPStatus.Unbounded)
            {
                objective = double.NaN; dualBound = double.NaN; gap = double.NaN;
                for (int j = 0; j < n; j++) xOut[j] = (double)0;
            }
            else
            {
                if (status == MIPStatus.Optimal) dualBound = incumbentObj;   // fully explored: proven

                if (haveIncumbent)
                {
                    objective = incumbentObj;
                    gap = (incumbentObj - dualBound) / math.max(1.0, math.abs(incumbentObj));
                    for (int j = 0; j < n; j++) xOut[j] = incumbentX[j];
                }
                else
                {
                    objective = double.PositiveInfinity;
                    gap = double.PositiveInfinity;
                    for (int j = 0; j < n; j++) xOut[j] = (double)0;
                }
            }

            basis.Dispose(); y.Dispose(); trialY.Dispose(); xNode.Dispose(); trialX.Dispose(); incumbentX.Dispose();
            xRound.Dispose(); uplocks.Dispose(); downlocks.Dispose();
            boundStack.Dispose(); heap.Dispose();
            pcUpSum.Dispose(); pcDownSum.Dispose(); pcUpCount.Dispose(); pcDownCount.Dispose();

            return new MIPInfo { objective = objective, dualBound = dualBound, gap = gap, nodes = nodes, lpIterations = totalLpIter, status = status };
        }

        // Rounding heuristic (docs/draft-spec-mip.md stage 4), ported from
        // HighsPrimalHeuristics::randomizedRounding's per-variable rounding rule: a variable with no
        // "up lock" (uplocks[j]==0) is safe to round up unconditionally; failing that, no "down lock"
        // rounds down; failing both (locked in both directions), floor a randomized point in the
        // fractional interval -- HiGHS's `floor(relaxationsol[i] + randgen.real(0.1, 0.9))`. Continuous
        // variables are left at their LP value. Two intentional deviations from tryRoundedPoint/
        // randomizedRounding, both required by this library's constraints (see the call sites below for
        // why, not just "simpler"):
        //  (a) HiGHS re-solves an LP with the rounded integers fixed to repair continuous variables and
        //      confirm feasibility; we have no such subsystem available here (no per-node LP re-solve
        //      budget was scoped for this heuristic) and the mini-spec calls for an O(mn) direct
        //      feasibility check against the original rows instead -- see TryRoundingHeuristic's callers.
        //  (b) HiGHS's `randgen` is a solver-wide RNG advanced continuously; MIP.solve has no public seed
        //      parameter and must stay bit-deterministic across repeated identical calls (open question 6
        //      in the spec), so this uses a fixed internal seed instead (see roundRng in SearchCore).
        // Bound handling follows HiGHS: rounded values are clamped into the CURRENT node's bounds and
        // feasibility is checked against them (not the root bounds -- third-review finding).
        // Installs the point as the new incumbent when feasible and better than the current one (or there
        // is none yet).
        internal static void TryRoundingHeuristic(doubleN xNode, NativeArray<byte> integrality, int n,
                                                   NativeArray<int> uplocks, NativeArray<int> downlocks,
                                                   ref Unity.Mathematics.Random rng,
                                                   doubleMxN A, doubleN b, NativeArray<ConstraintSense> senses, int m0,
                                                   doubleN c, doubleN curLB, doubleN curUB, doubleN xRound,
                                                   ref bool haveIncumbent, ref double incumbentObj, doubleN incumbentX)
        {
            for (int j = 0; j < n; j++)
            {
                if (integrality[j] == 0) { xRound[j] = xNode[j]; continue; }
                double v = (double)xNode[j];
                double r;
                if (uplocks[j] == 0) r = math.ceil(v - INTEGRALITY_TOL);
                else if (downlocks[j] == 0) r = math.floor(v + INTEGRALITY_TOL);
                else r = math.floor(v + (double)rng.NextFloat(0.1f, 0.9f));
                // Clamp into the node's bounds (HiGHS randomizedRounding clamps to localdom).
                // Round-inward first: node bounds are integral once branched, but ROOT bounds are the
                // user's raw values and may be fractional -- clamping to a fractional bound would
                // install a fractional "integer" incumbent.
                double lbj = math.ceil((double)curLB[j] - INTEGRALITY_TOL);
                double ubj = math.floor((double)curUB[j] + INTEGRALITY_TOL);
                if (r < lbj) r = lbj;
                if (r > ubj) r = ubj;
                xRound[j] = (double)r;
            }

            for (int j = 0; j < n; j++)
            {
                double v = (double)xRound[j];
                if (v < (double)curLB[j] - ROUNDING_FEAS_TOL || v > (double)curUB[j] + ROUNDING_FEAS_TOL)
                    return;
            }

            for (int i = 0; i < m0; i++)
            {
                double act = 0;
                for (int j = 0; j < n; j++) act += (double)A[i, j] * (double)xRound[j];
                double bi = (double)b[i];
                double tol = ROUNDING_FEAS_TOL * (1.0 + math.abs(bi));
                ConstraintSense se = senses[i];
                if (se == ConstraintSense.LessEqual) { if (act > bi + tol) return; }
                else if (se == ConstraintSense.GreaterEqual) { if (act < bi - tol) return; }
                else { if (math.abs(act - bi) > tol) return; }
            }

            double cand = 0;
            for (int j = 0; j < n; j++) cand += (double)c[j] * (double)xRound[j];
            if (haveIncumbent && cand >= incumbentObj - ABS_GAP) return;

            haveIncumbent = true;
            incumbentObj = cand;
            for (int j = 0; j < n; j++) incumbentX[j] = xRound[j];
        }
    }
}
