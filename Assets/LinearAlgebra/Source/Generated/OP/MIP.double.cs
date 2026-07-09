#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using System;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // Mixed-integer programming: branch & bound over the dual simplex (docs/draft-spec-mip.md).
    // Stage 2: most-fractional branching, DFS with backtracking, warm-started node LPs.
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
    // Warm start: one LPBasis persists across the whole search. The dual simplex's dual-feasibility
    // repair makes a stale basis (right after backtracking) a correct, not just fast, starting point.
    //
    // dualBound = min(root bound, open DFS-stack frames' parent bounds) -- sound but loose under pure
    // DFS (no best-bound queue until stage 3).
    internal struct doubleMIPNode
    {
        public int marker;         // bound-change stack length before this frame's own changes
        public int branchVar;
        public double floorV;      // down child's new UB
        public double ceilV;       // up child's new LB
        public bool downFirst;     // true: explore x_j <= floorV before x_j >= ceilV
        public byte state;         // 0 = first child in flight, 1 = second child in flight
        public double parentBound; // this node's own LP bound
    }

    public static partial class MIP
    {
        /// <summary>
        /// Solve the mixed-integer program min cᵀx s.t. A x {≤,=,≥} b (per-row <paramref name="senses"/>),
        /// xl ≤ x ≤ xu, x_j ∈ ℤ for every flagged <paramref name="integrality"/>[j]. Branch &amp; bound
        /// over the dense dual simplex (docs/draft-spec-mip.md stage 2: most-fractional branching, pure
        /// DFS, no propagation/pseudocost/heuristics).
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
        public static MIPInfo solve(in doubleMxN A, in doubleN b, in doubleN c,
                                    in NativeArray<ConstraintSense> senses,
                                    in doubleN xl, in doubleN xu,
                                    in NativeArray<byte> integrality,
                                    ref doubleN x, out double objective,
                                    int maxNodes = 0, int maxIter = 0)
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
                                  curLB, curUB, rowLB, rowUB, n, nY, mAug, maxNodes, maxIter, x);

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

        // Applies one child of `node`'s branch: the first child if isFirstChild (per node.downFirst),
        // else the other side. Pushes one BoundChange, undo-able back to node.marker.
        internal static void ApplyChild(ref UnsafeList<doubleBoundChange> boundStack, doubleMIPNode node, bool isFirstChild,
                                        doubleN curLB, doubleN curUB, doubleN xlRoot,
                                        doubleMxN Aaug, NativeArray<int> col,
                                        doubleN bAug, NativeArray<int> rowLB, NativeArray<int> rowUB)
        {
            bool down = isFirstChild ? node.downFirst : !node.downFirst;
            int j = node.branchVar;
            if (down)
                PushBoundChange(ref boundStack, j, true, curUB[j], node.floorV, curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB);
            else
                PushBoundChange(ref boundStack, j, false, curLB[j], node.ceilV, curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB);
        }

        // DFS search: explicit stack, no recursion. `xOut` (length n) gets the best incumbent, or zeros.
        internal static MIPInfo SearchCore(doubleMxN Aaug, doubleN bAug, doubleN costY, NativeArray<ConstraintSense> sensesAug,
                                           doubleN c, NativeArray<byte> integrality,
                                           NativeArray<byte> kind, NativeArray<int> col,
                                           doubleN xlRoot, doubleN xuRoot,
                                           doubleN curLB, doubleN curUB,
                                           NativeArray<int> rowLB, NativeArray<int> rowUB,
                                           int n, int nY, int mAug,
                                           int maxNodes, int maxIter,
                                           doubleN xOut)
        {
            var basis = new LPBasis(nY, mAug, Allocator.Temp);   // job-safe: unpopulated, seeded by first solve
            var y = new doubleN(nY, Allocator.Temp);
            var xNode = new doubleN(math.max(n, 1), Allocator.Temp);
            var incumbentX = new doubleN(math.max(n, 1), Allocator.Temp);

            var boundStack = new UnsafeList<doubleBoundChange>(64, Allocator.Temp);
            var nodeStack = new UnsafeList<doubleMIPNode>(64, Allocator.Temp);

            bool haveIncumbent = false;
            double incumbentObj = double.PositiveInfinity;
            double rootBound = double.NegativeInfinity;
            int nodes = 0;
            int totalLpIter = 0;
            MIPStatus status = MIPStatus.Optimal;
            bool solveNode = true;

            while (true)
            {
                if (solveNode)
                {
                    LPInfo info = LP.solve(in Aaug, in bAug, in costY, in sensesAug, ref y, out double _, ref basis, 0);
                    nodes++;
                    totalLpIter += info.iterations;

                    if (nodes == 1)
                    {
                        if (info.status == LPStatus.Infeasible) { status = MIPStatus.Infeasible; break; }
                        if (info.status == LPStatus.Unbounded) { status = MIPStatus.Unbounded; break; }
                        if (info.status == LPStatus.MaxIterations) { status = MIPStatus.MaxIterations; break; }
                    }

                    bool usable = info.status == LPStatus.Optimal;
                    double nodeObj = 0;
                    if (usable)
                    {
                        UnshiftToX(y, kind, col, xlRoot, xuRoot, n, xNode);
                        for (int j = 0; j < n; j++) nodeObj += (double)c[j] * (double)xNode[j];
                        if (nodes == 1) rootBound = nodeObj;
                    }

                    if ((maxNodes > 0 && nodes >= maxNodes) || (maxIter > 0 && totalLpIter >= maxIter))
                    {
                        status = (maxNodes > 0 && nodes >= maxNodes) ? MIPStatus.NodeLimit : MIPStatus.MaxIterations;
                        break;
                    }

                    // Non-root Infeasible/Unbounded/MaxIterations: pruned, unusable bound.
                    bool prune = !usable || (haveIncumbent && nodeObj >= incumbentObj - ABS_GAP);

                    if (!prune)
                    {
                        int branchVar = -1; double bestDist = 0;
                        for (int j = 0; j < n; j++)
                        {
                            if (integrality[j] == 0) continue;
                            double xd = (double)xNode[j];
                            double frac = xd - math.floor(xd);
                            double dist = math.min(frac, 1.0 - frac);
                            double tol = INTEGRALITY_TOL * math.max(1.0, math.abs(xd));
                            if (dist > tol && dist > bestDist) { bestDist = dist; branchVar = j; }
                        }

                        if (branchVar < 0)
                        {
                            haveIncumbent = true;
                            incumbentObj = nodeObj;
                            for (int j = 0; j < n; j++) incumbentX[j] = xNode[j];
                        }
                        else
                        {
                            double v = xNode[branchVar];
                            double floorV = math.floor(v);
                            double ceilV = math.ceil(v);
                            bool downFirst = (v - floorV) <= (double)0.5;

                            var node = new doubleMIPNode
                            {
                                marker = boundStack.Length,
                                branchVar = branchVar,
                                floorV = floorV,
                                ceilV = ceilV,
                                downFirst = downFirst,
                                state = 0,
                                parentBound = nodeObj,
                            };
                            nodeStack.Add(node);
                            ApplyChild(ref boundStack, node, true, curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB);
                            continue;   // descend into the first child
                        }
                    }

                    solveNode = false;   // leaf -> backtrack
                    continue;
                }
                else
                {
                    if (nodeStack.Length == 0) { status = haveIncumbent ? MIPStatus.Optimal : MIPStatus.Infeasible; break; }

                    int topIdx = nodeStack.Length - 1;
                    doubleMIPNode top = nodeStack[topIdx];
                    if (top.state == 0)
                    {
                        UndoToMarker(ref boundStack, top.marker, curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB);
                        ApplyChild(ref boundStack, top, false, curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB);
                        top.state = 1;
                        nodeStack[topIdx] = top;
                        solveNode = true;
                        continue;
                    }
                    else
                    {
                        UndoToMarker(ref boundStack, top.marker, curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB);
                        nodeStack.Length = topIdx;   // pop
                        continue;
                    }
                }
            }

            double objective, dualBound, gap;
            if (status == MIPStatus.Infeasible || status == MIPStatus.Unbounded)
            {
                objective = double.NaN; dualBound = double.NaN; gap = double.NaN;
                for (int j = 0; j < n; j++) xOut[j] = (double)0;
            }
            else
            {
                dualBound = rootBound;
                for (int k = 0; k < nodeStack.Length; k++) dualBound = math.min(dualBound, nodeStack[k].parentBound);
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

            basis.Dispose(); y.Dispose(); xNode.Dispose(); incumbentX.Dispose();
            boundStack.Dispose(); nodeStack.Dispose();

            return new MIPInfo { objective = objective, dualBound = dualBound, gap = gap, nodes = nodes, lpIterations = totalLpIter, status = status };
        }
    }
}
