using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // Pseudocost + reliability branching (docs/draft-spec-mip.md stage 3), cross-checked against
    // HiGHS's mip/HighsPseudocost.h and mip/HighsSearch.cpp (selectBranchingCandidate /
    // evalUnreliableBranchCands). Each integer variable tracks a running mean objective-degradation-
    // per-unit-fractionality for its up/down branch direction; a variable is trusted once both
    // directions have >= RELIABILITY observations, otherwise SelectBranchVariable strong-branches it
    // (a capped, throwaway child-LP solve) to seed real data before trusting the pseudocost estimate.
    public static partial class MIP
    {
        // unitGain = objDelta / |delta| (HighsPseudocost::addObservation); running mean via sum/count,
        // equivalent to HighsPseudocost's Welford update at the sample counts RELIABILITY bounds this
        // to. globalPCSum/globalPCCount is the fallback average used before a variable has any samples
        // of its own (HighsPseudocost::cost_total).
        internal static void AddPseudocostObservation(int j, bool isUp, double delta, double objDelta,
                                                       floatN pcUpSum, NativeArray<int> pcUpCount,
                                                       floatN pcDownSum, NativeArray<int> pcDownCount,
                                                       ref double globalPCSum, ref int globalPCCount)
        {
            if (delta == 0.0) return;   // defensive only -- callers only observe already-fractional vars
            double unitGain = objDelta / math.abs(delta);
            if (isUp) { pcUpSum[j] += (float)unitGain; pcUpCount[j]++; }
            else { pcDownSum[j] += (float)unitGain; pcDownCount[j]++; }
            globalPCSum += unitGain;
            globalPCCount++;
        }

        // Estimated objective degradation for branching j in direction `isUp` at fractional value v:
        // (up/down fractional distance) * (variable's own mean unit gain, or the global mean unit gain
        // as a fallback with zero samples of its own) -- HighsPseudocost::getPseudocostUp/Down.
        internal static double PseudocostEstimate(int j, bool isUp, float v,
                                                   floatN pcUpSum, NativeArray<int> pcUpCount,
                                                   floatN pcDownSum, NativeArray<int> pcDownCount,
                                                   double globalPCSum, int globalPCCount)
        {
            double vd = (double)v;
            double dist = isUp ? (math.ceil(vd) - vd) : (vd - math.floor(vd));
            int count = isUp ? pcUpCount[j] : pcDownCount[j];
            double unitCost;
            if (count > 0) unitCost = isUp ? (double)pcUpSum[j] / count : (double)pcDownSum[j] / count;
            else if (globalPCCount > 0) unitCost = globalPCSum / globalPCCount;
            else unitCost = 0.0;
            return dist * unitCost;
        }

        // Product-rule branching score: HighsPseudocost::getScore's costScore ranking, without its
        // cost_total^2 normalization/sigmoid remap -- those only matter when blending in the
        // inference/cutoff/conflict terms this stage does not have (no propagation/cuts/conflicts);
        // dropping them leaves the SAME ranking within one node's candidate set, since cost_total is
        // one constant shared by every candidate there.
        internal static double PseudocostScore(int j, float v,
                                               floatN pcUpSum, NativeArray<int> pcUpCount,
                                               floatN pcDownSum, NativeArray<int> pcDownCount,
                                               double globalPCSum, int globalPCCount)
        {
            double up = PseudocostEstimate(j, true, v, pcUpSum, pcUpCount, pcDownSum, pcDownCount, globalPCSum, globalPCCount);
            double down = PseudocostEstimate(j, false, v, pcUpSum, pcUpCount, pcDownSum, pcDownCount, globalPCSum, globalPCCount);
            return math.max(up, PSEUDOCOST_EPS) * math.max(down, PSEUDOCOST_EPS);
        }

        // One throwaway strong-branch trial: tightens j's bound (isUp: x_j >= ceil(v), else
        // x_j <= floor(v)), solves the child LP with a capped iteration budget on the SAME persistent
        // basis, records the objective-degradation observation if it reached optimality, then undoes
        // the bound change (Push/UndoToMarker) so the caller's live state is unchanged. A non-optimal
        // trial (infeasible/unbounded/iteration-capped) contributes no observation.
        internal static void StrongBranchTrial(int j, bool isUp, float v, double parentObj,
                                               ref UnsafeList<floatBoundChange> boundStack, ref LPBasis basis,
                                               floatMxN Aaug, floatN bAug, floatN costY,
                                               NativeArray<ConstraintSense> sensesAug,
                                               floatN c, NativeArray<byte> kind, NativeArray<int> col,
                                               floatN xlRoot, floatN xuRoot, floatN curLB, floatN curUB,
                                               NativeArray<int> rowLB, NativeArray<int> rowUB, int n,
                                               floatN trialY, floatN trialX,
                                               floatN pcUpSum, NativeArray<int> pcUpCount,
                                               floatN pcDownSum, NativeArray<int> pcDownCount,
                                               ref double globalPCSum, ref int globalPCCount,
                                               ref int totalLpIter)
        {
            int marker = boundStack.Length;
            float newBound = isUp ? math.ceil(v) : math.floor(v);
            if (isUp)
                PushBoundChange(ref boundStack, j, false, curLB[j], newBound, curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB);
            else
                PushBoundChange(ref boundStack, j, true, curUB[j], newBound, curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB);

            LPInfo info = LP.solve(in Aaug, in bAug, in costY, in sensesAug, ref trialY, out double _, ref basis, STRONG_BRANCH_ITER_CAP);
            totalLpIter += info.iterations;

            if (info.status == LPStatus.Optimal)
            {
                UnshiftToX(trialY, kind, col, xlRoot, xuRoot, n, trialX);
                double trialObj = 0;
                for (int k = 0; k < n; k++) trialObj += (double)c[k] * (double)trialX[k];
                double objDelta = math.max(0.0, trialObj - parentObj);
                double delta = (double)newBound - (double)v;
                AddPseudocostObservation(j, isUp, delta, objDelta, pcUpSum, pcUpCount, pcDownSum, pcDownCount, ref globalPCSum, ref globalPCCount);
            }

            UndoToMarker(ref boundStack, marker, curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB);
        }

        // Picks the branching variable at the current node: most-fractional before any pseudocost
        // history exists anywhere in the search (bootstrap), else the product-rule pseudocost score,
        // strong-branching the current best candidate's unreliable direction(s) until it is reliable
        // or the search-wide strong-branch budget is spent (HighsSearch::selectBranchingCandidate /
        // evalUnreliableBranchCands). Returns -1 when no integer variable is fractional.
        internal static int SelectBranchVariable(floatN xNode, NativeArray<byte> integrality, int n, double nodeObj,
                                                  floatN pcUpSum, NativeArray<int> pcUpCount,
                                                  floatN pcDownSum, NativeArray<int> pcDownCount,
                                                  ref double globalPCSum, ref int globalPCCount,
                                                  ref int sbCallsUsed, int sbBudget,
                                                  ref UnsafeList<floatBoundChange> boundStack, ref LPBasis basis,
                                                  floatMxN Aaug, floatN bAug, floatN costY,
                                                  NativeArray<ConstraintSense> sensesAug,
                                                  floatN c, NativeArray<byte> kind, NativeArray<int> col,
                                                  floatN xlRoot, floatN xuRoot, floatN curLB, floatN curUB,
                                                  NativeArray<int> rowLB, NativeArray<int> rowUB,
                                                  floatN trialY, floatN trialX, ref int totalLpIter)
        {
            var frac = new NativeArray<int>(math.max(n, 1), Allocator.Temp);
            int nc = 0;
            for (int j = 0; j < n; j++)
            {
                if (integrality[j] == 0) continue;
                double xd = (double)xNode[j];
                double f = xd - math.floor(xd);
                double dist = math.min(f, 1.0 - f);
                double tol = INTEGRALITY_TOL * math.max(1.0, math.abs(xd));
                if (dist > tol) frac[nc++] = j;
            }

            if (nc == 0) { frac.Dispose(); return -1; }

            if (globalPCCount == 0)
            {
                // Bootstrap: no pseudocost history anywhere yet -- fall back to most-fractional
                // (stage 2's rule) rather than strong-branch an arbitrary first candidate.
                int mostFrac = frac[0]; double bestDist = -1;
                for (int k = 0; k < nc; k++)
                {
                    int j = frac[k];
                    double xd = (double)xNode[j];
                    double f = xd - math.floor(xd);
                    double dist = math.min(f, 1.0 - f);
                    if (dist > bestDist) { bestDist = dist; mostFrac = j; }
                }
                frac.Dispose();
                return mostFrac;
            }

            while (true)
            {
                int best = frac[0]; double bestScore = -1.0;
                for (int k = 0; k < nc; k++)
                {
                    int j = frac[k];
                    double score = PseudocostScore(j, xNode[j], pcUpSum, pcUpCount, pcDownSum, pcDownCount, globalPCSum, globalPCCount);
                    if (score > bestScore) { bestScore = score; best = j; }
                }

                bool reliable = pcUpCount[best] >= RELIABILITY && pcDownCount[best] >= RELIABILITY;
                if (reliable || sbCallsUsed >= sbBudget) { frac.Dispose(); return best; }

                // Strong-branch whichever direction(s) of the current best candidate are still
                // unreliable; each call is a bounded LP solve against the persistent basis.
                float v = xNode[best];
                if (pcDownCount[best] < RELIABILITY && sbCallsUsed < sbBudget)
                {
                    StrongBranchTrial(best, false, v, nodeObj, ref boundStack, ref basis, Aaug, bAug, costY, sensesAug,
                                      c, kind, col, xlRoot, xuRoot, curLB, curUB, rowLB, rowUB, n, trialY, trialX,
                                      pcUpSum, pcUpCount, pcDownSum, pcDownCount, ref globalPCSum, ref globalPCCount, ref totalLpIter);
                    sbCallsUsed++;
                }
                if (pcUpCount[best] < RELIABILITY && sbCallsUsed < sbBudget)
                {
                    StrongBranchTrial(best, true, v, nodeObj, ref boundStack, ref basis, Aaug, bAug, costY, sensesAug,
                                      c, kind, col, xlRoot, xuRoot, curLB, curUB, rowLB, rowUB, n, trialY, trialX,
                                      pcUpSum, pcUpCount, pcDownSum, pcDownCount, ref globalPCSum, ref globalPCCount, ref totalLpIter);
                    sbCallsUsed++;
                }
                // loop back: re-score with the freshly-updated pseudocost (the best candidate may change)
            }
        }
    }
}
