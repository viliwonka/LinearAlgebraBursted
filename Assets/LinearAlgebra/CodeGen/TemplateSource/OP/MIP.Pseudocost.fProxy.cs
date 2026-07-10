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
                                                       fProxyN pcUpSum, NativeArray<int> pcUpCount,
                                                       fProxyN pcDownSum, NativeArray<int> pcDownCount,
                                                       ref double globalPCSum, ref int globalPCCount)
        {
            if (delta == 0.0) return;   // defensive only -- callers only observe already-fractional vars
            double unitGain = objDelta / math.abs(delta);
            if (isUp) { pcUpSum[j] += (fProxy)unitGain; pcUpCount[j]++; }
            else { pcDownSum[j] += (fProxy)unitGain; pcDownCount[j]++; }
            globalPCSum += unitGain;
            globalPCCount++;
        }

        // Estimated objective degradation for branching j in direction `isUp` at fractional value v:
        // (up/down fractional distance) * (variable's own mean unit gain, or the global mean unit gain
        // as a fallback with zero samples of its own) -- HighsPseudocost::getPseudocostUp/Down.
        internal static double PseudocostEstimate(int j, bool isUp, fProxy v,
                                                   fProxyN pcUpSum, NativeArray<int> pcUpCount,
                                                   fProxyN pcDownSum, NativeArray<int> pcDownCount,
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

        // Product-rule branching score, faithfully ported from HighsPseudocost::getScore(col, upcost,
        // downcost) (mip/HighsPseudocost.h): costScore = max(upcost,minThreshold)*max(downcost,minThreshold)
        // / max(minThreshold, cost_total^2), then mapScore(x) = 1 - 1/(1+x). upcost/downcost are
        // PseudocostEstimate above (== HighsPseudocost::getPseudocostUp/Down's 2-arg no-offset overload,
        // the one selectBranchingCandidate actually scores with: fractional-distance * own mean, falling
        // back to the running global-average pseudocost when the variable has zero samples of its own).
        // cost_total is that same running global average (globalPCSum/globalPCCount); minThreshold ==
        // PSEUDOCOST_EPS (both 1e-6) -- same clamp value AND placement as the source.
        // OMITTED (fidelity taxonomy -- subsystems this stage does not have): the conflictScore term (no
        // conflict analysis / no-good learning), the cutoffScore term (no cutoff-bound tracking), the
        // inferenceScore term (no propagation/inference statistics). OMITTED: degeneracyFactor weighting
        // (no LP-degeneracy detection) -- HiGHS only sets it > 1 while actively degenerate; fixed at its
        // non-degenerate default of 1.0, getScore's full expression collapses exactly to mapScore(costScore),
        // which is what this function returns.
        internal static double PseudocostScore(int j, fProxy v,
                                               fProxyN pcUpSum, NativeArray<int> pcUpCount,
                                               fProxyN pcDownSum, NativeArray<int> pcDownCount,
                                               double globalPCSum, int globalPCCount)
        {
            double up = PseudocostEstimate(j, true, v, pcUpSum, pcUpCount, pcDownSum, pcDownCount, globalPCSum, globalPCCount);
            double down = PseudocostEstimate(j, false, v, pcUpSum, pcUpCount, pcDownSum, pcDownCount, globalPCSum, globalPCCount);
            double costTotal = globalPCCount > 0 ? globalPCSum / globalPCCount : 0.0;   // HighsPseudocost::cost_total
            double costScore = math.max(up, PSEUDOCOST_EPS) * math.max(down, PSEUDOCOST_EPS)
                              / math.max(PSEUDOCOST_EPS, costTotal * costTotal);
            return 1.0 - 1.0 / (1.0 + costScore);   // HighsPseudocost::getScore's mapScore
        }

        // One throwaway strong-branch trial: tightens j's bound (isUp: x_j >= ceil(v), else
        // x_j <= floor(v)), solves the child LP with a capped iteration budget on the SAME persistent
        // basis, records the objective-degradation observation if it reached optimality, then undoes
        // the bound change (Push/UndoToMarker) so the caller's live state is unchanged. A non-optimal
        // trial (infeasible/unbounded/iteration-capped) contributes no observation.
        internal static void StrongBranchTrial(int j, bool isUp, fProxy v, double parentObj,
                                               ref UnsafeList<fProxyBoundChange> boundStack, ref LPBasis basis,
                                               ref fProxyLPCache cache,
                                               fProxyMxN Aaug, fProxyN bAug, fProxyN costY,
                                               NativeArray<ConstraintSense> sensesAug,
                                               fProxyN c, NativeArray<byte> kind, NativeArray<int> col,
                                               fProxyN xlRoot, fProxyN xuRoot, fProxyN curLB, fProxyN curUB,
                                               NativeArray<int> rowLB, NativeArray<int> rowUB, int n,
                                               fProxyN trialY, fProxyN trialX,
                                               fProxyN pcUpSum, NativeArray<int> pcUpCount,
                                               fProxyN pcDownSum, NativeArray<int> pcDownCount,
                                               ref double globalPCSum, ref int globalPCCount,
                                               ref int totalLpIter)
        {
            int marker = boundStack.Length;
            fProxy newBound = isUp ? math.ceil(v) : math.floor(v);
            if (isUp)
                PushBoundChange(ref boundStack, j, false, curLB[j], newBound, curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB, ref cache);
            else
                PushBoundChange(ref boundStack, j, true, curUB[j], newBound, curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB, ref cache);

            LPInfo info = LP.solve(in Aaug, in bAug, in costY, in sensesAug, ref trialY, out double _, ref basis, ref cache, STRONG_BRANCH_ITER_CAP);
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

            UndoToMarker(ref boundStack, marker, curLB, curUB, xlRoot, Aaug, col, bAug, rowLB, rowUB, ref cache);
        }

        // Picks the branching variable at the current node: most-fractional before any pseudocost
        // history exists anywhere in the search (bootstrap), else the product-rule pseudocost score,
        // strong-branching the current best candidate's unreliable direction(s) until it is reliable
        // or the search-wide strong-branch budget is spent (HighsSearch::selectBranchingCandidate /
        // evalUnreliableBranchCands). Returns -1 when no integer variable is fractional.
        internal static int SelectBranchVariable(fProxyN xNode, NativeArray<byte> integrality, int n, double nodeObj,
                                                  fProxyN pcUpSum, NativeArray<int> pcUpCount,
                                                  fProxyN pcDownSum, NativeArray<int> pcDownCount,
                                                  ref double globalPCSum, ref int globalPCCount,
                                                  ref int sbCallsUsed, int sbBudget,
                                                  ref UnsafeList<fProxyBoundChange> boundStack, ref LPBasis basis,
                                                  ref fProxyLPCache cache,
                                                  fProxyMxN Aaug, fProxyN bAug, fProxyN costY,
                                                  NativeArray<ConstraintSense> sensesAug,
                                                  fProxyN c, NativeArray<byte> kind, NativeArray<int> col,
                                                  fProxyN xlRoot, fProxyN xuRoot, fProxyN curLB, fProxyN curUB,
                                                  NativeArray<int> rowLB, NativeArray<int> rowUB,
                                                  fProxyN trialY, fProxyN trialX, ref int totalLpIter)
        {
            var frac = new NativeArray<int>(math.max(n, 1), Allocator.Temp);
            int nc = 0;
            for (int j = 0; j < n; j++)
            {
                if (integrality[j] == 0) continue;
                double xd = (double)xNode[j];
                double f = xd - math.floor(xd);
                double dist = math.min(f, 1.0 - f);
                // ABSOLUTE tolerance (HiGHS mip_feasibility_tolerance semantics). A relative form
                // (tol * max(1,|x|)) exceeds the max possible fractional distance 0.5 once |x| >= 5e5,
                // silently classifying every large-magnitude fractional value as integral.
                if (dist > INTEGRALITY_TOL) frac[nc++] = j;
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
                fProxy v = xNode[best];
                if (pcDownCount[best] < RELIABILITY && sbCallsUsed < sbBudget)
                {
                    StrongBranchTrial(best, false, v, nodeObj, ref boundStack, ref basis, ref cache, Aaug, bAug, costY, sensesAug,
                                      c, kind, col, xlRoot, xuRoot, curLB, curUB, rowLB, rowUB, n, trialY, trialX,
                                      pcUpSum, pcUpCount, pcDownSum, pcDownCount, ref globalPCSum, ref globalPCCount, ref totalLpIter);
                    sbCallsUsed++;
                }
                if (pcUpCount[best] < RELIABILITY && sbCallsUsed < sbBudget)
                {
                    StrongBranchTrial(best, true, v, nodeObj, ref boundStack, ref basis, ref cache, Aaug, bAug, costY, sensesAug,
                                      c, kind, col, xlRoot, xuRoot, curLB, curUB, rowLB, rowUB, n, trialY, trialX,
                                      pcUpSum, pcUpCount, pcDownSum, pcDownCount, ref globalPCSum, ref globalPCCount, ref totalLpIter);
                    sbCallsUsed++;
                }
                // loop back: re-score with the freshly-updated pseudocost (the best candidate may change)
            }
        }
    }
}
