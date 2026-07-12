# Release scan 2026-07-12 — area: qp-mip

Scanned 7 template files (core). Findings: total 3 — confirmed 3, uncertain 0, unverified 0, refuted 0; by severity: high 0, medium 1, low 2.

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QP.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QP.Info.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/MIP.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/MIP.Domain.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/MIP.Info.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/MIP.Pseudocost.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Optimize.fProxy.cs

## Findings

### 1. [medium/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/MIP.fProxy.cs:364 — The maxNodes/maxIter budget is checked AFTER the node LP solve but BEFORE the integral-incumbent extraction, so the last permitted node's integer-feasible solution is discarded; an integral root LP with maxNodes=1 returns NodeLimit with no incumbent instead of the optimal point.

**Evidence**

```csharp
if ((maxNodes > 0 && nodes >= maxNodes) || (maxIter > 0 && totalLpIter >= maxIter)) { status = (maxNodes > 0 && nodes >= maxNodes) ? MIPStatus.NodeLimit : MIPStatus.MaxIterations; break; }
```

Line 364-368. The only place a node's own LP solution becomes the incumbent is the branch block far below (line 397: `if (branchVar < 0) { haveIncumbent = true; incumbentObj = nodeObj; ... }`), which is never reached once this break fires. So when the budget is hit exactly at a node whose LP solution is integer-feasible (in particular an integral root LP solved with maxNodes=1), the code breaks here, `haveIncumbent` stays false, and the drain/finalize block writes `xOut[j]=0` with objective=+inf (line 505-507) even though a proven-optimal integer point was in hand.

**Verifier**

Traced MIP.fProxy.cs SearchCore: at nodes=1 with an integer-feasible root LP and maxNodes=1, the flow is nodes++ -> LP.solve (Optimal) -> nodes==1 special case only handles non-Optimal statuses -> compute nodeObj/frontierBound -> budget check at line 364 breaks with NodeLimit -> falls straight to the drain/finalize block. haveIncumbent is still false (line 397-402 is the only current-node install path and it lives inside the !prune branch that never runs), so line 505-507 writes objective=+Infinity and xOut[j]=0 despite the LP having already produced a proven-optimal integer point. Same failure mode applies whenever maxIter is exhausted at a node whose LP happens to be integer feasible.

**Suggested fix**

Move the node/iteration budget check to the top of the loop (before the LP solve of a NEW node), or extract the integral incumbent from the current node (run the branchVar<0 / rounding check) before breaking on the budget, so the final permitted node can still install its incumbent.

### 2. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QP.fProxy.cs:761 — Code comment records development history ('REPLACES the earlier Bland-style seam'), which the project comment policy requires to live in DEVLOG, not in source.

**Evidence**

```csharp
// Anti-cycling hardening: HiGHS-style deterministic bound perturbation (the exact pattern -- and lesson -- of LP.DualSimplexCore's own cost perturbation, see that file's header comment) REPLACES the earlier Bland-style seam.
```

Line 761-763. The 'REPLACES the earlier ... seam' clause is dev history / rejected-alternative narration, explicitly banned from code comments by CLAUDE.md.

**Verifier**

Lines 761-775 of Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QP.fProxy.cs contain the exact wording flagged: "HiGHS-style deterministic bound perturbation ... REPLACES the earlier Bland-style seam." That "REPLACES the earlier ... seam" phrasing is dev history / rejected-alternative narration, which CLAUDE.md's comment policy explicitly bans from source ("development history", "rejected alternatives" belong in DEVLOG.md). The rest of the block is a legitimate contract description, but the "REPLACES the earlier Bland-style seam" clause and the "and lesson -- of ... see that file's header comment" aside are the policy contradiction. Low-severity naming/policy issue; the fix is to strip the history clause and, if useful, move it to the folder DEVLOG.

**Suggested fix**

State the current contract only ('deterministic bound perturbation breaks exact ratio-test ties after degenCap zero-length steps'); move the 'earlier Bland-style seam' history to the folder DEVLOG.md.

### 3. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QP.fProxy.cs:543 — Code comment references an internal test/stage ('caught by the Stage-1 KKT-oracle check at k=2, n=8') and elsewhere ('caught by the LP-limit oracle test'); test-class/stage references belong in DEVLOG per the comment policy.

**Evidence**

```csharp
// ... (caught by the Stage-1 KKT-oracle check at k=2, n=8: k=1 has only one reflector at d=0 ...)
```

Line 543. Same pattern at line 874-876: `... -- caught by the LP-limit oracle test (Q=0 forces EVERY step through this exact path ...)`. These are notes about which test detects the bug / internal STAGE labels, which CLAUDE.md keeps out of code comments.

**Verifier**

Verified both cited sites in QP.fProxy.cs. Lines 541-543 contain "caught by the Stage-1 KKT-oracle check at k=2, n=8" (internal STAGE label + test-name + bug postmortem), and lines 874-876 contain "caught by the LP-limit oracle test (Q=0 forces EVERY step through this exact path...)" (test-name + debugging narration). CLAUDE.md explicitly forbids "internal spec/ticket references (... STAGE n)", "bug postmortems and debugging narration", and "notes to reviewers" in code comments, routing them to DEVLOG.md. The surrounding mathematical rationale (why leading-identity column restriction does not generalize) is legitimate contract/context and should stay; only the "caught by ... test" clauses violate policy.

**Suggested fix**

Keep the mathematical rationale (why the leading-identity column restriction does not generalize) but drop the 'caught by the ... oracle test at k=2,n=8' and 'Stage-1' references, relocating them to DEVLOG.md.

## Scanner notes

Verified clean (no finding warranted): QP null-space QR (FactorWorkingSetTranspose R-triangle read ordering, ApplyWorkingSetQtForward reflector replay, FormNullSpaceBasis full-width reflector application), the SolveReducedNewtonStep regularized rebuild from intact Z/QZ, the Harris ratio test pScale rescale/un-rescale and sentinel skips, all Allocator.Temp disposals on every early-return/unbounded path in eqpNullSpaceStep and qpActiveSetCore, MIP shift/split reformulation + bound-row activation (inert 0/0 vs coeff 1), PropagateFixpoint min/max activity bound derivations and ninf bookkeeping, uplocks/downlocks, HeapPush/HeapPopMin min-heap, and Optimize bisection/newton/goldenSection/gradientDescent/ladIRLS. Objective accumulation is consistently done in double, so float-template precision loss is confined to stored working data as intended. One non-defect observation: AddPseudocostObservation stores per-variable sums via `(fProxy)unitGain` (float in the float template) while globalPCSum accumulates in double; this is a minor precision asymmetry in a heuristic running-mean, not a correctness bug.
