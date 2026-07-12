# Release scan 2026-07-12 — area: mpc-qpseam (post-scan code)

{"total":2,"confirmed":2,"uncertain":0,"unverified":0,"refuted":0,"high":1,"medium":0,"low":1}

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/MPC.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/MPC.State.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/MPC.Info.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QP.fProxy.cs

## Findings

### 1. [high/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/MPC.State.fProxy.cs:495 — Prestabilized input-bound rows use predicted state x_{k+1} (Phi/Gamma block k) instead of x_k, an off-by-one that both mis-constrains the applied inputs and makes the warm-start guess infeasible.

**Evidence**

```
Rows are assembled as `sum += -Kstab[i,p] * Gamma[k*n+p, col]` (line 495) and
`sum += -Kstab[i,p] * Phi[k*n+p, q]` (line 502), plus `Arows[rowIdx, k*m+i] += 1`.
Per this file's own doc (lines 75-76) `Phi/Gamma` block k is the coefficient of
x_{k+1}, so the row encodes -Kstab*x_{k+1} + v_k <= uHi. But the physical relation
is u_k = -Kstab*x_k + v_k (DEVLOG 2026-07-12; ExtractU0 line 115-117 uses x0 for
stage 0; RecoverPhysicalUPlan line 287-288 uses x_k). Stage k needs x_k, i.e.
block (k-1) for k>=1 and identity/0 for k=0 (u_0 = -Kstab*x0 + v_0, no z-coupling).
The warm start makes this worse: BuildWarmStartGuess (MPC.fProxy.cs line 176-177)
sets v_k = uGuess_k + Kstab*x_k using x_k, so at the guess the row LHS =
uGuess_k + Kstab*(x_k - x_{k+1}) != uGuess_k, breaking the 'feasible by
construction' claim -> qpActiveSetCoreWarm reports Infeasible and MPC.solve
silently falls back every frame; when it does solve, the returned u_0's true
bound is never enforced.
```

**Verifier**

Traced the claim in Assets/LinearAlgebra/CodeGen/TemplateSource/OP/MPC.State.fProxy.cs lines 485-527. Phi/Gamma blocks are documented (and coded) to represent x_{k+1}'s coefficients (Phi block k = Acond^{k+1}, Gamma block (k,j) = Acond^{k-j}@B for j<=k). The physical prestabilization relation u_k = -Kstab*x_k + v_k, used consistently in ExtractU0 (MPC.fProxy.cs line 115), RecoverPhysicalUPlan (line 287), and BuildWarmStartGuess (line 176), requires x_k's coefficients (i.e. Phi/Gamma block k-1 for k>=1, and identity/zero for k=0). Instead the input-bound row assembly for stage k reads Phi[k*n+p, q] and Gamma[k*n+p, col] -- x_{k+1}'s coefficients -- producing rows that bound -Kstab@x_{k+1} + v_k rather than u_k = -Kstab@x_k + v_k. Evaluating the buggy row at the warm-start point v_k = u_guess_k + Kstab*x_guess_k gives LHS = u_guess_k + Kstab*(x_guess_k - x_guess_{k+1}), which is not bounded by uHi even with clipped u_guess -- the "feasible by construction" premise (MPC.fProxy.cs header) is broken for the prestab case. Independently, the true u_0 = -Kstab*x_0 + z[0:m] returned by ExtractU0 is never constrained to [uLo,uHi]. No test exercises Kstab (verified via read of MPCTests.fProxy.cs -- 0 mentions), and the DEVLOG entries for MPC (OP/DEVLOG.md lines 150-199) only validate closed-loop condensing identity and cond(H_cl), never the assembled Arows rows against u_k's physical bound. The claim's off-by-one is a genuine defect, not a documented design choice.

**Suggested fix**

Index the prestab input rows off x_k: use Phi/Gamma block (k-1) for k>=1 with qCoupling/z-coupling from that block, and for k=0 set qCoupling = -Kstab (x0 coupling) with zero z-coupling apart from the +/-1 on v_0. Add a prestabilization test that drives an input to saturation and checks the returned u_0 respects uLo/uHi and status==Optimal (no test currently exercises Kstab).

### 2. [low/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/MPC.fProxy.cs:87 — Fallback contract claims wstatus/z are left untouched, but qpActiveSetCoreWarm only short-circuits on Infeasible; an Unbounded return overwrites wstatusPersist and mutates z.

**Evidence**

```
MPC.solve's else-branch comment (lines 87-90) states s.z and s.wstatus are
'untouched' so the next frame retries from the last known-good state. That holds
for QPStatus.Infeasible (QP.fProxy.cs lines 832-843 return before allocating/
seeding wstatus and before qpActiveSetLoop). But QPStatus.Unbounded flows through
qpActiveSetLoop (which mutates x = s.z) and then unconditionally executes
`wstatusPersist.CopyFrom(wstatus)` (QP.fProxy.cs line 856), so on Unbounded the
persisted working set AND s.z are modified while MPCInfo still reports
iterations=0/activeSetChanges=0. Unbounded is unlikely for this bounded convex QP
but is not impossible (e.g. prestab with +/-INF box and a v-direction not pinned
by the general rows).
```

**Verifier**

Verified against the code paths at C:\Users\viliv\Documents\LinearAlgebraBursted\Assets\LinearAlgebra\CodeGen\TemplateSource\OP\MPC.fProxy.cs:60-90 and C:\Users\viliv\Documents\LinearAlgebraBursted\Assets\LinearAlgebra\CodeGen\TemplateSource\OP\QP.fProxy.cs:832-856 plus the qpActiveSetLoop Unbounded sites at :948 and :1010. Only the Infeasible branch short-circuits before wstatusPersist is written and before x can be mutated. On Unbounded, qpActiveSetLoop runs (potentially mutating x in prior iterations via `x[i] += alphaTake * p[i]` at :1045), then `wstatusPersist.CopyFrom(wstatus)` at :856 unconditionally overwrites the persisted working set. Consequently the inline fallback comment at MPC.fProxy.cs:87-90 that broadens the "untouched" guarantee to "Infeasible/Unbounded" is factually wrong; the file-header comment at :23-27 correctly limits it to Infeasible. This is a low-severity documentation-vs-behavior mismatch (Unbounded is defensive-only and unlikely for a well-posed convex MPC, matching the report's own "low" severity). The reporter's suggested fix (either short-circuit Unbounded like Infeasible or narrow the doc to Infeasible-only, plus snapshot z if a true prior-guess fallback is wanted) matches the actual gap.

Finding: CONFIRMED — low-severity contract mismatch.

File paths:
- C:\Users\viliv\Documents\LinearAlgebraBursted\Assets\LinearAlgebra\CodeGen\TemplateSource\OP\MPC.fProxy.cs (lines 23-27 header, 60-62 call site, 87-90 wrong comment, 93-100 zeroed diagnostics)
- C:\Users\viliv\Documents\LinearAlgebraBursted\Assets\LinearAlgebra\CodeGen\TemplateSource\OP\QP.fProxy.cs (lines 832-843 Infeasible short-circuit, 849-856 unconditional persist, 947-948 and 1004-1010 Unbounded sites, 1045 x mutation)

**Suggested fix**

Either short-circuit Unbounded the same way as Infeasible inside qpActiveSetCoreWarm (skip the wstatusPersist write and leave x unchanged), or narrow the MPC.solve fallback doc/behavior to acknowledge that only Infeasible preserves state and snapshot s.z before the QP so the fallback truly returns the prior guess.

## Scanner notes

Scanned all four target files in full plus MPCTests.fProxy.cs and the OP DEVLOG. Verified clean: the x2 QP-convention factor (H = 2*H_UU line 443, slack diag 2*rho2 line 446, gradient c = 2*GtQbar*(Phi x0 - r) line 234, deltaU c -= 2*S*uPrev line 243) is internally consistent with the 1/2 zHz+cz solver form; deltaU Hessian diagonal scale (2 interior / 1 terminal, lines 424-437) and cross-blocks are correct for symmetric S. Disposal across the qpActiveSetCore/Warm/Loop split is clean: qpActiveSetLoop no longer disposes wstatus/L/U (comment line 1172), both entry points dispose them exactly once (lines 764, 858), no double-free or leak; the Infeasible early-return frees Ax0/L/U before allocating wstatus. RepairWorkingSet re-admission (side-consistent, feasTol-gated, lines 1286-1311) matches SeedWorkingSet's pass 2. Soft-row slack guess (max(0, C x_{k+1} - d)) is consistent with the soft rows (which correctly constrain future states x_1..x_N) and the linear-vs-simulated trajectory match holds in both plain and prestab dynamics. MPCInfo iterations/activeSetChanges/maxSlackViolation/objective are honestly computed. Main gap enabling finding 1: the test battery covers only the non-prestab, non-deltaU path -- prestabilization and the deltaU penalty have zero test coverage.
