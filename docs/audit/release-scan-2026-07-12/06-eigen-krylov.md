# Release scan 2026-07-12 — area: eigen-krylov

Scanned 10 template files (core). Findings: total 4 — confirmed 4, uncertain 0, unverified 0, refuted 0; high 0, medium 0, low 4.

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Eigen.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Eigen.Info.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Eigen.LanczosWorkspace.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Eigen.SymWorkspace.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LOBPCG.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LOBPCG.Cache.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LOBPCG.Info.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.Guards.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.PBiCGStab.fProxy.cs

## Findings

### 1. [low/performance/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LOBPCG.Cache.fProxy.cs:121 — Cache fields AXnext and APnext are allocated (k x n each) but never read or written by the algorithm, wasting 2*k*n elements per workspace.

**Evidence**

```
Field decl: "public fProxyMxN Xnext, AXnext, Pnext, APnext;" with doc "AXnext/APnext are allocated but UNUSED; do not rely on their contents." and allocations in fProxyLOBPCGCache: "AXnext = arena.fProxyMat(k, n), ... APnext = arena.fProxyMat(k, n),".
```

UpdateActiveBlock writes only Xnext/Pnext then swaps X<->Xnext, P<->Pnext; AX/BX/AP/BP are recomputed fresh via Apply, so AXnext/APnext are never consumed.

**Verifier**

Grep across TemplateSource + generated float/double sources shows AXnext/APnext are only touched by the cache constructor, the size guard, and the distinct-buffer pointer sweep — never assigned, never read by any algorithm step. UpdateActiveBlock's own inline comment (LOBPCG.fProxy.cs:1163) says AX/AP-next-style mirror-combining is deliberately avoided because the caller re-Applies A/B immediately afterward, so no consumer exists by design. The field XML doc self-admits "AXnext/APnext are allocated but UNUSED; do not rely on their contents." with no retention rationale, so 2*k*n fProxy elements are wasted per workspace.

**Suggested fix**

Drop AXnext/APnext from fProxyLOBPCGCache and from fProxyLOBPCGCache(...) to save 2*k*n allocations per workspace, or document why they are retained as a deliberate reserve.

### 2. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LOBPCG.fProxy.cs:152 — Code comment states an empirical/benchmark verdict (a measured 0.87x residual-floor constant), which the project's contracts-only comment policy routes to DEVLOG.md.

**Evidence**

```
// ... the best residual achievable under that confinement is ~0.87x the frozen pair's lock residual. So lock with margin (0.087*tolerance induced floor) ...
```

An empirical measurement embedded in a code comment rather than a contract statement.

**Verifier**

The comment at LOBPCG.fProxy.cs:149-154 embeds an empirical/analytic factor ("~0.87x the frozen pair's lock residual", "0.087*tolerance induced floor") to justify the magic constant 0.1 in `lockTol = tolerance * 0.1`. CLAUDE.md's strict comment policy requires contracts only in code comments and explicitly routes rationale like measured floors, rejected alternatives, and derivations to DEVLOG.md. The contract statement is "lockTol = 0.1*tolerance"; the 0.87x justification is exactly the kind of numerical/perf-derived rationale that DEVLOG owns. Low severity as flagged.

**Suggested fix**

Move the empirical 0.87x rationale to the folder DEVLOG.md; keep only the contract (lockTol = 0.1*tolerance) in the comment.

### 3. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LOBPCG.fProxy.cs:349 — Code comment contains development history ('just like AX used to be'), which the contracts-only comment policy disallows in code.

**Evidence**

```
// ... P is reformed EVERY iteration from a combination of the CURRENT W and the OLD P (chained iteration to iteration, just like AX used to be) ...
```

'used to be' is dev history, not a contract.

**Verifier**

Lines 347-352 of Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LOBPCG.fProxy.cs literally contain the phrase "just like AX used to be" — an explicit reference to a prior code state, which CLAUDE.md's contracts-only comment policy classifies as development history ("an earlier version…", "changed from…") that must live in DEVLOG.md, not in code. "Same fix for AP/BP" reinforces the historical-remediation framing. The contract portion (AP/BP refreshed via fresh Apply each iteration because they feed the next iteration's Gram/H directly) can stay; the "used to be" clause should be relocated.

**Suggested fix**

Relocate the historical note to DEVLOG.md; keep the contract (AP/BP are refreshed via fresh Apply each iteration) in the comment.

### 4. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LOBPCG.Cache.fProxy.cs:107 — XML-doc for the AP field contains development narration ('this one mattered even more in practice'), a comment-policy violation.

**Evidence**

```
/// A applied to each row of P; recomputed via a FRESH A.Apply every iteration ... (this one mattered even more in practice -- an inaccurate AP fed directly into the NEXT iteration's Rayleigh-Ritz energy matrix, not just the residual check).
```

Development narration embedded in the AP field's XML doc rather than a contract statement.

**Verifier**

Lines 105-108 of Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LOBPCG.Cache.fProxy.cs contain the exact text "(this one mattered even more in practice -- an inaccurate AP fed directly into the NEXT iteration's Rayleigh-Ritz energy matrix, not just the residual check)" inside the AP field's <summary>. CLAUDE.md's strict comment policy limits code/XML docs to contracts only and explicitly bans "bug postmortems and debugging narration" and perf/benchmark rationale — those belong in the folder's DEVLOG.md. The parenthetical adds no contract (the contract "recomputed via a FRESH A.Apply every iteration" is already stated) and is exactly the "mattered in practice" rationale-narration the policy forbids. AX's sibling doc doesn't carry a symmetric note, confirming this is drift rather than intentional contract text.

**Suggested fix**

Trim to the contract ('AP = A applied to P, refreshed fresh each iteration'); move the 'mattered in practice' rationale to DEVLOG.md.

## Scanner notes

Genuine correctness search came back clean. Verified in detail: (1) Householder tred1/tred2 (v[m0]=x0-alpha, beta=2/vtv, p=beta*A*v, K=beta*(v.p)/2, symmetric rank-2 update) and the implicit-shift QL (pythag/copysign, deflation floored by global anorm) are faithful EISPACK; the transpose-in/transpose-out eigenvector accumulation with jacobiRotate on rows maps correctly back to columns and the descending selection sort carries columns along. (2) valuesQR is a faithful NR hqr port including exceptional shifts, p/q/r normalization guards (s2!=0), and column-update bound min(nn,k+3). (3) MINRES buffer rotations {r1<-r2, r2<-y} and {w1<-w2, w2<-w} plus combine3 exactly match Paige-Saunders (SOL minres.m); gamma floor guards div-by-zero. (4) CG/PCG/CGNE verify-at-exit recompute r=b-Ax fresh; bb==0 shortcut copies b (sanitizes NaN); breakdown guards are NaN-safe. (5) biCGStab/pbiCGStab report rr (x_old residual) on tt/omega breakdown correctly (r holds s, x not yet updated). (6) LSQR/LSMR damping folds, augmented-residual recovery in LstsqInfoTracked, and the LSMR ||r|| scalar recurrence are consistent; atbSq==0 and beta/alpha==0 early-outs are handled. (7) LOBPCG locking swaps move P/AP/BP/R with the pair; (d1) re-deflation aliases X-as-both-V-and-Against safely (active rows a<numActive disjoint from locked i>=numActive); guard-vector rank tie-break (j<i) yields exactly k wanted rows; Rayleigh-Ritz envelope validation and FactorGram ridge-retry are sound; all Allocator.Temp scratch is disposed on the single return paths. Aliasing guards (hand OR-chains and RequireDistinctBuffers) cover all live buffers. These files are float/double-only (no iProxy expansion), so no integer division/overflow pitfalls apply. Only comment-policy and one wasted-allocation item found.
