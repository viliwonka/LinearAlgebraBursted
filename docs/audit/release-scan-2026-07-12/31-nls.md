# Release scan 2026-07-12 — area: nls (post-scan code)

{"total":2,"confirmed":2,"uncertain":0,"unverified":0,"refuted":0,"high":0,"medium":0,"low":2}

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/NLS.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/NLS.Info.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Interfaces/ResidualFunction.fProxy.cs

## Findings

### 1. [low/numerical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/NLS.fProxy.cs:185 — All-columns-flat start (0 < LInf(J) <= flatThresh) leaves d all-zero, causing a divide-by-zero scaled gradient norm and mu=0 (undamped) solve.

**Evidence**

```
The whole-Jacobian stationary guard is `if (!(LInfJ0 > (fProxy)0))` (line 251 / 387), but
nlsUpdateScale classifies a column flat at `cn <= flatThresh` where
`flatThresh = Consts.fProxyEpsilon` (lines 48,56,268). If every column norm is <= epsilon
(i.e. 0 < LInf(J) <= epsilon), `maxRealColNorm` stays 0 (line 48), so
`effective = maxRealColNorm = 0` for all j and d stays at its pre-zeroed value. Then
nlsScaledGradNorm computes `math.abs((double)g[j] / (double)d[j])` with d[j]==0 -> inf/NaN
(line 185), and `mu = (fProxy)1e-3 * nlsMaxD2(in d)` = 0 (line 275 / 409) removes all
damping. Reported gradientNorm is inf/NaN and the solve degrades to FailedLinearSolve
instead of reporting a stationary point. The DEVLOG (2026-07-13 THIRD BUG) documents the
per-column flat floor but not the all-flat sub-case.
```

**Verifier**

Real defect at NLS.fProxy.cs lines 38-59, 251, 275, 341, 387, 409, 474.

Traced scenario: 0 < LInf(J) <= flatThresh (e.g., float32 J containing a single entry 1e-9 with flatThresh = fProxyEpsilon = 1.19e-7).

1. nlsUpdateScale (lines 42-58): if every column has cn <= flatThresh, first loop leaves maxRealColNorm = 0; second loop sets effective = maxRealColNorm = 0 for every j. d[] was zero-initialized (line 228 uses the default uninit=false path in fProxyN.cs:69-82, which calls ClearMemory), so d[j] stays 0 for every j.
2. LInfJ0 > 0 guard (lines 251, 387) passes because at least one J entry is nonzero, so we enter the else branch with all-zero d.
3. nlsScaledGradNorm (line 185) then divides g[j] by d[j]=0 -> 0/0=NaN (skipped by v>best because NaN comparisons return false) or nonzero/0=+/-inf (adopted). If any g[j] is nonzero, gnorm0 = inf.
4. mu = 1e-3 * nlsMaxD2(d) = 0 (lines 275, 409) — completely undamped, contrary to the file docstring at line 130 which says "d is floored away from zero".

Observable outcomes both wrong:
- With default gradTol = fProxySqrtEps > 0: stop = (inf <= gradTol * inf = inf) is true (inf <= inf is true in IEEE 754), status = Converged, but gnorm recomputed at line 341/474 is inf, so NLSInfo.gradientNorm returns inf.
- With user-supplied gradTol = 0: gradTol * inf = NaN, inf <= NaN is false, while loop entered with mu = 0, nlsSolveStep runs QR on essentially-zero augmented system, h non-finite returns false, status = FailedLinearSolve. This matches the claim's specific stated consequence.

DEVLOG 2026-07-13 THIRD BUG entry (OP/DEVLOG.md lines 61-105) explicitly documents the "column at-or-below flatThresh is floored at maxRealColNorm" mechanism but never enumerates the sub-case where NO column is above flatThresh — the fix breaks down there because there is no maxRealColNorm to floor against.

Test coverage gap: FlatParameterNoBlowup (NLSTests.fProxy.cs:243) covers only 1-of-4 flat; no test covers all-columns-flat.

The claim's "FailedLinearSolve" phrasing is one of the two possible defect outcomes (the one that manifests when user passes gradTol = 0). Under defaults, the outcome is Converged-with-inf-gradientNorm. Both are wrong; both stem from the identical d-all-zero root cause the claim identifies, at the exact lines cited. Severity low is accurate: requires all Jacobian columns to have norm below type epsilon, unusual but reachable (finite-difference roundoff on an insensitive model, degenerate residual, or a badly-scaled analytic Jacobian).

**Suggested fix**

Gate the early stationary branch at `LInfJ0 > flatThresh` (consistent with the per-column flat test) instead of `> 0`, or guard `maxRealColNorm == 0` inside nlsUpdateScale (report Converged/stationary, matching the existing LInfJ0-zero branch).

### 2. [low/performance/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/NLS.fProxy.cs:43 — nlsUpdateScale computes every column's L2 norm twice per call (redundant O(m*n) pass) on the LM hot path.

**Evidence**

```
Lines 43-49 loop over all columns computing `cn = math.sqrt(s)` solely to find
`maxRealColNorm`, then lines 51-58 recompute the identical `s`/`cn` for each column again
to set `effective`. The inner squared-sum
`for (int i = 0; i < m; i++) { fProxy v = J[i, j]; s += v * v; }` is byte-identical
between the two passes. nlsUpdateScale runs on every accepted iteration
(lines 269,313 / 404,447), so this doubles the column-norm work each iteration.
```

**Verifier**

Verified against NLS.fProxy.cs lines 38-59 and the OP DEVLOG.md NLS entries. The two inner-loop bodies at lines 45-47 and 53-55 are byte-identical: same `s = sum J[i,j]^2` accumulation, same `cn = math.sqrt(s)`, with no writes to J between them. `nlsUpdateScale` is invoked at lines 269, 313 (numeric core) and 404, 447 (analytic core); the two loop-internal calls (313, 447) run on every accepted LM iteration, so the O(m*n) squared-sum work is genuinely doubled per accepted iteration. The DEVLOG (lines 84-105) justifies WHY two passes exist (need `maxRealColNorm` from pass 1 before pass 2 can use it as the flat-column floor target) but never defends recomputing the identical norms — a scratch buffer of size n caching cn[j] between passes eliminates the redundancy without changing the algorithm. Row-major layout means the column stride is already cache-unfriendly, so double-passing over J compounds the cost. Severity "low" is fair (correctness unchanged, absolute cost bounded by typically-small n); the observation, evidence, call-site count, and suggested fix are all accurate. Not by-design.

**Suggested fix**

Compute each column norm once into a scratch/reuse buffer (or a single pass tracking both max and per-column values), then apply the floor in a second cheap loop over the cached norms.

## Scanner notes

Verified-correct (no defect): Nielsen accept factor max(1/3,1-(2rho-1)^3) & reject mu*=nu,nu*=2 (lines 317-319/330-331); predicted reduction 0.5*h^T(mu*D^2*h - g) matches the scaled damped normal equations (lines 198-208); scipy robust rescale jscale=sqrt(max(rho'+2*rho''*s,EPS)), rs=r*rho'/jscale, Js=J*jscale reproduces scale_for_robust_loss_function exactly and makes g=Js^T*rs the true robust gradient (jscale cancels) (lines 77-93,271); Huber/Cauchy/Tukey Rho/RhoPrime/RhoPrime2 are each internally self-consistent (Tukey uses a c2/3 objective-scale convention, but RhoPrime/RhoPrime2 are its correct d/ds derivatives, so the constant does not affect the minimizer); L2Loss is the exact identity of nlsApplyRobustScale; finite-difference forward/central step h=sqrt(max(epsfcn,eps))*max(|p|,1) matches MINPACK and pPert is fully reset each column; QR.solveInPlace scratch dims (u.N=m+n, w.N>=n, x=h.N=n, b=baug.N=m+n) all satisfied and Aaug/baug are fully rewritten each mu-retry (no stale rows, m constant); every allocation is Allocator.Temp and disposed on all exit paths incl. FailedLinearSolve; curveFit residual sign r=model-y matches the documented 0.5*sum(f-y)^2; default tolerances gradTol=fProxySqrtEps/stepTol=fProxyEpsilon scale per dtype; enums correctly placed in the non-templated singularFile to avoid CS0102. `d` is correctly zeroed (constructor bool arg is `uninit`, and d omits it) while working buffers use uninit:true and are fully written before read. Minor: curveFit takes `ref TModel model` but never writes fitted state back (model is stateless by IfProxyCurveModel contract, p carries the result) — harmless, could be `in`.
