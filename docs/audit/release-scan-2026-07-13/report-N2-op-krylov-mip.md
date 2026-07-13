# N2 narrow scan — TemplateSource/OP (Krylov.fProxy.cs .. MIP.Info.cs)

Scope: 21 alphabetically-sorted files from `Krylov.fProxy.cs` through `MIP.Info.cs`
inclusive (the actual alphabetical range in the folder; the brief's "24" appears to be
an overcount).

Files read line by line:
Krylov.fProxy.cs, LOBPCG.Cache.fProxy.cs, LOBPCG.Info.cs, LOBPCG.fProxy.cs,
LP.BarrodaleRoberts.fProxy.cs, LP.Cache.fProxy.cs, LP.DualSimplex.fProxy.cs,
LP.FrischNewton.fProxy.cs, LP.Info.cs, LP.InteriorPoint.fProxy.cs,
LP.RevisedSimplex.fProxy.cs, LP.Sparse.fProxy.cs, LP.fProxy.cs,
LQ.MinNormWorkspace.fProxy.cs, LQ.Workspace.fProxy.cs, LQ.fProxy.cs,
LQRP.Workspace.fProxy.cs, LQRP.fProxy.cs, LU.fProxy.cs, MIP.Domain.fProxy.cs,
MIP.Info.cs.

Applied every dimension per the brief (comment policy, error handling, numerics,
type-split, logic, naming, style) + the narrow-pass addendum sweep for role-swapped
InPlace wrappers, rename stragglers, missing InPlace suffix, [NoAlias] violations,
sibling-validation gaps, literal type keywords surviving substitution.

Note: this partition inherits an unusually strong DEVLOG trail (~200 lines specific to
these files documenting prior audit fixes, bug postmortems, and per-dtype thresholds).
Most numeric/tolerance choices are already justified there; findings below are
constrained to genuinely new observations.

---

## Findings

### MEDIUM

**M1. LQ.minNormSolve signature vs contract mismatch (ref where in fits)**  
Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LQ.fProxy.cs:511, :557, :651

Every LQ.minNormSolve overload takes the input matrix by `ref fProxyMxN A` (and the multi-RHS variant also takes `ref fProxyMxN B`), yet the XML docs for all three overloads state plainly "A is not modified" / "B is not modified", and the implementations only ever read A via `W.Data.CopyFrom(A.Data)` before working on the copy.

Sibling `LQ.decomp` at LQ.fProxy.cs:345 uses the correct `in fProxyMxN A` for the same non-destructive read; the settled solver-API-rework decision (docs/naming-style-guide.md) is that `ref` on a non-InPlace method signals mutation. The name `minNormSolve` (no `InPlace` suffix) plus the doc line both promise non-destructive behaviour -- the `ref` signature contradicts both.

Concrete impact: (a) callers reading only the signature will assume the buffer is a scratch, not their persistent A, and defensively copy it; (b) an `in` reference would also let read-only aliased callers pass through readonly wrappers the current signature rejects; (c) LQRP.decomp / LQ.decomp already use `in`, so consistency with the same-family sibling is violated. Compare to LQ.fProxy.cs:345 and LQRP.fProxy.cs:251 which both use `in fProxyMxN A` for non-destructive decomp.

Fix direction: change all three signatures to `in fProxyMxN A` (and the multi-RHS variant B to `in fProxyMxN B`). Not a behaviour change; the internal `.Data.CopyFrom` already treats both as read-only.

---

**M2. LOBPCGInfo.ToFixedString output does not match its own doc example**  
Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LOBPCG.Info.cs:60-69

The <summary> at line 60 documents the format as `LOBPCGInfo(Converged, iters=12, converged=4/4, maxResidual=1.23E-08)` -- i.e. X/Y where Y is the k that was requested. The actual builder at line 66 only interpolates the count, producing `converged=4` (no `/4`). The struct has no field holding the requested k, so the format cannot be produced without an API change; the doc is the incorrect party.

Fix direction: adjust the doc example to `converged=4` (match code), or add a `requestedK` field to LOBPCGInfo and update the format string to match the doc. The lightweight fix is the doc.

---

### LOW

**L1. LU.decompNoPivot / LU.decomp require caller-initialised L identity but never verify it**  
Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LU.fProxy.cs:25-73, :88-181

Both overloads' doc comments state "L must be caller-initialized to the identity matrix". The kernels only ever write below-diagonal entries of L (`L[j,k] = Ljk` for j > k). On/above-diagonal entries retain whatever the caller passed in -- pass an uninit-flagged buffer (`Allocator.Temp, false`) and the returned L has garbage on and above the diagonal. Size is validated (`A.M_Rows == L.M_Rows`); content is not.

An LU-consuming caller (`Blas.triLower(L)` or `LU.decompSolve(L, U, P, b)`) then reads those garbage entries as the unit lower-triangular factor's diagonal and above-diagonal zeros, silently producing a wrong solution. No NaN/Inf, no exception, no DirectSolveStatus signal.

Fix direction: initialise L to identity inside `decompNoPivot`/`decomp` themselves (one memset + one diagonal fill; O(m^2) but so is the rest of the routine). Alternatively: verify the diagonal-is-1 precondition in ENABLE_UNITY_COLLECTIONS_CHECKS.

---

**L2. LU exception messages inconsistently omit the method-name prefix**  
Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LU.fProxy.cs:105, :328, :616, :696, :798, :826

Every other size-check throw in LU.fProxy.cs prefixes with the method name (`"decomp: A needs to be square"`, `"solveInPlace: A_to_LU needs to be square"`, etc.). The pivot-size checks all use the bare string `"pivot size must equal matrix dimension"` -- no method name. When this fires from a test suite or job log, the caller cannot tell which of six overloads raised it.

Fix direction: prefix each with its own method name, matching the sibling checks in the same body.

---

**L3. LP.DualSimplex row-cost perturbation base 1e-12 is inert in the float variant**  
Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LP.DualSimplex.fProxy.cs:334, :344

`double rowPerturbBase = 1e-12;` then `perturbedCost[j] = cost[j] + (fProxy)((0.5 - r) * rowPerturbBase);`.

For row-index (logical) columns cost[j] is always 0 at build time, so the perturbation stores as ~5e-13 (representable in float) and does its tie-break job. But this contract is non-obvious: any future change that seeds nonzero cost on logical columns (e.g. Big-M-style artificial costs, or a warm-start caller pre-loading them) will find the perturbation silently vanishes in float because `1 + 5e-13 == 1` exactly in float arithmetic. The DEVLOG note is technically correct about REPRESENTATION but sidesteps the ARITHMETIC-precision point.

Fix direction: annotate the 1e-12 literal with a "requires cost[j]==0 for logical columns in float" comment, or use a per-dtype `//+choose[...]` literal (1e-7-scale for float, 1e-12 for double). Low severity: the current call path does maintain the "cost==0 on logicals" invariant.

---

**L4. 1e-13 fixed-variable threshold behaves per-dtype**  
Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LP.DualSimplex.fProxy.cs:123, :455  
Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LP.RevisedSimplex.fProxy.cs:220, :551

Every "is variable j fixed" test uses `upper[j] - lower[j] <= (fProxy)1e-13`. In float, 1e-13 is well below `math.ulp(1.0f) ~= 1.19e-7`, so at bounds near O(1) magnitude this check effectively degenerates to "upper - lower == 0 (bit-exact) or subnormal" -- a caller that passes upper = lower + 1e-10 (arithmetically a fixed variable to any practical tolerance) is treated as boxed in float but fixed in double, exposing a per-dtype behaviour split for the same input.

Fix direction: switch to a `//+choose[1e-7f|1e-13]` literal, or derive from `Consts.fProxyEpsilon`. Not urgent because the practical caller pattern (upper - lower is bit-exact zero for a fixed var) works identically on both dtypes. Same pattern is already handled correctly at LP.DualSimplex.fProxy.cs:134 (ratioTieTol) via `//+choose[1e-5f|1e-9]`.

---

**L5. LOBPCG initial-X random seed uses float-suffixed literals in the double variant**  
Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LOBPCG.fProxy.cs:114

`ws.X[i, c] = (fProxy)(seedRng.NextFloat() * 2f - 1f);`

The `2f` and `-1f` suffixes are literal `float`s that survive verbatim into the double variant. In this specific call `seedRng.NextFloat()` returns a `float` regardless, so the arithmetic is float-typed and the final cast to `double` loses no information the random source did not already lose. Functionally harmless; the deterministic-seed contract still holds.

Still: it is a "float literal in an fProxy template" leak the wide-scan pattern hunts for. If a future edit migrates the caller to a `NextDouble()`, the surviving `f` suffixes would silently downcast. Cheap fix: change to `(fProxy)2` and `(fProxy)1`.

---

### Not-a-finding notes (audit trail; no fix implied)

- **Krylov.fProxy.cs aliasing guards are thorough.** Every solver (cg/pcg/minres/biCGStab/cgls/lsqr/lsmr/cgne) opens with an unsafe pointer-uniqueness check via `RequireDistinctBuffers` (or an inline OR-chain for cg/pcg smaller sets). Guard set matches each solver's actual scratch -- no missing pair.
- **Verify-at-exit pattern is consistent** across cg, pcg, cgne, and cgls: on a candidate convergence the true residual is recomputed fresh via one extra Apply (+ApplyT for cgls) before returning Converged (Krylov.fProxy.cs:115-124, :312-320, :1030-1042, :2079-2087). MINRES, BiCGSTAB, LSQR, LSMR intentionally skip this per SolveInfo.cs:88 DEVLOG.
- **LOBPCG RequireDistinctBuffers covers all 23 scratch buffers** including generalized-B images (BX/BW/BP) and rowAux -- doc-comment on line 707 matches the ptrs array on lines 711-733.
- **LP.DualSimplex dual-feasibility repair uses ORIGINAL cost (not perturbedCost)** at LP.DualSimplex.fProxy.cs:443; matches DEVLOG-documented fix, correctly commented at :438-442.
- **LU-blocked path pivot identity vs unblocked form** explicitly checked at LU.fProxy.cs:191-196 -- bit-identical pivot choices, `k < m-1` bound preserved.
- **LQRP.lqrpKernel norm-downdate guard formula** at LQRP.fProxy.cs:180 uses the LAPACK `(1+ratio)*(1-ratio)` form (not `1 - ratio*ratio`) -- correct cancellation avoidance for near-1 ratios.
- **LP.Sparse warm-starts corrector PCG from affine-predictor dy** at LP.Sparse.fProxy.cs:262. Legitimate: same M and preconditioner (only rhsY changed).
- **All Allocator.Temp allocations matched by .Dispose() on every return path** in LP/MIP/LQRP/LU/Krylov cores. No leaks found.
- **No hand-editing of generated Assets/LinearAlgebra/Source/** from this partition scope.

---

## Summary table

| Severity | Count |
|----------|-------|
| HIGH     | 0     |
| MEDIUM   | 2     |
| LOW      | 5     |
| TOTAL    | 7     |

Areas confirmed clean (no real defects after full-line reading):
- Krylov solver family (cg/pcg/minres/biCGStab/cgls/lsqr/lsmr/cgne + Jacobi convenience wrappers): aliasing guards, verify-at-exit, allocator discipline, per-dtype tolerance handling all consistent.
- LOBPCG generalized-eigenproblem plumbing (B=I via fProxyIdentityOperator): fresh-matvec principle enforced, deflation cadence documented, safeguards 1/2/3 in place with justification.
- LP.RevisedSimplex Harris ratio test far-bound fallback: two-attempt structure intact.
- LP.DualSimplex + LP.Cache + LP.fProxy warm-solve cache flow: version tracking, factorsUsable propagation, and the ENABLE_UNITY_COLLECTIONS_CHECKS verify path all sound.
- MIP.Domain bound-change stack + UB-row inert/active state machine: Push/Undo symmetry holds; matrixVersion bump is on coefficient-flip transition only.
