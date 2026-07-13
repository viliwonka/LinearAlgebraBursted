# Report N14 — Krylov gap coverage (narrow scan)

Partition (two files that fell between other scanners' sort boundaries):
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.Guards.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.PBiCGStab.fProxy.cs

Read every line of both at full depth. Diffed pbiCGStab against biCGStab and the rest
of Krylov.fProxy.cs (cg/pcg/minres/cgls/lsqr). Verified RequireDistinctBuffers against
all five call sites, the IfProxyPreconditioner/IfProxyLinearOperator contracts, and the
fProxyILU0.Apply aliasing precondition. Checked OP/DEVLOG.md for rationale.

Bottom line: the preconditioned BiCGSTAB math is CORRECT for right-preconditioning
(x updated with the preconditioned directions pHat/sHat, true-unpreconditioned residual
convergence test, breakdown-status rnorm reporting all consistent with the sibling). No
HIGH findings. One MEDIUM sibling inconsistency and one LOW comment-policy item.

## Findings

### N14-1 (MEDIUM) — pbiCGStab default maxIterations breaks the family convention
Krylov.PBiCGStab.fProxy.cs:134 (and its doc at :130-131)

The BSR convenience overload defaults maxIterations to `2 * A.M_Rows`:
```
return pbiCGStab(in A, in M, in b, ref x, 2 * A.M_Rows, Consts.fProxySqrtEps);
```
Every other square-system solver in the family defaults to `A.M_Rows`, not `2*`:
biCGStab BSR (Krylov.fProxy.cs:872) and dense (:844), cg (:168/:202), pcg
(:363/:398/:434/:470), minres (:650/:681). biCGStab is itself the non-symmetric
BiCGSTAB, so "non-symmetric wants a bigger budget" cannot explain the divergence — the
un-preconditioned twin uses N. There is no OP/DEVLOG.md entry recording this as a
deliberate choice (searched: no pbiCGStab/BiCGSTAB default-iterations note).

Not an API lie — the `<summary>` at :130-131 honestly documents "2*A.M_Rows" — but a
caller reasonably expects the two BiCGSTAB variants to share a default iteration budget,
and they silently do not. A user porting from `biCGStab(A,b,x)` to
`pbiCGStab(A,M,b,x)` gets 2x the max work on non-convergence without any hint that the
budget doubled.

Failure scenario: a near-singular non-convergent system solved with the parameterless
overload runs up to 2N iterations under pbiCGStab vs N under biCGStab — same algorithm
family, different wall-time cap, no documented reason.

Fix direction: either change :134 to `A.M_Rows` to match every sibling, or, if 2N is
intentional (e.g. preconditioner-setup amortization), record the rationale in
OP/DEVLOG.md under `## Krylov` so it is not read as an accidental straggler.

### N14-2 (LOW) — Guards.cs header comment carries codegen/rejected-alternative rationale
Krylov.Guards.cs:7-11

```
// This lives in a //singularFile// partial (emitted ONCE, NOT multiplied into
// float/double) because RequireDistinctBuffers has no fProxy in its signature: if it
// were declared in the multiplying Krylov.fProxy.cs template it would be copied
// identically into both Krylov.float.cs and Krylov.double.cs -- two definitions of the
// same member in the same partial class -> CS0111.
```
Per CLAUDE.md (strict comment policy) code comments state contracts only; the
"why this structure / what the rejected placement would do (-> CS0111)" narration is
exactly the design-decision/rejected-alternative content that belongs in the folder's
DEVLOG.md. Borderline because the first clause (this is a type-agnostic helper emitted
once) is contract-ish and genuinely useful to a template editor, but the CS0111
cause-and-effect explanation is DEVLOG material.

Fix direction: trim the comment to the contract ("Type-agnostic Krylov aliasing guard;
emitted once via //singularFile// because it has no fProxy in its signature.") and move
the CS0111 rationale to OP/DEVLOG.md, e.g.
`## Krylov.Guards` / `- 2026-07-13 | RequireDistinctBuffers is a //singularFile// partial: declaring it in the multiplying Krylov.fProxy.cs template would duplicate it into the float and double partials -> CS0111. (was Krylov.Guards.cs:7-11)`

## Confirmed clean (verified, not assumed)

- **Preconditioned algorithm correctness.** Right-preconditioned BiCGSTAB is implemented
  correctly: pHat = M^-1 p, v = A pHat, sHat = M^-1 s, t = A sHat, and the solution is
  updated with the PRECONDITIONED directions `x += alpha*pHat` (line 83 early-exit and
  line 98) and `x += omega*sHat` (line 99) — not with p/s. The recurrence for p and the
  scalars rho/alpha/beta/omega stay in the original (unpreconditioned) space, matching the
  standard right-preconditioned BiCGSTAB. The p-update order
  (`p -= omega v` then `p = beta p + r`, lines 68-69) matches biCGStab:765-766.
- **Convergence test.** threshold = tolerance^2 * dot(b,b) and rr = dot(r,r) on the
  unpreconditioned residual — the doc's "true-residual convergence test" claim (line 12)
  is accurate and is the meaningful distinguishing property for the preconditioned
  variant (it converges on ||b-Ax||, not ||M^-1(b-Ax)||).
- **Breakdown-status rnorm.** Lines 65/76/92/96 report sqrt(rr) at breakdown; at each of
  those points x is still x_old (the alpha*pHat / omega*sHat commits are below), so rr is
  the residual of the committed iterate — same reasoning the biCGStab sibling documents at
  :792-796. The ss early-exit (line 84) correctly reports sqrt(ss) because x has just been
  advanced by alpha*pHat, making s the true residual of the new x.
- **Aliasing guard.** RequireDistinctBuffers is called with all 9 participating buffers
  (r,rHat0,p,v,t,x,b,pHat,sHat); the ptrs[] fill order (lines 34-36) matches the token
  order in the message string (line 37) exactly. pHat/sHat are correctly added vs the
  7-buffer biCGStab guard. Every M.Apply/A.Apply pair (p->pHat->v, r->sHat->t) has its
  in/out operands covered as distinct, satisfying both the IfProxyPreconditioner "z
  distinct from r" and IfProxyLinearOperator "y distinct from x" contracts, including
  fProxyILU0.Apply's own `z must not alias r` runtime check (fProxyILU0.cs:261).
- **Size validation.** All 9 vectors checked against A.Rows (lines 25-26); A square check
  (line 23); maxIterations>=1 (line 28) — matches sibling coverage.
- **Zero-b / NaN sanitize shortcut** (lines 41-45) matches the family idiom (copy b rather
  than multiply by 0).
- **RequireDistinctBuffers itself** (Guards.cs:18-24): O(count^2) pairwise compare over
  long-cast pointers, throws ArgumentException(who). Correct and used identically by
  minres(9)/biCGStab(7)/cgls(6)/lsqr(7)/pbiCGStab(9). The `who` string always names the
  method and buffer set. No defect.
- **Type-split.** Neither file has //+choose blocks. fProxy expands to float/double only
  (Krylov is inherently real, no int/uint variant). Every literal is guarded: `(fProxy)0`,
  `(fProxy)1`, `(fProxy)(-1)`, and Consts.fProxySqrtEps — no bare `f`-suffixed or
  precision-specific literal survives into the double variant. math.sqrt/math.isnan are
  valid for both float and double. No epsilon hardcoded to one precision.
- **Memory / arena.** The BSR convenience overload allocates 7 scratch vectors via
  b.fProxyTempVec (arena-owned, never hand-disposed) — identical pattern to cg/pcg/minres/
  biCGStab; not a leak. The zero-alloc primitive allocates nothing. The generic overload
  uses no managed types, no boxing, no LINQ.
- **Naming.** pbiCGStab / maxIterations / tolerance all conform to canon; no retired tokens
  (BSM/Elem/Linear/_OP), M_Rows kept as expected. Public surface matches sibling generic
  primitives (biCGStab<TOp> public).

Note (out of my partition, flagged for the test scanners): pbiCGStab's behavioral test
coverage lives in TemplateSourceTests/fProxy/SparseILU0Tests.fProxy.cs (N11/N12 scope) —
not audited here for happy-path-only edge gaps.

## Summary

| Severity | Count |
|----------|-------|
| HIGH     | 0     |
| MEDIUM   | 1     |
| LOW      | 1     |

Areas confirmed clean: preconditioned-BiCGSTAB numerics, convergence/breakdown rnorm
reporting, aliasing guard completeness and message/order match, size validation, type-split
literals, arena memory discipline, naming. No HIGH findings.
