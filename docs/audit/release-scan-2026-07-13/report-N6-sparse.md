# Release scan 2026-07-13 — N6 narrow pass: TemplateSource/Sparse (20 files, every line read)

Scanner: N6. Scope: `Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/*.cs` (all 20 templates,
~5,284 lines), all dimensions at once, plus the narrow-pass addendum patterns. Proxy rules
confirmed against `TemplateConverter.cs`/`GenUtils.cs` (fProxy → float, double; every file in this
partition is fProxy-family). `DEVLOG.md` present in the folder and consulted before flagging
leftovers; nothing below duplicates what DEVLOG already records.

---

## HIGH

### H1 — `fProxyILU0.Apply`: `stackalloc` inside the per-block-row loop → stack growth O(blockRows), stack-overflow risk
- **File:** `Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxyILU0.cs:313`
- **Defect:** the backward sweep allocates scratch inside the descending block-row loop:
  ```csharp
  for (int i = nb - 1; i >= 0; i--)
  {
      ...
      fProxy* w = stackalloc fProxy[16];
  ```
  C# releases `stackalloc` memory only on METHOD return, not per loop iteration, so each Apply
  call grows the stack by `nb * 16 * sizeof(fProxy)`. In the generated **double** variant that is
  128 bytes per block row — ~1.3 MB at nb = 10,240 (the exact scale the sparse gallery doc
  advertises, and Apply runs every pbiCGStab iteration), overflowing typical job-thread stacks.
  Whether Burst happens to hoist a constant-size alloca is not guaranteed, and the Mono/editor
  fallback definitely grows. Every other loop in this partition deliberately hoists —
  `UnsafeOP.Sparse.fProxy.cs:1361` even documents the scratch as "stackalloc-ed ONCE, reused every
  row", and `Arena.Sparse.fProxy.cs:112/153` hoist before their loops — so this is accidental
  drift, not a chosen pattern.
- **Fix direction:** hoist `fProxy* w = stackalloc fProxy[16];` above the `for (int i = ...)` loop
  (it is fully overwritten each iteration).

---

## MEDIUM

### M1 — `BSR.spMM`: no validation of `rows` against either operand row count
- **File:** `Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/SparseOP.fProxy.cs:92-107`
- **Defect:** `spMM(in A, in Vrows, ref AVrows, int rows)` checks both `N_Cols` but never that
  `rows <= Vrows.M_Rows`, `rows <= AVrows.M_Rows`, or `rows >= 0`. `rows` then drives a raw-pointer
  `MemClear((long)rows * AVrows.N_Cols * ...)` and per-row pointer walks (`V + rv*ldV`) — a caller
  passing `rows` larger than the actual row count gets silent out-of-bounds reads/writes; a
  negative `rows` feeds a negative byte count into MemClear. Sibling entry points (`spMV`,
  `spMVT`, `sweepLower/Upper`, `rowSquaredWeighted`, `columnNormsSquared`) validate every
  dimension they touch. (Addendum pattern 5 — same family as the wide `Blas.dotRows` gap.)
- **Fix direction:** add `rows` range checks against `Vrows.M_Rows`/`AVrows.M_Rows` (and `rows >= 0`)
  next to the existing `N_Cols` guards.

### M2 — `BSR.samePattern` ignores the `Symmetric` flag → `addScaledInPlace` can silently mix storage kinds
- **File:** `Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/Comp.Sparse.fProxy.cs:56-71` (gate used at :40)
- **Defect:** `samePattern` compares grid, Nnzb, RowPtr, ColInd — but not `A.Symmetric == B.Symmetric`.
  A symmetric-storage `y` and a full-storage strictly-lower-triangular `x` can have byte-identical
  RowPtr/ColInd; `addScaledInPlace(y, a, x)` then passes the guard and sums stored entries, but the
  LOGICAL matrices differ (the upper triangle of y is implicit-mirrored, that of x is zero), so the
  result is wrong with no error. The sibling `BSR.transpose(in A, ref At)`
  (`SparseOP.Transpose.fProxy.cs:34`) explicitly requires matching `Symmetric` flags —
  inconsistent guard for the same hazard.
- **Fix direction:** compare `Symmetric` in `samePattern` (or add the flag check in
  `addScaledInPlace`).

### M3 — `fProxyBSRBuilder` type doc lies: says value-restamping is "a later phase" while `Refill`/`BuildAssemblyCache` ship it
- **File:** `Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxyBSRBuilder.cs:16-17` (also :11 "ONCE", :178 "Phase 1 pattern-edit scope")
- **Defect:** the struct doc says "call ToBSR(arena) ONCE", "Editing the pattern after compression
  is out of scope for Phase 1 -- ... (re-stamping VALUES on a fixed pattern without a rebuild is a
  later phase)" — but the SAME partial struct (`fProxyBSRAssembly.fProxy.cs`) ships exactly that:
  `Clear()` + `BuildAssemblyCache` + `Refill` is the documented per-frame reuse path. Stale contract
  in the generated public package, plus "Phase 1"/"later phase" are internal spec references the
  comment policy forbids.
- **Fix direction:** rewrite the two sentences to point at Clear/BuildAssemblyCache/Refill; move any
  phase history to DEVLOG. Proposed DEVLOG line:
  `## fProxyBSRBuilder.cs` / `- 2026-07-13 | "Phase 1 / later phase" scoping note removed from type doc; value-restamping shipped as BuildAssemblyCache/Refill. (was fProxyBSRBuilder.cs:16-17,178)`

### M4 — `fProxyNormalJacobi.Apply` has no size validation (every sibling preconditioner validates)
- **File:** `Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxySparseLP.fProxy.cs:109-112`
- **Defect:** `Apply(in r, ref z)` loops `for (i < r.N) z[i] = r[i] * InvDiag[i];` with no check that
  `r.N == z.N == InvDiag.N`. If `r` is longer than `InvDiag` this is an out-of-bounds read (and
  write past `z`). `fProxyBlockJacobi.Apply`, `fProxySSOR.Apply`, `fProxyIC0.Apply`, and
  `fProxyILU0.Apply` all validate both lengths. (Addendum pattern 5.)
- **Fix direction:** add the same `r.N`/`z.N` (vs `InvDiag.N`) ArgumentException guards as the
  sibling preconditioners.

---

## LOW

### L1 — `fProxyILU0.InvertBlockInPlace`: dead `perm` array
- **File:** `fProxyILU0.cs:205-207` — `int* perm = stackalloc int[16]; ... perm[t] = t;` is
  initialized and never read (the Gauss–Jordan swaps both `m` and `inv` rows directly, which is
  correct without permutation tracking). Delete the two lines.

### L2 — `fProxyILU0.BlockMulRight`: `stackalloc` inside the `r` loop
- **File:** `fProxyILU0.cs:171` — same pattern as H1 but bounded (at most 16 iterations × 16
  elements, roughly 2 KB per call, released at method return), so correctness is fine; still drift
  vs the "stackalloc once, before the loop" convention used everywhere else in the partition.
  Hoist above the `r` loop.

### L3 — Gallery doc: wrong dense-size figure, and literal "float" in a double-generating template
- **File:** `Gallery.Sparse.fProxy.cs:12` — "a dense 10000×10000 matrix is ~800 MB in float": 10⁸
  entries × 4 B ≈ **400 MB** in float (800 MB is the double figure), and the sentence says "float"
  verbatim in the generated double variant too. Reword the size claim per-type or drop the number.

### L4 — `const float fProxySparseOffScale = 0.3f`: name token substitutes, type does not
- **File:** `Gallery.Sparse.fProxy.cs:25` — generates `const float doubleSparseOffScale = 0.3f;` in
  the double variant (addendum pattern 6). Behavior is fine (all uses cast, and
  `Random.NextFloat` is float-typed anyway), but the fProxy-prefixed name on a hardcoded-float
  const is a sore thumb in generated source. Either rename without the proxy token or type it
  `fProxy` with casts at the NextFloat call sites.

### L5 — Stale "future wrapper (Phase 2)" class comment
- **File:** `SparseOP.fProxy.cs:11-13` — "a future generic IfProxyLinearOperator wrapper (Phase 2)
  can forward Apply/ApplyT straight to spMV/spMVT" — `fProxyBSROperator` exists and does exactly
  this; "Phase 2" is an internal spec ref. Rewrite as present-tense contract
  ("fProxyBSROperator forwards Apply/ApplyT here").

### L6 — "(Q4 ruling)" internal ticket reference in a public XML doc
- **File:** `SparseOP.fProxy.cs:286` (sweepLower doc) — "FULL-storage BSR only (Q4 ruling)".
  Drop "(Q4 ruling)"; the constraint itself is already stated. Proposed DEVLOG line:
  `## SparseOP.fProxy.cs` / `- 2026-07-13 | sweepLower/sweepUpper full-storage-only is an owner ruling (Q4). (was SparseOP.fProxy.cs:286)`

### L7 — Audit narration in comment
- **File:** `fProxyBSR.cs:69` — "only a WHOLE-FIELD reassignment from outside this file would be
  unsafe, and there is none (grepped repo-wide)". "(grepped repo-wide)" is reviewer narration, not
  contract. Trim the parenthetical.

### L8 — Benchmark-verdict reference in kernel comment
- **File:** `UnsafeOP.Sparse.fProxy.cs:656` — "b=1 not paired -- mirrors bsrMatVecB1 (same A/B
  finding applies: trivial per-block work)". The A/B verdict already lives in the DEVLOG of this
  folder (R2/R8 entry); trim to "b=1 not paired -- mirrors bsrMatVecB1."

### L9 — `bsm` local names are BSM→BSR rename stragglers
- **File:** `fProxyBSRBuilder.cs:211,219,222,227-228,236,304,309,311-313,352` — locals named `bsm`
  survive the retired BSM token (addendum pattern 2). Locals only, generated-source cosmetic;
  rename to `bsr`.

### L10 — `[NoAlias]` self-pass idiom (library-wide, noted for completeness)
- **Files:** `Comp.Sparse.fProxy.cs:23` — `UnsafeOP.signFlip(A.Values.Ptr, A.Values.Ptr, ...)`
  passes the same pointer to both `[NoAlias]` parameters (`target`, `from` —
  `OP/UnsafeOP.fProxy.cs:758`); `Norms.Sparse.fProxy.cs:51` — `vecDot(vals.Ptr, vals.Ptr, ...)`.
  Semantically safe (same-index element-wise write / read-only reduction) and identical to the
  dense siblings (`OP/OP.Component.fProxy.cs:165`, `OP/NormsOP.fProxy.cs:235`), so this is a
  library-wide idiom rather than Sparse drift — but it does contradict the declared no-alias
  contract (addendum pattern 4). If addressed, address at the kernel declarations library-wide,
  not per call site.

### L11 — `fProxyBSRAssemblyCache` public fields are camelCase; every sibling struct is PascalCase
- **File:** `fProxyBSRAssembly.fProxy.cs:18-22` — `slotOfTriplet/tripletRow/tripletCol/nnzb/tripletCount`
  vs `fProxyBSR.BlockRows`, `fProxyIC0.L/Shift`, `fProxyLadOperator.Sp/Tm/Atr`, `fProxySSOR.ScaledD`.
  Visible public-surface inconsistency in the generated package.

### L12 — Arena factory record-slot leak if the chained ctor throws
- **File:** `Arena.Sparse.fProxy.cs:87-91` (`fProxyBlockJacobi` factory; same shape at :40-44 for
  `fProxyBSR`) — the record slot is `Allocate`d BEFORE the ctor runs; if the ctor throws (missing
  diagonal block, singular block, symmetric-shape violation) the slot stays alive with a default
  payload until arena disposal. No memory corruption (default UnsafeList dispose is a no-op) —
  just slot/bookkeeping leakage on an error path. Consider Allocate-after-validate or a try/Free.

### L13 — `fProxyNormalOperator.ApplyBlock` sizes both scratch vectors from `Vrows.N_Cols`
- **File:** `fProxySparseLP.fProxy.cs:79-92` — `rin` AND `rout` both use `cols = Vrows.N_Cols`,
  where the siblings (`fProxyLadOperator.ApplyBlock:189-190`, `fProxySlackAugmentedOperator:276-277`)
  use `Cols` for input and `Rows` for output. Correct today only because M is square
  (`Rows == Cols`); a mismatched `Vrows` fails later and less clearly than in the siblings.
  Copy-paste drift; use `Cols`/`Rows` like the siblings.

---

## Areas confirmed clean

- **UnsafeOP.Sparse kernels (all 40+):** every unrolled B1/B2/B3/B4/B6 specialization
  (matVec/matVecT/matVecSym/matMat/matMatSym/blockJacobiApply/sweepLower/sweepUpper) was checked
  index-by-index against its general runtime-BR fallback — transpose index pairings
  (`block[c*b+r]` mirrors), row/col bases, ascending-ColInd `break`/`continue` logic, and the
  `(D/diagScale + L/U) y = r  =>  y_i = diagScale·D_i⁻¹·acc` algebra are all correct; accumulation
  order matches the documented bit-identical fold.
- **IC0/ILU0 math:** IC(0) up-looking factorization (pattern intersection merge, `S·L_jj^{-T}`
  forward-solve rows, guarded Cholesky with NaN-safe `!(sum > pivotFloor)`), Manteuffel shift
  escalation with per-type `Consts.fProxyEpsilon`, and both triangular Apply sweeps (including
  the scatter-style backward pass) verified; ILU0 IKJ trailing update and Gauss–Jordan pivot
  inverse verified (aside from L1/L2/H1 above).
- **Builder/compression:** counting-sort + per-row insertion sort, duplicate summing, `RowPtr`
  fill, `ToBSRSymmetric` guards (upper-triplet rejection, per-type relative tolerance on diagonal
  symmetry), and `BuildAssemblyCache`/`Refill` slot mapping and topology validation are correct.
- **Transpose/mirror paths:** `BSR.transpose` (CSR-of-transpose histogram/scatter, per-block
  transpose), `Arena.fProxyBSRTranspose`, `fProxyBSRMirrorToFull`, `fProxyBSR.ToDense` symmetric
  mirror all correct; symmetric copy fast-path is right (lower-canonical storage, matching the
  2026-07-12 triangle-trust DEVLOG entry).
- **Guards for Symmetric storage:** `Norms.L1/L2` throw, `LInf` correctly does not;
  `columnNormsSquared`/`rowSquaredWeighted` throw; `sweepLower/Upper` throw; `spMVT` forwards to
  `spMV` with an argued-equivalent guard set; `Debug.Spy` display-only mirroring correct.
- **Gallery:** SPD dominance accounting (offBound both sides, duplicate draws over-count safely),
  Laplacian2D stencil and eigenvalue formula, seeding/determinism — correct.
- **Arena/record lifecycle:** dual-mode RowPtr/ColInd/Values, generation stamps, Dispose ordering
  (Free before native dispose), builder shared-State* model — consistent with the use-after-free
  fix recorded in DEVLOG; no new lifecycle bugs found.
- **Addendum sweeps:** no role-swapped InPlace wrappers (mulInPlace/signFlipInPlace/absInPlace/
  addScaledInPlace all mutate the receiver, matching kernel semantics); no `maxIter`/`tol`/
  `Solvers`/`MatrixMetrics` stragglers (only the `bsm` locals, L9); no missing-InPlace-suffix
  methods; `Export.Sparse` `//+choose[float|double]` / `["G9"|"G17"]` blocks line up with the
  fProxy type order.

## Summary

| Severity | Count |
|----------|-------|
| HIGH     | 1     |
| MEDIUM   | 4     |
| LOW      | 13    |

Files with no findings at all: `fProxyBSRRecords.fProxy.cs`, `fProxyBSROperator.cs`,
`Analysis.Sparse.fProxy.cs`, `SparseOP.Transpose.fProxy.cs`, `Debug.Sparse.fProxy.cs`,
`Export.Sparse.fProxy.cs`, `fProxyBlockJacobi.cs`, `fProxySSOR.cs`, `fProxyIC0.cs`.
