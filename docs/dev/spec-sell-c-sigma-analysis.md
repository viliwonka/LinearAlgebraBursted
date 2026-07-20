# SELL-C-σ architecture analysis (read-only reassessment, 2026-07-20)

Reassessment of adding **SELL-C-σ** (Sliced ELLPACK, chunk size C, σ-window row sorting) as a
second sparse format for this library's matrix-free iterative solvers, on **CPU Burst SIMD**
(deterministic, `+ - * / sqrt`-only regime — NOT GPU). Every claim below is grounded in the
current code with file:line citations.

Relationship to prior work: a detailed design draft already exists —
`docs/dev/draft-spec-sell-c-sigma.md` — and was **USER-RULED 2026-07-10 as POST-v1.0, nothing
implemented** (draft §11). The memory index also parks it "post-v1.0" (see
`next-phase-decision-menu.md`). This document independently re-verifies that draft's technical
claims against the shipped code, corrects two of them, and restates the bottom line. It does not
supersede the draft's staged build plan (draft §8), which remains valid if the feature is greenlit.

---

## 1. Usefulness for sparse iterative solving on CPU SIMD

### 1.1 How SELL-C-σ's SpMV works

SELL-C-σ is ELLPACK sliced into chunks of **C consecutive rows**, each chunk padded to its own
longest row and stored **column-major within the chunk**. The spMV holds C row-accumulators in
SIMD lanes; the inner loop over column-position `j` does one contiguous vector load of `val`/`col`,
gathers C entries of `x`, and one FMA into the C accumulators. Each of the C rows is summed
**left-to-right in its own lane** — no horizontal reduction, no cross-row reassociation. σ-sorting
reorders rows by descending nnz **within windows of σ rows** so that rows sharing a chunk have
similar length, which cuts the ELLPACK zero-padding (occupancy β = nnz / Σ C·chunkLen). This is
established format theory (draft §2, Kreutzer2014); I take it as given and focus on the fit to
*this* codebase.

### 1.2 The library's current sparse path (what SELL competes against)

The current format is **BSR** (CSR-of-dense-blocks). The operator wrapper is
`fProxyBSROperator` (`Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxyBSROperator.cs:18`),
whose `Apply` forwards straight to `BSR.spMV` (`fProxyBSROperator.cs:61`). The spMV kernels live in
`UnsafeOP.Sparse.fProxy.cs`, with fully-unrolled register-tile variants for square blocks
b ∈ {1,2,3,4,6} plus a general fallback (`UnsafeOP.Sparse.fProxy.cs:14`, dispatch table at
`:141-157`). The unrolled kernels (e.g. b=3 at `:198`, the FEM/cloth workhorse) are the ~3× win the
sparse DEVLOG refers to: for b≥2 the inner block-multiply is straight-line named scalars that Burst
register-allocates and auto-vectorizes.

**The critical case is b=1** — the scalar/unstructured regime. `bsrMatVecB1`
(`UnsafeOP.Sparse.fProxy.cs:161-173`) is:

```csharp
for (int k = rowStart; k < rowEnd; k++)
    y[br] += values[k] * x[colInd[k]];
```

This is a plain CSR gather loop: one serial accumulator chain per row, a non-contiguous `x` gather.
The library's whole SIMD strategy — two `fProxy4` accumulators = 8 independent lane-chains — is a
*reduction-splitting* trick that requires contiguous data and permission to run 8 partial sums.
Neither holds here: the per-row sum is a single dependency chain, and the DEVLOG records that the
2-accumulator pairing was **tried on the b=1 stencil and reverted as a no-win**
(`Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/DEVLOG.md`, "UnsafeOP.Sparse.fProxy.cs"
2026-07-11 entry: "b=1 kept single-chain, A/B'd as a no-win exception"; software prefetch on the
gather was 8-56% slower, also reverted). So under the library's FloatMode regime, **b=1 spMV is
structurally scalar and cannot be rescued in place.** That is precisely the gap SELL-C-σ fills.

### 1.3 Where SELL beats BSR, where it loses

The trade-off is real and the axis is clear:

- **BSR vectorizes DOWN a dense b×b block** (one shared column index amortized over b² values;
  contiguous block load). Wins whenever the matrix has natural vector-DOF blocks: elasticity/truss
  b=3 (`bsrMatVecSymB3`, `:469`), 6-DOF frames b=6 (`:553`), PDE systems. These are headline
  workloads — the demos (`Assets/Demos/12_Truss3D`, `13_BuildingFrame`, `09_TrussModal`) and the
  penalized-grid gallery generator (`Gallery.Sparse`, `fProxyPenalizedGrid3D`, b=3) all live here.
- **SELL vectorizes ACROSS C rows in a chunk**. Wins whenever there is NO block structure, i.e.
  where BSR degenerates to b=1 and runs the scalar loop above: graph/network Laplacians and
  adjacency matrices, circuit matrices (cf. `Assets/Demos/08_Circuit`), scalar finite-difference/
  finite-element stencils, scattered-data / RBF sparsifications, generic user triplets.

Expected magnitude on blockless matrices vs `bsrMatVecB1` (estimates; only the draft's Stage-4
kill-gate benchmark can confirm): **~1.5–3× in-cache** (breaks the per-row FMA dependency chain
into C lane-chains, streams `val`/`col` contiguously; the scalar `x`-gather is the new ceiling, so
not 8×), compressing to **~1.0–1.5× when DRAM-resident** (single-threaded Burst does not saturate
bandwidth, so some compute win survives — this is *more* favorable than the socket-saturated
multicore runs in the literature where SELL merely matches CSR), and multiplied by β wherever
padding is present. **On blocked matrices SELL loses to BSR** — it pays 4 B of column index per
nonzero and has no dense-block inner kernel to amortize it, and BSR's `Symmetric` lower-triangle
storage (`bsrMatVecSym*`, `:83`) has no SELL analogue. SELL must never be pitched for b≥2.

Net: SELL is a **complement, not a replacement** (draft §6 — I concur, verified against the code).
It targets exactly the one case the current kernels cannot vectorize.

---

## 2. Compatibility — "we only need to do the kernels?"

**Almost.** The operator abstraction is genuinely the whole integration surface, but the interface
is **four methods, not three** (the draft §0/§6 mentions only Apply/ApplyT/ApplyBlock and omits
`ApplyDot`).

`IfProxyLinearOperator` (`Assets/LinearAlgebra/CodeGen/TemplateSource/Interfaces/LinearOperator.fProxy.cs:15`)
requires:

| Member | Line | Semantics | Who consumes it |
|---|---|---|---|
| `Apply(x, y)` | `:21` | y = A·x | **every** solver (cg/pcg/minres/bicgstab/gmres/idr/tfqmr, all block variants, LOBPCG, eigensolvers) |
| `ApplyT(x, y)` | `:25` | y = Aᵀ·x | least-squares / least-norm: lsqr, lsmr, craig, cgls (`Krylov.LSQR.fProxy.cs:59,85,123`) |
| `ApplyDot(x, y)` | `:33` | y = A·x, returns dot(x,y) | cg's `pAp = A.ApplyDot(p, Ap)` (`Krylov.CG.fProxy.cs:199`); square-A only. Composes Apply + `Blas.dot` |
| `ApplyBlock(V, AV, rows)` | `:44` | AV[i,:] = A·V[i,:] for a block of rows | LOBPCG (`LOBPCG.fProxy.cs:153,160,413,418,482`) and block-Krylov (bcg/bfbcg/bcgrq/bgmres/…); intended for symmetric A |

Solvers are generic over `TOp : struct, IfProxyLinearOperator`, so each call site compiles to a
direct (non-virtual) call per specialization — the reason a new operator needs **zero solver
edits**. The concrete BSR wrapper implements exactly these four by forwarding to BSR kernels
(`fProxyBSROperator.cs:61-75`).

**A `fProxySELLOperator` needs all four**, but only two underlying kernels do real work:

1. **forward SELL spMV** — implements `Apply` directly; `ApplyDot` = Apply then `Blas.dot`
   (same compose the BSR and dense operators use — `LinearOperator.fProxy.cs:94`,
   `fProxyBSROperator.cs:72`; note the DEVLOG's finding that a *fused* dot kernel was a ~45%
   regression at b=1 and was deleted — so compose is correct here too); `ApplyBlock` = a row-loop
   over the forward spMV initially (BSR ships a real multivector `spMM` at `:605`; SELL can defer
   that to a later optimization pass, exactly as `fProxyColScaledOperator.ApplyBlock` loops over
   scalar Apply today — `LinearOperator.fProxy.cs:216`).
2. **transpose** — `ApplyT`. See §3.2; the recommended answer is a materialized `SELL(Aᵀ)` reusing
   the forward spMV, mirroring the shipped BSR two-arg operator pattern.

So the honest statement of the user's question: **yes, adding SELL is confined to the operator + its
kernels; no solver, preconditioner, or existing-BSR code changes.** The kernel work is: one forward
SIMD spMV (the payoff), one scalar reference spMV (correctness oracle), a scalar scatter `spMVT`
fallback, and the builder/transpose plumbing. `ApplyDot`/`ApplyBlock` are free compositions at
first. That is the payoff of the operator interface and the reason the feature is cheap to bolt on.

---

## 3. The catches

### 3.1 Determinism (this is better than the prompt supposes)

The task framing says SELL's chunked reduction "gives each row's dot a DIFFERENT summation order
than BSR." **That is not the case for the forward spMV, and it matters.** In SELL each output row
lives in its own SIMD lane and is summed **left-to-right in stored column order** — identical to the
scalar `bsrMatVecB1` fold (`UnsafeOP.Sparse.fProxy.cs:171`). The library already leans on exactly
this equivalence: the unrolled BSR kernels are documented as **bit-identical** to the general
kernel because `0 + p0 == p0` makes a zero-seeded running accumulator equal to the left-associative
sum (`UnsafeOP.Sparse.fProxy.cs:141-157`). If SELL preserves each row's within-row column order
(the builder controls this), its forward spMV is **bit-identical to BSR-b1 per row** — the *only*
reordering is the row **permutation** applied at output writeback, which changes *which* `y[i]` each
lane writes, never any individual row's accumulated value. Padding contributes exact `+0.0` terms
(`0 * x[padCol]`), inert except the `-0.0`/`+0.0` sign edge noted in draft §9.

Where does the permutation get applied? The draft's design (§5.2) — **permute only rows, keep column
indices unpermuted**, scatter each chunk's C results through `y[P[row]]` at writeback — is the right
call and keeps the permutation an internal storage detail: no solver, workspace, user vector, or
preconditioner ever sees permuted data. Roughly n scattered stores per spMV, negligible next to nnz
work. This is strictly cleaner than storing/un-permuting b and x at the operator boundary.

Cross-arch: the spMV is multiply + add only — no transcendentals — so it sits in the library's
cross-arch-deterministic core (memory `determinism-analysis`: `+ - * / sqrt` are cross-arch under
Strict; reassociation is forbidden, SIMD lane-parallelism is not). SELL introduces no reassociation.
**Verdict: SELL forward spMV is deterministic AND cross-arch-reproducible, and can be made
bit-identical to the existing BSR-b1 reference — a testable acceptance criterion, not a hope.** The
determinism contract is *not* a catch here; if anything it is a selling point, because the
alternatives (CSR5-style segmented sums) would break it (draft §10).

### 3.2 Transpose (the cgls/lsqr question)

Column-major chunks make in-place `Aᵀx` a scatter with intra-vector write conflicts (two lanes may
target the same `y` entry) — SIMD-hostile, scalar-at-best. But the library already hit and solved
this for BSR: `bsrMatVecTB1` (`UnsafeOP.Sparse.fProxy.cs:284-296`) is the scatter fallback, and the
shipped fix is to **materialize the transpose once** via `arena.fProxyBSRTranspose`
(`Sparse/Arena.Sparse.fProxy.cs:132`) and hand the operator both matrices through the two-arg
`fProxyBSROperator(in A, in aT)` ctor (`fProxyBSROperator.cs:51,63-69`), turning `ApplyT` into a
forward spMV over Aᵀ. SELL should copy this exactly: a two-arg `fProxySELLOperator(in A, in AT)`
where `AT = SELL(Aᵀ)` gets its own σ-sort and its own internal permutation, so `ApplyT` inherits the
full §1.3 speedup and stays deterministic. Cost: ≈2× matrix memory (precedented). A scalar scatter
`spMVT` over the untransposed storage ships as the one-arg fallback for memory-constrained callers,
mirroring BSR's one-arg ctor. With the two-arg pattern, lsqr/lsmr/craig/cgls work unchanged and get
faster; without it SELL would be a net loss for least-squares — so the pattern is the design.

### 3.3 Build / convert cost and bookkeeping

- **Construction** is the same order of work as `ToBSR`: count row lengths O(nnz), stable-sort rows
  within σ-windows O(n·log σ) using the existing `Indices` int-buffer machinery (draft §3 confirms
  `Pivot` is on the revised-simplex do-not-touch list, so SELL uses `Indices`), compute chunk
  offsets, one gather pass to fill padded `val`/`col`. Build from builder triplets or from a finished
  BSR-b1 (`toSELL(sigma)`).
- **Memory/padding** vs BSR-b1: same bytes/nnz divided by β, plus tiny `cs`/`cl` and 8 B/row for
  P/P⁻¹. Worst case β→1/C for a pathological single-dense-row chunk; σ = C² (=64) makes that
  construction impossible within a window. The builder must **compute and report β**, and docs must
  state SELL is a bad idea below β≈0.7 (draft §5.1). This is the honest failure mode.
- **Bookkeeping burden on callers: essentially none**, given the rows-only-permutation design
  (§3.1) — the operator hides P entirely. This is the single most important design decision for
  keeping the feature additive.
- **NaN/padding edge** (draft §9): padded `0 * x[padCol]` injects NaN if `x[padCol]` is non-finite;
  point `padCol` at the row's own first column to confine it, accept the empty-row residual hole,
  document + pin with a test. Same "document the pathology" precedent the integer-surface policy uses.

---

## 4. Verdict + effort

**Recommendation: qualified GO, POST-v1.0, behind the draft's kill-gate benchmark — unchanged from
the 2026-07-10 user ruling, and re-confirmed by this code re-read.** Nothing in the current code
changes the calculus; if anything the determinism story (§3.1) came out *stronger* than the draft
stated (bit-identical to BSR-b1, not merely "deterministic-but-different").

Why GO at all: `bsrMatVecB1` (`:161`) is the library's answer for blockless matrices today, it is
provably scalar under FloatMode (the pairing/prefetch reverts in the DEVLOG close off the in-place
alternatives), and SELL-C-σ is the standard, production-proven layout whose entire purpose is to
SIMD-vectorize that case without reordering any row sum. It enters through the operator interface
with zero solver edits.

Why qualified / why post-v1.0:
1. **It grows the public sparse surface** (a second format to document and explain) right when the
   release plan wants the API frozen. Purely additive, but not free to maintain.
2. **The win is workload-gated.** It only pays on blockless matrices that are also reasonably
   cache-resident and reasonably high-β. 4-lane `fProxy4` registers and scalar `x`-gathers cap the
   ceiling well below GPU/Phi headlines. Only a benchmark can say whether 1.5–3× survives Burst.
3. **The customer must be real.** SELL helps graph Laplacians, circuit/network matrices, scalar
   stencils, scattered-data LS. It does nothing for the physics-mesh demos (b∈{2,3,6}) or the LP/QP
   interior-point operators, which are current headline workloads and stay on BSR. The user already
   flagged the b=1/heat/fluids demos as the concrete target (draft §11.2), so the customer exists in
   principle — but it is a *future* demo, not shipped code today, which is itself an argument for
   post-v1.0.

**Kill gate (binding, from draft §7):** Stage-4 benchmark must show ≥1.5× spMV vs BSR-b1 in-cache on
unstructured matrices at β≥0.9, and no >10% regression in any measured cell the docs don't disclaim;
plus bit-identity vs the scalar reference per row. Miss ⇒ park the branch, write the numbers into the
spec, done.

**Effort estimate** (each stage = one coder round, suite green both dtypes before the next; matches
draft §8):

| Stage | Work | Rough size |
|---|---|---|
| 1. Format + reference | `fProxySELL` struct (arena/standalone, mirroring `fProxyBSR`), build from triplets and from BSR-b1, β stored, **scalar** reference spMV + scalar scatter spMVT in `UnsafeOP.Sparse`, kernel tests (empty rows, empty matrix, σ=1, C∤n tail, NaN-padding), one literature known-answer matrix | ~1 new source file + `SparseOP`/`UnsafeOP`/`Arena` additions + 1 test file |
| 2. SIMD kernel + operator | C=8 chunk kernel (2×`fProxy4`), **bit-identity test vs Stage-1 scalar** (the determinism acceptance criterion), `arena.fProxySELLTranspose`, one-arg + two-arg `fProxySELLOperator`, cg/cgls/lsqr/lsmr integration tests matched to a BSR operator | ~1 new source file + kernel + operator tests |
| 3. Gallery | `fProxyRandomSparseSkewed` (power-law/Zipf row degrees, seeded, SPD + rectangular) — the honest skewed-row test matrix the gallery lacks today; β assertions as tests | `Gallery.Sparse` addition + test |
| 4. Benchmark + gate | SELL vs BSR-b1 vs dense across cache-resident/DRAM, fills, σ∈{1,C²,256}, C∈{4,8} once, + cgls/lsqr end-to-end; apply kill gate; write numbers back into the spec; only then docs/README | 1 benchmark section |

Kernel variants strictly needed for full solver coverage: **forward SIMD spMV** (Apply, ApplyDot via
compose, ApplyBlock via row-loop) and **transpose** (ApplyT via materialized SELL(Aᵀ) forward spMV +
a scalar scatter fallback). A real SELL `spMM` for ApplyBlock/LOBPCG is an optional later
optimization, not a coverage requirement.

**Bottom line:** technically sound, cleanly additive, determinism-compatible (better than expected),
and it targets a genuine gap (`bsrMatVecB1` is scalar and unfixable in place). But the payoff is
gated on a blockless-matrix workload this library does not yet ship, so it stays **post-v1.0 behind
the benchmark kill-gate** — exactly where the user parked it. When the heat/fluids/graph demos become
real, revisit and run Stage 4; if the ≥1.5× gate misses on Burst's 4-lane registers, park it
permanently and record the numbers.
